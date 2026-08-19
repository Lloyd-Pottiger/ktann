use std::fmt;

use bytes::Bytes;
use foundationdb::options::StreamingMode;
use foundationdb::{Database, FdbError, RangeOption, Transaction};
use futures_util::TryStreamExt;
use futures_util::future::try_join_all;
use ktann::api::{Error, ErrorKind, Result};
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation,
    ReadOps, ReadTxn, ScanItem, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::KeyRange;

const FDB_MAX_KEY_BYTES: usize = 10_000;
const FDB_MAX_VALUE_BYTES: usize = 100_000;
const DEFAULT_MAX_MUTATIONS: usize = 10_000;
const DEFAULT_MAX_MUTATION_BYTES: usize = 1 << 20;
const MAX_SCAN_PAGE_BYTES: usize = 80 << 10;
const MAX_CONCURRENT_POINT_READS: usize = 1_024;
const MAX_BACKEND_NAMESPACE_BYTES: usize = u8::MAX as usize;
const PHYSICAL_PREFIX_HEADER: &[u8] = b"\0ktann\x01";

const FDB_TRANSACTION_TOO_OLD: i32 = 1007;
const FDB_TRANSACTION_TOO_LARGE: i32 = 2101;
const FDB_KEY_TOO_LARGE: i32 = 2102;
const FDB_VALUE_TOO_LARGE: i32 = 2103;

/// A caller-selected FoundationDB storage scope for KTANN Logical Indexes.
///
/// Namespace bytes are opaque and may be empty. They are length-delimited in
/// the physical key prefix, so distinct values always select disjoint key
/// ranges. `Debug` redacts the bytes because callers may treat them as
/// sensitive deployment metadata.
#[derive(Clone, Eq, PartialEq)]
pub struct BackendNamespace(Bytes);

impl BackendNamespace {
    /// Constructs a Backend Namespace from at most 255 opaque bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidArgument`] when the namespace is longer
    /// than 255 bytes.
    pub fn new(bytes: impl AsRef<[u8]>) -> Result<Self> {
        let bytes = bytes.as_ref();
        if bytes.len() > MAX_BACKEND_NAMESPACE_BYTES {
            return Err(Error::new(ErrorKind::InvalidArgument));
        }
        Ok(Self(Bytes::copy_from_slice(bytes)))
    }

    /// Returns the opaque namespace bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for BackendNamespace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BackendNamespace([REDACTED])")
    }
}

#[derive(Debug)]
struct PhysicalPrefix {
    bytes: Bytes,
}

impl PhysicalPrefix {
    fn new(namespace: &BackendNamespace) -> Self {
        let mut bytes = Vec::with_capacity(PHYSICAL_PREFIX_HEADER.len() + 1 + namespace.0.len());
        bytes.extend_from_slice(PHYSICAL_PREFIX_HEADER);
        bytes.push(namespace.0.len() as u8);
        bytes.extend_from_slice(&namespace.0);
        Self {
            bytes: Bytes::from(bytes),
        }
    }

    fn max_logical_key_bytes(&self) -> usize {
        FDB_MAX_KEY_BYTES - self.bytes.len()
    }

    fn validate_key(&self, logical_key: &[u8]) -> Result<()> {
        if logical_key.len() > self.max_logical_key_bytes() {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        Ok(())
    }

    fn encode_key(&self, logical_key: &[u8]) -> Result<Bytes> {
        self.validate_key(logical_key)?;
        let mut physical_key = Vec::with_capacity(self.bytes.len() + logical_key.len());
        physical_key.extend_from_slice(&self.bytes);
        physical_key.extend_from_slice(logical_key);
        Ok(Bytes::from(physical_key))
    }

    fn decode_key(&self, physical_key: &[u8]) -> Result<Bytes> {
        let logical_key = physical_key
            .strip_prefix(self.bytes.as_ref())
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        Ok(Bytes::copy_from_slice(logical_key))
    }
}

/// The FoundationDB adapter for KTANN's backend-neutral transaction seam.
///
/// The caller owns FoundationDB's process-global network lifecycle and passes
/// an already-open [`Database`]. The adapter owns one Backend Namespace within
/// that database and never performs internal transaction retries.
pub struct FoundationDbBackend {
    database: Database,
    namespace: BackendNamespace,
    prefix: PhysicalPrefix,
}

impl FoundationDbBackend {
    /// Constructs an adapter over `database` and one Backend Namespace.
    ///
    /// FoundationDB's network must already be running and must outlive this
    /// adapter and all transactions opened from it.
    #[must_use]
    pub fn new(database: Database, namespace: BackendNamespace) -> Self {
        let prefix = PhysicalPrefix::new(&namespace);
        Self {
            database,
            namespace,
            prefix,
        }
    }

    /// Returns this adapter's Backend Namespace.
    #[must_use]
    pub fn namespace(&self) -> &BackendNamespace {
        &self.namespace
    }

    async fn begin_transaction(&self) -> Result<Transaction> {
        let transaction = self.database.create_trx().map_err(map_operation_error)?;
        transaction
            .get_read_version()
            .await
            .map_err(map_operation_error)?;
        Ok(transaction)
    }
}

impl fmt::Debug for FoundationDbBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FoundationDbBackend")
            .field("namespace", &self.namespace)
            .finish_non_exhaustive()
    }
}

/// A consistent FoundationDB read transaction.
pub struct FoundationDbReadTxn<'backend> {
    transaction: Transaction,
    prefix: &'backend PhysicalPrefix,
}

impl fmt::Debug for FoundationDbReadTxn<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FoundationDbReadTxn")
            .finish_non_exhaustive()
    }
}

/// An atomic FoundationDB write transaction with read-your-writes behavior.
pub struct FoundationDbWriteTxn<'backend> {
    transaction: Transaction,
    prefix: &'backend PhysicalPrefix,
    mutation_count: usize,
    mutation_bytes: usize,
}

impl fmt::Debug for FoundationDbWriteTxn<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FoundationDbWriteTxn")
            .field("mutation_count", &self.mutation_count)
            .field("mutation_bytes", &self.mutation_bytes)
            .finish_non_exhaustive()
    }
}

impl Backend for FoundationDbBackend {
    type ReadTxn<'backend> = FoundationDbReadTxn<'backend>;
    type WriteTxn<'backend> = FoundationDbWriteTxn<'backend>;

    fn hard_limits(&self) -> HardLimits {
        HardLimits {
            max_key_bytes: self.prefix.max_logical_key_bytes(),
            max_value_bytes: FDB_MAX_VALUE_BYTES,
        }
    }

    fn admission_budget(&self) -> AdmissionBudget {
        AdmissionBudget {
            max_mutations: DEFAULT_MAX_MUTATIONS,
            max_mutation_bytes: DEFAULT_MAX_MUTATION_BYTES,
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            transactional_clear_range: true,
        }
    }

    async fn begin_read(&self) -> Result<FoundationDbReadTxn<'_>> {
        let transaction = self.begin_transaction().await?;
        Ok(FoundationDbReadTxn {
            transaction,
            prefix: &self.prefix,
        })
    }

    async fn begin_write(&self) -> Result<FoundationDbWriteTxn<'_>> {
        let transaction = self.begin_transaction().await?;
        Ok(FoundationDbWriteTxn {
            transaction,
            prefix: &self.prefix,
            mutation_count: 0,
            mutation_bytes: 0,
        })
    }
}

impl ReadOps for FoundationDbReadTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        get(&self.transaction, self.prefix, key, ReadMode::Snapshot).await
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        batch_get(&self.transaction, self.prefix, keys, ReadMode::Snapshot).await
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        scan(&self.transaction, self.prefix, range, limits).await
    }
}

impl ReadTxn for FoundationDbReadTxn<'_> {}

impl ReadOps for FoundationDbWriteTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        get(&self.transaction, self.prefix, key, ReadMode::Snapshot).await
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        batch_get(&self.transaction, self.prefix, keys, ReadMode::Snapshot).await
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        scan(&self.transaction, self.prefix, range, limits).await
    }
}

impl FoundationDbWriteTxn<'_> {
    fn charge(&mut self, mutation_count: usize, mutation_bytes: usize) -> Result<()> {
        let next_count = self
            .mutation_count
            .checked_add(mutation_count)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let next_bytes = self
            .mutation_bytes
            .checked_add(mutation_bytes)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        if next_count > DEFAULT_MAX_MUTATIONS || next_bytes > DEFAULT_MAX_MUTATION_BYTES {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        self.mutation_count = next_count;
        self.mutation_bytes = next_bytes;
        Ok(())
    }

    fn prepare_put(&self, key: Bytes, value: Bytes) -> Result<PreparedMutation> {
        validate_value(&value)?;
        let key = self.prefix.encode_key(&key)?;
        Ok(PreparedMutation::Put { key, value })
    }

    fn prepare_delete(&self, key: Bytes) -> Result<PreparedMutation> {
        let key = self.prefix.encode_key(&key)?;
        Ok(PreparedMutation::Delete { key })
    }
}

impl WriteTxn for FoundationDbWriteTxn<'_> {
    async fn get_for_update(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        get(&self.transaction, self.prefix, key, ReadMode::ForUpdate).await
    }

    async fn batch_get_for_update(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        batch_get(&self.transaction, self.prefix, keys, ReadMode::ForUpdate).await
    }

    async fn put(&mut self, key: Bytes, value: Bytes) -> Result<()> {
        let mutation = self.prepare_put(key, value)?;
        self.charge(1, mutation.charged_bytes()?)?;
        mutation.apply(&self.transaction);
        Ok(())
    }

    async fn insert(&mut self, key: Bytes, value: Bytes) -> Result<InsertOutcome> {
        validate_value(&value)?;
        let physical_key = self.prefix.encode_key(&key)?;
        let charged_bytes = physical_key
            .len()
            .checked_add(value.len())
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let existing = self
            .transaction
            .get(&physical_key, false)
            .await
            .map_err(map_operation_error)?;
        if existing.is_some() {
            return Ok(InsertOutcome::AlreadyExists);
        }
        self.charge(1, charged_bytes)?;
        self.transaction.set(&physical_key, &value);
        Ok(InsertOutcome::Inserted)
    }

    async fn delete(&mut self, key: Bytes) -> Result<()> {
        let mutation = self.prepare_delete(key)?;
        self.charge(1, mutation.charged_bytes()?)?;
        mutation.apply(&self.transaction);
        Ok(())
    }

    async fn batch_mutate(&mut self, mutations: Vec<Mutation>) -> Result<()> {
        if mutations.len() > DEFAULT_MAX_MUTATIONS.saturating_sub(self.mutation_count) {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let mut prepared = Vec::with_capacity(mutations.len());
        let mut charged_bytes = 0_usize;
        for mutation in mutations {
            let mutation = match mutation {
                Mutation::Put { key, value } => self.prepare_put(key, value)?,
                Mutation::Delete { key } => self.prepare_delete(key)?,
                _ => return Err(Error::new(ErrorKind::Unsupported)),
            };
            charged_bytes = charged_bytes
                .checked_add(mutation.charged_bytes()?)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            prepared.push(mutation);
        }
        self.charge(prepared.len(), charged_bytes)?;
        for mutation in prepared {
            mutation.apply(&self.transaction);
        }
        Ok(())
    }

    async fn clear_range(&mut self, range: &KeyRange) -> Result<()> {
        if range.start() >= range.end() {
            return Ok(());
        }
        let start = self.prefix.encode_key(range.start())?;
        let end = self.prefix.encode_key(range.end())?;
        let charged_bytes = start
            .len()
            .checked_add(end.len())
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        self.charge(1, charged_bytes)?;
        self.transaction.clear_range(&start, &end);
        Ok(())
    }

    async fn commit_with(self, start: CommitStart) -> Result<()> {
        start.begin()?;
        self.transaction
            .commit()
            .await
            .map(|_| ())
            .map_err(|error| map_commit_error(error.into()))
    }

    async fn rollback(self) {}
}

enum PreparedMutation {
    Put { key: Bytes, value: Bytes },
    Delete { key: Bytes },
}

#[derive(Clone, Copy)]
enum ReadMode {
    Snapshot,
    ForUpdate,
}

impl ReadMode {
    fn is_snapshot(self) -> bool {
        matches!(self, Self::Snapshot)
    }
}

impl PreparedMutation {
    fn charged_bytes(&self) -> Result<usize> {
        match self {
            Self::Put { key, value } => key
                .len()
                .checked_add(value.len())
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded)),
            Self::Delete { key } => Ok(key.len()),
        }
    }

    fn apply(self, transaction: &Transaction) {
        match self {
            Self::Put { key, value } => transaction.set(&key, &value),
            Self::Delete { key } => transaction.clear(&key),
        }
    }
}

async fn get(
    transaction: &Transaction,
    prefix: &PhysicalPrefix,
    key: Bytes,
    mode: ReadMode,
) -> Result<Option<Bytes>> {
    let key = prefix.encode_key(&key)?;
    transaction
        .get(&key, mode.is_snapshot())
        .await
        .map(|value| value.map(Bytes::from_owner))
        .map_err(map_operation_error)
}

async fn batch_get(
    transaction: &Transaction,
    prefix: &PhysicalPrefix,
    keys: Vec<Bytes>,
    mode: ReadMode,
) -> Result<Vec<Option<Bytes>>> {
    for key in &keys {
        prefix.validate_key(key)?;
    }
    let mut values = Vec::with_capacity(keys.len());
    for keys in keys.chunks(MAX_CONCURRENT_POINT_READS) {
        let keys = keys
            .iter()
            .map(|key| prefix.encode_key(key))
            .collect::<Result<Vec<_>>>()?;
        let reads = keys
            .iter()
            .map(|key| transaction.get(key, mode.is_snapshot()));
        values.extend(
            try_join_all(reads)
                .await
                .map_err(map_operation_error)?
                .into_iter()
                .map(|value| value.map(Bytes::from_owner)),
        );
    }
    Ok(values)
}

async fn scan(
    transaction: &Transaction,
    prefix: &PhysicalPrefix,
    range: &KeyRange,
    limits: ScanLimits,
) -> Result<ScanPage> {
    if limits.item_limit == 0 || limits.byte_limit == 0 {
        return Err(Error::new(ErrorKind::InvalidArgument));
    }
    if range.start() >= range.end() {
        return Ok(ScanPage::terminal(Vec::new()));
    }
    let start = prefix.encode_key(range.start())?;
    let end = prefix.encode_key(range.end())?;
    let byte_limit = limits.byte_limit.min(MAX_SCAN_PAGE_BYTES);
    let mut items = Vec::new();
    let mut item_bytes = 0_usize;
    let candidate_items = limits
        .item_limit
        .saturating_add(1)
        .min(byte_limit.saturating_add(1));
    let target_bytes = prefix
        .bytes
        .len()
        .saturating_mul(candidate_items)
        .saturating_add(byte_limit);
    let options = RangeOption {
        limit: Some(limits.item_limit.saturating_add(1)),
        target_bytes,
        mode: StreamingMode::WantAll,
        ..RangeOption::from(start.as_ref()..end.as_ref())
    };
    let mut batches = transaction.get_ranges(options, true);

    while let Some(batch) = batches.try_next().await.map_err(map_operation_error)? {
        for value in &batch {
            let logical_key = prefix.decode_key(value.key())?;
            let bytes = logical_key
                .len()
                .checked_add(value.value().len())
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            if !items.is_empty()
                && (items.len() >= limits.item_limit
                    || item_bytes
                        .checked_add(bytes)
                        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?
                        > byte_limit)
            {
                return ScanPage::continued(items, prefix.max_logical_key_bytes());
            }
            item_bytes = item_bytes
                .checked_add(bytes)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            items.push(ScanItem::new(
                logical_key,
                Bytes::copy_from_slice(value.value()),
            ));
        }
    }
    Ok(ScanPage::terminal(items))
}

fn validate_value(value: &[u8]) -> Result<()> {
    if value.len() > FDB_MAX_VALUE_BYTES {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    Ok(())
}

fn map_operation_error(error: FdbError) -> Error {
    let kind = if error.code() != FDB_TRANSACTION_TOO_OLD && error.is_retryable_not_committed() {
        ErrorKind::RetryableAbort
    } else {
        map_non_commit_error_kind(error.code())
    };
    Error::with_source(kind, error)
}

fn map_commit_error(error: FdbError) -> Error {
    let kind = if error.is_maybe_committed() {
        ErrorKind::CommitOutcomeUnknown
    } else if error.is_retryable_not_committed() {
        ErrorKind::RetryableAbort
    } else {
        map_non_commit_error_kind(error.code())
    };
    Error::with_source(kind, error)
}

fn map_non_commit_error_kind(code: i32) -> ErrorKind {
    match code {
        FDB_TRANSACTION_TOO_LARGE => ErrorKind::TransactionTooLarge,
        FDB_KEY_TOO_LARGE | FDB_VALUE_TOO_LARGE => ErrorKind::LimitExceeded,
        _ => ErrorKind::Backend,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_prefix_is_versioned_length_delimited_and_order_preserving() {
        let a = PhysicalPrefix::new(&BackendNamespace::new("a").expect("namespace"));
        let aa = PhysicalPrefix::new(&BackendNamespace::new("aa").expect("namespace"));

        assert_ne!(a.bytes, aa.bytes);
        assert_eq!(a.bytes.as_ref(), b"\0ktann\x01\x01a");
        assert!(a.encode_key(b"left").expect("key") < a.encode_key(b"right").expect("key"));
        assert!(!aa.encode_key(b"key").expect("key").starts_with(&a.bytes));
    }

    #[test]
    fn logical_key_limit_accounts_for_the_physical_prefix() {
        let prefix = PhysicalPrefix::new(&BackendNamespace::new("scope").expect("namespace"));
        let maximum = vec![0; prefix.max_logical_key_bytes()];
        assert_eq!(
            prefix.encode_key(&maximum).expect("maximum").len(),
            FDB_MAX_KEY_BYTES
        );
        assert_eq!(
            prefix
                .encode_key(&[maximum, vec![0]].concat())
                .expect_err("oversized")
                .kind(),
            ErrorKind::LimitExceeded,
        );
    }

    #[test]
    fn namespace_is_bounded_and_debug_is_redacted() {
        let namespace = BackendNamespace::new(Bytes::from_static(b"secret")).expect("namespace");
        assert_eq!(format!("{namespace:?}"), "BackendNamespace([REDACTED])");
        assert_eq!(
            BackendNamespace::new(vec![0; MAX_BACKEND_NAMESPACE_BYTES + 1])
                .expect_err("oversized")
                .kind(),
            ErrorKind::InvalidArgument,
        );
    }

    #[test]
    fn fdb_errors_map_to_stable_backend_categories() {
        assert_eq!(
            map_operation_error(FdbError::from_code(FDB_KEY_TOO_LARGE)).kind(),
            ErrorKind::LimitExceeded,
        );
        assert_eq!(
            map_operation_error(FdbError::from_code(FDB_TRANSACTION_TOO_LARGE)).kind(),
            ErrorKind::TransactionTooLarge,
        );
        assert_eq!(
            map_operation_error(FdbError::from_code(1009)).kind(),
            ErrorKind::RetryableAbort,
        );
        assert_eq!(
            map_operation_error(FdbError::from_code(FDB_TRANSACTION_TOO_OLD)).kind(),
            ErrorKind::Backend,
        );
        assert_eq!(
            map_commit_error(FdbError::from_code(1020)).kind(),
            ErrorKind::RetryableAbort,
        );
        assert_eq!(
            map_commit_error(FdbError::from_code(1021)).kind(),
            ErrorKind::CommitOutcomeUnknown,
        );
    }
}
