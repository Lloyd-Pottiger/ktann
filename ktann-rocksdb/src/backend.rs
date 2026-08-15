use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{Error, ErrorKind, Result};
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation,
    ReadOps, ReadTxn, ScanItem, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::KeyRange;
use rocksdb::{
    DBAccess, Error as RocksError, ErrorKind as RocksErrorKind, OptimisticTransactionDB,
    OptimisticTransactionOptions, ReadOptions, SnapshotWithThreadMode, Transaction, WriteOptions,
};

use crate::blocking::{AdmittedHandle, BlockingAdmission};
use crate::config::RocksDbConfig;

const ROCKSDB_MAX_PHYSICAL_KEY_BYTES: usize = u32::MAX as usize;
const ROCKSDB_MAX_VALUE_BYTES: usize = u32::MAX as usize;
const DEFAULT_MAX_MUTATIONS: usize = 10_000;
const DEFAULT_MAX_MUTATION_BYTES: usize = 1 << 20;
const MAX_SCAN_PAGE_BYTES: usize = 80 << 10;
const MAX_BATCH_POINT_READS: usize = 1_024;
const MAX_BACKEND_NAMESPACE_BYTES: usize = u8::MAX as usize;
const PHYSICAL_PREFIX_HEADER: &[u8] = b"\0ktann-rocksdb\x01";

/// A caller-selected RocksDB storage scope for KTANN Logical Indexes.
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
        ROCKSDB_MAX_PHYSICAL_KEY_BYTES - self.bytes.len()
    }

    fn validate_key(&self, logical_key: &[u8]) -> Result<()> {
        if logical_key.len() > self.max_logical_key_bytes() {
            return Err(limit_exceeded());
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
        if !physical_key.starts_with(self.bytes.as_ref()) {
            return Err(Error::new(ErrorKind::Corruption));
        }
        Ok(Bytes::copy_from_slice(&physical_key[self.bytes.len()..]))
    }
}

/// The RocksDB adapter for KTANN's backend-neutral transaction seam.
///
/// The caller owns the database configuration and lifecycle. Multiple adapter
/// instances may share one database through [`Arc`] while binding different
/// Backend Namespaces. The adapter never performs internal transaction retries.
pub struct RocksDbBackend {
    database: Arc<OptimisticTransactionDB>,
    namespace: BackendNamespace,
    prefix: PhysicalPrefix,
    config: RocksDbConfig,
    blocking: BlockingAdmission,
}

impl RocksDbBackend {
    /// Constructs an adapter over `database` and one Backend Namespace.
    ///
    /// An owned [`OptimisticTransactionDB`] or a shared
    /// `Arc<OptimisticTransactionDB>` may be passed. Database options remain
    /// caller-owned; transaction commits always keep WAL enabled and request a
    /// synchronous WAL flush.
    ///
    /// # Database requirements
    ///
    /// The database must use a comparator whose ordering and equality exactly
    /// match lexicographic byte ordering. Custom comparators with different
    /// semantics are unsupported because physical namespace isolation and
    /// ordered logical range scans depend on raw byte order.
    #[must_use]
    pub fn new(
        database: impl Into<Arc<OptimisticTransactionDB>>,
        namespace: BackendNamespace,
    ) -> Self {
        Self::with_config(database, namespace, RocksDbConfig::default())
    }

    /// Constructs an adapter with explicit process-local resource limits.
    ///
    /// The configured blocking limit applies to this adapter instance. Opening
    /// a transaction waits asynchronously for one resource slot, retains it
    /// through native cleanup, and enters RocksDB only from a Tokio multi-thread
    /// runtime's blocking section.
    #[must_use]
    pub fn with_config(
        database: impl Into<Arc<OptimisticTransactionDB>>,
        namespace: BackendNamespace,
        config: RocksDbConfig,
    ) -> Self {
        let prefix = PhysicalPrefix::new(&namespace);
        let blocking = BlockingAdmission::new(config.blocking_resource_limit());
        Self {
            database: database.into(),
            namespace,
            prefix,
            config,
            blocking,
        }
    }

    /// Returns this adapter's Backend Namespace.
    #[must_use]
    pub fn namespace(&self) -> &BackendNamespace {
        &self.namespace
    }

    /// Returns this adapter's process-local resource limits.
    #[must_use]
    pub fn config(&self) -> &RocksDbConfig {
        &self.config
    }
}

impl fmt::Debug for RocksDbBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RocksDbBackend")
            .field("namespace", &self.namespace)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// A consistent RocksDB read transaction.
pub struct RocksDbReadTxn<'backend> {
    snapshot: AdmittedHandle<'backend, SnapshotWithThreadMode<'backend, OptimisticTransactionDB>>,
    prefix: &'backend PhysicalPrefix,
}

impl fmt::Debug for RocksDbReadTxn<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RocksDbReadTxn")
            .finish_non_exhaustive()
    }
}

/// An atomic RocksDB write transaction with snapshot reads and read-your-writes.
pub struct RocksDbWriteTxn<'backend> {
    transaction: AdmittedHandle<'backend, Transaction<'backend, OptimisticTransactionDB>>,
    prefix: &'backend PhysicalPrefix,
    mutation_count: usize,
    mutation_bytes: usize,
}

impl fmt::Debug for RocksDbWriteTxn<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RocksDbWriteTxn")
            .field("mutation_count", &self.mutation_count)
            .field("mutation_bytes", &self.mutation_bytes)
            .finish_non_exhaustive()
    }
}

impl Backend for RocksDbBackend {
    type ReadTxn<'backend> = RocksDbReadTxn<'backend>;
    type WriteTxn<'backend> = RocksDbWriteTxn<'backend>;

    fn hard_limits(&self) -> HardLimits {
        HardLimits {
            max_key_bytes: self.prefix.max_logical_key_bytes(),
            max_value_bytes: ROCKSDB_MAX_VALUE_BYTES,
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
            transactional_clear_range: false,
        }
    }

    async fn begin_read(&self) -> Result<RocksDbReadTxn<'_>> {
        let snapshot = self
            .blocking
            .admit()
            .await?
            .open(|| self.database.snapshot())?;
        Ok(RocksDbReadTxn {
            snapshot,
            prefix: &self.prefix,
        })
    }

    async fn begin_write(&self) -> Result<RocksDbWriteTxn<'_>> {
        let transaction = self.blocking.admit().await?.open(|| {
            let mut write_options = WriteOptions::default();
            write_options.disable_wal(false);
            write_options.set_sync(true);
            let mut transaction_options = OptimisticTransactionOptions::default();
            transaction_options.set_snapshot(true);
            self.database
                .transaction_opt(&write_options, &transaction_options)
        })?;
        Ok(RocksDbWriteTxn {
            transaction,
            prefix: &self.prefix,
            mutation_count: 0,
            mutation_bytes: 0,
        })
    }
}

impl ReadOps for RocksDbReadTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        let key = self.prefix.encode_key(&key)?;
        self.snapshot.run(|snapshot| get(snapshot, &key))
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        for key in &keys {
            self.prefix.validate_key(key)?;
        }
        self.snapshot
            .run(|snapshot| batch_get(snapshot, self.prefix, keys))
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        self.snapshot
            .run(|snapshot| scan(snapshot, self.prefix, range, limits))
    }
}

impl ReadTxn for RocksDbReadTxn<'_> {}

impl ReadOps for RocksDbWriteTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        let key = self.prefix.encode_key(&key)?;
        self.transaction.run(|transaction| {
            let snapshot = transaction.snapshot();
            get(&snapshot, &key)
        })
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        for key in &keys {
            self.prefix.validate_key(key)?;
        }
        self.transaction.run(|transaction| {
            let snapshot = transaction.snapshot();
            batch_get(&snapshot, self.prefix, keys)
        })
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        self.transaction.run(|transaction| {
            let snapshot = transaction.snapshot();
            scan(&snapshot, self.prefix, range, limits)
        })
    }
}

impl RocksDbWriteTxn<'_> {
    fn next_charge(&self, mutation_count: usize, mutation_bytes: usize) -> Result<(usize, usize)> {
        let next_count = self
            .mutation_count
            .checked_add(mutation_count)
            .ok_or_else(limit_exceeded)?;
        let next_bytes = self
            .mutation_bytes
            .checked_add(mutation_bytes)
            .ok_or_else(limit_exceeded)?;
        if next_count > DEFAULT_MAX_MUTATIONS || next_bytes > DEFAULT_MAX_MUTATION_BYTES {
            return Err(limit_exceeded());
        }
        Ok((next_count, next_bytes))
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

impl WriteTxn for RocksDbWriteTxn<'_> {
    async fn get_for_update(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        let key = self.prefix.encode_key(&key)?;
        self.transaction.run(|transaction| {
            let snapshot = transaction.snapshot();
            let mut read_options = ReadOptions::default();
            read_options.set_snapshot(&snapshot);
            get_for_update(transaction, &key, &read_options)
        })
    }

    async fn batch_get_for_update(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        for key in &keys {
            self.prefix.validate_key(key)?;
        }
        self.transaction.run(|transaction| {
            let snapshot = transaction.snapshot();
            let mut read_options = ReadOptions::default();
            read_options.set_snapshot(&snapshot);
            let mut values = Vec::with_capacity(keys.len());
            for key in keys {
                let key = self.prefix.encode_key(&key)?;
                values.push(get_for_update(transaction, &key, &read_options)?);
            }
            Ok(values)
        })
    }

    async fn put(&mut self, key: Bytes, value: Bytes) -> Result<()> {
        let mutation = self.prepare_put(key, value)?;
        let charge = self.next_charge(1, mutation.charged_bytes()?)?;
        self.transaction
            .run(|transaction| mutation.apply(transaction))?;
        (self.mutation_count, self.mutation_bytes) = charge;
        Ok(())
    }

    async fn insert(&mut self, key: Bytes, value: Bytes) -> Result<InsertOutcome> {
        let mutation = self.prepare_put(key, value)?;
        let charged_bytes = mutation.charged_bytes()?;
        let (outcome, charge) = self.transaction.run(|transaction| {
            let snapshot = transaction.snapshot();
            let mut read_options = ReadOptions::default();
            read_options.set_snapshot(&snapshot);
            if get_for_update(transaction, mutation.key(), &read_options)?.is_some() {
                return Ok((InsertOutcome::AlreadyExists, None));
            }
            let charge = self.next_charge(1, charged_bytes)?;
            mutation.apply(transaction)?;
            Ok((InsertOutcome::Inserted, Some(charge)))
        })?;
        if let Some((mutation_count, mutation_bytes)) = charge {
            self.mutation_count = mutation_count;
            self.mutation_bytes = mutation_bytes;
        }
        Ok(outcome)
    }

    async fn delete(&mut self, key: Bytes) -> Result<()> {
        let mutation = self.prepare_delete(key)?;
        let charge = self.next_charge(1, mutation.charged_bytes()?)?;
        self.transaction
            .run(|transaction| mutation.apply(transaction))?;
        (self.mutation_count, self.mutation_bytes) = charge;
        Ok(())
    }

    async fn batch_mutate(&mut self, mutations: Vec<Mutation>) -> Result<()> {
        if mutations.len() > DEFAULT_MAX_MUTATIONS.saturating_sub(self.mutation_count) {
            return Err(limit_exceeded());
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
                .ok_or_else(limit_exceeded)?;
            prepared.push(mutation);
        }
        let charge = self.next_charge(prepared.len(), charged_bytes)?;
        self.transaction.run(|transaction| {
            for mutation in prepared {
                mutation.apply(transaction)?;
            }
            Ok(())
        })?;
        (self.mutation_count, self.mutation_bytes) = charge;
        Ok(())
    }

    async fn clear_range(&mut self, _range: &KeyRange) -> Result<()> {
        Err(Error::new(ErrorKind::Unsupported))
    }

    async fn commit_with(self, start: CommitStart) -> Result<()> {
        self.transaction.ensure_supported()?;
        start.begin()?;
        let (transaction, section) = self.transaction.into_section();
        section.run(|| transaction.commit().map_err(map_commit_error))
    }

    async fn rollback(self) {
        drop(self);
    }
}

enum PreparedMutation {
    Put { key: Bytes, value: Bytes },
    Delete { key: Bytes },
}

impl PreparedMutation {
    fn key(&self) -> &Bytes {
        match self {
            Self::Put { key, .. } | Self::Delete { key } => key,
        }
    }

    fn charged_bytes(&self) -> Result<usize> {
        match self {
            Self::Put { key, value } => key
                .len()
                .checked_add(value.len())
                .ok_or_else(limit_exceeded),
            Self::Delete { key } => Ok(key.len()),
        }
    }

    fn apply(self, transaction: &Transaction<'_, OptimisticTransactionDB>) -> Result<()> {
        match self {
            Self::Put { key, value } => transaction.put(&key, &value),
            Self::Delete { key } => transaction.delete(&key),
        }
        .map_err(map_operation_error)
    }
}

fn get<D: DBAccess>(snapshot: &SnapshotWithThreadMode<'_, D>, key: &[u8]) -> Result<Option<Bytes>> {
    snapshot
        .get(key)
        .map(|value| value.map(Bytes::from))
        .map_err(map_operation_error)
}

fn batch_get<D: DBAccess>(
    snapshot: &SnapshotWithThreadMode<'_, D>,
    prefix: &PhysicalPrefix,
    keys: Vec<Bytes>,
) -> Result<Vec<Option<Bytes>>> {
    let mut values = Vec::with_capacity(keys.len());
    for keys in keys.chunks(MAX_BATCH_POINT_READS) {
        let keys = keys
            .iter()
            .map(|key| prefix.encode_key(key))
            .collect::<Result<Vec<_>>>()?;
        for value in snapshot.multi_get(keys) {
            values.push(
                value
                    .map(|value| value.map(Bytes::from))
                    .map_err(map_operation_error)?,
            );
        }
    }
    Ok(values)
}

fn get_for_update(
    transaction: &Transaction<'_, OptimisticTransactionDB>,
    key: &[u8],
    read_options: &ReadOptions,
) -> Result<Option<Bytes>> {
    transaction
        .get_for_update_opt(key, true, read_options)
        .map(|value| value.map(Bytes::from))
        .map_err(map_operation_error)
}

fn scan<D: DBAccess>(
    snapshot: &SnapshotWithThreadMode<'_, D>,
    prefix: &PhysicalPrefix,
    range: &KeyRange,
    limits: ScanLimits,
) -> Result<ScanPage> {
    if limits.item_limit == 0 || limits.byte_limit == 0 {
        return Err(Error::new(ErrorKind::InvalidArgument));
    }
    if range.start() >= range.end() {
        return Ok(ScanPage::new(Vec::new(), None));
    }
    let start = prefix.encode_key(range.start())?;
    let end = prefix.encode_key(range.end())?;
    let byte_limit = limits.byte_limit.min(MAX_SCAN_PAGE_BYTES);
    let mut read_options = ReadOptions::default();
    read_options.set_iterate_lower_bound(start.to_vec());
    read_options.set_iterate_upper_bound(end.to_vec());
    read_options.set_total_order_seek(true);
    let mut iterator = snapshot.raw_iterator_opt(read_options);
    iterator.seek(&start);
    let mut items = Vec::new();
    let mut item_bytes = 0_usize;
    while let Some((physical_key, value)) = iterator.item() {
        let logical_key = prefix.decode_key(physical_key)?;
        let bytes = logical_key
            .len()
            .checked_add(value.len())
            .ok_or_else(limit_exceeded)?;
        if !items.is_empty()
            && (items.len() >= limits.item_limit
                || item_bytes.checked_add(bytes).ok_or_else(limit_exceeded)? > byte_limit)
        {
            return Ok(ScanPage::new(items, Some(logical_key)));
        }
        item_bytes = item_bytes.checked_add(bytes).ok_or_else(limit_exceeded)?;
        items.push(ScanItem::new(logical_key, Bytes::copy_from_slice(value)));
        iterator.next();
    }
    iterator.status().map_err(map_operation_error)?;
    Ok(ScanPage::new(items, None))
}

fn validate_value(value: &[u8]) -> Result<()> {
    if value.len() > ROCKSDB_MAX_VALUE_BYTES {
        return Err(limit_exceeded());
    }
    Ok(())
}

fn limit_exceeded() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

fn map_operation_error(error: RocksError) -> Error {
    let kind = map_operation_error_kind(error.kind());
    Error::with_source(kind, error)
}

/// Maps a native RocksDB error kind to a stable backend category for a
/// non-commit operation.
fn map_operation_error_kind(kind: RocksErrorKind) -> ErrorKind {
    match kind {
        RocksErrorKind::Corruption => ErrorKind::Corruption,
        RocksErrorKind::NotSupported => ErrorKind::Unsupported,
        RocksErrorKind::TimedOut
        | RocksErrorKind::Aborted
        | RocksErrorKind::Busy
        | RocksErrorKind::Expired
        | RocksErrorKind::TryAgain => ErrorKind::RetryableAbort,
        _ => ErrorKind::Backend,
    }
}

fn map_commit_error(error: RocksError) -> Error {
    let kind = map_commit_error_kind(error.kind());
    Error::with_source(kind, error)
}

/// Maps a native RocksDB error kind to a stable backend category for a commit.
fn map_commit_error_kind(kind: RocksErrorKind) -> ErrorKind {
    match kind {
        RocksErrorKind::IOError
        | RocksErrorKind::Incomplete
        | RocksErrorKind::ShutdownInProgress
        | RocksErrorKind::Unknown => ErrorKind::CommitOutcomeUnknown,
        _ => map_operation_error_kind(kind),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_prefix_is_backend_specific_length_delimited_and_order_preserving() {
        let a = PhysicalPrefix::new(&BackendNamespace::new("a").expect("namespace"));
        let aa = PhysicalPrefix::new(&BackendNamespace::new("aa").expect("namespace"));

        assert_ne!(a.bytes, aa.bytes);
        assert!(!aa.bytes.starts_with(&a.bytes));
        assert!(a.encode_key(b"a").expect("key") < a.encode_key(b"b").expect("key"));
        assert!(a.bytes.starts_with(PHYSICAL_PREFIX_HEADER));
        assert!(!a.bytes.starts_with(b"\0ktann\x01"));
    }

    #[test]
    fn hard_key_limit_accounts_for_the_complete_physical_prefix() {
        let prefix = PhysicalPrefix::new(&BackendNamespace::new("scope").expect("namespace"));
        assert_eq!(
            prefix.max_logical_key_bytes() + prefix.bytes.len(),
            ROCKSDB_MAX_PHYSICAL_KEY_BYTES,
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
    fn rocksdb_errors_map_to_stable_backend_categories() {
        assert_eq!(
            map_operation_error_kind(RocksErrorKind::Corruption),
            ErrorKind::Corruption,
        );
        assert_eq!(
            map_operation_error_kind(RocksErrorKind::NotSupported),
            ErrorKind::Unsupported,
        );
        assert_eq!(
            map_operation_error_kind(RocksErrorKind::Busy),
            ErrorKind::RetryableAbort,
        );
        assert_eq!(
            map_operation_error_kind(RocksErrorKind::TryAgain),
            ErrorKind::RetryableAbort,
        );
        assert_eq!(
            map_operation_error_kind(RocksErrorKind::NotFound),
            ErrorKind::Backend,
        );
        assert_eq!(
            map_commit_error_kind(RocksErrorKind::Busy),
            ErrorKind::RetryableAbort,
        );
        assert_eq!(
            map_commit_error_kind(RocksErrorKind::IOError),
            ErrorKind::CommitOutcomeUnknown,
        );
        assert_eq!(
            map_commit_error_kind(RocksErrorKind::Incomplete),
            ErrorKind::CommitOutcomeUnknown,
        );
        assert_eq!(
            map_commit_error_kind(RocksErrorKind::ShutdownInProgress),
            ErrorKind::CommitOutcomeUnknown,
        );
        assert_eq!(
            map_commit_error_kind(RocksErrorKind::Unknown),
            ErrorKind::CommitOutcomeUnknown,
        );
    }
}
