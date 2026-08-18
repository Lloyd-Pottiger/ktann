//! Typed logical operations over backend transactions and canonical codecs.

use std::collections::{BTreeMap, btree_map::Entry};
use std::fmt;

use bytes::Bytes;

use crate::api::{DataType, Error, ErrorKind, LogicalIndexId, MAX_FIELDS, Result, Value};

use super::backend::{
    AdmissionBudget, CommitStart, HardLimits, InsertOutcome, Mutation, ReadOps, ScanLimits,
    WriteTxn,
};
use super::keys::{self, KeyRange, LogicalKey, TreeKey};
use super::values::{
    IndexLifecycle, IndexManifest, OpaquePayload, PersistentValue, RecordLocation, ValueCodec,
    VectorRecord,
};

/// The ordered Tree Key field types of one Logical Index.
///
/// The schema is the authority for splitting an encoded Tree Key from its
/// trailing key components, so a range and its cursors carry it alongside the
/// index ID rather than trusting a caller to re-derive it from the same
/// Manifest. Ranges built under one schema cannot be reused under another.
#[derive(Clone, Copy, Eq, PartialEq)]
struct TreeKeySchema {
    types: [DataType; MAX_FIELDS],
    count: usize,
}

impl TreeKeySchema {
    /// The empty schema used by namespace-level bootstrap bindings.
    fn bootstrap() -> Self {
        Self {
            types: [DataType::Bool; MAX_FIELDS],
            count: 0,
        }
    }

    /// The Tree Key schema declared by one Index Manifest.
    fn for_index(manifest: &IndexManifest) -> Self {
        let (types, count) = manifest.tree_key_types();
        Self { types, count }
    }

    /// The ordered field types.
    fn types(&self) -> &[DataType] {
        &self.types[..self.count]
    }
}

/// A typed half-open range over one Logical Index.
///
/// Raw range bounds remain private so algorithm modules cannot compose logical
/// keys or resume a scan outside the range that produced its cursor. The range
/// binds both the Logical Index ID and the Tree Key schema required to decode
/// its keys.
#[derive(Clone, Eq, PartialEq)]
pub struct LogicalRange {
    index: LogicalIndexId,
    schema: TreeKeySchema,
    raw: KeyRange,
}

impl LogicalRange {
    /// Selects every key owned by one Logical Index.
    #[must_use]
    pub fn index(manifest: &IndexManifest) -> Self {
        Self {
            index: manifest.logical_index_id(),
            schema: TreeKeySchema::for_index(manifest),
            raw: keys::index_range(manifest.logical_index_id()),
        }
    }

    /// Selects every Tree Manifest in one Logical Index.
    #[must_use]
    pub fn tree_manifests(manifest: &IndexManifest) -> Self {
        Self {
            index: manifest.logical_index_id(),
            schema: TreeKeySchema::for_index(manifest),
            raw: keys::tree_manifest_range(manifest.logical_index_id()),
        }
    }

    /// Selects Tree Manifests matching leading Tree Key field values.
    pub fn tree_manifests_with_prefix(manifest: &IndexManifest, prefix: &[Value]) -> Result<Self> {
        let schema = TreeKeySchema::for_index(manifest);
        Ok(Self {
            index: manifest.logical_index_id(),
            schema,
            raw: keys::tree_manifest_prefix_range(
                manifest.logical_index_id(),
                schema.types(),
                prefix,
            )?,
        })
    }

    /// Selects Tree Manifests within one planned raw directory range.
    ///
    /// The Tree Key planner owns the bytes of `raw`; this constructor only
    /// binds them to the exact Logical Index ID and Tree Key schema needed to
    /// decode the returned keys.
    pub(crate) fn tree_manifest_plan(manifest: &IndexManifest, raw: KeyRange) -> Self {
        Self {
            index: manifest.logical_index_id(),
            schema: TreeKeySchema::for_index(manifest),
            raw,
        }
    }

    /// Selects every value owned by one partition.
    ///
    /// Returns `InvalidArgument` when `tree_key` does not match the Manifest's
    /// declared Tree Key schema.
    pub fn partition(
        manifest: &IndexManifest,
        tree_key: &TreeKey,
        partition: crate::api::PartitionKey,
    ) -> Result<Self> {
        let schema = TreeKeySchema::for_index(manifest);
        tree_key
            .validate(schema.types())
            .map_err(|_| Error::invalid_argument())?;
        Ok(Self {
            index: manifest.logical_index_id(),
            schema,
            raw: keys::partition_range(manifest.logical_index_id(), tree_key, partition),
        })
    }

    /// Selects every Leaf Entry owned by one partition.
    ///
    /// Returns `InvalidArgument` when `tree_key` does not match the Manifest's
    /// declared Tree Key schema.
    pub fn leaf_entries(
        manifest: &IndexManifest,
        tree_key: &TreeKey,
        partition: crate::api::PartitionKey,
    ) -> Result<Self> {
        let schema = TreeKeySchema::for_index(manifest);
        tree_key
            .validate(schema.types())
            .map_err(|_| Error::invalid_argument())?;
        Ok(Self {
            index: manifest.logical_index_id(),
            schema,
            raw: keys::leaf_entry_range(manifest.logical_index_id(), tree_key, partition),
        })
    }

    /// Selects every Child Entry owned by one partition.
    ///
    /// Returns `InvalidArgument` when `tree_key` does not match the Manifest's
    /// declared Tree Key schema.
    pub fn child_entries(
        manifest: &IndexManifest,
        tree_key: &TreeKey,
        partition: crate::api::PartitionKey,
    ) -> Result<Self> {
        let schema = TreeKeySchema::for_index(manifest);
        tree_key
            .validate(schema.types())
            .map_err(|_| Error::invalid_argument())?;
        Ok(Self {
            index: manifest.logical_index_id(),
            schema,
            raw: keys::child_entry_range(manifest.logical_index_id(), tree_key, partition),
        })
    }
}

impl fmt::Debug for LogicalRange {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LogicalRange")
            .field("index", &self.index)
            .field("bounds", &"[REDACTED]")
            .finish()
    }
}

/// An opaque continuation for one exact [`LogicalRange`].
#[derive(Clone, Eq, PartialEq)]
pub struct LogicalScanCursor {
    range: LogicalRange,
    next_start: Bytes,
}

impl fmt::Debug for LogicalScanCursor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LogicalScanCursor")
            .field("index", &self.range.index)
            .field("next_start", &"[REDACTED]")
            .finish()
    }
}

/// One decoded key/value item returned by a logical scan.
#[derive(Clone, PartialEq)]
pub struct LogicalScanItem {
    key: LogicalKey,
    value: PersistentValue,
}

impl LogicalScanItem {
    /// Returns the decoded Logical Key.
    #[must_use]
    pub const fn key(&self) -> &LogicalKey {
        &self.key
    }

    /// Returns the decoded persistent value.
    #[must_use]
    pub const fn value(&self) -> &PersistentValue {
        &self.value
    }
}

impl fmt::Debug for LogicalScanItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LogicalScanItem")
            .field("key", &self.key)
            .field("value_kind", &self.value.kind())
            .finish()
    }
}

/// One bounded decoded page from a logical scan.
#[derive(Clone, PartialEq)]
pub struct LogicalScanPage {
    items: Vec<LogicalScanItem>,
    next_cursor: Option<LogicalScanCursor>,
}

impl LogicalScanPage {
    /// Returns the ordered decoded items.
    #[must_use]
    pub fn items(&self) -> &[LogicalScanItem] {
        &self.items
    }

    /// Returns the continuation for the same range, if more data remains.
    #[must_use]
    pub const fn next_cursor(&self) -> Option<&LogicalScanCursor> {
        self.next_cursor.as_ref()
    }

    /// Consumes the page and returns its continuation, if any.
    #[must_use]
    pub fn into_next_cursor(self) -> Option<LogicalScanCursor> {
        self.next_cursor
    }
}

impl fmt::Debug for LogicalScanPage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LogicalScanPage")
            .field("items", &self.items.len())
            .field("has_next_cursor", &self.next_cursor.is_some())
            .finish()
    }
}

/// Codec state bound either to namespace bootstrap data or one Logical Index.
#[derive(Clone, Copy)]
struct LogicalBinding<'manifest> {
    manifest: Option<&'manifest IndexManifest>,
    schema: TreeKeySchema,
    allow_name_mapping: bool,
}

impl LogicalBinding<'static> {
    fn bootstrap() -> Self {
        Self {
            manifest: None,
            schema: TreeKeySchema::bootstrap(),
            allow_name_mapping: true,
        }
    }
}

impl<'manifest> LogicalBinding<'manifest> {
    fn for_index(manifest: &'manifest IndexManifest) -> Self {
        Self {
            manifest: Some(manifest),
            schema: TreeKeySchema::for_index(manifest),
            allow_name_mapping: false,
        }
    }

    /// A drop transition may atomically update the Index Name mapping and the
    /// complete index-owned range in one transaction, so it needs both the
    /// namespace name key and the bound index keyspace.
    fn for_drop(manifest: &'manifest IndexManifest) -> Self {
        Self {
            manifest: Some(manifest),
            schema: TreeKeySchema::for_index(manifest),
            allow_name_mapping: true,
        }
    }

    fn tree_key_types(&self) -> &[DataType] {
        self.schema.types()
    }

    fn codec(&self) -> ValueCodec<'manifest> {
        match self.manifest {
            Some(manifest) => ValueCodec::for_index(manifest),
            None => ValueCodec::bootstrap(),
        }
    }

    fn validate_input_key(&self, key: &LogicalKey) -> Result<()> {
        let valid = match (self.manifest, key.index()) {
            (None, None) => true,
            (None, Some(_)) => matches!(key, LogicalKey::Manifest(_)),
            (Some(manifest), Some(index)) => index == manifest.logical_index_id(),
            (Some(_), None) => {
                self.allow_name_mapping && matches!(key, LogicalKey::IndexNameDirectory(_))
            }
        };
        if !valid {
            return Err(Error::invalid_argument());
        }
        if let Some(tree_key) = key.tree_key() {
            self.validate_tree_key(tree_key)?;
        }
        Ok(())
    }

    fn validate_tree_key(&self, tree_key: &TreeKey) -> Result<()> {
        if self.manifest.is_none() {
            return Err(Error::invalid_argument());
        }
        tree_key
            .validate(self.schema.types())
            .map_err(|_| Error::invalid_argument())
    }

    fn validate_range(&self, range: &LogicalRange) -> Result<()> {
        match self.manifest {
            Some(manifest)
                if manifest.logical_index_id() == range.index && self.schema == range.schema =>
            {
                Ok(())
            }
            _ => Err(Error::invalid_argument()),
        }
    }

    fn compatible_with(&self, other: &LogicalBinding<'_>) -> bool {
        self.manifest == other.manifest && self.allow_name_mapping == other.allow_name_mapping
    }
}

fn encode_input_key(binding: &LogicalBinding<'_>, key: &LogicalKey) -> Result<Bytes> {
    binding.validate_input_key(key)?;
    keys::encode_key(key).map(Bytes::from)
}

fn decode_value(
    binding: &LogicalBinding<'_>,
    key: &LogicalKey,
    bytes: Option<Bytes>,
) -> Result<Option<PersistentValue>> {
    bytes
        .map(|bytes| binding.codec().decode(key, bytes))
        .transpose()
}

fn decode_batch(
    binding: &LogicalBinding<'_>,
    keys: &[LogicalKey],
    values: Vec<Option<Bytes>>,
) -> Result<Vec<Option<PersistentValue>>> {
    if keys.len() != values.len() {
        return Err(Error::new(ErrorKind::Backend));
    }
    keys.iter()
        .zip(values)
        .map(|(key, value)| decode_value(binding, key, value))
        .collect()
}

async fn scan_logical<T: ReadOps>(
    raw: &mut T,
    binding: &LogicalBinding<'_>,
    range: &LogicalRange,
    cursor: Option<&LogicalScanCursor>,
    limits: ScanLimits,
) -> Result<LogicalScanPage> {
    binding.validate_range(range)?;
    let raw_range = match cursor {
        Some(cursor) if cursor.range == *range => {
            KeyRange::new(cursor.next_start.to_vec(), range.raw.end().to_vec())
        }
        Some(_) => return Err(Error::invalid_argument()),
        None => range.raw.clone(),
    };
    let raw_page = raw.scan(&raw_range, limits).await?;
    let mut items = Vec::with_capacity(raw_page.items().len());
    let mut previous: Option<&Bytes> = None;
    for item in raw_page.items() {
        if item.key().as_ref() < raw_range.start()
            || item.key().as_ref() >= raw_range.end()
            || previous.is_some_and(|previous| previous >= item.key())
        {
            return Err(Error::new(ErrorKind::Backend));
        }
        let key = keys::decode_key(binding.tree_key_types(), item.key())?;
        let value = binding.codec().decode(&key, item.value().clone())?;
        items.push(LogicalScanItem { key, value });
        previous = Some(item.key());
    }

    let next_cursor = match raw_page.next_start() {
        Some(next_start)
            if !raw_page.items().is_empty()
                && raw_page
                    .items()
                    .last()
                    .is_some_and(|last| next_start > last.key())
                && next_start.as_ref() < range.raw.end() =>
        {
            Some(LogicalScanCursor {
                range: range.clone(),
                next_start: next_start.clone(),
            })
        }
        Some(_) => return Err(Error::new(ErrorKind::Backend)),
        None => None,
    };

    Ok(LogicalScanPage { items, next_cursor })
}

/// The closed read of one existing Vector Record group.
///
/// The group contains the Vector Record body and the authoritative Record
/// Location read from one transaction snapshot. The Opaque Payload is `None`
/// when the payload was not requested by the read operation; a requested but
/// absent payload is also `None`, and callers distinguish the two cases from
/// the request itself.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RecordGroupRead {
    record: VectorRecord,
    location: RecordLocation,
    payload: Option<OpaquePayload>,
}

impl RecordGroupRead {
    /// Returns the canonical Vector Record body.
    #[must_use]
    pub const fn record(&self) -> &VectorRecord {
        &self.record
    }

    /// Returns the authoritative Record Location.
    #[must_use]
    pub const fn location(&self) -> &RecordLocation {
        &self.location
    }

    /// Returns the Opaque Payload when it was requested and present.
    #[must_use]
    pub const fn payload(&self) -> Option<&OpaquePayload> {
        self.payload.as_ref()
    }

    /// Consumes the read and returns its owned parts.
    #[must_use]
    pub fn into_parts(self) -> (VectorRecord, RecordLocation, Option<OpaquePayload>) {
        (self.record, self.location, self.payload)
    }
}

/// A read transaction that exposes only typed logical operations.
pub struct ReadLogicalTxn<'manifest, T> {
    raw: T,
    binding: LogicalBinding<'manifest>,
}

impl<T> ReadLogicalTxn<'static, T> {
    /// Binds a read transaction to namespace values and Index Manifests.
    #[must_use]
    pub fn bootstrap(raw: T) -> Self {
        Self {
            raw,
            binding: LogicalBinding::bootstrap(),
        }
    }
}

impl<'manifest, T> ReadLogicalTxn<'manifest, T> {
    /// Binds a read transaction to one supported Index Manifest.
    pub fn for_index(raw: T, manifest: &'manifest IndexManifest) -> Result<Self> {
        Ok(Self {
            raw,
            binding: LogicalBinding::for_index(manifest),
        })
    }

    /// Unwraps the raw transaction so a manifest-validated read can rebind it.
    ///
    /// The raw transaction retains its snapshot. Manifest validation reads the
    /// Manifest with a bootstrap binding and then rebinds the same raw
    /// transaction to the validated Manifest, so the Record Group reads that
    /// follow share one consistent snapshot.
    pub(crate) fn into_raw(self) -> T {
        self.raw
    }
}

impl<T> fmt::Debug for ReadLogicalTxn<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReadLogicalTxn")
            .field(
                "index",
                &self.binding.manifest.map(IndexManifest::logical_index_id),
            )
            .finish_non_exhaustive()
    }
}

impl<'manifest, T> ReadLogicalTxn<'manifest, T> {
    /// Returns the Index Manifest this transaction is bound to, if any.
    pub(crate) fn bound_manifest(&self) -> Option<&'manifest IndexManifest> {
        self.binding.manifest
    }
}

impl<T: ReadOps> ReadLogicalTxn<'_, T> {
    /// Reads one typed value, returning `None` when the key is absent.
    pub async fn get(&mut self, key: LogicalKey) -> Result<Option<PersistentValue>> {
        let encoded = encode_input_key(&self.binding, &key)?;
        let value = self.raw.get(encoded).await?;
        decode_value(&self.binding, &key, value)
    }

    /// Reads typed values in input order while preserving duplicates and gaps.
    pub async fn batch_get(
        &mut self,
        keys: Vec<LogicalKey>,
    ) -> Result<Vec<Option<PersistentValue>>> {
        let encoded = keys
            .iter()
            .map(|key| encode_input_key(&self.binding, key))
            .collect::<Result<Vec<_>>>()?;
        let values = self.raw.batch_get(encoded).await?;
        decode_batch(&self.binding, &keys, values)
    }

    /// Reads one Vector Record group from this transaction's snapshot.
    ///
    /// Returns `None` when the Record ID is absent: neither the Vector Record
    /// nor its Record Location exists. A Record ID with only one side of the
    /// Record/Location pair present is [`ErrorKind::Corruption`], as is an
    /// Opaque Payload without its Vector Record.
    pub async fn read_record_group(
        &mut self,
        id: Bytes,
        include_payload: bool,
    ) -> Result<Option<RecordGroupRead>> {
        let mut groups = self.read_record_groups(vec![id], include_payload).await?;
        Ok(groups
            .pop()
            .expect("one input Record ID yields exactly one group read"))
    }

    /// Reads Vector Record groups in input order while preserving duplicates.
    ///
    /// The whole request reads from this transaction's one snapshot in a
    /// single bounded batch. Every input Record ID yields one result: `None`
    /// for a fully absent group, the validated group when the Vector Record
    /// and Record Location exist (plus the Opaque Payload when requested), or
    /// [`ErrorKind::Corruption`] for any partial group.
    pub async fn read_record_groups(
        &mut self,
        ids: Vec<Bytes>,
        include_payload: bool,
    ) -> Result<Vec<Option<RecordGroupRead>>> {
        let index = match self.binding.manifest {
            Some(manifest) => manifest.logical_index_id(),
            None => return Err(Error::invalid_argument()),
        };
        let keys_per_id = if include_payload { 3 } else { 2 };
        let key_count = ids
            .len()
            .checked_mul(keys_per_id)
            .ok_or_else(limit_exceeded)?;
        let mut keys = Vec::with_capacity(key_count);
        for id in &ids {
            keys.push(LogicalKey::Record {
                index,
                id: id.clone(),
            });
            keys.push(LogicalKey::Location {
                index,
                id: id.clone(),
            });
            if include_payload {
                keys.push(LogicalKey::Payload {
                    index,
                    id: id.clone(),
                });
            }
        }

        let encoded = keys
            .iter()
            .map(|key| encode_input_key(&self.binding, key))
            .collect::<Result<Vec<_>>>()?;
        // Guard the batch byte accounting with checked arithmetic; the backend
        // enforces its own hard key limits and batch ceiling on the raw call.
        encoded
            .iter()
            .try_fold(0_usize, |bytes, key| bytes.checked_add(key.len()))
            .ok_or_else(limit_exceeded)?;
        let values = self.raw.batch_get(encoded).await?;
        let values = decode_batch(&self.binding, &keys, values)?;

        let mut iter = values.into_iter();
        let mut groups = Vec::with_capacity(ids.len());
        for _ in &ids {
            let record = iter
                .next()
                .expect("the decoded batch has one entry per input key");
            let location = iter
                .next()
                .expect("the decoded batch has one entry per input key");
            let payload = if include_payload {
                iter.next()
                    .expect("the decoded batch has one entry per input key")
            } else {
                None
            };
            groups.push(match (record, location, payload) {
                (None, None, None) => None,
                (
                    Some(PersistentValue::VectorRecord(record)),
                    Some(PersistentValue::RecordLocation(location)),
                    payload,
                ) => {
                    let payload = match payload {
                        Some(PersistentValue::OpaquePayload(payload)) => Some(payload),
                        None => None,
                        Some(_) => return Err(corruption()),
                    };
                    Some(RecordGroupRead {
                        record,
                        location,
                        payload,
                    })
                }
                _ => return Err(corruption()),
            });
        }
        Ok(groups)
    }

    /// Scans one bounded typed page from an exact Logical Range.
    pub async fn scan(
        &mut self,
        range: &LogicalRange,
        cursor: Option<&LogicalScanCursor>,
        limits: ScanLimits,
    ) -> Result<LogicalScanPage> {
        scan_logical(&mut self.raw, &self.binding, range, cursor, limits).await
    }
}

/// Exact mutation work charged to one backend transaction.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransactionSize {
    mutations: usize,
    bytes: usize,
}

impl TransactionSize {
    /// Returns the number of point mutations.
    #[must_use]
    pub const fn mutations(self) -> usize {
        self.mutations
    }

    /// Returns total encoded key plus value bytes.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self.bytes
    }

    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            mutations: self.mutations.checked_add(other.mutations)?,
            bytes: self.bytes.checked_add(other.bytes)?,
        })
    }
}

/// A deterministic, bounded set of typed point mutations.
pub struct MutationBuilder<'manifest> {
    binding: LogicalBinding<'manifest>,
    hard_limits: HardLimits,
    budget: AdmissionBudget,
    entries: BTreeMap<Bytes, Option<Bytes>>,
    size: TransactionSize,
}

impl MutationBuilder<'static> {
    /// Creates a builder for namespace values and Index Manifests.
    #[must_use]
    pub fn bootstrap(hard_limits: HardLimits, budget: AdmissionBudget) -> Self {
        Self::new(LogicalBinding::bootstrap(), hard_limits, budget)
    }
}

impl<'manifest> MutationBuilder<'manifest> {
    /// Creates a builder bound to one supported Index Manifest.
    pub fn for_index(
        manifest: &'manifest IndexManifest,
        hard_limits: HardLimits,
        budget: AdmissionBudget,
    ) -> Result<Self> {
        Ok(Self::new(
            LogicalBinding::for_index(manifest),
            hard_limits,
            budget,
        ))
    }

    fn new(
        binding: LogicalBinding<'manifest>,
        hard_limits: HardLimits,
        budget: AdmissionBudget,
    ) -> Self {
        Self {
            binding,
            hard_limits,
            budget,
            entries: BTreeMap::new(),
            size: TransactionSize::default(),
        }
    }

    /// Returns the exact final size of this builder's point mutations.
    #[must_use]
    pub const fn size(&self) -> TransactionSize {
        self.size
    }

    /// Queues a typed put, superseding an earlier mutation to the same key.
    pub fn put(&mut self, key: LogicalKey, value: PersistentValue) -> Result<()> {
        let encoded_key = encode_input_key(&self.binding, &key)?;
        let encoded_value = Bytes::from(self.binding.codec().encode_for_key(&key, &value)?);
        self.queue(encoded_key, Some(encoded_value))
    }

    /// Queues a typed delete, superseding an earlier mutation to the same key.
    pub fn delete(&mut self, key: LogicalKey) -> Result<()> {
        let encoded_key = encode_input_key(&self.binding, &key)?;
        self.queue(encoded_key, None)
    }

    fn queue(&mut self, key: Bytes, value: Option<Bytes>) -> Result<()> {
        check_hard_limits(self.hard_limits, &key, value.as_ref())?;
        let new_bytes = mutation_bytes(&key, value.as_ref())?;
        let next = match self.entries.entry(key) {
            Entry::Occupied(mut entry) => {
                let old_bytes = mutation_bytes(entry.key(), entry.get().as_ref())?;
                let bytes = self
                    .size
                    .bytes
                    .checked_sub(old_bytes)
                    .and_then(|bytes| bytes.checked_add(new_bytes))
                    .ok_or_else(limit_exceeded)?;
                let next = TransactionSize {
                    mutations: self.size.mutations,
                    bytes,
                };
                check_budget(self.budget, next)?;
                entry.insert(value);
                next
            }
            Entry::Vacant(entry) => {
                let next = TransactionSize {
                    mutations: self
                        .size
                        .mutations
                        .checked_add(1)
                        .ok_or_else(limit_exceeded)?,
                    bytes: self
                        .size
                        .bytes
                        .checked_add(new_bytes)
                        .ok_or_else(limit_exceeded)?,
                };
                check_budget(self.budget, next)?;
                entry.insert(value);
                next
            }
        };
        self.size = next;
        Ok(())
    }

    fn into_backend_mutations(self) -> Vec<Mutation> {
        self.entries
            .into_iter()
            .map(|(key, value)| match value {
                Some(value) => Mutation::Put { key, value },
                None => Mutation::Delete { key },
            })
            .collect()
    }
}

impl fmt::Debug for MutationBuilder<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MutationBuilder")
            .field(
                "index",
                &self.binding.manifest.map(IndexManifest::logical_index_id),
            )
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

fn mutation_bytes(key: &Bytes, value: Option<&Bytes>) -> Result<usize> {
    key.len()
        .checked_add(value.map_or(0, Bytes::len))
        .ok_or_else(limit_exceeded)
}

fn check_hard_limits(limits: HardLimits, key: &Bytes, value: Option<&Bytes>) -> Result<()> {
    if key.len() > limits.max_key_bytes
        || value.is_some_and(|value| value.len() > limits.max_value_bytes)
    {
        Err(limit_exceeded())
    } else {
        Ok(())
    }
}

fn check_budget(budget: AdmissionBudget, size: TransactionSize) -> Result<()> {
    if size.mutations > budget.max_mutations || size.bytes > budget.max_mutation_bytes {
        Err(limit_exceeded())
    } else {
        Ok(())
    }
}

fn limit_exceeded() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

fn corruption() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// A write transaction that exposes typed logical reads and mutations.
pub struct WriteLogicalTxn<'manifest, T> {
    raw: T,
    binding: LogicalBinding<'manifest>,
    hard_limits: HardLimits,
    budget: AdmissionBudget,
    size: TransactionSize,
}

impl<T> WriteLogicalTxn<'static, T> {
    /// Binds a write transaction to namespace values and Index Manifests.
    #[must_use]
    pub fn bootstrap(raw: T, hard_limits: HardLimits, budget: AdmissionBudget) -> Self {
        Self {
            raw,
            binding: LogicalBinding::bootstrap(),
            hard_limits,
            budget,
            size: TransactionSize::default(),
        }
    }
}

impl<'manifest, T> WriteLogicalTxn<'manifest, T> {
    /// Binds a write transaction to one supported Index Manifest.
    pub fn for_index(
        raw: T,
        manifest: &'manifest IndexManifest,
        hard_limits: HardLimits,
        budget: AdmissionBudget,
    ) -> Result<Self> {
        Ok(Self {
            raw,
            binding: LogicalBinding::for_index(manifest),
            hard_limits,
            budget,
            size: TransactionSize::default(),
        })
    }

    /// Binds a write transaction to one Dropping Manifest transition.
    ///
    /// Drop completion must atomically remove the Index Name mapping and the
    /// index-owned range. This binding accepts the exact bound index keyspace
    /// plus the Index Name directory key, while rejecting the allocator and
    /// every other namespace key.
    pub fn for_drop(
        raw: T,
        manifest: &'manifest IndexManifest,
        hard_limits: HardLimits,
        budget: AdmissionBudget,
    ) -> Result<Self> {
        if manifest.lifecycle() != IndexLifecycle::Dropping {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            raw,
            binding: LogicalBinding::for_drop(manifest),
            hard_limits,
            budget,
            size: TransactionSize::default(),
        })
    }

    /// Returns the exact mutation work already applied to this transaction.
    #[must_use]
    pub const fn size(&self) -> TransactionSize {
        self.size
    }

    /// Returns the Index Manifest this transaction is bound to, if any.
    pub(crate) fn bound_manifest(&self) -> Option<&'manifest IndexManifest> {
        self.binding.manifest
    }

    /// Returns the conservative admission budget bound to this transaction.
    #[must_use]
    pub(crate) const fn admission_budget(&self) -> AdmissionBudget {
        self.budget
    }

    /// Unwraps the raw transaction so a lifecycle transition can rebind it.
    ///
    /// The raw transaction retains its snapshot, update-protected read set,
    /// and any pending mutations. Lifecycle drop uses this only before any
    /// mutation is queued, so the replacement typed binding starts with an
    /// exact zero mutation charge.
    pub(crate) fn into_raw(self) -> T {
        self.raw
    }

    /// Creates a builder bounded by this transaction's remaining budget.
    #[must_use]
    pub fn mutations(&self) -> MutationBuilder<'manifest> {
        let remaining = AdmissionBudget {
            max_mutations: self
                .budget
                .max_mutations
                .saturating_sub(self.size.mutations),
            max_mutation_bytes: self
                .budget
                .max_mutation_bytes
                .saturating_sub(self.size.bytes),
        };
        MutationBuilder::new(self.binding, self.hard_limits, remaining)
    }
}

impl<T> fmt::Debug for WriteLogicalTxn<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WriteLogicalTxn")
            .field(
                "index",
                &self.binding.manifest.map(IndexManifest::logical_index_id),
            )
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl<T: WriteTxn> WriteLogicalTxn<'_, T> {
    /// Reads one typed value from the transaction snapshot and pending writes.
    pub async fn get(&mut self, key: LogicalKey) -> Result<Option<PersistentValue>> {
        let encoded = encode_input_key(&self.binding, &key)?;
        let value = self.raw.get(encoded).await?;
        decode_value(&self.binding, &key, value)
    }

    /// Reads typed values in input order from the transaction snapshot.
    pub async fn batch_get(
        &mut self,
        keys: Vec<LogicalKey>,
    ) -> Result<Vec<Option<PersistentValue>>> {
        let encoded = keys
            .iter()
            .map(|key| encode_input_key(&self.binding, key))
            .collect::<Result<Vec<_>>>()?;
        let values = self.raw.batch_get(encoded).await?;
        decode_batch(&self.binding, &keys, values)
    }

    /// Reads one typed value and establishes a point conflict on its key.
    pub async fn get_for_update(&mut self, key: LogicalKey) -> Result<Option<PersistentValue>> {
        let encoded = encode_input_key(&self.binding, &key)?;
        let value = self.raw.get_for_update(encoded).await?;
        decode_value(&self.binding, &key, value)
    }

    /// Reads typed values and establishes point conflicts in input order.
    pub async fn batch_get_for_update(
        &mut self,
        keys: Vec<LogicalKey>,
    ) -> Result<Vec<Option<PersistentValue>>> {
        let encoded = keys
            .iter()
            .map(|key| encode_input_key(&self.binding, key))
            .collect::<Result<Vec<_>>>()?;
        let values = self.raw.batch_get_for_update(encoded).await?;
        decode_batch(&self.binding, &keys, values)
    }

    /// Scans one bounded typed page from this transaction's snapshot.
    pub async fn scan(
        &mut self,
        range: &LogicalRange,
        cursor: Option<&LogicalScanCursor>,
        limits: ScanLimits,
    ) -> Result<LogicalScanPage> {
        scan_logical(&mut self.raw, &self.binding, range, cursor, limits).await
    }

    /// Applies one typed put and charges its exact encoded size.
    pub async fn put(&mut self, key: LogicalKey, value: PersistentValue) -> Result<()> {
        let mut mutations = self.mutations();
        mutations.put(key, value)?;
        self.apply(mutations).await
    }

    /// Applies one typed delete and charges its exact encoded size.
    pub async fn delete(&mut self, key: LogicalKey) -> Result<()> {
        let mut mutations = self.mutations();
        mutations.delete(key)?;
        self.apply(mutations).await
    }

    /// Inserts a typed value only when its key is absent.
    pub async fn insert(
        &mut self,
        key: LogicalKey,
        value: PersistentValue,
    ) -> Result<InsertOutcome> {
        let encoded_key = encode_input_key(&self.binding, &key)?;
        let encoded_value = Bytes::from(self.binding.codec().encode_for_key(&key, &value)?);
        check_hard_limits(self.hard_limits, &encoded_key, Some(&encoded_value))?;
        let mutation_size = TransactionSize {
            mutations: 1,
            bytes: mutation_bytes(&encoded_key, Some(&encoded_value))?,
        };

        // An update-protected read both establishes a conflict on the target key
        // and decodes the existing value fail-closed, so a duplicate insert is
        // always validated and conflict-safe regardless of the remaining budget.
        if let Some(existing) = self.raw.get_for_update(encoded_key.clone()).await? {
            self.binding.codec().decode(&key, existing)?;
            return Ok(InsertOutcome::AlreadyExists);
        }

        let next = self
            .size
            .checked_add(mutation_size)
            .ok_or_else(limit_exceeded)?;
        check_budget(self.budget, next)?;
        let outcome = self.raw.insert(encoded_key, encoded_value).await?;
        if outcome == InsertOutcome::Inserted {
            self.size = next;
        }
        Ok(outcome)
    }

    /// Applies a validated builder in canonical encoded-key order.
    pub async fn apply(&mut self, builder: MutationBuilder<'_>) -> Result<()> {
        if !self.binding.compatible_with(&builder.binding) {
            return Err(Error::invalid_argument());
        }
        for (key, value) in &builder.entries {
            check_hard_limits(self.hard_limits, key, value.as_ref())?;
        }
        let next = self
            .size
            .checked_add(builder.size)
            .ok_or_else(limit_exceeded)?;
        check_budget(self.budget, next)?;
        self.raw
            .batch_mutate(builder.into_backend_mutations())
            .await?;
        self.size = next;
        Ok(())
    }

    /// Clears one typed range when the backend supports transactional clear.
    pub async fn clear_range(&mut self, range: &LogicalRange) -> Result<()> {
        self.binding.validate_range(range)?;
        let bytes = range
            .raw
            .start()
            .len()
            .checked_add(range.raw.end().len())
            .ok_or_else(limit_exceeded)?;
        let next = self
            .size
            .checked_add(TransactionSize {
                mutations: 1,
                bytes,
            })
            .ok_or_else(limit_exceeded)?;
        check_budget(self.budget, next)?;
        self.raw.clear_range(&range.raw).await?;
        self.size = next;
        Ok(())
    }

    /// Commits every read and mutation atomically, consuming the transaction.
    pub async fn commit(self) -> Result<()> {
        self.raw.commit().await
    }

    /// Commits after coordinating the Runtime's native commit boundary.
    ///
    /// The first commit attempt of one foreground operation uses the
    /// cancellation-guarded boundary; later attempts in a resumable lifecycle
    /// operation commit without re-arming the already-consumed boundary.
    pub async fn commit_with(self, start: CommitStart) -> Result<()> {
        self.raw.commit_with(start).await
    }

    /// Abandons every mutation, consuming the transaction.
    pub async fn rollback(self) {
        self.raw.rollback().await;
    }
}

#[cfg(test)]
mod tests {
    use crate::api::IndexName;

    use super::*;

    #[test]
    fn mutation_builder_emits_canonical_key_order() {
        let mut builder = MutationBuilder::bootstrap(
            HardLimits {
                max_key_bytes: 1_024,
                max_value_bytes: 1_024,
            },
            AdmissionBudget {
                max_mutations: 2,
                max_mutation_bytes: 2_048,
            },
        );
        builder
            .delete(LogicalKey::IndexNameDirectory(
                IndexName::new("z").expect("valid name"),
            ))
            .expect("queue z");
        builder
            .delete(LogicalKey::IndexNameDirectory(
                IndexName::new("a").expect("valid name"),
            ))
            .expect("queue a");

        let keys = builder
            .into_backend_mutations()
            .into_iter()
            .map(|mutation| match mutation {
                Mutation::Put { key, .. } | Mutation::Delete { key } => key,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            keys,
            vec![
                Bytes::from(keys::name_directory_key(
                    &IndexName::new("a").expect("valid name")
                )),
                Bytes::from(keys::name_directory_key(
                    &IndexName::new("z").expect("valid name")
                )),
            ]
        );
    }
}
