//! Canonical codecs for persistent logical values.
//!
//! Every value starts with a one-byte type tag and a one-byte codec version.
//! Integer and floating-point payloads use big-endian bytes; lengths use
//! fixed-width unsigned integers. Strings are unnormalized UTF-8. Finite
//! floating-point zero has exactly one representation: positive zero.
//!
//! The Index Manifest additionally declares the whole-format and value-codec
//! versions. Unsupported versions in bootstrap values or a Manifest are
//! [`ErrorKind::UnsupportedFormat`]. Once a supported Manifest has been opened,
//! an unknown tag, codec version, discriminant, malformed length, noncanonical
//! scalar, inconsistent identity, or trailing byte in an index-owned value is
//! [`ErrorKind::Corruption`].
//!
//! [`ValueCodec::bootstrap`] handles namespace values and Index Manifests.
//! [`ValueCodec::for_index`] binds all remaining codecs to the Manifest's exact
//! dimension, schema, Tree Key definition, and Bloom parameters. The module
//! never adds or interprets a backend physical prefix.
//!
//! # Version 1 layout
//!
//! `u16`, `u32`, `u64`, `i64`, `f32`, and `f64` below are fixed-width
//! big-endian values. `bytes32` is exactly 32 bytes. `sizedN<T>` is an unsigned
//! `N`-bit element count followed by that many `T` values; bare `sized<T>` uses
//! a `u32` count. Every row starts with `[tag: u8][codec: u8 = 1]`:
//!
//! ```text
//! 00 IndexIdAllocator   u64 high_water
//! 01 IndexNameEntry     u64 logical_index_id
//! 02 IndexManifest      u16 format, u8 declared_codec, u8 lifecycle,
//!                       u64 id, u32 dimension, u8 metric,
//!                       sized16<FieldSchema>, sized16<u16 tree_field>,
//!                       u32 min_entries, u32 max_entries, bytes32 seed
//! 03 TreeManifest       u64 root, u64 partition_key_high_water
//! 04 VectorRecord       sized16<u8 record_id>, sized<VectorComponent>,
//!                       sized16<TypedValue>
//! 05 OpaquePayload      sized<u8>
//! 06 RecordLocation     sized<u8 tree_key>, u64 leaf
//! 07 PartitionHeader    u32 level, u32 count, u64 epoch, u8 state
//! 08 PartitionCentroid  sized<VectorComponent>
//! 09 ChildEntry         u64 child, sized<VectorComponent>
//! 0a LeafEntry          sized16<u8 record_id>, sized16<TypedValue>,
//!                       sized<u8 RaBitQ7>
//! 0b PartitionSynopsis  sized16<FieldSynopsis>
//! 0c PartitionState     u8 state, u64 started_at, state-specific u64 keys
//! ```
//!
//! A vector is prefixed by its redundant `u32` dimension. A typed value uses
//! tag `0=NULL`, `1=Bool`, `2=I64`, `3=F64`, or `4=String`; Bool has one
//! canonical byte and String is `sized<u8>`. A Field Schema stores its
//! `sized8<u8>` name, type, nullable byte, synopsis tag, configured Bloom input,
//! and exact Bloom parameters. A Field Synopsis stores canonical NULL/non-NULL
//! flags, optional typed extrema, and the schema-governed fixed-size Bloom byte
//! string.

use std::cmp::Ordering;
use std::fmt;

use bytes::Bytes;

use crate::api::{
    DataType, Error, ErrorKind, FieldId, FieldSchema, IndexConfig, LogicalIndexId,
    MAX_ENCODED_SYNOPSIS_BYTES, MAX_FIELDS, MAX_STRING_BYTES, Metric, PartitionKey, Result,
    SynopsisConfig, Value,
};

use super::keys::{LogicalKey, MAX_RECORD_ID_BYTES, MAX_TREE_KEY_BYTES, TreeKey};

/// The whole persistent format version emitted and accepted by this build.
pub const FORMAT_VERSION: u16 = 1;

/// The logical value codec version emitted and accepted by this build.
pub const VALUE_CODEC_VERSION: u8 = 1;

/// The maximum encoded Opaque Payload size.
pub const MAX_PAYLOAD_BYTES: usize = crate::api::MAX_PAYLOAD_BYTES;

/// The maximum encoded Partition Synopsis size.
pub const MAX_SYNOPSIS_BYTES: usize = MAX_ENCODED_SYNOPSIS_BYTES;

const MAX_VALUE_BYTES: usize = 1024 * 1024;
const ROTATION_SEED_BYTES: usize = 32;

const TAG_INDEX_ID_ALLOCATOR: u8 = 0x00;
const TAG_INDEX_NAME_ENTRY: u8 = 0x01;
const TAG_INDEX_MANIFEST: u8 = 0x02;
const TAG_TREE_MANIFEST: u8 = 0x03;
const TAG_VECTOR_RECORD: u8 = 0x04;
const TAG_OPAQUE_PAYLOAD: u8 = 0x05;
const TAG_RECORD_LOCATION: u8 = 0x06;
const TAG_PARTITION_HEADER: u8 = 0x07;
const TAG_PARTITION_CENTROID: u8 = 0x08;
const TAG_CHILD_ENTRY: u8 = 0x09;
const TAG_LEAF_ENTRY: u8 = 0x0a;
const TAG_PARTITION_SYNOPSIS: u8 = 0x0b;
const TAG_PARTITION_STATE: u8 = 0x0c;

/// The persistent value family expected at a logical key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ValueKind {
    /// The namespace Logical Index ID allocator.
    IndexIdAllocator,
    /// An Index Name to Logical Index ID mapping.
    IndexNameEntry,
    /// An Index Manifest.
    IndexManifest,
    /// A Tree Manifest.
    TreeManifest,
    /// A Vector Record body.
    VectorRecord,
    /// An Opaque Payload.
    OpaquePayload,
    /// A Record Location.
    RecordLocation,
    /// A Partition Header.
    PartitionHeader,
    /// A partition's immutable centroid.
    PartitionCentroid,
    /// A Child Entry envelope.
    ChildEntry,
    /// A Leaf Entry envelope.
    LeafEntry,
    /// A Partition Synopsis.
    PartitionSynopsis,
    /// A persistent Partition State.
    PartitionState,
}

impl ValueKind {
    const fn tag(self) -> u8 {
        match self {
            Self::IndexIdAllocator => TAG_INDEX_ID_ALLOCATOR,
            Self::IndexNameEntry => TAG_INDEX_NAME_ENTRY,
            Self::IndexManifest => TAG_INDEX_MANIFEST,
            Self::TreeManifest => TAG_TREE_MANIFEST,
            Self::VectorRecord => TAG_VECTOR_RECORD,
            Self::OpaquePayload => TAG_OPAQUE_PAYLOAD,
            Self::RecordLocation => TAG_RECORD_LOCATION,
            Self::PartitionHeader => TAG_PARTITION_HEADER,
            Self::PartitionCentroid => TAG_PARTITION_CENTROID,
            Self::ChildEntry => TAG_CHILD_ENTRY,
            Self::LeafEntry => TAG_LEAF_ENTRY,
            Self::PartitionSynopsis => TAG_PARTITION_SYNOPSIS,
            Self::PartitionState => TAG_PARTITION_STATE,
        }
    }
}

/// The lifecycle state persisted in an Index Manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexLifecycle {
    /// The Logical Index accepts ordinary operations.
    Active,
    /// The Logical Index is being deleted.
    Dropping,
}

/// Exact format parameters for one persisted Bloom synopsis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BloomParameters {
    bit_count: u32,
    hash_count: u8,
}

impl BloomParameters {
    /// Creates bounded, nonzero Bloom parameters.
    pub fn new(bit_count: u32, hash_count: u8) -> Result<Self> {
        let byte_count = usize::try_from(bit_count)
            .ok()
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(Error::invalid_argument)?;
        if bit_count == 0 || hash_count == 0 || byte_count > MAX_SYNOPSIS_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            bit_count,
            hash_count,
        })
    }

    /// Returns the exact number of persisted bits.
    #[must_use]
    pub const fn bit_count(self) -> u32 {
        self.bit_count
    }

    /// Returns the exact number of hash probes.
    #[must_use]
    pub const fn hash_count(self) -> u8 {
        self.hash_count
    }

    fn byte_count(self) -> usize {
        usize::try_from(self.bit_count)
            .expect("u32 fits usize on supported targets")
            .div_ceil(8)
    }
}

/// The authoritative persistent metadata for one Logical Index.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexManifest {
    lifecycle: IndexLifecycle,
    logical_index_id: LogicalIndexId,
    config: IndexConfig,
    rotation_seed: [u8; ROTATION_SEED_BYTES],
    bloom_parameters: Box<[Option<BloomParameters>]>,
}

impl IndexManifest {
    /// Creates a supported version-1 Index Manifest.
    pub fn new(
        lifecycle: IndexLifecycle,
        logical_index_id: LogicalIndexId,
        config: IndexConfig,
        rotation_seed: [u8; ROTATION_SEED_BYTES],
        bloom_parameters: Vec<Option<BloomParameters>>,
    ) -> Result<Self> {
        let manifest = Self {
            lifecycle,
            logical_index_id,
            config,
            rotation_seed,
            bloom_parameters: bloom_parameters.into_boxed_slice(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Returns the whole persistent format version.
    #[must_use]
    pub const fn format_version(&self) -> u16 {
        FORMAT_VERSION
    }

    /// Returns the logical value codec version.
    #[must_use]
    pub const fn value_codec_version(&self) -> u8 {
        VALUE_CODEC_VERSION
    }

    /// Returns the persistent lifecycle state.
    #[must_use]
    pub const fn lifecycle(&self) -> IndexLifecycle {
        self.lifecycle
    }

    /// Returns the owned Logical Index ID.
    #[must_use]
    pub const fn logical_index_id(&self) -> LogicalIndexId {
        self.logical_index_id
    }

    /// Returns the immutable Logical Index configuration.
    #[must_use]
    pub const fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Returns the persistent rotation seed.
    #[must_use]
    pub const fn rotation_seed(&self) -> &[u8; ROTATION_SEED_BYTES] {
        &self.rotation_seed
    }

    /// Returns exact Bloom parameters aligned with the field schema.
    #[must_use]
    pub fn bloom_parameters(&self) -> &[Option<BloomParameters>] {
        &self.bloom_parameters
    }

    fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.bloom_parameters.len() != self.config.fields().len() {
            return Err(Error::invalid_argument());
        }
        let mut maximum_synopsis_size = 2_usize + 2;
        for (field, parameters) in self.config.fields().iter().zip(&self.bloom_parameters) {
            match (field.synopsis(), parameters) {
                (SynopsisConfig::MinMax, None) => {}
                (SynopsisConfig::MinMaxBloom { .. }, Some(parameters))
                    if parameters.byte_count() <= MAX_SYNOPSIS_BYTES => {}
                _ => return Err(Error::invalid_argument()),
            }
            let encoded_extrema = match field.data_type() {
                DataType::Bool => 2 * 2,
                DataType::I64 | DataType::F64 => 2 * 9,
                DataType::String => 2 * (1 + 4 + MAX_STRING_BYTES),
            };
            maximum_synopsis_size = maximum_synopsis_size
                .checked_add(1 + encoded_extrema)
                .and_then(|size| {
                    parameters.map_or(Some(size), |parameters| {
                        size.checked_add(4 + parameters.byte_count())
                    })
                })
                .ok_or_else(Error::invalid_argument)?;
        }
        if maximum_synopsis_size > MAX_SYNOPSIS_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}

/// A namespace allocator high-water mark; zero means no ID has been issued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexIdAllocator {
    high_water: u64,
}

impl IndexIdAllocator {
    /// Creates an allocator value.
    #[must_use]
    pub const fn new(high_water: u64) -> Self {
        Self { high_water }
    }

    /// Returns the greatest Logical Index ID ever reserved.
    #[must_use]
    pub const fn high_water(self) -> u64 {
        self.high_water
    }
}

/// An Index Name directory mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexNameEntry {
    logical_index_id: LogicalIndexId,
}

impl IndexNameEntry {
    /// Creates a directory mapping.
    #[must_use]
    pub const fn new(logical_index_id: LogicalIndexId) -> Self {
        Self { logical_index_id }
    }

    /// Returns the mapped Logical Index ID.
    #[must_use]
    pub const fn logical_index_id(self) -> LogicalIndexId {
        self.logical_index_id
    }
}

/// The directory and Partition Key allocator state for one Tree Key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TreeManifest {
    root: PartitionKey,
    partition_key_high_water: PartitionKey,
}

impl TreeManifest {
    /// Creates a Tree Manifest whose stable root is Partition Key 1.
    pub fn new(root: PartitionKey, partition_key_high_water: PartitionKey) -> Result<Self> {
        if root.get() != 1 || partition_key_high_water < root {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            root,
            partition_key_high_water,
        })
    }

    /// Returns the stable root Partition Key.
    #[must_use]
    pub const fn root(self) -> PartitionKey {
        self.root
    }

    /// Returns the greatest Partition Key reserved for the tree.
    #[must_use]
    pub const fn partition_key_high_water(self) -> PartitionKey {
        self.partition_key_high_water
    }
}

/// A persistent Vector Record body, excluding its separately stored payload.
#[derive(Clone, PartialEq)]
pub struct VectorRecord {
    record_id: Bytes,
    vector: Box<[f32]>,
    fields: Box<[Value]>,
}

impl VectorRecord {
    /// Creates a Vector Record body.
    #[must_use]
    pub fn new(
        record_id: Bytes,
        vector: impl Into<Box<[f32]>>,
        fields: impl Into<Box<[Value]>>,
    ) -> Self {
        Self {
            record_id,
            vector: vector.into(),
            fields: fields.into(),
        }
    }

    /// Returns the Record ID duplicated from the logical key.
    #[must_use]
    pub const fn record_id(&self) -> &Bytes {
        &self.record_id
    }

    /// Returns the original vector.
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// Returns the positional typed fields.
    #[must_use]
    pub fn fields(&self) -> &[Value] {
        &self.fields
    }
}

impl fmt::Debug for VectorRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("VectorRecord([REDACTED])")
    }
}

/// A separately stored Opaque Payload, including an existing empty payload.
#[derive(Clone, Eq, PartialEq)]
pub struct OpaquePayload(Bytes);

impl OpaquePayload {
    /// Creates a bounded Opaque Payload.
    pub fn new(bytes: Bytes) -> Result<Self> {
        if bytes.len() > MAX_PAYLOAD_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(Self(bytes))
    }

    /// Returns the opaque bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &Bytes {
        &self.0
    }
}

impl fmt::Debug for OpaquePayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OpaquePayload([REDACTED])")
    }
}

/// The authoritative tree and leaf membership of a Vector Record.
#[derive(Clone, Eq, PartialEq)]
pub struct RecordLocation {
    tree_key: TreeKey,
    leaf: PartitionKey,
}

impl RecordLocation {
    /// Creates a Record Location.
    #[must_use]
    pub const fn new(tree_key: TreeKey, leaf: PartitionKey) -> Self {
        Self { tree_key, leaf }
    }

    /// Returns the canonical Tree Key.
    #[must_use]
    pub const fn tree_key(&self) -> &TreeKey {
        &self.tree_key
    }

    /// Returns the authoritative Leaf Partition.
    #[must_use]
    pub const fn leaf(&self) -> PartitionKey {
        self.leaf
    }
}

impl fmt::Debug for RecordLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RecordLocation")
            .field("tree_key", &"[REDACTED]")
            .field("leaf", &self.leaf)
            .finish()
    }
}

/// The state discriminator duplicated in a Partition Header for traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PartitionState {
    /// The partition accepts its ordinary operations.
    Ready,
    /// Split target identities have been reserved.
    Splitting,
    /// A published split target is receiving entries.
    ReceivingSplit,
    /// The source is draining into two published targets.
    DrainingSplit,
    /// The source is draining into reselected Ready targets.
    Merging,
}

/// Small mutable operational metadata for one partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartitionHeader {
    level: u32,
    entry_count: u32,
    cache_epoch: u64,
    state: PartitionState,
}

impl PartitionHeader {
    /// Creates a structurally valid Partition Header.
    pub fn new(
        level: u32,
        entry_count: u32,
        cache_epoch: u64,
        state: PartitionState,
    ) -> Result<Self> {
        if level == 0 {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            level,
            entry_count,
            cache_epoch,
            state,
        })
    }

    /// Returns the tree level; leaves are level one.
    #[must_use]
    pub const fn level(self) -> u32 {
        self.level
    }

    /// Returns the exact number of entries.
    #[must_use]
    pub const fn entry_count(self) -> u32 {
        self.entry_count
    }

    /// Returns the persistent cache-validation epoch.
    #[must_use]
    pub const fn cache_epoch(self) -> u64 {
        self.cache_epoch
    }

    /// Returns the traversal state discriminator.
    #[must_use]
    pub const fn state(self) -> PartitionState {
        self.state
    }
}

/// A full-f32 immutable routing centroid.
#[derive(Clone, PartialEq)]
pub struct PartitionCentroid(Box<[f32]>);

impl PartitionCentroid {
    /// Creates an immutable centroid.
    #[must_use]
    pub fn new(components: impl Into<Box<[f32]>>) -> Self {
        Self(components.into())
    }

    /// Returns the centroid components.
    #[must_use]
    pub fn components(&self) -> &[f32] {
        &self.0
    }
}

impl fmt::Debug for PartitionCentroid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PartitionCentroid([REDACTED])")
    }
}

/// An internal-partition entry and its immutable routing projection.
#[derive(Clone, PartialEq)]
pub struct ChildEntry {
    child: PartitionKey,
    centroid: Box<[f32]>,
}

impl ChildEntry {
    /// Creates a Child Entry envelope.
    #[must_use]
    pub fn new(child: PartitionKey, centroid: impl Into<Box<[f32]>>) -> Self {
        Self {
            child,
            centroid: centroid.into(),
        }
    }

    /// Returns the child identity duplicated from the logical key.
    #[must_use]
    pub const fn child(&self) -> PartitionKey {
        self.child
    }

    /// Returns the immutable child centroid projection.
    #[must_use]
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }
}

impl fmt::Debug for ChildEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChildEntry")
            .field("child", &self.child)
            .field("centroid", &"[REDACTED]")
            .finish()
    }
}

/// A leaf-partition entry envelope.
///
/// The RaBitQ7 bytes have their exact dimension-derived length checked here;
/// their bit-level canonicality is owned by the RaBitQ7 module.
#[derive(Clone, PartialEq)]
pub struct LeafEntry {
    record_id: Bytes,
    fields: Box<[Value]>,
    rabitq7: Bytes,
}

impl LeafEntry {
    /// Creates a Leaf Entry envelope.
    #[must_use]
    pub fn new(record_id: Bytes, fields: impl Into<Box<[Value]>>, rabitq7: Bytes) -> Self {
        Self {
            record_id,
            fields: fields.into(),
            rabitq7,
        }
    }

    /// Returns the Record ID duplicated from the logical key.
    #[must_use]
    pub const fn record_id(&self) -> &Bytes {
        &self.record_id
    }

    /// Returns the exact typed filter projection.
    #[must_use]
    pub fn fields(&self) -> &[Value] {
        &self.fields
    }

    /// Returns the absolute RaBitQ7 bytes.
    #[must_use]
    pub const fn rabitq7(&self) -> &Bytes {
        &self.rabitq7
    }
}

impl fmt::Debug for LeafEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LeafEntry([REDACTED])")
    }
}

/// One field's conservative synopsis state.
#[derive(Clone, PartialEq)]
pub struct FieldSynopsis {
    has_null: bool,
    minimum: Option<Value>,
    maximum: Option<Value>,
    bloom: Option<Bytes>,
}

impl FieldSynopsis {
    /// Creates a field synopsis.
    #[must_use]
    pub fn new(
        has_null: bool,
        minimum: Option<Value>,
        maximum: Option<Value>,
        bloom: Option<Bytes>,
    ) -> Self {
        Self {
            has_null,
            minimum,
            maximum,
            bloom,
        }
    }

    /// Returns whether the synopsis has observed NULL.
    #[must_use]
    pub const fn has_null(&self) -> bool {
        self.has_null
    }

    /// Returns the minimum observed non-NULL value.
    #[must_use]
    pub const fn minimum(&self) -> Option<&Value> {
        self.minimum.as_ref()
    }

    /// Returns the maximum observed non-NULL value.
    #[must_use]
    pub const fn maximum(&self) -> Option<&Value> {
        self.maximum.as_ref()
    }

    /// Returns the optional fixed-size Bloom bytes.
    #[must_use]
    pub const fn bloom(&self) -> Option<&Bytes> {
        self.bloom.as_ref()
    }
}

impl fmt::Debug for FieldSynopsis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("FieldSynopsis([REDACTED])")
    }
}

/// Conservative synopses aligned with the complete Vector Record schema.
#[derive(Clone, PartialEq)]
pub struct PartitionSynopsis(Box<[FieldSynopsis]>);

impl PartitionSynopsis {
    /// Creates a Partition Synopsis envelope.
    #[must_use]
    pub fn new(fields: impl Into<Box<[FieldSynopsis]>>) -> Self {
        Self(fields.into())
    }

    /// Returns field synopses in schema order.
    #[must_use]
    pub fn fields(&self) -> &[FieldSynopsis] {
        &self.0
    }
}

impl fmt::Debug for PartitionSynopsis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PartitionSynopsis([REDACTED])")
    }
}

/// The durable topology state and references for one partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PartitionTransition {
    /// The partition accepts ordinary operations.
    Ready {
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
    /// A source has reserved two target identities.
    Splitting {
        /// The left target.
        left: PartitionKey,
        /// The right target.
        right: PartitionKey,
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
    /// A published target is receiving entries from one source.
    ReceivingSplit {
        /// The source partition.
        source: PartitionKey,
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
    /// A source is draining into two published targets.
    DrainingSplit {
        /// The left target.
        left: PartitionKey,
        /// The right target.
        right: PartitionKey,
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
    /// A source is draining into targets reselected per batch.
    Merging {
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
}

impl PartitionTransition {
    /// Returns the Header discriminator corresponding to this state.
    #[must_use]
    pub const fn state(self) -> PartitionState {
        match self {
            Self::Ready { .. } => PartitionState::Ready,
            Self::Splitting { .. } => PartitionState::Splitting,
            Self::ReceivingSplit { .. } => PartitionState::ReceivingSplit,
            Self::DrainingSplit { .. } => PartitionState::DrainingSplit,
            Self::Merging { .. } => PartitionState::Merging,
        }
    }

    /// Returns milliseconds since the Unix epoch when this state began.
    #[must_use]
    pub const fn started_at_unix_millis(self) -> u64 {
        match self {
            Self::Ready {
                started_at_unix_millis,
            }
            | Self::Splitting {
                started_at_unix_millis,
                ..
            }
            | Self::ReceivingSplit {
                started_at_unix_millis,
                ..
            }
            | Self::DrainingSplit {
                started_at_unix_millis,
                ..
            }
            | Self::Merging {
                started_at_unix_millis,
            } => started_at_unix_millis,
        }
    }
}

/// Any versioned logical value owned by the storage module.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PersistentValue {
    /// The namespace Logical Index ID allocator.
    IndexIdAllocator(IndexIdAllocator),
    /// An Index Name directory mapping.
    IndexNameEntry(IndexNameEntry),
    /// An Index Manifest.
    IndexManifest(IndexManifest),
    /// A Tree Manifest.
    TreeManifest(TreeManifest),
    /// A Vector Record body.
    VectorRecord(VectorRecord),
    /// An Opaque Payload.
    OpaquePayload(OpaquePayload),
    /// A Record Location.
    RecordLocation(RecordLocation),
    /// A Partition Header.
    PartitionHeader(PartitionHeader),
    /// A partition centroid.
    PartitionCentroid(PartitionCentroid),
    /// A Child Entry envelope.
    ChildEntry(ChildEntry),
    /// A Leaf Entry envelope.
    LeafEntry(LeafEntry),
    /// A Partition Synopsis.
    PartitionSynopsis(PartitionSynopsis),
    /// A persistent Partition State.
    PartitionState(PartitionTransition),
}

impl PersistentValue {
    /// Returns this value's persistent family.
    #[must_use]
    pub const fn kind(&self) -> ValueKind {
        match self {
            Self::IndexIdAllocator(_) => ValueKind::IndexIdAllocator,
            Self::IndexNameEntry(_) => ValueKind::IndexNameEntry,
            Self::IndexManifest(_) => ValueKind::IndexManifest,
            Self::TreeManifest(_) => ValueKind::TreeManifest,
            Self::VectorRecord(_) => ValueKind::VectorRecord,
            Self::OpaquePayload(_) => ValueKind::OpaquePayload,
            Self::RecordLocation(_) => ValueKind::RecordLocation,
            Self::PartitionHeader(_) => ValueKind::PartitionHeader,
            Self::PartitionCentroid(_) => ValueKind::PartitionCentroid,
            Self::ChildEntry(_) => ValueKind::ChildEntry,
            Self::LeafEntry(_) => ValueKind::LeafEntry,
            Self::PartitionSynopsis(_) => ValueKind::PartitionSynopsis,
            Self::PartitionState(_) => ValueKind::PartitionState,
        }
    }
}

/// A canonical value codec, optionally bound to one supported Index Manifest.
#[derive(Clone, Copy, Debug)]
pub struct ValueCodec<'a> {
    manifest: Option<&'a IndexManifest>,
}

impl ValueCodec<'static> {
    /// Creates the bootstrap codec for namespace values and Index Manifests.
    #[must_use]
    pub const fn bootstrap() -> Self {
        Self { manifest: None }
    }
}

impl<'a> ValueCodec<'a> {
    /// Creates a codec bound to a supported Index Manifest.
    #[must_use]
    pub const fn for_index(manifest: &'a IndexManifest) -> Self {
        Self {
            manifest: Some(manifest),
        }
    }

    /// Encodes one typed value to deterministic canonical bytes.
    ///
    /// Index-owned values other than the Index Manifest require a codec made
    /// with [`ValueCodec::for_index`].
    pub fn encode(self, value: &PersistentValue) -> Result<Vec<u8>> {
        let mut encoder = Encoder::new(value.kind());
        match value {
            PersistentValue::IndexIdAllocator(value) => encoder.u64(value.high_water),
            PersistentValue::IndexNameEntry(value) => {
                encoder.u64(value.logical_index_id.get());
            }
            PersistentValue::IndexManifest(value) => encode_index_manifest(&mut encoder, value)?,
            PersistentValue::TreeManifest(value) => {
                self.require_manifest()?;
                encoder.u64(value.root.get());
                encoder.u64(value.partition_key_high_water.get());
            }
            PersistentValue::VectorRecord(value) => {
                encode_vector_record(&mut encoder, self.require_manifest()?, value)?;
            }
            PersistentValue::OpaquePayload(value) => {
                self.require_manifest()?;
                encoder.sized_bytes(value.as_bytes(), MAX_PAYLOAD_BYTES)?;
            }
            PersistentValue::RecordLocation(value) => {
                let manifest = self.require_manifest()?;
                validate_tree_key(manifest, value.tree_key())?;
                encoder.sized_bytes(value.tree_key().as_bytes(), MAX_TREE_KEY_BYTES)?;
                encoder.u64(value.leaf.get());
            }
            PersistentValue::PartitionHeader(value) => {
                self.require_manifest()?;
                encoder.u32(value.level);
                encoder.u32(value.entry_count);
                encoder.u64(value.cache_epoch);
                encoder.u8(encode_state_kind(value.state));
            }
            PersistentValue::PartitionCentroid(value) => {
                encode_vector(
                    &mut encoder,
                    self.require_manifest()?.config().dimension(),
                    value.components(),
                )?;
            }
            PersistentValue::ChildEntry(value) => {
                encoder.u64(value.child.get());
                encode_vector(
                    &mut encoder,
                    self.require_manifest()?.config().dimension(),
                    value.centroid(),
                )?;
            }
            PersistentValue::LeafEntry(value) => {
                encode_leaf_entry(&mut encoder, self.require_manifest()?, value)?;
            }
            PersistentValue::PartitionSynopsis(value) => {
                encode_partition_synopsis(&mut encoder, self.require_manifest()?, value)?;
            }
            PersistentValue::PartitionState(value) => {
                self.require_manifest()?;
                encode_partition_state(&mut encoder, *value)?;
            }
        }
        encoder.finish()
    }

    /// Decodes owned backend bytes and rejects every noncanonical byte.
    ///
    /// Index-owned values other than the Index Manifest require a codec made
    /// with [`ValueCodec::for_index`]. Owned byte fields retain zero-copy slices
    /// of `bytes`.
    pub fn decode(self, key: &LogicalKey, bytes: Bytes) -> Result<PersistentValue> {
        let expected = value_kind_for_key(key);
        if expected == ValueKind::PartitionSynopsis && bytes.len() > MAX_SYNOPSIS_BYTES {
            return Err(corrupt());
        }
        let mut decoder = Decoder::framed(expected, bytes)?;
        let value = match expected {
            ValueKind::IndexIdAllocator => {
                PersistentValue::IndexIdAllocator(IndexIdAllocator::new(decoder.u64()?))
            }
            ValueKind::IndexNameEntry => PersistentValue::IndexNameEntry(IndexNameEntry::new(
                decode_logical_index_id(&mut decoder)?,
            )),
            ValueKind::IndexManifest => {
                PersistentValue::IndexManifest(decode_index_manifest(&mut decoder)?)
            }
            ValueKind::TreeManifest => {
                self.require_manifest()?;
                let root = decode_partition_key(&mut decoder)?;
                let high_water = decode_partition_key(&mut decoder)?;
                PersistentValue::TreeManifest(
                    TreeManifest::new(root, high_water).map_err(|_| corrupt())?,
                )
            }
            ValueKind::VectorRecord => PersistentValue::VectorRecord(decode_vector_record(
                &mut decoder,
                self.require_manifest()?,
            )?),
            ValueKind::OpaquePayload => {
                self.require_manifest()?;
                let payload = decoder.sized_bytes(MAX_PAYLOAD_BYTES)?;
                PersistentValue::OpaquePayload(OpaquePayload::new(payload).map_err(|_| corrupt())?)
            }
            ValueKind::RecordLocation => {
                let manifest = self.require_manifest()?;
                let bytes = decoder.sized_bytes(MAX_TREE_KEY_BYTES)?;
                let (types, type_count) = tree_key_types(manifest);
                let tree_key = TreeKey::from_encoded(&types[..type_count], &bytes)?;
                let leaf = decode_partition_key(&mut decoder)?;
                PersistentValue::RecordLocation(RecordLocation::new(tree_key, leaf))
            }
            ValueKind::PartitionHeader => {
                self.require_manifest()?;
                let level = decoder.u32()?;
                let entry_count = decoder.u32()?;
                let cache_epoch = decoder.u64()?;
                let state = decode_state_kind(decoder.u8()?)?;
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(level, entry_count, cache_epoch, state)
                        .map_err(|_| corrupt())?,
                )
            }
            ValueKind::PartitionCentroid => {
                let dimension = self.require_manifest()?.config().dimension();
                PersistentValue::PartitionCentroid(PartitionCentroid::new(decode_vector(
                    &mut decoder,
                    dimension,
                )?))
            }
            ValueKind::ChildEntry => {
                let child = decode_partition_key(&mut decoder)?;
                let dimension = self.require_manifest()?.config().dimension();
                PersistentValue::ChildEntry(ChildEntry::new(
                    child,
                    decode_vector(&mut decoder, dimension)?,
                ))
            }
            ValueKind::LeafEntry => PersistentValue::LeafEntry(decode_leaf_entry(
                &mut decoder,
                self.require_manifest()?,
            )?),
            ValueKind::PartitionSynopsis => PersistentValue::PartitionSynopsis(
                decode_partition_synopsis(&mut decoder, self.require_manifest()?)?,
            ),
            ValueKind::PartitionState => {
                self.require_manifest()?;
                PersistentValue::PartitionState(decode_partition_state(&mut decoder)?)
            }
        };
        decoder.finish()?;
        validate_key_identity(key, &value, self.manifest)?;
        Ok(value)
    }

    fn require_manifest(self) -> Result<&'a IndexManifest> {
        self.manifest.ok_or_else(Error::invalid_argument)
    }
}

fn encode_index_manifest(encoder: &mut Encoder, manifest: &IndexManifest) -> Result<()> {
    encoder.u16(FORMAT_VERSION);
    encoder.u8(VALUE_CODEC_VERSION);
    encoder.u8(match manifest.lifecycle {
        IndexLifecycle::Active => 0,
        IndexLifecycle::Dropping => 1,
    });
    encoder.u64(manifest.logical_index_id.get());
    let config = manifest.config();
    encoder.u32(u32::try_from(config.dimension()).map_err(|_| Error::invalid_argument())?);
    encoder.u8(encode_metric(config.metric()));
    encoder.u16(u16::try_from(config.fields().len()).map_err(|_| Error::invalid_argument())?);
    for (field, bloom) in config.fields().iter().zip(manifest.bloom_parameters()) {
        encoder.sized_u8_bytes(field.name().as_bytes())?;
        encoder.u8(encode_data_type(field.data_type()));
        encoder.bool(field.is_nullable());
        match field.synopsis() {
            SynopsisConfig::MinMax => encoder.u8(0),
            SynopsisConfig::MinMaxBloom {
                expected_distinct,
                false_positive_rate,
            } => {
                let parameters = bloom.ok_or_else(Error::invalid_argument)?;
                encoder.u8(1);
                encoder.u32(expected_distinct.get());
                encoder.f64(*false_positive_rate)?;
                encoder.u32(parameters.bit_count);
                encoder.u8(parameters.hash_count);
            }
        }
    }
    encoder
        .u16(u16::try_from(config.tree_key_fields().len()).map_err(|_| Error::invalid_argument())?);
    for field in config.tree_key_fields() {
        encoder.u16(field.0);
    }
    encoder.u32(config.min_partition_entries());
    encoder.u32(config.max_partition_entries());
    encoder.bytes(&manifest.rotation_seed);
    Ok(())
}

fn decode_index_manifest(decoder: &mut Decoder) -> Result<IndexManifest> {
    let format_version = decoder.u16()?;
    let declared_codec_version = decoder.u8()?;
    if format_version != FORMAT_VERSION || declared_codec_version != VALUE_CODEC_VERSION {
        return Err(unsupported());
    }
    let lifecycle = match decoder.u8()? {
        0 => IndexLifecycle::Active,
        1 => IndexLifecycle::Dropping,
        _ => return Err(corrupt()),
    };
    let logical_index_id = decode_logical_index_id(decoder)?;
    let dimension = usize::try_from(decoder.u32()?).map_err(|_| corrupt())?;
    let metric = decode_metric(decoder.u8()?)?;
    let field_count = usize::from(decoder.u16()?);
    if field_count > MAX_FIELDS {
        return Err(corrupt());
    }
    let mut fields = Vec::with_capacity(field_count);
    let mut bloom_parameters = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        let name_bytes = decoder.sized_u8_bytes()?;
        let name = std::str::from_utf8(&name_bytes).map_err(|_| corrupt())?;
        let data_type = decode_data_type(decoder.u8()?)?;
        let nullable = decoder.bool()?;
        let synopsis_tag = decoder.u8()?;
        let (synopsis, bloom) = match synopsis_tag {
            0 => (SynopsisConfig::MinMax, None),
            1 => {
                let expected_distinct =
                    std::num::NonZeroU32::new(decoder.u32()?).ok_or_else(corrupt)?;
                let false_positive_rate = decoder.canonical_f64()?;
                let bit_count = decoder.u32()?;
                let hash_count = decoder.u8()?;
                let bloom = BloomParameters::new(bit_count, hash_count).map_err(|_| corrupt())?;
                (
                    SynopsisConfig::MinMaxBloom {
                        expected_distinct,
                        false_positive_rate,
                    },
                    Some(bloom),
                )
            }
            _ => return Err(corrupt()),
        };
        let mut field = FieldSchema::new(name, data_type).map_err(|_| corrupt())?;
        if nullable {
            field = field.nullable();
        }
        field = field.with_synopsis(synopsis).map_err(|_| corrupt())?;
        fields.push(field);
        bloom_parameters.push(bloom);
    }
    let tree_key_count = usize::from(decoder.u16()?);
    if tree_key_count > field_count {
        return Err(corrupt());
    }
    let mut tree_key_fields = Vec::with_capacity(tree_key_count);
    for _ in 0..tree_key_count {
        tree_key_fields.push(FieldId(decoder.u16()?));
    }
    let minimum = decoder.u32()?;
    let maximum = decoder.u32()?;
    let rotation_seed = decoder.array::<ROTATION_SEED_BYTES>()?;
    let config = IndexConfig::new(dimension, metric)
        .and_then(|config| config.with_fields(fields))
        .and_then(|config| config.with_tree_key_fields(tree_key_fields))
        .and_then(|config| config.with_partition_entries(minimum, maximum))
        .and_then(|config| {
            config.validate()?;
            Ok(config)
        })
        .map_err(|_| corrupt())?;
    IndexManifest::new(
        lifecycle,
        logical_index_id,
        config,
        rotation_seed,
        bloom_parameters,
    )
    .map_err(|_| corrupt())
}

fn encode_vector_record(
    encoder: &mut Encoder,
    manifest: &IndexManifest,
    record: &VectorRecord,
) -> Result<()> {
    if record.record_id.is_empty() || record.record_id.len() > MAX_RECORD_ID_BYTES {
        return Err(Error::invalid_argument());
    }
    encoder.sized_u16_bytes(&record.record_id)?;
    encode_vector(encoder, manifest.config().dimension(), record.vector())?;
    encode_fields(encoder, manifest.config().fields(), record.fields())
}

fn decode_vector_record(decoder: &mut Decoder, manifest: &IndexManifest) -> Result<VectorRecord> {
    let record_id = decoder.sized_u16_bytes(MAX_RECORD_ID_BYTES)?;
    if record_id.is_empty() {
        return Err(corrupt());
    }
    let vector = decode_vector(decoder, manifest.config().dimension())?;
    let fields = decode_fields(decoder, manifest.config().fields())?;
    Ok(VectorRecord::new(record_id, vector, fields))
}

fn encode_leaf_entry(
    encoder: &mut Encoder,
    manifest: &IndexManifest,
    entry: &LeafEntry,
) -> Result<()> {
    if entry.record_id.is_empty() || entry.record_id.len() > MAX_RECORD_ID_BYTES {
        return Err(Error::invalid_argument());
    }
    encoder.sized_u16_bytes(&entry.record_id)?;
    encode_fields(encoder, manifest.config().fields(), &entry.fields)?;
    let expected = rabitq7_encoded_len(manifest.config().dimension())?;
    if entry.rabitq7.len() != expected {
        return Err(Error::invalid_argument());
    }
    encoder.sized_bytes(&entry.rabitq7, expected)
}

fn decode_leaf_entry(decoder: &mut Decoder, manifest: &IndexManifest) -> Result<LeafEntry> {
    let record_id = decoder.sized_u16_bytes(MAX_RECORD_ID_BYTES)?;
    if record_id.is_empty() {
        return Err(corrupt());
    }
    let fields = decode_fields(decoder, manifest.config().fields())?;
    let expected = rabitq7_encoded_len(manifest.config().dimension()).map_err(|_| corrupt())?;
    let rabitq7 = decoder.sized_bytes(expected)?;
    if rabitq7.len() != expected {
        return Err(corrupt());
    }
    Ok(LeafEntry::new(record_id, fields, rabitq7))
}

fn encode_partition_synopsis(
    encoder: &mut Encoder,
    manifest: &IndexManifest,
    synopsis: &PartitionSynopsis,
) -> Result<()> {
    let schema = manifest.config().fields();
    if synopsis.fields().len() != schema.len() {
        return Err(Error::invalid_argument());
    }
    encoder.u16(u16::try_from(schema.len()).map_err(|_| Error::invalid_argument())?);
    for ((field_synopsis, field), bloom_parameters) in synopsis
        .fields()
        .iter()
        .zip(schema)
        .zip(manifest.bloom_parameters())
    {
        validate_field_synopsis(field_synopsis, field, *bloom_parameters)?;
        let has_non_null = field_synopsis.minimum.is_some();
        encoder.u8(u8::from(field_synopsis.has_null) | (u8::from(has_non_null) << 1));
        if let (Some(minimum), Some(maximum)) = (&field_synopsis.minimum, &field_synopsis.maximum) {
            encode_typed_value(encoder, field.data_type(), field.is_nullable(), minimum)?;
            encode_typed_value(encoder, field.data_type(), field.is_nullable(), maximum)?;
        }
        if let Some(parameters) = bloom_parameters {
            let bloom = field_synopsis
                .bloom
                .as_ref()
                .ok_or_else(Error::invalid_argument)?;
            encoder.sized_bytes(bloom, parameters.byte_count())?;
        }
    }
    if encoder.len() > MAX_SYNOPSIS_BYTES {
        return Err(Error::invalid_argument());
    }
    Ok(())
}

fn decode_partition_synopsis(
    decoder: &mut Decoder,
    manifest: &IndexManifest,
) -> Result<PartitionSynopsis> {
    let schema = manifest.config().fields();
    if usize::from(decoder.u16()?) != schema.len() {
        return Err(corrupt());
    }
    let mut fields = Vec::with_capacity(schema.len());
    for (field, bloom_parameters) in schema.iter().zip(manifest.bloom_parameters()) {
        let flags = decoder.u8()?;
        if flags & !0b11 != 0 {
            return Err(corrupt());
        }
        let has_null = flags & 1 != 0;
        let has_non_null = flags & 2 != 0;
        let (minimum, maximum) = if has_non_null {
            (
                Some(decode_typed_value(decoder, field.data_type(), false)?),
                Some(decode_typed_value(decoder, field.data_type(), false)?),
            )
        } else {
            (None, None)
        };
        let bloom = if let Some(parameters) = bloom_parameters {
            let bloom = decoder.sized_bytes(parameters.byte_count())?;
            Some(bloom)
        } else {
            None
        };
        let synopsis = FieldSynopsis::new(has_null, minimum, maximum, bloom);
        validate_field_synopsis(&synopsis, field, *bloom_parameters).map_err(|_| corrupt())?;
        fields.push(synopsis);
    }
    Ok(PartitionSynopsis::new(fields))
}

fn encode_partition_state(encoder: &mut Encoder, transition: PartitionTransition) -> Result<()> {
    validate_transition(transition)?;
    encoder.u8(encode_state_kind(transition.state()));
    encoder.u64(transition.started_at_unix_millis());
    match transition {
        PartitionTransition::Ready { .. } | PartitionTransition::Merging { .. } => {}
        PartitionTransition::Splitting { left, right, .. }
        | PartitionTransition::DrainingSplit { left, right, .. } => {
            encoder.u64(left.get());
            encoder.u64(right.get());
        }
        PartitionTransition::ReceivingSplit { source, .. } => encoder.u64(source.get()),
    }
    Ok(())
}

fn decode_partition_state(decoder: &mut Decoder) -> Result<PartitionTransition> {
    let kind = decode_state_kind(decoder.u8()?)?;
    let started_at_unix_millis = decoder.u64()?;
    let transition = match kind {
        PartitionState::Ready => PartitionTransition::Ready {
            started_at_unix_millis,
        },
        PartitionState::Splitting => PartitionTransition::Splitting {
            left: decode_partition_key(decoder)?,
            right: decode_partition_key(decoder)?,
            started_at_unix_millis,
        },
        PartitionState::ReceivingSplit => PartitionTransition::ReceivingSplit {
            source: decode_partition_key(decoder)?,
            started_at_unix_millis,
        },
        PartitionState::DrainingSplit => PartitionTransition::DrainingSplit {
            left: decode_partition_key(decoder)?,
            right: decode_partition_key(decoder)?,
            started_at_unix_millis,
        },
        PartitionState::Merging => PartitionTransition::Merging {
            started_at_unix_millis,
        },
    };
    validate_transition(transition).map_err(|_| corrupt())?;
    Ok(transition)
}

fn validate_transition(transition: PartitionTransition) -> Result<()> {
    match transition {
        PartitionTransition::Splitting { left, right, .. }
        | PartitionTransition::DrainingSplit { left, right, .. }
            if left == right =>
        {
            Err(Error::invalid_argument())
        }
        _ => Ok(()),
    }
}

fn encode_vector(encoder: &mut Encoder, dimension: usize, vector: &[f32]) -> Result<()> {
    if vector.len() != dimension {
        return Err(Error::invalid_argument());
    }
    let encoded_bytes = dimension
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or_else(Error::invalid_argument)?;
    encoder.bytes.reserve(encoded_bytes);
    encoder.u32(u32::try_from(dimension).map_err(|_| Error::invalid_argument())?);
    for component in vector {
        encoder.f32(*component)?;
    }
    Ok(())
}

fn decode_vector(decoder: &mut Decoder, dimension: usize) -> Result<Box<[f32]>> {
    if usize::try_from(decoder.u32()?).map_err(|_| corrupt())? != dimension {
        return Err(corrupt());
    }
    let byte_count = dimension.checked_mul(4).ok_or_else(corrupt)?;
    if decoder.remaining() < byte_count {
        return Err(corrupt());
    }
    let mut vector = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        vector.push(decoder.canonical_f32()?);
    }
    Ok(vector.into_boxed_slice())
}

fn encode_fields(encoder: &mut Encoder, schema: &[FieldSchema], fields: &[Value]) -> Result<()> {
    if fields.len() != schema.len() {
        return Err(Error::invalid_argument());
    }
    encoder.u16(u16::try_from(fields.len()).map_err(|_| Error::invalid_argument())?);
    for (value, field) in fields.iter().zip(schema) {
        encode_typed_value(encoder, field.data_type(), field.is_nullable(), value)?;
    }
    Ok(())
}

fn decode_fields(decoder: &mut Decoder, schema: &[FieldSchema]) -> Result<Box<[Value]>> {
    if usize::from(decoder.u16()?) != schema.len() {
        return Err(corrupt());
    }
    let mut fields = Vec::with_capacity(schema.len());
    for field in schema {
        fields.push(decode_typed_value(
            decoder,
            field.data_type(),
            field.is_nullable(),
        )?);
    }
    Ok(fields.into_boxed_slice())
}

fn encode_typed_value(
    encoder: &mut Encoder,
    data_type: DataType,
    nullable: bool,
    value: &Value,
) -> Result<()> {
    match value {
        Value::Null if nullable => encoder.u8(0),
        Value::Bool(value) if data_type == DataType::Bool => {
            encoder.u8(1);
            encoder.bool(*value);
        }
        Value::I64(value) if data_type == DataType::I64 => {
            encoder.u8(2);
            encoder.i64(*value);
        }
        Value::F64(value) if data_type == DataType::F64 => {
            encoder.u8(3);
            encoder.f64(*value)?;
        }
        Value::String(value) if data_type == DataType::String => {
            if value.len() > MAX_STRING_BYTES {
                return Err(Error::invalid_argument());
            }
            encoder.u8(4);
            encoder.sized_bytes(value.as_bytes(), MAX_STRING_BYTES)?;
        }
        _ => return Err(Error::invalid_argument()),
    }
    Ok(())
}

fn decode_typed_value(decoder: &mut Decoder, data_type: DataType, nullable: bool) -> Result<Value> {
    match decoder.u8()? {
        0 if nullable => Ok(Value::Null),
        1 if data_type == DataType::Bool => Ok(Value::Bool(decoder.bool()?)),
        2 if data_type == DataType::I64 => Ok(Value::I64(decoder.i64()?)),
        3 if data_type == DataType::F64 => {
            Value::f64(decoder.canonical_f64()?).map_err(|_| corrupt())
        }
        4 if data_type == DataType::String => {
            let bytes = decoder.sized_bytes(MAX_STRING_BYTES)?;
            let value = std::str::from_utf8(&bytes).map_err(|_| corrupt())?;
            Value::string(value).map_err(|_| corrupt())
        }
        _ => Err(corrupt()),
    }
}

fn validate_field_synopsis(
    synopsis: &FieldSynopsis,
    field: &FieldSchema,
    bloom_parameters: Option<BloomParameters>,
) -> Result<()> {
    if synopsis.has_null && !field.is_nullable() {
        return Err(Error::invalid_argument());
    }
    match (&synopsis.minimum, &synopsis.maximum) {
        (None, None) => {}
        (Some(minimum), Some(maximum)) => {
            validate_non_null_value(minimum, field.data_type())?;
            validate_non_null_value(maximum, field.data_type())?;
            if compare_values(minimum, maximum)? == Ordering::Greater {
                return Err(Error::invalid_argument());
            }
        }
        _ => return Err(Error::invalid_argument()),
    }
    match (bloom_parameters, &synopsis.bloom) {
        (None, None) => {}
        (Some(parameters), Some(bytes)) if bytes.len() == parameters.byte_count() => {
            validate_bloom_padding(bytes, parameters).map_err(|_| Error::invalid_argument())?;
            if synopsis.minimum.is_none() && bytes.iter().any(|byte| *byte != 0) {
                return Err(Error::invalid_argument());
            }
            if synopsis.minimum.is_some() && bytes.iter().all(|byte| *byte == 0) {
                return Err(Error::invalid_argument());
            }
        }
        _ => return Err(Error::invalid_argument()),
    }
    Ok(())
}

fn validate_non_null_value(value: &Value, data_type: DataType) -> Result<()> {
    match (data_type, value) {
        (DataType::Bool, Value::Bool(_))
        | (DataType::I64, Value::I64(_))
        | (DataType::String, Value::String(_)) => {}
        (DataType::F64, Value::F64(value)) if value.is_finite() => {}
        _ => return Err(Error::invalid_argument()),
    }
    if let Value::String(value) = value {
        if value.len() > MAX_STRING_BYTES {
            return Err(Error::invalid_argument());
        }
    }
    Ok(())
}

fn compare_values(left: &Value, right: &Value) -> Result<Ordering> {
    match (left, right) {
        (Value::Bool(left), Value::Bool(right)) => Ok(left.cmp(right)),
        (Value::I64(left), Value::I64(right)) => Ok(left.cmp(right)),
        (Value::F64(left), Value::F64(right)) if left.is_finite() && right.is_finite() => {
            left.partial_cmp(right).ok_or_else(Error::invalid_argument)
        }
        (Value::String(left), Value::String(right)) => Ok(left.cmp(right)),
        _ => Err(Error::invalid_argument()),
    }
}

fn validate_bloom_padding(bytes: &[u8], parameters: BloomParameters) -> Result<()> {
    if bytes.len() != parameters.byte_count() {
        return Err(corrupt());
    }
    let used_bits = parameters.bit_count % 8;
    if used_bits != 0 {
        let mask = !((1_u8 << used_bits) - 1);
        if bytes.last().is_some_and(|byte| byte & mask != 0) {
            return Err(corrupt());
        }
    }
    Ok(())
}

fn validate_tree_key(manifest: &IndexManifest, tree_key: &TreeKey) -> Result<()> {
    let (types, type_count) = tree_key_types(manifest);
    tree_key
        .values(&types[..type_count])
        .map(|_| ())
        .map_err(|_| Error::invalid_argument())
}

fn tree_key_types(manifest: &IndexManifest) -> ([DataType; MAX_FIELDS], usize) {
    let mut types = [DataType::Bool; MAX_FIELDS];
    let field_ids = manifest.config().tree_key_fields();
    for (target, field_id) in types.iter_mut().zip(field_ids) {
        *target = manifest.config().fields()[usize::from(field_id.0)].data_type();
    }
    (types, field_ids.len())
}

fn rabitq7_encoded_len(dimension: usize) -> Result<usize> {
    let sign_bytes = dimension.checked_add(7).map(|value| value / 8);
    let magnitude_bytes = dimension
        .checked_mul(6)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8);
    sign_bytes
        .zip(magnitude_bytes)
        .and_then(|(signs, magnitudes)| 12_usize.checked_add(signs)?.checked_add(magnitudes))
        .ok_or_else(Error::invalid_argument)
}

const fn encode_metric(metric: Metric) -> u8 {
    match metric {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::InnerProduct => 2,
    }
}

fn decode_metric(tag: u8) -> Result<Metric> {
    match tag {
        0 => Ok(Metric::L2),
        1 => Ok(Metric::Cosine),
        2 => Ok(Metric::InnerProduct),
        _ => Err(corrupt()),
    }
}

const fn encode_data_type(data_type: DataType) -> u8 {
    match data_type {
        DataType::Bool => 0,
        DataType::I64 => 1,
        DataType::F64 => 2,
        DataType::String => 3,
    }
}

fn decode_data_type(tag: u8) -> Result<DataType> {
    match tag {
        0 => Ok(DataType::Bool),
        1 => Ok(DataType::I64),
        2 => Ok(DataType::F64),
        3 => Ok(DataType::String),
        _ => Err(corrupt()),
    }
}

const fn encode_state_kind(kind: PartitionState) -> u8 {
    match kind {
        PartitionState::Ready => 0,
        PartitionState::Splitting => 1,
        PartitionState::ReceivingSplit => 2,
        PartitionState::DrainingSplit => 3,
        PartitionState::Merging => 4,
    }
}

fn decode_state_kind(tag: u8) -> Result<PartitionState> {
    match tag {
        0 => Ok(PartitionState::Ready),
        1 => Ok(PartitionState::Splitting),
        2 => Ok(PartitionState::ReceivingSplit),
        3 => Ok(PartitionState::DrainingSplit),
        4 => Ok(PartitionState::Merging),
        _ => Err(corrupt()),
    }
}

fn decode_logical_index_id(decoder: &mut Decoder) -> Result<LogicalIndexId> {
    LogicalIndexId::new(decoder.u64()?).map_err(|_| corrupt())
}

fn decode_partition_key(decoder: &mut Decoder) -> Result<PartitionKey> {
    PartitionKey::new(decoder.u64()?).map_err(|_| corrupt())
}

const fn value_kind_for_key(key: &LogicalKey) -> ValueKind {
    match key {
        LogicalKey::IndexIdAllocator => ValueKind::IndexIdAllocator,
        LogicalKey::IndexNameDirectory(_) => ValueKind::IndexNameEntry,
        LogicalKey::Manifest(_) => ValueKind::IndexManifest,
        LogicalKey::Record { .. } => ValueKind::VectorRecord,
        LogicalKey::Location { .. } => ValueKind::RecordLocation,
        LogicalKey::Payload { .. } => ValueKind::OpaquePayload,
        LogicalKey::TreeManifest { .. } => ValueKind::TreeManifest,
        LogicalKey::Header { .. } => ValueKind::PartitionHeader,
        LogicalKey::Synopsis { .. } => ValueKind::PartitionSynopsis,
        LogicalKey::State { .. } => ValueKind::PartitionState,
        LogicalKey::Centroid { .. } => ValueKind::PartitionCentroid,
        LogicalKey::LeafEntry { .. } => ValueKind::LeafEntry,
        LogicalKey::ChildEntry { .. } => ValueKind::ChildEntry,
    }
}

fn validate_key_identity(
    key: &LogicalKey,
    value: &PersistentValue,
    manifest: Option<&IndexManifest>,
) -> Result<()> {
    if let (Some(expected), Some(actual)) = (manifest, logical_key_index(key)) {
        if expected.logical_index_id() != actual {
            return Err(corrupt());
        }
    }
    match (key, value) {
        (LogicalKey::Manifest(expected), PersistentValue::IndexManifest(actual))
            if *expected != actual.logical_index_id() =>
        {
            Err(corrupt())
        }
        (LogicalKey::Record { id, .. }, PersistentValue::VectorRecord(actual))
            if id != actual.record_id() =>
        {
            Err(corrupt())
        }
        (LogicalKey::LeafEntry { id, .. }, PersistentValue::LeafEntry(actual))
            if id != actual.record_id() =>
        {
            Err(corrupt())
        }
        (LogicalKey::ChildEntry { child, .. }, PersistentValue::ChildEntry(actual))
            if *child != actual.child() =>
        {
            Err(corrupt())
        }
        _ => Ok(()),
    }
}

const fn logical_key_index(key: &LogicalKey) -> Option<LogicalIndexId> {
    match key {
        LogicalKey::IndexIdAllocator | LogicalKey::IndexNameDirectory(_) => None,
        LogicalKey::Manifest(index) => Some(*index),
        LogicalKey::Record { index, .. }
        | LogicalKey::Location { index, .. }
        | LogicalKey::Payload { index, .. }
        | LogicalKey::TreeManifest { index, .. }
        | LogicalKey::Header { index, .. }
        | LogicalKey::Synopsis { index, .. }
        | LogicalKey::State { index, .. }
        | LogicalKey::Centroid { index, .. }
        | LogicalKey::LeafEntry { index, .. }
        | LogicalKey::ChildEntry { index, .. } => Some(*index),
    }
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

fn unsupported() -> Error {
    Error::new(ErrorKind::UnsupportedFormat)
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new(kind: ValueKind) -> Self {
        Self {
            bytes: vec![kind.tag(), VALUE_CODEC_VERSION],
        }
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn f32(&mut self, value: f32) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::invalid_argument());
        }
        self.u32(if value == 0.0 { 0 } else { value.to_bits() });
        Ok(())
    }

    fn f64(&mut self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::invalid_argument());
        }
        self.u64(if value == 0.0 { 0 } else { value.to_bits() });
        Ok(())
    }

    fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn sized_bytes(&mut self, value: &[u8], maximum: usize) -> Result<()> {
        if value.len() > maximum {
            return Err(Error::invalid_argument());
        }
        self.u32(u32::try_from(value.len()).map_err(|_| Error::invalid_argument())?);
        self.bytes(value);
        Ok(())
    }

    fn sized_u8_bytes(&mut self, value: &[u8]) -> Result<()> {
        self.u8(u8::try_from(value.len()).map_err(|_| Error::invalid_argument())?);
        self.bytes(value);
        Ok(())
    }

    fn sized_u16_bytes(&mut self, value: &[u8]) -> Result<()> {
        self.u16(u16::try_from(value.len()).map_err(|_| Error::invalid_argument())?);
        self.bytes(value);
        Ok(())
    }

    fn finish(self) -> Result<Vec<u8>> {
        if self.bytes.len() > MAX_VALUE_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(self.bytes)
    }
}

struct Decoder {
    bytes: Bytes,
    position: usize,
}

impl Decoder {
    fn framed(expected: ValueKind, bytes: Bytes) -> Result<Self> {
        if bytes.len() < 2 || bytes.len() > MAX_VALUE_BYTES || bytes[0] != expected.tag() {
            return Err(corrupt());
        }
        if bytes[1] != VALUE_CODEC_VERSION {
            return Err(
                if matches!(
                    expected,
                    ValueKind::IndexIdAllocator
                        | ValueKind::IndexNameEntry
                        | ValueKind::IndexManifest
                ) {
                    unsupported()
                } else {
                    corrupt()
                },
            );
        }
        Ok(Self { bytes, position: 2 })
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, length: usize) -> Result<&[u8]> {
        let end = self.position.checked_add(length).ok_or_else(corrupt)?;
        if end > self.bytes.len() {
            return Err(corrupt());
        }
        let start = self.position;
        self.position = end;
        Ok(&self.bytes[start..end])
    }

    fn take_bytes(&mut self, length: usize) -> Result<Bytes> {
        let end = self.position.checked_add(length).ok_or_else(corrupt)?;
        if end > self.bytes.len() {
            return Err(corrupt());
        }
        let value = self.bytes.slice(self.position..end);
        self.position = end;
        Ok(value)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.take(N)?.try_into().map_err(|_| corrupt())
    }

    fn u8(&mut self) -> Result<u8> {
        self.take(1)?.first().copied().ok_or_else(corrupt)
    }

    fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(corrupt()),
        }
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.array()?))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_be_bytes(self.array()?))
    }

    fn canonical_f32(&mut self) -> Result<f32> {
        let bits = self.u32()?;
        let value = f32::from_bits(bits);
        if !value.is_finite() || (value == 0.0 && bits != 0) {
            return Err(corrupt());
        }
        Ok(value)
    }

    fn canonical_f64(&mut self) -> Result<f64> {
        let bits = self.u64()?;
        let value = f64::from_bits(bits);
        if !value.is_finite() || (value == 0.0 && bits != 0) {
            return Err(corrupt());
        }
        Ok(value)
    }

    fn sized_bytes(&mut self, maximum: usize) -> Result<Bytes> {
        let length = usize::try_from(self.u32()?).map_err(|_| corrupt())?;
        if length > maximum {
            return Err(corrupt());
        }
        self.take_bytes(length)
    }

    fn sized_u8_bytes(&mut self) -> Result<Bytes> {
        let length = usize::from(self.u8()?);
        self.take_bytes(length)
    }

    fn sized_u16_bytes(&mut self, maximum: usize) -> Result<Bytes> {
        let length = usize::from(self.u16()?);
        if length > maximum {
            return Err(corrupt());
        }
        self.take_bytes(length)
    }

    fn finish(self) -> Result<()> {
        if self.position != self.bytes.len() {
            return Err(corrupt());
        }
        Ok(())
    }
}
