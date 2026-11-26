//! Namespace and ordered families for Logical Keys.
//!
//! This module owns the logical keyspace: the namespace/index scope tags, typed
//! entry-kind discriminators, raw identity and name components, and bounded ranges. It embeds canonical Tree
//! Key bytes without interpreting or re-escaping them. It defines no persistent
//! values and no backend physical prefixes; adapters prepend their own bounded
//! prefix and value codecs live in a sibling module.
//!
//! # Layout
//!
//! Every logical key begins with a one-byte scope tag:
//!
//! ```text
//! [ scope: u8 ]
//! ```
//!
//! `Namespace` scope (`0x00`) keys address the whole Backend Namespace:
//!
//! ```text
//! [ 0x00 ][ 0x00 ]            IndexIdAllocator
//! [ 0x00 ][ 0x01 ][ name ]    IndexNameDirectory(name)
//! ```
//!
//! `Index` scope (`0x01`) keys begin with a big-endian Logical Index ID, so one
//! Logical Index owns one contiguous logical range for drop:
//!
//! ```text
//! [ 0x01 ][ index_id: u64 BE ][ kind ][ ... ]
//! ```
//!
//! The index-scoped `kind` byte selects the family. `Manifest`, `RecordGroup`,
//! `TreeManifest`, and `Partition` address index-level objects. A Record Group
//! contains one self-terminating Record ID followed by the `Record`, `Location`,
//! or `Payload` subkind, keeping all three values adjacent by Record ID.
//! `TreeManifest` ends with the canonical Tree Key. `Partition` is followed by
//! the canonical Tree Key, the big-endian Partition Key, a subkind byte
//! (`Header`, `Synopsis`, `State`, `Centroid`, `LeafEntry`, `ChildEntry`,
//! `GraphDescriptor`, `GraphChunk`, `DrainHeader`, `DrainSynopsis`), and —
//! only for `LeafEntry`, `ChildEntry`, and `GraphChunk` — a terminal Record ID,
//! child Partition Key, or chunk ordinal.
//!
//! # Fail closed
//!
//! Decoders reject an unknown scope, an unknown kind or subkind,
//! truncated or overlong components, noncanonical scalars (including `-0.0`),
//! invalid UTF-8, and trailing bytes after a terminal component. Every decode
//! failure returns [`ErrorKind::Corruption`]; encode-time rejection of invalid
//! caller input returns [`ErrorKind::InvalidArgument`].
//!
//! Decoding is zero-copy: decoded Tree Key and Record ID components borrow the
//! input key's allocation, and validation walks the canonical encoding without
//! materializing field values. Only a Record ID containing `0x00` bytes, whose
//! escaped form is not contiguous in the key, decodes into one owned buffer.

use std::fmt;

use bytes::Bytes;

use crate::api::{
    DataType, Error, ErrorKind, IndexName, LogicalIndexId, PartitionKey, Result, Value,
};

#[doc(inline)]
pub use super::tree_key::{MAX_STRING_BYTES, MAX_TREE_KEY_BYTES, TreeKey};

use super::tree_key::take_array;
pub(crate) use super::tree_key::tree_key_hash;
pub(crate) use super::tree_key::{
    decode_escaped_terminated, push_escaped_terminated, scan_escaped_terminated,
};

/// The fixed encoded width of a [`LogicalIndexId`] in bytes.
pub const LOGICAL_INDEX_ID_BYTES: usize = 8;
/// The fixed encoded width of a [`PartitionKey`] in bytes.
pub const PARTITION_KEY_BYTES: usize = 8;

/// The maximum encoded length of a Record ID in bytes.
pub const MAX_RECORD_ID_BYTES: usize = crate::api::MAX_RECORD_ID_BYTES;
const SCOPE_NAMESPACE: u8 = 0x00;
const SCOPE_INDEX: u8 = 0x01;

const NS_INDEX_ID_ALLOCATOR: u8 = 0x00;
const NS_INDEX_NAME_DIRECTORY: u8 = 0x01;
const NS_BUILD_WORKSPACE: u8 = 0x02;
const NS_BUILD_SCHEDULE: u8 = 0x03;
const NS_BUILD_SCHEDULE_SLOT: u8 = 0x04;

const KIND_MANIFEST: u8 = 0x00;
const KIND_RECORD_GROUP: u8 = 0x01;
const KIND_TREE_MANIFEST: u8 = 0x03;
const KIND_PARTITION: u8 = 0x04;
const KIND_BUILD_DESCRIPTOR: u8 = 0x05;
const KIND_BUILD_PROGRESS: u8 = 0x06;
const KIND_BUILD_UNIT: u8 = 0x07;

const RECORD_VALUE: u8 = 0x00;
const RECORD_LOCATION: u8 = 0x01;
const RECORD_PAYLOAD: u8 = 0x02;

const SUB_HEADER: u8 = 0x00;
const SUB_SYNOPSIS: u8 = 0x01;
const SUB_STATE: u8 = 0x02;
const SUB_CENTROID: u8 = 0x03;
const SUB_LEAF_ENTRY: u8 = 0x04;
const SUB_CHILD_ENTRY: u8 = 0x05;
const SUB_GRAPH_DESCRIPTOR: u8 = 0x06;
const SUB_GRAPH_CHUNK: u8 = 0x07;
const SUB_DRAIN_HEADER: u8 = 0x08;
const SUB_DRAIN_SYNOPSIS: u8 = 0x09;

/// A storage-corruption error for a malformed or noncanonical decoded key.
fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// A fully decoded logical key.
///
/// `Debug` redacts the Index Name, Record ID, and Tree Key components while
/// retaining the safe Logical Index ID and Partition Key identifiers.
#[derive(Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum LogicalKey {
    /// The Logical Index ID allocator high-water mark.
    IndexIdAllocator,
    /// An Index Name directory entry mapping to its Logical Index ID.
    IndexNameDirectory(IndexName),
    /// The Index Manifest of one Logical Index.
    Manifest(LogicalIndexId),
    /// Immutable input identity and construction parameters of a Bulk Build.
    BuildDescriptor(LogicalIndexId),
    /// Fenced stage, entry, and multiset-digest progress of a Bulk Build.
    BuildProgress(LogicalIndexId),
    /// Durable workspace ownership and failure ledger, outside index data.
    BuildWorkspace(LogicalIndexId),
    /// Durable distributed scheduler queue and lease.
    BuildSchedule(LogicalIndexId),
    /// Renewable worker-slot lease for a collaborative scheduled Bulk Build.
    BuildScheduleSlot {
        /// The target Logical Index ID.
        index: LogicalIndexId,
        /// Zero-based worker slot (`0..distributed_workers`).
        slot: u32,
    },
    /// Per-stage lease and completion record for one Bulk Build unit or fence lane.
    BuildUnit {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// Stage discriminator (`0` = writer fence lane, `1..` = execution/validation stages).
        stage: u8,
        /// Zero-based unit ordinal within `stage`.
        unit: u32,
    },
    /// A Vector Record.
    Record {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The raw Record ID.
        id: Bytes,
    },
    /// A Record Location.
    Location {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The raw Record ID.
        id: Bytes,
    },
    /// An Opaque Payload.
    Payload {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The raw Record ID.
        id: Bytes,
    },
    /// A Tree Manifest directory entry.
    TreeManifest {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
    },
    /// A partition Header.
    Header {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The Partition Key.
        partition: PartitionKey,
    },
    /// A partition Synopsis.
    Synopsis {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The Partition Key.
        partition: PartitionKey,
    },
    /// A split-drain sidecar Header for a level-1 `ReceivingSplit` leaf.
    DrainHeader {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The Partition Key.
        partition: PartitionKey,
    },
    /// A split-drain sidecar Synopsis for a level-1 `ReceivingSplit` leaf.
    DrainSynopsis {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The Partition Key.
        partition: PartitionKey,
    },
    /// A partition State.
    State {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The Partition Key.
        partition: PartitionKey,
    },
    /// A partition's immutable centroid.
    Centroid {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The Partition Key.
        partition: PartitionKey,
    },
    /// A Leaf Entry.
    LeafEntry {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The Partition Key.
        partition: PartitionKey,
        /// The raw Record ID.
        id: Bytes,
    },
    /// Integrity metadata of a partition-local graph.
    GraphDescriptor {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The owning internal Partition Key.
        partition: PartitionKey,
    },
    /// One format-sized topology chunk.
    GraphChunk {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The owning internal Partition Key.
        partition: PartitionKey,
        /// Zero-based contiguous chunk ordinal.
        chunk: u32,
    },
    /// A Child Entry.
    ChildEntry {
        /// The owning Logical Index ID.
        index: LogicalIndexId,
        /// The canonical Tree Key.
        tree_key: TreeKey,
        /// The Partition Key.
        partition: PartitionKey,
        /// The child Partition Key.
        child: PartitionKey,
    },
}

impl LogicalKey {
    /// Returns the owning Logical Index ID for an index-scoped key.
    pub(crate) const fn index(&self) -> Option<LogicalIndexId> {
        match self {
            Self::IndexIdAllocator
            | Self::IndexNameDirectory(_)
            | Self::BuildWorkspace(_)
            | Self::BuildSchedule(_)
            | Self::BuildScheduleSlot { .. } => None,
            Self::Manifest(index)
            | Self::BuildDescriptor(index)
            | Self::BuildProgress(index)
            | Self::BuildUnit { index, .. } => Some(*index),
            Self::Record { index, .. }
            | Self::Location { index, .. }
            | Self::Payload { index, .. }
            | Self::TreeManifest { index, .. }
            | Self::Header { index, .. }
            | Self::Synopsis { index, .. }
            | Self::DrainHeader { index, .. }
            | Self::DrainSynopsis { index, .. }
            | Self::State { index, .. }
            | Self::Centroid { index, .. }
            | Self::LeafEntry { index, .. }
            | Self::GraphDescriptor { index, .. }
            | Self::GraphChunk { index, .. }
            | Self::ChildEntry { index, .. } => Some(*index),
        }
    }

    /// Returns the embedded Tree Key for a tree-local key.
    pub(crate) const fn tree_key(&self) -> Option<&TreeKey> {
        match self {
            Self::TreeManifest { tree_key, .. }
            | Self::Header { tree_key, .. }
            | Self::Synopsis { tree_key, .. }
            | Self::DrainHeader { tree_key, .. }
            | Self::DrainSynopsis { tree_key, .. }
            | Self::State { tree_key, .. }
            | Self::Centroid { tree_key, .. }
            | Self::LeafEntry { tree_key, .. }
            | Self::GraphDescriptor { tree_key, .. }
            | Self::GraphChunk { tree_key, .. }
            | Self::ChildEntry { tree_key, .. } => Some(tree_key),
            Self::IndexIdAllocator
            | Self::IndexNameDirectory(_)
            | Self::Manifest(_)
            | Self::BuildDescriptor(_)
            | Self::BuildProgress(_)
            | Self::BuildWorkspace(_)
            | Self::BuildSchedule(_)
            | Self::BuildScheduleSlot { .. }
            | Self::BuildUnit { .. }
            | Self::Record { .. }
            | Self::Location { .. }
            | Self::Payload { .. } => None,
        }
    }
}

impl fmt::Debug for LogicalKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexIdAllocator => formatter.write_str("IndexIdAllocator"),
            Self::IndexNameDirectory(_) => formatter.write_str("IndexNameDirectory([REDACTED])"),
            Self::Manifest(index) => formatter.debug_tuple("Manifest").field(index).finish(),
            Self::BuildSchedule(index) => {
                formatter.debug_tuple("BuildSchedule").field(index).finish()
            }
            Self::BuildScheduleSlot { index, slot } => formatter
                .debug_struct("BuildScheduleSlot")
                .field("index", index)
                .field("slot", slot)
                .finish(),
            Self::BuildWorkspace(index) => formatter
                .debug_tuple("BuildWorkspace")
                .field(index)
                .finish(),
            Self::BuildUnit { index, stage, unit } => formatter
                .debug_struct("BuildUnit")
                .field("index", index)
                .field("stage", stage)
                .field("unit", unit)
                .finish(),
            Self::BuildProgress(index) => {
                formatter.debug_tuple("BuildProgress").field(index).finish()
            }
            Self::BuildDescriptor(index) => formatter
                .debug_tuple("BuildDescriptor")
                .field(index)
                .finish(),
            Self::Record { index, .. } => formatter
                .debug_struct("Record")
                .field("index", index)
                .field("id", &"[REDACTED]")
                .finish(),
            Self::Location { index, .. } => formatter
                .debug_struct("Location")
                .field("index", index)
                .field("id", &"[REDACTED]")
                .finish(),
            Self::Payload { index, .. } => formatter
                .debug_struct("Payload")
                .field("index", index)
                .field("id", &"[REDACTED]")
                .finish(),
            Self::TreeManifest { index, .. } => formatter
                .debug_struct("TreeManifest")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .finish(),
            Self::Header {
                index, partition, ..
            } => formatter
                .debug_struct("Header")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .finish(),
            Self::Synopsis {
                index, partition, ..
            } => formatter
                .debug_struct("Synopsis")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .finish(),
            Self::DrainHeader {
                index, partition, ..
            } => formatter
                .debug_struct("DrainHeader")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .finish(),
            Self::DrainSynopsis {
                index, partition, ..
            } => formatter
                .debug_struct("DrainSynopsis")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .finish(),
            Self::State {
                index, partition, ..
            } => formatter
                .debug_struct("State")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .finish(),
            Self::Centroid {
                index, partition, ..
            } => formatter
                .debug_struct("Centroid")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .finish(),
            Self::LeafEntry {
                index, partition, ..
            } => formatter
                .debug_struct("LeafEntry")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .field("id", &"[REDACTED]")
                .finish(),
            Self::GraphDescriptor {
                index, partition, ..
            } => formatter
                .debug_struct("GraphDescriptor")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .finish(),
            Self::GraphChunk {
                index,
                partition,
                chunk,
                ..
            } => formatter
                .debug_struct("GraphChunk")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .field("chunk", chunk)
                .finish(),
            Self::ChildEntry {
                index, partition, ..
            } => formatter
                .debug_struct("ChildEntry")
                .field("index", index)
                .field("tree_key", &"[REDACTED]")
                .field("partition", partition)
                .field("child", &"[REDACTED]")
                .finish(),
        }
    }
}

/// A half-open `[start, end)` byte range over the logical keyspace.
///
/// `start` is inclusive and `end` is exclusive. `Debug` is redacted because a
/// range may embed Tree Key and Record ID bytes.
#[derive(Clone, Eq, PartialEq)]
pub struct KeyRange {
    start: Vec<u8>,
    end: Vec<u8>,
}

impl KeyRange {
    /// Constructs a half-open `[start, end)` range from explicit byte bounds.
    ///
    /// A range whose `start` is not less than `end` is empty and yields no
    /// keys. This is the generic constructor used for scan pagination; the
    /// codec range helpers below build their exact prefix ends themselves.
    #[must_use]
    pub fn new(start: Vec<u8>, end: Vec<u8>) -> Self {
        Self { start, end }
    }

    /// Selects every key under one encoded KTANN prefix. Its scope tag
    /// guarantees the prefix has a finite exclusive end.
    fn for_prefix(start: Vec<u8>) -> Self {
        let end = successor(&start);
        Self { start, end }
    }

    /// The inclusive start bound.
    #[must_use]
    pub fn start(&self) -> &[u8] {
        &self.start
    }

    /// The exclusive end bound.
    #[must_use]
    pub fn end(&self) -> &[u8] {
        &self.end
    }
}

impl fmt::Debug for KeyRange {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KeyRange")
            .field("start", &"[REDACTED]")
            .field("end", &"[REDACTED]")
            .finish()
    }
}

/// The smallest byte string strictly greater than every string with `prefix`.
///
/// Returns an empty slice only when `prefix` is all `0xFF`; no logical key
/// prefix is, because every key begins with a `0x00` or `0x01` scope tag.
pub(super) fn successor(prefix: &[u8]) -> Vec<u8> {
    let mut bytes = prefix.to_vec();
    while let Some(last) = bytes.last_mut() {
        if *last == 0xFF {
            bytes.pop();
        } else {
            *last += 1;
            return bytes;
        }
    }
    Vec::new()
}

/// Rejects an invalid Record ID on the encode path.
fn check_record_id(id: &[u8]) -> Result<()> {
    if id.is_empty() || id.len() > MAX_RECORD_ID_BYTES {
        return Err(Error::invalid_argument());
    }
    Ok(())
}

/// Validates and shares the terminal Record ID starting at `offset` in `key`.
fn slice_terminal_record_id(key: &Bytes, offset: usize) -> Result<Bytes> {
    let id = key.get(offset..).ok_or_else(corrupt)?;
    if id.is_empty() || id.len() > MAX_RECORD_ID_BYTES {
        return Err(corrupt());
    }
    Ok(key.slice(offset..))
}

/// Decodes a terminal Index Name.
fn decode_name(bytes: &[u8]) -> Result<IndexName> {
    let name = std::str::from_utf8(bytes).map_err(|_| corrupt())?;
    IndexName::new(name).map_err(|_| corrupt())
}

/// Builds the common `[ index scope ][ index id ]` key prefix.
fn index_prefix(index: LogicalIndexId) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1 + LOGICAL_INDEX_ID_BYTES);
    bytes.push(SCOPE_INDEX);
    bytes.extend_from_slice(&index.get().to_be_bytes());
    bytes
}

/// Builds the common `[ index prefix ][ partition ][ tree key ][ partition key ]`
/// prefix shared by every partition-local key.
fn partition_prefix(index: LogicalIndexId, tree_key: &TreeKey, partition: PartitionKey) -> Vec<u8> {
    let mut bytes = index_prefix(index);
    bytes.push(KIND_PARTITION);
    bytes.extend_from_slice(tree_key.as_bytes());
    bytes.extend_from_slice(&partition.get().to_be_bytes());
    bytes
}

/// The single Logical Index ID allocator key.
#[must_use]
pub fn index_id_allocator_key() -> Vec<u8> {
    vec![SCOPE_NAMESPACE, NS_INDEX_ID_ALLOCATOR]
}

/// The Index Name directory key for `name`.
#[must_use]
pub fn name_directory_key(name: &IndexName) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(2 + name.as_str().len());
    bytes.push(SCOPE_NAMESPACE);
    bytes.push(NS_INDEX_NAME_DIRECTORY);
    bytes.extend_from_slice(name.as_str().as_bytes());
    bytes
}

/// The Index Manifest key for `index`.
#[must_use]
pub fn manifest_key(index: LogicalIndexId) -> Vec<u8> {
    let mut bytes = index_prefix(index);
    bytes.push(KIND_MANIFEST);
    bytes
}

/// The immutable Bulk Build request key for `index`.
#[must_use]
pub fn build_descriptor_key(index: LogicalIndexId) -> Vec<u8> {
    let mut bytes = index_prefix(index);
    bytes.push(KIND_BUILD_DESCRIPTOR);
    bytes
}

/// The Bulk Build stage and multiset-digest progress key for `index`.
#[must_use]
pub fn build_progress_key(index: LogicalIndexId) -> Vec<u8> {
    let mut bytes = index_prefix(index);
    bytes.push(KIND_BUILD_PROGRESS);
    bytes
}

/// Namespace-owned workspace ledger key; survives index-prefix removal.
#[must_use]
pub fn build_workspace_key(index: LogicalIndexId) -> Vec<u8> {
    namespace_build_key(NS_BUILD_WORKSPACE, index)
}

/// Namespace scheduler queue key, surviving index-prefix deletion.
#[must_use]
pub fn build_schedule_key(index: LogicalIndexId) -> Vec<u8> {
    namespace_build_key(NS_BUILD_SCHEDULE, index)
}

/// Namespace collaborative scheduler slot key, surviving index-prefix deletion.
#[must_use]
pub fn build_schedule_slot_key(index: LogicalIndexId, slot: u32) -> Vec<u8> {
    let mut bytes = namespace_build_key(NS_BUILD_SCHEDULE_SLOT, index);
    bytes.extend_from_slice(&slot.to_be_bytes());
    bytes
}

/// Contiguous namespace range of every collaborative scheduler slot for `index`.
#[must_use]
pub(crate) fn build_schedule_slot_range(index: LogicalIndexId) -> KeyRange {
    KeyRange::for_prefix(namespace_build_key(NS_BUILD_SCHEDULE_SLOT, index))
}

fn namespace_build_key(kind: u8, index: LogicalIndexId) -> Vec<u8> {
    let mut bytes = vec![SCOPE_NAMESPACE, kind];
    bytes.extend_from_slice(&index.get().to_be_bytes());
    bytes
}

/// One stage-local Bulk Build unit or fence-lane key.
#[must_use]
pub fn build_unit_key(index: LogicalIndexId, stage: u8, unit: u32) -> Vec<u8> {
    let mut bytes = index_prefix(index);
    bytes.push(KIND_BUILD_UNIT);
    bytes.push(stage);
    bytes.extend_from_slice(&unit.to_be_bytes());
    bytes
}

/// Contiguous range of every `BuildUnit` in one stage of `index`.
#[must_use]
pub(crate) fn build_unit_stage_range(index: LogicalIndexId, stage: u8) -> KeyRange {
    let mut prefix = index_prefix(index);
    prefix.push(KIND_BUILD_UNIT);
    prefix.push(stage);
    KeyRange::for_prefix(prefix)
}

/// Contiguous serving-data range (`RecordGroup`, `TreeManifest`, and `Partition` keys
/// `[0x01..0x05)`) owned by `index`, excluding `Manifest` (`0x00`) and `Build*` (`0x05..`).
#[must_use]
pub(crate) fn serving_data_range(index: LogicalIndexId) -> KeyRange {
    let mut start = index_prefix(index);
    start.push(KIND_RECORD_GROUP);
    let mut end = index_prefix(index);
    end.push(KIND_BUILD_DESCRIPTOR);
    KeyRange::new(start, end)
}

/// Contiguous range of transient `BuildUnit` keys (`[0x07..0x08)`) owned by `index`,
/// preserving `BuildDescriptor` (`0x05`) and `BuildProgress` (`0x06`).
#[must_use]
pub(crate) fn transient_build_unit_range(index: LogicalIndexId) -> KeyRange {
    let mut prefix = index_prefix(index);
    prefix.push(KIND_BUILD_UNIT);
    KeyRange::for_prefix(prefix)
}

/// The Vector Record key for `id` in `index`.
pub fn record_key(index: LogicalIndexId, id: &Bytes) -> Result<Vec<u8>> {
    let mut bytes = record_group_prefix(index, id)?;
    bytes.push(RECORD_VALUE);
    Ok(bytes)
}

/// The Record Location key for `id` in `index`.
pub fn location_key(index: LogicalIndexId, id: &Bytes) -> Result<Vec<u8>> {
    let mut bytes = record_group_prefix(index, id)?;
    bytes.push(RECORD_LOCATION);
    Ok(bytes)
}

/// Returns the maximum encoded Record Location key length.
pub(crate) const fn maximum_location_key_len() -> usize {
    // Index prefix, record-group kind, maximally escaped terminated Record ID,
    // and Record Location subkind.
    1 + LOGICAL_INDEX_ID_BYTES + 1 + (2 * MAX_RECORD_ID_BYTES + 2) + 1
}

/// The Opaque Payload key for `id` in `index`.
pub fn payload_key(index: LogicalIndexId, id: &Bytes) -> Result<Vec<u8>> {
    let mut bytes = record_group_prefix(index, id)?;
    bytes.push(RECORD_PAYLOAD);
    Ok(bytes)
}

fn record_group_prefix(index: LogicalIndexId, id: &Bytes) -> Result<Vec<u8>> {
    check_record_id(id)?;
    let mut bytes = index_prefix(index);
    bytes.push(KIND_RECORD_GROUP);
    push_escaped_terminated(&mut bytes, id);
    Ok(bytes)
}

/// The Tree Manifest directory key for `tree_key` in `index`.
#[must_use]
pub fn tree_manifest_key(index: LogicalIndexId, tree_key: &TreeKey) -> Vec<u8> {
    let mut bytes = index_prefix(index);
    bytes.push(KIND_TREE_MANIFEST);
    bytes.extend_from_slice(tree_key.as_bytes());
    bytes
}

/// The partition Header key.
#[must_use]
pub fn header_key(index: LogicalIndexId, tree_key: &TreeKey, partition: PartitionKey) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_HEADER);
    bytes
}

/// The partition Synopsis key.
#[must_use]
pub fn synopsis_key(index: LogicalIndexId, tree_key: &TreeKey, partition: PartitionKey) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_SYNOPSIS);
    bytes
}

/// The split-drain sidecar Header key for a level-1 `ReceivingSplit` leaf.
#[must_use]
pub fn drain_header_key(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_DRAIN_HEADER);
    bytes
}

/// The split-drain sidecar Synopsis key for a level-1 `ReceivingSplit` leaf.
#[must_use]
pub fn drain_synopsis_key(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_DRAIN_SYNOPSIS);
    bytes
}

/// Returns one partition metadata key's exact encoded length.
pub(crate) fn partition_metadata_key_len(tree_key: &TreeKey) -> usize {
    // Index prefix, partition kind, Tree Key, Partition Key, and metadata subkind.
    1 + LOGICAL_INDEX_ID_BYTES + 1 + tree_key.as_bytes().len() + PARTITION_KEY_BYTES + 1
}

/// The partition State key.
#[must_use]
pub fn state_key(index: LogicalIndexId, tree_key: &TreeKey, partition: PartitionKey) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_STATE);
    bytes
}

/// The partition's immutable centroid key.
#[must_use]
pub fn centroid_key(index: LogicalIndexId, tree_key: &TreeKey, partition: PartitionKey) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_CENTROID);
    bytes
}

/// The partition-local graph descriptor key.
#[must_use]
pub fn graph_descriptor_key(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_GRAPH_DESCRIPTOR);
    bytes
}

/// One partition-local graph chunk key, ordered by its numeric ordinal.
#[must_use]
pub fn graph_chunk_key(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
    chunk: u32,
) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_GRAPH_CHUNK);
    bytes.extend_from_slice(&chunk.to_be_bytes());
    bytes
}

/// Range containing only one partition's graph chunks.
#[must_use]
pub fn graph_chunk_range(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> KeyRange {
    partition_subkind_range(index, tree_key, partition, SUB_GRAPH_CHUNK)
}

/// The Leaf Entry key for `id` in `partition`.
pub fn leaf_entry_key(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
    id: &Bytes,
) -> Result<Vec<u8>> {
    check_record_id(id)?;
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_LEAF_ENTRY);
    bytes.extend_from_slice(id);
    Ok(bytes)
}

/// Returns the maximum encoded Leaf Entry key length for `tree_key`.
pub(crate) fn maximum_leaf_entry_key_len(tree_key: &TreeKey) -> usize {
    partition_metadata_key_len(tree_key) + MAX_RECORD_ID_BYTES
}

/// The Child Entry key for child partition `child` in `partition`.
#[must_use]
pub fn child_entry_key(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
    child: PartitionKey,
) -> Vec<u8> {
    let mut bytes = partition_prefix(index, tree_key, partition);
    bytes.push(SUB_CHILD_ENTRY);
    bytes.extend_from_slice(&child.get().to_be_bytes());
    bytes
}

/// Encodes one typed Logical Key to canonical bytes.
pub(crate) fn encode_key(key: &LogicalKey) -> Result<Vec<u8>> {
    match key {
        LogicalKey::IndexIdAllocator => Ok(index_id_allocator_key()),
        LogicalKey::IndexNameDirectory(name) => Ok(name_directory_key(name)),
        LogicalKey::Manifest(index) => Ok(manifest_key(*index)),
        LogicalKey::BuildDescriptor(index) => Ok(build_descriptor_key(*index)),
        LogicalKey::BuildProgress(index) => Ok(build_progress_key(*index)),
        LogicalKey::BuildWorkspace(index) => Ok(build_workspace_key(*index)),
        LogicalKey::BuildSchedule(index) => Ok(build_schedule_key(*index)),
        LogicalKey::BuildScheduleSlot { index, slot } => Ok(build_schedule_slot_key(*index, *slot)),
        LogicalKey::BuildUnit { index, stage, unit } => Ok(build_unit_key(*index, *stage, *unit)),
        LogicalKey::Record { index, id } => record_key(*index, id),
        LogicalKey::Location { index, id } => location_key(*index, id),
        LogicalKey::Payload { index, id } => payload_key(*index, id),
        LogicalKey::TreeManifest { index, tree_key } => Ok(tree_manifest_key(*index, tree_key)),
        LogicalKey::Header {
            index,
            tree_key,
            partition,
        } => Ok(header_key(*index, tree_key, *partition)),
        LogicalKey::Synopsis {
            index,
            tree_key,
            partition,
        } => Ok(synopsis_key(*index, tree_key, *partition)),
        LogicalKey::DrainHeader {
            index,
            tree_key,
            partition,
        } => Ok(drain_header_key(*index, tree_key, *partition)),
        LogicalKey::DrainSynopsis {
            index,
            tree_key,
            partition,
        } => Ok(drain_synopsis_key(*index, tree_key, *partition)),
        LogicalKey::State {
            index,
            tree_key,
            partition,
        } => Ok(state_key(*index, tree_key, *partition)),
        LogicalKey::Centroid {
            index,
            tree_key,
            partition,
        } => Ok(centroid_key(*index, tree_key, *partition)),
        LogicalKey::LeafEntry {
            index,
            tree_key,
            partition,
            id,
        } => leaf_entry_key(*index, tree_key, *partition, id),
        LogicalKey::GraphDescriptor {
            index,
            tree_key,
            partition,
        } => Ok(graph_descriptor_key(*index, tree_key, *partition)),
        LogicalKey::GraphChunk {
            index,
            tree_key,
            partition,
            chunk,
        } => Ok(graph_chunk_key(*index, tree_key, *partition, *chunk)),
        LogicalKey::ChildEntry {
            index,
            tree_key,
            partition,
            child,
        } => Ok(child_entry_key(*index, tree_key, *partition, *child)),
    }
}

/// Decodes a complete logical key.
///
/// `types` is the ordered Tree Key field types and is required to split the
/// Tree Key from its trailing components; it must match the Logical Index's
/// immutable schema. Decoding is zero-copy: the decoded Tree Key and Record ID
/// components share `key`'s allocation.
pub fn decode_key(types: &[DataType], key: &Bytes) -> Result<LogicalKey> {
    let scope = *key.first().ok_or_else(corrupt)?;
    match scope {
        SCOPE_NAMESPACE => decode_namespace_key(&key[1..]),
        SCOPE_INDEX => decode_index_key(types, key, 1),
        _ => Err(corrupt()),
    }
}

/// Decodes a namespace-scope key body.
fn decode_namespace_key(body: &[u8]) -> Result<LogicalKey> {
    match body.first() {
        Some(&NS_INDEX_ID_ALLOCATOR) if body.len() == 1 => Ok(LogicalKey::IndexIdAllocator),
        Some(&NS_BUILD_WORKSPACE) if body.len() == 9 => Ok(LogicalKey::BuildWorkspace(
            LogicalIndexId::new(u64::from_be_bytes(body[1..].try_into().expect("fixed id")))
                .map_err(|_| corrupt())?,
        )),
        Some(&NS_BUILD_SCHEDULE) if body.len() == 9 => Ok(LogicalKey::BuildSchedule(
            LogicalIndexId::new(u64::from_be_bytes(body[1..].try_into().expect("fixed id")))
                .map_err(|_| corrupt())?,
        )),
        Some(&NS_BUILD_SCHEDULE_SLOT) if body.len() == 13 => {
            let index =
                LogicalIndexId::new(u64::from_be_bytes(body[1..9].try_into().expect("fixed id")))
                    .map_err(|_| corrupt())?;
            let slot = u32::from_be_bytes(body[9..13].try_into().expect("fixed slot"));
            Ok(LogicalKey::BuildScheduleSlot { index, slot })
        }
        Some(&NS_INDEX_NAME_DIRECTORY) => {
            Ok(LogicalKey::IndexNameDirectory(decode_name(&body[1..])?))
        }
        _ => Err(corrupt()),
    }
}

/// Decodes an index-scope key body starting at `offset` in `key`.
fn decode_index_key(types: &[DataType], key: &Bytes, offset: usize) -> Result<LogicalKey> {
    let body = &key[offset..];
    let index =
        LogicalIndexId::new(u64::from_be_bytes(take_array::<8>(body)?)).map_err(|_| corrupt())?;
    let kind = *body.get(LOGICAL_INDEX_ID_BYTES).ok_or_else(corrupt)?;
    let rest = offset + LOGICAL_INDEX_ID_BYTES + 1;

    match kind {
        KIND_BUILD_UNIT if body.len() == LOGICAL_INDEX_ID_BYTES + 6 => {
            let stage = body[LOGICAL_INDEX_ID_BYTES + 1];
            let unit = u32::from_be_bytes(
                body[LOGICAL_INDEX_ID_BYTES + 2..]
                    .try_into()
                    .expect("fixed u32"),
            );
            Ok(LogicalKey::BuildUnit { index, stage, unit })
        }
        KIND_BUILD_PROGRESS if body.len() == LOGICAL_INDEX_ID_BYTES + 1 => {
            Ok(LogicalKey::BuildProgress(index))
        }
        KIND_BUILD_DESCRIPTOR if body.len() == LOGICAL_INDEX_ID_BYTES + 1 => {
            Ok(LogicalKey::BuildDescriptor(index))
        }
        KIND_MANIFEST if body.len() == LOGICAL_INDEX_ID_BYTES + 1 => {
            Ok(LogicalKey::Manifest(index))
        }
        KIND_RECORD_GROUP => decode_record_group(index, key, rest),
        KIND_TREE_MANIFEST => Ok(LogicalKey::TreeManifest {
            index,
            tree_key: TreeKey::from_encoded(types, key.slice(rest..))?,
        }),
        KIND_PARTITION => decode_partition_key(types, index, key, rest),
        _ => Err(corrupt()),
    }
}

fn decode_record_group(index: LogicalIndexId, key: &Bytes, offset: usize) -> Result<LogicalKey> {
    let body = key.get(offset..).ok_or_else(corrupt)?;
    let scan = scan_escaped_terminated(body, MAX_RECORD_ID_BYTES)?;
    let id = if scan.escaped {
        // An escaped Record ID is not contiguous in the key, so decode it into
        // one owned buffer. IDs without a `0x00` byte take the borrowed path.
        Bytes::from(decode_escaped_terminated(body, &scan))
    } else {
        key.slice(offset..offset + scan.decoded_len)
    };
    if id.is_empty() {
        return Err(corrupt());
    }
    match key.get(offset + scan.consumed..) {
        Some([RECORD_VALUE]) => Ok(LogicalKey::Record { index, id }),
        Some([RECORD_LOCATION]) => Ok(LogicalKey::Location { index, id }),
        Some([RECORD_PAYLOAD]) => Ok(LogicalKey::Payload { index, id }),
        _ => Err(corrupt()),
    }
}

/// Decodes the remainder of a partition-scoped key body starting at `offset`
/// in `key`.
fn decode_partition_key(
    types: &[DataType],
    index: LogicalIndexId,
    key: &Bytes,
    offset: usize,
) -> Result<LogicalKey> {
    let rest = key.slice(offset..);
    let (tree_key, tree_key_len) = TreeKey::from_prefix(types, &rest)?;
    let after = &rest[tree_key_len..];

    let partition =
        PartitionKey::new(u64::from_be_bytes(take_array::<8>(after)?)).map_err(|_| corrupt())?;
    let subkind = *after.get(PARTITION_KEY_BYTES).ok_or_else(corrupt)?;
    let terminal_offset = offset + tree_key_len + PARTITION_KEY_BYTES + 1;
    let terminal = key.get(terminal_offset..).ok_or_else(corrupt)?;

    match subkind {
        SUB_HEADER if terminal.is_empty() => Ok(LogicalKey::Header {
            index,
            tree_key,
            partition,
        }),
        SUB_SYNOPSIS if terminal.is_empty() => Ok(LogicalKey::Synopsis {
            index,
            tree_key,
            partition,
        }),
        SUB_DRAIN_HEADER if terminal.is_empty() => Ok(LogicalKey::DrainHeader {
            index,
            tree_key,
            partition,
        }),
        SUB_DRAIN_SYNOPSIS if terminal.is_empty() => Ok(LogicalKey::DrainSynopsis {
            index,
            tree_key,
            partition,
        }),
        SUB_STATE if terminal.is_empty() => Ok(LogicalKey::State {
            index,
            tree_key,
            partition,
        }),
        SUB_GRAPH_DESCRIPTOR if terminal.is_empty() => Ok(LogicalKey::GraphDescriptor {
            index,
            tree_key,
            partition,
        }),
        SUB_GRAPH_CHUNK if terminal.len() == 4 => Ok(LogicalKey::GraphChunk {
            index,
            tree_key,
            partition,
            chunk: u32::from_be_bytes(terminal.try_into().map_err(|_| corrupt())?),
        }),
        SUB_CENTROID if terminal.is_empty() => Ok(LogicalKey::Centroid {
            index,
            tree_key,
            partition,
        }),
        SUB_LEAF_ENTRY => Ok(LogicalKey::LeafEntry {
            index,
            tree_key,
            partition,
            id: slice_terminal_record_id(key, terminal_offset)?,
        }),
        SUB_CHILD_ENTRY => {
            let child = PartitionKey::new(u64::from_be_bytes(take_array::<8>(terminal)?))
                .map_err(|_| corrupt())?;
            if terminal.len() != PARTITION_KEY_BYTES {
                return Err(corrupt());
            }
            Ok(LogicalKey::ChildEntry {
                index,
                tree_key,
                partition,
                child,
            })
        }
        _ => Err(corrupt()),
    }
}

/// Namespace-owned workspace records strictly after the supplied index ID.
pub(crate) fn build_workspace_range(after: Option<LogicalIndexId>) -> KeyRange {
    namespace_build_range(NS_BUILD_WORKSPACE, after)
}

/// Scheduled build records strictly after the supplied index ID.
pub(crate) fn build_schedule_range(after: Option<LogicalIndexId>) -> KeyRange {
    namespace_build_range(NS_BUILD_SCHEDULE, after)
}

/// Resume strictly after the last processed key, retaining the full namespace
/// range's exclusive end even when the last Logical Index ID is maximal.
fn namespace_build_range(kind: u8, after: Option<LogicalIndexId>) -> KeyRange {
    let mut range = KeyRange::for_prefix(vec![SCOPE_NAMESPACE, kind]);
    if let Some(index) = after {
        range.start = namespace_build_key(kind, index);
        range.start.push(0);
    }
    range
}

/// The contiguous range of every key owned by one Logical Index.
#[must_use]
pub fn index_range(index: LogicalIndexId) -> KeyRange {
    KeyRange::for_prefix(index_prefix(index))
}

/// The contiguous range of every Tree Manifest directory key in one index.
#[must_use]
pub fn tree_manifest_range(index: LogicalIndexId) -> KeyRange {
    let mut start = index_prefix(index);
    start.push(KIND_TREE_MANIFEST);
    KeyRange::for_prefix(start)
}

/// The contiguous range of every key owned by one partition.
#[must_use]
pub fn partition_range(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> KeyRange {
    KeyRange::for_prefix(partition_prefix(index, tree_key, partition))
}

/// Builds the range for one partition-local subkind.
fn partition_subkind_range(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
    subkind: u8,
) -> KeyRange {
    let mut start = partition_prefix(index, tree_key, partition);
    start.push(subkind);
    KeyRange::for_prefix(start)
}

/// The contiguous range of every Leaf Entry in one partition.
pub(crate) fn leaf_entry_range(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> KeyRange {
    partition_subkind_range(index, tree_key, partition, SUB_LEAF_ENTRY)
}

/// The contiguous range of every Child Entry in one partition.
pub(crate) fn child_entry_range(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> KeyRange {
    partition_subkind_range(index, tree_key, partition, SUB_CHILD_ENTRY)
}

/// The Tree Manifest directory range matching a Tree Key field prefix.
///
/// `prefix` holds the leading field values that select the trees to enumerate;
/// an empty prefix selects every tree. It must not name more fields than
/// `types`.
pub fn tree_manifest_prefix_range(
    index: LogicalIndexId,
    types: &[DataType],
    prefix: &[Value],
) -> Result<KeyRange> {
    if prefix.len() > types.len() {
        return Err(Error::invalid_argument());
    }
    let mut start = index_prefix(index);
    start.push(KIND_TREE_MANIFEST);
    TreeKey::append_fields(&types[..prefix.len()], prefix, &mut start)?;
    Ok(KeyRange::for_prefix(start))
}

/// The Tree Manifest directory range for one planned field expansion.
///
/// `prefix` carries the already encoded leading exact field values and `lower`
/// and `upper` carry the already encoded bounds of the next field; `None`
/// bounds are unbounded within that field. The exclusive end is the byte
/// successor of the prefix plus upper bound when present, and the byte
/// successor of the prefix alone otherwise, so the range always covers every
/// directory key under the prefix whose bounded field value starts at or after
/// `lower` and — for a string upper bound — a conservative superset of keys
/// strictly before `upper`. Typed planning rejects those byte-superset
/// artifacts during enumeration.
pub(crate) fn tree_manifest_plan_range(
    index: LogicalIndexId,
    prefix: &[u8],
    lower: Option<&[u8]>,
    upper: Option<&[u8]>,
) -> KeyRange {
    let start = {
        let mut start = index_prefix(index);
        start.push(KIND_TREE_MANIFEST);
        start.extend_from_slice(prefix);
        start.extend_from_slice(lower.unwrap_or_default());
        start
    };
    let mut end = index_prefix(index);
    end.push(KIND_TREE_MANIFEST);
    end.extend_from_slice(prefix);
    if let Some(upper) = upper {
        end.extend_from_slice(upper);
    }
    let end = successor(&end);
    KeyRange { start, end }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successor_increments_the_last_non_ff_byte() {
        assert_eq!(successor(b"\x01\x02\x03"), b"\x01\x02\x04");
        assert_eq!(successor(b"\x01\xff\xff"), b"\x02");
        assert_eq!(successor(b"\xff"), Vec::<u8>::new());
    }

    #[test]
    fn plan_range_without_bounds_is_the_directory_prefix() {
        let index = LogicalIndexId::new(7).expect("nonzero");
        let range = tree_manifest_plan_range(index, b"", None, None);
        assert_eq!(range.start(), tree_manifest_range(index).start());
        assert_eq!(range.end(), tree_manifest_range(index).end());
    }

    #[test]
    fn plan_range_upper_bound_is_the_byte_successor() {
        let index = LogicalIndexId::new(7).expect("nonzero");
        let range = tree_manifest_plan_range(index, b"\x01", None, Some(b"\x05\x00"));
        let mut expected_end = index_prefix(index);
        expected_end.push(KIND_TREE_MANIFEST);
        expected_end.extend_from_slice(b"\x01\x05\x00");
        assert_eq!(range.end(), successor(&expected_end));
        // The end is the successor of prefix + upper, so directory keys with a
        // strictly greater encoded field value are still covered as artifacts.
        assert!(range.end().len() >= expected_end.len());
    }

    #[test]
    fn plan_range_with_only_a_lower_bound_ends_at_the_prefix_successor() {
        let index = LogicalIndexId::new(7).expect("nonzero");
        let range = tree_manifest_plan_range(index, b"\x01", Some(b"\x05"), None);
        let mut expected_start = index_prefix(index);
        expected_start.push(KIND_TREE_MANIFEST);
        expected_start.extend_from_slice(b"\x01\x05");
        assert_eq!(range.start(), expected_start);
        // Without an upper bound the range reaches the end of the whole prefix,
        // so every field value above the lower bound stays covered.
        let mut prefix = index_prefix(index);
        prefix.push(KIND_TREE_MANIFEST);
        prefix.extend_from_slice(b"\x01");
        assert_eq!(range.end(), successor(&prefix));
    }
}
