//! Versioned namespace and ordered families for Logical Keys.
//!
//! This module owns the version-1 logical keyspace: the version framing, the
//! namespace/index scope tags, the typed entry-kind discriminators, the raw
//! identity and name components, and bounded ranges. It embeds canonical Tree
//! Key bytes without interpreting or re-escaping them. It defines no persistent
//! values and no backend physical prefixes; adapters prepend their own bounded
//! prefix and value codecs live in a sibling module.
//!
//! # Layout
//!
//! Every logical key begins with a one-byte [`KEY_VERSION`] and a one-byte
//! scope tag:
//!
//! ```text
//! [ version: u8 = KEY_VERSION ][ scope: u8 ]
//! ```
//!
//! `Namespace` scope (`0x00`) keys address the whole Backend Namespace:
//!
//! ```text
//! [ 0x01 ][ 0x00 ][ 0x00 ]            IndexIdAllocator
//! [ 0x01 ][ 0x00 ][ 0x01 ][ name ]    IndexNameDirectory(name)
//! ```
//!
//! `Index` scope (`0x01`) keys begin with a big-endian Logical Index ID, so one
//! Logical Index owns one contiguous logical range for drop:
//!
//! ```text
//! [ 0x01 ][ 0x01 ][ index_id: u64 BE ][ kind ][ ... ]
//! ```
//!
//! The index-scoped `kind` byte selects the family. `Manifest`, `RecordGroup`,
//! `TreeManifest`, and `Partition` address index-level objects. A Record Group
//! contains one self-terminating Record ID followed by the `Record`, `Location`,
//! or `Payload` subkind, keeping all three values adjacent by Record ID.
//! `TreeManifest` ends with the canonical Tree Key. `Partition` is followed by
//! the canonical Tree Key, the big-endian Partition Key, a subkind byte
//! (`Header`, `Synopsis`, `State`, `Centroid`, `LeafEntry`, `ChildEntry`), and —
//! only for `LeafEntry` and `ChildEntry` — a terminal Record ID or child
//! Partition Key.
//!
//! # Fail closed
//!
//! Decoders reject an unknown version or scope, an unknown kind or subkind,
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

pub(crate) use super::tree_key::tree_key_hash;
use super::tree_key::{
    decode_escaped_terminated, push_escaped_terminated, scan_escaped_terminated, take_array,
};

/// The single logical-key format version emitted and accepted by this build.
pub const KEY_VERSION: u8 = 1;

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

const KIND_MANIFEST: u8 = 0x00;
const KIND_RECORD_GROUP: u8 = 0x01;
const KIND_TREE_MANIFEST: u8 = 0x03;
const KIND_PARTITION: u8 = 0x04;

const RECORD_VALUE: u8 = 0x00;
const RECORD_LOCATION: u8 = 0x01;
const RECORD_PAYLOAD: u8 = 0x02;

const SUB_HEADER: u8 = 0x00;
const SUB_SYNOPSIS: u8 = 0x01;
const SUB_STATE: u8 = 0x02;
const SUB_CENTROID: u8 = 0x03;
const SUB_LEAF_ENTRY: u8 = 0x04;
const SUB_CHILD_ENTRY: u8 = 0x05;

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
            Self::IndexIdAllocator | Self::IndexNameDirectory(_) => None,
            Self::Manifest(index) => Some(*index),
            Self::Record { index, .. }
            | Self::Location { index, .. }
            | Self::Payload { index, .. }
            | Self::TreeManifest { index, .. }
            | Self::Header { index, .. }
            | Self::Synopsis { index, .. }
            | Self::State { index, .. }
            | Self::Centroid { index, .. }
            | Self::LeafEntry { index, .. }
            | Self::ChildEntry { index, .. } => Some(*index),
        }
    }

    /// Returns the embedded Tree Key for a tree-local key.
    pub(crate) const fn tree_key(&self) -> Option<&TreeKey> {
        match self {
            Self::TreeManifest { tree_key, .. }
            | Self::Header { tree_key, .. }
            | Self::Synopsis { tree_key, .. }
            | Self::State { tree_key, .. }
            | Self::Centroid { tree_key, .. }
            | Self::LeafEntry { tree_key, .. }
            | Self::ChildEntry { tree_key, .. } => Some(tree_key),
            Self::IndexIdAllocator
            | Self::IndexNameDirectory(_)
            | Self::Manifest(_)
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
/// prefix is, because every key begins with the `0x01` version byte.
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

/// Pushes the version and scope bytes onto a key under construction.
fn push_version_scope(out: &mut Vec<u8>, scope: u8) {
    out.push(KEY_VERSION);
    out.push(scope);
}

/// Builds the common `[ version ][ index scope ][ index id ]` key prefix.
fn index_prefix(index: LogicalIndexId) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(2 + LOGICAL_INDEX_ID_BYTES);
    push_version_scope(&mut bytes, SCOPE_INDEX);
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
    let mut bytes = Vec::with_capacity(3);
    push_version_scope(&mut bytes, SCOPE_NAMESPACE);
    bytes.push(NS_INDEX_ID_ALLOCATOR);
    bytes
}

/// The Index Name directory key for `name`.
#[must_use]
pub fn name_directory_key(name: &IndexName) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(3 + name.as_str().len());
    push_version_scope(&mut bytes, SCOPE_NAMESPACE);
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
    2 + LOGICAL_INDEX_ID_BYTES + 1 + (2 * MAX_RECORD_ID_BYTES + 1) + 1
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

/// Returns one partition metadata key's exact encoded length.
pub(crate) fn partition_metadata_key_len(tree_key: &TreeKey) -> usize {
    // Index prefix, partition kind, Tree Key, Partition Key, and metadata subkind.
    2 + LOGICAL_INDEX_ID_BYTES + 1 + tree_key.as_bytes().len() + PARTITION_KEY_BYTES + 1
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
    if key.first() != Some(&KEY_VERSION) {
        return Err(corrupt());
    }
    let scope = *key.get(1).ok_or_else(corrupt)?;
    match scope {
        SCOPE_NAMESPACE => decode_namespace_key(&key[2..]),
        SCOPE_INDEX => decode_index_key(types, key, 2),
        _ => Err(corrupt()),
    }
}

/// Decodes a namespace-scope key body.
fn decode_namespace_key(body: &[u8]) -> Result<LogicalKey> {
    match body.first() {
        Some(&NS_INDEX_ID_ALLOCATOR) if body.len() == 1 => Ok(LogicalKey::IndexIdAllocator),
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
        SUB_STATE if terminal.is_empty() => Ok(LogicalKey::State {
            index,
            tree_key,
            partition,
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

/// The contiguous range of every key owned by one Logical Index.
#[must_use]
pub fn index_range(index: LogicalIndexId) -> KeyRange {
    let start = index_prefix(index);
    let end = successor(&start);
    KeyRange { start, end }
}

/// The contiguous range of every Tree Manifest directory key in one index.
#[must_use]
pub fn tree_manifest_range(index: LogicalIndexId) -> KeyRange {
    let mut start = index_prefix(index);
    start.push(KIND_TREE_MANIFEST);
    let end = successor(&start);
    KeyRange { start, end }
}

/// The contiguous range of every key owned by one partition.
#[must_use]
pub fn partition_range(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> KeyRange {
    let start = partition_prefix(index, tree_key, partition);
    let end = successor(&start);
    KeyRange { start, end }
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
    let end = successor(&start);
    KeyRange { start, end }
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
    let end = successor(&start);
    Ok(KeyRange { start, end })
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
    fn index_range_is_contiguous_across_adjacent_ids() {
        let one = index_range(LogicalIndexId::new(1).expect("one is nonzero"));
        let two = index_range(LogicalIndexId::new(2).expect("two is nonzero"));
        assert_eq!(one.end(), two.start());
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
