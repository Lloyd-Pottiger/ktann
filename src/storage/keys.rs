//! Canonical ordered codecs for logical key components and key families.
//!
//! This module owns the version-1 logical keyspace: the version framing, the
//! namespace/index scope tags, the typed entry-kind discriminators, the raw
//! identity and name components, and the memcomparable Tree Key tuple codec. It
//! defines no persistent values and no backend physical prefixes; adapters
//! prepend their own bounded prefix and value codecs live in a sibling module.
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
//! # Tree Key memcomparable codec
//!
//! A Tree Key is the ordered tuple of the declared non-null Tree Key fields.
//! Each scalar encodes to a self-delimiting memcomparable byte string whose
//! order matches typed comparison:
//!
//! * `Bool` is one byte: `0x00` for `false`, `0x01` for `true`.
//! * `I64` is eight big-endian bytes of `bits ^ 0x8000_0000_0000_0000`.
//! * `F64` is eight bytes: `bits | 0x8000_0000_0000_0000` for the non-negative
//!   half (including `+0.0`) and `!bits` for the negative half, matching the
//!   total order of finite `f64`. `-0.0` is noncanonical and rejected.
//! * `String` bytes escape `0x00` as `0x00 0xFF` and terminate with a single
//!   `0x00`.
//!
//! The tuple is the concatenation of its field encodings in field order. That
//! preserves typed order and makes the encoding of any leading fields a byte
//! prefix of every tuple sharing those fields, so a field-prefix half-open range
//! is an ordinary byte-prefix range. For a `String` field the byte-prefix range
//! is a conservative superset — it can additionally include Tree Keys whose
//! pinned `String` value is extended by a leading `0x00` byte — and exact
//! predicate evaluation later rejects those.
//!
//! # Fail closed
//!
//! Decoders reject an unknown version or scope, an unknown kind or subkind,
//! truncated or overlong components, noncanonical scalars (including `-0.0`),
//! invalid UTF-8, and trailing bytes after a terminal component. Every decode
//! failure returns [`ErrorKind::Corruption`]; encode-time rejection of invalid
//! caller input returns [`ErrorKind::InvalidArgument`].

use std::fmt;

use bytes::Bytes;

use crate::api::{
    DataType, Error, ErrorKind, IndexName, LogicalIndexId, PartitionKey, Result, Value,
};

/// The single logical-key format version emitted and accepted by this build.
pub const KEY_VERSION: u8 = 1;

/// The fixed encoded width of a [`LogicalIndexId`] in bytes.
pub const LOGICAL_INDEX_ID_BYTES: usize = 8;
/// The fixed encoded width of a [`PartitionKey`] in bytes.
pub const PARTITION_KEY_BYTES: usize = 8;

/// The maximum encoded length of a Record ID in bytes.
pub const MAX_RECORD_ID_BYTES: usize = 256;
/// The maximum encoded length of a single String Tree Key field in bytes.
pub const MAX_STRING_BYTES: usize = 1_024;
/// The maximum encoded length of a complete Tree Key in bytes.
pub const MAX_TREE_KEY_BYTES: usize = 8 * 1_024;

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

/// The high bit used by the memcomparable `I64` and `F64` transforms.
const SIGN: u64 = 0x8000_0000_0000_0000;

/// A storage-corruption error for a malformed or noncanonical decoded key.
fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// The canonical memcomparable encoding of one Tree Key.
///
/// The bytes are opaque and always interpreted through the schema-derived field
/// types; byte order matches typed comparison. `Debug` is redacted because a
/// Tree Key carries caller field values.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TreeKey(Vec<u8>);

impl TreeKey {
    /// Encodes `values` as a memcomparable Tree Key under the ordered `types`.
    ///
    /// `types` and `values` must have the same length, and every value must be a
    /// canonical non-`Null` value of its declared type. A `String` longer than
    /// [`MAX_STRING_BYTES`], a non-finite `F64`, or a `-0.0` is rejected as
    /// [`ErrorKind::InvalidArgument`], as is any result longer than
    /// [`MAX_TREE_KEY_BYTES`].
    pub fn encode(types: &[DataType], values: &[Value]) -> Result<Self> {
        if types.len() != values.len() {
            return Err(Error::invalid_argument());
        }
        let mut bytes = Vec::new();
        for (ty, value) in types.iter().zip(values) {
            encode_scalar(*ty, value, &mut bytes)?;
        }
        if bytes.len() > MAX_TREE_KEY_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(Self(bytes))
    }

    /// The canonical encoded bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Decodes this Tree Key into its typed field values.
    ///
    /// Malformed, truncated, noncanonical, overlong, or trailing bytes fail as
    /// [`ErrorKind::Corruption`].
    pub fn values(&self, types: &[DataType]) -> Result<Vec<Value>> {
        let (values, consumed) = decode_tree_key(types, &self.0)?;
        if consumed != self.0.len() {
            return Err(corrupt());
        }
        Ok(values)
    }

    /// Validates and owns an already encoded Tree Key.
    ///
    /// Malformed, truncated, noncanonical, overlong, or trailing bytes fail as
    /// [`ErrorKind::Corruption`].
    pub(super) fn from_encoded(types: &[DataType], bytes: &[u8]) -> Result<Self> {
        decode_tree_key_only(types, bytes)
    }
}

impl fmt::Debug for TreeKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TreeKey([REDACTED])")
    }
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
fn successor(prefix: &[u8]) -> Vec<u8> {
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

/// Reads exactly `N` bytes as an array, failing closed on truncation.
fn take_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    let slice = bytes.get(..N).ok_or_else(corrupt)?;
    slice.try_into().map_err(|_| corrupt())
}

/// Encodes one scalar Tree Key field into `out`.
fn encode_scalar(ty: DataType, value: &Value, out: &mut Vec<u8>) -> Result<()> {
    match (ty, value) {
        (DataType::Bool, Value::Bool(value)) => out.push(u8::from(*value)),
        (DataType::I64, Value::I64(value)) => {
            out.extend_from_slice(&((*value as u64) ^ SIGN).to_be_bytes());
        }
        (DataType::F64, Value::F64(value)) => {
            if !value.is_finite() || (*value == 0.0 && value.is_sign_negative()) {
                return Err(Error::invalid_argument());
            }
            let bits = value.to_bits();
            let encoded = if bits & SIGN != 0 { !bits } else { bits | SIGN };
            out.extend_from_slice(&encoded.to_be_bytes());
        }
        (DataType::String, Value::String(value)) => {
            if value.len() > MAX_STRING_BYTES {
                return Err(Error::invalid_argument());
            }
            for &byte in value.as_bytes() {
                if byte == 0x00 {
                    out.extend_from_slice(&[0x00, 0xFF]);
                } else {
                    out.push(byte);
                }
            }
            out.push(0x00);
        }
        _ => return Err(Error::invalid_argument()),
    }
    Ok(())
}

/// Decodes one scalar Tree Key field, returning its value and consumed bytes.
fn decode_scalar(ty: DataType, bytes: &[u8]) -> Result<(Value, usize)> {
    match ty {
        DataType::Bool => match bytes.first() {
            Some(0x00) => Ok((Value::Bool(false), 1)),
            Some(0x01) => Ok((Value::Bool(true), 1)),
            _ => Err(corrupt()),
        },
        DataType::I64 => {
            let raw = take_array::<8>(bytes)?;
            Ok((Value::I64((u64::from_be_bytes(raw) ^ SIGN) as i64), 8))
        }
        DataType::F64 => {
            let encoded = u64::from_be_bytes(take_array::<8>(bytes)?);
            let bits = if encoded & SIGN != 0 {
                encoded ^ SIGN
            } else {
                !encoded
            };
            let value = f64::from_bits(bits);
            if !value.is_finite() || (value == 0.0 && value.is_sign_negative()) {
                return Err(corrupt());
            }
            Ok((Value::F64(value), 8))
        }
        DataType::String => {
            let mut value = Vec::new();
            let mut offset = 0;
            loop {
                let byte = *bytes.get(offset).ok_or_else(corrupt)?;
                if byte == 0x00 {
                    if bytes.get(offset + 1) == Some(&0xFF) {
                        value.push(0x00);
                        offset += 2;
                    } else {
                        offset += 1;
                        break;
                    }
                } else {
                    value.push(byte);
                    offset += 1;
                }
                if value.len() > MAX_STRING_BYTES {
                    return Err(corrupt());
                }
            }
            let string = String::from_utf8(value).map_err(|_| corrupt())?;
            Ok((Value::String(string), offset))
        }
    }
}

/// Decodes a Tree Key into its values and the number of bytes it consumed.
fn decode_tree_key(types: &[DataType], bytes: &[u8]) -> Result<(Vec<Value>, usize)> {
    if bytes.len() > MAX_TREE_KEY_BYTES {
        return Err(corrupt());
    }
    let mut values = Vec::with_capacity(types.len());
    let mut offset = 0;
    for ty in types {
        let (value, consumed) = decode_scalar(*ty, &bytes[offset..])?;
        offset += consumed;
        values.push(value);
    }
    Ok((values, offset))
}

/// Rejects an invalid Record ID on the encode path.
fn check_record_id(id: &[u8]) -> Result<()> {
    if id.is_empty() || id.len() > MAX_RECORD_ID_BYTES {
        return Err(Error::invalid_argument());
    }
    Ok(())
}

/// Decodes a terminal Record ID.
fn decode_record_id(bytes: &[u8]) -> Result<Bytes> {
    if bytes.is_empty() || bytes.len() > MAX_RECORD_ID_BYTES {
        return Err(corrupt());
    }
    Ok(Bytes::copy_from_slice(bytes))
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
    for &byte in id {
        if byte == 0 {
            bytes.extend_from_slice(&[0, 0xff]);
        } else {
            bytes.push(byte);
        }
    }
    bytes.push(0);
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

/// Decodes a complete logical key.
///
/// `types` is the ordered Tree Key field types and is required to split the
/// Tree Key from its trailing components; it must match the Logical Index's
/// immutable schema.
pub fn decode_key(types: &[DataType], key: &[u8]) -> Result<LogicalKey> {
    if key.first() != Some(&KEY_VERSION) {
        return Err(corrupt());
    }
    let scope = *key.get(1).ok_or_else(corrupt)?;
    let body = &key[2..];
    match scope {
        SCOPE_NAMESPACE => decode_namespace_key(body),
        SCOPE_INDEX => decode_index_key(types, body),
        _ => Err(corrupt()),
    }
}

/// Decodes a namespace-scope key body.
fn decode_namespace_key(body: &[u8]) -> Result<LogicalKey> {
    match body.first() {
        Some(&NS_INDEX_ID_ALLOCATOR) if body.len() == 1 => Ok(LogicalKey::IndexIdAllocator),
        Some(&NS_INDEX_ID_ALLOCATOR) => Err(corrupt()),
        Some(&NS_INDEX_NAME_DIRECTORY) => {
            Ok(LogicalKey::IndexNameDirectory(decode_name(&body[1..])?))
        }
        _ => Err(corrupt()),
    }
}

/// Decodes an index-scope key body.
fn decode_index_key(types: &[DataType], body: &[u8]) -> Result<LogicalKey> {
    let index =
        LogicalIndexId::new(u64::from_be_bytes(take_array::<8>(body)?)).map_err(|_| corrupt())?;
    let kind = *body.get(LOGICAL_INDEX_ID_BYTES).ok_or_else(corrupt)?;
    let rest = &body[LOGICAL_INDEX_ID_BYTES + 1..];

    match kind {
        KIND_MANIFEST if rest.is_empty() => Ok(LogicalKey::Manifest(index)),
        KIND_MANIFEST => Err(corrupt()),
        KIND_RECORD_GROUP => decode_record_group(index, rest),
        KIND_TREE_MANIFEST => Ok(LogicalKey::TreeManifest {
            index,
            tree_key: decode_tree_key_only(types, rest)?,
        }),
        KIND_PARTITION => decode_partition_key(types, index, rest),
        _ => Err(corrupt()),
    }
}

fn decode_record_group(index: LogicalIndexId, bytes: &[u8]) -> Result<LogicalKey> {
    let mut id = Vec::new();
    let mut offset = 0;
    loop {
        let byte = *bytes.get(offset).ok_or_else(corrupt)?;
        if byte == 0 {
            if bytes.get(offset + 1) == Some(&0xff) {
                id.push(0);
                offset += 2;
            } else {
                offset += 1;
                break;
            }
        } else {
            id.push(byte);
            offset += 1;
        }
        if id.len() > MAX_RECORD_ID_BYTES {
            return Err(corrupt());
        }
    }
    let id = decode_record_id(&id)?;
    match bytes.get(offset..) {
        Some([RECORD_VALUE]) => Ok(LogicalKey::Record { index, id }),
        Some([RECORD_LOCATION]) => Ok(LogicalKey::Location { index, id }),
        Some([RECORD_PAYLOAD]) => Ok(LogicalKey::Payload { index, id }),
        _ => Err(corrupt()),
    }
}

/// Decodes a terminal Tree Key, rejecting trailing bytes.
fn decode_tree_key_only(types: &[DataType], bytes: &[u8]) -> Result<TreeKey> {
    let (_, consumed) = decode_tree_key(types, bytes)?;
    if consumed != bytes.len() {
        return Err(corrupt());
    }
    Ok(TreeKey(bytes.to_vec()))
}

/// Decodes the remainder of a partition-scoped key body.
fn decode_partition_key(
    types: &[DataType],
    index: LogicalIndexId,
    rest: &[u8],
) -> Result<LogicalKey> {
    let (_, tree_key_len) = decode_tree_key(types, rest)?;
    let (tree_key_bytes, after) = rest.split_at(tree_key_len);
    let tree_key = TreeKey(tree_key_bytes.to_vec());

    let partition =
        PartitionKey::new(u64::from_be_bytes(take_array::<8>(after)?)).map_err(|_| corrupt())?;
    let subkind = *after.get(PARTITION_KEY_BYTES).ok_or_else(corrupt)?;
    let terminal = &after[PARTITION_KEY_BYTES + 1..];

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
            id: decode_record_id(terminal)?,
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
    for (ty, value) in types.iter().zip(prefix) {
        encode_scalar(*ty, value, &mut start)?;
    }
    let end = successor(&start);
    Ok(KeyRange { start, end })
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
}
