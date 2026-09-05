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
//! string. The nested RaBitQ7 payload is governed by the Manifest's whole-format
//! version. Its 12-byte header and LSB-first bit streams use the format-v1
//! little-endian layout even though the enclosing value codec is big-endian.

use bytes::Bytes;

use crate::api::{Error, ErrorKind, MAX_ENCODED_SYNOPSIS_BYTES, Result};

use super::keys::{LogicalKey, TreeKey};

mod authority;
mod data;
mod entry;
mod manifest;
mod record;
mod synopsis;
mod wire;

#[doc(inline)]
pub use authority::{
    PartitionCentroid, PartitionHeader, PartitionState, PartitionTransition, TreeManifest,
};
#[doc(inline)]
pub use entry::{ChildEntry, LeafEntry};
#[doc(inline)]
pub use manifest::{
    BloomParameters, IndexIdAllocator, IndexLifecycle, IndexManifest, IndexNameEntry,
};
#[doc(inline)]
pub use record::{OpaquePayload, RecordLocation, VectorRecord};
#[doc(inline)]
pub use synopsis::{FieldSynopsis, PartitionSynopsis};

use wire::{Decoder, Encoder};

/// The whole persistent format version emitted and accepted by this build.
pub const FORMAT_VERSION: u16 = 1;

/// The logical value codec version emitted and accepted by this build.
pub const VALUE_CODEC_VERSION: u8 = 1;

/// The maximum encoded Opaque Payload size.
pub const MAX_PAYLOAD_BYTES: usize = crate::api::MAX_PAYLOAD_BYTES;

/// The maximum encoded Partition Synopsis size.
pub const MAX_SYNOPSIS_BYTES: usize = MAX_ENCODED_SYNOPSIS_BYTES;

/// Maximum encoded value lengths involved in one leaf relocation batch.
pub(crate) struct LeafRelocationValueSizes {
    pub(crate) leaf_entry: usize,
    pub(crate) record_location: usize,
    pub(crate) partition_header: usize,
    pub(crate) partition_synopsis: usize,
}

/// Returns exact worst-case value lengths for the current Manifest and Tree Key.
pub(crate) fn leaf_relocation_value_sizes(
    manifest: &IndexManifest,
    tree_key: &TreeKey,
) -> Result<LeafRelocationValueSizes> {
    Ok(LeafRelocationValueSizes {
        leaf_entry: entry::maximum_leaf_entry_encoded_len(manifest)?,
        record_location: record::record_location_encoded_len(tree_key),
        partition_header: authority::PARTITION_HEADER_ENCODED_LEN,
        partition_synopsis: manifest.maximum_synopsis_encoded_len(),
    })
}

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

// The typed logical transactions decode the value family a key implies, so a
// mismatched kind is unreachable in a correct backend but must stay
// fail-closed. These extractors are the single home of that check.
macro_rules! expect_variant {
    ($name:ident, $variant:ident, $ty:ty) => {
        /// Extracts the typed value from a typed read, failing closed on a
        /// wrong-kind value.
        pub(crate) fn $name(value: Option<PersistentValue>) -> Result<Option<$ty>> {
            match value {
                Some(PersistentValue::$variant(value)) => Ok(Some(value)),
                Some(_) => Err(corrupt()),
                None => Ok(None),
            }
        }
    };
}

expect_variant!(expect_record, VectorRecord, VectorRecord);
expect_variant!(expect_location, RecordLocation, RecordLocation);
expect_variant!(expect_header, PartitionHeader, PartitionHeader);
expect_variant!(expect_centroid, PartitionCentroid, PartitionCentroid);
expect_variant!(expect_leaf_entry, LeafEntry, LeafEntry);
expect_variant!(expect_child_entry, ChildEntry, ChildEntry);
expect_variant!(expect_synopsis, PartitionSynopsis, PartitionSynopsis);
expect_variant!(expect_state, PartitionState, PartitionTransition);

/// Extracts a borrowed Child Entry from a scanned typed value.
pub(crate) fn expect_child_entry_ref(value: &PersistentValue) -> Result<&ChildEntry> {
    match value {
        PersistentValue::ChildEntry(entry) => Ok(entry),
        _ => Err(corrupt()),
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
            PersistentValue::IndexIdAllocator(value) => {
                manifest::encode_index_id_allocator(&mut encoder, *value);
            }
            PersistentValue::IndexNameEntry(value) => {
                manifest::encode_index_name_entry(&mut encoder, *value);
            }
            PersistentValue::IndexManifest(value) => {
                manifest::encode_index_manifest(&mut encoder, value)?;
            }
            PersistentValue::TreeManifest(value) => {
                self.require_manifest()?;
                authority::encode_tree_manifest(&mut encoder, *value);
            }
            PersistentValue::VectorRecord(value) => {
                record::encode_vector_record(&mut encoder, self.require_manifest()?, value)?;
            }
            PersistentValue::OpaquePayload(value) => {
                self.require_manifest()?;
                record::encode_opaque_payload(&mut encoder, value)?;
            }
            PersistentValue::RecordLocation(value) => {
                record::encode_record_location(&mut encoder, self.require_manifest()?, value)?;
            }
            PersistentValue::PartitionHeader(value) => {
                self.require_manifest()?;
                authority::encode_partition_header(&mut encoder, *value);
            }
            PersistentValue::PartitionCentroid(value) => {
                authority::encode_partition_centroid(
                    &mut encoder,
                    self.require_manifest()?,
                    value,
                )?;
            }
            PersistentValue::ChildEntry(value) => {
                entry::encode_child_entry(&mut encoder, self.require_manifest()?, value)?;
            }
            PersistentValue::LeafEntry(value) => {
                entry::encode_leaf_entry(&mut encoder, self.require_manifest()?, value)?;
            }
            PersistentValue::PartitionSynopsis(value) => {
                synopsis::encode_partition_synopsis(&mut encoder, self.require_manifest()?, value)?;
            }
            PersistentValue::PartitionState(value) => {
                self.require_manifest()?;
                authority::encode_partition_state(&mut encoder, *value)?;
            }
        }
        encoder.finish()
    }

    /// Encodes a value after validating its typed key family and identity.
    pub(crate) fn encode_for_key(
        self,
        key: &LogicalKey,
        value: &PersistentValue,
    ) -> Result<Vec<u8>> {
        if value_kind_for_key(key) != value.kind()
            || validate_key_identity(key, value, self.manifest).is_err()
        {
            return Err(Error::invalid_argument());
        }
        self.encode(value)
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
            ValueKind::IndexIdAllocator => PersistentValue::IndexIdAllocator(
                manifest::decode_index_id_allocator(&mut decoder)?,
            ),
            ValueKind::IndexNameEntry => {
                PersistentValue::IndexNameEntry(manifest::decode_index_name_entry(&mut decoder)?)
            }
            ValueKind::IndexManifest => {
                PersistentValue::IndexManifest(manifest::decode_index_manifest(&mut decoder)?)
            }
            ValueKind::TreeManifest => {
                self.require_manifest()?;
                PersistentValue::TreeManifest(authority::decode_tree_manifest(&mut decoder)?)
            }
            ValueKind::VectorRecord => PersistentValue::VectorRecord(record::decode_vector_record(
                &mut decoder,
                self.require_manifest()?,
            )?),
            ValueKind::OpaquePayload => {
                self.require_manifest()?;
                PersistentValue::OpaquePayload(record::decode_opaque_payload(&mut decoder)?)
            }
            ValueKind::RecordLocation => PersistentValue::RecordLocation(
                record::decode_record_location(&mut decoder, self.require_manifest()?)?,
            ),
            ValueKind::PartitionHeader => {
                self.require_manifest()?;
                PersistentValue::PartitionHeader(authority::decode_partition_header(&mut decoder)?)
            }
            ValueKind::PartitionCentroid => PersistentValue::PartitionCentroid(
                authority::decode_partition_centroid(&mut decoder, self.require_manifest()?)?,
            ),
            ValueKind::ChildEntry => PersistentValue::ChildEntry(entry::decode_child_entry(
                &mut decoder,
                self.require_manifest()?,
            )?),
            ValueKind::LeafEntry => PersistentValue::LeafEntry(entry::decode_leaf_entry(
                &mut decoder,
                self.require_manifest()?,
            )?),
            ValueKind::PartitionSynopsis => PersistentValue::PartitionSynopsis(
                synopsis::decode_partition_synopsis(&mut decoder, self.require_manifest()?)?,
            ),
            ValueKind::PartitionState => {
                self.require_manifest()?;
                PersistentValue::PartitionState(authority::decode_partition_state(&mut decoder)?)
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
    if let (Some(expected), Some(actual)) = (manifest, key.index()) {
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
        (LogicalKey::Manifest(_), PersistentValue::IndexManifest(actual))
            if manifest.is_some_and(|expected| !expected.has_same_immutable_identity(actual)) =>
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

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

fn unsupported() -> Error {
    Error::new(ErrorKind::UnsupportedFormat)
}
