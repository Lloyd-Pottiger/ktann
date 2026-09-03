//! Persistent Child and Leaf Entry values.

use std::fmt;

use bytes::Bytes;

use crate::api::{Error, PartitionKey, Result, Value};
use crate::search::rabitq::RaBitQ7;
use crate::storage::keys::MAX_RECORD_ID_BYTES;

use super::corrupt;
use super::data::{
    decode_fields, decode_vector, encode_fields, encode_vector, maximum_typed_value_len,
};
use super::manifest::IndexManifest;
use super::wire::{Decoder, Encoder};

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

    /// Consumes the entry and returns its Record ID, filter projection, and
    /// RaBitQ7 bytes without copying.
    #[must_use]
    pub fn into_parts(self) -> (Bytes, Box<[Value]>, Bytes) {
        (self.record_id, self.fields, self.rabitq7)
    }
}

impl fmt::Debug for LeafEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LeafEntry([REDACTED])")
    }
}

pub(super) fn encode_child_entry(
    encoder: &mut Encoder,
    manifest: &IndexManifest,
    entry: &ChildEntry,
) -> Result<()> {
    encoder.u64(entry.child.get());
    encode_vector(encoder, manifest.config().dimension(), entry.centroid())
}

/// Returns the exact encoded Child Entry length for one Manifest.
pub(super) fn child_entry_encoded_len(manifest: &IndexManifest) -> Result<usize> {
    // Frame, Child Partition Key, vector length, and full-f32 components.
    manifest
        .config()
        .dimension()
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(2 + 8 + 4))
        .ok_or_else(Error::invalid_argument)
}

pub(super) fn decode_child_entry(
    decoder: &mut Decoder,
    manifest: &IndexManifest,
) -> Result<ChildEntry> {
    let child = decoder.partition_key()?;
    let centroid = decode_vector(decoder, manifest.config().dimension())?;
    Ok(ChildEntry::new(child, centroid))
}

pub(super) fn encode_leaf_entry(
    encoder: &mut Encoder,
    manifest: &IndexManifest,
    entry: &LeafEntry,
) -> Result<()> {
    if entry.record_id.is_empty() || entry.record_id.len() > MAX_RECORD_ID_BYTES {
        return Err(Error::invalid_argument());
    }
    encoder.sized_u16_bytes(&entry.record_id)?;
    encode_fields(encoder, manifest.config().fields(), &entry.fields)?;
    let expected = RaBitQ7::encoded_len(manifest.config().dimension())?;
    if entry.rabitq7.len() != expected {
        return Err(Error::invalid_argument());
    }
    RaBitQ7::validate(&entry.rabitq7, manifest.config().dimension())
        .map_err(|_| Error::invalid_argument())?;
    encoder.sized_bytes(&entry.rabitq7, expected)
}

/// Returns the exact maximum encoded Leaf Entry length for one Manifest.
pub(super) fn maximum_leaf_entry_encoded_len(manifest: &IndexManifest) -> Result<usize> {
    let field_bytes = manifest
        .config()
        .fields()
        .iter()
        .try_fold(0_usize, |size, field| {
            size.checked_add(maximum_typed_value_len(field.data_type()))
                .ok_or_else(Error::invalid_argument)
        })?;
    let rabitq7_bytes = RaBitQ7::encoded_len(manifest.config().dimension())?;

    // Frame, sized Record ID, field count, typed fields, and sized RaBitQ7.
    2_usize
        .checked_add(2 + MAX_RECORD_ID_BYTES)
        .and_then(|size| size.checked_add(2 + field_bytes))
        .and_then(|size| size.checked_add(4 + rabitq7_bytes))
        .ok_or_else(Error::invalid_argument)
}

pub(super) fn decode_leaf_entry(
    decoder: &mut Decoder,
    manifest: &IndexManifest,
) -> Result<LeafEntry> {
    let record_id = decoder.sized_u16_bytes(MAX_RECORD_ID_BYTES)?;
    if record_id.is_empty() {
        return Err(corrupt());
    }
    let fields = decode_fields(decoder, manifest.config().fields())?;
    let expected = RaBitQ7::encoded_len(manifest.config().dimension()).map_err(|_| corrupt())?;
    let rabitq7 = decoder.sized_bytes(expected)?;
    if rabitq7.len() != expected {
        return Err(corrupt());
    }
    RaBitQ7::validate(&rabitq7, manifest.config().dimension())?;
    Ok(LeafEntry::new(record_id, fields, rabitq7))
}
