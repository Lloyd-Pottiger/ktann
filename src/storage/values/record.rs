//! Persistent Vector Record ownership values.

use std::fmt;

use bytes::Bytes;

use crate::api::{Error, PartitionKey, Result, Value};

use crate::storage::keys::{MAX_RECORD_ID_BYTES, MAX_TREE_KEY_BYTES, TreeKey};

use super::data::{decode_fields, decode_vector, encode_fields, encode_vector};
use super::manifest::IndexManifest;
use super::wire::{Decoder, Encoder};
use super::{MAX_PAYLOAD_BYTES, corrupt};

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

    /// Consumes the record body and returns its owned parts.
    #[must_use]
    pub fn into_parts(self) -> (Bytes, Box<[f32]>, Box<[Value]>) {
        (self.record_id, self.vector, self.fields)
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

    /// Consumes the payload and returns its opaque bytes.
    #[must_use]
    pub fn into_bytes(self) -> Bytes {
        self.0
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

    /// Consumes the location and returns its Tree Key and leaf Partition Key.
    #[must_use]
    pub fn into_parts(self) -> (TreeKey, PartitionKey) {
        (self.tree_key, self.leaf)
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

pub(super) fn encode_vector_record(
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

pub(super) fn decode_vector_record(
    decoder: &mut Decoder,
    manifest: &IndexManifest,
) -> Result<VectorRecord> {
    let record_id = decoder.sized_u16_bytes(MAX_RECORD_ID_BYTES)?;
    if record_id.is_empty() {
        return Err(corrupt());
    }
    let vector = decode_vector(decoder, manifest.config().dimension())?;
    let fields = decode_fields(decoder, manifest.config().fields())?;
    Ok(VectorRecord::new(record_id, vector, fields))
}

fn validate_tree_key(manifest: &IndexManifest, tree_key: &TreeKey) -> Result<()> {
    let (types, type_count) = manifest.tree_key_types();
    tree_key
        .validate(&types[..type_count])
        .map_err(|_| Error::invalid_argument())
}

pub(super) fn encode_opaque_payload(encoder: &mut Encoder, payload: &OpaquePayload) -> Result<()> {
    encoder.sized_bytes(payload.as_bytes(), MAX_PAYLOAD_BYTES)
}

pub(super) fn decode_opaque_payload(decoder: &mut Decoder) -> Result<OpaquePayload> {
    let payload = decoder.sized_bytes(MAX_PAYLOAD_BYTES)?;
    OpaquePayload::new(payload).map_err(|_| corrupt())
}

pub(super) fn encode_record_location(
    encoder: &mut Encoder,
    manifest: &IndexManifest,
    location: &RecordLocation,
) -> Result<()> {
    validate_tree_key(manifest, location.tree_key())?;
    encoder.sized_bytes(location.tree_key().as_bytes(), MAX_TREE_KEY_BYTES)?;
    encoder.u64(location.leaf.get());
    Ok(())
}

pub(super) fn decode_record_location(
    decoder: &mut Decoder,
    manifest: &IndexManifest,
) -> Result<RecordLocation> {
    let bytes = decoder.sized_bytes(MAX_TREE_KEY_BYTES)?;
    let (types, type_count) = manifest.tree_key_types();
    let tree_key = TreeKey::from_encoded(&types[..type_count], bytes)?;
    let leaf = decoder.partition_key()?;
    Ok(RecordLocation::new(tree_key, leaf))
}
