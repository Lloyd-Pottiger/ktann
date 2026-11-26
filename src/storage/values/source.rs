//! Schema-directed source records, before a Logical Index identity is allocated.

use bytes::Bytes;

use crate::api::{Error, IndexConfig, MAX_PAYLOAD_BYTES, Record, Result, Value, validate_id};
use crate::storage::keys::TreeKey;

use super::data::{
    decode_fields, decode_vector, decode_vector_into, encode_fields, encode_vector,
    validate_fields, validate_vector,
};
use super::record::{OpaquePayload, decode_record_id, encode_opaque_payload, encode_record_id};
use super::wire::{Decoder, Encoder};
use super::{IndexManifest, ValueKind, corrupt};

/// Projects the canonical `TreeKey` from positional `fields` using checked indexing.
pub(crate) fn project_tree_key(manifest: &IndexManifest, fields: &[Value]) -> Result<TreeKey> {
    let config = manifest.config();
    if config.tree_key_fields().is_empty() {
        return TreeKey::encode(&[], &[]);
    }
    let (types, count) = manifest.tree_key_types();
    let mut values = Vec::with_capacity(count);
    for field_id in config.tree_key_fields() {
        let val = fields
            .get(usize::from(field_id.0))
            .ok_or_else(Error::invalid_argument)?;
        values.push(val.clone());
    }
    TreeKey::encode(&types[..count], &values)
}

// This body is framed by the shared persistent bulk input format. Reuse
// the serving primitives so original vectors and typed fields have one canonical
// representation, without inventing an Index Manifest before name reservation.
pub(crate) fn encode(config: &IndexConfig, record: &mut Record) -> Result<Vec<u8>> {
    record.validate(config.dimension(), config.fields())?;
    let mut encoder = Encoder::new(ValueKind::VectorRecord);
    encode_record_id(&mut encoder, record.id())?;
    encode_vector(&mut encoder, config.dimension(), record.vector())?;
    encode_fields(&mut encoder, config.fields(), record.fields())?;
    encoder.bool(record.payload().is_some());
    if let Some(payload) = record.payload() {
        encoder.sized_bytes(payload, MAX_PAYLOAD_BYTES)?;
    }
    encoder.finish()
}

pub(crate) fn decode(config: &IndexConfig, bytes: Bytes) -> Result<Record> {
    let mut decoder = Decoder::framed(ValueKind::VectorRecord, bytes)?;
    let id = decode_record_id(&mut decoder)?;
    let vector = decode_vector(&mut decoder, config.dimension())?;
    let fields = decode_fields(&mut decoder, config.fields())?;
    let mut record = Record::new(id, vector, fields).map_err(|_| corrupt())?;
    if decoder.bool()? {
        record = record
            .with_payload(decoder.sized_bytes(MAX_PAYLOAD_BYTES)?)
            .map_err(|_| corrupt())?;
    }
    decoder.finish()?;
    Ok(record)
}

/// Decodes a canonical source frame into caller-reused `vector_out` without allocating
/// per-record `Box<[f32]>` buffers or pinning the raw frame buffer via `Bytes::slice`.
pub(crate) fn decode_into_buf(
    config: &IndexConfig,
    bytes: Bytes,
    vector_out: &mut [f32],
) -> Result<(Bytes, Box<[Value]>, Option<usize>)> {
    let mut decoder = Decoder::framed(ValueKind::VectorRecord, bytes)?;
    let id_view = decode_record_id(&mut decoder)?;
    validate_id(&id_view).map_err(|_| corrupt())?;
    let id = Bytes::copy_from_slice(&id_view);
    decode_vector_into(&mut decoder, config.dimension(), vector_out)?;
    let fields = decode_fields(&mut decoder, config.fields())?;
    let payload_len = if decoder.bool()? {
        Some(decoder.sized_bytes(MAX_PAYLOAD_BYTES)?.len())
    } else {
        None
    };
    decoder.finish()?;
    Ok((id, fields, payload_len))
}

/// Validates a canonical source frame in place, extracts its canonical `TreeKey` and canonical
/// field-projection bytes, and slices out the persistent `VectorRecord` wire body (plus optional
/// encoded `OpaquePayload`) without pinning large payload tails.
pub(crate) fn split_serving_bytes(
    manifest: &IndexManifest,
    bytes: Bytes,
) -> Result<(Bytes, TreeKey, Bytes, Bytes, Option<Bytes>)> {
    let config = manifest.config();
    let mut decoder = Decoder::framed(ValueKind::VectorRecord, bytes.clone())?;
    let id_view = decode_record_id(&mut decoder)?;
    validate_id(&id_view).map_err(|_| corrupt())?;
    let id = Bytes::copy_from_slice(&id_view);
    validate_vector(&mut decoder, config.dimension())?;
    let fields_start = decoder.position();
    let tree_key = if config.tree_key_fields().is_empty() {
        validate_fields(&mut decoder, config.fields())?;
        TreeKey::encode(&[], &[]).map_err(|_| corrupt())?
    } else {
        let fields = decode_fields(&mut decoder, config.fields())?;
        project_tree_key(manifest, &fields).map_err(|_| corrupt())?
    };
    let vec_record_end = decoder.position();
    let fields_bytes = Bytes::copy_from_slice(&bytes[fields_start..vec_record_end]);
    let record_bytes = if bytes.len().saturating_sub(vec_record_end) > 64 {
        Bytes::copy_from_slice(&bytes[..vec_record_end])
    } else {
        bytes.slice(..vec_record_end)
    };
    let payload_bytes = if decoder.bool()? {
        let raw_payload = decoder.sized_bytes(MAX_PAYLOAD_BYTES)?;
        let payload = OpaquePayload::new(raw_payload).map_err(|_| corrupt())?;
        let mut enc = Encoder::new(ValueKind::OpaquePayload);
        encode_opaque_payload(&mut enc, &payload)?;
        Some(Bytes::from(enc.finish()?))
    } else {
        None
    };
    decoder.finish()?;
    Ok((id, tree_key, fields_bytes, record_bytes, payload_bytes))
}

/// Encodes `(Record ID, typed fields)` for temporary coarse-shard spill files.
pub(crate) fn encode_row_meta(
    config: &IndexConfig,
    id: &Bytes,
    fields: &[Value],
) -> Result<Vec<u8>> {
    let mut encoder = Encoder::new(ValueKind::VectorRecord);
    encode_record_id(&mut encoder, id)?;
    encode_fields(&mut encoder, config.fields(), fields)?;
    encoder.finish()
}

/// Decodes `(Record ID, typed fields)` written by [`encode_row_meta`].
pub(crate) fn decode_row_meta(config: &IndexConfig, bytes: Bytes) -> Result<(Bytes, Box<[Value]>)> {
    let mut decoder = Decoder::framed(ValueKind::VectorRecord, bytes)?;
    let id_view = decode_record_id(&mut decoder)?;
    validate_id(&id_view).map_err(|_| corrupt())?;
    let id = Bytes::copy_from_slice(&id_view);
    let fields = decode_fields(&mut decoder, config.fields())?;
    decoder.finish()?;
    Ok((id, fields))
}
