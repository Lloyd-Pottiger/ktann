//! Schema-directed source records, before a Logical Index identity is allocated.

use bytes::Bytes;

use crate::api::{IndexConfig, MAX_PAYLOAD_BYTES, Record, Result};

use super::data::{decode_fields, decode_vector, encode_fields, encode_vector};
use super::record::{decode_record_id, encode_record_id};
use super::wire::{Decoder, Encoder};
use super::{ValueKind, corrupt};

// This body is framed by the independently versioned bulk input format. Reuse
// the serving primitives so original vectors and typed fields have one canonical
// representation, without inventing an Index Manifest before name reservation.
pub(crate) fn encode(config: &IndexConfig, mut record: Record) -> Result<Vec<u8>> {
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
