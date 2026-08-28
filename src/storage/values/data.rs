//! Shared schema-directed vector and field encoding.

use crate::api::{DataType, Error, FieldSchema, MAX_STRING_BYTES, Result, Value};

use super::corrupt;
use super::wire::{Decoder, Encoder};

/// Returns the exact maximum encoded length of one non-NULL typed value.
pub(super) const fn maximum_typed_value_len(data_type: DataType) -> usize {
    match data_type {
        DataType::Bool => 2,
        DataType::I64 | DataType::F64 => 9,
        DataType::String => 1 + 4 + MAX_STRING_BYTES,
    }
}

pub(super) fn encode_vector(encoder: &mut Encoder, dimension: usize, vector: &[f32]) -> Result<()> {
    if vector.len() != dimension {
        return Err(Error::invalid_argument());
    }
    let encoded_bytes = dimension
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or_else(Error::invalid_argument)?;
    encoder.reserve(encoded_bytes);
    encoder.u32(u32::try_from(dimension).map_err(|_| Error::invalid_argument())?);
    for component in vector {
        encoder.f32(*component)?;
    }
    Ok(())
}

pub(super) fn decode_vector(decoder: &mut Decoder, dimension: usize) -> Result<Box<[f32]>> {
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

pub(super) fn encode_fields(
    encoder: &mut Encoder,
    schema: &[FieldSchema],
    fields: &[Value],
) -> Result<()> {
    if fields.len() != schema.len() {
        return Err(Error::invalid_argument());
    }
    encoder.u16(u16::try_from(fields.len()).map_err(|_| Error::invalid_argument())?);
    for (value, field) in fields.iter().zip(schema) {
        encode_typed_value(encoder, field.data_type(), field.is_nullable(), value)?;
    }
    Ok(())
}

pub(super) fn decode_fields(decoder: &mut Decoder, schema: &[FieldSchema]) -> Result<Box<[Value]>> {
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

pub(super) fn encode_typed_value(
    encoder: &mut Encoder,
    data_type: DataType,
    nullable: bool,
    value: &Value,
) -> Result<()> {
    visit_typed_value_bytes(data_type, nullable, value, |bytes| encoder.bytes(bytes))
}

/// Visits the canonical wire chunks for one schema-directed typed value.
pub(super) fn visit_typed_value_bytes(
    data_type: DataType,
    nullable: bool,
    value: &Value,
    mut visit: impl FnMut(&[u8]),
) -> Result<()> {
    match value {
        Value::Null if nullable => visit(&[0]),
        Value::Bool(value) if data_type == DataType::Bool => {
            visit(&[1, u8::from(*value)]);
        }
        Value::I64(value) if data_type == DataType::I64 => {
            visit(&[2]);
            visit(&value.to_be_bytes());
        }
        Value::F64(value) if data_type == DataType::F64 => {
            if !value.is_finite() {
                return Err(Error::invalid_argument());
            }
            visit(&[3]);
            let bits = if *value == 0.0 { 0 } else { value.to_bits() };
            visit(&bits.to_be_bytes());
        }
        Value::String(value) if data_type == DataType::String => {
            if value.len() > MAX_STRING_BYTES {
                return Err(Error::invalid_argument());
            }
            visit(&[4]);
            visit(
                &u32::try_from(value.len())
                    .map_err(|_| Error::invalid_argument())?
                    .to_be_bytes(),
            );
            visit(value.as_bytes());
        }
        _ => return Err(Error::invalid_argument()),
    }
    Ok(())
}

pub(super) fn decode_typed_value(
    decoder: &mut Decoder,
    data_type: DataType,
    nullable: bool,
) -> Result<Value> {
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
