//! Persistent Partition Synopsis values.

use std::cmp::Ordering;
use std::fmt;

use bytes::Bytes;

use crate::api::{DataType, Error, FieldSchema, MAX_STRING_BYTES, Result, Value};

use super::MAX_SYNOPSIS_BYTES;
use super::corrupt;
use super::data::{decode_typed_value, encode_typed_value};
use super::manifest::{BloomParameters, IndexManifest};
use super::wire::{Decoder, Encoder};

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

pub(super) fn encode_partition_synopsis(
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

pub(super) fn decode_partition_synopsis(
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
    let used_bits = parameters.bit_count() % 8;
    if used_bits != 0 {
        let mask = !((1_u8 << used_bits) - 1);
        if bytes.last().is_some_and(|byte| byte & mask != 0) {
            return Err(corrupt());
        }
    }
    Ok(())
}
