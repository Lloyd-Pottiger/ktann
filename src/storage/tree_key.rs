//! Canonical memcomparable encoding for Tree Keys.
//!
//! A Tree Key is the ordered tuple of the declared non-null Tree Key fields.
//! Each scalar encodes to a self-delimiting byte string whose order matches
//! typed comparison:
//!
//! * `Bool` is one byte: `0x00` for `false`, `0x01` for `true`.
//! * `I64` is eight big-endian bytes of `bits ^ 0x8000_0000_0000_0000`.
//! * `F64` is eight bytes: `bits | 0x8000_0000_0000_0000` for non-negative
//!   values and `!bits` for negative values. `-0.0` is noncanonical.
//! * `String` escapes `0x00` as `0x00 0xFF` and ends with a single `0x00`.
//!
//! Concatenating fields preserves tuple order and makes the encoding of leading
//! fields a byte prefix of every Tree Key sharing those fields. A `String`
//! field-prefix range is a conservative superset because it may also include a
//! value extended by a leading `0x00`; exact predicate evaluation rejects that
//! case later.

use std::fmt;

use crate::api::{DataType, Error, ErrorKind, Result, Value};

/// The maximum encoded length of a single String Tree Key field in bytes.
pub const MAX_STRING_BYTES: usize = crate::api::MAX_STRING_BYTES;
/// The maximum encoded length of a complete Tree Key in bytes.
pub const MAX_TREE_KEY_BYTES: usize = 8 * 1_024;

/// The high bit used by the memcomparable `I64` and `F64` transforms.
const SIGN: u64 = 0x8000_0000_0000_0000;

/// A storage-corruption error for malformed or noncanonical encoded fields.
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
        let mut bytes = Vec::new();
        Self::append_fields(types, values, &mut bytes)?;
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
        decode_complete(types, &self.0)
    }

    /// Validates this complete encoding against ordered field types.
    pub(super) fn validate(&self, types: &[DataType]) -> Result<()> {
        decode_complete(types, &self.0).map(|_| ())
    }

    /// Appends an ordered field sequence without an intermediate key buffer.
    pub(super) fn append_fields(
        types: &[DataType],
        values: &[Value],
        out: &mut Vec<u8>,
    ) -> Result<()> {
        if types.len() != values.len() {
            return Err(Error::invalid_argument());
        }
        for (ty, value) in types.iter().zip(values) {
            encode_scalar(*ty, value, out)?;
        }
        Ok(())
    }

    /// Validates and owns an encoded Tree Key prefix from a larger key.
    pub(super) fn from_prefix(types: &[DataType], bytes: &[u8]) -> Result<(Self, usize)> {
        let (_, consumed) = decode_fields(types, bytes)?;
        Ok((Self(bytes[..consumed].to_vec()), consumed))
    }

    /// Validates and owns an already encoded complete Tree Key.
    ///
    /// Malformed, truncated, noncanonical, overlong, or trailing bytes fail as
    /// [`ErrorKind::Corruption`].
    pub(super) fn from_encoded(types: &[DataType], bytes: &[u8]) -> Result<Self> {
        decode_complete(types, bytes)?;
        Ok(Self(bytes.to_vec()))
    }
}

impl fmt::Debug for TreeKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TreeKey([REDACTED])")
    }
}

/// Reads exactly `N` bytes as an array, failing closed on truncation.
pub(super) fn take_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    let slice = bytes.get(..N).ok_or_else(corrupt)?;
    slice.try_into().map_err(|_| corrupt())
}

/// Appends `bytes` with the v1 tuple escaping: `0x00` becomes `0x00 0xFF` and
/// a single `0x00` terminates the string.
pub(super) fn push_escaped_terminated(out: &mut Vec<u8>, bytes: &[u8]) {
    for &byte in bytes {
        if byte == 0x00 {
            out.extend_from_slice(&[0x00, 0xFF]);
        } else {
            out.push(byte);
        }
    }
    out.push(0x00);
}

/// Decodes one tuple-escaped string, returning its raw bytes and the consumed
/// length including the terminator. Truncated input, a missing terminator, or
/// a decoded string longer than `maximum` fails closed.
pub(super) fn decode_escaped_terminated(bytes: &[u8], maximum: usize) -> Result<(Vec<u8>, usize)> {
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
        if value.len() > maximum {
            return Err(corrupt());
        }
    }
    Ok((value, offset))
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
            push_escaped_terminated(out, value.as_bytes());
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
            let (value, offset) = decode_escaped_terminated(bytes, MAX_STRING_BYTES)?;
            let string = String::from_utf8(value).map_err(|_| corrupt())?;
            Ok((Value::String(string), offset))
        }
    }
}

/// Decodes a complete Tree Key encoding, rejecting trailing bytes.
fn decode_complete(types: &[DataType], bytes: &[u8]) -> Result<Vec<Value>> {
    let (values, consumed) = decode_fields(types, bytes)?;
    if consumed != bytes.len() {
        return Err(corrupt());
    }
    Ok(values)
}

/// Decodes a Tree Key field sequence and reports the consumed prefix length.
fn decode_fields(types: &[DataType], bytes: &[u8]) -> Result<(Vec<Value>, usize)> {
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
