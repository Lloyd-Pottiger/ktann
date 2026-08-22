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

use bytes::Bytes;
use xxhash_rust::xxh3::xxh3_128_with_seed;

use crate::api::{DataType, Error, ErrorKind, Result, Value};

/// The maximum encoded length of a single String Tree Key field in bytes.
pub const MAX_STRING_BYTES: usize = crate::api::MAX_STRING_BYTES;
/// The maximum encoded length of a complete Tree Key in bytes.
pub const MAX_TREE_KEY_BYTES: usize = 8 * 1_024;

/// The high bit used by the memcomparable `I64` and `F64` transforms.
const SIGN: u64 = 0x8000_0000_0000_0000;

/// Domain-separates the redacted Tree Key hash from every other hash.
const TREE_KEY_HASH_DOMAIN: u64 = 0x4b54_414e_4e01_b1a0;
/// The second pass domain deriving the upper 16 hash bytes.
const TREE_KEY_HASH_SECOND_DOMAIN: u64 = 0x4b54_414e_4e01_b1a1;

/// A storage-corruption error for malformed or noncanonical encoded fields.
fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// The canonical memcomparable encoding of one Tree Key.
///
/// The bytes are opaque and always interpreted through the schema-derived field
/// types; byte order matches typed comparison. The buffer is reference-counted,
/// so decoding a persistent key shares the scanned key's allocation instead of
/// copying. `Debug` is redacted because a Tree Key carries caller field values.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TreeKey(Bytes);

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
        Ok(Self(Bytes::from(bytes)))
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
        check_complete(types, &self.0)
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

    /// Validates and shares an encoded Tree Key prefix of a larger key.
    ///
    /// The returned Tree Key borrows the leading `consumed` bytes of `bytes`
    /// instead of copying them.
    pub(super) fn from_prefix(types: &[DataType], bytes: &Bytes) -> Result<(Self, usize)> {
        let consumed = check_fields(types, bytes)?;
        Ok((Self(bytes.slice(..consumed)), consumed))
    }

    /// Validates and shares an already encoded complete Tree Key.
    ///
    /// Malformed, truncated, noncanonical, overlong, or trailing bytes fail as
    /// [`ErrorKind::Corruption`]. The Tree Key borrows `bytes` instead of
    /// copying it.
    pub(super) fn from_encoded(types: &[DataType], bytes: Bytes) -> Result<Self> {
        check_complete(types, &bytes)?;
        Ok(Self(bytes))
    }
}

impl fmt::Debug for TreeKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TreeKey([REDACTED])")
    }
}

/// The stable redacted Tree Key hash attached to verification issues and
/// trace spans (ADR 0019, design `runtime-operations.md` section 5).
#[must_use]
pub(crate) fn tree_key_hash(tree_key: &TreeKey) -> [u8; 32] {
    let first = xxh3_128_with_seed(tree_key.as_bytes(), TREE_KEY_HASH_DOMAIN);
    let second = xxh3_128_with_seed(
        &first.to_le_bytes(),
        TREE_KEY_HASH_SECOND_DOMAIN ^ (first as u64),
    );
    let mut hash = [0; 32];
    hash[..16].copy_from_slice(&first.to_le_bytes());
    hash[16..].copy_from_slice(&second.to_le_bytes());
    hash
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

/// Decodes one tuple-escaped string whose layout was already validated by
/// [`scan_escaped_terminated`], returning its raw bytes in one exactly sized
/// buffer. The scan guarantees the input is well-formed, so the walk cannot
/// fail.
pub(super) fn decode_escaped_terminated(bytes: &[u8], scan: &EscapedScan) -> Vec<u8> {
    debug_assert!(scan.consumed <= bytes.len());
    let mut value = Vec::with_capacity(scan.decoded_len);
    let mut offset = 0;
    while offset < scan.consumed - 1 {
        let byte = bytes[offset];
        if byte == 0x00 {
            value.push(0x00);
            offset += 2;
        } else {
            value.push(byte);
            offset += 1;
        }
    }
    value
}

/// The layout of one tuple-escaped string, scanned without decoding it.
pub(super) struct EscapedScan {
    /// The decoded length in bytes.
    pub(super) decoded_len: usize,
    /// The consumed length including the terminator.
    pub(super) consumed: usize,
    /// Whether the encoded form contains any `0x00 0xFF` escape pair.
    pub(super) escaped: bool,
}

/// Scans one tuple-escaped string without materializing it. Truncated input, a
/// missing terminator, or a decoded string longer than `maximum` fails closed.
pub(super) fn scan_escaped_terminated(bytes: &[u8], maximum: usize) -> Result<EscapedScan> {
    let mut scan = EscapedScan {
        decoded_len: 0,
        consumed: 0,
        escaped: false,
    };
    loop {
        let byte = *bytes.get(scan.consumed).ok_or_else(corrupt)?;
        if byte == 0x00 {
            if bytes.get(scan.consumed + 1) == Some(&0xFF) {
                scan.escaped = true;
                scan.consumed += 2;
            } else {
                scan.consumed += 1;
                return Ok(scan);
            }
        } else {
            scan.consumed += 1;
        }
        scan.decoded_len += 1;
        if scan.decoded_len > maximum {
            return Err(corrupt());
        }
    }
}

/// Validates one tuple-escaped String field without materializing it, returning
/// the consumed length including the terminator.
///
/// The escape pair decodes to `0x00`, an ASCII byte that no multi-byte UTF-8
/// sequence contains, so the decoded string is valid UTF-8 exactly when every
/// literal segment between escapes is; validating the borrowed segments keeps
/// the check allocation-free while accepting the same encodings as
/// [`decode_scalar`].
fn check_string(bytes: &[u8]) -> Result<usize> {
    let mut decoded = 0;
    let mut offset = 0;
    let mut segment = 0;
    loop {
        let byte = *bytes.get(offset).ok_or_else(corrupt)?;
        if byte == 0x00 {
            std::str::from_utf8(&bytes[segment..offset]).map_err(|_| corrupt())?;
            if bytes.get(offset + 1) == Some(&0xFF) {
                offset += 2;
                segment = offset;
            } else {
                return Ok(offset + 1);
            }
        } else {
            offset += 1;
        }
        decoded += 1;
        if decoded > MAX_STRING_BYTES {
            return Err(corrupt());
        }
    }
}

/// Validates one scalar Tree Key field without materializing its value,
/// returning the consumed length. Byte-exact with [`decode_scalar`]: both
/// accept exactly the canonical encodings.
fn check_scalar(ty: DataType, bytes: &[u8]) -> Result<usize> {
    match ty {
        DataType::Bool => match bytes.first() {
            Some(0x00 | 0x01) => Ok(1),
            _ => Err(corrupt()),
        },
        DataType::I64 => {
            take_array::<8>(bytes)?;
            Ok(8)
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
            Ok(8)
        }
        DataType::String => check_string(bytes),
    }
}

/// Validates a Tree Key field sequence and reports the consumed prefix length.
fn check_fields(types: &[DataType], bytes: &[u8]) -> Result<usize> {
    if bytes.len() > MAX_TREE_KEY_BYTES {
        return Err(corrupt());
    }
    let mut offset = 0;
    for ty in types {
        offset += check_scalar(*ty, &bytes[offset..])?;
    }
    Ok(offset)
}

/// Validates a complete Tree Key encoding, rejecting trailing bytes.
fn check_complete(types: &[DataType], bytes: &[u8]) -> Result<()> {
    let consumed = check_fields(types, bytes)?;
    if consumed != bytes.len() {
        return Err(corrupt());
    }
    Ok(())
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
            let scan = scan_escaped_terminated(bytes, MAX_STRING_BYTES)?;
            let value = decode_escaped_terminated(bytes, &scan);
            let string = String::from_utf8(value).map_err(|_| corrupt())?;
            Ok((Value::String(string), scan.consumed))
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
