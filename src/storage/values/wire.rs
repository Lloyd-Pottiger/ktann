//! Canonical framing and primitive wire encoding.

use bytes::Bytes;

use crate::api::{Error, LogicalIndexId, PartitionKey, Result};

use super::{MAX_VALUE_BYTES, VALUE_CODEC_VERSION, ValueKind, corrupt, unsupported};

pub(super) struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    pub(super) fn new(kind: ValueKind) -> Self {
        Self {
            bytes: vec![kind.tag(), VALUE_CODEC_VERSION],
        }
    }

    pub(super) fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(super) fn reserve(&mut self, additional: usize) {
        self.bytes.reserve(additional);
    }

    pub(super) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub(super) fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    pub(super) fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub(super) fn f32(&mut self, value: f32) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::invalid_argument());
        }
        self.u32(if value == 0.0 { 0 } else { value.to_bits() });
        Ok(())
    }

    pub(super) fn f64(&mut self, value: f64) -> Result<()> {
        self.u64(canonical_f64_bits(value)?);
        Ok(())
    }

    pub(super) fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    pub(super) fn sized_bytes(&mut self, value: &[u8], maximum: usize) -> Result<()> {
        if value.len() > maximum {
            return Err(Error::invalid_argument());
        }
        self.u32(u32::try_from(value.len()).map_err(|_| Error::invalid_argument())?);
        self.bytes(value);
        Ok(())
    }

    pub(super) fn sized_u8_bytes(&mut self, value: &[u8]) -> Result<()> {
        self.u8(u8::try_from(value.len()).map_err(|_| Error::invalid_argument())?);
        self.bytes(value);
        Ok(())
    }

    pub(super) fn sized_u16_bytes(&mut self, value: &[u8]) -> Result<()> {
        self.u16(u16::try_from(value.len()).map_err(|_| Error::invalid_argument())?);
        self.bytes(value);
        Ok(())
    }

    pub(super) fn finish(self) -> Result<Vec<u8>> {
        if self.bytes.len() > MAX_VALUE_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(self.bytes)
    }
}

/// Returns the canonical persisted bits of a finite value; positive zero is
/// the only canonical zero.
pub(super) fn canonical_f64_bits(value: f64) -> Result<u64> {
    if !value.is_finite() {
        return Err(Error::invalid_argument());
    }
    Ok(if value == 0.0 { 0 } else { value.to_bits() })
}

pub(super) struct Decoder {
    bytes: Bytes,
    position: usize,
}

impl Decoder {
    pub(super) fn framed(expected: ValueKind, bytes: Bytes) -> Result<Self> {
        if bytes.len() < 2 || bytes.len() > MAX_VALUE_BYTES || bytes[0] != expected.tag() {
            return Err(corrupt());
        }
        if bytes[1] != VALUE_CODEC_VERSION {
            return Err(
                if matches!(
                    expected,
                    ValueKind::IndexIdAllocator
                        | ValueKind::IndexNameEntry
                        | ValueKind::IndexManifest
                ) {
                    unsupported()
                } else {
                    corrupt()
                },
            );
        }
        Ok(Self { bytes, position: 2 })
    }

    pub(super) fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take_range(&mut self, length: usize) -> Result<(usize, usize)> {
        let end = self.position.checked_add(length).ok_or_else(corrupt)?;
        if end > self.bytes.len() {
            return Err(corrupt());
        }
        let start = self.position;
        self.position = end;
        Ok((start, end))
    }

    fn take(&mut self, length: usize) -> Result<&[u8]> {
        let (start, end) = self.take_range(length)?;
        Ok(&self.bytes[start..end])
    }

    fn take_bytes(&mut self, length: usize) -> Result<Bytes> {
        let (start, end) = self.take_range(length)?;
        Ok(self.bytes.slice(start..end))
    }

    pub(super) fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.take(N)?.try_into().map_err(|_| corrupt())
    }

    pub(super) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(super) fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(corrupt()),
        }
    }

    pub(super) fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.array()?))
    }

    pub(super) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.array()?))
    }

    pub(super) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.array()?))
    }

    pub(super) fn logical_index_id(&mut self) -> Result<LogicalIndexId> {
        LogicalIndexId::new(self.u64()?).map_err(|_| corrupt())
    }

    pub(super) fn partition_key(&mut self) -> Result<PartitionKey> {
        PartitionKey::new(self.u64()?).map_err(|_| corrupt())
    }

    pub(super) fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_be_bytes(self.array()?))
    }

    pub(super) fn canonical_f32(&mut self) -> Result<f32> {
        let bits = self.u32()?;
        let value = f32::from_bits(bits);
        if !value.is_finite() || (value == 0.0 && bits != 0) {
            return Err(corrupt());
        }
        Ok(value)
    }

    pub(super) fn canonical_f64(&mut self) -> Result<f64> {
        let bits = self.u64()?;
        let value = f64::from_bits(bits);
        if !value.is_finite() || (value == 0.0 && bits != 0) {
            return Err(corrupt());
        }
        Ok(value)
    }

    pub(super) fn sized_bytes(&mut self, maximum: usize) -> Result<Bytes> {
        let length = usize::try_from(self.u32()?).map_err(|_| corrupt())?;
        if length > maximum {
            return Err(corrupt());
        }
        self.take_bytes(length)
    }

    pub(super) fn sized_u8_bytes(&mut self) -> Result<Bytes> {
        let length = usize::from(self.u8()?);
        self.take_bytes(length)
    }

    pub(super) fn sized_u16_bytes(&mut self, maximum: usize) -> Result<Bytes> {
        let length = usize::from(self.u16()?);
        if length > maximum {
            return Err(corrupt());
        }
        self.take_bytes(length)
    }

    pub(super) fn finish(self) -> Result<()> {
        if self.position != self.bytes.len() {
            return Err(corrupt());
        }
        Ok(())
    }
}
