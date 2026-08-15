//! Canonical absolute RaBitQ7 bitstream encoding.

use bytes::Bytes;

use crate::api::{Error, ErrorKind, MAX_DIMENSION, Result};

use super::RaBitQ7;
use super::rounding::{next_down, next_up};

const HEADER_BYTES: usize = 12;
const MAX_MAGNITUDE: u8 = 63;
const MAGNITUDE_BITS: usize = 6;

pub(super) fn quantize(vector: &[f32]) -> Result<Bytes> {
    if !(1..=MAX_DIMENSION).contains(&vector.len()) {
        return Err(Error::invalid_argument());
    }
    let dimension = vector.len();
    let encoded_len = encoded_len(dimension)?;

    let mut max_abs = 0.0_f32;
    for &component in vector {
        if !component.is_finite() {
            return Err(Error::invalid_argument());
        }
        max_abs = max_abs.max(component.abs());
    }
    if max_abs == 0.0 {
        return zero_code(encoded_len);
    }

    let step = f64::from(max_abs) / f64::from(MAX_MAGNITUDE);
    let mut signed_codes = allocate_codes(dimension)?;
    let mut code_norm_squared = 0_u32;
    let mut numerator = 0.0_f64;
    for &component in vector {
        let magnitude = (f64::from(component.abs()) / step)
            .round()
            .clamp(0.0, f64::from(MAX_MAGNITUDE)) as u8;
        let code = if magnitude == 0 {
            0
        } else if component.is_sign_negative() {
            -(magnitude as i8)
        } else {
            magnitude as i8
        };
        signed_codes.push(code);
        numerator += f64::from(component) * f64::from(code);
        code_norm_squared = code_norm_squared
            .checked_add(u32::from(magnitude).pow(2))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    }

    if code_norm_squared == 0 {
        return zero_code(encoded_len);
    }

    let exact_scale = (numerator / f64::from(code_norm_squared)).max(0.0);
    let scale = exact_scale as f32;
    if !scale.is_finite() || scale.is_sign_negative() {
        return Err(Error::invalid_argument());
    }

    let reconstruction_error_upper = reconstruction_error_upper(vector, &signed_codes, scale)?;
    encode_payload(
        encoded_len,
        scale,
        code_norm_squared,
        reconstruction_error_upper,
        &signed_codes,
    )
}

pub(super) fn decode(encoded: &[u8], dimension: usize) -> Result<RaBitQ7> {
    encoded_len(dimension).map_err(|_| corrupt())?;
    let mut signed_codes = allocate_codes(dimension)?;
    let metadata = parse(encoded, dimension, |code| signed_codes.push(code))?;
    Ok(RaBitQ7 {
        scale: metadata.scale,
        code_norm_squared: metadata.code_norm_squared,
        reconstruction_error_upper: metadata.reconstruction_error_upper,
        signed_codes: signed_codes.into_boxed_slice(),
    })
}

pub(super) fn validate(encoded: &[u8], dimension: usize) -> Result<()> {
    parse(encoded, dimension, |_| {}).map(|_| ())
}

struct Metadata {
    scale: f32,
    code_norm_squared: u32,
    reconstruction_error_upper: f32,
}

fn parse(encoded: &[u8], dimension: usize, mut accept_code: impl FnMut(i8)) -> Result<Metadata> {
    let expected_len = encoded_len(dimension).map_err(|_| corrupt())?;
    if encoded.len() != expected_len || !padding_is_zero(encoded, dimension) {
        return Err(corrupt());
    }

    let scale_bits = read_u32_le(encoded, 0);
    let code_norm_squared = read_u32_le(encoded, 4);
    let error_bits = read_u32_le(encoded, 8);
    let scale = f32::from_bits(scale_bits);
    let reconstruction_error_upper = f32::from_bits(error_bits);
    if !canonical_nonnegative(scale) || !canonical_nonnegative(reconstruction_error_upper) {
        return Err(corrupt());
    }

    let sign_start = HEADER_BYTES;
    let magnitude_start = sign_start + sign_bytes(dimension);
    let mut actual_norm = 0_u32;
    for index in 0..dimension {
        let negative = encoded[sign_start + index / u8::BITS as usize]
            & (1_u8 << (index % u8::BITS as usize))
            != 0;
        let magnitude = decode_magnitude(&encoded[magnitude_start..], index);
        if magnitude == 0 && negative {
            return Err(corrupt());
        }
        let code = if negative {
            -(magnitude as i8)
        } else {
            magnitude as i8
        };
        accept_code(code);
        actual_norm = actual_norm
            .checked_add(u32::from(magnitude).pow(2))
            .ok_or_else(corrupt)?;
    }

    if actual_norm != code_norm_squared {
        return Err(corrupt());
    }
    if code_norm_squared == 0 && (scale_bits != 0 || error_bits != 0) {
        return Err(corrupt());
    }

    Ok(Metadata {
        scale,
        code_norm_squared,
        reconstruction_error_upper,
    })
}

fn zero_code(encoded_len: usize) -> Result<Bytes> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(encoded_len)
        .map_err(|error| Error::with_source(ErrorKind::LimitExceeded, error))?;
    bytes.resize(encoded_len, 0);
    Ok(Bytes::from(bytes))
}

fn allocate_codes(dimension: usize) -> Result<Vec<i8>> {
    let mut codes = Vec::new();
    codes
        .try_reserve_exact(dimension)
        .map_err(|error| Error::with_source(ErrorKind::LimitExceeded, error))?;
    Ok(codes)
}

fn reconstruction_error_upper(vector: &[f32], codes: &[i8], scale: f32) -> Result<f32> {
    let scale = f64::from(scale);
    let mut squared_error_upper = 0.0_f64;
    for (&component, &code) in vector.iter().zip(codes) {
        let reconstruction = scale * f64::from(code);
        let component = f64::from(component);
        let (difference_lower, difference_upper) = difference_bounds(component, reconstruction);
        let magnitude_upper = difference_lower.abs().max(difference_upper.abs());
        let square_upper = nonnegative_product_upper(magnitude_upper, magnitude_upper);
        squared_error_upper = nonnegative_sum_upper(squared_error_upper, square_upper);
    }
    ceil_sqrt_f32(squared_error_upper).ok_or_else(Error::invalid_argument)
}

fn difference_bounds(component: f64, reconstruction: f64) -> (f64, f64) {
    let difference = component - reconstruction;
    let roundoff = two_sum_roundoff(component, -reconstruction, difference);
    if roundoff > 0.0 {
        (difference, next_up(difference))
    } else if roundoff < 0.0 {
        (next_down(difference), difference)
    } else {
        (difference, difference)
    }
}

fn nonnegative_product_upper(left: f64, right: f64) -> f64 {
    let product = left * right;
    let roundoff = left.mul_add(right, -product);
    if roundoff > 0.0 {
        next_up(product)
    } else {
        product
    }
}

fn nonnegative_sum_upper(left: f64, right: f64) -> f64 {
    let sum = left + right;
    let roundoff = two_sum_roundoff(left, right, sum);
    if roundoff > 0.0 { next_up(sum) } else { sum }
}

fn two_sum_roundoff(left: f64, right: f64, sum: f64) -> f64 {
    let right_virtual = sum - left;
    let left_virtual = sum - right_virtual;
    let right_roundoff = right - right_virtual;
    let left_roundoff = left - left_virtual;
    left_roundoff + right_roundoff
}

fn ceil_sqrt_f32(squared_upper: f64) -> Option<f32> {
    let mut candidate = ceil_f32(squared_upper.sqrt())?;
    if f64::from(candidate) * f64::from(candidate) < squared_upper {
        candidate = f32::from_bits(candidate.to_bits().checked_add(1)?);
    }
    candidate.is_finite().then_some(candidate)
}

fn ceil_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let rounded = value as f32;
    if !rounded.is_finite() {
        return None;
    }
    if f64::from(rounded) >= value {
        return Some(rounded);
    }
    let next = f32::from_bits(rounded.to_bits().checked_add(1)?);
    next.is_finite().then_some(next)
}

fn encode_payload(
    encoded_len: usize,
    scale: f32,
    code_norm_squared: u32,
    reconstruction_error_upper: f32,
    signed_codes: &[i8],
) -> Result<Bytes> {
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(encoded_len)
        .map_err(|error| Error::with_source(ErrorKind::LimitExceeded, error))?;
    encoded.resize(encoded_len, 0);
    encoded[0..4].copy_from_slice(&scale.to_bits().to_le_bytes());
    encoded[4..8].copy_from_slice(&code_norm_squared.to_le_bytes());
    encoded[8..12].copy_from_slice(&reconstruction_error_upper.to_bits().to_le_bytes());

    let sign_start = HEADER_BYTES;
    let magnitude_start = sign_start + sign_bytes(signed_codes.len());
    for (index, &code) in signed_codes.iter().enumerate() {
        if code.is_negative() {
            encoded[sign_start + index / u8::BITS as usize] |= 1_u8 << (index % u8::BITS as usize);
        }
        encode_magnitude(&mut encoded[magnitude_start..], index, code.unsigned_abs());
    }
    Ok(Bytes::from(encoded))
}

fn encode_magnitude(bytes: &mut [u8], index: usize, magnitude: u8) {
    let bit_offset = index * MAGNITUDE_BITS;
    let byte_index = bit_offset / u8::BITS as usize;
    let shift = bit_offset % u8::BITS as usize;
    let shifted = u16::from(magnitude) << shift;
    bytes[byte_index] |= shifted as u8;
    if shift > u8::BITS as usize - MAGNITUDE_BITS {
        bytes[byte_index + 1] |= (shifted >> u8::BITS) as u8;
    }
}

fn decode_magnitude(bytes: &[u8], index: usize) -> u8 {
    let bit_offset = index * MAGNITUDE_BITS;
    let byte_index = bit_offset / u8::BITS as usize;
    let shift = bit_offset % u8::BITS as usize;
    let low = u16::from(bytes[byte_index]);
    let high = bytes
        .get(byte_index + 1)
        .copied()
        .map(u16::from)
        .unwrap_or(0);
    (((low | (high << u8::BITS)) >> shift) & u16::from(MAX_MAGNITUDE)) as u8
}

fn padding_is_zero(encoded: &[u8], dimension: usize) -> bool {
    let sign_len = sign_bytes(dimension);
    let sign_remainder = dimension % u8::BITS as usize;
    if sign_remainder != 0 {
        let used = (1_u8 << sign_remainder) - 1;
        if encoded[HEADER_BYTES + sign_len - 1] & !used != 0 {
            return false;
        }
    }

    let magnitude_bits = dimension * MAGNITUDE_BITS;
    let magnitude_remainder = magnitude_bits % u8::BITS as usize;
    if magnitude_remainder != 0 {
        let magnitude_start = HEADER_BYTES + sign_len;
        let magnitude_len = magnitude_bytes(dimension);
        let used = (1_u8 << magnitude_remainder) - 1;
        if encoded[magnitude_start + magnitude_len - 1] & !used != 0 {
            return false;
        }
    }
    true
}

fn canonical_nonnegative(value: f32) -> bool {
    value.is_finite() && !value.is_sign_negative()
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

pub(super) fn encoded_len(dimension: usize) -> Result<usize> {
    if !(1..=MAX_DIMENSION).contains(&dimension) {
        return Err(Error::invalid_argument());
    }
    HEADER_BYTES
        .checked_add(sign_bytes(dimension))
        .and_then(|length| length.checked_add(magnitude_bytes(dimension)))
        .ok_or_else(Error::invalid_argument)
}

const fn sign_bytes(dimension: usize) -> usize {
    dimension.div_ceil(u8::BITS as usize)
}

const fn magnitude_bytes(dimension: usize) -> usize {
    (dimension * MAGNITUDE_BITS).div_ceil(u8::BITS as usize)
}

const fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}
