//! Canonical RaBitQ7 encoding and bounded candidate selection.
//!
//! This module is the single seam for the persistent seven-bit code, scalar
//! f64 approximate distances, conservative intervals, and deterministic
//! overlap selection. Callers do not need to know the bit layout or the
//! directed-rounding rules that keep its intervals conservative.

mod codec;
mod interval;
mod rounding;
mod selection;

#[cfg(test)]
mod tests;

use std::fmt;

use bytes::Bytes;

use crate::api::Result;

pub(crate) use interval::RaBitQQuery;
#[cfg(test)]
pub(crate) use selection::OverlapSelection;
pub(crate) use selection::{ApproximateCandidate, select_global_overlap, select_leaf_overlap};

/// One borrowed canonical absolute RaBitQ7 payload.
pub(crate) struct RaBitQ7<'a> {
    header: CodeHeader,
    dimension: usize,
    signs: &'a [u8],
    magnitudes: &'a [u8],
}

/// Numeric metadata shared by packed storage and the decoded cache form.
#[derive(Clone, Copy)]
struct CodeHeader {
    scale: f32,
    code_norm_squared: u32,
    reconstruction_error_upper: f32,
}

/// A validated code expanded once for an immutable cached leaf body.
/// The packed payload is not retained alongside these signed components.
pub(crate) struct DecodedRaBitQ7 {
    header: CodeHeader,
    codes: Box<[i8]>,
}

impl DecodedRaBitQ7 {
    /// Expands a payload already validated by the Leaf Entry decoder.
    pub(crate) fn from_validated_leaf_bytes(encoded: &[u8], dimension: usize) -> Self {
        let packed = RaBitQ7::from_validated_leaf_bytes(encoded, dimension);
        let dimension = packed.dimension;
        let mut codes = Vec::with_capacity(dimension);
        // Decode each complete packed group once, including its sign nibble.
        for index in 0..dimension / 4 {
            let block = packed.code_block(index);
            codes.extend((0..4).map(|component| block.signed_code(component) as i8));
        }
        codes.extend((dimension / 4 * 4..dimension).map(|index| packed.signed_code(index)));
        Self {
            header: packed.header,
            codes: codes.into_boxed_slice(),
        }
    }

    /// Returns the retained signed-code allocation size in bytes.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Scores independent lanes with the scalar accumulation order in each lane.
    pub(crate) fn approximate_distances<const N: usize>(
        codes: [&Self; N],
        query: &RaBitQQuery<'_>,
    ) -> Result<[ApproximateDistance; N]> {
        interval::approximate_distances(codes, query)
    }
}

impl<'a> RaBitQ7<'a> {
    /// Returns the exact payload length for a dimension.
    pub(crate) fn encoded_len(dimension: usize) -> Result<usize> {
        codec::encoded_len(dimension)
    }

    /// Quantizes one finite, metric-preprocessed, rotated vector.
    pub(crate) fn quantize(vector: &[f32]) -> Result<Bytes> {
        codec::quantize(vector)
    }

    /// Borrows and validates one persistent payload without expanding its codes.
    pub(crate) fn decode(encoded: &'a [u8], dimension: usize) -> Result<Self> {
        codec::decode(encoded, dimension)
    }

    /// Validates a persistent payload without allocating or retaining a view.
    pub(crate) fn validate(encoded: &'a [u8], dimension: usize) -> Result<()> {
        Self::decode(encoded, dimension).map(|_| ())
    }

    /// Borrows bytes validated by the Leaf Entry decoder for this Manifest dimension.
    /// A general `LeafEntry::new` does not establish that invariant.
    pub(super) fn from_validated_leaf_bytes(encoded: &'a [u8], dimension: usize) -> Self {
        codec::from_validated_bytes(encoded, dimension)
    }

    /// Computes a scalar-f64 rough distance and conservative interval.
    #[cfg(test)]
    pub(crate) fn approximate_distance(
        &self,
        query: &RaBitQQuery<'_>,
    ) -> Result<ApproximateDistance> {
        interval::approximate_distance(self, query)
    }
}

impl fmt::Debug for RaBitQ7<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RaBitQ7([REDACTED])")
    }
}

/// A rough ranking value and a conservative interval around the exact value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ApproximateDistance {
    rough: f64,
    lower: f64,
    upper: f64,
}

impl ApproximateDistance {
    fn from_conservative_bounds(rough: f64, lower: f64, upper: f64) -> Result<Self> {
        interval::validate_distance(rough, lower, upper)?;
        Ok(Self {
            rough,
            lower,
            upper,
        })
    }

    /// Returns the rough scalar-f64 ranking value.
    pub(crate) const fn rough(self) -> f64 {
        self.rough
    }

    /// Returns the conservative lower endpoint.
    pub(crate) const fn lower(self) -> f64 {
        self.lower
    }

    /// Returns the conservative upper endpoint.
    pub(crate) const fn upper(self) -> f64 {
        self.upper
    }
}

/// Builds a degenerate conservative interval for tests outside this module.
#[cfg(test)]
pub(crate) fn test_approximate_distance(rough: f64) -> ApproximateDistance {
    ApproximateDistance::from_conservative_bounds(rough, rough, rough)
        .expect("a finite rough value forms a valid degenerate interval")
}
