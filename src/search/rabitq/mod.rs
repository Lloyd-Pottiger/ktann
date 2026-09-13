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
    scale: f32,
    code_norm_squared: u32,
    reconstruction_error_upper: f32,
    dimension: usize,
    signs: &'a [u8],
    magnitudes: &'a [u8],
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

    /// Borrows leaf bytes already validated by `cache::load_body`.
    ///
    /// Only use for an immutable body returned by `load_body`, with the same
    /// Manifest dimension. Its storage scan validated every payload before
    /// publication, even when cache insertion was disabled or skipped. A
    /// general `LeafEntry::new` does not establish this invariant.
    pub(super) fn from_validated_leaf_bytes(encoded: &'a [u8], dimension: usize) -> Self {
        codec::from_validated_bytes(encoded, dimension)
    }

    /// Computes a scalar-f64 rough distance and conservative interval.
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
