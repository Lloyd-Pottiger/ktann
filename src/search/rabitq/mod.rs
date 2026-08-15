//! Canonical format-v1 RaBitQ7 encoding and bounded candidate selection.
//!
//! This module is the single seam for the persistent seven-bit code, scalar
//! f64 approximate distances, conservative intervals, and deterministic
//! overlap selection. Callers do not need to know the bit layout or the
//! directed-rounding rules that keep its intervals conservative.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "mutation and search pipelines consume the RaBitQ7 module"
    )
)]

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
pub(crate) use selection::{
    ApproximateCandidate, OverlapSelection, select_global_overlap, select_leaf_overlap,
};

/// One decoded canonical absolute RaBitQ7 payload.
pub(crate) struct RaBitQ7 {
    scale: f32,
    code_norm_squared: u32,
    reconstruction_error_upper: f32,
    signed_codes: Box<[i8]>,
}

impl RaBitQ7 {
    /// Returns the exact format-v1 payload length for a dimension.
    pub(crate) fn encoded_len(dimension: usize) -> Result<usize> {
        codec::encoded_len(dimension)
    }

    /// Quantizes one finite, metric-preprocessed, rotated vector.
    pub(crate) fn quantize(vector: &[f32]) -> Result<Bytes> {
        codec::quantize(vector)
    }

    /// Decodes and validates one persistent format-v1 payload.
    pub(crate) fn decode(encoded: &[u8], dimension: usize) -> Result<Self> {
        codec::decode(encoded, dimension)
    }

    /// Validates a payload without retaining its expanded signed codes.
    pub(crate) fn validate(encoded: &[u8], dimension: usize) -> Result<()> {
        codec::validate(encoded, dimension)
    }

    /// Computes a scalar-f64 rough distance and conservative interval.
    pub(crate) fn approximate_distance(
        &self,
        query: &RaBitQQuery<'_>,
    ) -> Result<ApproximateDistance> {
        interval::approximate_distance(self, query)
    }
}

impl fmt::Debug for RaBitQ7 {
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
