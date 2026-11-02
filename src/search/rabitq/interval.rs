//! Integer query/code dots with conservative directed-rounding intervals.

use std::fmt;

use crate::api::{Error, MAX_DIMENSION, Metric, Result};

#[cfg(test)]
use super::RaBitQ7;
use super::rounding::{add_down, add_up, multiply_down, multiply_up, sqrt_up};
use super::{ApproximateDistance, CodeHeader, DecodedRaBitQ7};

// Integer coarse dots remain exact over the public dimension limit.
const _: () = assert!(MAX_DIMENSION <= i32::MAX as usize / 63);

/// One validated, metric-specific query prepared once per Search.
pub(crate) struct RaBitQQuery {
    metric: Metric,
    norm_squared: f64,
    norm_squared_lower: f64,
    norm_squared_upper: f64,
    /// Quantized query components, scale, and an upper bound on dot error.
    components: Vec<i16>,
    scale: f64,
    dot_error_factor: f64,
}

impl RaBitQQuery {
    /// Query-owned components charged to routing working memory.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.components.capacity() * size_of::<i16>()
    }

    /// Quantizes a rotated query and bounds its norm and quantization error.
    pub(crate) fn new(components: &[f32], metric: Metric) -> Result<Self> {
        if !(1..=MAX_DIMENSION).contains(&components.len()) {
            return Err(Error::invalid_argument());
        }

        let mut norm_squared = 0.0_f64;
        let mut norm_squared_lower = 0.0_f64;
        let mut norm_squared_upper = 0.0_f64;
        let mut max_component = 0.0_f64;
        for &component in components {
            if !component.is_finite() {
                return Err(Error::invalid_argument());
            }
            let component = f64::from(component);
            max_component = max_component.max(component.abs());
            norm_squared += component * component;
            norm_squared_lower = add_down(norm_squared_lower, multiply_down(component, component));
            norm_squared_upper = add_up(norm_squared_upper, multiply_up(component, component));
        }
        let norm_upper = sqrt_up(norm_squared_upper);
        // Use the most precise i16 query that keeps every partial sum in i32.
        let query_limit =
            f64::from(i16::MAX).min((i32::MAX as usize / (63 * components.len())) as f64);
        let scale = if max_component == 0.0 {
            1.0
        } else {
            max_component / query_limit
        };
        let mut quantized_components = Vec::with_capacity(components.len());
        let mut quantization_error_squared = 0.0;
        for &component in components {
            let component = f64::from(component);
            let quantized = (component / scale).round().clamp(-query_limit, query_limit) as i16;
            quantized_components.push(quantized);
            let lower = add_down(component, -multiply_up(scale, f64::from(quantized)));
            let upper = add_up(component, -multiply_down(scale, f64::from(quantized)));
            let error = lower.abs().max(upper.abs());
            quantization_error_squared =
                add_up(quantization_error_squared, multiply_up(error, error));
        }
        let quantization_error = sqrt_up(quantization_error_squared);
        Ok(Self {
            components: quantized_components,
            metric,
            norm_squared,
            norm_squared_lower,
            norm_squared_upper,
            scale,
            dot_error_factor: add_up(
                quantization_error,
                multiply_up(2.0 * f64::EPSILON, add_up(norm_upper, quantization_error)),
            ),
        })
    }
}

impl fmt::Debug for RaBitQQuery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RaBitQQuery([REDACTED])")
    }
}

#[cfg(test)]
pub(super) fn approximate_distance(
    code: &RaBitQ7<'_>,
    query: &RaBitQQuery,
) -> Result<ApproximateDistance> {
    if query.components.len() != code.dimension {
        return Err(Error::invalid_argument());
    }

    let sum: i32 = query
        .components
        .iter()
        .zip(code.signed_codes())
        .map(|(&query, code)| i32::from(query) * i32::from(code))
        .sum();
    finish_distance(
        &code.header,
        query,
        f64::from(sum) * query.scale * f64::from(code.header.scale),
    )
}

/// Computes one exact integer dot; only competitive rough scores need an interval.
/// Equality remains eligible so the candidate pool can break ties by Record ID.
pub(super) fn score(
    code: &DecodedRaBitQ7<&[i8]>,
    query: &RaBitQQuery,
    cutoff: Option<f64>,
) -> Result<Option<ApproximateDistance>> {
    if code.codes.len() != query.components.len() {
        return Err(Error::invalid_argument());
    }
    // Every product and partial sum fits i32 over the public dimension limit.
    // Contiguous components allow LLVM to vectorize this exact reduction.
    let sum: i32 = query
        .components
        .iter()
        .zip(code.codes)
        .map(|(&query, &code)| i32::from(query) * i32::from(code))
        .sum();
    let dot = f64::from(sum) * query.scale * f64::from(code.header.scale);
    if cutoff.is_some_and(|cutoff| rough_distance(&code.header, query, dot) > cutoff) {
        return Ok(None);
    }
    finish_distance(&code.header, query, dot).map(Some)
}

/// Metric-specific rough score shared by admission and interval construction.
fn rough_distance(header: &CodeHeader, query: &RaBitQQuery, dot: f64) -> f64 {
    match query.metric {
        Metric::InnerProduct => -dot,
        Metric::Cosine => 1.0 - dot,
        Metric::L2 => {
            let scale = f64::from(header.scale);
            (query.norm_squared + scale * scale * f64::from(header.code_norm_squared) - 2.0 * dot)
                .max(0.0)
        }
    }
}

/// Converts a completed dot product and its bounds to a metric-specific interval.
fn finish_distance(
    header: &CodeHeader,
    query: &RaBitQQuery,
    dot: f64,
) -> Result<ApproximateDistance> {
    let scale = f64::from(header.scale);
    let error_upper = f64::from(header.reconstruction_error_upper);
    // Cauchy–Schwarz bounds query quantization error by E * ||reconstruction||.
    // The query's error factor also covers the two final f64 multiplications.
    // Directed arithmetic keeps both this bound and reconstruction error outward.
    let reconstruction_norm_upper =
        multiply_up(scale, sqrt_up(f64::from(header.code_norm_squared)));
    let dot_error = multiply_up(query.dot_error_factor, reconstruction_norm_upper);
    let dot_lower = add_down(dot, -dot_error);
    let dot_upper = add_up(dot, dot_error);
    let rough = rough_distance(header, query, dot);
    let (center_lower, center_upper, radius_upper, clamp_lower) = match query.metric {
        Metric::InnerProduct => {
            let radius = multiply_up(sqrt_up(query.norm_squared_upper), error_upper);
            (-dot_upper, -dot_lower, radius, false)
        }
        Metric::Cosine => {
            let radius = multiply_up(sqrt_up(query.norm_squared_upper), error_upper);
            (
                add_down(1.0, -dot_upper),
                add_up(1.0, -dot_lower),
                radius,
                false,
            )
        }
        Metric::L2 => {
            let reconstruction_norm_squared_lower = multiply_down(
                multiply_down(scale, scale),
                f64::from(header.code_norm_squared),
            );
            let reconstruction_norm_squared_upper = multiply_up(
                multiply_up(scale, scale),
                f64::from(header.code_norm_squared),
            );
            let lower = add_down(
                add_down(query.norm_squared_lower, reconstruction_norm_squared_lower),
                -multiply_up(2.0, dot_upper),
            )
            .max(0.0);
            let upper = add_up(
                add_up(query.norm_squared_upper, reconstruction_norm_squared_upper),
                -multiply_down(2.0, dot_lower),
            )
            .max(0.0);
            let root_distance_upper = sqrt_up(upper);
            let linear_error = multiply_up(multiply_up(2.0, root_distance_upper), error_upper);
            let squared_error = multiply_up(error_upper, error_upper);
            let radius = add_up(linear_error, squared_error);
            (lower, upper, radius, true)
        }
    };

    let mut lower = add_down(center_lower, -radius_upper);
    if clamp_lower {
        lower = lower.max(0.0);
    }
    let upper = add_up(center_upper, radius_upper);
    ApproximateDistance::from_conservative_bounds(rough, lower, upper)
}

pub(super) fn validate_distance(rough: f64, lower: f64, upper: f64) -> Result<()> {
    if !rough.is_finite()
        || !lower.is_finite()
        || !upper.is_finite()
        || lower > rough
        || rough > upper
    {
        return Err(Error::invalid_argument());
    }
    Ok(())
}
