//! Scalar-f64 rough distances with conservative directed-rounding intervals.

use std::fmt;

use crate::api::{Error, MAX_DIMENSION, Metric, Result};

use super::rounding::{add_down, add_up, multiply_down, multiply_up, sqrt_up};
use super::{ApproximateDistance, RaBitQ7};

/// One validated, metric-specific query prepared once per Search.
pub(crate) struct RaBitQQuery<'a> {
    components: &'a [f32],
    metric: Metric,
    norm_squared: f64,
    norm_squared_lower: f64,
    norm_squared_upper: f64,
    dot_error_factor: f64,
}

impl<'a> RaBitQQuery<'a> {
    /// Validates a rotated query and precomputes its scalar-f64 norm bounds.
    pub(crate) fn new(components: &'a [f32], metric: Metric) -> Result<Self> {
        if !(1..=MAX_DIMENSION).contains(&components.len()) {
            return Err(Error::invalid_argument());
        }

        let mut norm_squared = 0.0_f64;
        let mut norm_squared_lower = 0.0_f64;
        let mut norm_squared_upper = 0.0_f64;
        for &component in components {
            if !component.is_finite() {
                return Err(Error::invalid_argument());
            }
            let component = f64::from(component);
            norm_squared += component * component;
            norm_squared_lower = add_down(norm_squared_lower, multiply_down(component, component));
            norm_squared_upper = add_up(norm_squared_upper, multiply_up(component, component));
        }
        Ok(Self {
            components,
            metric,
            norm_squared,
            norm_squared_lower,
            norm_squared_upper,
            dot_error_factor: multiply_up(
                components.len() as f64 * f64::EPSILON,
                sqrt_up(norm_squared_upper),
            ),
        })
    }
}

impl fmt::Debug for RaBitQQuery<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RaBitQQuery([REDACTED])")
    }
}

pub(super) fn approximate_distance(
    code: &RaBitQ7<'_>,
    query: &RaBitQQuery<'_>,
) -> Result<ApproximateDistance> {
    if query.components.len() != code.dimension {
        return Err(Error::invalid_argument());
    }

    let scale = f64::from(code.scale);
    let mut dot = 0.0_f64;
    for (&query_component, signed_code) in query.components.iter().zip(code.signed_codes()) {
        let query_component = f64::from(query_component);
        let reconstruction = scale * f64::from(signed_code);
        let product = query_component * reconstruction;
        dot += product;
    }

    finish_distance(code, query, dot)
}

/// Interleaves independent scores without changing any score's accumulation order.
/// Shared query loads and independent dependency chains reduce scoring work while
/// retaining the scalar rough value and the same conservative error bound.
pub(super) fn approximate_distances(
    codes: &[RaBitQ7<'_>; 4],
    query: &RaBitQQuery<'_>,
) -> Result<[ApproximateDistance; 4]> {
    if codes
        .iter()
        .any(|code| code.dimension != query.components.len())
    {
        return Err(Error::invalid_argument());
    }
    let scales = codes.each_ref().map(|code| f64::from(code.scale));
    let mut dots = [0.0_f64; 4];
    // Decode complete packed groups once while preserving scalar accumulation order.
    let mut blocks = query.components.chunks_exact(4);
    for (block, components) in blocks.by_ref().enumerate() {
        let decoded: [_; 4] = std::array::from_fn(|lane| codes[lane].code_block(block));
        for (component_index, &component) in components.iter().enumerate() {
            let component = f64::from(component);
            for lane in 0..4 {
                let reconstruction =
                    scales[lane] * f64::from(decoded[lane].signed_code(component_index));
                let product = component * reconstruction;
                dots[lane] += product;
            }
        }
    }
    let tail_start = query.components.len() - blocks.remainder().len();
    for (offset, &component) in blocks.remainder().iter().enumerate() {
        let component = f64::from(component);
        for lane in 0..4 {
            let reconstruction =
                scales[lane] * f64::from(codes[lane].signed_code(tail_start + offset));
            let product = component * reconstruction;
            dots[lane] += product;
        }
    }
    let [a, b, c, d] = std::array::from_fn(|lane| finish_distance(&codes[lane], query, dots[lane]));
    Ok([a?, b?, c?, d?])
}

/// Converts a completed dot product and its bounds to a metric-specific interval.
fn finish_distance(
    code: &RaBitQ7<'_>,
    query: &RaBitQQuery<'_>,
    dot: f64,
) -> Result<ApproximateDistance> {
    let scale = f64::from(code.scale);
    let error_upper = f64::from(code.reconstruction_error_upper);
    // Reconstruction is exact in f64 (f32 scale times a six-bit integer).
    // For n products accumulated in order, roundoff is bounded by
    // gamma_n * sum(abs(q_i * x_hat_i)), with u = 2^-53. Since n <= 16384,
    // gamma_n = n*u/(1-n*u) <= 2*n*u = n*EPSILON. Cauchy-Schwarz bounds
    // the absolute sum by ||q|| * ||x_hat||. All bound operations round up;
    // f32 inputs and six-bit codes cannot underflow or overflow this kernel.
    let reconstruction_norm_upper = multiply_up(scale, sqrt_up(f64::from(code.code_norm_squared)));
    let dot_error = multiply_up(query.dot_error_factor, reconstruction_norm_upper);
    let dot_lower = add_down(dot, -dot_error);
    let dot_upper = add_up(dot, dot_error);
    let (rough, center_lower, center_upper, radius_upper, clamp_lower) = match query.metric {
        Metric::InnerProduct => {
            let radius = multiply_up(sqrt_up(query.norm_squared_upper), error_upper);
            (-dot, -dot_upper, -dot_lower, radius, false)
        }
        Metric::Cosine => {
            let radius = multiply_up(sqrt_up(query.norm_squared_upper), error_upper);
            (
                1.0 - dot,
                add_down(1.0, -dot_upper),
                add_up(1.0, -dot_lower),
                radius,
                false,
            )
        }
        Metric::L2 => {
            let reconstruction_norm_squared = scale * scale * f64::from(code.code_norm_squared);
            let reconstruction_norm_squared_lower = multiply_down(
                multiply_down(scale, scale),
                f64::from(code.code_norm_squared),
            );
            let reconstruction_norm_squared_upper =
                multiply_up(multiply_up(scale, scale), f64::from(code.code_norm_squared));
            let rough = (query.norm_squared + reconstruction_norm_squared - 2.0 * dot).max(0.0);
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
            (rough, lower, upper, radius, true)
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
