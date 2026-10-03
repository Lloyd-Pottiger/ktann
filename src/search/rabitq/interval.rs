//! Scalar-f64 rough distances with conservative directed-rounding intervals.

use std::fmt;

use crate::api::{Error, MAX_DIMENSION, Metric, Result};

#[cfg(test)]
use super::RaBitQ7;
use super::rounding::{add_down, add_up, multiply_down, multiply_up, sqrt_up};
use super::{ApproximateDistance, CodeHeader, DecodedRaBitQ7};

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

#[cfg(test)]
pub(super) fn approximate_distance(
    code: &RaBitQ7<'_>,
    query: &RaBitQQuery<'_>,
) -> Result<ApproximateDistance> {
    if query.components.len() != code.dimension {
        return Err(Error::invalid_argument());
    }

    let scale = f64::from(code.header.scale);
    let mut dot = 0.0_f64;
    let mut odd_dot = 0.0_f64;
    for (index, (&query_component, signed_code)) in
        query.components.iter().zip(code.signed_codes()).enumerate()
    {
        let query_component = f64::from(query_component);
        let product = query_component * f64::from(signed_code);
        if index % 2 == 0 {
            dot += product;
        } else {
            odd_dot += product;
        }
    }

    finish_distance(&code.header, query, (dot + odd_dot) * scale)
}

/// Scores decoded lanes using two independent scalar-f64 partial sums.
///
/// Query/code products are exact in f64. The two sums and final scale need
/// at most `dimension` rounding steps per term, preserving the dot-error
/// bound in `finish_distance` while shortening the addition dependency chain.
pub(super) fn approximate_distances<const N: usize>(
    codes: [&DecodedRaBitQ7; N],
    query: &RaBitQQuery<'_>,
) -> Result<[ApproximateDistance; N]> {
    if codes
        .iter()
        .any(|code| code.codes.len() != query.components.len())
    {
        return Err(Error::invalid_argument());
    }
    let scales = codes.map(|code| f64::from(code.header.scale));
    let mut dots = [0.0_f64; N];
    let mut odd_dots = [0.0_f64; N];
    for (pair_index, pair) in query.components.as_chunks::<2>().0.iter().enumerate() {
        let index = pair_index * 2;
        let even = f64::from(pair[0]);
        let odd = f64::from(pair[1]);
        for lane in 0..N {
            dots[lane] += even * f64::from(codes[lane].codes[index]);
            odd_dots[lane] += odd * f64::from(codes[lane].codes[index + 1]);
        }
    }
    if !query.components.len().is_multiple_of(2) {
        let index = query.components.len() - 1;
        let component = f64::from(query.components[index]);
        for lane in 0..N {
            dots[lane] += component * f64::from(codes[lane].codes[index]);
        }
    }
    let mut distances = [ApproximateDistance {
        rough: 0.0,
        lower: 0.0,
        upper: 0.0,
    }; N];
    for (lane, distance) in distances.iter_mut().enumerate() {
        *distance = finish_distance(
            &codes[lane].header,
            query,
            (dots[lane] + odd_dots[lane]) * scales[lane],
        )?;
    }
    Ok(distances)
}

/// Converts a completed dot product and its bounds to a metric-specific interval.
fn finish_distance(
    header: &CodeHeader,
    query: &RaBitQQuery<'_>,
    dot: f64,
) -> Result<ApproximateDistance> {
    let scale = f64::from(header.scale);
    let error_upper = f64::from(header.reconstruction_error_upper);
    // Each query-component times signed code is exact in f64 (at most 30 bits).
    // Two partial sums, their reduction, and one final scale multiplication
    // have at most ceil(n/2)+1 rounded operations per term for n >= 2,
    // which is at most n. For n=1 only the final scale multiplication rounds.
    // Thus roundoff is bounded by
    // gamma_n * sum(abs(q_i * x_hat_i)), with u = 2^-53. Since n <= 16384,
    // gamma_n = n*u/(1-n*u) <= 2*n*u = n*EPSILON. Cauchy-Schwarz bounds
    // the absolute sum by ||q|| * ||x_hat||. All bound operations round up;
    // f32 inputs and six-bit codes cannot underflow or overflow this kernel.
    let reconstruction_norm_upper =
        multiply_up(scale, sqrt_up(f64::from(header.code_norm_squared)));
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
            let reconstruction_norm_squared = scale * scale * f64::from(header.code_norm_squared);
            let reconstruction_norm_squared_lower = multiply_down(
                multiply_down(scale, scale),
                f64::from(header.code_norm_squared),
            );
            let reconstruction_norm_squared_upper = multiply_up(
                multiply_up(scale, scale),
                f64::from(header.code_norm_squared),
            );
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
