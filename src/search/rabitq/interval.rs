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
        })
    }
}

impl fmt::Debug for RaBitQQuery<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RaBitQQuery([REDACTED])")
    }
}

pub(super) fn approximate_distance(
    code: &RaBitQ7,
    query: &RaBitQQuery<'_>,
) -> Result<ApproximateDistance> {
    if query.components.len() != code.signed_codes.len() {
        return Err(Error::invalid_argument());
    }

    let scale = f64::from(code.scale);
    let mut dot = 0.0_f64;
    let mut dot_lower = 0.0_f64;
    let mut dot_upper = 0.0_f64;
    for (&query_component, &signed_code) in query.components.iter().zip(&code.signed_codes) {
        let query_component = f64::from(query_component);
        let reconstruction = scale * f64::from(signed_code);
        let product = query_component * reconstruction;
        dot += product;
        dot_lower = add_down(dot_lower, multiply_down(query_component, reconstruction));
        dot_upper = add_up(dot_upper, multiply_up(query_component, reconstruction));
    }

    let error_upper = f64::from(code.reconstruction_error_upper);
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
