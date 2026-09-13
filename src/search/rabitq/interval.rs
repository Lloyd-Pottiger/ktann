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
    code: &RaBitQ7<'_>,
    query: &RaBitQQuery<'_>,
) -> Result<ApproximateDistance> {
    if query.components.len() != code.dimension {
        return Err(Error::invalid_argument());
    }

    let scale = f64::from(code.scale);
    let mut dot = 0.0_f64;
    let mut dot_lower = 0.0_f64;
    let mut dot_upper = 0.0_f64;
    for (&query_component, signed_code) in query.components.iter().zip(code.signed_codes()) {
        let query_component = f64::from(query_component);
        let reconstruction = scale * f64::from(signed_code);
        let product = query_component * reconstruction;
        dot += product;
        (dot_lower, dot_upper) = extend_dot_interval(dot_lower, dot_upper, product);
    }

    finish_distance(code, query, dot, dot_lower, dot_upper)
}

/// Interleaves independent scores without changing any score's accumulation order.
/// Shared query loads and independent dependency chains reduce scoring work while
/// retaining the scalar rough value and directed-rounding endpoints bit-for-bit.
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
    let mut lowers = [0.0_f64; 4];
    let mut uppers = [0.0_f64; 4];
    for (index, &component) in query.components.iter().enumerate() {
        let component = f64::from(component);
        for lane in 0..4 {
            let reconstruction = scales[lane] * f64::from(codes[lane].signed_code(index));
            let product = component * reconstruction;
            dots[lane] += product;
            (lowers[lane], uppers[lane]) = extend_dot_interval(lowers[lane], uppers[lane], product);
        }
    }
    let [a, b, c, d] = std::array::from_fn(|lane| {
        finish_distance(&codes[lane], query, dots[lane], lowers[lane], uppers[lane])
    });
    Ok([a?, b?, c?, d?])
}

/// Extends a dot interval under the validated f32-query / packed-code bounds.
///
/// A nonzero product is at least 2^-298 and below 2^262: neither underflow nor
/// overflow is possible. Its adjacent f64 values are also nonzero. With at most
/// MAX_DIMENSION (16,384) terms, even outward-rounded sums stay below 2^277.
/// These bounds let us share the product's neighbors and omit nonfinite cases
/// while preserving the general multiply/add rounding rules, including zeros.
fn extend_dot_interval(lower: f64, upper: f64, product: f64) -> (f64, f64) {
    if product == 0.0 {
        return (lower + product, upper + product);
    }
    let (product_lower, product_upper) = finite_neighbors(product);
    let lower_sum = lower + product_lower;
    let upper_sum = upper + product_upper;
    (
        if lower == 0.0 {
            lower_sum
        } else {
            finite_neighbors(lower_sum).0
        },
        if upper == 0.0 {
            upper_sum
        } else {
            finite_neighbors(upper_sum).1
        },
    )
}

/// Returns outward adjacent values for a finite, bounded dot product or sum.
fn finite_neighbors(value: f64) -> (f64, f64) {
    const SIGN: u64 = 1 << 63;
    let bits = value.to_bits();
    if bits & !SIGN == 0 {
        return (f64::from_bits(SIGN | 1), f64::from_bits(1));
    }
    if bits & SIGN == 0 {
        (f64::from_bits(bits - 1), f64::from_bits(bits + 1))
    } else {
        (f64::from_bits(bits + 1), f64::from_bits(bits - 1))
    }
}

/// Converts a completed dot product and its bounds to a metric-specific interval.
fn finish_distance(
    code: &RaBitQ7<'_>,
    query: &RaBitQQuery<'_>,
    dot: f64,
    dot_lower: f64,
    dot_upper: f64,
) -> Result<ApproximateDistance> {
    let scale = f64::from(code.scale);
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

#[cfg(test)]
mod tests {
    use super::extend_dot_interval;
    use crate::api::MAX_DIMENSION;
    use crate::search::rabitq::rounding::{add_down, add_up, multiply_down, multiply_up};

    #[test]
    fn finite_dot_updates_match_general_rounding_at_every_step() {
        let values = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0,
            -1.0,
            f32::MAX,
            -f32::MAX,
        ];
        for scale in [0.0, f32::from_bits(1), f32::MIN_POSITIVE, 1.0, f32::MAX] {
            for phase in 0..values.len() {
                let zero = if phase % 2 == 0 { 0.0 } else { -0.0 };
                let mut actual = (zero, zero);
                let mut expected = (zero, zero);
                for index in 0..MAX_DIMENSION {
                    let left = f64::from(values[(index * 37 + phase) % values.len()]);
                    let code = ((index * 19 + phase) % 127) as i16 - 63;
                    let right = f64::from(scale) * f64::from(code);
                    actual = extend_dot_interval(actual.0, actual.1, left * right);
                    expected = (
                        add_down(expected.0, multiply_down(left, right)),
                        add_up(expected.1, multiply_up(left, right)),
                    );
                    assert_eq!(
                        (actual.0.to_bits(), actual.1.to_bits()),
                        (expected.0.to_bits(), expected.1.to_bits())
                    );
                }
            }
        }
        // Cancellation can produce a zero sum even though both addends are nonzero.
        for product in [1.0, -1.0, 2.0_f64.powi(-298), -2.0_f64.powi(-298)] {
            for initial in [-multiply_down(product, 1.0), -multiply_up(product, 1.0)] {
                let actual = extend_dot_interval(initial, initial, product);
                let expected = (
                    add_down(initial, multiply_down(product, 1.0)),
                    add_up(initial, multiply_up(product, 1.0)),
                );
                assert_eq!(
                    (actual.0.to_bits(), actual.1.to_bits()),
                    (expected.0.to_bits(), expected.1.to_bits())
                );
            }
        }
    }
}
