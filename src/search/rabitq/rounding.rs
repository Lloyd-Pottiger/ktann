//! Directed scalar-f64 rounding used by conservative numeric bounds.

/// Returns the adjacent representable f64 toward positive infinity.
pub(super) fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }

    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Returns the adjacent representable f64 toward negative infinity.
pub(super) fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }

    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Adds two values and rounds the result upward.
pub(super) fn add_up(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        left + right
    } else {
        next_up(left + right)
    }
}

/// Adds two values and rounds the result downward.
pub(super) fn add_down(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        left + right
    } else {
        next_down(left + right)
    }
}

/// Multiplies two values and rounds the result upward.
pub(super) fn multiply_up(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        left * right
    } else {
        next_up(left * right)
    }
}

/// Multiplies two values and rounds the result downward.
pub(super) fn multiply_down(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        left * right
    } else {
        next_down(left * right)
    }
}

/// Takes a nonnegative square root and rounds the result upward.
pub(super) fn sqrt_up(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        next_up(value.sqrt())
    }
}
