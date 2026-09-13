//! Directed scalar-f64 rounding used by conservative numeric bounds.

// With the IEEE-754 sign bit removed, infinity precedes every NaN encoding.
// Preserve NaN bits unchanged; other nonzero values follow unsigned bit order,
// reversed for negatives. Each infinity is fixed only in its outward direction.
const SIGN_BIT: u64 = 0x8000_0000_0000_0000;
const MAGNITUDE_MASK: u64 = 0x7fff_ffff_ffff_ffff;
const INFINITY_BITS: u64 = 0x7ff0_0000_0000_0000;

/// Returns the adjacent representable f64 toward positive infinity.
pub(super) fn next_up(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & MAGNITUDE_MASK;
    if magnitude > INFINITY_BITS || bits == INFINITY_BITS {
        return value;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    f64::from_bits(if bits & SIGN_BIT == 0 {
        bits + 1
    } else {
        bits - 1
    })
}

/// Returns the adjacent representable f64 toward negative infinity.
pub(super) fn next_down(value: f64) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & MAGNITUDE_MASK;
    if magnitude > INFINITY_BITS || bits == (SIGN_BIT | INFINITY_BITS) {
        return value;
    }
    if magnitude == 0 {
        return f64::from_bits(SIGN_BIT | 1);
    }
    f64::from_bits(if bits & SIGN_BIT == 0 {
        bits - 1
    } else {
        bits + 1
    })
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
