use bytes::Bytes;
use num_bigint::BigInt;
use proptest::prelude::*;

use crate::api::{ErrorKind, Metric};

use super::{
    ApproximateCandidate, ApproximateDistance, RaBitQ7, RaBitQQuery, select_global_overlap,
    select_leaf_overlap,
};

#[test]
fn zero_vector_has_the_canonical_all_zero_payload() {
    let encoded = RaBitQ7::quantize(&[0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .expect("zero vector is encodable");
    assert_eq!(encoded.len(), 21);
    assert!(encoded.iter().all(|byte| *byte == 0));

    let decoded = RaBitQ7::decode(&encoded, 9).expect("zero payload is canonical");
    let components = [1.0; 9];
    let query = RaBitQQuery::new(&components, Metric::InnerProduct).expect("valid query");
    let distance = decoded
        .approximate_distance(&query)
        .expect("zero code has a finite distance");
    assert_eq!(distance.rough(), 0.0);
    assert!(distance.lower() <= 0.0 && distance.upper() >= 0.0);
}

#[test]
fn signed_extremes_and_half_step_ties_have_golden_bytes() {
    let encoded = RaBitQ7::quantize(&[-1.0, 1.0, 0.5, -0.5]).expect("vector is encodable");
    let expected: &[u8] = &[
        0x75, 0x9d, 0x81, 0x3c, 0x02, 0x27, 0x00, 0x00, 0x6f, 0xf4, 0x23, 0x3c, 0x09, 0xff, 0x0f,
        0x82,
    ];
    assert_eq!(encoded.as_ref(), expected);
    assert_eq!(independently_decode_codes(&encoded, 4), [-63, 63, 32, -32]);

    RaBitQ7::decode(&encoded, 4).expect("golden payload decodes");
}

#[test]
fn malformed_payloads_fail_closed() {
    let bytes = RaBitQ7::quantize(&[-1.0, 0.25, 0.0]).expect("fixture is encodable");

    assert_corruption(RaBitQ7::decode(&bytes[..bytes.len() - 1], 3));
    let mut trailing = bytes.to_vec();
    trailing.push(0);
    assert_corruption(RaBitQ7::decode(&trailing, 3));

    let mut bad_sign_padding = bytes.to_vec();
    bad_sign_padding[12] |= 0x80;
    assert_corruption(RaBitQ7::decode(&bad_sign_padding, 3));

    let mut bad_magnitude_padding = bytes.to_vec();
    let last = bad_magnitude_padding.len() - 1;
    bad_magnitude_padding[last] |= 0xfc;
    assert_corruption(RaBitQ7::decode(&bad_magnitude_padding, 3));

    let mut wrong_norm = bytes.to_vec();
    wrong_norm[4..8].copy_from_slice(&0_u32.to_le_bytes());
    assert_corruption(RaBitQ7::decode(&wrong_norm, 3));

    let mut non_finite_scale = bytes.to_vec();
    non_finite_scale[0..4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
    assert_corruption(RaBitQ7::decode(&non_finite_scale, 3));

    let mut negative_zero_error = bytes.to_vec();
    negative_zero_error[8..12].copy_from_slice(&(-0.0_f32).to_bits().to_le_bytes());
    assert_corruption(RaBitQ7::decode(&negative_zero_error, 3));

    let mut negative_zero_code = vec![0_u8; 14];
    negative_zero_code[12] = 1;
    assert_corruption(RaBitQ7::decode(&negative_zero_code, 1));

    let mut noncanonical_zero = vec![0_u8; 14];
    noncanonical_zero[0..4].copy_from_slice(&1.0_f32.to_bits().to_le_bytes());
    assert_corruption(RaBitQ7::decode(&noncanonical_zero, 1));
}

#[test]
fn caller_input_and_arithmetic_fail_as_invalid_argument() {
    assert_kind(RaBitQ7::quantize(&[]), ErrorKind::InvalidArgument);
    assert_kind(RaBitQ7::quantize(&[f32::NAN]), ErrorKind::InvalidArgument);

    let encoded = RaBitQ7::quantize(&[1.0, -1.0]).expect("fixture is encodable");
    let code = RaBitQ7::decode(&encoded, 2).expect("fixture decodes");
    let short_query = RaBitQQuery::new(&[1.0], Metric::L2).expect("finite query");
    assert_kind(
        code.approximate_distance(&short_query),
        ErrorKind::InvalidArgument,
    );
    assert_kind(
        RaBitQQuery::new(&[1.0, f32::INFINITY], Metric::L2),
        ErrorKind::InvalidArgument,
    );
}

#[test]
fn finite_subnormal_and_maximum_components_remain_encodable() {
    for vector in [[f32::from_bits(1)], [f32::MAX], [-f32::MAX]] {
        let encoded = RaBitQ7::quantize(&vector).expect("every finite f32 component is encodable");
        let decoded =
            RaBitQ7::decode(&encoded, 1).expect("encoder emits a canonical extreme payload");
        let query = RaBitQQuery::new(&[1.0], Metric::InnerProduct).expect("valid query");
        let interval = decoded
            .approximate_distance(&query)
            .expect("extreme payload keeps finite scalar-f64 bounds");
        let exact = -f64::from(vector[0]);
        assert!(interval.lower() <= exact && exact <= interval.upper());
    }

    assert_eq!(RaBitQ7::encoded_len(1).expect("valid dimension"), 14);
    assert_eq!(
        RaBitQ7::encoded_len(crate::api::MAX_DIMENSION).expect("maximum dimension"),
        14_348
    );
    assert_kind(RaBitQ7::encoded_len(0), ErrorKind::InvalidArgument);
}

#[test]
fn stored_error_is_rounded_above_an_independent_reconstruction() {
    let vector = [13.25_f32, -8.75, 0.125, 91.0, -44.5, 2.0, -0.0];
    let encoded = RaBitQ7::quantize(&vector).expect("fixture is encodable");
    assert_error_is_least_exact_upper(&vector, &encoded);
}

#[test]
fn maximum_dimension_error_bound_covers_exact_rational_reconstruction() {
    let mut vector = vec![f32::from_bits(0x4a5d_13d5); crate::api::MAX_DIMENSION];
    vector[0] = f32::from_bits(0xe76a_fd68);
    let encoded = RaBitQ7::quantize(&vector).expect("maximum dimension is encodable");
    assert_error_is_least_exact_upper(&vector, &encoded);
}

#[test]
fn debug_output_redacts_codes_and_query_components() {
    let encoded = RaBitQ7::quantize(&[1.0, -2.0]).expect("fixture is encodable");
    let code = RaBitQ7::decode(&encoded, 2).expect("fixture is canonical");
    let components = [3.0, -4.0];
    let query = RaBitQQuery::new(&components, Metric::L2).expect("query is valid");
    assert_eq!(format!("{code:?}"), "RaBitQ7([REDACTED])");
    assert_eq!(format!("{query:?}"), "RaBitQQuery([REDACTED])");
}

#[test]
fn conservative_intervals_cover_mixed_exponents_and_dot_cancellation() {
    let vector = [
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
    ];
    let query = [
        f32::MAX,
        f32::MAX,
        -f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        -1.0,
        1.0,
    ];
    let encoded = RaBitQ7::quantize(&vector).expect("mixed-exponent vector is encodable");
    let decoded = RaBitQ7::decode(&encoded, vector.len()).expect("payload is canonical");
    let dot = query
        .iter()
        .zip(vector)
        .map(|(&left, right)| f64::from(left) * f64::from(right))
        .sum::<f64>();
    let squared_l2 = query
        .iter()
        .zip(vector)
        .map(|(&left, right)| {
            let difference = f64::from(left) - f64::from(right);
            difference * difference
        })
        .sum::<f64>();
    for (metric, exact) in [
        (Metric::InnerProduct, -dot),
        (Metric::Cosine, 1.0 - dot),
        (Metric::L2, squared_l2),
    ] {
        let prepared = RaBitQQuery::new(&query, metric).expect("query is valid");
        let interval = decoded
            .approximate_distance(&prepared)
            .expect("interval stays finite");
        assert!(interval.lower() <= exact && exact <= interval.upper());
    }
}

#[test]
fn leaf_overlap_obeys_formula_cap_and_stable_ordering() {
    let candidates = (0_u16..300)
        .map(|index| candidate(index, f64::from(index), 0.0, 1_000.0 + f64::from(index)))
        .collect();
    let selection = select_leaf_overlap(candidates, 1, usize::MAX).expect("selection succeeds");
    assert!(selection.truncated());
    assert_eq!(selection.candidates().len(), 256);
    assert!(
        selection
            .candidates()
            .windows(2)
            .all(|pair| pair[0].distance().rough() <= pair[1].distance().rough())
    );

    let candidates = vec![candidate(2, 1.0, 0.0, 2.0), candidate(1, 1.0, 0.0, 2.0)];
    let selection = select_leaf_overlap(candidates, 1, 1).expect("selection succeeds");
    assert!(selection.truncated());
    assert_eq!(selection.candidates()[0].record_id().as_ref(), [0, 1]);
    assert_eq!(selection.candidates()[0].value(), &1);
}

#[test]
fn global_overlap_uses_kth_upper_endpoint_then_rerank_cap() {
    let candidates = vec![
        candidate(1, 10.0, 9.0, 11.0),
        candidate(2, 20.0, 19.0, 21.0),
        candidate(3, 30.0, 10.5, 31.0),
        candidate(4, 40.0, 22.0, 41.0),
    ];
    let selection = select_global_overlap(candidates, 2, 2).expect("selection succeeds");
    assert!(selection.truncated());
    let selected = selection.into_candidates();
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].record_id().as_ref(), [0, 1]);
    let (_, distance, value) = selected
        .into_iter()
        .nth(1)
        .expect("second candidate exists")
        .into_parts();
    assert_eq!(distance.rough(), 20.0);
    assert_eq!(value, 2);

    assert_kind(
        select_global_overlap::<u16>(Vec::new(), 0, 1),
        ErrorKind::InvalidArgument,
    );
    let empty = select_global_overlap(vec![candidate(1, 1.0, 0.0, 2.0)], 2, 0)
        .expect("zero budget is a valid hard cap");
    assert!(empty.truncated());
    assert!(empty.candidates().is_empty());
}

proptest! {
    #[test]
    fn payloads_are_deterministic_and_intervals_cover_brute_force(
        pairs in prop::collection::vec((-10_000_i16..=10_000, -10_000_i16..=10_000), 1..65)
    ) {
        let vector: Vec<f32> = pairs.iter().map(|(value, _)| f32::from(*value) / 8.0).collect();
        let query: Vec<f32> = pairs.iter().map(|(_, value)| f32::from(*value) / 16.0).collect();
        let first = RaBitQ7::quantize(&vector).expect("generated vector is encodable");
        let second = RaBitQ7::quantize(&vector).expect("same vector is encodable");
        prop_assert_eq!(&first, &second);
        assert_error_is_least_exact_upper(&vector, &first);
        let decoded = RaBitQ7::decode(&first, vector.len())
            .expect("encoder emits canonical bytes");

        let dot = query.iter().zip(&vector)
            .map(|(&left, &right)| f64::from(left) * f64::from(right))
            .sum::<f64>();
        let squared_l2 = query.iter().zip(&vector)
            .map(|(&left, &right)| {
                let difference = f64::from(left) - f64::from(right);
                difference * difference
            })
            .sum::<f64>();
        for (metric, exact) in [
            (Metric::InnerProduct, -dot),
            (Metric::Cosine, 1.0 - dot),
            (Metric::L2, squared_l2),
        ] {
            let prepared = RaBitQQuery::new(&query, metric).expect("generated query is valid");
            let interval = decoded.approximate_distance(&prepared)
                .expect("generated distance is finite");
            prop_assert!(interval.lower() <= exact,
                "lower {} excluded exact {} for {:?}", interval.lower(), exact, metric);
            prop_assert!(exact <= interval.upper(),
                "upper {} excluded exact {} for {:?}", interval.upper(), exact, metric);
        }
    }

    #[test]
    fn intervals_cover_brute_force_across_f32_exponents(
        vector in finite_f32(),
        query in finite_f32(),
    ) {
        let encoded = RaBitQ7::quantize(&[vector]).expect("finite scalar is encodable");
        assert_error_is_least_exact_upper(&[vector], &encoded);
        let decoded = RaBitQ7::decode(&encoded, 1).expect("encoded scalar is canonical");
        let dot = f64::from(query) * f64::from(vector);
        let difference = f64::from(query) - f64::from(vector);
        for (metric, exact) in [
            (Metric::InnerProduct, -dot),
            (Metric::Cosine, 1.0 - dot),
            (Metric::L2, difference * difference),
        ] {
            let components = [query];
            let prepared = RaBitQQuery::new(&components, metric).expect("finite query is valid");
            let interval = decoded.approximate_distance(&prepared)
                .expect("f32 inputs stay finite in scalar f64");
            prop_assert!(interval.lower() <= exact && exact <= interval.upper());
        }
    }

    #[test]
    fn optimized_overlap_selection_matches_brute_force_sorting(
        raw in prop::collection::vec((-1_000_i16..=1_000, 0_u8..=20, 0_u8..=20), 1..100),
        k in 1_usize..50,
        budget in 0_usize..300,
    ) {
        let specs: Vec<CandidateSpec> = raw.into_iter().enumerate().map(
            |(index, (rough, lower_width, upper_width))| CandidateSpec {
                id: index as u16,
                rough: f64::from(rough),
                lower: f64::from(rough) - f64::from(lower_width),
                upper: f64::from(rough) + f64::from(upper_width),
            },
        ).collect();

        let actual_leaf = select_leaf_overlap(build_candidates(&specs), k, budget)
            .expect("generated leaf selection is valid");
        let (expected_leaf, expected_leaf_truncated) = brute_force_leaf(&specs, k, budget);
        prop_assert_eq!(actual_leaf.truncated(), expected_leaf_truncated);
        prop_assert_eq!(selected_ids(actual_leaf), expected_leaf);

        let actual_global = select_global_overlap(build_candidates(&specs), k, budget)
            .expect("generated global selection is valid");
        let (expected_global, expected_global_truncated) = brute_force_global(&specs, k, budget);
        prop_assert_eq!(actual_global.truncated(), expected_global_truncated);
        prop_assert_eq!(selected_ids(actual_global), expected_global);
    }
}

#[derive(Clone, Copy)]
struct CandidateSpec {
    id: u16,
    rough: f64,
    lower: f64,
    upper: f64,
}

fn build_candidates(specs: &[CandidateSpec]) -> Vec<ApproximateCandidate<u16>> {
    specs
        .iter()
        .map(|spec| candidate(spec.id, spec.rough, spec.lower, spec.upper))
        .collect()
}

fn selected_ids(selection: super::OverlapSelection<u16>) -> Vec<u16> {
    selection
        .into_candidates()
        .into_iter()
        .map(|candidate| candidate.into_parts().2)
        .collect()
}

fn brute_force_leaf(specs: &[CandidateSpec], k: usize, budget: usize) -> (Vec<u16>, bool) {
    let rough_count = specs.len().min(k.saturating_mul(2).max(64));
    let mut upper_order = specs.to_vec();
    upper_order.sort_unstable_by(|left, right| left.upper.total_cmp(&right.upper));
    let threshold = upper_order[rough_count - 1].upper;

    let mut rough_order = specs.to_vec();
    rough_order.sort_unstable_by(compare_spec_rough);
    let rough_ids: Vec<u16> = rough_order
        .iter()
        .take(rough_count)
        .map(|spec| spec.id)
        .collect();
    let mut retained: Vec<CandidateSpec> = specs
        .iter()
        .copied()
        .filter(|spec| rough_ids.contains(&spec.id) || spec.lower <= threshold)
        .collect();
    let cap = rough_count.saturating_mul(4).min(budget);
    let truncated = retained.len() > cap;
    if truncated {
        retained.sort_unstable_by(compare_spec_local_cap);
        retained.truncate(cap);
    }
    retained.sort_unstable_by(compare_spec_rough);
    (
        retained.into_iter().map(|spec| spec.id).collect(),
        truncated,
    )
}

fn brute_force_global(specs: &[CandidateSpec], k: usize, budget: usize) -> (Vec<u16>, bool) {
    let threshold = if specs.len() < k {
        f64::INFINITY
    } else {
        let mut upper_order = specs.to_vec();
        upper_order.sort_unstable_by(|left, right| left.upper.total_cmp(&right.upper));
        upper_order[k - 1].upper
    };
    let mut retained: Vec<CandidateSpec> = specs
        .iter()
        .copied()
        .filter(|spec| spec.lower <= threshold)
        .collect();
    retained.sort_unstable_by(compare_spec_rough);
    let truncated = retained.len() > budget;
    retained.truncate(budget);
    (
        retained.into_iter().map(|spec| spec.id).collect(),
        truncated,
    )
}

fn compare_spec_rough(left: &CandidateSpec, right: &CandidateSpec) -> std::cmp::Ordering {
    left.rough
        .total_cmp(&right.rough)
        .then_with(|| left.id.cmp(&right.id))
}

fn compare_spec_local_cap(left: &CandidateSpec, right: &CandidateSpec) -> std::cmp::Ordering {
    left.lower
        .total_cmp(&right.lower)
        .then_with(|| compare_spec_rough(left, right))
}

fn candidate(index: u16, rough: f64, lower: f64, upper: f64) -> ApproximateCandidate<u16> {
    assert!(lower <= rough && rough <= upper);
    let distance = ApproximateDistance {
        rough,
        lower,
        upper,
    };
    ApproximateCandidate::new(
        Bytes::copy_from_slice(&index.to_be_bytes()),
        distance,
        index,
    )
}

fn finite_f32() -> impl Strategy<Value = f32> {
    any::<u32>().prop_filter_map("finite f32", |bits| {
        let value = f32::from_bits(bits);
        value.is_finite().then_some(value)
    })
}

fn independently_decode_codes(encoded: &[u8], dimension: usize) -> Vec<i8> {
    let magnitude_start = 12 + dimension.div_ceil(8);
    (0..dimension)
        .map(|index| {
            let negative = encoded[12 + index / 8] & (1 << (index % 8)) != 0;
            let bit_offset = index * 6;
            let byte_index = bit_offset / 8;
            let shift = bit_offset % 8;
            let low = u16::from(encoded[magnitude_start + byte_index]);
            let high = encoded
                .get(magnitude_start + byte_index + 1)
                .copied()
                .map(u16::from)
                .unwrap_or(0);
            let magnitude = (((low | (high << 8)) >> shift) & 0x3f) as i8;
            if negative { -magnitude } else { magnitude }
        })
        .collect()
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn assert_error_is_least_exact_upper(vector: &[f32], encoded: &[u8]) {
    let scale = scaled_f32_integer(f32::from_bits(read_u32_le(encoded, 0)));
    let signed_codes = independently_decode_codes(encoded, vector.len());
    let exact_squared_error = vector.iter().zip(signed_codes).fold(
        BigInt::from(0_u8),
        |sum, (&component, signed_code)| {
            let difference = scaled_f32_integer(component) - &scale * signed_code;
            sum + &difference * &difference
        },
    );

    let stored_bits = read_u32_le(encoded, 8);
    let stored = scaled_f32_integer(f32::from_bits(stored_bits));
    assert!(&stored * &stored >= exact_squared_error);
    if stored_bits != 0 {
        let previous = scaled_f32_integer(f32::from_bits(stored_bits - 1));
        assert!(&previous * &previous < exact_squared_error);
    }
}

fn scaled_f32_integer(value: f32) -> BigInt {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    let significand = if exponent == 0 {
        fraction
    } else {
        (1 << 23) | fraction
    };
    let shift = if exponent == 0 { 0 } else { exponent - 1 };
    let integer = BigInt::from(significand) << shift as usize;
    if bits >> 31 == 0 { integer } else { -integer }
}

fn assert_corruption<T>(result: crate::api::Result<T>) {
    assert_kind(result, ErrorKind::Corruption);
}

fn assert_kind<T>(result: crate::api::Result<T>, expected: ErrorKind) {
    let error = result.err().expect("operation must fail");
    assert_eq!(error.kind(), expected);
}
