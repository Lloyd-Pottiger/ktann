//! The brute-force exact oracle for search and recall assertions.
//!
//! The numeric contract mirrors `src/search/numeric.rs` exactly: L2 exposes
//! Euclidean distance, inner product its negated dot product, and cosine
//! `1 - cos`; accumulation is scalar f64. Truth ordering is `(distance, Record
//! ID)` byte order, matching the engine's deterministic hit order.
//!
//! Filter evaluation follows SQL three-valued logic: any comparison against
//! NULL is UNKNOWN and never qualifies a record.

use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{CompareOp, Metric, Value};

/// One in-memory mirror of a committed Vector Record.
#[derive(Clone)]
pub struct ModelRecord {
    /// The original vector.
    pub vector: Arc<[f32]>,
    /// The positional typed field values.
    pub fields: Box<[Value]>,
}

/// The caller-side model of one index: every live Vector Record by Record ID.
pub type Model = std::collections::BTreeMap<Bytes, ModelRecord>;

/// The exact scalar-f64 distance between a query and a record vector.
///
/// A zero-norm input under cosine has no defined distance; it maps to
/// positive infinity so it orders last, mirroring the engine rejecting it.
#[must_use]
pub fn exact_distance(metric: Metric, query: &[f32], record: &[f32]) -> f64 {
    distance_with_norm(metric, query, norm(query), record)
}

/// The exact distance with the loop-invariant query norm precomputed.
fn distance_with_norm(metric: Metric, query: &[f32], query_norm: f64, record: &[f32]) -> f64 {
    debug_assert_eq!(query.len(), record.len());
    match metric {
        Metric::L2 => query
            .iter()
            .zip(record)
            .map(|(left, right)| {
                let delta = f64::from(*left) - f64::from(*right);
                delta * delta
            })
            .sum::<f64>()
            .sqrt(),
        Metric::InnerProduct => -dot(query, record),
        Metric::Cosine => {
            let record_norm = norm(record);
            if query_norm == 0.0 || record_norm == 0.0 {
                f64::INFINITY
            } else {
                1.0 - dot(query, record) / (query_norm * record_norm)
            }
        }
        _ => unreachable!("format v1 has exactly three metrics"),
    }
}

/// The brute-force top-`k` over one model under one filter, in engine hit
/// order.
#[must_use]
pub fn truth(
    model: &Model,
    metric: Metric,
    query: &[f32],
    k: usize,
    filter: &dyn Fn(&ModelRecord) -> bool,
) -> Vec<(Bytes, f64)> {
    let query_norm = norm(query);
    let scored: Vec<(f64, &Bytes)> = model
        .iter()
        .filter(|(_, record)| filter(record))
        .map(|(id, record)| {
            (
                distance_with_norm(metric, query, query_norm, &record.vector),
                id,
            )
        })
        .collect();
    top_k(scored, k)
}

/// The brute-force top-`k` for aligned IDs and vectors without record fields.
#[must_use]
pub fn truth_vectors(
    ids: &[Bytes],
    vectors: &[Arc<[f32]>],
    metric: Metric,
    query: &[f32],
    k: usize,
) -> Vec<(Bytes, f64)> {
    assert_eq!(
        ids.len(),
        vectors.len(),
        "oracle IDs and vectors must align"
    );
    let query_norm = norm(query);
    let scored = ids
        .iter()
        .zip(vectors)
        .map(|(id, vector)| (distance_with_norm(metric, query, query_norm, vector), id))
        .collect();
    top_k(scored, k)
}

/// Selects the exact prefix under the engine's total distance/ID order.
fn top_k(mut scored: Vec<(f64, &Bytes)>, k: usize) -> Vec<(Bytes, f64)> {
    if k == 0 {
        return Vec::new();
    }
    if k < scored.len() {
        scored.select_nth_unstable_by(k, score_order);
        scored.truncate(k);
    }
    scored.sort_by(score_order);
    scored
        .into_iter()
        .map(|(distance, id)| (id.clone(), distance))
        .collect()
}

/// Orders finite oracle distances and then canonical Record ID bytes.
fn score_order(
    (left_distance, left_id): &(f64, &Bytes),
    (right_distance, right_id): &(f64, &Bytes),
) -> std::cmp::Ordering {
    left_distance
        .partial_cmp(right_distance)
        .expect("oracle distances are finite")
        .then_with(|| left_id.cmp(right_id))
}

/// The overlap of the predicted hit set with the truth set, in `[0, 1]`.
///
/// A short prediction counts as missing items, matching the CockroachDB
/// recall definition; budget truncation is reported separately by the caller.
#[must_use]
pub fn recall_ids<'a>(
    predicted: impl IntoIterator<Item = &'a Bytes>,
    truth: &[(Bytes, f64)],
) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let predicted: std::collections::BTreeSet<&Bytes> = predicted.into_iter().collect();
    let hits = truth
        .iter()
        .filter(|(id, _)| predicted.contains(id))
        .count();
    hits as f64 / truth.len() as f64
}

/// Evaluates one typed comparison under SQL three-valued logic: a NULL field
/// or a cross-domain comparison is UNKNOWN and never qualifies.
#[must_use]
pub fn compare_3vl(op: CompareOp, field: &Value, target: &Value) -> bool {
    let Some(ordering) = typed_order(field, target) else {
        return false;
    };
    match op {
        CompareOp::Eq => ordering == std::cmp::Ordering::Equal,
        CompareOp::NotEq => ordering != std::cmp::Ordering::Equal,
        CompareOp::Lt => ordering == std::cmp::Ordering::Less,
        CompareOp::LessOrEqual => ordering != std::cmp::Ordering::Greater,
        CompareOp::Gt => ordering == std::cmp::Ordering::Greater,
        CompareOp::GreaterOrEqual => ordering != std::cmp::Ordering::Less,
        _ => unreachable!("format v1 has exactly six comparison operators"),
    }
}

/// Orders two non-NULL same-domain values; anything else is unordered.
fn typed_order(left: &Value, right: &Value) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (Value::Bool(left), Value::Bool(right)) => Some(left.cmp(right)),
        (Value::I64(left), Value::I64(right)) => Some(left.cmp(right)),
        (Value::F64(left), Value::F64(right)) => left.partial_cmp(right),
        (Value::String(left), Value::String(right)) => Some(left.cmp(right)),
        _ => None,
    }
}

fn dot(query: &[f32], record: &[f32]) -> f64 {
    query
        .iter()
        .zip(record)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|component| f64::from(*component) * f64::from(*component))
        .sum::<f64>()
        .sqrt()
}
