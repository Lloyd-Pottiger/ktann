//! Independent numeric reference shared by core search tests.

use crate::api::Metric;

/// The independent brute-force numeric oracle over original f32 vectors.
pub(super) fn exact_distance(metric: Metric, query: &[f32], record: &[f32]) -> f64 {
    let dot = |left: &[f32], right: &[f32]| {
        left.iter()
            .zip(right)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
    };
    match metric {
        Metric::L2 => query
            .iter()
            .zip(record)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            .sqrt(),
        Metric::InnerProduct => -dot(query, record),
        Metric::Cosine => {
            let norm = |vector: &[f32]| {
                vector
                    .iter()
                    .map(|&component| f64::from(component).powi(2))
                    .sum::<f64>()
                    .sqrt()
            };
            1.0 - dot(query, record) / (norm(query) * norm(record))
        }
    }
}
