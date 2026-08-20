//! Deterministic bounded selection of interval-overlapping candidates.

use std::cmp::Ordering;

use bytes::Bytes;

use crate::api::{Error, ErrorKind, Result};

use super::super::numeric::compare_finite;
use super::ApproximateDistance;

/// One approximately ranked candidate with an opaque caller-owned value.
pub(crate) struct ApproximateCandidate<T> {
    record_id: Bytes,
    distance: ApproximateDistance,
    value: T,
}

impl<T> ApproximateCandidate<T> {
    /// Creates a candidate whose Record ID is the deterministic tie-breaker.
    pub(crate) const fn new(record_id: Bytes, distance: ApproximateDistance, value: T) -> Self {
        Self {
            record_id,
            distance,
            value,
        }
    }

    /// Returns the Record ID used for stable unsigned byte ordering.
    #[cfg(test)]
    pub(crate) const fn record_id(&self) -> &Bytes {
        &self.record_id
    }

    /// Returns the candidate's rough distance and conservative interval.
    #[cfg(test)]
    pub(crate) const fn distance(&self) -> ApproximateDistance {
        self.distance
    }

    /// Returns the caller-owned value.
    #[cfg(test)]
    pub(crate) const fn value(&self) -> &T {
        &self.value
    }

    /// Splits the candidate into its caller-owned fields.
    pub(crate) fn into_parts(self) -> (Bytes, ApproximateDistance, T) {
        (self.record_id, self.distance, self.value)
    }
}

/// A bounded deterministic overlap-selection result.
pub(crate) struct OverlapSelection<T> {
    candidates: Vec<ApproximateCandidate<T>>,
    truncated: bool,
}

impl<T> OverlapSelection<T> {
    /// Returns whether the supplied cap discarded an otherwise selected item.
    pub(crate) const fn truncated(&self) -> bool {
        self.truncated
    }

    /// Returns the selected candidates in rough-distance ranking order.
    #[cfg(test)]
    pub(crate) fn candidates(&self) -> &[ApproximateCandidate<T>] {
        &self.candidates
    }

    /// Consumes the result and returns its selected candidates.
    #[cfg(test)]
    pub(crate) fn into_candidates(self) -> Vec<ApproximateCandidate<T>> {
        self.candidates
    }

    /// Consumes the result and returns the carried values in ranking order.
    pub(crate) fn into_values(self) -> Vec<T> {
        self.candidates
            .into_iter()
            .map(|candidate| candidate.into_parts().2)
            .collect()
    }
}

/// Selects one Leaf Partition's rough top set plus interval overlaps.
///
/// The returned set is capped at `min(4*r, remaining_rerank_budget)`. A true
/// truncation bit is the source of `rabitq_overlap_truncated` for local
/// selection.
pub(crate) fn select_leaf_overlap<T>(
    mut candidates: Vec<ApproximateCandidate<T>>,
    k: usize,
    remaining_rerank_budget: usize,
) -> Result<OverlapSelection<T>> {
    if k == 0 {
        return Err(Error::invalid_argument());
    }
    if candidates.is_empty() {
        return Ok(OverlapSelection {
            candidates,
            truncated: false,
        });
    }

    let rough_count = candidates.len().min(
        k.checked_mul(2)
            .ok_or_else(Error::invalid_argument)?
            .max(64),
    );
    let overlap_threshold = nth_upper_endpoint(&mut candidates, rough_count - 1);

    candidates.select_nth_unstable_by(rough_count - 1, compare_rough);
    let mut position = 0_usize;
    candidates.retain(|candidate| {
        let keep = position < rough_count || candidate.distance.lower() <= overlap_threshold;
        position += 1;
        keep
    });

    let overlap_cap = rough_count
        .checked_mul(4)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?
        .min(remaining_rerank_budget);
    let truncated = candidates.len() > overlap_cap;
    if truncated {
        truncate_candidates(&mut candidates, overlap_cap, compare_local_cap);
    }
    candidates.sort_unstable_by(compare_rough);

    Ok(OverlapSelection {
        candidates,
        truncated,
    })
}

/// Selects globally overlapping candidates before exact reranking.
///
/// The global threshold is the kth-smallest upper endpoint, or positive
/// infinity when fewer than `k` candidates exist. The rerank budget then caps
/// survivors by rough distance and Record ID.
pub(crate) fn select_global_overlap<T>(
    mut candidates: Vec<ApproximateCandidate<T>>,
    k: usize,
    remaining_rerank_budget: usize,
) -> Result<OverlapSelection<T>> {
    if k == 0 {
        return Err(Error::invalid_argument());
    }
    let overlap_threshold = if candidates.len() < k {
        f64::INFINITY
    } else {
        nth_upper_endpoint(&mut candidates, k - 1)
    };
    candidates.retain(|candidate| candidate.distance.lower() <= overlap_threshold);
    let truncated = candidates.len() > remaining_rerank_budget;
    if truncated {
        truncate_candidates(&mut candidates, remaining_rerank_budget, compare_rough);
    }
    candidates.sort_unstable_by(compare_rough);
    Ok(OverlapSelection {
        candidates,
        truncated,
    })
}

fn nth_upper_endpoint<T>(candidates: &mut [ApproximateCandidate<T>], index: usize) -> f64 {
    let (_, candidate, _) = candidates.select_nth_unstable_by(index, |left, right| {
        left.distance.upper().total_cmp(&right.distance.upper())
    });
    candidate.distance.upper()
}

fn compare_rough<T>(left: &ApproximateCandidate<T>, right: &ApproximateCandidate<T>) -> Ordering {
    compare_finite(left.distance.rough(), right.distance.rough())
        .then_with(|| left.record_id.cmp(&right.record_id))
}

fn compare_local_cap<T>(
    left: &ApproximateCandidate<T>,
    right: &ApproximateCandidate<T>,
) -> Ordering {
    compare_finite(left.distance.lower(), right.distance.lower())
        .then_with(|| compare_rough(left, right))
}

/// Truncates to `cap`, keeping the strongest candidates under `compare`.
fn truncate_candidates<T>(
    candidates: &mut Vec<ApproximateCandidate<T>>,
    cap: usize,
    compare: impl Fn(&ApproximateCandidate<T>, &ApproximateCandidate<T>) -> Ordering,
) {
    if cap == 0 {
        candidates.clear();
    } else {
        candidates.select_nth_unstable_by(cap - 1, compare);
        candidates.truncate(cap);
    }
}
