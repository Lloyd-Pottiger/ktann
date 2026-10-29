//! Approximately ranked candidates with caller-owned snapshot references.

use bytes::Bytes;

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
    pub(crate) const fn record_id(&self) -> &Bytes {
        &self.record_id
    }

    /// Returns the conservative distance used by selection.
    pub(crate) const fn distance(&self) -> ApproximateDistance {
        self.distance
    }

    /// Returns the caller-owned value.
    pub(crate) const fn value(&self) -> &T {
        &self.value
    }

    /// Splits the candidate into its caller-owned fields.
    pub(crate) fn into_parts(self) -> (Bytes, ApproximateDistance, T) {
        (self.record_id, self.distance, self.value)
    }
}
