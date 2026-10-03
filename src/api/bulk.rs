//! Explicit controls for unpublished bulk construction.

use super::{Error, Result};

/// Controls one bounded-round, capacity-constrained bulk construction.
///
/// The byte limit applies to the supplied records (IDs, original vectors,
/// fields and payloads), not total resident memory. Construction additionally
/// retains one preprocessed vector per record and proportional membership,
/// centroid and verification workspace. The caller must provision that memory.
#[derive(Clone, Debug)]
pub struct BulkBuildOptions {
    pub(crate) rounds: usize,
    pub(crate) neighbors: usize,
    pub(crate) input_bytes: usize,
}

impl BulkBuildOptions {
    /// Creates controls with two refinement rounds and 32 neighbor centroids.
    /// `input_bytes` must be positive. Zero rounds provides the same builder
    /// without local refinement for controlled measurements.
    pub fn new(input_bytes: usize) -> Result<Self> {
        if input_bytes == 0 {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            rounds: 2,
            neighbors: 32,
            input_bytes,
        })
    }

    /// Selects zero through five local refinement rounds.
    pub fn with_refinement_rounds(mut self, rounds: usize) -> Result<Self> {
        if rounds > 5 {
            return Err(Error::invalid_argument());
        }
        self.rounds = rounds;
        Ok(self)
    }

    /// Selects one through 32 candidate neighbor centroids per leaf.
    pub fn with_neighbor_centroids(mut self, neighbors: usize) -> Result<Self> {
        if !(1..=32).contains(&neighbors) {
            return Err(Error::invalid_argument());
        }
        self.neighbors = neighbors;
        Ok(self)
    }
}
