//! Controls for caller-exclusive, offline refinement of an existing index.

use super::{Error, OperationOptions, Result};

/// Controls bounded-round refinement before serving an imported index.
///
/// The input limit covers loaded vectors, IDs and topology, not total resident
/// memory. Numerical planning and migration proposals require additional space.
#[derive(Clone, Debug)]
pub struct RefineOptions {
    pub(crate) rounds: usize,
    pub(crate) neighbors: usize,
    pub(crate) input_bytes: usize,
    pub(crate) operation_options: OperationOptions,
}

impl RefineOptions {
    /// Creates controls with two refinement rounds and 32 neighbor centroids.
    /// `input_bytes` must be positive. Zero rounds refreshes centroids without moving records.
    pub fn new(input_bytes: usize) -> Result<Self> {
        if input_bytes == 0 {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            rounds: 2,
            neighbors: 32,
            input_bytes,
            operation_options: OperationOptions::default(),
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
    /// Sets deadline and cancellation control for the operation.
    #[must_use]
    pub fn with_operation_options(mut self, options: OperationOptions) -> Self {
        self.operation_options = options;
        self
    }
}
