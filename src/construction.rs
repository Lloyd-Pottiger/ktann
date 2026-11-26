//! Training sample and resource budgets for Bulk Build construction.

use crate::api::{Error, ErrorKind, IndexConfig, MAX_RECORD_ID_BYTES, Result};

const BUFFER_BYTES: usize = 8 * 1024;
const ROW_OVERHEAD_BYTES: usize = 128;

/// Training sample and resource budgets for Bulk Build construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstructionOptions {
    /// Maximum deterministic training sample size.
    pub sample_items: usize,
    /// Conservative bound for algorithm-owned working buffers, excluding caller IO.
    pub memory_bytes: usize,
    /// Maximum simultaneous scratch-file bytes, including the input and all child outputs.
    pub scratch_bytes: u64,
}

impl ConstructionOptions {
    /// Checks dimension-independent invariants before persisting a build request.
    pub(crate) fn validate(self) -> Result<()> {
        if self.sample_items < 2 || self.memory_bytes == 0 || self.scratch_bytes == 0 {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }

    /// Validates that `memory_bytes` accommodates the training sample, partition capacities,
    /// and I/O working buffers for `config`, returning the remaining resident-row allowance.
    pub(crate) fn resident_rows(self, config: &IndexConfig) -> Result<usize> {
        config.validate()?;
        let dimension = config.dimension();
        let row_bound = dimension
            .checked_mul(4)
            .and_then(|n| n.checked_add(MAX_RECORD_ID_BYTES + ROW_OVERHEAD_BYTES))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let reserved_rows = self
            .sample_items
            .max(
                config
                    .max_leaf_partition_entries()
                    .max(config.max_internal_partition_entries()) as usize,
            )
            .checked_mul(8)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let reserved = row_bound
            .checked_mul(reserved_rows)
            .and_then(|n| n.checked_add(16 * BUFFER_BYTES))
            .and_then(|n| n.checked_add(dimension * 4))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        self.validate()?;
        if self.memory_bytes
            <= reserved
                .checked_add(row_bound)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?
        {
            return Err(Error::invalid_argument());
        }
        Ok((self.memory_bytes - reserved) / row_bound)
    }
}
