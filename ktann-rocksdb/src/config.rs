use std::thread;

use ktann::api::{Error, ErrorKind, Result};
use tokio::sync::Semaphore;

/// Process-local resource limits for one RocksDB adapter.
///
/// The blocking resource limit bounds live RocksDB snapshots and transactions.
/// Each native handle retains one slot through cleanup, so every synchronous
/// call runs under bounded adapter capacity without making `Drop` wait for it.
///
/// # Examples
///
/// ```
/// use ktann_rocksdb::RocksDbConfig;
///
/// let config = RocksDbConfig::default().with_blocking_resource_limit(4)?;
/// assert_eq!(config.blocking_resource_limit(), 4);
/// # Ok::<(), ktann::api::Error>(())
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct RocksDbConfig {
    blocking_resource_limit: usize,
}

impl Default for RocksDbConfig {
    fn default() -> Self {
        Self {
            blocking_resource_limit: thread::available_parallelism().map_or(1, usize::from),
        }
    }
}

impl RocksDbConfig {
    /// Sets the maximum number of live RocksDB transaction resource slots.
    ///
    /// Each read snapshot or write transaction reserves one slot through native
    /// cleanup. This also bounds synchronous calls because one transaction is
    /// accessed serially. A caller retaining `limit` transactions must release
    /// one before awaiting another transaction open.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidArgument`] when `limit` is zero or exceeds
    /// Tokio's semaphore capacity.
    pub fn with_blocking_resource_limit(mut self, limit: usize) -> Result<Self> {
        if limit == 0 || limit > Semaphore::MAX_PERMITS {
            return Err(Error::new(ErrorKind::InvalidArgument));
        }
        self.blocking_resource_limit = limit;
        Ok(self)
    }

    /// Returns the maximum number of live RocksDB transaction resource slots.
    #[must_use]
    pub const fn blocking_resource_limit(&self) -> usize {
        self.blocking_resource_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocking_limit_rejects_values_outside_the_semaphore_domain() {
        assert_eq!(
            RocksDbConfig::default()
                .with_blocking_resource_limit(0)
                .expect_err("zero limit")
                .kind(),
            ErrorKind::InvalidArgument,
        );
        assert_eq!(
            RocksDbConfig::default()
                .with_blocking_resource_limit(Semaphore::MAX_PERMITS + 1)
                .expect_err("oversized limit")
                .kind(),
            ErrorKind::InvalidArgument,
        );
    }
}
