use std::thread;

use ktann::api::{Error, ErrorKind, Result};
use tokio::sync::Semaphore;

/// Process-local resource limits for one RocksDB adapter.
///
/// The blocking resource limit bounds live RocksDB transaction actors. Each
/// actor owns one dedicated native thread, one snapshot or write transaction,
/// and one slot until native cleanup finishes.
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
    /// Sets the maximum number of live RocksDB transaction actors.
    ///
    /// Each read snapshot or write transaction reserves one dedicated thread
    /// actor through native cleanup. Existing transactions never reacquire
    /// admission for their calls, so retaining `limit` transactions cannot
    /// prevent them from making progress; only another transaction open waits.
    /// Dropping a handle closes its bounded actor channel without waiting
    /// synchronously.
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

    /// Returns the maximum number of live RocksDB transaction actors.
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
