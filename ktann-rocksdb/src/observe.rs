//! Privacy-safe adapter metrics (design `runtime-operations.md` section 5).
//!
//! The adapter emits through the `metrics` facade under the same `ktann.*`
//! namespace as the core library. Labels stay within the documented
//! allowlist: `backend` is the fixed adapter name and `outcome` a bounded
//! commit category. No key, value, namespace, or native error text ever
//! becomes a label.

use std::time::Duration;

use ktann::api::ErrorKind;

/// The fixed `backend` label of this adapter.
const BACKEND: &str = "rocksdb";

/// Native commit outcomes by backend and bounded outcome.
const COMMIT: &str = "ktann.backend.commit";
/// Async wait for a bounded native transaction actor slot, in seconds.
const BLOCKING_WAIT: &str = "ktann.backend.blocking.wait";
/// Duration a native actor held its bounded slot, in seconds.
const BLOCKING_HELD: &str = "ktann.backend.blocking.held";

/// The bounded `outcome` label of one native commit.
#[derive(Clone, Copy)]
pub(crate) enum CommitOutcome {
    /// The commit definitely succeeded.
    Committed,
    /// The commit definitely failed and nothing was committed.
    Retryable,
    /// Whether the commit succeeded cannot be determined.
    Unknown,
    /// The commit failed with any other classified error.
    Failed,
}

impl CommitOutcome {
    /// The bounded label value.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Committed => "committed",
            Self::Retryable => "retryable",
            Self::Unknown => "unknown",
            Self::Failed => "failed",
        }
    }

    /// Classifies one commit result by its stable error category.
    pub(crate) fn from_result(result: &ktann::api::Result<()>) -> Self {
        match result {
            Ok(()) => Self::Committed,
            Err(error) => match error.kind() {
                ErrorKind::RetryableAbort => Self::Retryable,
                ErrorKind::CommitOutcomeUnknown => Self::Unknown,
                _ => Self::Failed,
            },
        }
    }
}

/// Counts one native commit's bounded outcome.
pub(crate) fn commit(outcome: CommitOutcome) {
    metrics::counter!(COMMIT, "backend" => BACKEND, "outcome" => outcome.as_str()).increment(1);
}

/// Records one async wait for a native actor slot.
pub(crate) fn blocking_wait(duration: Duration) {
    metrics::histogram!(BLOCKING_WAIT, "backend" => BACKEND).record(duration.as_secs_f64());
}

/// Records the duration one native actor held its slot.
pub(crate) fn blocking_held(duration: Duration) {
    metrics::histogram!(BLOCKING_HELD, "backend" => BACKEND).record(duration.as_secs_f64());
}
