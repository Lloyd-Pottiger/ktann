//! Privacy-safe adapter metrics (design `runtime-operations.md` section 5).
//!
//! The adapter emits through the `metrics` facade under the same `ktann.*`
//! namespace as the core library. Labels stay within the documented
//! allowlist: `backend` is the fixed adapter name and `outcome` a bounded
//! commit category. No key, value, namespace, or native error text ever
//! becomes a label.

use ktann::api::ErrorKind;

/// The fixed `backend` label of this adapter.
const BACKEND: &str = "foundationdb";

/// Native commit outcomes by backend and bounded outcome.
const COMMIT: &str = "ktann.backend.commit";

/// Counts one native commit by its bounded outcome category.
pub(crate) fn commit(result: &ktann::api::Result<()>) {
    let outcome = match result {
        Ok(()) => "committed",
        Err(error) => match error.kind() {
            ErrorKind::RetryableAbort => "retryable",
            ErrorKind::CommitOutcomeUnknown => "unknown",
            _ => "failed",
        },
    };
    metrics::counter!(COMMIT, "backend" => BACKEND, "outcome" => outcome).increment(1);
}
