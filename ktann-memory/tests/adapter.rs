//! The same transaction contract exercised by the persistent adapters.

#[path = "../../tests/support/backend_contract.rs"]
mod backend_contract;

use backend_contract::{BackendHarness, Fault, FaultInjection, RestartMode};
use ktann_memory::MemoryBackend;

#[derive(Default)]
struct Harness(MemoryBackend);

impl BackendHarness for Harness {
    type Backend = MemoryBackend;

    fn backend(&self) -> &MemoryBackend {
        &self.0
    }

    fn fault_injection(&self) -> FaultInjection {
        FaultInjection::Unavailable
    }

    fn inject_fault(&self, _: Fault) {
        unreachable!("no injected faults")
    }

    fn restart_mode(&self) -> RestartMode {
        RestartMode::Ephemeral
    }

    fn restart(&self) -> Self {
        Self::default()
    }
}

#[tokio::test]
async fn shared_backend_contract() {
    let harness = Harness::default();
    backend_contract::run_suite(&harness).await;
    graph_storage::exercise(harness.backend()).await;
}

#[path = "../../tests/support/graph_storage.rs"]
mod graph_storage;
