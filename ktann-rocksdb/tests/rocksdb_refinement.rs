//! Offline centroid refinement publishes graph and graph-free indexes consistently.

use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use std::sync::Arc;

#[path = "../../tests/support/refinement_contract.rs"]
mod contract;
#[path = "../../tests/support/refinement.rs"]
mod fixture;
mod support;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rocksdb_refinement_preserves_membership_graphs_and_snapshots() {
    for graph in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let database = Arc::new(support::open_database(&directory.path().join("database")));
        let open = || {
            RocksDbBackend::new(
                Arc::clone(&database),
                BackendNamespace::new("offline-refinement").unwrap(),
            )
        };
        contract::run(open(), open(), graph).await;
    }
}
