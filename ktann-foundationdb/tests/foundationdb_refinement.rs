//! Native FoundationDB contract for the ordinary offline refinement API.

use foundationdb::Database;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

#[path = "../../tests/support/refinement_contract.rs"]
mod contract;
#[path = "../../tests/support/refinement.rs"]
mod fixture;
mod support;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_refinement_preserves_membership_and_snapshots() {
    let _network = support::boot_foundationdb();
    let cluster = std::env::var("FDB_CLUSTER_FILE").ok();
    let namespace = format!("ktann-offline-refinement-{}", std::process::id());
    let open = || {
        FoundationDbBackend::new(
            Database::new(cluster.as_deref()).unwrap(),
            BackendNamespace::new(&namespace).unwrap(),
        )
    };
    contract::run(open(), open()).await;
    support::clear_test_keys(&open()).await;
}
