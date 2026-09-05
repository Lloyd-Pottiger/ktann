//! Recall parity against a local FoundationDB 7.3 cluster (issue #100): the
//! public Runtime/Index API over the real adapter meets the recall contract
//! the deterministic-backend corpus pins. Durability across a cluster
//! restart is covered by the phased `foundationdb_durability` binary.

use std::path::Path;

use foundationdb::Database;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

#[path = "../../tests/support/adapter_recall.rs"]
mod adapter_recall;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;
mod support;

use support::{boot_foundationdb, clear_test_keys};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_recall_matches_the_corpus_contract() {
    let _network = boot_foundationdb();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let backend = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("open FoundationDB"),
        BackendNamespace::new("ktann-issue-100-recall").expect("namespace"),
    );
    clear_test_keys(&backend).await;

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/datadriven/data");
    let mut base = fixtures::read_vectors(&fixture_dir, "siftsmall_base.fvecs");
    base.truncate(1000);
    let mut queries = fixtures::read_vectors(&fixture_dir, "siftsmall_query.fvecs");
    queries.truncate(20);
    adapter_recall::run(backend, base, queries).await;

    // The runtime consumed and released its backend; clean up through a
    // fresh handle on the same namespace.
    let cleanup = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("reopen database"),
        BackendNamespace::new("ktann-issue-100-recall").expect("namespace"),
    );
    clear_test_keys(&cleanup).await;
}
