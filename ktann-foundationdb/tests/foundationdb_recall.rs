//! Recall parity against a local FoundationDB 7.3 cluster (issue #100): the
//! public Runtime/Index API over the real adapter meets the recall contract
//! the deterministic-backend corpus pins. Durability across a cluster
//! restart is covered by the phased `foundationdb_durability` binary.

use std::path::Path;

use foundationdb::Database;
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

#[path = "../../tests/support/adapter_recall.rs"]
mod adapter_recall;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

#[expect(
    unsafe_code,
    reason = "the FoundationDB binding requires one process-global network boot"
)]
fn boot_foundationdb() -> foundationdb::api::NetworkAutoStop {
    // SAFETY: this integration-test binary contains one test, so it starts the
    // process-global FoundationDB network exactly once. The returned guard is
    // kept alive until every database, backend, and transaction has dropped.
    unsafe { foundationdb::boot() }
}

async fn clear_test_keys(backend: &FoundationDbBackend) {
    let mut transaction = backend.begin_write().await.expect("begin cleanup");
    transaction
        .clear_range(&KeyRange::new(b"".to_vec(), b"\xff".to_vec()))
        .await
        .expect("clear test range");
    transaction.commit().await.expect("commit cleanup");
}

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
