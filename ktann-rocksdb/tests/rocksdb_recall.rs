//! Recall parity on the embedded RocksDB backend (issue #100): the public
//! Runtime/Index API over the real adapter meets the recall contract the
//! deterministic-backend corpus pins, including across an orderly close and
//! reopen of the database path.

use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::RuntimeConfig;
use ktann::runtime::Runtime;
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};

#[path = "../../tests/support/adapter_recall.rs"]
mod adapter_recall;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;
mod support;

use support::open_database;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rocksdb_recall_matches_the_corpus_contract() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let database_path = directory.path().join("database");
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/datadriven/data");
    let mut base = fixtures::read_vectors(&fixture_dir, "siftsmall_base.fvecs");
    base.truncate(1000);
    let mut queries = fixtures::read_vectors(&fixture_dir, "siftsmall_query.fvecs");
    queries.truncate(20);

    let namespace = BackendNamespace::new("ktann-issue-100-recall").expect("namespace");
    let database = Arc::new(open_database(&database_path));
    let backend = RocksDbBackend::new(Arc::clone(&database), namespace.clone());
    adapter_recall::run(backend, base.clone(), queries.clone()).await;

    // The runtime drained and released the database; an orderly close and a
    // fresh handle on the same path preserve both indexes and their recall.
    drop(database);
    let database = Arc::new(open_database(&database_path));
    let backend = RocksDbBackend::new(Arc::clone(&database), namespace);
    let runtime = Runtime::new(backend, RuntimeConfig::default()).expect("runtime");
    let index = runtime.open_index("split").await.expect("open split index");
    let value = adapter_recall::recall(&index, &base, &queries, None).await;
    assert!(
        value >= adapter_recall::SETTLED_RECALL_FLOOR,
        "recall after reopen {value} below the floor"
    );
    index
        .get(Bytes::from_static(b"r000042"), Default::default())
        .await
        .expect("get")
        .expect("record survives reopen");
    runtime.shutdown().await.expect("shutdown");
}
