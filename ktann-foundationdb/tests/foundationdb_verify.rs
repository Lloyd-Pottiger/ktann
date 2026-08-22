//! `Index::verify` against a local FoundationDB 7.3 cluster.
//!
//! ADR 0019 bounds one audit to exactly one snapshot lifetime, so a large
//! audit may exceed FoundationDB's ordinary transaction lifetime and must run
//! against an offline copy. This test's index is small enough to verify well
//! within it: the audit completes, reports no issues, and writes nothing.

use std::sync::Arc;

use bytes::Bytes;
use foundationdb::Database;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, Metric, Mutation, Record,
    RuntimeConfig, Value, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

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

fn config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2)
        .expect("valid config")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
}

fn record(id: u8) -> Record {
    Record::new(
        Bytes::copy_from_slice(&[b'r', id]),
        Arc::from([f32::from(id), 1.0]),
        vec![Value::I64(i64::from(id % 3))],
    )
    .expect("valid record")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_verify_completes_within_one_snapshot() {
    let _network = boot_foundationdb();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let backend = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("open FoundationDB"),
        BackendNamespace::new("ktann-issue-35-verify").expect("namespace"),
    );
    clear_test_keys(&backend).await;

    let runtime = Runtime::new(
        backend,
        RuntimeConfig::default()
            .with_maintenance(0, 1)
            .expect("valid maintenance config"),
    )
    .expect("runtime");
    let index = runtime
        .create_index("verify", config())
        .await
        .expect("create index");
    index
        .batch_mutate((0..12_u8).map(|id| Mutation::Insert(record(id))).collect())
        .await
        .expect("batch insert");

    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(
        report.complete && report.issues.is_empty(),
        "issues: {:?}",
        report.issues
    );
    assert_eq!(report.objects.vector_records, 12);
    assert_eq!(report.objects.record_locations, 12);
    assert_eq!(report.objects.entries, 12);

    // A dropped index fails closed rather than auditing a missing Manifest.
    runtime.drop_index("verify").await.expect("drop index");
    let error = index
        .verify(VerifyOptions::default())
        .await
        .expect_err("dropped index");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);

    runtime.shutdown().await.expect("shutdown");

    // The runtime consumed and released its backend; clean up through a
    // fresh handle on the same namespace.
    let cleanup = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("reopen database"),
        BackendNamespace::new("ktann-issue-35-verify").expect("namespace"),
    );
    clear_test_keys(&cleanup).await;
}
