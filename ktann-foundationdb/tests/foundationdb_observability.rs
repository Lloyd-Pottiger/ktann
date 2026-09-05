//! FoundationDB adapter metric assertions (issue #36), gated on a local
//! FoundationDB 7.3 cluster like the rest of the adapter integration suite.

use bytes::Bytes;
use foundationdb::Database;
use ktann::storage::backend::{Backend, WriteTxn};
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};
use metrics_util::debugging::DebuggingRecorder;

mod support;

use support::{boot_foundationdb, clear_test_keys};

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_adapter_emits_bounded_commit_metrics() {
    let _network = boot_foundationdb();
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder.install().expect("global recorder installs once");

    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let database = Database::new(cluster_file.as_deref()).expect("open FoundationDB");
    let backend = FoundationDbBackend::new(
        database,
        BackendNamespace::new("ktann-observability").expect("namespace"),
    );
    clear_test_keys(&backend).await;

    let mut write = backend.begin_write().await.expect("begin write");
    write
        .put(Bytes::from_static(b"key"), Bytes::from_static(b"value"))
        .await
        .expect("put");
    write.commit().await.expect("commit");

    let series: Vec<(String, Vec<(String, String)>)> = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(key, _unit, _description, _value)| {
            let mut labels: Vec<(String, String)> = key
                .key()
                .labels()
                .map(|label| (label.key().to_owned(), label.value().to_owned()))
                .collect();
            labels.sort();
            (key.key().name().to_owned(), labels)
        })
        .collect();
    let committed = series.iter().any(|(name, labels)| {
        name == "ktann.backend.commit"
            && *labels
                == vec![
                    ("backend".to_owned(), "foundationdb".to_owned()),
                    ("outcome".to_owned(), "committed".to_owned()),
                ]
    });
    assert!(
        committed,
        "missing bounded commit series; captured: {series:?}"
    );
}
