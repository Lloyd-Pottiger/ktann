//! RocksDB adapter metric assertions (issue #36).
//!
//! The adapter emits its blocking-admission and commit observations through
//! the `metrics` facade under the `ktann.*` namespace with only the bounded
//! `backend` and `outcome` labels (design `runtime-operations.md` section 5).

use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use ktann::storage::backend::{Backend, ReadOps, WriteTxn};
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use metrics_util::debugging::{DebuggingRecorder, Snapshotter};

mod support;

use support::open_database;

fn snapshotter() -> &'static Snapshotter {
    static SNAPSHOTTER: OnceLock<Snapshotter> = OnceLock::new();
    SNAPSHOTTER.get_or_init(|| {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        recorder.install().expect("global recorder installs once");
        snapshotter
    })
}

/// Captured series as `(name, sorted (label, value) pairs)`.
fn series(snapshotter: &Snapshotter) -> Vec<(String, Vec<(String, String)>)> {
    snapshotter
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
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rocksdb_adapter_emits_bounded_backend_metrics() {
    let snapshotter = snapshotter();
    let directory = tempfile::tempdir().expect("temporary database directory");
    let database = Arc::new(open_database(&directory.path().join("database")));
    let backend = RocksDbBackend::new(
        database,
        BackendNamespace::new("ktann-observability").expect("namespace"),
    );

    let mut write = backend.begin_write().await.expect("begin write");
    write
        .put(Bytes::from_static(b"key"), Bytes::from_static(b"value"))
        .await
        .expect("put");
    write.commit().await.expect("commit");
    let mut read = backend.begin_read().await.expect("begin read");
    assert_eq!(
        read.get(Bytes::from_static(b"key"))
            .await
            .expect("snapshot get"),
        Some(Bytes::from_static(b"value")),
    );
    drop(read);
    backend.shutdown().await;

    let series = series(snapshotter);
    let expected: &[(&str, &[(&str, &str)])] = &[
        (
            "ktann.backend.commit",
            &[("backend", "rocksdb"), ("outcome", "committed")],
        ),
        ("ktann.backend.blocking.wait", &[("backend", "rocksdb")]),
        ("ktann.backend.blocking.held", &[("backend", "rocksdb")]),
    ];
    for (name, expected_labels) in expected {
        let mut sorted: Vec<(String, String)> = expected_labels
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        sorted.sort();
        assert!(
            series
                .iter()
                .any(|(series_name, labels)| series_name == name && *labels == sorted),
            "missing series {name} with labels {expected_labels:?}; captured: {series:?}"
        );
    }
    // Every emitted label stays within the documented allowlist.
    for (name, labels) in &series {
        assert!(name.starts_with("ktann."), "metric {name} leaves namespace");
        for (key, value) in labels {
            let bounded = match key.as_str() {
                "backend" => value == "rocksdb",
                "outcome" => matches!(
                    value.as_str(),
                    "committed" | "retryable" | "unknown" | "failed"
                ),
                _ => false,
            };
            assert!(bounded, "metric {name} has unbounded label {key}={value}");
        }
    }
}
