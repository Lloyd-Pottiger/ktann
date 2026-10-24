//! Drain benchmark maintenance, including Header-based rediscovery.
use super::topology;
use ktann::{api::Index, storage::backend::Backend};
use ktann_benchmarks::MetricCapture;
use serde_json::json;
use std::time::{Duration, Instant};

/// Poll only committed Headers; full membership verification is outside timing.
pub(crate) async fn settle<B: Backend>(
    index: &Index<B>,
    raw: &B,
    metrics: &MetricCapture,
    records: u64,
) -> (f64, usize, serde_json::Value) {
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut quiet = 0;
    let mut observer_seconds = 0.0;
    let mut polls = 0;
    let mut rediscovered = 0;
    loop {
        assert!(Instant::now() < deadline, "maintenance did not settle");
        if metrics.fixup_backlog() != 0 {
            quiet = 0;
            tokio::time::sleep(Duration::from_millis(25)).await;
            continue;
        }
        let start = Instant::now();
        let state = topology::snapshot(raw, index.logical_index_id(), records)
            .await
            .unwrap();
        observer_seconds += start.elapsed().as_secs_f64();
        polls += 1;
        quiet = if state.ready && metrics.fixup_backlog() == 0 {
            quiet + 1
        } else {
            0
        };
        if quiet >= 3 {
            return (
                observer_seconds,
                polls,
                json!({"topology":state.facts,"rediscovered":rediscovered}),
            );
        }
        rediscovered += state.pending.len();
        index.benchmark_rediscover_fixups(state.pending).unwrap();
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktann::api::{IndexConfig, Metric, Record, RuntimeConfig};
    use ktann::runtime::Runtime;
    use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
    use std::sync::Arc;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn settled_small_index_uses_actual_record_count() {
        let metrics = MetricCapture::install().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        let db = Arc::new(rocksdb::OptimisticTransactionDB::open(&options, dir.path()).unwrap());
        let backend = RocksDbBackend::new(db, BackendNamespace::new("settle-test").unwrap());
        let (backend, _) = ktann_benchmarks::MeasuredBackend::new(backend);
        let runtime = Runtime::new(
            backend.clone(),
            RuntimeConfig::default().with_maintenance(0, 1024).unwrap(),
        )
        .unwrap();
        let index = runtime
            .create_index("test", IndexConfig::new(2, Metric::L2).unwrap())
            .await
            .unwrap();
        index
            .insert(Record::new(bytes::Bytes::from_static(b"one"), vec![1.0, 2.0], vec![]).unwrap())
            .await
            .unwrap();
        let (_, polls, state) = tokio::time::timeout(
            Duration::from_secs(5),
            settle(&index, &backend, &metrics, 1),
        )
        .await
        .expect("small settled index must not wait for 1M records");
        assert_eq!(polls, 3);
        assert_eq!(state["topology"]["records_from_leaf_headers"], 1);
        runtime.shutdown().await.unwrap();
    }
}
