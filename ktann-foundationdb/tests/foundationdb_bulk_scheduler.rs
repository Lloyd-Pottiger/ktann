//! Automatic scheduling across worker processes, including abrupt owner death.
use bytes::Bytes;
use foundationdb::Database;
use ktann::api::{
    BulkBuildStatus, BulkSchedulerOptions, BulkWorkerOptions, IndexConfig, Metric,
    OperationOptions, Record, RuntimeConfig, VerifyOptions,
};
use ktann::bulk::InputSnapshot;
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann::storage::backend::{Backend, ReadOps};
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};
use std::{
    process::{Child, Command},
    time::Duration,
};
mod support;

fn backend(namespace: &str) -> FoundationDbBackend {
    let cluster = std::env::var("FDB_CLUSTER_FILE").ok();
    FoundationDbBackend::new(
        Database::new(cluster.as_deref()).unwrap(),
        BackendNamespace::new(namespace).unwrap(),
    )
}
struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn spawn_worker(namespace: &str) -> Worker {
    Worker(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "foundationdb_scheduler_process",
                "--ignored",
                "--nocapture",
            ])
            .env("KTANN_FDB_SCHEDULER_CHILD", namespace)
            .spawn()
            .unwrap(),
    )
}

// One test entry point ensures each process boots the FDB network exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a local FoundationDB 7.3 client and cluster; spawns and kills test-owned workers"]
async fn foundationdb_scheduler_process() {
    let _network = support::boot_foundationdb();
    if let Ok(namespace) = std::env::var("KTANN_FDB_SCHEDULER_CHILD") {
        let runtime = Runtime::new(
            backend(&namespace),
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let options = BulkSchedulerOptions {
            max_jobs: 1,
            poll_interval: Duration::from_millis(50),
            lease_duration: Duration::from_secs(2),
        };
        let _ = runtime
            .run_bulk_scheduler(
                options,
                OperationOptions::default()
                    .with_deadline(std::time::Instant::now() + Duration::from_secs(90)),
            )
            .await;
        runtime.shutdown().await.unwrap();
        return;
    }
    let namespace = format!("ktann-scheduler-process-{}", std::process::id());
    let root = std::env::temp_dir().join(&namespace);
    std::fs::create_dir(&root).unwrap();
    let workspace = root.join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let raw = backend(&namespace);
    support::clear_test_keys(&raw).await;
    let runtime = Runtime::new(
        backend(&namespace),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let config = IndexConfig::new(32, Metric::L2)
        .unwrap()
        .with_leaf_partition_entries(32, 128)
        .and_then(|config| config.with_internal_partition_entries(32, 128))
        .unwrap();
    let source = InputSnapshot::create(
        &root.join("source"),
        config,
        32 * 1024 * 1024,
        (0..20_000_u64).map(|n| {
            Ok(Record::new(
                Bytes::copy_from_slice(&n.to_be_bytes()),
                (0..32)
                    .map(|d| ((n * 17 + d * 131) % 997) as f32)
                    .collect::<Vec<_>>(),
                vec![],
            )
            .unwrap())
        }),
    )
    .unwrap();
    let construction = ConstructionOptions {
        sample_items: 256,
        memory_bytes: 32 * 1024 * 1024,
        scratch_bytes: 128 * 1024 * 1024,
    };
    let job = runtime
        .start_bulk_build("scheduled", &source, construction)
        .await
        .unwrap();
    job.schedule(BulkWorkerOptions::new(workspace))
        .await
        .unwrap();
    let mut first = spawn_worker(&namespace);
    let queue_key: Bytes = ktann::storage::keys::build_schedule_key(job.logical_index_id()).into();
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            assert!(first.0.try_wait().unwrap().is_none());
            let bytes = raw
                .begin_read()
                .await
                .unwrap()
                .get(queue_key.clone())
                .await
                .unwrap()
                .expect("queue must exist before publication");
            if bytes[bytes.len() - 40..bytes.len() - 8]
                .iter()
                .any(|v| *v != 0)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    first.0.kill().unwrap();
    first.0.wait().unwrap();
    let mut second = spawn_worker(&namespace);
    tokio::time::timeout(Duration::from_secs(90), async {
        loop {
            assert!(second.0.try_wait().unwrap().is_none());
            match job.status().await.unwrap() {
                BulkBuildStatus::Published => {
                    if raw
                        .begin_read()
                        .await
                        .unwrap()
                        .get(queue_key.clone())
                        .await
                        .unwrap()
                        .is_none()
                    {
                        break;
                    }
                }
                BulkBuildStatus::Failed { kind } => panic!("job failed: {kind:?}"),
                _ => {}
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    let index = runtime.open_index("scheduled").await.unwrap();
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty());
    assert_eq!(report.objects.vector_records, 20_000);
    drop(second);
    runtime.drop_index("scheduled").await.unwrap();

    // Also exercise multi-process collaboration (`distributed_workers = 3`) on ONE single build,
    // including SIGKILL of one collaborating worker process mid-build and lease takeover!
    let collab_ws = root.join("collab-workspace");
    std::fs::create_dir(&collab_ws).unwrap();
    let collab_job = runtime
        .start_bulk_build("scheduled-collab", &source, construction)
        .await
        .unwrap();
    let mut collab_opts = BulkWorkerOptions::new(collab_ws);
    collab_opts.distributed_workers = 3;
    collab_opts.unit_lease = Duration::from_millis(500);
    collab_opts.parallelism = 2;
    collab_job.schedule(collab_opts).await.unwrap();
    let mut w1 = spawn_worker(&namespace);
    let mut w2 = spawn_worker(&namespace);
    let mut w3 = spawn_worker(&namespace);
    tokio::time::sleep(Duration::from_millis(150)).await;
    w1.0.kill().unwrap();
    w1.0.wait().unwrap();
    let collab_queue_key: Bytes =
        ktann::storage::keys::build_schedule_key(collab_job.logical_index_id()).into();
    tokio::time::timeout(Duration::from_secs(90), async {
        loop {
            assert!(w2.0.try_wait().unwrap().is_none());
            assert!(w3.0.try_wait().unwrap().is_none());
            match collab_job.status().await.unwrap() {
                BulkBuildStatus::Published => {
                    if raw
                        .begin_read()
                        .await
                        .unwrap()
                        .get(collab_queue_key.clone())
                        .await
                        .unwrap()
                        .is_none()
                    {
                        break;
                    }
                }
                BulkBuildStatus::Failed { kind } => panic!("collab job failed: {kind:?}"),
                _ => {}
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    let collab_idx = runtime.open_index("scheduled-collab").await.unwrap();
    let collab_rep = collab_idx.verify(VerifyOptions::default()).await.unwrap();
    assert!(collab_rep.complete && collab_rep.issues.is_empty());
    assert_eq!(collab_rep.objects.vector_records, 20_000);
    drop(w2);
    drop(w3);
    runtime.drop_index("scheduled-collab").await.unwrap();

    runtime.shutdown().await.unwrap();
    support::clear_test_keys(&raw).await;
    std::fs::remove_dir_all(root).unwrap();
}
