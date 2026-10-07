//! Real-adapter persistence and exact-byte checks for fenced serving loading.
use bytes::Bytes;
use ktann::api::{
    BulkBuildStatus, BulkLoadOptions, DataType, ErrorKind, FieldId, FieldSchema, IndexConfig,
    Metric, Record, RuntimeConfig, Value,
};
use ktann::bulk::{ForestArtifact, ForestOptions, InputSnapshot, ServingArtifact, ServingOptions};
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann::storage::backend::{Backend, ReadOps};
use std::path::Path;

pub async fn exercise<B: Backend>(backend: impl Fn() -> B, directory: &Path) {
    let config = IndexConfig::new(2, Metric::L2)
        .unwrap()
        .with_partition_entries(4, 16)
        .unwrap()
        .with_fields(vec![FieldSchema::new("bucket", DataType::I64).unwrap()])
        .unwrap()
        .with_tree_key_fields(vec![FieldId(0)])
        .unwrap();
    let source = InputSnapshot::create(
        &directory.join("source"),
        config,
        1024 * 1024,
        (0..128_u64).map(|id| {
            Record::new(
                Bytes::copy_from_slice(&id.to_be_bytes()),
                vec![id as f32, 1.0],
                vec![Value::I64((id % 2) as i64)],
            )
            .and_then(|r| r.with_payload(Bytes::from_static(b"data")))
        }),
    )
    .unwrap();
    let tree = ConstructionOptions {
        min_partition_entries: 4,
        max_partition_entries: 16,
        sample_items: 16,
        memory_bytes: 2 * 1024 * 1024,
        scratch_bytes: 16 * 1024 * 1024,
    };
    let raw = backend();
    let hard_limits = raw.hard_limits();
    let runtime = Runtime::new(
        raw,
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let job = runtime
        .start_bulk_build("bulk-loader", &source, tree)
        .await
        .unwrap();
    let id = job.logical_index_id();
    let (forest, _) = ForestArtifact::build(
        &directory.join("forest"),
        &source,
        *job.index_manifest().rotation_seed(),
        ForestOptions {
            tree,
            sort_memory_bytes: 1024 * 1024,
            sort_scratch_bytes: 16 * 1024 * 1024,
        },
        1024 * 1024,
    )
    .unwrap();
    let (serving, _) = ServingArtifact::build(
        &directory.join("serving"),
        &source,
        &forest,
        job.index_manifest(),
        ServingOptions {
            memory_bytes: 8 * 1024 * 1024,
            scratch_bytes: 32 * 1024 * 1024,
            hard_limits,
        },
        4 * 1024 * 1024,
    )
    .unwrap();
    let options = BulkLoadOptions {
        max_mutations: 17,
        max_bytes: 4096,
    };
    job.load_serving(&serving, options).await.unwrap();
    let root = directory.join("worker");
    std::fs::create_dir(&root).unwrap();
    let mut worker = ktann::api::BulkWorkerOptions::new(root);
    worker.sort_memory_bytes = 1024 * 1024;
    worker.sort_scratch_bytes = 16 * 1024 * 1024;
    worker.serving_memory_bytes = 8 * 1024 * 1024;
    worker.serving_scratch_bytes = 32 * 1024 * 1024;
    worker.max_artifact_bytes = 4 * 1024 * 1024;
    worker.load = options;
    job.run_worker(worker.clone()).await.unwrap();
    runtime.shutdown().await.unwrap();
    let runtime = Runtime::new(
        backend(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let reopened = runtime.open_bulk_build("bulk-loader").await.unwrap();
    assert_eq!(reopened.logical_index_id(), id);
    reopened.load_serving(&serving, options).await.unwrap();
    assert_eq!(
        reopened.status().await.unwrap(),
        BulkBuildStatus::Loaded {
            entries: serving.manifest().items()
        }
    );
    assert_eq!(
        runtime.open_index("bulk-loader").await.unwrap_err().kind(),
        ErrorKind::IndexBuilding
    );
    let raw = backend();
    let mut txn = raw.begin_read().await.unwrap();
    for entry in serving.reader().unwrap() {
        let entry = entry.unwrap();
        assert_eq!(txn.get(entry.key).await.unwrap(), Some(entry.value));
    }
    drop(txn);
    let index = reopened.publish().await.unwrap();
    let verified = index
        .verify(ktann::api::VerifyOptions::default())
        .await
        .unwrap();
    assert!(
        verified.complete && verified.issues.is_empty(),
        "{:?}",
        verified.issues
    );
    assert_eq!(verified.objects.vector_records, 128);
    assert_eq!(reopened.status().await.unwrap(), BulkBuildStatus::Published);
    runtime.drop_index("bulk-loader").await.unwrap();
    assert_eq!(
        runtime.cleanup_bulk_builds(10, None).await.unwrap().pending,
        0
    );
    let scheduled = runtime
        .start_bulk_build("bulk-scheduled", &source, tree)
        .await
        .unwrap();
    scheduled.schedule(worker).await.unwrap();
    let scheduler_runtime = runtime.clone();
    let scheduler = tokio::spawn(async move {
        scheduler_runtime
            .run_bulk_scheduler(
                ktann::api::BulkSchedulerOptions::default(),
                ktann::api::OperationOptions::default(),
            )
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            if scheduled.status().await.unwrap() == BulkBuildStatus::Published {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let index = runtime.open_index("bulk-scheduled").await.unwrap();
    let report = index
        .verify(ktann::api::VerifyOptions::default())
        .await
        .unwrap();
    assert!(report.complete && report.issues.is_empty());
    runtime.drop_index("bulk-scheduled").await.unwrap();
    runtime.shutdown().await.unwrap();
    scheduler.await.unwrap().unwrap();
}
