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

#[path = "graph_updates.rs"]
mod graph_updates;

/// Exercises both manual loading and worker acceptance across a real backend
/// reopen, then checks exact bytes before publication and ordinary verification.
pub async fn exercise<B: Backend>(backend: impl Fn() -> B, directory: &Path) {
    for (name, fixed, metric) in [
        ("mutable", false, Metric::L2),
        ("fixed-l2", true, Metric::L2),
        ("fixed-cosine", true, Metric::Cosine),
        ("fixed-ip", true, Metric::InnerProduct),
    ] {
        let path = directory.join(name);
        std::fs::create_dir(&path).unwrap();
        exercise_mode(&backend, &path, fixed, metric, false, &|_, _| {}).await;
    }
    exercise_updates(&backend, &directory.join("graph-updates"), &|_, _| {}).await;
}

/// The same graph-update corpus is used by all adapters; Memory adds commit faults.
pub async fn exercise_updates<B: Backend>(
    backend: &impl Fn() -> B,
    directory: &Path,
    fault: &dyn Fn(&'static str, usize),
) {
    std::fs::create_dir(directory).unwrap();
    exercise_mode(backend, directory, true, Metric::L2, true, fault).await;
}

async fn exercise_mode<B: Backend>(
    backend: &impl Fn() -> B,
    directory: &Path,
    fixed: bool,
    metric: Metric,
    updates: bool,
    fault: &dyn Fn(&'static str, usize),
) {
    let config = IndexConfig::new(2, metric)
        .unwrap()
        .with_leaf_partition_entries(if updates { 1 } else { 4 }, if updates { 2 } else { 16 })
        .and_then(|config| {
            config.with_internal_partition_entries(2, if updates { 2048 } else { 4 })
        })
        .unwrap()
        .with_fields(vec![FieldSchema::new("bucket", DataType::I64).unwrap()])
        .unwrap()
        .with_tree_key_fields(vec![FieldId(0)])
        .unwrap();
    let config = if fixed {
        config.with_fixed_graph(
            ktann::api::FixedGraphConfig::new(
                if updates { 16 } else { 4 },
                if updates { 64 } else { 16 },
                [3; 32],
            )
            .unwrap(),
        )
    } else {
        config
    };
    let record_count: u64 = if updates { 4096 } else { 128 };
    let source = InputSnapshot::create(
        &directory.join("source"),
        config,
        1024 * 1024,
        (0..record_count).map(|id| {
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
        sample_items: 16,
        memory_bytes: if updates {
            128 * 1024 * 1024
        } else {
            2 * 1024 * 1024
        },
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
            refinement_rounds: None,
            refinement_memory_bytes: 256 * 1024 * 1024,
            sample_items: tree.sample_items,
            tree_memory_bytes: tree.memory_bytes,
            tree_scratch_bytes: tree.scratch_bytes,
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
            graph_memory_bytes: 256 * 1024 * 1024,
            memory_bytes: 8 * 1024 * 1024,
            scratch_bytes: 32 * 1024 * 1024,
            hard_limits,
        },
        4 * 1024 * 1024,
    )
    .unwrap();
    let options = BulkLoadOptions {
        max_mutations: 17,
        max_bytes: if updates { 128 * 1024 } else { 4096 },
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
    let index = reopened.complete(worker.clone()).await.unwrap();
    let verified = index
        .verify(ktann::api::VerifyOptions::default())
        .await
        .unwrap();
    assert!(
        verified.complete && verified.issues.is_empty(),
        "{:?}",
        verified.issues
    );
    assert_eq!(verified.objects.vector_records, record_count);
    if updates {
        graph_updates::exercise(&raw, &index, id, fault).await;
    }
    // Exercise real storage reads for both serving modes after durable loading.
    // The default beam is exhaustive for this small two-Tree-Key fixture.
    let modes = if fixed {
        vec![
            ktann::api::RoutingMode::Exact,
            ktann::api::RoutingMode::PartitionLocalHnsw { ef_search: 1 },
        ]
    } else {
        vec![ktann::api::RoutingMode::Exact]
    };
    for mode in modes {
        for _ in 0..2 {
            let result = index
                .search(
                    ktann::api::SearchRequest::new(vec![0.0, 1.0], 8)
                        .unwrap()
                        .with_options(
                            ktann::api::SearchOptions::default()
                                .with_routing_mode(mode)
                                .unwrap(),
                        ),
                )
                .await
                .unwrap();
            let actual: Vec<_> = result
                .hits
                .iter()
                .map(|hit| {
                    (
                        u64::from_be_bytes(hit.id().as_ref().try_into().unwrap()),
                        hit.distance(),
                    )
                })
                .collect();
            // The beam covers every leaf in this fixture. This original-vector
            // oracle is independent of both routing implementations.
            let mut expected: Vec<_> = (0..record_count)
                .map(|id| {
                    let distance = match metric {
                        Metric::L2 => id as f64,
                        Metric::Cosine => 1.0 - 1.0 / ((id as f64).powi(2) + 1.0).sqrt(),
                        Metric::InnerProduct => -1.0,
                        _ => unreachable!(),
                    };
                    (id, distance)
                })
                .collect();
            expected.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            assert_eq!(actual.len(), 8);
            for (actual, expected) in actual.iter().zip(expected.iter().take(8)) {
                assert_eq!(actual.0, expected.0);
                assert!((actual.1 - expected.1).abs() < 1e-12);
            }
        }
    }

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
