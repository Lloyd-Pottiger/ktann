//! Fixed publication and operation capabilities through public entrypoints.
use super::*;
use ktann::api::{
    BulkBuildOptions, BulkBuildStatus, BulkWorkerOptions, LocalGraphConfig, RefineOptions,
    SearchRequest,
};

fn worker(dir: &Directory) -> BulkWorkerOptions {
    let root = dir.0.join("work");
    fs::create_dir(&root).unwrap();
    BulkWorkerOptions::new(root)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fixed_publication_preserves_exact_membership_and_rejects_every_mutation() {
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        for rounds in [None, Some(0), Some(2)] {
            for count in [0, 1, 73] {
                let dir = Directory::new();
                let backend = MemoryBackend::new();
                let runtime = runtime(backend);
                let config = config(metric, true)
                    .with_fixed_graph(LocalGraphConfig::new(2, 8, [9; 32]).unwrap());
                assert_eq!(
                    runtime
                        .create_index("ordinary", config.clone())
                        .await
                        .unwrap_err()
                        .kind(),
                    ErrorKind::InvalidArgument
                );
                let input = InputSnapshot::create(
                    &dir.0.join("source"),
                    config,
                    1 << 20,
                    (0..count).map(|id| Ok(record(id))),
                )
                .unwrap();
                let request = BulkBuildOptions {
                    construction: construction_options(),
                    refinement_rounds: rounds,
                };
                let job = runtime
                    .start_bulk_build("fixed", &input, request)
                    .await
                    .unwrap();
                assert_eq!(
                    runtime.open_index("fixed").await.unwrap_err().kind(),
                    ErrorKind::IndexBuilding
                );
                let options = worker(&dir);
                job.run_worker(options.clone()).await.unwrap();
                assert!(matches!(
                    job.status().await.unwrap(),
                    BulkBuildStatus::Loaded { .. }
                ));
                // Accepted Serving bytes survive loss of the source and a worker takeover.
                fs::remove_dir_all(dir.0.join("source")).unwrap();
                job.run_worker(options.clone()).await.unwrap();
                let index = job.complete(options.clone()).await.unwrap();
                let verification = index.verify(VerifyOptions::default()).await.unwrap();
                assert!(
                    verification.complete && verification.issues.is_empty(),
                    "{:?}",
                    verification.issues
                );
                assert_eq!(verification.objects.vector_records, count);
                assert_eq!(
                    index.insert(record(999)).await.unwrap_err().kind(),
                    ErrorKind::IndexReadOnly
                );
                assert_eq!(
                    index.upsert(record(0)).await.unwrap_err().kind(),
                    ErrorKind::IndexReadOnly
                );
                assert_eq!(
                    index
                        .delete(Bytes::from_static(b"missing"))
                        .await
                        .unwrap_err()
                        .kind(),
                    ErrorKind::IndexReadOnly
                );
                assert_eq!(
                    index.batch_mutate(Vec::new()).await.unwrap_err().kind(),
                    ErrorKind::IndexReadOnly
                );
                assert_eq!(
                    index
                        .refine(
                            RefineOptions::new(1 << 20)
                                .unwrap()
                                .with_refinement_rounds(0)
                                .unwrap()
                        )
                        .await
                        .unwrap_err()
                        .kind(),
                    ErrorKind::IndexReadOnly
                );
                let result = index
                    .search(SearchRequest::new(vec![1.0, 1.0, 1.0], 1).unwrap())
                    .await
                    .unwrap();
                assert_eq!(result.hits.len(), usize::from(count > 0));
                assert_eq!(
                    runtime
                        .open_index("fixed")
                        .await
                        .unwrap()
                        .logical_index_id(),
                    index.logical_index_id()
                );
                assert_eq!(
                    job.complete(options).await.unwrap().logical_index_id(),
                    index.logical_index_id()
                );
                runtime.drop_index("fixed").await.unwrap();
                runtime.shutdown().await.unwrap();
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fixed_preparation_failure_is_terminal_hidden_and_identity_bound() {
    for refinement in [false, true] {
        let dir = Directory::new();
        let runtime = runtime(MemoryBackend::new());
        let config = config(Metric::L2, false)
            .with_fixed_graph(LocalGraphConfig::new(2, 8, [1; 32]).unwrap());
        let input = InputSnapshot::create(
            &dir.0.join("source"),
            config,
            1 << 20,
            (0..73).map(|id| Ok(record(id))),
        )
        .unwrap();
        let request = BulkBuildOptions {
            construction: construction_options(),
            refinement_rounds: refinement.then_some(2),
        };
        let job = runtime
            .start_bulk_build("fixed", &input, request)
            .await
            .unwrap();
        let different = BulkBuildOptions {
            refinement_rounds: Some(0),
            ..request
        };
        assert!(
            runtime
                .start_bulk_build("fixed", &input, different)
                .await
                .is_err()
        );
        let mut options = worker(&dir);
        if refinement {
            options.refinement_memory_bytes = 1;
        } else {
            options.graph_memory_bytes = 1;
        }
        assert_eq!(
            job.run_worker(options.clone()).await.unwrap_err().kind(),
            ErrorKind::LimitExceeded
        );
        assert_eq!(
            job.status().await.unwrap(),
            BulkBuildStatus::Failed {
                kind: ErrorKind::LimitExceeded
            }
        );
        assert_eq!(
            runtime.open_index("fixed").await.unwrap_err().kind(),
            ErrorKind::IndexBuilding
        );
        assert_eq!(
            job.run_worker(options).await.unwrap_err().kind(),
            ErrorKind::LimitExceeded
        );
        job.abort().await.unwrap();
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_load_checks_refinement_identity_and_graph_bytes_before_publish() {
    let dir = Directory::new();
    let memory = MemoryBackend::new();
    let runtime = runtime(memory.clone());
    let config =
        config(Metric::L2, false).with_fixed_graph(LocalGraphConfig::new(2, 8, [1; 32]).unwrap());
    let input = InputSnapshot::create(
        &dir.0.join("source"),
        config,
        1 << 20,
        (0..73).map(|id| Ok(record(id))),
    )
    .unwrap();
    let job = runtime
        .start_bulk_build(
            "fixed",
            &input,
            BulkBuildOptions {
                construction: construction_options(),
                refinement_rounds: Some(0),
            },
        )
        .await
        .unwrap();
    let (forest, _) = ForestArtifact::build(
        &dir.0.join("wrong-forest"),
        &input,
        *job.index_manifest().rotation_seed(),
        options(),
        1 << 24,
    )
    .unwrap();
    let (serving, _) = ServingArtifact::build(
        &dir.0.join("wrong-serving"),
        &input,
        &forest,
        job.index_manifest(),
        serving_options(),
        1 << 24,
    )
    .unwrap();
    assert_eq!(
        job.load_serving(&serving, ktann::api::BulkLoadOptions::default())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidArgument
    );
    let options = worker(&dir);
    job.run_worker(options).await.unwrap();
    let mut txn = memory.begin_write().await.unwrap();
    let tree = ktann::storage::keys::TreeKey::encode(&[], &[]).unwrap();
    txn.delete(Bytes::from(keys::graph_chunk_key(
        job.logical_index_id(),
        &tree,
        ktann::api::PartitionKey::new(1).unwrap(),
        0,
    )))
    .await
    .unwrap();
    txn.commit().await.unwrap();
    assert_eq!(
        job.publish().await.unwrap_err().kind(),
        ErrorKind::Corruption
    );
    assert_eq!(
        runtime.open_index("fixed").await.unwrap_err().kind(),
        ErrorKind::IndexBuilding
    );
    job.abort().await.unwrap();
    runtime.shutdown().await.unwrap();
}
