//! Filtered public search against exact truth on a split bulk-built tree.

use super::*;
use ktann::api::{BulkWorkerOptions, CompareOp, Predicate, SearchOptions, SearchRequest};

/// Records are permuted independently of the numeric filter, with exact 1D truth.
/// The beam exceeds the number of matching records, so retained traversal must
/// exhaust eligible branches and recover every match despite sparse filtering.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filtered_search_finds_sparse_matches_across_tree_levels() {
    const COUNT: u64 = 512;
    const K: usize = 5;
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(MemoryBackend::new());
    let config = IndexConfig::new(1, Metric::L2)
        .unwrap()
        .with_fields(vec![FieldSchema::new("id", DataType::I64).unwrap()])
        .unwrap()
        .with_leaf_partition_entries(2, 8)
        .unwrap()
        .with_internal_partition_entries(2, 4)
        .unwrap();
    let vector = |id: u64| ((id * 73) % COUNT) as f32;
    let input = InputSnapshot::create(
        &dir.path().join("input"),
        config,
        1 << 20,
        (0..COUNT).map(|id| {
            Record::new(
                Bytes::copy_from_slice(&id.to_be_bytes()),
                vec![vector(id)],
                vec![Value::I64(i64::try_from(id).unwrap())],
            )
        }),
    )
    .unwrap();
    let job = runtime
        .start_bulk_build("filtered", &input, construction_options())
        .await
        .unwrap();
    let work = dir.path().join("work");
    fs::create_dir(&work).unwrap();
    let worker = BulkWorkerOptions::new(work);
    job.run_worker(worker.clone()).await.unwrap();
    let index = job.complete(worker).await.unwrap();

    let threshold = COUNT - K as u64;
    let mut truth: Vec<_> = (threshold..COUNT).collect();
    truth.sort_by(|&a, &b| {
        (vector(a) - 250.5)
            .abs()
            .total_cmp(&(vector(b) - 250.5).abs())
            .then(a.cmp(&b))
    });
    let truth: Vec<_> = truth
        .into_iter()
        .take(K)
        .map(|id| Bytes::copy_from_slice(&id.to_be_bytes()))
        .collect();
    let request = SearchRequest::new(vec![250.5], K).unwrap().with_options(
        SearchOptions::default()
            .with_routing_mode(ktann::api::RoutingMode::Exact)
            .unwrap()
            .with_leaf_beam_size(8)
            .unwrap()
            .with_visited_partitions(1024)
            .unwrap(),
    );
    let request = request.with_predicate(Predicate::Compare {
        field: FieldId(0),
        op: CompareOp::GreaterOrEqual,
        value: Value::I64(i64::try_from(threshold).unwrap()),
    });
    let result = index.search(request).await.unwrap();
    assert_eq!(
        result
            .hits
            .iter()
            .map(|hit| hit.id().clone())
            .collect::<Vec<_>>(),
        truth
    );
    assert!(!result.exhausted.visited_partitions);
    runtime.shutdown().await.unwrap();
}
