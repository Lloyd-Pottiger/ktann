//! Production routing through published fixed indexes; quality is not an exactness promise.
use super::*;
use ktann::api::{
    BulkBuildOptions, BulkWorkerOptions, CompareOp, LocalGraphConfig, OperationOptions, Predicate,
    RoutingMode, SearchOptions, SearchOutcome, SearchRequest,
};
use std::time::Instant;

const HNSW: RoutingMode = RoutingMode::PartitionLocalHnsw { ef_search: 1 };

fn vector(id: u64) -> Vec<f32> {
    // Repeated vectors exercise centroid and final-hit ties independently of IDs.
    let group = id / 3;
    vec![
        group as f32 + 1.0,
        (group % 7) as f32 + 1.0,
        (group % 11) as f32 + 1.0,
    ]
}

fn row(id: u64) -> Record {
    Record::new(
        Bytes::copy_from_slice(&id.to_be_bytes()),
        vector(id),
        vec![Value::I64((id % 3) as i64), Value::string("a").unwrap()],
    )
    .unwrap()
}

async fn build(
    backend: MemoryBackend,
    metric: Metric,
    keyed: bool,
    count: u64,
) -> Runtime<MemoryBackend> {
    let dir = Directory::new();
    let runtime = runtime(backend);
    let input = InputSnapshot::create(
        &dir.0.join("input"),
        config(metric, keyed).with_fixed_graph(LocalGraphConfig::new(2, 8, [41; 32]).unwrap()),
        1 << 20,
        (0..count).map(|id| Ok(row(id))),
    )
    .unwrap();
    let job = runtime
        .start_bulk_build(
            "routing",
            &input,
            BulkBuildOptions {
                construction: construction_options(),
                refinement_rounds: None,
            },
        )
        .await
        .unwrap();
    let work = dir.0.join("work");
    fs::create_dir(&work).unwrap();
    let worker = BulkWorkerOptions::new(work);
    job.run_worker(worker.clone()).await.unwrap();
    let index = job.complete(worker).await.unwrap();
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty(), "{report:?}");
    runtime
}

fn request(mode: RoutingMode, beam: u32, filtered: bool) -> SearchRequest {
    let request = SearchRequest::new(vec![1.0, 2.0, 3.0], 9)
        .unwrap()
        .with_options(
            SearchOptions::default()
                .with_routing_mode(mode)
                .unwrap()
                .with_leaf_beam_size(beam)
                .unwrap(),
        );
    if filtered {
        request.with_predicate(Predicate::Compare {
            field: FieldId(0),
            op: CompareOp::Eq,
            value: Value::I64(1),
        })
    } else {
        request
    }
}

fn signature(outcome: &SearchOutcome) -> Vec<(Vec<u8>, u64)> {
    outcome
        .hits
        .iter()
        .map(|hit| (hit.id().to_vec(), hit.distance().to_bits()))
        .collect()
}

// Independent original-vector oracle, with no routing/core helper calls.
fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f64 {
    let dot = || {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
    };
    match metric {
        Metric::L2 => a
            .iter()
            .zip(b)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            .sqrt(),
        Metric::InnerProduct => -dot(),
        Metric::Cosine => {
            let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            1.0 - dot() / (norm(a) * norm(b))
        }
        _ => unreachable!(),
    }
}

fn assert_valid(outcome: &SearchOutcome, metric: Metric, filtered: bool) {
    let mut seen = std::collections::HashSet::new();
    for hit in &outcome.hits {
        let id = u64::from_be_bytes(hit.id().as_ref().try_into().unwrap());
        assert!(seen.insert(id));
        if filtered {
            assert_eq!(id % 3, 1);
        }
        let expected = distance(metric, &[1.0, 2.0, 3.0], &vector(id));
        assert!((hit.distance() - expected).abs() <= 1e-12 * expected.abs().max(1.0));
    }
    assert!(outcome.hits.windows(2).all(|pair| {
        pair[0].distance() < pair[1].distance()
            || (pair[0].distance() == pair[1].distance() && pair[0].id() < pair[1].id())
    }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_modes_metrics_filters_ties_and_cache_turnover() {
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        for keyed in [false, true] {
            let memory = MemoryBackend::new();
            let builder = build(memory.clone(), metric, keyed, 129).await;
            let mut baselines = Vec::new();
            for cache_bytes in [0, 32 * 1024, 16 * 1024 * 1024] {
                let runtime = Runtime::new(
                    memory.clone(),
                    RuntimeConfig::default()
                        .with_maintenance(0, 1)
                        .unwrap()
                        .with_partition_cache_bytes(cache_bytes)
                        .unwrap(),
                )
                .unwrap();
                let index = runtime.open_index("routing").await.unwrap();
                let mut signatures = Vec::new();
                for filtered in [false, true] {
                    for beam in [2, 128] {
                        for mode in [RoutingMode::Exact, HNSW] {
                            let cold = index.search(request(mode, beam, filtered)).await.unwrap();
                            let warm = index.search(request(mode, beam, filtered)).await.unwrap();
                            assert_valid(&cold, metric, filtered);
                            assert_eq!(signature(&cold), signature(&warm));
                            assert_eq!(cold.usage, warm.usage);
                            assert_eq!(cold.exhausted, warm.exhausted);
                            assert_eq!(
                                cold.rabitq_overlap_truncated,
                                warm.rabitq_overlap_truncated
                            );
                            // Wide local beams enumerate every child in this fixture.
                            // This is a deterministic exhaustive case, not an ANN recall assertion.
                            if beam == 128 {
                                let mut oracle: Vec<_> = (0..129_u64)
                                    .filter(|id| !filtered || id % 3 == 1)
                                    .map(|id| (id, distance(metric, &[1.0, 2.0, 3.0], &vector(id))))
                                    .collect();
                                oracle.sort_by(|a, b| {
                                    a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0))
                                });
                                let ids: Vec<_> = cold
                                    .hits
                                    .iter()
                                    .map(|hit| {
                                        u64::from_be_bytes(hit.id().as_ref().try_into().unwrap())
                                    })
                                    .collect();
                                assert_eq!(
                                    ids,
                                    oracle.iter().take(9).map(|x| x.0).collect::<Vec<_>>()
                                );
                            }
                            signatures.push((signature(&cold), cold.usage, cold.exhausted));
                        }
                    }
                }
                if baselines.is_empty() {
                    baselines = signatures;
                } else {
                    assert_eq!(baselines, signatures);
                }
                runtime.shutdown().await.unwrap();
            }
            builder.shutdown().await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validation_empty_leaf_only_budgets_and_control() {
    assert_eq!(SearchOptions::default().routing_mode(), RoutingMode::Exact);
    assert_eq!(
        SearchOptions::default()
            .with_routing_mode(RoutingMode::PartitionLocalHnsw { ef_search: 0 })
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidArgument
    );
    let ordinary = runtime(MemoryBackend::new());
    let index = ordinary
        .create_index("ordinary", config(Metric::L2, false))
        .await
        .unwrap();
    assert_eq!(
        index
            .search(request(HNSW, 2, false))
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidArgument
    );
    ordinary.shutdown().await.unwrap();
    for count in [0, 1, 73] {
        let memory = MemoryBackend::new();
        let builder = build(memory.clone(), Metric::L2, true, count).await;
        let index = builder.open_index("routing").await.unwrap();
        for mode in [
            RoutingMode::Exact,
            HNSW,
            RoutingMode::PartitionLocalHnsw {
                ef_search: usize::MAX,
            },
        ] {
            let result = index.search(request(mode, 128, false)).await.unwrap();
            assert_eq!(result.hits.len(), (count as usize).min(9));
            let limited = request(mode, 2, false).with_options(
                SearchOptions::default()
                    .with_routing_mode(mode)
                    .unwrap()
                    .with_leaf_beam_size(2)
                    .unwrap()
                    .with_visited_partitions(1)
                    .unwrap(),
            );
            let result = index.search(limited).await.unwrap();
            if count == 73 {
                assert_eq!(result.usage.visited_partitions, 1);
                assert!(result.exhausted.visited_partitions);
            }
            let cancellation = tokio_util::sync::CancellationToken::new();
            cancellation.cancel();
            assert_eq!(
                index
                    .search_with_control(
                        request(mode, 2, false),
                        OperationOptions::default().with_cancellation(cancellation)
                    )
                    .await
                    .unwrap_err()
                    .kind(),
                ErrorKind::Cancelled
            );
            assert_eq!(
                index
                    .search_with_control(
                        request(mode, 2, false),
                        OperationOptions::default().with_deadline(Instant::now())
                    )
                    .await
                    .unwrap_err()
                    .kind(),
                ErrorKind::DeadlineExceeded
            );
        }
        let bounded = Runtime::new(
            memory,
            RuntimeConfig::default()
                .with_maintenance(0, 1)
                .unwrap()
                .with_routing_working_bytes(1)
                .unwrap(),
        )
        .unwrap();
        let index = bounded.open_index("routing").await.unwrap();
        for mode in [RoutingMode::Exact, HNSW] {
            let result = index.search(request(mode, 2, false)).await;
            if count == 73 {
                assert_eq!(result.unwrap_err().kind(), ErrorKind::LimitExceeded);
            } else {
                assert_eq!(result.unwrap().hits.len(), count as usize);
            }
        }
        bounded.shutdown().await.unwrap();
        builder.shutdown().await.unwrap();
    }
}

/// Removes a mandatory graph immediately after a search acquires its snapshot.
/// The old snapshot must succeed; a later cold search must fail closed.
struct SnapshotCut {
    memory: MemoryBackend,
    delete: std::sync::Arc<std::sync::Mutex<Option<Bytes>>>,
}

impl Backend for SnapshotCut {
    type ReadTxn<'a> = <MemoryBackend as Backend>::ReadTxn<'a>;
    type WriteTxn<'a> = <MemoryBackend as Backend>::WriteTxn<'a>;

    fn hard_limits(&self) -> HardLimits {
        self.memory.hard_limits()
    }
    fn admission_budget(&self) -> ktann::storage::backend::AdmissionBudget {
        self.memory.admission_budget()
    }
    fn capabilities(&self) -> ktann::storage::backend::Capabilities {
        self.memory.capabilities()
    }

    async fn begin_read(&self) -> ktann::api::Result<Self::ReadTxn<'_>> {
        let snapshot = self.memory.begin_read().await?;
        let delete = self.delete.lock().unwrap().take();
        if let Some(key) = delete {
            let mut write = self.memory.begin_write().await?;
            write.delete(key).await?;
            write.commit().await?;
        }
        Ok(snapshot)
    }

    async fn begin_write(&self) -> ktann::api::Result<Self::WriteTxn<'_>> {
        self.memory.begin_write().await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_search_graph_reads_share_snapshot_and_missing_graph_is_corruption() {
    for mode in [RoutingMode::Exact, HNSW] {
        let memory = MemoryBackend::new();
        let builder = build(memory.clone(), Metric::L2, false, 73).await;
        let delete = std::sync::Arc::new(std::sync::Mutex::new(None));
        let runtime = Runtime::new(
            SnapshotCut {
                memory: memory.clone(),
                delete: delete.clone(),
            },
            RuntimeConfig::default()
                .with_maintenance(0, 1)
                .unwrap()
                .with_partition_cache_bytes(0)
                .unwrap(),
        )
        .unwrap();
        let index = runtime.open_index("routing").await.unwrap();
        let tree = keys::TreeKey::encode(&[], &[]).unwrap();
        let key = Bytes::from(keys::graph_chunk_key(
            index.logical_index_id(),
            &tree,
            ktann::api::PartitionKey::new(1).unwrap(),
            0,
        ));
        *delete.lock().unwrap() = Some(key);
        let result = index.search(request(mode, 128, false)).await.unwrap();
        assert_eq!(result.hits.len(), 9);
        assert!(delete.lock().unwrap().is_none());
        assert_eq!(
            index
                .search(request(mode, 128, false))
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Corruption
        );
        runtime.shutdown().await.unwrap();
        builder.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_shared_beam_preserves_concentrated_parent_candidates() {
    use ktann::api::PartitionKey;
    use ktann::storage::{LogicalRange, backend::ScanLimits};
    use std::collections::BTreeMap;

    let memory = MemoryBackend::new();
    let runtime = build(memory.clone(), Metric::L2, false, 129).await;
    let index = runtime.open_index("routing").await.unwrap();
    let mut txn = ReadLogicalTxn::bootstrap(memory.begin_read().await.unwrap());
    let Some(PersistentValue::IndexManifest(manifest)) = txn
        .get(LogicalKey::Manifest(index.logical_index_id()))
        .await
        .unwrap()
    else {
        panic!("manifest")
    };
    let mut txn = ReadLogicalTxn::for_index(memory.begin_read().await.unwrap(), &manifest);
    let page = txn
        .scan(
            &LogicalRange::index(&manifest),
            None,
            ScanLimits {
                item_limit: 10000,
                byte_limit: 4 << 20,
            },
        )
        .await
        .unwrap();
    assert!(page.next_cursor().is_none());
    let mut children = BTreeMap::<PartitionKey, Vec<(PartitionKey, f64)>>::new();
    let mut levels = BTreeMap::new();
    let mut leaves = BTreeMap::<PartitionKey, Vec<u64>>::new();
    let mut root = None;
    for item in page.into_items() {
        match (item.key(), item.value()) {
            (LogicalKey::TreeManifest { .. }, PersistentValue::TreeManifest(tree)) => {
                root = Some(tree.root())
            }
            (LogicalKey::Header { partition, .. }, PersistentValue::PartitionHeader(header)) => {
                levels.insert(*partition, header.level());
            }
            (
                LogicalKey::ChildEntry {
                    partition: parent, ..
                },
                PersistentValue::ChildEntry(child),
            ) => {
                // Zero queries remain zero under rotation. This oracle evaluates stored
                // centroid norms directly, independently of the engine routing kernel.
                let score = child.centroid().iter().map(|&x| f64::from(x).powi(2)).sum();
                children
                    .entry(*parent)
                    .or_default()
                    .push((child.child(), score));
            }
            (LogicalKey::LeafEntry { partition, id, .. }, PersistentValue::LeafEntry(_)) => {
                leaves
                    .entry(*partition)
                    .or_default()
                    .push(u64::from_be_bytes(id.as_ref().try_into().unwrap()));
            }
            _ => {}
        }
    }
    let mut frontier = vec![root.unwrap()];
    let mut concentrated = false;
    while levels[&frontier[0]] > 1 {
        let next_level = levels[&frontier[0]] - 1;
        let width = (4_usize >> (next_level - 1)).max(1);
        let mut all = Vec::new();
        for parent in &frontier {
            all.extend(
                children[parent]
                    .iter()
                    .map(|&(child, distance)| (child, distance, *parent)),
            );
        }
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        all.truncate(width);
        let mut counts = BTreeMap::<PartitionKey, usize>::new();
        for &(_, _, parent) in &all {
            *counts.entry(parent).or_default() += 1;
        }
        if frontier.len() > 1 && counts.values().any(|&n| n > width.div_ceil(frontier.len())) {
            concentrated = true;
        }
        frontier = all.into_iter().map(|(child, _, _)| child).collect();
    }
    assert!(
        concentrated,
        "fixture must require more than a divided per-parent beam"
    );
    let mut expected: Vec<_> = frontier
        .iter()
        .flat_map(|leaf| &leaves[leaf])
        .copied()
        .collect();
    expected.sort_by(|a, b| {
        distance(Metric::L2, &[0.0; 3], &vector(*a))
            .partial_cmp(&distance(Metric::L2, &[0.0; 3], &vector(*b)))
            .unwrap()
            .then(a.cmp(b))
    });
    for mode in [
        RoutingMode::Exact,
        RoutingMode::PartitionLocalHnsw { ef_search: 64 },
    ] {
        let outcome = index
            .search(
                SearchRequest::new(vec![0.0; 3], 32).unwrap().with_options(
                    SearchOptions::default()
                        .with_routing_mode(mode)
                        .unwrap()
                        .with_leaf_beam_size(4)
                        .unwrap(),
                ),
            )
            .await
            .unwrap();
        let actual: Vec<_> = outcome
            .hits
            .iter()
            .map(|hit| u64::from_be_bytes(hit.id().as_ref().try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
    }
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn warm_public_routing_reuses_bodies_without_changing_logical_work() {
    let memory = MemoryBackend::new();
    let builder = build(memory.clone(), Metric::L2, false, 129).await;
    for mode in [RoutingMode::Exact, HNSW] {
        let runtime = Runtime::new(
            memory.clone(),
            RuntimeConfig::default()
                .with_maintenance(0, 1)
                .unwrap()
                .with_partition_cache_bytes(16 << 20)
                .unwrap(),
        )
        .unwrap();
        let index = runtime.open_index("routing").await.unwrap();
        memory.reset_operation_counts();
        let cold = index.search(request(mode, 4, false)).await.unwrap();
        let cold_reads = memory.operation_counts();
        memory.reset_operation_counts();
        let warm = index.search(request(mode, 4, false)).await.unwrap();
        let warm_reads = memory.operation_counts();
        assert_eq!(signature(&cold), signature(&warm));
        assert_eq!(cold.usage, warm.usage);
        assert!(cold_reads.scan + cold_reads.batch_scan > warm_reads.scan + warm_reads.batch_scan);
        assert_eq!(
            cold_reads.get, warm_reads.get,
            "cache warmth never skips Header/descriptor authority"
        );
        eprintln!(
            "{mode:?}: cold={cold_reads:?}, warm={warm_reads:?}, logical={:?}",
            cold.usage
        );
        runtime.shutdown().await.unwrap();
    }
    builder.shutdown().await.unwrap();
}

#[path = "routing/verify.rs"]
mod verify;
