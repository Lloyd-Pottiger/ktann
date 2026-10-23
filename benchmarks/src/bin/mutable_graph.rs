//! Repeated mutable graph quality experiment on held-out SIFT/Cohere queries.
//! Usage:
//! ```text
//! ktann-mutable-graph DATASET NEW_OUTPUT_DIRECTORY [RECORDS] [ROUNDS] [INTERNAL_MIN:MAX]
//! ```
#[path = "../../../tests/support/audit.rs"]
#[allow(dead_code)]
mod audit;
#[path = "../../../tests/support/oracle.rs"]
#[expect(dead_code, reason = "shared audit helpers require the oracle model")]
mod oracle;
use bytes::Bytes;
use ktann::api::{
    Index, IndexConfig, LocalGraphConfig, Metric, Record, RoutingMode, RuntimeConfig,
    SearchBudgets, SearchOptions, SearchRequest, VerifyOptions, VerifyReport,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;
use ktann_benchmarks::{
    MeasuredBackend, MetricCapture, ResourceSnapshot, limit_with_query_offset, load, load_large,
};
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use rocksdb::{OptimisticTransactionDB, Options};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

type Corpus = Vec<(Bytes, Arc<[f32]>)>;

async fn settle<B: Backend>(index: &Index<B>, raw: &B, metrics: &MetricCapture) {
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut quiet = 0;
    // A unit vector is also valid for cosine; the probe only rediscovers work.
    let mut probe = vec![0.0; index.config().dimension()];
    probe[0] = 1.0;
    let probe: Arc<[f32]> = probe.into();
    loop {
        if Instant::now() >= deadline {
            let partitions = audit::list_partitions(raw, index.logical_index_id())
                .await
                .unwrap();
            let snapshot = metrics.snapshot();
            eprintln!(
                "maintenance timeout: backlog={} partitions={} unfinished={} steps={:?} retries={:?}",
                metrics.fixup_backlog(),
                partitions.len(),
                partitions
                    .iter()
                    .filter(
                        |(_, _, h)| h.state() != ktann::storage::values::PartitionState::Ready
                            || h.entry_count() > index.config().max_partition_entries(h.level())
                    )
                    .count(),
                snapshot.counters_rendered("ktann.fixup.steps"),
                snapshot.counters_rendered("ktann.write.retries"),
            );
            panic!("maintenance did not settle");
        }
        // Running workers retain and yield their work. Rediscover only when
        // the queue is idle, so the observer does not continuously scan and
        // decode the entire corpus while measuring maintenance cost.
        if metrics.fixup_backlog() != 0 {
            quiet = 0;
            tokio::time::sleep(Duration::from_millis(25)).await;
            continue;
        }
        index
            .search(
                SearchRequest::new(probe.clone(), 1)
                    .unwrap()
                    .with_options(SearchOptions::default().with_leaf_beam_size(16384).unwrap()),
            )
            .await
            .unwrap();
        let partitions = audit::list_partitions(raw, index.logical_index_id())
            .await
            .unwrap();
        let ready = partitions.iter().all(|(_, _, h)| {
            h.state() == ktann::storage::values::PartitionState::Ready
                && h.entry_count() <= index.config().max_partition_entries(h.level())
        });
        quiet = if ready && metrics.fixup_backlog() == 0 {
            quiet + 1
        } else {
            0
        };
        if quiet >= 3 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        report.complete && report.issues.is_empty(),
        "{:?}",
        report.issues
    );
}

/// Exact neighbors for one unchanged corpus/query set, shared across routing modes.
fn ground_truth(metric: Metric, corpus: &Corpus, queries: &[Arc<[f32]>]) -> Vec<BTreeSet<Bytes>> {
    queries
        .iter()
        .map(|query| {
            let mut all: Vec<_> = corpus
                .iter()
                .map(|(id, vector)| (id, oracle::exact_distance(metric, query, vector)))
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(b.0)));
            all.into_iter()
                .take(100)
                .map(|(id, _)| id.clone())
                .collect()
        })
        .collect()
}

/// Audits the settled index once before comparing routing modes on its records.
async fn verify_records<B: Backend>(index: &Index<B>, count: usize) -> VerifyReport {
    let coverage = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(coverage.complete && coverage.issues.is_empty());
    assert_eq!(coverage.objects.vector_records as usize, count);
    coverage
}

async fn quality<B: Backend>(
    index: &Index<B>,
    truth: &[BTreeSet<Bytes>],
    coverage: &VerifyReport,
    queries: &[Arc<[f32]>],
    beam: u32,
    mode: RoutingMode,
) -> Value {
    let mut latencies = Vec::new();
    let mut recall = 0.0;
    let mut per_query = Vec::new();
    let options = SearchOptions::default()
        .with_leaf_beam_size(beam)
        .unwrap()
        .with_routing_mode(mode)
        .unwrap();
    for query in queries.iter().take(10) {
        index
            .search(
                SearchRequest::new(query.clone(), 100)
                    .unwrap()
                    .with_options(options),
            )
            .await
            .unwrap();
    }
    let resources = ResourceSnapshot::capture().unwrap();
    let started = Instant::now();
    for (query, expected) in queries.iter().zip(truth) {
        let start = Instant::now();
        let outcome = index
            .search(
                SearchRequest::new(query.clone(), 100)
                    .unwrap()
                    .with_options(options),
            )
            .await
            .unwrap();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        let query_recall = outcome
            .hits
            .iter()
            .filter(|hit| expected.contains(hit.id()))
            .count() as f64
            / expected.len() as f64;
        recall += query_recall;
        per_query.push(query_recall);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let after = ResourceSnapshot::capture().unwrap();
    let mut diagnostic_coverage = Vec::new();
    for (query, expected) in queries.iter().zip(truth) {
        let (_, diagnostic) = index
            .benchmark_search(
                SearchRequest::new(query.clone(), 100)
                    .unwrap()
                    .with_options(options),
                None,
                64,
                Some(expected.iter().cloned().collect()),
            )
            .await
            .unwrap();
        let diagnostic = diagnostic.unwrap();
        let n = expected.len() as f64;
        diagnostic_coverage.push(json!({
            "true_neighbor_leaf_coverage": diagnostic.neighbors_in_leaves as f64/n,
            "after_local_selection_coverage": diagnostic.neighbors_after_local_selection as f64/n,
            "reranked_coverage": diagnostic.neighbors_reranked as f64/n,
            "leaves_scanned": diagnostic.leaves.len(),
        }));
    }
    latencies.sort_by(f64::total_cmp);
    let mut sorted_recall = per_query.clone();
    sorted_recall.sort_by(f64::total_cmp);
    json!({"recall_at_100": recall / queries.len() as f64, "query_count": queries.len(), "per_query_recall": per_query, "untimed_coverage": diagnostic_coverage,
        "recall_min": sorted_recall[0], "recall_p05": sorted_recall[(sorted_recall.len()-1)*5/100],
        "recall_p50": sorted_recall[(sorted_recall.len()-1)*50/100],
        "verified_records": coverage.objects.vector_records, "verified_partitions": coverage.objects.partitions,
        "partitions_by_level": coverage.topology.partitions_by_level, "entries_by_level": coverage.topology.entries_by_level,
        "max_level": coverage.topology.max_level, "wall_seconds": elapsed, "cpu_seconds": after.cpu_seconds_since(resources),
        "peak_rss_bytes": after.peak_rss_bytes(), "p50_ms": latencies[(latencies.len()-1)*50/100],
        "p95_ms": latencies[(latencies.len()-1)*95/100]})
}

async fn paired_quality<B: Backend>(
    index: &Index<B>,
    corpus: &Corpus,
    queries: &[Arc<[f32]>],
    beam: u32,
) -> Value {
    let truth = ground_truth(index.config().metric(), corpus, queries);
    let coverage = verify_records(index, corpus.len()).await;
    let hnsw = quality(
        index,
        &truth,
        &coverage,
        queries,
        beam,
        RoutingMode::PartitionLocalHnsw { ef_search: 64 },
    )
    .await;
    let exact = quality(index, &truth, &coverage, queries, beam, RoutingMode::Exact).await;
    json!({"partition_local_hnsw": hnsw, "exact": exact})
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=5).contains(&args.len()) {
        return Err(
            "usage: ktann-mutable-graph DATASET NEW_OUTPUT [RECORDS] [ROUNDS] [INTERNAL_MIN:MAX]"
                .into(),
        );
    }
    let count: usize = args.get(2).map_or(Ok(16384), |s| s.parse())?;
    let rounds: usize = args.get(3).map_or(Ok(8), |s| s.parse())?;
    let (internal_min, internal_max): (u32, u32) = if let Some(bounds) = args.get(4) {
        let (minimum, maximum) = bounds
            .split_once(':')
            .ok_or("internal bounds must be MIN:MAX")?;
        (minimum.parse()?, maximum.parse()?)
    } else {
        (32, 128)
    };
    let metrics = MetricCapture::install()?;
    let output = Path::new(&args[1]);
    std::fs::create_dir(output)?;
    let loaded = if args[0] == "clustered" {
        load("clustered", count * 5, 200, 32, 7)?
    } else {
        load_large(&args[0])?
    };
    let calibration_queries = loaded.queries[..50].to_vec();
    let replacement_pool: Vec<_> = loaded
        .base
        .iter()
        .skip(count)
        .take(count * 4)
        .cloned()
        .collect();
    if replacement_pool.is_empty() {
        return Err("dataset has no reserved update vectors".into());
    }
    let data = limit_with_query_offset(loaded, count, 100, 100)?;
    let mut corpus: Corpus = data.ids.into_iter().zip(data.base).collect();
    let mut options = Options::default();
    options.create_if_missing(true);
    let database = Arc::new(OptimisticTransactionDB::open(
        &options,
        output.join("database"),
    )?);
    let raw = RocksDbBackend::new(database, BackendNamespace::new("mutable-graph-experiment")?);
    let (raw, counters) = MeasuredBackend::new(raw);
    let runtime_config = RuntimeConfig::default()
        .with_maintenance(2, 1024)?
        .with_default_search_budgets(SearchBudgets::new(65536, 16384, 125)?)?;
    let runtime = Runtime::new(raw.clone(), runtime_config)?;
    let metric = if args[0] == "cohere-1m" {
        Metric::Cosine
    } else {
        Metric::L2
    };
    let config = IndexConfig::new(corpus[0].1.len(), metric)?
        .with_leaf_partition_entries(32, 64)?
        .with_internal_partition_entries(internal_min, internal_max)?
        .with_local_graph(LocalGraphConfig::new(16, 64, [19; 32])?);
    let index = runtime.create_index("updates", config.clone()).await?;
    let mut observations = vec![json!({"phase": "configuration", "dataset": data.metadata,
        "metric": format!("{metric:?}"), "graph_seed_bytes": vec![19_u8; 32], "synthetic_data_seed": (args[0] == "clustered").then_some(7_u64),
        "search_bounds": {"scanned_tree_keys": 65536, "visited_partitions": 16384, "rerank_candidates": 125},
        "partition_bounds": {"leaf": [32,64], "internal": [internal_min,internal_max]},
        "maintenance_workers": 2, "fixup_capacity": 1024, "rounds": rounds,
        "query_windows": {"calibration": [0,50], "evaluation": [100,200]},
        "calibration_target": 0.92, "beam_candidates": [8,16,32,64,128,256],
        "update_pool_rows": [count, count + replacement_pool.len()],
        "update_pool_rule": "(round * floor(count/10) + offset) modulo pool length"})];
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&observations)?,
    )?;
    eprintln!("import: {count} records, metric={metric:?}");
    let initial_started = Instant::now();
    let initial_resources = ResourceSnapshot::capture()?;
    let initial_io = counters.snapshot();
    for (id, vector) in &corpus {
        index
            .insert(Record::new(id.clone(), vector.clone(), vec![])?)
            .await?;
    }
    settle(&index, &raw, &metrics).await;
    let initial_seconds = initial_started.elapsed().as_secs_f64();
    let initial_after = ResourceSnapshot::capture()?;
    let initial_io = counters.since(&initial_io);
    eprintln!("initial topology settled: {initial_seconds:.3}s");
    let beam = {
        let calibration_truth = ground_truth(metric, &corpus, &calibration_queries);
        let calibration_coverage = verify_records(&index, corpus.len()).await;
        let mut frozen_beam = None;
        for beam in [8, 16, 32, 64, 128, 256] {
            let measured = quality(
                &index,
                &calibration_truth,
                &calibration_coverage,
                &calibration_queries,
                beam,
                RoutingMode::PartitionLocalHnsw { ef_search: 64 },
            )
            .await;
            let recall = measured["recall_at_100"].as_f64().unwrap();
            eprintln!("calibration: beam={beam} recall={recall:.6}");
            observations.push(json!({"phase": "calibration", "beam": beam, "quality": measured}));
            std::fs::write(
                output.join("report.json"),
                serde_json::to_vec_pretty(&observations)?,
            )?;
            if recall >= 0.92 {
                frozen_beam = Some(beam);
                break;
            }
        }
        frozen_beam.ok_or("initial calibration did not reach recall 0.92")?
    };
    eprintln!("frozen beam={beam}; evaluation queries100..199");
    observations.push(json!({"phase": "initial", "frozen_beam": beam,
        "initial_import_and_settle_seconds": initial_seconds,
        "initial_import_cpu_seconds": initial_after.cpu_seconds_since(initial_resources),
        "initial_backend_io": initial_io,
        "quality": paired_quality(&index, &corpus, &data.queries, beam).await}));
    let stride = (count / 10).max(1);
    let originals: BTreeSet<_> = corpus.iter().map(|(id, _)| id.clone()).collect();
    let mut touched = BTreeSet::new();
    for round in 0..rounds {
        eprintln!("round {round}: update/delete/reinsert");
        let before = counters.snapshot();
        let steps_before = metrics.snapshot().counters_rendered("ktann.fixup.steps");
        let resources = ResourceSnapshot::capture()?;
        let start = Instant::now();
        // Reserve vectors outside the initial prefix so repeated updates preserve
        // the real dataset distribution without recursively cloning live vectors.
        let delete_start = round * stride % (corpus.len() - stride + 1);
        let update_start = (delete_start + corpus.len() / 2) % (corpus.len() - stride + 1);
        let replacement: Vec<_> = (0..stride)
            .map(|i| replacement_pool[(round * stride + i) % replacement_pool.len()].clone())
            .collect();
        for (offset, vector) in replacement.into_iter().enumerate() {
            let i = update_start + offset;
            touched.insert(corpus[i].0.clone());
            corpus[i].1 = vector;
            index
                .upsert(Record::new(
                    corpus[i].0.clone(),
                    corpus[i].1.clone(),
                    vec![],
                )?)
                .await?;
        }
        let removed: Vec<_> = corpus.drain(delete_start..delete_start + stride).collect();
        for (id, _) in &removed {
            touched.insert(id.clone());
            index.delete(id.clone()).await?;
        }
        settle(&index, &raw, &metrics).await;
        for (ordinal, (_, vector)) in removed.into_iter().enumerate() {
            let id = Bytes::from(format!("replacement-{round}-{ordinal}"));
            index
                .insert(Record::new(id.clone(), vector.clone(), vec![])?)
                .await?;
            touched.insert(id.clone());
            corpus.push((id, vector));
        }
        settle(&index, &raw, &metrics).await;
        let after = ResourceSnapshot::capture()?;
        let update_seconds = start.elapsed().as_secs_f64();
        let steps: std::collections::BTreeMap<_, _> = metrics
            .snapshot()
            .counters_rendered("ktann.fixup.steps")
            .into_iter()
            .map(|(key, value)| {
                let delta = value.saturating_sub(*steps_before.get(&key).unwrap_or(&0));
                (key, delta)
            })
            .collect();
        observations.push(json!({"phase": "round", "round": round, "mutation_count": stride*3, "unique_touched_ids": touched.len(),
            "original_ids_touched": touched.intersection(&originals).count(),
            "maintenance_steps_per_second": steps.values().sum::<u64>() as f64 / update_seconds,
            "maintenance_steps": steps, "update_and_maintenance_seconds": update_seconds, "mutations_per_second": (stride*3) as f64/update_seconds,
            "cpu_seconds": after.cpu_seconds_since(resources), "peak_rss_bytes": after.peak_rss_bytes(),
            "backend_io": counters.since(&before), "quality": paired_quality(&index, &corpus, &data.queries, beam).await}));
        std::fs::write(
            output.join("report.json"),
            serde_json::to_vec_pretty(&observations)?,
        )?;
    }
    eprintln!("fresh rebuild from final records");
    let rebuilt = runtime.create_index("rebuilt", config).await?;
    for (id, vector) in &corpus {
        rebuilt
            .insert(Record::new(id.clone(), vector.clone(), vec![])?)
            .await?;
    }
    settle(&rebuilt, &raw, &metrics).await;
    observations.push(json!({"phase": "rebuilt", "frozen_beam": beam,
        "quality": paired_quality(&rebuilt, &corpus, &data.queries, beam).await}));
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&observations)?,
    )?;
    runtime.shutdown().await?;
    Ok(())
}
