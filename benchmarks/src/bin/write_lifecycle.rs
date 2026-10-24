//! Fixed-budget write lifecycle and quality comparisons on cloned Cohere fixtures.
#[path = "../bridge/settle.rs"]
mod maintenance_settle;
#[path = "../../../tests/support/oracle.rs"]
#[expect(dead_code, reason = "shared exact-distance oracle")]
mod oracle;
#[path = "../bridge/topology.rs"]
mod topology;
use ktann::api::{
    BulkBuildOptions, BulkWorkerOptions, Index, IndexConfig, LocalGraphConfig, Metric, Mutation,
    Record, RoutingMode, RuntimeConfig, SearchBudgets, SearchOptions, SearchRequest, VerifyOptions,
};
use ktann::{
    bulk::InputSnapshot, construction::ConstructionOptions, runtime::Runtime,
    storage::backend::Backend,
};
#[cfg(test)]
use ktann_benchmarks as backend;
use ktann_benchmarks::{
    BenchmarkDataset, MeasuredBackend, MetricCapture, ResourceSnapshot, load_large,
};
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use rocksdb::{OptimisticTransactionDB, Options};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

/// All workloads are explicit so reports can reproduce the same update stream.
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    records: usize,
    rounds: usize,
    updates_per_round: usize,
    warm: bool,
    workers: usize,
    queries: usize,
}

fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<_> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, state as usize % (i + 1));
    }
    order
}

/// Update both directions of the vector-to-record permutation together.
fn swap_vectors(at_id: &mut [usize], owner: &mut [usize], a: usize, b: usize) {
    at_id.swap(a, b);
    owner[at_id[a]] = a;
    owner[at_id[b]] = b;
}

/// Vector swaps preserve the corpus; map original vector neighbors to current IDs.
async fn quality<B: Backend>(
    index: &Index<B>,
    data: &BenchmarkDataset,
    truth: &[Vec<usize>],
    owner: &[usize],
) -> ktann::api::Result<Value> {
    let options = SearchOptions::default()
        .with_leaf_beam_size(128)?
        .with_routing_mode(RoutingMode::PartitionLocalHnsw { ef_search: 64 })?;
    let mut per_query = Vec::new();
    for (query, expected) in data.queries[100..100 + truth.len()].iter().zip(truth) {
        let expected: BTreeSet<_> = expected.iter().map(|&v| &data.ids[owner[v]]).collect();
        let result = index
            .search(SearchRequest::new(query.clone(), 100)?.with_options(options))
            .await?;
        per_query.push(
            result
                .hits
                .iter()
                .filter(|h| expected.contains(h.id()))
                .count() as f64
                / 100.0,
        );
    }
    Ok(
        json!({"recall_at_100": per_query.iter().sum::<f64>() / per_query.len() as f64, "per_query":per_query}),
    )
}

async fn verify<B: Backend>(index: &Index<B>, count: usize) -> ktann::api::Result<String> {
    let report = index
        .verify(
            VerifyOptions::default()
                .with_object_limit(10_000_000)?
                .with_memory_limit_bytes(1 << 30)?,
        )
        .await?;
    assert!(report.complete && report.issues.is_empty(), "{report:?}");
    assert_eq!(report.objects.vector_records as usize, count);
    Ok(format!("{report:?}"))
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "fixture" | "run") {
        return Err("usage: ktann-write-lifecycle fixture|run OUTPUT SCENARIO.json".into());
    }
    let scenario: Scenario = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let n = scenario.records;
    if !(100..=1_000_000).contains(&n)
        || scenario.rounds == 0
        || scenario.updates_per_round == 0
        || scenario.updates_per_round > n
        || !scenario.updates_per_round.is_multiple_of(32)
        || scenario.queries == 0
        || scenario.queries > 900
    {
        return Err("invalid scenario bounds: records100..1M, rounds>0, updates<=records in batches32, queries1..900".into());
    }
    let output = Path::new(&args[2]);
    let building = args[1] == "fixture";
    if building {
        std::fs::create_dir(output)?;
        std::fs::create_dir(output.join("workspace"))?;
    }
    let metrics = MetricCapture::install()?;
    let data = load_large("cohere-1m")?;
    let mut options = Options::default();
    options.create_if_missing(building);
    let db = Arc::new(OptimisticTransactionDB::open(
        &options,
        output.join("database"),
    )?);
    let (raw, counters) = MeasuredBackend::new(RocksDbBackend::new(
        db,
        BackendNamespace::new("ktann-benchmark")?,
    ));
    let config = RuntimeConfig::default()
        .with_maintenance(if building { 0 } else { scenario.workers }, 1024)?
        .with_default_search_budgets(SearchBudgets::new(65536, 16384, 125)?)?;
    let runtime = Runtime::new(raw.clone(), config)?;
    if building {
        let mut seed = [0; 32];
        seed[0] = 7;
        let config = IndexConfig::new(768, Metric::Cosine)?
            .with_leaf_partition_entries(16, 128)?
            .with_internal_partition_entries(64, 8192)?
            .with_local_graph(LocalGraphConfig::new(16, 200, seed)?);
        let source = InputSnapshot::create(
            &output.join("source"),
            config,
            64 << 30,
            data.ids
                .iter()
                .zip(&data.base)
                .take(n)
                .map(|(id, v)| Record::new(id.clone(), v.clone(), vec![])),
        )?;
        let job = runtime
            .start_bulk_build(
                "benchmark",
                &source,
                BulkBuildOptions {
                    construction: ConstructionOptions {
                        sample_items: 1024,
                        memory_bytes: 1 << 30,
                        scratch_bytes: 64 << 30,
                    },
                    refinement_rounds: Some(2),
                },
            )
            .await?;
        // Fixture refinement retains vectors and planning arrays; the default
        // worker envelope is too small for the larger crossover fixtures.
        let mut worker = BulkWorkerOptions::new(output.join("workspace"));
        worker.refinement_memory_bytes = 8 << 30;
        let (index, _) = job
            .complete_with_report(worker, None, Default::default())
            .await?;
        std::fs::write(
            output.join("fixture.json"),
            serde_json::to_vec_pretty(
                &json!({"config":format!("{:?}", index.config()),"verify":verify(&index,n).await?}),
            )?,
        )?;
        runtime.shutdown().await?;
        return Ok(());
    }
    let index = runtime.open_index("benchmark").await?;
    let initial = verify(&index, n).await?;
    // Full Verify reads storage but does not populate the local graph cache.
    let id_index: HashMap<_, _> = data.ids.iter().enumerate().map(|(i, id)| (id, i)).collect();
    let truth: Vec<Vec<usize>> = if n == data.base.len() {
        data.ground_truth
            .as_ref()
            .ok_or("missing full-corpus truth")?[100..100 + scenario.queries]
            .iter()
            .map(|row| row.iter().take(100).map(|id| id_index[id]).collect())
            .collect()
    } else {
        data.queries[100..100 + scenario.queries]
            .iter()
            .map(|q| {
                let mut scores: Vec<_> = data.base[..n]
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (oracle::exact_distance(Metric::Cosine, q, v), i))
                    .collect();
                scores.sort_unstable_by(|a, b| {
                    a.0.total_cmp(&b.0).then(data.ids[a.1].cmp(&data.ids[b.1]))
                });
                scores[..100].iter().map(|s| s.1).collect()
            })
            .collect()
    };
    assert!(truth.iter().all(|row| row.len() == 100));
    let mut at_id: Vec<_> = (0..n).collect();
    let mut owner = at_id.clone();
    let baseline = if scenario.warm {
        Some(quality(&index, &data, &truth, &owner).await?)
    } else {
        None
    };
    let mut rounds = Vec::new();
    for round in 0..scenario.rounds {
        let order = permutation(n, 7 + round as u64);
        let mut mutations = Vec::with_capacity(scenario.updates_per_round);
        for pair in 0..scenario.updates_per_round / 2 {
            let offset = pair * 2;
            let a = order[offset];
            let b = order[(offset + 1) % n];
            swap_vectors(&mut at_id, &mut owner, a, b);
            for id in [a, b] {
                mutations.push(Mutation::Upsert(Record::new(
                    data.ids[id].clone(),
                    data.base[at_id[id]].clone(),
                    vec![],
                )?));
            }
        }
        let _ = metrics.snapshot();
        let before = counters.snapshot();
        let resources = ResourceSnapshot::capture()?;
        let start = Instant::now();
        for batch in mutations.chunks(32) {
            index.batch_mutate(batch.to_vec()).await?;
        }
        let foreground_seconds = start.elapsed().as_secs_f64();
        let foreground_resources = ResourceSnapshot::capture()?;
        let foreground_io = counters.since(&before);
        let foreground = metrics.snapshot();
        let settle = if scenario.workers > 0 {
            Some(maintenance_settle::settle(&index, &raw, &metrics, n as u64).await)
        } else {
            None
        };
        let total_seconds = start.elapsed().as_secs_f64();
        let after = ResourceSnapshot::capture()?;
        let io = counters.since(&before);
        let drain = metrics.snapshot();
        // Verification and held-out queries are outside every write measurement.
        let verification = verify(&index, n).await?;
        let quality = quality(&index, &data, &truth, &owner).await?;
        // Read existing write-path probes after timing; no new hot-path instrumentation.
        let stages: std::collections::BTreeMap<_, _> = [
            "graph_read",
            "graph_restore",
            "ownership",
            "workspace",
            "search",
            "merge",
        ]
        .into_iter()
        .map(|stage| {
            (
                stage,
                foreground
                    .histogram(&format!("ktann.experiment.{stage}"), &[])
                    .iter()
                    .sum::<f64>(),
            )
        })
        .collect();
        rounds.push(json!({"round":round,"foreground_seconds":foreground_seconds,"total_seconds":total_seconds,"foreground_cpu_seconds":foreground_resources.cpu_seconds_since(resources),"total_cpu_seconds":after.cpu_seconds_since(resources),"foreground_io":foreground_io,"total_io":io,"routing":foreground.routing_summary(),"write_stages_seconds":stages,"write_exact_bodies":foreground.counter("ktann.experiment.write_exact_bodies", &[]),"write_hnsw_bodies":foreground.counter("ktann.experiment.write_hnsw_bodies", &[]),"write_distances":foreground.counter("ktann.experiment.write_distances",&[]),"foreground_fixups":foreground.counters_rendered("ktann.fixup.steps"),"drain_fixups":drain.counters_rendered("ktann.fixup.steps"),"parent_proof_stale":foreground.counter("ktann.experiment.parent_proof_stale",&[])+drain.counter("ktann.experiment.parent_proof_stale",&[]),"settle":settle,"verify":verification,"quality":quality}));
        std::fs::write(
            output.join("report.json"),
            serde_json::to_vec_pretty(
                &json!({"scenario":scenario,"write_hnsw_min_nodes":std::env::var("KTANN_EXPERIMENT_WRITE_HNSW_MIN_NODES").ok(),"experiment_switches":{"write_hnsw":std::env::var_os("KTANN_EXPERIMENT_WRITE_HNSW").is_some(),"publish_ready":std::env::var_os("KTANN_EXPERIMENT_PUBLISH_READY").is_some(),"reuse_parent_proof":std::env::var_os("KTANN_EXPERIMENT_REUSE_PARENT_PROOF").is_some()},"query_budget":{"beam":128,"ef":64,"rerank":125,"k":100,"query_offset":100},"initial":initial,"baseline_quality":baseline,"rounds":rounds}),
            )?,
        )?;
        eprintln!(
            "round={round} foreground={foreground_seconds:.3} total={total_seconds:.3} recall={}",
            rounds.last().unwrap()["quality"]["recall_at_100"]
        );
    }
    runtime.shutdown().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remapped_truth_matches_changed_corpus_across_overlapping_swaps() {
        let vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        let query = [0.8, 0.2];
        let mut at_id: Vec<_> = (0..4).collect();
        let mut owner = at_id.clone();
        for (a, b) in [(0, 1), (1, 2), (0, 3), (2, 3)] {
            swap_vectors(&mut at_id, &mut owner, a, b);
            let original = (0..4)
                .min_by(|&a, &b| {
                    oracle::exact_distance(Metric::Cosine, &query, &vectors[a])
                        .total_cmp(&oracle::exact_distance(Metric::Cosine, &query, &vectors[b]))
                })
                .unwrap();
            let current = (0..4)
                .min_by(|&a, &b| {
                    oracle::exact_distance(Metric::Cosine, &query, &vectors[at_id[a]]).total_cmp(
                        &oracle::exact_distance(Metric::Cosine, &query, &vectors[at_id[b]]),
                    )
                })
                .unwrap();
            assert_eq!(owner[original], current);
            for (id, &vector) in at_id.iter().enumerate() {
                assert_eq!(owner[vector], id);
            }
        }
    }
}
