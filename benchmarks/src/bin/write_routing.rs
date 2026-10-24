//! Isolated full-scale write-routing experiment; never a production entrypoint.
#[path = "../bridge/settle.rs"]
mod maintenance_settle;
#[path = "../bridge/topology.rs"]
mod topology;
use ktann::api::{
    Mutation, Record, RuntimeConfig, SearchBudgets, SearchOptions, SearchRequest, VerifyOptions,
};
use ktann::runtime::Runtime;
#[cfg(test)]
use ktann_benchmarks as backend;
use ktann_benchmarks::{MeasuredBackend, MetricCapture, ResourceSnapshot, load_large};
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use maintenance_settle::settle;
use rocksdb::{OptimisticTransactionDB, Options};
use serde_json::json;
use std::{path::Path, sync::Arc, time::Instant};
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: ktann-write-routing OUTPUT WORKERS UPDATES (1..=500000)".into());
    }
    let output = Path::new(&args[1]);
    let workers: usize = args[2].parse()?;
    let updates: usize = args[3].parse()?;
    if !(1..=500_000).contains(&updates) {
        return Err("UPDATES must be between 1 and 500000".into());
    }
    let metrics = MetricCapture::install()?;
    let data = load_large("cohere-1m")?;
    assert_eq!(data.base.len(), 1_000_000);
    let mut options = Options::default();
    options.create_if_missing(false);
    let database = Arc::new(OptimisticTransactionDB::open(
        &options,
        output.join("database"),
    )?);
    let raw = RocksDbBackend::new(database, BackendNamespace::new("ktann-benchmark")?);
    let (raw, counters) = MeasuredBackend::new(raw);
    let config = RuntimeConfig::default()
        .with_maintenance(workers, 1024)?
        .with_default_search_budgets(SearchBudgets::new(65536, 16384, 125)?)?;
    let runtime = Runtime::new(raw.clone(), config)?;
    let index = runtime.open_index("benchmark").await?;
    let initial = index
        .verify(
            VerifyOptions::default()
                .with_object_limit(10_000_000)?
                .with_memory_limit_bytes(1 << 30)?,
        )
        .await?;
    assert!(initial.complete && initial.issues.is_empty(), "{initial:?}");
    assert_eq!(initial.objects.vector_records, 1_000_000);
    let search_options = SearchOptions::default().with_leaf_beam_size(128)?;
    // Fixed queries warm all internal graph bodies; their scores are not an experiment target.
    for query in &data.queries[200..300] {
        index
            .search(SearchRequest::new(query.clone(), 100)?.with_options(search_options))
            .await?;
    }
    // A deterministic full-corpus permutation gives distinct, spatially dispersed updates.
    let mut order: Vec<_> = (0..data.base.len()).collect();
    let mut state = 7_u64;
    for i in (1..order.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, state as usize % (i + 1));
    }
    let mutations = (0..updates)
        .map(|i| {
            Record::new(
                data.ids[order[i]].clone(),
                data.base[order[500_000 + i]].clone(),
                vec![],
            )
            .map(Mutation::Upsert)
        })
        .collect::<ktann::api::Result<Vec<_>>>()?;
    let _ = metrics.snapshot();
    let before = counters.snapshot();
    let resources = ResourceSnapshot::capture()?;
    let start = Instant::now();
    eprintln!("WRITE_BEGIN updates={updates} workers={workers}");
    for chunk in mutations.chunks(32) {
        index.batch_mutate(chunk.to_vec()).await?;
    }
    let foreground_seconds = start.elapsed().as_secs_f64();
    let foreground_resources = ResourceSnapshot::capture()?;
    let foreground_io = counters.since(&before);
    let foreground = metrics.snapshot();
    eprintln!("FOREGROUND_END seconds={foreground_seconds}");
    let settle_observer = if workers > 0 {
        Some(settle(&index, &raw, &metrics, 1_000_000).await)
    } else {
        None
    };
    let total_seconds = start.elapsed().as_secs_f64();
    let after = ResourceSnapshot::capture()?;
    let total_io = counters.since(&before);
    let maintenance = metrics.snapshot();
    eprintln!("WRITE_END foreground={foreground_seconds} total={total_seconds}");
    let final_verify = index
        .verify(
            VerifyOptions::default()
                .with_object_limit(10_000_000)?
                .with_memory_limit_bytes(1 << 30)?,
        )
        .await?;
    assert!(
        final_verify.complete && final_verify.issues.is_empty(),
        "{:?}",
        final_verify.issues
    );
    assert_eq!(final_verify.objects.vector_records, 1_000_000);
    let stages = [
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
    .collect::<std::collections::BTreeMap<_, _>>();
    let report = json!({"updates":updates,"batch":32,"workers":workers,"settle_observer":settle_observer,"initial":format!("{initial:?}"),"final":format!("{final_verify:?}"),
        "foreground_seconds":foreground_seconds,"total_seconds":total_seconds,
        "foreground_cpu_seconds":foreground_resources.cpu_seconds_since(resources),"total_cpu_seconds":after.cpu_seconds_since(resources),
        "foreground_io":foreground_io,"total_io":total_io,"write":foreground.write_attribution(),"routing":foreground.routing_summary(),
        "maintenance_write":maintenance.write_attribution(),"maintenance_routing":maintenance.routing_summary(),
        "fixups":foreground.counters_rendered("ktann.fixup.steps"),"fixup_executions":foreground.counters_rendered("ktann.fixup.execution"),"settle_executions":maintenance.counters_rendered("ktann.fixup.execution"),"settle_fixups":maintenance.counters_rendered("ktann.fixup.steps"),
        "foreground_maintenance_phases":foreground.histograms_by_label("ktann.experiment.maintenance_seconds", "stage"),"drain_maintenance_phases":maintenance.histograms_by_label("ktann.experiment.maintenance_seconds", "stage"),
        "parent_proof_reused":foreground.counter("ktann.experiment.parent_proof_reused", &[])+maintenance.counter("ktann.experiment.parent_proof_reused", &[]),"parent_proof_stale":foreground.counter("ktann.experiment.parent_proof_stale", &[])+maintenance.counter("ktann.experiment.parent_proof_stale", &[]),
        "ready_graph_offers":foreground.counter("ktann.experiment.ready_graph_offers", &[]),"settle_ready_graph_offers":maintenance.counter("ktann.experiment.ready_graph_offers", &[]),"stages_seconds":stages,"distances":foreground.counter("ktann.experiment.write_distances", &[])});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    runtime.shutdown().await?;
    Ok(())
}
