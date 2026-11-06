//! Import-to-search lifecycle measurement with continuous phase accounting.

use std::collections::BTreeMap;
use std::time::Instant;

use ktann::api::{Index, Mutation, RuntimeConfig, SearchRequest};
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;

use crate::backend::{BackendCounters, MeasuredBackend, subtract};
use crate::dataset::BenchmarkDataset;
use crate::metrics::MetricCapture;
use crate::oracle;
use crate::report::{
    BackendIo, ConvergencePhase, Distribution, ImportPhase, LifecycleMeasurements, OperationClass,
    OperationSummary, SearchPhase, SearchTruncation, Topology,
};
use crate::resource::ResourceSnapshot;
use crate::runner::construction::{mutation_batches, settle_and_drain_topology, settle_timeout};
use crate::runner::workload::{
    ExactTruth, budget_summaries, exact_truth, recall_summary, search_request,
};
use crate::runner::{error_at, phase_resources};
use crate::scenario::{CONVERGENCE_MAINTENANCE_WORKERS, ScenarioSpec, index_config};

/// Measures one fresh Index from atomic batch load submission through warm search.
pub(super) async fn run_lifecycle_case<B: Backend>(
    backend: MeasuredBackend<B>,
    import_runtime_config: RuntimeConfig,
    convergence_runtime_config: RuntimeConfig,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &BenchmarkDataset,
) -> Result<(Topology, LifecycleMeasurements), String> {
    let runtime = Runtime::new(backend.clone(), import_runtime_config)
        .map_err(|error| error_at("create import runtime", error))?;
    let index = runtime
        .create_index("benchmark", index_config(spec)?)
        .await
        .map_err(|error| error_at("create lifecycle index", error))?;
    let batches = mutation_batches(dataset, spec.import_batch_size, "construct import record")?;
    let truths = exact_truth(dataset, spec.metric, spec.k);
    let requests = lifecycle_requests(dataset, spec)?;
    let _ = metric_capture.snapshot();
    let (import, case_baseline) =
        run_import_phase(&index, batches, backend_counters, metric_capture).await?;
    let import_finished = Instant::now();
    let immediate_search =
        run_search_phase(&index, &requests, &truths, backend_counters, metric_capture)
            .await?
            .phase;

    let convergence_started = Instant::now();
    let convergence_deadline = convergence_started + settle_timeout(spec);
    let convergence_resources_before = ResourceSnapshot::capture()?;
    let convergence_backend_before = backend_counters.snapshot();
    let (runtime, index) = if spec.maintenance_workers == CONVERGENCE_MAINTENANCE_WORKERS {
        (runtime, index)
    } else {
        runtime
            .shutdown()
            .await
            .map_err(|error| error_at("shut down import-only runtime", error))?;
        drop(index);
        drop(runtime);
        let runtime = Runtime::new(backend.clone(), convergence_runtime_config.clone())
            .map_err(|error| error_at("create convergence runtime", error))?;
        let index = runtime
            .open_index("benchmark")
            .await
            .map_err(|error| error_at("reopen lifecycle index for convergence", error))?;
        (runtime, index)
    };
    let (topology, maintenance_drain_seconds) =
        settle_and_drain_topology(&index, dataset, spec, metric_capture, convergence_deadline)
            .await?;
    let convergence_completed = Instant::now();
    let convergence_wall_seconds = convergence_completed
        .duration_since(convergence_started)
        .as_secs_f64();
    let convergence_resources_after = ResourceSnapshot::capture()?;
    let convergence_backend_io = backend_counters.since(&convergence_backend_before);
    let convergence_metrics = metric_capture.snapshot();
    let convergence = ConvergencePhase {
        resources: phase_resources(
            convergence_wall_seconds,
            convergence_resources_before,
            convergence_resources_after,
            convergence_backend_io,
            &convergence_metrics,
            metric_capture.fixup_backlog(),
        ),
        from_import_finish_seconds: convergence_completed
            .duration_since(import_finished)
            .as_secs_f64(),
        maintenance_drain_seconds,
    };

    // Reopening the Runtime starts the cold search with an empty Partition Cache.
    let reset_started = Instant::now();
    let reset_resources_before = ResourceSnapshot::capture()?;
    let reset_backend_before = backend_counters.snapshot();
    runtime
        .shutdown()
        .await
        .map_err(|error| error_at("shut down convergence runtime", error))?;
    drop(index);
    drop(runtime);
    let search_runtime = Runtime::new(backend, convergence_runtime_config)
        .map_err(|error| error_at("create search runtime", error))?;
    let stable_index = search_runtime
        .open_index("benchmark")
        .await
        .map_err(|error| error_at("reopen lifecycle index", error))?;
    let reset_wall_seconds = reset_started.elapsed().as_secs_f64();
    let reset_resources_after = ResourceSnapshot::capture()?;
    let reset_backend_io = backend_counters.since(&reset_backend_before);
    let reset_metrics = metric_capture.snapshot();
    let cache_reset = phase_resources(
        reset_wall_seconds,
        reset_resources_before,
        reset_resources_after,
        reset_backend_io,
        &reset_metrics,
        metric_capture.fixup_backlog(),
    );

    let stable_cold_search = run_search_phase(
        &stable_index,
        &requests,
        &truths,
        backend_counters,
        metric_capture,
    )
    .await?
    .phase;
    let stable_warm = run_search_phase(
        &stable_index,
        &requests,
        &truths,
        backend_counters,
        metric_capture,
    )
    .await?;
    let case_wall_seconds = stable_warm
        .completed_at
        .duration_since(case_baseline.started)
        .as_secs_f64();
    let case_resources_after = stable_warm.resources_after;
    let case_backend_io = subtract(&stable_warm.backend_after, &case_baseline.backend_io);
    let stable_warm_search = stable_warm.phase;
    search_runtime
        .shutdown()
        .await
        .map_err(|error| error_at("shut down search runtime", error))?;

    let case_cpu_seconds = case_resources_after.cpu_seconds_since(case_baseline.resources);
    let lifecycle = LifecycleMeasurements {
        case_wall_seconds,
        case_cpu_seconds: Some(case_cpu_seconds),
        case_peak_rss_bytes: Some(case_resources_after.peak_rss_bytes()),
        case_backend_io,
        import,
        immediate_search,
        convergence,
        cache_reset,
        stable_cold_search,
        stable_warm_search,
    };
    Ok((topology, lifecycle))
}

/// Exact continuous-case baselines captured immediately before the first submit.
struct CaseBaseline {
    /// Monotonic first-submit timer boundary.
    started: Instant,
    /// Cumulative process resources at the same boundary.
    resources: ResourceSnapshot,
    /// Cumulative Backend IO at the same boundary.
    backend_io: BackendIo,
}

/// Constructs the fixed query set used at every lifecycle search boundary.
fn lifecycle_requests(
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
) -> Result<Vec<SearchRequest>, String> {
    (0..spec.measured_operations)
        .map(|ordinal| search_request(dataset, spec, ordinal % dataset.queries.len()))
        .collect()
}

/// Loads atomic batches sequentially and summarizes committed outcomes.
async fn run_import_phase<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    batches: Vec<Vec<Mutation>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
) -> Result<(ImportPhase, CaseBaseline), String> {
    if batches.is_empty() {
        return Err("lifecycle import has no batches".to_owned());
    }
    let resources_before = ResourceSnapshot::capture()?;
    let backend_before = backend_counters.snapshot();
    let started = Instant::now();
    let submitted_batches =
        u64::try_from(batches.len()).map_err(|_| "import batch count overflow".to_owned())?;
    let mut batch_latency_ms = Vec::with_capacity(batches.len());
    let mut failures = BTreeMap::new();
    let mut accepted_batches = 0_u64;
    let mut accepted_records = 0_u64;
    let mut submitted_records = 0_u64;
    for batch in batches {
        let records = batch.len();
        submitted_records += u64::try_from(records).map_err(|_| "import record count overflow")?;
        let batch_started = Instant::now();
        let result = index.batch_mutate(batch).await;
        batch_latency_ms.push(batch_started.elapsed().as_secs_f64() * 1_000.0);
        match result {
            Ok(outcomes) => {
                if outcomes.len() != records {
                    return Err("atomic batch outcome count differs from batch size".to_owned());
                }
                accepted_batches += 1;
                accepted_records +=
                    u64::try_from(records).map_err(|_| "import record count overflow")?;
            }
            Err(error) => {
                *failures.entry(format!("{:?}", error.kind())).or_default() += 1;
            }
        }
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    let resources_after = ResourceSnapshot::capture()?;
    let backend_io = backend_counters.since(&backend_before);
    let metrics = metric_capture.snapshot();
    if accepted_records != submitted_records {
        return Err(format!(
            "atomic batch load committed {accepted_records} of {submitted_records} records; batch failures: {failures:?}"
        ));
    }
    let phase = ImportPhase {
        resources: phase_resources(
            wall_seconds,
            resources_before,
            resources_after,
            backend_io,
            &metrics,
            metric_capture.fixup_backlog(),
        ),
        submitted_batches,
        accepted_batches,
        accepted_records,
        records_per_second: if wall_seconds > 0.0 {
            accepted_records as f64 / wall_seconds
        } else {
            0.0
        },
        batch_latency_ms: Distribution::from_samples(batch_latency_ms),
        batch_failures: failures,
        admission: metrics.admission_summary(),
    };
    Ok((
        phase,
        CaseBaseline {
            started,
            resources: resources_before,
            backend_io: backend_before,
        },
    ))
}

/// Executes the same fixed query set sequentially at one lifecycle boundary.
async fn run_search_phase<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    requests: &[SearchRequest],
    truths: &[ExactTruth],
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
) -> Result<CompletedSearchPhase, String> {
    let resources_before = ResourceSnapshot::capture()?;
    let backend_before = backend_counters.snapshot();
    let started = Instant::now();
    let mut latency_ms = Vec::with_capacity(requests.len());
    let mut first_query_latency_ms = None;
    let mut errors = BTreeMap::new();
    let mut outcomes = Vec::with_capacity(requests.len());
    for (ordinal, request) in requests.iter().enumerate() {
        let query_started = Instant::now();
        let result = index.search(request.clone()).await;
        let query_latency_ms = query_started.elapsed().as_secs_f64() * 1_000.0;
        match result {
            Ok(outcome) => {
                if ordinal == 0 {
                    first_query_latency_ms = Some(query_latency_ms);
                }
                latency_ms.push(query_latency_ms);
                outcomes.push((outcome, &truths[ordinal % truths.len()]))
            }
            Err(error) => *errors.entry(format!("{:?}", error.kind())).or_default() += 1,
        }
    }
    let completed_at = Instant::now();
    let wall_seconds = completed_at.duration_since(started).as_secs_f64();
    let resources_after = ResourceSnapshot::capture()?;
    let backend_after = backend_counters.snapshot();
    let backend_io = subtract(&backend_after, &backend_before);
    let metrics = metric_capture.snapshot();
    let attempted = u64::try_from(requests.len()).map_err(|_| "query count overflow")?;
    let accepted = u64::try_from(outcomes.len()).map_err(|_| "query count overflow")?;
    let recalls: Vec<_> = outcomes
        .iter()
        .map(|(outcome, truth)| oracle::recall_ids(outcome.hits.iter().map(|hit| hit.id()), truth))
        .collect();
    let truncation =
        outcomes
            .iter()
            .fold(SearchTruncation::default(), |mut summary, (outcome, _)| {
                summary.scanned_tree_keys += u64::from(outcome.exhausted.scanned_tree_keys);
                summary.visited_partitions += u64::from(outcome.exhausted.visited_partitions);
                summary.exact_rerank_candidates +=
                    u64::from(outcome.exhausted.exact_rerank_candidates);
                summary
            });
    let phase = SearchPhase {
        resources: phase_resources(
            wall_seconds,
            resources_before,
            resources_after,
            backend_io,
            &metrics,
            metric_capture.fixup_backlog(),
        ),
        first_query_latency_ms,
        search: OperationSummary {
            attempted,
            accepted,
            rejected: metrics.foreground_admission_rejections(OperationClass::Search),
            errors,
            latency_ms: Distribution::from_samples(latency_ms),
        },
        throughput_per_second: if wall_seconds > 0.0 {
            accepted as f64 / wall_seconds
        } else {
            0.0
        },
        recall_at_k: recall_summary(&recalls),
        truncation,
        search_budgets: budget_summaries(&metrics),
        visited_leaf_entries: Distribution::from_samples(
            metrics.histogram("ktann.search.leaf_entries", &[]),
        ),
        search_stages_ms: metrics.search_stage_summaries(),
        cache: metrics.cache_summary(),
        backend_admission: metrics.admission_summary(),
    };
    Ok(CompletedSearchPhase {
        phase,
        completed_at,
        resources_after,
        backend_after,
    })
}

/// Search report plus the exact final-operation continuous-case boundary.
struct CompletedSearchPhase {
    /// Serialized phase measurements.
    phase: SearchPhase,
    /// Instant captured immediately after the final public search returned.
    completed_at: Instant,
    /// Cumulative process resources at the same boundary.
    resources_after: ResourceSnapshot,
    /// Cumulative Backend IO at the same boundary.
    backend_after: BackendIo,
}
