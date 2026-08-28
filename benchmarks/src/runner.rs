//! Scenario definitions and the measured public-API execution path.

use std::collections::BTreeMap;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, ImportOptions, ImportSession, Index, IndexConfig,
    Metric, Mutation, OperationOptions, Record, RuntimeConfig, SearchBudgets, SearchOptions,
    SearchOutcome, SearchRequest, Value, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;

use crate::backend::{BackendCounters, MeasuredBackend};
use crate::dataset::{self, BenchmarkDataset};
use crate::metrics::{CapturedMetrics, MetricCapture};
use crate::report::{
    AdmissionTarget, BenchmarkReport, BudgetConfiguration, BudgetSummary, Configuration,
    ConvergencePhase, Distribution, Environment, ImportPhase, LifecycleMeasurements,
    MaintenanceSummary, OperationClass, OperationSummary, PhaseResources, RecallSummary,
    ReportMeasurements, SearchBudgetConfiguration, SearchPhase, SearchTruncation,
    SteadyStateMeasurements, Topology, WorkloadDispatch, WriteAmplification,
};
use crate::resource::ResourceSnapshot;

#[path = "../../tests/support/oracle.rs"]
#[expect(
    dead_code,
    reason = "the shared oracle also exposes filter helpers used only by integration tests"
)]
mod oracle;

/// Queue capacity keeps the default Import backlog watermark valid.
const FIXUP_QUEUE_CAPACITY: usize = 1_024;
/// Workers used after import diagnostics to converge persistent topology.
const CONVERGENCE_MAINTENANCE_WORKERS: usize = 2;

/// One immutable exact top-k result shared by repeated query operations.
type ExactTruth = Arc<[(Bytes, f64)]>;

/// A successful search retained until recall can run outside resource timing.
#[derive(Debug)]
struct RecallInput {
    /// Approximate hits returned inside the measured API call.
    outcome: SearchOutcome,
    /// Immutable exact result computed before any benchmark timing.
    truth: ExactTruth,
}

/// One fixed scenario in a named scale profile.
#[derive(Clone, Debug)]
pub struct ScenarioSpec {
    /// Stable scenario name used as the comparison key.
    pub name: &'static str,
    /// Public fixture or synthetic distribution.
    pub dataset: &'static str,
    /// Human-readable scale profile.
    pub profile: &'static str,
    /// Indexed vector count.
    pub base_vectors: usize,
    /// Held-out query count.
    pub query_vectors: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Replayable dataset and workload seed.
    pub seed: u64,
    /// Percentage of operations that are searches.
    pub search_percent: u8,
    /// Whether updates concentrate on a small conflict set.
    pub hot_updates: bool,
    /// Runtime Partition Cache capacity.
    pub partition_cache_bytes: u64,
    /// Runtime foreground concurrency and wait bound.
    pub foreground_limit: usize,
    /// RocksDB native actor bound; ignored by other Backends.
    pub blocking_resource_limit: Option<usize>,
    /// Concurrent clients in the timed workload.
    pub concurrency: usize,
    /// Dispatch policy for the bounded concurrent clients.
    pub dispatch: WorkloadDispatch,
    /// Operations outside the timed region that establish steady state.
    pub warmup_operations: usize,
    /// Operations in the timed region.
    pub measured_operations: usize,
    /// Search result count.
    pub k: usize,
    /// Per-request Search Budget and traversal overrides.
    pub search_options: SearchOptions,
    /// Logical Index maximum partition size.
    pub max_partition_entries: u32,
    /// Whether this scenario measures the import-to-search lifecycle.
    pub lifecycle: bool,
    /// Records per Import Session batch in a lifecycle scenario.
    pub import_batch_size: usize,
    /// Explicit Import Session in-flight batch limit.
    pub import_in_flight_batches: usize,
    /// Explicit Runtime Fixup backlog watermark for Import Session admission.
    pub import_backlog_watermark: usize,
    /// Background Structure Maintenance workers.
    pub maintenance_workers: usize,
}

/// Returns the complete deterministic scenario matrix for one scale profile.
///
/// # Errors
///
/// Returns an error when `profile` is neither `smoke` nor `full`.
pub fn scenarios(profile: &str) -> Result<Vec<ScenarioSpec>, String> {
    match profile {
        "smoke" => Ok(smoke_scenarios()),
        "full" => Ok(full_scenarios()),
        _ => Err(format!(
            "unknown profile `{profile}`; expected smoke or full"
        )),
    }
}

/// Small deterministic matrix used to catch runner and adapter breakage in CI.
fn smoke_scenarios() -> Vec<ScenarioSpec> {
    let common = ScenarioSpec {
        name: "ann-warm-cache",
        dataset: "clustered",
        profile: "smoke",
        base_vectors: 256,
        query_vectors: 8,
        dimension: 16,
        seed: 0x38_0001,
        search_percent: 100,
        hot_updates: false,
        partition_cache_bytes: 4 << 20,
        foreground_limit: 8,
        blocking_resource_limit: None,
        concurrency: 2,
        dispatch: WorkloadDispatch::Continuous,
        warmup_operations: 16,
        measured_operations: 64,
        k: 10,
        search_options: SearchOptions::default(),
        max_partition_entries: 32,
        lifecycle: false,
        import_batch_size: 32,
        import_in_flight_batches: 2,
        import_backlog_watermark: 512,
        maintenance_workers: 2,
    };
    vec![
        common.clone(),
        ScenarioSpec {
            name: "ann-cache-disabled",
            partition_cache_bytes: 0,
            ..common.clone()
        },
        ScenarioSpec {
            name: "mixed-search95-update5",
            search_percent: 95,
            concurrency: 4,
            measured_operations: 80,
            ..common.clone()
        },
        ScenarioSpec {
            name: "mixed-search50-update50-hot",
            search_percent: 50,
            hot_updates: true,
            concurrency: 4,
            measured_operations: 80,
            ..common.clone()
        },
        ScenarioSpec {
            name: "backend-admission-saturated",
            search_percent: 50,
            hot_updates: true,
            blocking_resource_limit: Some(1),
            concurrency: 16,
            foreground_limit: 4,
            dispatch: WorkloadDispatch::FixedWaves(admission_target()),
            measured_operations: 640,
            ..common.clone()
        },
        ScenarioSpec {
            name: "import-to-search-lifecycle",
            lifecycle: true,
            search_percent: 100,
            concurrency: 1,
            warmup_operations: 0,
            measured_operations: 8,
            hot_updates: false,
            blocking_resource_limit: None,
            dispatch: WorkloadDispatch::Continuous,
            ..common.clone()
        },
    ]
}

/// Representative public and synthetic baselines reserved for scheduled runs.
fn full_scenarios() -> Vec<ScenarioSpec> {
    let ann = |name, dataset, base_vectors, query_vectors, dimension, seed| ScenarioSpec {
        name,
        dataset,
        profile: "full",
        base_vectors,
        query_vectors,
        dimension,
        seed,
        search_percent: 100,
        hot_updates: false,
        partition_cache_bytes: 64 << 20,
        foreground_limit: 32,
        blocking_resource_limit: None,
        concurrency: 4,
        dispatch: WorkloadDispatch::Continuous,
        warmup_operations: query_vectors,
        measured_operations: query_vectors.saturating_mul(10),
        k: 10,
        search_options: SearchOptions::default(),
        max_partition_entries: 128,
        lifecycle: false,
        import_batch_size: 50,
        import_in_flight_batches: 4,
        import_backlog_watermark: 512,
        maintenance_workers: 2,
    };
    let clustered = ann("ann-clustered", "clustered", 5_000, 100, 128, 0x38_1001);
    vec![
        ann("ann-siftsmall", "siftsmall", 10_000, 100, 128, 0x38_1002),
        ann(
            "ann-fashion-mnist",
            "fashion-mnist",
            10_000,
            20,
            784,
            0x38_1003,
        ),
        clustered.clone(),
        ann("ann-skewed", "skewed", 5_000, 100, 128, 0x38_1004),
        ann("ann-duplicates", "duplicates", 5_000, 100, 128, 0x38_1005),
        ScenarioSpec {
            name: "ann-cache-disabled",
            partition_cache_bytes: 0,
            ..clustered.clone()
        },
        ScenarioSpec {
            name: "mixed-search95-update5",
            search_percent: 95,
            measured_operations: 2_000,
            concurrency: 8,
            ..clustered.clone()
        },
        ScenarioSpec {
            name: "mixed-search50-update50-hot",
            search_percent: 50,
            hot_updates: true,
            measured_operations: 2_000,
            concurrency: 8,
            ..clustered.clone()
        },
        ScenarioSpec {
            name: "backend-admission-saturated",
            search_percent: 50,
            hot_updates: true,
            blocking_resource_limit: Some(1),
            foreground_limit: 8,
            concurrency: 32,
            measured_operations: 2_000,
            dispatch: WorkloadDispatch::FixedWaves(admission_target()),
            ..clustered.clone()
        },
        ScenarioSpec {
            name: "import-to-search-lifecycle",
            dataset: "siftsmall",
            base_vectors: 10_000,
            query_vectors: 100,
            dimension: 128,
            seed: 0x38_1006,
            lifecycle: true,
            search_percent: 100,
            concurrency: 1,
            warmup_operations: 0,
            measured_operations: 100,
            hot_updates: false,
            blocking_resource_limit: None,
            dispatch: WorkloadDispatch::Continuous,
            // Keep one batch active so the fixed corpus is fully imported
            // while Structure Maintenance competes for the same partitions.
            import_in_flight_batches: 1,
            ..clustered.clone()
        },
    ]
}

/// Defines a sample-rich mixed region while preserving visible overload.
const fn admission_target() -> AdmissionTarget {
    AdmissionTarget {
        minimum_accepted_per_class: 100,
        minimum_rejection_rate: 0.25,
        maximum_rejection_rate: 0.75,
    }
}

/// Runs one scenario through KTANN's public Runtime and Index APIs.
///
/// # Errors
///
/// Returns an error when inputs are invalid, a public KTANN operation fails,
/// topology does not converge, verification finds an issue, or resource
/// sampling fails.
pub async fn run_scenario<B: Backend>(
    backend_name: &str,
    backend_runtime: String,
    backend: B,
    spec: &ScenarioSpec,
    reproduction_command: String,
    tokio_worker_threads: usize,
) -> Result<BenchmarkReport, String> {
    let dataset = dataset::load(
        spec.dataset,
        spec.base_vectors,
        spec.query_vectors,
        spec.dimension,
        spec.seed,
    )?;
    let (backend, backend_counters) = MeasuredBackend::new(backend);
    let admission = backend.admission_budget();
    let metric_capture = MetricCapture::install()?;
    let primary_runtime_config = runtime_config(spec, spec.maintenance_workers)?;
    let default_search_budgets = primary_runtime_config.default_search_budgets();
    let search_budgets =
        search_budget_configuration(default_search_budgets, spec.search_options, spec.k)?;
    let (topology, measurements) = if spec.lifecycle {
        let (topology, lifecycle) = run_lifecycle_case(
            backend,
            primary_runtime_config,
            runtime_config(spec, CONVERGENCE_MAINTENANCE_WORKERS)?,
            &backend_counters,
            &metric_capture,
            spec,
            &dataset,
        )
        .await?;
        (topology, ReportMeasurements::Lifecycle(Box::new(lifecycle)))
    } else {
        let runtime = Runtime::new(backend, primary_runtime_config)
            .map_err(|error| error_at("create runtime", error))?;
        let measured =
            run_with_runtime(&runtime, &backend_counters, &metric_capture, spec, &dataset).await;
        let shutdown = runtime
            .shutdown()
            .await
            .map_err(|error| error_at("shut down runtime", error));
        let (topology, measurements) = measured?;
        shutdown?;
        (
            topology,
            ReportMeasurements::SteadyState(Box::new(measurements)),
        )
    };

    Ok(BenchmarkReport {
        generated_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        reproduction_command,
        git_revision: git_revision(),
        environment: environment(tokio_worker_threads, backend_runtime),
        configuration: Configuration {
            backend: backend_name.to_owned(),
            scenario: spec.name.to_owned(),
            profile: spec.profile.to_owned(),
            seed: spec.seed,
            dimension: spec.dimension,
            metric: "l2".to_owned(),
            search_percent: spec.search_percent,
            hot_updates: spec.hot_updates,
            min_partition_entries: spec.max_partition_entries / 4,
            max_partition_entries: spec.max_partition_entries,
            partition_cache_bytes: spec.partition_cache_bytes,
            foreground_limit: spec.foreground_limit,
            maintenance_workers: spec.maintenance_workers,
            convergence_maintenance_workers: spec
                .lifecycle
                .then_some(CONVERGENCE_MAINTENANCE_WORKERS),
            fixup_queue_capacity: FIXUP_QUEUE_CAPACITY,
            search_budgets,
            leaf_beam_size_override: spec.search_options.leaf_beam_size(),
            blocking_resource_limit: spec.blocking_resource_limit,
            backend_max_mutations: admission.max_mutations,
            backend_max_mutation_bytes: admission.max_mutation_bytes,
            backend_mutation_key_overhead_bytes: admission.mutation_key_overhead_bytes,
            concurrency: spec.concurrency,
            dispatch: spec.dispatch,
            warmup_operations: spec.warmup_operations,
            measured_operations: spec.measured_operations,
            k: spec.k,
            import_batch_size: spec.lifecycle.then_some(spec.import_batch_size),
            import_in_flight_batches: spec.lifecycle.then_some(spec.import_in_flight_batches),
            import_backlog_watermark: spec.lifecycle.then_some(spec.import_backlog_watermark),
        },
        dataset: dataset.metadata,
        topology,
        measurements,
    })
}

/// Builds the complete process-local configuration used by benchmark workers.
fn runtime_config(
    spec: &ScenarioSpec,
    maintenance_workers: usize,
) -> Result<RuntimeConfig, String> {
    // The Fixup queue retains the default capacity so the default Import
    // Session backlog watermark remains within it. RuntimeConfig validates
    // those two process-local resource bounds together at Runtime creation.
    let config = RuntimeConfig::default()
        .with_foreground_operation_limit(spec.foreground_limit)
        .and_then(|config| config.with_maintenance(maintenance_workers, FIXUP_QUEUE_CAPACITY))
        .and_then(|config| config.with_attempts(32, 32))
        .and_then(|config| config.with_partition_cache_bytes(spec.partition_cache_bytes));
    let config = if spec.lifecycle {
        config.and_then(|config| {
            config.with_import_limits(spec.import_in_flight_batches, spec.import_backlog_watermark)
        })
    } else {
        config
    };
    config
        .and_then(|config| config.validate().map(|()| config))
        .map_err(|error| error_at("configure runtime", error))
}

/// Resolves the exact limits used by scenario requests through the public API.
fn search_budget_configuration(
    defaults: SearchBudgets,
    options: SearchOptions,
    k: usize,
) -> Result<SearchBudgetConfiguration, String> {
    let effective = options
        .resolve(defaults, k)
        .map_err(|error| error_at("resolve search budgets", error))?;
    Ok(SearchBudgetConfiguration {
        scanned_tree_keys: BudgetConfiguration {
            runtime_default: defaults.scanned_tree_keys(),
            request_override: options.scanned_tree_keys(),
            effective_limit: effective.scanned_tree_keys(),
        },
        visited_partitions: BudgetConfiguration {
            runtime_default: defaults.visited_partitions(),
            request_override: options.visited_partitions(),
            effective_limit: effective.visited_partitions(),
        },
        visited_leaf_entries: BudgetConfiguration {
            runtime_default: defaults.visited_leaf_entries(),
            request_override: options.visited_leaf_entries(),
            effective_limit: effective.visited_leaf_entries(),
        },
        exact_rerank_candidates: BudgetConfiguration {
            runtime_default: defaults.exact_rerank_candidates(),
            request_override: options.exact_rerank_candidates(),
            effective_limit: effective.exact_rerank_candidates(),
        },
    })
}

/// Measures one fresh Index from Import Session submission through warm search.
async fn run_lifecycle_case<B: Backend>(
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
    let truths = exact_truth(dataset, spec.k);
    let requests = lifecycle_requests(dataset, spec)?;
    let import_options = ImportOptions::default()
        .with_in_flight_batches(spec.import_in_flight_batches)
        .map_err(|error| error_at("configure Import Session", error))?;
    let import_session = index
        .import_session(import_options)
        .map_err(|error| error_at("open Import Session", error))?;

    let _ = metric_capture.snapshot();
    let (import, case_baseline) =
        run_import_phase(import_session, batches, backend_counters, metric_capture).await?;
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
            metric_capture,
        ),
        from_import_finish_seconds: convergence_completed
            .duration_since(import_finished)
            .as_secs_f64(),
        maintenance_drain_seconds,
    };

    // A new Runtime is the public, deterministic cache-reset boundary. It
    // preserves the stable persistent topology while ensuring the following
    // query pass starts with an empty process-local Partition Cache.
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
        metric_capture,
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
    let case_backend_io = stable_warm
        .backend_after
        .checked_sub(&case_baseline.backend_io)
        .ok_or_else(|| "continuous Backend counters decreased".to_owned())?;
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
    lifecycle
        .unattributed_wall_seconds()
        .ok_or_else(|| "lifecycle phases exceed the continuous wall time".to_owned())?;
    lifecycle
        .unattributed_cpu_seconds()
        .ok_or_else(|| "lifecycle phases exceed the continuous CPU time".to_owned())?;
    lifecycle
        .unattributed_backend_io()
        .ok_or_else(|| "lifecycle phases exceed the continuous Backend IO".to_owned())?;

    Ok((topology, lifecycle))
}

/// Exact continuous-case baselines captured immediately before the first submit.
struct CaseBaseline {
    /// Monotonic first-submit timer boundary.
    started: Instant,
    /// Cumulative process resources at the same boundary.
    resources: ResourceSnapshot,
    /// Cumulative Backend IO at the same boundary.
    backend_io: crate::report::BackendIo,
}

/// Builds the public Index configuration shared by steady and lifecycle cases.
fn index_config(spec: &ScenarioSpec) -> Result<IndexConfig, String> {
    IndexConfig::new(spec.dimension, Metric::L2)
        .and_then(|config| {
            config.with_fields(vec![
                FieldSchema::new("bucket", DataType::I64)?,
                FieldSchema::new("generation", DataType::I64)?,
            ])
        })
        .and_then(|config| config.with_tree_key_fields(vec![FieldId(0)]))
        .and_then(|config| {
            config
                .with_partition_entries(spec.max_partition_entries / 4, spec.max_partition_entries)
        })
        .map_err(|error| error_at("configure index", error))
}

/// Materializes bounded mutation batches before the first submission timer.
fn mutation_batches(
    dataset: &BenchmarkDataset,
    batch_size: usize,
    phase: &str,
) -> Result<Vec<Vec<Mutation>>, String> {
    dataset
        .base
        .chunks(batch_size)
        .enumerate()
        .map(|(batch, vectors)| {
            vectors
                .iter()
                .enumerate()
                .map(|(offset, vector)| {
                    let ordinal = batch
                        .checked_mul(batch_size)
                        .and_then(|ordinal| ordinal.checked_add(offset))
                        .ok_or_else(|| "import record ordinal overflow".to_owned())?;
                    Record::new(
                        dataset.ids[ordinal].clone(),
                        Arc::clone(vector),
                        vec![Value::I64(0), Value::I64(0)],
                    )
                    .map(Mutation::Insert)
                    .map_err(|error| error_at(phase, error))
                })
                .collect()
        })
        .collect()
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

/// Constructs one deterministic public search request.
fn search_request(
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
    query_index: usize,
) -> Result<SearchRequest, String> {
    SearchRequest::new(Arc::clone(&dataset.queries[query_index]), spec.k)
        .map(|request| request.with_options(spec.search_options))
        .map_err(|error| error_at("construct search request", error))
}

/// Executes one real bounded Import Session and summarizes accepted outcomes.
async fn run_import_phase<B: Backend>(
    mut session: ImportSession<MeasuredBackend<B>>,
    batches: Vec<Vec<Mutation>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
) -> Result<(ImportPhase, CaseBaseline), String> {
    let mut submitted_batch_sizes = Vec::with_capacity(batches.len());
    let mut submit_latency_ms = Vec::with_capacity(batches.len());
    let mut failures = BTreeMap::new();
    let mut resources_before = None;
    let mut backend_before = None;
    let mut started = None;
    for batch in batches {
        if started.is_none() {
            resources_before = Some(ResourceSnapshot::capture()?);
            backend_before = Some(backend_counters.snapshot());
        }
        let records = batch.len();
        let submit_started = Instant::now();
        started.get_or_insert(submit_started);
        match session.submit(batch).await {
            Ok(_) => submitted_batch_sizes.push(records),
            Err(error) => {
                *failures.entry(format!("{:?}", error.kind())).or_default() += 1;
            }
        }
        submit_latency_ms.push(submit_started.elapsed().as_secs_f64() * 1_000.0);
    }
    let started = started.ok_or_else(|| "lifecycle import has no batches".to_owned())?;
    let resources_before =
        resources_before.ok_or_else(|| "lifecycle import has no resource baseline".to_owned())?;
    let backend_before =
        backend_before.ok_or_else(|| "lifecycle import has no Backend IO baseline".to_owned())?;
    let results = session.finish().await;
    let wall_seconds = started.elapsed().as_secs_f64();
    let resources_after = ResourceSnapshot::capture()?;
    let backend_io = backend_counters.since(&backend_before);
    let metrics = metric_capture.snapshot();
    if results.len() != submitted_batch_sizes.len() {
        return Err("Import Session result count differs from submitted batches".to_owned());
    }
    let mut accepted_batches = 0_u64;
    let mut accepted_records = 0_u64;
    let submitted_records = submitted_batch_sizes.iter().try_fold(0_u64, |total, size| {
        total.checked_add(u64::try_from(*size).ok()?)
    });
    let submitted_records =
        submitted_records.ok_or_else(|| "import record count overflow".to_owned())?;
    for (batch_size, result) in submitted_batch_sizes.iter().zip(results) {
        match result.result {
            Ok(outcomes) => {
                accepted_batches = accepted_batches.saturating_add(1);
                accepted_records = accepted_records.saturating_add(
                    u64::try_from(outcomes.len()).map_err(|_| "import record count overflow")?,
                );
                if outcomes.len() != *batch_size {
                    return Err("Import Session outcome count differs from batch size".to_owned());
                }
            }
            Err(error) => {
                *failures.entry(format!("{:?}", error.kind())).or_default() += 1;
            }
        }
    }
    if accepted_records != submitted_records {
        return Err(format!(
            "Import Session accepted {accepted_records} of {submitted_records} records; batch failures: {failures:?}"
        ));
    }
    let submitted_batches = u64::try_from(submitted_batch_sizes.len())
        .map_err(|_| "import batch count overflow".to_owned())?;
    let phase = ImportPhase {
        resources: phase_resources(
            wall_seconds,
            resources_before,
            resources_after,
            backend_io,
            &metrics,
            metric_capture,
        ),
        submitted_batches,
        accepted_batches,
        accepted_records,
        records_per_second: if wall_seconds > 0.0 {
            accepted_records as f64 / wall_seconds
        } else {
            0.0
        },
        submit_latency_ms: Distribution::from_samples(submit_latency_ms),
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
    let backend_io = backend_after
        .checked_sub(&backend_before)
        .ok_or_else(|| "phase Backend counters decreased".to_owned())?;
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
                summary.visited_leaf_entries += u64::from(outcome.exhausted.visited_leaf_entries);
                summary.exact_rerank_candidates +=
                    u64::from(outcome.exhausted.exact_rerank_candidates);
                summary.rabitq_overlap += u64::from(outcome.rabitq_overlap_truncated);
                summary
            });
    let phase = SearchPhase {
        resources: phase_resources(
            wall_seconds,
            resources_before,
            resources_after,
            backend_io,
            &metrics,
            metric_capture,
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
        search_stages_ms: search_stage_summaries(&metrics),
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
    backend_after: crate::report::BackendIo,
}

/// Builds common resource and maintenance accounting at a phase boundary.
fn phase_resources(
    wall_seconds: f64,
    before: ResourceSnapshot,
    after: ResourceSnapshot,
    backend_io: crate::report::BackendIo,
    metrics: &CapturedMetrics,
    metric_capture: &MetricCapture,
) -> PhaseResources {
    PhaseResources {
        wall_seconds,
        cpu_seconds: Some(after.cpu_seconds_since(before)),
        peak_rss_bytes: Some(after.peak_rss_bytes()),
        backend_io,
        writes: metrics.write_attribution(),
        maintenance: MaintenanceSummary {
            admission: metrics.counters_rendered("ktann.fixup.admission"),
            execution: metrics.counters_rendered("ktann.fixup.execution"),
            steps: metrics.counters_rendered("ktann.fixup.steps"),
            drain_entries: metrics.distributions_rendered("ktann.fixup.drain.entries"),
            backlog_at_end: metric_capture.fixup_backlog(),
        },
    }
}

/// Converts measured search-stage histograms into milliseconds.
fn search_stage_summaries(metrics: &CapturedMetrics) -> BTreeMap<String, Distribution> {
    metrics
        .histograms_by_label("ktann.search.stage.duration", "stage")
        .into_iter()
        .map(|(stage, samples)| {
            (
                stage,
                Distribution::from_samples(samples).seconds_to_milliseconds(),
            )
        })
        .collect()
}

/// Owns setup, warmup, measurement, and the post-run invariant audit.
async fn run_with_runtime<B: Backend>(
    runtime: &Runtime<MeasuredBackend<B>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &BenchmarkDataset,
) -> Result<(Topology, SteadyStateMeasurements), String> {
    let index_config = index_config(spec)?;
    let index = runtime
        .create_index("benchmark", index_config)
        .await
        .map_err(|error| error_at("create index", error))?;

    // Setup is deliberately complete before any timer or counter baseline:
    // load transactions, split training, verification, oracle construction,
    // and cache warmup therefore cannot make measured operations look slower.
    load_index(&index, dataset).await?;
    let topology = settle_topology(&index, dataset, spec).await?;
    let truth = (spec.search_percent == 100).then(|| exact_truth(dataset, spec.k));
    let warmup = work_items(dataset, None, spec, spec.warmup_operations, 0)?;
    let _warmup_result =
        execute_items(index.clone(), warmup, spec.concurrency, spec.dispatch).await?;

    // Reset setup counters and histograms before the measured interval. Gauges
    // retain current process state and need no compatibility fallback.
    let _ = metric_capture.snapshot();
    wait_for_maintenance(metric_capture, settle_timeout(spec)).await?;
    let backend_before = backend_counters.snapshot();
    let measured_items = work_items(
        dataset,
        truth.as_deref(),
        spec,
        spec.measured_operations,
        spec.warmup_operations,
    )?;

    let resources_before = ResourceSnapshot::capture()?;
    let wall_started = Instant::now();
    let workload = execute_items(
        index.clone(),
        measured_items,
        spec.concurrency,
        spec.dispatch,
    )
    .await?;
    let wall_seconds = wall_started.elapsed().as_secs_f64();
    let metrics = metric_capture.snapshot();
    let maintenance_started = Instant::now();
    wait_for_maintenance(metric_capture, settle_timeout(spec)).await?;
    let maintenance_drain_seconds = maintenance_started.elapsed().as_secs_f64();
    let resources_after = ResourceSnapshot::capture()?;
    let backend_io = backend_counters.since(&backend_before);
    let recalls = workload.recall_values();

    // Verification is outside the timed region and protects the benchmark
    // itself from reporting fast results produced by corrupt persistent state.
    let final_report = index
        .verify(VerifyOptions::default())
        .await
        .map_err(|error| error_at("verify measured state", error))?;
    if !final_report.complete || !final_report.issues.is_empty() {
        return Err(format!(
            "post-run verification failed: complete={}, issues={}",
            final_report.complete,
            final_report.issues.len()
        ));
    }

    let successful_writes = workload.accepted_operations(OperationClass::Write);
    let write_amplification = (successful_writes > 0).then(|| WriteAmplification {
        successful_writes,
        logical_mutations_per_write: backend_io.mutation_operations as f64
            / successful_writes as f64,
        logical_bytes_per_write: backend_io.mutation_bytes as f64 / successful_writes as f64,
        write_retries: metrics.write_retries(),
    });
    let successful_operations = workload.successful_operations();
    let operations = workload.into_operation_summaries(&metrics);
    validate_admission_target(&operations, spec)?;
    let measurements = SteadyStateMeasurements {
        wall_seconds,
        maintenance_drain_seconds,
        cpu_seconds: Some(resources_after.cpu_seconds_since(resources_before)),
        peak_rss_bytes: Some(resources_after.peak_rss_bytes()),
        throughput_per_second: if wall_seconds > 0.0 {
            successful_operations as f64 / wall_seconds
        } else {
            0.0
        },
        operations,
        recall_at_k: recall_summary(&recalls),
        search_budgets: budget_summaries(&metrics),
        search_stages_ms: search_stage_summaries(&metrics),
        cache: metrics.cache_summary(),
        backend_admission: metrics.admission_summary(),
        backend_io,
        write_amplification,
    };
    Ok((topology, measurements))
}

/// Rejects pressure samples that do not occupy their declared operating region.
fn validate_admission_target(
    operations: &BTreeMap<OperationClass, OperationSummary>,
    spec: &ScenarioSpec,
) -> Result<(), String> {
    let WorkloadDispatch::FixedWaves(target) = spec.dispatch else {
        return Ok(());
    };
    for class in OperationClass::for_mix(spec.search_percent) {
        let accepted = operations.get(&class).map_or(0, |summary| summary.accepted);
        if accepted < target.minimum_accepted_per_class {
            return Err(format!(
                "admission sample has {} accepted {} operations; target requires at least {}",
                accepted,
                class.as_str(),
                target.minimum_accepted_per_class
            ));
        }
    }
    let attempted: u64 = operations.values().map(|summary| summary.attempted).sum();
    let rejected: u64 = operations.values().map(|summary| summary.rejected).sum();
    let rejection_rate = if attempted == 0 {
        0.0
    } else {
        rejected as f64 / attempted as f64
    };
    if rejection_rate < target.minimum_rejection_rate
        || rejection_rate > target.maximum_rejection_rate
    {
        return Err(format!(
            "admission sample has {rejected}/{attempted} rejected operations ({rejection_rate:.4}); target rate is [{:.4}, {:.4}]",
            target.minimum_rejection_rate, target.maximum_rejection_rate
        ));
    }
    Ok(())
}

/// Loads ordinary atomic mutation batches, retrying only definite exhaustion.
async fn load_index<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    dataset: &BenchmarkDataset,
) -> Result<(), String> {
    for mutations in mutation_batches(dataset, 50, "construct load record")? {
        let mut attempts = 0_u32;
        loop {
            match index.batch_mutate(mutations.clone()).await {
                Ok(_) => break,
                Err(error) if error.kind() == ErrorKind::ContentionExhausted && attempts < 32 => {
                    attempts += 1;
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(error) => return Err(error_at("load records", error)),
            }
        }
    }
    Ok(())
}

/// Drives relevant accesses until the structured topology is observably stable.
async fn settle_topology<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
) -> Result<Topology, String> {
    let deadline = Instant::now() + settle_timeout(spec);
    settle_topology_until(index, dataset, spec, deadline).await
}

/// Drives topology convergence without extending the caller's absolute deadline.
async fn settle_topology_until<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
    deadline: Instant,
) -> Result<Topology, String> {
    let mut previous = None;
    let mut stable_rounds = 0_u8;
    loop {
        let topology = verified_topology(index, "verify topology", deadline).await?;
        let structured = topology.partitions > 1 && topology.entries > topology.vector_records;
        stable_rounds = if structured {
            if previous.as_ref() == Some(&topology) {
                stable_rounds.saturating_add(1)
            } else {
                1
            }
        } else {
            0
        };
        if stable_rounds >= 3 {
            return Ok(topology);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "topology did not stabilize: records={}, partitions={}, entries={}",
                topology.vector_records, topology.partitions, topology.entries
            ));
        }
        previous = Some(topology.clone());

        // Search is the public access path that rediscovers cold intermediate
        // states. Cycling a few held-out queries makes progress demand-driven
        // without introducing a private maintenance control surface.
        for query in dataset.queries.iter().take(4) {
            let request = SearchRequest::new(Arc::clone(query), spec.k)
                .map(|request| request.with_options(spec.search_options))
                .map_err(|error| error_at("construct topology-settling search", error))?;
            index
                .search_with_control(request, OperationOptions::default().with_deadline(deadline))
                .await
                .map_err(|error| error_at("settle topology", error))?;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Repeats stable observation and maintenance drain until both agree.
async fn settle_and_drain_topology<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
    metric_capture: &MetricCapture,
    deadline: Instant,
) -> Result<(Topology, f64), String> {
    let mut maintenance_drain_seconds = 0.0;
    loop {
        let topology = settle_topology_until(index, dataset, spec, deadline).await?;
        let drain_started = Instant::now();
        wait_for_maintenance_until(metric_capture, deadline).await?;
        maintenance_drain_seconds += drain_started.elapsed().as_secs_f64();
        let drained = verified_topology(index, "verify drained topology", deadline).await?;
        if drained == topology {
            return Ok((drained, maintenance_drain_seconds));
        }
    }
}

/// Verifies persistent state and returns its observable topology counts.
async fn verified_topology<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    phase: &str,
    deadline: Instant,
) -> Result<Topology, String> {
    let report = index
        .verify(
            VerifyOptions::default()
                .with_operation_options(OperationOptions::default().with_deadline(deadline)),
        )
        .await
        .map_err(|error| error_at(phase, error))?;
    if !report.complete || !report.issues.is_empty() {
        return Err(format!(
            "{phase} failed: complete={}, issues={}",
            report.complete,
            report.issues.len()
        ));
    }
    Ok(Topology {
        vector_records: report.objects.vector_records,
        partitions: report.objects.partitions,
        entries: report.objects.entries,
    })
}

/// Returns the bounded setup and maintenance-drain allowance for a profile.
fn settle_timeout(spec: &ScenarioSpec) -> Duration {
    if spec.profile == "smoke" {
        Duration::from_secs(30)
    } else {
        Duration::from_secs(120)
    }
}

/// Waits until all maintenance caused before the accounting boundary finishes.
async fn wait_for_maintenance(
    metric_capture: &MetricCapture,
    timeout: Duration,
) -> Result<(), String> {
    wait_for_maintenance_until(metric_capture, Instant::now() + timeout).await
}

/// Waits for maintenance without extending an existing absolute deadline.
async fn wait_for_maintenance_until(
    metric_capture: &MetricCapture,
    deadline: Instant,
) -> Result<(), String> {
    let mut backlog = metric_capture.fixup_backlog();
    while backlog > 0 {
        if Instant::now() >= deadline {
            return Err(format!(
                "Structure Maintenance did not drain: backlog={backlog}"
            ));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        backlog = metric_capture.fixup_backlog();
    }
    Ok(())
}

/// One fully materialized operation; construction stays outside measurement.
enum WorkItem {
    /// One held-out ANN query and optional immutable brute-force truth.
    Search {
        /// Fully validated request constructed before operation timing starts.
        request: SearchRequest,
        /// Exact truth only for read-only workloads whose model stays valid.
        truth: Option<ExactTruth>,
    },
    /// One replacement upsert derived from a stable base vector.
    Upsert {
        /// Fully validated Record constructed before operation timing starts.
        record: Record,
    },
}

/// Measurements returned by one operation task.
#[derive(Debug)]
struct OperationObservation {
    /// Public operation class used for per-class admission reporting.
    class: OperationClass,
    /// End-to-end API latency; aggregation retains it only after success.
    latency_ms: f64,
    /// Optional search-recall input on success, or a stable failure category.
    result: Result<Option<RecallInput>, ErrorKind>,
}

/// Aggregation built only after all bounded operation tasks have joined.
#[derive(Debug, Default)]
struct WorkloadObservation {
    /// Attempt and outcome observations grouped by public operation class.
    operations: BTreeMap<OperationClass, OperationClassObservation>,
    /// Search results whose exact recall is computed after resource timing.
    recall_inputs: Vec<RecallInput>,
}

/// Raw samples for one public operation class.
#[derive(Debug, Default)]
struct OperationClassObservation {
    /// Public calls issued in the measured region.
    attempted: u64,
    /// Calls that returned success.
    accepted: u64,
    /// Accepted-call latency samples.
    latency_ms: Vec<f64>,
    /// All failures grouped by stable public error category.
    errors: BTreeMap<String, u64>,
}

/// Materializes deterministic operation choices before the timed region.
fn work_items(
    dataset: &BenchmarkDataset,
    truth: Option<&[ExactTruth]>,
    spec: &ScenarioSpec,
    count: usize,
    offset: usize,
) -> Result<Vec<WorkItem>, String> {
    let mut items = Vec::with_capacity(count);
    for operation in offset..offset.saturating_add(count) {
        let searches = is_search_operation(operation, spec.search_percent);
        if searches {
            let query_index = operation % dataset.queries.len();
            // Recall is meaningful only while the indexed vectors remain the
            // immutable oracle model. Mixed workloads still report budgets and
            // latency, but never compare against stale pre-update truth.
            let truth = truth.map(|truth| Arc::clone(&truth[query_index]));
            let request = search_request(dataset, spec, query_index)?;
            items.push(WorkItem::Search { request, truth });
        } else {
            let ordinal = if spec.hot_updates {
                operation % dataset.base.len().min(8)
            } else {
                operation % dataset.base.len()
            };
            let mut vector = dataset.base[ordinal].to_vec();
            // Toggle one finite component around the original vector. Every
            // operation remains a replacement rather than accumulating drift,
            // so the workload is replayable under arbitrary task scheduling.
            vector[0] += if operation % 2 == 0 { 0.001 } else { -0.001 };
            let generation = i64::try_from(operation).unwrap_or(i64::MAX);
            let record = Record::new(
                dataset.ids[ordinal].clone(),
                Arc::<[f32]>::from(vector),
                vec![Value::I64(0), Value::I64(generation)],
            )
            .map_err(|error| error_at("construct update record", error))?;
            items.push(WorkItem::Upsert { record });
        }
    }
    Ok(items)
}

/// Spreads a percentage mix uniformly instead of clustering one operation kind.
fn is_search_operation(operation: usize, search_percent: u8) -> bool {
    // Multiplication by the percentage permutes a short modulo-100 cycle. For
    // 95/5 this yields one update every 20 operations; for 50/50 it alternates.
    // Reducing the ordinal first keeps the arithmetic bounded independently of
    // how many warmup or measurement operations a future profile requests.
    (operation % 100) * usize::from(search_percent) % 100 < usize::from(search_percent)
}

/// Maintains bounded in-flight concurrency while retaining each task's latency.
async fn execute_items<B: Backend>(
    index: Index<MeasuredBackend<B>>,
    items: Vec<WorkItem>,
    concurrency: usize,
    dispatch: WorkloadDispatch,
) -> Result<WorkloadObservation, String> {
    match dispatch {
        WorkloadDispatch::Continuous => execute_continuous(index, items, concurrency).await,
        WorkloadDispatch::FixedWaves(_) => execute_fixed_waves(index, items, concurrency).await,
    }
}

/// Sustains pressure by replacing each client as soon as its operation ends.
async fn execute_continuous<B: Backend>(
    index: Index<MeasuredBackend<B>>,
    items: Vec<WorkItem>,
    concurrency: usize,
) -> Result<WorkloadObservation, String> {
    let mut aggregate = WorkloadObservation::default();
    let mut items = items.into_iter();
    let mut tasks = tokio::task::JoinSet::new();
    for item in items.by_ref().take(concurrency) {
        let index = index.clone();
        tasks.spawn(async move { execute_item(&index, item).await });
    }
    while let Some(result) = tasks.join_next().await {
        let observation = result.map_err(|error| format!("workload task failed: {error}"))?;
        aggregate.push(observation);
        // Replenish on each completion rather than waiting for the slowest
        // operation in a wave, so admission sees sustained client pressure.
        if let Some(item) = items.next() {
            let index = index.clone();
            tasks.spawn(async move { execute_item(&index, item).await });
        }
    }
    Ok(aggregate)
}

/// Applies repeatable overload bursts without letting fast rejections self-amplify.
async fn execute_fixed_waves<B: Backend>(
    index: Index<MeasuredBackend<B>>,
    items: Vec<WorkItem>,
    concurrency: usize,
) -> Result<WorkloadObservation, String> {
    let mut aggregate = WorkloadObservation::default();
    let mut items = items.into_iter();
    let mut tasks = tokio::task::JoinSet::new();
    loop {
        let wave_size = items.len().min(concurrency);
        if wave_size == 0 {
            break;
        }
        let start = Arc::new(tokio::sync::Barrier::new(wave_size + 1));
        for item in items.by_ref().take(wave_size) {
            let index = index.clone();
            let start = Arc::clone(&start);
            tasks.spawn(async move {
                start.wait().await;
                execute_item(&index, item).await
            });
        }
        start.wait().await;
        while let Some(result) = tasks.join_next().await {
            aggregate.push(result.map_err(|error| format!("workload task failed: {error}"))?);
        }
    }
    Ok(aggregate)
}

impl WorkloadObservation {
    /// Merges one completed task without treating failures as useful latency.
    fn push(&mut self, observation: OperationObservation) {
        let operation = self.operations.entry(observation.class).or_default();
        operation.attempted = operation.attempted.saturating_add(1);
        match observation.result {
            Err(error) => {
                *operation.errors.entry(format!("{error:?}")).or_default() += 1;
            }
            Ok(recall_input) => {
                operation.accepted = operation.accepted.saturating_add(1);
                operation.latency_ms.push(observation.latency_ms);
                if let Some(input) = recall_input {
                    self.recall_inputs.push(input);
                }
            }
        }
    }

    /// Counts all accepted public operations for foreground throughput.
    fn successful_operations(&self) -> u64 {
        self.operations
            .values()
            .map(|operation| operation.accepted)
            .sum()
    }

    /// Counts accepted calls for one public operation class.
    fn accepted_operations(&self, class: OperationClass) -> u64 {
        self.operations
            .get(&class)
            .map_or(0, |operation| operation.accepted)
    }

    /// Builds the stable per-class report schema from retained raw samples.
    fn into_operation_summaries(
        self,
        metrics: &CapturedMetrics,
    ) -> BTreeMap<OperationClass, OperationSummary> {
        self.operations
            .into_iter()
            .map(|(class, operation)| {
                let rejected = metrics.foreground_admission_rejections(class);
                (
                    class,
                    OperationSummary {
                        attempted: operation.attempted,
                        accepted: operation.accepted,
                        rejected,
                        errors: operation.errors,
                        latency_ms: Distribution::from_samples(operation.latency_ms),
                    },
                )
            })
            .collect()
    }

    /// Computes oracle overlap only after latency, wall, and CPU observations.
    fn recall_values(&self) -> Vec<f64> {
        self.recall_inputs
            .iter()
            .map(|input| {
                oracle::recall_ids(input.outcome.hits.iter().map(|hit| hit.id()), &input.truth)
            })
            .collect()
    }
}

/// Measures one public operation from immediately before its API call.
async fn execute_item<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    item: WorkItem,
) -> OperationObservation {
    let started = Instant::now();
    let (class, result) = match item {
        WorkItem::Search { request, truth } => (
            OperationClass::Search,
            index
                .search(request)
                .await
                .map(|outcome| truth.map(|truth| RecallInput { outcome, truth }))
                .map_err(|error| error.kind()),
        ),
        WorkItem::Upsert { record } => (
            OperationClass::Write,
            index
                .upsert(record)
                .await
                .map(|_| None)
                .map_err(|error| error.kind()),
        ),
    };
    OperationObservation {
        class,
        latency_ms: started.elapsed().as_secs_f64() * 1_000.0,
        result,
    }
}

/// Computes canonical exact L2 top-k truth before warmup and measurement.
fn exact_truth(dataset: &BenchmarkDataset, k: usize) -> Vec<ExactTruth> {
    dataset
        .queries
        .iter()
        .map(|query| {
            Arc::from(oracle::truth_vectors(
                &dataset.ids,
                &dataset.base,
                Metric::L2,
                query,
                k,
            ))
        })
        .collect()
}

/// Summarizes recall without turning a measured baseline into a pass/fail SLA.
fn recall_summary(recalls: &[f64]) -> Option<RecallSummary> {
    if recalls.is_empty() {
        return None;
    }
    Some(RecallSummary {
        queries: recalls.len() as u64,
        mean: recalls.iter().sum::<f64>() / recalls.len() as f64,
        min: recalls.iter().copied().fold(f64::INFINITY, f64::min),
    })
}

/// Converts the four public Search Budget dimensions without inventing work.
fn budget_summaries(metrics: &CapturedMetrics) -> BTreeMap<String, BudgetSummary> {
    [
        "scanned_tree_keys",
        "visited_partitions",
        "visited_leaf_entries",
        "exact_rerank_candidates",
    ]
    .into_iter()
    .map(|dimension| {
        let usage = metrics.histogram("ktann.search.budget.usage", &[("dimension", dimension)]);
        let exhausted_searches =
            metrics.counter("ktann.search.budget.exhausted", &[("dimension", dimension)]);
        (
            dimension.to_owned(),
            BudgetSummary {
                usage: Distribution::from_samples(usage),
                exhausted_searches,
            },
        )
    })
    .collect()
}

/// Captures host metadata with ordinary commands so the library stays portable.
fn environment(tokio_worker_threads: usize, backend_runtime: String) -> Environment {
    let logical_cpus = std::thread::available_parallelism().map_or(1, usize::from);
    let operating_system = command_output("uname", &["-a"]);
    let cpu_model = if cfg!(target_os = "macos") {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else {
        command_output(
            "sh",
            &[
                "-c",
                r"sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1",
            ],
        )
    };
    let memory_bytes = if cfg!(target_os = "macos") {
        command_output("sysctl", &["-n", "hw.memsize"]).parse().ok()
    } else {
        let kibibytes = command_output(
            "sh",
            &[
                "-c",
                r"sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo",
            ],
        )
        .parse::<u64>()
        .ok();
        kibibytes.and_then(|value| value.checked_mul(1_024))
    };
    Environment {
        operating_system,
        cpu_model,
        logical_cpus,
        memory_bytes,
        rustc: env!("KTANN_BENCH_BUILD_RUSTC").to_owned(),
        build_profile: env!("KTANN_BENCH_BUILD_PROFILE").to_owned(),
        build_features: build_features(),
        rustflags: env!("KTANN_BENCH_RUSTFLAGS").to_owned(),
        backend_runtime,
        tokio_worker_threads,
    }
}

/// Returns the deterministic additive feature identity of this executable.
fn build_features() -> String {
    let mut features = Vec::new();
    if cfg!(feature = "foundationdb") {
        features.push("foundationdb");
    }
    if cfg!(feature = "rocksdb") {
        features.push("rocksdb");
    }
    features.join(",")
}

/// Runs a metadata command and normalizes failure to an explicit marker.
fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

/// Identifies the measured commit and makes uncommitted inputs explicit.
fn git_revision() -> String {
    let revision = command_output("git", &["rev-parse", "HEAD"]);
    let dirty = Command::new("git")
        .args(["status", "--porcelain=v1", "--untracked-files=normal"])
        .output()
        .is_ok_and(|output| output.status.success() && !output.stdout.is_empty());
    if dirty {
        format!("{revision}-dirty")
    } else {
        revision
    }
}

/// Identifies a failed benchmark phase without exposing caller-derived data.
///
/// KTANN intentionally keeps public errors terse and privacy-safe. The phase
/// prefix restores enough operational context to diagnose a broken benchmark
/// while the suffix remains restricted to the stable public error category.
fn error_at(phase: &str, error: ktann::api::Error) -> String {
    format!("{phase}: {:?}", error.kind())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ktann::api::{SearchBudgets, SearchOptions};

    use crate::report::{Distribution, OperationClass, OperationSummary};

    use super::{
        CONVERGENCE_MAINTENANCE_WORKERS, is_search_operation, runtime_config, scenarios,
        search_budget_configuration, validate_admission_target,
    };

    fn operation_summary(attempted: u64, accepted: u64, rejected: u64) -> OperationSummary {
        let other_failures = attempted
            .checked_sub(accepted + rejected)
            .expect("outcomes fit attempts");
        let mut errors = BTreeMap::new();
        if rejected > 0 {
            errors.insert("LimitExceeded".to_owned(), rejected);
        }
        if other_failures > 0 {
            errors.insert("Backend".to_owned(), other_failures);
        }
        OperationSummary {
            attempted,
            accepted,
            rejected,
            errors,
            latency_ms: Distribution {
                count: accepted,
                ..Default::default()
            },
        }
    }

    #[test]
    fn every_scenario_has_a_jointly_valid_runtime_configuration() {
        for profile in ["smoke", "full"] {
            for scenario in scenarios(profile).expect("known profile") {
                for workers in [
                    scenario.maintenance_workers,
                    CONVERGENCE_MAINTENANCE_WORKERS,
                ] {
                    runtime_config(&scenario, workers).unwrap_or_else(|error| {
                        panic!("{} runtime configuration: {error}", scenario.name)
                    });
                }
            }
        }
    }

    #[test]
    fn measured_windows_preserve_each_declared_operation_mix() {
        for profile in ["smoke", "full"] {
            for scenario in scenarios(profile).expect("known profile") {
                let searches = (scenario.warmup_operations
                    ..scenario.warmup_operations + scenario.measured_operations)
                    .filter(|operation| is_search_operation(*operation, scenario.search_percent))
                    .count();
                assert_eq!(
                    searches * 100,
                    scenario.measured_operations * usize::from(scenario.search_percent),
                    "{} measured operation mix",
                    scenario.name
                );
            }
        }
    }

    #[test]
    fn admission_target_rejects_weak_samples() {
        let scenario = scenarios("full")
            .expect("known profile")
            .into_iter()
            .find(|scenario| scenario.name == "backend-admission-saturated")
            .expect("admission scenario");
        let mut operations = BTreeMap::from([
            (OperationClass::Search, operation_summary(200, 100, 100)),
            (OperationClass::Write, operation_summary(200, 100, 100)),
        ]);
        assert!(validate_admission_target(&operations, &scenario).is_ok());

        operations.insert(OperationClass::Search, operation_summary(200, 99, 100));
        assert!(validate_admission_target(&operations, &scenario).is_err());

        for summary in operations.values_mut() {
            *summary = operation_summary(500, 100, 400);
        }
        assert!(validate_admission_target(&operations, &scenario).is_err());
    }

    #[test]
    fn report_exposes_the_k_derived_effective_rerank_limit() {
        let budgets =
            search_budget_configuration(SearchBudgets::default(), SearchOptions::default(), 10)
                .expect("default search budgets resolve");

        let rerank = budgets.exact_rerank_candidates;
        assert_eq!(rerank.runtime_default, 65_536);
        assert_eq!(rerank.request_override, None);
        assert_eq!(rerank.effective_limit, 100);
    }

    #[test]
    fn report_distinguishes_a_request_budget_override() {
        let options = SearchOptions::default()
            .with_visited_partitions(64)
            .and_then(|options| options.with_exact_rerank_candidates(256))
            .expect("valid request overrides");
        let budgets = search_budget_configuration(SearchBudgets::default(), options, 10)
            .expect("overridden search budgets resolve");

        let partitions = budgets.visited_partitions;
        assert_eq!(partitions.runtime_default, 1_024);
        assert_eq!(partitions.request_override, Some(64));
        assert_eq!(partitions.effective_limit, 64);

        let rerank = budgets.exact_rerank_candidates;
        assert_eq!(rerank.runtime_default, 65_536);
        assert_eq!(rerank.request_override, Some(256));
        assert_eq!(rerank.effective_limit, 256);
    }
}
