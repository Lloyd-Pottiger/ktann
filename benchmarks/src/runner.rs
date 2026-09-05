//! Scenario definitions and the measured public-API execution path.

use std::collections::BTreeMap;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, ImportOptions, ImportSession, Index, IndexConfig,
    Metric, Mutation, OperationOptions, Record, RuntimeConfig, SearchBudgets, SearchOptions,
    SearchRequest, Value, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;

use crate::backend::{BackendCounters, MeasuredBackend};
use crate::dataset::{self, BenchmarkDataset};
use crate::metrics::{CapturedMetrics, MetricCapture};
use crate::report::{
    AdmissionTarget, BackendIo, BenchmarkReport, BudgetConfiguration, BudgetSummary, Configuration,
    ConvergencePhase, Distribution, Environment, ImportPhase, LifecycleMeasurements,
    MaintenanceSummary, OperationClass, OperationSummary, PartitionStateCounts, PhaseResources,
    QualityPoint, QualitySweepMeasurements, RecallSummary, ReportMeasurements,
    SearchBudgetConfiguration, SearchPhase, SearchTruncation, SteadyStateMeasurements, Topology,
    WorkloadDispatch, WriteAmplification, aggregate_rejection_rate,
};
use crate::resource::ResourceSnapshot;

#[path = "../../tests/support/oracle.rs"]
#[expect(
    dead_code,
    reason = "the shared oracle also exposes filter helpers used only by integration tests"
)]
mod oracle;

/// Queue capacity for demand-driven Structure Maintenance.
const FIXUP_QUEUE_CAPACITY: usize = 1_024;
/// Workers used after import diagnostics to converge persistent topology.
const CONVERGENCE_MAINTENANCE_WORKERS: usize = 2;

/// One immutable exact top-k result shared by repeated query operations.
type ExactTruth = Arc<[(Bytes, f64)]>;

/// A successful search retained until recall can run outside resource timing.
#[derive(Debug)]
struct RecallInput {
    /// Record IDs returned inside the measured API call.
    hit_ids: Vec<Bytes>,
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
    /// Query-window start used by bounded large diagnostic runs.
    pub query_offset: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Distance metric used by the Logical Index and supplied ground truth.
    pub metric: Metric,
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
    /// Per-level beam used while importing records into the tree.
    pub write_beam_size: u32,
    /// Ordered single-variable beam values; empty for ordinary scenarios.
    pub leaf_beam_sweep: Vec<u32>,
    /// Logical Index maximum partition size.
    pub max_partition_entries: u32,
    /// Whether this scenario measures the import-to-search lifecycle.
    pub lifecycle: bool,
    /// Records per Import Session batch in a lifecycle scenario.
    pub import_batch_size: usize,
    /// Explicit Import Session maximum accepted-batch ceiling.
    pub import_max_in_flight_batches: usize,
    /// Explicit Runtime Fixup backlog watermark for Import Session admission.
    pub import_backlog_watermark: usize,
    /// Background Structure Maintenance workers.
    pub maintenance_workers: usize,
}

/// Returns the complete deterministic scenario matrix for one scale profile.
///
/// # Errors
///
/// Returns an error when `profile` is unknown.
pub fn scenarios(profile: &str) -> Result<Vec<ScenarioSpec>, String> {
    match profile {
        "smoke" => Ok(smoke_scenarios()),
        "full" => Ok(full_scenarios()),
        "large" => large_scenarios(),
        _ => Err(format!(
            "unknown profile `{profile}`; expected smoke, full, or large"
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
        query_offset: 0,
        dimension: 16,
        metric: Metric::L2,
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
        write_beam_size: 8,
        leaf_beam_sweep: Vec::new(),
        max_partition_entries: 32,
        lifecycle: false,
        import_batch_size: 32,
        import_max_in_flight_batches: 2,
        import_backlog_watermark: 2,
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
        query_offset: 0,
        dimension,
        metric: Metric::L2,
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
        write_beam_size: 8,
        leaf_beam_sweep: Vec::new(),
        max_partition_entries: 128,
        lifecycle: false,
        import_batch_size: 50,
        import_max_in_flight_batches: 4,
        import_backlog_watermark: 2,
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
            // Adaptive admission starts at one and may probe up to four
            // concurrent batches after sustained conflict-free completions.
            import_max_in_flight_batches: 4,
            ..clustered.clone()
        },
    ]
}

/// Fixed public million-vector quality curves reserved for optimized workers.
fn large_scenarios() -> Result<Vec<ScenarioSpec>, String> {
    let search_options = SearchOptions::default()
        .with_scanned_tree_keys(1)
        .and_then(|options| options.with_visited_partitions(16_384))
        .and_then(|options| options.with_visited_leaf_entries(1_048_576))
        .map_err(|error| error_at("configure large search", error))?;
    let scenario = |name, dataset, dimension, metric| ScenarioSpec {
        name,
        dataset,
        profile: "large",
        base_vectors: 1_000_000,
        query_vectors: 1_000,
        query_offset: 0,
        dimension,
        metric,
        seed: 0x38_2001,
        search_percent: 100,
        hot_updates: false,
        partition_cache_bytes: 512 << 20,
        foreground_limit: 64,
        blocking_resource_limit: Some(64),
        concurrency: 16,
        dispatch: WorkloadDispatch::Continuous,
        warmup_operations: 1_000,
        measured_operations: 1_000,
        k: 10,
        search_options,
        write_beam_size: 8,
        leaf_beam_sweep: vec![1, 4, 8, 16, 32],
        // Keep the shared leaf/internal fanout below sqrt(1M) so the
        // million-vector corpus must form at least three searchable levels.
        max_partition_entries: 128,
        lifecycle: false,
        import_batch_size: 50,
        import_max_in_flight_batches: 4,
        import_backlog_watermark: 2,
        maintenance_workers: 2,
    };
    Ok(vec![
        scenario("quality-cohere-1m", "cohere-1m", 768, Metric::Cosine),
        scenario("quality-sift-1m", "sift-1m", 128, Metric::L2),
    ])
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
    let dataset_started = phase_started(spec, "dataset load");
    let mut dataset = if spec.profile == "large" {
        dataset::load_large(spec.dataset)?
    } else {
        dataset::load(
            spec.dataset,
            spec.base_vectors,
            spec.query_vectors,
            spec.dimension,
            spec.seed,
        )?
    };
    if spec.profile == "large"
        && (spec.base_vectors != dataset.base.len()
            || spec.query_offset != 0
            || spec.query_vectors != dataset.queries.len())
    {
        dataset = dataset::limit_with_query_offset(
            dataset,
            spec.base_vectors,
            spec.query_offset,
            spec.query_vectors,
        )?;
    }
    phase_completed(spec, "dataset load", dataset_started);
    let (backend, backend_counters) = MeasuredBackend::new(backend);
    let admission = backend.admission_budget();
    let metric_capture = MetricCapture::install()?;
    let primary_runtime_config = runtime_config(spec, spec.maintenance_workers)?;
    let index_config = index_config(spec)?;
    let default_search_budgets = primary_runtime_config.default_search_budgets();
    let search_budgets =
        search_budget_configuration(default_search_budgets, spec.search_options, spec.k)?;
    let (topology, measurements) = if !spec.leaf_beam_sweep.is_empty() {
        let runtime = Runtime::new(backend, primary_runtime_config)
            .map_err(|error| error_at("create runtime", error))?;
        let measured = run_quality_sweep(
            &runtime,
            &backend_counters,
            &metric_capture,
            spec,
            &mut dataset,
        )
        .await;
        let shutdown = runtime
            .shutdown()
            .await
            .map_err(|error| error_at("shut down runtime", error));
        let (topology, measurements) = measured?;
        shutdown?;
        (topology, ReportMeasurements::QualitySweep(measurements))
    } else if spec.lifecycle {
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
            dimension: index_config.dimension(),
            metric: metric_name(index_config.metric()).to_owned(),
            index_field_count: index_config.fields().len(),
            tree_key_field_count: index_config.tree_key_fields().len(),
            search_percent: spec.search_percent,
            hot_updates: spec.hot_updates,
            min_partition_entries: index_config.min_partition_entries(),
            max_partition_entries: index_config.max_partition_entries(),
            partition_cache_bytes: spec.partition_cache_bytes,
            foreground_limit: spec.foreground_limit,
            maintenance_workers: spec.maintenance_workers,
            convergence_maintenance_workers: spec
                .lifecycle
                .then_some(CONVERGENCE_MAINTENANCE_WORKERS),
            fixup_queue_capacity: FIXUP_QUEUE_CAPACITY,
            mutation_attempt_limit: 32,
            maintenance_attempt_limit: 32,
            search_budgets,
            write_beam_size: spec.write_beam_size,
            leaf_beam_size_override: spec.search_options.leaf_beam_size(),
            leaf_beam_sweep: spec.leaf_beam_sweep.clone(),
            blocking_resource_limit: spec.blocking_resource_limit,
            backend_max_mutations: admission.max_mutations,
            backend_max_mutation_bytes: admission.max_mutation_bytes,
            backend_mutation_key_overhead_bytes: admission.mutation_key_overhead_bytes,
            concurrency: spec.concurrency,
            dispatch: spec.dispatch,
            warmup_operations: spec.warmup_operations,
            measured_operations: spec.measured_operations,
            k: spec.k,
            import_batch_size: (spec.lifecycle || spec.profile == "large")
                .then_some(spec.import_batch_size),
            import_max_in_flight_batches: (spec.lifecycle || spec.profile == "large")
                .then_some(spec.import_max_in_flight_batches),
            import_backlog_watermark: (spec.lifecycle || spec.profile == "large")
                .then_some(spec.import_backlog_watermark),
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
        .and_then(|config| config.with_partition_cache_bytes(spec.partition_cache_bytes))
        .and_then(|config| config.with_write_beam_size(spec.write_beam_size));
    let config = if spec.lifecycle || spec.profile == "large" {
        config.and_then(|config| {
            config.with_import_limits(
                spec.import_max_in_flight_batches,
                spec.import_backlog_watermark,
            )
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
            request_override: None,
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
    let truths = exact_truth(dataset, spec.metric, spec.k);
    let requests = lifecycle_requests(dataset, spec)?;
    let import_options = ImportOptions::default()
        .with_max_in_flight_batches(spec.import_max_in_flight_batches)
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
    backend_io: BackendIo,
}

/// Returns the stable public name of a configured distance metric.
const fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cosine",
        Metric::InnerProduct => "inner_product",
        _ => "unknown",
    }
}

/// Builds the public Index configuration shared by steady and lifecycle cases.
fn index_config(spec: &ScenarioSpec) -> Result<IndexConfig, String> {
    let config = IndexConfig::new(spec.dimension, spec.metric).and_then(|config| {
        config.with_partition_entries(spec.max_partition_entries / 4, spec.max_partition_entries)
    });
    let config = if spec.profile == "large" {
        config
    } else {
        config
            .and_then(|config| {
                config.with_fields(vec![
                    FieldSchema::new("bucket", DataType::I64)?,
                    FieldSchema::new("generation", DataType::I64)?,
                ])
            })
            .and_then(|config| config.with_tree_key_fields(vec![FieldId(0)]))
    };
    config.map_err(|error| error_at("configure index", error))
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
            let start = batch
                .checked_mul(batch_size)
                .ok_or_else(|| "import record ordinal overflow".to_owned())?;
            mutation_batch(
                dataset,
                vectors,
                start,
                &[Value::I64(0), Value::I64(0)],
                phase,
            )
        })
        .collect()
}

/// Builds one bounded batch of insert mutations from aligned dataset rows.
fn mutation_batch(
    dataset: &BenchmarkDataset,
    vectors: &[Arc<[f32]>],
    start: usize,
    fields: &[Value],
    phase: &str,
) -> Result<Vec<Mutation>, String> {
    vectors
        .iter()
        .enumerate()
        .map(|(offset, vector)| {
            let ordinal = start
                .checked_add(offset)
                .ok_or_else(|| "load record ordinal overflow".to_owned())?;
            Record::new(
                dataset.ids[ordinal].clone(),
                Arc::clone(vector),
                fields.to_vec(),
            )
            .map(Mutation::Insert)
            .map_err(|error| error_at(phase, error))
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
    if batches.is_empty() {
        return Err("lifecycle import has no batches".to_owned());
    }
    let resources_before = ResourceSnapshot::capture()?;
    let backend_before = backend_counters.snapshot();
    let started = Instant::now();
    let mut submitted_batch_sizes = Vec::with_capacity(batches.len());
    let mut submit_latency_ms = Vec::with_capacity(batches.len());
    let mut failures = BTreeMap::new();
    for batch in batches {
        let records = batch.len();
        let submit_started = Instant::now();
        match session.submit(batch).await {
            Ok(_) => submitted_batch_sizes.push(records),
            Err(error) => {
                *failures.entry(format!("{:?}", error.kind())).or_default() += 1;
            }
        }
        submit_latency_ms.push(submit_started.elapsed().as_secs_f64() * 1_000.0);
    }
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
    backend_after: BackendIo,
}

/// Builds common resource and maintenance accounting at a phase boundary.
fn phase_resources(
    wall_seconds: f64,
    before: ResourceSnapshot,
    after: ResourceSnapshot,
    backend_io: BackendIo,
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

/// Owns setup, one steady measurement, and the post-run invariant audit.
async fn run_with_runtime<B: Backend>(
    runtime: &Runtime<MeasuredBackend<B>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &BenchmarkDataset,
) -> Result<(Topology, SteadyStateMeasurements), String> {
    let (index, topology) = prepare_index(runtime, metric_capture, spec, dataset).await?;
    let measurements = measure_steady_workload(
        &index,
        backend_counters,
        metric_capture,
        spec,
        dataset,
        None,
    )
    .await?;
    verify_measured_state(&index, spec).await?;
    Ok((topology, measurements))
}

/// Measures an ordered beam curve over one loaded and converged Logical Index.
async fn run_quality_sweep<B: Backend>(
    runtime: &Runtime<MeasuredBackend<B>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &mut BenchmarkDataset,
) -> Result<(Topology, QualitySweepMeasurements), String> {
    let (index, topology) = prepare_index(runtime, metric_capture, spec, dataset).await?;
    if topology.max_level.is_none_or(|level| level < 3) {
        return Err(format!(
            "large quality topology has max level {:?} with {} partitions by level {:?}; expected at least three searchable levels",
            topology.max_level, topology.partitions, topology.partitions_by_level
        ));
    }
    // Compute truth while the imported vectors are still available. Bounded
    // diagnostic datasets deliberately discard supplied full-corpus truth, so
    // this also preserves exact truth for the selected base/query subset.
    let truth = exact_truth(dataset, spec.metric, spec.k);
    // Release the imported million-vector corpus before measuring search.
    dataset.ids = Vec::new();
    dataset.base = Vec::new();
    let mut points = Vec::with_capacity(spec.leaf_beam_sweep.len());
    for beam in &spec.leaf_beam_sweep {
        let mut point = spec.clone();
        point.search_options = point
            .search_options
            .with_leaf_beam_size(*beam)
            .map_err(|error| error_at("configure quality point", error))?;
        let mut measurements = measure_steady_workload(
            &index,
            backend_counters,
            metric_capture,
            &point,
            dataset,
            Some(&truth),
        )
        .await?;
        // getrusage exposes only the process-lifetime high-water mark, which
        // setup already established and cannot attribute to one beam point.
        measurements.peak_rss_bytes = None;
        points.push(QualityPoint {
            leaf_beam_size: *beam,
            measurements,
        });
    }
    verify_measured_state(&index, spec).await?;
    validate_quality_frontier(&points, spec.measured_operations)?;
    Ok((topology, QualitySweepMeasurements { points }))
}

/// Creates, loads, and converges one fresh Logical Index outside measurement.
async fn prepare_index<B: Backend>(
    runtime: &Runtime<MeasuredBackend<B>>,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &BenchmarkDataset,
) -> Result<(Index<MeasuredBackend<B>>, Topology), String> {
    let index_config = index_config(spec)?;
    let index = runtime
        .create_index("benchmark", index_config)
        .await
        .map_err(|error| error_at("create index", error))?;

    // Setup is deliberately complete before any timer or counter baseline:
    // load transactions, split training, verification, oracle construction,
    // and cache warmup therefore cannot make measured operations look slower.
    let import_started = phase_started(spec, "import");
    load_index(&index, dataset, spec).await?;
    phase_completed(spec, "import", import_started);
    let deadline = Instant::now() + settle_timeout(spec);
    let (topology, _) =
        settle_and_drain_topology(&index, dataset, spec, metric_capture, deadline).await?;
    Ok((index, topology))
}

/// Measures one fixed workload after setup and before the invariant audit.
async fn measure_steady_workload<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &BenchmarkDataset,
    supplied_truth: Option<&[ExactTruth]>,
) -> Result<SteadyStateMeasurements, String> {
    let computed_truth = (supplied_truth.is_none() && spec.search_percent == 100)
        .then(|| exact_truth(dataset, spec.metric, spec.k));
    let truth = supplied_truth.or(computed_truth.as_deref());
    let warmup = work_items(dataset, None, spec, spec.warmup_operations, 0)?;
    let point = spec
        .search_options
        .leaf_beam_size()
        .map_or_else(|| "workload".to_owned(), |beam| format!("beam {beam}"));
    let warmup_phase = format!("{point} warmup");
    let warmup_started = phase_started(spec, &warmup_phase);
    let _warmup_result =
        execute_items(index.clone(), warmup, spec.concurrency, spec.dispatch).await?;
    phase_completed(spec, &warmup_phase, warmup_started);

    // Reset setup counters and histograms before the measured interval. Gauges
    // retain current process state and need no compatibility fallback.
    let _ = metric_capture.snapshot();
    wait_for_maintenance(metric_capture, settle_timeout(spec)).await?;
    let backend_before = backend_counters.snapshot();
    let measured_items = work_items(
        dataset,
        truth,
        spec,
        spec.measured_operations,
        spec.warmup_operations,
    )?;

    let resources_before = ResourceSnapshot::capture()?;
    let measured_phase = format!("{point} measurement");
    let measured_started = phase_started(spec, &measured_phase);
    let wall_started = Instant::now();
    let workload = execute_items(
        index.clone(),
        measured_items,
        spec.concurrency,
        spec.dispatch,
    )
    .await?;
    let wall_seconds = wall_started.elapsed().as_secs_f64();
    phase_completed(spec, &measured_phase, measured_started);
    let metrics = metric_capture.snapshot();
    let maintenance_started = Instant::now();
    wait_for_maintenance(metric_capture, settle_timeout(spec)).await?;
    let maintenance_drain_seconds = maintenance_started.elapsed().as_secs_f64();
    let resources_after = ResourceSnapshot::capture()?;
    let backend_io = backend_counters.since(&backend_before);
    let recalls = workload.recall_values();

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
    Ok(measurements)
}

/// Protects reports from measurements produced by corrupt persistent state.
async fn verify_measured_state<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    spec: &ScenarioSpec,
) -> Result<(), String> {
    let started = phase_started(spec, "final verification");
    let report = index
        .verify(verify_options(spec, None)?)
        .await
        .map_err(|error| error_at("verify measured state", error))?;
    if !report.complete || !report.issues.is_empty() {
        return Err(format!(
            "post-run verification failed: complete={}, issues={}",
            report.complete,
            report.issues.len()
        ));
    }
    phase_completed(spec, "final verification", started);
    Ok(())
}

/// Requires both quality and work to move across the declared beam curve.
fn validate_quality_frontier(
    points: &[QualityPoint],
    expected_searches: usize,
) -> Result<(), String> {
    let expected_searches = u64::try_from(expected_searches)
        .map_err(|_| "large quality search count exceeds u64".to_owned())?;
    let mut recalls = Vec::with_capacity(points.len());
    let mut leaf_work = Vec::with_capacity(points.len());
    for point in points {
        let search = point
            .measurements
            .operations
            .get(&OperationClass::Search)
            .ok_or_else(|| format!("leaf beam {} has no search results", point.leaf_beam_size))?;
        let recall =
            point.measurements.recall_at_k.as_ref().ok_or_else(|| {
                format!("leaf beam {} has no recall results", point.leaf_beam_size)
            })?;
        let leaf_budget = point
            .measurements
            .search_budgets
            .get("visited_leaf_entries")
            .ok_or_else(|| {
                format!(
                    "leaf beam {} has no Leaf Entry budget results",
                    point.leaf_beam_size
                )
            })?;
        if search.attempted != expected_searches
            || search.accepted != expected_searches
            || recall.queries != expected_searches
            || leaf_budget.usage.count != expected_searches
        {
            return Err(format!(
                "leaf beam {} completed {}/{} searches with {} recall and {} Leaf Entry samples; expected {expected_searches}",
                point.leaf_beam_size,
                search.accepted,
                search.attempted,
                recall.queries,
                leaf_budget.usage.count,
            ));
        }
        recalls.push(recall.mean);
        leaf_work.push(leaf_budget.usage.mean);
    }
    let recall_moves = recalls
        .iter()
        .copied()
        .reduce(f64::min)
        .zip(recalls.iter().copied().reduce(f64::max))
        .is_some_and(|(minimum, maximum)| maximum > minimum);
    let work_moves = leaf_work.windows(2).any(|window| window[1] > window[0]);
    if !recall_moves || !work_moves {
        return Err(
            "large quality sweep did not produce a nontrivial quality/work frontier".to_owned(),
        );
    }
    Ok(())
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
    let rejection_rate = aggregate_rejection_rate(operations);
    if rejection_rate < target.minimum_rejection_rate
        || rejection_rate > target.maximum_rejection_rate
    {
        let attempted: u64 = operations.values().map(|summary| summary.attempted).sum();
        let rejected: u64 = operations.values().map(|summary| summary.rejected).sum();
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
    spec: &ScenarioSpec,
) -> Result<(), String> {
    let mut import = if spec.profile == "large" {
        Some(
            index
                .import_session(
                    ImportOptions::default()
                        .with_max_in_flight_batches(spec.import_max_in_flight_batches)
                        .map_err(|error| error_at("configure load import", error))?,
                )
                .map_err(|error| error_at("open load import", error))?,
        )
    } else {
        None
    };
    let batch_size = if spec.profile == "large" {
        spec.import_batch_size
    } else {
        50
    };
    let fields = if spec.profile == "large" {
        Vec::new()
    } else {
        vec![Value::I64(0), Value::I64(0)]
    };
    for (batch, vectors) in dataset.base.chunks(batch_size).enumerate() {
        let start = batch
            .checked_mul(batch_size)
            .ok_or_else(|| "load record ordinal overflow".to_owned())?;
        let mutations = mutation_batch(dataset, vectors, start, &fields, "construct load record")?;
        if let Some(import) = import.as_mut() {
            import
                .submit(mutations)
                .await
                .map_err(|error| error_at("submit load records", error))?;
            continue;
        }
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
    if let Some(import) = import {
        for result in import.finish().await {
            result
                .result
                .map_err(|error| error_at("load records", error))?;
        }
    }
    Ok(())
}

/// Uses public searches to rediscover maintenance work after an unsettled audit.
async fn rediscover_topology_work<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
    deadline: Instant,
) -> Result<(), String> {
    // A complete large query set gives demand-driven maintenance representative
    // tree coverage. Ordinary profiles retain a bounded four-query probe.
    let query_limit = if spec.profile == "large" {
        dataset.queries.len()
    } else {
        4
    };
    for query in dataset.queries.iter().take(query_limit) {
        let request = SearchRequest::new(Arc::clone(query), spec.k)
            .map(|request| request.with_options(spec.search_options))
            .map_err(|error| error_at("construct topology-settling search", error))?;
        index
            .search_with_control(request, OperationOptions::default().with_deadline(deadline))
            .await
            .map_err(|error| error_at("settle topology", error))?;
    }
    Ok(())
}

/// Returns whether aggregate topology could be a fully settled search tree.
fn topology_is_settled(topology: &Topology, max_partition_entries: u32) -> bool {
    let leaf_entries = topology.entries_by_level.get(&1).copied().unwrap_or(0);
    let largest_leaf = topology.max_entries_by_level.get(&1).copied().unwrap_or(0);
    topology.partitions > 1
        && topology.entries > topology.vector_records
        && leaf_entries == topology.vector_records
        && largest_leaf <= max_partition_entries
        && topology.actionable_partitions == 0
        && topology.partition_states.transitional() == 0
}

/// Records a topology audit and returns consecutive unchanged rediscovery rounds.
fn observe_topology_progress(
    topology: &Topology,
    previous_topology: &mut Option<Topology>,
    unchanged_rounds: &mut u32,
) -> u32 {
    if previous_topology.as_ref() == Some(topology) {
        *unchanged_rounds = unchanged_rounds.saturating_add(1);
    } else {
        *unchanged_rounds = 0;
    }
    *previous_topology = Some(topology.clone());
    *unchanged_rounds
}

/// Drains scheduled work and accepts the first complete settled topology audit.
async fn settle_and_drain_topology<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
    metric_capture: &MetricCapture,
    deadline: Instant,
) -> Result<(Topology, f64), String> {
    const MAX_UNCHANGED_ROUNDS: u32 = 2;

    let mut maintenance_drain_seconds = 0.0;
    let mut round = 1_u32;
    let mut previous_topology = None;
    let mut unchanged_rounds = 0_u32;
    loop {
        let drain_phase = format!("maintenance drain {round}");
        let drain_started = Instant::now();
        phase_announced(spec, &drain_phase);
        wait_for_maintenance_until(metric_capture, deadline).await?;
        let drain_elapsed = drain_started.elapsed();
        maintenance_drain_seconds += drain_elapsed.as_secs_f64();
        phase_completed(spec, &drain_phase, drain_started);

        let verify_phase = format!("topology verification {round}");
        let verify_started = phase_started(spec, &verify_phase);
        let topology = verified_topology(index, spec, "verify topology", deadline).await?;
        phase_completed(spec, &verify_phase, verify_started);
        eprintln!(
            "[{}] topology: records={}, partitions={}, actionable={}, transitional={}, max_entries_by_level={:?}",
            spec.name,
            topology.vector_records,
            topology.partitions,
            topology.actionable_partitions,
            topology.partition_states.transitional(),
            topology.max_entries_by_level,
        );
        if topology_is_settled(&topology, spec.max_partition_entries) {
            return Ok((topology, maintenance_drain_seconds));
        }
        if observe_topology_progress(&topology, &mut previous_topology, &mut unchanged_rounds)
            >= MAX_UNCHANGED_ROUNDS
        {
            return Err(format!(
                "topology made no observable progress across {unchanged_rounds} rediscovery rounds: \
                 records={}, partitions={}, entries={}, actionable={}, transitional={}, \
                 max_entries_by_level={:?}",
                topology.vector_records,
                topology.partitions,
                topology.entries,
                topology.actionable_partitions,
                topology.partition_states.transitional(),
                topology.max_entries_by_level,
            ));
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "topology did not settle: records={}, partitions={}, entries={}, \
                 actionable={}, transitional={}, max_entries_by_level={:?}",
                topology.vector_records,
                topology.partitions,
                topology.entries,
                topology.actionable_partitions,
                topology.partition_states.transitional(),
                topology.max_entries_by_level,
            ));
        }
        let rediscovery_phase = format!("maintenance rediscovery {round}");
        let rediscovery_started = phase_started(spec, &rediscovery_phase);
        rediscover_topology_work(index, dataset, spec, deadline).await?;
        phase_completed(spec, &rediscovery_phase, rediscovery_started);
        round = round.saturating_add(1);
    }
}

/// Reports a benchmark phase boundary without changing its measured interval.
fn phase_started(spec: &ScenarioSpec, phase: &str) -> Instant {
    phase_announced(spec, phase);
    Instant::now()
}

/// Reports a benchmark phase start to the worker's inherited stderr.
fn phase_announced(spec: &ScenarioSpec, phase: &str) {
    eprintln!("[{}] {phase} started", spec.name);
}

/// Reports elapsed wall time for one completed benchmark phase.
fn phase_completed(spec: &ScenarioSpec, phase: &str, started: Instant) {
    eprintln!(
        "[{}] {phase} completed in {:.3}s",
        spec.name,
        started.elapsed().as_secs_f64()
    );
}

/// Verifies persistent state and returns its observable topology counts.
async fn verified_topology<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    spec: &ScenarioSpec,
    phase: &str,
    deadline: Instant,
) -> Result<Topology, String> {
    let report = index
        .verify(verify_options(spec, Some(deadline))?)
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
        partitions: report.topology.partitions,
        entries: report.objects.entries,
        trees: report.topology.trees,
        max_level: report.topology.max_level,
        partitions_by_level: report.topology.partitions_by_level,
        entries_by_level: report.topology.entries_by_level,
        max_entries_by_level: report.topology.max_entries_by_level,
        partition_states: PartitionStateCounts {
            ready: report.topology.partition_states.ready,
            splitting: report.topology.partition_states.splitting,
            receiving_split: report.topology.partition_states.receiving_split,
            draining_split: report.topology.partition_states.draining_split,
            merging: report.topology.partition_states.merging,
        },
        actionable_partitions: report.topology.actionable_partitions,
    })
}

/// Applies larger verification bounds only to the declared million-vector profile.
fn verify_options(spec: &ScenarioSpec, deadline: Option<Instant>) -> Result<VerifyOptions, String> {
    let options = if spec.profile == "large" {
        VerifyOptions::default()
            .with_object_limit(10_000_000)
            .and_then(|options| options.with_memory_limit_bytes(1 << 30))
            .map_err(|error| error_at("configure verification", error))?
    } else {
        VerifyOptions::default()
    };
    Ok(match deadline {
        Some(deadline) => {
            options.with_operation_options(OperationOptions::default().with_deadline(deadline))
        }
        None => options,
    })
}

/// Returns the bounded setup and maintenance-drain allowance for a profile.
fn settle_timeout(spec: &ScenarioSpec) -> Duration {
    match spec.profile {
        "smoke" => Duration::from_secs(30),
        "large" => Duration::from_secs(6 * 60 * 60),
        _ => Duration::from_secs(120),
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
            .map(|input| oracle::recall_ids(input.hit_ids.iter(), &input.truth))
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
                .map(|outcome| {
                    truth.map(|truth| RecallInput {
                        hit_ids: outcome.hits.iter().map(|hit| hit.id().clone()).collect(),
                        truth,
                    })
                })
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

/// Computes canonical exact top-k truth before warmup and measurement.
fn exact_truth(dataset: &BenchmarkDataset, metric: Metric, k: usize) -> Vec<ExactTruth> {
    if let Some(truth) = &dataset.ground_truth {
        return truth
            .iter()
            .map(|neighbors| {
                neighbors
                    .iter()
                    .take(k)
                    .cloned()
                    .map(|id| (id, 0.0))
                    .collect()
            })
            .collect();
    }
    dataset
        .queries
        .iter()
        .map(|query| {
            Arc::from(oracle::truth_vectors(
                &dataset.ids,
                &dataset.base,
                metric,
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

    use crate::report::{
        BudgetSummary, Distribution, OperationClass, OperationSummary, QualityPoint, RecallSummary,
        SteadyStateMeasurements, Topology,
    };

    use super::{
        CONVERGENCE_MAINTENANCE_WORKERS, is_search_operation, observe_topology_progress,
        runtime_config, scenarios, search_budget_configuration, topology_is_settled,
        validate_admission_target, validate_quality_frontier,
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

    fn quality_point(beam: u32, searches: u64, recall: f64, leaf_work: f64) -> QualityPoint {
        let mut measurements = SteadyStateMeasurements {
            recall_at_k: Some(RecallSummary {
                queries: searches,
                mean: recall,
                min: recall,
            }),
            ..Default::default()
        };
        measurements.operations.insert(
            OperationClass::Search,
            operation_summary(searches, searches, 0),
        );
        measurements.search_budgets.insert(
            "visited_leaf_entries".to_owned(),
            BudgetSummary {
                usage: Distribution {
                    count: searches,
                    mean: leaf_work,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        QualityPoint {
            leaf_beam_size: beam,
            measurements,
        }
    }

    #[test]
    fn every_scenario_has_a_jointly_valid_runtime_configuration() {
        for profile in ["smoke", "full", "large"] {
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
        for profile in ["smoke", "full", "large"] {
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
        assert_eq!(rerank.effective_limit, 64);
    }

    #[test]
    fn report_distinguishes_a_request_budget_override() {
        let options = SearchOptions::default()
            .with_visited_partitions(64)
            .expect("valid request override");
        let budgets = search_budget_configuration(SearchBudgets::default(), options, 10)
            .expect("overridden search budgets resolve");

        let partitions = budgets.visited_partitions;
        assert_eq!(partitions.runtime_default, 1_024);
        assert_eq!(partitions.request_override, Some(64));
        assert_eq!(partitions.effective_limit, 64);
    }

    #[test]
    fn large_profile_is_an_explicit_single_variable_beam_sweep() {
        for scenario in scenarios("large").expect("large profile") {
            assert_eq!(scenario.leaf_beam_sweep, [1, 4, 8, 16, 32]);
            assert_eq!(scenario.search_options.scanned_tree_keys(), Some(1));
            assert_eq!(scenario.search_options.visited_partitions(), Some(16_384));
            assert_eq!(
                scenario.search_options.visited_leaf_entries(),
                Some(1_048_576)
            );
            assert_eq!(scenario.search_options.leaf_beam_size(), None);
            let budgets = scenario
                .search_options
                .resolve(SearchBudgets::default(), scenario.k)
                .expect("large search budgets resolve");
            assert_eq!(budgets.exact_rerank_candidates(), 64);
            let max_partition_entries = usize::try_from(scenario.max_partition_entries)
                .expect("partition fanout fits usize");
            assert!(
                max_partition_entries * max_partition_entries < scenario.base_vectors,
                "the shared partition fanout must force a third topology level"
            );
        }
    }

    #[test]
    fn quality_frontier_requires_complete_results_for_every_beam() {
        let mut points = vec![
            quality_point(1, 2, 0.5, 10.0),
            quality_point(2, 2, 1.0, 20.0),
        ];
        assert!(validate_quality_frontier(&points, 2).is_ok());

        points[0].measurements.recall_at_k = None;
        assert!(validate_quality_frontier(&points, 2).is_err());
    }

    #[test]
    fn topology_settlement_rejects_insufficient_leaf_capacity() {
        let topology = Topology {
            vector_records: 1_000_000,
            partitions: 1_089,
            entries: 1_001_088,
            trees: 1,
            max_level: Some(3),
            partitions_by_level: BTreeMap::from([(1, 1_085), (2, 3), (3, 1)]),
            entries_by_level: BTreeMap::from([(1, 1_000_000), (2, 1_085), (3, 3)]),
            max_entries_by_level: BTreeMap::from([(1, 1_842), (2, 512), (3, 3)]),
            partition_states: Default::default(),
            actionable_partitions: 0,
        };

        assert!(topology_is_settled(&topology, 1_842));
        assert!(!topology_is_settled(&topology, 512));
    }

    #[test]
    fn topology_progress_counts_completed_unchanged_rediscoveries() {
        let first = Topology {
            vector_records: 1_000_000,
            partitions: 1_000,
            ..Default::default()
        };
        let mut previous = None;
        let mut unchanged_rounds = 0;

        assert_eq!(
            observe_topology_progress(&first, &mut previous, &mut unchanged_rounds),
            0
        );
        assert_eq!(
            observe_topology_progress(&first, &mut previous, &mut unchanged_rounds),
            1
        );

        let progressed = Topology {
            partitions: 1_001,
            ..first
        };
        assert_eq!(
            observe_topology_progress(&progressed, &mut previous, &mut unchanged_rounds),
            0
        );
        assert_eq!(
            observe_topology_progress(&progressed, &mut previous, &mut unchanged_rounds),
            1
        );
        assert_eq!(
            observe_topology_progress(&progressed, &mut previous, &mut unchanged_rounds),
            2
        );
    }
}
