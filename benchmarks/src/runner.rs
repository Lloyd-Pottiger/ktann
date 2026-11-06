//! Benchmark orchestration, shared phase accounting, and report metadata.

use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;

use crate::backend::{BackendCounters, MeasuredBackend};
use crate::dataset;
use crate::dataset::BenchmarkDataset;
use crate::metrics::{CapturedMetrics, MetricCapture};
use crate::report::{
    BackendIo, BenchmarkReport, Configuration, Environment, MaintenanceSummary, PhaseResources,
    ReportMeasurements, SteadyStateMeasurements, Topology,
};
use crate::resource::ResourceSnapshot;
use crate::runner::construction::{prepare_index, verify_measured_state};
use crate::runner::lifecycle::run_lifecycle_case;
use crate::runner::quality::run_quality_sweep;
use crate::runner::workload::measure_steady_workload;
use crate::scenario::{
    CONVERGENCE_MAINTENANCE_WORKERS, ScenarioSpec, index_config, metric_name, runtime_config,
    search_budget_configuration,
};

mod construction;
mod lifecycle;
mod quality;
mod workload;

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
            spec.query_vectors
                .checked_add(spec.query_offset)
                .ok_or("query window overflow")?,
            spec.dimension,
            spec.seed,
        )?
    };
    if spec.base_vectors != dataset.base.len()
        || spec.query_offset != 0
        || spec.query_vectors != dataset.queries.len()
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
    let fixup_queue_capacity = primary_runtime_config.fixup_queue_capacity();
    let mutation_attempt_limit = primary_runtime_config.foreground_attempts();
    let maintenance_attempt_limit = primary_runtime_config.fixup_attempts();
    let index_config = index_config(spec)?;
    let default_search_budgets = primary_runtime_config.default_search_budgets();
    let search_budgets =
        search_budget_configuration(default_search_budgets, spec.search_options, spec.k)?;
    let (topology, measurements) = if spec.lifecycle && spec.leaf_beam_sweep.is_empty() {
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
        let measured = if !spec.leaf_beam_sweep.is_empty() {
            run_quality_sweep(
                &runtime,
                &backend_counters,
                &metric_capture,
                spec,
                &mut dataset,
            )
            .await
            .map(|(topology, measurements)| {
                (
                    topology,
                    ReportMeasurements::QualitySweep(Box::new(measurements)),
                )
            })
        } else {
            run_steady_state(&runtime, &backend_counters, &metric_capture, spec, &dataset)
                .await
                .map(|(topology, measurements)| {
                    (
                        topology,
                        ReportMeasurements::SteadyState(Box::new(measurements)),
                    )
                })
        };
        // Always shut down, retaining the measurement error if both operations fail.
        let shutdown = runtime
            .shutdown()
            .await
            .map_err(|error| error_at("shut down runtime", error));
        let result = measured?;
        shutdown?;
        result
    };

    let report = BenchmarkReport {
        generated_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        reproduction_command,
        git_revision: git_revision(),
        environment: environment(tokio_worker_threads, backend_runtime),
        configuration: Configuration {
            routing_experiment: spec.routing_experiment.clone(),
            query_offset: spec.query_offset,
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
            min_leaf_partition_entries: index_config.min_leaf_partition_entries(),
            max_leaf_partition_entries: index_config.max_leaf_partition_entries(),
            min_internal_partition_entries: index_config.min_internal_partition_entries(),
            max_internal_partition_entries: index_config.max_internal_partition_entries(),
            partition_cache_bytes: spec.partition_cache_bytes,
            foreground_limit: spec.foreground_limit,
            maintenance_workers: spec.maintenance_workers,
            convergence_maintenance_workers: spec
                .lifecycle
                .then_some(CONVERGENCE_MAINTENANCE_WORKERS),
            fixup_queue_capacity,
            mutation_attempt_limit,
            maintenance_attempt_limit,
            search_budgets,
            write_beam_size: spec.write_beam_size,
            refinement_rounds: spec.refinement_rounds,
            refinement_neighbor_centroids: spec.refinement_rounds.map(|_| 32),
            refinement_input_limit_bytes: spec.refinement_rounds.map(|_| {
                spec.routing_experiment
                    .as_ref()
                    .map_or(32 << 30, |e| e.build_memory_bytes as u64)
            }),
            leaf_beam_size_override: spec.search_options.leaf_beam_size(),
            leaf_beam_sweep: spec.leaf_beam_sweep.clone(),
            query_concurrency_sweep: spec.query_concurrency_sweep.clone(),
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
        },
        dataset: dataset.metadata,
        topology,
        measurements,
    };
    report.validate()?;
    Ok(report)
}

/// Owns setup, one steady measurement, and the post-run invariant audit.
async fn run_steady_state<B: Backend>(
    runtime: &Runtime<MeasuredBackend<B>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &BenchmarkDataset,
) -> Result<(Topology, SteadyStateMeasurements), String> {
    let (index, topology, _) =
        prepare_index(runtime, backend_counters, metric_capture, spec, dataset).await?;
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

/// Builds common resource and maintenance accounting at a phase boundary.
fn phase_resources(
    wall_seconds: f64,
    before: ResourceSnapshot,
    after: ResourceSnapshot,
    backend_io: BackendIo,
    metrics: &CapturedMetrics,
    backlog_at_end: usize,
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
            backlog_at_end,
        },
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
/// Keep arbitrary source messages private, but retain FoundationDB's numeric
/// error code so snapshot expiry is distinguishable from other backend failures.
pub(crate) fn error_at(phase: &str, error: ktann::api::Error) -> String {
    format!("{phase}: {:?}", error.kind())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::report::{Distribution, OperationSummary};

    #[test]
    fn benchmark_error_keeps_arbitrary_source_messages_private() {
        let error = ktann::api::Error::with_source(
            ktann::api::ErrorKind::Backend,
            std::io::Error::other("private record contents"),
        );
        assert_eq!(
            super::error_at("construction", error),
            "construction: Backend"
        );
    }

    pub(super) fn operation_summary(
        attempted: u64,
        accepted: u64,
        rejected: u64,
    ) -> OperationSummary {
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
}
