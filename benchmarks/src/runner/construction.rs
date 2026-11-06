//! Index construction, Structure Maintenance convergence, and invariant verification.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use ktann::api::{
    ErrorKind, Index, IndexConfig, Mutation, OperationOptions, Record, RefineOptions,
    SearchRequest, Value, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;

use crate::backend::{BackendCounters, MeasuredBackend, subtract};
use crate::dataset::BenchmarkDataset;
use crate::metrics::MetricCapture;
use crate::report::{ConstructionMeasurements, ConstructionPhase, PartitionStateCounts, Topology};
use crate::resource::ResourceSnapshot;
use crate::runner::{error_at, phase_announced, phase_completed, phase_resources, phase_started};
use crate::scenario::{ScenarioSpec, index_config};

/// Materializes bounded mutation batches before the first submission timer.
pub(super) fn mutation_batches(
    dataset: &BenchmarkDataset,
    batch_size: usize,
    phase: &str,
) -> Result<Vec<Vec<Mutation>>, String> {
    dataset
        .ids
        .chunks(batch_size)
        .zip(dataset.base.chunks(batch_size))
        .map(|(ids, vectors)| mutation_batch(ids, vectors, &[Value::I64(0), Value::I64(0)], phase))
        .collect()
}

/// Builds one bounded batch of insert mutations from aligned dataset rows.
fn mutation_batch(
    ids: &[Bytes],
    vectors: &[Arc<[f32]>],
    fields: &[Value],
    phase: &str,
) -> Result<Vec<Mutation>, String> {
    ids.iter()
        .zip(vectors)
        .map(|(id, vector)| {
            Record::new(id.clone(), Arc::clone(vector), fields.to_vec())
                .map(Mutation::Insert)
                .map_err(|error| error_at(phase, error))
        })
        .collect()
}

/// Creates and measures a fresh Logical Index before the search interval.
pub(super) async fn prepare_index<B: Backend>(
    runtime: &Runtime<MeasuredBackend<B>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &BenchmarkDataset,
) -> Result<
    (
        Index<MeasuredBackend<B>>,
        Topology,
        ConstructionMeasurements,
    ),
    String,
> {
    if spec.routing_experiment.is_some() {
        return prepare_bulk_index(runtime, backend_counters, metric_capture, spec, dataset).await;
    }
    let index_config = index_config(spec)?;
    let index = runtime
        .create_index("benchmark", index_config)
        .await
        .map_err(|error| error_at("create index", error))?;

    // Dataset/index creation is excluded. Import keeps streaming its batches;
    // constructing every batch before this boundary would change memory use.
    drop(metric_capture.snapshot());
    let backend_before = backend_counters.snapshot();
    let resources_before = ResourceSnapshot::capture()?;
    let import_started = phase_started(spec, "import");
    load_index(&index, dataset, spec).await?;
    let import_completed = Instant::now();
    let import_seconds = import_completed
        .duration_since(import_started)
        .as_secs_f64();
    phase_completed(spec, "import", import_started);
    let import_after = ResourceSnapshot::capture()?;
    let import_backend = backend_counters.snapshot();
    let import_metrics = metric_capture.snapshot();
    let import_backlog = metric_capture.fixup_backlog();

    let deadline = Instant::now() + settle_timeout(spec);
    let (mut topology, _) =
        settle_and_drain_topology(&index, dataset, spec, metric_capture, deadline).await?;
    if let Some(rounds) = spec.refinement_rounds {
        wait_for_maintenance_until(metric_capture, deadline).await?;
        let options = RefineOptions::new(32 << 30)
            .and_then(|options| options.with_refinement_rounds(rounds))
            .map_err(|error| error_at("refinement options", error))?;
        let refinement_started = phase_started(spec, "refinement");
        index
            .refine(options)
            .await
            .map_err(|error| error_at("offline refinement", error))?;
        phase_completed(spec, "refinement", refinement_started);
        topology = verified_topology(
            &index,
            spec,
            "verify refined topology",
            Instant::now() + settle_timeout(spec),
        )
        .await?;
    }
    let completed = Instant::now();
    let convergence_seconds = completed.duration_since(import_completed).as_secs_f64();
    let wall_seconds = completed.duration_since(import_started).as_secs_f64();
    let resources_after = ResourceSnapshot::capture()?;
    let backend_after = backend_counters.snapshot();
    let convergence_metrics = metric_capture.snapshot();
    let convergence_backlog = metric_capture.fixup_backlog();
    // Render the captured phase samples only after the continuous timer stops.
    let import = ConstructionPhase {
        resources: phase_resources(
            import_seconds,
            resources_before,
            import_after,
            subtract(&import_backend, &backend_before),
            &import_metrics,
            import_backlog,
        ),
        admission: import_metrics.admission_summary(),
        cache: import_metrics.cache_summary(),
    };
    let refinement_rounds = convergence_metrics.counter("ktann.refine.rounds", &[]);
    let refinement_moves = convergence_metrics.counter("ktann.refine.moves", &[]);
    drop(import_metrics);
    log_import_diagnostics(spec, &import);
    let convergence = ConstructionPhase {
        resources: phase_resources(
            convergence_seconds,
            import_after,
            resources_after,
            subtract(&backend_after, &import_backend),
            &convergence_metrics,
            convergence_backlog,
        ),
        admission: convergence_metrics.admission_summary(),
        cache: convergence_metrics.cache_summary(),
    };
    let construction = ConstructionMeasurements {
        bulk: None,
        refinement_rounds: Some(refinement_rounds),
        refinement_moves: Some(refinement_moves),
        wall_seconds,
        cpu_seconds: Some(resources_after.cpu_seconds_since(resources_before)),
        peak_rss_bytes: Some(resources_after.peak_rss_bytes()),
        import,
        convergence,
    };
    Ok((index, topology, construction))
}

/// Renders the same captured import interval that is retained in the report.
fn log_import_diagnostics(spec: &ScenarioSpec, import: &ConstructionPhase) {
    let attempts = &import.resources.writes.attempts;
    if !attempts.is_empty() {
        eprintln!("[{}] import write attempts: {attempts:?}", spec.name);
    }
    for (labels, wait) in &import.resources.writes.commit_wait_ms {
        eprintln!(
            "[{}] import commit wait {labels}: count={} mean={:.3}ms p95={:.3}ms",
            spec.name, wait.count, wait.mean, wait.p95,
        );
    }
    let steps = &import.resources.maintenance.steps;
    if !steps.is_empty() {
        eprintln!("[{}] import-phase fixup steps: {steps:?}", spec.name);
    }
}

/// Protects reports from measurements produced by corrupt persistent state.
pub(super) async fn verify_measured_state<B: Backend>(
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

/// Loads ordinary atomic mutation batches, retrying only definite exhaustion.
async fn load_index<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
) -> Result<(), String> {
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
    for (ids, vectors) in dataset
        .ids
        .chunks(batch_size)
        .zip(dataset.base.chunks(batch_size))
    {
        let mutations = mutation_batch(ids, vectors, &fields, "construct load record")?;
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
fn topology_is_settled(topology: &Topology, config: &IndexConfig) -> bool {
    let leaf_entries = topology.entries_by_level.get(&1).copied().unwrap_or(0);
    topology.partitions > 1
        && topology.entries > topology.vector_records
        && leaf_entries == topology.vector_records
        && topology
            .max_entries_by_level
            .iter()
            .all(|(level, count)| *count <= config.max_partition_entries(*level))
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
pub(super) async fn settle_and_drain_topology<B: Backend>(
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
        if topology_is_settled(&topology, index.config()) {
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

/// Verifies persistent state and returns its observable topology counts.
pub(super) async fn verified_topology<B: Backend>(
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
pub(super) fn settle_timeout(spec: &ScenarioSpec) -> Duration {
    match spec.profile {
        "smoke" => Duration::from_secs(30),
        "large" => Duration::from_secs(6 * 60 * 60),
        _ => Duration::from_secs(120),
    }
}

/// Waits until all maintenance caused before the accounting boundary finishes.
pub(super) async fn wait_for_maintenance(
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

/// Bulk Build uses final refined centroids before graph encoding/publication.
/// All construction and verification are outside the measured query interval.
async fn prepare_bulk_index<B: Backend>(
    runtime: &Runtime<MeasuredBackend<B>>,
    backend_counters: &BackendCounters,
    capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &BenchmarkDataset,
) -> Result<
    (
        Index<MeasuredBackend<B>>,
        Topology,
        ConstructionMeasurements,
    ),
    String,
> {
    use ktann::api::BulkBuildOptions;
    use ktann::bulk::InputSnapshot;
    let experiment = spec
        .routing_experiment
        .as_ref()
        .expect("routing experiment");
    let root = tempfile::tempdir().map_err(|e| e.to_string())?;
    let config = index_config(spec)?;
    drop(capture.snapshot());
    let before = backend_counters.snapshot();
    let resources = ResourceSnapshot::capture()?;
    let started = Instant::now();
    let source = InputSnapshot::create(
        &root.path().join("input"),
        config,
        experiment.build_scratch_bytes,
        dataset
            .ids
            .iter()
            .zip(&dataset.base)
            .map(|(id, vector)| Record::new(id.clone(), Arc::clone(vector), vec![])),
    )
    .map_err(|e| error_at("capture bulk source", e))?;
    let capture_seconds = started.elapsed().as_secs_f64();
    let construction = experiment.construction();
    let job = runtime
        .start_bulk_build(
            "benchmark",
            &source,
            BulkBuildOptions {
                construction,
                refinement_rounds: spec.refinement_rounds.map(|rounds| rounds as u8),
            },
        )
        .await
        .map_err(|e| error_at("reserve bulk index", e))?;
    let worker = experiment.worker(root.path().join("workspace"));
    let settings = crate::report::bulk_build_settings(&construction, &worker);
    std::fs::create_dir(&worker.workspace).map_err(|e| e.to_string())?;
    let (index, stages) = job
        .complete_with_report(worker, None, OperationOptions::default())
        .await
        .map_err(|e| error_at("complete bulk index", e))?;
    let imported = ResourceSnapshot::capture()?;
    let imported_io = backend_counters.snapshot();
    let import_seconds = started.elapsed().as_secs_f64();
    let metrics = capture.snapshot();
    let graph_build_seconds = metrics.histogram("ktann.routing.build_seconds", &[]);
    let graph_native_peak_bytes = metrics.histogram("ktann.routing.native_build_peak_bytes", &[]);
    let stored_graph_bytes = metrics.counter("ktann.routing.stored_graph_bytes", &[]);
    let import = ConstructionPhase {
        resources: phase_resources(
            import_seconds,
            resources,
            imported,
            subtract(&imported_io, &before),
            &metrics,
            capture.fixup_backlog(),
        ),
        admission: metrics.admission_summary(),
        cache: metrics.cache_summary(),
    };
    let verification = Instant::now();
    let topology = verified_topology(
        &index,
        spec,
        "verify bulk index",
        Instant::now() + settle_timeout(spec),
    )
    .await?;
    let after = ResourceSnapshot::capture()?;
    let metrics = capture.snapshot();
    let convergence = ConstructionPhase {
        resources: phase_resources(
            verification.elapsed().as_secs_f64(),
            imported,
            after,
            subtract(&backend_counters.snapshot(), &imported_io),
            &metrics,
            capture.fixup_backlog(),
        ),
        admission: metrics.admission_summary(),
        cache: metrics.cache_summary(),
    };
    let mut bulk = serde_json::json!({
        "capture_seconds": capture_seconds, "forest_seconds": stages.forest.as_secs_f64(),
        "serving_seconds": stages.serving.as_secs_f64(), "load_seconds": stages.load.as_secs_f64(),
        "publish_seconds": stages.publish.as_secs_f64(), "source_sha256": source.manifest().sha256(),
        "source_bytes": source.manifest().bytes(), "rotation_seed": stages.rotation_seed,
        "graph_build_seconds": graph_build_seconds,
        "graph_native_peak_bytes": graph_native_peak_bytes, "stored_graph_bytes": stored_graph_bytes,
    });
    bulk.as_object_mut()
        .expect("bulk report object")
        .extend(settings);
    let measurements = ConstructionMeasurements {
        bulk: Some(bulk),
        refinement_rounds: None,
        refinement_moves: None,
        wall_seconds: started.elapsed().as_secs_f64(),
        cpu_seconds: Some(after.cpu_seconds_since(resources)),
        peak_rss_bytes: Some(after.peak_rss_bytes()),
        import,
        convergence,
    };
    Ok((index, topology, measurements))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

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

        assert!(topology_is_settled(
            &topology,
            &ktann::api::IndexConfig::new(2, ktann::api::Metric::L2)
                .unwrap()
                .with_leaf_partition_entries(64, 1_842)
                .unwrap()
        ));
        assert!(!topology_is_settled(
            &topology,
            &ktann::api::IndexConfig::new(2, ktann::api::Metric::L2).unwrap()
        ));
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
