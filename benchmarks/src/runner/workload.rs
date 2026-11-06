//! Bounded workload execution and steady-state measurement.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use ktann::api::{ErrorKind, Index, Metric, Record, SearchRequest, Value};
use ktann::storage::backend::Backend;

use crate::backend::{BackendCounters, MeasuredBackend};
use crate::dataset::BenchmarkDataset;
use crate::metrics::{CapturedMetrics, MetricCapture};
use crate::oracle;
use crate::report::{
    BudgetSummary, Distribution, OperationClass, OperationSummary, RecallSummary,
    SteadyStateMeasurements, WorkloadDispatch, WriteAmplification, aggregate_rejection_rate,
};
use crate::resource::ResourceSnapshot;
use crate::runner::construction::{settle_timeout, wait_for_maintenance};
use crate::runner::{error_at, phase_completed, phase_started};
use crate::scenario::ScenarioSpec;

/// One immutable exact top-k result shared by repeated query operations.
pub(super) type ExactTruth = Arc<[(Bytes, f64)]>;

/// A successful search retained until recall can run outside resource timing.
#[derive(Debug)]
struct RecallInput {
    /// Record IDs returned inside the measured API call.
    hit_ids: Vec<Bytes>,
    /// Immutable exact result computed before any benchmark timing.
    truth: ExactTruth,
}

/// Constructs one deterministic public search request.
pub(super) fn search_request(
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
    query_index: usize,
) -> Result<SearchRequest, String> {
    SearchRequest::new(Arc::clone(&dataset.queries[query_index]), spec.k)
        .map(|request| request.with_options(spec.search_options))
        .map_err(|error| error_at("construct search request", error))
}

/// Measures one fixed workload after setup and before the invariant audit.
pub(super) async fn measure_steady_workload<B: Backend>(
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
    let point = format!("{point} concurrency {}", spec.concurrency);
    let warmup_phase = format!("{point} warmup");
    let warmup_started = phase_started(spec, &warmup_phase);
    let _warmup_result =
        execute_items(index.clone(), warmup, spec.concurrency, spec.dispatch).await?;
    phase_completed(spec, &warmup_phase, warmup_started);

    // Reset setup counters and histograms; gauges retain current process state.
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
        quality_diagnostics: None,
        recall_at_k: recall_summary(&recalls),
        per_query_recall: recalls,
        routing: metrics.routing_summary(),
        wall_seconds,
        maintenance_drain_seconds,
        physical_read_bytes: resources_after
            .disk_io_bytes_since(resources_before)
            .map(|bytes| bytes.0),
        physical_write_bytes: resources_after
            .disk_io_bytes_since(resources_before)
            .map(|bytes| bytes.1),
        cpu_seconds: Some(resources_after.cpu_seconds_since(resources_before)),
        peak_rss_bytes: Some(resources_after.peak_rss_bytes()),
        throughput_per_second: if wall_seconds > 0.0 {
            successful_operations as f64 / wall_seconds
        } else {
            0.0
        },
        operations,
        search_budgets: budget_summaries(&metrics),
        visited_leaf_entries: Distribution::from_samples(
            metrics.histogram("ktann.search.leaf_entries", &[]),
        ),
        search_stages_ms: metrics.search_stage_summaries(),
        cache: metrics.cache_summary(),
        backend_admission: metrics.admission_summary(),
        backend_io,
        write_amplification,
    };
    Ok(measurements)
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
pub(super) fn exact_truth(dataset: &BenchmarkDataset, metric: Metric, k: usize) -> Vec<ExactTruth> {
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
pub(super) fn recall_summary(recalls: &[f64]) -> Option<RecallSummary> {
    if recalls.is_empty() {
        return None;
    }
    Some(RecallSummary {
        queries: recalls.len() as u64,
        mean: recalls.iter().sum::<f64>() / recalls.len() as f64,
        min: recalls.iter().copied().fold(f64::INFINITY, f64::min),
    })
}

/// Summarizes usage and exhaustion for the three Search Budget dimensions.
pub(super) fn budget_summaries(metrics: &CapturedMetrics) -> BTreeMap<String, BudgetSummary> {
    [
        "scanned_tree_keys",
        "visited_partitions",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::tests::operation_summary;
    use crate::scenario::scenarios;

    #[test]
    fn measured_windows_preserve_each_declared_operation_mix() {
        for profile in ["smoke", "full", "large", "diagnostic"] {
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
        let scenario = scenarios("diagnostic")
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
}
