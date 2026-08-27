//! Material-regression comparison for reports from equivalent environments.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::report::{
    AdmissionSummary, BackendIo, BenchmarkReport, BenchmarkSuite, BudgetSummary, CacheSummary,
    Distribution, LifecycleMeasurements, OperationClass, OperationSummary, REPORT_SCHEMA_VERSION,
    ReportMeasurements, SearchPhase, SteadyStateMeasurements,
};

/// Sub-millisecond p95 movement is not stable enough to classify by ratio.
const MINIMUM_LATENCY_REGRESSION_MS: f64 = 1.0;

/// Explicit policy for distinguishing material movement from ordinary noise.
#[derive(Clone, Copy, Debug)]
pub struct ComparisonPolicy {
    /// Maximum relative increase for latency, CPU, RSS, and logical work.
    pub maximum_relative_regression: f64,
    /// Maximum absolute decrease in recall@k.
    pub maximum_recall_drop: f64,
    /// Maximum absolute increase in admission-rejection rate.
    pub maximum_rejection_rate_increase: f64,
}

impl Default for ComparisonPolicy {
    fn default() -> Self {
        Self {
            maximum_relative_regression: 0.20,
            maximum_recall_drop: 0.02,
            maximum_rejection_rate_increase: 0.05,
        }
    }
}

/// Machine-readable result of comparing two benchmark suites.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ComparisonReport {
    /// Material regressions that make the comparison fail.
    pub regressions: Vec<String>,
    /// Non-failing context such as unavailable resource observations.
    pub notes: Vec<String>,
}

impl ComparisonReport {
    /// Returns whether any material regression was found.
    #[must_use]
    pub fn failed(&self) -> bool {
        !self.regressions.is_empty()
    }
}

/// Compares scenario reports only when their inputs and host are equivalent.
///
/// # Errors
///
/// Returns an error when policy values are invalid or the suites differ in
/// schema, scenario set, inputs, Backend limits, or runtime fingerprint.
pub fn compare(
    baseline: &BenchmarkSuite,
    candidate: &BenchmarkSuite,
    policy: ComparisonPolicy,
) -> Result<ComparisonReport, String> {
    if baseline.schema_version != REPORT_SCHEMA_VERSION
        || candidate.schema_version != REPORT_SCHEMA_VERSION
    {
        return Err("benchmark report schema mismatch".to_owned());
    }
    if !policy.maximum_relative_regression.is_finite()
        || policy.maximum_relative_regression < 0.0
        || !policy.maximum_recall_drop.is_finite()
        || policy.maximum_recall_drop < 0.0
        || !policy.maximum_rejection_rate_increase.is_finite()
        || policy.maximum_rejection_rate_increase < 0.0
    {
        return Err("comparison thresholds must be finite and nonnegative".to_owned());
    }
    let baseline = by_key(&baseline.reports)?;
    let candidate = by_key(&candidate.reports)?;
    if baseline.keys().ne(candidate.keys()) {
        return Err("baseline and candidate scenario sets differ".to_owned());
    }

    let mut result = ComparisonReport::default();
    for (key, baseline) in baseline {
        let candidate = candidate
            .get(&key)
            .ok_or_else(|| format!("candidate is missing scenario {key}"))?;
        ensure_comparable(baseline, candidate)?;
        compare_report(&key, baseline, candidate, policy, &mut result);
    }
    Ok(result)
}

/// Indexes scenario reports and rejects ambiguous duplicate comparison keys.
fn by_key(reports: &[BenchmarkReport]) -> Result<BTreeMap<String, &BenchmarkReport>, String> {
    let mut keyed = BTreeMap::new();
    for report in reports {
        let key = format!(
            "{}/{}",
            report.configuration.backend, report.configuration.scenario
        );
        if keyed.insert(key.clone(), report).is_some() {
            return Err(format!("duplicate scenario {key}"));
        }
    }
    Ok(keyed)
}

/// Timing comparisons are meaningful only under identical workload and host facts.
fn ensure_comparable(
    baseline: &BenchmarkReport,
    candidate: &BenchmarkReport,
) -> Result<(), String> {
    validate_operation_summaries(baseline)?;
    validate_operation_summaries(candidate)?;
    validate_lifecycle_measurements(baseline)?;
    validate_lifecycle_measurements(candidate)?;
    // Compare the schema type as a whole so adding a workload or admission
    // input cannot silently leave the comparability contract incomplete.
    if baseline.configuration != candidate.configuration
        || baseline.dataset.checksum_xxh3_128 != candidate.dataset.checksum_xxh3_128
    {
        return Err(format!(
            "scenario {}/{} has incomparable inputs",
            baseline.configuration.backend, baseline.configuration.scenario
        ));
    }
    if baseline.environment != candidate.environment {
        return Err(format!(
            "scenario {}/{} has a different hardware/runtime fingerprint",
            baseline.configuration.backend, baseline.configuration.scenario
        ));
    }
    if std::mem::discriminant(&baseline.measurements)
        != std::mem::discriminant(&candidate.measurements)
    {
        return Err(format!(
            "scenario {}/{} has different measurement kinds",
            baseline.configuration.backend, baseline.configuration.scenario
        ));
    }
    Ok(())
}

/// Rejects internally inconsistent lifecycle measurements at the JSON boundary.
fn validate_lifecycle_measurements(report: &BenchmarkReport) -> Result<(), String> {
    let ReportMeasurements::Lifecycle(lifecycle) = &report.measurements else {
        return Ok(());
    };
    if lifecycle.unattributed_wall_seconds().is_none() {
        return Err(format!(
            "scenario {}/{} has inconsistent lifecycle phase accounting",
            report.configuration.backend, report.configuration.scenario
        ));
    }
    match (
        lifecycle.case_cpu_seconds,
        lifecycle.unattributed_cpu_seconds(),
    ) {
        (Some(_), Some(_)) => {}
        (None, None) => {}
        _ => {
            return Err(format!(
                "scenario {}/{} has inconsistent lifecycle CPU accounting",
                report.configuration.backend, report.configuration.scenario
            ));
        }
    }
    if lifecycle.unattributed_backend_io().is_none() {
        return Err(format!(
            "scenario {}/{} has inconsistent lifecycle Backend IO accounting",
            report.configuration.backend, report.configuration.scenario
        ));
    }
    let configured_attempts =
        u64::try_from(report.configuration.measured_operations).map_err(|_| {
            format!(
                "scenario {}/{} measured operation count exceeds report range",
                report.configuration.backend, report.configuration.scenario
            )
        })?;
    for phase in [
        &lifecycle.immediate_search,
        &lifecycle.stable_cold_search,
        &lifecycle.stable_warm_search,
    ] {
        phase.search.validate().map_err(|error| {
            format!(
                "scenario {}/{} has invalid lifecycle search results: {error}",
                report.configuration.backend, report.configuration.scenario
            )
        })?;
        if phase.search.attempted != configured_attempts {
            return Err(format!(
                "scenario {}/{} lifecycle query attempts differ from its configuration",
                report.configuration.backend, report.configuration.scenario
            ));
        }
    }
    Ok(())
}

/// Rejects internally inconsistent operation summaries at the JSON boundary.
fn validate_operation_summaries(report: &BenchmarkReport) -> Result<(), String> {
    let ReportMeasurements::SteadyState(measurements) = &report.measurements else {
        return Ok(());
    };
    if !OperationClass::for_mix(report.configuration.search_percent)
        .eq(measurements.operations.keys().copied())
    {
        return Err(format!(
            "scenario {}/{} has operation classes inconsistent with its mix",
            report.configuration.backend, report.configuration.scenario
        ));
    }
    let attempted = measurements
        .operations
        .values()
        .try_fold(0_u64, |sum, summary| {
            summary.validate()?;
            sum.checked_add(summary.attempted)
                .ok_or("operation attempt count overflow")
        })
        .map_err(|error| {
            format!(
                "scenario {}/{} has invalid operation results: {error}",
                report.configuration.backend, report.configuration.scenario
            )
        })?;
    let configured_attempts =
        u64::try_from(report.configuration.measured_operations).map_err(|_| {
            format!(
                "scenario {}/{} measured operation count exceeds report range",
                report.configuration.backend, report.configuration.scenario
            )
        })?;
    if attempted != configured_attempts {
        return Err(format!(
            "scenario {}/{} operation attempts differ from its configuration",
            report.configuration.backend, report.configuration.scenario
        ));
    }
    Ok(())
}

/// Applies every issue #38 comparison family to one comparable scenario pair.
fn compare_report(
    key: &str,
    baseline: &BenchmarkReport,
    candidate: &BenchmarkReport,
    policy: ComparisonPolicy,
    result: &mut ComparisonReport,
) {
    match (&baseline.measurements, &candidate.measurements) {
        (ReportMeasurements::SteadyState(baseline), ReportMeasurements::SteadyState(candidate)) => {
            compare_steady_state(result, key, baseline, candidate, policy);
        }
        (ReportMeasurements::Lifecycle(baseline), ReportMeasurements::Lifecycle(candidate)) => {
            compare_lifecycle(result, key, baseline, candidate, policy);
        }
        _ => result
            .regressions
            .push(format!("{key}: measurement kind changed")),
    }
}

/// Compares one pair of steady-state scenario measurements.
fn compare_steady_state(
    result: &mut ComparisonReport,
    key: &str,
    baseline: &SteadyStateMeasurements,
    candidate: &SteadyStateMeasurements,
    policy: ComparisonPolicy,
) {
    compare_operations(
        result,
        key,
        &baseline.operations,
        &candidate.operations,
        policy,
    );
    relative_decrease(
        result,
        key,
        "throughput",
        baseline.throughput_per_second,
        candidate.throughput_per_second,
        policy.maximum_relative_regression,
    );
    compare_optional_resource(
        result,
        key,
        "CPU seconds",
        baseline.cpu_seconds,
        candidate.cpu_seconds,
        policy.maximum_relative_regression,
    );
    relative_regression(
        result,
        key,
        "maintenance drain seconds",
        baseline.maintenance_drain_seconds,
        candidate.maintenance_drain_seconds,
        policy.maximum_relative_regression,
    );
    compare_distributions(
        result,
        key,
        "search stage",
        &baseline.search_stages_ms,
        &candidate.search_stages_ms,
        policy.maximum_relative_regression,
    );
    compare_budgets(
        result,
        key,
        &baseline.search_budgets,
        &candidate.search_budgets,
        policy.maximum_relative_regression,
    );
    compare_cache(
        result,
        key,
        &baseline.cache,
        &candidate.cache,
        policy.maximum_relative_regression,
    );
    compare_admission(
        result,
        key,
        &baseline.backend_admission,
        &candidate.backend_admission,
        policy.maximum_relative_regression,
    );
    compare_backend_io(
        result,
        key,
        &baseline.backend_io,
        &candidate.backend_io,
        policy.maximum_relative_regression,
    );
    compare_optional_resource(
        result,
        key,
        "peak RSS bytes",
        baseline.peak_rss_bytes.map(|value| value as f64),
        candidate.peak_rss_bytes.map(|value| value as f64),
        policy.maximum_relative_regression,
    );
    compare_recall(
        result,
        key,
        "",
        baseline.recall_at_k.as_ref(),
        candidate.recall_at_k.as_ref(),
        policy.maximum_recall_drop,
    );
    match (
        baseline.write_amplification.as_ref(),
        candidate.write_amplification.as_ref(),
    ) {
        (Some(baseline), Some(candidate)) => {
            relative_regression(
                result,
                key,
                "logical mutations/write",
                baseline.logical_mutations_per_write,
                candidate.logical_mutations_per_write,
                policy.maximum_relative_regression,
            );
            relative_regression(
                result,
                key,
                "write retries",
                baseline.write_retries as f64,
                candidate.write_retries as f64,
                policy.maximum_relative_regression,
            );
            relative_regression(
                result,
                key,
                "logical bytes/write",
                baseline.logical_bytes_per_write,
                candidate.logical_bytes_per_write,
                policy.maximum_relative_regression,
            );
        }
        (None, None) => {}
        _ => result
            .regressions
            .push(format!("{key}: write amplification availability changed")),
    }
}

/// Compares the phase boundaries unique to import-to-search lifecycle cases.
fn compare_lifecycle(
    result: &mut ComparisonReport,
    key: &str,
    baseline: &LifecycleMeasurements,
    candidate: &LifecycleMeasurements,
    policy: ComparisonPolicy,
) {
    relative_regression(
        result,
        key,
        "lifecycle case wall seconds",
        baseline.case_wall_seconds,
        candidate.case_wall_seconds,
        policy.maximum_relative_regression,
    );
    compare_optional_resource(
        result,
        key,
        "lifecycle case CPU seconds",
        baseline.case_cpu_seconds,
        candidate.case_cpu_seconds,
        policy.maximum_relative_regression,
    );
    compare_optional_resource(
        result,
        key,
        "lifecycle case peak RSS bytes",
        baseline.case_peak_rss_bytes.map(|value| value as f64),
        candidate.case_peak_rss_bytes.map(|value| value as f64),
        policy.maximum_relative_regression,
    );
    relative_decrease(
        result,
        key,
        "import records/second",
        baseline.import.records_per_second,
        candidate.import.records_per_second,
        policy.maximum_relative_regression,
    );
    compare_distribution(
        result,
        key,
        "import submit latency",
        &baseline.import.submit_latency_ms,
        &candidate.import.submit_latency_ms,
        policy.maximum_relative_regression,
    );
    compare_count_maps(
        result,
        key,
        "import batch failure",
        &baseline.import.batch_failures,
        &candidate.import.batch_failures,
        policy.maximum_relative_regression,
    );
    relative_decrease(
        result,
        key,
        "import accepted records",
        baseline.import.accepted_records as f64,
        candidate.import.accepted_records as f64,
        0.0,
    );
    compare_admission(
        result,
        key,
        &baseline.import.admission,
        &candidate.import.admission,
        policy.maximum_relative_regression,
    );
    compare_phase_resources(
        result,
        key,
        "import",
        &baseline.import.resources,
        &candidate.import.resources,
        policy,
    );
    relative_regression(
        result,
        key,
        "finish-to-stable seconds",
        baseline.convergence.from_import_finish_seconds,
        candidate.convergence.from_import_finish_seconds,
        policy.maximum_relative_regression,
    );
    compare_phase_resources(
        result,
        key,
        "convergence",
        &baseline.convergence.resources,
        &candidate.convergence.resources,
        policy,
    );
    compare_phase_resources(
        result,
        key,
        "cache reset",
        &baseline.cache_reset,
        &candidate.cache_reset,
        policy,
    );
    relative_regression(
        result,
        key,
        "convergence maintenance drain seconds",
        baseline.convergence.maintenance_drain_seconds,
        candidate.convergence.maintenance_drain_seconds,
        policy.maximum_relative_regression,
    );
    compare_lifecycle_search(
        result,
        key,
        "immediate search",
        &baseline.immediate_search,
        &candidate.immediate_search,
        policy,
    );
    compare_lifecycle_search(
        result,
        key,
        "stable cold search",
        &baseline.stable_cold_search,
        &candidate.stable_cold_search,
        policy,
    );
    compare_lifecycle_search(
        result,
        key,
        "stable warm search",
        &baseline.stable_warm_search,
        &candidate.stable_warm_search,
        policy,
    );
}

/// Compares one fixed lifecycle query pass without mixing cache states.
fn compare_lifecycle_search(
    result: &mut ComparisonReport,
    key: &str,
    phase_name: &str,
    baseline: &SearchPhase,
    candidate: &SearchPhase,
    policy: ComparisonPolicy,
) {
    compare_distribution(
        result,
        key,
        &format!("{phase_name} latency"),
        &baseline.search.latency_ms,
        &candidate.search.latency_ms,
        policy.maximum_relative_regression,
    );
    relative_regression(
        result,
        key,
        &format!("{phase_name} admission rejections"),
        baseline.search.rejected as f64,
        candidate.search.rejected as f64,
        policy.maximum_relative_regression,
    );
    compare_non_admission_errors(
        result,
        key,
        &format!("{phase_name} error"),
        &baseline.search.errors,
        &candidate.search.errors,
        policy.maximum_relative_regression,
    );
    compare_optional_resource(
        result,
        key,
        &format!("{phase_name} first-query latency"),
        baseline.first_query_latency_ms,
        candidate.first_query_latency_ms,
        policy.maximum_relative_regression,
    );
    relative_decrease(
        result,
        key,
        &format!("{phase_name} throughput"),
        baseline.throughput_per_second,
        candidate.throughput_per_second,
        policy.maximum_relative_regression,
    );
    compare_recall(
        result,
        key,
        &format!("{phase_name} "),
        baseline.recall_at_k.as_ref(),
        candidate.recall_at_k.as_ref(),
        policy.maximum_recall_drop,
    );
    compare_budgets(
        result,
        key,
        &baseline.search_budgets,
        &candidate.search_budgets,
        policy.maximum_relative_regression,
    );
    compare_distributions(
        result,
        key,
        &format!("{phase_name} search stage"),
        &baseline.search_stages_ms,
        &candidate.search_stages_ms,
        policy.maximum_relative_regression,
    );
    compare_cache(
        result,
        key,
        &baseline.cache,
        &candidate.cache,
        policy.maximum_relative_regression,
    );
    compare_admission(
        result,
        key,
        &baseline.backend_admission,
        &candidate.backend_admission,
        policy.maximum_relative_regression,
    );
    for (name, baseline, candidate) in [
        (
            "scanned Tree Keys",
            baseline.truncation.scanned_tree_keys,
            candidate.truncation.scanned_tree_keys,
        ),
        (
            "visited partitions",
            baseline.truncation.visited_partitions,
            candidate.truncation.visited_partitions,
        ),
        (
            "visited Leaf Entries",
            baseline.truncation.visited_leaf_entries,
            candidate.truncation.visited_leaf_entries,
        ),
        (
            "exact rerank candidates",
            baseline.truncation.exact_rerank_candidates,
            candidate.truncation.exact_rerank_candidates,
        ),
        (
            "RaBitQ overlap",
            baseline.truncation.rabitq_overlap,
            candidate.truncation.rabitq_overlap,
        ),
    ] {
        relative_regression(
            result,
            key,
            &format!("{phase_name} {name} truncation"),
            baseline as f64,
            candidate as f64,
            policy.maximum_relative_regression,
        );
    }
    compare_phase_resources(
        result,
        key,
        phase_name,
        &baseline.resources,
        &candidate.resources,
        policy,
    );
}

/// Compares recall availability and its absolute mean drop.
fn compare_recall(
    result: &mut ComparisonReport,
    key: &str,
    prefix: &str,
    baseline: Option<&crate::report::RecallSummary>,
    candidate: Option<&crate::report::RecallSummary>,
    maximum_drop: f64,
) {
    match (baseline, candidate) {
        (Some(baseline), Some(candidate)) => {
            if baseline.mean - candidate.mean > maximum_drop {
                result.regressions.push(format!(
                    "{key}: {prefix}mean recall dropped from {:.4} to {:.4}",
                    baseline.mean, candidate.mean
                ));
            }
            if baseline.min - candidate.min > maximum_drop {
                result.regressions.push(format!(
                    "{key}: {prefix}minimum recall dropped from {:.4} to {:.4}",
                    baseline.min, candidate.min
                ));
            }
        }
        (None, None) => {}
        _ => result
            .regressions
            .push(format!("{key}: {prefix}recall availability changed")),
    }
}

/// Compares phase-local CPU, Backend IO, and maintenance work.
fn compare_phase_resources(
    result: &mut ComparisonReport,
    key: &str,
    phase_name: &str,
    baseline: &crate::report::PhaseResources,
    candidate: &crate::report::PhaseResources,
    policy: ComparisonPolicy,
) {
    relative_regression(
        result,
        key,
        &format!("{phase_name} wall seconds"),
        baseline.wall_seconds,
        candidate.wall_seconds,
        policy.maximum_relative_regression,
    );
    compare_optional_resource(
        result,
        key,
        &format!("{phase_name} CPU seconds"),
        baseline.cpu_seconds,
        candidate.cpu_seconds,
        policy.maximum_relative_regression,
    );
    compare_backend_io(
        result,
        key,
        &baseline.backend_io,
        &candidate.backend_io,
        policy.maximum_relative_regression,
    );
    compare_count_maps(
        result,
        key,
        &format!("{phase_name} Fixup admission"),
        &baseline.maintenance.admission,
        &candidate.maintenance.admission,
        policy.maximum_relative_regression,
    );
    compare_count_maps(
        result,
        key,
        &format!("{phase_name} Fixup execution"),
        &baseline.maintenance.execution,
        &candidate.maintenance.execution,
        policy.maximum_relative_regression,
    );
}

/// Compares each stable outcome category without hiding shifts in equal totals.
fn compare_count_maps(
    result: &mut ComparisonReport,
    key: &str,
    metric: &str,
    baseline: &BTreeMap<String, u64>,
    candidate: &BTreeMap<String, u64>,
    threshold: f64,
) {
    for category in baseline
        .keys()
        .chain(candidate.keys())
        .collect::<BTreeSet<_>>()
    {
        relative_regression(
            result,
            key,
            &format!("{metric} {category}"),
            baseline.get(category).copied().unwrap_or_default() as f64,
            candidate.get(category).copied().unwrap_or_default() as f64,
            threshold,
        );
    }
}

/// Compares stable operation errors while admission uses its aggregate policy.
fn compare_non_admission_errors(
    result: &mut ComparisonReport,
    key: &str,
    metric: &str,
    baseline: &BTreeMap<String, u64>,
    candidate: &BTreeMap<String, u64>,
    threshold: f64,
) {
    for category in baseline
        .keys()
        .chain(candidate.keys())
        .filter(|category| category.as_str() != "LimitExceeded")
        .collect::<BTreeSet<_>>()
    {
        relative_regression(
            result,
            key,
            &format!("{metric} {category}"),
            baseline.get(category).copied().unwrap_or_default() as f64,
            candidate.get(category).copied().unwrap_or_default() as f64,
            threshold,
        );
    }
}

/// Compares accepted latency and admission outcomes without amplifying count noise.
fn compare_operations(
    result: &mut ComparisonReport,
    scenario: &str,
    baseline: &BTreeMap<OperationClass, OperationSummary>,
    candidate: &BTreeMap<OperationClass, OperationSummary>,
    policy: ComparisonPolicy,
) {
    for (class, baseline) in baseline {
        let class_name = class.as_str();
        let candidate = candidate
            .get(class)
            .expect("validated reports have identical operation classes");
        compare_distribution(
            result,
            scenario,
            &format!("{class_name} latency"),
            &baseline.latency_ms,
            &candidate.latency_ms,
            policy.maximum_relative_regression,
        );
        compare_non_admission_errors(
            result,
            scenario,
            &format!("{class_name} error"),
            &baseline.errors,
            &candidate.errors,
            policy.maximum_relative_regression,
        );
    }
    let baseline_rate = aggregate_rejection_rate(baseline);
    let candidate_rate = aggregate_rejection_rate(candidate);
    if candidate_rate > baseline_rate + policy.maximum_rejection_rate_increase {
        result.regressions.push(format!(
            "{scenario}: aggregate rejection rate increased from {baseline_rate:.4} to {candidate_rate:.4}"
        ));
    }
}

/// Returns the admission-rejection rate across the fixed operation mix.
fn aggregate_rejection_rate(operations: &BTreeMap<OperationClass, OperationSummary>) -> f64 {
    let attempted: u64 = operations.values().map(|summary| summary.attempted).sum();
    let rejected: u64 = operations.values().map(|summary| summary.rejected).sum();
    if attempted == 0 {
        0.0
    } else {
        rejected as f64 / attempted as f64
    }
}

/// Compares named p95 distributions and treats a missing series as regression.
fn compare_distributions(
    result: &mut ComparisonReport,
    scenario: &str,
    family: &str,
    baseline: &BTreeMap<String, Distribution>,
    candidate: &BTreeMap<String, Distribution>,
    threshold: f64,
) {
    let names: BTreeSet<_> = baseline.keys().chain(candidate.keys()).collect();
    for name in names {
        match (baseline.get(name), candidate.get(name)) {
            (Some(baseline), Some(candidate)) => compare_distribution(
                result,
                scenario,
                &format!("{family} {name}"),
                baseline,
                candidate,
                threshold,
            ),
            _ => {
                let message = format!("{scenario}: {family} {name} availability changed");
                if !result.regressions.contains(&message) {
                    result.regressions.push(message);
                }
            }
        }
    }
}

/// Compares both usage and exhaustion for every public Search Budget.
fn compare_budgets(
    result: &mut ComparisonReport,
    scenario: &str,
    baseline: &BTreeMap<String, BudgetSummary>,
    candidate: &BTreeMap<String, BudgetSummary>,
    threshold: f64,
) {
    let names: BTreeSet<_> = baseline.keys().chain(candidate.keys()).collect();
    for name in names {
        match (baseline.get(name), candidate.get(name)) {
            (Some(baseline), Some(candidate)) => {
                relative_regression(
                    result,
                    scenario,
                    &format!("search budget {name} usage p95"),
                    baseline.usage.p95,
                    candidate.usage.p95,
                    threshold,
                );
                relative_regression(
                    result,
                    scenario,
                    &format!("search budget {name} exhausted searches"),
                    baseline.exhausted_searches as f64,
                    candidate.exhausted_searches as f64,
                    threshold,
                );
            }
            _ => {
                let message = format!("{scenario}: search budget {name} availability changed");
                if !result.regressions.contains(&message) {
                    result.regressions.push(message);
                }
            }
        }
    }
}

/// Compares miss/stale-miss ratio, installation work, and accounted capacity.
fn compare_cache(
    result: &mut ComparisonReport,
    scenario: &str,
    baseline: &CacheSummary,
    candidate: &CacheSummary,
    threshold: f64,
) {
    relative_regression(
        result,
        scenario,
        "cache miss ratio",
        cache_miss_ratio(&baseline.lookups),
        cache_miss_ratio(&candidate.lookups),
        threshold,
    );
    relative_regression(
        result,
        scenario,
        "cache installations",
        baseline.installs.values().sum::<u64>() as f64,
        candidate.installs.values().sum::<u64>() as f64,
        threshold,
    );
    compare_optional_resource(
        result,
        scenario,
        "cache accounted bytes",
        baseline.accounted_bytes.map(|value| value as f64),
        candidate.accounted_bytes.map(|value| value as f64),
        threshold,
    );
}

/// Returns the fraction of lookups that failed to yield a reusable entry.
fn cache_miss_ratio(lookups: &BTreeMap<String, u64>) -> f64 {
    let total = lookups.values().sum::<u64>();
    if total == 0 {
        return 0.0;
    }
    let misses = lookups
        .iter()
        .filter(|(labels, _)| {
            labels.contains("result=miss") || labels.contains("result=stale_miss")
        })
        .map(|(_, count)| *count)
        .sum::<u64>();
    misses as f64 / total as f64
}

/// Compares Backend blocking and Import admission p95 distributions.
fn compare_admission(
    result: &mut ComparisonReport,
    scenario: &str,
    baseline: &AdmissionSummary,
    candidate: &AdmissionSummary,
    threshold: f64,
) {
    for (name, baseline, candidate) in [
        (
            "blocking wait",
            &baseline.blocking_wait_ms,
            &candidate.blocking_wait_ms,
        ),
        (
            "blocking held",
            &baseline.blocking_held_ms,
            &candidate.blocking_held_ms,
        ),
    ] {
        compare_distribution(result, scenario, name, baseline, candidate, threshold);
    }
    compare_distributions(
        result,
        scenario,
        "import admission",
        &baseline.import_wait_ms,
        &candidate.import_wait_ms,
        threshold,
    );
}

/// Compares one distribution without treating a missing series as zero work.
fn compare_distribution(
    result: &mut ComparisonReport,
    scenario: &str,
    metric: &str,
    baseline: &Distribution,
    candidate: &Distribution,
    threshold: f64,
) {
    match (baseline.count > 0, candidate.count > 0) {
        (true, true) => relative_regression_with_floor(
            result,
            scenario,
            &format!("{metric} p95"),
            baseline.p95,
            candidate.p95,
            threshold,
            MINIMUM_LATENCY_REGRESSION_MS,
        ),
        (false, false) => {}
        _ => result
            .regressions
            .push(format!("{scenario}: {metric} availability changed")),
    }
}

/// Records a regression only when both relative and absolute movement matter.
fn relative_regression_with_floor(
    result: &mut ComparisonReport,
    key: &str,
    metric: &str,
    baseline: f64,
    candidate: f64,
    relative_threshold: f64,
    absolute_threshold: f64,
) {
    if candidate - baseline >= absolute_threshold
        && ((baseline == 0.0 && candidate > 0.0)
            || (baseline > 0.0 && candidate > baseline * (1.0 + relative_threshold)))
    {
        result.regressions.push(format!(
            "{key}: {metric} increased from {baseline:.4} to {candidate:.4}"
        ));
    }
}

/// Compares every Backend-boundary logical IO and commit outcome counter.
fn compare_backend_io(
    result: &mut ComparisonReport,
    scenario: &str,
    baseline: &BackendIo,
    candidate: &BackendIo,
    threshold: f64,
) {
    for (name, baseline, candidate) in [
        (
            "read transactions",
            baseline.read_transactions,
            candidate.read_transactions,
        ),
        (
            "write transactions",
            baseline.write_transactions,
            candidate.write_transactions,
        ),
        (
            "point-read keys",
            baseline.point_read_keys,
            candidate.point_read_keys,
        ),
        ("scans", baseline.scans, candidate.scans),
        ("items read", baseline.items_read, candidate.items_read),
        ("bytes read", baseline.bytes_read, candidate.bytes_read),
        (
            "mutation operations",
            baseline.mutation_operations,
            candidate.mutation_operations,
        ),
        (
            "mutation bytes",
            baseline.mutation_bytes,
            candidate.mutation_bytes,
        ),
        (
            "range clears",
            baseline.range_clears,
            candidate.range_clears,
        ),
        ("commits", baseline.commits, candidate.commits),
        (
            "retryable commits",
            baseline.retryable_commits,
            candidate.retryable_commits,
        ),
        (
            "unknown commits",
            baseline.unknown_commits,
            candidate.unknown_commits,
        ),
        (
            "failed commits",
            baseline.failed_commits,
            candidate.failed_commits,
        ),
    ] {
        relative_regression(
            result,
            scenario,
            &format!("Backend {name}"),
            baseline as f64,
            candidate as f64,
            threshold,
        );
    }
}

/// Compares an optional host observation and reports availability changes.
fn compare_optional_resource(
    result: &mut ComparisonReport,
    key: &str,
    metric: &str,
    baseline: Option<f64>,
    candidate: Option<f64>,
    threshold: f64,
) {
    match (baseline, candidate) {
        (Some(baseline), Some(candidate)) => {
            relative_regression(result, key, metric, baseline, candidate, threshold);
        }
        (None, None) => result
            .notes
            .push(format!("{key}: {metric} unavailable in both reports")),
        _ => result
            .regressions
            .push(format!("{key}: {metric} availability changed")),
    }
}

/// Records a material increase in a metric where lower is better.
fn relative_regression(
    result: &mut ComparisonReport,
    key: &str,
    metric: &str,
    baseline: f64,
    candidate: f64,
    threshold: f64,
) {
    if (baseline == 0.0 && candidate > 0.0)
        || (baseline > 0.0 && candidate > baseline * (1.0 + threshold))
    {
        result.regressions.push(format!(
            "{key}: {metric} increased from {baseline:.4} to {candidate:.4}"
        ));
    }
}

/// Records a material decrease in a metric where higher is better.
fn relative_decrease(
    result: &mut ComparisonReport,
    key: &str,
    metric: &str,
    baseline: f64,
    candidate: f64,
    threshold: f64,
) {
    if baseline > 0.0 && candidate < baseline * (1.0 - threshold) {
        result.regressions.push(format!(
            "{key}: {metric} decreased from {baseline:.4} to {candidate:.4}"
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::report::{
        BenchmarkReport, BenchmarkSuite, BudgetConfiguration, BudgetSummary, Configuration,
        DatasetMetadata, Distribution, Environment, LifecycleMeasurements, OperationClass,
        OperationSummary, REPORT_SCHEMA_VERSION, RecallSummary, ReportMeasurements,
        SearchBudgetConfiguration, SteadyStateMeasurements, WorkloadDispatch, WriteAmplification,
    };

    use super::{ComparisonPolicy, compare};

    fn budget_configuration(runtime_default: u32, effective_limit: u32) -> BudgetConfiguration {
        BudgetConfiguration {
            runtime_default,
            request_override: None,
            effective_limit,
        }
    }

    /// Produces the smallest complete report whose fields remain easy to vary.
    fn report() -> BenchmarkReport {
        BenchmarkReport {
            generated_unix_seconds: 1,
            reproduction_command: "ktann-bench run".to_owned(),
            git_revision: "revision".to_owned(),
            environment: Environment {
                operating_system: "os".to_owned(),
                cpu_model: "cpu".to_owned(),
                logical_cpus: 8,
                memory_bytes: Some(16 << 30),
                rustc: "rustc 1.85.0\nbinary: rustc".to_owned(),
                build_profile: "release".to_owned(),
                build_features: "rocksdb".to_owned(),
                rustflags: "-Ctarget-cpu=native".to_owned(),
                backend_runtime: "rocksdb=10.4.2".to_owned(),
                tokio_worker_threads: 4,
            },
            configuration: Configuration {
                backend: "rocksdb".to_owned(),
                scenario: "ann".to_owned(),
                profile: "smoke".to_owned(),
                seed: 38,
                dimension: 16,
                metric: "l2".to_owned(),
                search_percent: 100,
                hot_updates: false,
                min_partition_entries: 8,
                max_partition_entries: 32,
                partition_cache_bytes: 1024,
                foreground_limit: 8,
                maintenance_workers: 2,
                fixup_queue_capacity: 1_024,
                search_budgets: SearchBudgetConfiguration {
                    scanned_tree_keys: budget_configuration(4_096, 4_096),
                    visited_partitions: budget_configuration(1_024, 1_024),
                    visited_leaf_entries: budget_configuration(65_536, 65_536),
                    exact_rerank_candidates: budget_configuration(65_536, 100),
                },
                leaf_beam_size_override: None,
                blocking_resource_limit: Some(2),
                backend_max_mutations: 100,
                backend_max_mutation_bytes: 1_000,
                concurrency: 4,
                dispatch: WorkloadDispatch::Continuous,
                warmup_operations: 10,
                measured_operations: 100,
                k: 10,
                import_batch_size: None,
                import_in_flight_batches: None,
                import_backlog_watermark: None,
            },
            dataset: DatasetMetadata {
                name: "clustered".to_owned(),
                base_vectors: 100,
                query_vectors: 10,
                dimension: 16,
                checksum_xxh3_128: "checksum".to_owned(),
            },
            topology: Default::default(),
            measurements: ReportMeasurements::SteadyState(SteadyStateMeasurements {
                cpu_seconds: Some(1.0),
                peak_rss_bytes: Some(1_000),
                throughput_per_second: 100.0,
                operations: BTreeMap::from([(
                    OperationClass::Search,
                    OperationSummary {
                        attempted: 100,
                        accepted: 100,
                        latency_ms: Distribution {
                            count: 100,
                            p95: 10.0,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )]),
                recall_at_k: Some(RecallSummary {
                    queries: 10,
                    mean: 0.9,
                    min: 0.8,
                }),
                write_amplification: Some(WriteAmplification {
                    successful_writes: 10,
                    logical_mutations_per_write: 5.0,
                    logical_bytes_per_write: 500.0,
                    write_retries: 0,
                }),
                ..Default::default()
            }),
        }
    }

    fn steady_mut(report: &mut BenchmarkReport) -> &mut SteadyStateMeasurements {
        let ReportMeasurements::SteadyState(measurements) = &mut report.measurements else {
            panic!("fixture is not steady-state")
        };
        measurements
    }

    fn suite(report: BenchmarkReport) -> BenchmarkSuite {
        BenchmarkSuite {
            schema_version: REPORT_SCHEMA_VERSION,
            reproduction_command: "ktann-bench run".to_owned(),
            reports: vec![report],
        }
    }

    /// Updates the mutually exclusive operation outcomes as one valid summary.
    fn set_operation_outcomes(
        summary: &mut OperationSummary,
        accepted: u64,
        rejected: u64,
        other_failures: u64,
    ) {
        assert_eq!(accepted + rejected + other_failures, summary.attempted);
        summary.accepted = accepted;
        summary.rejected = rejected;
        summary.latency_ms.count = accepted;
        summary.errors.clear();
        if rejected > 0 {
            summary.errors.insert("LimitExceeded".to_owned(), rejected);
        }
        if other_failures > 0 {
            summary.errors.insert("Backend".to_owned(), other_failures);
        }
    }

    #[test]
    fn rejects_different_runtime_or_workload_inputs() {
        let baseline = suite(report());
        let mut changed_budget = report();
        changed_budget.configuration.backend_max_mutations += 1;
        assert!(
            compare(
                &baseline,
                &suite(changed_budget),
                ComparisonPolicy::default()
            )
            .is_err()
        );

        let mut changed_traversal = report();
        changed_traversal.configuration.leaf_beam_size_override = Some(16);
        assert!(
            compare(
                &baseline,
                &suite(changed_traversal),
                ComparisonPolicy::default()
            )
            .is_err()
        );

        let mut changed_runtime = report();
        changed_runtime.environment.tokio_worker_threads += 1;
        assert!(
            compare(
                &baseline,
                &suite(changed_runtime),
                ComparisonPolicy::default()
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_internally_inconsistent_operation_results() {
        let baseline = suite(report());
        let mut inconsistent = report();
        steady_mut(&mut inconsistent)
            .operations
            .get_mut(&OperationClass::Search)
            .expect("operation fixture")
            .accepted = 99;

        assert!(compare(&baseline, &suite(inconsistent), ComparisonPolicy::default()).is_err());
    }

    #[test]
    fn reports_material_resource_and_throughput_regressions() {
        let baseline = suite(report());
        let mut candidate = report();
        let measurements = steady_mut(&mut candidate);
        measurements.throughput_per_second = 70.0;
        measurements.cpu_seconds = Some(1.3);
        measurements.peak_rss_bytes = Some(1_300);
        measurements
            .write_amplification
            .as_mut()
            .expect("fixture has write amplification")
            .logical_bytes_per_write = 700.0;

        let comparison = compare(&baseline, &suite(candidate), ComparisonPolicy::default())
            .expect("reports are comparable");
        assert_eq!(comparison.regressions.len(), 4);
    }

    #[test]
    fn recall_uses_an_absolute_drop_threshold() {
        let baseline = suite(report());
        let mut candidate = report();
        steady_mut(&mut candidate)
            .recall_at_k
            .as_mut()
            .expect("fixture has recall")
            .mean = 0.87;

        let comparison = compare(&baseline, &suite(candidate), ComparisonPolicy::default())
            .expect("reports are comparable");
        assert_eq!(comparison.regressions.len(), 1);
        assert!(comparison.regressions[0].contains("mean recall dropped"));
    }

    #[test]
    fn lifecycle_comparison_catches_minimum_recall_and_failure_category_shifts() {
        let mut baseline_report = report();
        let mut baseline_lifecycle = LifecycleMeasurements::default();
        for phase in [
            &mut baseline_lifecycle.immediate_search,
            &mut baseline_lifecycle.stable_cold_search,
            &mut baseline_lifecycle.stable_warm_search,
        ] {
            phase.search.attempted = 100;
            phase.search.accepted = 100;
            phase.search.latency_ms.count = 100;
        }
        baseline_lifecycle.stable_warm_search.recall_at_k = Some(RecallSummary {
            queries: 10,
            mean: 0.9,
            min: 0.8,
        });
        baseline_lifecycle
            .import
            .batch_failures
            .insert("Backend".to_owned(), 1);
        baseline_report.measurements = ReportMeasurements::Lifecycle(baseline_lifecycle);

        let baseline = suite(baseline_report.clone());
        let mut candidate = baseline_report;
        let ReportMeasurements::Lifecycle(lifecycle) = &mut candidate.measurements else {
            panic!("fixture is not lifecycle")
        };
        lifecycle
            .stable_warm_search
            .recall_at_k
            .as_mut()
            .expect("recall fixture")
            .min = 0.7;
        lifecycle.import.batch_failures.clear();
        lifecycle
            .import
            .batch_failures
            .insert("CommitOutcomeUnknown".to_owned(), 1);

        let comparison = compare(&baseline, &suite(candidate), ComparisonPolicy::default())
            .expect("reports are comparable");
        assert!(
            comparison
                .regressions
                .iter()
                .any(|regression| regression.contains("minimum recall dropped"))
        );
        assert!(
            comparison
                .regressions
                .iter()
                .any(|regression| regression.contains("CommitOutcomeUnknown"))
        );
    }

    #[test]
    fn aggregate_rejection_rate_uses_an_absolute_noise_threshold() {
        let baseline = suite(report());
        let mut noisy = report();
        let noisy_search = steady_mut(&mut noisy)
            .operations
            .get_mut(&OperationClass::Search)
            .expect("operation fixture");
        set_operation_outcomes(noisy_search, 96, 4, 0);
        assert!(
            compare(&baseline, &suite(noisy), ComparisonPolicy::default())
                .expect("reports are comparable")
                .regressions
                .is_empty()
        );

        let mut material = report();
        let material_search = steady_mut(&mut material)
            .operations
            .get_mut(&OperationClass::Search)
            .expect("operation fixture");
        set_operation_outcomes(material_search, 94, 6, 0);
        let comparison = compare(&baseline, &suite(material), ComparisonPolicy::default())
            .expect("reports are comparable");
        assert!(
            comparison
                .regressions
                .iter()
                .any(|regression| regression.contains("rejection rate"))
        );
    }

    #[test]
    fn opposing_per_class_admission_noise_does_not_change_the_aggregate() {
        let mut baseline_report = report();
        baseline_report.configuration.search_percent = 50;
        let search = steady_mut(&mut baseline_report)
            .operations
            .get_mut(&OperationClass::Search)
            .expect("search fixture");
        search.attempted = 50;
        set_operation_outcomes(search, 25, 25, 0);
        let mut write = OperationSummary {
            attempted: 50,
            ..Default::default()
        };
        set_operation_outcomes(&mut write, 25, 25, 0);
        steady_mut(&mut baseline_report)
            .operations
            .insert(OperationClass::Write, write);

        let baseline = suite(baseline_report.clone());
        let mut candidate = baseline_report;
        set_operation_outcomes(
            steady_mut(&mut candidate)
                .operations
                .get_mut(&OperationClass::Search)
                .expect("search fixture"),
            21,
            29,
            0,
        );
        set_operation_outcomes(
            steady_mut(&mut candidate)
                .operations
                .get_mut(&OperationClass::Write)
                .expect("write fixture"),
            29,
            21,
            0,
        );

        assert!(
            compare(&baseline, &suite(candidate), ComparisonPolicy::default())
                .expect("reports are comparable")
                .regressions
                .is_empty()
        );
    }

    #[test]
    fn latency_requires_material_relative_and_absolute_movement() {
        let baseline = suite(report());
        let mut noisy = report();
        steady_mut(&mut noisy)
            .operations
            .get_mut(&OperationClass::Search)
            .expect("operation fixture")
            .latency_ms
            .p95 = 10.9;
        assert!(
            compare(&baseline, &suite(noisy), ComparisonPolicy::default())
                .expect("reports are comparable")
                .regressions
                .is_empty()
        );

        let mut material = report();
        steady_mut(&mut material)
            .operations
            .get_mut(&OperationClass::Search)
            .expect("operation fixture")
            .latency_ms
            .p95 = 13.0;
        assert!(
            compare(&baseline, &suite(material), ComparisonPolicy::default())
                .expect("reports are comparable")
                .regressions
                .iter()
                .any(|regression| regression.contains("search latency"))
        );
    }

    #[test]
    fn reports_disappearing_scenario_measurements() {
        let mut baseline_report = report();
        steady_mut(&mut baseline_report)
            .backend_admission
            .blocking_wait_ms = Distribution {
            count: 1,
            p95: 1.0,
            ..Default::default()
        };
        let baseline = suite(baseline_report);
        let mut candidate = report();
        let measurements = steady_mut(&mut candidate);
        measurements.cpu_seconds = None;
        measurements.recall_at_k = None;
        measurements.write_amplification = None;

        let comparison = compare(&baseline, &suite(candidate), ComparisonPolicy::default())
            .expect("reports are comparable");
        assert_eq!(comparison.regressions.len(), 4);
        assert!(
            comparison
                .regressions
                .iter()
                .any(|regression| regression.contains("CPU seconds availability"))
        );
        assert!(
            comparison
                .regressions
                .iter()
                .any(|regression| regression.contains("recall availability"))
        );
        assert!(
            comparison
                .regressions
                .iter()
                .any(|regression| regression.contains("write amplification availability"))
        );
        assert!(
            comparison
                .regressions
                .iter()
                .any(|regression| regression.contains("blocking wait availability"))
        );
    }

    #[test]
    fn compares_each_issue_38_measurement_family() {
        let mut baseline_report = report();
        let measurements = steady_mut(&mut baseline_report);
        measurements.maintenance_drain_seconds = 1.0;
        measurements.search_stages_ms.insert(
            "exact_reranking".to_owned(),
            Distribution {
                count: 1,
                p95: 1.0,
                ..Default::default()
            },
        );
        measurements.search_budgets.insert(
            "visited_partitions".to_owned(),
            BudgetSummary {
                usage: Distribution {
                    p95: 10.0,
                    ..Default::default()
                },
                exhausted_searches: 0,
            },
        );
        measurements
            .cache
            .lookups
            .insert("level=leaf,result=hit".to_owned(), 100);
        measurements.backend_admission.blocking_wait_ms = Distribution {
            count: 1,
            p95: 1.0,
            ..Default::default()
        };
        measurements.backend_io.bytes_read = 100;

        let baseline = suite(baseline_report.clone());
        let mut candidate = baseline_report;
        let measurements = steady_mut(&mut candidate);
        measurements.maintenance_drain_seconds = 2.0;
        let operation = measurements
            .operations
            .get_mut(&OperationClass::Search)
            .expect("operation fixture");
        set_operation_outcomes(operation, 99, 0, 1);
        measurements
            .search_stages_ms
            .get_mut("exact_reranking")
            .expect("stage fixture")
            .p95 = 2.0;
        measurements
            .search_budgets
            .get_mut("visited_partitions")
            .expect("budget fixture")
            .exhausted_searches = 1;
        measurements
            .cache
            .lookups
            .insert("level=leaf,result=miss".to_owned(), 1);
        measurements.backend_admission.blocking_wait_ms.p95 = 2.0;
        measurements.backend_io.bytes_read = 200;

        let comparison = compare(&baseline, &suite(candidate), ComparisonPolicy::default())
            .expect("reports are comparable");
        for expected in [
            "maintenance drain",
            "search error Backend",
            "search stage exact_reranking",
            "exhausted searches",
            "cache miss ratio",
            "blocking wait",
            "Backend bytes read",
        ] {
            assert!(
                comparison
                    .regressions
                    .iter()
                    .any(|regression| regression.contains(expected)),
                "missing regression for {expected}: {:?}",
                comparison.regressions
            );
        }
    }
}
