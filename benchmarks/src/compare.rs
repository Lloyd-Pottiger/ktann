//! Material-regression comparison for reports from equivalent environments.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::report::{
    AdmissionSummary, BackendIo, BenchmarkReport, BenchmarkSuite, BudgetSummary, CacheSummary,
    Distribution, REPORT_SCHEMA_VERSION,
};

/// Explicit policy for distinguishing material movement from ordinary noise.
#[derive(Clone, Copy, Debug)]
pub struct ComparisonPolicy {
    /// Maximum relative increase for latency, CPU, RSS, and logical work.
    pub maximum_relative_regression: f64,
    /// Maximum absolute decrease in recall@k.
    pub maximum_recall_drop: f64,
}

impl Default for ComparisonPolicy {
    fn default() -> Self {
        Self {
            maximum_relative_regression: 0.20,
            maximum_recall_drop: 0.02,
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
    relative_regression(
        result,
        key,
        "p95 latency",
        baseline.measurements.latency_ms.p95,
        candidate.measurements.latency_ms.p95,
        policy.maximum_relative_regression,
    );
    relative_decrease(
        result,
        key,
        "throughput",
        baseline.measurements.throughput_per_second,
        candidate.measurements.throughput_per_second,
        policy.maximum_relative_regression,
    );
    compare_optional_resource(
        result,
        key,
        "CPU seconds",
        baseline.measurements.cpu_seconds,
        candidate.measurements.cpu_seconds,
        policy.maximum_relative_regression,
    );
    relative_regression(
        result,
        key,
        "maintenance drain seconds",
        baseline.measurements.maintenance_drain_seconds,
        candidate.measurements.maintenance_drain_seconds,
        policy.maximum_relative_regression,
    );
    relative_regression(
        result,
        key,
        "failed operations",
        baseline.measurements.failed_operations as f64,
        candidate.measurements.failed_operations as f64,
        policy.maximum_relative_regression,
    );
    compare_distributions(
        result,
        key,
        "search stage",
        &baseline.measurements.search_stages_ms,
        &candidate.measurements.search_stages_ms,
        policy.maximum_relative_regression,
    );
    compare_budgets(
        result,
        key,
        &baseline.measurements.search_budgets,
        &candidate.measurements.search_budgets,
        policy.maximum_relative_regression,
    );
    compare_cache(
        result,
        key,
        &baseline.measurements.cache,
        &candidate.measurements.cache,
        policy.maximum_relative_regression,
    );
    compare_admission(
        result,
        key,
        &baseline.measurements.backend_admission,
        &candidate.measurements.backend_admission,
        policy.maximum_relative_regression,
    );
    compare_backend_io(
        result,
        key,
        &baseline.measurements.backend_io,
        &candidate.measurements.backend_io,
        policy.maximum_relative_regression,
    );
    compare_optional_resource(
        result,
        key,
        "peak RSS bytes",
        baseline
            .measurements
            .peak_rss_bytes
            .map(|value| value as f64),
        candidate
            .measurements
            .peak_rss_bytes
            .map(|value| value as f64),
        policy.maximum_relative_regression,
    );
    match (
        baseline.measurements.recall_at_k.as_ref(),
        candidate.measurements.recall_at_k.as_ref(),
    ) {
        (Some(baseline), Some(candidate)) => {
            if baseline.mean - candidate.mean > policy.maximum_recall_drop {
                result.regressions.push(format!(
                    "{key}: mean recall dropped from {:.4} to {:.4}",
                    baseline.mean, candidate.mean
                ));
            }
        }
        (None, None) => {}
        _ => result
            .regressions
            .push(format!("{key}: recall availability changed")),
    }
    match (
        baseline.measurements.write_amplification.as_ref(),
        candidate.measurements.write_amplification.as_ref(),
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
        (true, true) => relative_regression(
            result,
            scenario,
            &format!("{metric} p95"),
            baseline.p95,
            candidate.p95,
            threshold,
        ),
        (false, false) => {}
        _ => result
            .regressions
            .push(format!("{scenario}: {metric} availability changed")),
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
    use crate::report::{
        BenchmarkReport, BenchmarkSuite, BudgetConfiguration, BudgetSummary, Configuration,
        DatasetMetadata, Distribution, Environment, Measurements, REPORT_SCHEMA_VERSION,
        RecallSummary, SearchBudgetConfiguration, WriteAmplification,
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
                search_percent: 95,
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
                warmup_operations: 10,
                measured_operations: 100,
                k: 10,
            },
            dataset: DatasetMetadata {
                name: "clustered".to_owned(),
                base_vectors: 100,
                query_vectors: 10,
                dimension: 16,
                checksum_xxh3_128: "checksum".to_owned(),
            },
            topology: Default::default(),
            measurements: Measurements {
                cpu_seconds: Some(1.0),
                peak_rss_bytes: Some(1_000),
                throughput_per_second: 100.0,
                latency_ms: Distribution {
                    p95: 10.0,
                    ..Default::default()
                },
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
            },
        }
    }

    fn suite(report: BenchmarkReport) -> BenchmarkSuite {
        BenchmarkSuite {
            schema_version: REPORT_SCHEMA_VERSION,
            reproduction_command: "ktann-bench run".to_owned(),
            reports: vec![report],
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
    fn reports_material_resource_and_throughput_regressions() {
        let baseline = suite(report());
        let mut candidate = report();
        candidate.measurements.latency_ms.p95 = 13.0;
        candidate.measurements.throughput_per_second = 70.0;
        candidate.measurements.cpu_seconds = Some(1.3);
        candidate.measurements.peak_rss_bytes = Some(1_300);
        candidate
            .measurements
            .write_amplification
            .as_mut()
            .expect("fixture has write amplification")
            .logical_bytes_per_write = 700.0;

        let comparison = compare(&baseline, &suite(candidate), ComparisonPolicy::default())
            .expect("reports are comparable");
        assert_eq!(comparison.regressions.len(), 5);
    }

    #[test]
    fn recall_uses_an_absolute_drop_threshold() {
        let baseline = suite(report());
        let mut candidate = report();
        candidate
            .measurements
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
    fn reports_disappearing_scenario_measurements() {
        let mut baseline_report = report();
        baseline_report
            .measurements
            .backend_admission
            .blocking_wait_ms = Distribution {
            count: 1,
            p95: 1.0,
            ..Default::default()
        };
        let baseline = suite(baseline_report);
        let mut candidate = report();
        candidate.measurements.cpu_seconds = None;
        candidate.measurements.recall_at_k = None;
        candidate.measurements.write_amplification = None;

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
        baseline_report.measurements.maintenance_drain_seconds = 1.0;
        baseline_report.measurements.search_stages_ms.insert(
            "exact_reranking".to_owned(),
            Distribution {
                count: 1,
                p95: 1.0,
                ..Default::default()
            },
        );
        baseline_report.measurements.search_budgets.insert(
            "visited_partitions".to_owned(),
            BudgetSummary {
                usage: Distribution {
                    p95: 10.0,
                    ..Default::default()
                },
                exhausted_searches: 0,
            },
        );
        baseline_report
            .measurements
            .cache
            .lookups
            .insert("level=leaf,result=hit".to_owned(), 100);
        baseline_report
            .measurements
            .backend_admission
            .blocking_wait_ms = Distribution {
            count: 1,
            p95: 1.0,
            ..Default::default()
        };
        baseline_report.measurements.backend_io.bytes_read = 100;

        let baseline = suite(baseline_report.clone());
        let mut candidate = baseline_report;
        candidate.measurements.maintenance_drain_seconds = 2.0;
        candidate.measurements.failed_operations = 1;
        candidate
            .measurements
            .search_stages_ms
            .get_mut("exact_reranking")
            .expect("stage fixture")
            .p95 = 2.0;
        candidate
            .measurements
            .search_budgets
            .get_mut("visited_partitions")
            .expect("budget fixture")
            .exhausted_searches = 1;
        candidate
            .measurements
            .cache
            .lookups
            .insert("level=leaf,result=miss".to_owned(), 1);
        candidate
            .measurements
            .backend_admission
            .blocking_wait_ms
            .p95 = 2.0;
        candidate.measurements.backend_io.bytes_read = 200;

        let comparison = compare(&baseline, &suite(candidate), ComparisonPolicy::default())
            .expect("reports are comparable");
        for expected in [
            "maintenance drain",
            "failed operations",
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
