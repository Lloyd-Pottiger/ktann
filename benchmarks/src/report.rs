//! Versioned benchmark report schema and distribution summaries.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Current on-disk report schema.
pub const REPORT_SCHEMA_VERSION: u32 = 1;

/// Reports produced by one suite command on one comparable host.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkSuite {
    /// Report schema version.
    pub schema_version: u32,
    /// Exact command that reproduces the suite.
    pub reproduction_command: String,
    /// Scenario reports, each measured in an isolated worker process.
    pub reports: Vec<BenchmarkReport>,
}

/// One complete, reproducible benchmark result.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkReport {
    /// Unix timestamp at report creation.
    pub generated_unix_seconds: u64,
    /// Exact shell command that reproduces the run.
    pub reproduction_command: String,
    /// Git revision measured by the worker.
    pub git_revision: String,
    /// Hardware and runtime facts.
    pub environment: Environment,
    /// Benchmark inputs and process-local limits.
    pub configuration: Configuration,
    /// Fixed input data identity.
    pub dataset: DatasetMetadata,
    /// Stable topology facts after setup.
    pub topology: Topology,
    /// Measurements collected outside setup and warmup.
    pub measurements: Measurements,
}

/// Hardware and software metadata needed to judge comparability.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Environment {
    /// Operating-system and kernel description.
    pub operating_system: String,
    /// CPU model reported by the host.
    pub cpu_model: String,
    /// Logical CPU count visible to the process.
    pub logical_cpus: usize,
    /// Installed memory in bytes when available.
    pub memory_bytes: Option<u64>,
    /// Build-time `rustc --version --verbose` output.
    pub rustc: String,
    /// Cargo profile used to compile the benchmark executable.
    pub build_profile: String,
    /// Additive Cargo features compiled into the benchmark executable.
    pub build_features: String,
    /// Rust code-generation flags applied at build time.
    pub rustflags: String,
    /// Storage-engine client and server identity.
    pub backend_runtime: String,
    /// Tokio worker-thread count used by the benchmark.
    pub tokio_worker_threads: usize,
}

/// Reproducible benchmark configuration.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Configuration {
    /// Production Backend under measurement.
    pub backend: String,
    /// Workload scenario.
    pub scenario: String,
    /// Dataset/operation scale.
    pub profile: String,
    /// Replayable random seed.
    pub seed: u64,
    /// Exact vector dimension accepted by the Logical Index.
    pub dimension: usize,
    /// Stable distance metric name.
    pub metric: String,
    /// Percentage of workload operations that are searches.
    pub search_percent: u8,
    /// Whether writes target a small conflict set.
    pub hot_updates: bool,
    /// Logical Index merge threshold.
    pub min_partition_entries: u32,
    /// Logical Index split threshold.
    pub max_partition_entries: u32,
    /// Runtime Partition Cache capacity.
    pub partition_cache_bytes: u64,
    /// Foreground operation concurrency limit.
    pub foreground_limit: usize,
    /// Background Structure Maintenance worker count.
    pub maintenance_workers: usize,
    /// Pending-plus-running Fixup capacity.
    pub fixup_queue_capacity: usize,
    /// Runtime defaults, request overrides, and effective per-search limits.
    pub search_budgets: SearchBudgetConfiguration,
    /// Per-request leaf-level base beam override, when present.
    pub leaf_beam_size_override: Option<u32>,
    /// RocksDB native blocking-resource limit, when applicable.
    pub blocking_resource_limit: Option<usize>,
    /// Backend Admission Budget mutation-count ceiling.
    pub backend_max_mutations: usize,
    /// Backend Admission Budget mutation-byte ceiling.
    pub backend_max_mutation_bytes: usize,
    /// Concurrent workload clients.
    pub concurrency: usize,
    /// How clients are dispatched during the measured workload.
    pub dispatch: WorkloadDispatch,
    /// Operations executed before measurement.
    pub warmup_operations: usize,
    /// Operations included in the report.
    pub measured_operations: usize,
    /// Requested result count.
    pub k: usize,
}

/// Configuration and effective limit for one public Search Budget dimension.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BudgetConfiguration {
    /// Runtime default from which requests resolve this dimension.
    pub runtime_default: u32,
    /// Explicit per-request override used by the scenario, when present.
    pub request_override: Option<u32>,
    /// Concrete limit applied to every measured search in the scenario.
    pub effective_limit: u32,
}

/// Configuration for every public Search Budget dimension.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SearchBudgetConfiguration {
    /// Scanned Tree Key limit configuration.
    pub scanned_tree_keys: BudgetConfiguration,
    /// Visited partition limit configuration.
    pub visited_partitions: BudgetConfiguration,
    /// Visited Leaf Entry limit configuration.
    pub visited_leaf_entries: BudgetConfiguration,
    /// Exact-rerank candidate limit configuration.
    pub exact_rerank_candidates: BudgetConfiguration,
}

/// Identity and size of one fixed public or synthetic dataset.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DatasetMetadata {
    /// Canonical dataset/distribution name.
    pub name: String,
    /// Number of indexed vectors.
    pub base_vectors: usize,
    /// Number of held-out query vectors.
    pub query_vectors: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Stable xxh3-128 checksum over vector bits and IDs.
    pub checksum_xxh3_128: String,
}

/// Persistent topology facts after setup convergence.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Topology {
    /// Verified Vector Record count.
    pub vector_records: u64,
    /// Verified partition count.
    pub partitions: u64,
    /// Verified internal and Leaf Entry count.
    pub entries: u64,
}

/// Measurements attributed to one isolated scenario worker.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Measurements {
    /// Timed-region wall-clock duration.
    pub wall_seconds: f64,
    /// Time after foreground completion until its maintenance backlog drained.
    pub maintenance_drain_seconds: f64,
    /// User/system CPU through foreground work and maintenance drain.
    pub cpu_seconds: Option<f64>,
    /// Whole-worker peak RSS, including setup, warmup, and measurement.
    pub peak_rss_bytes: Option<u64>,
    /// Successful operations per wall-clock second.
    pub throughput_per_second: f64,
    /// Attempt, outcome, and successful-latency results by public operation class.
    pub operations: BTreeMap<OperationClass, OperationSummary>,
    /// Recall@k for successful searches.
    pub recall_at_k: Option<RecallSummary>,
    /// Search budget use for every public dimension.
    pub search_budgets: BTreeMap<String, BudgetSummary>,
    /// Approximate-selection and exact-reranking stage latency.
    pub search_stages_ms: BTreeMap<String, Distribution>,
    /// Partition Cache observations.
    pub cache: CacheSummary,
    /// Backend blocking/admission observations.
    pub backend_admission: AdmissionSummary,
    /// Backend-neutral logical KV work attempted in the timed region.
    pub backend_io: BackendIo,
    /// Logical KV work per successful write operation.
    pub write_amplification: Option<WriteAmplification>,
}

/// Scheduling shape used to apply bounded client pressure.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadDispatch {
    /// Replenish one client whenever any operation completes.
    Continuous,
    /// Submit fixed concurrent waves and let each wave drain before the next.
    FixedWaves(AdmissionTarget),
}

/// Public operation classes reported independently by mixed workloads.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationClass {
    /// Approximate search followed by exact reranking.
    Search,
    /// One replacement upsert.
    Write,
}

impl OperationClass {
    /// Returns the stable report key for this operation class.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Search => "search",
            Self::Write => "write",
        }
    }

    /// Returns the operation classes present in a search/write mix.
    pub(crate) fn for_mix(search_percent: u8) -> impl Iterator<Item = Self> {
        [
            (search_percent > 0).then_some(Self::Search),
            (search_percent < 100).then_some(Self::Write),
        ]
        .into_iter()
        .flatten()
    }
}

/// Observable operating region required from an admission-pressure scenario.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct AdmissionTarget {
    /// Successful samples required for every attempted operation class.
    pub minimum_accepted_per_class: u64,
    /// Inclusive lower bound for the overall admission-rejection rate.
    pub minimum_rejection_rate: f64,
    /// Inclusive upper bound for the overall admission-rejection rate.
    pub maximum_rejection_rate: f64,
}

/// Outcomes and successful latency for one public operation class.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct OperationSummary {
    /// Public calls issued in the measured region.
    pub attempted: u64,
    /// Calls that returned success.
    pub accepted: u64,
    /// Calls rejected by the Runtime foreground admission bound.
    pub rejected: u64,
    /// All failures grouped by stable KTANN error category.
    pub errors: BTreeMap<String, u64>,
    /// End-to-end latency of accepted calls in milliseconds.
    pub latency_ms: Distribution,
}

impl OperationSummary {
    /// Returns failures not attributed to Runtime foreground admission.
    pub(crate) fn non_admission_failures(&self) -> u64 {
        self.attempted
            .checked_sub(self.accepted)
            .and_then(|failures| failures.checked_sub(self.rejected))
            .expect("validated operation outcomes fit within attempts")
    }

    /// Checks that serialized outcome counts agree with each other.
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        let failures = self
            .attempted
            .checked_sub(self.accepted)
            .ok_or("accepted operations exceed attempts")?;
        let error_count = self
            .errors
            .values()
            .try_fold(0_u64, |sum, count| sum.checked_add(*count))
            .ok_or("operation error count overflow")?;
        if failures != error_count {
            return Err("operation errors do not account for failed attempts");
        }
        if self.rejected
            > self
                .errors
                .get("LimitExceeded")
                .copied()
                .unwrap_or_default()
        {
            return Err("admission rejections exceed limit-exceeded failures");
        }
        if self.latency_ms.count != self.accepted {
            return Err("accepted operation count differs from latency samples");
        }
        Ok(())
    }
}

/// A finite sample distribution.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Distribution {
    /// Number of observations.
    pub count: u64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Minimum.
    pub min: f64,
    /// Median.
    pub p50: f64,
    /// 95th percentile, nearest-rank.
    pub p95: f64,
    /// 99th percentile, nearest-rank.
    pub p99: f64,
    /// Maximum.
    pub max: f64,
}

impl Distribution {
    /// Summarizes finite samples in their current unit.
    #[must_use]
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        samples.retain(|sample| sample.is_finite());
        if samples.is_empty() {
            return Self::default();
        }
        samples.sort_by(f64::total_cmp);
        let count = samples.len();
        let sum: f64 = samples.iter().sum();
        Self {
            count: count as u64,
            mean: sum / count as f64,
            min: samples[0],
            p50: percentile(&samples, 50),
            p95: percentile(&samples, 95),
            p99: percentile(&samples, 99),
            max: samples[count - 1],
        }
    }

    /// Converts a seconds distribution into milliseconds.
    #[must_use]
    pub fn seconds_to_milliseconds(mut self) -> Self {
        for value in [
            &mut self.mean,
            &mut self.min,
            &mut self.p50,
            &mut self.p95,
            &mut self.p99,
            &mut self.max,
        ] {
            *value *= 1_000.0;
        }
        self
    }
}

/// Returns the nearest-rank percentile from finite samples sorted ascending.
fn percentile(samples: &[f64], percentile: usize) -> f64 {
    let rank = percentile
        .saturating_mul(samples.len())
        .div_ceil(100)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[rank]
}

/// Recall distribution across successful queries.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RecallSummary {
    /// Number of queries evaluated.
    pub queries: u64,
    /// Mean recall@k.
    pub mean: f64,
    /// Minimum per-query recall@k.
    pub min: f64,
}

/// One public Search Budget's use and exhaustion.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct BudgetSummary {
    /// Work charged per successful search.
    pub usage: Distribution,
    /// Searches where this budget prevented eligible work.
    pub exhausted_searches: u64,
}

/// Partition Cache lookup and capacity observations.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CacheSummary {
    /// Lookups by `level/result`.
    pub lookups: BTreeMap<String, u64>,
    /// Installs by `level/result`.
    pub installs: BTreeMap<String, u64>,
    /// Last reported accounted cache bytes.
    pub accounted_bytes: Option<u64>,
}

/// Backend resource admission distributions.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AdmissionSummary {
    /// RocksDB wait for a native actor slot, milliseconds.
    pub blocking_wait_ms: Distribution,
    /// RocksDB native actor hold time, milliseconds.
    pub blocking_held_ms: Distribution,
    /// Import gate waits, milliseconds, by gate.
    pub import_wait_ms: BTreeMap<String, Distribution>,
}

/// Backend-neutral attempted logical KV work.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct BackendIo {
    /// Read transactions opened.
    pub read_transactions: u64,
    /// Write transactions opened, including retries.
    pub write_transactions: u64,
    /// Point and batch point-read keys requested.
    pub point_read_keys: u64,
    /// Range scan calls.
    pub scans: u64,
    /// Logical KV items returned by reads.
    pub items_read: u64,
    /// Logical key/value bytes returned by reads.
    pub bytes_read: u64,
    /// Attempted point mutations, including retry attempts.
    pub mutation_operations: u64,
    /// Attempted logical mutation key/value bytes.
    pub mutation_bytes: u64,
    /// Attempted transactional range clears.
    pub range_clears: u64,
    /// Definitely committed write transactions.
    pub commits: u64,
    /// Definitely retryable commit outcomes.
    pub retryable_commits: u64,
    /// Unknown commit outcomes.
    pub unknown_commits: u64,
    /// Other failed commit outcomes.
    pub failed_commits: u64,
}

/// Logical write amplification derived from Backend calls.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct WriteAmplification {
    /// Successful user-visible writes.
    pub successful_writes: u64,
    /// Attempted logical KV mutation operations per successful write.
    pub logical_mutations_per_write: f64,
    /// Attempted logical mutation bytes per successful write.
    pub logical_bytes_per_write: f64,
    /// Whole-operation retries observed by KTANN.
    pub write_retries: u64,
}

#[cfg(test)]
mod tests {
    use super::Distribution;

    #[test]
    fn distribution_uses_nearest_rank_percentiles() {
        let distribution = Distribution::from_samples((1..=100).map(|n| n as f64).collect());
        assert_eq!(distribution.count, 100);
        assert_eq!(distribution.p50, 50.0);
        assert_eq!(distribution.p95, 95.0);
        assert_eq!(distribution.p99, 99.0);
    }
}
