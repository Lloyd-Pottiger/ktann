//! Command-line orchestration for isolated runs and report comparison.
//!
//! The public `run` command never measures scenarios in its own process. It
//! launches one hidden worker per scenario so process-global metrics, the
//! Partition Cache, and peak RSS each have an unambiguous owner. The public
//! `compare` command accepts only complete, comparable versioned suites.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
#[cfg(feature = "foundationdb")]
use std::time::{SystemTime, UNIX_EPOCH};

use crate::compare::{self, ComparisonPolicy};
use crate::report::{BenchmarkReport, BenchmarkSuite, REPORT_SCHEMA_VERSION};
use crate::runner::{self, ScenarioSpec};

const IMPORT_LIFECYCLE_SCENARIO: &str = "import-to-search-lifecycle";

/// Parses and executes one `ktann-bench` command.
///
/// # Errors
///
/// Returns an error for invalid options, unavailable Backend features,
/// scenario failures, incomparable reports, or JSON/file IO failures.
pub fn run<I, T>(arguments: I) -> Result<(), String>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let mut arguments = arguments.into_iter().map(Into::into);
    let _program = arguments.next();
    let command = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or_else(usage)?;
    let arguments: Vec<OsString> = arguments.collect();
    match command.as_str() {
        "run" => run_suite(parse_run_options(&arguments)?),
        "compare" => compare_reports(parse_compare_options(&arguments)?),
        // Each public suite scenario executes in a fresh process so metrics
        // recorder state, Partition Cache, and peak RSS cannot leak between
        // reports. This command is intentionally absent from the public help.
        "__worker" => run_worker(parse_worker_options(&arguments)?),
        _ => Err(usage()),
    }
}

/// Suite selection and output destination.
#[derive(Clone, Debug)]
struct RunOptions {
    profile: String,
    scenario: Option<String>,
    output: Option<PathBuf>,
    execution: ExecutionOptions,
}

/// Inputs for one isolated scenario process.
#[derive(Clone, Debug)]
struct WorkerOptions {
    profile: String,
    scenario: String,
    reproduction_command: String,
    execution: ExecutionOptions,
}

/// Execution settings shared by the suite and its scenario workers.
#[derive(Clone, Debug)]
struct ExecutionOptions {
    backend: String,
    worker_threads: usize,
    write_beam_size: Option<u32>,
    refinement_rounds: Option<usize>,
    base_vectors: Option<usize>,
    query_vectors: Option<usize>,
    query_offset: Option<usize>,
    max_partition_entries: Option<u32>,
    query_concurrency: Option<Vec<usize>>,
    partition_cache_bytes: Option<u64>,
    leaf_beam_size: Option<u32>,
    rocksdb_block_cache_bytes: Option<usize>,
    warmup_operations: Option<usize>,
    rocksdb_path: Option<PathBuf>,
    reuse_index: bool,
    lifecycle: LifecycleOverrides,
}

impl ExecutionOptions {
    fn parse(values: &BTreeMap<String, String>) -> Result<Self, String> {
        let backend = required(values, "backend")?;
        if backend != "rocksdb" && backend != "foundationdb" {
            return Err("--backend must be rocksdb or foundationdb".to_owned());
        }
        if values.contains_key("reuse-index") && !values.contains_key("rocksdb-path") {
            return Err("--reuse-index requires --rocksdb-path".to_owned());
        }
        if backend != "rocksdb"
            && (values.contains_key("rocksdb-block-cache-bytes")
                || values.contains_key("rocksdb-path"))
        {
            return Err("RocksDB cache and fixture options require --backend rocksdb".to_owned());
        }
        Ok(Self {
            backend,
            worker_threads: match values.get("worker-threads") {
                Some(value) => parse_positive(value, "worker-threads")?,
                None => default_worker_threads(),
            },
            write_beam_size: values
                .get("write-beam-size")
                .map(|value| parse_positive_u32(value, "write-beam-size"))
                .transpose()?,
            refinement_rounds: values
                .get("refinement-rounds")
                .map(|value| parse_nonnegative(value, "refinement-rounds"))
                .transpose()?,
            base_vectors: values
                .get("base-vectors")
                .map(|value| parse_positive(value, "base-vectors"))
                .transpose()?,
            query_vectors: values
                .get("query-vectors")
                .map(|value| parse_positive(value, "query-vectors"))
                .transpose()?,
            query_offset: values
                .get("query-offset")
                .map(|value| parse_nonnegative(value, "query-offset"))
                .transpose()?,
            max_partition_entries: values
                .get("max-partition-entries")
                .map(|value| parse_positive_u32(value, "max-partition-entries"))
                .transpose()?,
            query_concurrency: values
                .get("query-concurrency")
                .map(|value| {
                    value
                        .split(',')
                        .map(|item| parse_positive(item, "query-concurrency"))
                        .collect()
                })
                .transpose()?,
            partition_cache_bytes: values
                .get("partition-cache-bytes")
                .map(|value| {
                    value.parse::<u64>().map_err(|_| {
                        "--partition-cache-bytes must be a nonnegative 64-bit integer".to_owned()
                    })
                })
                .transpose()?,
            rocksdb_path: values.get("rocksdb-path").map(PathBuf::from),
            reuse_index: match values.get("reuse-index").map(String::as_str) {
                None | Some("false") => false,
                Some("true") => true,
                Some(_) => return Err("--reuse-index must be true or false".to_owned()),
            },
            rocksdb_block_cache_bytes: values
                .get("rocksdb-block-cache-bytes")
                .map(|value| parse_positive(value, "rocksdb-block-cache-bytes"))
                .transpose()?,
            warmup_operations: values
                .get("warmup-operations")
                .map(|value| parse_nonnegative(value, "warmup-operations"))
                .transpose()?,
            leaf_beam_size: values
                .get("leaf-beam-size")
                .map(|value| parse_positive_u32(value, "leaf-beam-size"))
                .transpose()?,
            lifecycle: LifecycleOverrides::parse(values)?,
        })
    }
}

const SCENARIO_OPTIONS: &[&str] = &[
    "backend",
    "profile",
    "scenario",
    "worker-threads",
    "query-concurrency",
    "partition-cache-bytes",
    "leaf-beam-size",
    "rocksdb-block-cache-bytes",
    "warmup-operations",
    "rocksdb-path",
    "reuse-index",
    "write-beam-size",
    "refinement-rounds",
    "base-vectors",
    "query-vectors",
    "query-offset",
    "max-partition-entries",
    "maintenance-workers",
    "import-max-in-flight-batches",
    "import-batch-size",
    "import-backlog-watermark",
];

/// Optional diagnostic bounds applied to the selected scenario.
#[derive(Clone, Debug, Default)]
struct LifecycleOverrides {
    maintenance_workers: Option<usize>,
    max_in_flight_batches: Option<usize>,
    batch_size: Option<usize>,
    backlog_watermark: Option<usize>,
}

impl LifecycleOverrides {
    fn parse(values: &BTreeMap<String, String>) -> Result<Self, String> {
        Ok(Self {
            maintenance_workers: values
                .get("maintenance-workers")
                .map(|value| parse_nonnegative(value, "maintenance-workers"))
                .transpose()?,
            max_in_flight_batches: values
                .get("import-max-in-flight-batches")
                .map(|value| parse_positive(value, "import-max-in-flight-batches"))
                .transpose()?,
            batch_size: values
                .get("import-batch-size")
                .map(|value| parse_positive(value, "import-batch-size"))
                .transpose()?,
            backlog_watermark: values
                .get("import-backlog-watermark")
                .map(|value| parse_positive(value, "import-backlog-watermark"))
                .transpose()?,
        })
    }

    fn is_empty(&self) -> bool {
        self.maintenance_workers.is_none()
            && self.max_in_flight_batches.is_none()
            && self.batch_size.is_none()
            && self.backlog_watermark.is_none()
    }

    fn apply(&self, scenario: &mut ScenarioSpec) {
        if let Some(workers) = self.maintenance_workers {
            scenario.maintenance_workers = workers;
        }
        if let Some(maximum) = self.max_in_flight_batches {
            scenario.import_max_in_flight_batches = maximum;
        }
        if let Some(batch_size) = self.batch_size {
            scenario.import_batch_size = batch_size;
        }
        if let Some(watermark) = self.backlog_watermark {
            scenario.import_backlog_watermark = watermark;
        }
    }
}

/// Report paths and comparison thresholds.
#[derive(Clone, Debug)]
struct CompareOptions {
    baseline: PathBuf,
    candidate: PathBuf,
    output: Option<PathBuf>,
    policy: ComparisonPolicy,
}

/// Validates public suite options at the process boundary.
fn parse_run_options(arguments: &[OsString]) -> Result<RunOptions, String> {
    let values = option_map(arguments, &[SCENARIO_OPTIONS, &["output"]].concat())?;
    let profile = values
        .get("profile")
        .cloned()
        .unwrap_or_else(|| "smoke".to_owned());
    let scenario = values.get("scenario").cloned();
    let execution = ExecutionOptions::parse(&values)?;
    if execution.rocksdb_path.is_some() && (profile != "large" || scenario.is_none()) {
        return Err("--rocksdb-path requires --profile large and one --scenario".to_owned());
    }
    if profile != "large"
        && (execution.query_concurrency.is_some() || execution.leaf_beam_size.is_some())
    {
        return Err("--query-concurrency and --leaf-beam-size require --profile large".to_owned());
    }
    if !execution.lifecycle.is_empty()
        && profile != "large"
        && scenario.as_deref() != Some(IMPORT_LIFECYCLE_SCENARIO)
    {
        return Err(format!(
            "lifecycle overrides require --scenario {IMPORT_LIFECYCLE_SCENARIO}"
        ));
    }
    Ok(RunOptions {
        profile,
        scenario,
        output: values.get("output").map(PathBuf::from),
        execution,
    })
}

/// Parses the internal worker command using the same execution settings.
fn parse_worker_options(arguments: &[OsString]) -> Result<WorkerOptions, String> {
    let values = option_map(
        arguments,
        &[SCENARIO_OPTIONS, &["reproduction-command"]].concat(),
    )?;
    Ok(WorkerOptions {
        profile: required(&values, "profile")?,
        scenario: required(&values, "scenario")?,
        reproduction_command: required(&values, "reproduction-command")?,
        execution: ExecutionOptions::parse(&values)?,
    })
}

/// Validates report paths and material-regression thresholds.
fn parse_compare_options(arguments: &[OsString]) -> Result<CompareOptions, String> {
    let defaults = ComparisonPolicy::default();
    let values = option_map(
        arguments,
        &[
            "baseline",
            "candidate",
            "output",
            "maximum-relative-regression",
            "maximum-recall-drop",
            "maximum-rejection-rate-increase",
        ],
    )?;
    let maximum_relative_regression = values
        .get("maximum-relative-regression")
        .map_or(Ok(defaults.maximum_relative_regression), |value| {
            parse_number(value, "maximum-relative-regression")
        })?;
    let maximum_recall_drop = values
        .get("maximum-recall-drop")
        .map_or(Ok(defaults.maximum_recall_drop), |value| {
            parse_number(value, "maximum-recall-drop")
        })?;
    let maximum_rejection_rate_increase = values
        .get("maximum-rejection-rate-increase")
        .map_or(Ok(defaults.maximum_rejection_rate_increase), |value| {
            parse_number(value, "maximum-rejection-rate-increase")
        })?;
    Ok(CompareOptions {
        baseline: PathBuf::from(required(&values, "baseline")?),
        candidate: PathBuf::from(required(&values, "candidate")?),
        output: values.get("output").map(PathBuf::from),
        policy: ComparisonPolicy {
            maximum_relative_regression,
            maximum_recall_drop,
            maximum_rejection_rate_increase,
        },
    })
}

/// Parses strict `--key value` pairs and rejects unknown or duplicate keys.
fn option_map(
    arguments: &[OsString],
    allowed: &[&str],
) -> Result<BTreeMap<String, String>, String> {
    if !arguments.len().is_multiple_of(2) {
        return Err("every option must have a value".to_owned());
    }
    let mut values = BTreeMap::new();
    for pair in arguments.as_chunks::<2>().0.iter() {
        let key = pair[0]
            .to_str()
            .and_then(|value| value.strip_prefix("--"))
            .ok_or_else(|| "options must use --key value syntax".to_owned())?;
        let value = pair[1]
            .to_str()
            .ok_or_else(|| format!("--{key} is not valid UTF-8"))?;
        if !allowed.contains(&key) {
            return Err(format!("unknown option --{key}"));
        }
        if values.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("duplicate option --{key}"));
        }
    }
    Ok(values)
}

/// Returns one required option after the shared parser has validated its key.
fn required(values: &BTreeMap<String, String>, key: &str) -> Result<String, String> {
    values
        .get(key)
        .cloned()
        .ok_or_else(|| format!("missing --{key}"))
}

/// Renders one optional value as a forwarded worker argument.
fn option_string<T: ToString>(value: Option<T>) -> Option<String> {
    value.map(|value| value.to_string())
}

/// Parses a strictly positive host-sized bound.
fn parse_positive(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("--{name} must be a positive integer"))
}

/// Parses a positive 32-bit bound passed through to the public configuration.
fn parse_positive_u32(value: &str, name: &str) -> Result<u32, String> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("--{name} must be a positive 32-bit integer"))
}

/// Parses a nonnegative host-sized bound.
fn parse_nonnegative(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("--{name} must be a nonnegative integer"))
}

/// Parses a finite nonnegative comparison threshold.
fn parse_number(value: &str, name: &str) -> Result<f64, String> {
    value
        .parse()
        .ok()
        .filter(|value: &f64| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("--{name} must be finite and nonnegative"))
}

/// Bounds the default executor size to keep runs comparable on large hosts.
fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(2, usize::from)
        .clamp(2, 8)
}

/// Runs every selected scenario in a fresh subprocess and assembles one suite.
fn run_suite(options: RunOptions) -> Result<(), String> {
    let all = runner::scenarios(&options.profile)?;
    let selected: Vec<_> = all
        .into_iter()
        .filter(|scenario| {
            options
                .scenario
                .as_ref()
                .is_none_or(|selected| scenario.name == selected)
        })
        .collect();
    if selected.is_empty() {
        return Err(format!(
            "profile `{}` has no scenario `{}`",
            options.profile,
            options.scenario.as_deref().unwrap_or("")
        ));
    }
    let reproduction_command = render_command(std::env::args_os());
    let executable = std::env::current_exe()
        .map_err(|error| format!("resolve benchmark executable: {error}"))?;
    let mut reports = Vec::with_capacity(selected.len());
    for scenario in selected {
        let mut command = Command::new(&executable);
        command
            .arg("__worker")
            .args(["--backend", &options.execution.backend])
            .args(["--profile", &options.profile])
            .args(["--scenario", scenario.name])
            .args(["--reproduction-command", &reproduction_command])
            .args([
                "--worker-threads",
                &options.execution.worker_threads.to_string(),
            ])
            .stderr(Stdio::inherit());
        let lifecycle = &options.execution.lifecycle;
        for (name, value) in [
            (
                "rocksdb-path",
                options
                    .execution
                    .rocksdb_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
            ),
            (
                "reuse-index",
                options.execution.reuse_index.then(|| "true".to_owned()),
            ),
            (
                "rocksdb-block-cache-bytes",
                option_string(options.execution.rocksdb_block_cache_bytes),
            ),
            (
                "warmup-operations",
                option_string(options.execution.warmup_operations),
            ),
            (
                "query-concurrency",
                options.execution.query_concurrency.as_ref().map(|values| {
                    values
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                }),
            ),
            (
                "partition-cache-bytes",
                option_string(options.execution.partition_cache_bytes),
            ),
            (
                "leaf-beam-size",
                option_string(options.execution.leaf_beam_size),
            ),
            (
                "write-beam-size",
                option_string(options.execution.write_beam_size),
            ),
            (
                "refinement-rounds",
                option_string(options.execution.refinement_rounds),
            ),
            (
                "base-vectors",
                option_string(options.execution.base_vectors),
            ),
            (
                "query-vectors",
                option_string(options.execution.query_vectors),
            ),
            (
                "query-offset",
                option_string(options.execution.query_offset),
            ),
            (
                "max-partition-entries",
                option_string(options.execution.max_partition_entries),
            ),
            (
                "maintenance-workers",
                option_string(lifecycle.maintenance_workers),
            ),
            (
                "import-max-in-flight-batches",
                option_string(lifecycle.max_in_flight_batches),
            ),
            ("import-batch-size", option_string(lifecycle.batch_size)),
            (
                "import-backlog-watermark",
                option_string(lifecycle.backlog_watermark),
            ),
        ] {
            if let Some(value) = value {
                command.args([format!("--{name}"), value]);
            }
        }
        let output = command
            .output()
            .map_err(|error| format!("start scenario {}: {error}", scenario.name))?;
        if !output.status.success() {
            return Err(format!(
                "scenario {} failed with status {}",
                scenario.name, output.status
            ));
        }
        let report: BenchmarkReport = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("decode scenario {} report: {error}", scenario.name))?;
        reports.push(report);
    }
    let suite = BenchmarkSuite {
        schema_version: REPORT_SCHEMA_VERSION,
        reproduction_command,
        reports,
    };
    write_json(options.output.as_deref(), &suite)
}

/// Builds one Tokio runtime and executes exactly one isolated scenario.
fn run_worker(options: WorkerOptions) -> Result<(), String> {
    let mut scenario = runner::scenarios(&options.profile)?
        .into_iter()
        .find(|scenario| scenario.name == options.scenario)
        .ok_or_else(|| format!("unknown scenario `{}`", options.scenario))?;
    options.execution.lifecycle.apply(&mut scenario);
    if options.execution.rocksdb_path.is_some() {
        if scenario.profile != "large" {
            return Err("--rocksdb-path requires --profile large".to_owned());
        }
        scenario.persisted_index = options.execution.rocksdb_path.clone();
        scenario.reuse_index = options.execution.reuse_index;
    }
    if let Some(count) = options.execution.warmup_operations {
        scenario.warmup_operations = count;
    }
    if let Some(concurrency) = &options.execution.query_concurrency {
        if scenario.profile != "large" {
            return Err("--query-concurrency requires --profile large".to_owned());
        }
        scenario.query_concurrency_sweep = concurrency.clone();
    }
    if let Some(bytes) = options.execution.partition_cache_bytes {
        scenario.partition_cache_bytes = bytes;
    }
    if let Some(beam) = options.execution.leaf_beam_size {
        if scenario.profile != "large" {
            return Err("--leaf-beam-size requires --profile large".to_owned());
        }
        scenario.leaf_beam_sweep = vec![beam];
    }
    if let Some(rounds) = options.execution.refinement_rounds {
        if rounds > 5 || scenario.leaf_beam_sweep.is_empty() {
            return Err("--refinement-rounds requires a quality sweep and 0..=5 rounds".to_owned());
        }
        scenario.refinement_rounds = Some(rounds);
    }
    if let Some(write_beam_size) = options.execution.write_beam_size {
        scenario.write_beam_size = write_beam_size;
    }
    if let Some(base_vectors) = options.execution.base_vectors {
        scenario.base_vectors = base_vectors;
    }
    if let Some(query_vectors) = options.execution.query_vectors {
        scenario.query_vectors = query_vectors;
    }
    if let Some(query_offset) = options.execution.query_offset {
        scenario.query_offset = query_offset;
    }
    if let Some(max_partition_entries) = options.execution.max_partition_entries {
        scenario.max_partition_entries = max_partition_entries;
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(options.execution.worker_threads)
        .enable_all()
        .build()
        .map_err(|error| format!("build Tokio runtime: {error}"))?;
    let report = runtime.block_on(run_backend(&options, &scenario))?;
    serde_json::to_writer(io::stdout().lock(), &report)
        .map_err(|error| format!("write worker report: {error}"))?;
    Ok(())
}

/// Dispatches a worker to the selected compile-time Backend adapter.
async fn run_backend(
    options: &WorkerOptions,
    scenario: &ScenarioSpec,
) -> Result<BenchmarkReport, String> {
    match options.execution.backend.as_str() {
        "rocksdb" => run_rocksdb(options, scenario).await,
        "foundationdb" => run_foundationdb(options, scenario).await,
        _ => Err(format!("unknown backend `{}`", options.execution.backend)),
    }
}

#[cfg(feature = "rocksdb")]
/// Opens an isolated RocksDB database and measures one scenario.
async fn run_rocksdb(
    options: &WorkerOptions,
    scenario: &ScenarioSpec,
) -> Result<BenchmarkReport, String> {
    use std::sync::Arc;

    use ktann_rocksdb::{BackendNamespace, RocksDbBackend, RocksDbConfig};
    use rocksdb::{BlockBasedOptions, Cache, OptimisticTransactionDB, Options};

    let temporary = if options.execution.rocksdb_path.is_none() {
        Some(tempfile::tempdir().map_err(|error| format!("create RocksDB tempdir: {error}"))?)
    } else {
        None
    };
    let directory = options
        .execution
        .rocksdb_path
        .as_deref()
        .or_else(|| temporary.as_ref().map(tempfile::TempDir::path))
        .expect("persistent or temporary directory");
    if options.execution.rocksdb_path.is_some() {
        if options.execution.reuse_index {
            if !directory.join("fixture.json").is_file() {
                return Err("persisted index has no completed fixture manifest".to_owned());
            }
        } else {
            fs::create_dir(directory)
                .map_err(|error| format!("create fresh persisted index directory: {error}"))?;
        }
    }
    let database_path = directory.join("database");
    let mut database_options = Options::default();
    database_options.create_if_missing(!options.execution.reuse_index);
    // RocksDB owns raw-vector residency independently of the Partition Cache.
    let cache_bytes = options
        .execution
        .rocksdb_block_cache_bytes
        .unwrap_or(8 * 1024 * 1024);
    let cache = Cache::new_lru_cache(cache_bytes);
    let mut table_options = BlockBasedOptions::default();
    table_options.set_block_cache(&cache);
    database_options.set_block_based_table_factory(&table_options);
    let database = Arc::new(
        OptimisticTransactionDB::open(&database_options, &database_path)
            .map_err(|error| format!("open RocksDB: {error}"))?,
    );
    let config = RocksDbConfig::default();
    let config = if let Some(limit) = scenario.blocking_resource_limit {
        config
            .with_blocking_resource_limit(limit)
            .map_err(|error| format!("RocksDB config: {error:?}"))?
    } else {
        config
    };
    // Report the effective adapter limit even when the scenario uses its default.
    let mut effective_scenario = scenario.clone();
    effective_scenario.blocking_resource_limit = Some(config.blocking_resource_limit());
    let backend = RocksDbBackend::with_config(
        Arc::clone(&database),
        BackendNamespace::new("ktann-benchmark")
            .map_err(|error| format!("namespace: {error:?}"))?,
        config,
    );
    let report = runner::run_scenario(
        "rocksdb",
        format!("rust-rocksdb=0.24.0; rocksdb=10.4.2; block-cache-bytes={cache_bytes}"),
        backend,
        &effective_scenario,
        options.reproduction_command.clone(),
        options.execution.worker_threads,
    )
    .await?;
    eprintln!(
        "rocksdb block cache: capacity={cache_bytes} usage={} pinned={}",
        cache.get_usage(),
        cache.get_pinned_usage()
    );
    Ok(report)
}

#[cfg(not(feature = "rocksdb"))]
/// Reports that this executable lacks RocksDB support.
async fn run_rocksdb(
    _options: &WorkerOptions,
    _scenario: &ScenarioSpec,
) -> Result<BenchmarkReport, String> {
    Err("RocksDB support requires the `rocksdb` feature".to_owned())
}

#[cfg(feature = "foundationdb")]
/// Measures one scenario in a unique, subsequently cleared FDB namespace.
async fn run_foundationdb(
    options: &WorkerOptions,
    scenario: &ScenarioSpec,
) -> Result<BenchmarkReport, String> {
    use foundationdb::Database;
    use ktann::storage::backend::{Backend as _, WriteTxn as _};
    use ktann::storage::keys::KeyRange;
    use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

    let _network = boot_foundationdb();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let started_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("read system time for FoundationDB namespace: {error}"))?
        .as_nanos();
    let namespace_text = format!(
        "ktann-benchmark-{}-{started_nanos}-{}",
        std::process::id(),
        scenario.name
    );
    let namespace =
        BackendNamespace::new(&namespace_text).map_err(|error| format!("namespace: {error:?}"))?;
    let database = Database::new(cluster_file.as_deref())
        .map_err(|error| format!("open FoundationDB: {error}"))?;
    let backend = FoundationDbBackend::new(database, namespace.clone());
    let result = runner::run_scenario(
        "foundationdb",
        foundationdb_runtime_identity()?,
        backend,
        scenario,
        options.reproduction_command.clone(),
        options.execution.worker_threads,
    )
    .await;

    // Time plus PID prevents an abnormal prior run from being reopened after
    // PID reuse. Successful and ordinary failed runs still clear their
    // persistent namespace before the worker exits.
    let cleanup = async {
        let cleanup_backend = FoundationDbBackend::new(
            Database::new(cluster_file.as_deref())
                .map_err(|error| format!("open FoundationDB cleanup handle: {error}"))?,
            namespace,
        );
        let mut transaction = cleanup_backend
            .begin_write()
            .await
            .map_err(|error| format!("begin FoundationDB cleanup: {error:?}"))?;
        transaction
            .clear_range(&KeyRange::new(Vec::new(), vec![0xff]))
            .await
            .map_err(|error| format!("clear FoundationDB benchmark namespace: {error:?}"))?;
        transaction
            .commit()
            .await
            .map_err(|error| format!("commit FoundationDB cleanup: {error:?}"))?;
        Ok::<(), String>(())
    }
    .await;

    // Scenario failure is the primary diagnostic. Cleanup is nevertheless a
    // required postcondition after success, and a secondary failure is kept
    // visible so persistent nightly data cannot accumulate unnoticed.
    match (result, cleanup) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(format!(
            "{error}; FoundationDB namespace cleanup also failed: {cleanup_error}"
        )),
    }
}

#[cfg(feature = "foundationdb")]
#[expect(
    unsafe_code,
    reason = "the FoundationDB binding requires one process-global network boot"
)]
/// Boots the single FoundationDB network owned by one worker process.
pub(crate) fn boot_foundationdb() -> foundationdb::api::NetworkAutoStop {
    // SAFETY: a worker executes exactly one scenario, boots the process-global
    // network once, and retains this guard until all handles have been dropped.
    unsafe { foundationdb::boot() }
}

/// Returns the linked client and connected server versions used by this run.
#[cfg(feature = "foundationdb")]
pub(crate) fn foundationdb_runtime_identity() -> Result<String, String> {
    let client = foundationdb_client_version();
    let server = Command::new("fdbcli")
        .args(["--exec", "status json"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| serde_json::from_slice::<serde_json::Value>(&output.stdout).ok())
        .and_then(|status| {
            let processes = status.pointer("/cluster/processes")?.as_object()?;
            let mut versions: Vec<_> = processes
                .values()
                .filter_map(|process| process.get("version")?.as_str())
                .map(str::to_owned)
                .collect();
            versions.sort();
            versions.dedup();
            (!versions.is_empty()).then(|| versions.join(","))
        })
        .ok_or_else(|| {
            "read FoundationDB server version with `fdbcli --exec status json`".to_owned()
        })?;
    Ok(format!(
        "foundationdb-client={client}; foundationdb-server={server}; api=730"
    ))
}

/// Reads the FoundationDB C client's static version string.
#[cfg(feature = "foundationdb")]
#[expect(
    unsafe_code,
    reason = "FoundationDB exposes its linked client version only through this C API"
)]
fn foundationdb_client_version() -> String {
    // SAFETY: FoundationDB documents this as a process-lifetime NUL-terminated
    // string owned by the client library; the pointer is never freed by Rust.
    let pointer = unsafe { foundationdb_sys::fdb_get_client_version() };
    if pointer.is_null() {
        return "unavailable".to_owned();
    }
    // SAFETY: the non-null pointer above refers to the documented C string.
    unsafe { std::ffi::CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(not(feature = "foundationdb"))]
/// Reports that this executable lacks FoundationDB support.
async fn run_foundationdb(
    _options: &WorkerOptions,
    _scenario: &ScenarioSpec,
) -> Result<BenchmarkReport, String> {
    Err("FoundationDB support requires the `foundationdb` feature".to_owned())
}

/// Loads two suites, applies the explicit policy, and emits structured findings.
fn compare_reports(options: CompareOptions) -> Result<(), String> {
    let baseline: BenchmarkSuite = read_json(&options.baseline)?;
    let candidate: BenchmarkSuite = read_json(&options.candidate)?;
    let comparison = compare::compare(&baseline, &candidate, options.policy)?;
    write_json(options.output.as_deref(), &comparison)?;
    if comparison.failed() {
        Err("material benchmark regressions detected".to_owned())
    } else {
        Ok(())
    }
}

/// Reads one complete versioned JSON artifact for comparison.
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("decode {}: {error}", path.display()))
}

/// Writes one complete JSON document atomically to stdout or the requested path.
fn write_json<T: serde::Serialize>(path: Option<&Path>, value: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("encode benchmark JSON: {error}"))?;
    if let Some(path) = path {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent)
            .map_err(|error| format!("create temporary output: {error}"))?;
        temporary
            .write_all(&bytes)
            .map_err(|error| format!("write temporary output: {error}"))?;
        temporary
            .persist(path)
            .map_err(|error| format!("persist {}: {}", path.display(), error.error))?;
    } else {
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(&bytes)
            .and_then(|()| stdout.write_all(b"\n"))
            .map_err(|error| format!("write stdout: {error}"))?;
    }
    Ok(())
}

/// Renders the current invocation with conservative POSIX shell quoting.
fn render_command(arguments: impl IntoIterator<Item = OsString>) -> String {
    arguments
        .into_iter()
        .map(|argument| shell_quote(&argument))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Quotes one argument so the recorded command preserves its boundary.
fn shell_quote(value: &OsStr) -> String {
    let value = value.to_string_lossy();
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"-_./:=+".contains(&byte))
    {
        value.into_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

/// Returns the stable help shown for missing or unknown public commands.
fn usage() -> String {
    "usage:\n  ktann-bench run --backend rocksdb|foundationdb [--profile smoke|full|large] [--scenario NAME] [--worker-threads N] [--query-concurrency N,...] [--partition-cache-bytes N] [--leaf-beam-size N] [--warmup-operations N] [--rocksdb-block-cache-bytes N] [--rocksdb-path PATH] [--reuse-index true|false] [--write-beam-size N] [--refinement-rounds 0..5] [--base-vectors N] [--query-vectors N] [--query-offset N] [--max-partition-entries N] [--maintenance-workers N] [--import-max-in-flight-batches N] [--import-batch-size N] [--import-backlog-watermark N] [--output PATH]\n  ktann-bench compare --baseline PATH --candidate PATH [--maximum-relative-regression N] [--maximum-recall-drop N] [--maximum-rejection-rate-increase N] [--output PATH]".to_owned()
}

#[cfg(test)]
mod tests {
    use super::{option_map, parse_compare_options, parse_run_options, shell_quote};

    #[test]
    fn native_cache_and_fixture_controls_are_validated() {
        let parse = |arguments: &[&str]| {
            parse_run_options(
                &arguments
                    .iter()
                    .map(std::ffi::OsString::from)
                    .collect::<Vec<_>>(),
            )
        };
        let options = parse(&[
            "--backend",
            "rocksdb",
            "--profile",
            "large",
            "--scenario",
            "quality-sift-1m",
            "--rocksdb-block-cache-bytes",
            "4096",
            "--warmup-operations",
            "0",
            "--rocksdb-path",
            "/tmp/fixture",
            "--reuse-index",
            "true",
        ])
        .unwrap();
        assert_eq!(options.execution.rocksdb_block_cache_bytes, Some(4096));
        assert_eq!(options.execution.warmup_operations, Some(0));
        assert!(options.execution.reuse_index);
        for arguments in [
            vec!["--backend", "rocksdb", "--rocksdb-block-cache-bytes", "0"],
            vec![
                "--backend",
                "foundationdb",
                "--rocksdb-block-cache-bytes",
                "4096",
            ],
            vec!["--backend", "rocksdb", "--reuse-index", "true"],
            vec!["--backend", "rocksdb", "--rocksdb-path", "/tmp/fixture"],
            vec![
                "--backend",
                "rocksdb",
                "--profile",
                "large",
                "--rocksdb-path",
                "/tmp/fixture",
            ],
        ] {
            assert!(parse(&arguments).is_err(), "{arguments:?}");
        }
    }

    #[test]
    fn large_search_diagnostics_parse_and_reject_invalid_counts() {
        let arguments = [
            "--backend",
            "rocksdb",
            "--profile",
            "large",
            "--query-concurrency",
            "1,4,16,4",
            "--partition-cache-bytes",
            "0",
            "--leaf-beam-size",
            "32",
        ]
        .map(std::ffi::OsString::from);
        let options = parse_run_options(&arguments).expect("large diagnostics");
        assert_eq!(options.execution.query_concurrency, Some(vec![1, 4, 16, 4]));
        assert_eq!(options.execution.partition_cache_bytes, Some(0));
        assert_eq!(options.execution.leaf_beam_size, Some(32));
        for counts in ["", "0", "1,0", "1,", "-1"] {
            let arguments = [
                "--backend",
                "rocksdb",
                "--profile",
                "large",
                "--query-concurrency",
                counts,
            ]
            .map(std::ffi::OsString::from);
            assert!(parse_run_options(&arguments).is_err(), "{counts}");
        }
        let arguments =
            ["--backend", "rocksdb", "--query-concurrency", "1"].map(std::ffi::OsString::from);
        assert!(parse_run_options(&arguments).is_err());
    }

    #[test]
    fn offline_refinement_option_accepts_zero_rounds() {
        let arguments = [
            "--backend",
            "rocksdb",
            "--scenario",
            "quality-cohere-1m",
            "--refinement-rounds",
            "0",
        ]
        .map(std::ffi::OsString::from);
        let options = parse_run_options(&arguments).expect("offline refinement option");
        assert_eq!(options.execution.refinement_rounds, Some(0));
        let mut invalid = arguments;
        invalid[5] = "-1".into();
        assert!(parse_run_options(&invalid).is_err());
    }

    #[test]
    fn lifecycle_parser_accepts_diagnostic_overrides() {
        let arguments = [
            "--backend",
            "rocksdb",
            "--scenario",
            "import-to-search-lifecycle",
            "--maintenance-workers",
            "0",
            "--import-max-in-flight-batches",
            "4",
            "--import-batch-size",
            "25",
            "--import-backlog-watermark",
            "1",
        ]
        .map(std::ffi::OsString::from);
        let options = parse_run_options(&arguments).expect("valid lifecycle overrides");
        assert_eq!(options.execution.lifecycle.maintenance_workers, Some(0));
        assert_eq!(options.execution.lifecycle.max_in_flight_batches, Some(4));
        assert_eq!(options.execution.lifecycle.batch_size, Some(25));
        assert_eq!(options.execution.lifecycle.backlog_watermark, Some(1));
    }

    #[test]
    fn lifecycle_overrides_require_the_lifecycle_scenario() {
        let arguments =
            ["--backend", "rocksdb", "--maintenance-workers", "0"].map(std::ffi::OsString::from);
        assert_eq!(
            parse_run_options(&arguments).map(|_| ()),
            Err("lifecycle overrides require --scenario import-to-search-lifecycle".to_owned())
        );
    }

    #[test]
    fn lifecycle_overrides_are_allowed_for_large_diagnostics() {
        let arguments = [
            "--backend",
            "rocksdb",
            "--profile",
            "large",
            "--maintenance-workers",
            "0",
        ]
        .map(std::ffi::OsString::from);
        let options = parse_run_options(&arguments).expect("large diagnostic override");
        assert_eq!(options.profile, "large");
        assert_eq!(options.execution.lifecycle.maintenance_workers, Some(0));
    }

    #[test]
    fn option_parser_rejects_duplicates() {
        let arguments = ["--profile", "smoke", "--profile", "full"].map(std::ffi::OsString::from);
        assert_eq!(
            option_map(&arguments, &["profile"]),
            Err("duplicate option --profile".to_owned())
        );
    }

    #[test]
    fn option_parser_rejects_unknown_keys() {
        let arguments = ["--profiel", "full"].map(std::ffi::OsString::from);
        assert_eq!(
            option_map(&arguments, &["profile"]),
            Err("unknown option --profiel".to_owned())
        );
    }

    #[test]
    fn compare_parser_honors_documented_threshold_names() {
        let arguments = [
            "--baseline",
            "base.json",
            "--candidate",
            "candidate.json",
            "--maximum-relative-regression",
            "0.10",
            "--maximum-recall-drop",
            "0.01",
            "--maximum-rejection-rate-increase",
            "0.03",
        ]
        .map(std::ffi::OsString::from);
        let options = parse_compare_options(&arguments).expect("documented options");
        assert_eq!(options.policy.maximum_relative_regression, 0.10);
        assert_eq!(options.policy.maximum_recall_drop, 0.01);
        assert_eq!(options.policy.maximum_rejection_rate_increase, 0.03);
    }

    #[test]
    fn shell_quote_preserves_argument_boundaries() {
        assert_eq!(shell_quote(std::ffi::OsStr::new("plain")), "plain");
        assert_eq!(
            shell_quote(std::ffi::OsStr::new("two words")),
            "'two words'"
        );
        assert_eq!(shell_quote(std::ffi::OsStr::new("a'b")), "'a'\\''b'");
    }
}
