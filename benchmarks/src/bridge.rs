//! Versioned, benchmark-only Unix socket bridge for spawned VectorDBBench clients.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use ktann::api::{
    ImportOptions, Index, IndexConfig, Metric, Mutation, OperationOptions, Record, RuntimeConfig,
    SearchBudgets, SearchOptions, SearchRequest,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::backend::{BackendCounters, MeasuredBackend};
use crate::resource::ResourceSnapshot;

mod topology;

/// Wire frames are a big-endian u32 length followed by UTF-8 JSON.
const VERSION: u32 = 1;
const MAX_FRAME: usize = 8 << 20;
const MAX_BATCH: usize = 50;
const MAX_CONNECTIONS: usize = 128;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(3600);

/// Private CLI options. A bridge owns exactly one isolated backend namespace.
struct Options {
    socket: PathBuf,
    database: PathBuf,
    report: PathBuf,
    backend: String,
}

/// Runs a single bridge until shutdown or SIGINT, then closes the Runtime.
///
/// # Errors
/// Returns setup, protocol listener, shutdown, and report persistence failures.
pub fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        socket: PathBuf::new(),
        database: PathBuf::new(),
        report: PathBuf::new(),
        backend: "rocksdb".into(),
    };
    while let Some(arg) = args.next() {
        let value = args.next().ok_or(
            "expected --socket PATH --database PATH --report PATH --backend rocksdb|foundationdb",
        )?;
        match arg.as_str() {
            "--socket" => options.socket = value.into(),
            "--database" => options.database = value.into(),
            "--report" => options.report = value.into(),
            "--backend" => options.backend = value,
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    if options.socket.as_os_str().is_empty()
        || options.report.as_os_str().is_empty()
        || options.database.as_os_str().is_empty()
    {
        return Err("--socket, --database, and --report are required (database is the FDB namespace for foundationdb)".into());
    }
    // Bind without unlinking: a stale path must never evict a live bridge.
    let executor = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    match options.backend.as_str() {
        #[cfg(feature = "rocksdb")]
        "rocksdb" => executor.block_on(async {
            use ktann_rocksdb::{BackendNamespace, RocksDbBackend, RocksDbConfig};
            let mut config = rocksdb::Options::default();
            config.create_if_missing(true);
            let db = Arc::new(
                rocksdb::OptimisticTransactionDB::open(&config, &options.database)
                    .map_err(|e| e.to_string())?,
            );
            let backend = RocksDbBackend::with_config(
                db,
                BackendNamespace::new("ktann-vdbbench").map_err(|e| e.to_string())?,
                RocksDbConfig::default()
                    .with_blocking_resource_limit(64)
                    .map_err(|e| e.to_string())?,
            );
            serve(
                backend,
                "rust-rocksdb=0.24.0; rocksdb=10.4.2".into(),
                &options,
            )
            .await
        }),
        #[cfg(feature = "foundationdb")]
        "foundationdb" => {
            // Network guard outlives every database handle and Tokio task.
            let _network = crate::cli::boot_foundationdb();
            let identity = crate::cli::foundationdb_runtime_identity()?;
            executor.block_on(async {
                use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};
                let namespace = options.database.to_str().ok_or("namespace must be UTF-8")?;
                if !namespace.starts_with("ktann-vdbbench-") {
                    return Err("FDB namespace must start with ktann-vdbbench-".into());
                }
                let database =
                    foundationdb::Database::new(std::env::var("FDB_CLUSTER_FILE").ok().as_deref())
                        .map_err(|e| e.to_string())?;
                let backend = FoundationDbBackend::new(
                    database,
                    BackendNamespace::new(namespace).map_err(|e| e.to_string())?,
                );
                serve(backend, identity, &options).await
            })
        }
        _ => Err("backend unavailable; build with rocksdb and/or foundationdb features".into()),
    }
}

/// Strict envelope; unsupported fields and versions never silently change semantics.
#[derive(Deserialize)]
struct Request {
    version: u32,
    #[serde(flatten)]
    operation: Operation,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Operation {
    Reset {
        dimension: usize,
        metric: String,
        dataset: String,
        leaf_budget: Option<u32>,
        leaf_beam: Option<u32>,
    },
    Insert {
        ids: Vec<i64>,
        vectors: Vec<Vec<f32>>,
    },
    Optimize {
        records: u64,
    },
    Search {
        vector: Vec<f32>,
        k: usize,
    },
    Health,
    Shutdown,
}

/// Lifecycle state is exclusive for mutation and shared for concurrent search.
struct State<B: Backend> {
    index: Option<Index<MeasuredBackend<B>>>,
    ready: bool,
    records: u64,
    dimension: usize,
    metric: String,
    dataset: String,
    search: SearchOptions,
    root_probe: Option<Arc<[f32]>>,
    started: Option<Instant>,
    insert_seconds: f64,
    optimize_seconds: f64,
    topology: Value,
}

impl<B: Backend> Default for State<B> {
    fn default() -> Self {
        Self {
            index: None,
            ready: false,
            records: 0,
            dimension: 0,
            metric: String::new(),
            dataset: String::new(),
            search: SearchOptions::default(),
            root_probe: None,
            started: None,
            insert_seconds: 0.0,
            optimize_seconds: 0.0,
            topology: Value::Null,
        }
    }
}

/// Fixed-size logarithmic histograms bound reporting memory even in long QPS tests.
struct Measurements {
    searches: u64,
    buckets: [u64; 64],
    search_seconds: f64,
    decode_seconds: f64,
    encode_seconds: f64,
    received_bytes: u64,
    sent_bytes: u64,
    budget: [u64; 4],
    exhausted: [u64; 5],
    last_search: Option<Instant>,
}
impl Default for Measurements {
    fn default() -> Self {
        Self {
            searches: 0,
            buckets: [0; 64],
            search_seconds: 0.0,
            decode_seconds: 0.0,
            encode_seconds: 0.0,
            received_bytes: 0,
            sent_bytes: 0,
            budget: [0; 4],
            exhausted: [0; 5],
            last_search: None,
        }
    }
}

/// Shared process owner. No KTANN handle crosses the Python process boundary.
struct Service<B: Backend> {
    runtime: Runtime<MeasuredBackend<B>>,
    backend: MeasuredBackend<B>,
    state: RwLock<State<B>>,
    measurements: Mutex<Measurements>,
    counters: BackendCounters,
    identity: String,
    limits: String,
    baseline: ResourceSnapshot,
    stop: CancellationToken,
}

/// Removes only the socket created by this process, including on setup failure.
struct SocketGuard(PathBuf);
impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn serve<B: Backend>(backend: B, identity: String, options: &Options) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let listener = UnixListener::bind(&options.socket).map_err(|e| e.to_string())?;
    let _socket = SocketGuard(options.socket.clone());
    std::fs::set_permissions(&options.socket, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| e.to_string())?;
    let limits = format!(
        "hard={:?}; admission={:?}",
        backend.hard_limits(),
        backend.admission_budget()
    );
    let (backend, counters) = MeasuredBackend::new(backend);
    let config = RuntimeConfig::default()
        .with_foreground_operation_limit(128)
        .and_then(|c| c.with_maintenance(2, 1024))
        .and_then(|c| c.with_attempts(32, 32))
        .and_then(|c| c.with_partition_cache_bytes(512 << 20))
        .and_then(|c| c.with_write_beam_size(8))
        .and_then(|c| c.with_import_limits(4, 1))
        .map_err(|e| e.to_string())?;
    let service = Arc::new(Service {
        runtime: Runtime::new(backend.clone(), config).map_err(|e| e.to_string())?,
        backend,
        state: RwLock::new(State::default()),
        measurements: Mutex::new(Measurements::default()),
        counters,
        identity,
        limits,
        baseline: ResourceSnapshot::capture()?,
        stop: CancellationToken::new(),
    });
    let permits = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    let mut tasks = JoinSet::new();
    let result = async {
        loop {
            tokio::select! {
                biased;
                _ = service.stop.cancelled() => break,
                signal = tokio::signal::ctrl_c() => { signal.map_err(|e| e.to_string())?; break; },
                Some(joined) = tasks.join_next(), if !tasks.is_empty() => { joined.map_err(|e| e.to_string())?; },
                permit = Arc::clone(&permits).acquire_owned() => {
                    let permit = permit.map_err(|e| e.to_string())?;
                    let accepted = tokio::select! { _ = service.stop.cancelled() => break, signal = tokio::signal::ctrl_c() => { signal.map_err(|e| e.to_string())?; break; }, accepted = listener.accept() => accepted };
                    let (stream, _) = accepted.map_err(|e| e.to_string())?;
                    let service = Arc::clone(&service);
                    tasks.spawn(async move { let _permit = permit; connection(stream, service).await; });
                }
            }
        }
        Ok::<(), String>(())
    }.await;
    service.stop.cancel();
    while let Some(joined) = tasks.join_next().await {
        joined.map_err(|e| e.to_string())?;
    }
    // Capture workload IO before deleting this bridge's dedicated Logical Index.
    let report = service.report(&options.backend).await;
    let cleanup = service
        .runtime
        .drop_index("vdbbench")
        .await
        .map_err(|e| e.to_string());
    let shutdown = service.runtime.shutdown().await.map_err(|e| e.to_string());
    let persisted = report.and_then(|report| write_report(&options.report, &report));
    result.and(cleanup).and(shutdown).and(persisted)
}

fn write_report(path: &Path, report: &Value) -> Result<(), String> {
    let temporary = path.with_extension("json.tmp");
    std::fs::write(
        &temporary,
        serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::rename(temporary, path).map_err(|e| e.to_string())
}

async fn connection<B: Backend>(mut stream: UnixStream, service: Arc<Service<B>>) {
    loop {
        let response = tokio::select! {
            _ = service.stop.cancelled() => return,
            result = tokio::time::timeout(REQUEST_TIMEOUT, exchange(&mut stream, &service)) => result,
        };
        if !matches!(response, Ok(Ok(()))) {
            return;
        }
    }
}

async fn exchange<B: Backend>(stream: &mut UnixStream, service: &Service<B>) -> Result<(), String> {
    let size = stream.read_u32().await.map_err(|e| e.to_string())? as usize;
    if size == 0 || size > MAX_FRAME {
        return Err("invalid frame length".into());
    }
    let mut data = vec![0; size];
    stream
        .read_exact(&mut data)
        .await
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    let parsed = serde_json::from_slice::<Request>(&data);
    let decoded = start.elapsed().as_secs_f64();
    let shutdown = matches!(
        &parsed,
        Ok(Request {
            version: VERSION,
            operation: Operation::Shutdown
        })
    );
    let result = match parsed {
        Ok(request) if request.version == VERSION => service.handle(request.operation).await,
        Ok(_) => Err(("protocol", "unsupported protocol version".into())),
        Err(error) => Err(("invalid_argument", error.to_string())),
    };
    let result = match result {
        Ok(value) => json!({"version":VERSION,"ok":true,"result":value}),
        Err((kind, message)) => {
            json!({"version":VERSION,"ok":false,"error":{"kind":kind,"message":message}})
        }
    };
    let start = Instant::now();
    let response = serde_json::to_vec(&result).map_err(|e| e.to_string())?;
    {
        let mut m = service
            .measurements
            .lock()
            .expect("measurement mutex poisoned");
        m.decode_seconds += decoded;
        m.encode_seconds += start.elapsed().as_secs_f64();
        m.received_bytes += size as u64;
        m.sent_bytes += response.len() as u64;
    }
    stream
        .write_u32(u32::try_from(response.len()).map_err(|e| e.to_string())?)
        .await
        .map_err(|e| e.to_string())?;
    stream
        .write_all(&response)
        .await
        .map_err(|e| e.to_string())?;
    if shutdown {
        service.stop.cancel();
    }
    Ok(())
}

/// Errors never invite automatic replay of inserts with unknown outcomes.
type BridgeResult = Result<Value, (&'static str, String)>;
fn api_error(error: ktann::api::Error) -> (&'static str, String) {
    ("ktann", format!("{:?}: {error}", error.kind()))
}
fn invalid(message: &str) -> (&'static str, String) {
    ("invalid_argument", message.into())
}

impl<B: Backend> Service<B> {
    async fn handle(&self, operation: Operation) -> BridgeResult {
        match operation {
            Operation::Health => {
                let Ok(state) = self.state.try_read() else {
                    return Ok(json!({"ready":false,"busy":true}));
                };
                Ok(
                    json!({"ready":state.ready,"records":state.records,"max_frame_bytes":MAX_FRAME,"max_batch":MAX_BATCH,"backend_identity":self.identity,"limits":self.limits}),
                )
            }
            Operation::Shutdown => Ok(json!({"shutdown":true})),
            Operation::Reset {
                dimension,
                metric,
                dataset,
                leaf_budget,
                leaf_beam,
            } => {
                if dataset.is_empty() || dataset.len() > 1024 {
                    return Err(invalid("dataset identity required (at most 1024 bytes)"));
                }
                let metric_kind = match metric.as_str() {
                    "L2" => Metric::L2,
                    "COSINE" => Metric::Cosine,
                    _ => return Err(invalid("only L2 and COSINE are supported")),
                };
                let config = IndexConfig::new(dimension, metric_kind)
                    .and_then(|c| c.with_partition_entries(32, 128))
                    .map_err(api_error)?;
                let mut search = SearchOptions::default();
                if let Some(budget) = leaf_budget {
                    search = search
                        .with_visited_leaf_entries(budget)
                        .map_err(api_error)?;
                }
                if let Some(beam) = leaf_beam {
                    search = search.with_leaf_beam_size(beam).map_err(api_error)?;
                }
                let mut state = self.state.write().await;
                // One case per bridge keeps all reported resource high-water marks attributable.
                if state.index.is_some() {
                    return Err(invalid("one case per bridge; restart for another reset"));
                }
                self.runtime
                    .drop_index("vdbbench")
                    .await
                    .map_err(api_error)?;
                let index = self
                    .runtime
                    .create_index("vdbbench", config)
                    .await
                    .map_err(api_error)?;
                state.index = Some(index);
                state.dimension = dimension;
                state.metric = metric;
                state.dataset = dataset;
                state.search = search;
                Ok(json!({"created":true}))
            }
            Operation::Insert { ids, vectors } => {
                if ids.is_empty() || ids.len() != vectors.len() || ids.len() > MAX_BATCH {
                    return Err(invalid("batch must contain 1..50 paired IDs and vectors"));
                }
                let mut state = self.state.write().await;
                if state.ready {
                    return Err(invalid("inserts after optimize are unsupported"));
                }
                let index = state
                    .index
                    .as_ref()
                    .ok_or_else(|| invalid("reset required"))?
                    .clone();
                let mut mutations = Vec::with_capacity(ids.len());
                let mut root_probe = None;
                for (id, vector) in ids.iter().zip(vectors) {
                    if vector.len() != state.dimension {
                        return Err(invalid("wrong vector dimension"));
                    }
                    let vector: Arc<[f32]> = vector.into();
                    if mutations.is_empty() && state.root_probe.is_none() {
                        root_probe = Some(Arc::clone(&vector));
                    }
                    let record =
                        Record::new(Bytes::copy_from_slice(&id.to_be_bytes()), vector, vec![])
                            .map_err(api_error)?;
                    mutations.push(Mutation::Insert(record));
                }
                let start = Instant::now();
                state.started.get_or_insert(start);
                let mut session = index
                    .import_session(ImportOptions::default())
                    .map_err(api_error)?;
                session.submit(mutations).await.map_err(api_error)?;
                let outcomes = session.finish().await;
                state.insert_seconds += start.elapsed().as_secs_f64();
                for outcome in outcomes {
                    outcome.result.map_err(api_error)?;
                }
                state.records += ids.len() as u64;
                if let Some(probe) = root_probe {
                    state.root_probe = Some(probe);
                }
                Ok(json!({"inserted":ids.len()}))
            }
            Operation::Optimize { records } => {
                let mut state = self.state.write().await;
                if state.records != records || records == 0 {
                    return Err(invalid("optimize record count mismatch or empty dataset"));
                }
                let index = state
                    .index
                    .as_ref()
                    .ok_or_else(|| invalid("reset required"))?
                    .clone();
                let start = Instant::now();
                let deadline = start + Duration::from_secs(3500);
                let mut previous_progress = None;
                let mut last_progress = Instant::now();
                let mut last_probe: Option<Instant> = None;
                // Rediscovery needs to touch the target, not execute a quality benchmark.
                let probe_options = SearchOptions::default()
                    .with_scanned_tree_keys(1)
                    .and_then(|s| s.with_visited_partitions(128))
                    .and_then(|s| s.with_visited_leaf_entries(128))
                    .and_then(|s| s.with_leaf_beam_size(1))
                    .map_err(api_error)?;

                loop {
                    if Instant::now() >= deadline {
                        return Err((
                            "timeout",
                            format!(
                                "topology readiness deadline exceeded; last snapshot: {}",
                                state.topology
                            ),
                        ));
                    }
                    let snapshot = tokio::time::timeout_at(
                        tokio::time::Instant::from_std(deadline),
                        topology::snapshot(&self.backend, index.logical_index_id(), records),
                    )
                    .await
                    .map_err(|_| {
                        (
                            "timeout",
                            format!(
                                "topology snapshot deadline exceeded; last snapshot: {}",
                                state.topology
                            ),
                        )
                    })?
                    .map_err(api_error)?;
                    state.topology = snapshot.facts;
                    if snapshot.ready {
                        state.ready = true;
                        state.optimize_seconds = start.elapsed().as_secs_f64();
                        return Ok(state.topology.clone());
                    }
                    if previous_progress != Some(snapshot.progress) {
                        eprintln!(
                            "optimize audit after {:.1}s: {}",
                            start.elapsed().as_secs_f64(),
                            state.topology
                        );
                        previous_progress = Some(snapshot.progress);
                        last_progress = Instant::now();
                    }
                    if last_progress.elapsed() >= Duration::from_secs(5)
                        && last_probe.is_none_or(|at| at.elapsed() >= Duration::from_secs(30))
                    {
                        last_probe = Some(Instant::now());
                        let root_probe = state
                            .root_probe
                            .iter()
                            .filter(|_| snapshot.needs_root_probe);
                        for vector in snapshot.probes.iter().chain(root_probe) {
                            index
                                .search_with_control(
                                    SearchRequest::new(Arc::clone(vector), 10)
                                        .map_err(api_error)?
                                        .with_options(probe_options),
                                    OperationOptions::default().with_deadline(deadline),
                                )
                                .await
                                .map_err(api_error)?;
                        }
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
            Operation::Search { vector, k } => {
                let state = self.state.read().await;
                if !state.ready {
                    return Err(invalid("optimize must succeed before search"));
                }
                let index = state
                    .index
                    .as_ref()
                    .ok_or_else(|| invalid("reset required"))?;
                let request = SearchRequest::new(vector, k)
                    .map_err(api_error)?
                    .with_options(state.search);
                let start = Instant::now();
                let result = index.search(request).await.map_err(api_error)?;
                let elapsed = start.elapsed();
                let ids: Vec<i64> = result
                    .hits
                    .iter()
                    .map(|hit| {
                        i64::from_be_bytes(
                            hit.id()
                                .as_ref()
                                .try_into()
                                .expect("bridge owns eight-byte Record IDs"),
                        )
                    })
                    .collect();
                let mut m = self
                    .measurements
                    .lock()
                    .expect("measurement mutex poisoned");
                m.searches += 1;
                m.search_seconds += elapsed.as_secs_f64();
                m.last_search = Some(Instant::now());
                let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
                let bucket = (64 - nanos.leading_zeros()).min(63) as usize;
                m.buckets[bucket] += 1;
                let u = result.usage;
                for (sum, value) in m.budget.iter_mut().zip([
                    u.scanned_tree_keys,
                    u.visited_partitions,
                    u.visited_leaf_entries,
                    u.exact_rerank_candidates,
                ]) {
                    *sum += u64::from(value);
                }
                let e = result.exhausted;
                for (sum, value) in m.exhausted.iter_mut().zip([
                    e.scanned_tree_keys,
                    e.visited_partitions,
                    e.visited_leaf_entries,
                    e.exact_rerank_candidates,
                    result.rabitq_overlap_truncated,
                ]) {
                    *sum += u64::from(value);
                }
                drop(m);
                Ok(json!({"ids":ids,"ktann_seconds":elapsed.as_secs_f64()}))
            }
        }
    }

    async fn report(&self, backend: &str) -> Result<Value, String> {
        let state = self.state.read().await;
        let m = self
            .measurements
            .lock()
            .expect("measurement mutex poisoned");
        let resources = ResourceSnapshot::capture()?;
        let mut cumulative = 0;
        let p50 = m.buckets.iter().position(|count| {
            cumulative += count;
            cumulative >= m.searches.div_ceil(2)
        });
        Ok(json!({
            "label": "KTANN plus benchmark bridge — companion report (not VectorDBBench canonical metrics)",
            "protocol_version": VERSION,
            "backend": backend,
            "backend_identity": self.identity,
            "backend_limits": self.limits,
            "dataset": state.dataset,
            "dimension": state.dimension,
            "metric": state.metric,
            "records": state.records,
            "ready": state.ready,
            "configuration": {
                "min_partition_entries": 32, "max_partition_entries": 128, "write_beam_size": 8,
                "maintenance_workers": 2, "partition_cache_bytes": 512_u64 << 20,
                "foreground_limit": 128, "import_max_in_flight_batches": 4, "import_backlog_watermark": 1,
                "readiness_header_slot_limit": 262144, "readiness_probe_limit": 32,
                "readiness_stall_seconds": 5, "readiness_probe_interval_seconds": 30,
                "readiness_probe_leaf_beam": 1, "readiness_probe_partition_budget": 128,
                "readiness_probe_leaf_entry_budget": 128
            },
            "continuous_first_insert_through_final_search_seconds": state.started.zip(m.last_search).map(|(a,b)| b.duration_since(a).as_secs_f64()),
            "phases": {
                "committed_import_seconds": state.insert_seconds,
                "optimize_seconds": state.optimize_seconds,
                "ktann_search_seconds_sum": m.search_seconds
            },
            "searches": m.searches,
            "ktann_search_p50_upper_bound_seconds": p50.filter(|_| m.searches > 0).map(|i| 2f64.powi(i as i32) / 1e9),
            "search_latency_log2_nanoseconds_histogram": m.buckets.to_vec(),
            "search_budgets": {
                "scanned_tree_keys": SearchBudgets::default().scanned_tree_keys(),
                "visited_partitions": SearchBudgets::default().visited_partitions(),
                "visited_leaf_entries": state.search.visited_leaf_entries().unwrap_or(SearchBudgets::default().visited_leaf_entries()),
                "leaf_beam": state.search.resolved_leaf_beam_size(),
                "exact_rerank_candidates": "KTANN default derived from k"
            },
            "search_budget_usage_totals": m.budget,
            "search_exhaustion_counts": m.exhausted,
            "topology": state.topology,
            "resource": { "peak_rss_bytes": resources.peak_rss_bytes(), "cpu_seconds": resources.cpu_seconds_since(self.baseline) },
            "backend_io": self.counters.snapshot(),
            "bridge": {
                "decode_seconds": m.decode_seconds, "encode_seconds": m.encode_seconds,
                "received_json_bytes": m.received_bytes, "sent_json_bytes": m.sent_bytes,
                "max_frame_bytes": MAX_FRAME, "max_batch": MAX_BATCH, "max_connections": MAX_CONNECTIONS
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::Request;

    #[test]
    fn envelope_rejects_unknown_operations_fields_and_noninteger_ids() {
        assert!(serde_json::from_str::<Request>(r#"{"version":1,"op":"health"}"#).is_ok());
        for input in [
            r#"{"version":1,"op":"delete","id":1}"#,
            r#"{"version":1,"op":"search","vector":[1],"k":1,"filter":{}}"#,
            r#"{"version":1,"op":"insert","ids":[1.5],"vectors":[[1]]}"#,
            r#"{"version":1,"op":"insert","ids":[9223372036854775808],"vectors":[[1]]}"#,
        ] {
            assert!(serde_json::from_str::<Request>(input).is_err(), "{input}");
        }
    }
}
