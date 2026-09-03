//! The data-driven integration corpus runner (issue #94).
//!
//! Executes every block of every `tests/datadriven/*.kddt` file against the
//! public Runtime/Index API on the deterministic backend, and diffs the
//! actual output against the recorded expectation. `KTANN_REWRITE=1 cargo test
//! --test e2e` regenerates the corpus instead of failing; a rewrite must be
//! reviewed like any other change.
//!
//! KDDT requests use a test-only default leaf beam of 8 so small settled trees
//! exercise approximate routing; production SearchOptions keep their default
//! of 128. A directive can override the test default with `beam-size=`.
//!
//! Directives:
//!
//! - `new-index name=N dimension=D metric=M fields=f:t[?],... tree-key-fields=I,... [min-entries=N] [max-entries=N] [write-beam=N]`
//!   starts a fresh backend, Runtime, and index; the harness model resets.
//!   Field types: `i64`, `f64`, `bool`, `string`; a `?` suffix makes the
//!   field nullable.
//! - `load dataset=SPEC tree=V|A..B [via=batch|single|import] [seed=N] [batch=N]`
//!   inserts a dataset through the public mutation API. SPECs are generated
//!   synthetically except `file:NAME[:N]`, which loads (the first N vectors
//!   of) a checked-in fixture from `tests/datadriven/data/` and ignores
//!   `seed`. Non-tree fields are filled deterministically (ordinal values;
//!   every seventh nullable field is NULL).
//! - `insert [tree=V]` / `upsert [tree=V]` — input lines `id: [v,v,...] [fI=value ...]`;
//!   prints `id: ok|created|replaced` or `id: error Kind`.
//! - `delete` — input lines of Record IDs; prints `id: true|false`.
//! - `get` — input lines of Record IDs; prints `id: present|absent`.
//! - `search k=K vector=[v,v,...] [where=F:op:value ...] [budget overrides] [beam-size=N]` —
//!   prints one `id: distance` line per hit, then exact budget usage, then
//!   any exhaustion flags.
//! - `recall k=K samples=N [query=SPEC] [query-seed=N] [where=...] [budgets]
//!   [beam-size=N] [min-recall=PERCENT]` — prints recall against the
//!   brute-force oracle plus the count of budget-truncated queries. A
//!   `query=` spec is capped at `samples` queries; without it, `samples`
//!   stride queries come from the loaded dataset. `min-recall=` adds a
//!   quality assertion without changing the recorded output.
//! - `inject-fault kind=abort|unknown-applied|unknown-not-applied` — queues one
//!   commit fault; unknown outcomes are recovered by read-back, and the model
//!   is synchronized per ADR 0012.
//! - `split-step tree=V [partition=N]` — one bounded split state-machine
//!   transition (`maintenance::split::advance`) on `partition=N`, or on the
//!   most over-full Ready partition of the tree when omitted; prints the
//!   transition (`began`/`exposed`/`drained`/`completed`/`idle`).
//! - `split tree=V [partition=N]` — drives one source to a completed split.
//! - `split-all tree=V` — settles every over-full partition of the tree,
//!   worst offender first.
//! - `merge-step tree=V [partition=N]` — one bounded merge state-machine
//!   transition (`maintenance::merge::advance`) on `partition=N`, or on an
//!   in-flight merge or the most under-full eligible Ready partition of the
//!   tree when omitted; prints the transition
//!   (`began`/`drained`/`stalled`/`completed`/`idle`).
//! - `merge tree=V [partition=N]` — drives one source to a completed merge.
//! - `load-index tree=V` — installs one exact persistent topology for the
//!   (empty) tree from annotated `format-tree`-shaped text: partition lines
//!   `pk=N level=L [state=S] [left=L right=R | source=N] [centroid=[...]]`
//!   nested by two-space indentation (first line `pk=1`, the stable root),
//!   with leaf record lines in `insert` syntax. In-flight split/merge
//!   intermediate states install directly, so corpus files can construct
//!   states that are tedious to reach by driving the state machines; later
//!   directives drive them exactly like state-machine output. Prints a
//!   one-line summary.
//! - `restart` — shuts the Runtime down and reopens the index on a reopened
//!   durable backend, simulating a process restart.
//! - `validate` — runs the exact-membership/topology audit against the model.
//! - `format-tree [entries] [tree=V]` — renders the reachable topology,
//!   including in-flight split states with their targets; `tree=V` restricts
//!   the output to one tree, keeping its ordinal in directory order.
//! - `drop-index` — drops the index.
//!
//! The split and merge directives drive the #10/#31 state machines explicitly
//! (foreground mutations never trigger structural maintenance on their own in
//! the current build), so corpus files decide the exact interleaving of
//! topology transitions, foreground mutations, and searches: every committed
//! intermediate state stays searchable and auditable.

use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{
    CompareOp, DataType, ErrorKind, FieldId, FieldSchema, ImportOptions, Index, IndexConfig,
    Metric, Mutation, PartitionKey, Predicate, Record, RuntimeConfig, SearchOptions, SearchRequest,
    Value,
};
use ktann::maintenance::merge::{self, Advance as MergeAdvance};
use ktann::maintenance::split::{self, Advance};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::keys::TreeKey;
use ktann::storage::values::PartitionState;

#[allow(dead_code)]
mod support;

use support::datadriven::{self, Directive, Mismatch};
use support::dataset::{self, Dataset};
use support::load_index::{FixturePartition, FixtureState, LoadFixture};
use support::oracle::{self, Model, ModelRecord};
use support::{CommitFault, DeterministicBackend, DeterministicConfig, Durability, SharedBackend};

/// The corpus directory, relative to the crate root.
const CORPUS_DIR: &str = "tests/datadriven";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_driven_corpus() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(CORPUS_DIR);
    let rewrite = datadriven::rewrite_enabled();
    let mut mismatches = Vec::new();
    for path in datadriven::corpus_files(&dir) {
        eprintln!("running {}", path.display());
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let directives = datadriven::parse(&path, &text);
        let mut harness = Harness::new();
        let mut outputs = Vec::with_capacity(directives.len());
        for directive in &directives {
            let actual = harness.execute(directive).await;
            // Compare immediately: a later directive may panic (e.g. a recall
            // after a failed load), which must not hide the real mismatch.
            if !rewrite && directive.expected != actual {
                let mismatch = Mismatch {
                    path: path.clone(),
                    line: directive.line,
                    raw_header: directive.raw_header.clone(),
                    expected: directive.expected.clone(),
                    actual: actual.clone(),
                };
                eprintln!("{mismatch}");
                mismatches.push(mismatch);
            }
            outputs.push(actual);
        }
        harness.shutdown().await;
        if rewrite {
            let rendered = datadriven::render(&directives, &outputs);
            if rendered != text {
                std::fs::write(&path, rendered)
                    .unwrap_or_else(|error| panic!("rewrite {}: {error}", path.display()));
            }
        }
    }
    if rewrite {
        eprintln!("corpus rewritten under {}", dir.display());
        return;
    }
    assert!(
        mismatches.is_empty(),
        "data-driven corpus mismatches (set KTANN_REWRITE=1 to regenerate):\n\n{}",
        mismatches
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

/// The oracle-side predicate filter over one modeled record.
type ModelFilter = Box<dyn Fn(&ModelRecord) -> bool>;

/// One corpus file's world: backend, runtime, index, and the caller model.
struct Harness {
    backend: Option<SharedBackend>,
    runtime: Option<Runtime<SharedBackend>>,
    index: Option<Index<SharedBackend>>,
    name: String,
    model: Model,
    dataset: Option<Dataset>,
    /// Deterministic clock for split-step, merge-step, and load-index
    /// `started_at` timestamps (and the load-index cache-epoch stamp).
    maintenance_clock: u64,
}

impl Harness {
    fn new() -> Self {
        Self {
            backend: None,
            runtime: None,
            index: None,
            name: "index".to_string(),
            model: Model::new(),
            dataset: None,
            maintenance_clock: 1_000,
        }
    }

    async fn shutdown(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown().await.expect("runtime shutdown");
        }
        self.index = None;
    }

    fn index(&self) -> &Index<SharedBackend> {
        self.index
            .as_ref()
            .expect("directive requires new-index first")
    }

    async fn execute(&mut self, directive: &Directive) -> String {
        match directive.command() {
            "new-index" => self.new_index(directive).await,
            "load" => self.load(directive).await,
            "insert" | "upsert" => self.mutate_lines(directive).await,
            "delete" => self.delete_lines(directive).await,
            "get" => self.get_lines(directive).await,
            "search" => self.search(directive).await,
            "recall" => self.recall(directive).await,
            "split-step" => self.split_step(directive).await,
            "split" => self.split_full(directive).await,
            "split-all" => self.split_all(directive).await,
            "merge-step" => self.merge_step(directive).await,
            "merge" => self.merge_full(directive).await,
            "load-index" => self.load_index(directive).await,
            "inject-fault" => self.inject_fault(directive),
            "restart" => self.restart().await,
            "validate" => self.validate().await,
            "format-tree" => self.format_tree(directive).await,
            "drop-index" => self.drop_index().await,
            other => panic!("unknown directive `{other}` at line {}", directive.line),
        }
    }

    /// `new-index` starts a completely fresh world, like a new memstore in the
    /// reference framework.
    async fn new_index(&mut self, directive: &Directive) -> String {
        self.shutdown().await;
        let name = directive.arg("name").unwrap_or("index").to_string();
        let dimension = directive.arg_usize("dimension", 0);
        assert!(dimension > 0, "new-index requires `dimension=`");
        let metric = match directive.arg("metric").unwrap_or("L2") {
            "L2" => Metric::L2,
            "Cosine" => Metric::Cosine,
            "InnerProduct" => Metric::InnerProduct,
            other => panic!("unknown metric `{other}` at line {}", directive.line),
        };
        let fields = parse_fields(directive);
        let mut config = IndexConfig::new(dimension, metric).expect("valid config");
        if !fields.is_empty() {
            config = config.with_fields(fields).expect("valid fields");
        }
        if let Some(tree_fields) = directive.arg("tree-key-fields") {
            let fields: Vec<FieldId> = tree_fields
                .split(',')
                .map(|index| FieldId(index.parse().expect("tree key field index")))
                .collect();
            config = config
                .with_tree_key_fields(fields)
                .expect("valid tree key fields");
        }
        if let (Some(minimum), Some(maximum)) =
            (directive.arg("min-entries"), directive.arg("max-entries"))
        {
            config = config
                .with_partition_entries(
                    minimum.parse().expect("min-entries"),
                    maximum.parse().expect("max-entries"),
                )
                .expect("valid partition entries");
        }
        let backend_config = DeterministicConfig {
            durability: Durability::Durable,
            ..DeterministicConfig::default()
        };
        let backend = SharedBackend::new(DeterministicBackend::new(backend_config));
        // The corpus drives the split/merge state machines through its own
        // split-step/merge-step directives and asserts every intermediate
        // state, so its Runtime runs without background maintenance workers;
        // demand-driven scheduling is covered by tests/maintenance_scheduling.rs.
        let mut runtime_config = support::manual_maintenance_config();
        if let Some(beam) = directive.arg("write-beam") {
            runtime_config = runtime_config
                .with_write_beam_size(beam.parse().expect("write-beam"))
                .expect("valid write beam");
        }
        let runtime = Runtime::new(backend.clone(), runtime_config).expect("runtime");
        match runtime.create_index(&name, config).await {
            Ok(index) => {
                self.backend = Some(backend);
                self.runtime = Some(runtime);
                self.index = Some(index);
                self.name = name.clone();
                self.model.clear();
                self.dataset = None;
                format!("created index {name:?}\n")
            }
            Err(error) => format!("error: {:?}\n", error.kind()),
        }
    }

    async fn load(&mut self, directive: &Directive) -> String {
        let spec = directive.require("dataset");
        let seed = directive.arg_u64("seed", 42);
        let dimension = self.index().config().dimension();
        let data = dataset::generate(spec, dimension, seed);
        let count = data.len();

        let via = Via::parse(directive);
        let batch_size = match via {
            Via::Single => 1,
            _ => directive.arg_usize("batch", 100),
        };

        // Every record is built exactly once; the model mirrors the same
        // committed chunks below.
        let tree = TreeRule::parse(directive);
        let records: Vec<Record> = (0..count)
            .map(|ordinal| {
                Record::new(
                    data.ids[ordinal].clone(),
                    data.vectors[ordinal].clone(),
                    fill_fields(
                        self.index().config(),
                        tree.as_ref().map(|rule| rule.value(ordinal)).as_deref(),
                        ordinal,
                        &[],
                    ),
                )
                .expect("dataset record")
            })
            .collect();

        match via {
            Via::Import => {
                // Serialized import keeps the corpus deterministic: pipelined
                // batches inserting into the same leaf legitimately race, and
                // a bounded-retry exhaustion under that contention is engine
                // behavior for engine tests, not for this corpus.
                let options = ImportOptions::default()
                    .with_max_in_flight_batches(1)
                    .expect("import options");
                let mut session = self
                    .index()
                    .import_session(options)
                    .expect("import session");
                for chunk in records.chunks(batch_size) {
                    let mutations = chunk.iter().cloned().map(Mutation::Insert).collect();
                    session.submit(mutations).await.expect("import submit");
                }
                for (chunk, result) in records.chunks(batch_size).zip(session.finish().await) {
                    match result.result {
                        Ok(outcomes) => {
                            assert_eq!(outcomes.len(), chunk.len());
                            self.accept_model_chunk(chunk);
                        }
                        Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => {
                            self.recover_chunk(chunk).await;
                        }
                        Err(error) => return format!("error: {:?}\n", error.kind()),
                    }
                }
            }
            Via::Batch | Via::Single => {
                for chunk in records.chunks(batch_size) {
                    let result = match via {
                        Via::Batch => self
                            .index()
                            .batch_mutate(chunk.iter().cloned().map(Mutation::Insert).collect())
                            .await
                            .map(|_| ()),
                        Via::Single => self.index().insert(chunk[0].clone()).await,
                        Via::Import => unreachable!("handled above"),
                    };
                    match result {
                        Ok(()) => self.accept_model_chunk(chunk),
                        Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => {
                            self.recover_chunk(chunk).await;
                        }
                        Err(error) => return format!("error: {:?}\n", error.kind()),
                    }
                }
            }
        }
        let summary = format!("loaded {count} records (dataset={spec}, seed={seed})\n");
        self.dataset = Some(data);
        summary
    }

    /// Mirrors one committed load chunk into the caller model.
    fn accept_model_chunk(&mut self, chunk: &[Record]) {
        for record in chunk {
            self.model.insert(
                record.id().clone(),
                ModelRecord {
                    vector: Arc::from(record.vector()),
                    fields: record.fields().to_vec().into_boxed_slice(),
                },
            );
        }
    }

    /// Recovers one load chunk after an unknown commit outcome by reading each
    /// record back, per the documented recovery protocol.
    async fn recover_chunk(&mut self, chunk: &[Record]) {
        for record in chunk {
            self.recover_one(record.id()).await;
        }
    }

    async fn mutate_lines(&mut self, directive: &Directive) -> String {
        let upsert = directive.command() == "upsert";
        let tree = TreeRule::parse(directive);
        let mut out = String::new();
        for (ordinal, line) in directive.input.iter().enumerate() {
            let (id, vector, explicit) = parse_record_line(line);
            let fields = fill_fields(
                self.index().config(),
                tree.as_ref().map(|rule| rule.value(ordinal)).as_deref(),
                ordinal,
                &explicit,
            );
            let record = Record::new(id.clone(), vector, fields).expect("record");
            let result = if upsert {
                self.index()
                    .upsert(record.clone())
                    .await
                    .map(|outcome| match outcome {
                        ktann::api::UpsertResult::Created => "created",
                        ktann::api::UpsertResult::Replaced => "replaced",
                        _ => unreachable!("format v1 upsert outcomes"),
                    })
            } else {
                self.index().insert(record.clone()).await.map(|_| "ok")
            };
            match result {
                Ok(label) => {
                    self.model.insert(
                        id.clone(),
                        ModelRecord {
                            vector: record.vector().into(),
                            fields: record.fields().to_vec().into_boxed_slice(),
                        },
                    );
                    out.push_str(&format!("{}: {label}\n", datadriven::show_id(&id)));
                }
                Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => {
                    let recovered = self.recover_one(&id).await;
                    out.push_str(&format!(
                        "{}: error CommitOutcomeUnknown (recovered: {recovered})\n",
                        datadriven::show_id(&id)
                    ));
                }
                Err(error) => out.push_str(&format!(
                    "{}: error {:?}\n",
                    datadriven::show_id(&id),
                    error.kind()
                )),
            }
        }
        out
    }

    async fn delete_lines(&mut self, directive: &Directive) -> String {
        let mut out = String::new();
        for line in &directive.input {
            let id = Bytes::from(line.trim().to_string());
            match self.index().delete(id.clone()).await {
                Ok(existed) => {
                    if existed {
                        self.model.remove(&id);
                    }
                    out.push_str(&format!("{}: {existed}\n", datadriven::show_id(&id)));
                }
                Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => {
                    let recovered = self.recover_one(&id).await;
                    out.push_str(&format!(
                        "{}: error CommitOutcomeUnknown (recovered: {recovered})\n",
                        datadriven::show_id(&id)
                    ));
                }
                Err(error) => out.push_str(&format!(
                    "{}: error {:?}\n",
                    datadriven::show_id(&id),
                    error.kind()
                )),
            }
        }
        out
    }

    async fn get_lines(&mut self, directive: &Directive) -> String {
        let mut out = String::new();
        for line in &directive.input {
            let id = Bytes::from(line.trim().to_string());
            match self.index().get(id.clone(), Default::default()).await {
                Ok(Some(_)) => out.push_str(&format!("{}: present\n", datadriven::show_id(&id))),
                Ok(None) => out.push_str(&format!("{}: absent\n", datadriven::show_id(&id))),
                Err(error) => out.push_str(&format!(
                    "{}: error {:?}\n",
                    datadriven::show_id(&id),
                    error.kind()
                )),
            }
        }
        out
    }

    /// Re-synchronizes one model entry after an unknown commit outcome.
    async fn recover_one(&mut self, id: &Bytes) -> &'static str {
        let stored = self
            .index()
            .get(id.clone(), Default::default())
            .await
            .expect("recovery read");
        match stored {
            Some(stored) => {
                self.model.insert(
                    id.clone(),
                    ModelRecord {
                        vector: Arc::from(stored.vector()),
                        fields: stored.fields().to_vec().into_boxed_slice(),
                    },
                );
                "present"
            }
            None => {
                self.model.remove(id);
                "absent"
            }
        }
    }

    async fn search(&mut self, directive: &Directive) -> String {
        let k = directive.arg_usize("k", 10);
        let vector = parse_vector(directive.require("vector"));
        let (request, _) = self.search_request_with(directive, k, vector);
        match self.index().search(request).await {
            Ok(outcome) => {
                let mut out = String::new();
                for hit in &outcome.hits {
                    out.push_str(&format!(
                        "{}: {}\n",
                        datadriven::show_id(hit.id()),
                        format_distance(hit.distance())
                    ));
                }
                out.push_str(&format!(
                    "usage: scanned_tree_keys={} visited_partitions={} visited_leaf_entries={} exact_rerank_candidates={}\n",
                    outcome.usage.scanned_tree_keys,
                    outcome.usage.visited_partitions,
                    outcome.usage.visited_leaf_entries,
                    outcome.usage.exact_rerank_candidates
                ));
                let mut exhausted = Vec::new();
                if outcome.exhausted.scanned_tree_keys {
                    exhausted.push("scanned_tree_keys");
                }
                if outcome.exhausted.visited_partitions {
                    exhausted.push("visited_partitions");
                }
                if outcome.exhausted.visited_leaf_entries {
                    exhausted.push("visited_leaf_entries");
                }
                if outcome.exhausted.exact_rerank_candidates {
                    exhausted.push("exact_rerank_candidates");
                }
                if !exhausted.is_empty() {
                    out.push_str(&format!("exhausted: {}\n", exhausted.join(" ")));
                }
                if outcome.rabitq_overlap_truncated {
                    out.push_str("truncated: rabitq_overlap\n");
                }
                out
            }
            Err(error) => format!("error: {:?}\n", error.kind()),
        }
    }

    async fn recall(&mut self, directive: &Directive) -> String {
        let k = directive.arg_usize("k", 10);
        let samples = directive.arg_usize("samples", 20);
        let metric = self.index().config().metric();
        let dimension = self.index().config().dimension();

        // The query set: a separately generated spec, or a deterministic
        // stride sample of the loaded dataset. A `query=` spec is capped at
        // `samples` queries to bound brute-force oracle cost.
        let queries: Vec<Arc<[f32]>> = match directive.arg("query") {
            Some(spec) => {
                let seed = directive.arg_u64("query-seed", 43);
                dataset::generate(spec, dimension, seed)
                    .vectors
                    .into_iter()
                    .take(samples)
                    .collect()
            }
            None => {
                let data = self.dataset.as_ref().unwrap_or_else(|| {
                    panic!(
                        "recall at line {} needs a loaded dataset or query= \
                         (a previous load may have failed)",
                        directive.line
                    )
                });
                (0..samples)
                    .map(|sample| data.vectors[sample * data.len() / samples].clone())
                    .collect()
            }
        };

        let mut total_recall = 0.0;
        let mut truncated = 0_usize;
        let mut total_partitions = 0_u64;
        let mut total_leaf_entries = 0_u64;
        let mut total_rerank = 0_u64;
        for query in &queries {
            let (request, filter) = self.search_request_with(directive, k, query.clone());
            let outcome = self.index().search(request).await.expect("recall search");
            let predicted: Vec<Bytes> = outcome.hits.iter().map(|hit| hit.id().clone()).collect();
            let truth = oracle::truth(&self.model, metric, query, k, &filter);
            total_recall += oracle::recall(&predicted, &truth);
            if outcome.exhausted != Default::default() || outcome.rabitq_overlap_truncated {
                truncated += 1;
            }
            total_partitions += u64::from(outcome.usage.visited_partitions);
            total_leaf_entries += u64::from(outcome.usage.visited_leaf_entries);
            total_rerank += u64::from(outcome.usage.exact_rerank_candidates);
        }
        let count = queries.len() as f64;
        let mean_recall = total_recall / count;
        if let Some(value) = directive.arg("min-recall") {
            let minimum: f64 = value.parse().unwrap_or_else(|_| {
                panic!(
                    "directive `{}` at line {}: `min-recall=` must be a finite percentage, got `{value}`",
                    directive.raw_header, directive.line
                )
            });
            assert!(
                minimum.is_finite() && (0.0..=100.0).contains(&minimum),
                "directive `{}` at line {}: `min-recall=` must be between 0 and 100, got `{minimum}`",
                directive.raw_header,
                directive.line
            );
            let actual = mean_recall * 100.0;
            assert!(
                actual >= minimum,
                "directive `{}` at line {}: recall {:.2}% is below the minimum {:.2}%",
                directive.raw_header,
                directive.line,
                actual,
                minimum
            );
        }
        format!(
            "recall@{k}: {:.2}% (samples={}, truncated={truncated})\nmean usage: visited_partitions={} visited_leaf_entries={} exact_rerank_candidates={}\n",
            mean_recall * 100.0,
            queries.len(),
            total_partitions / queries.len() as u64,
            total_leaf_entries / queries.len() as u64,
            total_rerank / queries.len() as u64,
        )
    }

    /// `split-step` performs one bounded state-machine transition
    /// (`split::advance`) on the `partition=` source, or on the most
    /// over-full Ready partition of the tree when the argument is omitted.
    async fn split_step(&mut self, directive: &Directive) -> String {
        let Some((tree_key, source)) = self.split_source(directive).await else {
            return "idle: nothing to split\n".to_string();
        };
        match self.advance_once(&tree_key, source).await {
            Ok(outcome) => format!("pk={}: {}\n", source.get(), describe_advance(&outcome)),
            Err(error) => format!("pk={}: error {:?}\n", source.get(), error.kind()),
        }
    }

    /// `split` drives one source partition's state machine to completion.
    async fn split_full(&mut self, directive: &Directive) -> String {
        let Some((tree_key, source)) = self.split_source(directive).await else {
            return "idle: nothing to split\n".to_string();
        };
        match self.run_split(&tree_key, source).await {
            Ok(summary) => summary,
            Err(error) => format!("pk={}: error {:?}\n", source.get(), error.kind()),
        }
    }

    /// `split-all` settles every over-full partition of the tree, one full
    /// split at a time, worst offender first.
    async fn split_all(&mut self, directive: &Directive) -> String {
        let tree_key = self.tree_key_arg(directive);
        let mut out = String::new();
        let mut splits = 0_usize;
        while let Some(source) = self.split_candidate(&tree_key).await {
            assert!(
                splits < 1_024,
                "split-all did not settle within 1024 splits"
            );
            match self.run_split(&tree_key, source).await {
                Ok(summary) => out.push_str(&summary),
                Err(error) => {
                    out.push_str(&format!("pk={}: error {:?}\n", source.get(), error.kind()));
                    return out;
                }
            }
            splits += 1;
        }
        out.push_str("settled\n");
        out
    }

    /// Resolves the split source for one directive: explicit `partition=N`,
    /// or the most over-full Ready partition of the tree when omitted.
    async fn split_source(&self, directive: &Directive) -> Option<(TreeKey, PartitionKey)> {
        let tree_key = self.tree_key_arg(directive);
        if let Some(raw) = directive.arg("partition") {
            let key = PartitionKey::new(raw.parse().expect("partition key"))
                .expect("split source Partition Key is nonzero");
            return Some((tree_key, key));
        }
        self.split_candidate(&tree_key)
            .await
            .map(|source| (tree_key, source))
    }

    /// The partition an argument-free split directive should drive: an
    /// in-flight split source first (smallest Partition Key, so a corpus
    /// resumes the machine it started), otherwise the most over-full Ready
    /// partition (ties: smallest key).
    async fn split_candidate(&self, tree_key: &TreeKey) -> Option<PartitionKey> {
        let backend = self
            .backend
            .as_ref()
            .expect("split directives require new-index first");
        let index = self.index().logical_index_id();
        let manifest = support::read_manifest(backend, index).await;
        let listing = support::audit::list_partitions(backend, index)
            .await
            .expect("partition listing");
        let in_flight = listing
            .iter()
            .filter(|(key, _, header)| {
                key == tree_key
                    && matches!(
                        header.state(),
                        PartitionState::Splitting | PartitionState::DrainingSplit
                    )
            })
            .map(|(_, partition, _)| *partition)
            .min();
        if in_flight.is_some() {
            return in_flight;
        }
        let maximum = manifest.config().max_partition_entries();
        listing
            .into_iter()
            .filter(|(key, _, header)| {
                key == tree_key
                    && header.state() == PartitionState::Ready
                    && header.entry_count() > maximum
            })
            .max_by(|left, right| {
                left.2
                    .entry_count()
                    .cmp(&right.2.entry_count())
                    .then_with(|| right.1.cmp(&left.1))
            })
            .map(|(_, partition, _)| partition)
    }

    /// Drives one source partition to a completed split; an immediately-idle
    /// source reports `idle`. The per-batch `moved` accounting is only
    /// visible through `split-step`: `advance` folds the final drain batch
    /// into `Completed`, so a full-split summary cannot sum moves honestly.
    async fn run_split(
        &mut self,
        tree_key: &TreeKey,
        source: PartitionKey,
    ) -> ktann::api::Result<String> {
        let mut targets = None;
        for _ in 0..4_096 {
            match self.advance_once(tree_key, source).await? {
                Advance::Idle => {
                    assert!(targets.is_none(), "split idled mid-machine");
                    return Ok(format!("pk={}: idle\n", source.get()));
                }
                Advance::Began { left, right } | Advance::Exposed { left, right } => {
                    targets = Some((left, right));
                }
                Advance::Corrected { .. } | Advance::Drained { .. } => {}
                Advance::Completed { .. } => {
                    return Ok(match targets {
                        Some((left, right)) => format!(
                            "pk={}: split into left={} right={}\n",
                            source.get(),
                            left.get(),
                            right.get()
                        ),
                        None => format!("pk={}: split resumed\n", source.get()),
                    });
                }
                other => panic!("unexpected split outcome {other:?}"),
            }
        }
        panic!(
            "split of pk={} did not complete within 4096 steps",
            source.get()
        );
    }

    /// One bounded split transition with the harness's deterministic clock.
    async fn advance_once(
        &mut self,
        tree_key: &TreeKey,
        source: PartitionKey,
    ) -> ktann::api::Result<Advance> {
        let backend = self
            .backend
            .as_ref()
            .expect("split directives require new-index first");
        let manifest = support::read_manifest(backend, self.index().logical_index_id()).await;
        let started_at = self.maintenance_clock;
        self.maintenance_clock += 100;
        split::advance(
            backend,
            &manifest,
            tree_key,
            source,
            started_at,
            &retry_policy(),
        )
        .await
    }

    /// `merge-step` performs one bounded state-machine transition
    /// (`merge::advance`) on the `partition=` source, or on an in-flight
    /// merge or the most under-full eligible Ready partition of the tree when
    /// the argument is omitted.
    async fn merge_step(&mut self, directive: &Directive) -> String {
        let Some((tree_key, source)) = self.merge_source(directive).await else {
            return "idle: nothing to merge\n".to_string();
        };
        match self.merge_advance_once(&tree_key, source).await {
            Ok(outcome) => format!(
                "pk={}: {}\n",
                source.get(),
                describe_merge_advance(&outcome)
            ),
            Err(error) => format!("pk={}: error {:?}\n", source.get(), error.kind()),
        }
    }

    /// `merge` drives one source partition's merge state machine to
    /// completion; a source that never begins reports `idle`, and a merge
    /// with no legal target reports `stalled`.
    async fn merge_full(&mut self, directive: &Directive) -> String {
        let Some((tree_key, source)) = self.merge_source(directive).await else {
            return "idle: nothing to merge\n".to_string();
        };
        match self.run_merge(&tree_key, source).await {
            Ok(summary) => summary,
            Err(error) => format!("pk={}: error {:?}\n", source.get(), error.kind()),
        }
    }

    /// Resolves the merge source for one directive: explicit `partition=N`,
    /// or an in-flight merge / the most under-full eligible Ready partition
    /// of the tree when omitted.
    async fn merge_source(&self, directive: &Directive) -> Option<(TreeKey, PartitionKey)> {
        let tree_key = self.tree_key_arg(directive);
        if let Some(raw) = directive.arg("partition") {
            let key = PartitionKey::new(raw.parse().expect("partition key"))
                .expect("merge source Partition Key is nonzero");
            return Some((tree_key, key));
        }
        self.merge_candidate(&tree_key)
            .await
            .map(|source| (tree_key, source))
    }

    /// The partition an argument-free merge directive should drive: an
    /// in-flight merge source first (smallest Partition Key, so a corpus
    /// resumes the machine it started), otherwise the most under-full
    /// eligible Ready non-root partition (fewest entries; ties: smallest
    /// key).
    async fn merge_candidate(&self, tree_key: &TreeKey) -> Option<PartitionKey> {
        let backend = self
            .backend
            .as_ref()
            .expect("merge directives require new-index first");
        let index = self.index().logical_index_id();
        let manifest = support::read_manifest(backend, index).await;
        let listing = support::audit::list_partitions(backend, index)
            .await
            .expect("partition listing");
        let in_flight = listing
            .iter()
            .filter(|(key, _, header)| key == tree_key && header.state() == PartitionState::Merging)
            .map(|(_, partition, _)| *partition)
            .min();
        if in_flight.is_some() {
            return in_flight;
        }
        let minimum = manifest.config().min_partition_entries();
        listing
            .into_iter()
            .filter(|(key, partition, header)| {
                key == tree_key
                    && partition.get() != 1
                    && header.state() == PartitionState::Ready
                    && header.entry_count() < minimum
            })
            .min_by(|left, right| {
                left.2
                    .entry_count()
                    .cmp(&right.2.entry_count())
                    .then_with(|| left.1.cmp(&right.1))
            })
            .map(|(_, partition, _)| partition)
    }

    /// Drives one source partition to a completed merge; an immediately-idle
    /// source reports `idle` and a target-less merge reports `stalled`.
    async fn run_merge(
        &mut self,
        tree_key: &TreeKey,
        source: PartitionKey,
    ) -> ktann::api::Result<String> {
        let mut began = false;
        for _ in 0..4_096 {
            match self.merge_advance_once(tree_key, source).await? {
                MergeAdvance::Idle => {
                    assert!(!began, "merge idled mid-machine");
                    return Ok(format!("pk={}: idle\n", source.get()));
                }
                MergeAdvance::Stalled => return Ok(format!("pk={}: stalled\n", source.get())),
                MergeAdvance::Began => began = true,
                MergeAdvance::Drained { .. } => {}
                MergeAdvance::Completed => return Ok(format!("pk={}: merged\n", source.get())),
                other => panic!("unexpected merge outcome {other:?}"),
            }
        }
        panic!(
            "merge of pk={} did not complete within 4096 steps",
            source.get()
        );
    }

    /// One bounded merge transition with the harness's deterministic clock.
    async fn merge_advance_once(
        &mut self,
        tree_key: &TreeKey,
        source: PartitionKey,
    ) -> ktann::api::Result<MergeAdvance> {
        let backend = self
            .backend
            .as_ref()
            .expect("merge directives require new-index first");
        let manifest = support::read_manifest(backend, self.index().logical_index_id()).await;
        let started_at = self.maintenance_clock;
        self.maintenance_clock += 100;
        merge::advance(
            backend,
            &manifest,
            tree_key,
            source,
            started_at,
            &retry_policy(),
        )
        .await
    }

    /// `load-index` installs one exact persistent topology state for the
    /// (empty) `tree=V` tree — including in-flight split/merge intermediate
    /// states — directly from the annotated text, so corpus files can
    /// construct states that are tedious to reach by driving the state
    /// machines (issue #100, item C2). Installed states are byte-equivalent
    /// to what the state machines persist; later directives drive them
    /// exactly like state-machine output.
    async fn load_index(&mut self, directive: &Directive) -> String {
        let tree_key = self.tree_key_arg(directive);
        let fixture = self.parse_load_fixture(directive);
        // The deterministic maintenance clock supplies both the persisted
        // started-at timestamps and a fresh cache epoch stamped on every
        // Header the installer writes: an earlier search may have warmed the
        // runtime partition cache under the tree's pre-install epochs.
        let started_at = self.maintenance_clock;
        self.maintenance_clock += 100;
        let cache_epoch = self.maintenance_clock;
        self.maintenance_clock += 100;
        let summary = support::load_index::install(
            self.backend
                .as_ref()
                .expect("load-index requires new-index first"),
            self.index(),
            &fixture,
            &tree_key,
            started_at,
            cache_epoch,
            directive.line,
        )
        .await;
        self.accept_model_chunk(&fixture.records);
        format!(
            "loaded {} records, {} partitions, max level {}\n",
            summary.records, summary.partitions, summary.max_level
        )
    }

    /// Parses the `load-index` input lines into a fixture: partition lines
    /// nested by two-space indentation (exactly like `format-tree` renders,
    /// minus the `tree N:` line), and leaf record lines in `insert` syntax
    /// with the Tree Key field filled from `tree=V`.
    fn parse_load_fixture(&self, directive: &Directive) -> LoadFixture {
        let line = directive.line;
        let config = self.index().config();
        let dimension = config.dimension();
        let tree_value = directive.require("tree");
        let mut fixture = LoadFixture::default();
        // The current nesting path: (depth, fixture position) pairs.
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for input in &directive.input {
            let indent = input.len() - input.trim_start_matches(' ').len();
            let content = &input[indent..];
            assert!(
                indent % 2 == 0 && !content.is_empty() && !content.starts_with(char::is_whitespace),
                "load-index at line {line}: bad indentation in `{input}`"
            );
            let depth = indent / 2;
            while stack.last().is_some_and(|&(top, _)| top >= depth) {
                stack.pop();
            }
            if content.starts_with("pk=") {
                let mut partition = parse_partition_line(content, line, dimension);
                partition.parent_line = match stack.last() {
                    Some(&(parent_depth, parent)) => {
                        assert!(
                            parent_depth == depth - 1,
                            "load-index at line {line}: `{input}` skips a nesting level"
                        );
                        Some(fixture.partitions[parent].key)
                    }
                    None => {
                        assert!(
                            depth == 0,
                            "load-index at line {line}: `{input}` skips a nesting level"
                        );
                        assert!(
                            fixture.partitions.is_empty(),
                            "load-index at line {line}: only the first (root) line sits at \
                             depth 0"
                        );
                        None
                    }
                };
                stack.push((depth, fixture.partitions.len()));
                fixture.partitions.push(partition);
            } else {
                let Some(&(parent_depth, parent)) = stack.last() else {
                    panic!(
                        "load-index at line {line}: record line `{input}` nests under no \
                         partition line"
                    )
                };
                assert!(
                    parent_depth == depth - 1,
                    "load-index at line {line}: `{input}` skips a nesting level"
                );
                let (id, vector, explicit) = parse_record_line(content);
                assert!(
                    vector.len() == dimension,
                    "load-index at line {line}: record vector has {} components, dimension is \
                     {dimension}",
                    vector.len()
                );
                let fields =
                    fill_fields(config, Some(tree_value), fixture.records.len(), &explicit);
                let record = Record::new(id.clone(), vector, fields).expect("record");
                fixture.partitions[parent].records.push(id);
                fixture.records.push(record);
            }
        }
        fixture
    }

    /// Encodes the `tree=V` argument as the Tree Key. The split, merge, and
    /// load-index directives support the corpus's single i64 tree-key field
    /// shape only.
    fn tree_key_arg(&self, directive: &Directive) -> TreeKey {
        let value = directive.require("tree");
        let config = self.index().config();
        let tree_fields = config.tree_key_fields();
        assert!(
            tree_fields.len() == 1,
            "split/merge directives need a single tree-key field"
        );
        let schema = &config.fields()[usize::from(tree_fields[0].0)];
        assert!(
            schema.data_type() == DataType::I64,
            "split/merge directives need an i64 tree-key field"
        );
        TreeKey::encode(
            &[DataType::I64],
            &[Value::I64(value.parse().expect("tree value"))],
        )
        .expect("canonical tree key")
    }

    fn search_request_with(
        &self,
        directive: &Directive,
        k: usize,
        vector: Arc<[f32]>,
    ) -> (SearchRequest, ModelFilter) {
        let mut request = SearchRequest::new(vector, k).expect("valid search request");
        let (predicate, filter) = self.predicate(directive);
        if let Some(predicate) = predicate {
            request = request.with_predicate(predicate);
        }
        request = request.with_options(search_options(directive));
        (request, filter)
    }

    /// Builds the API predicate and the matching oracle filter from `where=`
    /// arguments. Grammar: `F:op:value` with op one of
    /// `eq|ne|lt|le|gt|ge|in|isnull|notnull`; repeated arguments conjoin.
    fn predicate(&self, directive: &Directive) -> (Option<Predicate>, ModelFilter) {
        let fields = self.index().config().fields().to_vec();
        let mut predicates = Vec::new();
        let mut filters: Vec<ModelFilter> = Vec::new();
        for clause in directive.args_of("where") {
            let mut parts = clause.splitn(3, ':');
            let field_index: u16 = parts
                .next()
                .and_then(|part| part.parse().ok())
                .unwrap_or_else(|| panic!("bad where clause `{clause}`"));
            let op = parts
                .next()
                .unwrap_or_else(|| panic!("bad where clause `{clause}`"));
            let field = FieldId(field_index);
            let data_type = fields
                .get(usize::from(field_index))
                .unwrap_or_else(|| panic!("where field out of range in `{clause}`"))
                .data_type();
            match op {
                "isnull" => {
                    predicates.push(Predicate::IsNull(field));
                    filters.push(Box::new(move |record| {
                        matches!(record.fields[usize::from(field_index)], Value::Null)
                    }));
                }
                "notnull" => {
                    predicates.push(Predicate::IsNotNull(field));
                    filters.push(Box::new(move |record| {
                        !matches!(record.fields[usize::from(field_index)], Value::Null)
                    }));
                }
                "in" => {
                    let raw = parts.next().unwrap_or_else(|| panic!("in needs values"));
                    let values: Vec<Value> = raw
                        .split(',')
                        .map(|part| parse_typed(part, data_type))
                        .collect();
                    let oracle_values = values.clone();
                    predicates.push(Predicate::In { field, values });
                    filters.push(Box::new(move |record| {
                        oracle_values.iter().any(|value| {
                            oracle::compare_3vl(
                                CompareOp::Eq,
                                &record.fields[usize::from(field_index)],
                                value,
                            )
                        })
                    }));
                }
                other => {
                    let (api_op, oracle_op) = match other {
                        "eq" => (CompareOp::Eq, CompareOp::Eq),
                        "ne" => (CompareOp::NotEq, CompareOp::NotEq),
                        "lt" => (CompareOp::Lt, CompareOp::Lt),
                        "le" => (CompareOp::LessOrEqual, CompareOp::LessOrEqual),
                        "gt" => (CompareOp::Gt, CompareOp::Gt),
                        "ge" => (CompareOp::GreaterOrEqual, CompareOp::GreaterOrEqual),
                        unknown => panic!("unknown where op `{unknown}`"),
                    };
                    let value = parse_typed(
                        parts
                            .next()
                            .unwrap_or_else(|| panic!("where op needs a value")),
                        data_type,
                    );
                    predicates.push(Predicate::Compare {
                        field,
                        op: api_op,
                        value: value.clone(),
                    });
                    filters.push(Box::new(move |record| {
                        oracle::compare_3vl(
                            oracle_op,
                            &record.fields[usize::from(field_index)],
                            &value,
                        )
                    }));
                }
            }
        }
        let predicate = match predicates.len() {
            0 => None,
            1 => predicates.pop(),
            _ => Some(Predicate::And(predicates)),
        };
        (
            predicate,
            Box::new(move |record| filters.iter().all(|filter| filter(record))),
        )
    }

    fn inject_fault(&mut self, directive: &Directive) -> String {
        let kind = match directive.require("kind") {
            "abort" => CommitFault::Abort,
            "unknown-applied" => CommitFault::UnknownApplied,
            "unknown-not-applied" => CommitFault::UnknownNotApplied,
            other => panic!("unknown fault kind `{other}` at line {}", directive.line),
        };
        self.backend
            .as_ref()
            .expect("inject-fault requires new-index first")
            .inner()
            .push_fault(kind)
            .expect("push fault");
        "ok\n".to_string()
    }

    async fn restart(&mut self) -> String {
        self.shutdown().await;
        let backend = self
            .backend
            .take()
            .expect("restart requires new-index first");
        let backend = SharedBackend::new(backend.inner().reopen());
        let runtime =
            Runtime::new(backend.clone(), support::manual_maintenance_config()).expect("runtime");
        match runtime.open_index(&self.name).await {
            Ok(index) => {
                self.backend = Some(backend);
                self.runtime = Some(runtime);
                self.index = Some(index);
                "restarted\n".to_string()
            }
            Err(error) => format!("error: {:?}\n", error.kind()),
        }
    }

    async fn validate(&mut self) -> String {
        let backend = self
            .backend
            .as_ref()
            .expect("validate requires new-index first");
        let index = self.index().logical_index_id();
        match support::audit::run(backend, index, &self.model).await {
            Ok(report) => format!(
                "ok: records={} trees={} partitions={} max_level={}\n",
                report.records, report.trees, report.partitions, report.max_level
            ),
            Err(mismatch) => format!("audit mismatch: {mismatch}\n"),
        }
    }

    async fn format_tree(&mut self, directive: &Directive) -> String {
        let backend = self
            .backend
            .as_ref()
            .expect("format-tree requires new-index first");
        let index = self.index().logical_index_id();
        let filter = directive.arg("tree").map(|_| self.tree_key_arg(directive));
        match support::audit::render_tree(
            backend,
            index,
            directive.flag("entries"),
            filter.as_ref(),
        )
        .await
        {
            Ok(rendered) => rendered,
            Err(mismatch) => format!("audit mismatch: {mismatch}\n"),
        }
    }

    async fn drop_index(&mut self) -> String {
        let runtime = self
            .runtime
            .as_ref()
            .expect("drop-index requires new-index first");
        match runtime.drop_index(&self.name).await {
            Ok(()) => {
                self.index = None;
                self.model.clear();
                "dropped\n".to_string()
            }
            Err(error) => format!("error: {:?}\n", error.kind()),
        }
    }
}

/// Parses the `fields=` spec: `name:type[?]` entries separated by commas.
fn parse_fields(directive: &Directive) -> Vec<FieldSchema> {
    directive
        .arg("fields")
        .map(|spec| {
            spec.split(',')
                .map(|field| {
                    let mut parts = field.split(':');
                    let name = parts.next().expect("field name");
                    let type_part = parts
                        .next()
                        .unwrap_or_else(|| panic!("field `{name}` needs a type"));
                    let nullable = type_part.ends_with('?');
                    let data_type = match type_part.trim_end_matches('?') {
                        "i64" => DataType::I64,
                        "f64" => DataType::F64,
                        "bool" => DataType::Bool,
                        "string" => DataType::String,
                        other => panic!("unknown field type `{other}`"),
                    };
                    let schema = FieldSchema::new(name, data_type).expect("field schema");
                    if nullable { schema.nullable() } else { schema }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parses one `fI=value` field override from a record input line.
fn parse_record_line(line: &str) -> (Bytes, Arc<[f32]>, Vec<(usize, String)>) {
    let (id, rest) = line
        .split_once(':')
        .unwrap_or_else(|| panic!("record line needs `id: [vector]`, got `{line}`"));
    let mut tokens = rest.split_whitespace();
    let vector = parse_vector(
        tokens
            .next()
            .unwrap_or_else(|| panic!("record line `{line}` needs a vector")),
    );
    let mut fields = Vec::new();
    for token in tokens {
        let (name, value) = token
            .split_once('=')
            .unwrap_or_else(|| panic!("bad field override `{token}`"));
        let index = name
            .strip_prefix('f')
            .and_then(|index| index.parse().ok())
            .unwrap_or_else(|| panic!("field override must be `fI=value`, got `{name}`"));
        fields.push((index, value.to_string()));
    }
    (Bytes::from(id.to_string()), vector, fields)
}

/// Parses one `load-index` partition line: `pk=N level=L [state=S]` plus the
/// state's parameters (`left=L right=R` for Splitting/DrainingSplit,
/// `source=N` for ReceivingSplit) and an optional `centroid=[...]`. The
/// nesting parent and record lines are filled in by the caller.
fn parse_partition_line(content: &str, line: usize, dimension: usize) -> FixturePartition {
    let mut tokens = content.split_whitespace();
    let first = tokens
        .next()
        .unwrap_or_else(|| panic!("load-index at line {line}: empty partition line"));
    let key = parse_partition_key(
        first
            .strip_prefix("pk=")
            .expect("partition line starts with `pk=`"),
        line,
    );
    let mut level: Option<u32> = None;
    let mut state: Option<&str> = None;
    let mut left = None;
    let mut right = None;
    let mut source = None;
    let mut centroid = None;
    for token in tokens {
        let (name, value) = token
            .split_once('=')
            .unwrap_or_else(|| panic!("load-index at line {line}: bad partition token `{token}`"));
        match name {
            "level" => set_once(
                &mut level,
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("load-index at line {line}: bad level `{value}`")),
                name,
                line,
            ),
            "state" => set_once(&mut state, value, name, line),
            "left" => set_once(&mut left, parse_partition_key(value, line), name, line),
            "right" => set_once(&mut right, parse_partition_key(value, line), name, line),
            "source" => set_once(&mut source, parse_partition_key(value, line), name, line),
            "centroid" => {
                let components = parse_vector(value);
                assert!(
                    components.len() == dimension,
                    "load-index at line {line}: centroid has {} components, dimension is \
                     {dimension}",
                    components.len()
                );
                set_once(&mut centroid, components, name, line);
            }
            other => panic!("load-index at line {line}: unknown partition token `{other}=`"),
        }
    }
    let level = level.unwrap_or_else(|| {
        panic!(
            "load-index at line {line}: partition pk={} needs `level=`",
            key.get()
        )
    });
    let state = match state.unwrap_or("Ready") {
        name @ ("Ready" | "Merging") => {
            assert!(
                left.is_none() && right.is_none() && source.is_none(),
                "load-index at line {line}: {name} takes no `left=`/`right=`/`source=` parameters"
            );
            match name {
                "Ready" => FixtureState::Ready,
                _ => FixtureState::Merging,
            }
        }
        name @ ("Splitting" | "DrainingSplit") => {
            assert!(
                source.is_none(),
                "load-index at line {line}: {name} takes no `source=` parameter"
            );
            let left =
                left.unwrap_or_else(|| panic!("load-index at line {line}: {name} needs `left=`"));
            let right =
                right.unwrap_or_else(|| panic!("load-index at line {line}: {name} needs `right=`"));
            match name {
                "Splitting" => FixtureState::Splitting { left, right },
                _ => FixtureState::DrainingSplit { left, right },
            }
        }
        "ReceivingSplit" => {
            assert!(
                left.is_none() && right.is_none(),
                "load-index at line {line}: ReceivingSplit takes no `left=`/`right=` parameters"
            );
            FixtureState::ReceivingSplit {
                source: source.unwrap_or_else(|| {
                    panic!("load-index at line {line}: ReceivingSplit needs `source=`")
                }),
            }
        }
        other => panic!("load-index at line {line}: unknown partition state `{other}`"),
    };
    FixturePartition {
        key,
        level,
        state,
        centroid,
        parent_line: None,
        records: Vec::new(),
    }
}

/// Parses one partition-key value (of `pk=N`, `left=N`, `right=N`, or
/// `source=N`) into a Partition Key.
fn parse_partition_key(value: &str, line: usize) -> PartitionKey {
    let raw = value
        .parse()
        .unwrap_or_else(|_| panic!("load-index at line {line}: bad partition key `{value}`"));
    PartitionKey::new(raw)
        .unwrap_or_else(|_| panic!("load-index at line {line}: partition key `{value}` is zero"))
}

/// Sets one not-yet-seen partition-line token value.
fn set_once<T>(slot: &mut Option<T>, value: T, name: &str, line: usize) {
    assert!(
        slot.replace(value).is_none(),
        "load-index at line {line}: duplicate `{name}=` token"
    );
}

/// Parses one override value against the field's declared type.
fn parse_typed(raw: &str, data_type: DataType) -> Value {
    match data_type {
        DataType::I64 => Value::I64(raw.parse().expect("i64 value")),
        DataType::F64 => Value::f64(raw.parse().expect("f64 value")).expect("finite f64"),
        DataType::Bool => Value::Bool(raw.parse().expect("bool value")),
        DataType::String => Value::string(raw).expect("string value"),
        _ => unreachable!("format v1 field types"),
    }
}

/// Parses a `[v1,v2,...]` vector (no inner spaces) into f32 components.
fn parse_vector(raw: &str) -> Arc<[f32]> {
    let inner = raw
        .strip_prefix('[')
        .and_then(|raw| raw.strip_suffix(']'))
        .unwrap_or_else(|| panic!("vector must be `[v1,v2,...]`, got `{raw}`"));
    inner
        .split(',')
        .map(|component| component.parse().expect("f32 component"))
        .collect()
}

/// The `tree=` rule: a constant Tree Key value or an `A..B` round-robin
/// range, parsed once per directive.
enum TreeRule {
    Constant(String),
    Range { low: i64, span: i64 },
}

impl TreeRule {
    fn parse(directive: &Directive) -> Option<TreeRule> {
        directive.arg("tree").map(|rule| {
            if let Some((low, high)) = rule.split_once("..") {
                let low: i64 = low.parse().expect("tree range low");
                let high: i64 = high.parse().expect("tree range high");
                assert!(low <= high, "tree range must be ascending");
                TreeRule::Range {
                    low,
                    span: high - low + 1,
                }
            } else {
                TreeRule::Constant(rule.to_string())
            }
        })
    }

    /// Resolves the Tree Key value for one record ordinal.
    fn value(&self, ordinal: usize) -> String {
        match self {
            Self::Constant(value) => value.clone(),
            Self::Range { low, span } => (*low + (ordinal % *span as usize) as i64).to_string(),
        }
    }
}

/// The `via=` load path.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Via {
    Batch,
    Single,
    Import,
}

impl Via {
    fn parse(directive: &Directive) -> Via {
        match directive.arg("via").unwrap_or("batch") {
            "batch" => Via::Batch,
            "single" => Via::Single,
            "import" => Via::Import,
            other => panic!("unknown via `{other}` at line {}", directive.line),
        }
    }
}

/// Builds the full positional field list for a record: tree-key fields from
/// the tree rule, explicit `fI=` overrides, and deterministic defaults for the
/// rest (ordinal values; every seventh nullable field is NULL).
fn fill_fields(
    config: &IndexConfig,
    tree: Option<&str>,
    ordinal: usize,
    explicit: &[(usize, String)],
) -> Box<[Value]> {
    config
        .fields()
        .iter()
        .enumerate()
        .map(|(index, schema)| {
            if config
                .tree_key_fields()
                .iter()
                .any(|field| usize::from(field.0) == index)
            {
                let rule =
                    tree.unwrap_or_else(|| panic!("schema with tree key fields needs `tree=`"));
                return parse_typed(rule, schema.data_type());
            }
            if let Some((_, raw)) = explicit.iter().find(|(field, _)| *field == index) {
                return parse_typed(raw, schema.data_type());
            }
            if schema.is_nullable() && ordinal % 7 == 3 {
                return Value::Null;
            }
            match schema.data_type() {
                DataType::I64 => Value::I64(ordinal as i64),
                DataType::F64 => Value::f64(ordinal as f64 + 0.5).expect("finite"),
                DataType::Bool => Value::Bool(ordinal % 2 == 0),
                DataType::String => Value::string(format!("s{ordinal:06}")).expect("string"),
                _ => unreachable!("format v1 field types"),
            }
        })
        .collect()
}

/// Keeps small KDDT datasets approximate enough to exercise tree routing.
/// Production requests retain their independent default of 128.
const TEST_DEFAULT_LEAF_BEAM: u32 = 8;

/// Builds Search Budget overrides from directive arguments.
fn search_options(directive: &Directive) -> SearchOptions {
    let mut options = SearchOptions::default()
        .with_leaf_beam_size(TEST_DEFAULT_LEAF_BEAM)
        .expect("valid test default leaf beam");
    if let Some(value) = directive.arg("scanned-tree-keys") {
        options = options
            .with_scanned_tree_keys(value.parse().expect("scanned-tree-keys"))
            .expect("valid");
    }
    if let Some(value) = directive.arg("visited-partitions") {
        options = options
            .with_visited_partitions(value.parse().expect("visited-partitions"))
            .expect("valid");
    }
    if let Some(value) = directive.arg("visited-leaf-entries") {
        options = options
            .with_visited_leaf_entries(value.parse().expect("visited-leaf-entries"))
            .expect("valid");
    }
    if let Some(value) = directive.arg("beam-size") {
        options = options
            .with_leaf_beam_size(value.parse().expect("beam-size"))
            .expect("valid");
    }
    options
}

/// Formats an exact distance at fixed precision, canonicalizing negative zero.
fn format_distance(distance: f64) -> String {
    let rounded = (distance * 10_000.0).round() / 10_000.0;
    let rounded = if rounded == 0.0 { 0.0 } else { rounded };
    format!("{rounded:.4}")
}

/// Renders one split-step outcome for the corpus.
fn describe_advance(outcome: &Advance) -> String {
    match outcome {
        Advance::Idle => "idle".to_string(),
        Advance::Began { left, right } => {
            format!("began left={} right={}", left.get(), right.get())
        }
        Advance::Exposed { left, right } => {
            format!("exposed left={} right={}", left.get(), right.get())
        }
        Advance::Corrected { moved } => format!("corrected moved={moved}"),
        Advance::Drained { moved, remaining } => {
            format!("drained moved={moved} remaining={remaining}")
        }
        Advance::Completed { .. } => "completed".to_string(),
        other => panic!("unexpected split outcome {other:?}"),
    }
}

/// Renders one merge-step outcome for the corpus.
fn describe_merge_advance(outcome: &MergeAdvance) -> String {
    match outcome {
        MergeAdvance::Idle => "idle".to_string(),
        MergeAdvance::Began => "began".to_string(),
        MergeAdvance::Drained { moved, remaining } => {
            format!("drained moved={moved} remaining={remaining}")
        }
        MergeAdvance::Stalled => "stalled".to_string(),
        MergeAdvance::Completed => "completed".to_string(),
        other => panic!("unexpected merge outcome {other:?}"),
    }
}

/// The bounded fixup retry policy used by split and merge steps.
fn retry_policy() -> RetryPolicy {
    RetryPolicy::for_fixup(&RuntimeConfig::default())
}
