//! The data-driven integration corpus runner (issue #94).
//!
//! Executes every block of every `tests/datadriven/*.kddt` file against the
//! public Runtime/Index API on the deterministic backend, and diffs the
//! actual output against the recorded expectation. `KTANN_REWRITE=1 cargo test
//! --test e2e` regenerates the corpus instead of failing; a rewrite must be
//! reviewed like any other change.
//!
//! Directives:
//!
//! - `new-index name=N dimension=D metric=M fields=f:t[?],... tree-key-fields=I,... [min-entries=N] [max-entries=N]`
//!   starts a fresh backend, Runtime, and index; the harness model resets.
//!   Field types: `i64`, `f64`, `bool`, `string`; a `?` suffix makes the
//!   field nullable.
//! - `load dataset=SPEC tree=V|A..B [via=batch|single|import] [seed=N] [batch=N]`
//!   inserts a dataset through the public mutation API. SPECs are generated
//!   synthetically except `file:NAME`, which loads a checked-in fixture from
//!   `tests/datadriven/data/` and ignores `seed`. Non-tree fields are filled
//!   deterministically (ordinal values; every seventh nullable field is NULL).
//! - `insert [tree=V]` / `upsert [tree=V]` — input lines `id: [v,v,...] [fI=value ...]`;
//!   prints `id: ok|created|replaced` or `id: error Kind`.
//! - `delete` — input lines of Record IDs; prints `id: true|false`.
//! - `get` — input lines of Record IDs; prints `id: present|absent`.
//! - `search k=K vector=[v,v,...] [where=F:op:value ...] [budget overrides]` —
//!   prints one `id: distance` line per hit, then exact budget usage, then
//!   any exhaustion flags.
//! - `recall k=K samples=N [query=SPEC] [query-seed=N] [where=...] [budgets]` —
//!   prints recall against the brute-force oracle plus the count of
//!   budget-truncated queries. A `query=` spec is capped at `samples`
//!   queries; without it, `samples` stride queries come from the loaded
//!   dataset.
//! - `inject-fault kind=abort|unknown-applied|unknown-not-applied` — queues one
//!   commit fault; unknown outcomes are recovered by read-back, and the model
//!   is synchronized per ADR 0012.
//! - `restart` — shuts the Runtime down and reopens the index on a reopened
//!   durable backend, simulating a process restart.
//! - `validate` — runs the exact-membership/topology audit against the model.
//! - `format-tree [entries]` — renders the reachable topology.
//! - `stats` — prints committed keyspace size.
//! - `drop-index` — drops the index.
//!
//! Until the split state machine (#10) lands, every corpus tree is a single
//! level-1 root; `validate` and `format-tree` then start covering internal
//! levels and intermediate topology states without changing the corpus
//! format.

use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{
    CompareOp, DataType, ErrorKind, FieldId, FieldSchema, ImportOptions, Index, IndexConfig,
    Metric, Mutation, Predicate, Record, RuntimeConfig, SearchOptions, SearchRequest, Value,
};
use ktann::runtime::Runtime;

#[allow(dead_code)]
mod support;

use support::datadriven::{self, Directive, Mismatch};
use support::dataset::{self, Dataset};
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
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let directives = datadriven::parse(&path, &text);
        let mut harness = Harness::new();
        let mut outputs = Vec::with_capacity(directives.len());
        for directive in &directives {
            outputs.push(harness.execute(directive).await);
        }
        harness.shutdown().await;
        if rewrite {
            let rendered = datadriven::render(&directives, &outputs);
            if rendered != text {
                std::fs::write(&path, rendered)
                    .unwrap_or_else(|error| panic!("rewrite {}: {error}", path.display()));
            }
        } else {
            for (directive, actual) in directives.iter().zip(&outputs) {
                if directive.expected != *actual {
                    mismatches.push(Mismatch {
                        path: path.clone(),
                        line: directive.line,
                        raw_header: directive.raw_header.clone(),
                        expected: directive.expected.clone(),
                        actual: actual.clone(),
                    });
                }
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
            "inject-fault" => self.inject_fault(directive),
            "restart" => self.restart().await,
            "validate" => self.validate().await,
            "format-tree" => self.format_tree(directive).await,
            "stats" => self.stats(),
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
        let runtime = Runtime::new(backend.clone(), RuntimeConfig::default()).expect("runtime");
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
                let mut session = self
                    .index()
                    .import_session(ImportOptions::default())
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
                if outcome.rabitq_overlap_truncated {
                    exhausted.push("rabitq_overlap_truncated");
                }
                if !exhausted.is_empty() {
                    out.push_str(&format!("exhausted: {}\n", exhausted.join(" ")));
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
                let data = self
                    .dataset
                    .as_ref()
                    .expect("recall needs a loaded dataset or query=");
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
        format!(
            "recall@{k}: {:.2}% (samples={}, truncated={truncated})\nmean usage: visited_partitions={} visited_leaf_entries={} exact_rerank_candidates={}\n",
            total_recall / count * 100.0,
            queries.len(),
            total_partitions / queries.len() as u64,
            total_leaf_entries / queries.len() as u64,
            total_rerank / queries.len() as u64,
        )
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
        let runtime = Runtime::new(backend.clone(), RuntimeConfig::default()).expect("runtime");
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
        match support::audit::render_tree(backend, index, directive.flag("entries")).await {
            Ok(rendered) => rendered,
            Err(mismatch) => format!("audit mismatch: {mismatch}\n"),
        }
    }

    fn stats(&mut self) -> String {
        let backend = self
            .backend
            .as_ref()
            .expect("stats requires new-index first");
        format!(
            "keys={} bytes={}\n",
            backend.inner().db_key_count(),
            backend.inner().db_byte_count()
        )
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

/// Builds Search Budget overrides from directive arguments.
fn search_options(directive: &Directive) -> SearchOptions {
    let mut options = SearchOptions::default();
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
    if let Some(value) = directive.arg("exact-rerank") {
        options = options
            .with_exact_rerank_candidates(value.parse().expect("exact-rerank"))
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
