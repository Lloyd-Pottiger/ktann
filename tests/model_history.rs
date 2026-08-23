//! Replayable crash-history and model-validation coverage (issue #37): one
//! seeded script drives the public API through lifecycle transitions, atomic
//! Foreground Mutations (some armed with commit faults), manually advanced
//! split/merge Structure Maintenance, queue loss via crash/reopen, unknown
//! commit outcomes, cancellation, and shutdown, asserting the system
//! invariants after every step.
//!
//! After every step the harness runs the exact-membership audit (every
//! committed Vector Record has exactly one Record Location and one
//! corresponding Leaf Entry, exact partition counts, exactly one incoming
//! Child Entry per non-root partition, walkable topology), checks Partition
//! Key non-reuse from a fresh partition listing, and keeps the Logical Index
//! ID history so a recreated index can never reuse an identity. Search steps
//! validate every hit against the brute-force oracle; when no Search Budget
//! dimension reports exhaustion the full ordered hit list must equal the
//! oracle truth exactly.
//!
//! Determinism argument: the script is fully pre-generated from one
//! `support::Rng` seed before any async work; the driver is a single
//! sequential task; the Runtime runs with zero maintenance workers so
//! topology moves only when the script synchronously drives one bounded
//! `split::advance`/`merge::advance`; armed faults are pushed immediately
//! before the single operation they target and are consumed by that
//! operation's first commit (leftovers from an operation that fails before
//! commit are cleared at the end of the step); retry jitter is timing-only
//! and cannot change outcomes with a single non-conflicting driver. The seed
//! therefore reproduces the whole history. On failure the driver prints its
//! trace ring and a replay command:
//!
//! ```text
//! KTANN_MODEL_SEED=<seed> cargo test --test model_history model_history_replay -- --nocapture
//! ```
//!
//! `KTANN_MODEL_STEPS` overrides the default step count. CI runs four fixed
//! seeds at 160 steps; the expanded deterministic profile
//! (`KTANN_MODEL_PROFILE=expanded`) runs 24 seeds at 400 steps from a
//! distinct base seed.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use ktann::api::{
    CompareOp, DataType, Error, ErrorKind, FieldId, FieldSchema, GetOptions, Index, IndexConfig,
    LogicalIndexId, Metric, Mutation, MutationOutcome, OperationOptions, PartitionKey, Predicate,
    Record, RuntimeConfig, SearchRequest, StoredRecord, UpsertResult, Value,
};
use ktann::maintenance::merge::{self, Advance as MergeAdvance};
use ktann::maintenance::split::{self, Advance as SplitAdvance};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::keys::TreeKey;
use ktann::storage::values::{PartitionHeader, PartitionState};
use tokio_util::sync::CancellationToken;

#[allow(dead_code)]
mod support;

use support::oracle::{self, Model, ModelRecord};
use support::{
    CommitFault, DeterministicBackend, DeterministicConfig, Durability, Rng, SharedBackend, audit,
};

/// The vector dimension.
const DIMENSION: usize = 4;
/// The single Tree Key bucket: one tree concentrates split/merge churn.
const TREE_BUCKET: i64 = 1;
/// The second (non-Tree-Key) i64 field, the target of `eq` Filter Predicates.
const TAG_FIELD: u16 = 1;
/// The `tag` value domain.
const TAG_DOMAIN: u64 = 4;
/// The Record ID pool: `r{ordinal}` for ordinal in `0..ID_POOL`.
const ID_POOL: u64 = 24;
/// Partition entry bounds sized so the 24-ID pool saturates the single tree
/// right at the split threshold (7 entries) in growth phases, while shrink
/// phases pull the small post-split leaves under the merge threshold.
const MIN_PARTITION_ENTRIES: u32 = 3;
const MAX_PARTITION_ENTRIES: u32 = 6;
/// The armed-fault rate for mutations, batches, and maintenance steps.
const FAULT_PERCENT: u64 = 15;
/// The growth/shrink phase length in script steps: kind weights alternate so
/// the tree oscillates across the split and merge thresholds.
const PHASE_STEPS: usize = 32;
/// The `eq`-predicate rate for search steps.
const FILTER_PERCENT: u64 = 35;
/// Fixup attempts for manually driven advances: an armed abort is retried
/// inside the bounded advance; an unknown outcome is never retried.
const FIXUP_ATTEMPTS: u32 = 4;
/// Foreground attempts: one, so an armed abort surfaces to the caller (as
/// `ContentionExhausted` after the single attempt) rather than being retried
/// away; no contention exists with a single driver.
const FOREGROUND_ATTEMPTS: u32 = 1;
/// The fixed CI seeds and step count.
const CI_SEEDS: [u64; 4] = [
    0x5EED_0037_0000_0001,
    0x5EED_0037_0000_0002,
    0x5EED_0037_0000_0003,
    0x5EED_0037_0000_0004,
];
const CI_STEPS: usize = 160;
/// The expanded deterministic profile: 24 seeds at 400 steps from a distinct
/// base seed, run only under `KTANN_MODEL_PROFILE=expanded`.
const EXPANDED_SEEDS: u64 = 24;
const EXPANDED_STEPS: usize = 400;
const EXPANDED_BASE_SEED: u64 = 0x5EED_0037_E0E0_0001;
/// The per-script cap on drop/recreate steps: each drop wipes the tree, and
/// at the CI step count a third wipe leaves too little room for split/merge
/// churn.
const MAX_DROP_RECREATES: usize = 2;
/// The trace ring capacity: the last traces before a failure print.
const TRACE_CAPACITY: usize = 64;
/// The single Index Name the script recreates across restarts and drops.
const INDEX_NAME: &str = "model-history";
/// The absolute tolerance for engine-vs-oracle distance agreement.
const DISTANCE_TOLERANCE: f64 = 1e-6;

/// One pre-drawn Foreground Mutation item.
#[derive(Clone, Debug)]
enum DrawnMutation {
    /// `insert` on the drawn Record ID.
    Insert {
        ordinal: u64,
        vector: [f32; DIMENSION],
        tag: i64,
    },
    /// `upsert` on the drawn Record ID.
    Upsert {
        ordinal: u64,
        vector: [f32; DIMENSION],
        tag: i64,
    },
    /// `delete` on the drawn Record ID.
    Delete { ordinal: u64 },
}

impl DrawnMutation {
    /// The touched Record ID.
    fn id(&self) -> Bytes {
        record_id(match self {
            DrawnMutation::Insert { ordinal, .. }
            | DrawnMutation::Upsert { ordinal, .. }
            | DrawnMutation::Delete { ordinal } => *ordinal,
        })
    }

    /// The public-API form of this item.
    fn to_api(&self) -> Mutation {
        match self {
            DrawnMutation::Insert { vector, tag, .. } => {
                Mutation::Insert(record(&self.id(), vector, *tag))
            }
            DrawnMutation::Upsert { vector, tag, .. } => {
                Mutation::Upsert(record(&self.id(), vector, *tag))
            }
            DrawnMutation::Delete { .. } => Mutation::Delete(self.id()),
        }
    }

    /// Applies this item to a model with engine semantics: an insert wins
    /// only when absent, an upsert always writes, a delete removes.
    fn apply(&self, model: &mut Model) {
        match self {
            DrawnMutation::Insert { vector, tag, .. } => {
                model
                    .entry(self.id())
                    .or_insert_with(|| model_record(vector, *tag));
            }
            DrawnMutation::Upsert { vector, tag, .. } => {
                model.insert(self.id(), model_record(vector, *tag));
            }
            DrawnMutation::Delete { .. } => {
                model.remove(&self.id());
            }
        }
    }
}

/// One pre-generated scripted step.
#[derive(Debug)]
enum Step {
    /// One insert/upsert/delete, possibly armed with a commit fault.
    Mutate {
        mutation: DrawnMutation,
        fault: Option<CommitFault>,
    },
    /// One atomic batch of 2-5 mixed mutations, possibly armed.
    Batch {
        mutations: Vec<DrawnMutation>,
        fault: Option<CommitFault>,
    },
    /// One point read, checked against the model (read-your-writes).
    Get { ordinal: u64 },
    /// One search, occasionally with an `eq` Filter Predicate on `tag`.
    Search {
        vector: [f32; DIMENSION],
        k: usize,
        tag_filter: Option<i64>,
    },
    /// One bounded `split::advance`/`merge::advance` on the first candidate,
    /// possibly armed with a commit fault.
    Maintenance { fault: Option<CommitFault> },
    /// Process loss: the handles drop without shutdown, losing the
    /// process-local Fixup queue; the durable backend reopens.
    Crash,
    /// `shutdown`, then reopen (durable committed state carries forward).
    CleanRestart,
    /// One mutation with a pre-cancelled token: `Cancelled`, nothing applied.
    CancelledOp { mutation: DrawnMutation },
    /// One mutation with an expired deadline: `DeadlineExceeded`, nothing
    /// applied.
    DeadlineOp { mutation: DrawnMutation },
    /// `drop_index` then `create_index` under the same Index Name: a fresh
    /// Logical Index ID, an empty model, and a reset Partition Key history.
    DropRecreate,
}

/// Draws the complete script from one seed before any async work runs, so the
/// base seed reproduces the whole history.
fn generate_script(seed: u64, steps: usize) -> Vec<Step> {
    let mut rng = Rng(seed);
    let mut script = Vec::with_capacity(steps);
    let mut drop_recreates = 0_usize;
    while script.len() < steps {
        // Mutation kind weights alternate between growth and shrink phases,
        // so the single tree repeatedly crosses the split threshold in
        // growth phases and the merge threshold in shrink phases instead of
        // settling at one steady topology.
        let growth = script.len() / PHASE_STEPS % 2 == 0;
        let step = match rng.below(100) {
            0..=42 => Step::Mutate {
                mutation: draw_mutation(&mut rng, growth),
                fault: draw_fault(&mut rng),
            },
            43..=50 => Step::Batch {
                mutations: draw_batch(&mut rng, growth),
                fault: draw_fault(&mut rng),
            },
            51..=55 => Step::Get {
                ordinal: rng.below(ID_POOL),
            },
            56..=66 => Step::Search {
                vector: draw_vector(&mut rng),
                k: 1 + usize::try_from(rng.below(10)).expect("small k"),
                tag_filter: draw_tag_filter(&mut rng),
            },
            67..=86 => Step::Maintenance {
                fault: draw_fault(&mut rng),
            },
            87..=90 => Step::Crash,
            91..=93 => Step::CleanRestart,
            94..=95 => Step::CancelledOp {
                mutation: draw_mutation(&mut rng, growth),
            },
            96..=97 => Step::DeadlineOp {
                mutation: draw_mutation(&mut rng, growth),
            },
            _ => {
                if drop_recreates >= MAX_DROP_RECREATES {
                    Step::Get {
                        ordinal: rng.below(ID_POOL),
                    }
                } else {
                    drop_recreates += 1;
                    Step::DropRecreate
                }
            }
        };
        script.push(step);
    }
    script
}

/// Draws one insert/upsert/delete over the Record ID pool. Growth phases draw
/// 45% insert / 35% upsert / 20% delete; shrink phases draw 20% insert / 20%
/// upsert / 60% delete.
fn draw_mutation(rng: &mut Rng, growth: bool) -> DrawnMutation {
    let ordinal = rng.below(ID_POOL);
    draw_mutation_on(rng, ordinal, growth)
}

/// Draws the mutation kind for one fixed Record ID.
fn draw_mutation_on(rng: &mut Rng, ordinal: u64, growth: bool) -> DrawnMutation {
    let kind = if growth {
        match rng.below(10) {
            0..=4 => 0,
            5..=7 => 1,
            _ => 2,
        }
    } else {
        match rng.below(10) {
            0..=1 => 0,
            2..=3 => 1,
            _ => 2,
        }
    };
    match kind {
        0 => DrawnMutation::Insert {
            ordinal,
            vector: draw_vector(rng),
            tag: draw_tag(rng),
        },
        1 => DrawnMutation::Upsert {
            ordinal,
            vector: draw_vector(rng),
            tag: draw_tag(rng),
        },
        _ => DrawnMutation::Delete { ordinal },
    }
}

/// Draws one atomic batch of 2-5 items on distinct Record IDs (the public
/// contract rejects duplicate IDs inside one batch).
fn draw_batch(rng: &mut Rng, growth: bool) -> Vec<DrawnMutation> {
    let size = 2 + usize::try_from(rng.below(4)).expect("small batch");
    let mut ordinals = BTreeSet::new();
    while ordinals.len() < size {
        ordinals.insert(rng.below(ID_POOL));
    }
    ordinals
        .into_iter()
        .map(|ordinal| draw_mutation_on(rng, ordinal, growth))
        .collect()
}

/// Draws an optional commit fault: 15% of faultable steps arm one.
fn draw_fault(rng: &mut Rng) -> Option<CommitFault> {
    if rng.below(100) >= FAULT_PERCENT {
        return None;
    }
    Some(match rng.below(3) {
        0 => CommitFault::Abort,
        1 => CommitFault::UnknownApplied,
        _ => CommitFault::UnknownNotApplied,
    })
}

/// Draws an optional `eq` Filter Predicate target on the `tag` field.
fn draw_tag_filter(rng: &mut Rng) -> Option<i64> {
    if rng.below(100) < FILTER_PERCENT {
        Some(draw_tag(rng))
    } else {
        None
    }
}

/// One `tag` value in its small domain.
fn draw_tag(rng: &mut Rng) -> i64 {
    rng.below(TAG_DOMAIN) as i64
}

/// One deterministic finite vector, components in `[0, 1000)`.
fn draw_vector(rng: &mut Rng) -> [f32; DIMENSION] {
    let mut vector = [0.0_f32; DIMENSION];
    for component in &mut vector {
        *component = rng.below(1_000_000) as f32 / 1_000.0;
    }
    vector
}

/// The pool Record ID `r{ordinal}`.
fn record_id(ordinal: u64) -> Bytes {
    Bytes::from(format!("r{ordinal}"))
}

/// One scripted record in the single Tree Key bucket.
fn record(id: &Bytes, vector: &[f32; DIMENSION], tag: i64) -> Record {
    Record::new(
        id.clone(),
        Arc::from(&vector[..]),
        vec![Value::I64(TREE_BUCKET), Value::I64(tag)],
    )
    .expect("valid record")
}

/// The model mirror of one scripted record.
fn model_record(vector: &[f32; DIMENSION], tag: i64) -> ModelRecord {
    ModelRecord {
        vector: Arc::from(&vector[..]),
        fields: vec![Value::I64(TREE_BUCKET), Value::I64(tag)].into_boxed_slice(),
    }
}

/// A 4-dimensional L2 index with one i64 Tree Key field, one i64 filter
/// field, and small partition entry bounds so splits (and, after deletes,
/// merges) fire regularly.
fn index_config() -> IndexConfig {
    IndexConfig::new(DIMENSION, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
            FieldSchema::new("tag", DataType::I64).expect("valid field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(MIN_PARTITION_ENTRIES, MAX_PARTITION_ENTRIES)
        .expect("valid partition entries")
}

/// One drivable unit of Structure Maintenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Work {
    /// `split::advance`: an in-flight split or an over-full Ready partition.
    Split,
    /// `merge::advance`: an in-flight merge or an under-full Ready non-root.
    Merge,
}

/// Orders one partition listing into drivable candidates: in-flight machines
/// first (smallest Tree Key and Partition Key), then over-full Ready
/// partitions (most entries, largest key on ties), then under-full Ready
/// non-roots (fewest entries, smallest key on ties) — the same candidate
/// ordering as the seeded-interleaving driver in `concurrency.rs`, so the
/// maintenance step drives the single first candidate at drive time.
fn collect_candidates(
    listing: &[(TreeKey, PartitionKey, PartitionHeader)],
    config: &IndexConfig,
) -> Vec<(TreeKey, PartitionKey, Work)> {
    let mut candidates = Vec::new();

    let mut in_flight_splits: Vec<_> = listing
        .iter()
        .filter(|(_, _, header)| {
            matches!(
                header.state(),
                PartitionState::Splitting | PartitionState::DrainingSplit
            )
        })
        .collect();
    in_flight_splits.sort_by_key(|(tree_key, partition, _)| (tree_key.clone(), *partition));
    candidates.extend(
        in_flight_splits
            .into_iter()
            .map(|(tree_key, partition, _)| (tree_key.clone(), *partition, Work::Split)),
    );

    let mut in_flight_merges: Vec<_> = listing
        .iter()
        .filter(|(_, _, header)| header.state() == PartitionState::Merging)
        .collect();
    in_flight_merges.sort_by_key(|(tree_key, partition, _)| (tree_key.clone(), *partition));
    candidates.extend(
        in_flight_merges
            .into_iter()
            .map(|(tree_key, partition, _)| (tree_key.clone(), *partition, Work::Merge)),
    );

    let maximum = config.max_partition_entries();
    let mut over_full: Vec<_> = listing
        .iter()
        .filter(|(_, _, header)| {
            header.state() == PartitionState::Ready && header.entry_count() > maximum
        })
        .collect();
    over_full.sort_by(|left, right| {
        right
            .2
            .entry_count()
            .cmp(&left.2.entry_count())
            .then_with(|| right.1.cmp(&left.1))
    });
    candidates.extend(
        over_full
            .into_iter()
            .map(|(tree_key, partition, _)| (tree_key.clone(), *partition, Work::Split)),
    );

    let minimum = config.min_partition_entries();
    let mut under_full: Vec<_> = listing
        .iter()
        .filter(|(_, partition, header)| {
            partition.get() != 1
                && header.state() == PartitionState::Ready
                && header.entry_count() < minimum
        })
        .collect();
    under_full.sort_by(|left, right| {
        left.2
            .entry_count()
            .cmp(&right.2.entry_count())
            .then_with(|| left.1.cmp(&right.1))
    });
    candidates.extend(
        under_full
            .into_iter()
            .map(|(tree_key, partition, _)| (tree_key.clone(), *partition, Work::Merge)),
    );

    candidates
}

/// A bounded ring of step traces printed with the replay command on panic.
struct TraceLog {
    seed: u64,
    steps: usize,
    lines: VecDeque<String>,
}

impl TraceLog {
    fn new(seed: u64, steps: usize) -> Self {
        Self {
            seed,
            steps,
            lines: VecDeque::with_capacity(TRACE_CAPACITY),
        }
    }

    fn push(&mut self, line: String) {
        if self.lines.len() >= TRACE_CAPACITY {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }
}

impl Drop for TraceLog {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "model history failed: seed={} steps={}",
                self.seed, self.steps
            );
            eprintln!(
                "replay: KTANN_MODEL_SEED={} KTANN_MODEL_STEPS={} cargo test --test model_history model_history_replay -- --nocapture",
                self.seed, self.steps
            );
            eprintln!("last {} step traces (oldest first):", self.lines.len());
            for line in &self.lines {
                eprintln!("{line}");
            }
        }
    }
}

/// The Partition Key non-reuse tracker for the current Logical Index ID: a
/// key that disappears from a later listing (split source consumed, merge
/// source removed) is retired and must never reappear.
#[derive(Default)]
struct PartitionTracker {
    ever_seen: BTreeSet<(TreeKey, PartitionKey)>,
    retired: BTreeSet<(TreeKey, PartitionKey)>,
}

impl PartitionTracker {
    async fn check(&mut self, backend: &SharedBackend, index: LogicalIndexId, step: usize) {
        let listing = audit::list_partitions(backend, index)
            .await
            .expect("partition listing");
        let current: BTreeSet<(TreeKey, PartitionKey)> = listing
            .into_iter()
            .map(|(tree_key, partition, _)| (tree_key, partition))
            .collect();
        for reference in &current {
            assert!(
                !self.retired.contains(reference),
                "step {step}: Partition Key {} reappeared after retirement",
                reference.1.get()
            );
        }
        let gone: Vec<_> = self.ever_seen.difference(&current).cloned().collect();
        self.retired.extend(gone);
        self.ever_seen.extend(current);
    }

    fn reset(&mut self) {
        self.ever_seen.clear();
        self.retired.clear();
    }
}

/// Counters describing one case, printed at completion.
#[derive(Debug, Default)]
struct CaseStats {
    mutations: u64,
    batches: u64,
    gets: u64,
    searches: u64,
    advances: u64,
    splits: u64,
    merges: u64,
    crashes: u64,
    restarts: u64,
    drops: u64,
    control_rejected: u64,
    aborted: u64,
    unknown_recovered: u64,
}

/// The sequential driver: current Runtime and Index handles, the exact model,
/// identity histories, the logical maintenance clock, and the trace ring.
struct Driver {
    backend: SharedBackend,
    config: RuntimeConfig,
    retry: RetryPolicy,
    runtime: Option<Runtime<SharedBackend>>,
    index: Option<Index<SharedBackend>>,
    model: Model,
    current_id: LogicalIndexId,
    seen_index_ids: BTreeSet<LogicalIndexId>,
    tracker: PartitionTracker,
    clock: u64,
    stats: CaseStats,
    trace: TraceLog,
}

impl Driver {
    /// Opens a fresh case on a durable deterministic backend with zero
    /// maintenance workers, so topology moves only under manual drives.
    async fn new(seed: u64, steps: usize) -> Self {
        let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig {
            durability: Durability::Durable,
            ..DeterministicConfig::default()
        }));
        let config = support::manual_maintenance_config()
            .with_attempts(FIXUP_ATTEMPTS, FOREGROUND_ATTEMPTS)
            .expect("valid attempt bounds");
        let retry = RetryPolicy::for_fixup(&config);
        let runtime = Runtime::new(backend.clone(), config.clone()).expect("runtime is valid");
        let index = runtime
            .create_index(INDEX_NAME, index_config())
            .await
            .expect("create index");
        let current_id = index.logical_index_id();
        let mut seen_index_ids = BTreeSet::new();
        seen_index_ids.insert(current_id);
        Self {
            backend,
            config,
            retry,
            runtime: Some(runtime),
            index: Some(index),
            model: Model::new(),
            current_id,
            seen_index_ids,
            tracker: PartitionTracker::default(),
            clock: 1_000,
            stats: CaseStats::default(),
            trace: TraceLog::new(seed, steps),
        }
    }

    fn index(&self) -> &Index<SharedBackend> {
        self.index.as_ref().expect("live index handle")
    }

    fn runtime(&self) -> &Runtime<SharedBackend> {
        self.runtime.as_ref().expect("live runtime")
    }

    /// Runs one scripted step, then the after-every-step invariant checks.
    async fn step(&mut self, n: usize, step: &Step) {
        match step {
            Step::Mutate { mutation, fault } => self.step_mutate(n, mutation, *fault).await,
            Step::Batch { mutations, fault } => self.step_batch(n, mutations, *fault).await,
            Step::Get { ordinal } => {
                self.stats.gets += 1;
                let id = record_id(*ordinal);
                self.check_get(n, &id).await;
                self.trace.push(format!(
                    "step {n}: get {} -> agrees with the model",
                    show(&id)
                ));
            }
            Step::Search {
                vector,
                k,
                tag_filter,
            } => self.step_search(n, vector, *k, *tag_filter).await,
            Step::Maintenance { fault } => self.step_maintenance(n, *fault).await,
            Step::Crash => self.step_crash(n).await,
            Step::CleanRestart => self.step_clean_restart(n).await,
            Step::CancelledOp { mutation } => {
                self.step_control_op(n, mutation, Control::Cancel).await;
            }
            Step::DeadlineOp { mutation } => {
                self.step_control_op(n, mutation, Control::Deadline).await;
            }
            Step::DropRecreate => self.step_drop_recreate(n).await,
        }
    }

    /// The after-every-step invariants: exact membership against the model
    /// and Partition Key non-reuse from a fresh partition listing.
    async fn post_step(&mut self, n: usize) {
        let report = audit::run(&self.backend, self.current_id, &self.model)
            .await
            .unwrap_or_else(|failure| panic!("step {n}: exact-membership audit failed: {failure}"));
        assert_eq!(
            report.records,
            self.model.len(),
            "step {n}: audit record count disagrees with the model"
        );
        assert!(
            report.trees <= 1,
            "step {n}: every record carries bucket={TREE_BUCKET}"
        );
        self.tracker.check(&self.backend, self.current_id, n).await;
    }

    /// Runs one single mutation, agreeing with the model on definite
    /// outcomes, then applying it to the model.
    async fn step_mutate(
        &mut self,
        n: usize,
        mutation: &DrawnMutation,
        fault: Option<CommitFault>,
    ) {
        self.stats.mutations += 1;
        if let Some(fault) = fault {
            self.backend.inner().push_fault(fault).expect("push fault");
        }
        let id = mutation.id();
        let had = self.model.contains_key(&id);
        match mutation {
            DrawnMutation::Insert { vector, tag, .. } => {
                match self.index().insert(record(&id, vector, *tag)).await {
                    Ok(()) => {
                        assert!(
                            !had,
                            "step {n}: insert committed an already-held Record ID {}",
                            show(&id)
                        );
                        mutation.apply(&mut self.model);
                        self.trace
                            .push(format!("step {n}: insert {} -> ok", show(&id)));
                    }
                    Err(error) if error.kind() == ErrorKind::RecordAlreadyExists => {
                        assert!(
                            had,
                            "step {n}: RecordAlreadyExists for an unheld Record ID {}",
                            show(&id)
                        );
                        self.trace.push(format!(
                            "step {n}: insert {} -> RecordAlreadyExists",
                            show(&id)
                        ));
                    }
                    Err(error) => {
                        self.mutation_error(n, std::slice::from_ref(mutation), error)
                            .await;
                    }
                }
            }
            DrawnMutation::Upsert { vector, tag, .. } => {
                match self.index().upsert(record(&id, vector, *tag)).await {
                    Ok(result) => {
                        assert_eq!(
                            result == UpsertResult::Replaced,
                            had,
                            "step {n}: upsert result disagrees with the model for {}",
                            show(&id)
                        );
                        mutation.apply(&mut self.model);
                        self.trace
                            .push(format!("step {n}: upsert {} -> ok", show(&id)));
                    }
                    Err(error) => {
                        self.mutation_error(n, std::slice::from_ref(mutation), error)
                            .await;
                    }
                }
            }
            DrawnMutation::Delete { .. } => match self.index().delete(id.clone()).await {
                Ok(existed) => {
                    assert_eq!(
                        existed,
                        had,
                        "step {n}: delete outcome disagrees with the model for {}",
                        show(&id)
                    );
                    mutation.apply(&mut self.model);
                    self.trace
                        .push(format!("step {n}: delete {} -> ok", show(&id)));
                }
                Err(error) => {
                    self.mutation_error(n, std::slice::from_ref(mutation), error)
                        .await;
                }
            },
        }
        // An operation that fails before its first commit leaves its armed
        // fault queued; clear leftovers so later steps arm cleanly.
        if fault.is_some() {
            self.backend
                .inner()
                .set_fault_plan(Vec::new())
                .expect("clear fault plan");
        }
    }

    /// Runs one atomic batch: per-position outcomes must agree with the
    /// model, then every item applies to the model.
    async fn step_batch(
        &mut self,
        n: usize,
        mutations: &[DrawnMutation],
        fault: Option<CommitFault>,
    ) {
        self.stats.batches += 1;
        if let Some(fault) = fault {
            self.backend.inner().push_fault(fault).expect("push fault");
        }
        let api: Vec<Mutation> = mutations.iter().map(DrawnMutation::to_api).collect();
        match self.index().batch_mutate(api).await {
            Ok(outcomes) => {
                assert_eq!(
                    outcomes.len(),
                    mutations.len(),
                    "step {n}: batch outcome count disagrees with the input"
                );
                for (mutation, outcome) in mutations.iter().zip(&outcomes) {
                    let had = self.model.contains_key(&mutation.id());
                    match (mutation, outcome) {
                        (DrawnMutation::Insert { .. }, MutationOutcome::Inserted) => assert!(
                            !had,
                            "step {n}: batch insert committed an already-held Record ID {}",
                            show(&mutation.id())
                        ),
                        (DrawnMutation::Upsert { .. }, MutationOutcome::Upserted { replaced }) => {
                            assert_eq!(
                                *replaced,
                                had,
                                "step {n}: batch upsert outcome disagrees with the model for {}",
                                show(&mutation.id())
                            );
                        }
                        (DrawnMutation::Delete { .. }, MutationOutcome::Deleted { existed }) => {
                            assert_eq!(
                                *existed,
                                had,
                                "step {n}: batch delete outcome disagrees with the model for {}",
                                show(&mutation.id())
                            );
                        }
                        (mutation, outcome) => panic!(
                            "step {n}: batch outcome {outcome:?} mismatches {} {}",
                            describe_one(mutation),
                            show(&mutation.id())
                        ),
                    }
                }
                for mutation in mutations {
                    mutation.apply(&mut self.model);
                }
                self.trace
                    .push(format!("step {n}: batch ({} items) -> ok", mutations.len()));
            }
            Err(error) if error.kind() == ErrorKind::RecordAlreadyExists => {
                // One conflicting insert item fails the whole batch before
                // commit: nothing applied.
                assert!(
                    mutations
                        .iter()
                        .any(|mutation| matches!(mutation, DrawnMutation::Insert { .. })
                            && self.model.contains_key(&mutation.id())),
                    "step {n}: RecordAlreadyExists without a conflicting batch insert"
                );
                for mutation in mutations {
                    self.check_get(n, &mutation.id()).await;
                }
                self.trace.push(format!(
                    "step {n}: batch ({} items) -> RecordAlreadyExists, nothing applied",
                    mutations.len()
                ));
            }
            Err(error) => self.mutation_error(n, mutations, error).await,
        }
        if fault.is_some() {
            self.backend
                .inner()
                .set_fault_plan(Vec::new())
                .expect("clear fault plan");
        }
    }

    /// Handles the shared mutation error cases: a definite abort or
    /// contention exhaustion applied nothing (verified with reads), an
    /// unknown commit outcome is recovered by read-back, and anything else
    /// fails the case.
    async fn mutation_error(&mut self, n: usize, mutations: &[DrawnMutation], error: Error) {
        match error.kind() {
            ErrorKind::RetryableAbort | ErrorKind::ContentionExhausted => {
                for mutation in mutations {
                    self.check_get(n, &mutation.id()).await;
                }
                self.stats.aborted += 1;
                self.trace.push(format!(
                    "step {n}: {} -> {:?}, nothing applied",
                    describe(mutations),
                    error.kind()
                ));
            }
            ErrorKind::CommitOutcomeUnknown => self.recover_atomic(n, mutations).await,
            _ => panic!("step {n}: mutation failed: {error:?}"),
        }
    }

    /// Re-synchronizes the model after an unknown commit outcome (ADR 0012):
    /// reads back every touched Record ID, requiring the stored state to be
    /// exactly either the pre-operation model value or the intended new
    /// value — and, because one atomic commit cannot partially apply, the
    /// same side for every touched ID. The uncertain operation is never
    /// retried.
    async fn recover_atomic(&mut self, n: usize, mutations: &[DrawnMutation]) {
        let mut pre = Vec::with_capacity(mutations.len());
        let mut post = Vec::with_capacity(mutations.len());
        let mut intended = self.model.clone();
        for mutation in mutations {
            pre.push(self.model.get(&mutation.id()).cloned());
            mutation.apply(&mut intended);
            post.push(intended.get(&mutation.id()).cloned());
        }
        let mut recovered = Vec::with_capacity(mutations.len());
        for mutation in mutations {
            recovered.push(self.recover_one(n, &mutation.id()).await);
        }
        let all_pre = pre
            .iter()
            .zip(&recovered)
            .all(|(left, right)| records_match(left.as_ref(), right.as_ref()));
        let all_post = post
            .iter()
            .zip(&recovered)
            .all(|(left, right)| records_match(left.as_ref(), right.as_ref()));
        assert!(
            all_pre || all_post,
            "step {n}: unknown commit outcome recovered a partial application of {}",
            describe(mutations)
        );
        for (mutation, stored) in mutations.iter().zip(recovered) {
            match stored {
                Some(record) => self.model.insert(mutation.id(), record),
                None => self.model.remove(&mutation.id()),
            };
        }
        self.stats.unknown_recovered += 1;
        self.trace.push(format!(
            "step {n}: {} -> CommitOutcomeUnknown, recovered {}",
            describe(mutations),
            if all_post { "applied" } else { "not applied" }
        ));
    }

    /// Reads back one Record ID as its model mirror after an unknown outcome.
    async fn recover_one(&self, n: usize, id: &Bytes) -> Option<ModelRecord> {
        self.index()
            .get(id.clone(), GetOptions::default())
            .await
            .unwrap_or_else(|error| panic!("step {n}: recovery read failed: {error:?}"))
            .map(|stored| ModelRecord {
                vector: Arc::from(stored.vector()),
                fields: stored.fields().to_vec().into_boxed_slice(),
            })
    }

    /// One bounded search: every hit must be a modeled record at the exact
    /// oracle distance, hits must be in (distance, Record ID) order, and when
    /// no Search Budget dimension reports exhaustion the ordered hit list
    /// must equal the brute-force oracle truth exactly.
    async fn step_search(
        &mut self,
        n: usize,
        vector: &[f32; DIMENSION],
        k: usize,
        tag_filter: Option<i64>,
    ) {
        self.stats.searches += 1;
        let mut request = SearchRequest::new(Arc::from(&vector[..]), k).expect("valid search");
        if let Some(tag) = tag_filter {
            request = request.with_predicate(Predicate::Compare {
                field: FieldId(TAG_FIELD),
                op: CompareOp::Eq,
                value: Value::I64(tag),
            });
        }
        let outcome = self
            .index()
            .search(request)
            .await
            .unwrap_or_else(|error| panic!("step {n}: search failed: {error:?}"));
        assert!(
            outcome.hits.len() <= k,
            "step {n}: {} hits for k={k}",
            outcome.hits.len()
        );
        assert!(
            outcome.hits.windows(2).all(|pair| {
                pair[0].distance() < pair[1].distance()
                    || (pair[0].distance() == pair[1].distance() && pair[0].id() <= pair[1].id())
            }),
            "step {n}: hits are not in (distance, Record ID) order"
        );
        for hit in &outcome.hits {
            let modeled = self.model.get(hit.id()).unwrap_or_else(|| {
                panic!(
                    "step {n}: search hit {} is not in the model",
                    show(hit.id())
                )
            });
            let exact = oracle::exact_distance(Metric::L2, vector, &modeled.vector);
            assert!(
                (hit.distance() - exact).abs() <= DISTANCE_TOLERANCE,
                "step {n}: hit distance {} != oracle {exact} for {}",
                hit.distance(),
                show(hit.id())
            );
        }
        let exhausted = outcome.exhausted.scanned_tree_keys
            || outcome.exhausted.visited_partitions
            || outcome.exhausted.visited_leaf_entries
            || outcome.exhausted.exact_rerank_candidates
            || outcome.rabitq_overlap_truncated;
        if exhausted {
            self.trace.push(format!(
                "step {n}: search k={k} -> {} hits (budget exhausted, membership checked)",
                outcome.hits.len()
            ));
            return;
        }
        let truth = oracle::truth(
            &self.model,
            Metric::L2,
            vector,
            k,
            &|record: &ModelRecord| {
                tag_filter.is_none_or(|tag| {
                    oracle::compare_3vl(
                        CompareOp::Eq,
                        &record.fields[usize::from(TAG_FIELD)],
                        &Value::I64(tag),
                    )
                })
            },
        );
        assert_eq!(
            outcome.hits.len(),
            truth.len(),
            "step {n}: unexhausted search must return the exact oracle truth"
        );
        for (hit, (id, distance)) in outcome.hits.iter().zip(&truth) {
            assert_eq!(
                hit.id(),
                id,
                "step {n}: hit order disagrees with the oracle truth"
            );
            assert!(
                (hit.distance() - distance).abs() <= DISTANCE_TOLERANCE,
                "step {n}: hit distance disagrees with the oracle truth"
            );
        }
        self.trace.push(format!(
            "step {n}: search k={k} -> {} hits (exact)",
            outcome.hits.len()
        ));
    }

    /// Drives one bounded split/merge transition on the first candidate in
    /// the `concurrency.rs` ordering. `Idle`/`Stalled` are legal no-progress;
    /// fault-armed advances may surface abort/unknown errors, which are legal
    /// because transitions are idempotent and later steps rediscover them.
    async fn step_maintenance(&mut self, n: usize, fault: Option<CommitFault>) {
        let manifest = support::read_manifest(&self.backend, self.current_id).await;
        let listing = audit::list_partitions(&self.backend, self.current_id)
            .await
            .expect("partition listing");
        let Some((tree_key, partition, work)) = collect_candidates(&listing, manifest.config())
            .into_iter()
            .next()
        else {
            self.trace
                .push(format!("step {n}: maintenance -> no candidate"));
            return;
        };
        self.stats.advances += 1;
        self.clock += 100;
        if let Some(fault) = fault {
            self.backend.inner().push_fault(fault).expect("push fault");
        }
        match work {
            Work::Split => {
                match split::advance(
                    &self.backend,
                    &manifest,
                    &tree_key,
                    partition,
                    self.clock,
                    &self.retry,
                )
                .await
                {
                    Ok(advance) => {
                        if advance == SplitAdvance::Completed {
                            self.stats.splits += 1;
                        }
                        self.trace.push(format!(
                            "step {n}: maintenance split pk={} -> {advance:?}",
                            partition.get()
                        ));
                    }
                    Err(error) => self.maintenance_error(n, "split", partition, error),
                }
            }
            Work::Merge => {
                match merge::advance(
                    &self.backend,
                    &manifest,
                    &tree_key,
                    partition,
                    self.clock,
                    &self.retry,
                )
                .await
                {
                    Ok(advance) => {
                        if advance == MergeAdvance::Completed {
                            self.stats.merges += 1;
                        }
                        self.trace.push(format!(
                            "step {n}: maintenance merge pk={} -> {advance:?}",
                            partition.get()
                        ));
                    }
                    Err(error) => self.maintenance_error(n, "merge", partition, error),
                }
            }
        }
        if fault.is_some() {
            self.backend
                .inner()
                .set_fault_plan(Vec::new())
                .expect("clear fault plan");
        }
    }

    /// Legal maintenance no-progress errors; anything else fails the case.
    fn maintenance_error(&mut self, n: usize, kind: &str, partition: PartitionKey, error: Error) {
        if matches!(
            error.kind(),
            ErrorKind::RetryableAbort
                | ErrorKind::CommitOutcomeUnknown
                | ErrorKind::ContentionExhausted
        ) {
            self.trace.push(format!(
                "step {n}: maintenance {kind} pk={} -> {:?} (legal no-progress)",
                partition.get(),
                error.kind()
            ));
        } else {
            panic!("step {n}: maintenance {kind} advance failed: {error:?}");
        }
    }

    /// Process loss: the Index and Runtime handles drop without shutdown, so
    /// the process-local Fixup queue is lost. The stale Index handle must
    /// fail closed, never silently succeed; the durable committed state
    /// reopens under the same Logical Index ID.
    async fn step_crash(&mut self, n: usize) {
        self.stats.crashes += 1;
        let stale = self.index().clone();
        drop(self.index.take());
        drop(self.runtime.take());
        let error = stale
            .get(record_id(0), GetOptions::default())
            .await
            .expect_err("step {n}: the stale Index handle must fail after a crash");
        assert_eq!(
            error.kind(),
            ErrorKind::RuntimeClosed,
            "step {n}: the stale Index handle must fail closed, not silently succeed"
        );
        drop(stale);
        self.reopen(n).await;
        self.trace.push(format!(
            "step {n}: crash -> reopened Logical Index ID {}",
            self.current_id.get()
        ));
    }

    /// Clean shutdown, then reopen: an operation on the shut-down Runtime
    /// must fail `RuntimeClosed`, and the durable committed state reopens
    /// under the same Logical Index ID.
    async fn step_clean_restart(&mut self, n: usize) {
        self.stats.restarts += 1;
        self.runtime()
            .shutdown()
            .await
            .expect("step {n}: clean shutdown");
        let error = self
            .index()
            .get(record_id(0), GetOptions::default())
            .await
            .expect_err("step {n}: an operation on the shut-down Runtime must fail");
        assert_eq!(
            error.kind(),
            ErrorKind::RuntimeClosed,
            "step {n}: the shut-down Runtime must reject operations"
        );
        self.reopen(n).await;
        self.trace.push(format!(
            "step {n}: clean restart -> reopened Logical Index ID {}",
            self.current_id.get()
        ));
    }

    /// Reopens the durable backend and opens the index under a fresh Runtime.
    async fn reopen(&mut self, n: usize) {
        let reopened = self.backend.inner().reopen();
        self.backend = SharedBackend::new(reopened);
        let runtime =
            Runtime::new(self.backend.clone(), self.config.clone()).expect("runtime is valid");
        let index = runtime
            .open_index(INDEX_NAME)
            .await
            .expect("step {n}: reopen index after restart");
        assert_eq!(
            index.logical_index_id(),
            self.current_id,
            "step {n}: open_index must return the same Logical Index ID after restart"
        );
        self.runtime = Some(runtime);
        self.index = Some(index);
    }

    /// One mutation rejected before admission: pre-cancelled token or expired
    /// deadline. The model is unchanged, verified with a read.
    async fn step_control_op(&mut self, n: usize, mutation: &DrawnMutation, control: Control) {
        self.stats.control_rejected += 1;
        let (options, expected, label) = match control {
            Control::Cancel => {
                let cancellation = CancellationToken::new();
                cancellation.cancel();
                (
                    OperationOptions::default().with_cancellation(cancellation),
                    ErrorKind::Cancelled,
                    "pre-cancelled",
                )
            }
            Control::Deadline => (
                OperationOptions::default().with_deadline(Instant::now()),
                ErrorKind::DeadlineExceeded,
                "deadline-expired",
            ),
        };
        let result = match mutation {
            DrawnMutation::Insert { vector, tag, .. } => self
                .index()
                .insert_with_control(record(&mutation.id(), vector, *tag), options)
                .await
                .map(|_| ()),
            DrawnMutation::Upsert { vector, tag, .. } => self
                .index()
                .upsert_with_control(record(&mutation.id(), vector, *tag), options)
                .await
                .map(|_| ()),
            DrawnMutation::Delete { .. } => self
                .index()
                .delete_with_control(mutation.id(), options)
                .await
                .map(|_| ()),
        };
        let error = result.expect_err("step {n}: a {label} operation must not run");
        assert_eq!(
            error.kind(),
            expected,
            "step {n}: a {label} operation must fail {expected:?}"
        );
        self.check_get(n, &mutation.id()).await;
        self.trace.push(format!(
            "step {n}: {} {} -> {expected:?} before admission, model unchanged",
            describe_one(mutation),
            show(&mutation.id())
        ));
    }

    /// Drops the index and recreates it under the same Index Name and
    /// configuration: the stale handle must fail closed (never silently
    /// retarget), and the new Logical Index ID must differ from every
    /// previously seen identity. The model and the Partition Key history
    /// reset for the new identity.
    async fn step_drop_recreate(&mut self, n: usize) {
        self.stats.drops += 1;
        let stale = self.index().clone();
        self.runtime()
            .drop_index(INDEX_NAME)
            .await
            .expect("step {n}: drop index");
        let error = stale
            .get(record_id(0), GetOptions::default())
            .await
            .expect_err("step {n}: the dropped Index handle must reject operations");
        assert!(
            matches!(
                error.kind(),
                ErrorKind::IndexNotFound | ErrorKind::IndexDropping
            ),
            "step {n}: the dropped Index handle must fail closed, not silently retarget"
        );
        drop(stale);
        let index = self
            .runtime()
            .create_index(INDEX_NAME, index_config())
            .await
            .expect("step {n}: recreate index");
        let new_id = index.logical_index_id();
        assert!(
            self.seen_index_ids.insert(new_id),
            "step {n}: Logical Index ID {} was reused",
            new_id.get()
        );
        self.current_id = new_id;
        self.index = Some(index);
        self.model.clear();
        self.tracker.reset();
        self.trace.push(format!(
            "step {n}: drop/recreate -> new Logical Index ID {}",
            new_id.get()
        ));
    }

    /// Read-your-writes: the stored state of one Record ID must equal the
    /// model exactly (presence, vector, and fields).
    async fn check_get(&self, n: usize, id: &Bytes) {
        let stored = self
            .index()
            .get(id.clone(), GetOptions::default())
            .await
            .unwrap_or_else(|error| panic!("step {n}: get failed: {error:?}"));
        match (&stored, self.model.get(id)) {
            (Some(stored), Some(modeled)) => assert!(
                stored_matches(stored, modeled),
                "step {n}: stored Vector Record disagrees with the model for {}",
                show(id)
            ),
            (None, None) => {}
            (Some(_), None) => panic!("step {n}: {} is stored but not in the model", show(id)),
            (None, Some(_)) => panic!("step {n}: {} is in the model but not stored", show(id)),
        }
    }

    /// Shuts down the current Runtime and prints the case summary.
    async fn finish(mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown().await.expect("final shutdown");
        }
        eprintln!("model_history: complete: {:?}", self.stats);
    }
}

/// The kind of pre-admission rejection a control step exercises.
#[derive(Clone, Copy)]
enum Control {
    Cancel,
    Deadline,
}

/// Whether a stored Vector Record equals its model mirror exactly.
fn stored_matches(stored: &StoredRecord, modeled: &ModelRecord) -> bool {
    stored.vector() == &*modeled.vector && stored.fields() == &*modeled.fields
}

/// Whether two optional model records agree (None = absent).
fn records_match(left: Option<&ModelRecord>, right: Option<&ModelRecord>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => left.vector == right.vector && left.fields == right.fields,
        _ => false,
    }
}

/// The operation kind of one drawn mutation.
fn describe_one(mutation: &DrawnMutation) -> &'static str {
    match mutation {
        DrawnMutation::Insert { .. } => "insert",
        DrawnMutation::Upsert { .. } => "upsert",
        DrawnMutation::Delete { .. } => "delete",
    }
}

/// A short trace description of one mutation or one batch.
fn describe(mutations: &[DrawnMutation]) -> String {
    if mutations.len() == 1 {
        format!(
            "{} {}",
            describe_one(&mutations[0]),
            show(&mutations[0].id())
        )
    } else {
        format!("batch ({} items)", mutations.len())
    }
}

/// Record IDs are printable ASCII (`r{ordinal}`), safe for traces.
fn show(id: &Bytes) -> String {
    String::from_utf8_lossy(id).into_owned()
}

/// Runs one case: pre-generate the script from the seed, then drive it
/// sequentially, asserting the after-every-step invariants.
async fn run_case(seed: u64, steps: usize) {
    eprintln!("model_history: seed={seed} steps={steps}");
    let script = generate_script(seed, steps);
    let mut driver = Driver::new(seed, steps).await;
    for (n, step) in script.iter().enumerate() {
        driver.step(n, step).await;
        driver.post_step(n).await;
    }
    driver.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_history_seed_0() {
    run_case(CI_SEEDS[0], CI_STEPS).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_history_seed_1() {
    run_case(CI_SEEDS[1], CI_STEPS).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_history_seed_2() {
    run_case(CI_SEEDS[2], CI_STEPS).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_history_seed_3() {
    run_case(CI_SEEDS[3], CI_STEPS).await;
}

/// Replays one seed: `KTANN_MODEL_SEED=<seed>` (optionally with
/// `KTANN_MODEL_STEPS=<n>`, default 160). A no-op without the variable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_history_replay() {
    let Ok(raw_seed) = std::env::var("KTANN_MODEL_SEED") else {
        return;
    };
    let seed: u64 = raw_seed.parse().expect("KTANN_MODEL_SEED must be a u64");
    let steps = std::env::var("KTANN_MODEL_STEPS").map_or(CI_STEPS, |raw| {
        raw.parse().expect("KTANN_MODEL_STEPS must be a usize")
    });
    run_case(seed, steps).await;
}

/// The expanded deterministic profile: 24 seeds at 400 steps from a distinct
/// base seed. A no-op unless `KTANN_MODEL_PROFILE=expanded`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_history_expanded() {
    if std::env::var("KTANN_MODEL_PROFILE").as_deref() != Ok("expanded") {
        return;
    }
    for i in 0..EXPANDED_SEEDS {
        let seed = EXPANDED_BASE_SEED ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        run_case(seed, EXPANDED_STEPS).await;
    }
}
