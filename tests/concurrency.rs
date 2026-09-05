//! Seeded interleaving end-to-end coverage (issue #100, item D1), in the
//! spirit of CockroachDB vecindex's `TestIndexConcurrency`: several async
//! tasks run public-API mutations and searches against one shared Index while
//! a concurrent driver task advances the split/merge state machines, and the
//! exact-membership audit must pass over the final state.
//!
//! Replayability comes from pre-generated seeded operation scripts: each
//! mutation task draws its full script from `support::Rng` with a fixed
//! per-task seed derived from `BASE_SEED` before any task spawns, so the base
//! seed reproduces every operation. Tasks own disjoint Record ID ranges
//! (`t{TASK}-r{ORDINAL}`), so each operation's final effect depends only on
//! the owning task's earlier operations — the merged model, and therefore
//! exact membership, is interleaving-independent by construction. Topology
//! shape and split/merge timestamps legitimately vary with the schedule; the
//! audit checks membership and topology invariants, not golden bytes.
//!
//! Commit-fault injection is deliberately excluded: the deterministic
//! backend's fault queue is consumed in commit order, which is
//! scheduling-dependent under concurrency, and `CommitOutcomeUnknown` is
//! never retried by design — injecting it here would make the final model
//! unknowable. Fault-driven membership histories stay in `mutations.rs`.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, Index, IndexConfig, LogicalIndexId, Metric,
    PartitionKey, Record, RuntimeConfig, SearchRequest, Value,
};
use ktann::maintenance::merge::{self, Advance as MergeAdvance};
use ktann::maintenance::split::{self, Advance as SplitAdvance};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::keys::TreeKey;
use ktann::storage::values::{PartitionHeader, PartitionState};

#[allow(dead_code)]
mod support;

use support::oracle::{Model, ModelRecord};
use support::{DeterministicBackend, DeterministicConfig, Rng, SharedBackend, audit};

/// The base seed every per-task script derives from.
const BASE_SEED: u64 = 0x5EED_D100_0000_0001;
/// Mutation tasks; each owns `IDS_PER_TASK` disjoint Record IDs.
const TASKS: usize = 4;
/// Record IDs per task. Disjoint ranges make every operation's final effect
/// interleaving-independent.
const IDS_PER_TASK: u64 = 64;
/// Pre-generated operations per task script.
const OPS_PER_TASK: usize = 100;
/// The vector dimension.
const DIMENSION: usize = 4;
/// The single Tree Key bucket: one tree concentrates split/merge churn and
/// mutation contention in this test; cross-tree topology races are covered by
/// `tree_manifest.rs`.
const TREE_BUCKET: i64 = 1;
/// Partition entry bounds: splits fire every 17th insert into one leaf and
/// delete runs pull leaves under the merge threshold.
const MIN_PARTITION_ENTRIES: u32 = 8;
const MAX_PARTITION_ENTRIES: u32 = 16;
/// Whole-attempt bounds (default 8) raised so unlucky commit-order timing
/// does not exhaust retries under sustained contention.
const ATTEMPTS: u32 = 64;
/// Backstop on total maintenance advances; reaching it means the state
/// machines failed to settle.
const MAX_ADVANCES: u64 = 10_000;

/// One pre-generated scripted operation.
#[derive(Debug)]
enum Op {
    /// `insert` on the task's `ordinal`-th Record ID.
    Insert {
        ordinal: u64,
        vector: [f32; DIMENSION],
    },
    /// `upsert` on the task's `ordinal`-th Record ID.
    Upsert {
        ordinal: u64,
        vector: [f32; DIMENSION],
    },
    /// `delete` on the task's `ordinal`-th Record ID.
    Delete { ordinal: u64 },
    /// `search` with a drawn query vector and `k`.
    Search { vector: [f32; DIMENSION], k: usize },
}

/// Counters describing the maintenance driver's work.
#[derive(Debug, Default)]
struct DriverStats {
    /// Bounded advances attempted.
    advances: u64,
    /// Completed splits.
    splits: u64,
    /// Completed merges.
    merges: u64,
    /// Advances that declined (`Idle`/`Stalled`) or exhausted retries.
    retries: u64,
}

/// One drivable unit of Structure Maintenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Work {
    /// `split::advance`: an in-flight split or an over-full Ready partition.
    Split,
    /// `merge::advance`: an in-flight merge or an under-full Ready non-root.
    Merge,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seeded_interleaving_converges_with_exact_membership() {
    let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()));
    // No background workers: topology moves only when the driver task below
    // drives the public state machines, one bounded transition at a time.
    let config = support::manual_maintenance_config()
        .with_attempts(ATTEMPTS, ATTEMPTS)
        .expect("valid attempt bounds");
    let runtime = Runtime::new(backend.clone(), config.clone()).expect("runtime is valid");
    let index = runtime
        .create_index("concurrency", index_config())
        .await
        .expect("create index");
    let logical_index_id = index.logical_index_id();

    let tasks = spawn_scripts(&index);

    let mutations_done = Arc::new(AtomicBool::new(false));
    let maintenance_clock = Arc::new(AtomicU64::new(1_000));
    let driver = tokio::spawn(drive_maintenance(
        backend.clone(),
        logical_index_id,
        config,
        Arc::clone(&mutations_done),
        Arc::clone(&maintenance_clock),
    ));

    let model = join_models(tasks).await;
    mutations_done.store(true, Ordering::SeqCst);
    let stats = driver.await.expect("maintenance driver did not panic");

    // The audit reads across two snapshots and assumes quiescence, so it runs
    // only after every task has joined.
    let report = audit::run(&backend, logical_index_id, &model)
        .await
        .unwrap_or_else(|failure| panic!("exact-membership audit failed: {failure}"));
    assert_eq!(report.records, model.len());
    assert_eq!(report.trees, 1, "every record carries bucket={TREE_BUCKET}");
    assert!(
        report.partitions > 1,
        "splits must have fired under the load"
    );
    assert!(
        stats.splits > 0,
        "the driver must complete at least one split"
    );
    eprintln!(
        "converged: {} records, report {report:?}, driver {stats:?}",
        model.len()
    );
    runtime.shutdown().await.expect("runtime shutdown");
}

/// The same seeded interleaving, but with the Runtime's real background Fixup
/// workers performing all Structure Maintenance: foreground mutations and
/// searches offer discovered partitions to the demand-driven queue and the
/// workers settle the tree with no manual drive. The scripts are unchanged,
/// so the final model stays interleaving-independent and the same
/// exact-membership audit must pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seeded_interleaving_with_background_fixups_converges() {
    let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()));
    let config = RuntimeConfig::default()
        .with_maintenance(2, 64)
        .and_then(|config| config.with_attempts(ATTEMPTS, ATTEMPTS))
        .and_then(|config| config.with_import_limits(1, 1))
        .expect("valid runtime config");
    let runtime = Runtime::new(backend.clone(), config).expect("runtime is valid");
    let index = runtime
        .create_index("concurrency-fixups", index_config())
        .await
        .expect("create index");
    let logical_index_id = index.logical_index_id();

    let tasks = spawn_scripts(&index);
    let model = join_models(tasks).await;

    // Background workers settle the topology; each settle poll offers cold
    // work through an ordinary search.
    audit::settle(&index, &backend, model.len() as u32).await;

    let report = audit::run(&backend, logical_index_id, &model)
        .await
        .unwrap_or_else(|failure| panic!("exact-membership audit failed: {failure}"));
    assert_eq!(report.records, model.len());
    assert_eq!(report.trees, 1, "every record carries bucket={TREE_BUCKET}");
    assert!(
        report.partitions > 1,
        "background fixups must have split under the load"
    );
    runtime.shutdown().await.expect("runtime shutdown");
}

/// A 4-dimensional L2 index with one i64 Tree Key field and small partition
/// entry bounds so splits (and, after deletes, merges) fire regularly.
fn index_config() -> IndexConfig {
    IndexConfig::new(DIMENSION, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(MIN_PARTITION_ENTRIES, MAX_PARTITION_ENTRIES)
        .expect("valid partition entries")
}

/// Draws one task's script: 45% inserts, 20% upserts, 20% deletes, 15%
/// searches, all over the task's own Record ID range.
fn generate_script(task: usize) -> Vec<Op> {
    let mut rng = Rng(BASE_SEED ^ (task as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut script = Vec::with_capacity(OPS_PER_TASK);
    for _ in 0..OPS_PER_TASK {
        let draw = rng.below(100);
        let ordinal = rng.below(IDS_PER_TASK);
        let op = if draw < 45 {
            Op::Insert {
                ordinal,
                vector: draw_vector(&mut rng),
            }
        } else if draw < 65 {
            Op::Upsert {
                ordinal,
                vector: draw_vector(&mut rng),
            }
        } else if draw < 85 {
            Op::Delete { ordinal }
        } else {
            Op::Search {
                vector: draw_vector(&mut rng),
                k: 1 + usize::try_from(rng.below(8)).expect("small k"),
            }
        };
        script.push(op);
    }
    script
}

/// One deterministic finite vector, components in `[0, 1000)`.
fn draw_vector(rng: &mut Rng) -> [f32; DIMENSION] {
    let mut vector = [0.0_f32; DIMENSION];
    for component in &mut vector {
        *component = rng.below(1_000_000) as f32 / 1_000.0;
    }
    vector
}

/// Runs one task's script against its own Index clone, keeping the task-local
/// model exact: without fault injection every committed operation reports its
/// outcome definitely, and disjoint ID ranges mean only this task's earlier
/// operations can affect its IDs.
async fn run_script(task: usize, script: Vec<Op>, index: Index<SharedBackend>) -> Model {
    let mut model = Model::new();
    for (step, op) in script.into_iter().enumerate() {
        match op {
            Op::Insert { ordinal, vector } => {
                let id = record_id(task, ordinal);
                match index.insert(record(&id, &vector)).await {
                    Ok(()) => assert!(
                        model.insert(id, model_record(&vector)).is_none(),
                        "task {task} step {step}: insert committed an already-held Record ID"
                    ),
                    Err(error) if error.kind() == ErrorKind::RecordAlreadyExists => assert!(
                        model.contains_key(&id),
                        "task {task} step {step}: RecordAlreadyExists for an unheld Record ID"
                    ),
                    Err(error) => panic!("task {task} step {step}: insert failed: {error:?}"),
                }
            }
            Op::Upsert { ordinal, vector } => {
                let id = record_id(task, ordinal);
                index
                    .upsert(record(&id, &vector))
                    .await
                    .unwrap_or_else(|error| {
                        panic!("task {task} step {step}: upsert failed: {error:?}")
                    });
                model.insert(id, model_record(&vector));
            }
            Op::Delete { ordinal } => {
                let id = record_id(task, ordinal);
                let existed = index.delete(id.clone()).await.unwrap_or_else(|error| {
                    panic!("task {task} step {step}: delete failed: {error:?}")
                });
                assert_eq!(
                    model.remove(&id).is_some(),
                    existed,
                    "task {task} step {step}: delete outcome disagrees with the model"
                );
            }
            Op::Search { vector, k } => {
                // Hits legitimately vary with the interleaving; only the
                // bound is stable.
                let outcome = index
                    .search(SearchRequest::new(vector_arc(&vector), k).expect("valid search"))
                    .await
                    .unwrap_or_else(|error| {
                        panic!("task {task} step {step}: search failed: {error:?}")
                    });
                assert!(
                    outcome.hits.len() <= k,
                    "task {task} step {step}: {} hits for k={k}",
                    outcome.hits.len()
                );
            }
        }
    }
    model
}

/// Drives split/merge state machines one bounded transition at a time until
/// the mutation tasks have finished AND a full partition listing finds
/// nothing left to advance. In-flight machines go first, then over-full
/// splits, then under-full merges — started work finishes before more begins.
/// A candidate whose last advance declined (`Idle`/`Stalled`) or exhausted
/// contention retries is skipped until any other candidate makes progress;
/// when every candidate has declined and mutations are done, the remaining
/// under-full partitions have no legal target and the topology is settled.
async fn drive_maintenance(
    backend: SharedBackend,
    index: LogicalIndexId,
    config: RuntimeConfig,
    mutations_done: Arc<AtomicBool>,
    clock: Arc<AtomicU64>,
) -> DriverStats {
    let mut stats = DriverStats::default();
    let mut declined: BTreeSet<(TreeKey, PartitionKey)> = BTreeSet::new();
    let retry = RetryPolicy::for_fixup(&config);
    loop {
        assert!(
            stats.advances < MAX_ADVANCES,
            "maintenance did not settle within {MAX_ADVANCES} advances"
        );
        let manifest = support::read_manifest(&backend, index).await;
        let listing = audit::list_partitions(&backend, index)
            .await
            .expect("partition listing");
        let candidates = collect_candidates(&listing, manifest.config());
        let candidate = candidates
            .iter()
            .find(|(tree_key, partition, _)| !declined.contains(&(tree_key.clone(), *partition)))
            .cloned();
        let Some((tree_key, partition, work)) = candidate else {
            if mutations_done.load(Ordering::SeqCst) {
                break;
            }
            declined.clear();
            tokio::task::yield_now().await;
            continue;
        };
        let started_at = clock.fetch_add(100, Ordering::SeqCst);
        stats.advances += 1;
        let progress = match work {
            Work::Split => {
                match split::advance(
                    &backend, &manifest, &tree_key, partition, started_at, &retry,
                )
                .await
                {
                    Ok(SplitAdvance::Completed { .. }) => {
                        stats.splits += 1;
                        true
                    }
                    Ok(SplitAdvance::Idle) => false,
                    Ok(_) => true,
                    Err(error) if error.kind() == ErrorKind::ContentionExhausted => false,
                    Err(error) => panic!("split advance failed: {error:?}"),
                }
            }
            Work::Merge => {
                match merge::advance(
                    &backend, &manifest, &tree_key, partition, started_at, &retry,
                )
                .await
                {
                    Ok(MergeAdvance::Completed) => {
                        stats.merges += 1;
                        true
                    }
                    Ok(MergeAdvance::Idle | MergeAdvance::Stalled) => false,
                    Ok(_) => true,
                    Err(error) if error.kind() == ErrorKind::ContentionExhausted => false,
                    Err(error) => panic!("merge advance failed: {error:?}"),
                }
            }
        };
        if progress {
            declined.clear();
        } else {
            stats.retries += 1;
            declined.insert((tree_key, partition));
        }
        tokio::task::yield_now().await;
    }
    stats
}

/// Appends one ordered group of partitions as candidates for `work`.
fn push_candidates(
    candidates: &mut Vec<(TreeKey, PartitionKey, Work)>,
    group: Vec<&(TreeKey, PartitionKey, PartitionHeader)>,
    work: Work,
) {
    candidates.extend(
        group
            .into_iter()
            .map(|(tree_key, partition, _)| (tree_key.clone(), *partition, work)),
    );
}

/// Orders one partition listing into drivable candidates: in-flight machines
/// first (smallest Tree Key and Partition Key), then over-full Ready
/// partitions (most entries, largest key on ties), then under-full Ready
/// non-roots (fewest entries, smallest key on ties) — mirroring the e2e
/// corpus's split/merge candidate rules.
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
    push_candidates(&mut candidates, in_flight_splits, Work::Split);

    let mut in_flight_merges: Vec<_> = listing
        .iter()
        .filter(|(_, _, header)| header.state() == PartitionState::Merging)
        .collect();
    in_flight_merges.sort_by_key(|(tree_key, partition, _)| (tree_key.clone(), *partition));
    push_candidates(&mut candidates, in_flight_merges, Work::Merge);

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
    push_candidates(&mut candidates, over_full, Work::Split);

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
    push_candidates(&mut candidates, under_full, Work::Merge);

    candidates
}

/// Draws every task's script and spawns one task per script. Scripts are
/// fully drawn before any task spawns: the interleaving never influences
/// which operations exist, only when they commit.
fn spawn_scripts(index: &Index<SharedBackend>) -> Vec<tokio::task::JoinHandle<Model>> {
    let scripts: Vec<Vec<Op>> = (0..TASKS).map(generate_script).collect();
    scripts
        .into_iter()
        .enumerate()
        .map(|(task, script)| tokio::spawn(run_script(task, script, index.clone())))
        .collect()
}

/// Joins every mutation task and merges the task-local models, asserting the
/// disjoint Record ID ranges keep the merge exact.
async fn join_models(tasks: Vec<tokio::task::JoinHandle<Model>>) -> Model {
    let mut model = Model::new();
    for task in tasks {
        let local = task.await.expect("mutation task did not panic");
        let (before, added) = (model.len(), local.len());
        model.extend(local);
        assert_eq!(
            model.len(),
            before + added,
            "task Record ID ranges must be disjoint"
        );
    }
    model
}

/// The task-owned Record ID `t{TASK}-r{ORDINAL}`.
fn record_id(task: usize, ordinal: u64) -> Bytes {
    Bytes::from(format!("t{task}-r{ordinal}"))
}

/// One scripted record in the single Tree Key bucket.
fn record(id: &Bytes, vector: &[f32; DIMENSION]) -> Record {
    Record::new(
        id.clone(),
        vector_arc(vector),
        vec![Value::I64(TREE_BUCKET)],
    )
    .expect("valid record")
}

/// The model mirror of one scripted record.
fn model_record(vector: &[f32; DIMENSION]) -> ModelRecord {
    ModelRecord {
        vector: vector_arc(vector),
        fields: vec![Value::I64(TREE_BUCKET)].into_boxed_slice(),
    }
}

/// Shares one scripted vector.
fn vector_arc(vector: &[f32; DIMENSION]) -> Arc<[f32]> {
    Arc::from(&vector[..])
}
