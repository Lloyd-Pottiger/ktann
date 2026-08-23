//! Bounded maintenance scheduling contract tests (#32).
//!
//! The Runtime's process-local Fixup queue drives split and merge work units
//! discovered by ordinary mutations and searches. These tests prove the
//! demand-driven contract end to end on the deterministic backend:
//!
//! - mutations drive splits and merges to completion with no manual drive;
//! - queue, concurrency, and retry bounds never affect persistent
//!   correctness, including under saturation;
//! - losing the queue (shutdown) leaves searchable durable state and a later
//!   relevant access resumes it (cold recovery);
//! - an unknown commit outcome retires the worker without corruption, and
//!   rediscovery resumes the state machine idempotently.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use ktann::api::{
    DataType, FieldId, FieldSchema, Index, IndexConfig, Metric, Record, RuntimeConfig,
    SearchRequest, Value,
};
use ktann::runtime::Runtime;
use ktann::storage::values::PartitionState;

use support::oracle::{Model, ModelRecord};
use support::{CommitFault, DeterministicBackend, DeterministicConfig, SharedBackend, audit};

#[allow(dead_code)]
mod support;

/// One-dimensional L2 vectors sharded by one i64 Tree Key field, with tiny
/// partitions so a handful of records exercises splits and merges.
fn index_config(minimum: u32, maximum: u32) -> IndexConfig {
    IndexConfig::new(1, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(minimum, maximum)
        .expect("valid partition entries")
}

/// A Runtime configuration with the maintenance knobs under test. The import
/// backlog watermark must stay within the queue capacity, so it is pinned to
/// one for the small test queues.
fn runtime_config(workers: usize, capacity: usize, fixup_attempts: u32) -> RuntimeConfig {
    RuntimeConfig::default()
        .with_maintenance(workers, capacity)
        .and_then(|config| config.with_attempts(fixup_attempts, 8))
        .and_then(|config| config.with_import_limits(1, 1))
        .expect("valid runtime config")
}

fn backend() -> SharedBackend {
    SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()))
}

fn rid(value: u8) -> Bytes {
    Bytes::copy_from_slice(&[b'r', value])
}

fn record(id: u8, x: f32) -> Record {
    Record::new(rid(id), Arc::from([x]), vec![Value::I64(7)]).expect("valid record")
}

fn model_record(x: f32) -> ModelRecord {
    ModelRecord {
        vector: Arc::from([x]),
        fields: vec![Value::I64(7)].into_boxed_slice(),
    }
}

/// Inserts one record through the public API and mirrors it into the model.
async fn insert(index: &Index<SharedBackend>, model: &mut Model, id: u8, x: f32) {
    index.insert(record(id, x)).await.expect("insert");
    model.insert(rid(id), model_record(x));
}

/// The reachable partitions' states and the total exact leaf entry count.
async fn topology(
    backend: &SharedBackend,
    index: &Index<SharedBackend>,
) -> (Vec<PartitionState>, u32) {
    let partitions = audit::list_partitions(backend, index.logical_index_id())
        .await
        .expect("list partitions");
    let states = partitions
        .iter()
        .map(|(_, _, header)| header.state())
        .collect();
    // Only leaf entries count membership; internal entries count children.
    let entries = partitions
        .iter()
        .filter(|(_, _, header)| header.level() == 1)
        .map(|(_, _, header)| header.entry_count())
        .sum();
    (states, entries)
}

/// Polls until `condition` holds, with a generous real-time bound. The
/// deterministic backend makes maintenance fast; the bound only guards
/// against a broken implementation hanging the test.
async fn wait_until(
    description: &str,
    mut condition: impl FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>>,
) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !condition().await {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for: {description}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Asserts the quiescent end state: every partition Ready, and the full
/// persistent-state audit passes against the model.
async fn assert_converged(backend: &SharedBackend, index: &Index<SharedBackend>, model: &Model) {
    let (states, _) = topology(backend, index).await;
    assert!(
        states.iter().all(|state| *state == PartitionState::Ready),
        "all partitions Ready, got {states:?}"
    );
    audit::run(backend, index.logical_index_id(), model)
        .await
        .expect("audit passes");
}

/// Asserts every modeled record reads back through the public API.
async fn assert_records(index: &Index<SharedBackend>, model: &Model) {
    for id in model.keys() {
        let stored = index
            .get(id.clone(), Default::default())
            .await
            .expect("get")
            .expect("record exists");
        assert_eq!(stored.id(), id);
    }
}

/// Drives demand-driven rediscovery until the topology settles: every search
/// visits and offers cold partitions, so repeated searches converge the
/// forest without any manual state-machine drive.
async fn settle(index: &Index<SharedBackend>, backend: &SharedBackend, model: &Model) {
    audit::settle(index, backend, model.len() as u32).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inserts_drive_splits_to_completion() {
    let backend = backend();
    let runtime = Runtime::new(backend.clone(), runtime_config(2, 16, 8)).expect("runtime");
    let index = runtime
        .create_index("split", index_config(1, 4))
        .await
        .expect("create index");
    let mut model = Model::new();
    for id in 0..12_u8 {
        insert(&index, &mut model, id, f32::from(id)).await;
    }
    settle(&index, &backend, &model).await;
    assert_converged(&backend, &index, &model).await;
    assert_records(&index, &model).await;

    // A search over the converged index returns the exact nearest neighbor.
    let outcome = index
        .search(SearchRequest::new(Arc::from([5.1_f32]), 3).expect("request"))
        .await
        .expect("search");
    let first = outcome.hits.first().expect("a hit");
    assert_eq!(first.id(), &rid(5));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deletes_drive_merges_to_completion() {
    let backend = backend();
    let runtime = Runtime::new(backend.clone(), runtime_config(2, 16, 8)).expect("runtime");
    let index = runtime
        .create_index("merge", index_config(2, 4))
        .await
        .expect("create index");
    let mut model = Model::new();
    for id in 0..8_u8 {
        insert(&index, &mut model, id, f32::from(id)).await;
    }
    settle(&index, &backend, &model).await;
    assert_converged(&backend, &index, &model).await;
    let partitions_before = audit::list_partitions(&backend, index.logical_index_id())
        .await
        .expect("list partitions")
        .len();
    assert!(
        partitions_before > 2,
        "the inserts forced at least one split"
    );

    // Delete down to a single record, settling between deletes: at most one
    // under-minimum leaf begins a merge at a time, so two leaves can never
    // begin merging into each other and stall with no `Ready` target (merge
    // convergence is conditional on a legal target by design).
    for id in 0..7_u8 {
        assert!(index.delete(rid(id)).await.expect("delete"));
        model.remove(&rid(id));
        settle(&index, &backend, &model).await;
    }
    assert_converged(&backend, &index, &model).await;
    assert_records(&index, &model).await;
    let partitions_after = audit::list_partitions(&backend, index.logical_index_id())
        .await
        .expect("list partitions")
        .len();
    assert!(
        partitions_after < partitions_before,
        "merges shrank the forest: {partitions_before} -> {partitions_after}"
    );
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queue_loss_leaves_searchable_state_and_search_resumes_it() {
    let backend = backend();
    // One Fixup attempt per execution: the worker performs exactly one
    // state-machine step per offer, leaving a cold Splitting root.
    let runtime_a = Runtime::new(backend.clone(), runtime_config(1, 4, 1)).expect("runtime");
    let index_a = runtime_a
        .create_index("cold", index_config(1, 4))
        .await
        .expect("create index");
    let mut model = Model::new();
    for id in 0..5_u8 {
        insert(&index_a, &mut model, id, f32::from(id)).await;
    }
    // The single-step worker begins the split and retires.
    wait_until("split begins", || {
        let backend = backend.clone();
        let index = index_a.clone();
        Box::pin(async move {
            let (states, _) = topology(&backend, &index).await;
            states.contains(&PartitionState::Splitting)
        })
    })
    .await;
    // The intermediate state stays searchable while the queue is live.
    assert_records(&index_a, &model).await;
    // Dropping the Runtime loses the queue; the durable state is untouched.
    runtime_a.shutdown().await.expect("shutdown");

    // A fresh Runtime has an empty queue; the first relevant search
    // rediscovers the cold Splitting root and drives the split to completion.
    let runtime_b = Runtime::new(backend.clone(), runtime_config(2, 16, 8)).expect("runtime");
    let index_b = runtime_b.open_index("cold").await.expect("open index");
    settle(&index_b, &backend, &model).await;
    assert_converged(&backend, &index_b, &model).await;
    assert_records(&index_b, &model).await;
    runtime_b.shutdown().await.expect("shutdown");
}

/// Polls until some reachable partition is in `expected`, without offering
/// new work. Assumes the runtime drives one state-machine step per offer
/// (`fixup_attempts == 1`): because no new offer lands during the wait, the
/// observed state cannot overshoot past the target.
async fn wait_for_state(
    backend: &SharedBackend,
    index: &Index<SharedBackend>,
    description: &str,
    expected: PartitionState,
) {
    wait_until(description, || {
        let backend = backend.clone();
        let index = index.clone();
        Box::pin(async move {
            let (states, _) = topology(&backend, &index).await;
            states.contains(&expected)
        })
    })
    .await;
}

/// Offers one rediscovery pass through a search: with `fixup_attempts == 1`
/// each offer drives exactly one bounded state-machine step.
async fn drive_one_step(index: &Index<SharedBackend>) {
    let request = SearchRequest::new(Arc::from([0.0_f32]), 1).expect("valid request");
    let _ = index.search(request).await;
}

/// Drives a root split to `DrainingSplit` one deterministic step at a time,
/// injects `fault` into the worker's drain commit, and proves the worker
/// retires without losing the searchable state and that later rediscovery
/// completes the split idempotently.
///
/// With `fixup_attempts == 1` each offer runs exactly one state-machine step:
/// the fifth insert's offer begins the split, one search-driven offer exposes
/// the targets and starts the drain, and the next offer runs the faulted
/// drain batch. The commit history length is the deterministic witness that
/// the faulted step actually executed.
async fn unknown_outcome_retires_and_rediscovery_resumes(fault: CommitFault, name: &str) {
    let backend = backend();
    let runtime = Runtime::new(backend.clone(), runtime_config(1, 4, 1)).expect("runtime");
    let index = runtime
        .create_index(name, index_config(1, 4))
        .await
        .expect("create index");
    let mut model = Model::new();
    for id in 0..5_u8 {
        insert(&index, &mut model, id, f32::from(id)).await;
    }
    wait_for_state(&backend, &index, "split begins", PartitionState::Splitting).await;
    drive_one_step(&index).await;
    wait_for_state(
        &backend,
        &index,
        "drain starts",
        PartitionState::DrainingSplit,
    )
    .await;

    // The next commit attempt anywhere is the worker's drain batch; the
    // injected fault decides whether it applies, but the worker retires
    // either way because the outcome is unknown.
    backend.inner().push_fault(fault).expect("fault");
    let commits = backend.inner().history().len();
    drive_one_step(&index).await;
    wait_until("the faulted step ran and retired", || {
        let backend = backend.clone();
        Box::pin(async move { backend.inner().history().len() > commits })
    })
    .await;
    // The retired worker leaves the draining state intact and searchable.
    let (states, entries) = topology(&backend, &index).await;
    assert!(states.contains(&PartitionState::DrainingSplit));
    assert_eq!(entries, 5);
    assert_records(&index, &model).await;

    // A later relevant access rediscovers the drain and completes the split.
    settle(&index, &backend, &model).await;
    assert_converged(&backend, &index, &model).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_outcome_not_applied_retires_and_rediscovery_resumes() {
    // Nothing is applied, yet the commit outcome is unknown.
    unknown_outcome_retires_and_rediscovery_resumes(CommitFault::UnknownNotApplied, "fault").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_outcome_applied_is_resumed_idempotently() {
    // The drain batch applies but reports an unknown outcome; rediscovery
    // observes the applied batch and completes without a blind retry.
    unknown_outcome_retires_and_rediscovery_resumes(CommitFault::UnknownApplied, "fault2").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn saturated_queue_never_loses_correctness() {
    let backend = backend();
    // One worker and a single pending-or-running slot: most offers drop, and
    // only repeated relevant access converges the forest.
    let runtime = Runtime::new(backend.clone(), runtime_config(1, 1, 8)).expect("runtime");
    let index = runtime
        .create_index("saturated", index_config(1, 4))
        .await
        .expect("create index");
    let mut model = Model::new();
    for id in 0..24_u8 {
        insert(&index, &mut model, id, f32::from(id)).await;
    }
    settle(&index, &backend, &model).await;
    assert_converged(&backend, &index, &model).await;
    assert_records(&index, &model).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_with_pending_work_leaves_durable_state() {
    let backend = backend();
    let runtime = Runtime::new(backend.clone(), runtime_config(1, 1, 1)).expect("runtime");
    let index = runtime
        .create_index("shutdown", index_config(1, 4))
        .await
        .expect("create index");
    let mut model = Model::new();
    for id in 0..6_u8 {
        insert(&index, &mut model, id, f32::from(id)).await;
    }
    // Shutdown may interrupt the queue at any point; every durable state it
    // leaves is searchable.
    runtime.shutdown().await.expect("shutdown");
    let runtime = Runtime::new(backend.clone(), runtime_config(1, 4, 8)).expect("runtime");
    let index = runtime.open_index("shutdown").await.expect("open index");
    assert_records(&index, &model).await;
    settle(&index, &backend, &model).await;
    assert_converged(&backend, &index, &model).await;
    runtime.shutdown().await.expect("shutdown");
}
