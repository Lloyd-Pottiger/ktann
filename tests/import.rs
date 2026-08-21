//! Import Session admission, ordering, cancellation, and shutdown contract tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, GetOptions, ImportOptions, Index, IndexConfig,
    Metric, Mutation, MutationOutcome, Record, RuntimeConfig, Value,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome,
    Mutation as StorageMutation, ReadOps, ReadTxn, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::KeyRange;

use support::{
    CommitFault, DeterministicBackend, DeterministicConfig, DeterministicReadTxn,
    DeterministicWriteTxn,
};

#[allow(dead_code)]
mod support;

/// Holds chosen commits before their commit boundary until released.
///
/// The wait runs before [`CommitStart::begin`], so a held commit has not
/// crossed the Runtime's cancellation boundary and remains cancellable.
#[derive(Default)]
struct CommitGate {
    block_next: AtomicUsize,
    entered: AtomicUsize,
    released: AtomicBool,
}

impl CommitGate {
    /// Holds the next `commits` commit attempts until [`CommitGate::release`].
    fn hold_next(&self, commits: usize) {
        self.block_next.fetch_add(commits, Ordering::SeqCst);
    }

    async fn maybe_wait(&self) {
        let held = self
            .block_next
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |pending| {
                pending.checked_sub(1)
            })
            .is_ok();
        if !held {
            return;
        }
        self.entered.fetch_add(1, Ordering::SeqCst);
        while !self.released.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    }

    async fn wait_until_entered(&self, commits: usize) {
        while self.entered.load(Ordering::SeqCst) < commits {
            tokio::task::yield_now().await;
        }
    }

    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct GatedBackend {
    inner: Arc<DeterministicBackend>,
    gate: Arc<CommitGate>,
}

impl GatedBackend {
    fn new(inner: DeterministicBackend, gate: Arc<CommitGate>) -> Self {
        Self {
            inner: Arc::new(inner),
            gate,
        }
    }
}

impl Backend for GatedBackend {
    type ReadTxn<'backend> = DeterministicReadTxn<'backend>;

    type WriteTxn<'backend> = GatedWriteTxn<'backend>;

    fn hard_limits(&self) -> HardLimits {
        self.inner.hard_limits()
    }

    fn admission_budget(&self) -> AdmissionBudget {
        self.inner.admission_budget()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn shutdown(&self) {
        self.inner.shutdown().await;
    }

    async fn begin_read(&self) -> ktann::api::Result<Self::ReadTxn<'_>> {
        self.inner.begin_read().await
    }

    async fn begin_write(&self) -> ktann::api::Result<Self::WriteTxn<'_>> {
        Ok(GatedWriteTxn {
            inner: self.inner.begin_write().await?,
            gate: Arc::clone(&self.gate),
        })
    }
}

struct GatedWriteTxn<'backend> {
    inner: DeterministicWriteTxn<'backend>,
    gate: Arc<CommitGate>,
}

impl ReadOps for GatedWriteTxn<'_> {
    async fn get(&mut self, key: Bytes) -> ktann::api::Result<Option<Bytes>> {
        self.inner.get(key).await
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> ktann::api::Result<Vec<Option<Bytes>>> {
        self.inner.batch_get(keys).await
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> ktann::api::Result<ScanPage> {
        self.inner.scan(range, limits).await
    }
}

impl ReadTxn for GatedWriteTxn<'_> {}

impl WriteTxn for GatedWriteTxn<'_> {
    async fn get_for_update(&mut self, key: Bytes) -> ktann::api::Result<Option<Bytes>> {
        self.inner.get_for_update(key).await
    }

    async fn batch_get_for_update(
        &mut self,
        keys: Vec<Bytes>,
    ) -> ktann::api::Result<Vec<Option<Bytes>>> {
        self.inner.batch_get_for_update(keys).await
    }

    async fn put(&mut self, key: Bytes, value: Bytes) -> ktann::api::Result<()> {
        self.inner.put(key, value).await
    }

    async fn insert(&mut self, key: Bytes, value: Bytes) -> ktann::api::Result<InsertOutcome> {
        self.inner.insert(key, value).await
    }

    async fn delete(&mut self, key: Bytes) -> ktann::api::Result<()> {
        self.inner.delete(key).await
    }

    async fn batch_mutate(&mut self, mutations: Vec<StorageMutation>) -> ktann::api::Result<()> {
        self.inner.batch_mutate(mutations).await
    }

    async fn clear_range(&mut self, range: &KeyRange) -> ktann::api::Result<()> {
        self.inner.clear_range(range).await
    }

    async fn commit_with(self, start: CommitStart) -> ktann::api::Result<()> {
        self.gate.maybe_wait().await;
        self.inner.commit_with(start).await
    }

    async fn rollback(self) {
        self.inner.rollback().await;
    }
}

fn config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
}

fn rid(value: u8) -> Bytes {
    Bytes::copy_from_slice(&[b'r', value])
}

fn record(id: &[u8], x: f32, bucket: i64) -> Record {
    Record::new(
        Bytes::copy_from_slice(id),
        Arc::from([x, 0.0_f32]),
        vec![Value::I64(bucket)],
    )
    .expect("valid record")
}

fn insert(id: &[u8], x: f32, bucket: i64) -> Vec<Mutation> {
    vec![Mutation::Insert(record(id, x, bucket))]
}

async fn setup() -> (
    GatedBackend,
    Arc<CommitGate>,
    Runtime<GatedBackend>,
    Index<GatedBackend>,
) {
    setup_with(RuntimeConfig::default()).await
}

async fn setup_with(
    runtime_config: RuntimeConfig,
) -> (
    GatedBackend,
    Arc<CommitGate>,
    Runtime<GatedBackend>,
    Index<GatedBackend>,
) {
    let gate = Arc::new(CommitGate::default());
    let backend = GatedBackend::new(
        DeterministicBackend::new(DeterministicConfig::default()),
        Arc::clone(&gate),
    );
    let runtime = Runtime::new(backend.clone(), runtime_config).expect("runtime is valid");
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create index");
    (backend, gate, runtime, index)
}

fn import_options(in_flight: usize) -> ImportOptions {
    ImportOptions::default()
        .with_in_flight_batches(in_flight)
        .expect("positive in-flight limit")
}

async fn wait_until_present(index: &Index<GatedBackend>, id: &[u8]) {
    loop {
        let visible = index
            .get(Bytes::copy_from_slice(id), GetOptions::default())
            .await
            .expect("get")
            .is_some();
        if visible {
            break;
        }
        tokio::task::yield_now().await;
    }
}

async fn assert_record_present(index: &Index<GatedBackend>, id: &[u8]) {
    assert!(
        index
            .get(Bytes::copy_from_slice(id), GetOptions::default())
            .await
            .expect("get")
            .is_some(),
        "record must be present"
    );
}

async fn assert_record_absent(index: &Index<GatedBackend>, id: &[u8]) {
    assert!(
        index
            .get(Bytes::copy_from_slice(id), GetOptions::default())
            .await
            .expect("get")
            .is_none(),
        "record must be absent"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_issues_monotonic_tokens_and_finish_reports_submission_order() {
    let (_backend, _gate, runtime, index) = setup().await;
    let mut session = index.import_session(import_options(2)).expect("session");

    let mut tokens = Vec::new();
    for value in 0..4_u8 {
        tokens.push(
            session
                .submit(insert(&rid(value), f32::from(value), i64::from(value)))
                .await
                .expect("submit"),
        );
    }
    assert_eq!(
        tokens.iter().map(|token| token.get()).collect::<Vec<_>>(),
        vec![1, 2, 3, 4],
        "tokens increase monotonically from 1"
    );

    let results = session.finish().await;
    assert_eq!(results.len(), 4);
    for (token, entry) in tokens.iter().zip(&results) {
        assert_eq!(entry.token, *token, "outcomes appear in submission order");
        assert_eq!(
            entry.result.as_ref().expect("batch commits").as_slice(),
            &[MutationOutcome::Inserted]
        );
    }
    for value in 0..4_u8 {
        assert_record_present(&index, &rid(value)).await;
    }
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn out_of_order_completion_preserves_submission_order() {
    let (_backend, gate, runtime, index) = setup().await;
    let mut session = index.import_session(import_options(2)).expect("session");

    gate.hold_next(1);
    let token1 = session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect("submit 1");
    gate.wait_until_entered(1).await;

    let token2 = session
        .submit(insert(&rid(2), 2.0, 2))
        .await
        .expect("submit 2");
    // Batch 2 commits first while batch 1 is held before its commit boundary.
    wait_until_present(&index, &rid(2)).await;
    assert_record_absent(&index, &rid(1)).await;

    gate.release();
    let results = session.finish().await;
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].token, token1);
    assert_eq!(results[1].token, token2);
    for entry in &results {
        assert_eq!(
            entry.result.as_ref().expect("batch commits").as_slice(),
            &[MutationOutcome::Inserted]
        );
    }
    assert_record_present(&index, &rid(1)).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejected_batches_fail_validation_without_consuming_tokens() {
    let (_backend, _gate, runtime, index) = setup().await;
    let mut session = index
        .import_session(ImportOptions::default())
        .expect("session");

    let error = session
        .submit(vec![
            Mutation::Insert(record(&rid(1), 1.0, 1)),
            Mutation::Insert(record(&rid(1), 2.0, 2)),
        ])
        .await
        .expect_err("duplicate Record IDs are rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(error.position(), Some(1));

    let wrong_dimension = Record::new(rid(2), Arc::from([1.0_f32, 2.0, 3.0]), vec![Value::I64(1)])
        .expect("record shape is independent of the index");
    let error = session
        .submit(vec![Mutation::Insert(wrong_dimension)])
        .await
        .expect_err("a dimension mismatch is rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(error.position(), Some(0));

    let token1 = session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect("submit");
    assert_eq!(token1.get(), 1, "rejected submissions consume no token");
    let token2 = session.submit(Vec::new()).await.expect("empty submit");
    assert_eq!(token2.get(), 2, "an empty batch is accepted with a token");

    let results = session.finish().await;
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].token, token1);
    assert_eq!(
        results[0]
            .result
            .as_ref()
            .expect("batch commits")
            .as_slice(),
        &[MutationOutcome::Inserted]
    );
    assert_eq!(results[1].token, token2);
    assert!(
        results[1]
            .result
            .as_ref()
            .expect("empty batch succeeds")
            .is_empty()
    );
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_flight_limit_bounds_concurrent_commits() {
    let (_backend, gate, runtime, index) = setup().await;
    let mut session = index.import_session(import_options(1)).expect("session");

    // Hold every commit attempt in this test until released.
    gate.hold_next(64);
    session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect("submit 1");
    gate.wait_until_entered(1).await;

    let token2 = {
        let mut submit2 = std::pin::pin!(session.submit(insert(&rid(2), 2.0, 2)));
        tokio::select! {
            result = &mut submit2 => {
                panic!("submit must wait for a slot while batch 1 is held: {result:?}")
            }
            () = async {
                for _ in 0..64 {
                    tokio::task::yield_now().await;
                }
            } => {}
        }
        assert_eq!(
            gate.entered.load(Ordering::SeqCst),
            1,
            "the second batch never reaches commit while the slot is held"
        );

        gate.release();
        submit2.await.expect("submit 2")
    };
    assert_eq!(token2.get(), 2);
    let results = session.finish().await;
    assert_eq!(results.len(), 2);
    for entry in &results {
        assert_eq!(
            entry.result.as_ref().expect("batch commits").as_slice(),
            &[MutationOutcome::Inserted]
        );
    }
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_session_cancels_pre_commit_batches() {
    let (backend, gate, runtime, index) = setup().await;
    let mut session = index.import_session(import_options(1)).expect("session");

    // Hold every commit attempt in this test until released.
    gate.hold_next(64);
    session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect("submit");
    gate.wait_until_entered(1).await;

    drop(session);
    assert_record_absent(&index, &rid(1)).await;
    runtime
        .shutdown()
        .await
        .expect("shutdown drains the cancelled batch");

    // A cancelled batch never commits, even observed from a reopened index.
    let runtime = Runtime::new(backend.clone(), RuntimeConfig::default()).expect("runtime");
    let index = runtime.open_index("index").await.expect("open index");
    assert_record_absent(&index, &rid(1)).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn committed_batches_finish_without_a_session_observer() {
    let (_backend, _gate, runtime, index) = setup().await;
    let mut session = index
        .import_session(ImportOptions::default())
        .expect("session");

    session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect("submit");
    wait_until_present(&index, &rid(1)).await;

    drop(session);
    assert_record_present(&index, &rid(1)).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_commit_outcome_is_reported_in_place() {
    let (backend, _gate, runtime, index) = setup().await;
    // One in-flight slot serializes the batch commits, so the FIFO fault plan
    // lands deterministically: batch 1 commits, batch 2's outcome is unknown.
    backend
        .inner
        .set_fault_plan(vec![CommitFault::Normal, CommitFault::UnknownApplied])
        .expect("fault plan");
    let mut session = index.import_session(import_options(1)).expect("session");

    for value in 1..=3_u8 {
        session
            .submit(insert(&rid(value), f32::from(value), i64::from(value)))
            .await
            .expect("submit");
    }
    let results = session.finish().await;
    assert_eq!(results.len(), 3);
    assert_eq!(
        results[0]
            .result
            .as_ref()
            .expect("batch 1 commits")
            .as_slice(),
        &[MutationOutcome::Inserted]
    );
    let error = results[1]
        .result
        .as_ref()
        .expect_err("batch 2 has an unknown commit outcome");
    assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    assert_eq!(
        results[2]
            .result
            .as_ref()
            .expect("batch 3 commits")
            .as_slice(),
        &[MutationOutcome::Inserted]
    );

    // An unknown outcome is never reported as success: the batch did commit.
    assert_record_present(&index, &rid(2)).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_batch_does_not_discard_other_results() {
    let (_backend, _gate, runtime, index) = setup().await;
    index.insert(record(&rid(1), 1.0, 1)).await.expect("seed");
    let mut session = index.import_session(import_options(2)).expect("session");

    session
        .submit(insert(&rid(1), 9.0, 9))
        .await
        .expect("submit 1");
    session
        .submit(insert(&rid(2), 2.0, 2))
        .await
        .expect("submit 2");
    let results = session.finish().await;
    assert_eq!(results.len(), 2);
    let error = results[0]
        .result
        .as_ref()
        .expect_err("the duplicate insert fails");
    assert_eq!(error.kind(), ErrorKind::RecordAlreadyExists);
    assert_eq!(error.position(), Some(0));
    assert_eq!(
        results[1]
            .result
            .as_ref()
            .expect("batch 2 commits")
            .as_slice(),
        &[MutationOutcome::Inserted]
    );

    assert_record_present(&index, &rid(1)).await;
    assert_record_present(&index, &rid(2)).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_waits_for_admitted_batches_and_cancels_queued_ones() {
    let runtime_config = RuntimeConfig::default()
        .with_foreground_operation_limit(1)
        .expect("positive limit");
    let (_backend, gate, runtime, index) = setup_with(runtime_config).await;
    let mut session = index.import_session(import_options(2)).expect("session");

    // Batch 1 holds the only foreground permit inside the commit gate; batch 2
    // is admitted by the session but queues behind foreground admission.
    // Hold every commit attempt in this test until released.
    gate.hold_next(64);
    session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect("submit 1");
    gate.wait_until_entered(1).await;
    session
        .submit(insert(&rid(2), 2.0, 2))
        .await
        .expect("submit 2");

    let shutdown_done = Arc::new(AtomicBool::new(false));
    let shutdown = tokio::spawn({
        let runtime = runtime.clone();
        let shutdown_done = Arc::clone(&shutdown_done);
        async move {
            runtime.shutdown().await.expect("shutdown");
            shutdown_done.store(true, Ordering::SeqCst);
        }
    });
    // Wait until shutdown has actually stopped foreground admission before
    // releasing the gate: batch 1 holds the only foreground permit until its
    // commit completes, so batch 2 is guaranteed to still be queued exactly
    // when admission closes. Session construction is the side-effect-free
    // probe — it fails with RuntimeClosed once admission has stopped.
    let mut probes = 0_u32;
    loop {
        match index.import_session(ImportOptions::default()) {
            Ok(probe) => drop(probe),
            Err(error) if error.kind() == ErrorKind::RuntimeClosed => break,
            Err(error) => panic!("unexpected probe error: {error:?}"),
        }
        probes += 1;
        assert!(
            probes < 10_000,
            "shutdown did not stop foreground admission"
        );
        tokio::task::yield_now().await;
    }
    assert!(
        !shutdown_done.load(Ordering::SeqCst),
        "shutdown waits for the admitted batch"
    );
    gate.release();

    let results = session.finish().await;
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0]
            .result
            .as_ref()
            .expect("an admitted batch keeps its real result")
            .as_slice(),
        &[MutationOutcome::Inserted]
    );
    let error = results[1]
        .result
        .as_ref()
        .expect_err("the queued batch never began");
    assert_eq!(error.kind(), ErrorKind::RuntimeClosed);

    shutdown.await.expect("shutdown task");
    assert!(shutdown_done.load(Ordering::SeqCst));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn import_session_after_shutdown_is_runtime_closed() {
    let (_backend, _gate, runtime, index) = setup().await;
    runtime.shutdown().await.expect("shutdown");
    let error = index
        .import_session(ImportOptions::default())
        .expect_err("a closed Runtime rejects session construction");
    assert_eq!(error.kind(), ErrorKind::RuntimeClosed);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_after_shutdown_is_runtime_closed() {
    let (_backend, _gate, runtime, index) = setup().await;
    let mut session = index
        .import_session(ImportOptions::default())
        .expect("session");
    runtime.shutdown().await.expect("shutdown");

    let error = session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect_err("a closed Runtime rejects submission");
    assert_eq!(error.kind(), ErrorKind::RuntimeClosed);
    assert!(
        session.finish().await.is_empty(),
        "no batch was ever accepted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_debug_redacts_the_index_name() {
    let (_backend, _gate, runtime, _index) = setup().await;
    let index = runtime
        .create_index("import-redaction-secret", config())
        .await
        .expect("create index");
    let session = index
        .import_session(ImportOptions::default())
        .expect("session");
    let debug = format!("{session:?}");
    assert!(debug.contains("ImportSession"));
    assert!(!debug.contains("import-redaction-secret"));
    runtime.shutdown().await.expect("shutdown");
}
