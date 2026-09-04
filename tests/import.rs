//! Import Session admission, ordering, cancellation, and shutdown contract tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, GetOptions, ImportOptions, ImportSession, Index,
    IndexConfig, Metric, Mutation, MutationOutcome, Record, RuntimeConfig, SearchRequest, Value,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome,
    Mutation as StorageMutation, ReadOps, ReadTxn, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::KeyRange;

use support::{
    CommitFault, CommitOutcome, DeterministicBackend, DeterministicConfig, DeterministicReadTxn,
    DeterministicWriteTxn,
};

#[allow(dead_code)]
mod support;

/// A generous bound on every wait in these tests: a missed wakeup or a lost
/// Fixup offer must fail the test in seconds, never hang a CI job for hours
/// (issue #109).
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Holds chosen commits before their commit boundary until released.
///
/// The wait runs before [`CommitStart::begin`], so a held commit has not
/// crossed the Runtime's cancellation boundary and remains cancellable. Held
/// commits are released in entry order: [`CommitGate::release_one`] frees
/// exactly the earliest entered commit, [`CommitGate::release`] frees all.
///
/// Both waits are bounded by [`WAIT_TIMEOUT`]: a commit that is never
/// released, or an entry that never happens — for example because the Fixup
/// offer that should have triggered it was lost — panics instead of spinning
/// forever.
#[derive(Default)]
struct CommitGate {
    block_next: AtomicUsize,
    entered: AtomicUsize,
    released: AtomicUsize,
}

impl CommitGate {
    /// Holds the next `commits` commit attempts until released.
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
        let position = self.entered.fetch_add(1, Ordering::SeqCst);
        let released = async {
            while self.released.load(Ordering::SeqCst) <= position {
                tokio::task::yield_now().await;
            }
        };
        tokio::time::timeout(WAIT_TIMEOUT, released)
            .await
            .expect("a held commit was not released in time");
    }

    async fn wait_until_entered(&self, commits: usize) {
        let entered = async {
            while self.entered.load(Ordering::SeqCst) < commits {
                tokio::task::yield_now().await;
            }
        };
        tokio::time::timeout(WAIT_TIMEOUT, entered)
            .await
            .expect("the expected commits never entered the gate");
    }

    /// Releases every held commit.
    fn release(&self) {
        self.released.store(usize::MAX, Ordering::SeqCst);
    }

    /// Releases commits that have already entered while keeping later holds usable.
    fn release_entered(&self) {
        self.released
            .store(self.entered.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    /// Releases exactly the earliest entered held commit.
    fn release_one(&self) {
        self.released.fetch_add(1, Ordering::SeqCst);
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

    async fn batch_scan(
        &mut self,
        ranges: Vec<KeyRange>,
        limits: ScanLimits,
    ) -> ktann::api::Result<Vec<ScanPage>> {
        self.inner.batch_scan(ranges, limits).await
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
    setup_with_index(runtime_config, config()).await
}

async fn setup_with_index(
    runtime_config: RuntimeConfig,
    index_config: IndexConfig,
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
        .create_index("index", index_config)
        .await
        .expect("create index");
    (backend, gate, runtime, index)
}

fn import_options(in_flight: usize) -> ImportOptions {
    ImportOptions::default()
        .with_max_in_flight_batches(in_flight)
        .expect("positive maximum in-flight limit")
}

/// Keeps a limit-one session saturated for the controller's initial clean
/// window, then waits for the final warmup batch to finish at limit two.
async fn warm_import_concurrency(
    session: &mut ImportSession<GatedBackend>,
    gate: &CommitGate,
    index: &Index<GatedBackend>,
    first_value: u8,
) {
    const INITIAL_CLEAN_WINDOW: u8 = 8;

    let first_entry = gate.entered.load(Ordering::SeqCst) + 1;
    gate.hold_next(1);
    session
        .submit(insert(
            &rid(first_value),
            f32::from(first_value),
            i64::from(first_value),
        ))
        .await
        .expect("first warmup submit");
    gate.wait_until_entered(first_entry).await;

    for offset in 1..=INITIAL_CLEAN_WINDOW {
        let value = first_value + offset;
        gate.hold_next(1);
        let mut next =
            std::pin::pin!(
                session.submit(insert(&rid(value), f32::from(value), i64::from(value),))
            );
        assert_submit_pending(&mut next, "warmup must maintain admission demand").await;
        gate.release_one();
        next.await.expect("demanded warmup submit");
        gate.wait_until_entered(first_entry + usize::from(offset))
            .await;
    }
    gate.release_one();
    wait_until_present(index, &rid(first_value + INITIAL_CLEAN_WINDOW)).await;
}

async fn wait_until_present(index: &Index<GatedBackend>, id: &[u8]) {
    let deadline = Instant::now() + WAIT_TIMEOUT;
    loop {
        let visible = index
            .get(Bytes::copy_from_slice(id), GetOptions::default())
            .await
            .expect("get")
            .is_some();
        if visible {
            break;
        }
        assert!(Instant::now() < deadline, "the record never became visible");
        tokio::task::yield_now().await;
    }
}

fn aborted_commit_count(backend: &GatedBackend) -> usize {
    backend
        .inner
        .history()
        .iter()
        .filter(|entry| entry.outcome == CommitOutcome::Aborted)
        .count()
}

async fn wait_for_aborted_commits(backend: &GatedBackend, expected: usize) {
    let deadline = Instant::now() + WAIT_TIMEOUT;
    loop {
        if aborted_commit_count(backend) >= expected {
            return;
        }
        assert!(Instant::now() < deadline, "commit aborts were not observed");
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

    warm_import_concurrency(&mut session, &gate, &index, 20).await;

    let held_entry = gate.entered.load(Ordering::SeqCst) + 1;
    gate.hold_next(1);
    let token1 = session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect("submit 1");
    gate.wait_until_entered(held_entry).await;

    let token2 = session
        .submit(insert(&rid(2), 2.0, 2))
        .await
        .expect("submit 2");
    // Batch 2 commits first while batch 1 is held before its commit boundary.
    wait_until_present(&index, &rid(2)).await;
    assert_record_absent(&index, &rid(1)).await;

    gate.release();
    let results = session.finish().await;
    assert_eq!(results.len(), 11);
    let token1_position = results
        .iter()
        .position(|entry| entry.token == token1)
        .expect("batch 1 has an outcome");
    let token2_position = results
        .iter()
        .position(|entry| entry.token == token2)
        .expect("batch 2 has an outcome");
    assert!(token1_position < token2_position);
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
        assert_submit_pending(
            &mut submit2,
            "submit must wait for a slot while batch 1 is held",
        )
        .await;
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
async fn import_concurrency_grows_after_clean_work_and_contracts_on_contention() {
    let (_backend, gate, runtime, index) = setup().await;
    let mut session = index.import_session(import_options(2)).expect("session");

    // Sustained demand across clean, disjoint Tree Keys earns a probe at two.
    warm_import_concurrency(&mut session, &gate, &index, 0).await;

    // Both same-leaf batches reach commit concurrently, proving the learned
    // limit grew to two. Their conflict then contracts the session to one.
    let first_contender = gate.entered.load(Ordering::SeqCst) + 1;
    gate.hold_next(2);
    session
        .submit(insert(&rid(9), 9.0, 0))
        .await
        .expect("first contending submit");
    gate.wait_until_entered(first_contender).await;
    session
        .submit(insert(&rid(10), 10.0, 0))
        .await
        .expect("second contending submit");
    gate.wait_until_entered(first_contender + 1).await;
    gate.release_entered();
    wait_until_present(&index, &rid(9)).await;
    wait_until_present(&index, &rid(10)).await;

    // After the retryable conflict, only one new batch may run. The next
    // clean window must be earned before the controller probes two again.
    gate.hold_next(1);
    session
        .submit(insert(&rid(11), 11.0, 11))
        .await
        .expect("serial submit");
    gate.wait_until_entered(first_contender + 2).await;
    let token = {
        let mut next = std::pin::pin!(session.submit(insert(&rid(12), 12.0, 12)));
        assert_submit_pending(
            &mut next,
            "contention must contract the learned concurrency to one",
        )
        .await;
        assert_eq!(gate.entered.load(Ordering::SeqCst), first_contender + 2);
        gate.release();
        next.await.expect("submit after contracted slot opens")
    };
    assert_eq!(token.get(), 13);

    let results = session.finish().await;
    assert_eq!(results.len(), 13);
    assert!(results.iter().all(|entry| entry.result.is_ok()));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serial_clean_work_does_not_raise_import_concurrency() {
    let (_backend, gate, runtime, index) = setup().await;
    let mut session = index.import_session(import_options(2)).expect("session");

    // Clean completions without a waiting submission do not demonstrate that
    // overlapping writes are useful, regardless of how long they continue.
    for value in 0..16_u8 {
        session
            .submit(insert(&rid(value), f32::from(value), i64::from(value)))
            .await
            .expect("serial submit");
        wait_until_present(&index, &rid(value)).await;
    }

    let held_entry = gate.entered.load(Ordering::SeqCst) + 1;
    gate.hold_next(1);
    session
        .submit(insert(&rid(16), 16.0, 16))
        .await
        .expect("held serial submit");
    gate.wait_until_entered(held_entry).await;

    {
        let mut next = std::pin::pin!(session.submit(insert(&rid(17), 17.0, 17)));
        assert_submit_pending(
            &mut next,
            "unsaturated clean work must leave the learned limit at one",
        )
        .await;
        gate.release();
        next.await.expect("submit after the serial slot opens");
    }

    let results = session.finish().await;
    assert_eq!(results.len(), 18);
    assert!(results.iter().all(|entry| entry.result.is_ok()));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminal_contention_still_contracts_import_concurrency() {
    let runtime_config = RuntimeConfig::default()
        .with_attempts(2, 1)
        .expect("one foreground attempt");
    let (backend, gate, runtime, index) = setup_with(runtime_config).await;
    let mut session = index.import_session(import_options(2)).expect("session");
    warm_import_concurrency(&mut session, &gate, &index, 0).await;

    let aborted_before = aborted_commit_count(&backend);
    let first_contender = gate.entered.load(Ordering::SeqCst) + 1;
    gate.hold_next(2);
    session
        .submit(insert(&rid(9), 9.0, 0))
        .await
        .expect("first contending submit");
    gate.wait_until_entered(first_contender).await;
    session
        .submit(insert(&rid(10), 10.0, 0))
        .await
        .expect("second contending submit");
    gate.wait_until_entered(first_contender + 1).await;
    gate.release_entered();
    wait_for_aborted_commits(&backend, aborted_before + 1).await;

    let held_entry = gate.entered.load(Ordering::SeqCst) + 1;
    gate.hold_next(1);
    session
        .submit(insert(&rid(11), 11.0, 11))
        .await
        .expect("submit after terminal contention");
    gate.wait_until_entered(held_entry).await;
    {
        let mut next = std::pin::pin!(session.submit(insert(&rid(12), 12.0, 12)));
        assert_submit_pending(
            &mut next,
            "terminal contention must contract the next admission window",
        )
        .await;
        gate.release();
        next.await.expect("submit after contracted slot opens");
    }

    let results = session.finish().await;
    assert_eq!(results.len(), 13);
    let failures = results
        .iter()
        .filter_map(|entry| entry.result.as_ref().err())
        .collect::<Vec<_>>();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].kind(), ErrorKind::ContentionExhausted);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retry_waiters_retain_the_ceiling_and_precede_new_submissions() {
    let runtime_config = RuntimeConfig::default()
        .with_retry_backoff(Duration::from_millis(100), Duration::from_millis(100))
        .expect("fixed retry backoff");
    let (backend, gate, runtime, index) = setup_with(runtime_config).await;
    let mut session = index.import_session(import_options(2)).expect("session");
    warm_import_concurrency(&mut session, &gate, &index, 0).await;

    let aborted_before = aborted_commit_count(&backend);
    backend
        .inner
        .push_fault(CommitFault::Abort)
        .and_then(|()| backend.inner.push_fault(CommitFault::Abort))
        .expect("two retryable commit faults");

    let first_attempt = gate.entered.load(Ordering::SeqCst) + 1;
    gate.hold_next(4);
    session
        .submit(insert(&rid(30), 30.0, 30))
        .await
        .expect("first faulted submit");
    gate.wait_until_entered(first_attempt).await;
    session
        .submit(insert(&rid(31), 31.0, 31))
        .await
        .expect("second faulted submit");
    gate.wait_until_entered(first_attempt + 1).await;
    gate.release_entered();
    wait_for_aborted_commits(&backend, aborted_before + 2).await;

    {
        let mut next = std::pin::pin!(session.submit(insert(&rid(32), 32.0, 32)));
        assert_submit_pending(
            &mut next,
            "paused accepted batches retain the configured ceiling",
        )
        .await;

        gate.wait_until_entered(first_attempt + 2).await;
        assert_submit_pending(&mut next, "the first retry must precede a new submission").await;
        gate.release_one();

        gate.wait_until_entered(first_attempt + 3).await;
        assert_submit_pending(&mut next, "every retry waiter precedes a new submission").await;
        gate.release();
        next.await.expect("submit after both retry waiters finish");
    }

    let results = session.finish().await;
    assert_eq!(results.len(), 12);
    assert!(results.iter().all(|entry| entry.result.is_ok()));
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

    warm_import_concurrency(&mut session, &gate, &index, 20).await;

    // Batch 1 holds the only foreground permit inside the commit gate; batch 2
    // is admitted by the session but queues behind foreground admission.
    // Hold every commit attempt in this test until released.
    let held_entry = gate.entered.load(Ordering::SeqCst) + 1;
    gate.hold_next(64);
    let token1 = session
        .submit(insert(&rid(1), 1.0, 1))
        .await
        .expect("submit 1");
    gate.wait_until_entered(held_entry).await;
    let token2 = session
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
    assert_eq!(results.len(), 11);
    let admitted = results
        .iter()
        .find(|entry| entry.token == token1)
        .expect("the admitted token has an outcome");
    assert_eq!(
        admitted
            .result
            .as_ref()
            .expect("an admitted batch keeps its real result")
            .as_slice(),
        &[MutationOutcome::Inserted]
    );
    let queued = results
        .iter()
        .find(|entry| entry.token == token2)
        .expect("the queued token has an outcome");
    let error = queued
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

/// A Runtime configuration for backlog-gate tests: one worker, a two-slot
/// Fixup queue, one state-machine step per Fixup execution, and an import
/// backlog watermark of one, so a single running Fixup closes the gate.
fn gate_runtime_config() -> RuntimeConfig {
    RuntimeConfig::default()
        .with_maintenance(1, 2)
        .and_then(|config| config.with_attempts(1, 8))
        .and_then(|config| config.with_import_limits(1, 1))
        .expect("valid gate config")
}

/// Tiny partitions: at most four entries per leaf, so a handful of records
/// offers a split Fixup.
fn gate_index_config() -> IndexConfig {
    config()
        .with_partition_entries(1, 4)
        .expect("valid partition entries")
}

/// Runs one search purely as the relevant access that rediscovers
/// maintenance: visiting an over-threshold `Ready` partition or an in-flight
/// split re-offers it to the bounded Fixup queue (Demand-Driven Maintenance,
/// ADR 0006). The search outcome itself is irrelevant here.
async fn reoffer_maintenance(index: &Index<GatedBackend>) {
    let request = SearchRequest::new(Arc::from([0.0_f32, 0.0]), 1).expect("valid request");
    let _ = index.search(request).await;
}

/// Seeds one overfull `Ready` leaf without offering its Fixup.
///
/// The fifth insert commits with an unknown outcome. Its durable mutation is
/// visible, but the failed foreground operation cannot offer maintenance, so
/// a later search deterministically owns rediscovery.
async fn seed_cold_overfull(backend: &GatedBackend, index: &Index<GatedBackend>) {
    for value in 1..=4_u8 {
        index
            .insert(record(&rid(value), f32::from(value), 1))
            .await
            .expect("seed insert");
    }
    backend
        .inner
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("unknown-outcome fault");
    let error = index
        .insert(record(&rid(5), 5.0, 1))
        .await
        .expect_err("the applied insert reports an unknown outcome");
    assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    assert_record_present(index, &rid(5)).await;
}

/// Closes the backlog gate deterministically: searches rediscover the cold
/// overfull leaf and offer it, and the worker's next step commit is held
/// by `gate`, keeping one Fixup running — a backlog of one, exactly at the
/// watermark. `entered` is the expected held-commit count, including any
/// commits already held. The gate stays closed until `gate` releases the
/// commit.
///
/// The loop re-offers until the worker's commit is actually held because an
/// offer may still coalesce with a worker execution already in progress.
async fn hold_gate_closed(gate: &CommitGate, index: &Index<GatedBackend>, entered: usize) {
    gate.hold_next(1);
    let deadline = Instant::now() + WAIT_TIMEOUT;
    loop {
        reoffer_maintenance(index).await;
        if gate.entered.load(Ordering::SeqCst) >= entered {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the worker's step commit never reached the gate"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Asserts the pinned submit stays pending across many executor turns.
async fn assert_submit_pending<F>(submit: &mut std::pin::Pin<&mut F>, context: &str)
where
    F: std::future::Future,
    F::Output: std::fmt::Debug,
{
    tokio::select! {
        result = submit.as_mut() => panic!("{context}: {result:?}"),
        () = async {
            for _ in 0..64 {
                tokio::task::yield_now().await;
            }
        } => {}
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backlog_gate_blocks_and_releases_submit() {
    let (backend, gate, runtime, index) =
        setup_with_index(gate_runtime_config(), gate_index_config()).await;
    seed_cold_overfull(&backend, &index).await;
    let mut session = index.import_session(import_options(2)).expect("session");
    hold_gate_closed(&gate, &index, 1).await;

    // An empty batch does no storage work: it bypasses both the in-flight
    // slot and the backlog gate.
    let empty_token = session.submit(Vec::new()).await.expect("empty submit");
    assert_eq!(empty_token.get(), 1);

    // The backlog sits at the watermark, so a non-empty submit waits even
    // with every in-flight slot free.
    let token = {
        let mut submit = std::pin::pin!(session.submit(insert(&rid(9), 9.0, 1)));
        assert_submit_pending(
            &mut submit,
            "submit must wait while the backlog is at the watermark",
        )
        .await;

        // Draining the backlog below the watermark admits the waiting batch.
        gate.release();
        tokio::time::timeout(WAIT_TIMEOUT, submit.as_mut())
            .await
            .expect("the gate opens once the backlog drains")
            .expect("submit succeeds")
    };
    assert_eq!(token.get(), 2, "the gated submit issues the next token");

    let results = session.finish().await;
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].token, empty_token);
    assert!(
        results[0]
            .result
            .as_ref()
            .expect("empty batch succeeds")
            .is_empty()
    );
    assert_eq!(results[1].token, token);
    assert_eq!(
        results[1]
            .result
            .as_ref()
            .expect("batch commits")
            .as_slice(),
        &[MutationOutcome::Inserted]
    );
    assert_record_present(&index, &rid(9)).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backlog_gate_composes_with_in_flight_bound() {
    let (backend, gate, runtime, index) =
        setup_with_index(gate_runtime_config(), gate_index_config()).await;
    seed_cold_overfull(&backend, &index).await;
    let mut session = index.import_session(import_options(1)).expect("session");

    // Batch 1 admits while the gate is open and holds the session's only
    // in-flight slot; its commit is held before the boundary.
    gate.hold_next(1);
    let token1 = session
        .submit(insert(&rid(8), 8.0, 1))
        .await
        .expect("submit 1");
    gate.wait_until_entered(1).await;

    // The worker's step commit is held too: the slot is occupied and the
    // backlog sits at the watermark.
    hold_gate_closed(&gate, &index, 2).await;

    let token2 = {
        let mut submit2 = std::pin::pin!(session.submit(insert(&rid(9), 9.0, 1)));
        assert_submit_pending(&mut submit2, "submit must wait for the in-flight slot").await;

        // Releasing batch 1 frees the slot, but the closed gate alone still
        // blocks admission.
        gate.release_one();
        wait_until_present(&index, &rid(8)).await;
        assert_submit_pending(&mut submit2, "submit must wait for the backlog gate").await;

        // Releasing the worker's held step drains the backlog below the
        // watermark; with both bounds open the waiting batch admits and commits.
        gate.release_one();
        tokio::time::timeout(WAIT_TIMEOUT, submit2.as_mut())
            .await
            .expect("the gate opens once the backlog drains")
            .expect("submit 2 succeeds")
    };
    assert_eq!((token1.get(), token2.get()), (1, 2));

    let results = session.finish().await;
    assert_eq!(results.len(), 2);
    for entry in &results {
        assert_eq!(
            entry.result.as_ref().expect("batch commits").as_slice(),
            &[MutationOutcome::Inserted]
        );
    }
    assert_record_present(&index, &rid(9)).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_gated_submit_admits_nothing() {
    let (backend, gate, runtime, index) =
        setup_with_index(gate_runtime_config(), gate_index_config()).await;
    seed_cold_overfull(&backend, &index).await;
    let mut session = index.import_session(import_options(2)).expect("session");
    hold_gate_closed(&gate, &index, 1).await;

    {
        let mut gated = std::pin::pin!(session.submit(insert(&rid(9), 9.0, 1)));
        assert_submit_pending(&mut gated, "submit must wait for the backlog gate").await;
        // Dropping the waiting future cancels the wait: nothing admits and no
        // token or in-flight slot is consumed.
    }

    gate.release();
    let token = session
        .submit(insert(&rid(10), 10.0, 1))
        .await
        .expect("submit after the gate opens");
    assert_eq!(token.get(), 1, "the dropped gated submit consumed no token");
    let results = session.finish().await;
    assert_eq!(results.len(), 1);
    assert_record_absent(&index, &rid(9)).await;
    assert_record_present(&index, &rid(10)).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_releases_gated_submit() {
    let (backend, gate, runtime, index) =
        setup_with_index(gate_runtime_config(), gate_index_config()).await;
    seed_cold_overfull(&backend, &index).await;
    let mut session = index.import_session(import_options(2)).expect("session");

    // A batch admitted before the gate closes keeps its real result.
    let token1 = session
        .submit(insert(&rid(8), 8.0, 1))
        .await
        .expect("submit 1");
    wait_until_present(&index, &rid(8)).await;

    hold_gate_closed(&gate, &index, 1).await;
    let error = {
        let mut gated = std::pin::pin!(session.submit(insert(&rid(9), 9.0, 1)));
        assert_submit_pending(&mut gated, "submit must wait for the backlog gate").await;

        // Shutdown while gated: the waiting submit fails with RuntimeClosed and
        // admits nothing, even though shutdown cannot complete until the held
        // worker commit is released.
        let shutdown = tokio::spawn({
            let runtime = runtime.clone();
            async move { runtime.shutdown().await }
        });
        let error = tokio::select! {
            result = gated.as_mut() => {
                result.expect_err("a gated submit fails once the Runtime closes")
            }
            () = async {
                for _ in 0..10_000 {
                    tokio::task::yield_now().await;
                }
            } => panic!("shutdown must release a gated submit"),
        };
        gate.release();
        shutdown
            .await
            .expect("shutdown task")
            .expect("shutdown succeeds");
        error
    };
    assert_eq!(error.kind(), ErrorKind::RuntimeClosed);

    let results = session.finish().await;
    assert_eq!(results.len(), 1, "the gated submit admitted nothing");
    assert_eq!(results[0].token, token1);
    assert_eq!(
        results[0]
            .result
            .as_ref()
            .expect("an admitted batch keeps its real result")
            .as_slice(),
        &[MutationOutcome::Inserted]
    );
}
