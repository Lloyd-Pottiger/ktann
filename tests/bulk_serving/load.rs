//! Crash/retry and transaction fencing tests for the public serving loader.
use super::*;
use ktann::api::{BulkBuildJob, BulkBuildStatus, BulkLoadOptions, OperationOptions, Result};
use ktann::storage::backend::{
    AdmissionBudget, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation as RawMutation,
    ReadOps, ScanLimits, ScanPage,
};
use ktann::storage::keys::KeyRange;
use ktann::storage::values::BuildPhase;
use ktann_memory::test_support::{CommitFault, CommitOutcome, TestConfig};
use ktann_memory::{MemoryReadTxn, MemoryWriteTxn};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

fn load_options() -> BulkLoadOptions {
    BulkLoadOptions {
        max_mutations: 7,
        max_bytes: 2048,
    }
}
async fn reserve_fixture<B: Backend>(
    runtime: &Runtime<B>,
    dir: &Directory,
    count: u64,
) -> BulkBuildJob<B> {
    let input = InputSnapshot::create(
        &dir.0.join("source"),
        config(Metric::L2, true),
        1024 * 1024,
        (0..count).map(|id| Ok(record(id))),
    )
    .unwrap();
    runtime
        .start_bulk_build("bulk", &input, options().tree)
        .await
        .unwrap()
}
async fn fixture<B: Backend>(
    runtime: &Runtime<B>,
    memory: &MemoryBackend,
    dir: &Directory,
    count: u64,
) -> (BulkBuildJob<B>, ServingArtifact) {
    let job = reserve_fixture(runtime, dir, count).await;
    let input = InputSnapshot::open(
        &dir.0.join("source"),
        job.index_manifest().config().clone(),
        job.descriptor().input().clone(),
    )
    .unwrap();
    let (forest, _) = ForestArtifact::build(
        &dir.0.join("forest"),
        &input,
        *job.index_manifest().rotation_seed(),
        options(),
        1024 * 1024,
    )
    .unwrap();
    let (artifact, _) = ServingArtifact::build(
        &dir.0.join("serving"),
        &input,
        &forest,
        job.index_manifest(),
        serving_options(memory),
        4 * 1024 * 1024,
    )
    .unwrap();
    (job, artifact)
}
async fn checkpoint(
    memory: &MemoryBackend,
    job: &BulkBuildJob<impl Backend>,
) -> ktann::storage::values::BuildProgress {
    let mut txn = ReadLogicalTxn::bootstrap(memory.begin_read().await.unwrap());
    match txn
        .get(LogicalKey::BuildProgress(job.logical_index_id()))
        .await
        .unwrap()
    {
        Some(PersistentValue::BuildProgress(load)) => load,
        _ => panic!("missing checkpoint"),
    }
}
async fn assert_exact(memory: &MemoryBackend, artifact: &ServingArtifact) {
    let mut txn = memory.begin_read().await.unwrap();
    for entry in artifact.reader().unwrap() {
        let entry = entry.unwrap();
        assert_eq!(txn.get(entry.key).await.unwrap(), Some(entry.value));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claims_and_chunks_recover_unknown_outcomes_without_rewriting_committed_prefixes() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownApplied,
        CommitFault::UnknownNotApplied,
    ] {
        for at_claim in [false, true] {
            let dir = Directory::new();
            let memory = MemoryBackend::with_test_config(TestConfig::default());
            let runtime = runtime(memory.clone());
            let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
            let mut faults = vec![];
            if !at_claim {
                faults.push(CommitFault::Normal);
            }
            faults.push(fault);
            memory.set_fault_plan(faults).unwrap();
            let result = job.load_serving(&artifact, load_options()).await;
            if at_claim && fault != CommitFault::Abort {
                assert_eq!(result.unwrap_err().kind(), ErrorKind::CommitOutcomeUnknown);
                let reopened = runtime.open_bulk_build("bulk").await.unwrap();
                reopened
                    .load_serving(&artifact, load_options())
                    .await
                    .unwrap();
            } else {
                result.unwrap();
            }
            assert_eq!(
                job.status().await.unwrap(),
                BulkBuildStatus::Loaded {
                    entries: artifact.manifest().items()
                }
            );
            assert_exact(&memory, &artifact).await;
            let before = memory.history().len();
            job.load_serving(&artifact, load_options()).await.unwrap();
            assert_eq!(
                before,
                memory.history().len(),
                "complete task must not write again"
            );
            for h in memory.history().iter().skip(1) {
                assert!(h.mutations <= load_options().max_mutations);
                assert!(h.mutation_bytes <= load_options().max_bytes);
            }
            assert_eq!(
                runtime.open_index("bulk").await.unwrap_err().kind(),
                ErrorKind::IndexBuilding
            );
            runtime.shutdown().await.unwrap();
        }
    }
}

// Pause one chosen commit after its conflict reads, without sleeps or scheduler
// assumptions. All other transactions continue through the production adapter.
#[derive(Default)]
struct Gate {
    remaining: AtomicUsize,
    reached: Notify,
    release: Notify,
    write_slots: Option<Arc<tokio::sync::Semaphore>>,
    waiting_for_slot: Notify,
}
impl Gate {
    fn arm(&self, commit: usize) {
        self.remaining.store(commit, Ordering::SeqCst);
    }
    async fn wait(&self) {
        tokio::time::timeout(Duration::from_secs(10), self.reached.notified())
            .await
            .unwrap();
    }
}
#[derive(Clone)]
struct Gated {
    memory: MemoryBackend,
    gate: Arc<Gate>,
}
struct GatedTxn<'a> {
    inner: MemoryWriteTxn<'a>,
    gate: Arc<Gate>,
    _permit: Option<tokio::sync::OwnedSemaphorePermit>,
}
impl Backend for Gated {
    type ReadTxn<'a> = MemoryReadTxn;
    type WriteTxn<'a> = GatedTxn<'a>;
    fn hard_limits(&self) -> HardLimits {
        self.memory.hard_limits()
    }
    fn admission_budget(&self) -> AdmissionBudget {
        self.memory.admission_budget()
    }
    fn capabilities(&self) -> Capabilities {
        self.memory.capabilities()
    }
    async fn begin_read(&self) -> Result<MemoryReadTxn> {
        self.memory.begin_read().await
    }
    async fn begin_write(&self) -> Result<GatedTxn<'_>> {
        let permit = if let Some(slots) = &self.gate.write_slots {
            let permit = match slots.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => {
                    self.gate.waiting_for_slot.notify_one();
                    slots.clone().acquire_owned().await.unwrap()
                }
            };
            Some(permit)
        } else {
            None
        };
        Ok(GatedTxn {
            inner: self.memory.begin_write().await?,
            gate: self.gate.clone(),
            _permit: permit,
        })
    }
}
impl ReadOps for GatedTxn<'_> {
    async fn get(&mut self, k: Bytes) -> Result<Option<Bytes>> {
        self.inner.get(k).await
    }
    async fn batch_get(&mut self, k: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.inner.batch_get(k).await
    }
    async fn scan(&mut self, r: &KeyRange, l: ScanLimits) -> Result<ScanPage> {
        self.inner.scan(r, l).await
    }
    async fn batch_scan(&mut self, r: &[KeyRange], l: ScanLimits) -> Result<Vec<ScanPage>> {
        self.inner.batch_scan(r, l).await
    }
}
impl WriteTxn for GatedTxn<'_> {
    async fn get_for_update(&mut self, k: Bytes) -> Result<Option<Bytes>> {
        self.inner.get_for_update(k).await
    }
    async fn batch_get_for_update(&mut self, k: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.inner.batch_get_for_update(k).await
    }
    async fn put(&mut self, k: Bytes, v: Bytes) -> Result<()> {
        self.inner.put(k, v).await
    }
    async fn insert(&mut self, k: Bytes, v: Bytes) -> Result<InsertOutcome> {
        self.inner.insert(k, v).await
    }
    async fn delete(&mut self, k: Bytes) -> Result<()> {
        self.inner.delete(k).await
    }
    async fn batch_mutate(&mut self, m: Vec<RawMutation>) -> Result<()> {
        self.inner.batch_mutate(m).await
    }
    async fn clear_range(&mut self, r: &KeyRange) -> Result<()> {
        self.inner.clear_range(r).await
    }
    async fn commit_with(self, start: CommitStart) -> Result<()> {
        if self
            .gate
            .remaining
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_sub(1))
            == Ok(1)
        {
            self.gate.reached.notify_one();
            self.gate.release.notified().await;
        }
        self.inner.commit_with(start).await
    }
    async fn rollback(self) {
        self.inner.rollback().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn takeover_and_abort_fence_an_already_prepared_old_chunk() {
    for abort in [false, true] {
        let dir = Directory::new();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let gate = Arc::new(Gate::default());
        let runtime = Runtime::new(
            Gated {
                memory: memory.clone(),
                gate: gate.clone(),
            },
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
        gate.arm(2);
        let old = job.clone();
        let input = artifact.clone();
        let task = tokio::spawn(async move { old.load_serving(&input, load_options()).await });
        gate.wait().await;
        assert_eq!(checkpoint(&memory, &job).await.entries(), 0);
        if abort {
            job.abort().await.unwrap();
        } else {
            job.load_serving(&artifact, load_options()).await.unwrap();
            assert_eq!(checkpoint(&memory, &job).await.epoch(), 2);
        }
        gate.release.notify_one();
        let error = task.await.unwrap().unwrap_err();
        assert_eq!(
            error.kind(),
            if abort {
                ErrorKind::IndexNotFound
            } else {
                ErrorKind::BulkBuildSuperseded
            }
        );
        assert!(
            memory
                .history()
                .iter()
                .any(|h| h.outcome == CommitOutcome::Aborted)
        );
        if abort {
            assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Aborted);
            assert!(dir.0.join("serving/data.bin").exists());
        } else {
            assert_exact(&memory, &artifact).await;
        }
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_keeps_atomic_progress_and_reopen_resumes_without_prefix_writes() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
    gate.arm(3);
    let worker = job.clone();
    let input = artifact.clone();
    let cancellation = CancellationToken::new();
    let control = OperationOptions::default().with_cancellation(cancellation.clone());
    let task = tokio::spawn(async move {
        worker
            .load_serving_with_control(&input, load_options(), control)
            .await
    });
    gate.wait().await;
    assert_eq!(checkpoint(&memory, &job).await.entries(), 6);
    cancellation.cancel();
    gate.release.notify_one();
    assert_eq!(
        task.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    let progress = checkpoint(&memory, &job).await;
    assert!(progress.entries() >= 6 && !matches!(progress.phase(), BuildPhase::Loaded));
    let before = memory.history().len();
    let resumed = runtime.open_bulk_build("bulk").await.unwrap();
    resumed
        .load_serving(&artifact, load_options())
        .await
        .unwrap();
    let delta = memory.history();
    let data_writes: usize = delta[before..]
        .iter()
        .filter(|h| h.outcome == CommitOutcome::Committed)
        .map(|h| h.mutations - 1)
        .sum();
    assert_eq!(
        data_writes as u64,
        artifact.manifest().items() - progress.entries()
    );
    assert_exact(&memory, &artifact).await;
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_entry_count_remains_loading_until_eof_checkpoint_commits() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, artifact) = fixture(&runtime, &memory, &dir, 1).await;
    let options = BulkLoadOptions {
        max_mutations: 2,
        ..load_options()
    };
    // One claim, one commit per entry, then the separate EOF checkpoint.
    gate.arm(artifact.manifest().items() as usize + 2);
    let worker = job.clone();
    let input = artifact.clone();
    let cancellation = CancellationToken::new();
    let control = OperationOptions::default().with_cancellation(cancellation.clone());
    let task = tokio::spawn(async move {
        worker
            .load_serving_with_control(&input, options, control)
            .await
    });
    gate.wait().await;
    let total = artifact.manifest().items();
    assert_eq!(
        job.status().await.unwrap(),
        BulkBuildStatus::Loading {
            loaded_entries: total,
            total_entries: total,
        }
    );
    assert!(matches!(
        checkpoint(&memory, &job).await.phase(),
        BuildPhase::Loading { .. }
    ));
    // Cancellation cannot undo an in-flight commit. Force a definite abort,
    // then cancellation prevents the retry from completing the EOF transition.
    memory.set_fault_plan(vec![CommitFault::Abort]).unwrap();
    cancellation.cancel();
    gate.release.notify_one();
    assert_eq!(
        task.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    let before = memory.history().len();
    runtime
        .open_bulk_build("bulk")
        .await
        .unwrap()
        .load_serving(&artifact, options)
        .await
        .unwrap();
    assert_eq!(
        job.status().await.unwrap(),
        BulkBuildStatus::Loaded { entries: total }
    );
    // Recovery only claims and completes the checkpoint, without rewriting data.
    assert!(
        memory.history()[before..]
            .iter()
            .all(|entry| entry.mutations == 1)
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_load_and_checkpoint_codec_are_canonical_and_fail_closed() {
    let dir = Directory::new();
    let memory = MemoryBackend::new();
    let runtime = runtime(memory.clone());
    let (job, artifact) = fixture(&runtime, &memory, &dir, 0).await;
    job.load_serving(&artifact, load_options()).await.unwrap();
    let load = checkpoint(&memory, &job).await;
    assert!(matches!(load.phase(), BuildPhase::Loaded));
    assert_eq!(load.entries(), 0);
    let key = LogicalKey::BuildProgress(job.logical_index_id());
    let value = ValueCodec::bootstrap()
        .encode(&PersistentValue::BuildProgress(load.clone()))
        .unwrap();
    let mut expected = vec![14, 0, 0, 0, 89];
    expected.extend_from_slice(&artifact.manifest().encode());
    expected.extend_from_slice(&1_u64.to_be_bytes());
    expected.push(1); // Loaded has no duplicate count or digest.
    assert_eq!(value, expected);
    assert_eq!(
        keys::build_progress_key(job.logical_index_id()),
        [
            vec![1],
            job.logical_index_id().get().to_be_bytes().to_vec(),
            vec![6]
        ]
        .concat()
    );
    for (offset, replacement) in [(101, 0), (102, 4), (13, 2)] {
        let mut invalid = value.clone();
        invalid[offset] = replacement;
        assert_eq!(
            ValueCodec::bootstrap()
                .decode(&key, Bytes::from(invalid))
                .unwrap_err()
                .kind(),
            ErrorKind::Corruption
        );
    }
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn corrupt_input_never_completes_and_repaired_identical_bytes_resume() {
    let dir = Directory::new();
    let memory = MemoryBackend::new();
    let runtime = runtime(memory.clone());
    let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
    let path = dir.0.join("serving/data.bin");
    let original = fs::read(&path).unwrap();
    let mut corrupt = original.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    fs::write(&path, corrupt).unwrap();
    assert_eq!(
        job.load_serving(&artifact, load_options())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Corruption
    );
    let partial = checkpoint(&memory, &job).await;
    assert!(!matches!(partial.phase(), BuildPhase::Loaded));
    assert!(partial.entries() > 0);
    assert_eq!(
        runtime.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexBuilding
    );
    fs::write(path, original).unwrap();
    job.load_serving(&artifact, load_options()).await.unwrap();
    assert_exact(&memory, &artifact).await;
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admission_and_artifact_identity_are_enforced_before_data_writes() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let runtime = runtime(memory.clone());
    let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
    for limit in [0, 152] {
        let err = job
            .load_serving(
                &artifact,
                BulkLoadOptions {
                    max_mutations: 7,
                    max_bytes: limit,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(
            err.kind(),
            ErrorKind::InvalidArgument | ErrorKind::LimitExceeded
        ));
        assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Preparing);
    }
    assert_eq!(
        job.load_serving(
            &artifact,
            BulkLoadOptions {
                max_mutations: 7,
                max_bytes: 153
            }
        )
        .await
        .unwrap_err()
        .kind(),
        ErrorKind::LimitExceeded
    );
    assert_eq!(checkpoint(&memory, &job).await.entries(), 0);
    // Changing preparation parameters produces another sealed artifact identity;
    // the loader must retain the identity registered by the first attempt.
    let source = InputSnapshot::open(
        &dir.0.join("source"),
        job.index_manifest().config().clone(),
        job.descriptor().input().clone(),
    )
    .unwrap();
    let (forest, _) = ForestArtifact::build(
        &dir.0.join("other-forest"),
        &source,
        *job.index_manifest().rotation_seed(),
        options(),
        1024 * 1024,
    )
    .unwrap();
    let mut changed = serving_options(&memory);
    changed.memory_bytes += 1024 * 1024;
    let (other, _) = ServingArtifact::build(
        &dir.0.join("other-serving"),
        &source,
        &forest,
        job.index_manifest(),
        changed,
        4 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(
        job.load_serving(&other, load_options())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(checkpoint(&memory, &job).await.epoch(), 1);
    job.load_serving(&artifact, load_options()).await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn physical_namespace_overhead_and_checkpoint_share_the_chunk_budget() {
    let mut config = TestConfig::default();
    config.admission_budget.mutation_key_overhead_bytes = 100;
    let memory = MemoryBackend::with_test_config(config);
    let runtime = runtime(memory.clone());
    let dir = Directory::new();
    let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
    let before = memory.history().len();
    job.load_serving(
        &artifact,
        BulkLoadOptions {
            max_mutations: 7,
            max_bytes: 640,
        },
    )
    .await
    .unwrap();
    for h in &memory.history()[before..] {
        assert!(h.mutations <= 7 && h.mutation_bytes <= 640);
    }
    assert_exact(&memory, &artifact).await;
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_checks_the_exact_previously_committed_prefix_before_skipping_it() {
    use sha2::{Digest, Sha256};
    let dir = Directory::new();
    let memory = MemoryBackend::new();
    let runtime = runtime(memory.clone());
    let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
    let path = dir.0.join("serving/data.bin");
    let original = fs::read(&path).unwrap();
    let mut forged = original.clone();
    let length = u32::from_be_bytes(forged[41..45].try_into().unwrap()) as usize;
    let end = 45 + length;
    let key_length = u32::from_be_bytes(forged[45..49].try_into().unwrap()) as usize;
    let value = 49 + key_length;
    assert_eq!(forged[value], 4); // Vector Record, eight-byte ID and dimension.
    forged[value + 15..value + 19].copy_from_slice(&2_f32.to_be_bytes());
    let digest = Sha256::digest(&forged[45..end]);
    forged[end..end + 32].copy_from_slice(&digest);
    fs::write(&path, forged).unwrap();
    // Every frame is well formed, but final file identity must fail.
    assert_eq!(
        job.load_serving(&artifact, load_options())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Corruption
    );
    let before = checkpoint(&memory, &job).await;
    assert!(before.entries() > 0 && !matches!(before.phase(), BuildPhase::Loaded));
    fs::write(path, original).unwrap();
    // Repairing the file cannot bless a different prefix already in storage.
    assert_eq!(
        job.load_serving(&artifact, load_options())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Corruption
    );
    let after = checkpoint(&memory, &job).await;
    assert_eq!(before.entries(), after.entries());
    assert_eq!(before.prefix_sha256(), after.prefix_sha256());
    assert!(!matches!(after.phase(), BuildPhase::Loaded));
    job.abort().await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[path = "complete.rs"]
mod complete;

#[path = "scheduler.rs"]
mod scheduler;
