//! Transaction fencing, chunk budget bounds, and recovery tests for Bulk Build workers.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::*;
use ktann::api::{BulkBuildJob, BulkBuildStatus, BulkLoadOptions, OperationOptions, Result};
use ktann::storage::backend::{
    AdmissionBudget, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation as RawMutation,
    ScanLimits, ScanPage,
};
use ktann::storage::keys::KeyRange;
use ktann_memory::test_support::{CommitFault, CommitOutcome, TestConfig};
use ktann_memory::{MemoryReadTxn, MemoryWriteTxn};

pub(super) fn load_options() -> BulkLoadOptions {
    BulkLoadOptions {
        max_mutations: 7,
        max_bytes: 2048,
    }
}

pub(super) async fn reservation<B: Backend>(
    runtime: &Runtime<B>,
    dir: &tempfile::TempDir,
    count: u64,
) -> (BulkBuildJob<B>, InputSnapshot) {
    reservation_config(runtime, dir, count, config(Metric::L2, true)).await
}

pub(super) async fn reservation_config<B: Backend>(
    runtime: &Runtime<B>,
    dir: &tempfile::TempDir,
    count: u64,
    config: IndexConfig,
) -> (BulkBuildJob<B>, InputSnapshot) {
    let input = InputSnapshot::create(
        &dir.path().join("source"),
        config,
        1024 * 1024,
        (0..count).map(|id| Ok(record(id))),
    )
    .unwrap();
    let job = runtime
        .start_bulk_build("bulk", &input, construction_options())
        .await
        .unwrap();
    (job, input)
}

// Pause one chosen commit or snapshot read without sleeps or scheduler assumptions.
#[derive(Default)]
pub(super) struct Gate {
    pub(super) remaining: AtomicUsize,
    pub(super) read_remaining: AtomicUsize,
    pub(super) reached: Notify,
    pub(super) release: Notify,
    pub(super) write_slots: Option<Arc<tokio::sync::Semaphore>>,
    pub(super) waiting_for_slot: Notify,
}

impl Gate {
    pub(super) fn arm(&self, commit: usize) {
        self.remaining.store(commit, Ordering::SeqCst);
    }

    #[allow(dead_code)]
    pub(super) fn arm_read(&self, read: usize) {
        self.read_remaining.store(read, Ordering::SeqCst);
    }

    pub(super) async fn wait(&self) {
        tokio::time::timeout(Duration::from_secs(10), self.reached.notified())
            .await
            .unwrap();
    }
}

#[derive(Clone)]
pub(super) struct Gated {
    pub(super) memory: MemoryBackend,
    pub(super) gate: Arc<Gate>,
}

pub(super) struct GatedTxn<'a> {
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
        if self
            .gate
            .read_remaining
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_sub(1))
            == Ok(1)
        {
            self.gate.reached.notify_one();
            self.gate.release.notified().await;
        }
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
async fn claims_and_chunks_respect_budgets_and_recover_unknown_outcomes() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownApplied,
        CommitFault::UnknownNotApplied,
    ] {
        for at_claim in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let memory = MemoryBackend::with_test_config(TestConfig::default());
            let rt = runtime(memory.clone());
            let (job, _) = reservation(&rt, &dir, 19).await;
            let work = dir.path().join("work");
            fs::create_dir(&work).unwrap();
            let mut opts = BulkWorkerOptions::new(work);
            opts.parallelism = 1;
            opts.load = BulkLoadOptions {
                max_mutations: 24,
                max_bytes: 4096,
            };

            let mut faults = vec![];
            if !at_claim {
                faults.push(CommitFault::Normal);
            }
            faults.push(fault);
            memory.set_fault_plan(faults).unwrap();
            let result = job.run_worker(opts.clone()).await;
            if at_claim && fault != CommitFault::Abort {
                assert_eq!(result.unwrap_err().kind(), ErrorKind::CommitOutcomeUnknown);
                let reopened = rt.open_bulk_build("bulk").await.unwrap();
                reopened.run_worker(opts.clone()).await.unwrap();
            } else {
                result.unwrap();
            }
            assert!(matches!(
                job.status().await.unwrap(),
                BulkBuildStatus::Loaded { .. }
            ));

            // Calling run_worker again on a Loaded build writes no new backend mutations.
            let before = memory.history().len();
            job.run_worker(opts.clone()).await.unwrap();
            assert_eq!(before, memory.history().len());

            // Every committed transaction respects `opts.load` mutation and byte bounds.
            for h in memory
                .history()
                .iter()
                .filter(|h| h.outcome == CommitOutcome::Committed)
            {
                assert!(h.mutations <= opts.load.max_mutations);
                assert!(h.mutation_bytes <= opts.load.max_bytes);
            }
            let index = job.publish().await.unwrap();
            let verified = index.verify(VerifyOptions::default()).await.unwrap();
            assert!(verified.complete && verified.issues.is_empty());
            assert_eq!(verified.objects.vector_records, 19);
            rt.shutdown().await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_mid_load_recovers_cleanly_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let rt = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = reservation(&rt, &dir, 19).await;
    let work = dir.path().join("work");
    fs::create_dir(&work).unwrap();
    let mut opts = BulkWorkerOptions::new(work);
    opts.parallelism = 1;
    opts.load = load_options();

    gate.arm(3);
    let cancel = CancellationToken::new();
    let ctrl = OperationOptions::default().with_cancellation(cancel.clone());
    let worker_job = job.clone();
    let opts_c = opts.clone();
    let pending =
        tokio::spawn(async move { worker_job.run_worker_with_control(opts_c, ctrl).await });
    gate.wait().await;
    cancel.cancel();
    gate.release.notify_one();
    assert_eq!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );

    let reopened = rt.open_bulk_build("bulk").await.unwrap();
    let index = reopened.complete(opts).await.unwrap();
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty());
    assert_eq!(report.objects.vector_records, 19);
    rt.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn corrupt_input_segment_fails_closed_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::new();
    let rt = runtime(memory);
    let (job, _) = reservation(&rt, &dir, 19).await;
    let seg_path = dir.path().join("source/seg-00000000.bin");
    let orig = fs::read(&seg_path).unwrap();
    let mut damaged = orig.clone();
    *damaged.last_mut().unwrap() ^= 0xff;
    fs::write(&seg_path, damaged).unwrap();

    let work = dir.path().join("work");
    fs::create_dir(&work).unwrap();
    let opts = BulkWorkerOptions::new(work);
    assert_eq!(
        job.run_worker(opts).await.unwrap_err().kind(),
        ErrorKind::Corruption
    );
    assert_eq!(
        job.status().await.unwrap(),
        BulkBuildStatus::Failed {
            kind: ErrorKind::Corruption
        }
    );
    job.abort().await.unwrap();
    rt.shutdown().await.unwrap();
}

#[path = "complete.rs"]
mod complete;

#[path = "scheduler.rs"]
mod scheduler;
