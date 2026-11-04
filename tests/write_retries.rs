//! Whole-attempt retries include backend admission and precommit reads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use ktann::api::{
    Error, ErrorKind, GetOptions, IndexConfig, Metric, OperationOptions, PartitionKey, Record,
    Result, RuntimeConfig,
};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation,
    ReadOps, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::{KeyRange, LogicalKey, TreeKey};
use ktann::storage::{ReadLogicalTxn, values::PersistentValue};
use ktann_memory::test_support::{CommitFault, TestConfig};
use ktann_memory::{MemoryBackend, MemoryWriteTxn};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    Open,
    Read,
    Put,
}

/// Injects a bounded number of errors at one backend seam; real Memory
/// transactions retain snapshot, atomicity, and cleanup behavior.
#[derive(Clone)]
struct FaultBackend {
    memory: MemoryBackend,
    fault: Arc<Mutex<Option<(Stage, usize, ErrorKind)>>>,
    opened: Arc<AtomicUsize>,
    cancel: Arc<Mutex<Option<CancellationToken>>>,
}

impl FaultBackend {
    fn new() -> Self {
        Self {
            memory: MemoryBackend::with_test_config(TestConfig {
                max_active_transactions: 1,
                ..TestConfig::default()
            }),
            fault: Arc::new(Mutex::new(None)),
            opened: Arc::new(AtomicUsize::new(0)),
            cancel: Arc::new(Mutex::new(None)),
        }
    }

    fn arm(&self, stage: Stage, count: usize, kind: ErrorKind) {
        *self.fault.lock().unwrap() = Some((stage, count, kind));
        self.opened.store(0, Ordering::SeqCst);
    }

    fn check(&self, stage: Stage) -> Result<()> {
        let mut fault = self.fault.lock().unwrap();
        if let Some((target, remaining, kind)) = fault.as_mut()
            && *target == stage
            && *remaining > 0
        {
            *remaining -= 1;
            if let Some(cancel) = self.cancel.lock().unwrap().take() {
                cancel.cancel();
            }
            return Err(Error::new(*kind));
        }
        Ok(())
    }

    fn assert_attempts(&self, expected: usize) {
        assert_eq!(self.opened.load(Ordering::SeqCst), expected);
        assert_eq!(
            self.memory.active_transactions(),
            0,
            "every attempt releases its snapshot"
        );
    }
}

impl Backend for FaultBackend {
    type ReadTxn<'a> = <MemoryBackend as Backend>::ReadTxn<'a>;
    type WriteTxn<'a> = FaultTxn<'a>;
    fn hard_limits(&self) -> HardLimits {
        self.memory.hard_limits()
    }
    fn admission_budget(&self) -> AdmissionBudget {
        self.memory.admission_budget()
    }
    fn capabilities(&self) -> Capabilities {
        self.memory.capabilities()
    }
    async fn begin_read(&self) -> Result<Self::ReadTxn<'_>> {
        self.memory.begin_read().await
    }
    async fn begin_write(&self) -> Result<Self::WriteTxn<'_>> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        self.check(Stage::Open)?;
        Ok(FaultTxn {
            inner: self.memory.begin_write().await?,
            backend: self,
        })
    }
}

struct FaultTxn<'a> {
    inner: MemoryWriteTxn<'a>,
    backend: &'a FaultBackend,
}

impl ReadOps for FaultTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.inner.get(key).await
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.inner.batch_get(keys).await
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        self.inner.scan(range, limits).await
    }

    async fn batch_scan(
        &mut self,
        ranges: &[KeyRange],
        limits: ScanLimits,
    ) -> Result<Vec<ScanPage>> {
        self.inner.batch_scan(ranges, limits).await
    }
}

impl WriteTxn for FaultTxn<'_> {
    async fn get_for_update(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.backend.check(Stage::Read)?;
        self.inner.get_for_update(key).await
    }

    async fn batch_get_for_update(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.backend.check(Stage::Read)?;
        self.inner.batch_get_for_update(keys).await
    }

    async fn put(&mut self, key: Bytes, value: Bytes) -> Result<()> {
        self.backend.check(Stage::Put)?;
        self.inner.put(key, value).await
    }

    async fn insert(&mut self, key: Bytes, value: Bytes) -> Result<InsertOutcome> {
        self.inner.insert(key, value).await
    }

    async fn delete(&mut self, key: Bytes) -> Result<()> {
        self.backend.check(Stage::Put)?;
        self.inner.delete(key).await
    }

    async fn batch_mutate(&mut self, mutations: Vec<Mutation>) -> Result<()> {
        self.backend.check(Stage::Put)?;
        self.inner.batch_mutate(mutations).await
    }

    async fn clear_range(&mut self, range: &KeyRange) -> Result<()> {
        self.backend.check(Stage::Put)?;
        self.inner.clear_range(range).await
    }

    async fn commit_with(self, start: CommitStart) -> Result<()> {
        self.inner.commit_with(start).await
    }

    async fn rollback(self) {
        self.inner.rollback().await;
    }
}

fn config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2).unwrap()
}

fn runtime(backend: &FaultBackend) -> Runtime<FaultBackend> {
    Runtime::new(
        backend.clone(),
        RuntimeConfig::default()
            .with_maintenance(0, 1)
            .unwrap()
            .with_attempts(3, 3)
            .unwrap()
            .with_retry_backoff(Duration::from_nanos(1), Duration::from_nanos(1))
            .unwrap(),
    )
    .unwrap()
}

fn record() -> Record {
    Record::new(Bytes::from_static(b"r"), vec![1.0, 2.0], vec![]).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_retries_open_and_manifest_aborts() {
    for stage in [Stage::Open, Stage::Read] {
        let backend = FaultBackend::new();
        let runtime = runtime(&backend);
        let index = runtime.create_index("index", config()).await.unwrap();
        backend.arm(stage, 1, ErrorKind::RetryableAbort);
        index.insert(record()).await.unwrap();
        backend.assert_attempts(2);
        assert!(
            index
                .get(Bytes::from_static(b"r"), GetOptions::default())
                .await
                .unwrap()
                .is_some()
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_precommit_aborts_exhaust_and_other_errors_are_not_replayed() {
    for stage in [Stage::Open, Stage::Read] {
        for (kind, expected, attempts) in [
            (ErrorKind::RetryableAbort, ErrorKind::ContentionExhausted, 3),
            (ErrorKind::Backend, ErrorKind::Backend, 1),
            (ErrorKind::Corruption, ErrorKind::Corruption, 1),
        ] {
            let backend = FaultBackend::new();
            let runtime = runtime(&backend);
            let index = runtime.create_index("index", config()).await.unwrap();
            backend.arm(stage, 10, kind);
            assert_eq!(index.insert(record()).await.unwrap_err().kind(), expected);
            backend.assert_attempts(attempts);
            assert!(
                index
                    .get(Bytes::from_static(b"r"), GetOptions::default())
                    .await
                    .unwrap()
                    .is_none()
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_commit_after_precommit_retry_is_not_replayed() {
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        let backend = FaultBackend::new();
        let runtime = runtime(&backend);
        let index = runtime.create_index("index", config()).await.unwrap();
        backend.arm(Stage::Read, 1, ErrorKind::RetryableAbort);
        backend.memory.push_fault(fault).unwrap();
        assert_eq!(
            index.insert(record()).await.unwrap_err().kind(),
            ErrorKind::CommitOutcomeUnknown
        );
        backend.assert_attempts(2);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_and_deadline_interrupt_precommit_backoff() {
    for stage in [Stage::Open, Stage::Read] {
        for deadline in [false, true] {
            let backend = FaultBackend::new();
            let runtime = Runtime::new(
                backend.clone(),
                RuntimeConfig::default()
                    .with_maintenance(0, 1)
                    .unwrap()
                    .with_retry_backoff(Duration::from_secs(60), Duration::from_secs(60))
                    .unwrap(),
            )
            .unwrap();
            let index = runtime.create_index("index", config()).await.unwrap();
            let cancel = CancellationToken::new();
            let options = if deadline {
                OperationOptions::default()
                    .with_deadline(std::time::Instant::now() + Duration::from_millis(50))
            } else {
                *backend.cancel.lock().unwrap() = Some(cancel.clone());
                OperationOptions::default().with_cancellation(cancel)
            };
            backend.arm(stage, 10, ErrorKind::RetryableAbort);
            let result = tokio::time::timeout(
                Duration::from_secs(2),
                index.insert_with_control(record(), options),
            )
            .await
            .unwrap();
            assert_eq!(
                result.unwrap_err().kind(),
                if deadline {
                    ErrorKind::DeadlineExceeded
                } else {
                    ErrorKind::Cancelled
                }
            );
            // The caller may return before the owned task observes cancellation.
            runtime.shutdown().await.unwrap();
            backend.assert_attempts(1);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn maintenance_retries_manifest_and_open_with_its_own_budget() {
    for stage in [Stage::Open, Stage::Read] {
        for aborts in [1, 10] {
            let backend = FaultBackend::new();
            let runtime = runtime(&backend);
            let index = runtime.create_index("index", config()).await.unwrap();
            let raw = backend.begin_read().await.unwrap();
            let mut txn = ReadLogicalTxn::bootstrap(raw);
            let Some(PersistentValue::IndexManifest(manifest)) = txn
                .get(LogicalKey::Manifest(index.logical_index_id()))
                .await
                .unwrap()
            else {
                panic!("manifest");
            };
            drop(txn);
            backend.arm(stage, aborts, ErrorKind::RetryableAbort);
            let retry = RetryPolicy::for_fixup(
                &RuntimeConfig::default()
                    .with_attempts(2, 3)
                    .unwrap()
                    .with_retry_backoff(Duration::from_nanos(1), Duration::from_nanos(1))
                    .unwrap(),
            );
            let result = ktann::maintenance::split::begin_split(
                &backend,
                &manifest,
                &TreeKey::encode(&[], &[]).unwrap(),
                PartitionKey::new(1).unwrap(),
                0,
                &retry,
            )
            .await;
            if aborts == 1 {
                assert!(result.is_ok());
            } else {
                assert_eq!(result.unwrap_err().kind(), ErrorKind::ContentionExhausted);
            }
            backend.assert_attempts(2);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lifecycle_retries_complete_precommit_attempts() {
    for stage in [Stage::Open, Stage::Read, Stage::Put] {
        let backend = FaultBackend::new();
        let runtime = runtime(&backend);
        backend.arm(stage, 1, ErrorKind::RetryableAbort);
        let index = runtime.create_index("index", config()).await.unwrap();
        assert_eq!(
            index.logical_index_id().get(),
            1,
            "failed create allocates no durable identity"
        );
        backend.assert_attempts(2);
        backend.arm(stage, 1, ErrorKind::RetryableAbort);
        runtime.drop_index("index").await.unwrap();
        assert_eq!(
            runtime.open_index("index").await.unwrap_err().kind(),
            ErrorKind::IndexNotFound
        );
        assert_eq!(backend.memory.active_transactions(), 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lifecycle_precommit_errors_are_bounded_and_classified() {
    for stage in [Stage::Open, Stage::Read, Stage::Put] {
        for (kind, expected, attempts) in [
            (ErrorKind::RetryableAbort, ErrorKind::ContentionExhausted, 3),
            (ErrorKind::Backend, ErrorKind::Backend, 1),
        ] {
            let backend = FaultBackend::new();
            let runtime = runtime(&backend);
            backend.arm(stage, 10, kind);
            assert_eq!(
                runtime
                    .create_index("index", config())
                    .await
                    .unwrap_err()
                    .kind(),
                expected
            );
            backend.assert_attempts(attempts);
            backend.arm(stage, 0, kind);
            runtime.create_index("index", config()).await.unwrap();
            backend.arm(stage, 10, kind);
            assert_eq!(
                runtime.drop_index("index").await.unwrap_err().kind(),
                expected
            );
            backend.assert_attempts(attempts);
        }
    }
}
