//! Bulk construction publication, fencing and ordinary mutation contracts.
use bytes::Bytes;
use ktann::api::{
    BulkBuildOptions, DataType, ErrorKind, FieldId, FieldSchema, GetOptions, IndexConfig, Metric,
    OperationOptions, Record, SearchRequest, Value, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::ReadLogicalTxn;
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome,
    Mutation as StorageMutation, ReadOps, ReadTxn, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::{KeyRange, LogicalKey};
use ktann::storage::values::{IndexLifecycle, PersistentValue};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;
use support::{
    CommitFault, DeterministicBackend, DeterministicConfig, DeterministicReadTxn,
    DeterministicWriteTxn, SharedBackend, manual_maintenance_config,
};
use tokio_util::sync::CancellationToken;
#[allow(dead_code)]
mod support;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
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
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |pending| {
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

    /// Releases exactly the earliest entered held commit.
    fn release_one(&self) {
        self.released.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct GatedBackend {
    inner: Arc<DeterministicBackend>,
    gate: Arc<CommitGate>,
    read_gate: Arc<CommitGate>,
    scan_limit: usize,
}

impl GatedBackend {
    fn new(inner: Arc<DeterministicBackend>, gate: Arc<CommitGate>) -> Self {
        Self {
            inner,
            gate,
            read_gate: Arc::new(CommitGate::default()),
            scan_limit: usize::MAX,
        }
    }
}

impl Backend for GatedBackend {
    type ReadTxn<'backend> = GatedReadTxn<'backend>;

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
        Ok(GatedReadTxn {
            inner: self.inner.begin_read().await?,
            gate: Arc::clone(&self.read_gate),
            remaining_scans: self.scan_limit,
        })
    }

    async fn begin_write(&self) -> ktann::api::Result<Self::WriteTxn<'_>> {
        Ok(GatedWriteTxn {
            inner: self.inner.begin_write().await?,
            gate: Arc::clone(&self.gate),
        })
    }
}

/// A deterministic short-lived snapshot; only the scan budget is artificial.
struct GatedReadTxn<'backend> {
    inner: DeterministicReadTxn<'backend>,
    gate: Arc<CommitGate>,
    remaining_scans: usize,
}
impl GatedReadTxn<'_> {
    async fn admit_scans(&mut self, count: usize) -> ktann::api::Result<()> {
        self.gate.maybe_wait().await;
        self.remaining_scans = self
            .remaining_scans
            .checked_sub(count)
            .ok_or_else(|| ktann::api::Error::new(ErrorKind::Backend))?;
        Ok(())
    }
}
impl ReadOps for GatedReadTxn<'_> {
    async fn get(&mut self, key: Bytes) -> ktann::api::Result<Option<Bytes>> {
        self.inner.get(key).await
    }
    async fn batch_get(&mut self, keys: Vec<Bytes>) -> ktann::api::Result<Vec<Option<Bytes>>> {
        self.inner.batch_get(keys).await
    }
    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> ktann::api::Result<ScanPage> {
        self.admit_scans(1).await?;
        self.inner.scan(range, limits).await
    }
    async fn batch_scan(
        &mut self,
        ranges: &[KeyRange],
        limits: ScanLimits,
    ) -> ktann::api::Result<Vec<ScanPage>> {
        self.admit_scans(ranges.len()).await?;
        self.inner.batch_scan(ranges, limits).await
    }
}
impl ReadTxn for GatedReadTxn<'_> {}

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
        ranges: &[KeyRange],
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
    IndexConfig::new(2, Metric::Cosine)
        .unwrap()
        .with_partition_entries(2, 4)
        .unwrap()
        .with_fields(vec![FieldSchema::new("bucket", DataType::I64).unwrap()])
        .unwrap()
        .with_tree_key_fields(vec![FieldId(0)])
        .unwrap()
}
fn records(count: usize) -> Vec<Record> {
    (0..count)
        .map(|position| {
            Record::new(
                Bytes::from(format!("r{position:03}")),
                vec![1.0, (position as f32 - count as f32 / 2.0) / 4.0],
                vec![Value::I64((position % 2) as i64)],
            )
            .unwrap()
            .with_payload(Bytes::from(format!("p{position}")))
            .unwrap()
        })
        .collect()
}
fn options(rounds: usize) -> BulkBuildOptions {
    BulkBuildOptions::new(1 << 20)
        .unwrap()
        .with_refinement_rounds(rounds)
        .unwrap()
}
fn make_runtime(backend: SharedBackend) -> Runtime<SharedBackend> {
    Runtime::new(backend, manual_maintenance_config()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refined_build_preserves_exact_membership_fields_payload_and_mutations() {
    for rounds in [0, 2, 5] {
        let backend = SharedBackend::new(DeterministicBackend::default());
        let runtime = make_runtime(backend);
        let source = records(53);
        let index = runtime
            .build_index(
                "bulk",
                config(),
                source.clone(),
                options(rounds),
                OperationOptions::default(),
            )
            .await
            .unwrap();
        let report = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(report.complete && report.issues.is_empty(), "{report:?}");
        assert!(
            report
                .topology
                .max_entries_by_level
                .values()
                .all(|count| *count <= 4)
        );
        assert_eq!(report.topology.actionable_partitions, 0);
        for record in &source {
            let loaded = index
                .get(record.id().clone(), GetOptions::default().with_payload())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(loaded.vector(), record.vector());
            assert_eq!(loaded.fields(), record.fields());
            assert_eq!(
                loaded.payload(),
                &ktann::api::PayloadProjection::Present(record.payload().unwrap().clone())
            );
        }
        let hits = index
            .search(SearchRequest::new(vec![1.0, 0.0], 10).unwrap())
            .await
            .unwrap();
        assert!(!hits.hits.is_empty());
        let replacement =
            Record::new(source[0].id().clone(), vec![0.5, 1.0], vec![Value::I64(99)]).unwrap();
        index.upsert(replacement.clone()).await.unwrap();
        assert_eq!(
            index
                .get(replacement.id().clone(), GetOptions::default())
                .await
                .unwrap()
                .unwrap()
                .fields(),
            replacement.fields()
        );
        index.delete(source[1].id().clone()).await.unwrap();
        assert!(
            index
                .get(source[1].id().clone(), GetOptions::default())
                .await
                .unwrap()
                .is_none()
        );
        let inserted = Record::new(
            Bytes::from_static(b"new"),
            vec![1.0, 1.0],
            vec![Value::I64(99)],
        )
        .unwrap();
        index.insert(inserted).await.unwrap();
        let report = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(report.complete && report.issues.is_empty(), "{report:?}");
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_stage_is_replayed_and_unknown_publication_is_resolved() {
    // Six records in two root leaves: reservation, topology, records, publish.
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        let backend = SharedBackend::new(DeterministicBackend::default());
        backend
            .inner()
            .set_fault_plan(vec![
                CommitFault::UnknownApplied,
                fault,
                CommitFault::Normal,
                CommitFault::Normal,
                CommitFault::UnknownApplied,
            ])
            .unwrap();
        let runtime = make_runtime(backend);
        let index = runtime
            .build_index(
                "bulk",
                config(),
                records(6),
                options(2),
                OperationOptions::default(),
            )
            .await
            .unwrap();
        let report = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(report.complete && report.issues.is_empty(), "{report:?}");
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_publication_stays_unpublished_and_can_be_dropped_after_restart() {
    let persistent = DeterministicConfig {
        durability: support::Durability::Durable,
        ..Default::default()
    };
    let backend = SharedBackend::new(DeterministicBackend::new(persistent));
    backend
        .inner()
        .set_fault_plan(vec![
            CommitFault::Normal,
            CommitFault::Normal,
            CommitFault::Normal,
            CommitFault::UnknownNotApplied,
        ])
        .unwrap();
    let runtime = make_runtime(backend.clone());
    let error = runtime
        .build_index(
            "bulk",
            config(),
            records(6),
            options(2),
            OperationOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    assert_eq!(
        runtime.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexBuilding
    );
    assert_eq!(
        runtime
            .create_index("bulk", config())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::IndexBuilding
    );
    runtime.shutdown().await.unwrap();
    let reopened = make_runtime(SharedBackend::new(backend.inner().reopen()));
    assert_eq!(
        reopened.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexBuilding
    );
    reopened.drop_index("bulk").await.unwrap();
    reopened
        .build_index(
            "bulk",
            config(),
            records(6),
            options(0),
            OperationOptions::default(),
        )
        .await
        .unwrap();
    reopened.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_before_publication_retains_only_unpublished_construction() {
    let inner = Arc::new(DeterministicBackend::default());
    let gate = Arc::new(CommitGate::default());
    gate.hold_next(2);
    let runtime = Runtime::new(
        GatedBackend::new(inner, gate.clone()),
        manual_maintenance_config(),
    )
    .unwrap();
    let task_runtime = runtime.clone();
    let cancel = CancellationToken::new();
    let token = cancel.clone();
    let task = tokio::spawn(async move {
        task_runtime
            .build_index(
                "bulk",
                config(),
                records(6),
                options(2),
                OperationOptions::default().with_cancellation(token),
            )
            .await
    });
    gate.wait_until_entered(1).await;
    gate.release_one();
    gate.wait_until_entered(2).await;
    assert_eq!(
        runtime.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexBuilding
    );
    cancel.cancel();
    assert_eq!(
        task.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    runtime.drop_index("bulk").await.unwrap();
    assert_eq!(
        runtime.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexNotFound
    );
    gate.release();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_fences_in_flight_staging_and_new_name_owner() {
    let inner = Arc::new(DeterministicBackend::default());
    let gate = Arc::new(CommitGate::default());
    gate.hold_next(2);
    let runtime = Runtime::new(
        GatedBackend::new(inner, gate.clone()),
        manual_maintenance_config(),
    )
    .unwrap();
    let task_runtime = runtime.clone();
    let task = tokio::spawn(async move {
        task_runtime
            .build_index(
                "bulk",
                config(),
                records(6),
                options(2),
                OperationOptions::default(),
            )
            .await
    });
    gate.wait_until_entered(1).await;
    gate.release_one();
    gate.wait_until_entered(2).await;
    runtime.drop_index("bulk").await.unwrap();
    let replacement = runtime.create_index("bulk", config()).await.unwrap();
    gate.release();
    let error = task.await.unwrap().unwrap_err();
    assert!(matches!(
        error.kind(),
        ErrorKind::IndexNotFound | ErrorKind::IndexDropping
    ));
    assert_eq!(
        runtime.open_index("bulk").await.unwrap().logical_index_id(),
        replacement.logical_index_id()
    );
    assert!(
        replacement
            .get(Bytes::from_static(b"r000"), GetOptions::default())
            .await
            .unwrap()
            .is_none()
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ambiguous_reservation_cannot_borrow_competing_build_ownership() {
    let inner = Arc::new(DeterministicBackend::default());
    let gate = Arc::new(CommitGate::default());
    gate.hold_next(1);
    let loser = Runtime::new(
        GatedBackend::new(inner.clone(), gate.clone()),
        manual_maintenance_config(),
    )
    .unwrap();
    let task_runtime = loser.clone();
    let task = tokio::spawn(async move {
        task_runtime
            .build_index(
                "bulk",
                config(),
                records(6),
                options(2),
                OperationOptions::default(),
            )
            .await
    });
    gate.wait_until_entered(1).await;
    let winner_gate = Arc::new(CommitGate::default());
    let winner = Runtime::new(
        GatedBackend::new(inner.clone(), winner_gate),
        manual_maintenance_config(),
    )
    .unwrap();
    let published = winner
        .build_index(
            "bulk",
            config(),
            records(8),
            options(0),
            OperationOptions::default(),
        )
        .await
        .unwrap();
    inner.push_fault(CommitFault::UnknownNotApplied).unwrap();
    gate.release();
    assert_eq!(
        task.await.unwrap().unwrap_err().kind(),
        ErrorKind::CommitOutcomeUnknown
    );
    assert!(
        published
            .get(Bytes::from_static(b"r007"), GetOptions::default())
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        winner.open_index("bulk").await.unwrap().logical_index_id(),
        published.logical_index_id()
    );
    loser.shutdown().await.unwrap();
    winner.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn validation_does_not_reserve_name_and_adapter_budgets_bound_staging() {
    let mut config_backend = DeterministicConfig::default();
    config_backend.admission_budget.max_mutations = 4;
    let backend = SharedBackend::new(DeterministicBackend::new(config_backend));
    let runtime = make_runtime(backend);
    let mut duplicate = records(2);
    duplicate.push(duplicate[0].clone());
    assert_eq!(
        runtime
            .build_index(
                "bulk",
                config(),
                duplicate,
                options(2),
                OperationOptions::default()
            )
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::RecordAlreadyExists
    );
    assert_eq!(
        runtime.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexNotFound
    );
    let index = runtime
        .build_index(
            "bulk",
            config(),
            records(17),
            options(5),
            OperationOptions::default(),
        )
        .await
        .unwrap();
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn publication_preserves_older_unpublished_snapshot() {
    let inner = Arc::new(DeterministicBackend::default());
    let gate = Arc::new(CommitGate::default());
    gate.hold_next(2);
    let runtime = Runtime::new(
        GatedBackend::new(inner.clone(), gate.clone()),
        manual_maintenance_config(),
    )
    .unwrap();
    let task_runtime = runtime.clone();
    let task = tokio::spawn(async move {
        task_runtime
            .build_index(
                "bulk",
                config(),
                records(6),
                options(2),
                OperationOptions::default(),
            )
            .await
    });
    gate.wait_until_entered(1).await;
    gate.release_one();
    gate.wait_until_entered(2).await;
    let mut old = ReadLogicalTxn::bootstrap(inner.begin_read().await.unwrap());
    let Some(PersistentValue::IndexNameEntry(entry)) = old
        .get(LogicalKey::IndexNameDirectory(
            ktann::api::IndexName::new("bulk").unwrap(),
        ))
        .await
        .unwrap()
    else {
        panic!("reserved name");
    };
    gate.release();
    let index = task.await.unwrap().unwrap();
    let Some(PersistentValue::IndexManifest(manifest)) = old
        .get(LogicalKey::Manifest(entry.logical_index_id()))
        .await
        .unwrap()
    else {
        panic!("manifest in original snapshot");
    };
    assert!(matches!(
        manifest.lifecycle(),
        IndexLifecycle::Building { .. }
    ));
    assert_eq!(index.logical_index_id(), entry.logical_index_id());
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    drop(old);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_single_record_group_cannot_partially_commit_or_publish() {
    let mut config_backend = DeterministicConfig::default();
    config_backend.admission_budget.max_mutations = 3; // Reservation fits, payload group does not.
    let backend = SharedBackend::new(DeterministicBackend::new(config_backend));
    let runtime = make_runtime(backend.clone());
    let error = runtime
        .build_index(
            "bulk",
            config(),
            records(17),
            options(2),
            OperationOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    assert_eq!(
        runtime.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexBuilding
    );
    let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await.unwrap());
    let Some(PersistentValue::IndexNameEntry(entry)) = txn
        .get(LogicalKey::IndexNameDirectory(
            ktann::api::IndexName::new("bulk").unwrap(),
        ))
        .await
        .unwrap()
    else {
        panic!("reserved name");
    };
    let Some(PersistentValue::IndexManifest(manifest)) = txn
        .get(LogicalKey::Manifest(entry.logical_index_id()))
        .await
        .unwrap()
    else {
        panic!("building manifest");
    };
    drop(txn);
    let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
    let id = Bytes::from_static(b"r000");
    assert!(
        txn.get(LogicalKey::Record {
            index: entry.logical_index_id(),
            id: id.clone()
        })
        .await
        .unwrap()
        .is_none()
    );
    assert!(
        txn.get(LogicalKey::Location {
            index: entry.logical_index_id(),
            id: id.clone()
        })
        .await
        .unwrap()
        .is_none()
    );
    assert!(
        txn.get(LogicalKey::Payload {
            index: entry.logical_index_id(),
            id
        })
        .await
        .unwrap()
        .is_none()
    );
    drop(txn);
    runtime.drop_index("bulk").await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_construction_publishes_then_accepts_ordinary_insert() {
    let runtime = make_runtime(SharedBackend::new(DeterministicBackend::default()));
    let index = runtime
        .build_index(
            "bulk",
            config(),
            vec![],
            options(5),
            OperationOptions::default(),
        )
        .await
        .unwrap();
    index.insert(records(1).pop().unwrap()).await.unwrap();
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    runtime.shutdown().await.unwrap();
}

/// Construction is immutable during its audit; an Active index is not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn build_audit_renews_short_snapshots_but_online_verify_does_not() {
    let inner = Arc::new(DeterministicBackend::default());
    let mut backend = GatedBackend::new(inner.clone(), Arc::new(CommitGate::default()));
    backend.scan_limit = 1;
    let runtime = Runtime::new(backend, manual_maintenance_config()).unwrap();
    let index = runtime
        .build_index(
            "bulk",
            config(),
            records(1025),
            options(0),
            OperationOptions::default(),
        )
        .await
        .expect("immutable build must span short-lived read transactions");
    assert_eq!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Backend
    );
    let stable = Runtime::new(
        GatedBackend::new(inner, Arc::new(CommitGate::default())),
        manual_maintenance_config(),
    )
    .unwrap();
    let report = stable
        .open_index("bulk")
        .await
        .unwrap()
        .verify(VerifyOptions::default())
        .await
        .unwrap();
    assert!(report.complete && report.issues.is_empty());
    assert_eq!(report.objects.vector_records, 1025);
    stable.shutdown().await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_during_build_audit_cannot_publish_over_a_new_name_owner() {
    let backend = GatedBackend::new(
        Arc::new(DeterministicBackend::default()),
        Arc::new(CommitGate::default()),
    );
    let read_gate = Arc::clone(&backend.read_gate);
    read_gate.hold_next(1);
    let runtime = Runtime::new(backend, manual_maintenance_config()).unwrap();
    let builder = runtime.clone();
    let task = tokio::spawn(async move {
        builder
            .build_index(
                "bulk",
                config(),
                records(1025),
                options(0),
                OperationOptions::default(),
            )
            .await
    });
    read_gate.wait_until_entered(1).await;
    runtime.drop_index("bulk").await.unwrap();
    let replacement = runtime.create_index("bulk", config()).await.unwrap();
    read_gate.release();
    let error = task.await.unwrap().unwrap_err();
    assert!(matches!(
        error.kind(),
        ErrorKind::IndexNotFound | ErrorKind::IndexDropping
    ));
    assert_eq!(
        runtime.open_index("bulk").await.unwrap().logical_index_id(),
        replacement.logical_index_id()
    );
    let report = replacement.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty());
    assert_eq!(report.objects.vector_records, 0);
    runtime.shutdown().await.unwrap();
}
