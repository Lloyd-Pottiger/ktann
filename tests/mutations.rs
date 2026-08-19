//! Foreground mutation routing and retry contract tests.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, GetOptions, Index, IndexConfig, LogicalIndexId,
    Metric, Mutation, MutationOutcome, OperationOptions, PartitionKey, PayloadProjection, Record,
    RuntimeConfig, UpsertResult, Value,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::{AdmissionBudget, Backend, Capabilities, HardLimits, ScanLimits};
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    IndexLifecycle, IndexManifest, PartitionHeader, PersistentValue, RecordLocation,
};
use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};
use tokio_util::sync::CancellationToken;

use support::{
    CommitFault, CommitOutcome, DeterministicBackend, DeterministicConfig, DeterministicReadTxn,
    DeterministicWriteTxn, Rng,
};

#[allow(dead_code)]
mod support;

#[derive(Clone)]
struct SharedBackend {
    inner: Arc<DeterministicBackend>,
}

impl SharedBackend {
    fn new(inner: DeterministicBackend) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl Backend for SharedBackend {
    type ReadTxn<'backend> = DeterministicReadTxn<'backend>;

    type WriteTxn<'backend> = DeterministicWriteTxn<'backend>;

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
        self.inner.begin_write().await
    }
}

fn backend(config: DeterministicConfig) -> SharedBackend {
    SharedBackend::new(DeterministicBackend::new(config))
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

fn make_runtime(backend: SharedBackend) -> Runtime<SharedBackend> {
    Runtime::new(backend, RuntimeConfig::default()).expect("runtime is valid")
}

fn tree_key(bucket: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(bucket)]).expect("valid tree key")
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

fn record_with_payload(id: &[u8], x: f32, bucket: i64, payload: &[u8]) -> Record {
    record(id, x, bucket)
        .with_payload(Bytes::copy_from_slice(payload))
        .expect("valid payload")
}

async fn make_index(runtime: &Runtime<SharedBackend>) -> Index<SharedBackend> {
    runtime
        .create_index("index", config())
        .await
        .expect("create index")
}

async fn read_manifest(backend: &SharedBackend, index: LogicalIndexId) -> IndexManifest {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    match txn
        .get(LogicalKey::Manifest(index))
        .await
        .expect("read manifest")
    {
        Some(PersistentValue::IndexManifest(manifest)) => manifest,
        _ => panic!("committed manifest must exist"),
    }
}

async fn read_location(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    record_id: &[u8],
) -> Option<RecordLocation> {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, manifest).expect("bind index");
    match txn
        .get(LogicalKey::Location {
            index: manifest.logical_index_id(),
            id: Bytes::copy_from_slice(record_id),
        })
        .await
        .expect("read location")
    {
        Some(PersistentValue::RecordLocation(location)) => Some(location),
        None => None,
        _ => panic!("a Record Location key holds a Record Location value"),
    }
}

async fn read_header(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    bucket: i64,
    partition: u64,
) -> PartitionHeader {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, manifest).expect("bind index");
    match txn
        .get(LogicalKey::Header {
            index: manifest.logical_index_id(),
            tree_key: tree_key(bucket),
            partition: PartitionKey::new(partition).expect("valid partition key"),
        })
        .await
        .expect("read header")
    {
        Some(PersistentValue::PartitionHeader(header)) => header,
        _ => panic!("committed partition header must exist"),
    }
}

async fn leaf_member_ids(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    bucket: i64,
    partition: u64,
) -> BTreeSet<Bytes> {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, manifest).expect("bind index");
    let range = LogicalRange::leaf_entries(
        manifest,
        &tree_key(bucket),
        PartitionKey::new(partition).expect("valid partition key"),
    )
    .expect("leaf range");
    let page = txn
        .scan(
            &range,
            None,
            ScanLimits {
                item_limit: 1_024,
                byte_limit: 1 << 20,
            },
        )
        .await
        .expect("scan leaf");
    assert!(page.next_cursor().is_none(), "one page covers the leaf");
    page.items()
        .iter()
        .map(|item| match item.key() {
            LogicalKey::LeafEntry { id, .. } => id.clone(),
            _ => panic!("a Leaf Entry range holds only Leaf Entries"),
        })
        .collect()
}

async fn tree_exists(backend: &SharedBackend, manifest: &IndexManifest, bucket: i64) -> bool {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, manifest).expect("bind index");
    ktann::storage::tree_manifest::read_tree_manifest(&mut txn, &tree_key(bucket))
        .await
        .expect("read tree manifest")
        .is_some()
}

fn commit_outcomes(backend: &SharedBackend) -> Vec<CommitOutcome> {
    backend
        .inner
        .history()
        .iter()
        .map(|entry| entry.outcome)
        .collect()
}

async fn assert_record_absent(index: &Index<SharedBackend>, id: &[u8]) {
    assert!(
        index
            .get(
                Bytes::copy_from_slice(id),
                GetOptions::default().with_payload(),
            )
            .await
            .expect("get")
            .is_none(),
        "record must be absent"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn insert_commits_the_record_and_its_membership() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    index
        .insert(record_with_payload(&rid(1), 3.0, 5, b"p1"))
        .await
        .expect("insert");

    let stored = index
        .get(rid(1), GetOptions::default().with_payload())
        .await
        .expect("get")
        .expect("record exists");
    assert_eq!(stored.vector(), &[3.0, 0.0]);
    assert_eq!(stored.fields(), &[Value::I64(5)]);
    assert_eq!(
        stored.payload(),
        &PayloadProjection::Present(Bytes::from_static(b"p1"))
    );

    // The first insert lazily installed the tree with one searchable root.
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let location = read_location(&backend, &manifest, &rid(1))
        .await
        .expect("location exists");
    assert_eq!(location.tree_key(), &tree_key(5));
    assert_eq!(location.leaf(), PartitionKey::new(1).expect("pk"));
    assert_eq!(
        leaf_member_ids(&backend, &manifest, 5, 1).await,
        BTreeSet::from([rid(1)])
    );
    let header = read_header(&backend, &manifest, 5, 1).await;
    assert_eq!(header.entry_count(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upsert_creates_then_replaces_in_place() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    let created = index
        .upsert(record_with_payload(&rid(1), 1.0, 1, b"p1"))
        .await
        .expect("upsert create");
    assert_eq!(created, UpsertResult::Created);

    let replaced = index
        .upsert(record_with_payload(&rid(1), 2.0, 1, b"p2"))
        .await
        .expect("upsert replace");
    assert_eq!(replaced, UpsertResult::Replaced);

    let stored = index
        .get(rid(1), GetOptions::default().with_payload())
        .await
        .expect("get")
        .expect("record exists");
    assert_eq!(stored.vector(), &[2.0, 0.0]);
    assert_eq!(
        stored.payload(),
        &PayloadProjection::Present(Bytes::from_static(b"p2"))
    );

    // A same-tree replacement keeps the same leaf and the exact count.
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    assert_eq!(
        leaf_member_ids(&backend, &manifest, 1, 1).await,
        BTreeSet::from([rid(1)])
    );
    assert_eq!(
        read_header(&backend, &manifest, 1, 1).await.entry_count(),
        1
    );

    // A payload-less replacement deletes the old payload.
    let replaced = index
        .upsert(record(&rid(1), 3.0, 1))
        .await
        .expect("upsert without payload");
    assert_eq!(replaced, UpsertResult::Replaced);
    let stored = index
        .get(rid(1), GetOptions::default().with_payload())
        .await
        .expect("get")
        .expect("record exists");
    assert_eq!(stored.payload(), &PayloadProjection::Absent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upsert_moves_across_tree_keys_preserving_membership() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    index.insert(record(&rid(1), 1.0, 1)).await.expect("insert");
    let replaced = index
        .upsert(record(&rid(1), 2.0, 7))
        .await
        .expect("cross-tree upsert");
    assert_eq!(replaced, UpsertResult::Replaced);

    let stored = index
        .get(rid(1), GetOptions::default())
        .await
        .expect("get")
        .expect("record exists");
    assert_eq!(stored.vector(), &[2.0, 0.0]);
    assert_eq!(stored.fields(), &[Value::I64(7)]);

    // Exactly one membership exists: the new tree's leaf. The source tree's
    // leaf is empty with an exact zero count.
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let location = read_location(&backend, &manifest, &rid(1))
        .await
        .expect("location exists");
    assert_eq!(location.tree_key(), &tree_key(7));
    assert_eq!(
        leaf_member_ids(&backend, &manifest, 1, 1).await,
        BTreeSet::new()
    );
    assert_eq!(
        leaf_member_ids(&backend, &manifest, 7, 1).await,
        BTreeSet::from([rid(1)])
    );
    assert_eq!(
        read_header(&backend, &manifest, 1, 1).await.entry_count(),
        0
    );
    assert_eq!(
        read_header(&backend, &manifest, 7, 1).await.entry_count(),
        1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_is_idempotent_and_removes_membership() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    index
        .insert(record_with_payload(&rid(1), 1.0, 1, b"p1"))
        .await
        .expect("insert");
    assert!(index.delete(rid(1)).await.expect("delete"));
    assert_record_absent(&index, &rid(1)).await;
    assert!(!index.delete(rid(1)).await.expect("idempotent delete"));

    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    assert_eq!(read_location(&backend, &manifest, &rid(1)).await, None);
    assert_eq!(
        leaf_member_ids(&backend, &manifest, 1, 1).await,
        BTreeSet::new()
    );
    assert_eq!(
        read_header(&backend, &manifest, 1, 1).await.entry_count(),
        0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_mutate_commits_ordered_outcomes_atomically() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    index.insert(record(&rid(1), 1.0, 1)).await.expect("insert");
    index.insert(record(&rid(3), 1.0, 1)).await.expect("insert");
    let outcomes = index
        .batch_mutate(vec![
            Mutation::Insert(record(&rid(2), 2.0, 1)),
            Mutation::Upsert(record(&rid(1), 3.0, 2)),
            Mutation::Delete(rid(3)),
            Mutation::Delete(rid(9)),
        ])
        .await
        .expect("batch");
    assert_eq!(
        outcomes,
        vec![
            MutationOutcome::Inserted,
            MutationOutcome::Upserted { replaced: true },
            MutationOutcome::Deleted { existed: true },
            MutationOutcome::Deleted { existed: false },
        ]
    );

    let stored = index
        .get(rid(1), GetOptions::default())
        .await
        .expect("get 1")
        .expect("record 1 exists");
    assert_eq!(stored.vector(), &[3.0, 0.0]);
    assert_record_absent(&index, &rid(3)).await;
    let stored = index
        .get(rid(2), GetOptions::default())
        .await
        .expect("get 2")
        .expect("record 2 exists");
    assert_eq!(stored.vector(), &[2.0, 0.0]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_mutate_is_all_or_nothing_on_item_failure() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    index.insert(record(&rid(1), 1.0, 1)).await.expect("insert");
    let keys_before = backend.inner.db_key_count();

    let error = index
        .batch_mutate(vec![
            Mutation::Insert(record(&rid(2), 2.0, 1)),
            Mutation::Insert(record(&rid(1), 9.0, 1)),
        ])
        .await
        .expect_err("duplicate Record ID at position 1");
    assert_eq!(error.kind(), ErrorKind::RecordAlreadyExists);
    assert_eq!(error.position(), Some(1));

    // The first item's mutation is rolled back with the failed batch, and the
    // duplicate target's committed record is unchanged.
    assert_record_absent(&index, &rid(2)).await;
    let stored = index
        .get(rid(1), GetOptions::default())
        .await
        .expect("get 1")
        .expect("record 1 exists");
    assert_eq!(stored.vector(), &[1.0, 0.0]);
    assert_eq!(stored.fields(), &[Value::I64(1)]);
    assert_eq!(backend.inner.db_key_count(), keys_before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_batch_succeeds_without_storage_work() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;
    let keys_before = backend.inner.db_key_count();

    let outcomes = index.batch_mutate(Vec::new()).await.expect("empty batch");
    assert_eq!(outcomes, Vec::new());
    assert_eq!(backend.inner.db_key_count(), keys_before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_records_fail_validation_before_storage_work() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;
    let keys_before = backend.inner.db_key_count();

    // A wrong vector dimension is rejected before any transaction begins.
    let wrong_dimension = Record::new(rid(1), Arc::from([1.0_f32, 0.0, 0.0]), vec![Value::I64(1)])
        .expect("record shape is valid");
    let error = index
        .insert(wrong_dimension)
        .await
        .expect_err("dimension mismatch");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);

    // A wrong field type fails the whole batch at the item's position, and
    // the valid first item is never committed.
    let wrong_type = Record::new(rid(3), Arc::from([1.0_f32, 0.0]), vec![Value::Bool(true)])
        .expect("record shape is valid");
    let error = index
        .batch_mutate(vec![
            Mutation::Insert(record(&rid(2), 1.0, 1)),
            Mutation::Insert(wrong_type),
        ])
        .await
        .expect_err("field type mismatch");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(error.position(), Some(1));

    assert_record_absent(&index, &rid(2)).await;
    assert_eq!(backend.inner.db_key_count(), keys_before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retryable_abort_replays_the_whole_mutation() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    backend
        .inner
        .push_fault(CommitFault::Abort)
        .expect("fault fits the plan");
    index.insert(record(&rid(1), 1.0, 1)).await.expect("insert");

    // The first attempt aborted and the whole mutation replayed and committed.
    let outcomes = commit_outcomes(&backend);
    let mut insert_outcomes = outcomes.iter().rev().take(2).copied().collect::<Vec<_>>();
    insert_outcomes.reverse();
    assert_eq!(
        insert_outcomes,
        vec![CommitOutcome::Aborted, CommitOutcome::Committed]
    );

    let stored = index
        .get(rid(1), GetOptions::default())
        .await
        .expect("get")
        .expect("record exists");
    assert_eq!(stored.vector(), &[1.0, 0.0]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn contention_exhaustion_is_reported_and_nothing_commits() {
    let backend = backend(DeterministicConfig::default());
    let runtime = Runtime::new(
        backend.clone(),
        RuntimeConfig::default()
            .with_attempts(1, 3)
            .expect("valid attempts"),
    )
    .expect("runtime is valid");
    let index = make_index(&runtime).await;

    backend
        .inner
        .set_fault_plan(vec![CommitFault::Abort; 8])
        .expect("fault plan");
    let error = index
        .insert(record(&rid(1), 1.0, 1))
        .await
        .expect_err("three aborted attempts exhaust the policy");
    assert_eq!(error.kind(), ErrorKind::ContentionExhausted);

    let outcomes = commit_outcomes(&backend);
    let aborted = outcomes
        .iter()
        .filter(|outcome| **outcome == CommitOutcome::Aborted)
        .count();
    assert_eq!(aborted, 3, "exactly the configured attempts ran");
    assert_record_absent(&index, &rid(1)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_commit_outcome_is_returned_without_retry() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    // Unknown-applied: the mutation lands but the caller learns nothing.
    backend
        .inner
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("fault plan");
    let error = index
        .insert(record(&rid(1), 1.0, 1))
        .await
        .expect_err("unknown outcome");
    assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    assert!(
        index
            .get(rid(1), GetOptions::default())
            .await
            .expect("get")
            .is_some(),
        "the applied mutation is committed even though its outcome is unknown"
    );

    // Unknown-not-applied: nothing lands, and the operation is not retried.
    backend
        .inner
        .set_fault_plan(vec![CommitFault::UnknownNotApplied])
        .expect("fault plan");
    let history_before = backend.inner.history().len();
    let error = index
        .insert(record(&rid(2), 2.0, 1))
        .await
        .expect_err("unknown outcome");
    assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    assert_record_absent(&index, &rid(2)).await;
    assert_eq!(
        backend.inner.history().len(),
        history_before + 1,
        "an unknown outcome never enters the retry loop"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_and_deadline_fail_before_storage_work() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;
    let keys_before = backend.inner.db_key_count();

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = index
        .insert_with_control(
            record(&rid(1), 1.0, 1),
            OperationOptions::default().with_cancellation(cancellation),
        )
        .await
        .expect_err("cancelled before admission");
    assert_eq!(error.kind(), ErrorKind::Cancelled);

    let error = index
        .insert_with_control(
            record(&rid(1), 1.0, 1),
            OperationOptions::default().with_deadline(Instant::now()),
        )
        .await
        .expect_err("expired deadline");
    assert_eq!(error.kind(), ErrorKind::DeadlineExceeded);

    assert_eq!(backend.inner.db_key_count(), keys_before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_closed_runtime_rejects_mutations() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;
    runtime.shutdown().await.expect("shutdown");

    let error = index
        .insert(record(&rid(1), 1.0, 1))
        .await
        .expect_err("runtime closed");
    assert_eq!(error.kind(), ErrorKind::RuntimeClosed);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropping_or_dropped_index_fails_closed() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let dropping = make_index(&runtime).await;

    // Flip the persisted Manifest to Dropping underneath the open handle.
    let manifest = read_manifest(&backend, dropping.logical_index_id()).await;
    {
        let raw = backend.begin_write().await.expect("begin write");
        let limits = backend.hard_limits();
        let budget = backend.admission_budget();
        let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);
        txn.put(
            LogicalKey::Manifest(manifest.logical_index_id()),
            PersistentValue::IndexManifest(
                manifest
                    .with_lifecycle(IndexLifecycle::Dropping)
                    .expect("dropping manifest"),
            ),
        )
        .await
        .expect("put dropping manifest");
        txn.commit().await.expect("commit dropping manifest");
    }
    let error = dropping
        .insert(record(&rid(1), 1.0, 1))
        .await
        .expect_err("dropping index");
    assert_eq!(error.kind(), ErrorKind::IndexDropping);

    // A completed drop removes the Manifest; the stale handle fails closed.
    let dropped = runtime
        .create_index("index-2", config())
        .await
        .expect("create second index");
    runtime.drop_index("index-2").await.expect("drop");
    let error = dropped
        .insert(record(&rid(1), 1.0, 1))
        .await
        .expect_err("dropped index");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_admission_limits_are_enforced() {
    let backend = backend(DeterministicConfig {
        admission_budget: AdmissionBudget {
            max_mutations: 3,
            max_mutation_bytes: 1 << 20,
        },
        ..DeterministicConfig::default()
    });
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    let error = index
        .insert(record(&rid(1), 1.0, 1))
        .await
        .expect_err("the mutation exceeds the admission budget");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    assert_record_absent(&index, &rid(1)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutations_work_on_an_index_without_tree_key_fields() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index(
            "fieldless",
            IndexConfig::new(2, Metric::L2).expect("valid config"),
        )
        .await
        .expect("create index");

    let record =
        |x: f32| Record::new(rid(1), Arc::from([x, 0.0_f32]), Vec::new()).expect("valid record");
    index.insert(record(1.0)).await.expect("insert");
    let stored = index
        .get(rid(1), GetOptions::default())
        .await
        .expect("get")
        .expect("record exists");
    assert_eq!(stored.vector(), &[1.0, 0.0]);
    assert!(stored.fields().is_empty());

    assert_eq!(
        index.upsert(record(2.0)).await.expect("upsert"),
        UpsertResult::Replaced
    );
    assert!(index.delete(rid(1)).await.expect("delete"));
    assert_record_absent(&index, &rid(1)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_mutations_converge_with_exact_membership() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    // Racing first inserts on one Tree Key conflict on tree creation and the
    // leaf Header; whole-attempt retries converge both mutations.
    let mut tasks = Vec::new();
    for value in 0..8_u8 {
        let index = index.clone();
        tasks.push(tokio::spawn(async move {
            index.insert(record(&rid(value), 1.0, 1)).await
        }));
    }
    for task in tasks {
        task.await
            .expect("mutation task did not panic")
            .expect("every racing insert commits");
    }

    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let members = leaf_member_ids(&backend, &manifest, 1, 1).await;
    assert_eq!(members.len(), 8);
    assert_eq!(
        read_header(&backend, &manifest, 1, 1).await.entry_count(),
        8
    );
    for value in 0..8_u8 {
        assert!(
            index
                .get(rid(value), GetOptions::default())
                .await
                .expect("get")
                .is_some()
        );
    }
}

/// Runs a seeded model history over insert, upsert, and delete with injected
/// abort faults, asserting exact membership after every committed operation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seeded_model_history_with_abort_faults_preserves_membership() {
    let backend = backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = make_index(&runtime).await;

    // A deterministic fault plan: at most two consecutive definite aborts, so
    // every operation commits within the default attempt bound.
    let mut fault_rng = Rng(0x9e37_90ab_cdef_1235);
    let mut plan = Vec::new();
    let mut consecutive = 0_u32;
    while plan.len() < 512 {
        if consecutive < 2 && fault_rng.below(4) == 0 {
            plan.push(CommitFault::Abort);
            consecutive += 1;
        } else {
            plan.push(CommitFault::Normal);
            consecutive = 0;
        }
    }
    backend.inner.set_fault_plan(plan).expect("fault plan");

    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    let mut model: HashMap<Bytes, (f32, i64)> = HashMap::new();
    for _ in 0..150 {
        let id = rid(rng.below(24) as u8);
        match rng.below(3) {
            0 => {
                let x = rng.below(100) as f32;
                let bucket = rng.below(3) as i64;
                match index.insert(record(&id, x, bucket)).await {
                    Ok(()) => assert!(
                        model.insert(id.clone(), (x, bucket)).is_none(),
                        "a committed insert means the model had no record"
                    ),
                    Err(error) if error.kind() == ErrorKind::RecordAlreadyExists => {
                        assert!(model.contains_key(&id));
                    }
                    Err(error) => panic!("unexpected insert error: {error:?}"),
                }
            }
            1 => {
                let x = rng.below(100) as f32;
                let bucket = rng.below(3) as i64;
                let expected = model.contains_key(&id);
                let result = index.upsert(record(&id, x, bucket)).await.expect("upsert");
                assert_eq!(result == UpsertResult::Replaced, expected);
                model.insert(id.clone(), (x, bucket));
            }
            _ => {
                let existed = index.delete(id.clone()).await.expect("delete");
                assert_eq!(existed, model.remove(&id).is_some());
            }
        }

        // Read-your-writes: the touched Record ID reflects the model.
        let stored = index
            .get(id.clone(), GetOptions::default())
            .await
            .expect("get");
        match (stored, model.get(&id)) {
            (None, None) => {}
            (Some(stored), Some((x, bucket))) => {
                assert_eq!(stored.vector(), &[*x, 0.0]);
                assert_eq!(stored.fields(), &[Value::I64(*bucket)]);
            }
            (stored, expected) => panic!("model mismatch: {stored:?} against {expected:?}"),
        }
    }

    // Final consistency pass: every modeled record reads back and each tree's
    // scanned Leaf Entry membership and exact Header count match the model.
    for value in 0..24_u8 {
        let stored = index
            .get(rid(value), GetOptions::default())
            .await
            .expect("get");
        match (stored, model.get(&rid(value))) {
            (None, None) => {}
            (Some(stored), Some((x, bucket))) => {
                assert_eq!(stored.vector(), &[*x, 0.0]);
                assert_eq!(stored.fields(), &[Value::I64(*bucket)]);
            }
            (stored, expected) => panic!("final mismatch: {stored:?} against {expected:?}"),
        }
    }
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    for bucket in 0..3_i64 {
        let expected: BTreeSet<Bytes> = model
            .iter()
            .filter(|(_, (_, b))| *b == bucket)
            .map(|(id, _)| id.clone())
            .collect();
        if expected.is_empty() && !tree_exists(&backend, &manifest, bucket).await {
            // A tree is installed lazily; an untouched bucket may have none.
            continue;
        }
        assert_eq!(
            leaf_member_ids(&backend, &manifest, bucket, 1).await,
            expected
        );
        assert_eq!(
            read_header(&backend, &manifest, bucket, 1)
                .await
                .entry_count() as usize,
            expected.len()
        );
    }
}
