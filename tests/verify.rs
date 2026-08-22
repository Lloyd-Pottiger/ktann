//! Public bounded read-only verification contract tests (issue #35).
//!
//! Every test drives the public `Index::verify` API against the deterministic
//! in-memory backend. A healthy index verifies complete and issue-free —
//! including legal in-flight split states and under concurrent foreground
//! commits — and each injected invariant violation surfaces as a structured,
//! redacted issue of its documented kind. Reaching the issue, object, or
//! memory limit stops the audit with `complete: false`; cancellation and
//! deadline surface as errors. Verification never writes persistent data.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, Index, IndexConfig, Metric, Mutation,
    OperationOptions, PartitionKey, Record, RuntimeConfig, Value, VerifyIssueKind, VerifyOptions,
    VerifyReport,
};
use ktann::maintenance::split::{self, Advance};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{
    IndexIdAllocator, IndexManifest, LeafEntry, OpaquePayload, PartitionHeader, PartitionState,
    PartitionSynopsis, PersistentValue, RecordLocation, ValueCodec,
};
use ktann::storage::{ReadLogicalTxn, WriteLogicalTxn};
use tokio_util::sync::CancellationToken;

use support::{DeterministicBackend, DeterministicConfig, SharedBackend, audit};

#[allow(dead_code)]
mod support;

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

fn tree_key(bucket: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(bucket)]).expect("valid tree key")
}

/// A one-dimensional L2 index: `bucket` is the Tree Key field and `score` a
/// nullable non-key field for projection and synopsis coverage.
fn config() -> IndexConfig {
    config_with(1, None)
}

fn config_with(dimension: usize, partition_entries: Option<(u32, u32)>) -> IndexConfig {
    let config = IndexConfig::new(dimension, Metric::L2)
        .expect("valid config")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
            FieldSchema::new("score", DataType::I64)
                .expect("valid field")
                .nullable(),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields");
    match partition_entries {
        Some((minimum, maximum)) => config
            .with_partition_entries(minimum, maximum)
            .expect("valid partition entries"),
        None => config,
    }
}

fn rid(id: u32) -> Bytes {
    Bytes::from(format!("r{id}"))
}

fn record(id: u32, x: f32, bucket: i64) -> Record {
    Record::new(
        rid(id),
        Arc::from([x]),
        vec![Value::I64(bucket), Value::I64(i64::from(id))],
    )
    .expect("valid record")
}

fn record_wide(id: u32, dimension: usize, bucket: i64) -> Record {
    Record::new(
        rid(id),
        Arc::from(vec![id as f32; dimension]),
        vec![Value::I64(bucket), Value::I64(i64::from(id))],
    )
    .expect("valid record")
}

async fn setup(
    config: IndexConfig,
) -> (SharedBackend, Runtime<SharedBackend>, Index<SharedBackend>) {
    let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()));
    let runtime =
        Runtime::new(backend.clone(), support::manual_maintenance_config()).expect("runtime");
    let index = runtime
        .create_index("verify", config)
        .await
        .expect("create index");
    (backend, runtime, index)
}

async fn insert_all(index: &Index<SharedBackend>, records: Vec<Record>) {
    // The deterministic backend's default admission budget bounds one
    // transaction, so loads commit in bounded batches.
    for chunk in records.chunks(100) {
        index
            .batch_mutate(chunk.iter().cloned().map(Mutation::Insert).collect())
            .await
            .expect("batch insert");
    }
}

/// Two records in one tree's single root leaf: `r0` scores 0, `r1` scores 1.
async fn two_record_setup() -> (
    SharedBackend,
    Runtime<SharedBackend>,
    Index<SharedBackend>,
    IndexManifest,
) {
    let (backend, runtime, index) = setup(config()).await;
    insert_all(&index, vec![record(0, 0.0, 1), record(1, 1.0, 1)]).await;
    let manifest = support::read_manifest(&backend, index.logical_index_id()).await;
    (backend, runtime, index, manifest)
}

/// Writes one typed but semantically inconsistent value, bypassing the
/// mutation protocol's invariants.
async fn typed_put(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: LogicalKey,
    value: PersistentValue,
) {
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind index");
    txn.put(key, value).await.expect("typed put");
    txn.commit().await.expect("commit");
}

async fn raw_put(backend: &SharedBackend, key: Vec<u8>, value: Bytes) {
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.put(Bytes::from(key), value).await.expect("raw put");
    txn.commit().await.expect("commit");
}

async fn raw_delete(backend: &SharedBackend, key: Vec<u8>) {
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.delete(Bytes::from(key)).await.expect("raw delete");
    txn.commit().await.expect("commit");
}

async fn read_leaf_entry(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    bucket: i64,
    partition: PartitionKey,
    id: Bytes,
) -> LeafEntry {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, manifest).expect("bind index");
    match txn
        .get(LogicalKey::LeafEntry {
            index: manifest.logical_index_id(),
            tree_key: tree_key(bucket),
            partition,
            id,
        })
        .await
        .expect("read leaf entry")
    {
        Some(PersistentValue::LeafEntry(entry)) => entry,
        other => panic!("leaf entry must exist, got {other:?}"),
    }
}

fn kinds(report: &VerifyReport) -> Vec<VerifyIssueKind> {
    report.issues.iter().map(|issue| issue.kind).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_index_verifies_complete() {
    let (_backend, runtime, index) = setup(config()).await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(report.complete);
    assert!(report.issues.is_empty());
    assert_eq!(report.objects.vector_records, 0);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn healthy_index_verifies_complete_and_writes_nothing() {
    let (backend, runtime, index) = setup(config()).await;
    insert_all(
        &index,
        (0..6_u32)
            .map(|id| record(id, id as f32, i64::from(id % 3)))
            .collect(),
    )
    .await;
    // Six records across three trees, each a single Ready root leaf.

    backend.inner().reset_operation_counts();
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(report.complete, "issues: {:?}", report.issues);
    assert!(report.issues.is_empty());
    assert_eq!(report.objects.vector_records, 6);
    assert_eq!(report.objects.record_locations, 6);
    assert_eq!(report.objects.entries, 6);
    // Three root leaves, each with a Header, a Synopsis, and a State.
    assert_eq!(report.objects.partitions, 9);
    // The allocator, the Manifest, six Record/Location pairs, three Tree
    // Manifests, nine partition bodies, and six Leaf Entries.
    assert_eq!(report.objects.total, 32);

    // The audit is read-only: no mutation or clear calls reached the backend.
    let counts = backend.inner().operation_counts();
    assert_eq!(counts.put, 0);
    assert_eq!(counts.insert, 0);
    assert_eq!(counts.delete, 0);
    assert_eq!(counts.batch_mutate, 0);
    assert_eq!(counts.clear_range, 0);
    assert!(counts.get > 0 && counts.scan > 0);

    // The report is deterministic across runs of the same snapshot state.
    let again = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(again.complete && again.issues.is_empty());
    assert_eq!(again.objects, report.objects);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inflight_split_states_verify_clean() {
    let (backend, runtime, index) = setup(config_with(1, Some((1, 2)))).await;
    insert_all(
        &index,
        (0..4_u32).map(|id| record(id, id as f32, 1)).collect(),
    )
    .await;
    let manifest = support::read_manifest(&backend, index.logical_index_id()).await;
    let retry = RetryPolicy::for_fixup(&RuntimeConfig::default());

    // Verify the over-full Ready leaf, then every committed intermediate
    // split state — reserved-unexposed targets, exposed targets receiving,
    // draining — and the converged post-split tree. Every legal state is
    // complete and issue-free.
    let mut clock = 1_000_u64;
    for step in 0..16_u32 {
        let report = index
            .verify(VerifyOptions::default())
            .await
            .expect("verify");
        assert!(
            report.complete && report.issues.is_empty(),
            "step {step} issues: {:?}",
            report.issues
        );
        let outcome = split::advance(&backend, &manifest, &tree_key(1), pk(1), clock, &retry)
            .await
            .expect("advance");
        clock += 100;
        if matches!(outcome, Advance::Idle | Advance::Completed) {
            break;
        }
    }
    // The split must actually have happened: the tree grew beyond its root.
    let partitions = audit::list_partitions(&backend, index.logical_index_id())
        .await
        .expect("list partitions");
    assert!(partitions.len() > 1, "the root split never began");
    assert!(
        partitions
            .iter()
            .all(|(_, _, header)| header.state() == PartitionState::Ready)
    );
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(report.complete && report.issues.is_empty());

    // Phase two: over-fill one leaf below the internal root and drive its
    // non-root split. Unlike a root split's targets, these carry ordinary
    // incoming Child Entries from their parent.
    insert_all(&index, (10..14_u32).map(|id| record(id, 5.0, 1)).collect()).await;
    let over_full = audit::list_partitions(&backend, index.logical_index_id())
        .await
        .expect("list partitions")
        .into_iter()
        .find(|(_, _, header)| header.level() == 1 && header.entry_count() > 2)
        .map(|(_, partition, _)| partition)
        .expect("an over-full leaf exists");
    for step in 0..16_u32 {
        let report = index
            .verify(VerifyOptions::default())
            .await
            .expect("verify");
        assert!(
            report.complete && report.issues.is_empty(),
            "non-root step {step} issues: {:?}",
            report.issues
        );
        let outcome = split::advance(&backend, &manifest, &tree_key(1), over_full, clock, &retry)
            .await
            .expect("advance");
        clock += 100;
        if matches!(outcome, Advance::Idle | Advance::Completed) {
            break;
        }
    }
    let partitions = audit::list_partitions(&backend, index.logical_index_id())
        .await
        .expect("list partitions");
    assert!(
        partitions
            .iter()
            .all(|(_, _, header)| header.state() == PartitionState::Ready)
    );
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(report.complete && report.issues.is_empty());
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_leaf_entry_is_reported_and_the_audit_continues() {
    let (backend, runtime, index, _manifest) = two_record_setup().await;
    raw_put(
        &backend,
        keys::leaf_entry_key(index.logical_index_id(), &tree_key(1), pk(1), &rid(0))
            .expect("leaf entry key"),
        Bytes::from_static(b"not-a-canonical-leaf-entry"),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    // Findings do not truncate: the audit covered everything else.
    assert!(report.complete);
    // The malformed entry reports InvalidEncoding; the exact Header count
    // still says two while one entry decoded, and the record with its
    // location never joined and reports Membership at the end of the walk.
    assert_eq!(
        kinds(&report),
        vec![
            VerifyIssueKind::InvalidEncoding,
            VerifyIssueKind::CountMismatch,
            VerifyIssueKind::Membership,
        ]
    );
    let issue = &report.issues[0];
    assert_eq!(issue.partition_key, pk(1));
    assert_eq!(issue.record_id, Some(rid(0)));
    assert_ne!(issue.tree_key_hash, [0; 32]);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_key_is_reported_with_sentinel_context() {
    let (backend, runtime, index, _manifest) = two_record_setup().await;
    let mut key = keys::manifest_key(index.logical_index_id());
    key.push(0x00); // a Manifest key with trailing bytes never decodes
    raw_put(&backend, key, Bytes::from_static(b"garbage")).await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(report.complete);
    assert_eq!(kinds(&report), vec![VerifyIssueKind::InvalidEncoding]);
    // An undecodable key carries no Tree Key, partition, or Record ID
    // context: the zero hash and Partition Key 1 sentinels stand in.
    let issue = &report.issues[0];
    assert_eq!(issue.tree_key_hash, [0; 32]);
    assert_eq!(issue.partition_key, pk(1));
    assert_eq!(issue.record_id, None);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dangling_location_and_orphan_entry_are_membership() {
    let (backend, runtime, index, _manifest) = two_record_setup().await;
    // The Vector Record of r0 disappears; its Location and Leaf Entry remain.
    raw_delete(
        &backend,
        keys::record_key(index.logical_index_id(), &rid(0)).expect("record key"),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(report.complete);
    assert_eq!(
        kinds(&report),
        vec![VerifyIssueKind::Membership, VerifyIssueKind::Membership]
    );
    assert!(
        report
            .issues
            .iter()
            .all(|issue| issue.record_id == Some(rid(0)))
    );
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_without_location_is_membership() {
    let (backend, runtime, index, _manifest) = two_record_setup().await;
    raw_delete(
        &backend,
        keys::location_key(index.logical_index_id(), &rid(0)).expect("location key"),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(
        kinds(&report),
        vec![VerifyIssueKind::Membership, VerifyIssueKind::Membership]
    );
    // The record-side finding has no location context; the entry-side one
    // names the Leaf Entry's position.
    assert_eq!(report.issues[0].tree_key_hash, [0; 32]);
    assert_ne!(report.issues[1].tree_key_hash, [0; 32]);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_payload_is_membership() {
    let (backend, runtime, index, manifest) = two_record_setup().await;
    let payload = ValueCodec::for_index(&manifest)
        .encode(&PersistentValue::OpaquePayload(
            OpaquePayload::new(Bytes::from_static(b"blob")).expect("payload"),
        ))
        .expect("encode payload");
    raw_put(
        &backend,
        keys::payload_key(index.logical_index_id(), &rid(99)).expect("payload key"),
        Bytes::from(payload),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(kinds(&report), vec![VerifyIssueKind::Membership]);
    assert_eq!(report.issues[0].record_id, Some(rid(99)));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn divergent_location_is_membership_on_both_sides() {
    let (backend, runtime, index, manifest) = two_record_setup().await;
    // The Location names a leaf that is not where the Leaf Entry lives.
    typed_put(
        &backend,
        &manifest,
        LogicalKey::Location {
            index: index.logical_index_id(),
            id: rid(0),
        },
        PersistentValue::RecordLocation(RecordLocation::new(tree_key(1), pk(99))),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(
        kinds(&report),
        vec![VerifyIssueKind::Membership, VerifyIssueKind::Membership]
    );
    assert_eq!(report.issues[0].partition_key, pk(1));
    assert_eq!(report.issues[1].partition_key, pk(99));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wrong_header_count_is_count_mismatch() {
    let (backend, runtime, index, manifest) = two_record_setup().await;
    typed_put(
        &backend,
        &manifest,
        LogicalKey::Header {
            index: index.logical_index_id(),
            tree_key: tree_key(1),
            partition: pk(1),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 3, 9, PartitionState::Ready).expect("header"),
        ),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(kinds(&report), vec![VerifyIssueKind::CountMismatch]);
    assert_eq!(report.issues[0].partition_key, pk(1));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_partition_state_is_reachability() {
    let (backend, runtime, index, _manifest) = two_record_setup().await;
    raw_delete(
        &backend,
        keys::state_key(index.logical_index_id(), &tree_key(1), pk(1)),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(kinds(&report), vec![VerifyIssueKind::Reachability]);
    assert_eq!(report.issues[0].partition_key, pk(1));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_synopsis_is_not_conservative() {
    let (backend, runtime, index, manifest) = two_record_setup().await;
    // Overwrite the leaf's Synopsis with the canonical empty one: it no
    // longer covers the entries the audit recomputes.
    typed_put(
        &backend,
        &manifest,
        LogicalKey::Synopsis {
            index: index.logical_index_id(),
            tree_key: tree_key(1),
            partition: pk(1),
        },
        PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(&manifest)),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(
        kinds(&report),
        vec![VerifyIssueKind::SynopsisNotConservative]
    );
    assert_eq!(report.issues[0].partition_key, pk(1));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edited_leaf_projection_is_record_projection_mismatch() {
    let (backend, runtime, index, manifest) = two_record_setup().await;
    // Rewrite r0's Leaf Entry with a different non-key field. The new value
    // stays inside the stored Synopsis, so only the projection check fires.
    let original = read_leaf_entry(&backend, &manifest, 1, pk(1), rid(0)).await;
    typed_put(
        &backend,
        &manifest,
        LogicalKey::LeafEntry {
            index: index.logical_index_id(),
            tree_key: tree_key(1),
            partition: pk(1),
            id: rid(0),
        },
        PersistentValue::LeafEntry(LeafEntry::new(
            rid(0),
            vec![Value::I64(1), Value::I64(1)],
            original.rabitq7().clone(),
        )),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(
        kinds(&report),
        vec![VerifyIssueKind::RecordProjectionMismatch]
    );
    assert_eq!(report.issues[0].record_id, Some(rid(0)));
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn allocator_below_index_is_count_mismatch() {
    let (backend, runtime, index, _manifest) = two_record_setup().await;
    let encoded = ValueCodec::bootstrap()
        .encode(&PersistentValue::IndexIdAllocator(IndexIdAllocator::new(0)))
        .expect("encode allocator");
    raw_put(
        &backend,
        keys::index_id_allocator_key(),
        Bytes::from(encoded),
    )
    .await;
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(kinds(&report), vec![VerifyIssueKind::CountMismatch]);
    // The allocator is namespace state: sentinel context.
    assert_eq!(report.issues[0].tree_key_hash, [0; 32]);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn object_and_memory_limits_truncate() {
    // The two-record index owns exactly twelve logical objects: the
    // allocator, the Manifest, two Record/Location pairs, one Tree Manifest,
    // three partition bodies, and two Leaf Entries.
    let (_backend, runtime, index, _manifest) = two_record_setup().await;

    // The exact object boundary still completes; one less truncates.
    let exact = index
        .verify(
            VerifyOptions::default()
                .with_object_limit(12)
                .expect("valid limit"),
        )
        .await
        .expect("verify");
    assert!(exact.complete);
    assert_eq!(exact.objects.total, 12);
    let truncated = index
        .verify(
            VerifyOptions::default()
                .with_object_limit(11)
                .expect("valid limit"),
        )
        .await
        .expect("verify");
    assert!(!truncated.complete);
    assert_eq!(truncated.objects.total, 11);

    // A tiny memory limit cannot hold the first scan page.
    let truncated = index
        .verify(
            VerifyOptions::default()
                .with_memory_limit_bytes(64)
                .expect("valid limit"),
        )
        .await
        .expect("verify");
    assert!(!truncated.complete);
    assert_eq!(truncated.objects.total, 1);
    assert!(truncated.issues.is_empty());
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn issue_limit_truncates() {
    let (backend, runtime, index, _manifest) = two_record_setup().await;
    for suffix in [0x00_u8, 0x01_u8] {
        let mut key = keys::manifest_key(index.logical_index_id());
        key.push(suffix);
        raw_put(&backend, key, Bytes::from_static(b"garbage")).await;
    }
    let report = index
        .verify(
            VerifyOptions::default()
                .with_issue_limit(1)
                .expect("valid limit"),
        )
        .await
        .expect("verify");
    assert!(!report.complete);
    assert_eq!(report.issues.len(), 1);
    assert_eq!(report.issues[0].kind, VerifyIssueKind::InvalidEncoding);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_or_expired_verify_is_an_error() {
    let (_backend, runtime, index, _manifest) = two_record_setup().await;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let options = VerifyOptions::default()
        .with_operation_options(OperationOptions::default().with_cancellation(cancellation));
    let error = index.verify(options).await.expect_err("cancelled verify");
    assert_eq!(error.kind(), ErrorKind::Cancelled);

    let options = VerifyOptions::default()
        .with_operation_options(OperationOptions::default().with_deadline(Instant::now()));
    let error = index.verify(options).await.expect_err("expired verify");
    assert_eq!(error.kind(), ErrorKind::DeadlineExceeded);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancellation_stops_a_long_audit() {
    let (_backend, runtime, index) = setup(config_with(8, None)).await;
    insert_all(
        &index,
        (0..4_000_u32)
            .map(|id| record_wide(id, 8, i64::from(id % 3)))
            .collect(),
    )
    .await;

    let cancellation = CancellationToken::new();
    let options = VerifyOptions::default().with_operation_options(
        OperationOptions::default().with_cancellation(cancellation.clone()),
    );
    let audit = tokio::spawn({
        let index = index.clone();
        async move { index.verify(options).await }
    });
    // The audit recomputes every record's quantization; four thousand
    // records take far longer than this delay, so the token fires mid-walk.
    tokio::time::sleep(Duration::from_millis(10)).await;
    cancellation.cancel();
    let error = audit
        .await
        .expect("audit task did not panic")
        .expect_err("cancelled audit");
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_foreground_mutations_never_produce_phantom_issues() {
    let (_backend, runtime, index) = setup(config()).await;
    insert_all(
        &index,
        (0..300_u32)
            .map(|id| record(id, id as f32, i64::from(id % 3)))
            .collect(),
    )
    .await;

    let audit = tokio::spawn({
        let index = index.clone();
        async move { index.verify(VerifyOptions::default()).await }
    });
    let mut tasks = Vec::new();
    for worker in 0..4_u32 {
        let index = index.clone();
        tasks.push(tokio::spawn(async move {
            for n in 0..25_u32 {
                let id = 1_000 + worker * 25 + n;
                index
                    .insert(record(id, id as f32, i64::from(id % 3)))
                    .await
                    .expect("insert");
            }
        }));
    }
    // The audit's pinned snapshot is a committed healthy state no matter how
    // the concurrent inserts interleave it.
    let report = audit
        .await
        .expect("audit task did not panic")
        .expect("verify");
    assert!(report.complete, "issues: {:?}", report.issues);
    assert!(report.issues.is_empty());
    for task in tasks {
        task.await.expect("insert task did not panic");
    }
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_index_verify_fails_closed() {
    let (_backend, runtime, index) = setup(config()).await;
    runtime.drop_index("verify").await.expect("drop index");
    let error = index
        .verify(VerifyOptions::default())
        .await
        .expect_err("dropped index");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn corruption_in_one_index_does_not_leak_into_another() {
    let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()));
    let runtime =
        Runtime::new(backend.clone(), support::manual_maintenance_config()).expect("runtime");
    let first = runtime
        .create_index("first", config())
        .await
        .expect("create first");
    let second = runtime
        .create_index("second", config())
        .await
        .expect("create second");
    insert_all(&first, vec![record(0, 0.0, 1)]).await;
    insert_all(&second, vec![record(1, 1.0, 1)]).await;

    let mut key = keys::manifest_key(second.logical_index_id());
    key.push(0x00);
    raw_put(&backend, key, Bytes::from_static(b"garbage")).await;

    let clean = first
        .verify(VerifyOptions::default())
        .await
        .expect("verify first");
    assert!(clean.complete && clean.issues.is_empty());
    let dirty = second
        .verify(VerifyOptions::default())
        .await
        .expect("verify second");
    assert_eq!(kinds(&dirty), vec![VerifyIssueKind::InvalidEncoding]);
    assert_eq!(dirty.issues[0].logical_index_id, second.logical_index_id());
    runtime.shutdown().await.expect("shutdown");
}
