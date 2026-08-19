//! Typed atomic record-membership contract tests.

use std::collections::{BTreeMap, BTreeSet, btree_map};

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, LogicalIndexId, Metric, PartitionKey,
    Value,
};
use ktann::storage::backend::{Backend, ScanLimits};
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::membership::{self, DeleteOutcome};
use ktann::storage::values::{
    ChildEntry, IndexLifecycle, IndexManifest, LeafEntry, OpaquePayload, PartitionHeader,
    PartitionState, PartitionSynopsis, PersistentValue, RecordLocation, VectorRecord,
};
use ktann::storage::{
    LogicalRange, ReadLogicalTxn, RecordGroupRead, WriteLogicalTxn, tree_manifest,
};

use support::{CommitFault, CommitOutcome, DeterministicBackend, Rng};

#[allow(dead_code)]
mod support;

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
}

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

/// A one-dimensional L2 index with one I64 field that is also the Tree Key.
fn manifest() -> IndexManifest {
    let config = IndexConfig::new(1, Metric::L2)
        .expect("valid config")
        .with_fields(vec![FieldSchema::new("a", DataType::I64).expect("field")])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields");
    IndexManifest::new(IndexLifecycle::Active, id(7), config, [7; 32], vec![None])
        .expect("valid manifest")
}

fn tree_key(value: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(value)]).expect("canonical key")
}

fn rid(value: u8) -> Bytes {
    Bytes::copy_from_slice(&[b'r', value])
}

fn record(rid: &[u8], field: i64) -> VectorRecord {
    VectorRecord::new(
        Bytes::copy_from_slice(rid),
        vec![1.0_f32],
        vec![Value::I64(field)],
    )
}

/// A legal dimension-1 Leaf Entry; the all-zero code is the canonical encoding
/// of the zero vector, matching the established storage-operations fixture.
fn entry(rid: &[u8], field: i64) -> LeafEntry {
    LeafEntry::new(
        Bytes::copy_from_slice(rid),
        vec![Value::I64(field)],
        Bytes::from_static(&[0; 14]),
    )
}

fn location(tree_key: &TreeKey, leaf: u64) -> RecordLocation {
    RecordLocation::new(tree_key.clone(), pk(leaf))
}

fn payload(tag: &'static [u8]) -> OpaquePayload {
    OpaquePayload::new(Bytes::from_static(tag)).expect("bounded payload")
}

async fn write_txn<'b, 'm>(
    backend: &'b DeterministicBackend,
    manifest: &'m IndexManifest,
) -> WriteLogicalTxn<'m, <DeterministicBackend as Backend>::WriteTxn<'b>> {
    let raw = backend.begin_write().await.expect("begin write");
    WriteLogicalTxn::for_index(
        raw,
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind manifest")
}

async fn read_txn<'b, 'm>(
    backend: &'b DeterministicBackend,
    manifest: &'m IndexManifest,
) -> ReadLogicalTxn<'m, <DeterministicBackend as Backend>::ReadTxn<'b>> {
    let raw = backend.begin_read().await.expect("begin read");
    ReadLogicalTxn::for_index(raw, manifest).expect("bind manifest")
}

async fn create_tree(backend: &DeterministicBackend, manifest: &IndexManifest, key: &TreeKey) {
    let mut txn = write_txn(backend, manifest).await;
    tree_manifest::create_tree(&mut txn, key, 0)
        .await
        .expect("create tree");
    txn.commit().await.expect("commit tree");
}

/// Seeds the grown root shape: internal root PK 1 at level 2 with leaf
/// children PK 2 (centroid 0.0) and PK 3 (centroid 10.0), each with its Header
/// and empty Synopsis installed.
async fn seed_grown_root(backend: &DeterministicBackend, manifest: &IndexManifest, key: &TreeKey) {
    create_tree(backend, manifest, key).await;
    let mut txn = write_txn(backend, manifest).await;
    let header_key = |partition| LogicalKey::Header {
        index: id(7),
        tree_key: key.clone(),
        partition,
    };
    let header = |level| {
        PersistentValue::PartitionHeader(
            PartitionHeader::new(level, 0, 0, PartitionState::Ready).expect("header"),
        )
    };
    txn.put(header_key(pk(1)), header(2))
        .await
        .expect("root header");
    txn.put(header_key(pk(2)), header(1))
        .await
        .expect("left header");
    txn.put(header_key(pk(3)), header(1))
        .await
        .expect("right header");
    for (partition, centroid) in [(pk(2), 0.0_f32), (pk(3), 10.0_f32)] {
        txn.put(
            LogicalKey::ChildEntry {
                index: id(7),
                tree_key: key.clone(),
                partition: pk(1),
                child: partition,
            },
            PersistentValue::ChildEntry(ChildEntry::new(partition, vec![centroid])),
        )
        .await
        .expect("child edge");
        txn.put(
            LogicalKey::Synopsis {
                index: id(7),
                tree_key: key.clone(),
                partition,
            },
            PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(manifest)),
        )
        .await
        .expect("leaf synopsis");
    }
    txn.commit().await.expect("commit grown root");
}

async fn insert_committed(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    record: &VectorRecord,
    payload: Option<&OpaquePayload>,
    target: &RecordLocation,
    entry: &LeafEntry,
) {
    let mut txn = write_txn(backend, manifest).await;
    membership::insert_record(&mut txn, record, payload, target, entry)
        .await
        .expect("insert");
    txn.commit().await.expect("commit insert");
}

async fn read_header_at(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> PartitionHeader {
    let mut txn = read_txn(backend, manifest).await;
    match txn
        .get(LogicalKey::Header {
            index: id(7),
            tree_key: tree_key.clone(),
            partition,
        })
        .await
        .expect("read header")
    {
        Some(PersistentValue::PartitionHeader(header)) => header,
        _ => panic!("committed partition header must exist"),
    }
}

async fn read_synopsis_at(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> PartitionSynopsis {
    let mut txn = read_txn(backend, manifest).await;
    match txn
        .get(LogicalKey::Synopsis {
            index: id(7),
            tree_key: tree_key.clone(),
            partition,
        })
        .await
        .expect("read synopsis")
    {
        Some(PersistentValue::PartitionSynopsis(synopsis)) => synopsis,
        _ => panic!("committed partition synopsis must exist"),
    }
}

async fn read_entry_at(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    location: &RecordLocation,
    record_id: &Bytes,
) -> Option<LeafEntry> {
    let mut txn = read_txn(backend, manifest).await;
    match txn
        .get(LogicalKey::LeafEntry {
            index: id(7),
            tree_key: location.tree_key().clone(),
            partition: location.leaf(),
            id: record_id.clone(),
        })
        .await
        .expect("read entry")
    {
        Some(PersistentValue::LeafEntry(entry)) => Some(entry),
        None => None,
        _ => panic!("a Leaf Entry key holds a Leaf Entry value"),
    }
}

async fn read_group(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    record_id: Bytes,
    include_payload: bool,
) -> Option<RecordGroupRead> {
    read_txn(backend, manifest)
        .await
        .read_record_group(record_id, include_payload)
        .await
        .expect("read group")
}

async fn leaf_member_ids(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> BTreeSet<Bytes> {
    let mut txn = read_txn(backend, manifest).await;
    let range = LogicalRange::leaf_entries(manifest, tree_key, partition).expect("leaf range");
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

#[tokio::test]
async fn insert_commits_the_complete_record_group() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;

    let target = location(&key, 1);
    insert_committed(
        &backend,
        &manifest,
        &record(&rid(1), 5),
        Some(&payload(b"p1")),
        &target,
        &entry(&rid(1), 5),
    )
    .await;

    let group = read_group(&backend, &manifest, rid(1), true)
        .await
        .expect("record exists");
    assert_eq!(group.record(), &record(&rid(1), 5));
    assert_eq!(group.location(), &target);
    assert_eq!(group.payload(), Some(&payload(b"p1")));
    assert_eq!(
        read_entry_at(&backend, &manifest, &target, &rid(1)).await,
        Some(entry(&rid(1), 5))
    );

    let header = read_header_at(&backend, &manifest, &key, pk(1)).await;
    assert_eq!(
        header,
        PartitionHeader::new(1, 1, 1, PartitionState::Ready).expect("header")
    );

    let synopsis = read_synopsis_at(&backend, &manifest, &key, pk(1)).await;
    assert!(!synopsis.fields()[0].has_null());
    assert_eq!(synopsis.fields()[0].minimum(), Some(&Value::I64(5)));
    assert_eq!(synopsis.fields()[0].maximum(), Some(&Value::I64(5)));
}

#[tokio::test]
async fn duplicate_insert_is_record_already_exists_and_changes_nothing() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;
    let target = location(&key, 1);
    insert_committed(
        &backend,
        &manifest,
        &record(&rid(1), 5),
        None,
        &target,
        &entry(&rid(1), 5),
    )
    .await;

    let mut txn = write_txn(&backend, &manifest).await;
    let error = membership::insert_record(
        &mut txn,
        &record(&rid(1), 9),
        None,
        &target,
        &entry(&rid(1), 9),
    )
    .await
    .expect_err("duplicate Record ID");
    assert_eq!(error.kind(), ErrorKind::RecordAlreadyExists);
    // The Record unique insert runs first, so nothing else was staged.
    assert_eq!(txn.size().mutations(), 0);
    txn.rollback().await;

    let group = read_group(&backend, &manifest, rid(1), false)
        .await
        .expect("record exists");
    assert_eq!(group.record(), &record(&rid(1), 5));
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 1, 1, PartitionState::Ready).expect("header")
    );
}

#[tokio::test]
async fn same_leaf_replace_rewrites_entry_payload_and_epoch() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;
    let target = location(&key, 1);
    insert_committed(
        &backend,
        &manifest,
        &record(&rid(1), 5),
        Some(&payload(b"p1")),
        &target,
        &entry(&rid(1), 5),
    )
    .await;

    // A payload-carrying replacement updates the entry and swaps the payload.
    {
        let mut txn = write_txn(&backend, &manifest).await;
        membership::replace_record(
            &mut txn,
            &record(&rid(1), 7),
            Some(&payload(b"p2")),
            &target,
            &target,
            &entry(&rid(1), 7),
        )
        .await
        .expect("replace");
        txn.commit().await.expect("commit replace");
    }

    let group = read_group(&backend, &manifest, rid(1), true)
        .await
        .expect("record exists");
    assert_eq!(group.record(), &record(&rid(1), 7));
    assert_eq!(group.location(), &target);
    assert_eq!(group.payload(), Some(&payload(b"p2")));
    assert_eq!(
        read_entry_at(&backend, &manifest, &target, &rid(1)).await,
        Some(entry(&rid(1), 7))
    );
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 1, 2, PartitionState::Ready).expect("header")
    );

    // A payload-less replacement deletes the old payload.
    {
        let mut txn = write_txn(&backend, &manifest).await;
        membership::replace_record(
            &mut txn,
            &record(&rid(1), 9),
            None,
            &target,
            &target,
            &entry(&rid(1), 9),
        )
        .await
        .expect("replace");
        txn.commit().await.expect("commit replace");
    }

    let group = read_group(&backend, &manifest, rid(1), true)
        .await
        .expect("record exists");
    assert_eq!(group.record(), &record(&rid(1), 9));
    assert_eq!(group.payload(), None);
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 1, 3, PartitionState::Ready).expect("header")
    );
}

#[tokio::test]
async fn cross_leaf_and_cross_tree_moves_retarget_membership_and_counts() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_grown_root(&backend, &manifest, &key).await;
    let source = location(&key, 2);
    let target = location(&key, 3);
    insert_committed(
        &backend,
        &manifest,
        &record(&rid(1), 5),
        None,
        &source,
        &entry(&rid(1), 5),
    )
    .await;
    let source_synopsis_before = read_synopsis_at(&backend, &manifest, &key, pk(2)).await;

    // A same-tree cross-leaf move from PK 2 to PK 3.
    let mut txn = write_txn(&backend, &manifest).await;
    membership::replace_record(
        &mut txn,
        &record(&rid(1), 5),
        None,
        &source,
        &target,
        &entry(&rid(1), 5),
    )
    .await
    .expect("move");
    txn.commit().await.expect("commit move");

    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(2)).await,
        PartitionHeader::new(1, 0, 2, PartitionState::Ready).expect("header")
    );
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(3)).await,
        PartitionHeader::new(1, 1, 1, PartitionState::Ready).expect("header")
    );
    // The internal root is not part of the membership mutation.
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(2, 0, 0, PartitionState::Ready).expect("header")
    );

    let group = read_group(&backend, &manifest, rid(1), false)
        .await
        .expect("record exists");
    assert_eq!(group.location(), &target);
    assert_eq!(
        read_entry_at(&backend, &manifest, &source, &rid(1)).await,
        None
    );
    assert_eq!(
        read_entry_at(&backend, &manifest, &target, &rid(1)).await,
        Some(entry(&rid(1), 5))
    );

    // Synopses are monotone: the source keeps its historical state
    // byte-identical while the target expands with the moved projection.
    assert_eq!(
        read_synopsis_at(&backend, &manifest, &key, pk(2)).await,
        source_synopsis_before
    );
    let target_synopsis = read_synopsis_at(&backend, &manifest, &key, pk(3)).await;
    assert_eq!(target_synopsis.fields()[0].minimum(), Some(&Value::I64(5)));
    assert_eq!(target_synopsis.fields()[0].maximum(), Some(&Value::I64(5)));

    // A cross-tree move carries a second record from the first tree's PK 2
    // leaf (now empty) into the second tree's root.
    let other = tree_key(2);
    create_tree(&backend, &manifest, &other).await;
    let other_target = location(&other, 1);
    insert_committed(
        &backend,
        &manifest,
        &record(&rid(2), 3),
        None,
        &source,
        &entry(&rid(2), 3),
    )
    .await;
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(2)).await,
        PartitionHeader::new(1, 1, 3, PartitionState::Ready).expect("header")
    );

    let mut txn = write_txn(&backend, &manifest).await;
    membership::replace_record(
        &mut txn,
        &record(&rid(2), 3),
        None,
        &source,
        &other_target,
        &entry(&rid(2), 3),
    )
    .await
    .expect("cross-tree move");
    txn.commit().await.expect("commit move");

    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(2)).await,
        PartitionHeader::new(1, 0, 4, PartitionState::Ready).expect("header")
    );
    assert_eq!(
        read_header_at(&backend, &manifest, &other, pk(1)).await,
        PartitionHeader::new(1, 1, 1, PartitionState::Ready).expect("header")
    );
    let group = read_group(&backend, &manifest, rid(2), false)
        .await
        .expect("record exists");
    assert_eq!(group.location(), &other_target);
    assert_eq!(
        read_entry_at(&backend, &manifest, &source, &rid(2)).await,
        None
    );
    assert_eq!(
        read_entry_at(&backend, &manifest, &other_target, &rid(2)).await,
        Some(entry(&rid(2), 3))
    );
    assert_eq!(
        read_synopsis_at(&backend, &manifest, &other, pk(1))
            .await
            .fields()[0]
            .minimum(),
        Some(&Value::I64(3))
    );

    // The first move's membership is undisturbed by the cross-tree move.
    let group = read_group(&backend, &manifest, rid(1), false)
        .await
        .expect("record exists");
    assert_eq!(group.location(), &target);
    assert_eq!(
        read_entry_at(&backend, &manifest, &target, &rid(1)).await,
        Some(entry(&rid(1), 5))
    );
}

#[tokio::test]
async fn delete_removes_the_whole_group_and_is_idempotent() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;
    let target = location(&key, 1);
    insert_committed(
        &backend,
        &manifest,
        &record(&rid(1), 5),
        Some(&payload(b"p1")),
        &target,
        &entry(&rid(1), 5),
    )
    .await;

    let mut txn = write_txn(&backend, &manifest).await;
    assert_eq!(
        membership::delete_record(&mut txn, &rid(1))
            .await
            .expect("delete"),
        DeleteOutcome::Deleted
    );
    txn.commit().await.expect("commit delete");

    assert_eq!(read_group(&backend, &manifest, rid(1), true).await, None);
    assert_eq!(
        read_entry_at(&backend, &manifest, &target, &rid(1)).await,
        None
    );
    let payload_value = read_txn(&backend, &manifest)
        .await
        .get(LogicalKey::Payload {
            index: id(7),
            id: rid(1),
        })
        .await
        .expect("read payload");
    assert_eq!(payload_value, None);
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 0, 2, PartitionState::Ready).expect("header")
    );

    // The second delete is a no-op that stages nothing.
    let mut txn = write_txn(&backend, &manifest).await;
    assert_eq!(
        membership::delete_record(&mut txn, &rid(1))
            .await
            .expect("idempotent delete"),
        DeleteOutcome::NotFound
    );
    assert_eq!(txn.size().mutations(), 0);
    txn.rollback().await;
}

#[tokio::test]
async fn delete_uses_the_stored_location_not_routing() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_grown_root(&backend, &manifest, &key).await;
    let stored = location(&key, 2);
    insert_committed(
        &backend,
        &manifest,
        &record(&rid(1), 5),
        None,
        &stored,
        &entry(&rid(1), 5),
    )
    .await;

    let mut txn = write_txn(&backend, &manifest).await;
    assert_eq!(
        membership::delete_record(&mut txn, &rid(1))
            .await
            .expect("delete"),
        DeleteOutcome::Deleted
    );
    txn.commit().await.expect("commit delete");

    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(2)).await,
        PartitionHeader::new(1, 0, 2, PartitionState::Ready).expect("header")
    );
    // No other partition is touched.
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(3)).await,
        PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("header")
    );
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(2, 0, 0, PartitionState::Ready).expect("header")
    );
}

#[tokio::test]
async fn rollback_discards_the_whole_mutation() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;

    let mut txn = write_txn(&backend, &manifest).await;
    membership::insert_record(
        &mut txn,
        &record(&rid(1), 5),
        Some(&payload(b"p1")),
        &location(&key, 1),
        &entry(&rid(1), 5),
    )
    .await
    .expect("insert");
    txn.rollback().await;

    assert_eq!(read_group(&backend, &manifest, rid(1), true).await, None);
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("header")
    );
}

#[tokio::test]
async fn concurrent_leaf_mutations_conflict_and_retry_cleanly() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;
    let target = location(&key, 1);

    // Both transactions update-protect the same leaf Header.
    let mut first = write_txn(&backend, &manifest).await;
    membership::insert_record(
        &mut first,
        &record(&rid(1), 1),
        None,
        &target,
        &entry(&rid(1), 1),
    )
    .await
    .expect("first insert");
    let mut second = write_txn(&backend, &manifest).await;
    membership::insert_record(
        &mut second,
        &record(&rid(2), 2),
        None,
        &target,
        &entry(&rid(2), 2),
    )
    .await
    .expect("second insert");

    first.commit().await.expect("first commit wins");
    let error = second.commit().await.expect_err("header conflict");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // Exactly the winner's membership is committed, with the exact count.
    assert!(
        read_group(&backend, &manifest, rid(1), false)
            .await
            .is_some()
    );
    assert_eq!(read_group(&backend, &manifest, rid(2), false).await, None);
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 1, 1, PartitionState::Ready).expect("header")
    );

    // The retried attempt observes the winner and commits cleanly.
    let mut retry = write_txn(&backend, &manifest).await;
    membership::insert_record(
        &mut retry,
        &record(&rid(2), 2),
        None,
        &target,
        &entry(&rid(2), 2),
    )
    .await
    .expect("retried insert");
    retry.commit().await.expect("retry commits");
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 2, 2, PartitionState::Ready).expect("header")
    );
}

#[tokio::test]
async fn unknown_commit_outcomes_surface_and_preserve_membership() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;
    let target = location(&key, 1);

    // Unknown-applied: the mutation lands but the outcome is unknown.
    backend
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("fault plan");
    let mut txn = write_txn(&backend, &manifest).await;
    membership::insert_record(
        &mut txn,
        &record(&rid(1), 1),
        None,
        &target,
        &entry(&rid(1), 1),
    )
    .await
    .expect("insert");
    let error = txn.commit().await.expect_err("unknown outcome");
    assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    assert_eq!(
        backend.history().last().expect("history").outcome,
        CommitOutcome::UnknownApplied
    );
    assert!(
        read_group(&backend, &manifest, rid(1), false)
            .await
            .is_some()
    );
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 1, 1, PartitionState::Ready).expect("header")
    );
    assert_eq!(
        leaf_member_ids(&backend, &manifest, &key, pk(1)).await,
        BTreeSet::from([rid(1)])
    );

    // Unknown-not-applied: nothing lands but the outcome is unknown.
    backend
        .set_fault_plan(vec![CommitFault::UnknownNotApplied])
        .expect("fault plan");
    let mut txn = write_txn(&backend, &manifest).await;
    membership::insert_record(
        &mut txn,
        &record(&rid(2), 2),
        None,
        &target,
        &entry(&rid(2), 2),
    )
    .await
    .expect("insert");
    let error = txn.commit().await.expect_err("unknown outcome");
    assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    assert_eq!(
        backend.history().last().expect("history").outcome,
        CommitOutcome::UnknownNotApplied
    );
    assert_eq!(read_group(&backend, &manifest, rid(2), false).await, None);
    assert_eq!(
        read_header_at(&backend, &manifest, &key, pk(1)).await,
        PartitionHeader::new(1, 1, 1, PartitionState::Ready).expect("header")
    );
    assert_eq!(
        leaf_member_ids(&backend, &manifest, &key, pk(1)).await,
        BTreeSet::from([rid(1)])
    );
}

#[tokio::test]
async fn invalid_caller_input_is_rejected_before_any_write() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;
    let target = location(&key, 1);

    // Record and Leaf Entry identities must agree.
    let mut txn = write_txn(&backend, &manifest).await;
    let error = membership::insert_record(
        &mut txn,
        &record(&rid(1), 5),
        None,
        &target,
        &entry(&rid(9), 5),
    )
    .await
    .expect_err("mismatched identity");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(txn.size().mutations(), 0);
    txn.rollback().await;

    // An unbound transaction has no Logical Index to mutate.
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn =
        WriteLogicalTxn::bootstrap(raw, backend.hard_limits(), backend.admission_budget());
    let error = membership::insert_record(
        &mut txn,
        &record(&rid(1), 5),
        None,
        &target,
        &entry(&rid(1), 5),
    )
    .await
    .expect_err("unbound transaction");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    let error = membership::delete_record(&mut txn, &rid(1))
        .await
        .expect_err("unbound transaction");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    txn.rollback().await;
}

#[tokio::test]
async fn replace_fails_closed_on_absent_or_stale_membership() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_grown_root(&backend, &manifest, &key).await;
    let source = location(&key, 2);
    insert_committed(
        &backend,
        &manifest,
        &record(&rid(1), 5),
        None,
        &source,
        &entry(&rid(1), 5),
    )
    .await;

    // Replacing a record that does not exist is corruption, not an insert.
    let mut txn = write_txn(&backend, &manifest).await;
    let error = membership::replace_record(
        &mut txn,
        &record(&rid(9), 5),
        None,
        &source,
        &source,
        &entry(&rid(9), 5),
    )
    .await
    .expect_err("absent record");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;

    // A stale expected location mismatches the authoritative stored location.
    let mut txn = write_txn(&backend, &manifest).await;
    let error = membership::replace_record(
        &mut txn,
        &record(&rid(1), 6),
        None,
        &location(&key, 3),
        &location(&key, 3),
        &entry(&rid(1), 6),
    )
    .await
    .expect_err("stale location");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;

    let group = read_group(&backend, &manifest, rid(1), false)
        .await
        .expect("record exists");
    assert_eq!(group.record(), &record(&rid(1), 5));
    assert_eq!(group.location(), &source);
}

#[tokio::test]
async fn a_target_that_no_longer_accepts_writes_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_tree(&backend, &manifest, &key).await;

    // A Merging leaf is stale topology for a foreground write target.
    let mut txn = write_txn(&backend, &manifest).await;
    txn.put(
        LogicalKey::Header {
            index: id(7),
            tree_key: key.clone(),
            partition: pk(1),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 0, 0, PartitionState::Merging).expect("header"),
        ),
    )
    .await
    .expect("put merging header");
    txn.commit().await.expect("commit header");

    let mut txn = write_txn(&backend, &manifest).await;
    let error = membership::insert_record(
        &mut txn,
        &record(&rid(1), 5),
        None,
        &location(&key, 1),
        &entry(&rid(1), 5),
    )
    .await
    .expect_err("merging target");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;

    assert_eq!(read_group(&backend, &manifest, rid(1), false).await, None);
}

/// Verifies the committed state against the model: every modeled record reads
/// back consistently, every absent Record ID reads back absent, and each
/// leaf's Header count and scanned Leaf Entry membership match the model's
/// per-leaf membership exactly.
async fn verify_model(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    model: &BTreeMap<u8, (i64, RecordLocation)>,
    leaves: &[(TreeKey, u64)],
    id_space: u8,
) {
    for value in 0..id_space {
        let record_id = rid(value);
        match model.get(&value) {
            Some((field, location)) => {
                let group = read_group(backend, manifest, record_id.clone(), false)
                    .await
                    .expect("modeled record exists");
                assert_eq!(group.record(), &record(&record_id, *field));
                assert_eq!(group.location(), location);
                assert_eq!(
                    read_entry_at(backend, manifest, location, &record_id).await,
                    Some(entry(&record_id, *field))
                );
            }
            None => assert_eq!(read_group(backend, manifest, record_id, false).await, None),
        }
    }
    for (tree_key, leaf) in leaves {
        let expected: BTreeSet<Bytes> = model
            .iter()
            .filter(|(_, (_, location))| {
                location.tree_key() == tree_key && location.leaf() == pk(*leaf)
            })
            .map(|(value, _)| rid(*value))
            .collect();
        let header = read_header_at(backend, manifest, tree_key, pk(*leaf)).await;
        assert_eq!(
            usize::try_from(header.entry_count()).expect("count fits"),
            expected.len(),
            "exact leaf count"
        );
        assert_eq!(
            leaf_member_ids(backend, manifest, tree_key, pk(*leaf)).await,
            expected,
            "exact leaf membership"
        );
    }
}

#[tokio::test]
async fn membership_matches_a_seeded_model() {
    const SEEDS: u64 = 8;
    const OPS: u64 = 128;
    const ID_SPACE: u8 = 12;

    for seed in 1..=SEEDS {
        let backend = DeterministicBackend::default();
        let manifest = manifest();
        let first = tree_key(1);
        let second = tree_key(2);
        seed_grown_root(&backend, &manifest, &first).await;
        create_tree(&backend, &manifest, &second).await;
        let leaves = [(first.clone(), 2), (first, 3), (second, 1)];
        let mut model: BTreeMap<u8, (i64, RecordLocation)> = BTreeMap::new();
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);

        for _ in 0..OPS {
            let record_id = u8::try_from(rng.below(u64::from(ID_SPACE))).expect("id space");
            let field = i64::try_from(rng.below(1_000)).expect("field range");
            let target = {
                let (tree_key, leaf) = &leaves[usize::try_from(rng.below(3)).expect("leaf count")];
                location(tree_key, *leaf)
            };
            match rng.below(4) {
                0 | 1 => {
                    let mut txn = write_txn(&backend, &manifest).await;
                    let result = membership::insert_record(
                        &mut txn,
                        &record(&rid(record_id), field),
                        None,
                        &target,
                        &entry(&rid(record_id), field),
                    )
                    .await;
                    match model.entry(record_id) {
                        btree_map::Entry::Occupied(_) => {
                            let error = result.expect_err("duplicate insert");
                            assert_eq!(error.kind(), ErrorKind::RecordAlreadyExists, "seed {seed}");
                            txn.rollback().await;
                        }
                        btree_map::Entry::Vacant(entry) => {
                            result.expect("insert");
                            txn.commit().await.expect("commit insert");
                            entry.insert((field, target));
                        }
                    }
                }
                2 => {
                    let mut txn = write_txn(&backend, &manifest).await;
                    let Some((_, expected)) = model.get(&record_id).cloned() else {
                        let error = membership::replace_record(
                            &mut txn,
                            &record(&rid(record_id), field),
                            None,
                            &target,
                            &target,
                            &entry(&rid(record_id), field),
                        )
                        .await
                        .expect_err("replace of an absent record");
                        assert_eq!(error.kind(), ErrorKind::Corruption, "seed {seed}");
                        txn.rollback().await;
                        continue;
                    };
                    // Half the replacements stay in the same leaf; the rest
                    // move to a randomly chosen leaf in either tree.
                    let target = if rng.below(2) == 0 {
                        expected.clone()
                    } else {
                        target
                    };
                    membership::replace_record(
                        &mut txn,
                        &record(&rid(record_id), field),
                        None,
                        &expected,
                        &target,
                        &entry(&rid(record_id), field),
                    )
                    .await
                    .expect("replace");
                    txn.commit().await.expect("commit replace");
                    model.insert(record_id, (field, target));
                }
                _ => {
                    let mut txn = write_txn(&backend, &manifest).await;
                    let outcome = membership::delete_record(&mut txn, &rid(record_id))
                        .await
                        .expect("delete");
                    let expected = if model.remove(&record_id).is_some() {
                        DeleteOutcome::Deleted
                    } else {
                        DeleteOutcome::NotFound
                    };
                    assert_eq!(outcome, expected, "seed {seed}");
                    txn.commit().await.expect("commit delete");
                }
            }
            verify_model(&backend, &manifest, &model, &leaves, ID_SPACE).await;
        }
    }
}
