//! Searchable split state machine contract tests (#10).
//!
//! Every committed split phase must stay searchable and preserve exact
//! membership; moves atomically update target/source state and Record
//! Location; work per transaction is bounded; completion uses exact zero
//! counts; crashes and conflicts at every transition are covered.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, Index, IndexConfig, Metric, PartitionKey, Record,
    RuntimeConfig, Value,
};
use ktann::maintenance::routing::route_leaf;
use ktann::maintenance::split::{self, Advance};
use ktann::maintenance::training::train_split_centroids;
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::backend::{Backend, Capabilities, ScanLimits, WriteTxn};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexManifest, LeafEntry, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionSynopsis, PartitionTransition, PersistentValue, RecordLocation,
};
use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn, topology, tree_manifest};

use support::{
    CommitFault, DeterministicBackend, DeterministicConfig, Rng, SharedBackend, read_manifest,
};

#[allow(dead_code)]
mod support;

fn backend() -> SharedBackend {
    SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()))
}

fn backend_with_clear() -> SharedBackend {
    let config = DeterministicConfig {
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        ..DeterministicConfig::default()
    };
    SharedBackend::new(DeterministicBackend::new(config))
}

/// A one-dimensional L2 index over one i64 tree-key field with a maximum of
/// four entries per partition, so small fixtures trigger splits.
fn config() -> IndexConfig {
    IndexConfig::new(1, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(1, 4)
        .expect("valid partition entries")
}

fn make_runtime(backend: SharedBackend) -> Runtime<SharedBackend> {
    // These suites drive the state machines by hand; background maintenance
    // workers would race the manual drives.
    Runtime::new(backend, support::manual_maintenance_config()).expect("runtime is valid")
}

fn retry() -> RetryPolicy {
    RetryPolicy::for_fixup(&RuntimeConfig::default())
}

fn tree_key(bucket: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(bucket)]).expect("canonical key")
}

fn rid(value: u8) -> Bytes {
    Bytes::copy_from_slice(&[b'r', value])
}

fn record(id: &[u8], x: f32, bucket: i64) -> Record {
    Record::new(
        Bytes::copy_from_slice(id),
        Arc::from([x]),
        vec![Value::I64(bucket)],
    )
    .expect("valid record")
}

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

async fn read_txn<'b, 'm>(
    backend: &'b SharedBackend,
    manifest: &'m IndexManifest,
) -> ReadLogicalTxn<'m, <SharedBackend as Backend>::ReadTxn<'b>> {
    let raw = backend.begin_read().await.expect("begin read");
    ReadLogicalTxn::for_index(raw, manifest).expect("bind manifest")
}

async fn write_txn<'b, 'm>(
    backend: &'b SharedBackend,
    manifest: &'m IndexManifest,
) -> WriteLogicalTxn<'m, <SharedBackend as Backend>::WriteTxn<'b>> {
    let raw = backend.begin_write().await.expect("begin write");
    WriteLogicalTxn::for_index(
        raw,
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind manifest")
}

/// Installs the Tree Manifest and initial leaf root so fixtures can grow the
/// root shape from a committed empty root.
async fn create_committed_tree(backend: &SharedBackend, manifest: &IndexManifest, key: &TreeKey) {
    let mut txn = write_txn(backend, manifest).await;
    tree_manifest::create_tree(&mut txn, key, 100)
        .await
        .expect("create tree");
    txn.commit().await.expect("commit tree");
}

async fn header_of(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Option<PartitionHeader> {
    match read_txn(backend, manifest)
        .await
        .get(LogicalKey::Header {
            index: manifest.logical_index_id(),
            tree_key: key.clone(),
            partition,
        })
        .await
        .expect("read header")
    {
        Some(PersistentValue::PartitionHeader(header)) => Some(header),
        None => None,
        other => panic!("wrong header kind: {other:?}"),
    }
}

async fn state_of(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Option<PartitionTransition> {
    match read_txn(backend, manifest)
        .await
        .get(LogicalKey::State {
            index: manifest.logical_index_id(),
            tree_key: key.clone(),
            partition,
        })
        .await
        .expect("read state")
    {
        Some(PersistentValue::PartitionState(state)) => Some(state),
        None => None,
        other => panic!("wrong state kind: {other:?}"),
    }
}

async fn centroid_of(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Option<PartitionCentroid> {
    match read_txn(backend, manifest)
        .await
        .get(LogicalKey::Centroid {
            index: manifest.logical_index_id(),
            tree_key: key.clone(),
            partition,
        })
        .await
        .expect("read centroid")
    {
        Some(PersistentValue::PartitionCentroid(centroid)) => Some(centroid),
        None => None,
        other => panic!("wrong centroid kind: {other:?}"),
    }
}

async fn synopsis_of(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Option<PartitionSynopsis> {
    match read_txn(backend, manifest)
        .await
        .get(LogicalKey::Synopsis {
            index: manifest.logical_index_id(),
            tree_key: key.clone(),
            partition,
        })
        .await
        .expect("read synopsis")
    {
        Some(PersistentValue::PartitionSynopsis(synopsis)) => Some(synopsis),
        None => None,
        other => panic!("wrong synopsis kind: {other:?}"),
    }
}

async fn location_of(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    id: &Bytes,
) -> Option<RecordLocation> {
    match read_txn(backend, manifest)
        .await
        .get(LogicalKey::Location {
            index: manifest.logical_index_id(),
            id: id.clone(),
        })
        .await
        .expect("read location")
    {
        Some(PersistentValue::RecordLocation(location)) => Some(location),
        None => None,
        other => panic!("wrong location kind: {other:?}"),
    }
}

async fn leaf_entry_of(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
    id: &Bytes,
) -> Option<LeafEntry> {
    match read_txn(backend, manifest)
        .await
        .get(LogicalKey::LeafEntry {
            index: manifest.logical_index_id(),
            tree_key: key.clone(),
            partition,
            id: id.clone(),
        })
        .await
        .expect("read entry")
    {
        Some(PersistentValue::LeafEntry(entry)) => Some(entry),
        None => None,
        other => panic!("wrong entry kind: {other:?}"),
    }
}

async fn edge_of(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    parent: PartitionKey,
    child: PartitionKey,
) -> Option<ChildEntry> {
    match read_txn(backend, manifest)
        .await
        .get(LogicalKey::ChildEntry {
            index: manifest.logical_index_id(),
            tree_key: key.clone(),
            partition: parent,
            child,
        })
        .await
        .expect("read edge")
    {
        Some(PersistentValue::ChildEntry(entry)) => Some(entry),
        None => None,
        other => panic!("wrong edge kind: {other:?}"),
    }
}

async fn scan_child_entries(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Vec<ChildEntry> {
    let range = LogicalRange::child_entries(manifest, key, partition).expect("range");
    let mut txn = read_txn(backend, manifest).await;
    let mut entries = Vec::new();
    let mut cursor = None;
    loop {
        let page = txn
            .scan(
                &range,
                cursor.as_ref(),
                ScanLimits {
                    item_limit: 64,
                    byte_limit: 1 << 20,
                },
            )
            .await
            .expect("scan children");
        for item in page.items() {
            match item.value() {
                PersistentValue::ChildEntry(entry) => entries.push(entry.clone()),
                other => panic!("wrong child kind: {other:?}"),
            }
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            return entries;
        }
    }
}

async fn scan_leaf_entries(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Vec<LeafEntry> {
    let range = LogicalRange::leaf_entries(manifest, key, partition).expect("range");
    let mut txn = read_txn(backend, manifest).await;
    let mut entries = Vec::new();
    let mut cursor = None;
    loop {
        let page = txn
            .scan(
                &range,
                cursor.as_ref(),
                ScanLimits {
                    item_limit: 64,
                    byte_limit: 1 << 20,
                },
            )
            .await
            .expect("scan leaves");
        for item in page.items() {
            match item.value() {
                PersistentValue::LeafEntry(entry) => entries.push(entry.clone()),
                other => panic!("wrong leaf kind: {other:?}"),
            }
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            return entries;
        }
    }
}

/// The set of leaves a traversal can reach, with exact-count validation.
///
/// Descent follows Child Entries from the root; while the root is `Splitting`
/// or `DrainingSplit` its targets are reachable only through the root's
/// persisted state, so they join the frontier there (ADR 0006). Every
/// internal body's scanned Child Entry count must equal its exact Header
/// count, and no partition may be discovered twice.
async fn reachable_leaves(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
) -> BTreeMap<PartitionKey, PartitionHeader> {
    let mut leaves = BTreeMap::new();
    let mut visited = BTreeSet::new();
    let mut frontier = vec![pk(1)];
    while let Some(partition) = frontier.pop() {
        assert!(
            visited.insert(partition),
            "partition {partition:?} discovered twice"
        );
        let header = header_of(backend, manifest, key, partition)
            .await
            .unwrap_or_else(|| panic!("partition {partition:?} must have a Header"));
        // A partition's Header and State discriminators always agree.
        let state = state_of(backend, manifest, key, partition)
            .await
            .unwrap_or_else(|| panic!("partition {partition:?} must have a State"));
        assert_eq!(header.state(), state.state(), "header/state agreement");
        if partition == pk(1) {
            match state {
                // While Splitting, root targets may not be exposed yet;
                // while Draining, they must exist.
                PartitionTransition::Splitting { left, right, .. } => {
                    for target in [left, right] {
                        if header_of(backend, manifest, key, target).await.is_some() {
                            frontier.push(target);
                        }
                    }
                }
                PartitionTransition::DrainingSplit { left, right, .. } => {
                    frontier.push(left);
                    frontier.push(right);
                }
                _ => {}
            }
        }
        if header.level() == 1 {
            let entries = scan_leaf_entries(backend, manifest, key, partition).await;
            assert_eq!(
                header.entry_count() as usize,
                entries.len(),
                "exact leaf entry count"
            );
            leaves.insert(partition, header);
        } else {
            let children = scan_child_entries(backend, manifest, key, partition).await;
            assert_eq!(
                header.entry_count() as usize,
                children.len(),
                "exact child entry count"
            );
            for entry in children {
                frontier.push(entry.child());
            }
        }
    }
    leaves
}

/// Asserts the exact-membership invariant for one tree: every record has
/// exactly one Record Location naming a reachable leaf, one corresponding
/// Leaf Entry, and the reachable leaves' exact counts sum to the record
/// count.
async fn assert_exact_membership(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    records: &[(Bytes, f32)],
) {
    let leaves = reachable_leaves(backend, manifest, key).await;
    for (id, _) in records {
        let location = location_of(backend, manifest, id)
            .await
            .unwrap_or_else(|| panic!("record must have a location"));
        assert_eq!(location.tree_key(), key, "location names the tree");
        assert!(
            leaves.contains_key(&location.leaf()),
            "location leaf must be reachable"
        );
        let entry = leaf_entry_of(backend, manifest, key, location.leaf(), id)
            .await
            .unwrap_or_else(|| panic!("leaf entry must exist"));
        assert_eq!(entry.record_id(), id);
    }
    let total: u32 = leaves.values().map(|header| header.entry_count()).sum();
    assert_eq!(total as usize, records.len(), "total membership");
}

/// Asserts that routing reaches a live leaf for every record's vector.
async fn assert_searchable(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    records: &[(Bytes, f32)],
) {
    assert_exact_membership(backend, manifest, key, records).await;
    for (_, x) in records {
        route_leaf(&mut read_txn(backend, manifest).await, key, &[*x])
            .await
            .expect("route")
            .expect("tree exists");
    }
}

/// Seeds records 0..count at x = 0.0, 1.0, ... into one tree through the
/// public mutation API.
async fn seed_records(index: &Index<SharedBackend>, bucket: i64, count: u8) -> Vec<(Bytes, f32)> {
    let mut records = Vec::new();
    for n in 0..count {
        let x = f32::from(n);
        index
            .insert(record(&rid(n), x, bucket))
            .await
            .expect("insert");
        records.push((rid(n), x));
    }
    records
}

/// Drives `advance` until the partition is no longer making progress,
/// returning the observed outcome sequence.
async fn drive_to_completion(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Vec<Advance> {
    let mut outcomes = Vec::new();
    for step in 0..1_000_u32 {
        let outcome = split::advance(
            backend,
            manifest,
            key,
            partition,
            10_000 + u64::from(step),
            &retry(),
        )
        .await
        .expect("advance");
        match outcome {
            Advance::Idle | Advance::Completed => {
                outcomes.push(outcome);
                return outcomes;
            }
            _ => outcomes.push(outcome),
        }
    }
    panic!("split did not converge in 1000 steps");
}

/// Drains one split source to exact zero, one bounded batch at a time,
/// returning the total moved entries.
async fn drain_to_zero(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    source: PartitionKey,
) -> usize {
    let mut moved_total = 0_usize;
    loop {
        match split::drain_batch(backend, manifest, key, source, &retry())
            .await
            .expect("drain")
        {
            split::DrainStep::Drained { moved, remaining } => {
                moved_total += moved;
                if remaining == 0 {
                    return moved_total;
                }
            }
            other => panic!("unexpected drain outcome {other:?}"),
        }
    }
}

/// Asserts the error kind one injected commit fault produces.
fn assert_fault_kind(fault: CommitFault, error: &ktann::api::Error) {
    let expected = match fault {
        CommitFault::Abort => ErrorKind::RetryableAbort,
        _ => ErrorKind::CommitOutcomeUnknown,
    };
    assert_eq!(error.kind(), expected, "fault {fault:?}");
}

// ---------------------------------------------------------------------------
// Lifecycle: root leaf split, end to end, with foreground interleaving.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn root_leaf_split_runs_end_to_end_and_stays_searchable() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let mut records = seed_records(&index, 1, 6).await;

    // The leaf root is above the maximum of four entries.
    let root_header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("root header");
    assert_eq!(root_header.entry_count(), 6);
    assert_eq!(root_header.state(), PartitionState::Ready);

    // Begin reserves the target pair and persists Splitting.
    let start = split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    let topology::SplitStart::Started { left, right } = start else {
        panic!("split must start, got {start:?}");
    };
    assert_eq!((left, right), (pk(2), pk(3)));
    assert_eq!(
        state_of(&backend, &manifest, &key, pk(1)).await,
        Some(PartitionTransition::Splitting {
            left,
            right,
            started_at_unix_millis: 1_000,
        })
    );
    // Every committed phase is searchable.
    assert_searchable(&backend, &manifest, &key, &records).await;

    // Re-driving begin is idempotent and never reserves a second pair.
    let again = split::begin_split(&backend, &manifest, &key, pk(1), 1_001, &retry())
        .await
        .expect("begin again");
    assert_eq!(
        again,
        topology::SplitStart::AlreadySplitting { left, right }
    );

    // Expose publishes both targets parentless, owned by the root's slot.
    let trained = train_split_centroids(&mut read_txn(&backend, &manifest).await, &key, pk(1))
        .await
        .expect("train");
    let exposed = split::expose_targets(&backend, &manifest, &key, pk(1), 1_100, &retry())
        .await
        .expect("expose");
    assert_eq!(exposed, split::TargetExposure::Exposed { left, right });
    for target in [left, right] {
        let header = header_of(&backend, &manifest, &key, target)
            .await
            .expect("target header");
        assert_eq!(header.level(), 1);
        assert_eq!(header.entry_count(), 0);
        assert_eq!(header.state(), PartitionState::ReceivingSplit);
        assert_eq!(
            state_of(&backend, &manifest, &key, target).await,
            Some(PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 1_100,
            })
        );
        // Root targets have no parent Child Entry yet.
        assert_eq!(
            edge_of(&backend, &manifest, &key, pk(1), target).await,
            None
        );
    }
    // The published centroids are the trained routing model in creation order.
    assert_eq!(
        centroid_of(&backend, &manifest, &key, left).await.as_ref(),
        Some(trained.left())
    );
    assert_eq!(
        centroid_of(&backend, &manifest, &key, right).await.as_ref(),
        Some(trained.right())
    );
    assert_searchable(&backend, &manifest, &key, &records).await;

    // While Splitting, inserts still land in the source.
    index.insert(record(&rid(6), 6.0, 1)).await.expect("insert");
    records.push((rid(6), 6.0));
    assert_eq!(
        location_of(&backend, &manifest, &rid(6))
            .await
            .expect("location")
            .leaf(),
        pk(1)
    );

    // Advance to DrainingSplit; the source count is deliberately ignored.
    let advanced = split::advance_to_draining(&backend, &manifest, &key, pk(1), 1_200, &retry())
        .await
        .expect("advance");
    assert_eq!(advanced, topology::DrainStart::Advanced);
    assert_eq!(
        state_of(&backend, &manifest, &key, pk(1)).await,
        Some(PartitionTransition::DrainingSplit {
            left,
            right,
            started_at_unix_millis: 1_200,
        })
    );
    assert_searchable(&backend, &manifest, &key, &records).await;

    // Snapshot the source entries to prove verbatim RaBitQ7 copies later.
    let before: BTreeMap<Bytes, LeafEntry> = scan_leaf_entries(&backend, &manifest, &key, pk(1))
        .await
        .into_iter()
        .map(|entry| (entry.record_id().clone(), entry))
        .collect();

    // During Draining: a new insert routes directly to the nearer target; an
    // upsert whose Record Location names the source relocates atomically; a
    // delete follows the exact location.
    index.insert(record(&rid(7), 5.5, 1)).await.expect("insert");
    records.push((rid(7), 5.5));
    let insert_leaf = location_of(&backend, &manifest, &rid(7))
        .await
        .expect("location")
        .leaf();
    assert!(
        insert_leaf == left || insert_leaf == right,
        "inserts redirect"
    );

    index
        .upsert(record(&rid(0), 0.25, 1))
        .await
        .expect("upsert");
    records[0] = (rid(0), 0.25);
    let upsert_leaf = location_of(&backend, &manifest, &rid(0))
        .await
        .expect("location")
        .leaf();
    assert!(
        upsert_leaf == left || upsert_leaf == right,
        "upsert relocates"
    );

    assert!(index.delete(rid(6)).await.expect("delete"));
    records.retain(|(id, _)| *id != rid(6));
    assert_eq!(location_of(&backend, &manifest, &rid(6)).await, None);
    assert_searchable(&backend, &manifest, &key, &records).await;

    // Drain to exact zero in bounded batches.
    let mut moved_total = 0_usize;
    loop {
        match split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
            .await
            .expect("drain")
        {
            split::DrainStep::Drained { moved, remaining } => {
                moved_total += moved;
                assert_searchable(&backend, &manifest, &key, &records).await;
                if remaining == 0 {
                    break;
                }
            }
            other => panic!("unexpected drain outcome {other:?}"),
        }
    }
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(1))
            .await
            .expect("root header")
            .entry_count(),
        0
    );

    // Completion converts Partition Key 1 in place into a Ready internal root.
    let completed = split::complete_split(&backend, &manifest, &key, pk(1), 1_300, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::SplitCompletion::Completed);

    let root_header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("root header");
    assert_eq!(root_header.level(), 2);
    assert_eq!(root_header.entry_count(), 2);
    assert_eq!(root_header.state(), PartitionState::Ready);
    assert_eq!(
        state_of(&backend, &manifest, &key, pk(1)).await,
        Some(PartitionTransition::Ready {
            started_at_unix_millis: 1_300,
        })
    );
    // The root holds both target Child Entries with their persisted centroids.
    for target in [left, right] {
        let edge = edge_of(&backend, &manifest, &key, pk(1), target)
            .await
            .expect("target edge");
        let centroid = centroid_of(&backend, &manifest, &key, target)
            .await
            .expect("target centroid");
        assert_eq!(edge.centroid(), centroid.components());
        let target_header = header_of(&backend, &manifest, &key, target)
            .await
            .expect("target header");
        assert_eq!(target_header.state(), PartitionState::Ready);
    }
    // The obsolete leaf root Synopsis is gone.
    assert!(
        read_txn(&backend, &manifest)
            .await
            .get(LogicalKey::Synopsis {
                index: manifest.logical_index_id(),
                tree_key: key.clone(),
                partition: pk(1),
            })
            .await
            .expect("read synopsis")
            .is_none()
    );

    // Exact membership: every record moved to its nearer target with its
    // Record Location repointed and its Leaf Entry bytes copied verbatim.
    let left_centroid = centroid_of(&backend, &manifest, &key, left)
        .await
        .expect("left centroid");
    let right_centroid = centroid_of(&backend, &manifest, &key, right)
        .await
        .expect("right centroid");
    for (id, x) in &records {
        let location = location_of(&backend, &manifest, id)
            .await
            .expect("location");
        assert!(location.leaf() == left || location.leaf() == right);
        // The nearer-target rule with the Partition Key tie-break.
        let left_distance = (x - left_centroid.components()[0]).abs();
        let right_distance = (x - right_centroid.components()[0]).abs();
        let expected = if right_distance < left_distance {
            right
        } else {
            left
        };
        assert_eq!(location.leaf(), expected, "nearer target for x = {x}");
        let entry = leaf_entry_of(&backend, &manifest, &key, location.leaf(), id)
            .await
            .expect("entry");
        // The upsert re-encoded r0's vector; every other entry moved with
        // its absolute RaBitQ7 payload copied verbatim.
        if id != &rid(0) {
            if let Some(before_entry) = before.get(id) {
                assert_eq!(&entry, before_entry, "RaBitQ7 payload copied verbatim");
            }
        }
    }
    // The drain moved every snapshotted source entry except the upsert's
    // relocation and the delete.
    assert_eq!(
        moved_total,
        before.len() - 2,
        "drain moved every source entry"
    );
    assert_searchable(&backend, &manifest, &key, &records).await;

    // Completion is idempotent: the source is no longer Draining.
    let again = split::complete_split(&backend, &manifest, &key, pk(1), 1_400, &retry())
        .await
        .expect("complete again");
    assert_eq!(again, topology::SplitCompletion::NotDraining);

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Lifecycle: non-root leaf split installs edges in the parent and completes.
// ---------------------------------------------------------------------------

async fn split_root_into_two_leaves(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
) -> (PartitionKey, PartitionKey) {
    split::begin_split(backend, manifest, key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    split::expose_targets(backend, manifest, key, pk(1), 1_100, &retry())
        .await
        .expect("expose");
    split::advance_to_draining(backend, manifest, key, pk(1), 1_200, &retry())
        .await
        .expect("advance");
    drain_to_zero(backend, manifest, key, pk(1)).await;
    let completed = split::complete_split(backend, manifest, key, pk(1), 1_300, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::SplitCompletion::Completed);
    (pk(2), pk(3))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_root_leaf_split_installs_edges_and_removes_the_source() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let mut records = seed_records(&index, 1, 6).await;

    // Root split: leaves pk 2 (x in {0,1,2}) and pk 3 (x in {3,4,5}).
    let (left_leaf, _) = split_root_into_two_leaves(&backend, &manifest, &key).await;
    assert_searchable(&backend, &manifest, &key, &records).await;

    // Grow the left leaf past the maximum: x in {0.25, 0.75, 1.25} route left.
    for (n, x) in [(10, 0.25), (11, 0.75), (12, 1.25)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let source_header = header_of(&backend, &manifest, &key, left_leaf)
        .await
        .expect("source header");
    assert_eq!(source_header.entry_count(), 6);

    // Begin: targets come from the never-reused allocator after the root
    // split's reservations.
    let start = split::begin_split(&backend, &manifest, &key, left_leaf, 2_000, &retry())
        .await
        .expect("begin");
    let topology::SplitStart::Started { left, right } = start else {
        panic!("split must start, got {start:?}");
    };
    assert_eq!((left, right), (pk(4), pk(5)));
    assert_searchable(&backend, &manifest, &key, &records).await;

    // Expose installs each target's Child Entry in the source's parent.
    let exposed = split::expose_targets(&backend, &manifest, &key, left_leaf, 2_100, &retry())
        .await
        .expect("expose");
    assert_eq!(exposed, split::TargetExposure::Exposed { left, right });
    for target in [left, right] {
        let edge = edge_of(&backend, &manifest, &key, pk(1), target)
            .await
            .expect("target edge in parent");
        assert_eq!(edge.child(), target);
    }
    // The parent count grew exactly with the installed edges.
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(1))
            .await
            .expect("root header")
            .entry_count(),
        4
    );
    assert_searchable(&backend, &manifest, &key, &records).await;

    split::advance_to_draining(&backend, &manifest, &key, left_leaf, 2_200, &retry())
        .await
        .expect("advance");
    drain_to_zero(&backend, &manifest, &key, left_leaf).await;
    assert_searchable(&backend, &manifest, &key, &records).await;

    // Completion removes the source edge and the whole source prefix.
    let completed = split::complete_split(&backend, &manifest, &key, left_leaf, 2_300, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::SplitCompletion::Completed);
    assert_eq!(
        edge_of(&backend, &manifest, &key, pk(1), left_leaf).await,
        None,
        "source edge removed"
    );
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(1))
            .await
            .expect("root header")
            .entry_count(),
        3,
        "parent count decremented"
    );
    // The deterministic backend has no transactional range clear: the source
    // prefix was removed by bounded point deletes.
    assert_eq!(header_of(&backend, &manifest, &key, left_leaf).await, None);
    assert_eq!(state_of(&backend, &manifest, &key, left_leaf).await, None);
    assert_eq!(
        centroid_of(&backend, &manifest, &key, left_leaf).await,
        None
    );
    assert!(
        scan_leaf_entries(&backend, &manifest, &key, left_leaf)
            .await
            .is_empty()
    );
    for target in [left, right] {
        assert_eq!(
            state_of(&backend, &manifest, &key, target).await,
            Some(PartitionTransition::Ready {
                started_at_unix_millis: 2_300,
            })
        );
    }
    assert_searchable(&backend, &manifest, &key, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Lifecycle: completion with transactional range clear.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completion_uses_range_clear_when_the_backend_supports_it() {
    let backend = backend_with_clear();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 6).await;

    let (left_leaf, _) = split_root_into_two_leaves(&backend, &manifest, &key).await;
    let mut records = records;
    for (n, x) in [(10, 0.25), (11, 0.75), (12, 1.25)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let outcomes = drive_to_completion(&backend, &manifest, &key, left_leaf).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));

    assert_eq!(header_of(&backend, &manifest, &key, left_leaf).await, None);
    assert_eq!(state_of(&backend, &manifest, &key, left_leaf).await, None);
    assert_searchable(&backend, &manifest, &key, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Lifecycle: internal (Child Entry) splits, root and non-root.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn root_internal_split_moves_child_entries_and_rises_one_level() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let mut records = seed_records(&index, 1, 6).await;

    // Depth two, then grow the left leaf past the maximum twice so the root
    // accumulates four children and the next leaf split pushes it over.
    let (left_leaf, right_leaf) = split_root_into_two_leaves(&backend, &manifest, &key).await;
    for (n, x) in [(10, 0.25), (11, 0.75), (12, 1.25)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let outcomes = drive_to_completion(&backend, &manifest, &key, left_leaf).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    // Root children: {3, 4, 5}.
    for (n, x) in [(13, 3.25), (14, 3.75), (15, 4.25)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let outcomes = drive_to_completion(&backend, &manifest, &key, right_leaf).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    // Root children: {4, 5, 6, 7} — at the maximum, not above it.
    let root_header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("root header");
    assert_eq!(root_header.entry_count(), 4);
    assert_searchable(&backend, &manifest, &key, &records).await;

    // One more leaf split makes the root exceed the maximum. Two inserts near
    // x = 1.5 route to the same left-side leaf and push it over four entries.
    for (n, x) in [(16, 1.5), (17, 1.625)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let over = {
        let leaves = reachable_leaves(&backend, &manifest, &key).await;
        leaves
            .iter()
            .find(|(_, header)| header.entry_count() > 4)
            .map(|(partition, _)| *partition)
            .expect("a leaf above the maximum")
    };
    let outcomes = drive_to_completion(&backend, &manifest, &key, over).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    let root_header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("root header");
    assert_eq!(root_header.entry_count(), 5);
    assert_eq!(root_header.level(), 2);

    // The root is an internal partition above the maximum: its split moves
    // Child Entries — no Record Location or Synopsis work — and converts
    // Partition Key 1 in place to level 3.
    let outcomes = drive_to_completion(&backend, &manifest, &key, pk(1)).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    let root_header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("root header");
    assert_eq!(root_header.level(), 3);
    assert_eq!(root_header.entry_count(), 2);
    assert_eq!(root_header.state(), PartitionState::Ready);
    assert_searchable(&backend, &manifest, &key, &records).await;

    // The two level-2 targets are Ready and partition the former children.
    let children = scan_child_entries(&backend, &manifest, &key, pk(1)).await;
    assert_eq!(children.len(), 2);
    let mut child_count = 0_usize;
    for edge in children {
        let header = header_of(&backend, &manifest, &key, edge.child())
            .await
            .expect("target header");
        assert_eq!(header.state(), PartitionState::Ready);
        assert_eq!(header.level(), 2);
        child_count += header.entry_count() as usize;
    }
    assert_eq!(
        child_count, 5,
        "the five former root children moved exactly"
    );

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_root_internal_split_moves_child_entries() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);

    // A depth-3 fixture with exact counts: root PK 1 at level 3 over two
    // level-2 internals; the left internal PK 2 holds five Child Entries —
    // above the maximum of four — and is the split subject. Internal split
    // training reads Child Entry centroids only, so no Vector Records exist.
    let index_id = manifest.logical_index_id();
    {
        let mut txn = write_txn(&backend, &manifest).await;
        tree_manifest::create_tree(&mut txn, &key, 100)
            .await
            .expect("create tree");
        // Reserve the fixture's Partition Keys so the split allocates fresh
        // never-reused targets.
        let reservation = tree_manifest::reserve_partition_keys(&mut txn, &key, 17)
            .await
            .expect("reserve fixture keys");
        assert_eq!(reservation.last(), pk(18));

        let mut puts: Vec<(LogicalKey, PersistentValue)> = Vec::new();
        let header = |partition: PartitionKey, level: u32, count: u32| {
            (
                LogicalKey::Header {
                    index: index_id,
                    tree_key: key.clone(),
                    partition,
                },
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(level, count, 0, PartitionState::Ready).expect("header"),
                ),
            )
        };
        let state = |partition: PartitionKey| {
            (
                LogicalKey::State {
                    index: index_id,
                    tree_key: key.clone(),
                    partition,
                },
                PersistentValue::PartitionState(PartitionTransition::Ready {
                    started_at_unix_millis: 100,
                }),
            )
        };
        let edge = |parent: PartitionKey, child: PartitionKey, centroid: f32| {
            (
                LogicalKey::ChildEntry {
                    index: index_id,
                    tree_key: key.clone(),
                    partition: parent,
                    child,
                },
                PersistentValue::ChildEntry(ChildEntry::new(child, vec![centroid])),
            )
        };
        puts.push(header(pk(1), 3, 2));
        puts.push(state(pk(1)));
        puts.push(edge(pk(1), pk(2), 2.0));
        puts.push(edge(pk(1), pk(3), 20.0));
        puts.push(header(pk(2), 2, 5));
        puts.push(state(pk(2)));
        puts.push(header(pk(3), 2, 2));
        puts.push(state(pk(3)));
        puts.push(edge(pk(3), pk(15), 19.0));
        puts.push(edge(pk(3), pk(16), 21.0));
        for (child, centroid) in [
            (pk(10), 0.0),
            (pk(11), 1.0),
            (pk(12), 2.0),
            (pk(13), 3.0),
            (pk(14), 4.0),
        ] {
            puts.push(edge(pk(2), child, centroid));
        }
        for leaf in [pk(10), pk(11), pk(12), pk(13), pk(14), pk(15), pk(16)] {
            puts.push(header(leaf, 1, 0));
            puts.push(state(leaf));
        }
        for (key, value) in puts {
            txn.put(key, value).await.expect("put fixture");
        }
        txn.commit().await.expect("commit fixture");
    }

    // The split reserves the never-reused pair after the fixture's range.
    let start = split::begin_split(&backend, &manifest, &key, pk(2), 2_000, &retry())
        .await
        .expect("begin");
    let topology::SplitStart::Started { left, right } = start else {
        panic!("split must start, got {start:?}");
    };
    assert_eq!((left, right), (pk(19), pk(20)));

    // The trained centroids come from the five Child Entry centroids: the
    // balanced deterministic assignment puts {pk 10, pk 11} left.
    let exposed = split::expose_targets(&backend, &manifest, &key, pk(2), 2_100, &retry())
        .await
        .expect("expose");
    assert_eq!(exposed, split::TargetExposure::Exposed { left, right });
    assert_eq!(
        centroid_of(&backend, &manifest, &key, left)
            .await
            .expect("left centroid")
            .components(),
        &[0.5]
    );
    assert_eq!(
        centroid_of(&backend, &manifest, &key, right)
            .await
            .expect("right centroid")
            .components(),
        &[3.0]
    );
    // Both targets were installed into the source's parent with their
    // centroids, growing the root's exact count to four.
    for target in [left, right] {
        let edge = edge_of(&backend, &manifest, &key, pk(1), target)
            .await
            .expect("target edge");
        assert_eq!(edge.child(), target);
    }
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(1))
            .await
            .expect("root header")
            .entry_count(),
        4
    );

    split::advance_to_draining(&backend, &manifest, &key, pk(2), 2_200, &retry())
        .await
        .expect("advance");

    // Drain: Child Entries move to the nearer persisted centroid with no
    // Record Location, Vector Record, or Synopsis work.
    drain_to_zero(&backend, &manifest, &key, pk(2)).await;
    let left_children: Vec<PartitionKey> = scan_child_entries(&backend, &manifest, &key, left)
        .await
        .into_iter()
        .map(|entry| entry.child())
        .collect();
    let right_children: Vec<PartitionKey> = scan_child_entries(&backend, &manifest, &key, right)
        .await
        .into_iter()
        .map(|entry| entry.child())
        .collect();
    assert_eq!(left_children, vec![pk(10), pk(11)]);
    assert_eq!(right_children, vec![pk(12), pk(13), pk(14)]);

    let completed = split::complete_split(&backend, &manifest, &key, pk(2), 2_300, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::SplitCompletion::Completed);
    assert_eq!(header_of(&backend, &manifest, &key, pk(2)).await, None);
    assert_eq!(edge_of(&backend, &manifest, &key, pk(1), pk(2)).await, None);
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(1))
            .await
            .expect("root header")
            .entry_count(),
        3
    );

    // Every level-1 partition remains reachable and routing descends through
    // the new targets.
    let leaves = reachable_leaves(&backend, &manifest, &key).await;
    assert_eq!(
        leaves.keys().copied().collect::<Vec<_>>(),
        vec![pk(10), pk(11), pk(12), pk(13), pk(14), pk(15), pk(16)]
    );
    let route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[0.75])
        .await
        .expect("route")
        .expect("tree exists");
    assert_eq!(route.leaf(), pk(11));
    assert_eq!(route.parent(), Some(left));

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Rediscovery: advance alone converges a cold split.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn advance_rediscovers_and_converges_a_cold_split() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 6).await;

    // No worker has run: advance performs each bounded step in turn.
    let outcomes = drive_to_completion(&backend, &manifest, &key, pk(1)).await;
    let mut kinds = outcomes.iter().map(std::mem::discriminant);
    let expected = [
        std::mem::discriminant(&Advance::Began {
            left: pk(2),
            right: pk(3),
        }),
        std::mem::discriminant(&Advance::Exposed {
            left: pk(2),
            right: pk(3),
        }),
    ];
    for expected in expected {
        assert_eq!(kinds.next(), Some(expected));
    }
    assert_eq!(outcomes.last(), Some(&Advance::Completed));

    // A settled partition is Idle, and a never-created or removed partition
    // has nothing to maintain.
    assert_eq!(
        split::advance(&backend, &manifest, &key, pk(1), 20_000, &retry())
            .await
            .expect("settled"),
        Advance::Idle
    );
    assert_eq!(
        split::advance(&backend, &manifest, &key, pk(99), 20_000, &retry())
            .await
            .expect("unknown partition is idle"),
        Advance::Idle
    );

    // A ReceivingSplit target is never split-eligible while its source drains:
    // run a second split and stop midway, then probe the target.
    let mut records = records;
    for (n, x) in [(10, 0.25), (11, 0.75), (12, 1.25), (13, 1.5), (14, 0.5)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let leaves = reachable_leaves(&backend, &manifest, &key).await;
    let (&over, _) = leaves
        .iter()
        .find(|(_, header)| header.entry_count() > 4)
        .expect("a leaf above the maximum");
    split::begin_split(&backend, &manifest, &key, over, 30_000, &retry())
        .await
        .expect("begin");
    split::expose_targets(&backend, &manifest, &key, over, 30_100, &retry())
        .await
        .expect("expose");
    let state = state_of(&backend, &manifest, &key, over)
        .await
        .expect("source state");
    let PartitionTransition::Splitting { left, .. } = state else {
        panic!("source must be Splitting");
    };
    // The target is over-threshold-empty but ReceivingSplit: not eligible.
    assert_eq!(
        split::advance(&backend, &manifest, &key, left, 30_200, &retry())
            .await
            .expect("target advance"),
        Advance::Idle
    );
    let outcomes = drive_to_completion(&backend, &manifest, &key, over).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    assert_searchable(&backend, &manifest, &key, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn advance_converges_a_split_whose_source_shrank_to_one_entry() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 6).await;

    // Begin the split while the root is over the maximum, then let foreground
    // deletes legally shrink the Splitting source to one entry (ADR 0014).
    assert_eq!(
        split::advance(&backend, &manifest, &key, pk(1), 10_000, &retry())
            .await
            .expect("begin"),
        Advance::Began {
            left: pk(2),
            right: pk(3)
        }
    );
    for n in 0..5_u8 {
        assert!(index.delete(rid(n)).await.expect("delete"));
    }
    let remaining = &records[5..];
    assert_searchable(&backend, &manifest, &key, remaining).await;

    // Exposure trains on the shrunken snapshot; the machine must advance on
    // this valid persistent state instead of reporting Corruption (#113).
    let outcomes = drive_to_completion(&backend, &manifest, &key, pk(1)).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    assert_searchable(&backend, &manifest, &key, remaining).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn advance_converges_a_split_whose_source_emptied_out() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 6).await;

    assert_eq!(
        split::advance(&backend, &manifest, &key, pk(1), 10_000, &retry())
            .await
            .expect("begin"),
        Advance::Began {
            left: pk(2),
            right: pk(3)
        }
    );
    for (id, _) in &records {
        assert!(index.delete(id.clone()).await.expect("delete"));
    }
    assert_searchable(&backend, &manifest, &key, &[]).await;

    // An empty Splitting source trains two zero centroids, drains nothing,
    // and completes through the ordinary zero-count completion (#113).
    let outcomes = drive_to_completion(&backend, &manifest, &key, pk(1)).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    assert_searchable(&backend, &manifest, &key, &[]).await;

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Crash and unknown-outcome recovery at every transition.
// ---------------------------------------------------------------------------

/// Builds a committed over-maximum leaf root with six records and returns its
/// manifest; the caller drives the split with explicit transactions.
async fn seed_over_max_root(backend: &SharedBackend) -> (IndexManifest, TreeKey) {
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    seed_records(&index, 1, 6).await;
    let manifest = read_manifest(backend, index.logical_index_id()).await;
    runtime.shutdown().await.expect("shutdown");
    (manifest, tree_key(1))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn begin_split_recovers_from_every_commit_outcome() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownNotApplied,
        CommitFault::UnknownApplied,
    ] {
        let backend = backend();
        let (manifest, key) = seed_over_max_root(&backend).await;

        backend.inner().push_fault(fault).expect("push fault");
        let mut txn = write_txn(&backend, &manifest).await;
        let started = topology::begin_split(&mut txn, &key, pk(1), 1_000)
            .await
            .expect("begin op");
        let topology::SplitStart::Started { left, right } = started else {
            panic!("split must start, got {started:?}");
        };
        let error = txn.commit().await.expect_err("injected fault");
        assert_fault_kind(fault, &error);

        // Re-driving observes exactly the committed outcome: either the split
        // still needs to start, or the persisted pair is adopted.
        let mut retry_txn = write_txn(&backend, &manifest).await;
        let redriven = topology::begin_split(&mut retry_txn, &key, pk(1), 1_001)
            .await
            .expect("redriven begin");
        match fault {
            CommitFault::UnknownApplied => assert_eq!(
                redriven,
                topology::SplitStart::AlreadySplitting { left, right },
                "the committed pair is adopted, never a second reservation"
            ),
            _ => assert_eq!(
                redriven,
                topology::SplitStart::Started { left, right },
                "nothing was applied, so the same pair is reserved again"
            ),
        }
        retry_txn.commit().await.expect("retry commits");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn target_creation_recovers_from_every_commit_outcome() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownNotApplied,
        CommitFault::UnknownApplied,
    ] {
        let backend = backend();
        let (manifest, key) = seed_over_max_root(&backend).await;
        split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
            .await
            .expect("begin");
        let trained = train_split_centroids(&mut read_txn(&backend, &manifest).await, &key, pk(1))
            .await
            .expect("train");

        backend.inner().push_fault(fault).expect("push fault");
        let mut txn = write_txn(&backend, &manifest).await;
        let created =
            topology::create_split_target(&mut txn, &key, pk(1), pk(2), trained.left(), 1_100)
                .await
                .expect("create op");
        assert_eq!(created, topology::TargetInstall::Created);
        let error = txn.commit().await.expect_err("injected fault");
        assert_fault_kind(fault, &error);

        let mut retry_txn = write_txn(&backend, &manifest).await;
        let redriven = topology::create_split_target(
            &mut retry_txn,
            &key,
            pk(1),
            pk(2),
            trained.left(),
            1_101,
        )
        .await
        .expect("redriven create");
        match fault {
            CommitFault::UnknownApplied => {
                // The persisted centroid stands; the re-driven centroid (same
                // value here) is discarded and the original start time kept.
                assert_eq!(redriven, topology::TargetInstall::AlreadyExists);
                assert_eq!(
                    state_of(&backend, &manifest, &key, pk(2)).await,
                    Some(PartitionTransition::ReceivingSplit {
                        source: pk(1),
                        started_at_unix_millis: 1_100,
                    })
                );
            }
            _ => assert_eq!(redriven, topology::TargetInstall::Created),
        }
        retry_txn.commit().await.expect("retry commits");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn advance_and_finalize_recover_from_every_commit_outcome() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownNotApplied,
        CommitFault::UnknownApplied,
    ] {
        let backend = backend();
        let (manifest, key) = seed_over_max_root(&backend).await;
        split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
            .await
            .expect("begin");
        split::expose_targets(&backend, &manifest, &key, pk(1), 1_100, &retry())
            .await
            .expect("expose");

        // Advance under the injected fault.
        backend.inner().push_fault(fault).expect("push fault");
        let mut txn = write_txn(&backend, &manifest).await;
        topology::advance_to_draining(&mut txn, &key, pk(1), 1_200)
            .await
            .expect("advance op");
        let error = txn.commit().await.expect_err("injected fault");
        assert_fault_kind(fault, &error);
        let mut retry_txn = write_txn(&backend, &manifest).await;
        let redriven = topology::advance_to_draining(&mut retry_txn, &key, pk(1), 1_201)
            .await
            .expect("redriven advance");
        match fault {
            CommitFault::UnknownApplied => {
                assert_eq!(redriven, topology::DrainStart::AlreadyDraining)
            }
            _ => assert_eq!(redriven, topology::DrainStart::Advanced),
        }
        retry_txn.commit().await.expect("retry commits");

        // Drain everything, then finalize under the injected fault.
        drain_to_zero(&backend, &manifest, &key, pk(1)).await;
        backend.inner().push_fault(fault).expect("push fault");
        let mut txn = write_txn(&backend, &manifest).await;
        topology::finalize_split(
            &mut txn,
            &key,
            pk(1),
            1_300,
            topology::SourceRemoval::PointDeletes,
        )
        .await
        .expect("finalize op");
        let error = txn.commit().await.expect_err("injected fault");
        assert_fault_kind(fault, &error);
        let mut retry_txn = write_txn(&backend, &manifest).await;
        let redriven = topology::finalize_split(
            &mut retry_txn,
            &key,
            pk(1),
            1_301,
            topology::SourceRemoval::PointDeletes,
        )
        .await
        .expect("redriven finalize");
        match fault {
            // The committed finalize converted the root; it is no longer
            // Draining, which is the documented rediscovery signal.
            CommitFault::UnknownApplied => {
                assert_eq!(redriven, topology::SplitCompletion::NotDraining)
            }
            _ => assert_eq!(redriven, topology::SplitCompletion::Completed),
        }
        retry_txn.commit().await.expect("retry commits");

        let root_header = header_of(&backend, &manifest, &key, pk(1))
            .await
            .expect("root header");
        assert_eq!(root_header.level(), 2);
        assert_eq!(root_header.state(), PartitionState::Ready);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_recovers_from_unknown_outcomes_without_losing_membership() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 6).await;
    split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    split::expose_targets(&backend, &manifest, &key, pk(1), 1_100, &retry())
        .await
        .expect("expose");
    split::advance_to_draining(&backend, &manifest, &key, pk(1), 1_200, &retry())
        .await
        .expect("advance");

    // The first batch's commit reports an unknown outcome after applying.
    backend
        .inner()
        .push_fault(CommitFault::UnknownApplied)
        .expect("push fault");
    let error = split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect_err("unknown outcome");
    assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    // Never retried blindly; rediscovery skips the already-moved prefix.
    assert_searchable(&backend, &manifest, &key, &records).await;
    loop {
        match split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
            .await
            .expect("redriven drain")
        {
            split::DrainStep::Drained { remaining: 0, .. } => break,
            split::DrainStep::Drained { .. } => {
                assert_searchable(&backend, &manifest, &key, &records).await;
            }
            other => panic!("unexpected drain outcome {other:?}"),
        }
    }
    let completed = split::complete_split(&backend, &manifest, &key, pk(1), 1_300, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::SplitCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restarted_process_rediscovers_the_durable_split_state() {
    let durable = DeterministicConfig {
        durability: support::Durability::Durable,
        ..DeterministicConfig::default()
    };
    let backend = SharedBackend::new(DeterministicBackend::new(durable));
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 6).await;
    split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    split::expose_targets(&backend, &manifest, &key, pk(1), 1_100, &retry())
        .await
        .expect("expose");
    split::advance_to_draining(&backend, &manifest, &key, pk(1), 1_200, &retry())
        .await
        .expect("advance");
    split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect("one batch");
    runtime.shutdown().await.expect("shutdown");

    // The process is gone; a reopened backend rediscovers the durable
    // DrainingSplit state and converges it.
    let reopened = SharedBackend::new(backend.inner().reopen());
    let outcomes = drive_to_completion(&reopened, &manifest, &key, pk(1)).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    assert_searchable(&reopened, &manifest, &key, &records).await;
}

// ---------------------------------------------------------------------------
// Conflicts abort bounded steps and retry from fresh snapshots.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_concurrent_source_write_aborts_begin_but_not_exposure() {
    let backend = backend();
    let (manifest, key) = seed_over_max_root(&backend).await;

    // A concurrent insert into the source conflicts with begin's
    // update-protected Header read.
    let mut attempt = write_txn(&backend, &manifest).await;
    topology::begin_split(&mut attempt, &key, pk(1), 1_000)
        .await
        .expect("begin op");
    let mut concurrent = write_txn(&backend, &manifest).await;
    let header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("header");
    concurrent
        .put(
            LogicalKey::Header {
                index: manifest.logical_index_id(),
                tree_key: key.clone(),
                partition: pk(1),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, header.entry_count() + 1, 1, PartitionState::Ready)
                    .expect("header"),
            ),
        )
        .await
        .expect("concurrent write");
    concurrent.commit().await.expect("concurrent commits");
    let error = attempt.commit().await.expect_err("begin conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // The retried begin observes the new count and starts.
    let mut retried = write_txn(&backend, &manifest).await;
    let started = topology::begin_split(&mut retried, &key, pk(1), 1_001)
        .await
        .expect("retried begin");
    assert!(matches!(started, topology::SplitStart::Started { .. }));
    retried.commit().await.expect("retry commits");

    // Exposure deliberately ignores concurrent source data changes: an insert
    // into the Splitting source does not conflict with target creation.
    let trained = train_split_centroids(&mut read_txn(&backend, &manifest).await, &key, pk(1))
        .await
        .expect("train");
    let runtime = make_runtime(backend.clone());
    let index = runtime.open_index("index").await.expect("open");
    let mut attempt = write_txn(&backend, &manifest).await;
    topology::create_split_target(&mut attempt, &key, pk(1), pk(2), trained.left(), 1_100)
        .await
        .expect("create op");
    index
        .insert(record(&rid(20), 2.5, 1))
        .await
        .expect("insert");
    attempt
        .commit()
        .await
        .expect("creation does not conflict with source writes");
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_concurrent_transition_aborts_target_creation_and_advance() {
    let backend = backend();
    let (manifest, key) = seed_over_max_root(&backend).await;
    split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    let trained = train_split_centroids(&mut read_txn(&backend, &manifest).await, &key, pk(1))
        .await
        .expect("train");

    // A concurrent worker exposing the same target conflicts on the target
    // keys through the unique insert's protected existence check.
    let mut attempt = write_txn(&backend, &manifest).await;
    topology::create_split_target(&mut attempt, &key, pk(1), pk(2), trained.left(), 1_100)
        .await
        .expect("create op");
    split::expose_targets(&backend, &manifest, &key, pk(1), 1_101, &retry())
        .await
        .expect("concurrent expose");
    let error = attempt
        .commit()
        .await
        .expect_err("duplicate creation conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // A concurrent advance conflicts with a stale create attempt: the source
    // State is update-protected by creation even when the target turns out to
    // already exist, so the stale attempt's commit aborts and must retry.
    let mut stale = write_txn(&backend, &manifest).await;
    let outcome =
        topology::create_split_target(&mut stale, &key, pk(1), pk(2), trained.left(), 1_102)
            .await
            .expect("stale create op");
    assert_eq!(outcome, topology::TargetInstall::AlreadyExists);
    split::advance_to_draining(&backend, &manifest, &key, pk(1), 1_200, &retry())
        .await
        .expect("concurrent advance");
    let error = stale
        .commit()
        .await
        .expect_err("the source-state protection conflicts with the advance");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // Once the source is DrainingSplit, a create attempt for a named target
    // verifies the exposed target instead of recreating it, and commits.
    let mut txn = write_txn(&backend, &manifest).await;
    let outcome =
        topology::create_split_target(&mut txn, &key, pk(1), pk(2), trained.left(), 1_300)
            .await
            .expect("verify existing");
    assert_eq!(outcome, topology::TargetInstall::AlreadyExists);
    txn.commit().await.expect("verification writes nothing");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stale_worker_cannot_recreate_a_target_after_completion() {
    let backend = backend();
    let (manifest, key) = seed_over_max_root(&backend).await;
    split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    let trained = train_split_centroids(&mut read_txn(&backend, &manifest).await, &key, pk(1))
        .await
        .expect("train");
    let outcomes = drive_to_completion(&backend, &manifest, &key, pk(1)).await;
    assert_eq!(outcomes.last(), Some(&Advance::Completed));
    let key_count = backend.inner().db_key_count();

    // The split is complete: the source State now says Ready (the root was
    // converted), so the stale creation attempt abandons without writing.
    let mut txn = write_txn(&backend, &manifest).await;
    let outcome =
        topology::create_split_target(&mut txn, &key, pk(1), pk(2), trained.left(), 9_999)
            .await
            .expect("stale create");
    assert_eq!(outcome, topology::TargetInstall::SourceAdvanced);
    txn.commit().await.expect("nothing written");
    assert_eq!(
        backend.inner().db_key_count(),
        key_count,
        "no orphan writes"
    );

    // A stale advance or drain likewise has nothing to do.
    let mut txn = write_txn(&backend, &manifest).await;
    let outcome = topology::advance_to_draining(&mut txn, &key, pk(1), 9_999)
        .await
        .expect("stale advance");
    assert_eq!(outcome, topology::DrainStart::NotSplitting);
    txn.rollback().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_concurrent_drain_move_conflicts_with_a_foreground_delete() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let mut records = seed_records(&index, 1, 6).await;
    split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    split::expose_targets(&backend, &manifest, &key, pk(1), 1_100, &retry())
        .await
        .expect("expose");
    split::advance_to_draining(&backend, &manifest, &key, pk(1), 1_200, &retry())
        .await
        .expect("advance");

    // The drain's update-protected entry read conflicts with a concurrent
    // delete of the same record; the batch retries and skips the moved entry.
    let mut attempt = write_txn(&backend, &manifest).await;
    let candidate = topology::read_leaf_drain_candidates(&mut attempt, &key, pk(1), &[rid(3)])
        .await
        .expect("read candidate")
        .into_iter()
        .next()
        .expect("one slot")
        .expect("candidate exists");
    assert!(index.delete(rid(3)).await.expect("delete"));
    records.retain(|(id, _)| *id != rid(3));
    topology::relocate_leaf_entries(
        &mut attempt,
        &key,
        pk(1),
        vec![(candidate, pk(3))],
        topology::Movement::Split,
    )
    .await
    .expect("relocate op");
    let error = attempt.commit().await.expect_err("delete conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // The retried batch skips the vanished entry; membership stays exact.
    drain_to_zero(&backend, &manifest, &key, pk(1)).await;
    assert_searchable(&backend, &manifest, &key, &records).await;
    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Fail-closed corruption paths.
// ---------------------------------------------------------------------------

/// Installs a drained non-root split fixture: root PK 1 at level 2 with leaf
/// children PK 2 (the drained source) and PK 3, and exposed targets PK 4 and
/// PK 5. Returns after committing the fixture.
async fn seed_completable_non_root_split(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
) {
    let mut txn = write_txn(backend, manifest).await;
    tree_manifest::create_tree(&mut txn, key, 100)
        .await
        .expect("create tree");
    txn.commit().await.expect("commit tree");

    let index = manifest.logical_index_id();
    let mut txn = write_txn(backend, manifest).await;
    let entries: Vec<(LogicalKey, PersistentValue)> = vec![
        (
            LogicalKey::Header {
                index,
                tree_key: key.clone(),
                partition: pk(1),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(2, 2, 0, PartitionState::Ready).expect("header"),
            ),
        ),
        // The drained source.
        (
            LogicalKey::Header {
                index,
                tree_key: key.clone(),
                partition: pk(2),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 0, 3, PartitionState::DrainingSplit).expect("header"),
            ),
        ),
        (
            LogicalKey::State {
                index,
                tree_key: key.clone(),
                partition: pk(2),
            },
            PersistentValue::PartitionState(PartitionTransition::DrainingSplit {
                left: pk(4),
                right: pk(5),
                started_at_unix_millis: 200,
            }),
        ),
        // The sibling leaf.
        (
            LogicalKey::Header {
                index,
                tree_key: key.clone(),
                partition: pk(3),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("header"),
            ),
        ),
        (
            LogicalKey::State {
                index,
                tree_key: key.clone(),
                partition: pk(3),
            },
            PersistentValue::PartitionState(PartitionTransition::Ready {
                started_at_unix_millis: 100,
            }),
        ),
        (
            LogicalKey::ChildEntry {
                index,
                tree_key: key.clone(),
                partition: pk(1),
                child: pk(2),
            },
            PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.0])),
        ),
        (
            LogicalKey::ChildEntry {
                index,
                tree_key: key.clone(),
                partition: pk(1),
                child: pk(3),
            },
            PersistentValue::ChildEntry(ChildEntry::new(pk(3), vec![10.0])),
        ),
    ];
    for (key, value) in entries {
        txn.put(key, value).await.expect("put fixture");
    }
    for target in [pk(4), pk(5)] {
        txn.put(
            LogicalKey::Header {
                index,
                tree_key: key.clone(),
                partition: target,
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 3, 1, PartitionState::ReceivingSplit).expect("header"),
            ),
        )
        .await
        .expect("put target header");
        txn.put(
            LogicalKey::State {
                index,
                tree_key: key.clone(),
                partition: target,
            },
            PersistentValue::PartitionState(PartitionTransition::ReceivingSplit {
                source: pk(2),
                started_at_unix_millis: 150,
            }),
        )
        .await
        .expect("put target state");
        txn.put(
            LogicalKey::Centroid {
                index,
                tree_key: key.clone(),
                partition: target,
            },
            PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![1.0])),
        )
        .await
        .expect("put target centroid");
    }
    txn.commit().await.expect("commit fixture");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_fails_closed_on_a_missing_or_duplicate_incoming_edge() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    seed_completable_non_root_split(&backend, &manifest, &key).await;

    // Missing edge: remove the source's only incoming Child Entry.
    let mut txn = write_txn(&backend, &manifest).await;
    txn.delete(LogicalKey::ChildEntry {
        index: manifest.logical_index_id(),
        tree_key: key.clone(),
        partition: pk(1),
        child: pk(2),
    })
    .await
    .expect("delete edge");
    txn.commit().await.expect("commit");
    let mut txn = write_txn(&backend, &manifest).await;
    let error = topology::finalize_split(
        &mut txn,
        &key,
        pk(2),
        300,
        topology::SourceRemoval::PointDeletes,
    )
    .await
    .expect_err("missing incoming edge");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;

    // Duplicate edge: a second reachable level-2 partition also references
    // the source. The fixture rises one level so the parent level holds two
    // bodies: root PK 1 at level 3 over internals PK 6 and PK 7, with the
    // source's real edge in PK 6 and a duplicate in PK 7.
    seed_completable_non_root_split(&backend, &manifest, &key).await;
    let index_id = manifest.logical_index_id();
    let mut txn = write_txn(&backend, &manifest).await;
    let fixture: Vec<(LogicalKey, PersistentValue)> = vec![
        (
            LogicalKey::Header {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(1),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(3, 2, 0, PartitionState::Ready).expect("header"),
            ),
        ),
        (
            LogicalKey::Header {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(6),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(2, 2, 0, PartitionState::Ready).expect("header"),
            ),
        ),
        (
            LogicalKey::State {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(6),
            },
            PersistentValue::PartitionState(PartitionTransition::Ready {
                started_at_unix_millis: 100,
            }),
        ),
        (
            LogicalKey::Header {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(7),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(2, 1, 0, PartitionState::Ready).expect("header"),
            ),
        ),
        (
            LogicalKey::State {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(7),
            },
            PersistentValue::PartitionState(PartitionTransition::Ready {
                started_at_unix_millis: 100,
            }),
        ),
        (
            LogicalKey::ChildEntry {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(1),
                child: pk(6),
            },
            PersistentValue::ChildEntry(ChildEntry::new(pk(6), vec![0.0])),
        ),
        (
            LogicalKey::ChildEntry {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(1),
                child: pk(7),
            },
            PersistentValue::ChildEntry(ChildEntry::new(pk(7), vec![10.0])),
        ),
        (
            LogicalKey::ChildEntry {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(6),
                child: pk(2),
            },
            PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.0])),
        ),
        (
            LogicalKey::ChildEntry {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(6),
                child: pk(3),
            },
            PersistentValue::ChildEntry(ChildEntry::new(pk(3), vec![10.0])),
        ),
        // The duplicate incoming reference.
        (
            LogicalKey::ChildEntry {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(7),
                child: pk(2),
            },
            PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.5])),
        ),
    ];
    for (key, value) in fixture {
        txn.put(key, value).await.expect("put duplicate edge");
    }
    // The source's original level-2 parent edges are replaced by the depth-3
    // shape.
    txn.delete(LogicalKey::ChildEntry {
        index: index_id,
        tree_key: key.clone(),
        partition: pk(1),
        child: pk(2),
    })
    .await
    .expect("delete level-2 edge");
    txn.delete(LogicalKey::ChildEntry {
        index: index_id,
        tree_key: key.clone(),
        partition: pk(1),
        child: pk(3),
    })
    .await
    .expect("delete level-2 sibling edge");
    txn.commit().await.expect("commit duplicate");
    let mut txn = write_txn(&backend, &manifest).await;
    let error = topology::finalize_split(
        &mut txn,
        &key,
        pk(2),
        300,
        topology::SourceRemoval::PointDeletes,
    )
    .await
    .expect_err("duplicate incoming edge");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_requires_exact_zero_count_and_draining_state() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    seed_completable_non_root_split(&backend, &manifest, &key).await;

    // A nonzero exact count refuses completion without an entry rescan.
    let mut txn = write_txn(&backend, &manifest).await;
    txn.put(
        LogicalKey::Header {
            index: manifest.logical_index_id(),
            tree_key: key.clone(),
            partition: pk(2),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 1, 3, PartitionState::DrainingSplit).expect("header"),
        ),
    )
    .await
    .expect("put nonzero header");
    txn.commit().await.expect("commit");
    let mut txn = write_txn(&backend, &manifest).await;
    let outcome = topology::finalize_split(
        &mut txn,
        &key,
        pk(2),
        300,
        topology::SourceRemoval::PointDeletes,
    )
    .await
    .expect("finalize");
    assert_eq!(outcome, topology::SplitCompletion::NotDrained);
    txn.rollback().await;

    // A non-Draining source has nothing to complete.
    let mut txn = write_txn(&backend, &manifest).await;
    let outcome = topology::finalize_split(
        &mut txn,
        &key,
        pk(3),
        300,
        topology::SourceRemoval::PointDeletes,
    )
    .await
    .expect("finalize a Ready partition");
    assert_eq!(outcome, topology::SplitCompletion::NotDraining);
    txn.rollback().await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_fails_closed_on_an_inconsistent_entry() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 6).await;
    split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    split::expose_targets(&backend, &manifest, &key, pk(1), 1_100, &retry())
        .await
        .expect("expose");
    split::advance_to_draining(&backend, &manifest, &key, pk(1), 1_200, &retry())
        .await
        .expect("advance");

    // Corrupt one Record Location through the raw seam: it names a leaf that
    // is not the source.
    let mut raw = backend.inner().begin_write().await.expect("begin write");
    raw.put(
        Bytes::from(
            keys::location_key(manifest.logical_index_id(), &rid(2)).expect("location key"),
        ),
        Bytes::from_static(b"invalid"),
    )
    .await
    .expect("raw put");
    raw.commit().await.expect("commit");

    let error = split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect_err("malformed location");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    // The corrupted source stays put for offline diagnosis.
    assert_searchable_present(&backend, &manifest, &key, &records, &[rid(2)]).await;

    runtime.shutdown().await.expect("shutdown");
}

/// Membership assertion tolerant of one corrupted record: every other record
/// keeps exact membership.
async fn assert_searchable_present(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    records: &[(Bytes, f32)],
    except: &[Bytes],
) {
    let kept: Vec<(Bytes, f32)> = records
        .iter()
        .filter(|(id, _)| !except.contains(id))
        .cloned()
        .collect();
    let leaves = reachable_leaves(backend, manifest, key).await;
    for (id, _) in &kept {
        let location = location_of(backend, manifest, id).await.expect("location");
        assert!(leaves.contains_key(&location.leaf()));
    }
}

// ---------------------------------------------------------------------------
// Model history: mutations interleaved with splits and abort faults.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seeded_model_history_interleaving_mutations_and_splits() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);

    let mut rng = Rng(0x5eed_u64);
    let mut model: BTreeMap<Bytes, f32> = BTreeMap::new();
    for step in 0..240_u64 {
        match rng.below(10) {
            // Upsert a record (create or replace).
            0..=4 => {
                let id = rid(rng.below(40) as u8);
                let x = (rng.below(400) as f32) / 10.0;
                match index.upsert(record(&id, x, 1)).await {
                    Ok(_) => {
                        model.insert(id, x);
                    }
                    Err(error) => {
                        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
                        // Recover through the documented idempotent readback.
                        let group = index
                            .get(id.clone(), Default::default())
                            .await
                            .expect("get");
                        match group {
                            Some(stored) => {
                                model.insert(id, stored.vector()[0]);
                            }
                            None => {
                                model.remove(&id);
                            }
                        }
                    }
                }
            }
            // Delete a record.
            5 => {
                let id = rid(rng.below(40) as u8);
                match index.delete(id.clone()).await {
                    Ok(_) => {
                        model.remove(&id);
                    }
                    Err(error) => {
                        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
                        let group = index
                            .get(id.clone(), Default::default())
                            .await
                            .expect("get");
                        if group.is_none() {
                            model.remove(&id);
                        }
                    }
                }
            }
            // Inject one definite abort into the next commit.
            6 => {
                backend
                    .inner()
                    .push_fault(CommitFault::Abort)
                    .expect("fault");
            }
            // Advance one random partition's split state.
            _ => {
                let partitions = all_partitions(&backend, &manifest, &key).await;
                if !partitions.is_empty() {
                    let partition = partitions[rng.below(partitions.len() as u64) as usize];
                    if let Err(error) = split::advance(
                        &backend,
                        &manifest,
                        &key,
                        partition,
                        50_000 + step,
                        &retry(),
                    )
                    .await
                    {
                        let header = header_of(&backend, &manifest, &key, partition).await;
                        let state = state_of(&backend, &manifest, &key, partition).await;
                        panic!(
                            "advance {partition:?} failed at step {step}: {error:?}; header {header:?}; state {state:?}"
                        );
                    }
                }
            }
        }
        if step % 25 == 0 {
            let model_records: Vec<(Bytes, f32)> =
                model.iter().map(|(id, x)| (id.clone(), *x)).collect();
            assert_exact_membership(&backend, &manifest, &key, &model_records).await;
        }
    }

    // Drive every remaining intermediate state to completion, then verify the
    // final tree against the model.
    let mut settled = false;
    for _ in 0..500 {
        let partitions = all_partitions(&backend, &manifest, &key).await;
        let mut progressed = false;
        let mut blocked = false;
        for partition in &partitions {
            let outcome =
                match split::advance(&backend, &manifest, &key, *partition, 60_000, &retry()).await
                {
                    Ok(outcome) => outcome,
                    // A child split waits while its parent drains; advancing the
                    // parent in this bounded loop makes a later attempt succeed.
                    Err(error) if error.kind() == ErrorKind::ContentionExhausted => {
                        blocked = true;
                        continue;
                    }
                    Err(error) => panic!("advance: {error:?}"),
                };
            if outcome != Advance::Idle {
                progressed = true;
            }
        }
        if !progressed {
            assert!(!blocked, "split convergence stalled on parent maintenance");
            settled = true;
            break;
        }
    }
    assert!(settled, "split convergence exceeded the bounded drive");
    for partition in all_partitions(&backend, &manifest, &key).await {
        assert_eq!(
            header_of(&backend, &manifest, &key, partition)
                .await
                .expect("reachable partition header")
                .state(),
            PartitionState::Ready,
            "partition {partition:?} did not settle"
        );
    }
    let model_records: Vec<(Bytes, f32)> = model.iter().map(|(id, x)| (id.clone(), *x)).collect();
    assert_searchable(&backend, &manifest, &key, &model_records).await;

    runtime.shutdown().await.expect("shutdown");
}

/// Enumerates every partition of one tree, following Child Entries and the
/// root's persisted split target slots.
async fn all_partitions(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
) -> Vec<PartitionKey> {
    let mut seen = BTreeSet::new();
    let mut frontier = vec![pk(1)];
    while let Some(partition) = frontier.pop() {
        if !seen.insert(partition) {
            continue;
        }
        let Some(header) = header_of(backend, manifest, key, partition).await else {
            continue;
        };
        if let Some(state) = state_of(backend, manifest, key, partition).await {
            if partition == pk(1) {
                if let PartitionTransition::Splitting { left, right, .. }
                | PartitionTransition::DrainingSplit { left, right, .. } = state
                {
                    frontier.push(left);
                    frontier.push(right);
                }
            }
        }
        if header.level() > 1 {
            for entry in scan_child_entries(backend, manifest, key, partition).await {
                frontier.push(entry.child());
            }
        }
    }
    seen.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Recovery and fail-closed regressions.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn steps_on_a_completed_non_root_split_are_harmless_noops() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    seed_completable_non_root_split(&backend, &manifest, &key).await;

    // Complete the split: both targets promote and the source is removed.
    let mut txn = write_txn(&backend, &manifest).await;
    let completed = topology::finalize_split(
        &mut txn,
        &key,
        pk(2),
        300,
        topology::SourceRemoval::PointDeletes,
    )
    .await
    .expect("finalize op");
    assert_eq!(completed, topology::SplitCompletion::Completed);
    txn.commit().await.expect("commit finalize");
    assert!(state_of(&backend, &manifest, &key, pk(2)).await.is_none());
    assert!(header_of(&backend, &manifest, &key, pk(2)).await.is_none());

    // A competing or recovering worker that re-drives any step of the
    // finished split observes the removal and gets a graceful outcome, never
    // a spurious Corruption (maintenance.md §3).
    let start = split::begin_split(&backend, &manifest, &key, pk(2), 400, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::SplitStart::NotEligible);
    let exposure = split::expose_targets(&backend, &manifest, &key, pk(2), 400, &retry())
        .await
        .expect("expose");
    assert_eq!(exposure, split::TargetExposure::SourceAdvanced);
    let drained = split::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("drain");
    assert_eq!(drained, split::DrainStep::SourceAdvanced);
    let completion = split::complete_split(&backend, &manifest, &key, pk(2), 400, &retry())
        .await
        .expect("complete");
    assert_eq!(completion, topology::SplitCompletion::Completed);
    let advance = split::advance(&backend, &manifest, &key, pk(2), 400, &retry())
        .await
        .expect("advance");
    assert_eq!(advance, Advance::Idle);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn steps_on_a_dropped_index_report_the_lifecycle_error() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    runtime.drop_index("index").await.expect("drop");

    // The read phase validates the persisted Manifest like every other
    // entry point, instead of reporting the missing topology keys as
    // Corruption.
    let error = split::advance(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect_err("a dropped index rejects maintenance");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);
    let error = split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect_err("a dropped index rejects draining");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_moves_bounded_batches_and_refreshes_target_authority() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 10).await;

    split::begin_split(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    split::expose_targets(&backend, &manifest, &key, pk(1), 1_100, &retry())
        .await
        .expect("expose");
    split::advance_to_draining(&backend, &manifest, &key, pk(1), 1_200, &retry())
        .await
        .expect("advance");

    // Newly exposed targets start empty with a zero epoch.
    for target in [pk(2), pk(3)] {
        let header = header_of(&backend, &manifest, &key, target)
            .await
            .expect("target header");
        assert_eq!(header.entry_count(), 0);
        assert_eq!(header.cache_epoch(), 0);
    }

    // The configured threshold caps this small fixture at eight entries, so
    // the exact count drives two bounded batches.
    let first = split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect("first batch");
    assert_eq!(
        first,
        split::DrainStep::Drained {
            moved: 8,
            remaining: 2
        }
    );
    let second = split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect("second batch");
    assert_eq!(
        second,
        split::DrainStep::Drained {
            moved: 2,
            remaining: 0
        }
    );

    // Each target's exact count, cache epoch, and synopsis reflect exactly
    // the entries it received: the epoch bumped once per moved entry and the
    // synopsis expanded monotonically from the canonical empty value.
    for target in [pk(2), pk(3)] {
        let entries = scan_leaf_entries(&backend, &manifest, &key, target).await;
        let header = header_of(&backend, &manifest, &key, target)
            .await
            .expect("target header");
        assert_eq!(header.entry_count() as usize, entries.len());
        assert_eq!(
            header.cache_epoch(),
            u64::from(header.entry_count()),
            "one epoch bump per moved entry"
        );
        let mut expected = PartitionSynopsis::empty(&manifest);
        for entry in &entries {
            expected.expand(&manifest, entry.fields()).expect("expand");
        }
        assert_eq!(
            synopsis_of(&backend, &manifest, &key, target).await,
            Some(expected),
            "target synopsis is exactly the moved entries' expansion"
        );
    }

    let completion = split::complete_split(&backend, &manifest, &key, pk(1), 1_300, &retry())
        .await
        .expect("complete");
    assert_eq!(completion, topology::SplitCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_fails_closed_when_the_count_disagrees_with_the_entries() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);

    // A DrainingSplit leaf root whose exact count claims one entry while the
    // entry range is empty: within one snapshot the two must agree. The
    // drain read phase fails before any target key is read, so the fixture
    // needs only the source's authority pair.
    create_committed_tree(&backend, &manifest, &key).await;
    let mut txn = write_txn(&backend, &manifest).await;
    let index_id = manifest.logical_index_id();
    txn.put(
        LogicalKey::Header {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(1),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 1, 0, PartitionState::DrainingSplit).expect("header"),
        ),
    )
    .await
    .expect("put source header");
    txn.put(
        LogicalKey::State {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(1),
        },
        PersistentValue::PartitionState(PartitionTransition::DrainingSplit {
            left: pk(2),
            right: pk(3),
            started_at_unix_millis: 200,
        }),
    )
    .await
    .expect("put source state");
    txn.commit().await.expect("commit fixture");

    let error = split::drain_batch(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect_err("a positive count with an empty entry range is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_root_finalize_recovers_from_an_unknown_commit_outcome() {
    for (backend, removal) in [
        (backend(), topology::SourceRemoval::PointDeletes),
        (
            backend_with_clear(),
            topology::SourceRemoval::TransactionalClear,
        ),
    ] {
        let runtime = make_runtime(backend.clone());
        let index = runtime
            .create_index("index", config())
            .await
            .expect("create");
        let manifest = read_manifest(&backend, index.logical_index_id()).await;
        let key = tree_key(1);
        seed_completable_non_root_split(&backend, &manifest, &key).await;

        // Finalize under an injected applied-but-unknown commit outcome.
        backend
            .inner()
            .push_fault(CommitFault::UnknownApplied)
            .expect("push fault");
        let mut txn = write_txn(&backend, &manifest).await;
        let completed = topology::finalize_split(&mut txn, &key, pk(2), 300, removal)
            .await
            .expect("finalize op");
        assert_eq!(completed, topology::SplitCompletion::Completed);
        let error = txn.commit().await.expect_err("injected fault");
        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);

        // Re-driving observes the removed source and reports completion
        // instead of failing or repeating the topology switch.
        let mut retry_txn = write_txn(&backend, &manifest).await;
        let redriven = topology::finalize_split(&mut retry_txn, &key, pk(2), 301, removal)
            .await
            .expect("redriven finalize");
        assert_eq!(redriven, topology::SplitCompletion::Completed);
        retry_txn.commit().await.expect("retry commits");

        // The switched topology stands: both targets are Ready and the
        // source is gone. (Their incoming edges are installed at target
        // creation, which this synthetic fixture omits.)
        for target in [pk(4), pk(5)] {
            let header = header_of(&backend, &manifest, &key, target)
                .await
                .expect("target header");
            assert_eq!(header.state(), PartitionState::Ready);
        }
        assert!(header_of(&backend, &manifest, &key, pk(2)).await.is_none());
        runtime.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn advance_fails_closed_on_a_torn_target_without_committing() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);

    // A Splitting leaf root whose left target carries only its State: Header
    // and Centroid are missing — a torn committed state.
    create_committed_tree(&backend, &manifest, &key).await;
    let mut txn = write_txn(&backend, &manifest).await;
    let index_id = manifest.logical_index_id();
    txn.put(
        LogicalKey::Header {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(1),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 6, 0, PartitionState::Splitting).expect("header"),
        ),
    )
    .await
    .expect("put source header");
    txn.put(
        LogicalKey::State {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(1),
        },
        PersistentValue::PartitionState(PartitionTransition::Splitting {
            left: pk(2),
            right: pk(3),
            started_at_unix_millis: 200,
        }),
    )
    .await
    .expect("put source state");
    txn.put(
        LogicalKey::State {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(2),
        },
        PersistentValue::PartitionState(PartitionTransition::ReceivingSplit {
            source: pk(1),
            started_at_unix_millis: 200,
        }),
    )
    .await
    .expect("put torn target state");
    for (key_part, value) in [
        (
            LogicalKey::Header {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(3),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 0, 0, PartitionState::ReceivingSplit).expect("header"),
            ),
        ),
        (
            LogicalKey::State {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(3),
            },
            PersistentValue::PartitionState(PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 200,
            }),
        ),
        (
            LogicalKey::Centroid {
                index: index_id,
                tree_key: key.clone(),
                partition: pk(3),
            },
            PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![1.0])),
        ),
    ] {
        txn.put(key_part, value).await.expect("put target");
    }
    txn.commit().await.expect("commit fixture");

    // The transition refuses to commit the source into a DrainingSplit that
    // drain and completion could only wedge behind...
    let mut txn = write_txn(&backend, &manifest).await;
    let error = topology::advance_to_draining(&mut txn, &key, pk(1), 300)
        .await
        .expect_err("a torn target fails closed");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;

    // ...and the source stays Splitting and therefore fully writable.
    assert_eq!(
        state_of(&backend, &manifest, &key, pk(1)).await,
        Some(PartitionTransition::Splitting {
            left: pk(2),
            right: pk(3),
            started_at_unix_millis: 200,
        })
    );

    runtime.shutdown().await.expect("shutdown");
}
