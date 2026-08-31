//! Searchable merge state machine contract tests (#31).
//!
//! Every committed merge phase must stay searchable and preserve exact
//! membership; each bounded drain batch reselects the nearest Ready same-level
//! target with no persisted target or cursor; completion uses the exact zero
//! count and removes the source's incoming reference and prefix without
//! touching any target state; stalls never revert to Ready; crashes and
//! conflicts at every transition are covered (ADR 0008).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, Index, IndexConfig, Metric, PartitionKey, Record,
    RuntimeConfig, SearchRequest, Value,
};
use ktann::maintenance::routing::route_leaf;
use ktann::maintenance::{merge, split};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::backend::{Backend, Capabilities, ScanLimits, WriteTxn};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexManifest, LeafEntry, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionSynopsis, PartitionTransition, PersistentValue, RecordLocation,
};
use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn, topology, tree_manifest};

use support::oracle::{Model, ModelRecord};
use support::{
    CommitFault, DeterministicBackend, DeterministicConfig, Durability, Rng, SharedBackend, audit,
    read_manifest,
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

/// A one-dimensional L2 index over one i64 tree-key field with a minimum of
/// two and a maximum of four entries per partition, so small fixtures trigger
/// both splits and merges.
fn config() -> IndexConfig {
    IndexConfig::new(1, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(2, 4)
        .expect("valid partition entries")
}

/// A one-dimensional L2 index with a minimum of thirteen entries per
/// partition, so budget-focused fixtures can exercise multiple drain batches.
fn wide_config() -> IndexConfig {
    IndexConfig::new(1, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(13, 32)
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
/// persisted state, so they join the frontier there (ADR 0006). A `Merging`
/// partition keeps its incoming edge until completion, so it is covered
/// unchanged. Every internal body's scanned Child Entry count must equal its
/// exact Header count, and no partition may be discovered twice.
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

/// Runs the persistent-state audit against the caller's record model.
async fn run_audit(backend: &SharedBackend, manifest: &IndexManifest, records: &[(Bytes, f32)]) {
    let model: Model = records
        .iter()
        .map(|(id, x)| {
            (
                id.clone(),
                ModelRecord {
                    vector: Arc::from([*x]),
                    fields: Box::from([Value::I64(1)]),
                },
            )
        })
        .collect();
    audit::run(backend, manifest.logical_index_id(), &model)
        .await
        .expect("audit");
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

/// Drives `split::advance` until the partition is no longer making progress,
/// returning the observed outcome sequence.
async fn drive_split_to_completion(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Vec<split::Advance> {
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
            split::Advance::Idle | split::Advance::Completed { .. } => {
                outcomes.push(outcome);
                return outcomes;
            }
            _ => outcomes.push(outcome),
        }
    }
    panic!("split did not converge in 1000 steps");
}

/// Drives `merge::advance` until the partition is no longer making progress,
/// returning the observed outcome sequence.
async fn drive_merge_to_completion(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    partition: PartitionKey,
) -> Vec<merge::Advance> {
    let mut outcomes = Vec::new();
    for step in 0..1_000_u32 {
        let outcome = merge::advance(
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
            merge::Advance::Idle | merge::Advance::Completed | merge::Advance::Stalled => {
                outcomes.push(outcome);
                return outcomes;
            }
            _ => outcomes.push(outcome),
        }
    }
    panic!("merge did not converge in 1000 steps");
}

/// Drains one merge source to exact zero, one bounded batch at a time,
/// returning the total moved entries.
async fn merge_drain_to_zero(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    source: PartitionKey,
) -> usize {
    let mut moved_total = 0_usize;
    loop {
        match merge::drain_batch(backend, manifest, key, source, &retry())
            .await
            .expect("drain")
        {
            merge::DrainStep::Drained { moved, remaining } => {
                moved_total += moved;
                if remaining == 0 {
                    return moved_total;
                }
            }
            other => panic!("unexpected drain outcome {other:?}"),
        }
    }
}

/// Runs one full merge of `source`: begin, drain to zero, complete.
async fn merge_partition(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    source: PartitionKey,
) {
    let start = merge::begin_merge(backend, manifest, key, source, 5_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);
    merge_drain_to_zero(backend, manifest, key, source).await;
    let completed = merge::complete_merge(backend, manifest, key, source, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
}

/// Deletes every id through the public API, keeping the model in step.
async fn delete_ids(index: &Index<SharedBackend>, records: &mut Vec<(Bytes, f32)>, ids: &[Bytes]) {
    for id in ids {
        assert!(index.delete(id.clone()).await.expect("delete"));
        records.retain(|(record_id, _)| record_id != id);
    }
}

/// The Record IDs held by one leaf, in Leaf Entry key order.
async fn leaf_ids(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    leaf: PartitionKey,
) -> Vec<Bytes> {
    scan_leaf_entries(backend, manifest, key, leaf)
        .await
        .into_iter()
        .map(|entry| entry.record_id().clone())
        .collect()
}

/// Deletes entries of one leaf until exactly `keep` remain, keeping the
/// entries with the largest vectors; returns the kept (id, x) pairs in
/// ascending-x order.
async fn trim_leaf(
    index: &Index<SharedBackend>,
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    leaf: PartitionKey,
    keep: usize,
    records: &mut Vec<(Bytes, f32)>,
) -> Vec<(Bytes, f32)> {
    let mut entries: Vec<(Bytes, f32)> = leaf_ids(backend, manifest, key, leaf)
        .await
        .into_iter()
        .map(|id| {
            let x = records
                .iter()
                .find(|(record_id, _)| record_id == &id)
                .expect("modeled record")
                .1;
            (id, x)
        })
        .collect();
    entries.sort_by(|left, right| left.1.total_cmp(&right.1));
    let delete_count = entries.len() - keep;
    let deleted: Vec<Bytes> = entries[..delete_count]
        .iter()
        .map(|(id, _)| id.clone())
        .collect();
    delete_ids(index, records, &deleted).await;
    entries.split_off(delete_count)
}

/// The single persisted routing centroid component of one partition.
async fn leaf_centroid(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    leaf: PartitionKey,
) -> f32 {
    centroid_of(backend, manifest, key, leaf)
        .await
        .expect("leaf centroid")
        .components()[0]
}

/// The canonical merge target for one vector: the nearest candidate by
/// squared distance with the Partition Key tie-break (ADR 0008). The
/// one-dimensional L2 rotation is the identity, so routing space is the raw
/// vector space.
fn expected_target(x: f32, targets: &[(PartitionKey, f32)]) -> PartitionKey {
    let mut best: Option<(f64, PartitionKey)> = None;
    for &(target, centroid) in targets {
        let delta = f64::from(x) - f64::from(centroid);
        let distance = delta * delta;
        if best.is_none_or(|(best_distance, best_partition)| {
            distance < best_distance || (distance == best_distance && target < best_partition)
        }) {
            best = Some((distance, target));
        }
    }
    best.expect("at least one target").1
}

/// Asserts the error kind one injected commit fault produces.
fn assert_fault_kind(fault: CommitFault, error: &ktann::api::Error) {
    let expected = match fault {
        CommitFault::Abort => ErrorKind::RetryableAbort,
        _ => ErrorKind::CommitOutcomeUnknown,
    };
    assert_eq!(error.kind(), expected, "fault {fault:?}");
}

/// A committed two-leaf tree: six records split off the root, leaving PK 2
/// (x in {0,1,2}) and PK 3 (x in {3,4,5}) under the level-2 root.
async fn two_leaf_tree(
    backend: &SharedBackend,
) -> (
    Runtime<SharedBackend>,
    Index<SharedBackend>,
    IndexManifest,
    TreeKey,
    Vec<(Bytes, f32)>,
) {
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 6).await;
    let outcomes = drive_split_to_completion(backend, &manifest, &key, pk(1)).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    for leaf in [pk(2), pk(3)] {
        let header = header_of(backend, &manifest, &key, leaf)
            .await
            .expect("leaf header");
        assert_eq!(header.entry_count(), 3, "balanced root split");
        assert_eq!(header.state(), PartitionState::Ready);
    }
    (runtime, index, manifest, key, records)
}

/// A committed three-leaf tree: the two-leaf tree with the left leaf grown
/// past the maximum and split, leaving PK 3 (x in {3,4,5}) and PK 4/PK 5
/// sharing x in {0, 0.25, 0.75, 1, 1.25, 2}.
async fn three_leaf_tree(
    backend: &SharedBackend,
) -> (
    Runtime<SharedBackend>,
    Index<SharedBackend>,
    IndexManifest,
    TreeKey,
    Vec<(Bytes, f32)>,
) {
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(backend).await;
    for (n, x) in [(10, 0.25), (11, 0.75), (12, 1.25)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let outcomes = drive_split_to_completion(backend, &manifest, &key, pk(2)).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    let leaves = reachable_leaves(backend, &manifest, &key).await;
    assert_eq!(
        leaves.keys().copied().collect::<Vec<_>>(),
        vec![pk(3), pk(4), pk(5)]
    );
    (runtime, index, manifest, key, records)
}

/// A committed wide-config two-leaf tree: 33 records split off the root.
async fn wide_two_leaf_tree(
    backend: &SharedBackend,
) -> (
    Runtime<SharedBackend>,
    Index<SharedBackend>,
    IndexManifest,
    TreeKey,
    Vec<(Bytes, f32)>,
) {
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", wide_config())
        .await
        .expect("create");
    let manifest = read_manifest(backend, index.logical_index_id()).await;
    let key = tree_key(1);
    let records = seed_records(&index, 1, 33).await;
    let outcomes = drive_split_to_completion(backend, &manifest, &key, pk(1)).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    (runtime, index, manifest, key, records)
}

/// A committed wide-config three-leaf tree: the right leaf grown past the
/// maximum (x = 33..49 all route right) and split.
async fn wide_three_leaf_tree(
    backend: &SharedBackend,
) -> (
    Runtime<SharedBackend>,
    Index<SharedBackend>,
    IndexManifest,
    TreeKey,
    Vec<(Bytes, f32)>,
) {
    let (runtime, index, manifest, key, mut records) = wide_two_leaf_tree(backend).await;
    for n in 33..50_u8 {
        let x = f32::from(n);
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let outcomes = drive_split_to_completion(backend, &manifest, &key, pk(3)).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    let leaves = reachable_leaves(backend, &manifest, &key).await;
    assert_eq!(leaves.len(), 3);
    (runtime, index, manifest, key, records)
}

/// A wide-config fixture with the middle leaf (by entry x-order) begun as a
/// merge source: exactly twelve source entries (two drain batches) and two
/// Ready targets with their persisted routing centroids.
struct MiddleLeafMerge {
    runtime: Runtime<SharedBackend>,
    index: Index<SharedBackend>,
    manifest: IndexManifest,
    key: TreeKey,
    records: Vec<(Bytes, f32)>,
    source: PartitionKey,
    targets: [(PartitionKey, f32); 2],
    kept: Vec<(Bytes, f32)>,
}

async fn begin_middle_leaf_merge(backend: &SharedBackend) -> MiddleLeafMerge {
    let (runtime, index, manifest, key, mut records) = wide_three_leaf_tree(backend).await;
    // The merge source is the middle leaf by entry x-order, so its entries
    // straddle the two targets' midpoint and both receive entries.
    let leaves = reachable_leaves(backend, &manifest, &key).await;
    let mut by_mean: Vec<(f32, PartitionKey)> = Vec::new();
    for leaf in leaves.keys() {
        let ids = leaf_ids(backend, &manifest, &key, *leaf).await;
        let total: f32 = ids
            .iter()
            .map(|id| {
                records
                    .iter()
                    .find(|(record_id, _)| record_id == id)
                    .expect("modeled record")
                    .1
            })
            .sum();
        by_mean.push((total / ids.len() as f32, *leaf));
    }
    by_mean.sort_by(|left, right| left.0.total_cmp(&right.0));
    let source = by_mean[1].1;
    let targets = [
        (
            by_mean[0].1,
            leaf_centroid(backend, &manifest, &key, by_mean[0].1).await,
        ),
        (
            by_mean[2].1,
            leaf_centroid(backend, &manifest, &key, by_mean[2].1).await,
        ),
    ];
    let kept = trim_leaf(&index, backend, &manifest, &key, source, 12, &mut records).await;
    assert_eq!(kept.len(), 12);
    let start = merge::begin_merge(backend, &manifest, &key, source, 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);
    MiddleLeafMerge {
        runtime,
        index,
        manifest,
        key,
        records,
        source,
        targets,
        kept,
    }
}

// ---------------------------------------------------------------------------
// Lifecycle: non-root leaf merge, end to end, with foreground interleaving.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_root_leaf_merge_runs_end_to_end_and_stays_searchable() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = wide_two_leaf_tree(&backend).await;
    let source = pk(2);
    let target = pk(3);

    // Trim the source below the minimum of thirteen: exactly twelve entries
    // remain, two bounded drain batches.
    let kept = trim_leaf(&index, &backend, &manifest, &key, source, 12, &mut records).await;
    let target_before = header_of(&backend, &manifest, &key, target)
        .await
        .expect("target header");
    let source_header = header_of(&backend, &manifest, &key, source)
        .await
        .expect("source header");
    assert_eq!(source_header.entry_count(), 12);
    assert_eq!(source_header.state(), PartitionState::Ready);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // Snapshot the source entries to prove verbatim RaBitQ7 copies later.
    let before: BTreeMap<Bytes, LeafEntry> = scan_leaf_entries(&backend, &manifest, &key, source)
        .await
        .into_iter()
        .map(|entry| (entry.record_id().clone(), entry))
        .collect();

    // Begin marks only the source Merging; its incoming edge stays.
    let start = merge::begin_merge(&backend, &manifest, &key, source, 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);
    assert_eq!(
        state_of(&backend, &manifest, &key, source).await,
        Some(PartitionTransition::Merging {
            started_at_unix_millis: 1_000,
        })
    );
    let source_header = header_of(&backend, &manifest, &key, source)
        .await
        .expect("source header");
    assert_eq!(source_header.state(), PartitionState::Merging);
    assert_eq!(source_header.entry_count(), 12);
    assert!(
        edge_of(&backend, &manifest, &key, pk(1), source)
            .await
            .is_some(),
        "a Merging source keeps its incoming edge until completion"
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // Re-driving begin is idempotent and keeps the original start time.
    let again = merge::begin_merge(&backend, &manifest, &key, source, 1_001, &retry())
        .await
        .expect("begin again");
    assert_eq!(again, topology::MergeStart::AlreadyMerging);
    assert_eq!(
        state_of(&backend, &manifest, &key, source).await,
        Some(PartitionTransition::Merging {
            started_at_unix_millis: 1_000,
        })
    );

    // During Merging: a new insert reroutes to the Ready target, an upsert
    // whose Record Location names the source relocates atomically, and an
    // exact delete follows the location into the source.
    index
        .insert(record(&rid(200), 20.5, 1))
        .await
        .expect("insert");
    records.push((rid(200), 20.5));
    assert_eq!(
        location_of(&backend, &manifest, &rid(200))
            .await
            .expect("location")
            .leaf(),
        target,
        "no insert enters a Merging leaf"
    );

    index
        .upsert(record(&kept[0].0, 25.5, 1))
        .await
        .expect("upsert");
    let upserted = kept[0].0.clone();
    if let Some(slot) = records.iter_mut().find(|(id, _)| id == &upserted) {
        slot.1 = 25.5;
    }
    assert_eq!(
        location_of(&backend, &manifest, &upserted)
            .await
            .expect("location")
            .leaf(),
        target,
        "upsert relocates atomically"
    );
    assert!(
        leaf_entry_of(&backend, &manifest, &key, source, &upserted)
            .await
            .is_none(),
        "the source entry is gone"
    );

    let deleted = kept[1].0.clone();
    assert!(index.delete(deleted.clone()).await.expect("delete"));
    records.retain(|(id, _)| id != &deleted);
    assert_eq!(location_of(&backend, &manifest, &deleted).await, None);
    let source_header = header_of(&backend, &manifest, &key, source)
        .await
        .expect("source header");
    assert_eq!(source_header.entry_count(), 10);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // Drain: two bounded batches of eight and two, driven by the exact count.
    let first = merge::drain_batch(&backend, &manifest, &key, source, &retry())
        .await
        .expect("first batch");
    assert_eq!(
        first,
        merge::DrainStep::Drained {
            moved: 8,
            remaining: 2
        }
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;
    let second = merge::drain_batch(&backend, &manifest, &key, source, &retry())
        .await
        .expect("second batch");
    assert_eq!(
        second,
        merge::DrainStep::Drained {
            moved: 2,
            remaining: 0
        }
    );
    assert_eq!(
        header_of(&backend, &manifest, &key, source)
            .await
            .expect("source header")
            .entry_count(),
        0
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // Completion removes the incoming edge and the whole source prefix, and
    // changes no target state.
    let completed = merge::complete_merge(&backend, &manifest, &key, source, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_eq!(
        edge_of(&backend, &manifest, &key, pk(1), source).await,
        None
    );
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(1))
            .await
            .expect("root header")
            .entry_count(),
        1,
        "parent count decremented"
    );
    assert_eq!(header_of(&backend, &manifest, &key, source).await, None);
    assert_eq!(state_of(&backend, &manifest, &key, source).await, None);
    assert_eq!(centroid_of(&backend, &manifest, &key, source).await, None);
    assert_eq!(synopsis_of(&backend, &manifest, &key, source).await, None);
    assert!(
        scan_leaf_entries(&backend, &manifest, &key, source)
            .await
            .is_empty()
    );
    let target_header = header_of(&backend, &manifest, &key, target)
        .await
        .expect("target header");
    assert_eq!(target_header.state(), PartitionState::Ready);
    assert_eq!(
        target_header.entry_count(),
        target_before.entry_count() + 12,
        "insert + upsert relocation + ten drained entries"
    );
    assert_eq!(
        target_header.cache_epoch(),
        target_before.cache_epoch() + 12,
        "one epoch bump per received entry"
    );

    // Every drained entry sits at the target with its Record Location
    // repointed and its Leaf Entry bytes copied verbatim.
    for (id, _) in &kept[2..] {
        let location = location_of(&backend, &manifest, id)
            .await
            .expect("location");
        assert_eq!(location.leaf(), target);
        let entry = leaf_entry_of(&backend, &manifest, &key, target, id)
            .await
            .expect("entry");
        assert_eq!(
            &entry,
            before.get(id).expect("snapshotted entry"),
            "RaBitQ7 payload copied verbatim"
        );
    }
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // Completion is idempotent: the source's authority values are gone.
    let again = merge::complete_merge(&backend, &manifest, &key, source, &retry())
        .await
        .expect("complete again");
    assert_eq!(again, topology::MergeCompletion::Completed);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_moves_bounded_batches_to_per_entry_reselected_targets() {
    let backend = backend();
    let fixture = begin_middle_leaf_merge(&backend).await;
    let MiddleLeafMerge {
        runtime,
        index: _,
        manifest,
        key,
        records,
        source,
        targets,
        kept,
    } = fixture;

    // The target synopses and counts before the merge, for exact checks.
    let mut target_before = BTreeMap::new();
    for (target, _) in targets {
        let header = header_of(&backend, &manifest, &key, target)
            .await
            .expect("target header");
        let synopsis = synopsis_of(&backend, &manifest, &key, target)
            .await
            .expect("target synopsis");
        target_before.insert(target, (header, synopsis));
    }
    let source_synopsis = synopsis_of(&backend, &manifest, &key, source)
        .await
        .expect("source synopsis");

    // The first bounded batch moves the eight smallest entries; the second
    // moves the remaining four.
    let first = merge::drain_batch(&backend, &manifest, &key, source, &retry())
        .await
        .expect("first batch");
    assert_eq!(
        first,
        merge::DrainStep::Drained {
            moved: 8,
            remaining: 4
        }
    );
    // The source Synopsis does not shrink while draining (ADR 0008).
    assert_eq!(
        synopsis_of(&backend, &manifest, &key, source).await,
        Some(source_synopsis.clone())
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    let mut kept_by_id = kept.clone();
    kept_by_id.sort_by(|left, right| left.0.cmp(&right.0));
    for (position, (id, x)) in kept_by_id.iter().enumerate() {
        let location = location_of(&backend, &manifest, id)
            .await
            .expect("location");
        if position < 8 {
            assert_eq!(
                location.leaf(),
                expected_target(*x, &targets),
                "the nearest Ready target receives the entry"
            );
        } else {
            assert_eq!(location.leaf(), source, "not yet drained");
        }
    }

    let second = merge::drain_batch(&backend, &manifest, &key, source, &retry())
        .await
        .expect("second batch");
    assert_eq!(
        second,
        merge::DrainStep::Drained {
            moved: 4,
            remaining: 0
        }
    );
    assert_eq!(
        synopsis_of(&backend, &manifest, &key, source).await,
        Some(source_synopsis)
    );

    // Different entries moved to different targets, each to its nearest
    // Ready target by the canonical rule.
    let mut used: BTreeSet<PartitionKey> = BTreeSet::new();
    for (id, x) in &kept {
        let location = location_of(&backend, &manifest, id)
            .await
            .expect("location");
        let expected = expected_target(*x, &targets);
        assert_eq!(location.leaf(), expected);
        used.insert(location.leaf());
    }
    assert_eq!(used.len(), 2, "both Ready targets received entries");

    // Each target's exact count, cache epoch, and synopsis reflect exactly
    // the entries it received.
    for (target, _) in targets {
        let (header_before, synopsis_before) =
            target_before.get(&target).expect("snapshotted target");
        let mut entries = Vec::new();
        for (id, x) in &kept {
            if expected_target(*x, &targets) == target {
                entries.push(
                    leaf_entry_of(&backend, &manifest, &key, target, id)
                        .await
                        .expect("moved entry"),
                );
            }
        }
        let header = header_of(&backend, &manifest, &key, target)
            .await
            .expect("target header");
        assert_eq!(
            header.entry_count(),
            header_before.entry_count() + entries.len() as u32,
            "exactly the received entries"
        );
        assert_eq!(
            header.cache_epoch(),
            header_before.cache_epoch() + entries.len() as u64,
            "one epoch bump per moved entry"
        );
        let mut expected = synopsis_before.clone();
        for entry in &entries {
            expected.expand(&manifest, entry.fields()).expect("expand");
        }
        assert_eq!(
            synopsis_of(&backend, &manifest, &key, target).await,
            Some(expected),
            "target synopsis is exactly the moved entries' expansion"
        );
    }

    let completed = merge::complete_merge(&backend, &manifest, &key, source, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Lifecycle: an internal (Child Entry) merge.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_internal_merge_moves_child_entries_and_removes_the_source() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;

    // Grow to depth three, mirroring the split suite's root-internal fixture:
    // split each leaf past the maximum, then one more leaf split pushes the
    // root over and its split rises to level 3 with two level-2 children.
    for (n, x) in [(10, 0.25), (11, 0.75), (12, 1.25)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    drive_split_to_completion(&backend, &manifest, &key, pk(2)).await;
    for (n, x) in [(13, 3.25), (14, 3.75), (15, 4.25)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    drive_split_to_completion(&backend, &manifest, &key, pk(3)).await;
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
    drive_split_to_completion(&backend, &manifest, &key, over).await;
    let outcomes = drive_split_to_completion(&backend, &manifest, &key, pk(1)).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    let root_header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("root header");
    assert_eq!(root_header.level(), 3);
    assert_eq!(root_header.entry_count(), 2);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // The two level-2 internals share the five leaves; work on the smaller.
    let internals: Vec<PartitionKey> = scan_child_entries(&backend, &manifest, &key, pk(1))
        .await
        .into_iter()
        .map(|entry| entry.child())
        .collect();
    assert_eq!(internals.len(), 2);
    let mut counts = Vec::new();
    for internal in &internals {
        let header = header_of(&backend, &manifest, &key, *internal)
            .await
            .expect("internal header");
        assert_eq!(header.level(), 2);
        counts.push(header.entry_count());
    }
    let (source, other) = if counts[0] <= counts[1] {
        (internals[0], internals[1])
    } else {
        (internals[1], internals[0])
    };
    let other_before = header_of(&backend, &manifest, &key, other)
        .await
        .expect("other internal header");

    // Shrink the source below the minimum of two by merging its leaf
    // children away until one Child Entry remains.
    while header_of(&backend, &manifest, &key, source)
        .await
        .expect("source header")
        .entry_count()
        > 1
    {
        let child = scan_child_entries(&backend, &manifest, &key, source)
            .await
            .first()
            .expect("a child")
            .child();
        let ids = leaf_ids(&backend, &manifest, &key, child).await;
        delete_ids(&index, &mut records, &ids).await;
        merge_partition(&backend, &manifest, &key, child).await;
        assert_searchable(&backend, &manifest, &key, &records).await;
        run_audit(&backend, &manifest, &records).await;
    }
    let source_header = header_of(&backend, &manifest, &key, source)
        .await
        .expect("source header");
    assert_eq!(source_header.entry_count(), 1);
    assert_eq!(source_header.state(), PartitionState::Ready);
    let remaining_child = scan_child_entries(&backend, &manifest, &key, source)
        .await
        .first()
        .expect("one child")
        .child();

    // Begin marks only the internal Merging; its incoming edge stays.
    let start = merge::begin_merge(&backend, &manifest, &key, source, 3_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);
    assert_eq!(
        state_of(&backend, &manifest, &key, source).await,
        Some(PartitionTransition::Merging {
            started_at_unix_millis: 3_000,
        })
    );
    assert!(
        edge_of(&backend, &manifest, &key, pk(1), source)
            .await
            .is_some()
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // The drain moves the one Child Entry to the other Ready level-2
    // internal — no Record Location, Vector Record, or Synopsis work.
    let step = merge::drain_batch(&backend, &manifest, &key, source, &retry())
        .await
        .expect("drain");
    assert_eq!(
        step,
        merge::DrainStep::Drained {
            moved: 1,
            remaining: 0
        }
    );
    assert_eq!(
        edge_of(&backend, &manifest, &key, source, remaining_child).await,
        None
    );
    assert!(
        edge_of(&backend, &manifest, &key, other, remaining_child)
            .await
            .is_some(),
        "the Child Entry moved to the same-level Ready internal"
    );
    assert_eq!(
        header_of(&backend, &manifest, &key, other)
            .await
            .expect("other header")
            .entry_count(),
        other_before.entry_count() + 1
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // Completion removes the source's incoming edge and its prefix.
    let completed = merge::complete_merge(&backend, &manifest, &key, source, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_eq!(
        edge_of(&backend, &manifest, &key, pk(1), source).await,
        None
    );
    let root_header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("root header");
    assert_eq!(root_header.entry_count(), 1);
    assert_eq!(root_header.level(), 3, "the root never collapses");
    assert_eq!(header_of(&backend, &manifest, &key, source).await, None);
    assert_eq!(state_of(&backend, &manifest, &key, source).await, None);
    assert!(
        scan_child_entries(&backend, &manifest, &key, source)
            .await
            .is_empty()
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Lifecycle: completion under both backend capability branches.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completion_removes_exactly_the_source_prefix_with_and_without_range_clear() {
    for backend in [backend(), backend_with_clear()] {
        let runtime = make_runtime(backend.clone());
        let index = runtime
            .create_index("index", config())
            .await
            .expect("create");
        let manifest = read_manifest(&backend, index.logical_index_id()).await;
        let key = tree_key(1);
        let mut records = seed_records(&index, 1, 6).await;
        let outcomes = drive_split_to_completion(&backend, &manifest, &key, pk(1)).await;
        assert!(matches!(
            outcomes.last(),
            Some(split::Advance::Completed { .. })
        ));
        trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;

        let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
            .await
            .expect("begin");
        assert_eq!(start, topology::MergeStart::Started);
        let moved = merge_drain_to_zero(&backend, &manifest, &key, pk(2)).await;
        assert_eq!(moved, 1);

        let keys_before = backend.inner().db_key_count();
        let completed = merge::complete_merge(&backend, &manifest, &key, pk(2), &retry())
            .await
            .expect("complete");
        assert_eq!(completed, topology::MergeCompletion::Completed);
        // Exactly the source's four fixed metadata keys — Header, State,
        // Centroid, Synopsis — plus its incoming Child Entry are removed, by
        // one transactional range clear or by bounded point deletes.
        assert_eq!(
            backend.inner().db_key_count(),
            keys_before - 5,
            "exactly the source prefix and its incoming edge are removed"
        );
        assert_eq!(edge_of(&backend, &manifest, &key, pk(1), pk(2)).await, None);
        assert_eq!(header_of(&backend, &manifest, &key, pk(2)).await, None);
        assert_eq!(state_of(&backend, &manifest, &key, pk(2)).await, None);
        assert_eq!(centroid_of(&backend, &manifest, &key, pk(2)).await, None);
        assert_eq!(synopsis_of(&backend, &manifest, &key, pk(2)).await, None);
        // The target's authority values are untouched.
        let target_header = header_of(&backend, &manifest, &key, pk(3))
            .await
            .expect("target header");
        assert_eq!(target_header.state(), PartitionState::Ready);
        assert_eq!(target_header.entry_count(), 4);
        assert_searchable(&backend, &manifest, &key, &records).await;
        run_audit(&backend, &manifest, &records).await;

        runtime.shutdown().await.expect("shutdown");
    }
}

// ---------------------------------------------------------------------------
// Rediscovery: advance alone converges a cold merge, even across a restart.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn advance_rediscovers_and_converges_a_cold_merge() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
    trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;

    // A settled partition and a never-created partition have nothing to do.
    assert_eq!(
        merge::advance(&backend, &manifest, &key, pk(3), 1_000, &retry())
            .await
            .expect("settled"),
        merge::Advance::Idle
    );
    assert_eq!(
        merge::advance(&backend, &manifest, &key, pk(99), 1_000, &retry())
            .await
            .expect("unknown partition is idle"),
        merge::Advance::Idle
    );

    // No worker has run: advance begins the eligible under-minimum partition.
    let began = merge::advance(&backend, &manifest, &key, pk(2), 1_001, &retry())
        .await
        .expect("advance");
    assert_eq!(began, merge::Advance::Began);
    assert_eq!(
        state_of(&backend, &manifest, &key, pk(2)).await,
        Some(PartitionTransition::Merging {
            started_at_unix_millis: 1_001,
        })
    );
    assert_searchable(&backend, &manifest, &key, &records).await;

    // The next pass drains the single entry and, at exact zero, completes.
    let completed = merge::advance(&backend, &manifest, &key, pk(2), 1_002, &retry())
        .await
        .expect("advance");
    assert_eq!(completed, merge::Advance::Completed);
    assert_eq!(header_of(&backend, &manifest, &key, pk(2)).await, None);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // A completed merge has nothing left to advance.
    assert_eq!(
        merge::advance(&backend, &manifest, &key, pk(2), 1_003, &retry())
            .await
            .expect("advance"),
        merge::Advance::Idle
    );

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restarted_process_rediscovers_the_durable_merge_state() {
    let durable = DeterministicConfig {
        durability: Durability::Durable,
        ..DeterministicConfig::default()
    };
    let backend = SharedBackend::new(DeterministicBackend::new(durable));
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
    trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;
    let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);
    runtime.shutdown().await.expect("shutdown");

    // The process is gone; a reopened backend rediscovers the durable Merging
    // state and converges it with advance alone.
    let reopened = SharedBackend::new(backend.inner().reopen());
    let outcomes = drive_merge_to_completion(&reopened, &manifest, &key, pk(2)).await;
    assert_eq!(outcomes.last(), Some(&merge::Advance::Completed));
    assert_searchable(&reopened, &manifest, &key, &records).await;
    run_audit(&reopened, &manifest, &records).await;
}

// ---------------------------------------------------------------------------
// Crash and unknown-outcome recovery at every transition.
// ---------------------------------------------------------------------------

/// Builds a committed two-leaf tree with PK 2 trimmed below the minimum — a
/// merge-eligible leaf — and returns its manifest; the caller drives the
/// merge with explicit transactions.
async fn seed_mergeable_leaf(backend: &SharedBackend) -> (IndexManifest, TreeKey) {
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(backend).await;
    trim_leaf(&index, backend, &manifest, &key, pk(2), 1, &mut records).await;
    runtime.shutdown().await.expect("shutdown");
    (manifest, key)
}

/// Builds a committed merge drained to exact zero: PK 2 is Merging with an
/// empty entry range, ready for completion.
async fn seed_drained_merge(
    backend: &SharedBackend,
) -> (IndexManifest, TreeKey, Vec<(Bytes, f32)>) {
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(backend).await;
    trim_leaf(&index, backend, &manifest, &key, pk(2), 1, &mut records).await;
    let start = merge::begin_merge(backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);
    let moved = merge_drain_to_zero(backend, &manifest, &key, pk(2)).await;
    assert_eq!(moved, 1);
    runtime.shutdown().await.expect("shutdown");
    (manifest, key, records)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn begin_merge_recovers_from_every_commit_outcome() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownNotApplied,
        CommitFault::UnknownApplied,
    ] {
        let backend = backend();
        let (manifest, key) = seed_mergeable_leaf(&backend).await;

        backend.inner().push_fault(fault).expect("push fault");
        let mut txn = write_txn(&backend, &manifest).await;
        let started = topology::begin_merge(&mut txn, &key, pk(2), 1_000)
            .await
            .expect("begin op");
        assert_eq!(started, topology::MergeStart::Started);
        let error = txn.commit().await.expect_err("injected fault");
        assert_fault_kind(fault, &error);

        // Re-driving observes exactly the committed outcome: either the merge
        // still needs to start, or the persisted Merging state is adopted.
        let mut retry_txn = write_txn(&backend, &manifest).await;
        let redriven = topology::begin_merge(&mut retry_txn, &key, pk(2), 1_001)
            .await
            .expect("redriven begin");
        match fault {
            CommitFault::UnknownApplied => assert_eq!(
                redriven,
                topology::MergeStart::AlreadyMerging,
                "the committed state is adopted, never restarted"
            ),
            _ => assert_eq!(
                redriven,
                topology::MergeStart::Started,
                "nothing was applied, so the merge starts"
            ),
        }
        retry_txn.commit().await.expect("retry commits");

        let expected_start = match fault {
            CommitFault::UnknownApplied => 1_000,
            _ => 1_001,
        };
        assert_eq!(
            state_of(&backend, &manifest, &key, pk(2)).await,
            Some(PartitionTransition::Merging {
                started_at_unix_millis: expected_start,
            })
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_recovers_from_unknown_outcomes_without_losing_membership() {
    for fault in [CommitFault::UnknownNotApplied, CommitFault::UnknownApplied] {
        let backend = backend();
        let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
        trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;
        let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
            .await
            .expect("begin");
        assert_eq!(start, topology::MergeStart::Started);

        // The batch's commit reports an unknown outcome; it may or may not
        // have applied.
        backend.inner().push_fault(fault).expect("push fault");
        let error = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
            .await
            .expect_err("unknown outcome");
        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
        // Never retried blindly; rediscovery observes the persisted state.
        assert_searchable(&backend, &manifest, &key, &records).await;
        let redriven = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
            .await
            .expect("redriven drain");
        match fault {
            CommitFault::UnknownApplied => assert_eq!(
                redriven,
                merge::DrainStep::Drained {
                    moved: 0,
                    remaining: 0
                },
                "the applied batch stands; nothing is left to move"
            ),
            _ => assert_eq!(
                redriven,
                merge::DrainStep::Drained {
                    moved: 1,
                    remaining: 0
                },
                "nothing was applied, so the batch moves the entry"
            ),
        }
        let completed = merge::complete_merge(&backend, &manifest, &key, pk(2), &retry())
            .await
            .expect("complete");
        assert_eq!(completed, topology::MergeCompletion::Completed);
        assert_searchable(&backend, &manifest, &key, &records).await;
        run_audit(&backend, &manifest, &records).await;

        runtime.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_recovers_from_every_commit_outcome() {
    for (clear, removal) in [
        (false, topology::SourceRemoval::PointDeletes),
        (true, topology::SourceRemoval::TransactionalClear),
    ] {
        for fault in [
            CommitFault::Abort,
            CommitFault::UnknownNotApplied,
            CommitFault::UnknownApplied,
        ] {
            let backend = if clear {
                backend_with_clear()
            } else {
                backend()
            };
            let (manifest, key, records) = seed_drained_merge(&backend).await;

            backend.inner().push_fault(fault).expect("push fault");
            let mut txn = write_txn(&backend, &manifest).await;
            let completed = topology::finalize_merge(&mut txn, &key, pk(2), removal)
                .await
                .expect("finalize op");
            assert_eq!(completed, topology::MergeCompletion::Completed);
            let error = txn.commit().await.expect_err("injected fault");
            assert_fault_kind(fault, &error);

            // Re-driving observes the removed source and reports completion
            // whether or not the faulted commit applied.
            let mut retry_txn = write_txn(&backend, &manifest).await;
            let redriven = topology::finalize_merge(&mut retry_txn, &key, pk(2), removal)
                .await
                .expect("redriven finalize");
            assert_eq!(redriven, topology::MergeCompletion::Completed);
            retry_txn.commit().await.expect("retry commits");

            // The completed topology stands: the source is gone, its incoming
            // edge is removed, and no target state changed.
            assert_eq!(header_of(&backend, &manifest, &key, pk(2)).await, None);
            assert_eq!(state_of(&backend, &manifest, &key, pk(2)).await, None);
            assert_eq!(edge_of(&backend, &manifest, &key, pk(1), pk(2)).await, None);
            assert_eq!(
                header_of(&backend, &manifest, &key, pk(3))
                    .await
                    .expect("target header")
                    .state(),
                PartitionState::Ready
            );
            assert_searchable(&backend, &manifest, &key, &records).await;
            run_audit(&backend, &manifest, &records).await;
        }
    }
}

// ---------------------------------------------------------------------------
// Conflicts abort bounded steps and retry from fresh snapshots.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_concurrent_source_write_aborts_begin_merge() {
    let backend = backend();
    let (manifest, key) = seed_mergeable_leaf(&backend).await;

    // A concurrent write to the source Header conflicts with begin's
    // update-protected authority reads.
    let mut attempt = write_txn(&backend, &manifest).await;
    let started = topology::begin_merge(&mut attempt, &key, pk(2), 1_000)
        .await
        .expect("begin op");
    assert_eq!(started, topology::MergeStart::Started);
    let header = header_of(&backend, &manifest, &key, pk(2))
        .await
        .expect("header");
    let mut concurrent = write_txn(&backend, &manifest).await;
    concurrent
        .put(
            LogicalKey::Header {
                index: manifest.logical_index_id(),
                tree_key: key.clone(),
                partition: pk(2),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(
                    1,
                    header.entry_count(),
                    header.cache_epoch() + 1,
                    PartitionState::Ready,
                )
                .expect("header"),
            ),
        )
        .await
        .expect("concurrent write");
    concurrent.commit().await.expect("concurrent commits");
    let error = attempt.commit().await.expect_err("begin conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // The retried begin observes the touched Header and starts: the count is
    // still below the minimum.
    let mut retried = write_txn(&backend, &manifest).await;
    let started = topology::begin_merge(&mut retried, &key, pk(2), 1_001)
        .await
        .expect("retried begin");
    assert_eq!(started, topology::MergeStart::Started);
    retried.commit().await.expect("retry commits");
    assert_eq!(
        state_of(&backend, &manifest, &key, pk(2)).await,
        Some(PartitionTransition::Merging {
            started_at_unix_millis: 1_001,
        })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_concurrent_delete_conflicts_with_a_drain_batch_and_the_retry_skips_it() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = wide_two_leaf_tree(&backend).await;
    let source = pk(2);
    let target = pk(3);
    trim_leaf(&index, &backend, &manifest, &key, source, 12, &mut records).await;
    let start = merge::begin_merge(&backend, &manifest, &key, source, 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);

    // The drain's update-protected entry read conflicts with a concurrent
    // delete of the same record.
    let victim = leaf_ids(&backend, &manifest, &key, source)
        .await
        .first()
        .expect("a first entry")
        .clone();
    let mut attempt = write_txn(&backend, &manifest).await;
    let candidate = topology::read_leaf_drain_candidates(
        &mut attempt,
        &key,
        source,
        std::slice::from_ref(&victim),
    )
    .await
    .expect("read candidate")
    .into_iter()
    .next()
    .expect("one slot")
    .expect("candidate exists");
    assert!(index.delete(victim.clone()).await.expect("delete"));
    records.retain(|(id, _)| id != &victim);
    topology::relocate_leaf_entries(
        &mut attempt,
        &key,
        source,
        vec![(candidate, target)],
        topology::Movement::Merge,
    )
    .await
    .expect("relocate op");
    let error = attempt.commit().await.expect_err("delete conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // The retried batch skips the vanished entry: eleven remain, eight move.
    let first = merge::drain_batch(&backend, &manifest, &key, source, &retry())
        .await
        .expect("retried batch");
    assert_eq!(
        first,
        merge::DrainStep::Drained {
            moved: 8,
            remaining: 3
        }
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    let moved = merge_drain_to_zero(&backend, &manifest, &key, source).await;
    assert_eq!(moved, 3);
    let completed = merge::complete_merge(&backend, &manifest, &key, source, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_concurrent_target_transition_aborts_the_relocate_and_the_next_batch_reselects() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = three_leaf_tree(&backend).await;

    // Grow one right-side leaf past the maximum so it can begin a split.
    for (n, x) in [
        (20, 0.1),
        (21, 0.2),
        (22, 0.3),
        (23, 0.9),
        (24, 1.1),
        (25, 1.3),
    ] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let mut split_target = None;
    for leaf in [pk(4), pk(5)] {
        let header = header_of(&backend, &manifest, &key, leaf)
            .await
            .expect("leaf header");
        if header.entry_count() > 4 {
            split_target = Some(leaf);
            break;
        }
    }
    let split_target = split_target.expect("one leaf above the maximum");
    let other_target = if split_target == pk(4) { pk(5) } else { pk(4) };

    // The merge source is PK 3, trimmed to one entry.
    let kept = trim_leaf(&index, &backend, &manifest, &key, pk(3), 1, &mut records).await;
    let start = merge::begin_merge(&backend, &manifest, &key, pk(3), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);

    // A drain attempt relocating to the chosen target conflicts with the
    // target's concurrent transition out of Ready.
    let mut attempt = write_txn(&backend, &manifest).await;
    let candidate = topology::read_leaf_drain_candidates(
        &mut attempt,
        &key,
        pk(3),
        std::slice::from_ref(&kept[0].0),
    )
    .await
    .expect("read candidate")
    .into_iter()
    .next()
    .expect("one slot")
    .expect("candidate exists");
    topology::relocate_leaf_entries(
        &mut attempt,
        &key,
        pk(3),
        vec![(candidate, split_target)],
        topology::Movement::Merge,
    )
    .await
    .expect("relocate op");
    let split_start = split::begin_split(&backend, &manifest, &key, split_target, 2_000, &retry())
        .await
        .expect("concurrent begin split");
    assert!(matches!(split_start, topology::SplitStart::Started { .. }));
    let error = attempt
        .commit()
        .await
        .expect_err("the target transition conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // The next batch reselects deterministically: the splitting target is
    // skipped and the entry moves to the remaining Ready target.
    let step = merge::drain_batch(&backend, &manifest, &key, pk(3), &retry())
        .await
        .expect("reselecting batch");
    assert_eq!(
        step,
        merge::DrainStep::Drained {
            moved: 1,
            remaining: 0
        }
    );
    assert_eq!(
        location_of(&backend, &manifest, &kept[0].0)
            .await
            .expect("location")
            .leaf(),
        other_target
    );
    let completed = merge::complete_merge(&backend, &manifest, &key, pk(3), &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // The interrupted split still converges on its own state machine.
    let outcomes = drive_split_to_completion(&backend, &manifest, &key, split_target).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Target reselection (ADR 0008).
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_target_leaving_ready_between_batches_is_skipped_by_reselection() {
    let backend = backend();
    let fixture = begin_middle_leaf_merge(&backend).await;
    let MiddleLeafMerge {
        runtime,
        index,
        manifest,
        key,
        mut records,
        source,
        targets,
        kept,
    } = fixture;

    // The first batch moves the eight smallest entries to their nearest
    // Ready targets.
    let first = merge::drain_batch(&backend, &manifest, &key, source, &retry())
        .await
        .expect("first batch");
    assert_eq!(
        first,
        merge::DrainStep::Drained {
            moved: 8,
            remaining: 4
        }
    );

    // The remaining four entries' nearest target leaves Ready: it is grown
    // past the maximum and begins a split.
    let mut remaining = kept.clone();
    remaining.sort_by(|left, right| left.0.cmp(&right.0));
    let flip = expected_target(remaining[8].1, &targets);
    let other = if flip == targets[0].0 {
        targets[1].0
    } else {
        targets[0].0
    };
    let centroid = leaf_centroid(&backend, &manifest, &key, flip).await;
    for n in 0..16_u8 {
        let x = centroid - 2.0 + f32::from(n) * 0.5;
        let id = rid(100 + n);
        index.insert(record(&id, x, 1)).await.expect("insert");
        assert_eq!(
            location_of(&backend, &manifest, &id)
                .await
                .expect("location")
                .leaf(),
            flip,
            "the growth inserts land on the flipping target"
        );
        records.push((id, x));
    }
    let split_start = split::begin_split(&backend, &manifest, &key, flip, 2_000, &retry())
        .await
        .expect("begin split");
    assert!(matches!(split_start, topology::SplitStart::Started { .. }));

    // The next batch reselects: every remaining entry moves to the one
    // remaining Ready target, however near the splitting target's centroid.
    let second = merge::drain_batch(&backend, &manifest, &key, source, &retry())
        .await
        .expect("second batch");
    assert_eq!(
        second,
        merge::DrainStep::Drained {
            moved: 4,
            remaining: 0
        }
    );
    for (id, _) in &remaining[8..] {
        assert_eq!(
            location_of(&backend, &manifest, id)
                .await
                .expect("location")
                .leaf(),
            other,
            "a non-Ready target is skipped"
        );
    }
    let completed = merge::complete_merge(&backend, &manifest, &key, source, &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_nearest_ready_target_wins_with_the_partition_key_tie_break() {
    // (near centroid of PK 4, near centroid of PK 5, expected target): first
    // the plainly nearer target wins even with the larger Partition Key, then
    // an exact distance tie breaks to the smaller Partition Key.
    for (centroid_a, centroid_b, expected) in
        [(100.0_f32, 6.0_f32, pk(5)), (4.0_f32, 6.0_f32, pk(4))]
    {
        let backend = backend();
        let (runtime, index, manifest, key, mut records) = three_leaf_tree(&backend).await;
        // The merge source is PK 3, trimmed to its last entry (x = 5.0).
        let kept = trim_leaf(&index, &backend, &manifest, &key, pk(3), 1, &mut records).await;
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].1, 5.0);

        // Craft the two targets' incoming-edge centroids: persisted centroids
        // are routing models, so overwriting them is a legal topology.
        for (child, centroid) in [(pk(4), centroid_a), (pk(5), centroid_b)] {
            let mut txn = write_txn(&backend, &manifest).await;
            txn.put(
                LogicalKey::ChildEntry {
                    index: manifest.logical_index_id(),
                    tree_key: key.clone(),
                    partition: pk(1),
                    child,
                },
                PersistentValue::ChildEntry(ChildEntry::new(child, vec![centroid])),
            )
            .await
            .expect("put edge");
            txn.commit().await.expect("commit edge");
        }

        let start = merge::begin_merge(&backend, &manifest, &key, pk(3), 1_000, &retry())
            .await
            .expect("begin");
        assert_eq!(start, topology::MergeStart::Started);
        let step = merge::drain_batch(&backend, &manifest, &key, pk(3), &retry())
            .await
            .expect("drain");
        assert_eq!(
            step,
            merge::DrainStep::Drained {
                moved: 1,
                remaining: 0
            }
        );
        assert_eq!(
            location_of(&backend, &manifest, &kept[0].0)
                .await
                .expect("location")
                .leaf(),
            expected,
            "centroids {centroid_a} vs {centroid_b}"
        );
        let completed = merge::complete_merge(&backend, &manifest, &key, pk(3), &retry())
            .await
            .expect("complete");
        assert_eq!(completed, topology::MergeCompletion::Completed);
        assert_searchable(&backend, &manifest, &key, &records).await;
        run_audit(&backend, &manifest, &records).await;

        runtime.shutdown().await.expect("shutdown");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_target_may_cross_the_split_threshold_while_receiving() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;

    // Grow the target to exactly the maximum of four.
    index
        .insert(record(&rid(10), 3.5, 1))
        .await
        .expect("insert");
    records.push((rid(10), 3.5));
    let target_header = header_of(&backend, &manifest, &key, pk(3))
        .await
        .expect("target header");
    assert_eq!(target_header.entry_count(), 4);

    trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;
    let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);
    let step = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("drain");
    assert_eq!(
        step,
        merge::DrainStep::Drained {
            moved: 1,
            remaining: 0
        }
    );

    // The target now exceeds the maximum and stays Ready: crossing the split
    // threshold is legal and starts no split by itself.
    let target_header = header_of(&backend, &manifest, &key, pk(3))
        .await
        .expect("target header");
    assert_eq!(target_header.entry_count(), 5);
    assert_eq!(target_header.state(), PartitionState::Ready);
    let completed = merge::complete_merge(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // A later split of the over-maximum target converges normally.
    let outcomes = drive_split_to_completion(&backend, &manifest, &key, pk(3)).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Stall rules: no Ready target means no progress, never a revert to Ready.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn begin_without_a_legal_ready_target_starts_nothing() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
    trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;

    // The only same-level candidate leaves Ready: it begins a split.
    for (n, x) in [(10, 3.5), (11, 4.5), (12, 5.5)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let split_start = split::begin_split(&backend, &manifest, &key, pk(3), 2_000, &retry())
        .await
        .expect("begin split");
    assert!(matches!(split_start, topology::SplitStart::Started { .. }));

    // Begin validates the target set before changing anything: no state
    // starts, and the source stays Ready.
    let before = state_of(&backend, &manifest, &key, pk(2)).await;
    let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::NoReadyTarget);
    assert_eq!(state_of(&backend, &manifest, &key, pk(2)).await, before);
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(2))
            .await
            .expect("source header")
            .state(),
        PartitionState::Ready
    );

    // Rediscovery reports the same: nothing to begin.
    assert_eq!(
        merge::advance(&backend, &manifest, &key, pk(2), 1_001, &retry())
            .await
            .expect("advance"),
        merge::Advance::Idle
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merging_source_stalls_searchable_until_a_ready_target_returns() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
    let kept = trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;
    assert_eq!(kept.len(), 1);

    // The merge begins while a Ready target exists.
    let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);

    // Then the only target leaves Ready.
    for (n, x) in [(10, 3.5), (11, 4.5), (12, 5.5)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let split_start = split::begin_split(&backend, &manifest, &key, pk(3), 2_000, &retry())
        .await
        .expect("begin split");
    assert!(matches!(split_start, topology::SplitStart::Started { .. }));

    // The drain reports the stall; advance surfaces it. The source stays
    // Merging — it never reverts to Ready.
    let step = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("stalled drain");
    assert_eq!(step, merge::DrainStep::NoReadyTarget);
    let advance = merge::advance(&backend, &manifest, &key, pk(2), 1_001, &retry())
        .await
        .expect("advance");
    assert_eq!(advance, merge::Advance::Stalled);
    assert_eq!(
        state_of(&backend, &manifest, &key, pk(2)).await,
        Some(PartitionTransition::Merging {
            started_at_unix_millis: 1_000,
        })
    );
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(2))
            .await
            .expect("source header")
            .entry_count(),
        1
    );

    // The stalled source stays searchable: exact membership holds and the
    // public search path visits it as an ordinary body.
    assert_exact_membership(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;
    let outcome = index
        .search(SearchRequest::new(Arc::from([kept[0].1]), 1).expect("request"))
        .await
        .expect("search");
    assert_eq!(outcome.hits.len(), 1);
    assert_eq!(outcome.hits[0].id(), &kept[0].0);

    // Once a Ready target exists again — the target's split completed,
    // publishing two Ready leaves — a later drain succeeds.
    let outcomes = drive_split_to_completion(&backend, &manifest, &key, pk(3)).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    let step = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("recovered drain");
    assert_eq!(
        step,
        merge::DrainStep::Drained {
            moved: 1,
            remaining: 0
        }
    );
    let completed = merge::complete_merge(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Foreground interaction with a Merging leaf.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_writes_reroute_around_a_merging_leaf() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
    let kept = trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;
    let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);

    // An insert whose vector is nearest the Merging leaf's centroid lands on
    // the reselected Ready leaf instead.
    index
        .insert(record(&rid(10), 1.0, 1))
        .await
        .expect("insert");
    records.push((rid(10), 1.0));
    assert_eq!(
        location_of(&backend, &manifest, &rid(10))
            .await
            .expect("location")
            .leaf(),
        pk(3),
        "no insert enters a Merging leaf"
    );

    // An upsert whose Record Location names the source relocates atomically:
    // the source entry, its count, and the location move in one commit.
    let relocated = kept[0].0.clone();
    index
        .upsert(record(&relocated, 4.75, 1))
        .await
        .expect("upsert");
    if let Some(slot) = records.iter_mut().find(|(id, _)| id == &relocated) {
        slot.1 = 4.75;
    }
    let location = location_of(&backend, &manifest, &relocated)
        .await
        .expect("location");
    assert_eq!(location.leaf(), pk(3));
    assert!(
        leaf_entry_of(&backend, &manifest, &key, pk(2), &relocated)
            .await
            .is_none()
    );
    assert!(
        leaf_entry_of(&backend, &manifest, &key, pk(3), &relocated)
            .await
            .is_some()
    );
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(2))
            .await
            .expect("source header")
            .entry_count(),
        0,
        "the relocation decremented the exact source count"
    );
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    // The source is drained by the foreground; the merge completes normally.
    let step = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("drain");
    assert_eq!(
        step,
        merge::DrainStep::Drained {
            moved: 0,
            remaining: 0
        }
    );
    let completed = merge::complete_merge(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_writes_fail_with_contention_exhausted_when_no_ready_target_remains() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
    let kept = trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;
    let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);

    // The only same-level alternative leaves Ready.
    for (n, x) in [(10, 3.5), (11, 4.5), (12, 5.5)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let split_start = split::begin_split(&backend, &manifest, &key, pk(3), 2_000, &retry())
        .await
        .expect("begin split");
    assert!(matches!(split_start, topology::SplitStart::Started { .. }));

    // An insert descending into the Merging leaf retries bounded and then
    // reports ContentionExhausted rather than growing the source.
    let error = index
        .insert(record(&rid(20), 1.9, 1))
        .await
        .expect_err("no Ready target");
    assert_eq!(error.kind(), ErrorKind::ContentionExhausted);
    // An upsert of a record located in the source cannot relocate either.
    let error = index
        .upsert(record(&kept[0].0, 2.5, 1))
        .await
        .expect_err("no Ready target");
    assert_eq!(error.kind(), ErrorKind::ContentionExhausted);
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(2))
            .await
            .expect("source header")
            .entry_count(),
        1,
        "the source never grows"
    );

    // An exact delete follows the stored location and works throughout.
    assert!(index.delete(kept[0].0.clone()).await.expect("delete"));
    records.retain(|(id, _)| id != &kept[0].0);
    assert_eq!(
        header_of(&backend, &manifest, &key, pk(2))
            .await
            .expect("source header")
            .entry_count(),
        0
    );
    assert_exact_membership(&backend, &manifest, &key, &records).await;

    // Once the split publishes Ready targets, foreground writes succeed and
    // the stalled merge completes.
    let outcomes = drive_split_to_completion(&backend, &manifest, &key, pk(3)).await;
    assert!(matches!(
        outcomes.last(),
        Some(split::Advance::Completed { .. })
    ));
    index
        .insert(record(&rid(20), 1.9, 1))
        .await
        .expect("insert");
    records.push((rid(20), 1.9));
    let step = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("drain");
    assert_eq!(
        step,
        merge::DrainStep::Drained {
            moved: 0,
            remaining: 0
        }
    );
    let completed = merge::complete_merge(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("complete");
    assert_eq!(completed, topology::MergeCompletion::Completed);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_root_never_merges() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);

    // A lone root leaf holds one record — below the minimum — but is the
    // root: never merge-eligible.
    index.insert(record(&rid(0), 0.0, 1)).await.expect("insert");
    let root_header = header_of(&backend, &manifest, &key, pk(1))
        .await
        .expect("root header");
    assert_eq!(root_header.entry_count(), 1);
    assert!(root_header.entry_count() < manifest.config().min_partition_entries());
    let start = merge::begin_merge(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::NotEligible);
    assert_eq!(
        merge::advance(&backend, &manifest, &key, pk(1), 1_001, &retry())
            .await
            .expect("advance"),
        merge::Advance::Idle
    );

    // An empty committed root is likewise ineligible.
    let empty_key = tree_key(2);
    create_committed_tree(&backend, &manifest, &empty_key).await;
    let start = merge::begin_merge(&backend, &manifest, &empty_key, pk(1), 1_002, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::NotEligible);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn begin_reports_not_eligible_for_ineligible_sources() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;

    // At or above the minimum: nothing to merge.
    let start = merge::begin_merge(&backend, &manifest, &key, pk(3), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::NotEligible);

    // An intermediate-state partition cannot start another structural change.
    for (n, x) in [(10, 3.5), (11, 4.5), (12, 5.5)] {
        index.insert(record(&rid(n), x, 1)).await.expect("insert");
        records.push((rid(n), x));
    }
    let split_start = split::begin_split(&backend, &manifest, &key, pk(3), 2_000, &retry())
        .await
        .expect("begin split");
    assert!(matches!(split_start, topology::SplitStart::Started { .. }));
    let start = merge::begin_merge(&backend, &manifest, &key, pk(3), 1_001, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::NotEligible);

    // A partition that never existed has nothing to start.
    let start = merge::begin_merge(&backend, &manifest, &key, pk(99), 1_002, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::NotEligible);

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Fail-closed corruption paths.
// ---------------------------------------------------------------------------

/// Installs a drained non-root merge fixture: root PK 1 at level 2 with leaf
/// children PK 2 (the drained Merging source) and PK 3 (Ready). Returns after
/// committing the fixture.
async fn seed_completable_non_root_merge(
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
                PartitionHeader::new(1, 0, 3, PartitionState::Merging).expect("header"),
            ),
        ),
        (
            LogicalKey::State {
                index,
                tree_key: key.clone(),
                partition: pk(2),
            },
            PersistentValue::PartitionState(PartitionTransition::Merging {
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
    txn.commit().await.expect("commit fixture");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_requires_merging_state_and_an_exact_zero_count() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    seed_completable_non_root_merge(&backend, &manifest, &key).await;

    // A nonzero exact count refuses completion without an entry rescan.
    let mut txn = write_txn(&backend, &manifest).await;
    txn.put(
        LogicalKey::Header {
            index: manifest.logical_index_id(),
            tree_key: key.clone(),
            partition: pk(2),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 1, 3, PartitionState::Merging).expect("header"),
        ),
    )
    .await
    .expect("put nonzero header");
    txn.commit().await.expect("commit");
    let mut txn = write_txn(&backend, &manifest).await;
    let outcome =
        topology::finalize_merge(&mut txn, &key, pk(2), topology::SourceRemoval::PointDeletes)
            .await
            .expect("finalize");
    assert_eq!(outcome, topology::MergeCompletion::NotDrained);
    txn.rollback().await;

    // A non-Merging source has nothing to complete.
    let mut txn = write_txn(&backend, &manifest).await;
    let outcome =
        topology::finalize_merge(&mut txn, &key, pk(3), topology::SourceRemoval::PointDeletes)
            .await
            .expect("finalize a Ready partition");
    assert_eq!(outcome, topology::MergeCompletion::NotMerging);
    txn.rollback().await;

    runtime.shutdown().await.expect("shutdown");
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
    seed_completable_non_root_merge(&backend, &manifest, &key).await;

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
    let error =
        topology::finalize_merge(&mut txn, &key, pk(2), topology::SourceRemoval::PointDeletes)
            .await
            .expect_err("missing incoming edge");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;

    // Duplicate edge: a second reachable level-2 partition also references
    // the source. The fixture rises one level so the parent level holds two
    // bodies: root PK 1 at level 3 over internals PK 6 and PK 7, with the
    // source's real edge in PK 6 and a duplicate in PK 7.
    seed_completable_non_root_merge(&backend, &manifest, &key).await;
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
    let error =
        topology::finalize_merge(&mut txn, &key, pk(2), topology::SourceRemoval::PointDeletes)
            .await
            .expect_err("duplicate incoming edge");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;

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

    // A Merging leaf whose exact count claims one entry while the entry
    // range is empty: within one snapshot the two must agree. The drain read
    // phase fails before any candidate is read, so the fixture needs only
    // the root edge and the source's authority pair.
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
            PartitionHeader::new(2, 1, 0, PartitionState::Ready).expect("header"),
        ),
    )
    .await
    .expect("put root header");
    txn.put(
        LogicalKey::ChildEntry {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(1),
            child: pk(2),
        },
        PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.0])),
    )
    .await
    .expect("put edge");
    txn.put(
        LogicalKey::Header {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(2),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 1, 0, PartitionState::Merging).expect("header"),
        ),
    )
    .await
    .expect("put source header");
    txn.put(
        LogicalKey::State {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(2),
        },
        PersistentValue::PartitionState(PartitionTransition::Merging {
            started_at_unix_millis: 200,
        }),
    )
    .await
    .expect("put source state");
    txn.commit().await.expect("commit fixture");

    let error = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect_err("a positive count with an empty entry range is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    let error = merge::advance(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect_err("advance drains and fails closed the same way");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_fails_closed_on_a_malformed_record_location() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
    let kept = trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;
    let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::Started);

    // Corrupt the draining record's Location through the raw seam.
    let mut raw = backend.inner().begin_write().await.expect("begin write");
    raw.put(
        Bytes::from(
            keys::location_key(manifest.logical_index_id(), &kept[0].0).expect("location key"),
        ),
        Bytes::from_static(b"invalid"),
    )
    .await
    .expect("raw put");
    raw.commit().await.expect("commit");

    let error = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect_err("malformed location");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    // The corrupted source stays Merging for offline diagnosis.
    assert_eq!(
        state_of(&backend, &manifest, &key, pk(2)).await,
        Some(PartitionTransition::Merging {
            started_at_unix_millis: 1_000,
        })
    );

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_and_advance_fail_closed_on_torn_authority_values() {
    let backend = backend();
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create");
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    let index_id = manifest.logical_index_id();

    // A State without its Header: a torn committed state.
    let mut txn = write_txn(&backend, &manifest).await;
    txn.put(
        LogicalKey::State {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(2),
        },
        PersistentValue::PartitionState(PartitionTransition::Merging {
            started_at_unix_millis: 200,
        }),
    )
    .await
    .expect("put torn state");
    txn.commit().await.expect("commit");
    let error = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect_err("state without header");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    let error = merge::advance(&backend, &manifest, &key, pk(2), 1_000, &retry())
        .await
        .expect_err("state without header");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    // A Header without its State is equally torn.
    let mut txn = write_txn(&backend, &manifest).await;
    txn.delete(LogicalKey::State {
        index: index_id,
        tree_key: key.clone(),
        partition: pk(2),
    })
    .await
    .expect("delete state");
    txn.put(
        LogicalKey::Header {
            index: index_id,
            tree_key: key.clone(),
            partition: pk(2),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 0, 0, PartitionState::Merging).expect("header"),
        ),
    )
    .await
    .expect("put torn header");
    txn.commit().await.expect("commit");
    let error = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect_err("header without state");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    let error = merge::advance(&backend, &manifest, &key, pk(2), 1_001, &retry())
        .await
        .expect_err("header without state");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Recovery and fail-closed regressions.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn steps_on_a_completed_merge_are_harmless_noops() {
    let backend = backend();
    let (runtime, index, manifest, key, mut records) = two_leaf_tree(&backend).await;
    trim_leaf(&index, &backend, &manifest, &key, pk(2), 1, &mut records).await;
    merge_partition(&backend, &manifest, &key, pk(2)).await;
    assert!(header_of(&backend, &manifest, &key, pk(2)).await.is_none());
    assert!(state_of(&backend, &manifest, &key, pk(2)).await.is_none());

    // A competing or recovering worker that re-drives any step of the
    // finished merge observes the removal and gets a graceful outcome, never
    // a spurious Corruption (maintenance.md §3).
    let start = merge::begin_merge(&backend, &manifest, &key, pk(2), 400, &retry())
        .await
        .expect("begin");
    assert_eq!(start, topology::MergeStart::NotEligible);
    let drained = merge::drain_batch(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("drain");
    assert_eq!(drained, merge::DrainStep::SourceAdvanced);
    let completion = merge::complete_merge(&backend, &manifest, &key, pk(2), &retry())
        .await
        .expect("complete");
    assert_eq!(completion, topology::MergeCompletion::Completed);
    let advance = merge::advance(&backend, &manifest, &key, pk(2), 400, &retry())
        .await
        .expect("advance");
    assert_eq!(advance, merge::Advance::Idle);
    assert_searchable(&backend, &manifest, &key, &records).await;
    run_audit(&backend, &manifest, &records).await;

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

    // Every entry point validates the persisted Manifest instead of
    // reporting the missing topology keys as Corruption.
    let error = merge::advance(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect_err("a dropped index rejects maintenance");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);
    let error = merge::drain_batch(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect_err("a dropped index rejects draining");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);
    let error = merge::begin_merge(&backend, &manifest, &key, pk(1), 1_000, &retry())
        .await
        .expect_err("a dropped index rejects beginning");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);
    let error = merge::complete_merge(&backend, &manifest, &key, pk(1), &retry())
        .await
        .expect_err("a dropped index rejects completion");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);

    runtime.shutdown().await.expect("shutdown");
}

// ---------------------------------------------------------------------------
// Model history: mutations interleaved with splits, merges, and abort faults.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seeded_model_history_interleaving_mutations_splits_and_merges() {
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
        match rng.below(12) {
            // Upsert a record (create or replace).
            0..=4 => {
                let id = rid(rng.below(40) as u8);
                let x = (rng.below(400) as f32) / 10.0;
                match index.upsert(record(&id, x, 1)).await {
                    Ok(_) => {
                        model.insert(id, x);
                    }
                    Err(error) => {
                        // An unknown outcome is recovered by readback; a
                        // ContentionExhausted means the descent met a stalled
                        // merge — a definite non-application (ADR 0008).
                        assert!(
                            matches!(
                                error.kind(),
                                ErrorKind::CommitOutcomeUnknown | ErrorKind::ContentionExhausted
                            ),
                            "unexpected upsert error: {error:?}"
                        );
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
            7..=8 => {
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
                            "split advance {partition:?} failed at step {step}: {error:?}; header {header:?}; state {state:?}"
                        );
                    }
                }
            }
            // Advance one random partition's merge state.
            _ => {
                let partitions = all_partitions(&backend, &manifest, &key).await;
                if !partitions.is_empty() {
                    let partition = partitions[rng.below(partitions.len() as u64) as usize];
                    if let Err(error) = merge::advance(
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
                            "merge advance {partition:?} failed at step {step}: {error:?}; header {header:?}; state {state:?}"
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
                    Err(error) => panic!("split advance: {error:?}"),
                };
            if outcome != split::Advance::Idle {
                progressed = true;
            }
            let outcome = merge::advance(&backend, &manifest, &key, *partition, 60_000, &retry())
                .await
                .expect("merge advance");
            if outcome == merge::Advance::Stalled {
                blocked = true;
            }
            if !matches!(outcome, merge::Advance::Idle | merge::Advance::Stalled) {
                progressed = true;
            }
        }
        if !progressed {
            assert!(
                !blocked,
                "topology convergence stalled on parent maintenance"
            );
            settled = true;
            break;
        }
    }
    assert!(settled, "topology convergence exceeded the bounded drive");
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
    run_audit(&backend, &manifest, &model_records).await;

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
