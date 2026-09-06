//! Shared probes and drivers for split/merge state-machine tests.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, Index, IndexConfig, Metric, PartitionKey, Record,
    RuntimeConfig, Value,
};
use ktann::maintenance::routing::route_leaf;
use ktann::maintenance::{merge, split};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::backend::{Backend, Capabilities, ScanLimits};
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexManifest, LeafEntry, PartitionCentroid, PartitionHeader, PartitionSynopsis,
    PartitionTransition, PersistentValue, RecordLocation,
};
use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn, tree_manifest};

use super::{CommitFault, DeterministicBackend, DeterministicConfig, SharedBackend};

pub fn backend() -> SharedBackend {
    SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()))
}

pub fn backend_with_clear() -> SharedBackend {
    let config = DeterministicConfig {
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        ..DeterministicConfig::default()
    };
    SharedBackend::new(DeterministicBackend::new(config))
}

/// A one-dimensional L2 index over one i64 tree-key field with the given
/// partition entry bounds, so small fixtures trigger splits and merges.
pub fn config(minimum: u32, maximum: u32) -> IndexConfig {
    IndexConfig::new(1, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(minimum, maximum)
        .expect("valid partition entries")
}

pub fn make_runtime(backend: SharedBackend) -> Runtime<SharedBackend> {
    // These suites drive the state machines by hand; background maintenance
    // workers would race the manual drives.
    Runtime::new(backend, super::manual_maintenance_config()).expect("runtime is valid")
}

pub fn retry() -> RetryPolicy {
    RetryPolicy::for_fixup(&RuntimeConfig::default())
}

pub fn tree_key(bucket: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(bucket)]).expect("canonical key")
}

pub fn rid(value: u8) -> Bytes {
    Bytes::copy_from_slice(&[b'r', value])
}

pub fn record(id: &[u8], x: f32, bucket: i64) -> Record {
    Record::new(
        Bytes::copy_from_slice(id),
        Arc::from([x]),
        vec![Value::I64(bucket)],
    )
    .expect("valid record")
}

pub fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

pub async fn read_txn<'b, 'm>(
    backend: &'b SharedBackend,
    manifest: &'m IndexManifest,
) -> ReadLogicalTxn<'m, <SharedBackend as Backend>::ReadTxn<'b>> {
    let raw = backend.begin_read().await.expect("begin read");
    ReadLogicalTxn::for_index(raw, manifest).expect("bind manifest")
}

pub async fn write_txn<'b, 'm>(
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
pub async fn create_committed_tree(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
) {
    let mut txn = write_txn(backend, manifest).await;
    tree_manifest::create_tree(&mut txn, key, 100)
        .await
        .expect("create tree");
    txn.commit().await.expect("commit tree");
}

pub async fn header_of(
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

pub async fn state_of(
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

pub async fn centroid_of(
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

pub async fn synopsis_of(
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

pub async fn location_of(
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

pub async fn leaf_entry_of(
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

pub async fn edge_of(
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

pub async fn scan_child_entries(
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

pub async fn scan_leaf_entries(
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
pub async fn reachable_leaves(
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
pub async fn assert_exact_membership(
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
pub async fn assert_searchable(
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
pub async fn seed_records(
    index: &Index<SharedBackend>,
    bucket: i64,
    count: u8,
) -> Vec<(Bytes, f32)> {
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
pub async fn drive_split_to_completion(
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
pub async fn drive_merge_to_completion(
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

/// Asserts the error kind one injected commit fault produces.
pub fn assert_fault_kind(fault: CommitFault, error: &ktann::api::Error) {
    let expected = match fault {
        CommitFault::Abort => ErrorKind::RetryableAbort,
        _ => ErrorKind::CommitOutcomeUnknown,
    };
    assert_eq!(error.kind(), expected, "fault {fault:?}");
}

/// Enumerates every partition of one tree, following Child Entries and the
/// root's persisted split target slots.
pub async fn all_partitions(
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
