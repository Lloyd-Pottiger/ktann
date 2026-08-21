//! The persistent-state audit and deterministic tree rendering shared by the
//! data-driven corpus.
//!
//! [`run`] proves the exact-membership invariant against the caller's model:
//! every modeled Vector Record has exactly one Record Location and one
//! corresponding Leaf Entry, every reachable partition's exact Header count
//! matches its scanned entry set, levels decrement by exactly one, and every
//! non-root partition has exactly one incoming Child Entry. It is the corpus
//! form of the focused assertion helpers used by the state-machine suites,
//! and stays valid for every committed topology state.
//!
//! [`render_tree`] renders the reachable topology deterministically so corpus
//! files can diff structure step by step.
//!
//! The Manifest read and the walk use two snapshots (the read transaction
//! wrapper has no rebind-with-manifest escape hatch); corpus audits run
//! quiescent, so the gap is unreachable in practice.

use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;
use ktann::api::{LogicalIndexId, PartitionKey};
use ktann::storage::backend::{Backend, ReadOps, ScanLimits};
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{IndexManifest, PartitionHeader, PersistentValue};
use ktann::storage::{LogicalRange, LogicalScanItem, ReadLogicalTxn};

use super::oracle::Model;

/// The page bound for one audit scan step; paging loops until exhaustion.
const AUDIT_SCAN: ScanLimits = ScanLimits {
    item_limit: 256,
    byte_limit: 1 << 20,
};

/// The membership read batch size.
const AUDIT_BATCH: usize = 256;

/// A partition within one tree; Partition Keys are only tree-unique.
type PartitionRef = (TreeKey, PartitionKey);

/// Counts describing one successful audit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuditReport {
    /// The modeled records verified against storage.
    pub records: usize,
    /// The trees enumerated from the directory.
    pub trees: usize,
    /// The reachable partitions across all trees.
    pub partitions: usize,
    /// The deepest tree level observed.
    pub max_level: u32,
}

/// One reachable partition found by the tree walk.
struct PartitionVisit {
    header: PartitionHeader,
    /// Leaf partitions only: the scanned entry Record IDs, in key order.
    leaf_entries: Vec<Bytes>,
}

/// Reads the persisted Index Manifest of one Logical Index.
async fn read_manifest<B: Backend>(
    backend: &B,
    index: LogicalIndexId,
) -> Result<IndexManifest, String> {
    let raw = backend
        .begin_read()
        .await
        .map_err(|error| format!("begin read: {error:?}"))?;
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    match txn
        .get(LogicalKey::Manifest(index))
        .await
        .map_err(|error| format!("read manifest: {error:?}"))?
    {
        Some(PersistentValue::IndexManifest(manifest)) => Ok(manifest),
        other => Err(format!(
            "manifest must decode as an Index Manifest, got {other:?}"
        )),
    }
}

/// Opens one bound read transaction for the audit walk.
async fn open_walk_txn<'b, 'm, B: Backend>(
    backend: &'b B,
    manifest: &'m IndexManifest,
) -> Result<ReadLogicalTxn<'m, B::ReadTxn<'b>>, String> {
    let raw = backend
        .begin_read()
        .await
        .map_err(|error| format!("begin read: {error:?}"))?;
    ReadLogicalTxn::for_index(raw, manifest).map_err(|error| format!("bind manifest: {error:?}"))
}

/// Scans one logical range to exhaustion in bounded pages.
async fn scan_all<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    range: &LogicalRange,
) -> Result<Vec<LogicalScanItem>, String> {
    let mut items = Vec::new();
    let mut cursor = None;
    loop {
        let page = txn
            .scan(range, cursor.as_ref(), AUDIT_SCAN)
            .await
            .map_err(|error| format!("scan: {error:?}"))?;
        let (page_items, next) = page.into_parts();
        items.extend(page_items);
        cursor = next;
        if cursor.is_none() {
            break;
        }
    }
    Ok(items)
}

/// Enumerates the Tree Manifest directory of one Logical Index.
async fn enumerate_trees<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
) -> Result<Vec<(TreeKey, PartitionKey)>, String> {
    let mut trees = Vec::new();
    for item in scan_all(txn, &LogicalRange::tree_manifests(manifest)).await? {
        match (item.key(), item.value()) {
            (LogicalKey::TreeManifest { tree_key, .. }, PersistentValue::TreeManifest(tree)) => {
                trees.push((tree_key.clone(), tree.root()));
            }
            (key, value) => {
                return Err(format!(
                    "directory holds {key:?} with value kind {:?}",
                    value.kind()
                ));
            }
        }
    }
    Ok(trees)
}

/// Reads one partition Header, requiring agreement with its State.
async fn read_partition_header<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<PartitionHeader, String> {
    let header = match txn
        .get(LogicalKey::Header {
            index: manifest.logical_index_id(),
            tree_key: tree_key.clone(),
            partition,
        })
        .await
        .map_err(|error| format!("read header: {error:?}"))?
    {
        Some(PersistentValue::PartitionHeader(header)) => header,
        other => {
            return Err(format!(
                "partition {} header must exist, got {other:?}",
                partition.get()
            ));
        }
    };
    match txn
        .get(LogicalKey::State {
            index: manifest.logical_index_id(),
            tree_key: tree_key.clone(),
            partition,
        })
        .await
        .map_err(|error| format!("read state: {error:?}"))?
    {
        Some(PersistentValue::PartitionState(transition)) => {
            if transition.state() != header.state() {
                return Err(format!(
                    "partition {} header/state disagree: {:?} vs {:?}",
                    partition.get(),
                    header.state(),
                    transition.state()
                ));
            }
        }
        other => {
            return Err(format!(
                "partition {} state must exist, got {other:?}",
                partition.get()
            ));
        }
    }
    Ok(header)
}

/// Scans one leaf partition's complete entry set, in key order.
async fn scan_leaf_ids<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Vec<Bytes>, String> {
    let range = LogicalRange::leaf_entries(manifest, tree_key, partition)
        .map_err(|error| format!("leaf range: {error:?}"))?;
    let mut ids = Vec::new();
    for item in scan_all(txn, &range).await? {
        match item.value() {
            PersistentValue::LeafEntry(entry) => ids.push(entry.record_id().clone()),
            other => {
                return Err(format!(
                    "partition {} holds a non-leaf entry of kind {:?}",
                    partition.get(),
                    other.kind()
                ));
            }
        }
    }
    Ok(ids)
}

/// Scans one internal partition's complete Child Entry set, in key order.
async fn scan_child_partitions<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Vec<PartitionKey>, String> {
    let range = LogicalRange::child_entries(manifest, tree_key, partition)
        .map_err(|error| format!("child range: {error:?}"))?;
    let mut children = Vec::new();
    for item in scan_all(txn, &range).await? {
        match item.value() {
            PersistentValue::ChildEntry(entry) => children.push(entry.child()),
            other => {
                return Err(format!(
                    "partition {} holds a non-child entry of kind {:?}",
                    partition.get(),
                    other.kind()
                ));
            }
        }
    }
    Ok(children)
}

/// The structural outcome of walking one tree.
struct TreeWalk {
    /// Every reachable partition, keyed by its tree-qualified reference.
    visits: BTreeMap<PartitionRef, PartitionVisit>,
    /// The deepest level observed.
    max_level: u32,
}

/// Walks one tree from its root, validating structure, and records every
/// reachable partition.
async fn walk_tree<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    root: PartitionKey,
) -> Result<TreeWalk, String> {
    let mut walk = TreeWalk {
        visits: BTreeMap::new(),
        max_level: 0,
    };
    let mut incoming: BTreeMap<PartitionRef, PartitionRef> = BTreeMap::new();
    let mut stack = vec![root];
    while let Some(partition) = stack.pop() {
        let reference = (tree_key.clone(), partition);
        if walk.visits.contains_key(&reference) {
            return Err(format!("partition {} reached twice", partition.get()));
        }
        let header = read_partition_header(txn, manifest, tree_key, partition).await?;
        walk.max_level = walk.max_level.max(header.level());

        let leaf_entries = if header.level() == 1 {
            let ids = scan_leaf_ids(txn, manifest, tree_key, partition).await?;
            if ids.len() != header.entry_count() as usize {
                return Err(format!(
                    "partition {} exact count {} != {} scanned leaf entries",
                    partition.get(),
                    header.entry_count(),
                    ids.len()
                ));
            }
            ids
        } else {
            let children = scan_child_partitions(txn, manifest, tree_key, partition).await?;
            if children.len() != header.entry_count() as usize {
                return Err(format!(
                    "partition {} exact count {} != {} scanned child entries",
                    partition.get(),
                    header.entry_count(),
                    children.len()
                ));
            }
            for child in children {
                let child_ref = (tree_key.clone(), child);
                if incoming
                    .insert(child_ref.clone(), reference.clone())
                    .is_some()
                {
                    return Err(format!("partition {} has two incoming edges", child.get()));
                }
                stack.push(child);
            }
            Vec::new()
        };
        walk.visits.insert(
            reference,
            PartitionVisit {
                header,
                leaf_entries,
            },
        );
    }
    Ok(walk)
}

/// Runs the full audit of one Logical Index against the caller's model.
pub async fn run<B: Backend>(
    backend: &B,
    index: LogicalIndexId,
    model: &Model,
) -> Result<AuditReport, String> {
    let manifest = read_manifest(backend, index).await?;
    let mut txn = open_walk_txn(backend, &manifest).await?;
    let trees = enumerate_trees(&mut txn, &manifest).await?;
    let mut visits = BTreeMap::new();
    let mut max_level = 0;
    for (tree_key, root) in &trees {
        let walk = walk_tree(&mut txn, &manifest, tree_key, *root).await?;
        max_level = max_level.max(walk.max_level);
        visits.extend(walk.visits);
    }

    // Exact membership, batched: each modeled record's Location names a
    // reachable leaf holding its Leaf Entry, and the record body round-trips.
    let mut located: BTreeMap<PartitionRef, BTreeSet<Bytes>> = BTreeMap::new();
    let ids: Vec<&Bytes> = model.keys().collect();
    for chunk in ids.chunks(AUDIT_BATCH) {
        let locations = txn
            .batch_get(
                chunk
                    .iter()
                    .map(|id| LogicalKey::Location {
                        index,
                        id: (*id).clone(),
                    })
                    .collect(),
            )
            .await
            .map_err(|error| format!("read locations: {error:?}"))?;
        let mut entry_keys = Vec::with_capacity(chunk.len());
        let mut record_keys = Vec::with_capacity(chunk.len());
        let mut leaves = Vec::with_capacity(chunk.len());
        for (id, location) in chunk.iter().zip(locations) {
            let location = match location {
                Some(PersistentValue::RecordLocation(location)) => location,
                other => {
                    return Err(format!(
                        "record must have one Record Location, got {other:?}"
                    ));
                }
            };
            let leaf_ref = (location.tree_key().clone(), location.leaf());
            let visit = visits.get(&leaf_ref).ok_or_else(|| {
                format!(
                    "location names unreachable partition {}",
                    location.leaf().get()
                )
            })?;
            if visit.header.level() != 1 {
                return Err(format!(
                    "location names non-leaf partition {}",
                    location.leaf().get()
                ));
            }
            record_keys.push(LogicalKey::Record {
                index,
                id: (*id).clone(),
            });
            entry_keys.push(LogicalKey::LeafEntry {
                index,
                tree_key: location.tree_key().clone(),
                partition: location.leaf(),
                id: (*id).clone(),
            });
            leaves.push(leaf_ref);
        }
        record_keys.append(&mut entry_keys);
        let authorities = txn
            .batch_get(record_keys)
            .await
            .map_err(|error| format!("read records and entries: {error:?}"))?;
        let (records, entries) = authorities.split_at(chunk.len());
        for position in 0..chunk.len() {
            let id: &Bytes = chunk[position];
            match &records[position] {
                Some(PersistentValue::VectorRecord(stored)) => {
                    let modeled = &model[id];
                    if stored.vector() != &*modeled.vector || stored.fields() != &*modeled.fields {
                        return Err("stored record body disagrees with the model".to_string());
                    }
                }
                other => return Err(format!("record must exist, got {other:?}")),
            }
            match &entries[position] {
                Some(PersistentValue::LeafEntry(entry)) if entry.record_id() == id => {}
                other => {
                    return Err(format!(
                        "leaf entry for the location must exist, got {other:?}"
                    ));
                }
            }
            located
                .entry(leaves[position].clone())
                .or_default()
                .insert(id.clone());
        }
    }

    // The reverse direction: the scanned leaf entry sets equal the located
    // sets, so no entry lacks or duplicates a modeled record.
    for (reference, visit) in &visits {
        if visit.header.level() != 1 {
            continue;
        }
        let scanned: BTreeSet<&Bytes> = visit.leaf_entries.iter().collect();
        let expected: BTreeSet<&Bytes> = located
            .get(reference)
            .map(|ids| ids.iter().collect())
            .unwrap_or_default();
        if scanned != expected {
            return Err(format!(
                "partition {} leaf entries disagree with record locations",
                reference.1.get()
            ));
        }
    }

    Ok(AuditReport {
        records: model.len(),
        trees: trees.len(),
        partitions: visits.len(),
        max_level,
    })
}

/// Renders the reachable topology of every tree, deterministically.
///
/// One line per tree, then one line per partition indented by depth:
/// `pk=N level=L state=State count=C`. With `entries`, each leaf's Record IDs
/// follow, one per line; corpus authors use that flag only on small fixtures.
pub async fn render_tree<B: Backend>(
    backend: &B,
    index: LogicalIndexId,
    entries: bool,
) -> Result<String, String> {
    let manifest = read_manifest(backend, index).await?;
    let mut txn = open_walk_txn(backend, &manifest).await?;
    let mut out = String::new();
    let trees = enumerate_trees(&mut txn, &manifest).await?;
    for (ordinal, (tree_key, root)) in trees.iter().enumerate() {
        out.push_str(&format!("tree {ordinal}: root pk={}\n", root.get()));
        render_partition(&mut txn, &manifest, tree_key, *root, 1, entries, &mut out).await?;
    }
    Ok(out)
}

/// Renders one partition subtree, recursively.
async fn render_partition<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    depth: usize,
    entries: bool,
    out: &mut String,
) -> Result<(), String> {
    let header = read_partition_header(txn, manifest, tree_key, partition).await?;
    let indent = "  ".repeat(depth);
    out.push_str(&format!(
        "{indent}pk={} level={} state={:?} count={}\n",
        partition.get(),
        header.level(),
        header.state(),
        header.entry_count()
    ));
    if header.level() == 1 {
        if entries {
            for id in scan_leaf_ids(txn, manifest, tree_key, partition).await? {
                out.push_str(&format!("{indent}  {}\n", super::datadriven::show_id(&id)));
            }
        }
    } else {
        for child in scan_child_partitions(txn, manifest, tree_key, partition).await? {
            Box::pin(render_partition(
                txn,
                manifest,
                tree_key,
                child,
                depth + 1,
                entries,
                out,
            ))
            .await?;
        }
    }
    Ok(())
}
