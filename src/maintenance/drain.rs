//! The two-phase bounded drain batch machinery shared by the split and merge
//! state machines.
//!
//! Draining stores no durable cursor: a short read snapshot fixes the
//! source's current smallest entries — as many Leaf Entries as the current
//! schema, Backend budget, and configured split threshold safely admit, or at
//! most [`DRAIN_BATCH_INTERNAL`] Child Entries by source level — and a
//! following write transaction revalidates and moves exactly those
//! candidates. Successful movement deletes that prefix, so every batch starts
//! at the current smallest entry, and entries a competing worker already
//! moved are simply absent from the next snapshot.

use bytes::Bytes;

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{AdmissionBudget, ReadOps, ScanLimits, WriteTxn};
use crate::storage::keys::TreeKey;
use crate::storage::topology::{self, LeafDrainEntry, Movement};
use crate::storage::values::{ChildEntry, IndexManifest, PartitionHeader, PersistentValue};
use crate::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

/// The number of Child Entries one drain batch moves.
///
/// An internal move writes only the entry and both Headers, so 128 moves stay
/// far below every adapter admission budget.
pub const DRAIN_BATCH_INTERNAL: usize = 128;

/// Maximum encoded bytes in one drain candidate scan page.
const DRAIN_SCAN_BYTES: usize = 1_048_576;

/// Applies the threshold and source-size bounds to a budget-safe leaf batch.
///
/// The configured split threshold caps the batch at one ordinary partition's
/// worth of entries, limiting conflict rollback and failure blast radius
/// relative to the index's ordinary partition size; the Backend budget remains
/// the hard upper bound.
fn leaf_drain_batch_limit(
    max_leaf_partition_entries: u32,
    source_entries: u32,
    budget_limit: usize,
) -> usize {
    (max_leaf_partition_entries as usize)
        .min(budget_limit)
        .min(source_entries as usize)
}

/// One drain batch fixed by the read snapshot: leaf Record IDs or internal
/// child Partition Keys.
pub(super) enum DrainBatch {
    Leaf(Vec<(PartitionKey, Bytes)>),
    Child(Vec<PartitionKey>),
}

/// Routes the Leaf Entries still present in the write attempt, in snapshot order.
pub(super) async fn route_leaves<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    candidates: &[(PartitionKey, Bytes)],
    kernel: &VectorKernel,
    mut target: impl FnMut(&[f32]) -> Result<PartitionKey>,
) -> Result<Vec<(LeafDrainEntry, PartitionKey)>> {
    let entries = topology::read_leaf_drain_candidates(txn, tree_key, candidates).await?;
    let mut moves = Vec::new();
    for entry in entries.into_iter().flatten() {
        let vector = kernel
            .preprocess(entry.record().vector())
            .map_err(|_| Error::new(ErrorKind::Corruption))?;
        moves.push((entry, target(&vector)?));
    }
    Ok(moves)
}

/// Routes the Child Entries still present using their persisted centroids.
pub(super) async fn route_children<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    children: &[PartitionKey],
    mut target: impl FnMut(&[f32]) -> Result<PartitionKey>,
) -> Result<Vec<(ChildEntry, PartitionKey)>> {
    let entries = topology::read_child_drain_candidates(txn, tree_key, source, children).await?;
    let mut moves = Vec::new();
    for entry in entries.into_iter().flatten() {
        let target = target(entry.centroid())?;
        moves.push((entry, target));
    }
    Ok(moves)
}

/// Fixes one drain batch from the read snapshot: the source's current
/// smallest entries, or `None` when the exact count is already zero.
///
/// The caller supplies the Header from a validated authority pair. An exact
/// count that disagrees with the scanned entry set is Corruption.
pub(super) async fn next_drain_batch<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    header: PartitionHeader,
    movement: Movement,
    budget: AdmissionBudget,
) -> Result<Option<DrainBatch>> {
    if header.entry_count() == 0 {
        return Ok(None);
    }
    let leaf = header.level() == 1;
    let (range, item_limit) = if leaf {
        let budget_limit =
            topology::leaf_relocation_batch_limit(manifest, tree_key, movement, budget)?;
        let limit = leaf_drain_batch_limit(
            manifest.config().max_leaf_partition_entries(),
            header.entry_count(),
            budget_limit,
        );
        (
            LogicalRange::leaf_entries(manifest, tree_key, source)?,
            limit,
        )
    } else {
        (
            LogicalRange::child_entries(manifest, tree_key, source)?,
            DRAIN_BATCH_INTERNAL,
        )
    };
    let limits = ScanLimits {
        item_limit,
        byte_limit: DRAIN_SCAN_BYTES,
    };
    let page = txn.scan(&range, None, limits).await?;
    if page.items().is_empty() {
        // The exact count and the entry set disagree within one snapshot.
        return Err(Error::new(ErrorKind::Corruption));
    }
    let batch = if leaf {
        let mut entries = Vec::with_capacity(page.items().len());
        for item in page.items() {
            let PersistentValue::LeafEntry(entry) = item.value() else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            entries.push((source, entry.record_id().clone()));
        }
        DrainBatch::Leaf(entries)
    } else {
        let mut children = Vec::with_capacity(page.items().len());
        for item in page.items() {
            let PersistentValue::ChildEntry(entry) = item.value() else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            children.push(entry.child());
        }
        DrainBatch::Child(children)
    };
    Ok(Some(batch))
}

#[cfg(test)]
mod tests {
    use super::leaf_drain_batch_limit;

    #[test]
    fn leaf_batch_combines_budget_threshold_and_source_bounds() {
        // The smallest of the three bounds wins: threshold, budget, source.
        assert_eq!(leaf_drain_batch_limit(32, 100, 1_000), 32);
        assert_eq!(leaf_drain_batch_limit(128, 100, 1_000), 100);
        assert_eq!(leaf_drain_batch_limit(128, 100, 17), 17);
        assert_eq!(leaf_drain_batch_limit(128, 9, 1_000), 9);
    }
}
