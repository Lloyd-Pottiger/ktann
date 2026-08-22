//! The two-phase bounded drain batch machinery shared by the split and merge
//! state machines.
//!
//! Draining stores no durable cursor: a short read snapshot fixes the
//! source's current smallest entries — at most [`DRAIN_BATCH_LEAF`] Leaf
//! Entries or [`DRAIN_BATCH_INTERNAL`] Child Entries by source level — and a
//! following write transaction revalidates and moves exactly those
//! candidates. Successful movement deletes that prefix, so every batch starts
//! at the current smallest entry, and entries a competing worker already
//! moved are simply absent from the next snapshot.

use bytes::Bytes;

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::storage::backend::{ReadOps, ScanLimits};
use crate::storage::keys::TreeKey;
use crate::storage::values::{IndexManifest, PartitionHeader, PersistentValue};
use crate::storage::{LogicalRange, ReadLogicalTxn};

/// The number of Leaf Entries one drain batch moves.
///
/// Each leaf move writes the target entry, both Headers, and the Record
/// Location, and may rewrite the target Synopsis (at most 64 KiB), so eight
/// moves stay within the most conservative adapter admission budget
/// (1,000 mutations / 1 MiB) even when every Synopsis rewrite is maximal.
pub const DRAIN_BATCH_LEAF: usize = 8;

/// The number of Child Entries one drain batch moves.
///
/// An internal move writes only the entry and both Headers, so 128 moves stay
/// far below every adapter admission budget.
pub const DRAIN_BATCH_INTERNAL: usize = 128;

/// The bound on one drain candidate scan page.
const DRAIN_SCAN_LIMITS: ScanLimits = ScanLimits {
    item_limit: DRAIN_BATCH_INTERNAL,
    byte_limit: 1_048_576,
};

/// One drain batch fixed by the read snapshot: leaf Record IDs or internal
/// child Partition Keys.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DrainBatch {
    Leaf(Vec<Bytes>),
    Child(Vec<PartitionKey>),
}

impl DrainBatch {
    /// Whether the batch holds no candidates.
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Leaf(ids) => ids.is_empty(),
            Self::Child(children) => children.is_empty(),
        }
    }
}

/// Scans the source's current smallest drain candidates from the snapshot.
pub(crate) async fn read_drain_batch<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    level: u32,
) -> Result<DrainBatch> {
    if level == 1 {
        let range = LogicalRange::leaf_entries(manifest, tree_key, source)?;
        let limits = ScanLimits {
            item_limit: DRAIN_BATCH_LEAF,
            ..DRAIN_SCAN_LIMITS
        };
        let page = txn.scan(&range, None, limits).await?;
        let mut ids = Vec::new();
        for item in page.items() {
            let PersistentValue::LeafEntry(entry) = item.value() else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            ids.push(entry.record_id().clone());
        }
        Ok(DrainBatch::Leaf(ids))
    } else {
        let range = LogicalRange::child_entries(manifest, tree_key, source)?;
        let page = txn.scan(&range, None, DRAIN_SCAN_LIMITS).await?;
        let mut children = Vec::new();
        for item in page.items() {
            let PersistentValue::ChildEntry(entry) = item.value() else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            children.push(entry.child());
        }
        Ok(DrainBatch::Child(children))
    }
}

/// Fixes one drain batch from the read snapshot: the source's current
/// smallest entries, or `None` when the exact count is already zero.
///
/// A missing Header alongside a present State, and an exact count that
/// disagrees with the scanned entry set, are Corruption.
pub(crate) async fn next_drain_batch<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    source_header: Option<PartitionHeader>,
) -> Result<Option<DrainBatch>> {
    let header = source_header.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    if header.entry_count() == 0 {
        return Ok(None);
    }
    let batch = read_drain_batch(txn, manifest, tree_key, source, header.level()).await?;
    if batch.is_empty() {
        // The exact count and the entry set disagree within one snapshot.
        return Err(Error::new(ErrorKind::Corruption));
    }
    Ok(Some(batch))
}
