//! Atomic Tree Manifest creation and Partition Key allocation.
//!
//! Creation installs one Tree Manifest and its searchable empty root in the
//! same transaction. Update-protected reservations advance the per-tree
//! high-water mark; allocated Partition Keys are never shared or reused.

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::storage::backend::{InsertOutcome, ReadOps, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    IndexManifest, PartitionHeader, PartitionState, PartitionSynopsis, PartitionTransition,
    PersistentValue, TreeManifest, expect_tree_manifest,
};
use crate::storage::{ReadLogicalTxn, WriteLogicalTxn};

/// The default number of Partition Keys reserved in one transaction.
pub const DEFAULT_PARTITION_KEY_RESERVATION: u32 = 1_024;

/// The outcome of lazily creating one Tree Key's tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TreeCreation {
    /// This transaction installed the Tree Manifest and initial leaf root.
    Created,
    /// Another committed transaction already created the tree.
    AlreadyExists,
}

/// One transactionally reserved range of never-reused Partition Keys.
///
/// The reservation is the inclusive range `[next, last]`. A later reservation
/// starts after `last`, so ranges from concurrent or sequential reservations
/// are disjoint and monotonic; unused keys remain valid gaps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartitionKeyReservation {
    next: PartitionKey,
    last: PartitionKey,
}

impl PartitionKeyReservation {
    /// Returns the first reserved Partition Key.
    #[must_use]
    pub const fn next(self) -> PartitionKey {
        self.next
    }

    /// Returns the last reserved Partition Key, inclusive.
    #[must_use]
    pub const fn last(self) -> PartitionKey {
        self.last
    }

    /// Returns the number of reserved Partition Keys.
    #[must_use]
    pub const fn count(self) -> u64 {
        self.last.get() - self.next.get() + 1
    }
}

/// Reads one Tree Manifest from the transaction snapshot.
pub async fn read_tree_manifest<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    tree_key: &TreeKey,
) -> Result<Option<TreeManifest>> {
    let key = tree_manifest_key_for(txn.require_manifest()?, tree_key);
    expect_tree_manifest(txn.get(key).await?)
}

/// Reads one Tree Manifest and establishes a conflict on its key.
pub async fn read_tree_manifest_for_update<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
) -> Result<Option<TreeManifest>> {
    let key = tree_manifest_key_for(txn.require_manifest()?, tree_key);
    expect_tree_manifest(txn.get_for_update(key).await?)
}

/// Lazily creates the tree for one Tree Key.
///
/// The commit installs the Tree Manifest with stable root Partition Key 1 and
/// high-water mark 1, the root Header (level 1, zero entries, epoch zero,
/// Ready), the canonical empty Synopsis, and the Ready State with the supplied
/// start time. When another transaction already created the tree this call
/// returns [`TreeCreation::AlreadyExists`] without changing any committed
/// value.
pub async fn create_tree<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    started_at_unix_millis: u64,
) -> Result<TreeCreation> {
    let manifest = txn.require_manifest()?;
    let index = manifest.logical_index_id();
    let root = partition_key(1)?;
    let outcome = txn
        .insert(
            LogicalKey::TreeManifest {
                index,
                tree_key: tree_key.clone(),
            },
            PersistentValue::TreeManifest(TreeManifest::new(root, root)?),
        )
        .await?;
    if outcome != InsertOutcome::Inserted {
        return Ok(TreeCreation::AlreadyExists);
    }
    txn.put(
        LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: root,
        },
        PersistentValue::PartitionHeader(PartitionHeader::new(1, 0, 0, PartitionState::Ready)?),
    )
    .await?;
    txn.put(
        LogicalKey::Synopsis {
            index,
            tree_key: tree_key.clone(),
            partition: root,
        },
        PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(manifest)),
    )
    .await?;
    txn.put(
        LogicalKey::State {
            index,
            tree_key: tree_key.clone(),
            partition: root,
        },
        PersistentValue::PartitionState(PartitionTransition::Ready {
            started_at_unix_millis,
        }),
    )
    .await?;
    Ok(TreeCreation::Created)
}

/// Reserves the next `count` Partition Keys for one tree.
///
/// The update-protected high-water mark read serializes reservations: the
/// returned range starts exactly after the previously persisted high-water
/// mark, so ranges are stable, monotonic, disjoint, and never reused. Near
/// exhaustion the final nonzero suffix may be shorter than `count`; once no
/// legal key remains the reservation fails with [`ErrorKind::IdExhausted`].
/// The Tree Manifest for `tree_key` must already exist.
pub async fn reserve_partition_keys<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    count: u32,
) -> Result<PartitionKeyReservation> {
    if count == 0 {
        return Err(Error::invalid_argument());
    }
    let key = tree_manifest_key_for(txn.require_manifest()?, tree_key);
    let manifest = expect_tree_manifest(txn.get_for_update(key.clone()).await?)?
        .ok_or_else(Error::invalid_argument)?;
    let high_water = manifest.partition_key_high_water().get();
    if high_water == u64::MAX {
        return Err(Error::new(ErrorKind::IdExhausted));
    }
    let next = high_water + 1;
    let last = high_water.saturating_add(u64::from(count));
    txn.put(
        key,
        PersistentValue::TreeManifest(TreeManifest::new(manifest.root(), partition_key(last)?)?),
    )
    .await?;
    Ok(PartitionKeyReservation {
        next: partition_key(next)?,
        last: partition_key(last)?,
    })
}

/// Builds the typed Tree Manifest key for one transaction binding.
fn tree_manifest_key_for(manifest: &IndexManifest, tree_key: &TreeKey) -> LogicalKey {
    LogicalKey::TreeManifest {
        index: manifest.logical_index_id(),
        tree_key: tree_key.clone(),
    }
}

/// Constructs a Partition Key from checked caller-validated arithmetic.
///
/// The reservation arithmetic above guarantees `1..=u64::MAX`; a violation is
/// an internal error, reported fail-closed.
fn partition_key(value: u64) -> Result<PartitionKey> {
    PartitionKey::new(value).map_err(|_| Error::new(ErrorKind::Corruption))
}
