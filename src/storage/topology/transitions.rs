//! Atomic split/merge state transitions and centroid projection updates.
//! Each operation leaves committed topology searchable and rolls back on error.

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::storage::backend::{Capabilities, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::membership::{added_entry, expect_inserted, removed_entry, touched_entry};
use crate::storage::tree_manifest::reserve_partition_keys;
use crate::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionSynopsis, PartitionTransition, PersistentValue, expect_centroid, expect_child_entry,
    expect_header, expect_state,
};
use crate::storage::{LogicalRange, WriteLogicalTxn};

use super::authority::{authority_keys, authority_pair, expect_agreement};
use super::discovery::{lock_incoming_edge, same_level_candidates};
use super::{corrupt, root_partition};

/// The number of split targets one split reserves and exposes; binary fanout
/// is a fixed persistent format protocol choice.
const SPLIT_TARGETS: u32 = 2;

/// The outcome of [`begin_split`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SplitStart {
    /// This transaction transitioned the source to `Splitting`.
    Started {
        /// The reserved left target.
        left: PartitionKey,
        /// The reserved right target.
        right: PartitionKey,
    },
    /// The source was already `Splitting`; the persisted targets are
    /// returned. This is the recovery path after an unknown commit outcome.
    AlreadySplitting {
        /// The persisted left target.
        left: PartitionKey,
        /// The persisted right target.
        right: PartitionKey,
    },
    /// The source is not a `Ready` partition above the split threshold.
    NotEligible,
}

/// The outcome of [`expose_split_targets`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SplitExposure {
    /// This transaction unique-created both targets and advanced the source
    /// to `DrainingSplit`.
    Exposed,
    /// The source was already `DrainingSplit` naming these targets; nothing
    /// was written. This is the recovery path after an unknown commit
    /// outcome.
    AlreadyExposed,
    /// The source no longer `Splitting`-names the targets (the split
    /// completed); the step wrote nothing and must be abandoned.
    SourceAdvanced,
    /// The source's current parent cannot accept the new Child Entries (it
    /// is draining, merging, or lacks graph headroom); the step wrote nothing
    /// and must be abandoned so a later attempt can rediscover the source's parent.
    ParentNotAccepting,
}

/// The outcome of [`finalize_split`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SplitCompletion {
    /// The split completed: both targets are `Ready` and the source topology
    /// is switched.
    Completed {
        /// The updated parent when this call completed a non-root split.
        /// `None` denotes a root split or an idempotent re-drive.
        parent: Option<PartitionKey>,
    },
    /// The source is `DrainingSplit` but its exact entry count is not zero.
    NotDrained,
    /// The source is not `DrainingSplit` (anymore). A split that was draining
    /// reached this state only through a completed finalize, so re-driving a
    /// finalize after an unknown commit outcome observes this outcome.
    NotDraining,
}

/// Starts the split of one `Ready` partition above the split threshold.
///
/// The update-protected source Header and State reads revalidate eligibility:
/// the source must be `Ready` with an exact entry count above the configured
/// maximum. One transaction reserves the two never-reused target Partition
/// Keys and writes `Splitting { left, right }` on the source State and Header
/// discriminator. A near-exhaustion reservation shorter than two keys fails
/// with [`ErrorKind::IdExhausted`] and must not be committed.
///
/// Observing `Splitting { left, right }` returns the persisted targets, so
/// re-driving after an unknown commit outcome never reserves a second pair.
/// A source with no authority values — a completed split removed them, or it
/// never existed — reports [`SplitStart::NotEligible`]; a half-present pair
/// is Corruption.
pub async fn begin_split<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
) -> Result<SplitStart> {
    let manifest = txn.require_manifest()?;
    let index = manifest.logical_index_id();
    let (header_key, state_key) = authority_keys(index, tree_key, source);
    let Some((header, state)) = authority_pair(txn, header_key.clone(), state_key.clone()).await?
    else {
        // Both authority values are gone: a completed split already removed
        // the source (or it never existed), so there is nothing to start.
        return Ok(SplitStart::NotEligible);
    };

    match state {
        PartitionTransition::Splitting { left, right, .. } => {
            return Ok(SplitStart::AlreadySplitting { left, right });
        }
        PartitionTransition::Ready { .. } => {}
        // ReceivingSplit cannot split until its own source drains; a source in
        // any other state has nothing to (re)start.
        _ => return Ok(SplitStart::NotEligible),
    }
    if header.entry_count() <= manifest.config().max_partition_entries(header.level()) {
        return Ok(SplitStart::NotEligible);
    }

    let reservation = reserve_partition_keys(txn, tree_key, SPLIT_TARGETS).await?;
    if reservation.count() != u64::from(SPLIT_TARGETS) {
        // A short final suffix cannot host both targets; the rolled-back
        // reservation leaves the high-water mark untouched.
        return Err(Error::new(ErrorKind::IdExhausted));
    }
    let left = reservation.next();
    let right = reservation.last();

    txn.put(
        state_key,
        PersistentValue::PartitionState(PartitionTransition::Splitting {
            left,
            right,
            started_at_unix_millis,
        }),
    )
    .await?;
    txn.put(
        header_key,
        PersistentValue::PartitionHeader(with_state(header, PartitionState::Splitting)?),
    )
    .await?;
    Ok(SplitStart::Started { left, right })
}

/// Unique-creates both split targets, installs their incoming Child Entries,
/// and advances the source from `Splitting` to `DrainingSplit`, atomically.
///
/// Exposure is one transaction (ADR 0014): both targets are created as
/// `ReceivingSplit { source }` with their persisted centroids, a non-root
/// source's current parent gains both Child Entries, and the source flips to
/// `DrainingSplit`, so no committed state holds a partially exposed split. The
/// update-protected source authority pair aborts the commit against a
/// concurrent transition or foreground write to the source; the caller's
/// whole-step retry absorbs that, and exhaustion leaves the searchable
/// `Splitting` state for a later access to rediscover. The first successful
/// exposure fixes both target centroids forever: a competing attempt conflicts
/// on the source State before commit and retries onto the persisted
/// `DrainingSplit` state, so no worker ever overwrites a published centroid.
///
/// For a non-root source the same transaction rediscovers the source's unique
/// incoming Child Entry by an exact bounded root-down scan (ADR 0007),
/// update-protects the traversed parent Header and the source edge, verifies
/// that the parent still contains the source and accepts new children, and
/// inserts both target edges. Root targets have no parent: they occupy the
/// exclusive target slots named by Partition Key 1's persisted state until
/// root completion exposes them.
///
/// The transition deliberately ignores the source's entries, count, and cache
/// epoch: training and publication never restart or revalidate because of
/// concurrent foreground writes (ADR 0014). Under the atomic protocol a
/// `DrainingSplit` source always has both targets complete, so the
/// [`SplitExposure::AlreadyExposed`] re-drive only checks that the persisted
/// state names the caller's targets and writes nothing; a `Splitting` source
/// whose targets already exist is a torn committed state the unique inserts
/// below fail closed on.
///
/// Only [`SplitExposure::Exposed`] carries writes; every other outcome leaves
/// the transaction without mutations.
pub async fn expose_split_targets<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    targets: [(PartitionKey, &PartitionCentroid); 2],
    started_at_unix_millis: u64,
) -> Result<SplitExposure> {
    expose_split_targets_with_parent(txn, tree_key, source, targets, started_at_unix_millis, None)
        .await
}

/// Uses a parent only after the caller validates its complete discovery proof.
pub(crate) async fn expose_split_targets_with_parent<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    targets: [(PartitionKey, &PartitionCentroid); 2],
    started_at_unix_millis: u64,
    parent: Option<PartitionKey>,
) -> Result<SplitExposure> {
    let manifest = txn.require_manifest()?;
    let index = manifest.logical_index_id();
    let (source_header_key, source_state_key) = authority_keys(index, tree_key, source);
    // The two targets in the order the persisted `Splitting` state names them.
    let [(left, left_centroid), (right, right_centroid)] = targets;

    // The source authority pair is update-protected so a concurrent
    // transition or foreground write to the source aborts the commit; its
    // absence means a completed split already removed it.
    let Some((source_header, state)) =
        authority_pair(txn, source_header_key.clone(), source_state_key.clone()).await?
    else {
        return Ok(SplitExposure::SourceAdvanced);
    };
    let level = source_header.level();
    let draining = matches!(state, PartitionTransition::DrainingSplit { .. });
    match state {
        PartitionTransition::Splitting {
            left: persisted_left,
            right: persisted_right,
            ..
        }
        | PartitionTransition::DrainingSplit {
            left: persisted_left,
            right: persisted_right,
            ..
        } => {
            if left != persisted_left || right != persisted_right {
                return Err(Error::invalid_argument());
            }
        }
        // Ready, ReceivingSplit, or Merging: the split completed or never
        // began; a stale worker must abandon, not recreate.
        _ => return Ok(SplitExposure::SourceAdvanced),
    }
    if draining {
        return Ok(SplitExposure::AlreadyExposed);
    }

    // For a non-root source, discover and validate the incoming topology
    // before writing anything. The parent Header's update-protected read both
    // validates the acceptance rules and feeds the exact-count update below.
    let parent = if source == root_partition() {
        None
    } else {
        let parent = lock_incoming_edge(txn, tree_key, source, level, parent).await?;
        let parent_header = expect_header(
            txn.get_for_update(LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: parent,
            })
            .await?,
        )?
        .ok_or_else(corrupt)?;
        if parent_header.level() != level.checked_add(1).ok_or_else(corrupt)? {
            return Err(corrupt());
        }
        match parent_header.state() {
            PartitionState::Ready | PartitionState::Splitting | PartitionState::ReceivingSplit => {}
            // A draining or merging parent is completing its own maintenance;
            // abandoning lets a later attempt rediscover the moved edge.
            _ => return Ok(SplitExposure::ParentNotAccepting),
        }
        if manifest.config().local_graph().is_some()
            && (parent_header.state() == PartitionState::ReceivingSplit
                || parent_header.entry_count() > manifest.config().max_internal_partition_entries())
        {
            // A receiving parent must finish draining its source before adding
            // new edges: spare receiver capacity does not imply spare capacity
            // in their combined graph rewrite. Other parents reserve two slots
            // for exposure and must split before consuming further headroom.
            return Ok(SplitExposure::ParentNotAccepting);
        }
        Some((parent, parent_header))
    };

    // Every key of each target partition is a unique insert: under atomic
    // exposure the partition cannot exist yet and Partition Keys are never
    // reused, so an existing key is a torn committed state.
    for (target, centroid) in [(left, left_centroid), (right, right_centroid)] {
        let (target_header_key, target_state_key) = authority_keys(index, tree_key, target);
        expect_inserted(
            txn.insert(
                target_header_key,
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(level, 0, 0, PartitionState::ReceivingSplit)
                        .map_err(|_| corrupt())?,
                ),
            )
            .await?,
        )?;
        expect_inserted(
            txn.insert(
                target_state_key,
                PersistentValue::PartitionState(PartitionTransition::ReceivingSplit {
                    source,
                    started_at_unix_millis,
                }),
            )
            .await?,
        )?;
        expect_inserted(
            txn.insert(
                LogicalKey::Centroid {
                    index,
                    tree_key: tree_key.clone(),
                    partition: target,
                },
                PersistentValue::PartitionCentroid(centroid.clone()),
            )
            .await?,
        )?;
        if level == 1 {
            expect_inserted(
                txn.insert(
                    LogicalKey::Synopsis {
                        index,
                        tree_key: tree_key.clone(),
                        partition: target,
                    },
                    PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(manifest)),
                )
                .await?,
            )?;
        }
        if let Some((parent, _)) = &parent {
            expect_inserted(
                txn.insert(
                    LogicalKey::ChildEntry {
                        index,
                        tree_key: tree_key.clone(),
                        partition: *parent,
                        child: target,
                    },
                    PersistentValue::ChildEntry(ChildEntry::from_centroid(
                        target,
                        centroid.clone(),
                    )),
                )
                .await?,
            )?;
        }
    }
    if let Some((parent, parent_header)) = parent {
        txn.put(
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: parent,
            },
            PersistentValue::PartitionHeader(added_entry(added_entry(parent_header)?)?),
        )
        .await?;
    }

    txn.put(
        source_state_key,
        PersistentValue::PartitionState(PartitionTransition::DrainingSplit {
            left,
            right,
            started_at_unix_millis,
        }),
    )
    .await?;
    txn.put(
        source_header_key,
        PersistentValue::PartitionHeader(with_state(source_header, PartitionState::DrainingSplit)?),
    )
    .await?;
    Ok(SplitExposure::Exposed)
}

/// Updates one nonroot centroid and its parent's routing projection atomically.
///
/// Both partitions must be Ready, with a matching existing child edge and
/// centroid. A changed projection advances only the parent's cache epoch;
/// membership counts, states, and leaf encodings remain unchanged.
/// Callers must roll back the transaction on error.
pub(crate) async fn update_centroid<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    partition: PartitionKey,
    parent: PartitionKey,
    centroid: PartitionCentroid,
) -> Result<()> {
    if partition == root_partition() || partition == parent {
        return Err(corrupt());
    }
    let index = txn.require_manifest()?.logical_index_id();
    let child_header_key = LogicalKey::Header {
        index,
        tree_key: tree_key.clone(),
        partition,
    };
    let parent_header_key = LogicalKey::Header {
        index,
        tree_key: tree_key.clone(),
        partition: parent,
    };
    let centroid_key = LogicalKey::Centroid {
        index,
        tree_key: tree_key.clone(),
        partition,
    };
    let edge_key = LogicalKey::ChildEntry {
        index,
        tree_key: tree_key.clone(),
        partition: parent,
        child: partition,
    };
    let mut values = txn
        .batch_get_for_update(vec![
            child_header_key,
            parent_header_key.clone(),
            centroid_key.clone(),
            edge_key.clone(),
        ])
        .await?
        .into_iter();
    let child_header = expect_header(
        values
            .next()
            .ok_or_else(|| Error::new(ErrorKind::Backend))?,
    )?
    .ok_or_else(corrupt)?;
    let parent_header = expect_header(
        values
            .next()
            .ok_or_else(|| Error::new(ErrorKind::Backend))?,
    )?
    .ok_or_else(corrupt)?;
    let old_centroid = expect_centroid(
        values
            .next()
            .ok_or_else(|| Error::new(ErrorKind::Backend))?,
    )?
    .ok_or_else(corrupt)?;
    let edge = expect_child_entry(
        values
            .next()
            .ok_or_else(|| Error::new(ErrorKind::Backend))?,
    )?
    .ok_or_else(corrupt)?;
    if child_header.state() != PartitionState::Ready
        || parent_header.state() != PartitionState::Ready
        || child_header.level().checked_add(1) != Some(parent_header.level())
        || edge.child() != partition
        || edge.model() != &old_centroid
    {
        return Err(corrupt());
    }
    if centroid == old_centroid {
        return Ok(());
    }
    let parent_header = touched_entry(parent_header)?;
    txn.put(
        centroid_key,
        PersistentValue::PartitionCentroid(centroid.clone()),
    )
    .await?;
    txn.put(
        edge_key,
        PersistentValue::ChildEntry(ChildEntry::from_centroid(partition, centroid)),
    )
    .await?;
    txn.put(
        parent_header_key,
        PersistentValue::PartitionHeader(parent_header),
    )
    .await?;
    Ok(())
}

/// How [`finalize_split`] removes a drained non-root source's prefix.
///
/// After exact count zero the prefix holds only the four fixed metadata keys,
/// so both forms are bounded; only the exact-count-zero invariant makes the
/// point form complete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceRemoval {
    /// One transactional range clear of the whole partition prefix, on
    /// backends advertising `transactional_clear_range`.
    TransactionalClear,
    /// Bounded point deletes of the four fixed metadata keys.
    PointDeletes,
}

impl SourceRemoval {
    /// Selects the removal form one backend's declared capabilities support.
    pub fn for_capabilities(capabilities: Capabilities) -> Self {
        if capabilities.transactional_clear_range {
            Self::TransactionalClear
        } else {
            Self::PointDeletes
        }
    }
}

/// Completes one fully drained split in a single final transaction.
///
/// The exact update-protected source Header count is the sole proof of
/// emptiness; no entry scan is repeated (ADR 0014). The transaction
/// revalidates the `DrainingSplit { left, right }` state, promotes both
/// targets from `ReceivingSplit` to `Ready`, and switches the topology:
///
/// - **Non-root**: the source's unique incoming Child Entry — rediscovered by
///   an exact bounded root-down scan and update-protected — is removed with
///   the parent's exact count, and the source prefix is removed per
///   `source_removal`.
/// - **Root**: Partition Key 1 is converted in place to a `Ready` internal
///   root one level above the drained source containing the two target Child
///   Entries with their persisted centroids; the obsolete leaf Synopsis is
///   deleted and no range is cleared because the root survives.
pub async fn finalize_split<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
    source_removal: SourceRemoval,
) -> Result<SplitCompletion> {
    finalize_split_with_parent(
        txn,
        tree_key,
        source,
        started_at_unix_millis,
        source_removal,
        None,
    )
    .await
}

/// Uses a parent only after the caller validates its complete discovery proof.
pub(crate) async fn finalize_split_with_parent<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
    source_removal: SourceRemoval,
    parent: Option<PartitionKey>,
) -> Result<SplitCompletion> {
    let manifest = txn.require_manifest()?;
    let index = manifest.logical_index_id();
    let (header_key, state_key) = authority_keys(index, tree_key, source);
    let Some((header, state)) = authority_pair(txn, header_key.clone(), state_key.clone()).await?
    else {
        // Both authority values are gone: a previous committed finalize
        // completed this split.
        return Ok(SplitCompletion::Completed { parent: None });
    };
    let PartitionTransition::DrainingSplit { left, right, .. } = state else {
        return Ok(SplitCompletion::NotDraining);
    };
    if header.entry_count() != 0 {
        return Ok(SplitCompletion::NotDrained);
    }
    let level = header.level();

    // Promote both targets; a target in any other state contradicts the
    // committed DrainingSplit source and is Corruption. One update-protected
    // batch reads both targets' authority values.
    let target_keys = [left, right].map(|target| authority_keys(index, tree_key, target));
    let mut target_values = txn
        .batch_get_for_update(vec![
            target_keys[0].1.clone(),
            target_keys[0].0.clone(),
            target_keys[1].1.clone(),
            target_keys[1].0.clone(),
        ])
        .await?
        .into_iter();
    for (target_header_key, target_state_key) in target_keys {
        let (Some(state_value), Some(header_value)) = (target_values.next(), target_values.next())
        else {
            // The typed batch read returns exactly one value per input key.
            return Err(Error::new(ErrorKind::Backend));
        };
        let target_state = expect_state(state_value)?.ok_or_else(corrupt)?;
        match target_state {
            PartitionTransition::ReceivingSplit { source: s, .. } if s == source => {}
            _ => return Err(corrupt()),
        }
        let target_header = expect_header(header_value)?.ok_or_else(corrupt)?;
        expect_agreement(target_header, target_state)?;
        if target_header.level() != level {
            return Err(corrupt());
        }
        txn.put(
            target_state_key,
            PersistentValue::PartitionState(PartitionTransition::Ready {
                started_at_unix_millis,
            }),
        )
        .await?;
        txn.put(
            target_header_key,
            PersistentValue::PartitionHeader(with_state(target_header, PartitionState::Ready)?),
        )
        .await?;
    }

    if source == root_partition() {
        // The root converts in place: it gains one level and the two target
        // Child Entries carrying the persisted target centroids, read in one
        // batch.
        let mut centroids = txn
            .batch_get(
                [left, right]
                    .map(|target| LogicalKey::Centroid {
                        index,
                        tree_key: tree_key.clone(),
                        partition: target,
                    })
                    .into(),
            )
            .await?
            .into_iter();
        let mut promoted = header;
        for target in [left, right] {
            let Some(centroid_value) = centroids.next() else {
                // The typed batch read returns exactly one value per input key.
                return Err(Error::new(ErrorKind::Backend));
            };
            let centroid = expect_centroid(centroid_value)?.ok_or_else(corrupt)?;
            // The drained root holds no entries, so both Child Entries are
            // unique inserts.
            expect_inserted(
                txn.insert(
                    LogicalKey::ChildEntry {
                        index,
                        tree_key: tree_key.clone(),
                        partition: root_partition(),
                        child: target,
                    },
                    PersistentValue::ChildEntry(ChildEntry::from_centroid(
                        target,
                        centroid.clone(),
                    )),
                )
                .await?,
            )?;
            promoted = added_entry(promoted)?;
        }
        // The leaf root's Synopsis is obsolete one level up.
        txn.delete(LogicalKey::Synopsis {
            index,
            tree_key: tree_key.clone(),
            partition: root_partition(),
        })
        .await?;
        let promoted = PartitionHeader::new(
            promoted.level().checked_add(1).ok_or_else(corrupt)?,
            promoted.entry_count(),
            promoted.cache_epoch(),
            PartitionState::Ready,
        )
        .map_err(|_| corrupt())?;
        txn.put(header_key, PersistentValue::PartitionHeader(promoted))
            .await?;
        txn.put(
            state_key,
            PersistentValue::PartitionState(PartitionTransition::Ready {
                started_at_unix_millis,
            }),
        )
        .await?;
        return Ok(SplitCompletion::Completed { parent: None });
    }

    // Remove the source's unique incoming reference, rediscovered on this
    // transaction's snapshot and update-protected (ADR 0007).
    let parent = remove_incoming_edge(txn, tree_key, source, level, parent).await?;
    remove_source_prefix(
        txn,
        manifest,
        tree_key,
        source,
        header_key,
        state_key,
        source_removal,
    )
    .await?;
    Ok(SplitCompletion::Completed {
        parent: Some(parent),
    })
}

/// The outcome of [`begin_merge`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MergeStart {
    /// This transaction transitioned the source to `Merging`.
    Started,
    /// The source was already `Merging`; nothing was written. This is the
    /// recovery path after an unknown commit outcome.
    AlreadyMerging,
    /// The source is not an eligible non-root `Ready` partition below the
    /// merge threshold, or its authority values are already gone.
    NotEligible,
    /// The source is eligible but no legal same-level `Ready` target exists,
    /// so no state starts (ADR 0008).
    NoReadyTarget,
}

/// Starts the merge of one eligible `Ready` partition by marking it
/// `Merging`.
///
/// The update-protected source Header and State reads revalidate eligibility:
/// the source must be `Ready`, non-root, and below the configured minimum
/// entry count; an intermediate-state partition cannot start another
/// structural change, and roots never merge or collapse. The same transaction
/// locks the source's unique incoming Child Entry — rediscovered by the exact
/// bounded root-down scan and update-protected (ADR 0007) — so a concurrent
/// edge move aborts the commit instead of starting a merge against a stale
/// parent observation. Before the transition it validates that at least one
/// legal same-level `Ready` target exists; when none exists no state starts.
/// The target validation reads the current snapshot without update protection
/// on purpose: merging persists no target, and a target that disappears later
/// stalls the drain, never corrupts it.
///
/// Observing `Merging` reports [`MergeStart::AlreadyMerging`], so re-driving
/// after an unknown commit outcome never restarts anything. A source with no
/// authority values — a completed merge removed them, or it never existed —
/// reports [`MergeStart::NotEligible`]; a half-present pair is Corruption.
pub async fn begin_merge<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
) -> Result<MergeStart> {
    let manifest = txn.require_manifest()?;
    let index = manifest.logical_index_id();
    let (header_key, state_key) = authority_keys(index, tree_key, source);
    let Some((header, state)) = authority_pair(txn, header_key.clone(), state_key.clone()).await?
    else {
        // Both authority values are gone: a completed merge already removed
        // the source (or it never existed), so there is nothing to start.
        return Ok(MergeStart::NotEligible);
    };

    match state {
        PartitionTransition::Merging { .. } => return Ok(MergeStart::AlreadyMerging),
        PartitionTransition::Ready { .. } => {}
        // An intermediate-state partition cannot start a merge.
        _ => return Ok(MergeStart::NotEligible),
    }
    if source == root_partition() {
        return Ok(MergeStart::NotEligible);
    }
    if header.entry_count() >= manifest.config().min_partition_entries(header.level()) {
        return Ok(MergeStart::NotEligible);
    }

    // One same-level enumeration answers both begin questions: the source's
    // own incoming edge (its candidate occurrence) and whether a legal
    // target exists.
    let candidates = same_level_candidates(txn, manifest, tree_key, header.level()).await?;

    // Lock the source's unique incoming reference before the transition: a
    // concurrent edge move aborts the commit, so the merge never starts
    // against a stale parent observation. Zero or duplicate occurrences of
    // the source contradict the one-incoming-edge invariant (ADR 0007).
    let mut occurrences = candidates
        .iter()
        .filter(|candidate| candidate.partition() == source);
    let Some(own) = occurrences.next() else {
        return Err(corrupt());
    };
    if occurrences.next().is_some() {
        return Err(corrupt());
    }
    expect_child_entry(
        txn.get_for_update(LogicalKey::ChildEntry {
            index,
            tree_key: tree_key.clone(),
            partition: own.parent(),
            child: source,
        })
        .await?,
    )?
    .ok_or_else(corrupt)?;

    if !candidates
        .iter()
        .any(|candidate| candidate.is_legal_merge_target(source))
    {
        return Ok(MergeStart::NoReadyTarget);
    }

    txn.put(
        state_key,
        PersistentValue::PartitionState(PartitionTransition::Merging {
            started_at_unix_millis,
        }),
    )
    .await?;
    txn.put(
        header_key,
        PersistentValue::PartitionHeader(with_state(header, PartitionState::Merging)?),
    )
    .await?;
    Ok(MergeStart::Started)
}

/// The outcome of [`finalize_merge`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MergeCompletion {
    /// The merge completed: the source's incoming reference is removed and
    /// its prefix is gone.
    Completed,
    /// The source is `Merging` but its exact entry count is not zero.
    NotDrained,
    /// The source is not `Merging` (anymore). A merging source left the state
    /// only through a completed finalize, so re-driving a finalize after an
    /// unknown commit outcome observes this outcome.
    NotMerging,
}

/// Completes one fully drained merge in a single final transaction.
///
/// The exact update-protected source Header count is the sole proof of
/// emptiness; no entry rescan is performed. The transaction revalidates the
/// `Merging` state, removes the source's unique incoming Child Entry with the
/// parent's exact count, and removes the source prefix per `source_removal` —
/// one transactional range clear, or bounded point deletes of the four fixed
/// metadata keys. Merging persists no targets, so no target state changes and
/// no tombstone remains (ADR 0008). Roots never merge, so a `Merging` root is
/// Corruption.
pub async fn finalize_merge<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    source_removal: SourceRemoval,
) -> Result<MergeCompletion> {
    finalize_merge_with_parent(txn, tree_key, source, source_removal, None).await
}

/// Uses a parent only after the caller validates its complete discovery proof.
pub(crate) async fn finalize_merge_with_parent<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    source_removal: SourceRemoval,
    parent: Option<PartitionKey>,
) -> Result<MergeCompletion> {
    let manifest = txn.require_manifest()?;
    let (header_key, state_key) = authority_keys(manifest.logical_index_id(), tree_key, source);
    let Some((header, state)) = authority_pair(txn, header_key.clone(), state_key.clone()).await?
    else {
        // Both authority values are gone: a previous committed finalize
        // completed this merge.
        return Ok(MergeCompletion::Completed);
    };
    if !matches!(state, PartitionTransition::Merging { .. }) {
        return Ok(MergeCompletion::NotMerging);
    }
    if source == root_partition() {
        // Roots never merge, so a Merging root contradicts the protocol
        // regardless of its count.
        return Err(corrupt());
    }
    if header.entry_count() != 0 {
        return Ok(MergeCompletion::NotDrained);
    }
    let level = header.level();
    remove_incoming_edge(txn, tree_key, source, level, parent).await?;
    remove_source_prefix(
        txn,
        manifest,
        tree_key,
        source,
        header_key,
        state_key,
        source_removal,
    )
    .await?;
    Ok(MergeCompletion::Completed)
}

/// Removes one drained non-root source's unique incoming Child Entry and
/// decrements the parent's exact count, atomically.
///
/// The edge is rediscovered on this transaction's snapshot by the exact
/// bounded root-down scan and update-protected (ADR 0007): a concurrent edge
/// move aborts the commit with a retryable conflict, and a missing or
/// duplicated edge fails closed. A parent whose own maintenance moved the
/// edge is rediscovered afresh on the next attempt.
async fn remove_incoming_edge<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    level: u32,
    parent: Option<PartitionKey>,
) -> Result<PartitionKey> {
    let index = txn.require_manifest()?.logical_index_id();
    let parent = lock_incoming_edge(txn, tree_key, source, level, parent).await?;
    let parent_header_key = LogicalKey::Header {
        index,
        tree_key: tree_key.clone(),
        partition: parent,
    };
    let parent_header =
        expect_header(txn.get_for_update(parent_header_key.clone()).await?)?.ok_or_else(corrupt)?;
    if parent_header.level() != level.checked_add(1).ok_or_else(corrupt)? {
        return Err(corrupt());
    }
    txn.delete(LogicalKey::ChildEntry {
        index,
        tree_key: tree_key.clone(),
        partition: parent,
        child: source,
    })
    .await?;
    txn.put(
        parent_header_key,
        PersistentValue::PartitionHeader(removed_entry(parent_header)?),
    )
    .await?;
    Ok(parent)
}

/// Removes one fully drained source's whole partition prefix per the
/// backend's capability branch.
///
/// After exact count zero the prefix holds only the four fixed metadata keys,
/// so both forms are bounded; only the exact-count-zero invariant makes the
/// point form complete without a rescan. A leaf-only Synopsis key is simply
/// absent for an internal source, so its point delete is a no-op.
async fn remove_source_prefix<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    header_key: LogicalKey,
    state_key: LogicalKey,
    source_removal: SourceRemoval,
) -> Result<()> {
    let index = manifest.logical_index_id();
    match source_removal {
        SourceRemoval::TransactionalClear => {
            txn.clear_range(&LogicalRange::partition(manifest, tree_key, source)?)
                .await?;
        }
        SourceRemoval::PointDeletes => {
            for key in [
                LogicalKey::Synopsis {
                    index,
                    tree_key: tree_key.clone(),
                    partition: source,
                },
                LogicalKey::Centroid {
                    index,
                    tree_key: tree_key.clone(),
                    partition: source,
                },
                state_key,
                header_key,
            ] {
                txn.delete(key).await?;
            }
        }
    }
    Ok(())
}

/// Returns `header` with only its state discriminator replaced.
fn with_state(header: PartitionHeader, state: PartitionState) -> Result<PartitionHeader> {
    PartitionHeader::new(
        header.level(),
        header.entry_count(),
        header.cache_epoch(),
        state,
    )
    .map_err(|_| corrupt())
}
