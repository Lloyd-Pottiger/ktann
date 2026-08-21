//! Typed atomic topology operations for the expose-then-drain split protocol
//! (ADR 0014).
//!
//! These operations are the only writers of partition topology: partition
//! creation, split state transitions, Child Entry installation and removal,
//! and the structural entry moves that drain a split source into its two
//! targets. Each operation performs one bounded, atomically committable step
//! inside the caller's transaction; the multi-transaction driver lives in
//! [`crate::maintenance::split`]. Every authoritative read a step depends on is
//! update-protected, so a concurrent transition, foreground mutation, or
//! adjacent-level move aborts the commit with [`ErrorKind::RetryableAbort`]
//! and the whole step retries from a fresh snapshot.
//!
//! # Contract
//!
//! - **Expose-then-drain.** A split begins by reserving two never-reused
//!   target Partition Keys and marking the source `Splitting`; each target is
//!   then unique-created as `ReceivingSplit { source }` with its persisted
//!   centroid before the source advances to `DrainingSplit`. Repeating a
//!   committed step is harmless: every operation recognizes the state a
//!   previous committed attempt left and reports it instead of failing, so the
//!   documented recovery from an unknown commit outcome is to re-drive the
//!   same step.
//! - **One incoming reference.** A non-root target's creation and its Child
//!   Entry insertion into the source's current parent are one atomic step; a
//!   root target is instead owned by the exclusive target slot named by
//!   Partition Key 1's persisted `Splitting` state until root completion
//!   converts the root in place (ADR 0007). Target creation abandons without
//!   writing when the source no longer names the target or the discovered
//!   parent cannot accept a new child.
//! - **Exact movement.** Draining moves one bounded batch per transaction:
//!   each moved entry's target insert, source delete, and — for leaves —
//!   Record Location change commit atomically, with both exact Header counts
//!   and cache epochs and the target Synopsis read and written once per
//!   partition per batch. A source entry removed by a concurrent completed
//!   mutation is skipped; a remaining membership mismatch is Corruption.
//! - **Zero-count completion.** The exact update-protected source Header count
//!   is the sole authority for emptiness; completion never rescans entries
//!   (ADR 0014). After exact count zero a partition prefix holds only its four
//!   fixed metadata keys, so the final transaction removes the source with one
//!   transactional range clear when the backend supports it, or with bounded
//!   point deletes of those keys otherwise.
//! - **Fail closed.** Missing or mismatched authority values, a level or state
//!   disagreement, a duplicate or missing incoming reference, and an
//!   impossible count are Corruption. On any returned error the caller must
//!   not commit the transaction; rolling back leaves no partial change.

use bytes::Bytes;

use crate::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result};
use crate::storage::backend::{InsertOutcome, ScanLimits, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::membership::{added_entry, expect_inserted, removed_entry};
use crate::storage::tree_manifest::reserve_partition_keys;
use crate::storage::values::{
    ChildEntry, LeafEntry, PartitionCentroid, PartitionHeader, PartitionState, PartitionSynopsis,
    PartitionTransition, PersistentValue, RecordLocation, VectorRecord, expect_centroid,
    expect_child_entry, expect_child_entry_ref, expect_header, expect_leaf_entry, expect_location,
    expect_record, expect_state, expect_synopsis,
};
use crate::storage::{LogicalRange, LogicalReader, WriteLogicalTxn};

/// The number of split targets one split reserves and exposes; binary fanout
/// is a fixed format-v1 protocol choice.
const SPLIT_TARGETS: u32 = 2;

/// The bound on one incoming-edge discovery or entry scan page.
///
/// Paging only shapes I/O: discovery walks whole tree levels and every page is
/// bounded, so work stays proportional to the scanned level's actual size.
const DISCOVERY_SCAN_LIMITS: ScanLimits = ScanLimits {
    item_limit: 128,
    byte_limit: 1_048_576,
};

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

/// The outcome of [`create_split_target`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TargetInstall {
    /// This transaction unique-created the target and, for a non-root source,
    /// inserted its Child Entry into the source's current parent.
    Created,
    /// A previous committed attempt already created the target; its persisted
    /// centroid stands and nothing was written.
    AlreadyExists,
    /// The source no longer `Splitting`-names the target (the split advanced
    /// or completed); the step wrote nothing and must be abandoned.
    SourceAdvanced,
    /// The source's current parent cannot accept a new Child Entry (it is
    /// itself draining or merging); the step wrote nothing and must be
    /// abandoned so a later attempt can rediscover the source's parent.
    ParentNotAccepting,
}

/// The outcome of [`advance_to_draining`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DrainStart {
    /// This transaction transitioned the source to `DrainingSplit`.
    Advanced,
    /// The source was already `DrainingSplit`; nothing was written.
    AlreadyDraining,
    /// The source is not `Splitting` (anymore); nothing was written.
    NotSplitting,
}

/// The outcome of [`finalize_split`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SplitCompletion {
    /// The split completed: both targets are `Ready` and the source topology
    /// is switched.
    Completed,
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
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
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
    if header.entry_count() <= manifest.config().max_partition_entries() {
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

/// Unique-creates one split target and, for a non-root source, installs its
/// incoming Child Entry in the source's current parent, atomically.
///
/// The step update-protects the source State and verifies that it is still
/// `Splitting` and names `target`, so a stale worker cannot recreate or
/// relink a target after the source advances. The first successful creation
/// fixes the target's persisted centroid forever; a later attempt observes
/// [`TargetInstall::AlreadyExists`] and the persisted centroid stands. The
/// target is created at the source's level with an exact zero count and, for
/// a leaf split, the canonical empty Synopsis that leaf membership requires.
///
/// For a non-root source the same transaction rediscovers the source's unique
/// incoming Child Entry by an exact bounded root-down scan (ADR 0007),
/// update-protects the traversed parent Header and the source edge, verifies
/// that the parent still contains the source and accepts a new child, and
/// inserts the target edge. A root target has no parent: it occupies the
/// exclusive target slot named by Partition Key 1's persisted `Splitting`
/// state until root completion exposes it.
///
/// Only [`TargetInstall::Created`] carries writes; every other outcome leaves
/// the transaction without mutations.
pub async fn create_split_target<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    target: PartitionKey,
    centroid: &PartitionCentroid,
    started_at_unix_millis: u64,
) -> Result<TargetInstall> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let index = manifest.logical_index_id();
    let (source_header_key, source_state_key) = authority_keys(index, tree_key, source);

    // The source State is update-protected so a concurrent transition aborts
    // the commit; its absence means a completed split already removed it.
    let Some(state) = expect_state(txn.get_for_update(source_state_key).await?)? else {
        return Ok(TargetInstall::SourceAdvanced);
    };
    let (left, right) = match state {
        PartitionTransition::Splitting { left, right, .. } => (left, right),
        PartitionTransition::DrainingSplit { left, right, .. } => {
            // Advancing verified both targets exist, so a named target absent
            // in the same snapshot is a torn committed state.
            if target != left && target != right {
                return Err(Error::invalid_argument());
            }
            return match read_target_state(txn, index, tree_key, source, target).await? {
                Some(()) => Ok(TargetInstall::AlreadyExists),
                None => Err(corrupt()),
            };
        }
        // Ready, ReceivingSplit, or Merging: the split completed or never
        // began; a stale worker must abandon, not recreate.
        _ => return Ok(TargetInstall::SourceAdvanced),
    };
    if target != left && target != right {
        return Err(Error::invalid_argument());
    }

    // A previous committed attempt is recognized before any discovery work;
    // its persisted centroid stands.
    let (target_header_key, target_state_key) = authority_keys(index, tree_key, target);
    if read_header_opt(txn, index, tree_key, target)
        .await?
        .is_some()
    {
        return match read_target_state(txn, index, tree_key, source, target).await? {
            Some(()) => Ok(TargetInstall::AlreadyExists),
            None => Err(corrupt()),
        };
    }

    let source_header = expect_header(txn.get(source_header_key).await?)?.ok_or_else(corrupt)?;
    expect_agreement(source_header, state)?;
    let level = source_header.level();

    // For a non-root source, discover and validate the incoming topology
    // before writing anything. The parent Header's update-protected read both
    // validates the acceptance rules and feeds the exact-count update below.
    let parent = if source == root_partition() {
        None
    } else {
        let Some((parent, _)) = find_incoming_edge(txn, tree_key, source, level).await? else {
            // A non-root partition has exactly one incoming Child Entry in
            // every committed state (ADR 0007).
            return Err(corrupt());
        };
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
            _ => return Ok(TargetInstall::ParentNotAccepting),
        }
        // The discovery scan ran on this transaction's snapshot, so the source
        // edge must still decode; the update-protected re-read establishes the
        // commit-time conflict that reroutes a concurrent edge move.
        expect_child_entry(
            txn.get_for_update(LogicalKey::ChildEntry {
                index,
                tree_key: tree_key.clone(),
                partition: parent,
                child: source,
            })
            .await?,
        )?
        .ok_or_else(corrupt)?;
        Some((parent, parent_header))
    };

    // Every key of the target partition is a unique insert: the partition did
    // not exist in this snapshot and Partition Keys are never reused, so an
    // existing key is a torn committed state.
    let created = txn
        .insert(
            target_header_key,
            PersistentValue::PartitionHeader(
                PartitionHeader::new(level, 0, 0, PartitionState::ReceivingSplit)
                    .map_err(|_| corrupt())?,
            ),
        )
        .await?;
    if created != InsertOutcome::Inserted {
        return Err(corrupt());
    }
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
    if let Some((parent, parent_header)) = parent {
        expect_inserted(
            txn.insert(
                LogicalKey::ChildEntry {
                    index,
                    tree_key: tree_key.clone(),
                    partition: parent,
                    child: target,
                },
                PersistentValue::ChildEntry(ChildEntry::new(
                    target,
                    centroid.components().to_vec(),
                )),
            )
            .await?,
        )?;
        txn.put(
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: parent,
            },
            PersistentValue::PartitionHeader(added_entry(parent_header)?),
        )
        .await?;
    }
    Ok(TargetInstall::Created)
}

/// Advances one fully exposed split source from `Splitting` to
/// `DrainingSplit`.
///
/// The transition update-protects the source and verifies that both persisted
/// targets identify it as their source and are complete — State, Header, and
/// Centroid present and in agreement at the source's level — so the source
/// never commits into a `DrainingSplit` that the fail-closed drain and
/// completion steps would wedge behind. It deliberately ignores the source's
/// entries, count, and cache epoch: training and publication never restart or
/// revalidate because of concurrent foreground writes (ADR 0014). The Header
/// write that keeps the state discriminator in agreement can still conflict
/// with a concurrent foreground write to the source; the caller's whole-step
/// retry absorbs that, and exhaustion leaves a searchable `Splitting` state
/// for a later access to rediscover.
pub async fn advance_to_draining<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
) -> Result<DrainStart> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let index = manifest.logical_index_id();
    let (header_key, state_key) = authority_keys(index, tree_key, source);
    let Some((header, state)) = authority_pair(txn, header_key.clone(), state_key.clone()).await?
    else {
        // Both authority values are gone: a completed split already removed
        // the source, so there is nothing to advance.
        return Ok(DrainStart::NotSplitting);
    };
    let (left, right) = match state {
        PartitionTransition::Splitting { left, right, .. } => (left, right),
        PartitionTransition::DrainingSplit { .. } => return Ok(DrainStart::AlreadyDraining),
        _ => return Ok(DrainStart::NotSplitting),
    };

    // Both targets must identify this source; one update-protected batch
    // establishes the commit-time conflicts on both target States.
    let mut target_states = txn
        .batch_get_for_update(
            [left, right]
                .map(|target| LogicalKey::State {
                    index,
                    tree_key: tree_key.clone(),
                    partition: target,
                })
                .into(),
        )
        .await?
        .into_iter();
    let (Some(left_state), Some(right_state)) = (target_states.next(), target_states.next()) else {
        // The typed batch read returns exactly one value per input key.
        return Err(Error::new(ErrorKind::Backend));
    };

    // Each target must also carry its Header and Centroid; committing the
    // transition with a torn target would wedge the source behind the
    // fail-closed drain and completion checks. These are plain reads: the
    // update-protected States above already conflict with any concurrent
    // target transition, and target counts and epochs may change freely.
    let mut target_parts = txn
        .batch_get(
            [left, right]
                .into_iter()
                .flat_map(|target| {
                    [
                        LogicalKey::Header {
                            index,
                            tree_key: tree_key.clone(),
                            partition: target,
                        },
                        LogicalKey::Centroid {
                            index,
                            tree_key: tree_key.clone(),
                            partition: target,
                        },
                    ]
                })
                .collect(),
        )
        .await?
        .into_iter();
    for state_value in [left_state, right_state] {
        match expect_state(state_value)? {
            Some(PartitionTransition::ReceivingSplit { source: s, .. }) if s == source => {}
            _ => return Err(corrupt()),
        }
        let (Some(header_value), Some(centroid_value)) = (target_parts.next(), target_parts.next())
        else {
            // The typed batch read returns exactly one value per input key.
            return Err(Error::new(ErrorKind::Backend));
        };
        let target_header = expect_header(header_value)?.ok_or_else(corrupt)?;
        if target_header.state() != PartitionState::ReceivingSplit
            || target_header.level() != header.level()
        {
            return Err(corrupt());
        }
        expect_centroid(centroid_value)?.ok_or_else(corrupt)?;
    }

    txn.put(
        state_key,
        PersistentValue::PartitionState(PartitionTransition::DrainingSplit {
            left,
            right,
            started_at_unix_millis,
        }),
    )
    .await?;
    txn.put(
        header_key,
        PersistentValue::PartitionHeader(with_state(header, PartitionState::DrainingSplit)?),
    )
    .await?;
    Ok(DrainStart::Advanced)
}

/// One verified leaf drain candidate: the source Leaf Entry and its current
/// Vector Record, read from the caller's transaction with the entry and its
/// Record Location update-protected.
///
/// The Vector Record is read without update protection on purpose: a
/// concurrent same-leaf replacement rewrites the entry in the same commit, so
/// the entry conflict covers the pair, and the consistent snapshot keeps the
/// decoded record in agreement with the entry.
pub struct LeafDrainEntry {
    entry: LeafEntry,
    record: VectorRecord,
}

impl LeafDrainEntry {
    /// Returns the source Leaf Entry, copied verbatim to the target on a move.
    #[must_use]
    pub const fn entry(&self) -> &LeafEntry {
        &self.entry
    }

    /// Returns the current Vector Record used for the routing decision.
    #[must_use]
    pub const fn record(&self) -> &VectorRecord {
        &self.record
    }
}

impl std::fmt::Debug for LeafDrainEntry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LeafDrainEntry([REDACTED])")
    }
}

/// Re-reads one batch of leaf drain candidates inside the drain write
/// transaction.
///
/// Returns one slot per input Record ID, in input order. A `None` slot means
/// the source entry is gone — exactly the committed trace of a concurrent
/// foreground mutation, and never an error. A remaining entry whose Record
/// Location or Vector Record is absent or inconsistent is Corruption. The
/// reads are three batches (entries and Locations update-protected, Records
/// plain), not per-entry round trips.
pub async fn read_leaf_drain_candidates<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    record_ids: &[Bytes],
) -> Result<Vec<Option<LeafDrainEntry>>> {
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();
    let entry_keys: Vec<LogicalKey> = record_ids
        .iter()
        .map(|id| LogicalKey::LeafEntry {
            index,
            tree_key: tree_key.clone(),
            partition: source,
            id: id.clone(),
        })
        .collect();
    let mut entries = txn.batch_get_for_update(entry_keys).await?.into_iter();

    // Validate present entries, then batch their Locations and Records.
    let mut present: Vec<(usize, LeafEntry)> = Vec::new();
    let mut slots: Vec<Option<LeafDrainEntry>> = Vec::with_capacity(record_ids.len());
    for (position, id) in record_ids.iter().enumerate() {
        let Some(entry_value) = entries.next() else {
            // The typed batch read returns exactly one value per input key.
            return Err(Error::new(ErrorKind::Backend));
        };
        match expect_leaf_entry(entry_value)? {
            Some(entry) if entry.record_id() == id => {
                present.push((position, entry));
                slots.push(None);
            }
            Some(_) => return Err(corrupt()),
            None => slots.push(None),
        }
    }

    let location_keys: Vec<LogicalKey> = present
        .iter()
        .map(|(_, entry)| LogicalKey::Location {
            index,
            id: entry.record_id().clone(),
        })
        .collect();
    let record_keys: Vec<LogicalKey> = present
        .iter()
        .map(|(_, entry)| LogicalKey::Record {
            index,
            id: entry.record_id().clone(),
        })
        .collect();
    let mut locations = txn.batch_get_for_update(location_keys).await?.into_iter();
    let mut records = txn.batch_get(record_keys).await?.into_iter();

    for (position, entry) in present {
        let (Some(location_value), Some(record_value)) = (locations.next(), records.next()) else {
            return Err(Error::new(ErrorKind::Backend));
        };
        let location = expect_location(location_value)?.ok_or_else(corrupt)?;
        if location.tree_key() != tree_key || location.leaf() != source {
            return Err(corrupt());
        }
        let record = expect_record(record_value)?.ok_or_else(corrupt)?;
        if record.record_id() != entry.record_id() {
            return Err(corrupt());
        }
        slots[position] = Some(LeafDrainEntry { entry, record });
    }
    Ok(slots)
}

/// Atomically moves one batch of verified leaf entries from the split source
/// to their chosen targets.
///
/// Each move copies the Leaf Entry — including its absolute RaBitQ7 payload —
/// unchanged to the target, unique-inserts it, deletes the source entry, and
/// repoints the Record Location. The batch reads the source Header and each
/// receiving target's Header and Synopsis once, accumulates the exact counts,
/// cache epochs, and synopsis expansions in memory, and writes each authority
/// value back once, so a batch never pays per-entry authority round trips.
/// The source must be `DrainingSplit` and every target must be a
/// `ReceivingSplit` leaf: the exact-membership invariant holds only while
/// movement runs inside the split protocol. Returns the number of moved
/// entries.
pub async fn relocate_leaf_entries<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    moves: Vec<(LeafDrainEntry, PartitionKey)>,
) -> Result<usize> {
    if moves.is_empty() {
        return Ok(0);
    }
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let index = manifest.logical_index_id();

    // One update-protected read of each touched partition Header, and one of
    // each receiving target's Synopsis: a batch never pays per-partition
    // authority round trips.
    let mut targets: Vec<PartitionKey> = moves.iter().map(|(_, target)| *target).collect();
    targets.sort_unstable();
    targets.dedup();
    let mut header_keys = Vec::with_capacity(targets.len() + 1);
    for target in &targets {
        header_keys.push(LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: *target,
        });
    }
    header_keys.push(LogicalKey::Header {
        index,
        tree_key: tree_key.clone(),
        partition: source,
    });
    let mut headers = txn.batch_get_for_update(header_keys).await?.into_iter();
    let mut target_headers: Vec<(PartitionKey, PartitionHeader)> = Vec::new();
    for target in &targets {
        let Some(value) = headers.next() else {
            return Err(Error::new(ErrorKind::Backend));
        };
        let header = expect_header(value)?.ok_or_else(corrupt)?;
        if header.level() != 1 || header.state() != PartitionState::ReceivingSplit {
            return Err(corrupt());
        }
        target_headers.push((*target, header));
    }
    let Some(source_value) = headers.next() else {
        return Err(Error::new(ErrorKind::Backend));
    };
    let source_header = expect_header(source_value)?.ok_or_else(corrupt)?;
    // Movement is legal only inside the split protocol: the source must be
    // draining into exactly these `ReceivingSplit` targets.
    if source_header.level() != 1 || source_header.state() != PartitionState::DrainingSplit {
        return Err(corrupt());
    }

    let synopsis_keys: Vec<LogicalKey> = targets
        .iter()
        .map(|target| LogicalKey::Synopsis {
            index,
            tree_key: tree_key.clone(),
            partition: *target,
        })
        .collect();
    let mut synopsis_values = txn.batch_get_for_update(synopsis_keys).await?.into_iter();
    let mut target_synopses: Vec<PartitionSynopsis> = Vec::with_capacity(targets.len());
    for _ in &targets {
        let Some(value) = synopsis_values.next() else {
            return Err(Error::new(ErrorKind::Backend));
        };
        target_synopses.push(expect_synopsis(value)?.ok_or_else(corrupt)?);
    }

    let moved = moves.len();
    for (drain, target) in &moves {
        let id = drain.entry.record_id().clone();
        expect_inserted(
            txn.insert(
                LogicalKey::LeafEntry {
                    index,
                    tree_key: tree_key.clone(),
                    partition: *target,
                    id: id.clone(),
                },
                PersistentValue::LeafEntry(drain.entry.clone()),
            )
            .await?,
        )?;
        txn.delete(LogicalKey::LeafEntry {
            index,
            tree_key: tree_key.clone(),
            partition: source,
            id: id.clone(),
        })
        .await?;
        txn.put(
            LogicalKey::Location { index, id },
            PersistentValue::RecordLocation(RecordLocation::new(tree_key.clone(), *target)),
        )
        .await?;
    }

    // Write each touched authority value back once.
    let mut source_header = source_header;
    for _ in &moves {
        source_header = removed_entry(source_header)?;
    }
    txn.put(
        LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: source,
        },
        PersistentValue::PartitionHeader(source_header),
    )
    .await?;
    for ((target, header), mut synopsis) in target_headers.into_iter().zip(target_synopses) {
        let mut header = header;
        let original = synopsis.clone();
        for (drain, move_target) in &moves {
            if move_target == &target {
                header = added_entry(header)?;
                synopsis.expand(manifest, drain.entry.fields())?;
            }
        }
        let synopsis_changed = synopsis != original;
        txn.put(
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: target,
            },
            PersistentValue::PartitionHeader(header),
        )
        .await?;
        if synopsis_changed {
            txn.put(
                LogicalKey::Synopsis {
                    index,
                    tree_key: tree_key.clone(),
                    partition: target,
                },
                PersistentValue::PartitionSynopsis(synopsis),
            )
            .await?;
        }
    }
    Ok(moved)
}

/// Re-reads one batch of internal drain candidates inside the drain write
/// transaction.
///
/// Returns one slot per input child Partition Key, in input order; a `None`
/// slot means the source Child Entry is gone — the committed trace of a
/// concurrent move — and never an error. A key/value identity mismatch is
/// Corruption.
pub async fn read_child_drain_candidates<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    children: &[PartitionKey],
) -> Result<Vec<Option<ChildEntry>>> {
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();
    let keys: Vec<LogicalKey> = children
        .iter()
        .map(|child| LogicalKey::ChildEntry {
            index,
            tree_key: tree_key.clone(),
            partition: source,
            child: *child,
        })
        .collect();
    let values = txn.batch_get_for_update(keys).await?;
    let mut slots = Vec::with_capacity(children.len());
    for (child, value) in children.iter().zip(values) {
        match expect_child_entry(value)? {
            Some(entry) if entry.child() == *child => slots.push(Some(entry)),
            Some(_) => return Err(corrupt()),
            None => slots.push(None),
        }
    }
    Ok(slots)
}

/// Atomically moves one batch of verified Child Entries from the split source
/// to their chosen targets.
///
/// Internal movement applies the same exact source-delete/target-insert rule
/// as leaf movement and updates both exact Header counts and cache epochs —
/// read and written once per partition per batch — but touches neither Record
/// Location, Vector Record, nor Synopsis (ADR 0014): exact Child Entry
/// ownership is the coordination point between adjacent-level maintenance.
/// The source must be `DrainingSplit` and every target must be
/// `ReceivingSplit` at the source's level: the exact-membership invariant
/// holds only while movement runs inside the split protocol. Returns the
/// number of moved entries.
pub async fn relocate_child_entries<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    moves: Vec<(ChildEntry, PartitionKey)>,
) -> Result<usize> {
    if moves.is_empty() {
        return Ok(0);
    }
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let index = manifest.logical_index_id();

    let mut targets: Vec<PartitionKey> = moves.iter().map(|(_, target)| *target).collect();
    targets.sort_unstable();
    targets.dedup();
    let mut header_keys = Vec::with_capacity(targets.len() + 1);
    for target in &targets {
        header_keys.push(LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: *target,
        });
    }
    header_keys.push(LogicalKey::Header {
        index,
        tree_key: tree_key.clone(),
        partition: source,
    });
    let mut headers = txn.batch_get_for_update(header_keys).await?.into_iter();
    let mut target_headers: Vec<(PartitionKey, PartitionHeader)> = Vec::new();
    for target in &targets {
        let Some(value) = headers.next() else {
            return Err(Error::new(ErrorKind::Backend));
        };
        let header = expect_header(value)?.ok_or_else(corrupt)?;
        if header.state() != PartitionState::ReceivingSplit {
            return Err(corrupt());
        }
        target_headers.push((*target, header));
    }
    let Some(source_value) = headers.next() else {
        return Err(Error::new(ErrorKind::Backend));
    };
    let source_header = expect_header(source_value)?.ok_or_else(corrupt)?;
    // Movement is legal only inside the split protocol: the source must be
    // draining into exactly these `ReceivingSplit` targets.
    if source_header.state() != PartitionState::DrainingSplit {
        return Err(corrupt());
    }
    for (_, target_header) in &target_headers {
        if target_header.level() != source_header.level() {
            return Err(corrupt());
        }
    }

    let moved = moves.len();
    for (entry, target) in &moves {
        expect_inserted(
            txn.insert(
                LogicalKey::ChildEntry {
                    index,
                    tree_key: tree_key.clone(),
                    partition: *target,
                    child: entry.child(),
                },
                PersistentValue::ChildEntry(entry.clone()),
            )
            .await?,
        )?;
        txn.delete(LogicalKey::ChildEntry {
            index,
            tree_key: tree_key.clone(),
            partition: source,
            child: entry.child(),
        })
        .await?;
    }

    let mut source_header = source_header;
    for _ in &moves {
        source_header = removed_entry(source_header)?;
    }
    txn.put(
        LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: source,
        },
        PersistentValue::PartitionHeader(source_header),
    )
    .await?;
    for (target, mut header) in target_headers {
        for (_, move_target) in &moves {
            if move_target == &target {
                header = added_entry(header)?;
            }
        }
        txn.put(
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: target,
            },
            PersistentValue::PartitionHeader(header),
        )
        .await?;
    }
    Ok(moved)
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
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let index = manifest.logical_index_id();
    let (header_key, state_key) = authority_keys(index, tree_key, source);
    let Some((header, state)) = authority_pair(txn, header_key.clone(), state_key.clone()).await?
    else {
        // Both authority values are gone: a previous committed finalize
        // completed this split.
        return Ok(SplitCompletion::Completed);
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
                    PersistentValue::ChildEntry(ChildEntry::new(
                        target,
                        centroid.components().to_vec(),
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
        return Ok(SplitCompletion::Completed);
    }

    // Remove the source's unique incoming reference, rediscovered on this
    // transaction's snapshot and update-protected (ADR 0007).
    let Some((parent, _)) = find_incoming_edge(txn, tree_key, source, level).await? else {
        return Err(corrupt());
    };
    expect_child_entry(
        txn.get_for_update(LogicalKey::ChildEntry {
            index,
            tree_key: tree_key.clone(),
            partition: parent,
            child: source,
        })
        .await?,
    )?
    .ok_or_else(corrupt)?;
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

    match source_removal {
        SourceRemoval::TransactionalClear => {
            txn.clear_range(&LogicalRange::partition(manifest, tree_key, source)?)
                .await?;
        }
        SourceRemoval::PointDeletes => {
            // After exact count zero the prefix holds only the four fixed
            // metadata keys; the entry ranges are provably empty without a
            // rescan (ADR 0014).
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
    Ok(SplitCompletion::Completed)
}

/// Rediscovers one partition's unique incoming Child Entry by an exact
/// bounded root-down scan of every Child Entry at the required parent level
/// (ADR 0007).
///
/// Partitions persist no reverse parent pointers, so the scan walks from the
/// root down to `level + 1`, reading every partition Header at each level to
/// prove the level invariant, and matches the child Partition Key. While the
/// root is `Splitting` or `DrainingSplit` its targets have no incoming edge
/// of their own, so the exclusive target slots named by the root State are
/// added to the root's level. A duplicate or missing match is Corruption.
async fn find_incoming_edge<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    partition: PartitionKey,
    level: u32,
) -> Result<Option<(PartitionKey, ChildEntry)>> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let index = manifest.logical_index_id();
    let parent_level = level.checked_add(1).ok_or_else(corrupt)?;
    let root = root_partition();
    let root_header = read_header(txn, index, tree_key, root).await?;
    if root_header.level() < parent_level {
        return Err(corrupt());
    }

    let mut current_level = root_header.level();
    let mut bodies = vec![root];
    loop {
        // While the root exposes its split targets only through its own
        // State, those targets sit at the root's level without an incoming
        // edge; add them exactly once. While the root is merely Splitting a
        // target may not be exposed yet: an absent target holds no Child
        // Entries and is skipped. While DrainingSplit both targets exist
        // because advancing verified them.
        if current_level == root_header.level() {
            if let Some(root_state) = read_state(txn, index, tree_key, root).await? {
                let draining = matches!(root_state, PartitionTransition::DrainingSplit { .. });
                let targets = match root_state {
                    PartitionTransition::Splitting { left, right, .. }
                    | PartitionTransition::DrainingSplit { left, right, .. } => [left, right],
                    _ => [root, root],
                };
                for target in targets {
                    if target == root || bodies.contains(&target) {
                        continue;
                    }
                    let exists = read_header_opt(txn, index, tree_key, target)
                        .await?
                        .is_some();
                    if !exists && !draining {
                        continue;
                    }
                    bodies.push(target);
                }
            }
        }

        if current_level == parent_level {
            // Only the parent level's matches are collected.
            let mut found: Option<(PartitionKey, ChildEntry)> = None;
            for body in &bodies {
                let range = LogicalRange::child_entries(manifest, tree_key, *body)?;
                let mut cursor = None;
                loop {
                    let page = txn
                        .scan(&range, cursor.as_ref(), DISCOVERY_SCAN_LIMITS)
                        .await?;
                    for item in page.items() {
                        let entry = expect_child_entry_ref(item.value())?;
                        if entry.child() == partition {
                            if found.is_some() {
                                return Err(corrupt());
                            }
                            found = Some((*body, entry.clone()));
                        }
                    }
                    cursor = page.into_next_cursor();
                    if cursor.is_none() {
                        break;
                    }
                }
            }
            return Ok(found);
        }

        // Descend: intermediate levels contribute only child Partition Keys.
        let mut next = Vec::new();
        for body in &bodies {
            let header = read_header(txn, index, tree_key, *body).await?;
            if header.level() != current_level {
                return Err(corrupt());
            }
            let range = LogicalRange::child_entries(manifest, tree_key, *body)?;
            let mut cursor = None;
            loop {
                let page = txn
                    .scan(&range, cursor.as_ref(), DISCOVERY_SCAN_LIMITS)
                    .await?;
                for item in page.items() {
                    next.push(expect_child_entry_ref(item.value())?.child());
                }
                cursor = page.into_next_cursor();
                if cursor.is_none() {
                    break;
                }
            }
        }
        if next.is_empty() {
            // A level above one that cannot be reached contradicts the
            // level-decrement invariant.
            return Err(corrupt());
        }
        bodies = next;
        current_level -= 1;
    }
}

/// Reads one partition Header that must exist.
///
/// This is the single home of the typed authority read; read-only and write
/// transactions share it through [`LogicalReader`]. A missing Header on a
/// referenced partition is Corruption.
pub(crate) async fn read_header<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<PartitionHeader> {
    read_header_opt(reader, index, tree_key, partition)
        .await?
        .ok_or_else(corrupt)
}

/// Reads one partition Header, which may be absent after a completed split.
pub(crate) async fn read_header_opt<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Option<PartitionHeader>> {
    expect_header(
        reader
            .get(LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition,
            })
            .await?,
    )
}

/// Reads one partition State, which may be absent after a completed split.
pub(crate) async fn read_state<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Option<PartitionTransition>> {
    expect_state(
        reader
            .get(LogicalKey::State {
                index,
                tree_key: tree_key.clone(),
                partition,
            })
            .await?,
    )
}

/// Reads one partition's authority pair in one batched plain read, without
/// classifying presence or agreement.
///
/// One batched call covers what would otherwise be two sequential point
/// reads; the caller classifies the pair because the meaning of a half-present
/// pair depends on the state machine step.
pub(crate) async fn read_authority_opt<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<(Option<PartitionHeader>, Option<PartitionTransition>)> {
    let (header_key, state_key) = authority_keys(index, tree_key, partition);
    let mut values = reader
        .batch_get(vec![header_key, state_key])
        .await?
        .into_iter();
    let (Some(header), Some(state)) = (values.next(), values.next()) else {
        // The typed batch read returns exactly one value per input key.
        return Err(Error::new(ErrorKind::Backend));
    };
    Ok((expect_header(header)?, expect_state(state)?))
}

/// Reads one visited partition's Header and State in one batch, failing
/// closed when either is missing, of the wrong kind, or in disagreement.
///
/// Every reachable partition carries both authority values in every committed
/// state: creation installs them together, and completion removes them
/// together with the partition's last incoming reference.
pub(crate) async fn read_authority<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<(PartitionHeader, PartitionTransition)> {
    match read_authority_opt(reader, index, tree_key, partition).await? {
        (Some(header), Some(state)) => {
            expect_agreement(header, state)?;
            Ok((header, state))
        }
        _ => Err(corrupt()),
    }
}

/// Reads one partition's Header and State with update protection in one
/// batch.
pub(crate) async fn authority_for_update<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    header_key: LogicalKey,
    state_key: LogicalKey,
) -> Result<(Option<PartitionHeader>, Option<PartitionTransition>)> {
    let mut values = txn
        .batch_get_for_update(vec![header_key, state_key])
        .await?
        .into_iter();
    let (Some(header), Some(state)) = (values.next(), values.next()) else {
        // The typed batch read returns exactly one value per input key.
        return Err(Error::new(ErrorKind::Backend));
    };
    Ok((expect_header(header)?, expect_state(state)?))
}

/// Reads one partition's authority pair with update protection, classifying
/// presence and agreement.
///
/// Both values present and in agreement is `Some`; both gone — a completed
/// split removed them, or the partition never existed — is `None`; a
/// half-present or disagreeing pair is Corruption.
async fn authority_pair<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    header_key: LogicalKey,
    state_key: LogicalKey,
) -> Result<Option<(PartitionHeader, PartitionTransition)>> {
    match authority_for_update(txn, header_key, state_key).await? {
        (Some(header), Some(state)) => {
            expect_agreement(header, state)?;
            Ok(Some((header, state)))
        }
        (None, None) => Ok(None),
        _ => Err(corrupt()),
    }
}

/// Verifies that one target's persisted State names `source`, without a
/// conflict.
async fn read_target_state<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    source: PartitionKey,
    target: PartitionKey,
) -> Result<Option<()>> {
    match read_state(txn, index, tree_key, target).await? {
        Some(PartitionTransition::ReceivingSplit { source: s, .. }) if s == source => Ok(Some(())),
        Some(_) => Err(corrupt()),
        None => Ok(None),
    }
}

/// Builds the Header and State keys of one partition.
fn authority_keys(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> (LogicalKey, LogicalKey) {
    (
        LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition,
        },
        LogicalKey::State {
            index,
            tree_key: tree_key.clone(),
            partition,
        },
    )
}

/// The stable root Partition Key of every tree.
///
/// The Tree Manifest constructor rejects any root other than Partition Key 1,
/// so construction cannot fail.
fn root_partition() -> PartitionKey {
    match PartitionKey::new(1) {
        Ok(root) => root,
        Err(_) => unreachable!("Partition Key 1 is nonzero"),
    }
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

/// Verifies that a Header discriminator and a State value agree.
fn expect_agreement(header: PartitionHeader, state: PartitionTransition) -> Result<()> {
    if header.state() != state.state() {
        return Err(corrupt());
    }
    Ok(())
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}
