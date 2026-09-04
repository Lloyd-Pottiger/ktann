//! Typed atomic record-membership operations.
//!
//! The exact-membership invariant (ADR 0001) gives every committed Vector
//! Record exactly one Record Location and one corresponding Leaf Entry. These
//! primitives are the only foreground writers of that membership: each one
//! performs the complete mutation — Vector Record, Record Location, Opaque
//! Payload, Leaf Entry, exact Header count and cache epoch, and the target
//! Synopsis — inside the caller's one transaction, so a commit either installs
//! the whole mutation or nothing. Routing and retry orchestration live in the
//! maintenance module; these functions take the caller's exact expected and
//! target locations and never route.
//!
//! Record-group writes (Vector Record, Record Location, Opaque Payload, Leaf
//! Entry) are queued into the caller's [`MutationBuilder`] and become visible
//! only when it is applied, so a whole mutation batch lands in canonical key
//! order with one backend write call; the queue replaces backend unique
//! inserts with update-protected existence checks. Exact Header counts and
//! Synopsis expansions accumulate per leaf in a [`LeafAccumulator`] and join
//! the queued writes when it is flushed, so a batch touching one leaf N times
//! reads and writes the leaf's Header and Synopsis once; later items observe
//! earlier items' adjustments through the accumulator. Deferral is exact
//! because a validated batch holds at most one item per Record ID; callers
//! must enforce that before queueing.
//!
//! Every authoritative read a mutation depends on is update-protected, so a
//! concurrent change to the same membership or leaf Header aborts the commit
//! with [`ErrorKind::RetryableAbort`] instead of producing a partial write. A
//! missing, extra, or mismatched authoritative value is
//! [`ErrorKind::Corruption`] and fails closed without repair. On any returned
//! error the caller must not commit the transaction; rolling back leaves no
//! partial change.

use std::collections::{BTreeMap, btree_map::Entry};

use bytes::Bytes;

use crate::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result, Value};
use crate::storage::backend::{InsertOutcome, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    IndexManifest, LeafEntry, OpaquePayload, PartitionHeader, PartitionSynopsis, PersistentValue,
    RecordLocation, VectorRecord, expect_header, expect_leaf_entry, expect_location, expect_record,
    expect_synopsis,
};
use crate::storage::{MutationBuilder, WriteLogicalTxn};

/// The outcome of an idempotent delete.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DeleteOutcome {
    /// The record existed and its whole membership group was deleted; carries
    /// the exact location it was deleted from, so the caller may offer the
    /// shrunken leaf to demand-driven maintenance.
    Deleted {
        /// The deleted record's authoritative Record Location.
        location: RecordLocation,
    },
    /// No Vector Record with the Record ID exists; nothing was touched.
    NotFound,
}

/// Internal delete result carrying the changed Header for scheduling.
pub(crate) enum DeleteReport {
    /// The record existed and its leaf Header changed.
    Deleted {
        location: RecordLocation,
        header: PartitionHeader,
    },
    /// No record existed and no Header changed.
    NotFound,
}

/// Partition Headers produced by one replacement upsert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReplacementHeaders {
    source: Option<PartitionHeader>,
    target: PartitionHeader,
}

impl ReplacementHeaders {
    /// Returns the shrunken source Header for a cross-leaf replacement.
    #[must_use]
    pub(crate) const fn source(self) -> Option<PartitionHeader> {
        self.source
    }

    /// Returns the target Header after the replacement.
    #[must_use]
    pub(crate) const fn target(self) -> PartitionHeader {
        self.target
    }
}

/// Per-leaf Header and Synopsis write state for one mutation batch.
///
/// The membership operations adjust exact Header counts and expand Synopses
/// against this accumulator instead of writing each adjustment straight to
/// the transaction: a leaf touched by N items is read once, adjusted once per
/// item in input order with the same checked arithmetic, and written back
/// once at [`flush`](Self::flush), so a batch's Header and Synopsis writes
/// collapse from two puts per item to one per touched leaf. The first access
/// to one leaf reads its Header or Synopsis update-protected through the
/// transaction, establishing the same commit-time conflict the unbatched
/// sequence establishes; later items observe their batch-mates' adjustments
/// through the accumulator rather than through read-your-writes. The
/// committed values and every per-item error are identical to the unbatched
/// sequence.
#[derive(Default)]
pub(crate) struct LeafAccumulator {
    headers: BTreeMap<(TreeKey, PartitionKey), PartitionHeader>,
    synopses: BTreeMap<(TreeKey, PartitionKey), SynopsisExpansion>,
}

/// One leaf's accumulated Synopsis state: the current expansion and whether
/// it differs from the stored Synopsis.
struct SynopsisExpansion {
    synopsis: PartitionSynopsis,
    changed: bool,
}

impl LeafAccumulator {
    /// Creates an empty accumulator.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns one leaf's current Header: the accumulated adjustments when
    /// the batch already touched the leaf, otherwise its stored Header read
    /// update-protected. A missing Header is [`ErrorKind::Corruption`].
    async fn header<T: WriteTxn>(
        &mut self,
        txn: &mut WriteLogicalTxn<'_, T>,
        index: LogicalIndexId,
        location: &RecordLocation,
    ) -> Result<PartitionHeader> {
        let key = (location.tree_key().clone(), location.leaf());
        if let Some(header) = self.headers.get(&key) {
            return Ok(*header);
        }
        let header = read_header(txn, index, location).await?;
        self.headers.insert(key, header);
        Ok(header)
    }

    /// Stores one item's adjusted Header as the leaf's current state.
    fn adjust_header(&mut self, location: &RecordLocation, header: PartitionHeader) {
        self.headers
            .insert((location.tree_key().clone(), location.leaf()), header);
    }

    /// Expands one leaf's Synopsis with one exact Leaf projection.
    ///
    /// The stored Synopsis was fully validated at decode, so a projection
    /// mismatch here is caller input; a missing Synopsis is corruption. The
    /// update-protected read on the first access already establishes the
    /// commit-time conflict, so an unchanged expansion is never written.
    async fn expand_synopsis<T: WriteTxn>(
        &mut self,
        txn: &mut WriteLogicalTxn<'_, T>,
        manifest: &IndexManifest,
        target: &RecordLocation,
        projection: &[Value],
    ) -> Result<()> {
        let key = (target.tree_key().clone(), target.leaf());
        let expansion = match self.synopses.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let stored = expect_synopsis(
                    txn.get_for_update(synopsis_key(manifest.logical_index_id(), target))
                        .await?,
                )?
                .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
                entry.insert(SynopsisExpansion {
                    synopsis: stored,
                    changed: false,
                })
            }
        };
        expansion.changed |= expansion.synopsis.expand(manifest, projection)?;
        Ok(())
    }

    /// Queues every accumulated Header and each changed Synopsis into the
    /// caller's builder, so a whole batch lands in one backend write call.
    pub(crate) fn flush(
        self,
        index: LogicalIndexId,
        deferred: &mut MutationBuilder<'_>,
    ) -> Result<()> {
        for ((tree_key, partition), header) in self.headers {
            deferred.put(
                LogicalKey::Header {
                    index,
                    tree_key,
                    partition,
                },
                PersistentValue::PartitionHeader(header),
            )?;
        }
        for ((tree_key, partition), expansion) in self.synopses {
            if expansion.changed {
                deferred.put(
                    LogicalKey::Synopsis {
                        index,
                        tree_key,
                        partition,
                    },
                    PersistentValue::PartitionSynopsis(expansion.synopsis),
                )?;
            }
        }
        Ok(())
    }
}

/// The two sinks one membership operation writes into: the caller's deferred
/// record-group builder and the per-leaf Header/Synopsis accumulator. They
/// travel together because every membership write joins the batch's single
/// backend write call — the record groups queued directly, the accumulated
/// leaf writes at the accumulator's flush.
pub(crate) struct WriteSinks<'a, 'manifest> {
    /// The shared builder record-group writes queue into.
    pub(crate) deferred: &'a mut MutationBuilder<'manifest>,
    /// The per-leaf Header and Synopsis write state.
    pub(crate) leaves: &'a mut LeafAccumulator,
}

/// Inserts one new Vector Record's complete membership at `target`.
///
/// The Record existence check runs first, so an existing Record ID fails with
/// [`ErrorKind::RecordAlreadyExists`] before any other write. The Record
/// Location, the optional Opaque Payload, and the Leaf Entry must not exist;
/// any of them already existing is [`ErrorKind::Corruption`]. All four checks
/// are update-protected and the writes are queued into `deferred`. The target
/// leaf Header must exist at level 1 in a write-accepting state (`Ready`,
/// `Splitting`, or `ReceivingSplit`) and the target Synopsis must exist; a
/// missing or non-write-accepting Header and a missing Synopsis are
/// [`ErrorKind::Corruption`]. The Header's exact count and cache epoch
/// increase by one, and the Synopsis expands with the entry's exact
/// projection.
///
/// `record` and `entry` must carry the same Record ID. On any error the
/// caller must not commit the transaction.
pub async fn insert_record<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    deferred: &mut MutationBuilder<'_>,
    record: &VectorRecord,
    payload: Option<&OpaquePayload>,
    target: &RecordLocation,
    entry: &LeafEntry,
) -> Result<()> {
    let mut leaves = LeafAccumulator::new();
    insert_record_with_header(
        txn,
        &mut WriteSinks {
            deferred: &mut *deferred,
            leaves: &mut leaves,
        },
        record,
        payload,
        target,
        entry,
    )
    .await?;
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();
    leaves.flush(index, deferred)
}

/// Inserts one record and returns the target's final Header for scheduling.
///
/// Follows [`insert_record`], with every write queued into the shared
/// `writes`: Header and Synopsis adjustments accumulate until its flush.
pub(crate) async fn insert_record_with_header<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    writes: &mut WriteSinks<'_, '_>,
    record: &VectorRecord,
    payload: Option<&OpaquePayload>,
    target: &RecordLocation,
    entry: &LeafEntry,
) -> Result<PartitionHeader> {
    let manifest = validated_input(txn, record, entry)?;
    let index = manifest.logical_index_id();
    let id = record.record_id();

    // The update-protected existence check aborts the commit when a concurrent
    // transaction creates the same Record ID, keeping insert-if-absent exact.
    if txn.get_for_update(record_key(index, id)).await?.is_some() {
        return Err(Error::new(ErrorKind::RecordAlreadyExists));
    }
    writes.deferred.put(
        record_key(index, id),
        PersistentValue::VectorRecord(record.clone()),
    )?;
    expect_absent(txn, location_key(index, id)).await?;
    writes.deferred.put(
        location_key(index, id),
        PersistentValue::RecordLocation(target.clone()),
    )?;
    if let Some(payload) = payload {
        let key = payload_key(index, id);
        expect_absent(txn, key.clone()).await?;
        writes
            .deferred
            .put(key, PersistentValue::OpaquePayload(payload.clone()))?;
    }
    let header = writes.leaves.header(txn, index, target).await?;
    let adjusted = added_entry(expect_write_target(header)?)?;
    writes.leaves.adjust_header(target, adjusted);
    writes
        .leaves
        .expand_synopsis(txn, manifest, target, entry.fields())
        .await?;
    let entry_key = entry_key(index, target, id);
    expect_absent(txn, entry_key.clone()).await?;
    writes
        .deferred
        .put(entry_key, PersistentValue::LeafEntry(entry.clone()))?;
    Ok(adjusted)
}

/// Replaces one existing Vector Record's complete membership, last-write-wins.
///
/// The caller routes insert versus replace from a same-snapshot read and
/// carries the record's exact current location in `expected`; the stored
/// Record must exist with the same Record ID and the stored Record Location
/// must equal `expected`, or the mutation fails closed with
/// [`ErrorKind::Corruption`]. The new Record body replaces the old one, the
/// Location moves to `target` when it differs, and the Opaque Payload follows
/// upsert semantics: `Some` replaces the old payload while `None` deletes it.
///
/// A same-leaf replacement (`expected == target`) requires the stored Leaf
/// Entry to exist, rewrites it, and bumps the leaf cache epoch with its exact
/// count unchanged. A cross-leaf replacement — including a cross-tree move,
/// since keys embed the Tree Key — deletes the source Leaf Entry (which must
/// exist), inserts the target Leaf Entry, decrements the source Header
/// count (which must be positive), and increments the target Header count;
/// only the cache epochs move by one on both sides. A source accepts the
/// move-out in any state because it follows the exact stored location; a
/// target must satisfy the same leaf validation as an insert. Synopses only
/// ever expand: the target Synopsis expands with the entry's exact projection
/// and the source Synopsis is left untouched.
///
/// `record` and `entry` must carry the same Record ID. Record-group writes are
/// queued into `deferred`; Header and Synopsis adjustments accumulate and join
/// the queued writes. On any error the caller must not commit the
/// transaction.
pub async fn replace_record<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    deferred: &mut MutationBuilder<'_>,
    record: &VectorRecord,
    payload: Option<&OpaquePayload>,
    expected: &RecordLocation,
    target: &RecordLocation,
    entry: &LeafEntry,
) -> Result<()> {
    let mut leaves = LeafAccumulator::new();
    replace_record_with_headers(
        txn,
        &mut WriteSinks {
            deferred: &mut *deferred,
            leaves: &mut leaves,
        },
        record,
        payload,
        expected,
        target,
        entry,
    )
    .await?;
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();
    leaves.flush(index, deferred)
}

/// Replaces one record and returns final changed Headers for scheduling.
///
/// Follows [`replace_record`], with every write queued into the shared
/// `writes`: Header and Synopsis adjustments accumulate until its flush.
pub(crate) async fn replace_record_with_headers<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    writes: &mut WriteSinks<'_, '_>,
    record: &VectorRecord,
    payload: Option<&OpaquePayload>,
    expected: &RecordLocation,
    target: &RecordLocation,
    entry: &LeafEntry,
) -> Result<ReplacementHeaders> {
    let manifest = validated_input(txn, record, entry)?;
    let index = manifest.logical_index_id();
    let id = record.record_id();

    let record_key = record_key(index, id);
    let location_key = location_key(index, id);
    let existing = expect_record(txn.get_for_update(record_key.clone()).await?)?
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    if existing.record_id() != id {
        return Err(Error::new(ErrorKind::Corruption));
    }
    let location = expect_location(txn.get_for_update(location_key.clone()).await?)?
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    if &location != expected {
        return Err(Error::new(ErrorKind::Corruption));
    }

    writes
        .deferred
        .put(record_key, PersistentValue::VectorRecord(record.clone()))?;
    if target != expected {
        writes.deferred.put(
            location_key,
            PersistentValue::RecordLocation(target.clone()),
        )?;
    }
    let payload_key = payload_key(index, id);
    match payload {
        Some(payload) => writes
            .deferred
            .put(payload_key, PersistentValue::OpaquePayload(payload.clone()))?,
        None => writes.deferred.delete(payload_key)?,
    }

    let target_entry_key = entry_key(index, target, id);
    let source = if expected == target {
        // The replaced entry must exist; the update-protected read also
        // establishes the conflict on it.
        expect_leaf_entry(txn.get_for_update(target_entry_key.clone()).await?)?
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        writes
            .deferred
            .put(target_entry_key, PersistentValue::LeafEntry(entry.clone()))?;
        None
    } else {
        let source_entry_key = entry_key(index, expected, id);
        expect_leaf_entry(txn.get_for_update(source_entry_key.clone()).await?)?
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        writes.deferred.delete(source_entry_key)?;
        expect_absent(txn, target_entry_key.clone()).await?;
        writes
            .deferred
            .put(target_entry_key, PersistentValue::LeafEntry(entry.clone()))?;

        // The source follows the exact stored location, so any state is legal
        // for it; only the target must be a write-accepting leaf.
        let source = writes.leaves.header(txn, index, expected).await?;
        let adjusted = removed_entry(source)?;
        writes.leaves.adjust_header(expected, adjusted);
        Some(adjusted)
    };

    let target_header = expect_write_target(writes.leaves.header(txn, index, target).await?)?;
    let adjusted = if expected == target {
        touched_entry(target_header)?
    } else {
        added_entry(target_header)?
    };
    writes.leaves.adjust_header(target, adjusted);
    writes
        .leaves
        .expand_synopsis(txn, manifest, target, entry.fields())
        .await?;
    Ok(ReplacementHeaders {
        source,
        target: adjusted,
    })
}

/// Deletes one Vector Record's complete membership, idempotently.
///
/// Delete never routes: the exact tree and leaf come from the stored Record
/// Location. An absent Vector Record returns [`DeleteOutcome::NotFound`] and
/// touches nothing else; otherwise the Location and the Leaf Entry at the
/// stored location must exist or the mutation fails closed with
/// [`ErrorKind::Corruption`]. The Record, Location, Leaf Entry, and any
/// Opaque Payload deletes are queued into `deferred`, the leaf Header count
/// decrements (and must be positive) with its cache epoch bumped, and the
/// Synopsis stays untouched because synopses never shrink. On any error the
/// caller must not commit the transaction.
pub async fn delete_record<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    deferred: &mut MutationBuilder<'_>,
    id: &Bytes,
) -> Result<DeleteOutcome> {
    let mut leaves = LeafAccumulator::new();
    let report = delete_record_with_header(
        txn,
        &mut WriteSinks {
            deferred: &mut *deferred,
            leaves: &mut leaves,
        },
        id,
    )
    .await?;
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();
    leaves.flush(index, deferred)?;
    Ok(match report {
        DeleteReport::Deleted { location, .. } => DeleteOutcome::Deleted { location },
        DeleteReport::NotFound => DeleteOutcome::NotFound,
    })
}

/// Deletes one record and returns the changed Header for maintenance discovery.
///
/// Follows [`delete_record`], with every write queued into the shared
/// `writes`: the Header adjustment accumulates until its flush.
pub(crate) async fn delete_record_with_header<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    writes: &mut WriteSinks<'_, '_>,
    id: &Bytes,
) -> Result<DeleteReport> {
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();

    let record_key = record_key(index, id);
    if expect_record(txn.get_for_update(record_key.clone()).await?)?.is_none() {
        return Ok(DeleteReport::NotFound);
    }
    let location_key = location_key(index, id);
    let location = expect_location(txn.get_for_update(location_key.clone()).await?)?
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    let entry_key = entry_key(index, &location, id);
    expect_leaf_entry(txn.get_for_update(entry_key.clone()).await?)?
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;

    writes.deferred.delete(record_key)?;
    writes.deferred.delete(location_key)?;
    writes.deferred.delete(entry_key)?;
    writes.deferred.delete(payload_key(index, id))?;
    let header = removed_entry(writes.leaves.header(txn, index, &location).await?)?;
    writes.leaves.adjust_header(&location, header);
    Ok(DeleteReport::Deleted { location, header })
}

/// Reads the authoritative Record Locations of one batch of Record IDs with
/// update protection, in input order.
///
/// One batched backend read establishes every conflict the upsert routing
/// decision and the replacement's exact expected location depend on. A fully
/// absent Record ID returns `None` at its position; a half-present
/// Record/Location pair is [`ErrorKind::Corruption`] at that position.
pub async fn read_locations_for_update<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    ids: &[Bytes],
) -> Result<Vec<Option<RecordLocation>>> {
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();
    let mut keys = Vec::with_capacity(ids.len().saturating_mul(2));
    for id in ids {
        keys.push(record_key(index, id));
        keys.push(location_key(index, id));
    }
    let mut values = txn.batch_get_for_update(keys).await?.into_iter();
    let mut locations = Vec::with_capacity(ids.len());
    for (position, _) in ids.iter().enumerate() {
        let pair = match (values.next(), values.next()) {
            (Some(record), Some(location)) => classify_location_pair(record, location)
                .map_err(|error| error.at_position(position))?,
            // The typed batch read returns exactly one value per input key.
            _ => return Err(Error::new(ErrorKind::Backend)),
        };
        locations.push(pair);
    }
    Ok(locations)
}

/// Validates one Record/Location pair, failing closed on a wrong value family
/// or a half-present pair.
fn classify_location_pair(
    record: Option<PersistentValue>,
    location: Option<PersistentValue>,
) -> Result<Option<RecordLocation>> {
    match (expect_record(record)?, expect_location(location)?) {
        (None, None) => Ok(None),
        (Some(_), Some(location)) => Ok(Some(location)),
        _ => Err(Error::new(ErrorKind::Corruption)),
    }
}

/// One batch item's membership read set to warm before the apply loop.
///
/// Each set mirrors exactly what the corresponding membership operation —
/// [`insert_record_with_header`], [`replace_record_with_headers`], or
/// [`delete_record_with_header`] — reads with update protection.
pub(crate) enum MembershipPrefetch<'a> {
    /// An insert's existence checks: the Record, the Location, the Opaque
    /// Payload when the record carries one, and the Leaf Entry at the routed
    /// target.
    Insert {
        id: &'a Bytes,
        payload: bool,
        target: &'a RecordLocation,
    },
    /// A replacement's reads: the stored Record and Location, the target Leaf
    /// Entry, and the source Leaf Entry when the move crosses leaves.
    Replace {
        id: &'a Bytes,
        expected: &'a RecordLocation,
        target: &'a RecordLocation,
    },
    /// A delete's Record and Location. The Leaf Entry key follows from the
    /// stored Location, so it warms in a second wave.
    Delete { id: &'a Bytes },
}

/// The chunk bound for batched membership warming reads. One chunk matches
/// the production adapters' internal point-read batch, so a logical batch
/// never re-chunks below the adapter, and stays well under every backend's
/// batch limit regardless of the caller's batch size.
const MEMBERSHIP_READ_CHUNK: usize = 1_024;

/// Warms the transaction-local read cache with every item's membership read
/// set in bounded update-protected batches.
///
/// The per-item membership operations re-read exactly these keys through the
/// checked typed path when they apply; served from the warmed cache, they
/// cost no backend round trip per record. The batched reads establish the
/// per-key conflicts the per-item reads rely on; a delete of an absent record
/// additionally protects its (necessarily absent) Location, which exact
/// membership only ever writes together with the already-protected Record.
///
/// This is an optimization only: it validates nothing and never fails. A key
/// left unwarmed by a failed batch — raw values fetched before the failure
/// stay cached — is re-read by the item's own checked path, which reproduces
/// the error with the item's input position attached.
pub(crate) async fn prefetch_membership_for_update<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    items: &[MembershipPrefetch<'_>],
) {
    let Some(index) = txn.bound_manifest().map(IndexManifest::logical_index_id) else {
        return;
    };
    let mut keys = Vec::new();
    for item in items {
        match *item {
            MembershipPrefetch::Insert {
                id,
                payload,
                target,
            } => {
                keys.push(record_key(index, id));
                keys.push(location_key(index, id));
                if payload {
                    keys.push(payload_key(index, id));
                }
                keys.push(entry_key(index, target, id));
            }
            MembershipPrefetch::Replace {
                id,
                expected,
                target,
            } => {
                keys.push(record_key(index, id));
                keys.push(location_key(index, id));
                keys.push(entry_key(index, target, id));
                if expected != target {
                    keys.push(entry_key(index, expected, id));
                }
            }
            MembershipPrefetch::Delete { id } => {
                keys.push(record_key(index, id));
                keys.push(location_key(index, id));
            }
        }
    }
    if !read_chunks_for_update(txn, keys).await {
        return;
    }
    // A delete's Leaf Entry key follows from its stored Location, so it warms
    // in a second wave; the Record and Location re-reads here are cache hits
    // from the first wave. An absent record or a half-present pair defers to
    // the checked delete path, which reports it with the item's position.
    let mut entry_keys = Vec::new();
    for item in items {
        let &MembershipPrefetch::Delete { id } = item else {
            continue;
        };
        let record = txn.get_for_update(record_key(index, id)).await;
        let location = txn.get_for_update(location_key(index, id)).await;
        let (Ok(record), Ok(location)) = (record, location) else {
            continue;
        };
        let (Ok(Some(_)), Ok(Some(location))) = (expect_record(record), expect_location(location))
        else {
            continue;
        };
        entry_keys.push(entry_key(index, &location, id));
    }
    read_chunks_for_update(txn, entry_keys).await;
}

/// Reads `keys` in bounded update-protected batches, returning false when a
/// batch fails and the remaining keys are left unwarmed.
async fn read_chunks_for_update<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    keys: Vec<LogicalKey>,
) -> bool {
    for chunk in keys.chunks(MEMBERSHIP_READ_CHUNK) {
        if txn.batch_get_for_update(chunk.to_vec()).await.is_err() {
            return false;
        }
    }
    true
}

/// Validates caller identity agreement and returns the bound Manifest.
fn validated_input<'manifest, T>(
    txn: &WriteLogicalTxn<'manifest, T>,
    record: &VectorRecord,
    entry: &LeafEntry,
) -> Result<&'manifest IndexManifest> {
    if record.record_id() != entry.record_id() {
        return Err(Error::invalid_argument());
    }
    txn.bound_manifest().ok_or_else(Error::invalid_argument)
}

/// Reads one update-protected leaf Header that must exist.
async fn read_header<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: LogicalIndexId,
    location: &RecordLocation,
) -> Result<PartitionHeader> {
    expect_header(txn.get_for_update(header_key(index, location)).await?)?
        .ok_or_else(|| Error::new(ErrorKind::Corruption))
}

/// Validates that a decoded Header names a write-accepting leaf.
fn expect_write_target(header: PartitionHeader) -> Result<PartitionHeader> {
    if header.level() == 1 && header.state().accepts_writes() {
        Ok(header)
    } else {
        Err(Error::new(ErrorKind::Corruption))
    }
}

/// Returns `header` with its exact count increased by one and epoch bumped.
pub(crate) fn added_entry(header: PartitionHeader) -> Result<PartitionHeader> {
    adjust_header(
        header,
        header
            .entry_count()
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?,
    )
}

/// Returns `header` with its exact count decreased by one and epoch bumped.
///
/// A zero count on decrement is an impossible count and fails closed.
pub(crate) fn removed_entry(header: PartitionHeader) -> Result<PartitionHeader> {
    adjust_header(
        header,
        header
            .entry_count()
            .checked_sub(1)
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?,
    )
}

/// Returns `header` with its count unchanged and epoch bumped.
fn touched_entry(header: PartitionHeader) -> Result<PartitionHeader> {
    adjust_header(header, header.entry_count())
}

/// Rebuilds `header` with `entry_count` and the next cache epoch.
///
/// The input Header was structurally valid at decode, so a construction or
/// epoch-overflow failure here is impossible arithmetic and fails closed.
fn adjust_header(header: PartitionHeader, entry_count: u32) -> Result<PartitionHeader> {
    let cache_epoch = header
        .cache_epoch()
        .checked_add(1)
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    PartitionHeader::new(header.level(), entry_count, cache_epoch, header.state())
        .map_err(|_| Error::new(ErrorKind::Corruption))
}

/// Asserts a record-group key is absent, failing closed on corruption.
///
/// The update-protected read establishes the conflict that keeps the queued
/// write exact: a concurrent commit creating the key aborts this transaction.
async fn expect_absent<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    key: LogicalKey,
) -> Result<()> {
    if txn.get_for_update(key).await?.is_some() {
        return Err(Error::new(ErrorKind::Corruption));
    }
    Ok(())
}

/// Maps a duplicate unique insert of authoritative state to Corruption.
pub(crate) fn expect_inserted(outcome: InsertOutcome) -> Result<()> {
    match outcome {
        InsertOutcome::Inserted => Ok(()),
        InsertOutcome::AlreadyExists => Err(Error::new(ErrorKind::Corruption)),
    }
}

fn record_key(index: LogicalIndexId, id: &Bytes) -> LogicalKey {
    LogicalKey::Record {
        index,
        id: id.clone(),
    }
}

fn location_key(index: LogicalIndexId, id: &Bytes) -> LogicalKey {
    LogicalKey::Location {
        index,
        id: id.clone(),
    }
}

fn payload_key(index: LogicalIndexId, id: &Bytes) -> LogicalKey {
    LogicalKey::Payload {
        index,
        id: id.clone(),
    }
}

fn header_key(index: LogicalIndexId, location: &RecordLocation) -> LogicalKey {
    LogicalKey::Header {
        index,
        tree_key: location.tree_key().clone(),
        partition: location.leaf(),
    }
}

fn synopsis_key(index: LogicalIndexId, location: &RecordLocation) -> LogicalKey {
    LogicalKey::Synopsis {
        index,
        tree_key: location.tree_key().clone(),
        partition: location.leaf(),
    }
}

fn entry_key(index: LogicalIndexId, location: &RecordLocation, id: &Bytes) -> LogicalKey {
    LogicalKey::LeafEntry {
        index,
        tree_key: location.tree_key().clone(),
        partition: location.leaf(),
        id: id.clone(),
    }
}
