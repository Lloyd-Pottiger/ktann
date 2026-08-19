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
//! Every authoritative read a mutation depends on is update-protected, so a
//! concurrent change to the same membership or leaf Header aborts the commit
//! with [`ErrorKind::RetryableAbort`] instead of producing a partial write. A
//! missing, extra, or mismatched authoritative value is
//! [`ErrorKind::Corruption`] and fails closed without repair. On any returned
//! error the caller must not commit the transaction; rolling back leaves no
//! partial change.

use bytes::Bytes;

use crate::api::{Error, ErrorKind, LogicalIndexId, Result, Value};
use crate::storage::WriteLogicalTxn;
use crate::storage::backend::{InsertOutcome, WriteTxn};
use crate::storage::keys::LogicalKey;
use crate::storage::values::{
    IndexManifest, LeafEntry, OpaquePayload, PartitionHeader, PartitionState, PartitionSynopsis,
    PersistentValue, RecordLocation, VectorRecord,
};

/// The outcome of an idempotent delete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DeleteOutcome {
    /// The record existed and its whole membership group was deleted.
    Deleted,
    /// No Vector Record with the Record ID exists; nothing was touched.
    NotFound,
}

/// Inserts one new Vector Record's complete membership at `target`.
///
/// The Record unique insert runs first, so an existing Record ID fails with
/// [`ErrorKind::RecordAlreadyExists`] before any other write. The Record
/// Location, the optional Opaque Payload, and the Leaf Entry are unique
/// inserts; any of them already existing is [`ErrorKind::Corruption`]. The
/// target leaf Header must exist at level 1 in a write-accepting state
/// (`Ready`, `Splitting`, or `ReceivingSplit`) and the target Synopsis must
/// exist; a missing or non-write-accepting Header and a missing Synopsis are
/// [`ErrorKind::Corruption`]. The Header's exact count and cache epoch
/// increase by one, and the Synopsis expands with the entry's exact
/// projection.
///
/// `record` and `entry` must carry the same Record ID. On any error the
/// caller must not commit the transaction.
pub async fn insert_record<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    record: &VectorRecord,
    payload: Option<&OpaquePayload>,
    target: &RecordLocation,
    entry: &LeafEntry,
) -> Result<()> {
    let manifest = validated_input(txn, record, entry)?;
    let index = manifest.logical_index_id();
    let id = record.record_id();

    let outcome = txn
        .insert(
            record_key(index, id),
            PersistentValue::VectorRecord(record.clone()),
        )
        .await?;
    if outcome != InsertOutcome::Inserted {
        return Err(Error::new(ErrorKind::RecordAlreadyExists));
    }
    expect_inserted(
        txn.insert(
            location_key(index, id),
            PersistentValue::RecordLocation(target.clone()),
        )
        .await?,
    )?;
    if let Some(payload) = payload {
        expect_inserted(
            txn.insert(
                payload_key(index, id),
                PersistentValue::OpaquePayload(payload.clone()),
            )
            .await?,
        )?;
    }
    let header = read_header(txn, index, target).await?;
    put_header(
        txn,
        index,
        target,
        added_entry(expect_write_target(header)?)?,
    )
    .await?;
    expand_synopsis(txn, manifest, target, entry.fields()).await?;
    expect_inserted(
        txn.insert(
            entry_key(index, target, id),
            PersistentValue::LeafEntry(entry.clone()),
        )
        .await?,
    )?;
    Ok(())
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
/// exist), unique-inserts the target Leaf Entry, decrements the source Header
/// count (which must be positive), and increments the target Header count;
/// only the cache epochs move by one on both sides. A source accepts the
/// move-out in any state because it follows the exact stored location; a
/// target must satisfy the same leaf validation as an insert. Synopses only
/// ever expand: the target Synopsis expands with the entry's exact projection
/// and the source Synopsis is left untouched.
///
/// `record` and `entry` must carry the same Record ID. On any error the
/// caller must not commit the transaction.
pub async fn replace_record<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    record: &VectorRecord,
    payload: Option<&OpaquePayload>,
    expected: &RecordLocation,
    target: &RecordLocation,
    entry: &LeafEntry,
) -> Result<()> {
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

    txn.put(record_key, PersistentValue::VectorRecord(record.clone()))
        .await?;
    if target != expected {
        txn.put(
            location_key,
            PersistentValue::RecordLocation(target.clone()),
        )
        .await?;
    }
    match payload {
        Some(payload) => {
            txn.put(
                payload_key(index, id),
                PersistentValue::OpaquePayload(payload.clone()),
            )
            .await?;
        }
        None => txn.delete(payload_key(index, id)).await?,
    }

    let target_entry_key = entry_key(index, target, id);
    if expected == target {
        // The replaced entry must exist; the update-protected read also
        // establishes the conflict on it.
        expect_entry(txn.get_for_update(target_entry_key.clone()).await?)?
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        txn.put(target_entry_key, PersistentValue::LeafEntry(entry.clone()))
            .await?;
    } else {
        let source_entry_key = entry_key(index, expected, id);
        expect_entry(txn.get_for_update(source_entry_key.clone()).await?)?
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        txn.delete(source_entry_key).await?;
        expect_inserted(
            txn.insert(target_entry_key, PersistentValue::LeafEntry(entry.clone()))
                .await?,
        )?;

        // The source follows the exact stored location, so any state is legal
        // for it; only the target must be a write-accepting leaf.
        let source = read_header(txn, index, expected).await?;
        put_header(txn, index, expected, removed_entry(source)?).await?;
    }

    let target_header = expect_write_target(read_header(txn, index, target).await?)?;
    let adjusted = if expected == target {
        touched_entry(target_header)?
    } else {
        added_entry(target_header)?
    };
    put_header(txn, index, target, adjusted).await?;
    expand_synopsis(txn, manifest, target, entry.fields()).await?;
    Ok(())
}

/// Deletes one Vector Record's complete membership, idempotently.
///
/// Delete never routes: the exact tree and leaf come from the stored Record
/// Location. An absent Vector Record returns [`DeleteOutcome::NotFound`] and
/// touches nothing else; otherwise the Location and the Leaf Entry at the
/// stored location must exist or the mutation fails closed with
/// [`ErrorKind::Corruption`]. The Record, Location, Leaf Entry, and any
/// Opaque Payload are deleted, the leaf Header count decrements (and must be
/// positive) with its cache epoch bumped, and the Synopsis stays untouched
/// because synopses never shrink. On any error the caller must not commit the
/// transaction.
pub async fn delete_record<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    id: &Bytes,
) -> Result<DeleteOutcome> {
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();

    let record_key = record_key(index, id);
    if expect_record(txn.get_for_update(record_key.clone()).await?)?.is_none() {
        return Ok(DeleteOutcome::NotFound);
    }
    let location_key = location_key(index, id);
    let location = expect_location(txn.get_for_update(location_key.clone()).await?)?
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    let entry_key = entry_key(index, &location, id);
    expect_entry(txn.get_for_update(entry_key.clone()).await?)?
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;

    txn.delete(record_key).await?;
    txn.delete(location_key).await?;
    txn.delete(entry_key).await?;
    txn.delete(payload_key(index, id)).await?;
    let header = read_header(txn, index, &location).await?;
    put_header(txn, index, &location, removed_entry(header)?).await?;
    Ok(DeleteOutcome::Deleted)
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

/// Writes one adjusted leaf Header back through the budgeted mutation path.
async fn put_header<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: LogicalIndexId,
    location: &RecordLocation,
    header: PartitionHeader,
) -> Result<()> {
    txn.put(
        header_key(index, location),
        PersistentValue::PartitionHeader(header),
    )
    .await
}

/// Expands the target Synopsis with one exact Leaf projection and puts it back
/// only when the expansion changed it.
///
/// The stored Synopsis was fully validated at decode, so a projection mismatch
/// here is caller input; a missing Synopsis is corruption. The update-protected
/// read already establishes the commit-time conflict, so a byte-identical
/// expansion is left unwritten.
async fn expand_synopsis<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    target: &RecordLocation,
    projection: &[Value],
) -> Result<()> {
    let key = synopsis_key(manifest.logical_index_id(), target);
    let mut synopsis = expect_synopsis(txn.get_for_update(key.clone()).await?)?
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    let before = synopsis.clone();
    synopsis.expand(manifest, projection)?;
    if synopsis != before {
        txn.put(key, PersistentValue::PartitionSynopsis(synopsis))
            .await?;
    }
    Ok(())
}

/// Validates that a decoded Header names a write-accepting leaf.
fn expect_write_target(header: PartitionHeader) -> Result<PartitionHeader> {
    match header.state() {
        PartitionState::Ready | PartitionState::Splitting | PartitionState::ReceivingSplit
            if header.level() == 1 =>
        {
            Ok(header)
        }
        _ => Err(Error::new(ErrorKind::Corruption)),
    }
}

/// Returns `header` with its exact count increased by one and epoch bumped.
fn added_entry(header: PartitionHeader) -> Result<PartitionHeader> {
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
fn removed_entry(header: PartitionHeader) -> Result<PartitionHeader> {
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

/// Maps a duplicate unique insert of authoritative state to Corruption.
fn expect_inserted(outcome: InsertOutcome) -> Result<()> {
    match outcome {
        InsertOutcome::Inserted => Ok(()),
        InsertOutcome::AlreadyExists => Err(Error::new(ErrorKind::Corruption)),
    }
}

/// Extracts the Vector Record from a typed read, failing closed on a
/// wrong-kind value.
fn expect_record(value: Option<PersistentValue>) -> Result<Option<VectorRecord>> {
    match value {
        Some(PersistentValue::VectorRecord(record)) => Ok(Some(record)),
        Some(_) => Err(Error::new(ErrorKind::Corruption)),
        None => Ok(None),
    }
}

/// Extracts the Record Location from a typed read, failing closed on a
/// wrong-kind value.
fn expect_location(value: Option<PersistentValue>) -> Result<Option<RecordLocation>> {
    match value {
        Some(PersistentValue::RecordLocation(location)) => Ok(Some(location)),
        Some(_) => Err(Error::new(ErrorKind::Corruption)),
        None => Ok(None),
    }
}

/// Extracts the Partition Header from a typed read, failing closed on a
/// wrong-kind value.
fn expect_header(value: Option<PersistentValue>) -> Result<Option<PartitionHeader>> {
    match value {
        Some(PersistentValue::PartitionHeader(header)) => Ok(Some(header)),
        Some(_) => Err(Error::new(ErrorKind::Corruption)),
        None => Ok(None),
    }
}

/// Extracts the Partition Synopsis from a typed read, failing closed on a
/// wrong-kind value.
fn expect_synopsis(value: Option<PersistentValue>) -> Result<Option<PartitionSynopsis>> {
    match value {
        Some(PersistentValue::PartitionSynopsis(synopsis)) => Ok(Some(synopsis)),
        Some(_) => Err(Error::new(ErrorKind::Corruption)),
        None => Ok(None),
    }
}

/// Extracts the Leaf Entry from a typed read, failing closed on a wrong-kind
/// value.
fn expect_entry(value: Option<PersistentValue>) -> Result<Option<LeafEntry>> {
    match value {
        Some(PersistentValue::LeafEntry(entry)) => Ok(Some(entry)),
        Some(_) => Err(Error::new(ErrorKind::Corruption)),
        None => Ok(None),
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
