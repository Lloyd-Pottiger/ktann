//! Runtime-owned validated reads: point and batch Vector Record reads, plus
//! the shared Structure Maintenance authority preflight.
//!
//! `get` and `batch_get` validate the persisted Index Manifest and read every
//! requested Record Group from one consistent backend snapshot. Manifest
//! validation fails closed: a missing Manifest is `IndexNotFound`, a Dropping
//! Manifest is `IndexDropping`, and a Manifest whose immutable identity does
//! not match the opened handle is `Corruption`. The storage layer's typed
//! Record Group read detects a partial Record/Location pair or a dangling
//! Opaque Payload as `Corruption`.

use std::sync::Arc;

use bytes::Bytes;

use crate::api::{Error, ErrorKind, PartitionKey, PayloadProjection, Result, StoredRecord};
use crate::storage::backend::{Backend, ReadOps, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    IndexLifecycle, IndexManifest, PartitionHeader, PartitionTransition, PersistentValue,
};
use crate::storage::{ReadLogicalTxn, RecordGroupRead, WriteLogicalTxn, topology};

use super::OperationContext;

/// Reads one Vector Record as a public projection.
pub(crate) async fn get_record<B: Backend>(
    context: &mut OperationContext<B>,
    handle_manifest: &IndexManifest,
    id: Bytes,
    include_payload: bool,
) -> Result<Option<StoredRecord>> {
    context.checkpoint()?;
    let backend = context.backend();
    let mut txn = open_validated_read(backend.as_ref(), handle_manifest).await?;
    let group = txn.read_record_group(id, include_payload).await?;
    context.checkpoint()?;
    Ok(group.map(|group| stored_record(include_payload, group)))
}

/// Reads Vector Records in input order while preserving duplicates and gaps.
pub(crate) async fn batch_get_records<B: Backend>(
    context: &mut OperationContext<B>,
    handle_manifest: &IndexManifest,
    ids: Vec<Bytes>,
    include_payload: bool,
) -> Result<Vec<Option<StoredRecord>>> {
    context.checkpoint()?;
    let backend = context.backend();
    let mut txn = open_validated_read(backend.as_ref(), handle_manifest).await?;
    let groups = txn.read_record_groups(ids, include_payload).await?;
    context.checkpoint()?;
    Ok(groups
        .into_iter()
        .map(|group| group.map(|group| stored_record(include_payload, group)))
        .collect())
}

/// Validates the persisted Manifest of the opened handle in one snapshot.
pub(crate) async fn validate_manifest<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    handle: &IndexManifest,
) -> Result<IndexManifest> {
    opened_manifest(
        txn.get(LogicalKey::Manifest(handle.logical_index_id()))
            .await?,
        handle,
    )
}

/// Opens one read transaction bound to the Logical Index, validating the
/// persisted Active Manifest of the opened handle first.
///
/// A dropped Logical Index reports `IndexNotFound`/`IndexDropping` instead of
/// a misleading Corruption from missing data keys. Validation proves the
/// persisted Manifest carries the handle's exact immutable identity, so
/// binding the handle manifest is equivalent to binding the persisted one.
pub(crate) async fn open_validated_read<'b, 'm, B: Backend>(
    backend: &'b B,
    handle_manifest: &'m IndexManifest,
) -> Result<ReadLogicalTxn<'m, B::ReadTxn<'b>>> {
    let raw = backend.begin_read().await?;
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    validate_manifest(&mut txn, handle_manifest).await?;
    ReadLogicalTxn::for_index(txn.into_raw(), handle_manifest)
}

/// Opens one validated read snapshot and reads one partition's authority
/// pair from it: the shared Structure Maintenance preflight.
///
/// The Manifest validation of [`open_validated_read`] runs first, then one
/// batched plain read covers both authority values, all in the same
/// consistent snapshot. The transaction stays open so the caller can fix more
/// of its step — a drain batch, the same-level candidates — from that
/// snapshot; a caller that needs only the pair drops it immediately.
pub(crate) async fn open_authority_read<'b, 'm, B: Backend>(
    backend: &'b B,
    handle_manifest: &'m IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<(
    ReadLogicalTxn<'m, B::ReadTxn<'b>>,
    Option<(PartitionHeader, PartitionTransition)>,
)> {
    let mut txn = open_validated_read(backend, handle_manifest).await?;
    let pair = topology::read_authority_pair(
        &mut txn,
        handle_manifest.logical_index_id(),
        tree_key,
        partition,
    )
    .await?;
    Ok((txn, pair))
}

/// Validates the persisted Manifest of the opened handle, with update
/// protection on the Manifest key.
///
/// The conflict aborts the transaction if a concurrent drop transition
/// commits, so neither a Foreground Mutation nor a Structure Maintenance step
/// commits into a Logical Index whose deletion has begun.
pub(crate) async fn validated_active_manifest<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    handle: &IndexManifest,
) -> Result<IndexManifest> {
    opened_manifest(
        txn.get_for_update(LogicalKey::Manifest(handle.logical_index_id()))
            .await?,
        handle,
    )
}

/// Classifies the persisted Manifest of the opened handle in one snapshot.
///
/// The handle never retargets to a newer Logical Index, so the persisted
/// Manifest must be Active and carry the handle's exact immutable identity.
/// A supported but Dropping Manifest fails with `IndexDropping`; an absent
/// Manifest means the Logical Index no longer exists.
pub(crate) fn opened_manifest(
    value: Option<PersistentValue>,
    handle: &IndexManifest,
) -> Result<IndexManifest> {
    match value {
        Some(PersistentValue::IndexManifest(current)) => match current.lifecycle() {
            IndexLifecycle::Active if current.has_same_immutable_identity(handle) => Ok(current),
            IndexLifecycle::Active => Err(Error::new(ErrorKind::Corruption)),
            IndexLifecycle::Dropping => Err(Error::new(ErrorKind::IndexDropping)),
        },
        Some(_) => Err(Error::new(ErrorKind::Corruption)),
        None => Err(Error::new(ErrorKind::IndexNotFound)),
    }
}

/// Projects one validated Record Group into the public point-read shape.
///
/// The Record Location is validated but never exposed. The Opaque Payload
/// projection is closed by the request: `NotLoaded` when the payload was not
/// requested, `Absent` for a requested but missing payload, and `Present` for
/// the decoded bytes.
fn stored_record(include_payload: bool, group: RecordGroupRead) -> StoredRecord {
    let (record, _location, payload) = group.into_parts();
    let (id, vector, fields) = record.into_parts();
    let payload = match (include_payload, payload) {
        (true, Some(payload)) => PayloadProjection::Present(payload.into_bytes()),
        (true, None) => PayloadProjection::Absent,
        (false, _) => PayloadProjection::NotLoaded,
    };
    StoredRecord::new(id, Arc::from(vector), fields, payload)
}
