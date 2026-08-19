//! Runtime-owned point and batch Vector Record reads.
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

use crate::api::{Error, ErrorKind, PayloadProjection, Result, StoredRecord};
use crate::storage::backend::{Backend, ReadOps};
use crate::storage::keys::LogicalKey;
use crate::storage::values::{IndexLifecycle, IndexManifest, PersistentValue};
use crate::storage::{ReadLogicalTxn, RecordGroupRead};

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
    let raw = backend.begin_read().await?;
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    let current = validate_manifest(&mut txn, handle_manifest).await?;
    let raw = txn.into_raw();
    let mut txn = ReadLogicalTxn::for_index(raw, &current)?;
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
    let raw = backend.begin_read().await?;
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    let current = validate_manifest(&mut txn, handle_manifest).await?;
    let raw = txn.into_raw();
    let mut txn = ReadLogicalTxn::for_index(raw, &current)?;
    let groups = txn.read_record_groups(ids, include_payload).await?;
    context.checkpoint()?;
    Ok(groups
        .into_iter()
        .map(|group| group.map(|group| stored_record(include_payload, group)))
        .collect())
}

/// Validates the persisted Manifest of the opened handle in one snapshot.
///
/// The handle never retargets to a newer Logical Index, so the persisted
/// Manifest must be Active and carry the handle's exact immutable identity.
/// A supported but Dropping Manifest fails with `IndexDropping`; an absent
/// Manifest means the Logical Index no longer exists.
async fn validate_manifest<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    handle: &IndexManifest,
) -> Result<IndexManifest> {
    match txn
        .get(LogicalKey::Manifest(handle.logical_index_id()))
        .await?
    {
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
