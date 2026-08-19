//! Foreground mutation routing and whole-operation retry orchestration.
//!
//! One mutation operation — a single insert, upsert, or delete, or one atomic
//! batch — is validated in the API layer, then runs here as a sequence of
//! whole attempts (ADR 0012). Each attempt opens a fresh write transaction,
//! update-protects and validates the Active Index Manifest, routes every
//! record through the current searchable topology, and applies the typed
//! atomic membership operations in input order. One commit either installs
//! every mutation or none; there is no record revision, no partial outcome,
//! and no repair branch.
//!
//! - **Routing.** Insert and upsert targets come from
//!   [`routing::route_leaf_for_write_preprocessed`]: the record's Tree Key
//!   selects the tree, which is lazily created on the first write, and the
//!   preprocessed routing vector descends to the nearest leaf. Upsert reads
//!   the stored Record Location with update protection to choose between
//!   insert and replacement; a replacement across Tree Keys or leaves is one
//!   atomic move. Delete never routes and follows the exact stored location.
//! - **Retry.** A definite backend abort anywhere in an attempt discards
//!   every route and replays the complete operation from a fresh snapshot
//!   under the bounded contention policy; exhaustion returns
//!   `ContentionExhausted`. A commit of unknown outcome is never retried and
//!   returns `CommitOutcomeUnknown`, so an older request can never overwrite
//!   a later successful write.
//! - **Errors.** Any item failure fails the whole operation with the input
//!   position attached and no partial outcomes. Cancellation and deadline are
//!   checked before every attempt and before commit; once commit starts, the
//!   real result wins.
//!
//! Offering maintenance after a committed mutation arrives with the Fixup
//! runtime (#10, #31); losing it never affects correctness because every
//! committed topology state remains searchable.

use crate::api::{Error, ErrorKind, Mutation, MutationOutcome, Record, Result};
use crate::runtime::lifecycle::RetryPolicy;
use crate::runtime::{OperationContext, reads};
use crate::search::numeric::VectorKernel;
use crate::search::rabitq::RaBitQ7;
use crate::storage::WriteLogicalTxn;
use crate::storage::backend::{Backend, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::membership;
use crate::storage::values::{
    IndexManifest, LeafEntry, OpaquePayload, RecordLocation, VectorRecord,
};

use super::routing;

/// Runs one validated, non-empty mutation batch as a bounded sequence of
/// whole attempts.
///
/// `mutations` must already be validated against the handle's immutable
/// configuration; the public operations reject or short-circuit empty batches
/// before admission. The returned outcomes correspond to the inputs in order
/// and are produced only after the single atomic commit succeeds; any failure
/// returns one error for the entire operation.
pub(crate) async fn mutate<B: Backend>(
    context: &mut OperationContext<B>,
    handle_manifest: &IndexManifest,
    mutations: &[Mutation],
    retry: RetryPolicy,
) -> Result<Vec<MutationOutcome>> {
    let kernel = routing::kernel_for(handle_manifest)?;
    let prepared = prepare_all(handle_manifest, &kernel, mutations)?;
    let mut failed_attempts = 0_u32;
    loop {
        context.checkpoint()?;
        match run_attempt(context, handle_manifest, &kernel, mutations, &prepared).await {
            Ok(outcomes) => return Ok(outcomes),
            Err(error) if error.kind() == ErrorKind::RetryableAbort => {
                if retry.would_exhaust(failed_attempts) {
                    return Err(Error::new(ErrorKind::ContentionExhausted));
                }
                retry.wait(failed_attempts).await;
                failed_attempts += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Runs one complete attempt in a single fresh write transaction.
async fn run_attempt<B: Backend>(
    context: &mut OperationContext<B>,
    handle_manifest: &IndexManifest,
    kernel: &VectorKernel,
    mutations: &[Mutation],
    prepared: &[Option<PreparedRecord>],
) -> Result<Vec<MutationOutcome>> {
    let backend = context.backend();
    let hard_limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let raw = backend.begin_write().await?;
    let mut txn = WriteLogicalTxn::bootstrap(raw, hard_limits, budget);
    let current = match validate_manifest(&mut txn, handle_manifest).await {
        Ok(current) => current,
        Err(error) => {
            txn.rollback().await;
            return Err(error);
        }
    };
    let raw = txn.into_raw();
    let mut txn = WriteLogicalTxn::for_index(raw, &current, hard_limits, budget)?;
    match apply_all(&mut txn, kernel, mutations, prepared).await {
        Ok(outcomes) => context
            .commit(move |start| txn.commit_with(start))
            .await
            .map(|()| outcomes),
        Err(error) => {
            txn.rollback().await;
            Err(error)
        }
    }
}

/// Validates the persisted Manifest of the opened handle in this transaction.
///
/// The update-protected read both validates Active state and conflicts with a
/// concurrent drop transition, so a mutation never commits into a Logical
/// Index whose deletion has begun.
async fn validate_manifest<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    handle: &IndexManifest,
) -> Result<IndexManifest> {
    reads::opened_manifest(
        txn.get_for_update(LogicalKey::Manifest(handle.logical_index_id()))
            .await?,
        handle,
    )
}

/// Applies every mutation in input order inside the attempt transaction.
///
/// Items route and apply one at a time; batched routing and batched writes
/// are #84.
async fn apply_all<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    kernel: &VectorKernel,
    mutations: &[Mutation],
    prepared: &[Option<PreparedRecord>],
) -> Result<Vec<MutationOutcome>> {
    debug_assert_eq!(mutations.len(), prepared.len());
    let started_at = now_unix_millis();
    let mut outcomes = Vec::with_capacity(mutations.len());
    for (position, (mutation, prepared)) in mutations.iter().zip(prepared).enumerate() {
        let outcome = apply_one(txn, kernel, mutation, prepared.as_ref(), started_at)
            .await
            .map_err(|error| error.at_position(position))?;
        outcomes.push(outcome);
    }
    Ok(outcomes)
}

/// Routes and applies one mutation item inside the attempt transaction.
async fn apply_one<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    kernel: &VectorKernel,
    mutation: &Mutation,
    prepared: Option<&PreparedRecord>,
    started_at: u64,
) -> Result<MutationOutcome> {
    match mutation {
        Mutation::Insert(_) => {
            let prepared = prepared.ok_or_else(|| Error::new(ErrorKind::Backend))?;
            let target = route_target(txn, kernel, prepared, started_at).await?;
            membership::insert_record(
                txn,
                &prepared.record,
                prepared.payload.as_ref(),
                &target,
                &prepared.entry,
            )
            .await?;
            Ok(MutationOutcome::Inserted)
        }
        Mutation::Upsert(record) => {
            let prepared = prepared.ok_or_else(|| Error::new(ErrorKind::Backend))?;
            let target = route_target(txn, kernel, prepared, started_at).await?;
            match membership::read_location_for_update(txn, record.id()).await? {
                None => {
                    membership::insert_record(
                        txn,
                        &prepared.record,
                        prepared.payload.as_ref(),
                        &target,
                        &prepared.entry,
                    )
                    .await?;
                    Ok(MutationOutcome::Upserted { replaced: false })
                }
                Some(expected) => {
                    membership::replace_record(
                        txn,
                        &prepared.record,
                        prepared.payload.as_ref(),
                        &expected,
                        &target,
                        &prepared.entry,
                    )
                    .await?;
                    Ok(MutationOutcome::Upserted { replaced: true })
                }
            }
        }
        Mutation::Delete(id) => {
            let outcome = membership::delete_record(txn, id).await?;
            Ok(MutationOutcome::Deleted {
                existed: matches!(outcome, membership::DeleteOutcome::Deleted),
            })
        }
    }
}

/// Routes one prepared record to its target Leaf Partition.
async fn route_target<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    kernel: &VectorKernel,
    prepared: &PreparedRecord,
    started_at: u64,
) -> Result<RecordLocation> {
    let route = routing::route_leaf_for_write_preprocessed(
        txn,
        &prepared.tree_key,
        kernel,
        &prepared.routing,
        started_at,
    )
    .await?;
    Ok(RecordLocation::new(prepared.tree_key.clone(), route.leaf()))
}

/// Derives every routed item's persistent projection once per operation.
///
/// Preparation is a pure function of the validated Records and the immutable
/// Manifest, so one derivation serves every retried attempt. An item failure
/// is reported with its input position before any storage work begins.
fn prepare_all(
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    mutations: &[Mutation],
) -> Result<Vec<Option<PreparedRecord>>> {
    mutations
        .iter()
        .enumerate()
        .map(|(position, mutation)| match mutation {
            Mutation::Insert(record) | Mutation::Upsert(record) => {
                PreparedRecord::new(manifest, kernel, record)
                    .map(Some)
                    .map_err(|error| error.at_position(position))
            }
            Mutation::Delete(_) => Ok(None),
        })
        .collect()
}

/// The snapshot-independent projection of one caller Record: its Tree Key,
/// routing vector, persistent values, and Leaf Entry.
struct PreparedRecord {
    tree_key: TreeKey,
    routing: Box<[f32]>,
    record: VectorRecord,
    payload: Option<OpaquePayload>,
    entry: LeafEntry,
}

impl PreparedRecord {
    /// Derives one validated caller Record's persistent projection.
    fn new(manifest: &IndexManifest, kernel: &VectorKernel, record: &Record) -> Result<Self> {
        let config = manifest.config();
        let mut tree_values = Vec::with_capacity(config.tree_key_fields().len());
        for field_id in config.tree_key_fields() {
            let value = record
                .fields()
                .get(usize::from(field_id.0))
                .ok_or_else(Error::invalid_argument)?;
            tree_values.push(value.clone());
        }
        let (types, type_count) = manifest.tree_key_types();
        let tree_key = TreeKey::encode(&types[..type_count], &tree_values)?;
        let routing = kernel.preprocess(record.vector())?;
        let rabitq7 = RaBitQ7::quantize(&routing)?;
        Ok(Self {
            tree_key,
            routing,
            record: VectorRecord::new(
                record.id().clone(),
                Box::from(record.vector()),
                Box::from(record.fields()),
            ),
            payload: record
                .payload()
                .map(|payload| OpaquePayload::new(payload.clone()))
                .transpose()?,
            entry: LeafEntry::new(record.id().clone(), Box::from(record.fields()), rabitq7),
        })
    }
}

/// Returns the current Unix time in milliseconds, or zero for a clock before
/// the epoch. State-start times only diagnose stalls and never grant
/// ownership, so a degraded clock cannot affect correctness.
fn now_unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|elapsed| u64::try_from(elapsed.as_millis()).ok())
        .unwrap_or(0)
}
