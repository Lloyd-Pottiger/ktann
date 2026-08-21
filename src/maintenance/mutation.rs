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
//! - **Routing.** Insert and upsert targets come from one grouped descent per
//!   distinct Tree Key
//!   ([`routing::route_leaves_for_write_preprocessed`]): the Tree Manifest and
//!   every visited internal partition are read once per attempt, not once per
//!   record. Upsert reads every stored Record Location with one batched
//!   update-protected read to choose between insert and replacement; a
//!   replacement across Tree Keys or leaves is one atomic move. Delete never
//!   routes and follows the exact stored location.
//! - **Writes.** Every item queues its record-group writes into one shared
//!   builder applied once in canonical key order; exact Header counts and
//!   Synopsis expansions still apply per item so later items observe them.
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
//! runtime (#31); losing it never affects correctness because every
//! committed topology state remains searchable.

use std::collections::BTreeMap;

use crate::api::{Error, ErrorKind, Mutation, MutationOutcome, Record, Result};
use crate::runtime::lifecycle::RetryPolicy;
use crate::runtime::{OperationContext, writes};
use crate::search::numeric::VectorKernel;
use crate::search::rabitq::RaBitQ7;
use crate::storage::backend::{Backend, WriteTxn};
use crate::storage::keys::TreeKey;
use crate::storage::values::{
    IndexManifest, LeafEntry, OpaquePayload, RecordLocation, VectorRecord,
};
use crate::storage::{MutationBuilder, WriteLogicalTxn, membership};

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
    let backend = context.backend();
    writes::run_write_attempts(
        backend.as_ref(),
        Some(context),
        handle_manifest,
        &retry,
        |txn| writes::boxed_step(apply_all(txn, &kernel, mutations, &prepared)),
    )
    .await
}

/// Applies every mutation in input order inside the attempt transaction.
///
/// Routing and the upsert membership reads are batched across the whole
/// operation first — one grouped descent per distinct Tree Key, one batched
/// update-protected read of every upsert's authoritative Record Location. The
/// membership operations then apply in input order and queue their
/// record-group writes into one builder, which is applied once in canonical
/// key order at the end.
async fn apply_all<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    kernel: &VectorKernel,
    mutations: &[Mutation],
    prepared: &[Option<PreparedRecord>],
) -> Result<Vec<MutationOutcome>> {
    debug_assert_eq!(mutations.len(), prepared.len());
    let started_at = now_unix_millis();
    let targets = route_all(txn, kernel, prepared, started_at).await?;
    let expected = read_locations(txn, mutations).await?;
    let mut deferred = txn.mutations();
    let mut outcomes = Vec::with_capacity(mutations.len());
    for (position, mutation) in mutations.iter().enumerate() {
        let routed = prepared[position].as_ref().zip(targets[position].as_ref());
        let outcome = apply_one(
            txn,
            mutation,
            routed,
            expected[position].as_ref(),
            &mut deferred,
        )
        .await
        .map_err(|error| error.at_position(position))?;
        outcomes.push(outcome);
    }
    txn.apply(deferred).await?;
    Ok(outcomes)
}

/// One Tree Key's routing group: the items routed together in one descent.
struct RouteGroup<'a> {
    tree_key: &'a TreeKey,
    /// The group's first input position, used for error attribution.
    first_position: usize,
    members: Vec<(usize, &'a PreparedRecord)>,
}

/// Routes every Insert/Upsert item to its target leaf with one batched
/// descent per distinct Tree Key, aligned to the input positions.
///
/// Groups are processed in first-appearance order. A group-level routing
/// failure is a property of the tree's topology rather than of one record, so
/// it is reported at the group's first input position.
async fn route_all<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    kernel: &VectorKernel,
    prepared: &[Option<PreparedRecord>],
    started_at: u64,
) -> Result<Vec<Option<RecordLocation>>> {
    let mut targets: Vec<Option<RecordLocation>> = vec![None; prepared.len()];
    let mut group_index: BTreeMap<&[u8], usize> = BTreeMap::new();
    let mut groups: Vec<RouteGroup<'_>> = Vec::new();
    for (position, prepared) in prepared.iter().enumerate() {
        let Some(prepared) = prepared else { continue };
        match group_index.get(prepared.tree_key.as_bytes()) {
            Some(&group) => groups[group].members.push((position, prepared)),
            None => {
                group_index.insert(prepared.tree_key.as_bytes(), groups.len());
                groups.push(RouteGroup {
                    tree_key: &prepared.tree_key,
                    first_position: position,
                    members: vec![(position, prepared)],
                });
            }
        }
    }
    for group in groups {
        let routings: Vec<&[f32]> = group
            .members
            .iter()
            .map(|(_, prepared)| &*prepared.routing)
            .collect();
        let routes = routing::route_leaves_for_write_preprocessed(
            txn,
            group.tree_key,
            kernel,
            &routings,
            started_at,
        )
        .await
        .map_err(|error| error.at_position(group.first_position))?;
        for ((position, prepared), route) in group.members.into_iter().zip(routes) {
            targets[position] = Some(RecordLocation::new(prepared.tree_key.clone(), route.leaf()));
        }
    }
    Ok(targets)
}

/// Reads the stored Record Locations of every upsert item with one batched
/// update-protected read, aligned to the input positions.
async fn read_locations<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    mutations: &[Mutation],
) -> Result<Vec<Option<RecordLocation>>> {
    let mut upsert_positions = Vec::new();
    let mut ids = Vec::new();
    for (position, mutation) in mutations.iter().enumerate() {
        if let Mutation::Upsert(record) = mutation {
            upsert_positions.push(position);
            ids.push(record.id().clone());
        }
    }
    let mut expected = vec![None; mutations.len()];
    if ids.is_empty() {
        return Ok(expected);
    }
    let locations = membership::read_locations_for_update(txn, &ids)
        .await
        .map_err(|error| match error.position() {
            Some(subset) => error.at_position(upsert_positions[subset]),
            None => error,
        })?;
    for (position, location) in upsert_positions.into_iter().zip(locations) {
        expected[position] = location;
    }
    Ok(expected)
}

/// Applies one mutation item inside the attempt transaction, queueing
/// record-group writes into `deferred`.
///
/// `routed` carries the prepared record and its target location; exactly the
/// insert/upsert items are prepared and routed by `apply_all`, so a missing
/// pair for those items is an internal contract violation, not caller input.
async fn apply_one<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    mutation: &Mutation,
    routed: Option<(&PreparedRecord, &RecordLocation)>,
    expected: Option<&RecordLocation>,
    deferred: &mut MutationBuilder<'_>,
) -> Result<MutationOutcome> {
    match mutation {
        Mutation::Insert(_) => {
            let (prepared, target) = routed.ok_or_else(|| Error::new(ErrorKind::Backend))?;
            membership::insert_record(
                txn,
                deferred,
                &prepared.record,
                prepared.payload.as_ref(),
                target,
                &prepared.entry,
            )
            .await?;
            Ok(MutationOutcome::Inserted)
        }
        Mutation::Upsert(_) => {
            let (prepared, target) = routed.ok_or_else(|| Error::new(ErrorKind::Backend))?;
            match expected {
                None => {
                    membership::insert_record(
                        txn,
                        deferred,
                        &prepared.record,
                        prepared.payload.as_ref(),
                        target,
                        &prepared.entry,
                    )
                    .await?;
                    Ok(MutationOutcome::Upserted { replaced: false })
                }
                Some(expected) => {
                    membership::replace_record(
                        txn,
                        deferred,
                        &prepared.record,
                        prepared.payload.as_ref(),
                        expected,
                        target,
                        &prepared.entry,
                    )
                    .await?;
                    Ok(MutationOutcome::Upserted { replaced: true })
                }
            }
        }
        Mutation::Delete(id) => {
            let outcome = membership::delete_record(txn, deferred, id).await?;
            Ok(MutationOutcome::Deleted {
                existed: matches!(outcome, membership::DeleteOutcome::Deleted),
            })
        }
    }
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
