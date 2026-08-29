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
//!   routes and follows the exact stored location. A descent that meets a
//!   `Merging` partition reselects the nearest `Ready` same-level candidate,
//!   so no insert enters a merge source and an upsert whose Record Location
//!   still names the source relocates atomically; when no `Ready` target
//!   exists the whole operation retries under the bounded policy and surfaces
//!   `ContentionExhausted` on exhaustion (ADR 0008).
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
//! Offering maintenance after a committed mutation is demand-driven and
//! best-effort: the committed attempt's routed targets, replacement and delete
//! sources, and any draining split sources rerouted around during descent are
//! returned for the Runtime's Fixup queue. Losing them never affects
//! correctness because every committed topology state remains searchable.

use std::collections::{BTreeMap, BTreeSet};

use crate::api::{Error, ErrorKind, Mutation, MutationOutcome, PartitionKey, Record, Result};
use crate::observe::labels::Operation;
use crate::runtime::import::ImportPermit;
use crate::runtime::lifecycle::{RetryPolicy, now_unix_millis};
use crate::runtime::{OperationContext, writes};
use crate::search::numeric::VectorKernel;
use crate::search::rabitq::RaBitQ7;
use crate::storage::backend::{Backend, WriteTxn};
use crate::storage::keys::TreeKey;
use crate::storage::values::{
    IndexManifest, LeafEntry, OpaquePayload, PartitionHeader, RecordLocation, VectorRecord,
};
use crate::storage::{MutationBuilder, WriteLogicalTxn, membership};

use super::{fixup, routing};

/// The result of one committed mutation batch: per-item outcomes and the
/// partitions worth offering to demand-driven maintenance.
///
/// The maintenance list holds only partitions whose final committed Header is
/// actionable, plus draining split sources the descent rerouted around. It is
/// coalesced per partition and remains best-effort discovery, not correctness
/// state.
pub(crate) struct MutationReport {
    /// Per-item outcomes aligned to the validated input batch.
    pub(crate) outcomes: Vec<MutationOutcome>,
    /// Discovered `(Tree Key, Partition Key)` maintenance candidates.
    pub(crate) maintenance: Vec<(TreeKey, PartitionKey)>,
}

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
    operation: Operation,
    mut import_permit: Option<&mut ImportPermit<B>>,
) -> Result<MutationReport> {
    let kernel = routing::kernel_for(handle_manifest)?;
    let prepared = prepare_all(handle_manifest, &kernel, mutations)?;
    let backend = context.backend();
    let mut failed_attempts = 0_u32;
    loop {
        let outcome = writes::run_write_attempts_with_optional_import_permit(
            backend.as_ref(),
            Some(&mut *context),
            handle_manifest,
            &retry,
            operation,
            &mut import_permit,
            |txn| writes::boxed_step(apply_all(txn, &kernel, mutations, &prepared)),
        )
        .await?;
        match outcome {
            ApplyOutcome::Applied(report) => return Ok(report),
            // A Merging leaf blocked routing and no Ready same-level target
            // exists: the attempt staged no writes, so the whole operation
            // retries from a fresh snapshot under the bounded policy, and
            // exhaustion returns ContentionExhausted (ADR 0008).
            ApplyOutcome::NoReadyMergeTarget => {
                writes::wait_before_retry(
                    &retry,
                    operation,
                    &mut failed_attempts,
                    &mut import_permit,
                )
                .await?;
            }
        }
    }
}

/// The outcome of one whole mutation attempt.
enum ApplyOutcome {
    /// Every mutation applied; the attempt commits.
    Applied(MutationReport),
    /// Routing found no `Ready` same-level merge target; the attempt staged
    /// no writes and the operation retries.
    NoReadyMergeTarget,
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
) -> Result<ApplyOutcome> {
    debug_assert_eq!(mutations.len(), prepared.len());
    let started_at = now_unix_millis();
    let (targets, draining_sources) = match route_all(txn, kernel, prepared, started_at).await? {
        RouteAll::Routed {
            targets,
            draining_sources,
        } => (targets, draining_sources),
        // Routing queues no writes, so an attempt abandoned here commits
        // nothing and retries whole.
        RouteAll::NoReadyMergeTarget => return Ok(ApplyOutcome::NoReadyMergeTarget),
    };
    let expected = read_locations(txn, mutations).await?;
    let mut deferred = txn.mutations();
    let mut outcomes = Vec::with_capacity(mutations.len());
    let mut changed_headers = BTreeMap::new();
    for (position, mutation) in mutations.iter().enumerate() {
        let routed = prepared[position].as_ref().zip(targets[position].as_ref());
        // Only a committed attempt's discoveries reach the caller, so
        // collecting them inside the item application is safe even though
        // aborted attempts repeat it.
        let outcome = apply_one(
            txn,
            mutation,
            routed,
            expected[position].as_ref(),
            &mut deferred,
            &mut changed_headers,
        )
        .await
        .map_err(|error| error.at_position(position))?;
        outcomes.push(outcome);
    }
    txn.apply(deferred).await?;
    let mut maintenance: BTreeSet<(TreeKey, PartitionKey)> = draining_sources.into_iter().collect();
    let config = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .config();
    for ((tree_key, partition), header) in changed_headers {
        if fixup::is_actionable(config, partition, header) {
            maintenance.insert((tree_key, partition));
        }
    }
    Ok(ApplyOutcome::Applied(MutationReport {
        outcomes,
        maintenance: maintenance.into_iter().collect(),
    }))
}

/// One Tree Key's routing group: the items routed together in one descent.
struct RouteGroup<'a> {
    tree_key: &'a TreeKey,
    /// The group's first input position, used for error attribution.
    first_position: usize,
    members: Vec<(usize, &'a PreparedRecord)>,
}

/// The outcome of routing every Insert/Upsert item to its target leaf.
enum RouteAll {
    /// Every item routed, aligned to the input positions, with the draining
    /// split sources the descents rerouted around.
    Routed {
        /// The routed target locations, aligned to the input positions.
        targets: Vec<Option<RecordLocation>>,
        /// The `DrainingSplit` sources discovered during the descents.
        draining_sources: Vec<(TreeKey, PartitionKey)>,
    },
    /// A `Merging` leaf blocked routing and no `Ready` same-level target
    /// exists.
    NoReadyMergeTarget,
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
) -> Result<RouteAll> {
    let mut targets: Vec<Option<RecordLocation>> = vec![None; prepared.len()];
    let mut draining_sources = Vec::new();
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
        let routes = match routing::route_leaves_for_write_preprocessed(
            txn,
            group.tree_key,
            kernel,
            &routings,
            started_at,
        )
        .await
        .map_err(|error| error.at_position(group.first_position))?
        {
            routing::GroupedDescent::Routed(routes) => routes,
            routing::GroupedDescent::NoReadyMergeTarget => {
                return Ok(RouteAll::NoReadyMergeTarget);
            }
        };
        // A descent rerouted by a draining split source is the relevant
        // access that resumes its drain; group members often share one
        // source, so offer each distinct source once per group.
        let sources: BTreeSet<PartitionKey> = routes
            .iter()
            .filter_map(|route| route.draining_source())
            .collect();
        draining_sources.extend(
            sources
                .into_iter()
                .map(|source| (group.tree_key.clone(), source)),
        );
        for ((position, prepared), route) in group.members.into_iter().zip(routes) {
            targets[position] = Some(RecordLocation::new(prepared.tree_key.clone(), route.leaf()));
        }
    }
    Ok(RouteAll::Routed {
        targets,
        draining_sources,
    })
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
/// Successful items retain each changed partition's final Header in
/// `changed_headers`; the batch filters and coalesces those observations only
/// after every item has applied.
async fn apply_one<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    mutation: &Mutation,
    routed: Option<(&PreparedRecord, &RecordLocation)>,
    expected: Option<&RecordLocation>,
    deferred: &mut MutationBuilder<'_>,
    changed_headers: &mut BTreeMap<(TreeKey, PartitionKey), PartitionHeader>,
) -> Result<MutationOutcome> {
    match mutation {
        Mutation::Insert(_) => {
            let (prepared, target) = routed.ok_or_else(|| Error::new(ErrorKind::Backend))?;
            let header = membership::insert_record_with_header(
                txn,
                deferred,
                &prepared.record,
                prepared.payload.as_ref(),
                target,
                &prepared.entry,
            )
            .await?;
            record_header(changed_headers, target, header);
            Ok(MutationOutcome::Inserted)
        }
        Mutation::Upsert(_) => {
            let (prepared, target) = routed.ok_or_else(|| Error::new(ErrorKind::Backend))?;
            let (outcome, target_header) = match expected {
                None => {
                    let header = membership::insert_record_with_header(
                        txn,
                        deferred,
                        &prepared.record,
                        prepared.payload.as_ref(),
                        target,
                        &prepared.entry,
                    )
                    .await?;
                    (MutationOutcome::Upserted { replaced: false }, header)
                }
                Some(expected) => {
                    let headers = membership::replace_record_with_headers(
                        txn,
                        deferred,
                        &prepared.record,
                        prepared.payload.as_ref(),
                        expected,
                        target,
                        &prepared.entry,
                    )
                    .await?;
                    if let Some(source) = headers.source() {
                        record_header(changed_headers, expected, source);
                    }
                    (
                        MutationOutcome::Upserted { replaced: true },
                        headers.target(),
                    )
                }
            };
            record_header(changed_headers, target, target_header);
            Ok(outcome)
        }
        Mutation::Delete(id) => {
            let report = membership::delete_record_with_header(txn, deferred, id).await?;
            let existed = match report {
                membership::DeleteReport::Deleted { location, header } => {
                    record_header(changed_headers, &location, header);
                    true
                }
                membership::DeleteReport::NotFound => false,
            };
            Ok(MutationOutcome::Deleted { existed })
        }
    }
}

/// Retains the final Header observed for one changed partition in this batch.
fn record_header(
    changed_headers: &mut BTreeMap<(TreeKey, PartitionKey), PartitionHeader>,
    location: &RecordLocation,
    header: PartitionHeader,
) {
    changed_headers.insert((location.tree_key().clone(), location.leaf()), header);
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
