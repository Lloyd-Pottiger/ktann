//! Atomic graph publication around the existing topology state machines.
//!
//! A snapshot fixes each changed internal body. Native graph work finishes
//! before opening a write transaction; exact source fences then protect both
//! topology and graph publication. A stale plan is discarded and rediscovered.

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::observe::labels::Operation;
use crate::runtime::{GraphMaintenance, reads, writes};
use crate::storage::backend::{Backend, ReadOps, ScanLimits, WriteTxn};
use crate::storage::graph_transitions::{GraphTransitionSource, PreparedGraphTransition};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::membership::{added_entry, removed_entry};
use crate::storage::values::{
    ChildEntry, IndexManifest, PartitionHeader, PartitionState, PartitionTransition,
    PersistentValue, expect_centroid,
};
use crate::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn, topology};

use super::{fixup, merge, routing, split, training};

/// One changed body together with its exact final authority header.
struct BodyPlan {
    source: GraphTransitionSource,
    header: Option<PartitionHeader>,
    children: Vec<ChildEntry>,
}

impl BodyPlan {
    async fn read<T: ReadOps>(
        read: &mut ReadLogicalTxn<'_, T>,
        tree: &TreeKey,
        partition: PartitionKey,
        working_bytes: &mut usize,
    ) -> Result<Self> {
        let source =
            GraphTransitionSource::read(read, tree, partition, *working_bytes, &|| Ok(())).await?;
        let next_bytes = source
            .children()
            .iter()
            .try_fold(0_usize, |bytes, entry| {
                bytes.checked_add(
                    std::mem::size_of::<ChildEntry>() + entry.centroid().len() * 4 + 64,
                )
            })
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        *working_bytes = working_bytes
            .checked_sub(
                source
                    .retained_bytes()?
                    .checked_add(next_bytes)
                    .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?,
            )
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        Ok(Self {
            header: source.header(),
            children: source.children().to_vec(),
            source,
        })
    }

    fn add(
        &mut self,
        child: PartitionKey,
        centroid: &[f32],
        working_bytes: &mut usize,
    ) -> Result<()> {
        let position = self
            .children
            .binary_search_by_key(&child, ChildEntry::child)
            .err()
            .ok_or_else(corrupt)?;
        let header = added_entry(self.header.ok_or_else(corrupt)?)?;
        admit_plan_entry(&mut self.children, centroid.len(), working_bytes)?;
        self.children
            .insert(position, ChildEntry::new(child, centroid.to_vec()));
        self.header = Some(header);
        Ok(())
    }

    fn remove(&mut self, partition: PartitionKey) -> Result<()> {
        let position = self
            .children
            .binary_search_by_key(&partition, ChildEntry::child)
            .map_err(|_| corrupt())?;
        self.children.remove(position);
        self.header = Some(removed_entry(self.header.ok_or_else(corrupt)?)?);
        Ok(())
    }
}

/// Admits a centroid copy and any replacement backing buffer before allocating.
/// The old buffer remains charged, covering transient overlap during growth.
fn admit_plan_entry<T>(
    entries: &mut Vec<T>,
    dimensions: usize,
    working_bytes: &mut usize,
) -> Result<()> {
    let limit = || Error::new(ErrorKind::LimitExceeded);
    let capacity = if entries.len() == entries.capacity() {
        entries.capacity().checked_mul(2).ok_or_else(limit)?.max(4)
    } else {
        0
    };
    let bytes = dimensions
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(64))
        .and_then(|bytes| bytes.checked_add(capacity.checked_mul(std::mem::size_of::<T>())?))
        .ok_or_else(limit)?;
    let remaining = working_bytes.checked_sub(bytes).ok_or_else(limit)?;
    if capacity != 0 {
        entries
            .try_reserve_exact(capacity - entries.len())
            .map_err(|_| limit())?;
    }
    *working_bytes = remaining;
    Ok(())
}

/// Existing topology operation whose changed bodies are prepared together.
enum Edit {
    Expose {
        left: PartitionKey,
        right: PartitionKey,
        centroids: training::SplitCentroids,
    },
    Drain {
        moves: Vec<(ChildEntry, PartitionKey)>,
        movement: topology::Movement,
        remaining: u32,
        other_target: Option<(PartitionKey, PartitionHeader)>,
    },
    SplitComplete {
        left: PartitionKey,
        right: PartitionKey,
    },
    MergeComplete,
}

/// Topology discovery retained for planning against the same snapshot.
struct BodyParticipants {
    keys: Vec<PartitionKey>,
    parent: Option<PartitionKey>,
    merge_candidates: Vec<topology::LevelCandidate>,
}

/// Discovers ownership from topology metadata before loading graph payloads.
/// Split drains also own the other sibling, whose header guides balancing.
async fn body_participants<T: ReadOps>(
    read: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree: &TreeKey,
    partition: PartitionKey,
    header: PartitionHeader,
    state: &PartitionTransition,
) -> Result<BodyParticipants> {
    let mut participants = vec![partition];
    let mut parent = None;
    let mut merge_candidates = Vec::new();
    match state {
        PartitionTransition::Splitting { .. }
        | PartitionTransition::DrainingSplit { .. }
        | PartitionTransition::Merging { .. }
            if matches!(state, PartitionTransition::Splitting { .. })
                || header.entry_count() == 0 =>
        {
            if partition != topology::root_partition()
                && let Some((found, _)) =
                    topology::find_incoming_edge(read, manifest, tree, partition, header.level())
                        .await?
            {
                participants.push(found);
                parent = Some(found);
            }
        }
        PartitionTransition::DrainingSplit { left, right, .. } => {
            participants.extend([*left, *right]);
        }
        PartitionTransition::Merging { .. } => {
            // The first child fixes the receiver for this bounded merge batch.
            // Discovery needs one centroid, not the source or receiver graph.
            let range = LogicalRange::child_entries(manifest, tree, partition)?;
            let page = read
                .scan(
                    &range,
                    None,
                    ScanLimits {
                        item_limit: 1,
                        byte_limit: manifest.config().dimension() * 4
                            + tree.as_bytes().len()
                            + 1024,
                    },
                )
                .await?;
            let (items, _) = page.into_parts();
            let entry = match items.into_iter().next().map(|item| item.into_value()) {
                Some(PersistentValue::ChildEntry(entry)) => entry,
                _ => return Err(corrupt()),
            };
            let ceiling = manifest.config().max_internal_partition_entries() + 2;
            let candidates: Vec<_> =
                topology::same_level_candidates(read, manifest, tree, header.level())
                    .await?
                    .into_iter()
                    .filter(|candidate| candidate.header().entry_count() < ceiling)
                    .collect();
            let kernel = routing::kernel_for(manifest)?;
            if let Some((_, target)) =
                routing::nearest_ready_candidate(&kernel, entry.centroid(), partition, &candidates)?
            {
                participants.push(target.partition());
            }
            merge_candidates = candidates;
        }
        _ => {}
    }
    participants.sort_unstable();
    participants.dedup();
    Ok(BodyParticipants {
        keys: participants,
        parent,
        merge_candidates,
    })
}

/// Advances one graph partition by one committed topology step.
///
/// Reuse the same executor across concurrent drivers to retain native CPU
/// admission. Bound simultaneous drivers as each owns a snapshot plan before
/// native admission. Unknown commit outcomes are
/// returned without replay; the next call rediscovers the persisted state.
pub async fn advance<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree: &TreeKey,
    partition: PartitionKey,
    now_unix_millis: u64,
    retry: &crate::runtime::RetryPolicy,
    executor: &GraphMaintenance,
) -> Result<()> {
    if manifest.config().local_graph().is_none() {
        return Err(Error::invalid_argument());
    }
    let Some(authority) = reads::open_authority_read(backend, manifest, tree, partition).await?
    else {
        return Ok(());
    };
    let outcome = advance_observed(
        backend,
        manifest,
        tree,
        partition,
        fixup::StepPolicy {
            now_unix_millis,
            recovery: None,
            retry,
            graph: Some(executor),
        },
        authority,
    )
    .await?;
    match outcome {
        fixup::Advance::Idle => Ok(()),
        fixup::Advance::Split(result) => result.map(|_| ()),
        fixup::Advance::Merge(result) => result.map(|_| ()),
    }
}

/// Executes a graph-aware maintenance step using the worker's bounded executor.
pub(crate) async fn advance_observed<'b, B: Backend>(
    backend: &'b B,
    manifest: &IndexManifest,
    tree: &TreeKey,
    partition: PartitionKey,
    policy: fixup::StepPolicy<'_>,
    authority: reads::PartitionRead<'_, B::ReadTxn<'b>>,
) -> Result<fixup::Advance> {
    let executor = policy.graph.ok_or_else(Error::invalid_argument)?;
    let now = policy.now_unix_millis;
    let retry = policy.retry;
    let header = authority.header;
    let state = authority.state;
    // Beginning maintenance changes State alone and retains the same body epoch.
    if matches!(state, PartitionTransition::Ready { .. }) {
        return if header.entry_count() > manifest.config().max_partition_entries(header.level()) {
            Ok(fixup::Advance::Split(
                split::advance_observed(backend, manifest, tree, partition, now, retry, authority)
                    .await,
            ))
        } else {
            Ok(fixup::Advance::Merge(
                merge::advance_observed(backend, manifest, tree, partition, now, retry, authority)
                    .await,
            ))
        };
    }
    // Leaf movement changes exact membership, not an internal routing graph.
    if header.level() == 1 && header.entry_count() > 0 {
        match state {
            PartitionTransition::DrainingSplit { .. } => {
                drop(authority);
                return Ok(fixup::Advance::Split(
                    split::drain_batch(backend, manifest, tree, partition, retry)
                        .await
                        .map(|step| match step {
                            split::DrainStep::Drained { moved, remaining } => {
                                split::Advance::Drained { moved, remaining }
                            }
                            _ => split::Advance::Idle,
                        }),
                ));
            }
            PartitionTransition::Merging { .. } => {
                drop(authority);
                return Ok(fixup::Advance::Merge(
                    merge::drain_batch(backend, manifest, tree, partition, retry)
                        .await
                        .map(|step| match step {
                            merge::DrainStep::Drained { moved, remaining } => {
                                merge::Advance::Drained { moved, remaining }
                            }
                            merge::DrainStep::NoReadyTarget => merge::Advance::Stalled,
                            _ => merge::Advance::Idle,
                        }),
                ));
            }
            _ => {}
        }
    }
    drop(authority);
    let mut failures = 0;
    // Keep body ownership while refreshing a stale plan, so another local
    // publisher cannot invalidate the replacement snapshot during that read.
    let mut locked_bodies = None;
    loop {
        let Some(observed) = reads::open_authority_read(backend, manifest, tree, partition).await?
        else {
            return Ok(fixup::Advance::Idle);
        };
        let reads::PartitionRead {
            txn: mut read,
            header,
            state,
        } = observed;
        let participants =
            body_participants(&mut read, manifest, tree, partition, header, &state).await?;
        if locked_bodies
            .as_ref()
            .is_none_or(|(held, _)| *held != participants.keys)
        {
            drop(locked_bodies.take());
            let guards = executor
                .lock_bodies(manifest.logical_index_id(), tree, participants.keys.clone())
                .await?;
            let waited = guards.waited;
            locked_bodies = Some((participants.keys.clone(), guards));
            if waited {
                // Refresh after another local publisher held these bodies.
                // Uncontended acquisition reuses this snapshot; exact source
                // fences still protect publication against concurrent writers.
                continue;
            }
        }
        let mut bodies = Vec::new();
        let mut read_bytes = executor.working_bytes();
        let edit = match state {
            PartitionTransition::Splitting { left, right, .. } => {
                // Training uses the same existing deterministic split trainer.
                let centroids = training::train_split_centroids(read, tree, partition).await?;
                read = reads::open_validated_read(backend, manifest).await?;
                if partition != topology::root_partition() {
                    let Some(parent) = participants.parent else {
                        return Ok(fixup::Advance::Idle);
                    };
                    if read
                        .get(LogicalKey::ChildEntry {
                            index: manifest.logical_index_id(),
                            tree_key: tree.clone(),
                            partition: parent,
                            child: partition,
                        })
                        .await?
                        .is_none()
                    {
                        drop(locked_bodies.take());
                        continue;
                    }
                    let mut body = BodyPlan::read(&mut read, tree, parent, &mut read_bytes).await?;
                    let parent_header = body.header.ok_or_else(corrupt)?;
                    if !matches!(
                        parent_header.state(),
                        PartitionState::Ready | PartitionState::Splitting
                    ) || parent_header.entry_count()
                        > manifest.config().max_internal_partition_entries()
                    {
                        return Ok(fixup::Advance::Idle);
                    }
                    body.add(left, centroids.left().components(), &mut read_bytes)?;
                    body.add(right, centroids.right().components(), &mut read_bytes)?;
                    bodies.push(body);
                }
                Edit::Expose {
                    left,
                    right,
                    centroids,
                }
            }
            PartitionTransition::DrainingSplit { left, right, .. } if header.entry_count() == 0 => {
                let mut source_body =
                    BodyPlan::read(&mut read, tree, partition, &mut read_bytes).await?;
                if partition == topology::root_partition() {
                    for target in [left, right] {
                        let centroid = centroid(&mut read, manifest, tree, target).await?;
                        source_body.add(target, centroid.components(), &mut read_bytes)?;
                    }
                    let promoted = source_body.header.ok_or_else(corrupt)?;
                    source_body.header = Some(
                        PartitionHeader::new(
                            promoted.level().checked_add(1).ok_or_else(corrupt)?,
                            promoted.entry_count(),
                            promoted.cache_epoch(),
                            PartitionState::Ready,
                        )
                        .map_err(|_| corrupt())?,
                    );
                } else {
                    let Some(parent) = participants.parent else {
                        return Ok(fixup::Advance::Idle);
                    };
                    let mut body = BodyPlan::read(&mut read, tree, parent, &mut read_bytes).await?;
                    body.remove(partition)?;
                    bodies.push(body);
                    source_body.header = None;
                }
                bodies.push(source_body);
                Edit::SplitComplete { left, right }
            }
            PartitionTransition::Merging { .. } if header.entry_count() == 0 => {
                let Some(parent) = participants.parent else {
                    return Ok(fixup::Advance::Idle);
                };
                let mut body = BodyPlan::read(&mut read, tree, parent, &mut read_bytes).await?;
                body.remove(partition)?;
                bodies.push(body);
                let mut source_body =
                    BodyPlan::read(&mut read, tree, partition, &mut read_bytes).await?;
                source_body.header = None;
                bodies.push(source_body);
                Edit::MergeComplete
            }
            PartitionTransition::DrainingSplit { left, right, .. } => {
                let mut source_body =
                    BodyPlan::read(&mut read, tree, partition, &mut read_bytes).await?;
                let mut headers = read
                    .batch_get(
                        [left, right]
                            .map(|partition| LogicalKey::Header {
                                index: manifest.logical_index_id(),
                                tree_key: tree.clone(),
                                partition,
                            })
                            .into(),
                    )
                    .await?
                    .into_iter();
                let left_header =
                    crate::storage::values::expect_header(headers.next().ok_or_else(corrupt)?)?
                        .ok_or_else(corrupt)?;
                let right_header =
                    crate::storage::values::expect_header(headers.next().ok_or_else(corrupt)?)?
                        .ok_or_else(corrupt)?;
                let left_centroid = centroid(&mut read, manifest, tree, left).await?;
                let right_centroid = centroid(&mut read, manifest, tree, right).await?;
                let kernel = routing::kernel_for(manifest)?;
                let balance = split::SplitBalance::new(
                    header.entry_count(),
                    left_header.entry_count(),
                    right_header.entry_count(),
                    manifest.config().min_partition_entries(header.level()),
                );
                let moves = split_drain_moves(
                    &source_body.children,
                    &kernel,
                    [(left, &left_centroid), (right, &right_centroid)],
                    balance,
                    &mut read_bytes,
                )?;
                let target = moves.first().ok_or_else(corrupt)?.1;
                let mut receiver = BodyPlan::read(&mut read, tree, target, &mut read_bytes).await?;
                for (entry, _) in &moves {
                    receiver.add(entry.child(), entry.centroid(), &mut read_bytes)?;
                }
                for (entry, _) in &moves {
                    source_body.remove(entry.child())?;
                }
                let remaining = source_body.header.ok_or_else(corrupt)?.entry_count();
                let other_target = Some(if target == left {
                    (right, right_header)
                } else {
                    (left, left_header)
                });
                bodies.push(source_body);
                bodies.push(receiver);
                Edit::Drain {
                    moves,
                    movement: topology::Movement::Split,
                    remaining,
                    other_target,
                }
            }
            PartitionTransition::Merging { .. } => {
                let mut source_body =
                    BodyPlan::read(&mut read, tree, partition, &mut read_bytes).await?;
                let ceiling = manifest.config().max_internal_partition_entries() + 2;
                let candidates = participants.merge_candidates;
                let kernel = routing::kernel_for(manifest)?;
                let mut moves = Vec::new();
                let mut receiver = None;
                for entry in source_body
                    .children
                    .iter()
                    .take(super::drain::DRAIN_BATCH_INTERNAL)
                {
                    let Some((_, target)) = routing::nearest_ready_candidate(
                        &kernel,
                        entry.centroid(),
                        partition,
                        &candidates,
                    )?
                    else {
                        break;
                    };
                    let target = target.partition();
                    if moves.first().is_some_and(|(_, first)| *first != target) {
                        continue;
                    }
                    if receiver.is_none() {
                        receiver =
                            Some(BodyPlan::read(&mut read, tree, target, &mut read_bytes).await?);
                    }
                    let body = receiver.as_mut().ok_or_else(corrupt)?;
                    if body.header.ok_or_else(corrupt)?.entry_count() >= ceiling {
                        break;
                    }
                    body.add(entry.child(), entry.centroid(), &mut read_bytes)?;
                    admit_plan_entry(&mut moves, entry.centroid().len(), &mut read_bytes)?;
                    moves.push((entry.clone(), target));
                }
                if moves.is_empty() {
                    return Ok(fixup::Advance::Merge(Ok(merge::Advance::Stalled)));
                }
                for (entry, _) in &moves {
                    source_body.remove(entry.child())?;
                }
                let remaining = source_body.header.ok_or_else(corrupt)?.entry_count();
                bodies.push(source_body);
                bodies.push(receiver.ok_or_else(corrupt)?);
                Edit::Drain {
                    moves,
                    movement: topology::Movement::Merge,
                    remaining,
                    other_target: None,
                }
            }
            _ => return Ok(fixup::Advance::Idle),
        };
        drop(read);
        let (partitions, body_guards) = locked_bodies.take().ok_or_else(corrupt)?;
        if bodies
            .iter()
            .any(|body| !partitions.contains(&body.source.partition()))
        {
            // An independent owner can relocate an incoming edge. Rediscover
            // its new participants before preparing any native graph.
            drop(body_guards);
            continue;
        }
        let edit_bytes = match &edit {
            Edit::Expose { centroids, .. } => {
                (centroids.left().components().len() + centroids.right().components().len()) * 4
            }
            Edit::Drain { moves, .. } => moves
                .iter()
                .try_fold(
                    moves.capacity() * std::mem::size_of::<(ChildEntry, PartitionKey)>(),
                    |bytes, (entry, _)| bytes.checked_add(entry.centroid().len() * 4),
                )
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?,
            Edit::SplitComplete { .. } | Edit::MergeComplete => 0,
        };
        let (prepared, body_guards) = executor
            .prepare(move |bytes, check| {
                let bytes = bytes
                    .checked_sub(edit_bytes)
                    .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
                let prepared = GraphTransitionSource::prepare_batch(
                    bodies
                        .into_iter()
                        .map(|body| (body.source, body.header, body.children))
                        .collect(),
                    bytes,
                    check,
                )?;
                Ok((prepared, body_guards))
            })
            .await?;
        let operation = if matches!(
            edit,
            Edit::MergeComplete
                | Edit::Drain {
                    movement: topology::Movement::Merge,
                    ..
                }
        ) {
            Operation::MergeFixup
        } else {
            Operation::SplitFixup
        };
        let removal = topology::SourceRemoval::for_capabilities(backend.capabilities());
        let outcome =
            writes::run_write_attempts(backend, None, manifest, retry, operation, |txn| {
                writes::boxed_step(apply(txn, tree, partition, now, removal, &edit, &prepared))
            })
            .await?;
        if let Some(outcome) = outcome {
            match &outcome {
                fixup::Advance::Split(Ok(split::Advance::Drained { moved, .. })) => {
                    crate::observe::metrics::fixup_drain_step(
                        crate::observe::labels::FixupKind::Split,
                        *moved,
                    )
                }
                fixup::Advance::Merge(Ok(merge::Advance::Drained { moved, .. })) => {
                    crate::observe::metrics::fixup_drain_step(
                        crate::observe::labels::FixupKind::Merge,
                        *moved,
                    )
                }
                _ => {}
            }
            return Ok(outcome);
        }
        locked_bodies = Some((partitions, body_guards));
        retry.wait_or_exhaust(operation, &mut failures).await?;
    }
}

/// Selects one receiver from a bounded candidate page. Skipped candidates stay
/// in the source and do not consume the split's minimum-occupancy reservation.
fn split_drain_moves(
    children: &[ChildEntry],
    kernel: &crate::search::numeric::VectorKernel,
    targets: [(PartitionKey, &crate::storage::values::PartitionCentroid); 2],
    mut balance: split::SplitBalance,
    working_bytes: &mut usize,
) -> Result<Vec<(ChildEntry, PartitionKey)>> {
    let [(left, left_centroid), (right, right_centroid)] = targets;
    let mut moves = Vec::new();
    for entry in children.iter().take(super::drain::DRAIN_BATCH_INTERNAL) {
        let mut trial = balance;
        let target = split::nearer_target(
            kernel,
            entry.centroid(),
            left,
            left_centroid,
            right,
            right_centroid,
            &mut trial,
        )?;
        if moves
            .first()
            .is_some_and(|(_, receiver)| *receiver != target)
        {
            continue;
        }
        balance = trial;
        admit_plan_entry(&mut moves, entry.centroid().len(), working_bytes)?;
        moves.push((entry.clone(), target));
    }
    Ok(moves)
}

async fn centroid<T: ReadOps>(
    read: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree: &TreeKey,
    partition: PartitionKey,
) -> Result<crate::storage::values::PartitionCentroid> {
    expect_centroid(
        read.get(LogicalKey::Centroid {
            index: manifest.logical_index_id(),
            tree_key: tree.clone(),
            partition,
        })
        .await?,
    )?
    .ok_or_else(corrupt)
}

/// Protects all sources before the first topology edit and publishes in the same commit.
async fn apply<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree: &TreeKey,
    source: PartitionKey,
    now: u64,
    removal: topology::SourceRemoval,
    edit: &Edit,
    graphs: &[PreparedGraphTransition],
) -> Result<Option<fixup::Advance>> {
    for graph in graphs {
        if !graph.protect(txn).await? {
            return Ok(None);
        }
    }
    if let Edit::Drain {
        other_target: Some((partition, expected)),
        ..
    } = edit
    {
        let index = txn.require_manifest()?.logical_index_id();
        let actual = crate::storage::values::expect_header(
            txn.get_for_update(LogicalKey::Header {
                index,
                tree_key: tree.clone(),
                partition: *partition,
            })
            .await?,
        )?;
        if actual != Some(*expected) {
            return Ok(None);
        }
    }
    let outcome = match edit {
        Edit::Expose {
            left,
            right,
            centroids,
        } => match topology::expose_split_targets(
            txn,
            tree,
            source,
            [(*left, centroids.left()), (*right, centroids.right())],
            now,
        )
        .await?
        {
            topology::SplitExposure::Exposed => {
                fixup::Advance::Split(Ok(split::Advance::Exposed {
                    left: *left,
                    right: *right,
                }))
            }
            _ => return Ok(None),
        },
        Edit::Drain {
            moves,
            movement,
            remaining,
            ..
        } => {
            let moved =
                topology::relocate_child_entries(txn, tree, source, moves.clone(), *movement)
                    .await?;
            if matches!(movement, topology::Movement::Split) {
                fixup::Advance::Split(Ok(split::Advance::Drained {
                    moved,
                    remaining: *remaining,
                }))
            } else {
                fixup::Advance::Merge(Ok(merge::Advance::Drained {
                    moved,
                    remaining: *remaining,
                }))
            }
        }
        Edit::SplitComplete { left, right } => {
            match topology::finalize_split(txn, tree, source, now, removal).await? {
                topology::SplitCompletion::Completed { parent } => {
                    fixup::Advance::Split(Ok(split::Advance::Completed {
                        left: *left,
                        right: *right,
                        parent,
                    }))
                }
                _ => return Ok(None),
            }
        }
        Edit::MergeComplete => match topology::finalize_merge(txn, tree, source, removal).await? {
            topology::MergeCompletion::Completed => {
                fixup::Advance::Merge(Ok(merge::Advance::Completed))
            }
            _ => return Ok(None),
        },
    };
    let mut builder = txn.mutations();
    for graph in graphs {
        graph.validate_target(txn).await?;
        graph.append_authority(&mut builder)?;
    }
    if txn
        .affected_bytes()
        .checked_add(builder.affected_bytes()?)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?
        > 900_000
    {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    for graph in graphs {
        graph.append_payload(&mut builder)?;
    }
    txn.apply(builder).await?;
    Ok(Some(outcome))
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Metric;
    use crate::storage::values::PartitionCentroid;

    #[test]
    fn plan_admission_rejects_before_growing_high_dimension_payload_buffers() {
        let dimensions = 16_384;
        let mut entries: Vec<ChildEntry> = Vec::new();
        let first_bytes = dimensions * 4 + 64 + 4 * std::mem::size_of::<ChildEntry>();
        let mut remaining = first_bytes - 1;
        assert_eq!(
            admit_plan_entry(&mut entries, dimensions, &mut remaining)
                .unwrap_err()
                .kind(),
            ErrorKind::LimitExceeded,
        );
        assert_eq!(remaining, first_bytes - 1);
        assert_eq!(entries.capacity(), 0);

        remaining = first_bytes;
        admit_plan_entry(&mut entries, dimensions, &mut remaining).unwrap();
        assert_eq!(remaining, 0);
        let capacity = entries.capacity();
        entries.resize_with(capacity, || {
            ChildEntry::new(PartitionKey::new(10).unwrap(), vec![0.0; dimensions])
        });
        let growth_bytes = dimensions * 4 + 64 + capacity * 2 * std::mem::size_of::<ChildEntry>();
        remaining = growth_bytes - 1;
        assert_eq!(
            admit_plan_entry(&mut entries, dimensions, &mut remaining)
                .unwrap_err()
                .kind(),
            ErrorKind::LimitExceeded,
        );
        assert_eq!(entries.capacity(), capacity);
        assert_eq!(entries.len(), capacity);
    }

    #[test]
    fn split_drain_rejects_high_dimension_copies_without_budget() {
        let dimensions = 16_384;
        let left = PartitionKey::new(2).unwrap();
        let right = PartitionKey::new(3).unwrap();
        let centers = [
            PartitionCentroid::new(vec![0.0; dimensions]),
            PartitionCentroid::new(vec![1.0; dimensions]),
        ];
        let children = vec![ChildEntry::new(
            PartitionKey::new(10).unwrap(),
            vec![0.0; dimensions],
        )];
        let kernel =
            crate::search::numeric::VectorKernel::new(dimensions, Metric::L2, [7; 32]).unwrap();
        let mut remaining = dimensions * 4 - 1;
        assert_eq!(
            split_drain_moves(
                &children,
                &kernel,
                [(left, &centers[0]), (right, &centers[1])],
                split::SplitBalance::new(1, 0, 0, 0),
                &mut remaining,
            )
            .unwrap_err()
            .kind(),
            ErrorKind::LimitExceeded,
        );
        assert_eq!(remaining, dimensions * 4 - 1);
    }

    #[test]
    fn split_batch_keeps_noncontiguous_children_for_one_receiver() {
        let left = PartitionKey::new(2).unwrap();
        let right = PartitionKey::new(3).unwrap();
        let centers = [
            PartitionCentroid::new(vec![0.0]),
            PartitionCentroid::new(vec![10.0]),
        ];
        let kernel = crate::search::numeric::VectorKernel::new(1, Metric::L2, [7; 32]).unwrap();
        for count in [8, super::super::drain::DRAIN_BATCH_INTERNAL + 4] {
            let children: Vec<_> = (0..count)
                .map(|i| {
                    ChildEntry::new(
                        PartitionKey::new(i as u64 + 10).unwrap(),
                        vec![if i % 2 == 0 { 0.0 } else { 10.0 }],
                    )
                })
                .collect();
            let mut working_bytes = usize::MAX;
            let moves = split_drain_moves(
                &children,
                &kernel,
                [(left, &centers[0]), (right, &centers[1])],
                split::SplitBalance::new(count as u32, 0, 0, 3),
                &mut working_bytes,
            )
            .unwrap();
            let expected = count.min(super::super::drain::DRAIN_BATCH_INTERNAL) / 2;
            assert_eq!(moves.len(), expected);
            for (ordinal, (entry, target)) in moves.into_iter().enumerate() {
                assert_eq!(entry.child().get(), ordinal as u64 * 2 + 10);
                assert_eq!(target, left);
            }
        }
    }
}
