//! The bounded multi-transaction reselect-then-drain merge driver (ADR 0008).
//!
//! This module drives one merge through its durable state machine —
//! `Ready → Merging →` completed, with entries moving to per-batch reselected
//! `Ready` targets and no target state changing at completion — composing the
//! typed topology operations of [`crate::storage::topology`] into
//! independently retryable short transactions. It owns no queue, worker, or
//! lease: every step is one bounded transaction that any process may drive,
//! and every committed intermediate state stays searchable, so a crash
//! between steps loses nothing that a later access cannot rediscover
//! (Demand-Driven Maintenance, ADR 0006).
//!
//! # Contract
//!
//! - **Whole-step retry.** Every step runs as a sequence of whole attempts
//!   from fresh snapshots under the caller's bounded [`RetryPolicy`];
//!   exhaustion returns `ContentionExhausted`. A commit of unknown outcome is
//!   never retried (ADR 0012): the caller recovers by re-driving the same
//!   step, which observes the persisted state and proceeds idempotently.
//! - **No persisted target.** Merging stores no fixed target and no drain
//!   cursor. Each bounded batch performs ordinary same-level routing, skips
//!   the source and every non-Ready candidate, and selects the nearest
//!   current `Ready` target by routing distance with the canonical Partition
//!   Key tie-break. A target that leaves `Ready`
//!   between the read snapshot and the write attempt discards the route and
//!   starts target selection again from a fresh snapshot, with no durable or
//!   process-local target affinity. Different entries may move to different
//!   targets, and a target may cross the split threshold.
//! - **Bounded work.** Each step commits at most one transition or one schema-
//!   and Backend-budget-bounded drain batch — small enough to stay within the
//!   adapter's conservative admission budget. Every batch starts at the
//!   source's current smallest entry because successful moves delete that
//!   prefix.
//! - **Zero-count completion.** The exact source Header count is the sole
//!   proof that draining is complete; the final transaction removes the
//!   incoming reference and the source prefix per the backend's capability
//!   branch, changing no target state and leaving no tombstone.
//! - **Rediscovery.** [`advance`] inspects one partition's durable state and
//!   performs the next bounded step: beginning an eligible under-minimum
//!   `Ready` partition, draining one batch of a `Merging` one, and completing
//!   at exact count zero. A merge with no legal target never starts; a
//!   `Merging` partition whose targets disappeared stalls searchable and
//!   never reverts to `Ready`.

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::observe::labels::{FixupKind, Operation};
use crate::observe::metrics;
use crate::runtime::RetryPolicy;
use crate::runtime::{reads, writes};
use crate::storage::backend::{Backend, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    IndexManifest, PartitionHeader, PartitionState, PartitionTransition, expect_header,
};
use crate::storage::{WriteLogicalTxn, topology};

use super::drain::{self, DrainBatch};
use super::routing;

/// The outcome of one bounded [`drain_batch`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DrainStep {
    /// The batch committed; `remaining` is the source's exact entry count
    /// afterwards.
    Drained {
        /// The number of entries this batch moved.
        moved: usize,
        /// The source's exact entry count after the batch.
        remaining: u32,
    },
    /// The source is no longer `Merging`; a competing worker completed the
    /// merge.
    SourceAdvanced,
    /// The source is not `Merging`; no merge has begun.
    NotMerging,
    /// No legal same-level `Ready` target exists under current topology, so
    /// nothing moved: the source stays a searchable `Merging` partition and a
    /// later access may retry (ADR 0008).
    NoReadyTarget,
}

/// What one [`advance`] rediscovery pass did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Advance {
    /// Nothing to do: the partition is not merge-eligible, has no legal
    /// target to begin with, or another state machine owns it.
    Idle,
    /// The partition began a merge.
    Began,
    /// One drain batch committed.
    Drained {
        /// The number of entries this batch moved.
        moved: usize,
        /// The source's exact entry count after the batch.
        remaining: u32,
    },
    /// The partition is `Merging` but no legal `Ready` target exists; the
    /// fixup retires and a later access may rediscover it.
    Stalled,
    /// The merge completed.
    Completed,
}

/// Runs [`topology::begin_merge`] as one bounded whole-retrying step.
pub async fn begin_merge<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
    retry: &RetryPolicy,
) -> Result<topology::MergeStart> {
    writes::run_write_attempts(
        backend,
        None,
        manifest,
        retry,
        Operation::MergeFixup,
        |txn| {
            writes::boxed_step(topology::begin_merge(
                txn,
                tree_key,
                source,
                started_at_unix_millis,
            ))
        },
    )
    .await
}

/// Moves one bounded batch of source entries to reselected `Ready` targets.
///
/// A short read snapshot first fixes the batch — the source's current
/// smallest entries, using the largest safe Leaf Entry batch for the current
/// schema and Backend budget or the fixed internal-entry bound — and the
/// same-level candidate set. The write transaction revalidates the `Merging`
/// state, then re-reads each candidate with update protection: an entry
/// removed by a concurrent completed mutation is skipped and a remaining
/// membership mismatch is Corruption. Every remaining entry routes against
/// the current candidate centroids to the nearest `Ready` target with the
/// Partition Key tie-break and moves atomically — Leaf Entries with their
/// Record Location and target Synopsis, Child Entries with exact counts
/// alone. A chosen target that left `Ready` in between discards the route and
/// reselects from a fresh snapshot under the bounded policy (ADR 0008).
pub async fn drain_batch<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    retry: &RetryPolicy,
) -> Result<DrainStep> {
    let mut failed_attempts = 0_u32;
    loop {
        // The read phase fixes the batch and the candidate set from one
        // consistent snapshot; one batched read covers both source authority
        // values.
        let mut read = reads::open_validated_read(backend, manifest).await?;
        let pair =
            topology::read_authority_pair(&mut read, manifest.logical_index_id(), tree_key, source)
                .await?;
        let Some((source_header, state)) = pair else {
            // A completed merge removed both authority values.
            return Ok(DrainStep::SourceAdvanced);
        };
        if !matches!(state, PartitionTransition::Merging { .. }) {
            return Ok(DrainStep::NotMerging);
        }
        let level = source_header.level();
        let Some(batch) = drain::next_drain_batch(
            &mut read,
            manifest,
            tree_key,
            source,
            Some(source_header),
            topology::Movement::Merge,
            backend.admission_budget(),
        )
        .await?
        else {
            return Ok(DrainStep::Drained {
                moved: 0,
                remaining: 0,
            });
        };
        let candidates =
            topology::same_level_candidates(&mut read, manifest, tree_key, level).await?;
        if !candidates
            .iter()
            .any(|candidate| candidate.is_legal_merge_target(source))
        {
            return Ok(DrainStep::NoReadyTarget);
        }
        drop(read);

        let plan = DrainPlan {
            batch,
            kernel: routing::kernel_for(manifest)?,
            candidates,
        };
        match writes::run_write_attempts(
            backend,
            None,
            manifest,
            retry,
            Operation::MergeFixup,
            |txn| writes::boxed_step(drain_attempt(txn, manifest, tree_key, source, &plan)),
        )
        .await?
        {
            Attempt::Step(step) => {
                if let DrainStep::Drained { moved, .. } = step {
                    metrics::fixup_drain_step(FixupKind::Merge, moved);
                }
                return Ok(step);
            }
            // A chosen target left `Ready` between the read snapshot and the
            // attempt: discard the route and start target selection again,
            // bounded by the same policy as the whole-step runner.
            Attempt::Reselect => {
                retry
                    .wait_or_exhaust(Operation::MergeFixup, &mut failed_attempts)
                    .await?
            }
        }
    }
}

/// Runs [`topology::finalize_merge`] as one bounded whole-retrying step,
/// clearing the source prefix transactionally when the backend supports it
/// and with bounded point deletes otherwise.
pub async fn complete_merge<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    retry: &RetryPolicy,
) -> Result<topology::MergeCompletion> {
    let removal = topology::SourceRemoval::for_capabilities(backend.capabilities());
    writes::run_write_attempts(
        backend,
        None,
        manifest,
        retry,
        Operation::MergeFixup,
        |txn| writes::boxed_step(topology::finalize_merge(txn, tree_key, source, removal)),
    )
    .await
}

/// Performs the next bounded merge step for one partition, whatever durable
/// state it is rediscovered in (Demand-Driven Maintenance).
///
/// Any process may call this for any index it has open; the steps are
/// idempotent and every committed intermediate state is searchable, so a cold
/// intermediate state resumes exactly where it stopped. One call performs at
/// most one begin, one drain batch, or one completion. A partition with no
/// durable authority values has nothing to maintain: it either never existed
/// or its merge already completed, so the call is an idle no-op. A
/// `Splitting`, `ReceivingSplit`, or `DrainingSplit` partition belongs to the
/// split state machine ([`crate::maintenance::split::advance`]).
pub async fn advance<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    started_at_unix_millis: u64,
    retry: &RetryPolicy,
) -> Result<Advance> {
    let mut read = reads::open_validated_read(backend, manifest).await?;
    let pair =
        topology::read_authority_pair(&mut read, manifest.logical_index_id(), tree_key, partition)
            .await?;
    drop(read);
    // Nothing was ever persisted here, or a completed merge already removed
    // every value: nothing to advance.
    let Some(authority) = pair else {
        return Ok(Advance::Idle);
    };
    advance_observed(
        backend,
        manifest,
        tree_key,
        partition,
        started_at_unix_millis,
        retry,
        authority,
    )
    .await
}

/// Runs one merge step from an already validated Header and State pair.
///
/// The shared Fixup dispatcher uses this entry point after its single
/// preflight read. Public direct drivers retain [`advance`] as the complete
/// read-and-dispatch operation.
pub(crate) async fn advance_observed<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    started_at_unix_millis: u64,
    retry: &RetryPolicy,
    authority: (PartitionHeader, PartitionTransition),
) -> Result<Advance> {
    let (header, state) = authority;
    if matches!(state, PartitionTransition::Merging { .. }) {
        metrics::fixup_state_age(
            FixupKind::Merge,
            started_at_unix_millis,
            state.started_at_unix_millis(),
        );
    }

    match state {
        PartitionTransition::Ready { .. } => {
            // Roots never merge; skipping them here keeps rediscovery of a
            // small tree's under-full root from committing an empty write
            // transaction on every pass.
            if partition == topology::root_partition()
                || header.entry_count() >= manifest.config().min_partition_entries()
            {
                return Ok(Advance::Idle);
            }
            match begin_merge(
                backend,
                manifest,
                tree_key,
                partition,
                started_at_unix_millis,
                retry,
            )
            .await?
            {
                topology::MergeStart::Started | topology::MergeStart::AlreadyMerging => {
                    Ok(Advance::Began)
                }
                // Not merge-eligible (root or an intermediate state) or no
                // legal target exists: no state starts.
                topology::MergeStart::NotEligible | topology::MergeStart::NoReadyTarget => {
                    Ok(Advance::Idle)
                }
            }
        }
        PartitionTransition::Merging { .. } => {
            match drain_batch(backend, manifest, tree_key, partition, retry).await? {
                DrainStep::Drained { moved, remaining } => {
                    if remaining == 0 {
                        match complete_merge(backend, manifest, tree_key, partition, retry).await? {
                            topology::MergeCompletion::Completed
                            | topology::MergeCompletion::NotMerging => Ok(Advance::Completed),
                            topology::MergeCompletion::NotDrained => {
                                Ok(Advance::Drained { moved, remaining })
                            }
                        }
                    } else {
                        Ok(Advance::Drained { moved, remaining })
                    }
                }
                DrainStep::SourceAdvanced => Ok(Advance::Completed),
                DrainStep::NotMerging => Ok(Advance::Idle),
                DrainStep::NoReadyTarget => Ok(Advance::Stalled),
            }
        }
        _ => Ok(Advance::Idle),
    }
}

/// One fixed drain plan: the entry batch fixed by the read snapshot and the
/// same-level target candidate set the write phase revalidates and moves
/// against.
struct DrainPlan {
    batch: DrainBatch,
    kernel: crate::search::numeric::VectorKernel,
    candidates: Vec<topology::LevelCandidate>,
}

/// The outcome of one drain attempt inside the attempt transaction.
enum Attempt {
    /// The attempt produced a final step outcome.
    Step(DrainStep),
    /// A chosen target left `Ready` between the read snapshot and this
    /// attempt; the route is discarded and target selection starts again.
    Reselect,
}

/// Runs one drain attempt inside the attempt transaction.
async fn drain_attempt<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    plan: &DrainPlan,
) -> Result<Attempt> {
    // Revalidate the durable state before moving anything; one batched
    // update-protected read covers both source authority values.
    let index = manifest.logical_index_id();
    let (source_header, state) = topology::authority_for_update(
        txn,
        LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: source,
        },
        LogicalKey::State {
            index,
            tree_key: tree_key.clone(),
            partition: source,
        },
    )
    .await?;
    let (source_header, state) = match (source_header, state) {
        // A completed merge removed both authority values.
        (None, None) => return Ok(Attempt::Step(DrainStep::SourceAdvanced)),
        (Some(header), Some(state)) => (header, state),
        // Completion removes the pair atomically, so a half-present pair is
        // a torn committed state.
        _ => return Err(Error::new(ErrorKind::Corruption)),
    };
    if !matches!(state, PartitionTransition::Merging { .. }) {
        // Merging never reverts to another persisted state and completion
        // removes both authority values, so a present non-Merging State
        // contradicts the persisted protocol.
        return Err(Error::new(ErrorKind::Corruption));
    }
    if source_header.state() != PartitionState::Merging {
        return Err(Error::new(ErrorKind::Corruption));
    }
    let level = source_header.level();

    let moved = match &plan.batch {
        DrainBatch::Leaf(record_ids) => {
            let candidates =
                topology::read_leaf_drain_candidates(txn, tree_key, source, record_ids).await?;
            let mut moves = Vec::new();
            for candidate in candidates.into_iter().flatten() {
                // A `None` slot is a concurrently removed entry: skipped.
                let routing = plan
                    .kernel
                    .preprocess(candidate.record().vector())
                    .map_err(|_| Error::new(ErrorKind::Corruption))?;
                let target = routing::nearest_ready_candidate(
                    &plan.kernel,
                    &routing,
                    source,
                    &plan.candidates,
                )?
                // The read phase proved at least one Ready candidate, and the
                // plan's fixed candidate set cannot shrink inside the attempt.
                .ok_or_else(|| Error::new(ErrorKind::Backend))?
                .partition();
                moves.push((candidate, target));
            }
            if !revalidate_targets(txn, tree_key, level, &moves).await? {
                return Ok(Attempt::Reselect);
            }
            topology::relocate_leaf_entries(txn, tree_key, source, moves, topology::Movement::Merge)
                .await?
        }
        DrainBatch::Child(children) => {
            let candidates =
                topology::read_child_drain_candidates(txn, tree_key, source, children).await?;
            let mut moves = Vec::new();
            for entry in candidates.into_iter().flatten() {
                let target = routing::nearest_ready_candidate(
                    &plan.kernel,
                    entry.centroid(),
                    source,
                    &plan.candidates,
                )?
                .ok_or_else(|| Error::new(ErrorKind::Backend))?
                .partition();
                moves.push((entry, target));
            }
            if !revalidate_targets(txn, tree_key, level, &moves).await? {
                return Ok(Attempt::Reselect);
            }
            topology::relocate_child_entries(
                txn,
                tree_key,
                source,
                moves,
                topology::Movement::Merge,
            )
            .await?
        }
    };
    let remaining = source_header
        .entry_count()
        .checked_sub(u32::try_from(moved).map_err(|_| Error::new(ErrorKind::Corruption))?)
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    Ok(Attempt::Step(DrainStep::Drained { moved, remaining }))
}

/// Update-protects and revalidates every distinct chosen target inside the
/// attempt transaction.
///
/// A target must still be a `Ready` partition at the source's level: a target
/// that left `Ready` — or vanished through its own completed merge — between
/// the read snapshot and this attempt discards the route (`false`), and
/// target selection starts again from a fresh snapshot (ADR 0008). The
/// update-protected reads establish the commit-time conflict with any
/// concurrent target transition, so a target that changes only afterwards
/// aborts the commit and the whole step retries.
async fn revalidate_targets<T: WriteTxn, E>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    level: u32,
    moves: &[(E, PartitionKey)],
) -> Result<bool> {
    let index = txn
        .bound_manifest()
        .ok_or_else(Error::invalid_argument)?
        .logical_index_id();
    let mut targets: Vec<PartitionKey> = moves.iter().map(|(_, target)| *target).collect();
    targets.sort_unstable();
    targets.dedup();
    let keys: Vec<LogicalKey> = targets
        .iter()
        .map(|target| LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: *target,
        })
        .collect();
    for value in txn.batch_get_for_update(keys).await? {
        // A target whose own merge completed in between is simply gone.
        let Some(header) = expect_header(value)? else {
            return Ok(false);
        };
        if header.level() != level {
            return Err(Error::new(ErrorKind::Corruption));
        }
        if header.state() != PartitionState::Ready {
            return Ok(false);
        }
    }
    Ok(true)
}
