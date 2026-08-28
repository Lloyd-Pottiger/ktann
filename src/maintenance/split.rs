//! The bounded multi-transaction expose-then-drain split driver (ADR 0014).
//!
//! This module drives one split through its durable state machine —
//! `Ready → Splitting → DrainingSplit →` completed, with targets exposed as
//! `ReceivingSplit` and promoted to `Ready` at completion — composing the
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
//! - **Bounded work.** Each step commits at most one transition, one target
//!   installation, or one fixed-size drain batch
//!   ([`DRAIN_BATCH_LEAF`] / [`DRAIN_BATCH_INTERNAL`]) — small enough to stay
//!   within every adapter's conservative admission budget. Draining stores no
//!   durable cursor: every batch starts at the source's current smallest
//!   entry because successful moves delete that prefix.
//! - **Training without locks.** Target centroids are trained from one
//!   consistent read snapshot while the source is `Splitting`, outside any
//!   write transaction, via [`training::train_split_centroids`]. Published
//!   centroids are routing models; concurrent source writes never restart
//!   training, and an already-created target always keeps its persisted
//!   centroid.
//! - **Viable target counts.** Draining normally routes by the persisted
//!   target centroids, but exact remaining and target counts reserve enough
//!   entries for each target to reach the configured minimum when possible.
//! - **Rediscovery.** [`advance`] inspects one partition's durable state and
//!   performs the next bounded step: beginning an over-maximum `Ready`
//!   partition, exposing and advancing a `Splitting` one, draining one batch
//!   of a `DrainingSplit` one, and completing at exact count zero. Repeating
//!   `advance` converges a split; abandoning it at any point leaves a
//!   searchable state.

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::observe::labels::{FixupKind, Operation};
use crate::observe::metrics;
use crate::runtime::RetryPolicy;
use crate::runtime::{reads, writes};
use crate::storage::backend::{Backend, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    IndexManifest, PartitionCentroid, PartitionHeader, PartitionState, PartitionTransition,
    expect_centroid, expect_header,
};
use crate::storage::{ReadLogicalTxn, WriteLogicalTxn, topology};

use super::drain::{self, DrainBatch};
pub use super::drain::{DRAIN_BATCH_INTERNAL, DRAIN_BATCH_LEAF};
use super::routing;
use super::training;

/// The outcome of [`expose_targets`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TargetExposure {
    /// Both targets are published as `ReceivingSplit` with their persisted
    /// centroids (by this or a previous committed attempt).
    Exposed {
        /// The left target.
        left: PartitionKey,
        /// The right target.
        right: PartitionKey,
    },
    /// The source is no longer `Splitting` or `DrainingSplit`; a competing
    /// worker's progress made this step obsolete.
    SourceAdvanced,
}

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
    /// The source is no longer `DrainingSplit`; a competing worker completed
    /// the split.
    SourceAdvanced,
    /// The source is not `DrainingSplit` yet (still `Splitting`).
    NotDraining,
}

/// What one [`advance`] rediscovery pass did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Advance {
    /// Nothing to do: the partition is `Ready` at or below the split
    /// threshold, a `ReceivingSplit` target waiting for its source, or
    /// `Merging` (advanced by [`super::merge`]).
    Idle,
    /// The partition began a split.
    Began {
        /// The reserved left target.
        left: PartitionKey,
        /// The reserved right target.
        right: PartitionKey,
    },
    /// The partition's targets are exposed and it is now `DrainingSplit`.
    Exposed {
        /// The left target.
        left: PartitionKey,
        /// The right target.
        right: PartitionKey,
    },
    /// One drain batch committed.
    Drained {
        /// The number of entries this batch moved.
        moved: usize,
        /// The source's exact entry count after the batch.
        remaining: u32,
    },
    /// The split completed.
    Completed,
}

/// Runs [`topology::begin_split`] as one bounded whole-retrying step.
pub async fn begin_split<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
    retry: &RetryPolicy,
) -> Result<topology::SplitStart> {
    writes::run_write_attempts(
        backend,
        None,
        manifest,
        retry,
        Operation::SplitFixup,
        |txn| {
            writes::boxed_step(topology::begin_split(
                txn,
                tree_key,
                source,
                started_at_unix_millis,
            ))
        },
    )
    .await
}

/// Trains and exposes both split targets of one `Splitting` source.
///
/// Training reads the complete source through one consistent snapshot and
/// runs outside any write transaction; the two target installations then
/// commit independently, each pinning its trained centroid only when it wins
/// the unique creation. A competing or recovering worker that observes an
/// already-created target leaves its persisted centroid untouched — the
/// per-target unique creation is exactly ADR 0014's pinning rule: the first
/// successful creation fixes that target's centroid forever, and no worker
/// ever overwrites it. Centroids are routing models, not authority (exact
/// movement preserves membership regardless of their freshness), so a pair
/// trained from one snapshot is never revalidated against later source
/// writes.
///
/// A target installation whose discovered parent cannot accept a new child is
/// abandoned and retried from a fresh snapshot under the retry policy, so a
/// later attempt rediscovers the source's current parent. That outer retry
/// deliberately shares the same bounded policy as the whole-step runner's
/// inner one: the total work is bounded by the product of the two (still a
/// small constant), and exhaustion retires the worker for a later access to
/// rediscover.
pub async fn expose_targets<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
    retry: &RetryPolicy,
) -> Result<TargetExposure> {
    // One consistent snapshot classifies the source and trains both target
    // centroids without holding KV locks.
    let mut read = reads::open_validated_read(backend, manifest).await?;
    let Some(state) = read_source_state(&mut read, tree_key, source).await? else {
        return Ok(TargetExposure::SourceAdvanced);
    };
    let (left, right) = match state {
        PartitionTransition::Splitting { left, right, .. } => (left, right),
        // Advancing proves both targets were exposed.
        PartitionTransition::DrainingSplit { left, right, .. } => {
            return Ok(TargetExposure::Exposed { left, right });
        }
        _ => return Ok(TargetExposure::SourceAdvanced),
    };
    let centroids = training::train_split_centroids(&mut read, tree_key, source).await?;
    drop(read);

    for (target, centroid) in [(left, centroids.left()), (right, centroids.right())] {
        let mut failed_attempts = 0_u32;
        let install = loop {
            let outcome = writes::run_write_attempts(
                backend,
                None,
                manifest,
                retry,
                Operation::SplitFixup,
                |txn| {
                    writes::boxed_step(topology::create_split_target(
                        txn,
                        tree_key,
                        source,
                        target,
                        centroid,
                        started_at_unix_millis,
                    ))
                },
            )
            .await?;
            match outcome {
                // The parent rejected a new child: abandon and rediscover
                // from a fresh snapshot under the same bounded policy.
                topology::TargetInstall::ParentNotAccepting => {
                    retry
                        .wait_or_exhaust(Operation::SplitFixup, &mut failed_attempts)
                        .await?;
                }
                other => break other,
            }
        };
        match install {
            topology::TargetInstall::Created | topology::TargetInstall::AlreadyExists => {}
            topology::TargetInstall::SourceAdvanced => return Ok(TargetExposure::SourceAdvanced),
            topology::TargetInstall::ParentNotAccepting => {
                // Unreachable: the loop above only breaks on other outcomes.
                return Err(Error::new(ErrorKind::Backend));
            }
        }
    }
    Ok(TargetExposure::Exposed { left, right })
}

/// Runs [`topology::advance_to_draining`] as one bounded whole-retrying step.
pub async fn advance_to_draining<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
    retry: &RetryPolicy,
) -> Result<topology::DrainStart> {
    writes::run_write_attempts(
        backend,
        None,
        manifest,
        retry,
        Operation::SplitFixup,
        |txn| {
            writes::boxed_step(topology::advance_to_draining(
                txn,
                tree_key,
                source,
                started_at_unix_millis,
            ))
        },
    )
    .await
}

/// Moves one bounded batch of source entries to their nearer split target.
///
/// A short read snapshot first fixes the batch: the source's current smallest
/// entries, at most [`DRAIN_BATCH_LEAF`] Leaf Entries or
/// [`DRAIN_BATCH_INTERNAL`] Child Entries by source level. The write
/// transaction revalidates the `DrainingSplit` state, then re-reads each
/// candidate with update protection: an entry removed by a concurrent
/// completed mutation is skipped and a remaining membership mismatch is
/// Corruption. Every remaining entry routes against the two persisted target
/// centroids and moves atomically — Leaf Entries with their Record Location and
/// target Synopsis, Child Entries with exact counts alone (ADR 0014). Exact
/// remaining and target counts reserve the last necessary entries for each
/// target to reach the configured minimum before nearest routing could make
/// that impossible.
pub async fn drain_batch<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    retry: &RetryPolicy,
) -> Result<DrainStep> {
    // The read phase fixes the batch from one consistent snapshot; one
    // batched read covers both source authority values.
    let mut read = reads::open_validated_read(backend, manifest).await?;
    let pair =
        topology::read_authority_pair(&mut read, manifest.logical_index_id(), tree_key, source)
            .await?;
    let Some((source_header, state)) = pair else {
        // A completed split removed both authority values.
        return Ok(DrainStep::SourceAdvanced);
    };
    let (left, right) = match state {
        PartitionTransition::DrainingSplit { left, right, .. } => (left, right),
        PartitionTransition::Splitting { .. } => return Ok(DrainStep::NotDraining),
        _ => return Ok(DrainStep::SourceAdvanced),
    };
    let Some(batch) =
        drain::next_drain_batch(&mut read, manifest, tree_key, source, Some(source_header)).await?
    else {
        return Ok(DrainStep::Drained {
            moved: 0,
            remaining: 0,
        });
    };
    drop(read);

    let plan = DrainPlan {
        batch,
        kernel: routing::kernel_for(manifest)?,
        left,
        right,
    };
    let step = writes::run_write_attempts(
        backend,
        None,
        manifest,
        retry,
        Operation::SplitFixup,
        |txn| writes::boxed_step(drain_attempt(txn, manifest, tree_key, source, &plan)),
    )
    .await?;
    if let DrainStep::Drained { moved, .. } = step {
        metrics::fixup_drain_step(FixupKind::Split, moved);
    }
    Ok(step)
}

/// One fixed drain batch: the candidates fixed by the read snapshot and the
/// routing data the write phase revalidates and moves against.
struct DrainPlan {
    batch: DrainBatch,
    kernel: crate::search::numeric::VectorKernel,
    left: PartitionKey,
    right: PartitionKey,
}

/// Runs one drain attempt inside the attempt transaction.
async fn drain_attempt<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    plan: &DrainPlan,
) -> Result<DrainStep> {
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
        // A completed split removed both authority values.
        (None, None) => return Ok(DrainStep::SourceAdvanced),
        (Some(header), Some(state)) => (header, state),
        // Completion removes or converts the pair atomically, so a
        // half-present pair is a torn committed state.
        _ => return Err(Error::new(ErrorKind::Corruption)),
    };
    match state {
        // The DrainingSplit target pair is fixed at the transition, so a
        // different pair contradicts the persisted protocol.
        PartitionTransition::DrainingSplit {
            left: l, right: r, ..
        } if l == plan.left && r == plan.right => {}
        PartitionTransition::DrainingSplit { .. } => {
            return Err(Error::new(ErrorKind::Corruption));
        }
        _ => return Ok(DrainStep::SourceAdvanced),
    }
    if source_header.state() != PartitionState::DrainingSplit {
        return Err(Error::new(ErrorKind::Corruption));
    }

    // Protect the exact target counts that shape this batch; relocation reuses
    // these cached Headers through commit. The immutable centroid reads stay
    // plain so unrelated target lifecycle work does not add conflicts.
    let mut header_values = txn
        .batch_get_for_update(
            [plan.left, plan.right]
                .map(|partition| LogicalKey::Header {
                    index,
                    tree_key: tree_key.clone(),
                    partition,
                })
                .into(),
        )
        .await?
        .into_iter();
    let (Some(left_header_value), Some(right_header_value)) =
        (header_values.next(), header_values.next())
    else {
        // The typed batch read returns exactly one value per input key.
        return Err(Error::new(ErrorKind::Backend));
    };
    let (Some(left_header), Some(right_header)) = (
        expect_header(left_header_value)?,
        expect_header(right_header_value)?,
    ) else {
        return Err(Error::new(ErrorKind::Corruption));
    };

    let mut centroid_values = txn
        .batch_get(
            [plan.left, plan.right]
                .map(|partition| LogicalKey::Centroid {
                    index,
                    tree_key: tree_key.clone(),
                    partition,
                })
                .into(),
        )
        .await?
        .into_iter();
    let (Some(left_centroid_value), Some(right_centroid_value)) =
        (centroid_values.next(), centroid_values.next())
    else {
        return Err(Error::new(ErrorKind::Backend));
    };
    let (Some(left_centroid), Some(right_centroid)) = (
        expect_centroid(left_centroid_value)?,
        expect_centroid(right_centroid_value)?,
    ) else {
        return Err(Error::new(ErrorKind::Corruption));
    };

    let mut balance = SplitBalance::new(
        source_header.entry_count(),
        left_header.entry_count(),
        right_header.entry_count(),
        manifest.config().min_partition_entries(),
    );
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
                let target = nearer_target(
                    &plan.kernel,
                    &routing,
                    plan.left,
                    &left_centroid,
                    plan.right,
                    &right_centroid,
                    &mut balance,
                )?;
                moves.push((candidate, target));
            }
            topology::relocate_leaf_entries(txn, tree_key, source, moves, topology::Movement::Split)
                .await?
        }
        DrainBatch::Child(children) => {
            let candidates =
                topology::read_child_drain_candidates(txn, tree_key, source, children).await?;
            let mut moves = Vec::new();
            for entry in candidates.into_iter().flatten() {
                let target = nearer_target(
                    &plan.kernel,
                    entry.centroid(),
                    plan.left,
                    &left_centroid,
                    plan.right,
                    &right_centroid,
                    &mut balance,
                )?;
                moves.push((entry, target));
            }
            topology::relocate_child_entries(
                txn,
                tree_key,
                source,
                moves,
                topology::Movement::Split,
            )
            .await?
        }
    };
    let remaining = source_header
        .entry_count()
        .checked_sub(u32::try_from(moved).map_err(|_| Error::new(ErrorKind::Corruption))?)
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    Ok(DrainStep::Drained { moved, remaining })
}

/// Runs [`topology::finalize_split`] as one bounded whole-retrying step,
/// clearing the source prefix transactionally when the backend supports it
/// and with bounded point deletes otherwise.
pub async fn complete_split<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    started_at_unix_millis: u64,
    retry: &RetryPolicy,
) -> Result<topology::SplitCompletion> {
    let removal = topology::SourceRemoval::for_capabilities(backend.capabilities());
    writes::run_write_attempts(
        backend,
        None,
        manifest,
        retry,
        Operation::SplitFixup,
        |txn| {
            writes::boxed_step(topology::finalize_split(
                txn,
                tree_key,
                source,
                started_at_unix_millis,
                removal,
            ))
        },
    )
    .await
}

/// Performs the next bounded split step for one partition, whatever durable
/// state it is rediscovered in (Demand-Driven Maintenance).
///
/// Any process may call this for any index it has open; the steps are
/// idempotent and every committed intermediate state is searchable, so a cold
/// intermediate state resumes exactly where it stopped. One call performs at
/// most one begin, one expose-and-advance sequence, one drain batch, or one
/// completion. A partition with no durable authority values has nothing to
/// maintain: it either never existed or its maintenance already completed, so
/// the call is an idle no-op.
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
    // Nothing was ever persisted here, or a completed split already removed
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

/// Runs one split step from an already validated Header and State pair.
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
    if matches!(
        state,
        PartitionTransition::Splitting { .. } | PartitionTransition::DrainingSplit { .. }
    ) {
        metrics::fixup_state_age(
            FixupKind::Split,
            started_at_unix_millis,
            state.started_at_unix_millis(),
        );
    }

    match state {
        PartitionTransition::Ready { .. } => {
            if header.entry_count() <= manifest.config().max_partition_entries() {
                return Ok(Advance::Idle);
            }
            match begin_split(
                backend,
                manifest,
                tree_key,
                partition,
                started_at_unix_millis,
                retry,
            )
            .await?
            {
                topology::SplitStart::Started { left, right }
                | topology::SplitStart::AlreadySplitting { left, right } => {
                    Ok(Advance::Began { left, right })
                }
                topology::SplitStart::NotEligible => Ok(Advance::Idle),
            }
        }
        PartitionTransition::Splitting { .. } => {
            match expose_targets(
                backend,
                manifest,
                tree_key,
                partition,
                started_at_unix_millis,
                retry,
            )
            .await?
            {
                TargetExposure::Exposed { left, right } => {
                    match advance_to_draining(
                        backend,
                        manifest,
                        tree_key,
                        partition,
                        started_at_unix_millis,
                        retry,
                    )
                    .await?
                    {
                        topology::DrainStart::Advanced | topology::DrainStart::AlreadyDraining => {
                            Ok(Advance::Exposed { left, right })
                        }
                        topology::DrainStart::NotSplitting => Ok(Advance::Idle),
                    }
                }
                TargetExposure::SourceAdvanced => Ok(Advance::Idle),
            }
        }
        PartitionTransition::DrainingSplit { .. } => {
            match drain_batch(backend, manifest, tree_key, partition, retry).await? {
                DrainStep::Drained { moved, remaining } => {
                    if remaining == 0 {
                        match complete_split(
                            backend,
                            manifest,
                            tree_key,
                            partition,
                            started_at_unix_millis,
                            retry,
                        )
                        .await?
                        {
                            topology::SplitCompletion::Completed
                            | topology::SplitCompletion::NotDraining => Ok(Advance::Completed),
                            topology::SplitCompletion::NotDrained => {
                                Ok(Advance::Drained { moved, remaining })
                            }
                        }
                    } else {
                        Ok(Advance::Drained { moved, remaining })
                    }
                }
                DrainStep::SourceAdvanced => Ok(Advance::Completed),
                DrainStep::NotDraining => Ok(Advance::Idle),
            }
        }
        PartitionTransition::ReceivingSplit { .. } | PartitionTransition::Merging { .. } => {
            Ok(Advance::Idle)
        }
    }
}

/// Chooses the nearer of two split targets for one routing-space vector.
///
/// Both centroids are persisted routing models, so kernel errors here are
/// fail-closed Corruption rather than caller error. `balance` reserves entries
/// needed to reach the deterministic half-total target quotas.
fn nearer_target(
    kernel: &crate::search::numeric::VectorKernel,
    routing: &[f32],
    left: PartitionKey,
    left_centroid: &PartitionCentroid,
    right: PartitionKey,
    right_centroid: &PartitionCentroid,
    balance: &mut SplitBalance,
) -> Result<PartitionKey> {
    let left_distance = kernel
        .routing_distance(routing, left_centroid.components())
        .map_err(|_| Error::new(ErrorKind::Corruption))?;
    let right_distance = kernel
        .routing_distance(routing, right_centroid.components())
        .map_err(|_| Error::new(ErrorKind::Corruption))?;
    balance.choose(left, left_distance, right, right_distance)
}

/// Exact remaining counts that keep both split targets viable while preserving
/// nearest-centroid routing whenever possible.
struct SplitBalance {
    source_remaining: u64,
    left_needed: u64,
    right_needed: u64,
}

impl SplitBalance {
    fn new(source: u32, left: u32, right: u32, minimum: u32) -> Self {
        Self {
            source_remaining: u64::from(source),
            left_needed: u64::from(minimum.saturating_sub(left)),
            right_needed: u64::from(minimum.saturating_sub(right)),
        }
    }

    fn choose(
        &mut self,
        left: PartitionKey,
        left_distance: f64,
        right: PartitionKey,
        right_distance: f64,
    ) -> Result<PartitionKey> {
        self.source_remaining = self
            .source_remaining
            .checked_sub(1)
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        let attainable = self
            .left_needed
            .checked_add(self.right_needed)
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?
            <= self.source_remaining.saturating_add(1);
        let target = if attainable && self.left_needed > self.source_remaining {
            left
        } else if attainable && self.right_needed > self.source_remaining {
            right
        } else {
            routing::nearer_of_two(left, left_distance, right, right_distance)
        };
        if target == left {
            self.left_needed = self.left_needed.saturating_sub(1);
        } else {
            self.right_needed = self.right_needed.saturating_sub(1);
        }
        Ok(target)
    }
}

/// Reads one split source's State from a snapshot, returning `None` when a
/// completed non-root split removed both authority values.
///
/// Re-driving any step of a finished split observes the absence and is
/// harmless (maintenance.md §3); a lone surviving Header is a torn committed
/// state. One batched read covers both authority values.
async fn read_source_state<T: crate::storage::backend::ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
) -> Result<Option<PartitionTransition>> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let (header, state) =
        topology::read_authority_opt(txn, manifest.logical_index_id(), tree_key, source).await?;
    match (header, state) {
        (_, Some(state)) => Ok(Some(state)),
        (Some(_), None) => Err(Error::new(ErrorKind::Corruption)),
        (None, None) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partition(value: u64) -> PartitionKey {
        PartitionKey::new(value).expect("nonzero Partition Key")
    }

    #[test]
    fn split_balance_reserves_the_minimum_after_nearest_routing_fills_one_side() {
        let left = partition(2);
        let right = partition(3);
        let mut balance = SplitBalance::new(9, 0, 0, 3);
        let mut counts = (0_u32, 0_u32);

        for _ in 0..9 {
            match balance
                .choose(left, 0.0, right, 1.0)
                .expect("one source entry remains")
            {
                target if target == left => counts.0 += 1,
                target if target == right => counts.1 += 1,
                _ => unreachable!("the policy returns one of its two targets"),
            }
        }

        assert_eq!(counts, (6, 3));
    }
}
