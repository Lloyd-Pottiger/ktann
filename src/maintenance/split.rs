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
//!   installation, one bounded corrective pull, or one schema- and
//!   Backend-budget-bounded drain batch —
//!   small enough to stay within the adapter's conservative admission budget.
//!   Draining stores no durable cursor: every batch starts at the source's current smallest
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
//!   partition, exposing and advancing a `Splitting` one, running one bounded
//!   same-level corrective pass and then draining one batch of a
//!   `DrainingSplit` one, and completing at exact count zero. Repeating
//!   `advance` converges a split; abandoning it at any point leaves a
//!   searchable state.

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::observe::labels::{FixupKind, Operation};
use crate::observe::metrics;
use crate::runtime::RetryPolicy;
use crate::runtime::{reads, writes};
use crate::storage::backend::{Backend, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionTransition, PersistentValue, expect_centroid, expect_child_entry_ref, expect_header,
};
use crate::storage::{LogicalRange, LogicalReader, ReadLogicalTxn, WriteLogicalTxn, topology};

pub use super::drain::DRAIN_BATCH_INTERNAL;
use super::drain::{self, DrainBatch};
use super::routing;
use super::training;

/// Aggregate routing-vector budget for one corrective pull step.
const CORRECTIVE_PULL_SCREEN_ITEMS: usize = 8;

/// Maximum Ready candidate pages inspected by one corrective pull step.
const CORRECTIVE_PULL_CANDIDATES: usize = 4;

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
    /// One same-level corrective pull batch committed into the split targets.
    Corrected {
        /// The number of entries moved from a Ready sibling or cousin.
        moved: usize,
    },
    /// One drain batch committed.
    Drained {
        /// The number of entries this batch moved.
        moved: usize,
        /// The source's exact entry count after the batch.
        remaining: u32,
    },
    /// The split completed and promoted both targets to `Ready`.
    Completed {
        /// The promoted left target.
        left: PartitionKey,
        /// The promoted right target.
        right: PartitionKey,
        /// The updated parent, or `None` when the root split in place.
        parent: Option<PartitionKey>,
    },
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

/// Moves one bounded batch of source entries to their nearer split target or a
/// strictly better Ready sibling.
///
/// A short read snapshot first fixes the batch: the source's current smallest
/// entries, using the largest safe Leaf Entry batch for the current schema and
/// Backend budget or at most [`DRAIN_BATCH_INTERNAL`] Child Entries. The write
/// transaction revalidates the `DrainingSplit` state, then re-reads each
/// candidate with update protection: an entry removed by a concurrent
/// completed mutation is skipped and a remaining membership mismatch is
/// Corruption. Every remaining entry routes against the two persisted target
/// centroids — or to a bounded same-level `Ready` candidate with remaining
/// capacity when that candidate is strictly closer and both target minima stay
/// attainable — and moves atomically — Leaf Entries with their Record Location
/// and target Synopsis, Child Entries with exact counts alone (ADR 0014). Exact
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
    drain_batch_with_discovery(backend, manifest, tree_key, source, retry, None).await
}

/// Runs a drain while reusing a preceding no-pull maintenance-beam snapshot.
async fn drain_batch_with_discovery<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    retry: &RetryPolicy,
    prepared: Option<CorrectionDiscovery>,
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
    let movement = topology::Movement::Split { left, right };
    let Some(batch) = drain::next_drain_batch(
        &mut read,
        manifest,
        tree_key,
        source,
        Some(source_header),
        movement,
        backend.admission_budget(),
    )
    .await?
    else {
        return Ok(DrainStep::Drained {
            moved: 0,
            remaining: 0,
        });
    };
    let discovery = if let Some(prepared) = prepared {
        prepared
    } else {
        CorrectionDiscovery::discover(
            &mut read,
            manifest,
            tree_key,
            source_header.level(),
            source,
            left,
            right,
        )
        .await?
    };
    drop(read);

    let plan = DrainPlan { batch, discovery };
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
    discovery: CorrectionDiscovery,
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
    let [left, right] = plan.discovery.targets.keys();
    let Some((source_header, state)) =
        topology::partition_authority_for_update(txn, tree_key, source).await?
    else {
        return Ok(DrainStep::SourceAdvanced);
    };
    match state {
        // The DrainingSplit target pair is fixed at the transition, so a
        // different pair contradicts the persisted protocol.
        state if state.is_draining_split_of(left, right) => {}
        PartitionTransition::DrainingSplit { .. } => {
            return Err(Error::new(ErrorKind::Corruption));
        }
        _ => return Ok(DrainStep::SourceAdvanced),
    }
    // Protect the exact target counts that shape this batch; relocation reuses
    // these cached Headers through commit. The immutable centroid reads stay
    // plain so unrelated target lifecycle work does not add conflicts.
    let mut header_values = txn
        .batch_get_for_update(
            [left, right]
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

    let mut balance = SplitBalance::new(
        source_header.entry_count(),
        left_header.entry_count(),
        right_header.entry_count(),
        manifest.config().min_partition_entries(),
    );
    let mut ready_capacity = ready_target_capacity(
        txn,
        manifest,
        tree_key,
        source_header.level(),
        &plan.discovery.candidates,
    )
    .await?;
    let targets = &plan.discovery.targets;
    let moved = match &plan.batch {
        DrainBatch::Leaf(record_ids) => {
            let candidates =
                topology::read_leaf_drain_candidates(txn, tree_key, source, record_ids).await?;
            let mut moves = Vec::new();
            for candidate in candidates.into_iter().flatten() {
                // A `None` slot is a concurrently removed entry: skipped.
                let routing = plan
                    .discovery
                    .kernel
                    .preprocess(candidate.record().vector())
                    .map_err(|_| Error::new(ErrorKind::Corruption))?;
                let target = corrective_split_target(
                    &plan.discovery.kernel,
                    &routing,
                    targets,
                    &plan.discovery.candidates,
                    &mut ready_capacity,
                    &mut balance,
                )?;
                moves.push((candidate, target));
            }
            topology::relocate_leaf_entries(
                txn,
                tree_key,
                source,
                moves,
                topology::Movement::Split { left, right },
            )
            .await?
        }
        DrainBatch::Child(children) => {
            let candidates =
                topology::read_child_drain_candidates(txn, tree_key, source, children).await?;
            let mut moves = Vec::new();
            for entry in candidates.into_iter().flatten() {
                let target = corrective_split_target(
                    &plan.discovery.kernel,
                    entry.centroid(),
                    targets,
                    &plan.discovery.candidates,
                    &mut ready_capacity,
                    &mut balance,
                )?;
                moves.push((entry, target));
            }
            topology::relocate_child_entries(
                txn,
                tree_key,
                source,
                moves,
                topology::Movement::Split { left, right },
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
        PartitionTransition::DrainingSplit { left, right, .. } => {
            let correction =
                corrective_pull_batch(backend, manifest, tree_key, partition, left, right, retry)
                    .await?;
            let prepared = match correction {
                CorrectiveStep::NoMove(prepared) => prepared,
                CorrectiveStep::Moved { moved } => {
                    metrics::fixup_drain_step(FixupKind::Split, moved);
                    return Ok(Advance::Corrected { moved });
                }
            };
            match drain_batch_with_discovery(
                backend, manifest, tree_key, partition, retry, prepared,
            )
            .await?
            {
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
                            topology::SplitCompletion::Completed { parent } => {
                                Ok(Advance::Completed {
                                    left,
                                    right,
                                    parent,
                                })
                            }
                            topology::SplitCompletion::NotDraining => Ok(Advance::Completed {
                                left,
                                right,
                                parent: None,
                            }),
                            topology::SplitCompletion::NotDrained => {
                                Ok(Advance::Drained { moved, remaining })
                            }
                        }
                    } else {
                        Ok(Advance::Drained { moved, remaining })
                    }
                }
                DrainStep::SourceAdvanced => Ok(Advance::Completed {
                    left,
                    right,
                    parent: None,
                }),
                DrainStep::NotDraining => Ok(Advance::Idle),
            }
        }
        PartitionTransition::ReceivingSplit { .. } | PartitionTransition::Merging { .. } => {
            Ok(Advance::Idle)
        }
    }
}

/// Outcome of one bounded sibling/cousin pull attempt.
enum CorrectiveStep {
    /// One atomic movement batch committed.
    Moved { moved: usize },
    /// Nothing moved: a root split, stale family state, a zero source count,
    /// full targets, no donor candidate, or no sampled entry with a strictly
    /// better split target.
    NoMove(Option<CorrectionDiscovery>),
}

/// Reusable same-snapshot routing data from a no-pull corrective pass.
struct CorrectionDiscovery {
    kernel: crate::search::numeric::VectorKernel,
    targets: SplitTargets,
    candidates: Vec<topology::LevelCandidate>,
}

impl CorrectionDiscovery {
    /// Reads the reusable routing data for one corrective pass from a snapshot.
    async fn discover<R: LogicalReader>(
        reader: &mut R,
        manifest: &IndexManifest,
        tree_key: &TreeKey,
        level: u32,
        source: PartitionKey,
        left: PartitionKey,
        right: PartitionKey,
    ) -> Result<Self> {
        let kernel = routing::kernel_for(manifest)?;
        let (left_centroid, right_centroid) =
            read_target_centroids(reader, manifest, tree_key, left, right).await?;
        let targets = SplitTargets::new(left, left_centroid, right, right_centroid);
        // A root split has no same-level siblings to correct against.
        let candidates = if source == topology::root_partition() {
            Vec::new()
        } else {
            corrective_candidates(reader, manifest, tree_key, level, source, &targets, &kernel)
                .await?
        };
        Ok(Self {
            kernel,
            targets,
            candidates,
        })
    }
}

/// The persisted identities and routing models of one split's targets.
struct SplitTargets {
    left: (PartitionKey, PartitionCentroid),
    right: (PartitionKey, PartitionCentroid),
}

impl SplitTargets {
    /// Builds the pair in its canonical left/right order.
    fn new(
        left: PartitionKey,
        left_centroid: PartitionCentroid,
        right: PartitionKey,
        right_centroid: PartitionCentroid,
    ) -> Self {
        Self {
            left: (left, left_centroid),
            right: (right, right_centroid),
        }
    }

    /// Returns both persisted target Partition Keys.
    fn keys(&self) -> [PartitionKey; 2] {
        [self.left.0, self.right.0]
    }

    /// Returns both target routing models in canonical order.
    fn iter(&self) -> impl Iterator<Item = (PartitionKey, &[f32])> {
        [
            (self.left.0, self.left.1.components()),
            (self.right.0, self.right.1.components()),
        ]
        .into_iter()
    }
}

/// Pulls one bounded same-level Ready batch into a strictly better split target.
async fn corrective_pull_batch<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    split_source: PartitionKey,
    left: PartitionKey,
    right: PartitionKey,
    retry: &RetryPolicy,
) -> Result<CorrectiveStep> {
    if split_source == topology::root_partition() {
        return Ok(CorrectiveStep::NoMove(None));
    }
    let mut read = reads::open_validated_read(backend, manifest).await?;
    let Some((split_header, split_state)) = topology::read_authority_pair(
        &mut read,
        manifest.logical_index_id(),
        tree_key,
        split_source,
    )
    .await?
    else {
        return Ok(CorrectiveStep::NoMove(None));
    };
    if !split_state.is_draining_split_of(left, right) {
        return Ok(CorrectiveStep::NoMove(None));
    }
    if split_header.entry_count() == 0 {
        return Ok(CorrectiveStep::NoMove(None));
    }
    let discovery = CorrectionDiscovery::discover(
        &mut read,
        manifest,
        tree_key,
        split_header.level(),
        split_source,
        left,
        right,
    )
    .await?;
    if discovery.candidates.is_empty() {
        return Ok(CorrectiveStep::NoMove(Some(discovery)));
    }
    let movement = topology::Movement::CorrectivePull {
        split_source,
        left,
        right,
    };
    let capacity = read_corrective_capacity(
        &mut read,
        manifest,
        tree_key,
        split_header.level(),
        left,
        right,
    )
    .await?;
    if capacity.iter().all(|(_, remaining)| *remaining == 0) {
        return Ok(CorrectiveStep::NoMove(Some(discovery)));
    }
    let source_batch_limit = drain::relocation_batch_limit(
        manifest,
        tree_key,
        split_header,
        topology::Movement::Split { left, right },
        backend.admission_budget(),
    )?;
    let scan_round = usize::try_from(split_header.entry_count())
        .map_err(|_| Error::new(ErrorKind::Corruption))?
        .div_ceil(source_batch_limit);
    let targets = &discovery.targets;
    let donors: Vec<_> = discovery
        .candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.header().level() == 1 || candidate.header().entry_count() > 1
        })
        .collect();
    if donors.is_empty() {
        return Ok(CorrectiveStep::NoMove(Some(discovery)));
    }
    let mut selected = None;
    let candidate_count = donors.len().min(CORRECTIVE_PULL_CANDIDATES);
    let first_candidate = scan_round
        .saturating_sub(1)
        .checked_mul(candidate_count)
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?
        % donors.len();
    for offset in 0..candidate_count {
        let donor = (first_candidate + offset) % donors.len();
        let (position, candidate) = donors[donor];
        let scan = CorrectivePullScan {
            manifest,
            tree_key,
            targets,
            kernel: &discovery.kernel,
            target_capacity: &capacity,
            screen_limit: CORRECTIVE_PULL_SCREEN_ITEMS / candidate_count
                + usize::from(offset < CORRECTIVE_PULL_SCREEN_ITEMS % candidate_count),
            round: scan_round,
        };
        let batch_limit = drain::relocation_batch_limit(
            manifest,
            tree_key,
            *candidate.header(),
            movement,
            backend.admission_budget(),
        )?;
        let batch_limit = if candidate.header().level() == 1 {
            batch_limit
        } else {
            let retained_source_entry = usize::try_from(
                candidate
                    .header()
                    .entry_count()
                    .checked_sub(1)
                    .ok_or_else(|| Error::new(ErrorKind::Corruption))?,
            )
            .map_err(|_| Error::new(ErrorKind::Corruption))?;
            batch_limit.min(retained_source_entry)
        };
        let batch = corrective_candidate_batch(&mut read, candidate, batch_limit, &scan).await?;
        if let Some(batch) = batch {
            selected = Some((position, batch));
            break;
        }
    }
    let Some((candidate, batch)) = selected else {
        return Ok(CorrectiveStep::NoMove(Some(discovery)));
    };
    let plan = CorrectivePlan {
        candidate,
        batch,
        discovery,
    };
    drop(read);

    let attempt = writes::run_write_attempts(
        backend,
        None,
        manifest,
        retry,
        Operation::SplitFixup,
        |txn| {
            writes::boxed_step(corrective_pull_attempt(
                txn,
                manifest,
                tree_key,
                split_source,
                &plan,
            ))
        },
    )
    .await?;
    Ok(if attempt == 0 {
        CorrectiveStep::NoMove(Some(plan.discovery))
    } else {
        CorrectiveStep::Moved { moved: attempt }
    })
}

/// Immutable policy and routing data for one corrective candidate scan.
struct CorrectivePullScan<'a> {
    manifest: &'a IndexManifest,
    tree_key: &'a TreeKey,
    targets: &'a SplitTargets,
    kernel: &'a crate::search::numeric::VectorKernel,
    target_capacity: &'a [(PartitionKey, u32); 2],
    screen_limit: usize,
    round: usize,
}

/// Selects one budget-safe qualifying batch from a candidate's bounded page.
async fn corrective_candidate_batch<R: LogicalReader>(
    reader: &mut R,
    candidate: &topology::LevelCandidate,
    batch_limit: usize,
    scan: &CorrectivePullScan<'_>,
) -> Result<Option<DrainBatch>> {
    let capacity_limit = scan
        .target_capacity
        .iter()
        .try_fold(0_usize, |total, (_, remaining)| {
            total.checked_add(*remaining as usize)
        })
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    let selection_limit = batch_limit.min(capacity_limit);
    if selection_limit == 0 || scan.screen_limit == 0 || scan.round == 0 {
        return Err(Error::new(ErrorKind::Corruption));
    }
    let source = candidate.partition();
    let range = if candidate.header().level() == 1 {
        LogicalRange::leaf_entries(scan.manifest, scan.tree_key, source)?
    } else {
        LogicalRange::child_entries(scan.manifest, scan.tree_key, source)?
    };
    let page = reader.scan(&range, None, drain::DRAIN_SCAN_LIMITS).await?;
    if page.items().is_empty() {
        return Err(Error::new(ErrorKind::Corruption));
    }
    let sampled = corrective_sample_indices(page.items().len(), scan.screen_limit, scan.round);
    let mut capacity = *scan.target_capacity;
    if candidate.header().level() == 1 {
        let record_ids: Vec<_> = sampled
            .iter()
            .map(|position| &page.items()[*position])
            .map(|item| match item.value() {
                PersistentValue::LeafEntry(entry) => Ok(entry.record_id().clone()),
                _ => Err(Error::new(ErrorKind::Corruption)),
            })
            .collect::<Result<_>>()?;
        let mut selected = Vec::with_capacity(selection_limit.min(record_ids.len()));
        let keys = record_ids
            .iter()
            .map(|id| LogicalKey::Record {
                index: scan.manifest.logical_index_id(),
                id: id.clone(),
            })
            .collect();
        let records = reader.batch_get(keys).await?;
        if records.len() != record_ids.len() {
            return Err(Error::new(ErrorKind::Backend));
        }
        for (record_id, value) in record_ids.iter().zip(records) {
            let Some(PersistentValue::VectorRecord(record)) = value else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            if record.record_id() != record_id {
                return Err(Error::new(ErrorKind::Corruption));
            }
            let routing = scan
                .kernel
                .preprocess(record.vector())
                .map_err(|_| Error::new(ErrorKind::Corruption))?;
            if corrective_pull_target(
                scan.kernel,
                &routing,
                candidate.centroid(),
                scan.targets,
                &mut capacity,
            )?
            .is_some()
            {
                selected.push(record_id.clone());
                if selected.len() == selection_limit {
                    break;
                }
            }
        }
        return Ok((!selected.is_empty()).then_some(DrainBatch::Leaf(selected)));
    }

    let mut selected = Vec::with_capacity(selection_limit.min(sampled.len()));
    for position in sampled {
        let item = &page.items()[position];
        let entry = expect_child_entry_ref(item.value())?;
        if corrective_pull_target(
            scan.kernel,
            entry.centroid(),
            candidate.centroid(),
            scan.targets,
            &mut capacity,
        )?
        .is_some()
        {
            selected.push(entry.child());
            if selected.len() == selection_limit {
                break;
            }
        }
    }
    Ok((!selected.is_empty()).then_some(DrainBatch::Child(selected)))
}

/// Spreads one rotating deterministic sample across a bounded entry page.
fn corrective_sample_indices(length: usize, limit: usize, round: usize) -> Vec<usize> {
    let count = length.min(limit);
    let offset = round.saturating_sub(1) % length;
    if count == 1 {
        return vec![offset];
    }
    (0..count)
        .map(|sample| (offset + sample * (length - 1) / (count - 1)) % length)
        .collect()
}

/// Reads the split targets' remaining corrective capacity from one snapshot.
async fn read_corrective_capacity<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    level: u32,
    left: PartitionKey,
    right: PartitionKey,
) -> Result<[(PartitionKey, u32); 2]> {
    let keys = [left, right]
        .map(|partition| LogicalKey::Header {
            index: manifest.logical_index_id(),
            tree_key: tree_key.clone(),
            partition,
        })
        .into();
    let headers = reader.batch_get(keys).await?;
    if headers.len() != 2 {
        return Err(Error::new(ErrorKind::Backend));
    }
    let maximum = manifest.config().max_partition_entries();
    let mut capacity = [(left, 0), (right, 0)];
    for ((_, remaining), value) in capacity.iter_mut().zip(headers) {
        let header = expect_header(value)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        if header.level() != level || header.state() != PartitionState::ReceivingSplit {
            return Err(Error::new(ErrorKind::Corruption));
        }
        *remaining = maximum.saturating_sub(header.entry_count());
    }
    Ok(capacity)
}

/// One bounded corrective pull plan fixed by a consistent snapshot.
struct CorrectivePlan {
    candidate: usize,
    batch: DrainBatch,
    discovery: CorrectionDiscovery,
}

/// Revalidates and atomically applies one corrective pull plan.
async fn corrective_pull_attempt<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    split_source: PartitionKey,
    plan: &CorrectivePlan,
) -> Result<usize> {
    let index = manifest.logical_index_id();
    let candidate = plan
        .discovery
        .candidates
        .get(plan.candidate)
        .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    let source = candidate.partition();
    let [left, right] = plan.discovery.targets.keys();
    let Some((source_header, source_state)) =
        topology::partition_authority_for_update(txn, tree_key, source).await?
    else {
        return Ok(0);
    };
    if !matches!(source_state, PartitionTransition::Ready { .. }) {
        return Ok(0);
    }

    let Some((split_header, split_state)) =
        topology::partition_authority_for_update(txn, tree_key, split_source).await?
    else {
        return Ok(0);
    };
    if split_header.level() != source_header.level() {
        return Err(Error::new(ErrorKind::Corruption));
    }
    if !split_state.is_draining_split_of(left, right) {
        return Ok(0);
    }

    let mut target_headers = txn
        .batch_get_for_update(
            [left, right]
                .map(|partition| LogicalKey::Header {
                    index,
                    tree_key: tree_key.clone(),
                    partition,
                })
                .into(),
        )
        .await?
        .into_iter();
    let (Some(left_header), Some(right_header)) = (target_headers.next(), target_headers.next())
    else {
        return Err(Error::new(ErrorKind::Backend));
    };
    let (Some(left_header), Some(right_header)) =
        (expect_header(left_header)?, expect_header(right_header)?)
    else {
        return Ok(0);
    };
    if [left_header, right_header].iter().any(|header| {
        header.level() != source_header.level() || header.state() != PartitionState::ReceivingSplit
    }) {
        return Ok(0);
    }
    let maximum = manifest.config().max_partition_entries();
    let mut capacity = [
        (left, maximum.saturating_sub(left_header.entry_count())),
        (right, maximum.saturating_sub(right_header.entry_count())),
    ];
    if capacity.iter().all(|(_, remaining)| *remaining == 0) {
        return Ok(0);
    }
    let movement = topology::Movement::CorrectivePull {
        split_source,
        left,
        right,
    };
    let targets = &plan.discovery.targets;
    let moved = match &plan.batch {
        DrainBatch::Leaf(record_ids) => {
            let entries =
                topology::read_leaf_drain_candidates(txn, tree_key, source, record_ids).await?;
            let mut moves = Vec::new();
            for entry in entries.into_iter().flatten() {
                let routing = plan
                    .discovery
                    .kernel
                    .preprocess(entry.record().vector())
                    .map_err(|_| Error::new(ErrorKind::Corruption))?;
                if let Some(target) = corrective_pull_target(
                    &plan.discovery.kernel,
                    &routing,
                    candidate.centroid(),
                    targets,
                    &mut capacity,
                )? {
                    moves.push((entry, target));
                }
            }
            topology::relocate_leaf_entries(txn, tree_key, source, moves, movement).await?
        }
        DrainBatch::Child(children) => {
            let move_limit = usize::try_from(source_header.entry_count().saturating_sub(1))
                .map_err(|_| Error::new(ErrorKind::Corruption))?;
            if move_limit == 0 {
                return Ok(0);
            }
            let entries =
                topology::read_child_drain_candidates(txn, tree_key, source, children).await?;
            let mut moves = Vec::new();
            for entry in entries.into_iter().flatten() {
                if moves.len() == move_limit {
                    break;
                }
                if let Some(target) = corrective_pull_target(
                    &plan.discovery.kernel,
                    entry.centroid(),
                    candidate.centroid(),
                    targets,
                    &mut capacity,
                )? {
                    moves.push((entry, target));
                }
            }
            topology::relocate_child_entries(txn, tree_key, source, moves, movement).await?
        }
    };
    Ok(moved)
}

/// Chooses the nearer of two split targets for one routing-space vector.
///
/// Both centroids are persisted routing models, so kernel errors here are
/// fail-closed Corruption rather than caller error. `balance` reserves entries
/// needed to reach the deterministic half-total target quotas.
fn corrective_split_target(
    kernel: &crate::search::numeric::VectorKernel,
    routing: &[f32],
    targets: &SplitTargets,
    corrective_candidates: &[topology::LevelCandidate],
    ready_capacity: &mut BTreeMap<PartitionKey, u32>,
    balance: &mut SplitBalance,
) -> Result<PartitionKey> {
    let (left, left_centroid) = (targets.left.0, &targets.left.1);
    let (right, right_centroid) = (targets.right.0, &targets.right.1);
    let left_distance = kernel
        .routing_distance(routing, left_centroid.components())
        .map_err(|_| Error::new(ErrorKind::Corruption))?;
    let right_distance = kernel
        .routing_distance(routing, right_centroid.components())
        .map_err(|_| Error::new(ErrorKind::Corruption))?;
    let current_distance = left_distance.min(right_distance);
    let best_ready = strictly_better_owner(
        kernel,
        routing,
        current_distance,
        corrective_candidates.iter().filter_map(|candidate| {
            (ready_capacity
                .get(&candidate.partition())
                .copied()
                .unwrap_or(0)
                > 0)
            .then_some((candidate.partition(), candidate.centroid()))
        }),
    )?;
    if let Some(target) = best_ready {
        if balance.release_to_ready()? {
            let capacity = ready_capacity
                .get_mut(&target)
                .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
            *capacity = capacity
                .checked_sub(1)
                .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
            return Ok(target);
        }
    }
    balance.choose(left, left_distance, right, right_distance)
}

/// Chooses the nearest candidate only when it strictly improves ownership.
fn strictly_better_owner<'a>(
    kernel: &crate::search::numeric::VectorKernel,
    routing: &[f32],
    current_distance: f64,
    candidates: impl Iterator<Item = (PartitionKey, &'a [f32])>,
) -> Result<Option<PartitionKey>> {
    let mut best: Option<(f64, PartitionKey)> = None;
    for (partition, centroid) in candidates {
        let distance = kernel
            .routing_distance(routing, centroid)
            .map_err(|_| Error::new(ErrorKind::Corruption))?;
        if best
            .as_ref()
            .is_none_or(|&(best_distance, best_partition)| {
                routing::nearer_of_two(partition, distance, best_partition, best_distance)
                    == partition
            })
        {
            best = Some((distance, partition));
        }
    }
    Ok(best
        .filter(|(distance, _)| *distance < current_distance)
        .map(|(_, partition)| partition))
}

/// Selects a split target only when it strictly beats the current Ready owner.
fn corrective_pull_target(
    kernel: &crate::search::numeric::VectorKernel,
    routing: &[f32],
    source_centroid: &[f32],
    targets: &SplitTargets,
    capacity: &mut [(PartitionKey, u32); 2],
) -> Result<Option<PartitionKey>> {
    let current_distance = kernel
        .routing_distance(routing, source_centroid)
        .map_err(|_| Error::new(ErrorKind::Corruption))?;
    let target = strictly_better_owner(
        kernel,
        routing,
        current_distance,
        targets.iter().filter(|(partition, _)| {
            capacity
                .iter()
                .any(|(candidate, remaining)| candidate == partition && *remaining > 0)
        }),
    )?;
    if let Some(target) = target {
        let (_, remaining) = capacity
            .iter_mut()
            .find(|(candidate, _)| *candidate == target)
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        *remaining = remaining
            .checked_sub(1)
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    }
    Ok(target)
}

/// Loads the immutable target centroids from one consistent read snapshot.
async fn read_target_centroids<R: LogicalReader>(
    txn: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    left: PartitionKey,
    right: PartitionKey,
) -> Result<(PartitionCentroid, PartitionCentroid)> {
    let index = manifest.logical_index_id();
    let mut values = txn
        .batch_get(
            [left, right]
                .map(|partition| LogicalKey::Centroid {
                    index,
                    tree_key: tree_key.clone(),
                    partition,
                })
                .into(),
        )
        .await?
        .into_iter();
    let (Some(left), Some(right)) = (values.next(), values.next()) else {
        return Err(Error::new(ErrorKind::Backend));
    };
    Ok((
        expect_centroid(left)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?,
        expect_centroid(right)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?,
    ))
}

/// One internal edge retained by the split correction's maintenance beam.
#[derive(Clone)]
struct CorrectiveBeamEntry {
    distance: f64,
    parent: PartitionKey,
    entry: ChildEntry,
}

/// The maintenance beam's total order: nearer distance, then child, then parent.
fn corrective_beam_order(
    left: (f64, PartitionKey, PartitionKey),
    right: (f64, PartitionKey, PartitionKey),
) -> std::cmp::Ordering {
    crate::search::numeric::compare_finite(left.0, right.0)
        .then_with(|| left.1.cmp(&right.1))
        .then_with(|| left.2.cmp(&right.2))
}

impl Ord for CorrectiveBeamEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        corrective_beam_order(
            (self.distance, self.entry.child(), self.parent),
            (other.distance, other.entry.child(), other.parent),
        )
    }
}

impl PartialOrd for CorrectiveBeamEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CorrectiveBeamEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for CorrectiveBeamEntry {}

/// Finds a bounded same-level candidate set with an independent maintenance beam.
async fn corrective_candidates<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    level: u32,
    source: PartitionKey,
    targets: &SplitTargets,
    kernel: &crate::search::numeric::VectorKernel,
) -> Result<Vec<topology::LevelCandidate>> {
    let index = manifest.logical_index_id();
    let root = topology::root_partition();
    let (root_header, _) = topology::read_authority(reader, index, tree_key, root).await?;
    if root_header.level() <= level {
        return if root == source {
            Ok(Vec::new())
        } else {
            Err(Error::new(ErrorKind::Corruption))
        };
    }

    let excluded = [source, targets.left.0, targets.right.0];
    let mut current_level = root_header.level();
    let mut frontier = vec![root];
    loop {
        let next_level = current_level
            .checked_sub(1)
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        let mut scanned = BTreeSet::new();
        let mut nearest = BinaryHeap::with_capacity(topology::CORRECTIVE_PARTITION_BEAM);
        let scan = CorrectiveScan {
            manifest,
            tree_key,
            expected_level: current_level,
            excluded: (next_level == level).then_some(&excluded),
            targets,
            kernel,
        };
        'frontier: for partition in frontier {
            if scanned.contains(&partition) {
                continue;
            }
            for (body, header) in
                corrective_family_bodies(reader, manifest, tree_key, partition, current_level)
                    .await?
            {
                if !scanned.contains(&body) && scanned.len() == topology::CORRECTIVE_PARTITION_BEAM
                {
                    break 'frontier;
                }
                if !scanned.insert(body) {
                    continue;
                }
                scan_corrective_body(reader, body, header, &scan, &mut nearest).await?;
            }
        }
        if nearest.is_empty() {
            return Ok(Vec::new());
        }
        let selected = nearest.into_sorted_vec();
        if next_level == level {
            let keys = selected
                .iter()
                .map(|candidate| LogicalKey::Header {
                    index,
                    tree_key: tree_key.clone(),
                    partition: candidate.entry.child(),
                })
                .collect();
            let headers = reader.batch_get(keys).await?;
            let mut candidates = Vec::with_capacity(selected.len());
            for (candidate, value) in selected.into_iter().zip(headers) {
                let header =
                    expect_header(value)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
                if header.level() != level {
                    return Err(Error::new(ErrorKind::Corruption));
                }
                if header.state() == PartitionState::Ready && header.entry_count() > 0 {
                    candidates.push(topology::LevelCandidate::new(
                        candidate.parent,
                        candidate.entry,
                        header,
                    ));
                }
            }
            return Ok(candidates);
        }
        frontier = selected
            .into_iter()
            .map(|candidate| candidate.entry.child())
            .collect();
        current_level = next_level;
    }
}

/// Returns the searchable bodies represented by one transitional partition.
async fn corrective_family_bodies<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    expected_level: u32,
) -> Result<Vec<(PartitionKey, PartitionHeader)>> {
    let index = manifest.logical_index_id();
    let (header, state) = topology::read_authority(reader, index, tree_key, partition).await?;
    if header.level() != expected_level {
        return Err(Error::new(ErrorKind::Corruption));
    }
    match state {
        source_state @ PartitionTransition::DrainingSplit { .. } => {
            topology::split_family_bodies(
                reader,
                index,
                tree_key,
                partition,
                header,
                source_state,
                expected_level,
            )
            .await
        }
        PartitionTransition::ReceivingSplit { source, .. } => {
            let (source_header, source_state) =
                topology::read_authority(reader, index, tree_key, source).await?;
            if source_header.level() != expected_level {
                return Err(Error::new(ErrorKind::Corruption));
            }
            match source_state {
                PartitionTransition::Splitting { left, right, .. }
                    if partition == left || partition == right =>
                {
                    Ok(vec![(source, source_header)])
                }
                source_state @ PartitionTransition::DrainingSplit { left, right, .. }
                    if partition == left || partition == right =>
                {
                    topology::split_family_bodies(
                        reader,
                        index,
                        tree_key,
                        source,
                        source_header,
                        source_state,
                        expected_level,
                    )
                    .await
                }
                _ => Err(Error::new(ErrorKind::Corruption)),
            }
        }
        PartitionTransition::Ready { .. }
        | PartitionTransition::Splitting { .. }
        | PartitionTransition::Merging { .. } => Ok(vec![(partition, header)]),
    }
}

/// Immutable inputs shared by every body expansion in one beam level.
struct CorrectiveScan<'a> {
    manifest: &'a IndexManifest,
    tree_key: &'a TreeKey,
    expected_level: u32,
    excluded: Option<&'a [PartitionKey; 3]>,
    targets: &'a SplitTargets,
    kernel: &'a crate::search::numeric::VectorKernel,
}

/// Expands one bounded internal body into the nearest maintenance-beam edges.
async fn scan_corrective_body<R: LogicalReader>(
    reader: &mut R,
    body: PartitionKey,
    header: PartitionHeader,
    scan: &CorrectiveScan<'_>,
    nearest: &mut BinaryHeap<CorrectiveBeamEntry>,
) -> Result<()> {
    if header.level() != scan.expected_level || header.level() == 1 {
        return Err(Error::new(ErrorKind::Corruption));
    }
    let range = LogicalRange::child_entries(scan.manifest, scan.tree_key, body)?;
    let page = reader.scan(&range, None, drain::DRAIN_SCAN_LIMITS).await?;
    for item in page.items() {
        let entry = expect_child_entry_ref(item.value())?;
        if scan
            .excluded
            .is_some_and(|partitions| partitions.contains(&entry.child()))
        {
            continue;
        }
        let left_distance = scan
            .kernel
            .routing_distance(entry.centroid(), scan.targets.left.1.components())
            .map_err(|_| Error::new(ErrorKind::Corruption))?;
        let right_distance = scan
            .kernel
            .routing_distance(entry.centroid(), scan.targets.right.1.components())
            .map_err(|_| Error::new(ErrorKind::Corruption))?;
        let distance = left_distance.min(right_distance);
        let admitted = nearest.len() < topology::CORRECTIVE_PARTITION_BEAM
            || nearest.peek().is_some_and(|worst| {
                corrective_beam_order(
                    (distance, entry.child(), body),
                    (worst.distance, worst.entry.child(), worst.parent),
                )
                .is_lt()
            });
        if admitted {
            if nearest.len() == topology::CORRECTIVE_PARTITION_BEAM {
                nearest.pop();
            }
            nearest.push(CorrectiveBeamEntry {
                distance,
                parent: body,
                entry: entry.clone(),
            });
        }
    }
    Ok(())
}

/// Reads current Ready target counts and returns remaining corrective capacity.
async fn ready_target_capacity<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    level: u32,
    candidates: &[topology::LevelCandidate],
) -> Result<BTreeMap<PartitionKey, u32>> {
    let maximum = manifest.config().max_partition_entries();
    let keys: Vec<_> = candidates
        .iter()
        .map(|candidate| LogicalKey::Header {
            index: manifest.logical_index_id(),
            tree_key: tree_key.clone(),
            partition: candidate.partition(),
        })
        .collect();
    let mut capacity = BTreeMap::new();
    for (candidate, value) in candidates.iter().zip(txn.batch_get(keys).await?) {
        let Some(header) = expect_header(value)? else {
            continue;
        };
        if header.level() != level {
            return Err(Error::new(ErrorKind::Corruption));
        }
        if header.state() == PartitionState::Ready && header.entry_count() < maximum {
            capacity.insert(candidate.partition(), maximum - header.entry_count());
        }
    }
    Ok(capacity)
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

    /// Reserves one source entry for a Ready sibling only when both split
    /// target minima remain attainable afterwards.
    fn release_to_ready(&mut self) -> Result<bool> {
        let Some(remaining) = self.source_remaining.checked_sub(1) else {
            return Err(Error::new(ErrorKind::Corruption));
        };
        let needed = self
            .left_needed
            .checked_add(self.right_needed)
            .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        if needed > remaining {
            return Ok(false);
        }
        self.source_remaining = remaining;
        Ok(true)
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

    #[test]
    fn split_balance_reserves_both_minima_before_releasing_to_ready() {
        let left = partition(2);
        let right = partition(3);
        let mut balance = SplitBalance::new(2, 0, 0, 1);

        assert_eq!(
            balance
                .choose(left, 0.0, right, 1.0)
                .expect("first reserved entry"),
            left
        );
        assert!(!balance.release_to_ready().expect("reservation is valid"));
        assert_eq!(
            balance
                .choose(left, 0.0, right, 1.0)
                .expect("second reserved entry"),
            right
        );

        let mut releasable = SplitBalance::new(3, 0, 0, 1);
        assert!(
            releasable
                .release_to_ready()
                .expect("one surplus entry is releasable")
        );
    }

    #[test]
    fn corrective_samples_rotate_across_the_whole_page() {
        assert_eq!(corrective_sample_indices(10, 4, 1), vec![0, 3, 6, 9]);
        assert_eq!(corrective_sample_indices(10, 4, 2), vec![1, 4, 7, 0]);
        assert_eq!(corrective_sample_indices(4, 8, 1), vec![0, 1, 2, 3]);
    }
}
