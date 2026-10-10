//! Bounded, deduplicating process-local Structure Maintenance scheduling.
//!
//! One Logical Index ID, Tree Key, and Partition Key occupies one admission
//! slot across pending and running work. Queue capacity, worker count, retry
//! backoff, and steps per execution are bounded. Duplicate and saturated
//! offers are counted; lost work is rediscovered from searchable durable
//! states by later relevant accesses (ADR 0006).
//!
//! Workers settle completed work, stall merges without a legal target,
//! yield successful unfinished work to the queue tail, and retire errors or
//! cancelled work without changing durable state. A completed split replaces
//! its admission with bounded offers for both targets and the updated parent
//! in one queue transition, then publishes the final backlog.
//!
//! Shutdown cancels pending work and lets an admitted bounded transaction
//! finish. Worker lifecycle guards keep the backend alive until all workers
//! stop. The split and merge state machines live in [`crate::maintenance`].

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, MutexGuard};

use tracing::{Instrument as _, Span};

use crate::api::{LogicalIndexId, PartitionKey, Result};
use crate::maintenance::{fixup as maintenance_fixup, merge, split};
use crate::observe::labels::{FixupAdmission, FixupExecution, FixupKind, FixupStepResult};
use crate::observe::{metrics, trace};
use crate::storage::backend::Backend;
use crate::storage::keys::TreeKey;
use crate::storage::values::IndexManifest;

use super::RuntimeInner;
use super::lifecycle::{RetryPolicy, now_unix_millis};

/// One unit of demand-driven Structure Maintenance: one partition of one tree
/// of one Logical Index.
///
/// The key identifies rediscovery, not a persisted task: execution re-reads
/// the durable state and performs whatever step comes next, so a stale or
/// repeated key is harmless. `Debug` is safe: the Tree Key's own `Debug` is
/// redacted.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FixupKey {
    index: LogicalIndexId,
    tree_key: TreeKey,
    partition: PartitionKey,
}

/// One admitted Fixup: its rediscovery key and the offering handle's Manifest.
///
/// Execution validates the persisted Active Manifest against this identity on
/// every step, so a Fixup offered before an index drop settles or retires
/// without touching the replacement index.
struct FixupOffer {
    key: FixupKey,
    manifest: Arc<IndexManifest>,
    /// A yielded worker keeps advancing without waiting for recovery age.
    continuing: bool,
}

/// Cumulative, privacy-safe observations of the process-local Fixup queue.
///
/// Counts only; the `ktann.fixup.*` facade series are emitted alongside
/// (design `runtime-operations.md` §5). Saturating arithmetic keeps the
/// counters themselves bounded and panic-free.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FixupStats {
    /// Keys that took a queue slot.
    enqueued: u64,
    /// Offers dropped because the key was already pending or running.
    duplicate: u64,
    /// Offers dropped because the queue was full.
    saturated: u64,
    /// Executions that reached a no-more-work outcome (idle or completed).
    settled: u64,
    /// Executions that stopped on a stalled merge with no legal target.
    stalled: u64,
    /// Executions that yielded after making a full step budget of progress.
    yielded: u64,
    /// Executions that stopped early because of an error or shutdown.
    retired: u64,
}

impl FixupStats {
    /// Counts one finished execution's outcome.
    fn record(&mut self, execution: FixupExecution) {
        let counter = match execution {
            FixupExecution::Settled => &mut self.settled,
            FixupExecution::Stalled => &mut self.stalled,
            FixupExecution::Yielded => &mut self.yielded,
            FixupExecution::Retired => &mut self.retired,
        };
        *counter = counter.saturating_add(1);
    }
}

/// The bounded, deduplicating process-local Fixup queue.
///
/// `admitted` holds every key with a slot, pending or running, which is what
/// makes an offer during execution coalesce instead of queueing behind the
/// running execution: the executor keeps advancing the partition within its
/// bounded step budget, then yields unfinished successful work to the queue.
pub(crate) struct FixupQueue {
    pending: VecDeque<FixupOffer>,
    admitted: HashSet<FixupKey>,
    capacity: usize,
    stats: FixupStats,
}

impl FixupQueue {
    pub(crate) fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0, "the Runtime configuration validates it");
        Self {
            pending: VecDeque::new(),
            admitted: HashSet::new(),
            capacity,
            stats: FixupStats::default(),
        }
    }

    /// The process-local maintenance backlog: pending plus running Fixups.
    fn backlog(&self) -> usize {
        self.admitted.len()
    }

    /// Admits one offered Fixup under the deduplication and capacity policy.
    ///
    /// The Manifest Arc is cloned only on the enqueue path; duplicate and
    /// saturated offers drop without touching its refcount.
    fn offer(&mut self, key: FixupKey, manifest: &Arc<IndexManifest>) -> FixupAdmission {
        if self.admitted.contains(&key) {
            self.stats.duplicate = self.stats.duplicate.saturating_add(1);
            return FixupAdmission::Duplicate;
        }
        if self.backlog() >= self.capacity {
            self.stats.saturated = self.stats.saturated.saturating_add(1);
            return FixupAdmission::Saturated;
        }
        self.admitted.insert(key.clone());
        self.pending.push_back(FixupOffer {
            key,
            manifest: Arc::clone(manifest),
            continuing: false,
        });
        self.stats.enqueued = self.stats.enqueued.saturating_add(1);
        FixupAdmission::Enqueued
    }

    /// Takes the oldest pending Fixup into running state.
    ///
    /// The key keeps its slot while running, so concurrent duplicate offers
    /// still coalesce. [`FixupQueue::finish`] releases the slot.
    fn pop(&mut self) -> Option<FixupOffer> {
        self.pending.pop_front()
    }

    /// Releases one running Fixup's slot after its execution ended, returning
    /// the remaining backlog.
    ///
    /// Called from the worker's running guard, so it also runs when the
    /// worker task is aborted mid-execution: an interrupted Fixup never leaks
    /// its admission slot.
    fn finish(&mut self, key: &FixupKey) -> usize {
        self.admitted.remove(key);
        self.backlog()
    }

    /// Returns one successfully progressing Fixup to the queue tail without
    /// releasing its bounded admission slot.
    fn yield_back(&mut self, offer: &FixupOffer) -> usize {
        debug_assert!(self.admitted.contains(&offer.key));
        self.pending.push_back(FixupOffer {
            key: offer.key.clone(),
            manifest: Arc::clone(&offer.manifest),
            continuing: true,
        });
        self.backlog()
    }

    /// Drops every pending Fixup, keeping only running admissions, and returns
    /// the remaining backlog.
    ///
    /// Shutdown cancels work that has not begun; running executions release
    /// their own slots through [`FixupQueue::finish`].
    fn drain_pending(&mut self) -> usize {
        while let Some(offer) = self.pending.pop_front() {
            self.admitted.remove(&offer.key);
        }
        self.backlog()
    }
}

impl<B: Backend> RuntimeInner<B> {
    /// Includes pending and running work; worker handoff stays under the same lock.
    /// Refinement's caller must exclude new offers from concurrent index operations.
    pub(crate) fn has_fixups(&self, index: LogicalIndexId) -> bool {
        self.lock_fixups()
            .admitted
            .iter()
            .any(|key| key.index == index)
    }

    /// Offers one batch of discovered partitions to the bounded Fixup queue.
    ///
    /// Best-effort by contract: an offer after shutdown began, a duplicate
    /// key, and a full queue all drop work that a later relevant access may
    /// rediscover. Every discovery of one batch shares the offering handle's
    /// Manifest identity.
    pub(crate) fn offer_fixups<I>(&self, manifest: &Arc<IndexManifest>, partitions: I)
    where
        I: IntoIterator<Item = (TreeKey, PartitionKey)>,
    {
        // With no workers nothing consumes the queue; drop immediately.
        if self.config().maintenance_workers() == 0 {
            return;
        }
        if !self.is_accepting() {
            return;
        }
        let index = manifest.logical_index_id();
        let mut enqueued = 0_u64;
        let mut duplicate = 0_u64;
        let mut saturated = 0_u64;
        {
            let mut queue = self.lock_fixups();
            for (tree_key, partition) in partitions {
                let key = FixupKey {
                    index,
                    tree_key,
                    partition,
                };
                match queue.offer(key, manifest) {
                    FixupAdmission::Enqueued => {
                        self.fixup_available.notify_one();
                        enqueued += 1;
                    }
                    FixupAdmission::Duplicate => duplicate += 1,
                    FixupAdmission::Saturated => saturated += 1,
                }
            }
        }
        // Facade calls stay outside the queue lock.
        report_admissions(enqueued, duplicate, saturated);
        self.report_fixup_backlog(metrics::fixup_backlog);
    }

    /// Returns the cumulative Fixup queue statistics.
    pub(crate) fn fixup_stats(&self) -> FixupStats {
        self.lock_fixups().stats
    }

    /// Starts the configured maintenance workers.
    ///
    /// Workers are counted in the Runtime's lifecycle activity from
    /// construction until they stop, so the backend is never released while a
    /// worker could still be executing a step against it.
    pub(crate) fn start_maintenance(self: &Arc<Self>) {
        for _ in 0..self.config().maintenance_workers() {
            {
                let mut lifecycle = self.lock_lifecycle();
                debug_assert!(lifecycle.is_accepting());
                lifecycle.begin_activity();
            }
            let guard = WorkerGuard {
                inner: Arc::clone(self),
            };
            drop(self.executor.spawn(run_worker(guard)));
        }
    }

    /// Stops Fixup admission and cancels queued work that has not begun.
    ///
    /// Called with the lifecycle transition to closing; running executions
    /// finish their bounded step and stop at the next cancellation point.
    pub(crate) fn stop_maintenance(&self) {
        self.maintenance_cancel.cancel();
        self.lock_fixups().drain_pending();
        self.report_fixup_backlog(metrics::fixup_backlog);
    }

    /// Returns the backend for maintenance work, or `None` once it is gone.
    fn maintenance_backend(&self) -> Option<Arc<B>> {
        let lifecycle = self.lock_lifecycle();
        if !lifecycle.is_accepting() {
            return None;
        }
        lifecycle.backend()
    }

    /// Waits for the next Fixup, or returns `None` once maintenance stopped.
    ///
    /// The notification is registered before the queue is checked so an offer
    /// landing between the check and the wait still wakes the worker.
    async fn next_fixup(&self) -> Option<FixupOffer> {
        loop {
            let notified = self.fixup_available.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.maintenance_cancel.is_cancelled() {
                self.lock_fixups().drain_pending();
                self.report_fixup_backlog(metrics::fixup_backlog);
                return None;
            }
            if let Some(offer) = self.lock_fixups().pop() {
                return Some(offer);
            }
            tokio::select! {
                biased;
                () = self.maintenance_cancel.cancelled() => {
                    self.lock_fixups().drain_pending();
                    self.report_fixup_backlog(metrics::fixup_backlog);
                    return None;
                }
                () = notified => {}
            }
        }
    }

    /// Publishes the latest backlog without racing an older gauge update.
    ///
    /// A concurrent or reentrant reporter leaves publication to the active one.
    /// Recheck after releasing the reporting guard: a queue change while the
    /// callback ran must either be covered by another reporter or trigger a
    /// fresh publication here. Recorder callbacks never hold the queue lock.
    fn report_fixup_backlog(&self, mut publish: impl FnMut(usize)) {
        loop {
            let reporting = match self.fixup_reporting.try_lock() {
                Ok(guard) => guard,
                Err(std::sync::TryLockError::WouldBlock) => return,
                // A recorder panic cannot damage the queue or this unit guard.
                Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            };
            let backlog = self.lock_fixups().backlog();
            publish(backlog);
            drop(reporting);
            if self.lock_fixups().backlog() == backlog {
                return;
            }
        }
    }

    fn lock_fixups(&self) -> MutexGuard<'_, FixupQueue> {
        self.fixups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Reports one batch of queue-offer outcomes, skipping zero counts.
fn report_admissions(enqueued: u64, duplicate: u64, saturated: u64) {
    for (admission, count) in [
        (FixupAdmission::Enqueued, enqueued),
        (FixupAdmission::Duplicate, duplicate),
        (FixupAdmission::Saturated, saturated),
    ] {
        if count > 0 {
            metrics::fixup_admission(admission, count);
        }
    }
}

/// Runs one maintenance worker until maintenance stops.
///
/// Each execution is guarded, so an aborted or panicking worker still
/// releases its Fixup's admission slot; the durable state it leaves is
/// searchable and a later access rediscovers it. The worker's lifecycle
/// activity is released by [`WorkerGuard`]'s drop, so even a task dropped
/// without polling — an ungraceful executor shutdown — cannot leak the
/// Runtime's drain accounting.
async fn run_worker<B: Backend>(guard: WorkerGuard<B>) {
    let inner = &guard.inner;
    while let Some(offer) = inner.next_fixup().await {
        let driven = {
            let mut running = RunningFixup {
                inner,
                offer: &offer,
                yield_back: false,
                split_followups: None,
            };
            let span = trace::fixup_span(offer.key.index, offer.key.partition, &offer.key.tree_key);
            let mut driven = drive(inner, &offer).instrument(span).await;
            running.yield_back = driven.execution == FixupExecution::Yielded;
            running.split_followups = driven.split_followups.take();
            driven
        };
        metrics::fixup_execution(driven.execution);
        inner.lock_fixups().stats.record(driven.execution);
    }
}

/// Releases a running Fixup's queue slot even when the worker is aborted.
struct RunningFixup<'a, B: Backend> {
    inner: &'a RuntimeInner<B>,
    offer: &'a FixupOffer,
    yield_back: bool,
    split_followups: Option<([PartitionKey; 2], Option<PartitionKey>)>,
}

impl<B: Backend> Drop for RunningFixup<'_, B> {
    fn drop(&mut self) {
        let split_followups = if self.inner.is_accepting() {
            self.split_followups.take()
        } else {
            None
        };
        let (can_yield, enqueued, duplicate, saturated) = {
            let mut queue = self.inner.lock_fixups();
            let can_yield = self.yield_back && !self.inner.maintenance_cancel.is_cancelled();
            if can_yield {
                queue.yield_back(self.offer);
                (can_yield, 0, 0, 0)
            } else {
                queue.finish(&self.offer.key);
                let mut enqueued = 0_u64;
                let mut duplicate = 0_u64;
                let mut saturated = 0_u64;
                if !self.inner.maintenance_cancel.is_cancelled()
                    && let Some((targets, parent)) = split_followups
                {
                    for partition in targets.into_iter().chain(parent) {
                        let key = FixupKey {
                            index: self.offer.key.index,
                            tree_key: self.offer.key.tree_key.clone(),
                            partition,
                        };
                        match queue.offer(key, &self.offer.manifest) {
                            FixupAdmission::Enqueued => enqueued += 1,
                            FixupAdmission::Duplicate => duplicate += 1,
                            FixupAdmission::Saturated => saturated += 1,
                        }
                    }
                }
                (can_yield, enqueued, duplicate, saturated)
            }
        };
        report_admissions(enqueued, duplicate, saturated);
        self.inner.report_fixup_backlog(metrics::fixup_backlog);
        if can_yield {
            self.inner.fixup_available.notify_one();
        } else {
            for _ in 0..enqueued {
                self.inner.fixup_available.notify_one();
            }
        }
    }
}

/// Owns one worker's lifecycle activity from spawn until the task is gone.
struct WorkerGuard<B: Backend> {
    inner: Arc<RuntimeInner<B>>,
}

impl<B: Backend> Drop for WorkerGuard<B> {
    fn drop(&mut self) {
        self.inner.clone().finish_activity();
    }
}

/// Drives one offered partition through its next bounded state-machine steps.
///
/// Rediscovered intermediate states wait for the configured recovery age.
/// A progressing worker retains permission across queue yields.
/// One step reads the durable authority once and dispatches directly to the
/// actionable split or merge state machine. Progress continues within the
/// configured step budget; successful step exhaustion yields the same key to
/// the queue tail. Any error — a
/// contended step that exhausted its bounded retries, an unknown commit
/// outcome, or Corruption — retires the Fixup, and a later relevant access
/// rediscovers the durable state exactly where it stopped. Corruption is not
/// repaired here; it surfaces to the foreground path that reads the same
/// state.
async fn drive<B: Backend>(inner: &Arc<RuntimeInner<B>>, offer: &FixupOffer) -> DrivenFixup {
    let Some(backend) = inner.maintenance_backend() else {
        return DrivenFixup::new(FixupExecution::Retired);
    };
    let retry = RetryPolicy::for_fixup(inner.config());
    let attempts = inner.config().fixup_attempts();
    let mut recovery_timeout = if offer.continuing {
        None
    } else {
        Some(inner.config().stalled_timeout(offer.manifest.config()))
    };
    for _ in 0..attempts {
        if inner.maintenance_cancel.is_cancelled() {
            return DrivenFixup::new(FixupExecution::Retired);
        }
        let started_at = now_unix_millis();
        let key = &offer.key;
        let step = maintenance_fixup::advance(
            &*backend,
            &offer.manifest,
            &key.tree_key,
            key.partition,
            maintenance_fixup::StepPolicy {
                now_unix_millis: started_at,
                recovery_timeout: recovery_timeout.take(),
                retry: &retry,
            },
        )
        .await;
        match step {
            Ok(maintenance_fixup::Advance::Idle) => {
                return DrivenFixup::new(FixupExecution::Settled);
            }
            Ok(maintenance_fixup::Advance::Split(split_step)) => {
                observe_split_step(&split_step);
                trace::fixup_kind(&Span::current(), FixupKind::Split);
                match split_step {
                    Ok(split::Advance::Idle) => {
                        return DrivenFixup::new(FixupExecution::Settled);
                    }
                    Ok(split::Advance::Completed {
                        left,
                        right,
                        parent,
                    }) => {
                        return DrivenFixup {
                            execution: FixupExecution::Settled,
                            split_followups: Some(([left, right], parent)),
                        };
                    }
                    Ok(
                        split::Advance::Began { .. }
                        | split::Advance::Exposed { .. }
                        | split::Advance::Drained { .. },
                    ) => {}
                    Err(_) => return DrivenFixup::new(FixupExecution::Retired),
                }
            }
            Ok(maintenance_fixup::Advance::Merge(merge_step)) => {
                observe_merge_step(&merge_step);
                trace::fixup_kind(&Span::current(), FixupKind::Merge);
                match merge_step {
                    Ok(merge::Advance::Idle | merge::Advance::Completed) => {
                        return DrivenFixup::new(FixupExecution::Settled);
                    }
                    Ok(merge::Advance::Stalled) => {
                        return DrivenFixup::new(FixupExecution::Stalled);
                    }
                    Ok(merge::Advance::Began | merge::Advance::Drained { .. }) => {}
                    Err(_) => return DrivenFixup::new(FixupExecution::Retired),
                }
            }
            // The shared authority preflight failed before it could identify
            // a split or merge owner. The execution outcome records the
            // retirement without fabricating a state-machine step.
            Err(_) => return DrivenFixup::new(FixupExecution::Retired),
        }
    }
    // Every budgeted step made progress and left work remaining. Preserve the
    // admission slot while returning the key to the queue tail so other work
    // remains fair without requiring another foreground access.
    DrivenFixup::new(FixupExecution::Yielded)
}

/// One bounded Fixup execution and any newly Ready split targets it discovered.
struct DrivenFixup {
    execution: FixupExecution,
    split_followups: Option<([PartitionKey; 2], Option<PartitionKey>)>,
}

impl DrivenFixup {
    const fn new(execution: FixupExecution) -> Self {
        Self {
            execution,
            split_followups: None,
        }
    }
}

/// Records one split advance result without exposing partition identity.
fn observe_split_step(step: &Result<split::Advance>) {
    let result = match step {
        Ok(split::Advance::Idle) => FixupStepResult::Idle,
        Ok(split::Advance::Began { .. }) => FixupStepResult::Began,
        Ok(split::Advance::Exposed { .. }) => FixupStepResult::Exposed,
        // The committed drain boundary records both the step and moved count,
        // including a final drain followed by completion in this advance.
        Ok(split::Advance::Drained { .. }) => return,
        Ok(split::Advance::Completed { .. }) => FixupStepResult::Completed,
        Err(_) => FixupStepResult::Failed,
    };
    metrics::fixup_step(FixupKind::Split, result);
}

/// Records one merge advance result without exposing partition identity.
fn observe_merge_step(step: &Result<merge::Advance>) {
    let result = match step {
        Ok(merge::Advance::Idle) => FixupStepResult::Idle,
        Ok(merge::Advance::Began) => FixupStepResult::Began,
        // The committed drain boundary records both the step and moved count,
        // including a final drain followed by completion in this advance.
        Ok(merge::Advance::Drained { .. }) => return,
        Ok(merge::Advance::Stalled) => FixupStepResult::Stalled,
        Ok(merge::Advance::Completed) => FixupStepResult::Completed,
        Err(_) => FixupStepResult::Failed,
    };
    metrics::fixup_step(FixupKind::Merge, result);
}

#[cfg(test)]
mod tests {
    use std::future::{Future, Ready, ready};
    use std::time::Duration;

    use bytes::Bytes;

    use super::*;
    use crate::api::{
        DataType, Error, ErrorKind, IndexConfig, Metric, Result, RuntimeConfig, Value,
    };
    use crate::runtime::Runtime;
    use crate::storage::backend::{
        AdmissionBudget, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation, ReadOps,
        ReadTxn, ScanLimits, ScanPage, WriteTxn,
    };
    use crate::storage::keys::KeyRange;
    use crate::storage::values::IndexLifecycle;

    fn manifest(id: u64) -> Arc<IndexManifest> {
        Arc::new(
            IndexManifest::new(
                IndexLifecycle::Active,
                LogicalIndexId::new(id).expect("nonzero"),
                IndexConfig::new(1, Metric::L2).expect("valid config"),
                [0_u8; 32],
                vec![],
            )
            .expect("valid manifest"),
        )
    }

    fn tree_key(bucket: i64) -> TreeKey {
        TreeKey::encode(&[DataType::I64], &[Value::I64(bucket)]).expect("canonical key")
    }

    fn key(manifest: &Arc<IndexManifest>, bucket: i64, partition: u64) -> FixupKey {
        FixupKey {
            index: manifest.logical_index_id(),
            tree_key: tree_key(bucket),
            partition: PartitionKey::new(partition).expect("nonzero"),
        }
    }

    #[test]
    fn duplicate_offers_coalesce_while_pending() {
        let manifest = manifest(1);
        let mut queue = FixupQueue::new(4);
        assert_eq!(
            queue.offer(key(&manifest, 1, 1), &manifest),
            FixupAdmission::Enqueued
        );
        assert_eq!(
            queue.offer(key(&manifest, 1, 1), &manifest),
            FixupAdmission::Duplicate
        );
        assert_eq!(queue.pending.len(), 1);
        assert_eq!(queue.stats.enqueued, 1);
        assert_eq!(queue.stats.duplicate, 1);
        // A distinct partition of the same tree and the same partition of a
        // distinct tree are both new keys.
        assert_eq!(
            queue.offer(key(&manifest, 1, 2), &manifest),
            FixupAdmission::Enqueued
        );
        assert_eq!(
            queue.offer(key(&manifest, 2, 1), &manifest),
            FixupAdmission::Enqueued
        );
        assert_eq!(queue.pending.len(), 3);
    }

    #[test]
    fn saturation_drops_offers_and_running_slots_count() {
        let manifest = manifest(1);
        let mut queue = FixupQueue::new(2);
        assert_eq!(
            queue.offer(key(&manifest, 1, 1), &manifest),
            FixupAdmission::Enqueued
        );
        assert_eq!(
            queue.offer(key(&manifest, 1, 2), &manifest),
            FixupAdmission::Enqueued
        );
        // Pending plus running reached the capacity.
        assert_eq!(
            queue.offer(key(&manifest, 1, 3), &manifest),
            FixupAdmission::Saturated
        );
        assert_eq!(queue.stats.saturated, 1);

        // A running Fixup keeps its slot against new offers.
        let first = queue.pop().expect("one pending offer");
        assert_eq!(queue.backlog(), 2);
        assert_eq!(
            queue.offer(key(&manifest, 1, 3), &manifest),
            FixupAdmission::Saturated
        );
        // A duplicate of the running key still coalesces rather than failing.
        assert_eq!(
            queue.offer(key(&manifest, 1, 1), &manifest),
            FixupAdmission::Duplicate
        );

        queue.finish(&first.key);
        assert_eq!(queue.backlog(), 1);
        // The still-pending second offer keeps its slot; the freed slot
        // admits again.
        assert_eq!(queue.admitted.len(), 1);
        assert_eq!(
            queue.offer(key(&manifest, 1, 3), &manifest),
            FixupAdmission::Enqueued
        );
        let second = queue.pop().expect("pending offer");
        assert_eq!(second.key.partition.get(), 2);
        let third = queue.pop().expect("pending offer");
        assert_eq!(third.key.partition.get(), 3);
        queue.finish(&second.key);
        queue.finish(&third.key);
        assert!(queue.admitted.is_empty());
    }

    #[test]
    fn pop_is_fifo_and_drain_keeps_running() {
        let manifest = manifest(1);
        let mut queue = FixupQueue::new(4);
        for partition in 1..=3_u64 {
            assert_eq!(
                queue.offer(key(&manifest, 1, partition), &manifest),
                FixupAdmission::Enqueued
            );
        }
        let first = queue.pop().expect("first pending");
        assert_eq!(first.key.partition.get(), 1);
        queue.drain_pending();
        // Pending work is cancelled; the running admission survives.
        assert!(queue.pending.is_empty());
        assert_eq!(queue.backlog(), 1);
        assert_eq!(queue.admitted.len(), 1);
        assert!(queue.pop().is_none());
        queue.finish(&first.key);
        assert!(queue.admitted.is_empty());
        assert_eq!(queue.backlog(), 0);
    }

    /// A backend whose every operation fails: Fixup executions retire
    /// immediately, which exercises worker pickup and slot hygiene.
    struct FailingBackend;

    struct FailingTxn;

    fn failing<T>() -> Ready<Result<T>> {
        ready(Err(Error::new(ErrorKind::Backend)))
    }

    impl ReadOps for FailingTxn {
        fn get(&mut self, _key: Bytes) -> impl Future<Output = Result<Option<Bytes>>> + Send {
            failing()
        }

        fn batch_get(
            &mut self,
            _keys: Vec<Bytes>,
        ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send {
            failing()
        }

        fn scan(
            &mut self,
            _range: &KeyRange,
            _limits: ScanLimits,
        ) -> impl Future<Output = Result<ScanPage>> + Send {
            failing()
        }

        fn batch_scan(
            &mut self,
            _ranges: &[KeyRange],
            _limits: ScanLimits,
        ) -> impl Future<Output = Result<Vec<ScanPage>>> + Send {
            failing()
        }
    }

    impl ReadTxn for FailingTxn {}

    impl WriteTxn for FailingTxn {
        fn get_for_update(
            &mut self,
            _key: Bytes,
        ) -> impl Future<Output = Result<Option<Bytes>>> + Send {
            failing()
        }

        fn batch_get_for_update(
            &mut self,
            _keys: Vec<Bytes>,
        ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send {
            failing()
        }

        fn put(&mut self, _key: Bytes, _value: Bytes) -> impl Future<Output = Result<()>> + Send {
            failing()
        }

        fn insert(
            &mut self,
            _key: Bytes,
            _value: Bytes,
        ) -> impl Future<Output = Result<InsertOutcome>> + Send {
            failing()
        }

        fn delete(&mut self, _key: Bytes) -> impl Future<Output = Result<()>> + Send {
            failing()
        }

        fn batch_mutate(
            &mut self,
            _mutations: Vec<Mutation>,
        ) -> impl Future<Output = Result<()>> + Send {
            failing()
        }

        fn clear_range(&mut self, _range: &KeyRange) -> impl Future<Output = Result<()>> + Send {
            failing()
        }

        async fn commit_with(self, _start: CommitStart) -> Result<()> {
            Err(Error::new(ErrorKind::Backend))
        }

        fn rollback(self) -> impl Future<Output = ()> + Send {
            ready(())
        }
    }

    impl Backend for FailingBackend {
        type ReadTxn<'backend> = FailingTxn;
        type WriteTxn<'backend> = FailingTxn;

        fn hard_limits(&self) -> HardLimits {
            HardLimits {
                max_key_bytes: usize::MAX,
                max_value_bytes: usize::MAX,
            }
        }

        fn admission_budget(&self) -> AdmissionBudget {
            AdmissionBudget {
                max_mutations: usize::MAX,
                max_mutation_bytes: usize::MAX,
                mutation_key_overhead_bytes: 0,
            }
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                transactional_clear_range: false,
            }
        }

        async fn shutdown(&self) {}

        fn begin_read(&self) -> impl Future<Output = Result<Self::ReadTxn<'_>>> + Send + '_ {
            ready(Ok(FailingTxn))
        }

        fn begin_write(&self) -> impl Future<Output = Result<Self::WriteTxn<'_>>> + Send + '_ {
            ready(Ok(FailingTxn))
        }
    }

    fn failing_runtime(workers: usize, capacity: usize) -> Runtime<FailingBackend> {
        let config = RuntimeConfig::default()
            .with_maintenance(workers, capacity)
            .expect("valid maintenance config");
        Runtime::new(FailingBackend, config).expect("multi-thread runtime")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backlog_publication_catches_a_concurrent_drain() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;

        let runtime = failing_runtime(0, 4);
        let inner = Arc::clone(&runtime.handle.inner);
        let manifest = manifest(1);
        assert_eq!(
            inner.lock_fixups().offer(key(&manifest, 0, 1), &manifest),
            FixupAdmission::Enqueued
        );
        let observed = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let reporting_inner = Arc::clone(&inner);
        let reporting_observed = Arc::clone(&observed);
        let reporter = std::thread::spawn(move || {
            let mut paused = false;
            reporting_inner.report_fixup_backlog(|backlog| {
                if !paused {
                    paused = true;
                    entered_tx.send(()).expect("test is waiting");
                    release_rx
                        .recv_timeout(Duration::from_secs(5))
                        .expect("release reporter");
                }
                reporting_observed.store(backlog, Ordering::SeqCst);
            });
        });
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("publisher started");
        assert_eq!(inner.lock_fixups().drain_pending(), 0);
        inner.report_fixup_backlog(|backlog| observed.store(backlog, Ordering::SeqCst));
        release_tx.send(()).expect("reporter is waiting");
        reporter.join().expect("reporter finished");
        assert_eq!(
            observed.load(Ordering::SeqCst),
            0,
            "quiescent queue must not retain an older nonzero gauge"
        );
        runtime.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backlog_publication_allows_reentrant_queue_changes() {
        let runtime = failing_runtime(0, 4);
        let inner = &runtime.handle.inner;
        let manifest = manifest(1);
        inner.lock_fixups().offer(key(&manifest, 0, 1), &manifest);
        let observed = std::cell::Cell::new(usize::MAX);
        inner.report_fixup_backlog(|backlog| {
            if backlog != 0 {
                inner.lock_fixups().drain_pending();
                inner.report_fixup_backlog(|current| observed.set(current));
            }
            observed.set(backlog);
        });
        assert_eq!(observed.get(), 0);
        runtime.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn backlog_publication_recovers_after_recorder_panic() {
        let runtime = failing_runtime(0, 4);
        let inner = &runtime.handle.inner;
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            inner.report_fixup_backlog(|_| panic!("recorder failed"));
        }));
        assert!(failed.is_err());
        let mut observed = None;
        inner.report_fixup_backlog(|backlog| observed = Some(backlog));
        assert_eq!(observed, Some(0));
        runtime.shutdown().await.expect("shutdown");
    }

    /// Polls `condition` with a generous real-time bound.
    async fn eventually(mut condition: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !condition() {
            assert!(
                std::time::Instant::now() < deadline,
                "condition not met within the time bound"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn workers_execute_and_retire_without_leaking_slots() {
        let runtime = failing_runtime(2, 4);
        let manifest = manifest(1);
        runtime.handle.inner.offer_fixups(
            &manifest,
            [(tree_key(1), PartitionKey::new(1).expect("nonzero"))],
        );
        eventually(|| runtime.handle.inner.lock_fixups().admitted.is_empty()).await;
        let stats = runtime.handle.inner.fixup_stats();
        assert_eq!(stats.enqueued, 1);
        assert_eq!(stats.retired, 1);
        assert_eq!(stats.settled, 0);
        let (pending, backlog) = {
            let queue = runtime.handle.inner.lock_fixups();
            (queue.pending.len(), queue.backlog())
        };
        assert_eq!((pending, backlog), (0, 0));
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn saturated_queue_drops_and_later_offers_admit() {
        let runtime = failing_runtime(1, 1);
        let manifest = manifest(1);
        let first = (tree_key(1), PartitionKey::new(1).expect("nonzero"));
        let second = (tree_key(1), PartitionKey::new(2).expect("nonzero"));
        // One batch holds the queue lock across both admissions, so a worker
        // cannot retire the first offer before the second encounters capacity.
        runtime
            .handle
            .inner
            .offer_fixups(&manifest, [first, second.clone()]);
        eventually(|| runtime.handle.inner.lock_fixups().admitted.is_empty()).await;
        let stats = runtime.handle.inner.fixup_stats();
        assert_eq!(stats.enqueued, 1);
        assert_eq!(stats.saturated, 1);
        assert_eq!(stats.duplicate, 0);
        // After the first retired, the second key admits and retires too.
        runtime.handle.inner.offer_fixups(&manifest, [second]);
        eventually(|| {
            let queue = runtime.handle.inner.lock_fixups();
            queue.admitted.is_empty() && queue.stats.retired == 2
        })
        .await;
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_cancels_pending_and_stops_workers() {
        let runtime = failing_runtime(1, 8);
        let manifest = manifest(1);
        for partition in 1..=8_u64 {
            runtime.handle.inner.offer_fixups(
                &manifest,
                [(tree_key(1), PartitionKey::new(partition).expect("nonzero"))],
            );
        }
        runtime.shutdown().await.expect("shutdown succeeds");
        {
            let queue = runtime.handle.inner.lock_fixups();
            assert!(queue.pending.is_empty());
            assert!(queue.admitted.is_empty());
            assert_eq!(queue.backlog(), 0);
        }
        // Offers after shutdown are dropped without being counted or run.
        let retired = runtime.handle.inner.fixup_stats().retired;
        runtime.handle.inner.offer_fixups(
            &manifest,
            [(tree_key(1), PartitionKey::new(9).expect("nonzero"))],
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
        let stats = runtime.handle.inner.fixup_stats();
        assert_eq!(stats.retired, retired);
        assert!(runtime.handle.inner.lock_fixups().pending.is_empty());
    }
}
