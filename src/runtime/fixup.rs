//! Bounded process-local Fixup scheduling (design `runtime-operations.md` §3).
//!
//! A Fixup is one demand-driven unit of Structure Maintenance: one partition
//! of one tree of one Logical Index, offered by a relevant mutation or search
//! and rediscovered from durable state when executed. This module owns the
//! Runtime's bounded, deduplicating process-local queue, the maintenance
//! workers that execute queued Fixups, and their shutdown behavior. The
//! split/merge state machines themselves live in [`crate::maintenance`].
//!
//! # Contract
//!
//! - **Bounded everything.** Pending plus running Fixups never exceed the
//!   configured queue capacity, worker count is fixed at construction, every
//!   execution runs at most the configured number of whole state-machine steps
//!   with the capped jittered backoff of [`RetryPolicy::for_fixup`], and queue
//!   memory is proportional to capacity. Overload drops offers; it never
//!   affects persistent correctness.
//! - **Deduplicating admission.** One key — Logical Index ID, Tree Key, and
//!   Partition Key — occupies at most one slot, whether pending or running.
//!   A duplicate offer coalesces into the admitted one; a saturated queue
//!   drops the offer. Both outcomes are counted in [`FixupStats`].
//! - **Loss is safe.** The queue is process-local and may be lost at any
//!   time: every committed state-machine state remains searchable, and a later
//!   relevant access re-offers the partition (Demand-Driven Maintenance, ADR
//!   0006). There is no durable scan, queue, leader, lease, or claim.
//! - **Bounded execution.** One execution drives one partition through the
//!   split or merge state machine for at most the configured steps. It settles
//!   when no work remains, stalls when a merge has no legal target, and
//!   retires on error or step exhaustion; retirement never removes durable
//!   state, so rediscovery resumes where the state machine stopped.
//! - **Shutdown.** Stopping admission cancels pending work immediately and
//!   lets an admitted step finish its bounded transaction; workers are
//!   accounted in the Runtime's lifecycle so the backend is released only
//!   after every worker has stopped.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, MutexGuard};

use crate::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result};
use crate::maintenance::{merge, split};
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
}

/// The outcome of offering one Fixup key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Admission {
    /// The key took a queue slot.
    Enqueued,
    /// The key is already pending or running; the offer coalesced into it.
    Duplicate,
    /// Pending plus running Fixups reached the configured capacity; the offer
    /// was dropped.
    Saturated,
}

/// Cumulative, privacy-safe observations of the process-local Fixup queue.
///
/// Counts only; the permanent metrics and tracing labels arrive with the
/// observability work. Saturating arithmetic keeps the counters themselves
/// bounded and panic-free.
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
    /// Executions that stopped early: error, step exhaustion, or shutdown.
    retired: u64,
}

impl FixupStats {
    /// Counts one finished execution's outcome.
    fn record(&mut self, execution: Execution) {
        let counter = match execution {
            Execution::Settled => &mut self.settled,
            Execution::Stalled => &mut self.stalled,
            Execution::Retired => &mut self.retired,
        };
        *counter = counter.saturating_add(1);
    }
}

/// The outcome of one Fixup execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Execution {
    /// The partition reached a state with no work for either state machine.
    Settled,
    /// The partition is `Merging` with no legal target; rediscovery may retry.
    Stalled,
    /// Execution stopped early: an error (including an unknown commit
    /// outcome), the bounded step count, cancellation, or backend release.
    Retired,
}

/// The bounded, deduplicating process-local Fixup queue.
///
/// `admitted` holds every key with a slot, pending or running, which is what
/// makes an offer during execution coalesce instead of queueing behind the
/// running execution: the executor keeps advancing the partition within its
/// bounded step budget, and a later access re-offers after it retires.
pub(crate) struct FixupQueue {
    pending: VecDeque<FixupOffer>,
    admitted: HashSet<FixupKey>,
    running: usize,
    capacity: usize,
    stats: FixupStats,
}

impl FixupQueue {
    pub(crate) fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0, "the Runtime configuration validates it");
        Self {
            pending: VecDeque::new(),
            admitted: HashSet::new(),
            running: 0,
            capacity,
            stats: FixupStats::default(),
        }
    }

    /// The process-local maintenance backlog: pending plus running Fixups.
    fn backlog(&self) -> usize {
        self.pending.len().saturating_add(self.running)
    }

    /// Admits one offered Fixup under the deduplication and capacity policy.
    ///
    /// The Manifest Arc is cloned only on the enqueue path; duplicate and
    /// saturated offers drop without touching its refcount.
    fn offer(&mut self, key: FixupKey, manifest: &Arc<IndexManifest>) -> Admission {
        if self.admitted.contains(&key) {
            self.stats.duplicate = self.stats.duplicate.saturating_add(1);
            return Admission::Duplicate;
        }
        if self.backlog() >= self.capacity {
            self.stats.saturated = self.stats.saturated.saturating_add(1);
            return Admission::Saturated;
        }
        self.admitted.insert(key.clone());
        self.pending.push_back(FixupOffer {
            key,
            manifest: Arc::clone(manifest),
        });
        self.stats.enqueued = self.stats.enqueued.saturating_add(1);
        Admission::Enqueued
    }

    /// Takes the oldest pending Fixup into running state.
    ///
    /// The key keeps its slot while running, so concurrent duplicate offers
    /// still coalesce. [`FixupQueue::finish`] releases the slot.
    fn pop(&mut self) -> Option<FixupOffer> {
        let offer = self.pending.pop_front()?;
        self.running = self.running.saturating_add(1);
        Some(offer)
    }

    /// Releases one running Fixup's slot after its execution ended.
    ///
    /// Called from the worker's running guard, so it also runs when the
    /// worker task is aborted mid-execution: an interrupted Fixup never leaks
    /// its admission slot.
    fn finish(&mut self, key: &FixupKey) {
        self.running = self.running.saturating_sub(1);
        self.admitted.remove(key);
    }

    /// Drops every pending Fixup, keeping only running admissions.
    ///
    /// Shutdown cancels work that has not begun; running executions release
    /// their own slots through [`FixupQueue::finish`].
    fn drain_pending(&mut self) {
        while let Some(offer) = self.pending.pop_front() {
            self.admitted.remove(&offer.key);
        }
    }
}

impl<B: Backend> RuntimeInner<B> {
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
        let mut queue = self.lock_fixups();
        for (tree_key, partition) in partitions {
            let key = FixupKey {
                index,
                tree_key,
                partition,
            };
            if queue.offer(key, manifest) == Admission::Enqueued {
                self.fixup_available.notify_one();
            }
        }
    }

    /// Returns the cumulative Fixup queue statistics.
    pub(crate) fn fixup_stats(&self) -> FixupStats {
        self.lock_fixups().stats
    }

    /// Waits for the Import Session backlog gate (design
    /// `runtime-operations.md` §4): admission pauses until the process-local
    /// Fixup backlog drops below the configured watermark.
    ///
    /// The wait is cancellation-safe and wakes on every queue slot release
    /// that opens the gate; once the Runtime stops accepting work it fails
    /// with [`ErrorKind::RuntimeClosed`] instead of waiting for the backlog.
    pub(crate) async fn wait_for_backlog_below(&self) -> Result<()> {
        let watermark = self.config().import_backlog_watermark();
        loop {
            // Register the waiter before checking the backlog so a slot
            // release landing between the check and the wait still wakes us.
            let notified = self.fixup_released.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.lock_fixups().backlog() < watermark {
                return Ok(());
            }
            if self.maintenance_cancel.is_cancelled() {
                return Err(Error::new(ErrorKind::RuntimeClosed));
            }
            tokio::select! {
                biased;
                () = self.maintenance_cancel.cancelled() => {
                    return Err(Error::new(ErrorKind::RuntimeClosed));
                }
                () = notified => {}
            }
        }
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
                return None;
            }
            if let Some(offer) = self.lock_fixups().pop() {
                return Some(offer);
            }
            tokio::select! {
                biased;
                () = self.maintenance_cancel.cancelled() => {
                    self.lock_fixups().drain_pending();
                    return None;
                }
                () = notified => {}
            }
        }
    }

    fn lock_fixups(&self) -> MutexGuard<'_, FixupQueue> {
        match self.fixups.lock() {
            Ok(fixups) => fixups,
            Err(poisoned) => poisoned.into_inner(),
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
        let _running = RunningFixup {
            inner,
            key: &offer.key,
        };
        let execution = drive(inner, &offer).await;
        inner.lock_fixups().stats.record(execution);
    }
}

/// Releases a running Fixup's queue slot even when the worker is aborted.
struct RunningFixup<'a, B: Backend> {
    inner: &'a RuntimeInner<B>,
    key: &'a FixupKey,
}

impl<B: Backend> Drop for RunningFixup<'_, B> {
    fn drop(&mut self) {
        // Wake gated Import Session submissions only when this release opens
        // the backlog gate; while it stays closed no waiter could proceed.
        let gate_open = {
            let mut queue = self.inner.lock_fixups();
            queue.finish(self.key);
            queue.backlog() < self.inner.config().import_backlog_watermark()
        };
        if gate_open {
            self.inner.fixup_released.notify_waiters();
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
/// One step is one [`split::advance`] pass, falling back to one
/// [`merge::advance`] pass when the split state machine has nothing to do
/// (`Ready` at or below the split threshold, `ReceivingSplit`, or `Merging`).
/// Progress continues within the configured step budget; any error — a
/// contended step that exhausted its bounded retries, an unknown commit
/// outcome, or Corruption — retires the Fixup, and a later relevant access
/// rediscovers the durable state exactly where it stopped. Corruption is not
/// repaired here; it surfaces to the foreground path that reads the same
/// state.
async fn drive<B: Backend>(inner: &Arc<RuntimeInner<B>>, offer: &FixupOffer) -> Execution {
    let Some(backend) = inner.maintenance_backend() else {
        return Execution::Retired;
    };
    let retry = RetryPolicy::for_fixup(inner.config());
    let attempts = inner.config().fixup_attempts();
    for _ in 0..attempts {
        if inner.maintenance_cancel.is_cancelled() {
            return Execution::Retired;
        }
        let started_at = now_unix_millis();
        let key = &offer.key;
        let split_step = split::advance(
            &*backend,
            &offer.manifest,
            &key.tree_key,
            key.partition,
            started_at,
            &retry,
        )
        .await;
        match split_step {
            Ok(split::Advance::Idle) => {
                match merge::advance(
                    &*backend,
                    &offer.manifest,
                    &key.tree_key,
                    key.partition,
                    started_at,
                    &retry,
                )
                .await
                {
                    Ok(merge::Advance::Idle | merge::Advance::Completed) => {
                        return Execution::Settled;
                    }
                    Ok(merge::Advance::Stalled) => return Execution::Stalled,
                    Ok(merge::Advance::Began | merge::Advance::Drained { .. }) => {}
                    Err(_) => return Execution::Retired,
                }
            }
            Ok(split::Advance::Completed) => return Execution::Settled,
            Ok(
                split::Advance::Began { .. }
                | split::Advance::Exposed { .. }
                | split::Advance::Drained { .. },
            ) => {}
            Err(_) => return Execution::Retired,
        }
    }
    // The step budget ran out with work possibly remaining: the Fixup
    // retires and a later relevant access re-offers the partition.
    Execution::Retired
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
            Admission::Enqueued
        );
        assert_eq!(
            queue.offer(key(&manifest, 1, 1), &manifest),
            Admission::Duplicate
        );
        assert_eq!(queue.pending.len(), 1);
        assert_eq!(queue.stats.enqueued, 1);
        assert_eq!(queue.stats.duplicate, 1);
        // A distinct partition of the same tree and the same partition of a
        // distinct tree are both new keys.
        assert_eq!(
            queue.offer(key(&manifest, 1, 2), &manifest),
            Admission::Enqueued
        );
        assert_eq!(
            queue.offer(key(&manifest, 2, 1), &manifest),
            Admission::Enqueued
        );
        assert_eq!(queue.pending.len(), 3);
    }

    #[test]
    fn saturation_drops_offers_and_running_slots_count() {
        let manifest = manifest(1);
        let mut queue = FixupQueue::new(2);
        assert_eq!(
            queue.offer(key(&manifest, 1, 1), &manifest),
            Admission::Enqueued
        );
        assert_eq!(
            queue.offer(key(&manifest, 1, 2), &manifest),
            Admission::Enqueued
        );
        // Pending plus running reached the capacity.
        assert_eq!(
            queue.offer(key(&manifest, 1, 3), &manifest),
            Admission::Saturated
        );
        assert_eq!(queue.stats.saturated, 1);

        // A running Fixup keeps its slot against new offers.
        let first = queue.pop().expect("one pending offer");
        assert_eq!(queue.running, 1);
        assert_eq!(
            queue.offer(key(&manifest, 1, 3), &manifest),
            Admission::Saturated
        );
        // A duplicate of the running key still coalesces rather than failing.
        assert_eq!(
            queue.offer(key(&manifest, 1, 1), &manifest),
            Admission::Duplicate
        );

        queue.finish(&first.key);
        assert_eq!(queue.running, 0);
        // The still-pending second offer keeps its slot; the freed slot
        // admits again.
        assert_eq!(queue.admitted.len(), 1);
        assert_eq!(
            queue.offer(key(&manifest, 1, 3), &manifest),
            Admission::Enqueued
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
                Admission::Enqueued
            );
        }
        let first = queue.pop().expect("first pending");
        assert_eq!(first.key.partition.get(), 1);
        queue.drain_pending();
        // Pending work is cancelled; the running admission survives.
        assert!(queue.pending.is_empty());
        assert_eq!(queue.running, 1);
        assert_eq!(queue.admitted.len(), 1);
        assert!(queue.pop().is_none());
        queue.finish(&first.key);
        assert!(queue.admitted.is_empty());
        assert_eq!(queue.running, 0);
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
            .and_then(|config| config.with_import_limits(1, 1))
            .expect("valid maintenance config");
        Runtime::new(FailingBackend, config).expect("multi-thread runtime")
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
        let (pending, running) = {
            let queue = runtime.handle.inner.lock_fixups();
            (queue.pending.len(), queue.running)
        };
        assert_eq!((pending, running), (0, 0));
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn duplicate_offers_during_execution_coalesce() {
        let runtime = failing_runtime(1, 4);
        let manifest = manifest(1);
        let key = (tree_key(1), PartitionKey::new(1).expect("nonzero"));
        for _ in 0..8 {
            runtime.handle.inner.offer_fixups(&manifest, [key.clone()]);
        }
        eventually(|| {
            let queue = runtime.handle.inner.lock_fixups();
            queue.admitted.is_empty() && queue.stats.enqueued == 1
        })
        .await;
        let stats = runtime.handle.inner.fixup_stats();
        assert_eq!(stats.enqueued, 1);
        assert!(stats.duplicate >= 1);
        assert_eq!(stats.retired, 1);
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn saturated_queue_drops_and_later_offers_admit() {
        let runtime = failing_runtime(1, 1);
        let manifest = manifest(1);
        let first = (tree_key(1), PartitionKey::new(1).expect("nonzero"));
        let second = (tree_key(1), PartitionKey::new(2).expect("nonzero"));
        runtime
            .handle
            .inner
            .offer_fixups(&manifest, [first.clone()]);
        // With capacity one, the second distinct key is dropped while the
        // first holds a pending or running slot.
        runtime
            .handle
            .inner
            .offer_fixups(&manifest, [second.clone()]);
        eventually(|| runtime.handle.inner.lock_fixups().admitted.is_empty()).await;
        let stats = runtime.handle.inner.fixup_stats();
        let dropped = stats.saturated + stats.duplicate;
        assert!(dropped >= 1, "the second offer coalesced or dropped");
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
            assert_eq!(queue.running, 0);
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
