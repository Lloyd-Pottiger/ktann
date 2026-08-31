//! Import Session batch admission and ordered outcome collection.
//!
//! One coordinator owns the process-local state of a single Import Session: a
//! adaptive bounded-concurrency controller, the monotonically increasing Batch
//! Token sequence, and the accepted batch tasks in submission order. Every
//! admitted batch runs the ordinary foreground mutation path
//! ([`Index::run_mutations`]),
//! so its atomicity, retry, error, and maintenance behavior is identical to a
//! normal `batch_mutate` and import changes neither Foreground Mutation
//! atomicity nor Logical Index lifecycle.
//!
//! Admission is bounded in both directions: a session starts with one active
//! batch, learns additional concurrency from saturated clean completions and retryable
//! conflicts up to its configured ceiling, and has no session-internal queue:
//! a caller waiting for a slot or the backlog gate holds its own
//! unsubmitted batch, while admitted batches additionally pass the Runtime's
//! bounded foreground admission. Session memory is bounded by the in-flight
//! batch payloads plus one outcome entry per accepted token, collected on
//! `drain`.
//!
//! Dropping the coordinator aborts every incomplete batch task. Tokio drops an
//! aborted task's future, which runs the caller-drop guard inside
//! `run_foreground`: a batch whose commit has not started is cancelled, while
//! a started commit keeps running detached under the Runtime's in-flight guard
//! and finishes without an Import Session result consumer.
//!
//! `submit_batch` also waits for the Runtime's Structure Maintenance backlog
//! gate (design `runtime-operations.md` §4): a non-empty batch admits only
//! once the process-local Fixup backlog — pending plus running — is below the
//! configured watermark. The gate is process-local backpressure, never a
//! durable or cluster-wide barrier; losing it cannot affect persistent
//! correctness, and an empty batch does no storage work so it bypasses the
//! gate exactly like it bypasses the in-flight slot.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::api::{
    BatchToken, Error, ErrorKind, ImportBatchResult, Index, Mutation, MutationOutcome,
    OperationOptions, Result,
};
use crate::observe::labels::{ImportConcurrencyAdjustment, ImportGate, Operation};
use crate::observe::metrics;
use crate::runtime::RuntimeInner;
use crate::storage::backend::Backend;

/// Clean completion windows required before probing one more concurrent batch.
///
/// Import starts conservatively because a new tree has one writable leaf. A
/// sustained clean window lets the session probe additional concurrency, while
/// the first retryable conflict immediately contracts the window.
const CLEAN_WINDOWS_PER_INCREASE: usize = 8;
/// Maximum clean window after repeated failed concurrency probes.
const MAX_CLEAN_WINDOWS_PER_INCREASE: usize = 64;

/// Mutable state for one session's additive-increase/multiplicative-decrease
/// admission controller.
struct ImportConcurrencyState {
    active: usize,
    submission_waiting: bool,
    waiting_retries: usize,
    limit: usize,
    clean_completions: usize,
    clean_window: usize,
}

/// Learns the useful in-flight batch count from observed write contention.
///
/// Partition count alone does not describe write parallelism: two batches can
/// overlap the same hot leaves in a large tree, while disjoint Tree Keys can be
/// independent in a small forest. This controller therefore starts at one,
/// raises the limit only after clean committed work, and contracts before a
/// contended batch retries. The configured maximum in-flight value is a hard
/// ceiling, not the concurrency used for the whole session.
struct ImportConcurrency {
    maximum: usize,
    state: Mutex<ImportConcurrencyState>,
    changed: Notify,
}

impl ImportConcurrency {
    /// Creates a controller whose first batch runs alone.
    fn new(maximum: usize) -> Arc<Self> {
        debug_assert!(maximum > 0, "import concurrency ceilings are validated");
        Arc::new(Self {
            maximum,
            state: Mutex::new(ImportConcurrencyState {
                active: 0,
                submission_waiting: false,
                waiting_retries: 0,
                limit: 1,
                clean_completions: 0,
                clean_window: CLEAN_WINDOWS_PER_INCREASE,
            }),
            changed: Notify::new(),
        })
    }

    /// Acquires accepted-batch and active-attempt capacity.
    async fn acquire<B: Backend>(
        self: &Arc<Self>,
        runtime: Arc<RuntimeInner<B>>,
    ) -> ImportPermit<B> {
        let mut waiter = SubmissionWaiter::new(self);
        self.wait_until(|state| {
            if state.active + state.waiting_retries < self.maximum
                && state.active < state.limit
                && state.waiting_retries == 0
            {
                waiter.unregister(state);
                state.active += 1;
                true
            } else {
                waiter.register(state);
                false
            }
        })
        .await;
        ImportPermit {
            concurrency: Arc::clone(self),
            runtime,
            state: ImportPermitState::Active,
            contended: false,
        }
    }

    /// Releases active capacity and records one completed batch.
    fn finish(&self, permit: &mut ImportPermitState, clean: bool) {
        if matches!(permit, ImportPermitState::Released) {
            return;
        }
        let mut state = self.lock_state();
        let saturated = matches!(permit, ImportPermitState::Active)
            && state.active == state.limit
            && state.submission_waiting
            && state.waiting_retries == 0;
        match permit {
            ImportPermitState::Active => {
                debug_assert!(state.active > 0);
                state.active -= 1;
            }
            ImportPermitState::WaitingRetry => {
                debug_assert!(state.waiting_retries > 0);
                state.waiting_retries -= 1;
            }
            ImportPermitState::Released => unreachable!("released permits return above"),
        }
        *permit = ImportPermitState::Released;
        let mut increased_to = None;
        if clean && saturated && state.limit < self.maximum {
            state.clean_completions = state.clean_completions.saturating_add(1);
            let threshold = state.limit.saturating_mul(state.clean_window);
            if state.clean_completions >= threshold {
                state.limit += 1;
                state.clean_completions = 0;
                increased_to = Some(state.limit);
            }
        }
        drop(state);
        if let Some(limit) = increased_to {
            metrics::import_concurrency_adjusted(ImportConcurrencyAdjustment::Increased, limit);
        }
        self.changed.notify_waiters();
    }

    /// Removes a contended batch from the active window and contracts it.
    fn pause_for_contention(&self, permit: &mut ImportPermitState) {
        debug_assert!(matches!(permit, ImportPermitState::Active));
        let mut state = self.lock_state();
        debug_assert!(state.active > 0);
        state.active -= 1;
        state.waiting_retries += 1;
        *permit = ImportPermitState::WaitingRetry;
        let previous = state.limit;
        state.limit = (state.limit / 2).max(1);
        let decreased_to = (state.limit < previous).then_some(state.limit);
        state.clean_completions = 0;
        if decreased_to.is_some() {
            state.clean_window = state
                .clean_window
                .saturating_mul(2)
                .min(MAX_CLEAN_WINDOWS_PER_INCREASE);
        }
        drop(state);
        if let Some(limit) = decreased_to {
            metrics::import_concurrency_adjusted(ImportConcurrencyAdjustment::Decreased, limit);
        }
        self.changed.notify_waiters();
    }

    /// Re-enters the active window after contention has subsided.
    async fn resume(&self, permit: &mut ImportPermitState) {
        debug_assert!(matches!(permit, ImportPermitState::WaitingRetry));
        self.wait_until(|state| {
            if state.active < state.limit {
                debug_assert!(state.waiting_retries > 0);
                state.waiting_retries -= 1;
                state.active += 1;
                *permit = ImportPermitState::Active;
                true
            } else {
                false
            }
        })
        .await;
        self.changed.notify_waiters();
    }

    /// Runs one capacity check with Notify registration ordered to avoid lost
    /// wakeups between checking the state and awaiting a change.
    async fn wait_until(&self, mut acquire: impl FnMut(&mut ImportConcurrencyState) -> bool) {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut state = self.lock_state();
                if acquire(&mut state) {
                    return;
                }
            }
            notified.await;
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, ImportConcurrencyState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

/// Cancellation-safe registration of demand for one new batch.
struct SubmissionWaiter<'a> {
    concurrency: &'a ImportConcurrency,
    registered: bool,
}

impl<'a> SubmissionWaiter<'a> {
    fn new(concurrency: &'a ImportConcurrency) -> Self {
        Self {
            concurrency,
            registered: false,
        }
    }

    fn register(&mut self, state: &mut ImportConcurrencyState) {
        if !self.registered {
            debug_assert!(!state.submission_waiting);
            state.submission_waiting = true;
            self.registered = true;
        }
    }

    fn unregister(&mut self, state: &mut ImportConcurrencyState) {
        if self.registered {
            debug_assert!(state.submission_waiting);
            state.submission_waiting = false;
            self.registered = false;
        }
    }
}

impl Drop for SubmissionWaiter<'_> {
    fn drop(&mut self) {
        if self.registered {
            let mut state = self.concurrency.lock_state();
            debug_assert!(state.submission_waiting);
            state.submission_waiting = false;
            self.registered = false;
        }
    }
}

/// One batch's place in an Import Session's learned concurrency window.
///
/// Retryable conflicts pause and reacquire the permit before the next whole
/// attempt. This lets already-running batches finish and releases any
/// Structure Maintenance they discovered before the retry consumes another
/// bounded attempt.
pub(crate) struct ImportPermit<B: Backend> {
    concurrency: Arc<ImportConcurrency>,
    runtime: Arc<RuntimeInner<B>>,
    state: ImportPermitState,
    contended: bool,
}

/// One accepted batch's mutually exclusive controller state.
enum ImportPermitState {
    Active,
    WaitingRetry,
    Released,
}

impl<B: Backend> ImportPermit<B> {
    /// Contracts admission as soon as an attempt reports contention.
    pub(crate) fn observe_contention(&mut self) {
        self.contended = true;
        self.concurrency.pause_for_contention(&mut self.state);
    }

    /// Waits for maintenance to quiesce and reacquires capacity before retry.
    pub(crate) async fn resume_after_backoff(&mut self) {
        self.runtime.wait_for_backlog_before_retry().await;
        self.concurrency.resume(&mut self.state).await;
    }

    /// Records a successful batch and releases its active capacity.
    pub(crate) fn complete(mut self) {
        let clean = !self.contended;
        self.concurrency.finish(&mut self.state, clean);
    }
}

impl<B: Backend> Drop for ImportPermit<B> {
    fn drop(&mut self) {
        self.concurrency.finish(&mut self.state, false);
    }
}

/// Coordinates bounded batch admission for one Import Session.
///
/// The coordinator is neither cloneable nor shared: the owning public session
/// serializes `submit` calls through `&mut self`, and `drain` consumes it.
pub(crate) struct ImportCoordinator<B: Backend> {
    index: Index<B>,
    concurrency: Arc<ImportConcurrency>,
    next_token: u64,
    batches: Vec<(BatchToken, JoinHandle<Result<Vec<MutationOutcome>>>)>,
}

impl<B: Backend> ImportCoordinator<B> {
    /// Creates a coordinator that learns concurrency up to
    /// `max_in_flight_batches` accepted concurrent batches.
    ///
    /// `max_in_flight_batches` is positive: both the Runtime configuration default
    /// and the `ImportOptions` override are validated upstream.
    pub(crate) fn new(index: Index<B>, max_in_flight_batches: usize) -> Self {
        debug_assert!(
            max_in_flight_batches > 0,
            "import maximum in-flight limits are validated"
        );
        Self {
            index,
            concurrency: ImportConcurrency::new(max_in_flight_batches),
            next_token: 0,
            batches: Vec::new(),
        }
    }

    /// Admits one validated, caller-owned mutation batch and returns its token.
    ///
    /// `mutations` must already be validated against the index's immutable
    /// configuration. An empty batch does no storage work — matching ordinary
    /// `batch_mutate` — so it is admitted without occupying an in-flight slot
    /// or waiting for the backlog gate.
    ///
    /// A non-empty batch waits for an in-flight slot and then for the Fixup
    /// backlog gate before admitting; dropping this future mid-wait admits
    /// nothing. Admission fails with [`ErrorKind::RuntimeClosed`] once the
    /// Runtime stops accepting work; the residual race with shutdown is
    /// decided by the batch's own foreground admission and reported as its
    /// outcome.
    pub(crate) async fn submit_batch(&mut self, mutations: Vec<Mutation>) -> Result<BatchToken> {
        let runtime = Arc::clone(self.index.runtime());
        if !runtime.is_accepting() {
            return Err(Error::new(ErrorKind::RuntimeClosed));
        }
        if mutations.is_empty() {
            let token = self.issue_token()?;
            let task = runtime.executor.spawn(async { Ok::<_, Error>(Vec::new()) });
            self.batches.push((token, task));
            return Ok(token);
        }
        let slot_wait = Instant::now();
        let permit = self.concurrency.acquire(Arc::clone(&runtime)).await;
        metrics::import_wait(ImportGate::InFlightSlot, slot_wait.elapsed());
        if !runtime.is_accepting() {
            return Err(Error::new(ErrorKind::RuntimeClosed));
        }
        let backlog_wait = Instant::now();
        runtime.wait_for_backlog_below().await?;
        metrics::import_wait(ImportGate::Backlog, backlog_wait.elapsed());
        let token = self.issue_token()?;
        let index = self.index.clone();
        let task = runtime.executor.spawn(async move {
            index
                .run_import_mutations(
                    Operation::BatchMutate,
                    mutations,
                    OperationOptions::default(),
                    permit,
                )
                .await
        });
        self.batches.push((token, task));
        Ok(token)
    }

    /// Issues the next monotonically increasing token within this session.
    fn issue_token(&mut self) -> Result<BatchToken> {
        let value = self
            .next_token
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        self.next_token = value;
        // Token values count up from 1 and never wrap, so they are nonzero.
        Ok(BatchToken::new(value).expect("token values are nonzero"))
    }

    /// Waits for every accepted batch and returns the results in submission
    /// order.
    ///
    /// Batches may complete out of order; awaiting each task at its submission
    /// position restores the contract without discarding any known result. A
    /// panicked batch task reports [`ErrorKind::Backend`] at its own position.
    /// The tasks stay in `self.batches` while they are awaited, so dropping
    /// this future aborts the still-incomplete batches through the same
    /// cancellation path as dropping the session.
    pub(crate) async fn drain(mut self) -> Vec<ImportBatchResult> {
        let mut results = Vec::with_capacity(self.batches.len());
        for index in 0..self.batches.len() {
            let (token, task) = &mut self.batches[index];
            results.push(ImportBatchResult {
                token: *token,
                result: super::join_task(task).await,
            });
        }
        results
    }
}

impl<B: Backend> Drop for ImportCoordinator<B> {
    fn drop(&mut self) {
        for (_token, task) in &self.batches {
            task.abort();
        }
    }
}
