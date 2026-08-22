//! Import Session batch admission and ordered outcome collection.
//!
//! One coordinator owns the process-local state of a single Import Session: a
//! bounded in-flight slot semaphore, the monotonically increasing Batch Token
//! sequence, and the accepted batch tasks in submission order. Every admitted
//! batch runs the ordinary foreground mutation path ([`Index::run_mutations`]),
//! so its atomicity, retry, error, and maintenance behavior is identical to a
//! normal `batch_mutate` and import changes neither Foreground Mutation
//! atomicity nor Logical Index lifecycle.
//!
//! Admission is bounded in both directions: at most the session's in-flight
//! limit of batches execute concurrently, and no session-internal queue
//! exists — a caller waiting for a slot holds its own unsubmitted batch, while
//! admitted batches additionally pass the Runtime's bounded foreground
//! admission. Session memory is bounded by the in-flight batch payloads plus
//! one outcome entry per accepted token, collected on `drain`.
//!
//! Dropping the coordinator aborts every incomplete batch task. Tokio drops an
//! aborted task's future, which runs the caller-drop guard inside
//! `run_foreground`: a batch whose commit has not started is cancelled, while
//! a started commit keeps running detached under the Runtime's in-flight guard
//! and finishes without an Import Session observer.
//!
//! `submit_batch` will also wait for the Runtime's Structure Maintenance
//! backlog watermark against the Fixup queue; the gate wiring is tracked by
//! #89, and the admission bounds here are independent of it.

use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

use crate::api::{
    BatchToken, Error, ErrorKind, ImportBatchResult, Index, Mutation, MutationOutcome,
    OperationOptions, Result,
};
use crate::storage::backend::Backend;

/// Coordinates bounded batch admission for one Import Session.
///
/// The coordinator is neither cloneable nor shared: the owning public session
/// serializes `submit` calls through `&mut self`, and `drain` consumes it.
pub(crate) struct ImportCoordinator<B: Backend> {
    index: Index<B>,
    slots: Arc<Semaphore>,
    next_token: u64,
    batches: Vec<(BatchToken, JoinHandle<Result<Vec<MutationOutcome>>>)>,
}

impl<B: Backend> ImportCoordinator<B> {
    /// Creates a coordinator admitting at most `in_flight_batches` concurrent
    /// batches.
    ///
    /// `in_flight_batches` is positive: both the Runtime configuration default
    /// and the `ImportOptions` override are validated upstream.
    pub(crate) fn new(index: Index<B>, in_flight_batches: usize) -> Self {
        debug_assert!(
            in_flight_batches > 0,
            "import in-flight limits are validated"
        );
        Self {
            index,
            slots: Arc::new(Semaphore::new(in_flight_batches)),
            next_token: 0,
            batches: Vec::new(),
        }
    }

    /// Admits one validated, caller-owned mutation batch and returns its token.
    ///
    /// `mutations` must already be validated against the index's immutable
    /// configuration. An empty batch does no storage work — matching ordinary
    /// `batch_mutate` — so it is admitted without occupying an in-flight slot.
    ///
    /// Admission fails with [`ErrorKind::RuntimeClosed`] once the Runtime stops
    /// accepting work; the residual race with shutdown is decided by the
    /// batch's own foreground admission and reported as its outcome.
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
        let slot = self
            .slots
            .clone()
            .acquire_owned()
            .await
            .expect("the session slot semaphore is never closed");
        if !runtime.is_accepting() {
            return Err(Error::new(ErrorKind::RuntimeClosed));
        }
        let token = self.issue_token()?;
        let index = self.index.clone();
        let task = runtime.executor.spawn(async move {
            let result = index
                .run_mutations(mutations, OperationOptions::default())
                .await;
            drop(slot);
            result
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
