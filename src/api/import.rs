//! Bounded Import Sessions.

use std::fmt;

use crate::runtime::import::ImportCoordinator;
use crate::storage::backend::Backend;

use super::{BatchToken, ImportBatchResult, Index, Mutation, Result};

/// A non-cloneable process-local coordinator that submits bounded waves of
/// ordinary atomic mutation batches into one Index.
///
/// `submit` applies ordinary batch validation, waits for an in-flight slot
/// and the Runtime's Structure Maintenance backlog gate, admits exactly one
/// ordinary atomic mutation operation, and returns a process-local
/// [`BatchToken`]. The
/// gate pauses admission while the process-local Fixup backlog is at or above
/// the configured watermark; it is process-local backpressure, never a durable
/// or cluster-wide barrier. Tokens are monotonically increasing within
/// the session and have no persistent or transaction identity. Accepted
/// batches may execute concurrently within the session's in-flight bound and
/// behave exactly like [`Index::batch_mutate`]: the same atomicity, retry,
/// error, and maintenance behavior.
///
/// `finish` closes admission, waits for every accepted batch, and returns one
/// [`ImportBatchResult`] per token in submission order; a failed batch is
/// reported in place and never discards other known results. Dropping the
/// session instead cancels batches whose commit has not started, while started
/// commits finish under the Runtime's in-flight guard without an observer.
///
/// Admission, concurrency, queued bytes, and memory are bounded: at most the
/// configured in-flight limit of batches execute concurrently, there is no
/// session-internal queue (a caller waiting in `submit` holds its own
/// unsubmitted batch), and every admitted batch still passes the Runtime's
/// bounded foreground admission. No import state is persistent, and import
/// never claims a cluster-wide maintenance barrier or an atomic whole-import
/// result.
///
/// `Debug` redacts the Index Name.
pub struct ImportSession<B: Backend> {
    index: Index<B>,
    coordinator: ImportCoordinator<B>,
}

impl<B: Backend> ImportSession<B> {
    pub(crate) fn new(index: Index<B>, in_flight_batches: usize) -> Self {
        let coordinator = ImportCoordinator::new(index.clone(), in_flight_batches);
        Self { index, coordinator }
    }

    /// Validates and submits one atomic mutation batch, returning its token.
    ///
    /// Validation matches [`Index::batch_mutate`]: an invalid batch fails with
    /// [`crate::api::ErrorKind::InvalidArgument`] at the offending input
    /// position and no token is issued. An empty batch is accepted with an
    /// empty result and never occupies an in-flight slot or waits for the
    /// backlog gate. Waiting for a slot or the gate is bounded by the caller:
    /// dropping this future before it returns cancels the wait and admits
    /// nothing.
    ///
    /// # Errors
    ///
    /// Returns [`crate::api::ErrorKind::RuntimeClosed`] once the Runtime stops
    /// accepting work. A batch accepted before shutdown always reports its
    /// real result through `finish` instead.
    pub async fn submit(&mut self, mut mutations: Vec<Mutation>) -> Result<BatchToken> {
        self.index.validate(&mut mutations)?;
        self.coordinator.submit_batch(mutations).await
    }

    /// Closes admission, waits for every accepted batch, and returns the batch
    /// results in submission order.
    ///
    /// Batches may complete out of order; the returned order is the submission
    /// order. A failed batch reports its error in place without discarding
    /// other results, so the caller can select the first failure without
    /// losing outcome information. `finish` waits only for the session's own
    /// batches: process-local or cluster-wide topology convergence remains
    /// demand-driven and separately observable. Dropping this future cancels
    /// the still-incomplete batches exactly like dropping the session.
    #[must_use]
    pub async fn finish(self) -> Vec<ImportBatchResult> {
        self.coordinator.drain().await
    }
}

impl<B: Backend> fmt::Debug for ImportSession<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImportSession")
            .field("name", &"[REDACTED]")
            .field("logical_index_id", &self.index.logical_index_id())
            .finish()
    }
}
