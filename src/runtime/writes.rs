//! Whole-attempt write transaction scaffolding (ADR 0012).
//!
//! This module is the single home of the begin-write → manifest-validation →
//! whole-retry scaffolding shared by foreground mutations and Structure
//! Maintenance steps. Each attempt opens a fresh write transaction,
//! update-protects and validates the persisted Active Index Manifest so no
//! operation commits into a dropping Logical Index, runs the caller's step,
//! and commits. A definite abort replays the whole step from a fresh snapshot
//! under the caller's bounded [`RetryPolicy`]; exhaustion returns
//! `ContentionExhausted`. A commit of unknown outcome is returned, never
//! retried (ADR 0012): the caller recovers by re-driving the same operation,
//! which observes the persisted state and proceeds idempotently.
//!
//! A foreground operation passes its [`OperationContext`] so every attempt
//! honors the caller's cancellation and deadline and the commit crosses the
//! Runtime's native commit boundary; a maintenance step has no caller control
//! and commits plainly.

use std::future::Future;
use std::time::Instant;

use crate::api::{ErrorKind, Result};
use crate::observe::labels::{Operation, WriteAttemptOutcome};
use crate::observe::metrics;
use crate::storage::WriteLogicalTxn;
use crate::storage::backend::Backend;
use crate::storage::values::IndexManifest;

use super::OperationContext;
use super::import::ImportPermit;
use super::lifecycle::RetryPolicy;
use super::reads;

/// One boxed attempt step future, tied to the transaction borrow.
///
/// The explicit `Send` bound keeps the whole-attempt loop's future `Send`
/// under the Runtime's foreground executor; the higher-ranked transaction
/// lifetime would otherwise defeat the `Send` analysis.
pub(crate) type StepFuture<'a, O> = std::pin::Pin<Box<dyn Future<Output = Result<O>> + Send + 'a>>;

/// Opens one write transaction bound to the Logical Index, update-protecting
/// and validating the persisted Active Manifest of the opened handle first.
///
/// The Manifest conflict aborts the transaction if a concurrent drop
/// transition commits. Validation proves the persisted Manifest carries the
/// handle's exact immutable identity, so binding the handle manifest is
/// equivalent to binding the persisted one.
pub(crate) async fn open_validated_write<'b, 'm, B: Backend>(
    backend: &'b B,
    handle_manifest: &'m IndexManifest,
) -> Result<WriteLogicalTxn<'m, B::WriteTxn<'b>>> {
    let raw = backend.begin_write().await?;
    let hard_limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let mut txn = WriteLogicalTxn::bootstrap(raw, hard_limits, budget);
    if let Err(error) = reads::validated_active_manifest(&mut txn, handle_manifest).await {
        txn.rollback().await;
        return Err(error);
    }
    WriteLogicalTxn::for_index(txn.into_raw(), handle_manifest, hard_limits, budget)
}

/// Runs one bounded write operation as a sequence of whole attempts.
///
/// Each attempt opens a fresh manifest-validated write transaction, runs
/// `step`, and commits; the returned outcome is produced only after the
/// commit succeeds. A definite abort discards the attempt and replays the
/// whole step under `retry`; a commit of unknown outcome is returned, never
/// retried (ADR 0012). With `context` — a foreground operation — every
/// attempt checkpoints the caller's cancellation and deadline first and the
/// commit crosses the Runtime's native commit boundary; without it — a
/// maintenance step — the attempt commits plainly.
pub(crate) async fn run_write_attempts<'b, 'm, B: Backend, O>(
    backend: &'b B,
    context: Option<&mut OperationContext<B>>,
    handle_manifest: &'m IndexManifest,
    retry: &RetryPolicy,
    operation: Operation,
    step: impl for<'a> FnMut(&'a mut WriteLogicalTxn<'m, B::WriteTxn<'b>>) -> StepFuture<'a, O>,
) -> Result<O> {
    let mut permit = None;
    run_write_attempts_with_optional_import_permit(
        backend,
        context,
        handle_manifest,
        retry,
        operation,
        &mut permit,
        step,
    )
    .await
}

/// Applies one bounded retry delay around an optional slow-path control.
pub(crate) async fn wait_before_retry<B: Backend>(
    retry: &RetryPolicy,
    operation: Operation,
    failed_attempts: &mut u32,
    permit: &mut Option<&mut ImportPermit<B>>,
) -> Result<()> {
    if let Some(permit) = permit.as_deref_mut() {
        permit.observe_contention();
    }
    retry.wait_or_exhaust(operation, failed_attempts).await?;
    if let Some(permit) = permit.as_deref_mut() {
        permit.resume_after_backoff().await;
    }
    Ok(())
}

/// Records the native commit wait and outcome of one finished commit call.
fn observe_commit(operation: Operation, committed: &Result<()>, started: Instant) {
    metrics::write_commit_finished(
        operation,
        WriteAttemptOutcome::from_result(committed),
        started.elapsed(),
    );
}

/// Shared whole-attempt loop behind the ordinary and observed entry points.
pub(crate) async fn run_write_attempts_with_optional_import_permit<'b, 'm, B: Backend, O>(
    backend: &'b B,
    mut context: Option<&mut OperationContext<B>>,
    handle_manifest: &'m IndexManifest,
    retry: &RetryPolicy,
    operation: Operation,
    permit: &mut Option<&mut ImportPermit<B>>,
    mut step: impl for<'a> FnMut(&'a mut WriteLogicalTxn<'m, B::WriteTxn<'b>>) -> StepFuture<'a, O>,
) -> Result<O> {
    let mut failed_attempts = 0_u32;
    loop {
        if let Some(context) = context.as_deref_mut() {
            context.checkpoint()?;
        }
        let mut txn = open_validated_write(backend, handle_manifest).await?;
        let error = match step(&mut txn).await {
            Ok(outcome) => {
                let size = txn.size();
                let committed = match context.as_deref_mut() {
                    Some(context) => {
                        context
                            .commit(move |start| async move {
                                let commit_started = Instant::now();
                                let committed = txn.commit_with(start).await;
                                observe_commit(operation, &committed, commit_started);
                                committed
                            })
                            .await
                    }
                    None => {
                        let commit_started = Instant::now();
                        let committed = txn.commit().await;
                        observe_commit(operation, &committed, commit_started);
                        committed
                    }
                };
                let attempt_outcome = WriteAttemptOutcome::from_result(&committed);
                metrics::write_attempt_finished(
                    operation,
                    attempt_outcome,
                    size.mutations(),
                    size.bytes(),
                );
                match committed {
                    Ok(()) => return Ok(outcome),
                    // The commit boundary is included in the whole-attempt
                    // retry; an unknown outcome is returned, never retried.
                    Err(error) => error,
                }
            }
            Err(error) => {
                let size = txn.size();
                txn.rollback().await;
                metrics::write_attempt_finished(
                    operation,
                    WriteAttemptOutcome::from_error(error.kind()),
                    size.mutations(),
                    size.bytes(),
                );
                error
            }
        };
        if error.kind() != ErrorKind::RetryableAbort {
            return Err(error);
        }
        wait_before_retry(retry, operation, &mut failed_attempts, permit).await?;
    }
}

/// Boxes one attempt step future with the `Send` bound
/// [`run_write_attempts`] requires.
pub(crate) fn boxed_step<'a, O>(
    future: impl Future<Output = Result<O>> + Send + 'a,
) -> StepFuture<'a, O> {
    Box::pin(future)
}
