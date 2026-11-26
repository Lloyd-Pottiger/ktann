//! Unified sharded Bulk Build worker claims, transaction chunking, and workspace reclamation.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio_util::sync::{CancellationToken, DropGuard};

use crate::api::{
    BulkBuildReport, BulkLoadOptions, BulkWorkerOptions, Error, ErrorKind, LogicalIndexId,
    OperationOptions, PartitionKey, Result,
};
use crate::bulk::{LtHash, ShardPlan};
use crate::observe::labels::Operation;
use crate::storage::backend::{AdmissionBudget, Backend, HardLimits, ScanLimits, WriteTxn};
use crate::storage::keys::{self, LogicalKey};
use crate::storage::values::{
    BuildDescriptor, BuildPhase, BuildProgress, BuildWorkspace, IndexLifecycle, IndexManifest,
    MAX_SPLIT_KEYS, PersistentValue,
};
use crate::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

use super::{
    OperationContext,
    bulk_scheduler::{
        BulkAuthority, acquire_unscheduled_peer_slot, authorize_raw,
        ensure_no_peer_scheduler_leases, non_zero_token, release_unscheduled_peer_slot,
        retire_orphaned_schedule, run_authority_lease_renewals,
    },
    lifecycle::RetryPolicy,
};

#[path = "bulk_pipeline.rs"]
pub(crate) mod bulk_pipeline;

pub(super) const MAX_WRITER_LANES: usize = 16;
const MAX_TXN_AFFECTED_BYTES: usize = 8 * 1024 * 1024;
const BUILD_UNIT_MUTATION_BYTES: usize = 296;

#[derive(Clone, Copy)]
pub(crate) struct WriteFence {
    pub epoch: u64,
    pub owner: [u8; 32],
}

/// Coordinates background lease heartbeats with foreground chunk/completion commits
/// on the same `BuildUnit` so a worker never conflict-aborts its own transactions.
#[derive(Clone)]
pub(crate) struct UnitLease {
    pub stage: u8,
    pub unit: u32,
    pub owner: [u8; 32],
    pub duration: Duration,
    pub lock: Arc<tokio::sync::Mutex<()>>,
}

/// Outcome of one prepare-and-commit write attempt before retry policy resolution.
pub(crate) enum CommitStep<T> {
    Done(T),
    PrepareAbort,
    CommitFailed(Error),
    Fatal(Error),
}

impl<T> CommitStep<T> {
    pub(crate) async fn resolve(
        self,
        op: Operation,
        retry: RetryPolicy,
        attempts: &mut u32,
    ) -> Result<Option<T>> {
        match self {
            Self::Done(v) => Ok(Some(v)),
            Self::PrepareAbort => {
                retry.wait_or_exhaust(op, attempts).await?;
                Ok(None)
            }
            Self::CommitFailed(e) => {
                retry.after_commit_error(op, attempts, e).await?;
                Ok(None)
            }
            Self::Fatal(e) => Err(e),
        }
    }
}

pub(crate) async fn commit_raw_prepared<W: WriteTxn, T>(
    prepared: Result<(Option<WriteLogicalTxn<'_, W>>, T)>,
) -> CommitStep<T> {
    match prepared {
        Ok((None, out)) => CommitStep::Done(out),
        Ok((Some(txn), out)) => match txn.commit().await {
            Ok(()) => CommitStep::Done(out),
            Err(e) => CommitStep::CommitFailed(e),
        },
        Err(e) if e.kind() == ErrorKind::RetryableAbort => CommitStep::PrepareAbort,
        Err(e) => CommitStep::Fatal(e),
    }
}

pub(crate) async fn lock_sched_read(
    lock: &Option<Arc<tokio::sync::RwLock<()>>>,
) -> Option<tokio::sync::RwLockReadGuard<'_, ()>> {
    match lock {
        Some(l) => Some(l.read().await),
        None => None,
    }
}

/// Executes a non-context write transaction step with scheduler read-lock protection and retries.
pub(crate) async fn retry_raw_write<T, F, Fut>(
    control: &OperationOptions,
    sched_lock: Option<&tokio::sync::RwLock<()>>,
    op: Operation,
    retry: RetryPolicy,
    mut step: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = CommitStep<T>>,
{
    let mut attempts = 0;
    loop {
        super::check_control(control)?;
        let guard = match sched_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
        let outcome = step().await;
        drop(guard);
        if let Some(done) = outcome.resolve(op, retry, &mut attempts).await? {
            return Ok(done);
        }
    }
}

/// Executes a read-only backend step with bounded `RetryableAbort` retries and control checks.
pub(crate) async fn retry_read<T, F, Fut>(
    control: &OperationOptions,
    op: Operation,
    retry: RetryPolicy,
    mut step: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut attempts = 0;
    loop {
        super::check_control(control)?;
        match step().await {
            Ok(v) => return Ok(v),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry.wait_or_exhaust(op, &mut attempts).await?;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Binds a drop-triggered cancellation token to `base` so dropping the owning async future
/// immediately cancels synchronous CPU/disk loops running inside `spawn_blocking`.
pub(crate) fn drop_guarded_options(base: &OperationOptions) -> (OperationOptions, DropGuard) {
    let token = match base.cancellation() {
        Some(parent) => parent.child_token(),
        None => CancellationToken::new(),
    };
    let guard = token.clone().drop_guard();
    (base.clone().with_cancellation(token), guard)
}

/// Shared root advisory lock retained across native workspace IO; exclusive cleanup
/// cannot delete directories while any local worker holds a shared lock.
pub(crate) fn lock_root(root: &Path, exclusive: bool) -> Result<File> {
    let path = root.join(".ktann-build-lock");
    if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(Error::invalid_argument());
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(io)?;
    let result = if exclusive {
        lock.try_lock()
    } else {
        lock.try_lock_shared()
    };
    result.map_err(|e| match e {
        std::fs::TryLockError::WouldBlock => Error::new(ErrorKind::BulkBuildBusy),
        std::fs::TryLockError::Error(e) => io(e),
    })?;
    Ok(lock)
}

/// Runs synchronous CPU/disk work while retaining foreground CPU admission and drop-cancellation.
pub(crate) async fn blocking<B: Backend, T: Send + 'static>(
    context: &OperationContext<B>,
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    context.compute(|_options| work()).await
}

/// Reads and conflict-protects the exact handle identity before lifecycle decisions.
pub(crate) async fn manifest<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    expected: &IndexManifest,
) -> Result<IndexManifest> {
    match txn
        .get_for_update(LogicalKey::Manifest(expected.logical_index_id()))
        .await?
    {
        Some(PersistentValue::IndexManifest(m)) if m.has_same_immutable_identity(expected) => Ok(m),
        None => Err(Error::new(ErrorKind::IndexNotFound)),
        _ => Err(corrupt()),
    }
}

/// Entry-point lifecycle check (`Active` rejects reservation/schedule/manual restart with `InvalidArgument`).
pub(crate) async fn building<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    expected: &IndexManifest,
) -> Result<()> {
    match manifest(txn, expected).await?.lifecycle() {
        IndexLifecycle::Building => Ok(()),
        IndexLifecycle::Dropping => Err(Error::new(ErrorKind::IndexDropping)),
        IndexLifecycle::Active => Err(Error::invalid_argument()),
    }
}

/// In-flight lifecycle check for worker tasks, validation slices, and sealing (`Active` means
/// a peer already published this build and supersedes our in-flight attempt).
pub(crate) async fn building_in_flight<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    expected: &IndexManifest,
) -> Result<()> {
    match manifest(txn, expected).await?.lifecycle() {
        IndexLifecycle::Building => Ok(()),
        IndexLifecycle::Dropping => Err(Error::new(ErrorKind::IndexDropping)),
        IndexLifecycle::Active => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
    }
}

pub(crate) async fn normalize_options<B: Backend>(
    context: &OperationContext<B>,
    mut options: BulkWorkerOptions,
) -> Result<BulkWorkerOptions> {
    options.validate()?;
    let root = options.workspace.clone();
    options.workspace = blocking(context, move || {
        let root = fs::canonicalize(root).map_err(io)?;
        if !fs::metadata(&root).map_err(io)?.is_dir() {
            return Err(Error::invalid_argument());
        }
        Ok(root)
    })
    .await?;
    options.validate()?;
    Ok(options)
}

async fn fill_report_counts_from_progress<B: Backend>(
    context: &OperationContext<B>,
    id: LogicalIndexId,
    retry: RetryPolicy,
    report: &mut BulkBuildReport,
) -> Result<()> {
    if report.records > 0 || report.entries > 0 {
        return Ok(());
    }
    let backend = context.backend();
    let val = retry_read(&context.options, Operation::PublishBulkBuild, retry, || {
        let backend = Arc::clone(&backend);
        async move {
            let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
            rtxn.get(LogicalKey::BuildProgress(id)).await
        }
    })
    .await?;
    if let Some(PersistentValue::BuildProgress(p)) = val {
        report.records = p.total_records;
        report.trees = p.trees;
        report.partitions = p.partitions;
        report.entries = p.total_entries;
    }
    Ok(())
}

async fn run_with_optional_peer_slot<B, T, F>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    peer_slot: Option<ActivePeerSlot>,
    options: &BulkWorkerOptions,
    retry: RetryPolicy,
    work: F,
) -> Result<T>
where
    B: Backend,
    F: AsyncFnOnce(&mut OperationContext<B>) -> Result<T>,
{
    let Some((slot, owner, initial_exp, lease_lock)) = peer_slot else {
        return work(context).await;
    };
    let backend = context.backend();
    let auth = BulkAuthority::UnscheduledPeer { slot, owner };
    let dist = options.distributed_workers;
    let lease = options.unit_lease;
    let outcome = tokio::select! {
        res = work(context) => res,
        err = run_authority_lease_renewals(backend, lease_lock, id, auth, dist, lease, initial_exp) => Err(err),
    };
    release_unscheduled_peer_slot(context, id, slot, owner, retry).await?;
    outcome
}

type ActivePeerSlot = (u32, [u8; 32], u64, Arc<tokio::sync::RwLock<()>>);

async fn maybe_acquire_peer_slot<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    options: &BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<Option<ActivePeerSlot>> {
    if options.distributed_workers <= 1 || context.bulk_authority.is_some() {
        return Ok(None);
    }
    let (slot, owner, initial_exp) =
        acquire_unscheduled_peer_slot(context, index, options, retry).await?;
    let lease_lock = Arc::new(tokio::sync::RwLock::new(()));
    context.bulk_lease_lock = Some(Arc::clone(&lease_lock));
    Ok(Some((slot, owner, initial_exp, lease_lock)))
}

async fn complete_attempt<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    options: &BulkWorkerOptions,
    retry: RetryPolicy,
    report: &mut BulkBuildReport,
) -> Result<IndexManifest> {
    let started = Instant::now();
    let publication =
        super::bulk_publish::publish(context, index.clone(), descriptor.clone(), retry).await;
    report.publish += started.elapsed();
    match publication {
        Err(error) if error.kind() == ErrorKind::BulkBuildBusy => {
            let normalized = normalize_options(context, options.clone()).await?;
            let pub_elapsed = report.publish;
            *report = run_normalized(
                context,
                index.clone(),
                descriptor.clone(),
                normalized,
                retry,
            )
            .await?;
            report.publish = pub_elapsed;
            let started = Instant::now();
            let publication =
                super::bulk_publish::publish(context, index.clone(), descriptor.clone(), retry)
                    .await;
            report.publish += started.elapsed();
            publication
        }
        other => other,
    }
}

/// Resumes the durable phase and atomically publishes the original identity.
pub(crate) async fn complete<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    descriptor: BuildDescriptor,
    options: BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<(IndexManifest, BulkBuildReport)> {
    options.validate()?;
    let mut attempts = 0;
    let mut report = BulkBuildReport::default();
    let id = index.logical_index_id();
    loop {
        context.checkpoint()?;
        let peer_slot = if super::bulk_publish::already_active_retry(context, &index, retry)
            .await?
            .is_some()
        {
            Ok(None)
        } else {
            maybe_acquire_peer_slot(context, &index, &options, retry).await
        };
        let result = match peer_slot {
            Ok(slot_opt) => {
                run_with_optional_peer_slot(context, id, slot_opt, &options, retry, async |ctx| {
                    complete_attempt(ctx, &index, &descriptor, &options, retry, &mut report).await
                })
                .await
            }
            Err(error) => Err(error),
        };

        let result = match result {
            Ok(active) => Ok(active),
            Err(error) => {
                match super::bulk_publish::already_active_retry(context, &index, retry).await {
                    Ok(Some(active)) => Ok(active),
                    _ => Err(error),
                }
            }
        };
        match result {
            Ok(active) => {
                fill_report_counts_from_progress(context, id, retry, &mut report).await?;
                let started = Instant::now();
                match cleanup(context, id, retry).await {
                    Err(error)
                        if error.kind() == ErrorKind::BulkBuildBusy
                            && (options.distributed_workers > 1
                                || context.bulk_authority.is_none()) => {}
                    result => result?,
                }
                report.publish += started.elapsed();
                report.rotation_seed = *index.rotation_seed();
                return Ok((active, report));
            }
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::BulkBuildBusy | ErrorKind::BulkBuildSuperseded
                ) && !retry.would_exhaust(attempts) =>
            {
                retry.wait(attempts).await;
                attempts += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Runs sharded construction and backend emission, leaving the index in `Loaded`.
pub(crate) async fn run<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    descriptor: BuildDescriptor,
    options: BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<BulkBuildReport> {
    let options = normalize_options(context, options).await?;
    let id = index.logical_index_id();
    let peer_slot = maybe_acquire_peer_slot(context, &index, &options, retry).await?;
    let options_ref = options.clone();
    run_with_optional_peer_slot(context, id, peer_slot, &options_ref, retry, async |ctx| {
        run_normalized(ctx, index, descriptor, options, retry).await
    })
    .await
}

async fn run_normalized<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    descriptor: BuildDescriptor,
    options: BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<BulkBuildReport> {
    if options.distributed_workers > 1
        && super::bulk_publish::already_active_retry(context, &index, retry)
            .await?
            .is_some()
    {
        let mut report = BulkBuildReport {
            rotation_seed: *index.rotation_seed(),
            ..BulkBuildReport::default()
        };
        fill_report_counts_from_progress(context, index.logical_index_id(), retry, &mut report)
            .await?;
        return Ok(report);
    }

    let root = options.workspace.clone();
    let lock = Arc::new(blocking(context, move || lock_root(&root, false)).await?);
    let (state, fence, already_loaded) =
        claim(context, &index, &descriptor, options, retry).await?;
    if let Some(p) = already_loaded {
        return Ok(BulkBuildReport {
            records: p.total_records,
            trees: p.trees,
            partitions: p.partitions,
            entries: p.total_entries,
            rotation_seed: *index.rotation_seed(),
            ..BulkBuildReport::default()
        });
    }

    {
        let dir = state.directory();
        let guard = lock.clone();
        blocking(context, move || {
            let _lock = guard;
            fs::create_dir_all(&dir).map_err(io)?;
            File::open(dir.parent().expect("workspace root"))
                .and_then(|f| f.sync_all())
                .map_err(io)
        })
        .await?;
    }

    let in_memory = state.options.distributed_workers == 1
        && fits_in_memory(&index, &descriptor, &state.options);
    let result = bulk_pipeline::run_pipeline(
        context,
        &index,
        &descriptor,
        &state,
        fence,
        in_memory,
        lock,
        retry,
    )
    .await;
    if let Err(error) = &result {
        record_failure(context, &index, &state, Some(fence), error.kind(), retry).await?;
    }
    result
}

fn fits_in_memory(
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    options: &BulkWorkerOptions,
) -> bool {
    let items = descriptor.input().items();
    if items == 0 {
        return true;
    }
    let effective_mem = options.memory_bytes.min(descriptor.options().memory_bytes) as u64;
    let dim = index.config().dimension() as u64;
    let per_row = dim.saturating_mul(4).saturating_add(224);
    let wave_size = (options.parallelism.clamp(1, 16)) as u64;
    let est_segments = items.div_ceil(16_384).max(1);
    let avg_seg_bytes = descriptor.input().bytes() / est_segments;
    let wave_frame_bytes = avg_seg_bytes.saturating_mul(wave_size.min(est_segments));
    let seg_overhead = items.min(16_384).saturating_mul(per_row);
    let needed = items
        .saturating_mul(per_row)
        .saturating_add(wave_frame_bytes)
        .saturating_add(seg_overhead)
        .saturating_add(4096);
    needed <= effective_mem
}

/// Claims or joins the workspace and installs `BuildWorkspace.writer_token`.
async fn claim<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    options: BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<(BuildWorkspace, WriteFence, Option<BuildProgress>)> {
    let ws_token = non_zero_token()?;
    let writer_token = non_zero_token()?;

    let _ = effective_budget(context.backend().admission_budget(), options.load)?;
    let id = index.logical_index_id();

    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let prepared = async {
            let mut txn = WriteLogicalTxn::bootstrap(
                backend.begin_write().await?,
                backend.hard_limits(),
                backend.admission_budget(),
            );
            let current_manifest = manifest(&mut txn, index).await?;
            match current_manifest.lifecycle() {
                IndexLifecycle::Building => {}
                IndexLifecycle::Dropping => return Err(Error::new(ErrorKind::IndexDropping)),
                IndexLifecycle::Active if options.distributed_workers > 1 => {
                    if let Some(PersistentValue::BuildProgress(p)) =
                        txn.get(LogicalKey::BuildProgress(id)).await?
                    {
                        txn.rollback().await;
                        return Ok((
                            None,
                            BuildWorkspace {
                                options: options.clone(),
                                hard_limits: backend.hard_limits(),
                                token: ws_token,
                                writer_token,
                                epoch: p.epoch,
                                failure: None,
                            },
                            WriteFence {
                                epoch: p.epoch,
                                owner: writer_token,
                            },
                            Some(p),
                        ));
                    }
                    return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                }
                IndexLifecycle::Active => return Err(Error::invalid_argument()),
            }
            super::bulk_scheduler::authorize(context, &mut txn, id).await?;
            match txn.get(LogicalKey::BuildDescriptor(id)).await? {
                Some(PersistentValue::BuildDescriptor(d)) if d == *descriptor => {}
                _ => return Err(corrupt()),
            }

            let existing_progress = match txn.get_for_update(LogicalKey::BuildProgress(id)).await? {
                None => None,
                Some(PersistentValue::BuildProgress(p)) => Some(p),
                _ => return Err(corrupt()),
            };

            let (state, fence) = match txn.get_for_update(LogicalKey::BuildWorkspace(id)).await? {
                None => {
                    if existing_progress
                        .as_ref()
                        .is_some_and(|p| p.phase == BuildPhase::Validating)
                    {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    let state = BuildWorkspace {
                        options: options.clone(),
                        hard_limits: backend.hard_limits(),
                        token: ws_token,
                        writer_token,
                        epoch: 1,
                        failure: None,
                    };
                    (
                        state,
                        WriteFence {
                            epoch: 1,
                            owner: writer_token,
                        },
                    )
                }
                Some(PersistentValue::BuildWorkspace(mut w)) => {
                    if w.options != options || w.hard_limits != backend.hard_limits() {
                        return Err(Error::invalid_argument());
                    }
                    if let Some(kind) = w.failure {
                        return Err(Error::new(kind));
                    }
                    if w.writer_token == [0; 32]
                        || existing_progress
                            .as_ref()
                            .is_some_and(|p| p.phase == BuildPhase::Validating)
                    {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    if let Some(p) = existing_progress.as_ref()
                        && p.phase == BuildPhase::Loaded
                    {
                        let loaded_prog = p.clone();
                        txn.rollback().await;
                        return Ok((
                            None,
                            w.clone(),
                            WriteFence {
                                epoch: w.epoch,
                                owner: w.writer_token,
                            },
                            Some(loaded_prog),
                        ));
                    }
                    if options.distributed_workers > 1 {
                        let existing_fence = WriteFence {
                            epoch: w.epoch,
                            owner: w.writer_token,
                        };
                        if existing_progress
                            .as_ref()
                            .is_some_and(|p| p.epoch == w.epoch)
                        {
                            txn.rollback().await;
                            return Ok((None, w, existing_fence, None));
                        }
                        (w, existing_fence)
                    } else {
                        w.epoch = w
                            .epoch
                            .checked_add(1)
                            .ok_or_else(|| Error::new(ErrorKind::IdExhausted))?;
                        w.writer_token = writer_token;
                        (
                            w.clone(),
                            WriteFence {
                                epoch: w.epoch,
                                owner: writer_token,
                            },
                        )
                    }
                }
                _ => return Err(corrupt()),
            };

            txn.put(
                LogicalKey::BuildWorkspace(id),
                PersistentValue::BuildWorkspace(state.clone()),
            )
            .await?;
            if existing_progress.is_none() || options.distributed_workers == 1 {
                let disk_mode =
                    options.distributed_workers > 1 || !fits_in_memory(index, descriptor, &options);
                let total_records = descriptor.input().items();
                let est_entries = total_records.saturating_mul(4).saturating_add(64);
                let progress = BuildProgress {
                    phase: if disk_mode {
                        BuildPhase::Preparing
                    } else {
                        BuildPhase::Loading
                    },
                    epoch: fence.epoch,
                    total_records,
                    loaded_entries: 0,
                    total_entries: if total_records == 0 { 0 } else { est_entries },
                    trees: 0,
                    partitions: 0,
                    expected_kv_lthash: LtHash::ZERO,
                    validation_split_keys: Vec::new(),
                };
                txn.put(
                    LogicalKey::BuildProgress(id),
                    PersistentValue::BuildProgress(progress),
                )
                .await?;
            }
            Ok((Some(txn), state, fence, None))
        }
        .await;

        let (txn_opt, state, fence, loaded_opt) = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        if loaded_opt.is_some() || txn_opt.is_none() {
            return Ok((state, fence, loaded_opt));
        }
        let txn = txn_opt.expect("prepared write txn");
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok((state, fence, None)),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    }
}

pub(crate) fn effective_budget(
    mut budget: AdmissionBudget,
    load: BulkLoadOptions,
) -> Result<AdmissionBudget> {
    budget.max_mutations = budget.max_mutations.min(load.max_mutations);
    budget.max_mutation_bytes = budget.max_mutation_bytes.min(load.max_bytes);
    if budget.max_mutations < 2 || budget.max_mutation_bytes == 0 {
        return Err(Error::invalid_argument());
    }
    Ok(budget)
}

/// Advances `BuildProgress` (`Preparing -> Loading` and/or `loaded_entries += delta`).
#[expect(
    clippy::too_many_arguments,
    reason = "progress advancement accepts explicit backend/control/fencing handles from both OperationContext and StageEnv"
)]
pub(crate) async fn advance_loading_progress<B: Backend>(
    backend: &B,
    control: &OperationOptions,
    sched_lease_lock: Option<&tokio::sync::RwLock<()>>,
    bulk_authority: Option<BulkAuthority>,
    index: &IndexManifest,
    fence: WriteFence,
    delta_entries: u64,
    retry: Option<RetryPolicy>,
) -> Result<()> {
    if delta_entries == 0 && retry.is_none() {
        return Ok(());
    }
    let id = index.logical_index_id();
    let step_once = || async {
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        building_in_flight(&mut txn, index).await?;
        authorize_raw(&mut txn, id, bulk_authority).await?;
        check_writer_fence(&mut txn, index, fence).await?;
        if let Some(PersistentValue::BuildProgress(mut p)) =
            txn.get_for_update(LogicalKey::BuildProgress(id)).await?
            && p.epoch == fence.epoch
            && matches!(p.phase, BuildPhase::Preparing | BuildPhase::Loading)
            && (delta_entries > 0 || p.phase == BuildPhase::Preparing)
        {
            p.phase = BuildPhase::Loading;
            p.loaded_entries = p.loaded_entries.saturating_add(delta_entries);
            if p.loaded_entries > p.total_entries {
                p.total_entries = p.loaded_entries;
            }
            txn.put(
                LogicalKey::BuildProgress(id),
                PersistentValue::BuildProgress(p),
            )
            .await?;
            return Ok::<_, Error>((Some(txn), ()));
        }
        txn.rollback().await;
        Ok((None, ()))
    };

    if let Some(policy) = retry {
        retry_raw_write(
            control,
            sched_lease_lock,
            Operation::RunBulkBuild,
            policy,
            || async { commit_raw_prepared(step_once().await).await },
        )
        .await
    } else {
        let _guard = match sched_lease_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
        if let Ok((Some(txn), ())) = step_once().await {
            let _ = txn.commit().await;
        }
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "shared loaded-sealing tail computes split keys, commits BuildProgress, and populates report counts"
)]
pub(crate) async fn finalize_and_seal_loaded<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    fence: WriteFence,
    plan: &ShardPlan,
    hard_limits: HardLimits,
    total_records: u64,
    total_trees: u64,
    total_partitions: u64,
    total_entries: u64,
    expected_kv_lthash: LtHash,
    retry: RetryPolicy,
    report: &mut BulkBuildReport,
) -> Result<()> {
    let split_keys =
        build_validation_split_keys(index.logical_index_id(), plan, total_entries, hard_limits)?;
    seal_loaded(
        context,
        index,
        fence,
        total_records,
        total_trees,
        total_partitions,
        total_entries,
        expected_kv_lthash,
        split_keys,
        retry,
    )
    .await?;
    report.records = total_records;
    report.trees = total_trees;
    report.partitions = total_partitions;
    report.entries = total_entries;
    Ok(())
}

/// Builds up to `MAX_SPLIT_KEYS` (`255`) balanced validation split keys strictly inside
/// `serving_data_range(index_id)` (`[0x01..0x05)`), bounded so `BuildProgress` always fits
/// well inside `hard_limits.max_value_bytes`.
pub(crate) fn build_validation_split_keys(
    index_id: LogicalIndexId,
    plan: &ShardPlan,
    total_entries: u64,
    hard_limits: HardLimits,
) -> Result<Vec<Bytes>> {
    if total_entries < 4_096 {
        return Ok(Vec::new());
    }
    let target_splits = (total_entries / 32_768).clamp(2, MAX_SPLIT_KEYS as u64) as usize;
    let rg_budget = (target_splits * 3 / 5).max(1);
    let part_budget = target_splits.saturating_sub(rg_budget + 1).max(1);

    let mut split_keys = Vec::with_capacity(target_splits + 8);
    if !plan.id_bounds.is_empty() {
        let target_rg = rg_budget.min(plan.id_bounds.len());
        for i in 0..target_rg {
            let pos = (i * plan.id_bounds.len()) / target_rg;
            split_keys.push(Bytes::from(keys::record_key(
                index_id,
                &plan.id_bounds[pos],
            )?));
        }
    }
    split_keys.push(Bytes::from(
        keys::tree_manifest_range(index_id).start().to_vec(),
    ));

    let total_heavy_shards: u64 = plan
        .heavy_trees
        .iter()
        .map(|h| u64::from(h.shard_count))
        .sum();
    if let Some(non_zero_heavy) = std::num::NonZeroU64::new(total_heavy_shards) {
        for heavy in &plan.heavy_trees {
            split_keys.push(Bytes::from(keys::header_key(
                index_id,
                &heavy.tree_key,
                PartitionKey::new(1)?,
            )));
            let tree_splits =
                ((u64::from(heavy.shard_count) * part_budget as u64) / non_zero_heavy) as usize;
            let approx_leaves = u64::from(heavy.shard_count).saturating_mul(64);
            if tree_splits > 0 && approx_leaves >= 4 {
                for s in 1..=tree_splits {
                    let pk_num = ((approx_leaves * s as u64) / (tree_splits as u64 + 1)).max(2);
                    split_keys.push(Bytes::from(keys::header_key(
                        index_id,
                        &heavy.tree_key,
                        PartitionKey::new(pk_num)?,
                    )));
                }
            }
        }
    }

    let serving = keys::serving_data_range(index_id);
    split_keys.retain(|k| serving.start() < k.as_ref() && k.as_ref() < serving.end());
    split_keys.sort_unstable();
    split_keys.dedup();
    if split_keys.len() > MAX_SPLIT_KEYS {
        let step = split_keys.len() as f64 / MAX_SPLIT_KEYS as f64;
        split_keys = (0..MAX_SPLIT_KEYS)
            .map(|i| split_keys[((i as f64 * step) as usize).min(split_keys.len() - 1)].clone())
            .collect();
        split_keys.dedup();
    }

    let max_split_bytes = hard_limits
        .max_value_bytes
        .saturating_sub(1024)
        .min(64 * 1024);
    while split_keys.iter().map(|k| 4 + k.len()).sum::<usize>() > max_split_bytes
        && split_keys.len() > 1
    {
        let next_cap = (split_keys.len() / 2).max(1);
        let step = split_keys.len() as f64 / next_cap as f64;
        split_keys = (0..next_cap)
            .map(|i| split_keys[((i as f64 * step) as usize).min(split_keys.len() - 1)].clone())
            .collect();
        split_keys.dedup();
    }
    if split_keys.iter().map(|k| 4 + k.len()).sum::<usize>() > max_split_bytes {
        split_keys.clear();
    }
    Ok(split_keys)
}

pub(crate) fn validate_split_keys_in_index(id: LogicalIndexId, split_keys: &[Bytes]) -> Result<()> {
    let serving = keys::serving_data_range(id);
    if split_keys
        .iter()
        .any(|k| k.as_ref() <= serving.start() || k.as_ref() >= serving.end())
        || split_keys.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(corrupt());
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum RangeClearGuard {
    StaleServing(WriteFence),
    PublishedTransient,
}

async fn check_range_clear_guard<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: &IndexManifest,
    bulk_authority: Option<BulkAuthority>,
    guard: RangeClearGuard,
) -> Result<bool> {
    let id = index.logical_index_id();
    match guard {
        RangeClearGuard::StaleServing(fence) => {
            check_writer_fence(txn, index, fence).await?;
            authorize_raw(txn, id, bulk_authority).await?;
            Ok(true)
        }
        RangeClearGuard::PublishedTransient => {
            match txn.get_for_update(LogicalKey::Manifest(id)).await? {
                None => Ok(false),
                Some(PersistentValue::IndexManifest(m))
                    if m.logical_index_id() == id && m.lifecycle() == IndexLifecycle::Active =>
                {
                    Ok(true)
                }
                Some(PersistentValue::IndexManifest(m))
                    if m.logical_index_id() == id && m.lifecycle() == IndexLifecycle::Dropping =>
                {
                    Ok(false)
                }
                Some(PersistentValue::IndexManifest(m))
                    if m.logical_index_id() == id && m.lifecycle() == IndexLifecycle::Building =>
                {
                    Err(Error::invalid_argument())
                }
                _ => Err(corrupt()),
            }
        }
    }
}

/// Clears a logical key range via `transactional_clear_range` when supported or bounded
/// point-delete batches otherwise, sharing retry and pagination logic between epoch recovery
/// and post-publication `BuildUnit` reclamation.
async fn clear_index_range<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    range: LogicalRange,
    guard: RangeClearGuard,
    op: Operation,
    budget: AdmissionBudget,
    retry: RetryPolicy,
) -> Result<()> {
    let sched_lock = context.bulk_lease_lock.clone();
    let bulk_authority = context.bulk_authority;
    if context.backend().capabilities().transactional_clear_range {
        let mut attempts = 0;
        loop {
            let step = {
                context.checkpoint()?;
                let _sched_guard = lock_sched_read(&sched_lock).await;
                let backend = context.backend();
                let prepared = async {
                    let mut txn = WriteLogicalTxn::for_index(
                        backend.begin_write().await?,
                        index,
                        backend.hard_limits(),
                        backend.admission_budget(),
                    );
                    if !check_range_clear_guard(&mut txn, index, bulk_authority, guard).await? {
                        txn.rollback().await;
                        return Ok::<_, Error>((None, ()));
                    }
                    txn.clear_range(&range).await?;
                    Ok((Some(txn), ()))
                }
                .await;
                context.commit_prepared(prepared).await
            };
            if step.resolve(op, retry, &mut attempts).await?.is_some() {
                return Ok(());
            }
        }
    }

    let max_del_muts = budget.max_mutations.max(1);
    let max_del_bytes = budget.max_mutation_bytes.max(1);
    let mut cursor = None;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let page = retry_read(&context.options, op, retry, || {
            let backend = Arc::clone(&backend);
            let cursor_ref = cursor.clone();
            let range_ref = &range;
            async move {
                let mut rtxn = ReadLogicalTxn::for_index(backend.begin_read().await?, index);
                rtxn.scan(
                    range_ref,
                    cursor_ref.as_ref(),
                    ScanLimits {
                        item_limit: max_del_muts,
                        byte_limit: max_del_bytes,
                    },
                )
                .await
            }
        })
        .await?;
        if page.items().is_empty() {
            return Ok(());
        }
        let next_cursor = page.next_cursor().cloned();
        let del_batches = split_delete_keys(
            page.into_items().into_iter().map(|it| it.key().clone()),
            budget,
        )?;
        for batch_keys in del_batches {
            let mut attempts = 0;
            loop {
                let step = {
                    context.checkpoint()?;
                    let _sched_guard = lock_sched_read(&sched_lock).await;
                    let backend = context.backend();
                    let prepared = async {
                        let mut txn = WriteLogicalTxn::for_index(
                            backend.begin_write().await?,
                            index,
                            backend.hard_limits(),
                            budget,
                        );
                        if !check_range_clear_guard(&mut txn, index, bulk_authority, guard).await? {
                            txn.rollback().await;
                            return Ok::<_, Error>((None, false));
                        }
                        let mut batch = txn.mutations();
                        for lk in &batch_keys {
                            batch.delete(lk.clone())?;
                        }
                        txn.apply(batch).await?;
                        Ok((Some(txn), true))
                    }
                    .await;
                    context.commit_prepared(prepared).await
                };
                if let Some(proceed) = step.resolve(op, retry, &mut attempts).await? {
                    if !proceed {
                        return Ok(());
                    }
                    break;
                }
            }
        }
        match next_cursor {
            Some(nc) => cursor = Some(nc),
            None => return Ok(()),
        }
    }
}

/// Clears any serving KV keys (`0x01..0x05`) left by an earlier aborted epoch before rewriting.
pub(crate) async fn clear_stale_serving_data<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    fence: WriteFence,
    budget: AdmissionBudget,
    retry: RetryPolicy,
) -> Result<()> {
    clear_index_range(
        context,
        index,
        LogicalRange::serving_data(index),
        RangeClearGuard::StaleServing(fence),
        Operation::RunBulkBuild,
        budget,
        retry,
    )
    .await
}

fn split_delete_keys(
    keys_iter: impl Iterator<Item = LogicalKey>,
    budget: AdmissionBudget,
) -> Result<Vec<Vec<LogicalKey>>> {
    let max_muts = budget.max_mutations.max(1);
    let max_bytes = budget.max_mutation_bytes.max(1);
    let mut batches = Vec::new();
    let mut cur = Vec::new();
    let mut cur_bytes = 0_usize;
    for lk in keys_iter {
        let raw_len = keys::encode_key(&lk)?.len();
        let cost = raw_len
            .checked_add(budget.mutation_key_overhead_bytes)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        if cost > max_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        if !cur.is_empty() && (cur.len() >= max_muts || cur_bytes + cost > max_bytes) {
            batches.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
        cur_bytes += cost;
        cur.push(lk);
    }
    if !cur.is_empty() {
        batches.push(cur);
    }
    Ok(batches)
}

pub(super) fn split_into_chunks(
    kvs: &[(Bytes, Bytes)],
    budget: AdmissionBudget,
    has_unit_fence: bool,
) -> Result<Vec<Vec<(Bytes, Bytes)>>> {
    let reserved_muts = usize::from(has_unit_fence);
    let reserved_bytes = if has_unit_fence {
        BUILD_UNIT_MUTATION_BYTES
            .checked_add(budget.mutation_key_overhead_bytes)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?
    } else {
        0
    };
    let max_muts = budget
        .max_mutations
        .checked_sub(reserved_muts)
        .filter(|&m| m >= 1)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let max_bytes = budget
        .max_mutation_bytes
        .checked_sub(reserved_bytes)
        .filter(|&b| b >= 1)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let base_affected = 5_usize
        .saturating_mul(32_usize.saturating_add(budget.mutation_key_overhead_bytes))
        .saturating_mul(3)
        .saturating_add(512);

    let mut chunks = Vec::new();
    let mut cur = Vec::new();
    let mut cur_bytes = 0_usize;
    let mut cur_affected = base_affected;

    for (k, v) in kvs {
        let item_bytes = k
            .len()
            .checked_add(v.len())
            .and_then(|n| n.checked_add(budget.mutation_key_overhead_bytes))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let item_affected = k
            .len()
            .checked_add(budget.mutation_key_overhead_bytes)
            .and_then(|kb| kb.checked_mul(3))
            .and_then(|n| n.checked_add(v.len()))
            .and_then(|n| n.checked_add(64))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;

        if item_bytes > max_bytes
            || base_affected
                .checked_add(item_affected)
                .is_none_or(|tot| tot > MAX_TXN_AFFECTED_BYTES)
        {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        if !cur.is_empty()
            && (cur.len() >= max_muts
                || cur_bytes + item_bytes > max_bytes
                || cur_affected + item_affected > MAX_TXN_AFFECTED_BYTES)
        {
            chunks.push(std::mem::take(&mut cur));
            cur_bytes = 0;
            cur_affected = base_affected;
        }
        cur_bytes += item_bytes;
        cur_affected += item_affected;
        cur.push((k.clone(), v.clone()));
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    Ok(chunks)
}

#[expect(
    clippy::too_many_arguments,
    reason = "detached async writer task owns cloned backend, control, and explicit fencing tokens"
)]
pub(super) async fn commit_one_chunk<B: Backend>(
    backend: Arc<B>,
    control: &OperationOptions,
    bulk_authority: Option<BulkAuthority>,
    sched_lease_lock: Option<&tokio::sync::RwLock<()>>,
    index: &IndexManifest,
    fence: WriteFence,
    unit_fence: Option<&UnitLease>,
    chunk: Vec<(Bytes, Bytes)>,
    budget: AdmissionBudget,
    retry: RetryPolicy,
) -> Result<()> {
    let id = index.logical_index_id();
    let chunk_ref = &chunk;
    retry_raw_write(
        control,
        sched_lease_lock,
        Operation::RunBulkBuild,
        retry,
        || async {
            let _lease_guard = match unit_fence {
                Some(lease) => Some(lease.lock.lock().await),
                None => None,
            };
            let prepared = async {
                let mut txn = WriteLogicalTxn::for_index(
                    backend.begin_write().await?,
                    index,
                    backend.hard_limits(),
                    budget,
                );

                let mut warm = Vec::with_capacity(5);
                warm.push(LogicalKey::Manifest(id));
                warm.push(LogicalKey::BuildWorkspace(id));
                warm.push(LogicalKey::BuildSchedule(id));
                if let Some(auth) = bulk_authority {
                    warm.push(auth.slot_key(id));
                }
                if let Some(lease) = unit_fence {
                    warm.push(LogicalKey::BuildUnit {
                        index: id,
                        stage: lease.stage,
                        unit: lease.unit,
                    });
                }
                txn.warm_for_update(warm).await?;

                check_writer_fence(&mut txn, index, fence).await?;
                authorize_raw(&mut txn, id, bulk_authority).await?;
                if let Some(lease) = unit_fence {
                    let ukey = LogicalKey::BuildUnit {
                        index: id,
                        stage: lease.stage,
                        unit: lease.unit,
                    };
                    match txn.get_for_update(ukey.clone()).await? {
                        Some(PersistentValue::BuildUnit(mut u))
                            if u.epoch == fence.epoch && u.owner == lease.owner && !u.complete =>
                        {
                            u.expires_ms = u
                                .expires_ms
                                .max(super::bulk_scheduler::expiry(lease.duration)?);
                            txn.put(ukey, PersistentValue::BuildUnit(u)).await?;
                        }
                        _ => return Err(Error::new(ErrorKind::BulkBuildSuperseded)),
                    }
                }
                let mut batch = txn.mutations();
                for (k, v) in chunk_ref {
                    batch.put_encoded(k.clone(), v.clone())?;
                }
                txn.apply(batch).await?;
                Ok((Some(txn), ()))
            }
            .await;
            commit_raw_prepared(prepared).await
        },
    )
    .await
}

/// Conflict-protects `BuildWorkspace(id)` without mutating it, verifying that `fence` remains the
/// active writer authority for `index`.
pub(crate) async fn check_writer_fence<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: &IndexManifest,
    fence: WriteFence,
) -> Result<()> {
    building_in_flight(txn, index).await?;
    let id = index.logical_index_id();
    match txn.get_for_update(LogicalKey::BuildWorkspace(id)).await? {
        Some(PersistentValue::BuildWorkspace(w)) => {
            if let Some(kind) = w.failure {
                return Err(Error::new(kind));
            }
            if w.epoch == fence.epoch && w.writer_token == fence.owner && fence.owner != [0; 32] {
                Ok(())
            } else {
                Err(Error::new(ErrorKind::BulkBuildSuperseded))
            }
        }
        None => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
        _ => Err(corrupt()),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "sealing Loaded commits exact counts, multiset digest, and validation split keys atomically"
)]
pub(crate) async fn seal_loaded<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    fence: WriteFence,
    total_records: u64,
    trees: u64,
    partitions: u64,
    total_entries: u64,
    expected_kv_lthash: LtHash,
    validation_split_keys: Vec<Bytes>,
    retry: RetryPolicy,
) -> Result<()> {
    let id = index.logical_index_id();
    validate_split_keys_in_index(id, &validation_split_keys)?;
    let sched_lock = context.bulk_lease_lock.clone();
    let mut attempts = 0;
    loop {
        let step = {
            context.checkpoint()?;
            let _sched_guard = lock_sched_read(&sched_lock).await;
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                check_writer_fence(&mut txn, index, fence).await?;
                super::bulk_scheduler::authorize(context, &mut txn, id).await?;

                let cur = match txn.get_for_update(LogicalKey::BuildProgress(id)).await? {
                    Some(PersistentValue::BuildProgress(p)) => p,
                    _ => return Err(corrupt()),
                };
                if cur.phase == BuildPhase::Loaded
                    && cur.epoch == fence.epoch
                    && cur.total_entries == total_entries
                    && cur.expected_kv_lthash == expected_kv_lthash
                {
                    txn.rollback().await;
                    return Ok((None, ()));
                }
                if cur.phase == BuildPhase::Validating || cur.epoch != fence.epoch {
                    return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                }

                let next = BuildProgress {
                    phase: BuildPhase::Loaded,
                    epoch: fence.epoch,
                    loaded_entries: total_entries,
                    total_entries,
                    total_records,
                    trees,
                    partitions,
                    expected_kv_lthash,
                    validation_split_keys: validation_split_keys.clone(),
                };
                txn.put(
                    LogicalKey::BuildProgress(id),
                    PersistentValue::BuildProgress(next),
                )
                .await?;
                Ok((Some(txn), ()))
            }
            .await;
            context.commit_prepared(prepared).await
        };
        if step
            .resolve(Operation::RunBulkBuild, retry, &mut attempts)
            .await?
            .is_some()
        {
            return Ok(());
        }
    }
}

/// Persists terminal failures for the current workspace epoch; transient errors,
/// cancellation, contention, and superseded ownership leave the build resumable.
pub(crate) async fn record_failure<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    before: &BuildWorkspace,
    worker_fence: Option<WriteFence>,
    kind: ErrorKind,
    retry: RetryPolicy,
) -> Result<()> {
    if !matches!(
        kind,
        ErrorKind::Corruption
            | ErrorKind::InvalidArgument
            | ErrorKind::LimitExceeded
            | ErrorKind::Other
            | ErrorKind::UnsupportedFormat
            | ErrorKind::IdExhausted
            | ErrorKind::RecordAlreadyExists
            | ErrorKind::TransactionTooLarge
    ) {
        return Ok(());
    }
    let sched_lock = context.bulk_lease_lock.clone();
    let mut attempts = 0;
    let id = index.logical_index_id();
    loop {
        if context.checkpoint().is_err() {
            return Ok(());
        }
        let step = {
            let _sched_guard = lock_sched_read(&sched_lock).await;
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                match building(&mut txn, index).await {
                    Ok(()) => {}
                    Err(e)
                        if matches!(
                            e.kind(),
                            ErrorKind::IndexDropping
                                | ErrorKind::IndexNotFound
                                | ErrorKind::InvalidArgument
                        ) =>
                    {
                        txn.rollback().await;
                        return Ok((None, ()));
                    }
                    Err(e) => return Err(e),
                }
                if super::bulk_scheduler::authorize(context, &mut txn, id)
                    .await
                    .is_err()
                {
                    txn.rollback().await;
                    return Ok((None, ()));
                }
                if let Some(fence) = worker_fence {
                    if check_writer_fence(&mut txn, index, fence).await.is_err() {
                        txn.rollback().await;
                        return Ok((None, ()));
                    }
                    if matches!(
                        txn.get_for_update(LogicalKey::BuildProgress(id)).await?,
                        Some(PersistentValue::BuildProgress(p)) if p.phase == BuildPhase::Validating
                    ) {
                        txn.rollback().await;
                        return Ok((None, ()));
                    }
                }
                let key = LogicalKey::BuildWorkspace(id);
                let mut current = match txn.get_for_update(key.clone()).await? {
                    Some(PersistentValue::BuildWorkspace(w)) => w,
                    _ => {
                        txn.rollback().await;
                        return Ok((None, ()));
                    }
                };
                if current.failure.is_some()
                    || current.epoch != before.epoch
                    || current.token != before.token
                {
                    txn.rollback().await;
                    return Ok((None, ()));
                }
                current.failure = Some(kind);
                txn.put(key, PersistentValue::BuildWorkspace(current))
                    .await?;
                Ok((Some(txn), ()))
            }
            .await;
            context.commit_prepared(prepared).await
        };
        match step {
            CommitStep::Done(()) => return Ok(()),
            CommitStep::PrepareAbort => {
                if retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await
                    .is_err()
                {
                    return Ok(());
                }
            }
            CommitStep::CommitFailed(e)
                if matches!(e.kind(), ErrorKind::Cancelled | ErrorKind::DeadlineExceeded) =>
            {
                return Ok(());
            }
            CommitStep::CommitFailed(e) => {
                retry
                    .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                    .await?;
            }
            CommitStep::Fatal(e) => return Err(e),
        }
    }
}

/// Reclaims generated child directories under exclusive `.ktann-build-lock` and removes
/// transient post-publication and orphaned scheduler metadata once no peer holds an active lease.
pub(crate) async fn cleanup<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    retry: RetryPolicy,
) -> Result<()> {
    let backend = context.backend();
    let (active_manifest, state_opt) =
        retry_read(&context.options, Operation::CleanupBulkBuild, retry, || {
            let backend = Arc::clone(&backend);
            async move {
                let mut snapshot = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
                let active_manifest = match snapshot.get(LogicalKey::Manifest(id)).await? {
                    None => None,
                    Some(PersistentValue::IndexManifest(m)) if m.logical_index_id() == id => {
                        if m.lifecycle() == IndexLifecycle::Building {
                            return Err(Error::invalid_argument());
                        }
                        (m.lifecycle() == IndexLifecycle::Active).then_some(m)
                    }
                    _ => return Err(corrupt()),
                };
                let state = match snapshot.get(LogicalKey::BuildWorkspace(id)).await? {
                    None => None,
                    Some(PersistentValue::BuildWorkspace(w)) => Some(w),
                    _ => return Err(corrupt()),
                };
                Ok((active_manifest, state))
            }
        })
        .await?;

    if let Some(active) = &active_manifest {
        let budget = context.backend().admission_budget();
        clear_index_range(
            context,
            active,
            LogicalRange::transient_build_state(active),
            RangeClearGuard::PublishedTransient,
            Operation::CleanupBulkBuild,
            budget,
            retry,
        )
        .await?;
    }
    let Some(state) = state_opt else {
        retire_orphaned_schedule(context, id, retry).await?;
        return Ok(());
    };

    let lock = {
        let mut lock_attempts = 0;
        loop {
            let root_c = state.options.workspace.clone();
            match blocking(context, move || lock_root(&root_c, true)).await {
                Ok(file) => break Arc::new(file),
                Err(e)
                    if e.kind() == ErrorKind::BulkBuildBusy
                        && context.bulk_authority.is_some()
                        && !retry.would_exhaust(lock_attempts) =>
                {
                    retry.wait(lock_attempts).await;
                    lock_attempts += 1;
                }
                Err(e) => return Err(e),
            }
        }
    };

    let sched_lock = context.bulk_lease_lock.clone();
    let mut attempts = 0;
    loop {
        let step = {
            context.checkpoint()?;
            let _sched_guard = lock_sched_read(&sched_lock).await;
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                if let Some(value) = txn.get_for_update(LogicalKey::Manifest(id)).await? {
                    match value {
                        PersistentValue::IndexManifest(m)
                            if m.logical_index_id() == id
                                && m.lifecycle() != IndexLifecycle::Building => {}
                        _ => return Err(Error::invalid_argument()),
                    }
                }
                ensure_no_peer_scheduler_leases(
                    context,
                    &mut txn,
                    id,
                    state.options.distributed_workers,
                )
                .await?;
                let key = LogicalKey::BuildWorkspace(id);
                match txn.get_for_update(key.clone()).await? {
                    None => {
                        txn.rollback().await;
                        return Ok((None, ()));
                    }
                    Some(PersistentValue::BuildWorkspace(w)) if w == state => {}
                    _ => return Err(corrupt()),
                }
                let path = state.directory();
                let lock = lock.clone();
                blocking(context, move || {
                    let _lock = lock;
                    match fs::remove_dir_all(&path) {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(io(e)),
                    };
                    File::open(path.parent().expect("root"))
                        .and_then(|f| f.sync_all())
                        .map_err(io)
                })
                .await?;
                txn.delete(key).await?;
                Ok((Some(txn), ()))
            }
            .await;
            context.commit_prepared(prepared).await
        };
        if step
            .resolve(Operation::CleanupBulkBuild, retry, &mut attempts)
            .await?
            .is_some()
        {
            break;
        }
    }

    drop(lock);
    retire_orphaned_schedule(context, id, retry).await
}

pub(crate) fn io(e: std::io::Error) -> Error {
    Error::with_source(ErrorKind::Other, e)
}

pub(crate) fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// Visits a bounded page of namespace cleanup records across both `BuildWorkspace` and
/// orphaned `BuildSchedule` entries.
pub(crate) async fn cleanup_pending<B: Backend>(
    context: &mut OperationContext<B>,
    maximum: usize,
    after: Option<LogicalIndexId>,
    retry: RetryPolicy,
) -> Result<crate::api::BulkCleanupPage> {
    if maximum == 0 || maximum > 1000 {
        return Err(Error::invalid_argument());
    }
    let backend = context.backend();
    let (ids, has_more) = {
        let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
        let ws_range = LogicalRange::build_workspaces(after);
        let sched_range = LogicalRange::build_schedules(after);
        let ws_page = rtxn
            .scan(
                &ws_range,
                None,
                ScanLimits {
                    item_limit: maximum,
                    byte_limit: 1024 * 1024,
                },
            )
            .await?;
        let sched_page = rtxn
            .scan(
                &sched_range,
                None,
                ScanLimits {
                    item_limit: maximum,
                    byte_limit: 1024 * 1024,
                },
            )
            .await?;

        let mut merged = BTreeSet::new();
        for item in ws_page.items() {
            match *item.key() {
                LogicalKey::BuildWorkspace(id) => {
                    merged.insert(id);
                }
                _ => return Err(corrupt()),
            }
        }
        for item in sched_page.items() {
            match *item.key() {
                LogicalKey::BuildSchedule(id) => {
                    merged.insert(id);
                }
                _ => return Err(corrupt()),
            }
        }
        let total_candidates = merged.len();
        let ids: Vec<LogicalIndexId> = merged.into_iter().take(maximum).collect();
        let has_more = total_candidates > maximum
            || ws_page.next_cursor().is_some()
            || sched_page.next_cursor().is_some();
        (ids, has_more)
    };

    let next = if has_more { ids.last().copied() } else { None };
    let mut result = crate::api::BulkCleanupPage {
        reclaimed: 0,
        pending: 0,
        next,
    };
    for id in ids {
        context.checkpoint()?;
        let mut read = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
        let building = match read.get(LogicalKey::Manifest(id)).await? {
            Some(PersistentValue::IndexManifest(m)) => m.lifecycle() == IndexLifecycle::Building,
            None => false,
            _ => return Err(corrupt()),
        };
        drop(read);
        if building {
            result.pending += 1;
            continue;
        }
        match cleanup(context, id, retry).await {
            Ok(()) => result.reclaimed += 1,
            Err(e) if e.kind() == ErrorKind::BulkBuildBusy => result.pending += 1,
            Err(e) => return Err(e),
        }
    }
    Ok(result)
}
