//! Sharded Bulk Build execution: zero-disk in-memory fast path, collaborative disk BuildUnits,
//! and namespace-owned workspace reclamation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio_util::sync::{CancellationToken, DropGuard};

use crate::api::{
    BulkBuildReport, BulkLoadOptions, BulkWorkerOptions, Error, ErrorKind, LogicalIndexId,
    OperationOptions, PartitionKey, Result,
};
use crate::bulk::sharded::{
    EncodedLeafShard, EncodedLocationBatch, IdFrameEntry, ShardRows, ShardState,
    assign_leaf_partition_keys, build_and_encode_upper_tree, build_shard_leaves,
    compute_shard_remap, encode_leaf_shard_opts, encode_record_group_bucket,
    ingest_segment_into_slab, quantize_shards_and_extract_locations, refine_shards_parallel,
    verify_id_bucket,
};
use crate::bulk::{InputSnapshot, LtHash, ShardPlan};
use crate::observe::labels::Operation;
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{AdmissionBudget, Backend, HardLimits, ScanLimits, WriteTxn};
use crate::storage::keys::{self, LogicalKey, TreeKey};
use crate::storage::values::{
    BuildDescriptor, BuildPhase, BuildProgress, BuildUnit, BuildWorkspace, IndexLifecycle,
    IndexManifest, MAX_SPLIT_KEYS, PartitionCentroid, PersistentValue,
};
use crate::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

use super::{
    OperationContext,
    bulk_scheduler::{
        BulkAuthority, acquire_unscheduled_peer_slot, release_unscheduled_peer_slot,
        renew_unscheduled_peer_slot, retire_orphaned_schedule,
    },
    lifecycle::RetryPolicy,
};

#[path = "bulk_disk.rs"]
pub(crate) mod bulk_disk;

/// Number of independent writer-fence lanes (`BuildUnit { stage: 0, unit: 0..16 }`).
pub(crate) const STAGE_FENCE: u8 = 0;
pub(crate) const FENCE_LANES: u32 = 16;
const MAX_TXN_AFFECTED_BYTES: usize = 8 * 1024 * 1024;
const BUILD_UNIT_MUTATION_BYTES: usize = 296;

type UpperTreeInput = (TreeKey, u64, Vec<(PartitionKey, PartitionCentroid)>);

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

/// In-flight lifecycle check for worker lanes, validation slices, and sealing (`Active` means
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
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let res = async {
            let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
            rtxn.get(LogicalKey::BuildProgress(id)).await
        }
        .await;
        match res {
            Ok(Some(PersistentValue::BuildProgress(p))) => {
                report.records = p.total_records;
                report.trees = p.trees;
                report.partitions = p.partitions;
                report.entries = p.total_entries;
                return Ok(());
            }
            Ok(_) => return Ok(()),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    }
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

async fn run_peer_slot_renewals<B: Backend>(
    backend: Arc<B>,
    lease_lock: Arc<tokio::sync::RwLock<()>>,
    id: LogicalIndexId,
    slot: u32,
    owner: [u8; 32],
    lease: Duration,
    mut confirmed: u64,
) -> Error {
    let tick = (lease / 3).max(Duration::from_millis(5));
    let mut timer = tokio::time::interval(tick);
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    timer.tick().await;
    loop {
        timer.tick().await;
        if let Err(e) = renew_unscheduled_peer_slot(
            backend.as_ref(),
            &lease_lock,
            id,
            slot,
            owner,
            lease,
            &mut confirmed,
        )
        .await
        {
            return e;
        }
    }
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
            Ok(Some((slot, owner, initial_exp, lease_lock))) => {
                let backend = context.backend();
                let lease = options.unit_lease;
                let outcome = tokio::select! {
                    res = complete_attempt(context, &index, &descriptor, &options, retry, &mut report) => res,
                    err = run_peer_slot_renewals(backend, lease_lock, id, slot, owner, lease, initial_exp) => Err(err),
                };
                release_unscheduled_peer_slot(context, id, slot, owner, retry).await?;
                outcome
            }
            Ok(None) => {
                complete_attempt(context, &index, &descriptor, &options, retry, &mut report).await
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
    if let Some((slot, owner, initial_exp, lease_lock)) =
        maybe_acquire_peer_slot(context, &index, &options, retry).await?
    {
        let backend = context.backend();
        let lease = options.unit_lease;
        let outcome = tokio::select! {
            res = run_normalized(context, index, descriptor, options, retry) => res,
            err = run_peer_slot_renewals(backend, lease_lock, id, slot, owner, lease, initial_exp) => Err(err),
        };
        release_unscheduled_peer_slot(context, id, slot, owner, retry).await?;
        outcome
    } else {
        run_normalized(context, index, descriptor, options, retry).await
    }
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

    // Ensure the workspace's owned child directory exists so cleanup owns a concrete directory.
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

    let use_in_memory = state.options.distributed_workers == 1
        && fits_in_memory(&index, &descriptor, &state.options);

    let result = if use_in_memory {
        run_in_memory(context, &index, &descriptor, &state, fence, lock, retry).await
    } else {
        bulk_disk::run_disk_units(context, &index, &descriptor, &state, fence, lock, retry).await
    };
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

async fn stage_extra_fence_lanes<B: Backend>(
    context: &OperationContext<B>,
    index: &IndexManifest,
    fence: WriteFence,
    retry: RetryPolicy,
) -> Result<()> {
    let id = index.logical_index_id();
    let max_per_batch = context
        .backend()
        .admission_budget()
        .max_mutations
        .saturating_sub(2)
        .clamp(1, FENCE_LANES as usize) as u32;
    for batch_start in (1..FENCE_LANES).step_by(max_per_batch as usize) {
        let batch_end = (batch_start + max_per_batch).min(FENCE_LANES);
        let mut attempts = 0;
        loop {
            context.checkpoint()?;
            let _lease_guard = match &context.bulk_lease_lock {
                Some(l) => Some(l.read().await),
                None => None,
            };
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                building_in_flight(&mut txn, index).await?;
                super::bulk_scheduler::authorize(context, &mut txn, id).await?;
                for lane in batch_start..batch_end {
                    txn.put(
                        LogicalKey::BuildUnit {
                            index: id,
                            stage: STAGE_FENCE,
                            unit: lane,
                        },
                        PersistentValue::BuildUnit(BuildUnit::fence_lane(fence.epoch, fence.owner)),
                    )
                    .await?;
                }
                Ok::<_, Error>(txn)
            }
            .await;
            let txn = match prepared {
                Ok(txn) => txn,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    drop(_lease_guard);
                    retry
                        .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                        .await?;
                    continue;
                }
                Err(e) => return Err(e),
            };
            match txn.commit().await {
                Ok(()) => break,
                Err(e) => {
                    drop(_lease_guard);
                    retry
                        .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                        .await?;
                }
            }
        }
    }
    Ok(())
}

/// Claims or joins the workspace and installs the 16 writer-fence lanes.
async fn claim<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    options: BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<(BuildWorkspace, WriteFence, Option<BuildProgress>)> {
    let mut ws_token = [0_u8; 32];
    getrandom::fill(&mut ws_token).map_err(|_| Error::new(ErrorKind::Other))?;
    let mut fence_owner = [0_u8; 32];
    getrandom::fill(&mut fence_owner).map_err(|_| Error::new(ErrorKind::Other))?;

    let _ = effective_budget(context.backend().admission_budget(), options.load)?;
    let id = index.logical_index_id();
    let inline_all_lanes =
        context.backend().admission_budget().max_mutations >= (FENCE_LANES as usize + 3);

    let mut attempts = 0;
    let (state, fence) = loop {
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
                                epoch: p.epoch,
                                failure: None,
                            },
                            WriteFence {
                                epoch: p.epoch,
                                owner: ws_token,
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
                        epoch: 1,
                        failure: None,
                    };
                    let owner = if options.distributed_workers > 1 {
                        ws_token
                    } else {
                        fence_owner
                    };
                    (state, WriteFence { epoch: 1, owner })
                }
                Some(PersistentValue::BuildWorkspace(mut w)) => {
                    if w.options != options || w.hard_limits != backend.hard_limits() {
                        return Err(Error::invalid_argument());
                    }
                    if let Some(kind) = w.failure {
                        return Err(Error::new(kind));
                    }
                    if existing_progress
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
                                owner: w.token,
                            },
                            Some(loaded_prog),
                        ));
                    }
                    if options.distributed_workers > 1 {
                        let existing_fence = WriteFence {
                            epoch: w.epoch,
                            owner: w.token,
                        };
                        // Joining collaborative peers observe an already-initialized epoch without
                        // rewriting `BuildWorkspace` or `STAGE_FENCE` lanes (zero writer conflicts!).
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
                        (
                            w.clone(),
                            WriteFence {
                                epoch: w.epoch,
                                owner: fence_owner,
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
            let lanes_in_txn = if inline_all_lanes { FENCE_LANES } else { 1 };
            for lane in 0..lanes_in_txn {
                txn.put(
                    LogicalKey::BuildUnit {
                        index: id,
                        stage: STAGE_FENCE,
                        unit: lane,
                    },
                    PersistentValue::BuildUnit(BuildUnit::fence_lane(fence.epoch, fence.owner)),
                )
                .await?;
            }
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
            Ok(()) => break (state, fence),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    };

    if !inline_all_lanes {
        stage_extra_fence_lanes(context, index, fence, retry).await?;
    }
    Ok((state, fence, None))
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

pub(crate) async fn bump_loading_progress<B: Backend>(
    context: &OperationContext<B>,
    index: &IndexManifest,
    fence: WriteFence,
    delta_entries: u64,
) -> Result<()> {
    if delta_entries == 0 {
        return Ok(());
    }
    let _lease_guard = match &context.bulk_lease_lock {
        Some(l) => Some(l.read().await),
        None => None,
    };
    let id = index.logical_index_id();
    let backend = context.backend();
    let mut txn = WriteLogicalTxn::bootstrap(
        backend.begin_write().await?,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    if building_in_flight(&mut txn, index).await.is_err()
        || super::bulk_scheduler::authorize(context, &mut txn, id)
            .await
            .is_err()
        || check_lane_fence(&mut txn, index, fence, 0).await.is_err()
    {
        txn.rollback().await;
        return Ok(());
    }
    if let Some(PersistentValue::BuildProgress(mut p)) =
        txn.get_for_update(LogicalKey::BuildProgress(id)).await?
        && p.epoch == fence.epoch
        && matches!(p.phase, BuildPhase::Preparing | BuildPhase::Loading)
    {
        p.phase = BuildPhase::Loading;
        p.loaded_entries = p.loaded_entries.saturating_add(delta_entries);
        if p.loaded_entries > p.total_entries {
            p.total_entries = p.loaded_entries;
        }
        if txn
            .put(
                LogicalKey::BuildProgress(id),
                PersistentValue::BuildProgress(p),
            )
            .await
            .is_ok()
        {
            let _ = txn.commit().await;
            return Ok(());
        }
    }
    txn.rollback().await;
    Ok(())
}

/// Zero-disk in-memory sharded execution with wave-pipelined ingestion and emission.
async fn run_in_memory<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &BuildWorkspace,
    fence: WriteFence,
    lock: Arc<File>,
    retry: RetryPolicy,
) -> Result<BulkBuildReport> {
    let mut report = BulkBuildReport {
        in_memory_mode: true,
        rotation_seed: *index.rotation_seed(),
        ..BulkBuildReport::default()
    };
    let parallelism = state.options.parallelism.max(1);
    let budget = effective_budget(context.backend().admission_budget(), state.options.load)?;
    let effective_mem = state
        .options
        .memory_bytes
        .min(descriptor.options().memory_bytes);

    // Before planning or sealing, clear any serving KVs left by an aborted earlier epoch.
    if fence.epoch > 1 {
        clear_stale_serving_data(context, index, fence, budget, retry).await?;
    }

    // 1. Open InputSnapshot and build ShardPlan.
    let plan_start = Instant::now();
    let (snapshot, plan) = {
        let desc = descriptor.clone();
        let manifest = index.clone();
        let guard = lock.clone();
        context
            .compute(move |control| {
                let _lock = guard;
                let check = || super::check_control(control);
                let snap = InputSnapshot::open(
                    desc.source(),
                    manifest.config().clone(),
                    desc.input().clone(),
                )?;
                let plan =
                    ShardPlan::build(&snap, &manifest, desc.options(), effective_mem, &check)?;
                Ok((snap, plan))
            })
            .await?
    };
    report.plan = plan_start.elapsed();

    if snapshot.manifest().items() == 0 {
        if let Some(rounds) = descriptor.refinement_rounds() {
            let mut empty_shards = Vec::new();
            refine_shards_parallel(
                index,
                &plan,
                &mut empty_shards,
                rounds,
                effective_mem,
                parallelism,
                &|| Ok(()),
            )?;
        }
        seal_loaded(
            context,
            index,
            fence,
            0,
            0,
            0,
            0,
            LtHash::ZERO,
            Vec::new(),
            retry,
        )
        .await?;
        return Ok(report);
    }

    // 2. Parallel segment ingestion directly into a single contiguous vector slab (`1.0x` exact size).
    let ingest_start = Instant::now();
    let num_buckets = plan.id_bucket_count() as usize;
    let num_shards = plan.total_shards as usize;
    let dim = index.config().dimension();
    let total_items_usize = usize::try_from(snapshot.manifest().items())
        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let slab_floats = total_items_usize
        .checked_mul(dim)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;

    let (slab, all_ids_by_bucket, mut all_rows_by_shard, total_records, input_id_lthash) = {
        let snap_c = snapshot.clone();
        let manifest_c = index.clone();
        let plan_c = plan.clone();
        let limits = state.hard_limits;
        let guard = lock.clone();

        context
            .compute(move |control| {
                let _lock = guard;
                let check = || super::check_control(control);
                let mut raw_slab = vec![0.0_f32; slab_floats];
                let segs = snap_c.segments();
                let seg_count = segs.len();
                let mut all_ids: Vec<Vec<IdFrameEntry>> = vec![Vec::new(); num_buckets];
                let mut shard_metas: Vec<Vec<crate::bulk::sharded::CompactMeta>> =
                    vec![Vec::new(); num_shards];
                let mut total_records = 0_u64;
                let mut input_id_lthash = LtHash::ZERO;

                let wave_size = parallelism.clamp(1, 16);
                let mut slab_rest: &mut [f32] = raw_slab.as_mut_slice();
                let mut base_row = 0_usize;

                for wave_start in (0..seg_count).step_by(wave_size) {
                    let wave_end = (wave_start + wave_size).min(seg_count);
                    let mut seg_slices = Vec::with_capacity(wave_end - wave_start);
                    for (seg_idx, seg_meta) in segs[wave_start..wave_end].iter().enumerate() {
                        let items = usize::try_from(seg_meta.items).map_err(|_| corrupt())?;
                        let floats = items.checked_mul(dim).ok_or_else(corrupt)?;
                        if floats > slab_rest.len() {
                            return Err(corrupt());
                        }
                        let (head, tail) = slab_rest.split_at_mut(floats);
                        slab_rest = tail;
                        seg_slices.push((wave_start + seg_idx, base_row, head));
                        base_row += items;
                    }

                    let ingested = std::thread::scope(|scope| {
                        let mut handles = Vec::with_capacity(seg_slices.len());
                        for (seg_idx, seg_base, slice) in seg_slices {
                            let s_ref = &snap_c;
                            let m_ref = &manifest_c;
                            let p_ref = &plan_c;
                            handles.push(scope.spawn(move || {
                                ingest_segment_into_slab(
                                    s_ref, seg_idx, seg_base, slice, m_ref, p_ref, limits, &check,
                                )
                            }));
                        }
                        let mut out = Vec::with_capacity(handles.len());
                        for h in handles {
                            out.push(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
                        }
                        Ok::<_, Error>(out)
                    })?;

                    for seg in ingested {
                        total_records += seg.records;
                        input_id_lthash.add_assign(seg.id_lthash);
                        for (b, ids) in seg.ids_by_bucket.into_iter().enumerate() {
                            all_ids[b].extend(ids);
                        }
                        for (s, metas) in seg.metas_by_shard.into_iter().enumerate() {
                            shard_metas[s].extend(metas);
                        }
                    }
                }
                if !slab_rest.is_empty() {
                    return Err(corrupt());
                }

                let slab = Arc::new(raw_slab);
                let rows_by_shard: Vec<ShardRows> = shard_metas
                    .into_iter()
                    .map(|rows| {
                        let mut sr = ShardRows::with_slab(dim, Arc::clone(&slab));
                        sr.rows = rows;
                        sr
                    })
                    .collect();
                Ok((slab, all_ids, rows_by_shard, total_records, input_id_lthash))
            })
            .await?
    };
    let mut total_entries = 0_u64;
    let mut expected_kv_lthash = LtHash::ZERO;
    report.ingest = ingest_start.elapsed();

    // 3. Parallel global ID dedup + shard remap + sharded leaf construction.
    let construct_start = Instant::now();
    let (mut shards, dedup_records, dedup_id_lthash, verified_id_refs) = {
        let manifest_c = index.clone();
        let plan_c = plan.clone();
        let sample_items = descriptor.options().sample_items;
        let slab_c = Arc::clone(&slab);
        drop(slab);
        context
            .compute(move |control| {
                let check = || super::check_control(control);
                let (dedup_records, dedup_lt, verified_buckets) =
                    verify_all_id_buckets_parallel(all_ids_by_bucket, parallelism)?;

                let kernel = VectorKernel::new(
                    manifest_c.config().dimension(),
                    manifest_c.config().metric(),
                    *manifest_c.rotation_seed(),
                )?;
                let counts: Vec<u64> = all_rows_by_shard
                    .iter()
                    .map(|s| s.rows.len() as u64)
                    .collect();
                let remap = compute_shard_remap(
                    &kernel,
                    &plan_c,
                    &counts,
                    u64::from(manifest_c.config().min_leaf_partition_entries()),
                    u64::from(manifest_c.config().max_leaf_partition_entries()),
                );
                for src in 0..all_rows_by_shard.len() {
                    let dst = remap[src] as usize;
                    if dst != src && !all_rows_by_shard[src].rows.is_empty() {
                        let mut moved = std::mem::replace(
                            &mut all_rows_by_shard[src],
                            ShardRows::with_slab(
                                manifest_c.config().dimension(),
                                Arc::clone(&slab_c),
                            ),
                        );
                        all_rows_by_shard[dst].rows.append(&mut moved.rows);
                    }
                }
                drop(slab_c);

                let chunk_size = all_rows_by_shard.len().div_ceil(parallelism.max(1)).max(1);
                let mut row_chunks = Vec::new();
                let mut iter = all_rows_by_shard.into_iter();
                while iter.len() > 0 {
                    row_chunks.push(iter.by_ref().take(chunk_size).collect::<Vec<_>>());
                }
                let shards = std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(row_chunks.len());
                    for chunk in row_chunks {
                        let m_ref = &manifest_c;
                        handles.push(scope.spawn(move || -> Result<Vec<ShardState>> {
                            let mut out = Vec::with_capacity(chunk.len());
                            for rows in chunk {
                                if rows.rows.is_empty() {
                                    out.push(ShardState {
                                        rows,
                                        leaves: Vec::new(),
                                    });
                                } else {
                                    out.push(build_shard_leaves(
                                        m_ref,
                                        sample_items,
                                        rows,
                                        &check,
                                    )?);
                                }
                            }
                            Ok(out)
                        }));
                    }
                    let mut built = Vec::new();
                    for h in handles {
                        built.extend(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
                    }
                    Ok::<_, Error>(built)
                })?;
                Ok((shards, dedup_records, dedup_lt, verified_buckets))
            })
            .await?
    };
    if dedup_records != total_records || dedup_id_lthash != input_id_lthash {
        return Err(corrupt());
    }
    report.construct = construct_start.elapsed();

    // 4. Optional parallel halo refinement (including `Some(0)` centroid refresh & admission).
    if let Some(rounds) = descriptor.refinement_rounds() {
        let refine_start = Instant::now();
        let manifest_c = index.clone();
        let plan_c = plan.clone();
        shards = context
            .compute(move |control| {
                let check = || super::check_control(control);
                refine_shards_parallel(
                    &manifest_c,
                    &plan_c,
                    &mut shards,
                    rounds,
                    effective_mem,
                    parallelism,
                    &check,
                )?;
                Ok(shards)
            })
            .await?;
        report.refine = refine_start.elapsed();
    }

    // 5. Global ascending key-order emission.
    let emit_start = Instant::now();
    let writer_lanes = if context.backend().capabilities().transactional_clear_range {
        parallelism.clamp(1, 8)
    } else {
        1
    };
    let (assigned, tree_totals) = assign_leaf_partition_keys(&shards)?;
    let leaf_partitions: u64 = tree_totals.values().map(|(_, c)| *c).sum();

    let mut child_by_tree: BTreeMap<Bytes, Vec<(PartitionKey, PartitionCentroid)>> =
        BTreeMap::new();
    for (shard, keys) in shards.iter().zip(&assigned) {
        for (leaf, ak) in shard.leaves.iter().zip(keys) {
            if !ak.is_single_leaf_root {
                child_by_tree
                    .entry(Bytes::copy_from_slice(leaf.tree_key.as_bytes()))
                    .or_default()
                    .push((ak.partition_key, leaf.centroid.clone()));
            }
        }
    }

    let (shards, assigned, locations_by_bucket) = {
        let manifest_c = index.clone();
        let plan_c = plan.clone();
        context
            .compute(move |control| {
                let check = || super::check_control(control);
                let locs = quantize_shards_and_extract_locations(
                    &manifest_c,
                    &plan_c,
                    &mut shards,
                    &assigned,
                    parallelism,
                    &check,
                )?;
                Ok((shards, assigned, locs))
            })
            .await?
    };

    let cpu_adm_base = context.cpu_keep_alive();

    // 5A. Emit `[0x01 Record + Payload + Location]` FIRST in strict ascending `id_bucket` order.
    let mut loc_records = 0_u64;
    let mut loc_mem_lt = LtHash::ZERO;
    let bucket_wave_size = parallelism.clamp(1, 4);
    let mut bucket_iter = verified_id_refs.into_iter().zip(locations_by_bucket);
    let mut pending_rec_kvs: Vec<(Bytes, Bytes)> = Vec::new();

    while bucket_iter.len() > 0 {
        context.checkpoint()?;
        let wave: Vec<_> = bucket_iter.by_ref().take(bucket_wave_size).collect();
        let manifest_c = index.clone();
        let snap_c = snapshot.clone();
        let limits = state.hard_limits;
        let (control, _drop_guard) = drop_guarded_options(&context.options);
        let guard = lock.clone();
        let adm = cpu_adm_base.clone();

        let encode_fut = async move {
            let _drop_guard = _drop_guard;
            tokio::task::spawn_blocking(move || -> Result<Vec<(u64, EncodedLocationBatch)>> {
                let _adm = adm;
                let _lock = guard;
                super::check_control(&control)?;
                let check = || super::check_control(&control);
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(wave.len());
                    for (id_refs, locs) in wave {
                        let m_ref = &manifest_c;
                        let s_ref = &snap_c;
                        let rec_count = locs.len() as u64;
                        handles.push(scope.spawn(
                            move || -> Result<(u64, EncodedLocationBatch)> {
                                let mut seg_reader = s_ref.frame_reader();
                                let batch = encode_record_group_bucket(
                                    m_ref,
                                    &mut seg_reader,
                                    id_refs,
                                    locs,
                                    limits,
                                    &check,
                                )?;
                                Ok((rec_count, batch))
                            },
                        ));
                    }
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        out.push(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
                    }
                    Ok(out)
                })
            })
            .await
            .map_err(|e| Error::with_source(ErrorKind::Other, e))?
        };

        let to_commit = std::mem::take(&mut pending_rec_kvs);
        let committed_len = to_commit.len() as u64;
        let commit_fut = commit_kv_batches(
            context,
            index,
            fence,
            None,
            to_commit,
            writer_lanes,
            budget,
            retry,
        );
        let (_, encoded_buckets) = tokio::try_join!(commit_fut, encode_fut)?;
        if committed_len > 0 {
            bump_loading_progress(context, index, fence, committed_len).await?;
        }

        let mut wave_kvs = Vec::new();
        for (cnt, (bkvs, bkv_lt, bm_lt)) in encoded_buckets {
            loc_records += cnt;
            total_entries += bkvs.len() as u64;
            expected_kv_lthash.add_assign(bkv_lt);
            loc_mem_lt.add_assign(bm_lt);
            wave_kvs.extend(bkvs);
        }
        pending_rec_kvs = wave_kvs;
    }

    if !pending_rec_kvs.is_empty() {
        let committed_len = pending_rec_kvs.len() as u64;
        commit_kv_batches(
            context,
            index,
            fence,
            None,
            pending_rec_kvs,
            writer_lanes,
            budget,
            retry,
        )
        .await?;
        bump_loading_progress(context, index, fence, committed_len).await?;
    }

    // 5B. Emit `[0x03 TreeManifest]` + `[0x04 Partitions]` SECOND in strict ascending key order.
    let (upper_kvs, upper_kv_lt, upper_def_lt, upper_ref_lt, internal_parts) = {
        let manifest_c = index.clone();
        let limits = state.hard_limits;
        let sample_items = descriptor.options().sample_items;
        let graph_mem = state.options.graph_memory_bytes;
        let tree_list: Vec<UpperTreeInput> = tree_totals
            .iter()
            .map(|(raw, (tk, total))| {
                let children = child_by_tree.remove(raw).unwrap_or_default();
                (tk.clone(), *total, children)
            })
            .collect();

        context
            .compute(move |control| {
                let check = || super::check_control(control);
                let mut upper_kvs = Vec::new();
                let mut upper_kv_lt = LtHash::ZERO;
                let mut upper_def_lt = LtHash::ZERO;
                let mut upper_ref_lt = LtHash::ZERO;
                let mut internal_parts = 0_u64;
                for (tk, total_leaves, children) in tree_list {
                    let enc = build_and_encode_upper_tree(
                        &manifest_c,
                        &tk,
                        total_leaves,
                        children,
                        sample_items,
                        graph_mem,
                        limits,
                        parallelism,
                        &check,
                    )?;
                    internal_parts += enc.partitions;
                    upper_kv_lt.add_assign(enc.kv_lthash);
                    upper_def_lt.add_assign(enc.edge_def_lthash);
                    upper_ref_lt.add_assign(enc.edge_ref_lthash);
                    upper_kvs.extend(enc.kvs);
                }
                upper_kvs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                Ok((
                    upper_kvs,
                    upper_kv_lt,
                    upper_def_lt,
                    upper_ref_lt,
                    internal_parts,
                ))
            })
            .await?
    };

    total_entries += upper_kvs.len() as u64;
    expected_kv_lthash.add_assign(upper_kv_lt);

    let upper_split_key = if tree_totals.len() == 1 {
        let (_, (single_tk, single_leaves)) = tree_totals.iter().next().expect("single tree");
        if *single_leaves >= 2 {
            Bytes::from(keys::header_key(
                index.logical_index_id(),
                single_tk,
                PartitionKey::new(2)?,
            ))
        } else {
            Bytes::from(
                keys::tree_manifest_range(index.logical_index_id())
                    .end()
                    .to_vec(),
            )
        }
    } else {
        Bytes::from(
            keys::tree_manifest_range(index.logical_index_id())
                .end()
                .to_vec(),
        )
    };
    let split_pos = upper_kvs.partition_point(|(k, _)| k < &upper_split_key);
    let mut post_leaf_upper_kvs = upper_kvs;
    let pre_leaf_upper_kvs: Vec<_> = post_leaf_upper_kvs.drain(..split_pos).collect();
    let pre_upper_len = pre_leaf_upper_kvs.len() as u64;

    commit_kv_batches(
        context,
        index,
        fence,
        None,
        pre_leaf_upper_kvs,
        writer_lanes,
        budget,
        retry,
    )
    .await?;
    bump_loading_progress(context, index, fence, pre_upper_len).await?;

    let mut leaf_membership_lt = LtHash::ZERO;
    let mut edge_def_lt = LtHash::ZERO;

    let leaf_wave_size = parallelism.clamp(1, 8);
    let mut shard_iter = shards.into_iter().zip(assigned);
    let mut pending_leaf_kvs: Vec<(Bytes, Bytes)> = Vec::new();

    while shard_iter.len() > 0 {
        context.checkpoint()?;
        let wave: Vec<_> = shard_iter.by_ref().take(leaf_wave_size).collect();
        let manifest_c = index.clone();
        let plan_c = plan.clone();
        let limits = state.hard_limits;
        let (control, _drop_guard) = drop_guarded_options(&context.options);
        let adm = cpu_adm_base.clone();

        let encode_fut = async move {
            let _drop_guard = _drop_guard;
            tokio::task::spawn_blocking(move || -> Result<Vec<EncodedLeafShard>> {
                let _adm = adm;
                super::check_control(&control)?;
                let check = || super::check_control(&control);
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(wave.len());
                    for (shard, keys) in wave {
                        let m_ref = &manifest_c;
                        let p_ref = &plan_c;
                        handles.push(scope.spawn(move || {
                            if shard.leaves.is_empty() {
                                return Ok(None);
                            }
                            let enc = encode_leaf_shard_opts(
                                m_ref, p_ref, &shard, &keys, limits, false, &check,
                            )?;
                            drop(shard);
                            Ok(Some(enc))
                        }));
                    }
                    let mut out = Vec::with_capacity(handles.len());
                    for h in handles {
                        if let Some(enc) = h.join().map_err(|_| Error::new(ErrorKind::Other))?? {
                            out.push(enc);
                        }
                    }
                    Ok(out)
                })
            })
            .await
            .map_err(|e| Error::with_source(ErrorKind::Other, e))?
        };

        let to_commit = std::mem::take(&mut pending_leaf_kvs);
        let committed_len = to_commit.len() as u64;
        let commit_fut = commit_kv_batches(
            context,
            index,
            fence,
            None,
            to_commit,
            writer_lanes,
            budget,
            retry,
        );
        let (_, encoded_wave) = tokio::try_join!(commit_fut, encode_fut)?;
        if committed_len > 0 {
            bump_loading_progress(context, index, fence, committed_len).await?;
        }

        let mut wave_kvs = Vec::new();
        for enc in encoded_wave {
            total_entries += enc.kvs.len() as u64;
            expected_kv_lthash.add_assign(enc.kv_lthash);
            leaf_membership_lt.add_assign(enc.membership_lthash);
            edge_def_lt.add_assign(enc.edge_def_lthash);
            wave_kvs.extend(enc.kvs);
        }
        wave_kvs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        pending_leaf_kvs = wave_kvs;
    }
    drop(shard_iter);

    if !pending_leaf_kvs.is_empty() {
        let committed_len = pending_leaf_kvs.len() as u64;
        commit_kv_batches(
            context,
            index,
            fence,
            None,
            pending_leaf_kvs,
            writer_lanes,
            budget,
            retry,
        )
        .await?;
        bump_loading_progress(context, index, fence, committed_len).await?;
    }

    let post_upper_len = post_leaf_upper_kvs.len() as u64;
    commit_kv_batches(
        context,
        index,
        fence,
        None,
        post_leaf_upper_kvs,
        writer_lanes,
        budget,
        retry,
    )
    .await?;
    bump_loading_progress(context, index, fence, post_upper_len).await?;

    if loc_records != total_records
        || leaf_membership_lt != loc_mem_lt
        || edge_def_lt.wrapping_add(upper_def_lt) != upper_ref_lt
    {
        return Err(corrupt());
    }
    report.emit = emit_start.elapsed();

    let total_trees = tree_totals.len() as u64;
    let total_partitions = leaf_partitions + internal_parts;
    let split_keys = build_validation_split_keys(
        index.logical_index_id(),
        &plan,
        total_entries,
        state.hard_limits,
    )?;
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
    Ok(report)
}

type VerifiedBucketsResult = (u64, LtHash, Vec<Vec<IdFrameEntry>>);

fn verify_all_id_buckets_parallel(
    buckets: Vec<Vec<IdFrameEntry>>,
    parallelism: usize,
) -> Result<VerifiedBucketsResult> {
    let chunk_size = buckets.len().div_ceil(parallelism.max(1)).max(1);
    let mut batches = Vec::new();
    let mut iter = buckets.into_iter();
    while iter.len() > 0 {
        batches.push(iter.by_ref().take(chunk_size).collect::<Vec<_>>());
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(batches.len());
        for batch in batches {
            handles.push(scope.spawn(move || -> Result<VerifiedBucketsResult> {
                let mut total = 0_u64;
                let mut lt = LtHash::ZERO;
                let mut out_buckets = Vec::with_capacity(batch.len());
                for ids in batch {
                    let (c, h, sorted) = verify_id_bucket(ids)?;
                    total += c;
                    lt.add_assign(h);
                    out_buckets.push(sorted);
                }
                Ok((total, lt, out_buckets))
            }));
        }
        let mut total = 0_u64;
        let mut lt = LtHash::ZERO;
        let mut all_buckets = Vec::new();
        for h in handles {
            let (c, part, b_list) = h.join().map_err(|_| Error::new(ErrorKind::Other))??;
            total += c;
            lt.add_assign(part);
            all_buckets.extend(b_list);
        }
        Ok((total, lt, all_buckets))
    })
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
            let tree_splits = ((u64::from(heavy.shard_count) * part_budget as u64)
                / non_zero_heavy.get()) as usize;
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

    // Cap total encoded byte footprint (`4 + key.len()` per split key) inside `max_value_bytes`.
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

/// Clears any serving KV keys (`0x01..0x05`) left by an earlier aborted epoch before rewriting.
pub(crate) async fn clear_stale_serving_data<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    fence: WriteFence,
    budget: AdmissionBudget,
    retry: RetryPolicy,
) -> Result<()> {
    let id = index.logical_index_id();
    let range = LogicalRange::serving_data(index);
    let sched_lock = context.bulk_lease_lock.clone();
    if context.backend().capabilities().transactional_clear_range {
        let mut attempts = 0;
        loop {
            context.checkpoint()?;
            let _lease_guard = match &sched_lock {
                Some(l) => Some(l.read().await),
                None => None,
            };
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::for_index(
                    backend.begin_write().await?,
                    index,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                check_lane_fence(&mut txn, index, fence, 0).await?;
                super::bulk_scheduler::authorize(context, &mut txn, id).await?;
                txn.clear_range(&range).await?;
                Ok::<_, Error>(txn)
            }
            .await;
            let txn = match prepared {
                Ok(txn) => txn,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    drop(_lease_guard);
                    retry
                        .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                        .await?;
                    continue;
                }
                Err(e) => return Err(e),
            };
            match context.commit(|start| txn.commit_with(start)).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    drop(_lease_guard);
                    retry
                        .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                        .await?;
                }
            }
        }
    }

    let max_del_muts = budget.max_mutations.max(1);
    let max_del_bytes = budget.max_mutation_bytes.max(1);
    let mut cursor = None;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let page = {
            let mut rtxn = ReadLogicalTxn::for_index(backend.begin_read().await?, index);
            rtxn.scan(
                &range,
                cursor.as_ref(),
                ScanLimits {
                    item_limit: max_del_muts,
                    byte_limit: max_del_bytes,
                },
            )
            .await?
        };
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
                context.checkpoint()?;
                let _lease_guard = match &sched_lock {
                    Some(l) => Some(l.read().await),
                    None => None,
                };
                let prepared = async {
                    let mut txn = WriteLogicalTxn::for_index(
                        backend.begin_write().await?,
                        index,
                        backend.hard_limits(),
                        budget,
                    );
                    check_lane_fence(&mut txn, index, fence, 0).await?;
                    super::bulk_scheduler::authorize(context, &mut txn, id).await?;
                    let mut batch = txn.mutations();
                    for lk in &batch_keys {
                        batch.delete(lk.clone())?;
                    }
                    txn.apply(batch).await?;
                    Ok::<_, Error>(txn)
                }
                .await;
                let txn = match prepared {
                    Ok(txn) => txn,
                    Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                        drop(_lease_guard);
                        retry
                            .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                            .await?;
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                match context.commit(|start| txn.commit_with(start)).await {
                    Ok(()) => break,
                    Err(e) => {
                        drop(_lease_guard);
                        retry
                            .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                            .await?;
                    }
                }
            }
        }
        match next_cursor {
            Some(nc) => cursor = Some(nc),
            None => return Ok(()),
        }
    }
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

/// Splits pre-encoded `(key, value)` pairs into budget-admitted chunks and commits them
/// across up to `min(parallelism, FENCE_LANES)` concurrent writer lanes.
#[expect(
    clippy::too_many_arguments,
    reason = "lane fencing, optional unit lease check, and retry/budget parameters are orthogonal"
)]
pub(crate) async fn commit_kv_batches<B: Backend>(
    context: &OperationContext<B>,
    index: &IndexManifest,
    fence: WriteFence,
    unit_fence: Option<&UnitLease>,
    kvs: Vec<(Bytes, Bytes)>,
    parallelism: usize,
    budget: AdmissionBudget,
    retry: RetryPolicy,
) -> Result<()> {
    if kvs.is_empty() {
        return Ok(());
    }
    let chunks = split_into_chunks(&kvs, budget, unit_fence.is_some())?;
    let concurrency = parallelism.clamp(1, FENCE_LANES as usize).min(chunks.len());

    if concurrency <= 1 || unit_fence.is_some() {
        let lane = unit_fence.map_or(0, |l| l.unit % FENCE_LANES);
        for chunk in chunks {
            commit_one_chunk(
                context.backend(),
                &context.options,
                context.bulk_authority,
                context.bulk_lease_lock.as_deref(),
                index,
                fence,
                lane,
                unit_fence,
                chunk,
                budget,
                retry,
            )
            .await?;
        }
        return Ok(());
    }

    let queue = Arc::new(std::sync::Mutex::new(chunks.into_iter()));
    let mut workers = tokio::task::JoinSet::<Result<()>>::new();
    for lane in 0..concurrency {
        let q = Arc::clone(&queue);
        let backend = context.backend();
        let control = context.options.clone();
        let bulk_auth = context.bulk_authority;
        let sched_lock = context.bulk_lease_lock.clone();
        let manifest = index.clone();
        let lane_id = lane as u32;
        workers.spawn(async move {
            loop {
                super::check_control(&control)?;
                let next_chunk = q.lock().unwrap_or_else(|e| e.into_inner()).next();
                let Some(chunk) = next_chunk else {
                    return Ok(());
                };
                commit_one_chunk(
                    Arc::clone(&backend),
                    &control,
                    bulk_auth,
                    sched_lock.as_deref(),
                    &manifest,
                    fence,
                    lane_id,
                    None,
                    chunk,
                    budget,
                    retry,
                )
                .await?;
            }
        });
    }
    while let Some(res) = workers.join_next().await {
        res.map_err(|e| Error::with_source(ErrorKind::Other, e))??;
    }
    Ok(())
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
    reason = "detached async lane worker owns cloned backend, control, and explicit fencing tokens"
)]
pub(super) async fn commit_one_chunk<B: Backend>(
    backend: Arc<B>,
    control: &OperationOptions,
    bulk_authority: Option<BulkAuthority>,
    sched_lease_lock: Option<&tokio::sync::RwLock<()>>,
    index: &IndexManifest,
    fence: WriteFence,
    lane: u32,
    unit_fence: Option<&UnitLease>,
    chunk: Vec<(Bytes, Bytes)>,
    budget: AdmissionBudget,
    retry: RetryPolicy,
) -> Result<()> {
    let mut attempts = 0;
    let id = index.logical_index_id();
    loop {
        super::check_control(control)?;
        let _sched_guard = match sched_lease_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
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
            warm.push(LogicalKey::BuildUnit {
                index: id,
                stage: STAGE_FENCE,
                unit: lane,
            });
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

            check_lane_fence(&mut txn, index, fence, lane).await?;
            super::bulk_scheduler::authorize_raw(&mut txn, id, bulk_authority).await?;
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
            for (k, v) in &chunk {
                batch.put_encoded(k.clone(), v.clone())?;
            }
            txn.apply(batch).await?;
            Ok(txn)
        }
        .await;

        let txn = match prepared {
            Ok(txn) => txn,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_lease_guard);
                drop(_sched_guard);
                retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        match txn.commit().await {
            Ok(()) => return Ok(()),
            Err(e) => {
                drop(_lease_guard);
                drop(_sched_guard);
                retry
                    .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                    .await?;
            }
        }
    }
}

pub(crate) async fn check_lane_fence<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: &IndexManifest,
    fence: WriteFence,
    lane: u32,
) -> Result<()> {
    building_in_flight(txn, index).await?;
    let id = index.logical_index_id();
    match txn
        .get_for_update(LogicalKey::BuildUnit {
            index: id,
            stage: STAGE_FENCE,
            unit: lane,
        })
        .await?
    {
        Some(PersistentValue::BuildUnit(u)) if u.epoch == fence.epoch && u.owner == fence.owner => {
            Ok(())
        }
        Some(PersistentValue::BuildUnit(_)) => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
        None => match txn.get(LogicalKey::Manifest(id)).await? {
            None => Err(Error::new(ErrorKind::IndexNotFound)),
            Some(PersistentValue::IndexManifest(m))
                if m.lifecycle() == IndexLifecycle::Dropping =>
            {
                Err(Error::new(ErrorKind::IndexDropping))
            }
            Some(PersistentValue::IndexManifest(_)) => {
                Err(Error::new(ErrorKind::BulkBuildSuperseded))
            }
            _ => Err(corrupt()),
        },
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
        context.checkpoint()?;
        let _sched_guard = match &sched_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
        let backend = context.backend();
        let prepared = async {
            let mut txn = WriteLogicalTxn::bootstrap(
                backend.begin_write().await?,
                backend.hard_limits(),
                backend.admission_budget(),
            );
            building_in_flight(&mut txn, index).await?;
            super::bulk_scheduler::authorize(context, &mut txn, id).await?;
            check_lane_fence(&mut txn, index, fence, 0).await?;

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
                return Ok(None);
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
            Ok(Some(txn))
        }
        .await;

        let txn_opt = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        let Some(txn) = txn_opt else {
            return Ok(());
        };
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                drop(_sched_guard);
                retry
                    .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                    .await?;
            }
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
    let mut after = before.clone();
    after.failure = Some(kind);
    let sched_lock = context.bulk_lease_lock.clone();
    let mut attempts = 0;
    let id = index.logical_index_id();
    loop {
        if context.checkpoint().is_err() {
            return Ok(());
        }
        let _sched_guard = match &sched_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
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
                    return Ok(None);
                }
                Err(e) => return Err(e),
            }
            if super::bulk_scheduler::authorize(context, &mut txn, id)
                .await
                .is_err()
            {
                txn.rollback().await;
                return Ok(None);
            }
            if let Some(fence) = worker_fence {
                if check_lane_fence(&mut txn, index, fence, 0).await.is_err() {
                    txn.rollback().await;
                    return Ok(None);
                }
                if matches!(
                    txn.get_for_update(LogicalKey::BuildProgress(id)).await?,
                    Some(PersistentValue::BuildProgress(p)) if p.phase == BuildPhase::Validating
                ) {
                    txn.rollback().await;
                    return Ok(None);
                }
            }
            let key = LogicalKey::BuildWorkspace(id);
            let current = match txn.get_for_update(key.clone()).await? {
                Some(PersistentValue::BuildWorkspace(w)) => w,
                _ => {
                    txn.rollback().await;
                    return Ok(None);
                }
            };
            if current == after || current.failure.is_some() {
                txn.rollback().await;
                return Ok(None);
            }
            if current.epoch != before.epoch || current.token != before.token {
                txn.rollback().await;
                return Ok(None);
            }
            txn.put(key, PersistentValue::BuildWorkspace(after.clone()))
                .await?;
            Ok(Some(txn))
        }
        .await;

        let txn_opt = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                if retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                continue;
            }
            Err(e) => return Err(e),
        };
        let Some(txn) = txn_opt else {
            return Ok(());
        };
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(()),
            Err(e) if matches!(e.kind(), ErrorKind::Cancelled | ErrorKind::DeadlineExceeded) => {
                return Ok(());
            }
            Err(e) => {
                drop(_sched_guard);
                retry
                    .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                    .await?;
            }
        }
    }
}

async fn ensure_no_peer_scheduler_leases<B: Backend, T: WriteTxn>(
    context: &OperationContext<B>,
    txn: &mut WriteLogicalTxn<'_, T>,
    id: LogicalIndexId,
    distributed_workers: usize,
) -> Result<()> {
    let now = super::bulk_scheduler::now_ms()?;
    if let Some(PersistentValue::BuildSchedule(s)) =
        txn.get_for_update(LogicalKey::BuildSchedule(id)).await?
        && s.options.distributed_workers <= 1
        && s.owner != [0; 32]
        && s.expires_ms > now
        && context.bulk_authority
            != Some(BulkAuthority::Scheduled {
                slot: 0,
                owner: s.owner,
            })
    {
        return Err(Error::new(ErrorKind::BulkBuildBusy));
    }
    let slots = distributed_workers.max(1) as u32;
    if slots > 1 {
        let keys: Vec<LogicalKey> = (0..slots)
            .map(|slot| LogicalKey::BuildScheduleSlot { index: id, slot })
            .collect();
        txn.warm_for_update(keys).await?;
        let max_inline_del = txn.admission_budget().max_mutations.saturating_sub(2);
        let mut expired_to_del = Vec::new();
        for slot in 0..slots {
            let skey = LogicalKey::BuildScheduleSlot { index: id, slot };
            match txn.get_for_update(skey.clone()).await? {
                None => {}
                Some(PersistentValue::BuildUnit(u)) => {
                    let is_self = matches!(
                        context.bulk_authority,
                        Some(
                            BulkAuthority::Scheduled { slot: s, owner }
                                | BulkAuthority::UnscheduledPeer { slot: s, owner }
                        ) if s == slot && owner == u.owner
                    );
                    if !is_self && u.owner != [0; 32] && u.expires_ms > now {
                        return Err(Error::new(ErrorKind::BulkBuildBusy));
                    }
                    if !is_self && expired_to_del.len() < max_inline_del {
                        expired_to_del.push(skey);
                    }
                }
                Some(_) => return Err(corrupt()),
            }
        }
        for skey in expired_to_del {
            txn.delete(skey).await?;
        }
    }
    Ok(())
}

/// Reclaims transient `BuildUnit` (`0x07`) keys from a published `Active` index while
/// preserving compact `BuildDescriptor` (`0x05`) and `BuildProgress` (`0x06`).
async fn reclaim_published_build_records<B: Backend>(
    context: &mut OperationContext<B>,
    active_manifest: &IndexManifest,
    retry: RetryPolicy,
) -> Result<()> {
    let id = active_manifest.logical_index_id();
    let range = LogicalRange::transient_build_state(active_manifest);
    let sched_lock = context.bulk_lease_lock.clone();
    if context.backend().capabilities().transactional_clear_range {
        let mut attempts = 0;
        loop {
            context.checkpoint()?;
            let _sched_guard = match &sched_lock {
                Some(l) => Some(l.read().await),
                None => None,
            };
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::for_index(
                    backend.begin_write().await?,
                    active_manifest,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                match txn.get_for_update(LogicalKey::Manifest(id)).await? {
                    None => {
                        txn.rollback().await;
                        return Ok(None);
                    }
                    Some(PersistentValue::IndexManifest(m))
                        if m.logical_index_id() == id
                            && m.lifecycle() == IndexLifecycle::Active => {}
                    Some(PersistentValue::IndexManifest(m))
                        if m.logical_index_id() == id
                            && m.lifecycle() == IndexLifecycle::Dropping =>
                    {
                        txn.rollback().await;
                        return Ok(None);
                    }
                    Some(PersistentValue::IndexManifest(m))
                        if m.logical_index_id() == id
                            && m.lifecycle() == IndexLifecycle::Building =>
                    {
                        return Err(Error::invalid_argument());
                    }
                    _ => return Err(corrupt()),
                }
                txn.clear_range(&range).await?;
                Ok(Some(txn))
            }
            .await;
            let txn_opt = match prepared {
                Ok(v) => v,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    drop(_sched_guard);
                    retry
                        .wait_or_exhaust(Operation::CleanupBulkBuild, &mut attempts)
                        .await?;
                    continue;
                }
                Err(e) => return Err(e),
            };
            let Some(txn) = txn_opt else {
                return Ok(());
            };
            match context.commit(|start| txn.commit_with(start)).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    drop(_sched_guard);
                    retry
                        .after_commit_error(Operation::CleanupBulkBuild, &mut attempts, e)
                        .await?;
                }
            }
        }
    }

    let budget = context.backend().admission_budget();
    let max_del_muts = budget.max_mutations.max(1);
    let max_del_bytes = budget.max_mutation_bytes.max(1);
    let mut cursor = None;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let page = {
            let mut rtxn = ReadLogicalTxn::for_index(backend.begin_read().await?, active_manifest);
            rtxn.scan(
                &range,
                cursor.as_ref(),
                ScanLimits {
                    item_limit: max_del_muts,
                    byte_limit: max_del_bytes,
                },
            )
            .await?
        };
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
                context.checkpoint()?;
                let _sched_guard = match &sched_lock {
                    Some(l) => Some(l.read().await),
                    None => None,
                };
                let prepared = async {
                    let mut txn = WriteLogicalTxn::bootstrap(
                        backend.begin_write().await?,
                        backend.hard_limits(),
                        budget,
                    );
                    match txn.get_for_update(LogicalKey::Manifest(id)).await? {
                        None => {
                            txn.rollback().await;
                            return Ok(None);
                        }
                        Some(PersistentValue::IndexManifest(m))
                            if m.logical_index_id() == id
                                && m.lifecycle() == IndexLifecycle::Active => {}
                        Some(PersistentValue::IndexManifest(m))
                            if m.logical_index_id() == id
                                && m.lifecycle() == IndexLifecycle::Dropping =>
                        {
                            txn.rollback().await;
                            return Ok(None);
                        }
                        Some(PersistentValue::IndexManifest(m))
                            if m.logical_index_id() == id
                                && m.lifecycle() == IndexLifecycle::Building =>
                        {
                            return Err(Error::invalid_argument());
                        }
                        _ => return Err(corrupt()),
                    }
                    let mut batch = txn.mutations();
                    for lk in &batch_keys {
                        batch.delete(lk.clone())?;
                    }
                    txn.apply(batch).await?;
                    Ok(Some(txn))
                }
                .await;

                let txn_opt = match prepared {
                    Ok(v) => v,
                    Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                        drop(_sched_guard);
                        retry
                            .wait_or_exhaust(Operation::CleanupBulkBuild, &mut attempts)
                            .await?;
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                let Some(txn) = txn_opt else {
                    return Ok(());
                };
                match context.commit(|start| txn.commit_with(start)).await {
                    Ok(()) => break,
                    Err(e) => {
                        drop(_sched_guard);
                        retry
                            .after_commit_error(Operation::CleanupBulkBuild, &mut attempts, e)
                            .await?;
                    }
                }
            }
        }
        match next_cursor {
            Some(nc) => cursor = Some(nc),
            None => return Ok(()),
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
    let mut read_attempts = 0;
    let (active_manifest, state_opt) = loop {
        context.checkpoint()?;
        let backend = context.backend();
        let res = async {
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
        .await;
        match res {
            Ok(pair) => break pair,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::CleanupBulkBuild, &mut read_attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    };

    if let Some(active) = &active_manifest {
        reclaim_published_build_records(context, active, retry).await?;
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
        context.checkpoint()?;
        let _sched_guard = match &sched_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
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
                    return Ok(None);
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
            Ok(Some(txn))
        }
        .await;

        let txn_opt = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                retry
                    .wait_or_exhaust(Operation::CleanupBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        if let Some(txn) = txn_opt {
            match context.commit(|start| txn.commit_with(start)).await {
                Ok(()) => break,
                Err(e) => {
                    drop(_sched_guard);
                    retry
                        .after_commit_error(Operation::CleanupBulkBuild, &mut attempts, e)
                        .await?;
                }
            }
        } else {
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
