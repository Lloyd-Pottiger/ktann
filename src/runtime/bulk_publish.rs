//! Parallel slice validation of frozen backend bytes and atomic `Active` publication.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;

use crate::api::{Error, ErrorKind, Result};
use crate::bulk::LtHash;
use crate::observe::labels::Operation;
use crate::storage::backend::{Backend, ReadTxn, ScanLimits};
use crate::storage::keys::{self, LogicalKey};
use crate::storage::values::{
    BuildDescriptor, BuildPhase, BuildProgress, BuildUnit, BuildWorkspace, IndexLifecycle,
    IndexManifest, PersistentValue, ValueCodec,
};
use crate::storage::{LogicalRange, LogicalScanCursor, ReadLogicalTxn, WriteLogicalTxn};

use super::{
    OperationContext,
    bulk_scheduler::authorize_raw,
    bulk_worker::{
        self, CommitStep,
        bulk_pipeline::{
            StageEnv, StageFence, UnitOutput, drive_stage, fence_validation, init_stage_units,
        },
        corrupt, effective_budget, lock_sched_read, retry_ctx_write, retry_read,
        validate_split_keys_in_index,
    },
    lifecycle::RetryPolicy,
};

pub(crate) const STAGE_VALIDATE: u8 = 40;
const READ_SNAPSHOT_ROTATE_INTERVAL: Duration = Duration::from_secs(2);

pub(crate) async fn publish<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    descriptor: BuildDescriptor,
    retry: RetryPolicy,
) -> Result<IndexManifest> {
    let result = publish_inner(context, index.clone(), descriptor, retry).await;
    if result.is_err()
        && let Ok(Some(active)) = already_active_retry(context, &index, retry).await
    {
        return Ok(active);
    }
    result
}

async fn publish_inner<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    descriptor: BuildDescriptor,
    retry: RetryPolicy,
) -> Result<IndexManifest> {
    context.checkpoint()?;
    let id = index.logical_index_id();
    let backend = context.backend();
    let state_or_active = retry_read(&context.options, Operation::PublishBulkBuild, retry, || {
        let backend = Arc::clone(&backend);
        let expected = &index;
        async move {
            let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
            if let Some(active) = read_active(&mut txn, expected).await? {
                return Ok((Some(active), None));
            }
            match txn.get(LogicalKey::BuildWorkspace(id)).await? {
                Some(PersistentValue::BuildWorkspace(state)) => Ok((None, Some(state))),
                None => Err(Error::new(ErrorKind::BulkBuildBusy)),
                _ => Err(corrupt()),
            }
        }
    })
    .await?;

    let state = match state_or_active {
        (Some(active), _) => return Ok(active),
        (None, Some(state)) => state,
        (None, None) => return Err(corrupt()),
    };
    if let Some(kind) = state.failure {
        return Err(Error::new(kind));
    }

    let result = async {
        let progress = begin_validation(context, &index, &descriptor, &state, retry).await?;
        validate_all_slices(context, &index, &state, &progress, retry).await?;
        activate(context, &index, &descriptor, &state, &progress, retry).await
    }
    .await;
    if let Err(e) = &result
        && e.kind() != ErrorKind::LimitExceeded
    {
        bulk_worker::record_failure(context, &index, &state, None, e.kind(), retry).await?;
    }
    result
}

/// Recognizes publication for the handle's immutable identity across transient read aborts.
pub(super) async fn already_active_retry<B: Backend>(
    context: &OperationContext<B>,
    expected: &IndexManifest,
    retry: RetryPolicy,
) -> Result<Option<IndexManifest>> {
    let backend = context.backend();
    retry_read(&context.options, Operation::PublishBulkBuild, retry, || {
        let backend = Arc::clone(&backend);
        async move {
            let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
            read_active(&mut txn, expected).await
        }
    })
    .await
}

async fn read_active<T: ReadTxn>(
    txn: &mut ReadLogicalTxn<'_, T>,
    expected: &IndexManifest,
) -> Result<Option<IndexManifest>> {
    match txn
        .get(LogicalKey::Manifest(expected.logical_index_id()))
        .await?
    {
        Some(PersistentValue::IndexManifest(m)) if m.has_same_immutable_identity(expected) => {
            match m.lifecycle() {
                IndexLifecycle::Active => Ok(Some(m)),
                IndexLifecycle::Building => Ok(None),
                IndexLifecycle::Dropping => Err(Error::new(ErrorKind::IndexDropping)),
            }
        }
        None => Err(Error::new(ErrorKind::IndexNotFound)),
        _ => Err(corrupt()),
    }
}

/// Atomically clears `BuildWorkspace.writer_token` and flips `BuildProgress` to `Validating`,
/// initializing `STAGE_VALIDATE` `BuildUnit`s inline when they fit in one transaction.
async fn begin_validation<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &BuildWorkspace,
    retry: RetryPolicy,
) -> Result<BuildProgress> {
    let id = index.logical_index_id();
    let backend = context.backend();
    let budget = backend.admission_budget();
    let bulk_authority = context.bulk_authority;

    retry_ctx_write(
        context,
        backend.as_ref(),
        budget,
        Operation::PublishBulkBuild,
        retry,
        |mut txn| async move {
            bulk_worker::building_in_flight(&mut txn, index).await?;
            authorize_raw(&mut txn, id, bulk_authority).await?;
            let ws = txn.get_for_update(LogicalKey::BuildWorkspace(id)).await?;
            match txn.get(LogicalKey::BuildDescriptor(id)).await? {
                Some(PersistentValue::BuildDescriptor(d)) if d == *descriptor => {}
                _ => return Err(corrupt()),
            }
            let progress = match txn.get_for_update(LogicalKey::BuildProgress(id)).await? {
                None => return Err(Error::new(ErrorKind::BulkBuildBusy)),
                Some(PersistentValue::BuildProgress(p)) => p,
                _ => return Err(corrupt()),
            };
            let mut w = match ws {
                Some(PersistentValue::BuildWorkspace(w)) => {
                    if let Some(kind) = w.failure {
                        return Err(Error::new(kind));
                    }
                    if w.token != state.token
                        || w.epoch != state.epoch
                        || w.options != state.options
                        || w.hard_limits != state.hard_limits
                    {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    w
                }
                None if progress.phase == BuildPhase::Validating => return Err(corrupt()),
                None => return Err(Error::new(ErrorKind::BulkBuildSuperseded)),
                _ => return Err(corrupt()),
            };
            validate_split_keys_in_index(id, &progress.validation_split_keys)?;
            if progress.total_records != descriptor.input().items() {
                return Err(corrupt());
            }
            if progress.epoch != state.epoch {
                return Err(Error::new(ErrorKind::BulkBuildSuperseded));
            }
            match progress.phase {
                BuildPhase::Preparing | BuildPhase::Loading => {
                    return Err(Error::new(ErrorKind::BulkBuildBusy));
                }
                BuildPhase::Validating => {
                    if w.writer_token != [0; 32] {
                        return Err(corrupt());
                    }
                    txn.rollback().await;
                    return Ok((None, progress));
                }
                BuildPhase::Loaded => {}
            }

            w.writer_token = [0; 32];
            txn.put(
                LogicalKey::BuildWorkspace(id),
                PersistentValue::BuildWorkspace(w),
            )
            .await?;
            let slices = progress.validation_slices();
            if budget.max_mutations >= (slices as usize + 2) {
                for s in 0..slices {
                    txn.put(
                        LogicalKey::BuildUnit {
                            index: id,
                            stage: STAGE_VALIDATE,
                            unit: s,
                        },
                        PersistentValue::BuildUnit(BuildUnit::unclaimed(progress.epoch)),
                    )
                    .await?;
                }
            }
            let next = BuildProgress {
                phase: BuildPhase::Validating,
                ..progress
            };
            txn.put(
                LogicalKey::BuildProgress(id),
                PersistentValue::BuildProgress(next.clone()),
            )
            .await?;
            Ok((Some(txn), next))
        },
    )
    .await
}

fn slice_logical_range(
    manifest: &IndexManifest,
    split_keys: &[Bytes],
    slice: u32,
) -> Result<LogicalRange> {
    let id = manifest.logical_index_id();
    validate_split_keys_in_index(id, split_keys)?;
    let serving = keys::serving_data_range(id);
    let idx = slice as usize;
    if idx > split_keys.len() {
        return Err(corrupt());
    }
    let start = if idx == 0 {
        serving.start().to_vec()
    } else {
        split_keys[idx - 1].to_vec()
    };
    let end = if idx < split_keys.len() {
        split_keys[idx].to_vec()
    } else {
        serving.end().to_vec()
    };
    LogicalRange::validation_slice(manifest, start, end)
}

async fn validate_all_slices<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    state: &BuildWorkspace,
    progress: &BuildProgress,
    retry: RetryPolicy,
) -> Result<()> {
    let slices = progress.validation_slices();
    let budget = effective_budget(context.backend().admission_budget(), state.options.load)?;
    let concurrency = state.options.parallelism.clamp(1, 16).min(slices as usize);
    let split_keys = Arc::new(progress.validation_split_keys.clone());

    let env = StageEnv {
        backend: context.backend(),
        control: context.options.clone(),
        bulk_authority: context.bulk_authority,
        sched_lease_lock: context.bulk_lease_lock.clone(),
        distributed_workers: state.options.distributed_workers,
        writer_lanes: 1,
        commit_sem: Arc::new(tokio::sync::Semaphore::new(1)),
        admission: context.cpu_keep_alive(),
        manifest: index.clone(),
        fence: StageFence::Validate {
            ws_token: state.token,
            epoch: progress.epoch,
        },
        unit_lease: state.options.unit_lease,
        budget,
        retry,
    };

    if context.backend().admission_budget().max_mutations < (slices as usize + 2) {
        init_stage_units(&env, STAGE_VALIDATE, slices).await?;
    }
    drive_stage(&env, STAGE_VALIDATE, slices, concurrency, {
        let env = env.clone();
        move |slice, _lease| {
            let env_c = env.clone();
            let splits = Arc::clone(&split_keys);
            async move { scan_validation_slice(&env_c, &splits, slice).await }
        }
    })
    .await?;
    Ok(())
}

/// Scans `[slice_start, slice_end)` across rotated frozen snapshots via `scan_raw_validated`,
/// decoding every serving key and value fail-closed and returning `(kv_count, kv_lthash)`.
async fn scan_validation_slice<B: Backend>(
    env: &StageEnv<B>,
    split_keys: &[Bytes],
    slice: u32,
) -> Result<UnitOutput> {
    let range = slice_logical_range(&env.manifest, split_keys, slice)?;
    let epoch = env.fence.epoch();
    let page_items = env.budget.max_mutations.max(1);
    let page_bytes = env.budget.max_mutation_bytes.max(1);
    let mut cursor: Option<LogicalScanCursor> = None;
    let mut slice_count = 0_u64;
    let mut slice_lt = LtHash::ZERO;
    let mut rtxn: Option<(ReadLogicalTxn<'_, B::ReadTxn<'_>>, Instant)> = None;

    loop {
        super::check_control(&env.control)?;
        let mut page_attempts = 0;
        let (raw_items, next_cursor) = loop {
            super::check_control(&env.control)?;
            let need_fresh = rtxn
                .as_ref()
                .is_none_or(|(_, opened)| opened.elapsed() >= READ_SNAPSHOT_ROTATE_INTERVAL);
            if need_fresh {
                rtxn = None;
                match open_frozen_validation_snapshot(env.backend.as_ref(), &env.manifest, epoch)
                    .await
                {
                    Ok(fresh) => rtxn = Some((fresh, Instant::now())),
                    Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                        env.retry
                            .wait_or_exhaust(Operation::PublishBulkBuild, &mut page_attempts)
                            .await?;
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
            let (active_rtxn, _) = rtxn.as_mut().expect("snapshot open");
            match active_rtxn
                .scan_raw_validated(
                    &range,
                    cursor.as_ref(),
                    ScanLimits {
                        item_limit: page_items,
                        byte_limit: page_bytes,
                    },
                )
                .await
            {
                Ok(page) => break page,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    rtxn = None;
                    env.retry
                        .wait_or_exhaust(Operation::PublishBulkBuild, &mut page_attempts)
                        .await?;
                }
                Err(e) => return Err(e),
            }
        };

        if !raw_items.is_empty() {
            let manifest_c = env.manifest.clone();
            let (c_delta, lt_delta) = env
                .spawn_cpu(move |ctrl| {
                    let (types, tcount) = manifest_c.tree_key_types();
                    let codec = ValueCodec::for_index(&manifest_c);
                    let mut count = 0_u64;
                    let mut lt = LtHash::ZERO;
                    for (i, (k, v)) in raw_items.into_iter().enumerate() {
                        if i.is_multiple_of(256) {
                            super::check_control(&ctrl)?;
                        }
                        let lk = keys::decode_key(&types[..tcount], &k)?;
                        if lk.index() != Some(manifest_c.logical_index_id()) {
                            return Err(corrupt());
                        }
                        codec.decode(&lk, v.clone())?;
                        count = count.checked_add(1).ok_or_else(corrupt)?;
                        lt.add_assign(LtHash::from_kv(&k, &v));
                    }
                    Ok((count, lt))
                })
                .await?;
            slice_count = slice_count.checked_add(c_delta).ok_or_else(corrupt)?;
            slice_lt.add_assign(lt_delta);
        }

        match next_cursor {
            Some(nc) => cursor = Some(nc),
            None => break,
        }
    }
    drop(rtxn);
    Ok(UnitOutput {
        kv_count: slice_count,
        kv_lthash: slice_lt,
        ..UnitOutput::default()
    })
}

/// Opens a fresh `ReadLogicalTxn` bound to `expected_manifest` after verifying in the same
/// snapshot that `Manifest`, `BuildWorkspace`, and `BuildProgress` remain frozen in
/// `(Building, Validating, expected_epoch)`.
async fn open_frozen_validation_snapshot<'a, B: Backend>(
    backend: &'a B,
    expected_manifest: &'a IndexManifest,
    expected_epoch: u64,
) -> Result<ReadLogicalTxn<'a, B::ReadTxn<'a>>> {
    let id = expected_manifest.logical_index_id();
    let mut boot = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
    let fetched = boot
        .batch_get(vec![
            LogicalKey::Manifest(id),
            LogicalKey::BuildProgress(id),
            LogicalKey::BuildWorkspace(id),
        ])
        .await?;
    if fetched.len() != 3 {
        return Err(Error::new(ErrorKind::Backend));
    }
    bulk_worker::check_manifest_building_in_flight(fetched[0].as_ref(), expected_manifest)?;
    match &fetched[2] {
        Some(PersistentValue::BuildWorkspace(w)) => {
            if let Some(kind) = w.failure {
                return Err(Error::new(kind));
            }
            if w.epoch != expected_epoch || w.writer_token != [0; 32] {
                return Err(Error::new(ErrorKind::BulkBuildSuperseded));
            }
        }
        _ => return Err(corrupt()),
    }
    match &fetched[1] {
        Some(PersistentValue::BuildProgress(p))
            if p.phase == BuildPhase::Validating && p.epoch == expected_epoch => {}
        Some(PersistentValue::BuildProgress(_)) => {
            return Err(Error::new(ErrorKind::BulkBuildSuperseded));
        }
        _ => return Err(corrupt()),
    }
    Ok(ReadLogicalTxn::for_index(
        boot.into_raw(),
        expected_manifest,
    ))
}

/// Verifies the complete slice inventory sum against `BuildProgress` and `BuildDescriptor`,
/// clears `validation_split_keys` on `BuildProgress`, and atomically activates the `IndexManifest`.
async fn activate<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &BuildWorkspace,
    progress: &BuildProgress,
    retry: RetryPolicy,
) -> Result<IndexManifest> {
    let active = index.clone().with_lifecycle(IndexLifecycle::Active);
    let id = index.logical_index_id();
    let slices = progress.validation_slices();
    let unit_range = LogicalRange::build_units(id, STAGE_VALIDATE);
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

                let mut warm = vec![
                    LogicalKey::Manifest(id),
                    LogicalKey::BuildWorkspace(id),
                    LogicalKey::BuildProgress(id),
                    LogicalKey::BuildSchedule(id),
                ];
                if let Some(auth) = context.bulk_authority {
                    warm.push(auth.slot_key(id));
                }
                txn.warm_for_update(warm).await?;

                let current = bulk_worker::manifest(&mut txn, index).await?;
                if current.lifecycle() == IndexLifecycle::Active {
                    txn.rollback().await;
                    return Ok((None, current));
                }
                let cur_prog =
                    fence_validation(&mut txn, index, state.token, progress.epoch).await?;
                authorize_raw(&mut txn, id, context.bulk_authority).await?;
                match txn.get(LogicalKey::BuildDescriptor(id)).await? {
                    Some(PersistentValue::BuildDescriptor(d)) if d == *descriptor => {}
                    _ => return Err(corrupt()),
                }
                if cur_prog.total_records != descriptor.input().items()
                    || cur_prog.validation_slices() != slices
                {
                    return Err(corrupt());
                }

                let mut cursor = None;
                let mut seen_slices = 0_u32;
                let mut total_verified = 0_u64;
                let mut total_lt = LtHash::ZERO;
                loop {
                    let page = txn
                        .scan(
                            &unit_range,
                            cursor.as_ref(),
                            ScanLimits {
                                item_limit: 260,
                                byte_limit: 512 * 1024,
                            },
                        )
                        .await?;
                    for item in page.items() {
                        let PersistentValue::BuildUnit(u) = item.value() else {
                            return Err(corrupt());
                        };
                        if !u.complete || u.epoch != cur_prog.epoch {
                            return Err(Error::new(ErrorKind::BulkBuildBusy));
                        }
                        seen_slices += 1;
                        total_verified =
                            total_verified.checked_add(u.kv_count).ok_or_else(corrupt)?;
                        total_lt.add_assign(u.kv_lthash);
                    }
                    cursor = page.into_next_cursor();
                    if cursor.is_none() {
                        break;
                    }
                }
                if seen_slices != slices {
                    return Err(Error::new(ErrorKind::BulkBuildBusy));
                }
                if total_verified != cur_prog.total_entries
                    || total_lt != cur_prog.expected_kv_lthash
                {
                    return Err(corrupt());
                }

                let mut compact_prog = cur_prog;
                compact_prog.validation_split_keys.clear();
                txn.put(
                    LogicalKey::BuildProgress(id),
                    PersistentValue::BuildProgress(compact_prog),
                )
                .await?;
                txn.put(
                    LogicalKey::Manifest(id),
                    PersistentValue::IndexManifest(active.clone()),
                )
                .await?;
                Ok((Some(txn), active.clone()))
            }
            .await;
            context.commit_prepared(prepared).await
        };

        match step {
            CommitStep::Done(published) => return Ok(published),
            CommitStep::PrepareAbort => {
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
            }
            CommitStep::CommitFailed(e) if e.kind() == ErrorKind::CommitOutcomeUnknown => {
                if let Some(active_now) = already_active_retry(context, index, retry).await? {
                    return Ok(active_now);
                }
                if retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await
                    .is_err()
                {
                    return Err(e);
                }
            }
            CommitStep::CommitFailed(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
            }
            CommitStep::CommitFailed(e) | CommitStep::Fatal(e) => return Err(e),
        }
    }
}
