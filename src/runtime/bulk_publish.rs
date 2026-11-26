//! Parallel slice validation of frozen backend bytes and atomic `Active` publication.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;

use crate::api::{Error, ErrorKind, OperationOptions, Result};
use crate::bulk::LtHash;
use crate::observe::labels::Operation;
use crate::storage::backend::{Backend, ScanLimits, WriteTxn};
use crate::storage::keys::{self, LogicalKey};
use crate::storage::values::{
    BuildDescriptor, BuildPhase, BuildProgress, BuildUnit, BuildWorkspace, IndexLifecycle,
    IndexManifest, PersistentValue, ValueCodec,
};
use crate::storage::{LogicalRange, LogicalScanCursor, ReadLogicalTxn, WriteLogicalTxn};

use super::{
    OperationContext,
    bulk_scheduler::{BulkAuthority, authorize_raw, expiry, now_ms},
    bulk_worker::{
        self, FENCE_LANES, STAGE_FENCE, corrupt, drop_guarded_options, effective_budget,
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
    let state = {
        let mut attempts = 0;
        loop {
            context.checkpoint()?;
            let backend = context.backend();
            let res = async {
                let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
                if let Some(active) = read_active(&mut txn, &index).await? {
                    return Ok((Some(active), None));
                }
                match txn.get(LogicalKey::BuildWorkspace(id)).await? {
                    Some(PersistentValue::BuildWorkspace(state)) => Ok((None, Some(state))),
                    None => Err(Error::new(ErrorKind::BulkBuildBusy)),
                    _ => Err(corrupt()),
                }
            }
            .await;
            match res {
                Ok((Some(active), _)) => return Ok(active),
                Ok((None, Some(state))) => break state,
                Ok((None, None)) => return Err(corrupt()),
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    retry
                        .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                        .await?;
                }
                Err(e) => return Err(e),
            }
        }
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
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let res = async {
            let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
            read_active(&mut txn, expected).await
        }
        .await;
        match res {
            Ok(opt) => return Ok(opt),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn read_active<T: crate::storage::backend::ReadTxn>(
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

async fn fence_validation<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: &IndexManifest,
    state: &BuildWorkspace,
    expected_epoch: u64,
) -> Result<BuildProgress> {
    bulk_worker::building_in_flight(txn, index).await?;
    let id = index.logical_index_id();
    let ws = txn.get_for_update(LogicalKey::BuildWorkspace(id)).await?;
    let prog = match txn.get(LogicalKey::BuildProgress(id)).await? {
        Some(PersistentValue::BuildProgress(p))
            if p.phase == BuildPhase::Validating && p.epoch == expected_epoch =>
        {
            validate_split_keys_in_index(id, &p.validation_split_keys)?;
            p
        }
        Some(PersistentValue::BuildProgress(_)) => {
            return Err(Error::new(ErrorKind::BulkBuildSuperseded));
        }
        _ => return Err(corrupt()),
    };
    match ws {
        Some(PersistentValue::BuildWorkspace(w)) => {
            if let Some(kind) = w.failure {
                return Err(Error::new(kind));
            }
            if w.token != state.token || w.epoch != expected_epoch {
                return Err(Error::new(ErrorKind::BulkBuildSuperseded));
            }
        }
        None => return Err(corrupt()),
        _ => return Err(corrupt()),
    }
    Ok(prog)
}

/// Pre-stages `STAGE_VALIDATE` `BuildUnit`s (and revokes extra `STAGE_FENCE` lanes `1..16` when
/// `max_mutations` is tight) in bounded batches, then atomically revokes remaining writer lanes
/// and flips `BuildProgress` from `Loaded` to `Validating`.
async fn begin_validation<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &BuildWorkspace,
    retry: RetryPolicy,
) -> Result<BuildProgress> {
    let id = index.logical_index_id();
    let (loaded_prog, slices) = {
        let mut attempts = 0;
        loop {
            context.checkpoint()?;
            let backend = context.backend();
            let res = async {
                let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
                if read_active(&mut rtxn, index).await?.is_some() {
                    return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                }
                match rtxn.get(LogicalKey::BuildDescriptor(id)).await? {
                    Some(PersistentValue::BuildDescriptor(d)) if d == *descriptor => {}
                    _ => return Err(corrupt()),
                }
                let p = match rtxn.get(LogicalKey::BuildProgress(id)).await? {
                    None => return Err(Error::new(ErrorKind::BulkBuildBusy)),
                    Some(PersistentValue::BuildProgress(p)) => p,
                    _ => return Err(corrupt()),
                };
                validate_split_keys_in_index(id, &p.validation_split_keys)?;
                if p.total_records != descriptor.input().items() {
                    return Err(corrupt());
                }
                if p.epoch != state.epoch {
                    return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                }
                match p.phase {
                    BuildPhase::Preparing | BuildPhase::Loading => {
                        Err(Error::new(ErrorKind::BulkBuildBusy))
                    }
                    BuildPhase::Validating | BuildPhase::Loaded => {
                        let s = p.validation_slices();
                        Ok((p, s))
                    }
                }
            }
            .await;
            match res {
                Ok(v) => break v,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    retry
                        .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                        .await?;
                }
                Err(e) => return Err(e),
            }
        }
    };
    if loaded_prog.phase == BuildPhase::Validating {
        return Ok(loaded_prog);
    }

    let max_muts = context.backend().admission_budget().max_mutations;
    let inline_all = max_muts >= (slices as usize + FENCE_LANES as usize + 2);
    if !inline_all {
        let per_batch = max_muts.saturating_sub(2).clamp(1, 256) as u32;
        for batch_start in (0..slices).step_by(per_batch as usize) {
            let batch_end = (batch_start + per_batch).min(slices);
            let mut attempts = 0;
            loop {
                context.checkpoint()?;
                let _sched_guard = match &context.bulk_lease_lock {
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
                    bulk_worker::building_in_flight(&mut txn, index).await?;
                    super::bulk_scheduler::authorize(context, &mut txn, id).await?;
                    let p = match txn.get(LogicalKey::BuildProgress(id)).await? {
                        Some(PersistentValue::BuildProgress(p)) => p,
                        _ => return Err(corrupt()),
                    };
                    if p.epoch != state.epoch {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    if p.phase == BuildPhase::Validating {
                        txn.rollback().await;
                        return Ok(None);
                    }
                    if p.phase != BuildPhase::Loaded {
                        return Err(Error::new(ErrorKind::BulkBuildBusy));
                    }
                    let mut any = false;
                    for s in batch_start..batch_end {
                        let ukey = LogicalKey::BuildUnit {
                            index: id,
                            stage: STAGE_VALIDATE,
                            unit: s,
                        };
                        match txn.get_for_update(ukey.clone()).await? {
                            Some(PersistentValue::BuildUnit(u)) if u.epoch == state.epoch => {}
                            _ => {
                                txn.put(
                                    ukey,
                                    PersistentValue::BuildUnit(BuildUnit::unclaimed(state.epoch)),
                                )
                                .await?;
                                any = true;
                            }
                        }
                    }
                    if !any {
                        txn.rollback().await;
                        return Ok(None);
                    }
                    Ok(Some(txn))
                }
                .await;
                let txn_opt = match prepared {
                    Ok(v) => v,
                    Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                        drop(_sched_guard);
                        retry
                            .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                            .await?;
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                let Some(txn) = txn_opt else {
                    break;
                };
                match txn.commit().await {
                    Ok(()) => break,
                    Err(e) => {
                        drop(_sched_guard);
                        retry
                            .after_commit_error(Operation::PublishBulkBuild, &mut attempts, e)
                            .await?;
                    }
                }
            }
        }
    }

    // If `max_mutations < FENCE_LANES + 2`, revoke lanes `1..FENCE_LANES` in bounded pre-batches
    // first so the final atomic `Loaded -> Validating` commit needs only `2` mutations (`lane 0` + `BuildProgress`).
    let inline_all_fences = inline_all || max_muts >= (FENCE_LANES as usize + 2);
    if !inline_all_fences {
        let fence_batch = max_muts.saturating_sub(1).clamp(1, FENCE_LANES as usize) as u32;
        for f_start in (1..FENCE_LANES).step_by(fence_batch as usize) {
            let f_end = (f_start + fence_batch).min(FENCE_LANES);
            let mut attempts = 0;
            loop {
                context.checkpoint()?;
                let _sched_guard = match &context.bulk_lease_lock {
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
                    bulk_worker::building_in_flight(&mut txn, index).await?;
                    super::bulk_scheduler::authorize(context, &mut txn, id).await?;
                    for lane in f_start..f_end {
                        txn.delete(LogicalKey::BuildUnit {
                            index: id,
                            stage: STAGE_FENCE,
                            unit: lane,
                        })
                        .await?;
                    }
                    Ok::<_, Error>(txn)
                }
                .await;
                let txn = match prepared {
                    Ok(txn) => txn,
                    Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                        drop(_sched_guard);
                        retry
                            .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                            .await?;
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                match txn.commit().await {
                    Ok(()) => break,
                    Err(e) => {
                        drop(_sched_guard);
                        retry
                            .after_commit_error(Operation::PublishBulkBuild, &mut attempts, e)
                            .await?;
                    }
                }
            }
        }
    }

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
            bulk_worker::building_in_flight(&mut txn, index).await?;
            super::bulk_scheduler::authorize(context, &mut txn, id).await?;
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
            match ws {
                Some(PersistentValue::BuildWorkspace(w)) => {
                    if let Some(kind) = w.failure {
                        return Err(Error::new(kind));
                    }
                    if w != *state {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                }
                None if progress.phase == BuildPhase::Validating => return Err(corrupt()),
                None => return Err(Error::new(ErrorKind::BulkBuildSuperseded)),
                _ => return Err(corrupt()),
            }
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
                    txn.rollback().await;
                    return Ok((None, progress));
                }
                BuildPhase::Loaded => {}
            }

            let lanes_to_del = if inline_all_fences { FENCE_LANES } else { 1 };
            for lane in 0..lanes_to_del {
                txn.delete(LogicalKey::BuildUnit {
                    index: id,
                    stage: STAGE_FENCE,
                    unit: lane,
                })
                .await?;
            }
            if inline_all {
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
        }
        .await;

        let (txn_opt, next) = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        let Some(txn) = txn_opt else {
            return Ok(next);
        };
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(next),
            Err(e) => {
                drop(_sched_guard);
                retry
                    .after_commit_error(Operation::PublishBulkBuild, &mut attempts, e)
                    .await?;
            }
        }
    }
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
    let mut worker_token = [0_u8; 32];
    getrandom::fill(&mut worker_token).map_err(|_| Error::new(ErrorKind::Other))?;

    let concurrency = state.options.parallelism.clamp(1, 16).min(slices as usize);
    if concurrency <= 1 {
        for s in 0..slices {
            context.checkpoint()?;
            validate_one_slice_task(
                context.backend(),
                context.options.clone(),
                context.bulk_authority,
                context.bulk_lease_lock.clone(),
                context.cpu_keep_alive(),
                index.clone(),
                state.clone(),
                progress.clone(),
                s,
                worker_token,
                budget.max_mutations,
                budget.max_mutation_bytes,
                retry,
            )
            .await?;
        }
    } else {
        let next_slice = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let mut tasks = tokio::task::JoinSet::<Result<()>>::new();
        for _ in 0..concurrency {
            let counter = Arc::clone(&next_slice);
            let backend = context.backend();
            let control = context.options.clone();
            let auth = context.bulk_authority;
            let sched_lock = context.bulk_lease_lock.clone();
            let admission = context.cpu_keep_alive();
            let manifest = index.clone();
            let ws = state.clone();
            let prog = progress.clone();
            tasks.spawn(async move {
                loop {
                    super::check_control(&control)?;
                    let s = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if s >= slices {
                        break;
                    }
                    validate_one_slice_task(
                        Arc::clone(&backend),
                        control.clone(),
                        auth,
                        sched_lock.clone(),
                        admission.clone(),
                        manifest.clone(),
                        ws.clone(),
                        prog.clone(),
                        s,
                        worker_token,
                        budget.max_mutations,
                        budget.max_mutation_bytes,
                        retry,
                    )
                    .await?;
                }
                Ok(())
            });
        }
        while let Some(res) = tasks.join_next().await {
            res.map_err(|e| Error::with_source(ErrorKind::Other, e))??;
        }
    }

    if state.options.distributed_workers <= 1 {
        return Ok(());
    }

    loop {
        context.checkpoint()?;
        let incomplete = find_incomplete_validation_slice(
            context.backend().as_ref(),
            index,
            progress.epoch,
            slices,
            retry,
        )
        .await?;
        let Some((s, ready_to_claim)) = incomplete else {
            return Ok(());
        };
        if ready_to_claim {
            validate_one_slice_task(
                context.backend(),
                context.options.clone(),
                context.bulk_authority,
                context.bulk_lease_lock.clone(),
                context.cpu_keep_alive(),
                index.clone(),
                state.clone(),
                progress.clone(),
                s,
                worker_token,
                budget.max_mutations,
                budget.max_mutation_bytes,
                retry,
            )
            .await?;
        } else {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
}

async fn find_incomplete_validation_slice<B: Backend>(
    backend: &B,
    expected_manifest: &IndexManifest,
    expected_epoch: u64,
    slices: u32,
    retry: RetryPolicy,
) -> Result<Option<(u32, bool)>> {
    let id = expected_manifest.logical_index_id();
    let range = LogicalRange::build_units(id, STAGE_VALIDATE);
    let mut attempts = 0;

    loop {
        let res = async {
            let now = now_ms()?;
            let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
            match rtxn.get(LogicalKey::Manifest(id)).await? {
                None => return Err(Error::new(ErrorKind::IndexNotFound)),
                Some(PersistentValue::IndexManifest(m))
                    if m.has_same_immutable_identity(expected_manifest) =>
                {
                    match m.lifecycle() {
                        IndexLifecycle::Building => {}
                        IndexLifecycle::Dropping => {
                            return Err(Error::new(ErrorKind::IndexDropping));
                        }
                        IndexLifecycle::Active => return Ok(None),
                    }
                }
                _ => return Err(corrupt()),
            }
            if let Some(PersistentValue::BuildWorkspace(w)) =
                rtxn.get(LogicalKey::BuildWorkspace(id)).await?
                && let Some(kind) = w.failure
            {
                return Err(Error::new(kind));
            }

            let mut cursor = None;
            let mut seen = 0_u32;
            let mut waiting_on_peer = None;
            loop {
                let page = rtxn
                    .scan(
                        &range,
                        cursor.as_ref(),
                        ScanLimits {
                            item_limit: 260,
                            byte_limit: 512 * 1024,
                        },
                    )
                    .await?;
                for item in page.items() {
                    let LogicalKey::BuildUnit { unit, .. } = *item.key() else {
                        return Err(corrupt());
                    };
                    let PersistentValue::BuildUnit(u) = item.value() else {
                        return Err(corrupt());
                    };
                    if u.epoch != expected_epoch {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    seen += 1;
                    if !u.complete {
                        if u.owner == [0; 32] || u.expires_ms <= now {
                            return Ok(Some((unit, true)));
                        }
                        waiting_on_peer = Some((unit, false));
                    }
                }
                cursor = page.into_next_cursor();
                if cursor.is_none() {
                    break;
                }
            }
            if seen != slices {
                return Err(corrupt());
            }
            Ok(waiting_on_peer)
        }
        .await;

        match res {
            Ok(v) => return Ok(v),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Opens a fresh `ReadLogicalTxn` bound to `expected_manifest` after verifying in the same
/// snapshot that `Manifest` and `BuildProgress` remain frozen in `(Building, Validating, expected_epoch)`.
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
    match &fetched[0] {
        None => return Err(Error::new(ErrorKind::IndexNotFound)),
        Some(PersistentValue::IndexManifest(m))
            if m.has_same_immutable_identity(expected_manifest) =>
        {
            match m.lifecycle() {
                IndexLifecycle::Building => {}
                IndexLifecycle::Dropping => return Err(Error::new(ErrorKind::IndexDropping)),
                IndexLifecycle::Active => {
                    return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                }
            }
        }
        _ => return Err(corrupt()),
    }
    if let Some(PersistentValue::BuildWorkspace(w)) = &fetched[2]
        && let Some(kind) = w.failure
    {
        return Err(Error::new(kind));
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

#[expect(
    clippy::too_many_arguments,
    reason = "self-contained slice task owns cloned handles for parallel JoinSet execution"
)]
async fn validate_one_slice_task<B: Backend>(
    backend: Arc<B>,
    control: OperationOptions,
    bulk_authority: Option<BulkAuthority>,
    sched_lease_lock: Option<Arc<tokio::sync::RwLock<()>>>,
    admission: Option<Arc<dyn std::any::Any + Send + Sync>>,
    index: IndexManifest,
    state: BuildWorkspace,
    progress: BuildProgress,
    slice: u32,
    worker_token: [u8; 32],
    page_items: usize,
    page_bytes: usize,
    retry: RetryPolicy,
) -> Result<()> {
    let id = index.logical_index_id();
    let ukey = LogicalKey::BuildUnit {
        index: id,
        stage: STAGE_VALIDATE,
        unit: slice,
    };
    let lease_lock = Arc::new(tokio::sync::Mutex::new(()));

    let initial_expires_ms = if state.options.distributed_workers > 1 {
        let mut attempts = 0;
        let claimed = loop {
            super::check_control(&control)?;
            let _sched_guard = match &sched_lease_lock {
                Some(l) => Some(l.read().await),
                None => None,
            };
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
                    ukey.clone(),
                ];
                if let Some(auth) = bulk_authority {
                    warm.push(auth.slot_key(id));
                }
                txn.warm_for_update(warm).await?;
                fence_validation(&mut txn, &index, &state, progress.epoch).await?;
                authorize_raw(&mut txn, id, bulk_authority).await?;
                let now = now_ms()?;
                let mut u = match txn.get_for_update(ukey.clone()).await? {
                    Some(PersistentValue::BuildUnit(u)) if u.epoch == progress.epoch => u,
                    Some(PersistentValue::BuildUnit(_)) => {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    _ => return Err(corrupt()),
                };
                if u.complete {
                    txn.rollback().await;
                    return Ok((None, None));
                }
                if u.owner == worker_token {
                    let exp = u.expires_ms;
                    txn.rollback().await;
                    return Ok((None, Some(exp)));
                }
                if u.owner != [0; 32] && u.expires_ms > now {
                    txn.rollback().await;
                    return Ok((None, None));
                }
                let exp = expiry(state.options.unit_lease)?;
                u.owner = worker_token;
                u.expires_ms = exp;
                txn.put(ukey.clone(), PersistentValue::BuildUnit(u)).await?;
                Ok((Some(txn), Some(exp)))
            }
            .await;

            let (txn_opt, exp_opt) = match prepared {
                Ok(v) => v,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    drop(_sched_guard);
                    if retry.would_exhaust(attempts) {
                        break None;
                    }
                    retry.wait(attempts).await;
                    attempts += 1;
                    continue;
                }
                Err(e) => return Err(e),
            };
            let Some(txn) = txn_opt else {
                break exp_opt;
            };
            match txn.commit().await {
                Ok(()) => break exp_opt,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    drop(_sched_guard);
                    if retry.would_exhaust(attempts) {
                        break None;
                    }
                    retry.wait(attempts).await;
                    attempts += 1;
                }
                Err(e) => {
                    drop(_sched_guard);
                    retry
                        .after_commit_error(Operation::PublishBulkBuild, &mut attempts, e)
                        .await?;
                }
            }
        };
        let Some(exp) = claimed else {
            return Ok(());
        };
        exp
    } else {
        let mut attempts = 0;
        let already_done = loop {
            super::check_control(&control)?;
            let res = async {
                let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
                rtxn.get(ukey.clone()).await
            }
            .await;
            match res {
                Ok(Some(PersistentValue::BuildUnit(u))) => {
                    break u.complete && u.epoch == progress.epoch;
                }
                Ok(_) => break false,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    retry
                        .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                        .await?;
                }
                Err(e) => return Err(e),
            }
        };
        if already_done {
            return Ok(());
        }
        0
    };

    // Scan `[slice_start, slice_end)` across rotated frozen snapshots via `scan_raw_validated`,
    // decoding every serving key and value fail-closed and computing `(kv_count, kv_lthash)`.
    let scan_fut = async {
        let range = slice_logical_range(&index, &progress.validation_split_keys, slice)?;
        let mut cursor: Option<LogicalScanCursor> = None;
        let mut slice_count = 0_u64;
        let mut slice_lt = LtHash::ZERO;

        let mut rtxn: Option<(ReadLogicalTxn<'_, B::ReadTxn<'_>>, Instant)> = None;

        loop {
            super::check_control(&control)?;
            let mut page_attempts = 0;
            let (raw_items, next_cursor) = loop {
                super::check_control(&control)?;
                let need_fresh = rtxn
                    .as_ref()
                    .is_none_or(|(_, opened)| opened.elapsed() >= READ_SNAPSHOT_ROTATE_INTERVAL);
                if need_fresh {
                    rtxn = None;
                    match open_frozen_validation_snapshot(backend.as_ref(), &index, progress.epoch)
                        .await
                    {
                        Ok(fresh) => rtxn = Some((fresh, Instant::now())),
                        Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                            retry
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
                            item_limit: page_items.max(1),
                            byte_limit: page_bytes.max(1),
                        },
                    )
                    .await
                {
                    Ok(page) => break page,
                    Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                        rtxn = None;
                        retry
                            .wait_or_exhaust(Operation::PublishBulkBuild, &mut page_attempts)
                            .await?;
                    }
                    Err(e) => return Err(e),
                }
            };

            if !raw_items.is_empty() {
                let manifest_c = index.clone();
                let (ctrl, _drop_guard) = drop_guarded_options(&control);
                let adm = admission.clone();
                let (c_delta, lt_delta) =
                    tokio::task::spawn_blocking(move || -> Result<(u64, LtHash)> {
                        let _adm = adm;
                        super::check_control(&ctrl)?;
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
                    .await
                    .map_err(|e| Error::with_source(ErrorKind::Other, e))??;

                slice_count = slice_count.checked_add(c_delta).ok_or_else(corrupt)?;
                slice_lt.add_assign(lt_delta);
            }

            match next_cursor {
                Some(nc) => cursor = Some(nc),
                None => break,
            }
        }
        drop(rtxn);
        Ok::<(u64, LtHash), Error>((slice_count, slice_lt))
    };

    let (slice_count, slice_lt) = if state.options.distributed_workers > 1 {
        let tick = (state.options.unit_lease / 3).max(Duration::from_millis(5));
        let lock_ref = Arc::clone(&lease_lock);
        let sched_lock_ref = sched_lease_lock.clone();
        let renewals = async {
            let mut confirmed = initial_expires_ms;
            let mut timer = tokio::time::interval(tick);
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            timer.tick().await;
            loop {
                timer.tick().await;
                let Ok(_guard) = lock_ref.try_lock() else {
                    continue;
                };
                let _sched_guard = match &sched_lock_ref {
                    Some(l) => Some(l.read().await),
                    None => None,
                };
                let step = async {
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
                        ukey.clone(),
                    ];
                    if let Some(auth) = bulk_authority {
                        warm.push(auth.slot_key(id));
                    }
                    txn.warm_for_update(warm).await?;
                    fence_validation(&mut txn, &index, &state, progress.epoch).await?;
                    authorize_raw(&mut txn, id, bulk_authority).await?;
                    match txn.get_for_update(ukey.clone()).await? {
                        Some(PersistentValue::BuildUnit(mut u))
                            if u.epoch == progress.epoch
                                && u.owner == worker_token
                                && !u.complete =>
                        {
                            let next_exp = u.expires_ms.max(expiry(state.options.unit_lease)?);
                            u.expires_ms = next_exp;
                            txn.put(ukey.clone(), PersistentValue::BuildUnit(u)).await?;
                            txn.commit().await?;
                            Ok(next_exp)
                        }
                        _ => Err(Error::new(ErrorKind::BulkBuildBusy)),
                    }
                }
                .await;
                match step {
                    Ok(next_exp) => confirmed = confirmed.max(next_exp),
                    Err(e)
                        if matches!(
                            e.kind(),
                            ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                        ) =>
                    {
                        if now_ms()? >= confirmed {
                            break Err(e);
                        }
                    }
                    Err(e) => break Err(e),
                }
            }
        };
        let outcome = tokio::select! {
            res = scan_fut => res,
            res = renewals => res,
        };
        match outcome {
            Ok(pair) => pair,
            Err(e) if e.kind() == ErrorKind::BulkBuildBusy => return Ok(()),
            Err(e) => return Err(e),
        }
    } else {
        scan_fut.await?
    };

    let mut attempts = 0;
    loop {
        super::check_control(&control)?;
        let _sched_guard = match &sched_lease_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
        let _guard = lease_lock.lock().await;
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
                ukey.clone(),
            ];
            if let Some(auth) = bulk_authority {
                warm.push(auth.slot_key(id));
            }
            txn.warm_for_update(warm).await?;
            fence_validation(&mut txn, &index, &state, progress.epoch).await?;
            authorize_raw(&mut txn, id, bulk_authority).await?;
            let mut u = match txn.get_for_update(ukey.clone()).await? {
                Some(PersistentValue::BuildUnit(u)) if u.epoch == progress.epoch => u,
                Some(PersistentValue::BuildUnit(_)) => {
                    return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                }
                _ => return Err(corrupt()),
            };
            if u.complete {
                txn.rollback().await;
                return Ok(None);
            }
            if state.options.distributed_workers > 1 && u.owner != worker_token {
                txn.rollback().await;
                return Ok(None);
            }
            u.owner = worker_token;
            u.complete = true;
            u.kv_count = slice_count;
            u.kv_lthash = slice_lt;
            txn.put(ukey.clone(), PersistentValue::BuildUnit(u)).await?;
            Ok(Some(txn))
        }
        .await;

        let txn_opt = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_guard);
                drop(_sched_guard);
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        let Some(txn) = txn_opt else {
            return Ok(());
        };
        match txn.commit().await {
            Ok(()) => return Ok(()),
            Err(e) => {
                drop(_guard);
                drop(_sched_guard);
                retry
                    .after_commit_error(Operation::PublishBulkBuild, &mut attempts, e)
                    .await?;
            }
        }
    }
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
            let cur_prog = fence_validation(&mut txn, index, state, progress.epoch).await?;
            super::bulk_scheduler::authorize(context, &mut txn, id).await?;
            match txn.get(LogicalKey::BuildDescriptor(id)).await? {
                Some(PersistentValue::BuildDescriptor(d)) if d == *descriptor => {}
                _ => return Err(corrupt()),
            }
            if cur_prog.total_records != descriptor.input().items()
                || cur_prog.validation_slices() != slices
            {
                return Err(corrupt());
            }

            // Protected `BuildWorkspace` + `BuildProgress` reads fence the validation epoch, and
            // completed `STAGE_VALIDATE` units are immutable; scan them cleanly via `LogicalRange`.
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
                    total_verified = total_verified.checked_add(u.kv_count).ok_or_else(corrupt)?;
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
            if total_verified != cur_prog.total_entries || total_lt != cur_prog.expected_kv_lthash {
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

        let (txn_opt, published) = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        let Some(txn) = txn_opt else {
            return Ok(published);
        };
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(published),
            Err(e) if e.kind() == ErrorKind::CommitOutcomeUnknown => {
                drop(_sched_guard);
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
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    }
}
