//! Namespace queue discovery, renewable claims, and collaborative worker slots.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio_util::sync::CancellationToken;

use crate::api::{
    BulkSchedulerOptions, BulkWorkerOptions, Error, ErrorKind, LogicalIndexId, OperationOptions,
    Result,
};
use crate::observe::labels::Operation;
use crate::storage::backend::{Backend, ReadOps, ScanLimits, WriteTxn};
use crate::storage::keys::LogicalKey;
use crate::storage::values::{
    BuildSchedule, BuildUnit, IndexLifecycle, IndexManifest, PersistentValue,
};
use crate::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

use super::{OperationContext, Runtime, bulk_worker, lifecycle::RetryPolicy};

/// Explicit write authority for scheduled jobs and unscheduled multi-worker peers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BulkAuthority {
    Scheduled { slot: u32, owner: [u8; 32] },
    UnscheduledPeer { slot: u32, owner: [u8; 32] },
}

impl BulkAuthority {
    #[must_use]
    pub(crate) const fn slot_key(self, id: LogicalIndexId) -> LogicalKey {
        let slot = match self {
            Self::Scheduled { slot, .. } | Self::UnscheduledPeer { slot, .. } => slot,
        };
        LogicalKey::BuildScheduleSlot { index: id, slot }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DriveOutcome {
    Published,
    AbortedOrDropped,
}

enum SlotClaim {
    AlreadyOwned { slot: u32, expires_ms: u64 },
    Claimed { slot: u32, expires_ms: u64 },
    Busy,
}

pub(crate) fn non_zero_token() -> Result<[u8; 32]> {
    let mut token = [0_u8; 32];
    getrandom::fill(&mut token).map_err(|e| Error::with_source(ErrorKind::Other, e))?;
    if token == [0_u8; 32] {
        token[0] = 1;
    }
    Ok(token)
}

pub(crate) fn now_ms() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .ok_or_else(|| Error::new(ErrorKind::Other))
}

pub(crate) fn expiry(delay: Duration) -> Result<u64> {
    now_ms()?
        .checked_add(u64::try_from(delay.as_millis()).map_err(|_| Error::invalid_argument())?)
        .ok_or_else(Error::invalid_argument)
}

/// Checks scheduler or collaborative-peer authority using explicit tokens.
pub(crate) async fn authorize_raw<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    id: LogicalIndexId,
    authority: Option<BulkAuthority>,
) -> Result<()> {
    match (
        authority,
        txn.get_for_update(LogicalKey::BuildSchedule(id)).await?,
    ) {
        (None, None) => Ok(()),
        (None, Some(PersistentValue::BuildSchedule(_))) => {
            Err(Error::new(ErrorKind::BulkBuildBusy))
        }
        (Some(BulkAuthority::UnscheduledPeer { slot, owner }), None) => {
            match txn
                .get_for_update(LogicalKey::BuildScheduleSlot { index: id, slot })
                .await?
            {
                Some(PersistentValue::BuildUnit(u)) if u.owner == owner => Ok(()),
                _ => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
            }
        }
        (Some(BulkAuthority::UnscheduledPeer { .. }), Some(PersistentValue::BuildSchedule(_))) => {
            Err(Error::new(ErrorKind::BulkBuildBusy))
        }
        (
            Some(BulkAuthority::Scheduled { slot: 0, owner }),
            Some(PersistentValue::BuildSchedule(s)),
        ) if s.options.distributed_workers <= 1 && s.owner == owner => Ok(()),
        (
            Some(BulkAuthority::Scheduled { slot, owner }),
            Some(PersistentValue::BuildSchedule(s)),
        ) if s.options.distributed_workers > 1
            && (slot as usize) < s.options.distributed_workers =>
        {
            match txn
                .get_for_update(LogicalKey::BuildScheduleSlot { index: id, slot })
                .await?
            {
                Some(PersistentValue::BuildUnit(u)) if u.owner == owner => Ok(()),
                _ => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
            }
        }
        (Some(BulkAuthority::Scheduled { .. }), None | Some(PersistentValue::BuildSchedule(_))) => {
            Err(Error::new(ErrorKind::BulkBuildSuperseded))
        }
        _ => Err(bulk_worker::corrupt()),
    }
}

pub(crate) async fn authorize<B: Backend, T: WriteTxn>(
    context: &OperationContext<B>,
    txn: &mut WriteLogicalTxn<'_, T>,
    id: LogicalIndexId,
) -> Result<()> {
    authorize_raw(txn, id, context.bulk_authority).await
}

/// Scans `0..slots` for an existing claim by `owner` first (for idempotent recovery after
/// `CommitOutcomeUnknown`), then claims the first free or expired slot.
async fn scan_and_claim_slot<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    id: LogicalIndexId,
    slots: u32,
    owner: [u8; 32],
    lease: Duration,
    require_all_idle: bool,
) -> Result<SlotClaim> {
    let now = now_ms()?;
    let mut first_free = None;
    for slot in 0..slots {
        let skey = LogicalKey::BuildScheduleSlot { index: id, slot };
        match txn.get_for_update(skey).await? {
            Some(PersistentValue::BuildUnit(u)) if u.owner == owner => {
                if !require_all_idle || slot == 0 {
                    return Ok(SlotClaim::AlreadyOwned {
                        slot,
                        expires_ms: u.expires_ms,
                    });
                }
            }
            None => {
                if first_free.is_none() {
                    first_free = Some(slot);
                }
            }
            Some(PersistentValue::BuildUnit(u)) if u.owner == [0; 32] || u.expires_ms <= now => {
                if first_free.is_none() {
                    first_free = Some(slot);
                }
            }
            Some(PersistentValue::BuildUnit(_)) => {
                if require_all_idle {
                    return Ok(SlotClaim::Busy);
                }
            }
            Some(_) => return Err(bulk_worker::corrupt()),
        }
    }
    let chosen = if require_all_idle {
        Some(0)
    } else {
        first_free
    };
    let Some(slot) = chosen else {
        return Ok(SlotClaim::Busy);
    };
    let exp = expiry(lease)?;
    let mut unit = BuildUnit::unclaimed(1);
    unit.owner = owner;
    unit.expires_ms = exp;
    txn.put(
        LogicalKey::BuildScheduleSlot { index: id, slot },
        PersistentValue::BuildUnit(unit),
    )
    .await?;
    Ok(SlotClaim::Claimed {
        slot,
        expires_ms: exp,
    })
}

/// Acquires an ephemeral renewable `BuildScheduleSlot` lease for unscheduled `distributed_workers > 1`
/// callers so concurrent peers never reclaim the shared workspace out from under one another.
pub(crate) async fn acquire_unscheduled_peer_slot<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    options: &BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<(u32, [u8; 32], u64)> {
    let owner = non_zero_token()?;
    let id = index.logical_index_id();
    let slots = options.distributed_workers.max(1) as u32;
    let mut attempts = 0;

    loop {
        let step = {
            context.checkpoint()?;
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                let mut warm = Vec::with_capacity(slots as usize + 2);
                warm.push(LogicalKey::Manifest(id));
                warm.push(LogicalKey::BuildSchedule(id));
                for slot in 0..slots {
                    warm.push(LogicalKey::BuildScheduleSlot { index: id, slot });
                }
                txn.warm_for_update(warm).await?;

                match bulk_worker::manifest(&mut txn, index).await?.lifecycle() {
                    IndexLifecycle::Building | IndexLifecycle::Active => {}
                    IndexLifecycle::Dropping => return Err(Error::new(ErrorKind::IndexDropping)),
                }
                if txn
                    .get_for_update(LogicalKey::BuildSchedule(id))
                    .await?
                    .is_some()
                {
                    return Err(Error::new(ErrorKind::BulkBuildBusy));
                }
                match scan_and_claim_slot(&mut txn, id, slots, owner, options.unit_lease, false)
                    .await?
                {
                    SlotClaim::AlreadyOwned { slot, expires_ms } => {
                        txn.rollback().await;
                        Ok((None, (slot, expires_ms)))
                    }
                    SlotClaim::Claimed { slot, expires_ms } => Ok((Some(txn), (slot, expires_ms))),
                    SlotClaim::Busy => Err(Error::new(ErrorKind::BulkBuildBusy)),
                }
            }
            .await;
            context.commit_prepared(prepared).await
        };

        if let Some((slot, exp)) = step
            .resolve(Operation::RunBulkBuild, retry, &mut attempts)
            .await?
        {
            context.bulk_authority = Some(BulkAuthority::UnscheduledPeer { slot, owner });
            return Ok((slot, owner, exp));
        }
    }
}

/// Runs periodic lease renewals for either an `UnscheduledPeer` slot or a `Scheduled` job/slot.
pub(crate) async fn run_authority_lease_renewals<B: Backend>(
    backend: Arc<B>,
    lease_lock: Arc<tokio::sync::RwLock<()>>,
    id: LogicalIndexId,
    authority: BulkAuthority,
    distributed_workers: usize,
    lease: Duration,
    mut confirmed_expires_ms: u64,
) -> Error {
    let tick = (lease / 3).max(Duration::from_millis(5));
    let mut timer = tokio::time::interval(tick);
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    timer.tick().await;
    loop {
        timer.tick().await;
        if let Err(err) = renew_authority_once(
            backend.as_ref(),
            &lease_lock,
            id,
            authority,
            distributed_workers,
            lease,
            &mut confirmed_expires_ms,
        )
        .await
        {
            return err;
        }
    }
}

async fn renew_authority_once<B: Backend>(
    backend: &B,
    lease_lock: &tokio::sync::RwLock<()>,
    id: LogicalIndexId,
    authority: BulkAuthority,
    distributed_workers: usize,
    lease: Duration,
    confirmed_expires_ms: &mut u64,
) -> Result<()> {
    let _guard = tokio::time::timeout(Duration::from_millis(5), lease_lock.write())
        .await
        .ok();
    let outcome = async {
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        match authority {
            BulkAuthority::Scheduled { owner, .. } if distributed_workers <= 1 => {
                let key = LogicalKey::BuildSchedule(id);
                match txn.get_for_update(key.clone()).await? {
                    Some(PersistentValue::BuildSchedule(mut s)) if s.owner == owner => {
                        let next_exp = s.expires_ms.max(expiry(lease)?);
                        s.expires_ms = next_exp;
                        txn.put(key, PersistentValue::BuildSchedule(s)).await?;
                        txn.commit().await?;
                        Ok(next_exp)
                    }
                    _ => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
                }
            }
            BulkAuthority::Scheduled { slot, owner } => {
                let skey = LogicalKey::BuildScheduleSlot { index: id, slot };
                txn.warm_for_update(vec![LogicalKey::BuildSchedule(id), skey.clone()])
                    .await?;
                if txn.get(LogicalKey::BuildSchedule(id)).await?.is_none() {
                    return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                }
                renew_slot_record(txn, skey, owner, lease).await
            }
            BulkAuthority::UnscheduledPeer { slot, owner } => {
                let skey = LogicalKey::BuildScheduleSlot { index: id, slot };
                txn.warm_for_update(vec![LogicalKey::BuildSchedule(id), skey.clone()])
                    .await?;
                if txn
                    .get_for_update(LogicalKey::BuildSchedule(id))
                    .await?
                    .is_some()
                {
                    return Err(Error::new(ErrorKind::BulkBuildBusy));
                }
                renew_slot_record(txn, skey, owner, lease).await
            }
        }
    }
    .await;

    match outcome {
        Ok(next_exp) => {
            *confirmed_expires_ms = (*confirmed_expires_ms).max(next_exp);
            Ok(())
        }
        Err(e)
            if matches!(
                e.kind(),
                ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
            ) =>
        {
            if now_ms()? < *confirmed_expires_ms {
                Ok(())
            } else {
                Err(e)
            }
        }
        Err(e) => Err(e),
    }
}

async fn renew_slot_record<T: WriteTxn>(
    mut txn: WriteLogicalTxn<'_, T>,
    skey: LogicalKey,
    owner: [u8; 32],
    lease: Duration,
) -> Result<u64> {
    match txn.get_for_update(skey.clone()).await? {
        Some(PersistentValue::BuildUnit(mut u)) if u.owner == owner => {
            let next_exp = u.expires_ms.max(expiry(lease)?);
            u.expires_ms = next_exp;
            txn.put(skey, PersistentValue::BuildUnit(u)).await?;
            txn.commit().await?;
            Ok(next_exp)
        }
        _ => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
    }
}

pub(crate) async fn release_unscheduled_peer_slot<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    slot: u32,
    owner: [u8; 32],
    retry: RetryPolicy,
) -> Result<()> {
    context.bulk_authority = None;
    context.bulk_lease_lock = None;
    let backend = context.backend();
    let mut attempts = 0;
    loop {
        if context.checkpoint().is_err() {
            return Ok(());
        }
        let step = bulk_worker::commit_raw_prepared(
            async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                let skey = LogicalKey::BuildScheduleSlot { index: id, slot };
                match txn.get_for_update(skey.clone()).await? {
                    Some(PersistentValue::BuildUnit(u)) if u.owner == owner => {
                        txn.delete(skey).await?;
                        Ok::<_, Error>((Some(txn), ()))
                    }
                    _ => {
                        txn.rollback().await;
                        Ok((None, ()))
                    }
                }
            }
            .await,
        )
        .await;

        match step {
            bulk_worker::CommitStep::Done(()) | bulk_worker::CommitStep::Fatal(_) => {
                return Ok(());
            }
            bulk_worker::CommitStep::PrepareAbort => {
                if retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await
                    .is_err()
                {
                    return Ok(());
                }
            }
            bulk_worker::CommitStep::CommitFailed(e)
                if matches!(
                    e.kind(),
                    ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                ) =>
            {
                if retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await
                    .is_err()
                {
                    return Ok(());
                }
            }
            bulk_worker::CommitStep::CommitFailed(_) => return Ok(()),
        }
    }
}

/// Scans `0..slots` in a write transaction, returning `(active_peer_count, expired_peer_keys)`.
async fn inspect_peer_slots<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    id: LogicalIndexId,
    slots: u32,
    self_slot: Option<(u32, [u8; 32])>,
    now: u64,
) -> Result<(usize, Vec<LogicalKey>)> {
    let keys: Vec<LogicalKey> = (0..slots)
        .map(|slot| LogicalKey::BuildScheduleSlot { index: id, slot })
        .collect();
    txn.warm_for_update(keys).await?;
    let mut active = 0_usize;
    let mut expired = Vec::new();
    for slot in 0..slots {
        let skey = LogicalKey::BuildScheduleSlot { index: id, slot };
        let Some(val) = txn.get_for_update(skey.clone()).await? else {
            continue;
        };
        let PersistentValue::BuildUnit(u) = val else {
            return Err(bulk_worker::corrupt());
        };
        if self_slot == Some((slot, u.owner)) {
            continue;
        }
        if u.owner != [0; 32] && u.expires_ms > now {
            active += 1;
        } else {
            expired.push(skey);
        }
    }
    Ok((active, expired))
}

/// Verifies no other scheduler or peer worker holds an unexpired lease on `id`, deleting
/// expired peer slot records inline within the caller's mutation budget.
pub(crate) async fn ensure_no_peer_scheduler_leases<B: Backend, T: WriteTxn>(
    context: &OperationContext<B>,
    txn: &mut WriteLogicalTxn<'_, T>,
    id: LogicalIndexId,
    distributed_workers: usize,
) -> Result<()> {
    let now = now_ms()?;
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
        let self_slot = context.bulk_authority.map(|a| match a {
            BulkAuthority::Scheduled { slot, owner }
            | BulkAuthority::UnscheduledPeer { slot, owner } => (slot, owner),
        });
        let (active, expired) = inspect_peer_slots(txn, id, slots, self_slot, now).await?;
        if active > 0 {
            return Err(Error::new(ErrorKind::BulkBuildBusy));
        }
        let max_inline_del = txn.admission_budget().max_mutations.saturating_sub(2);
        for skey in expired.into_iter().take(max_inline_del) {
            txn.delete(skey).await?;
        }
    }
    Ok(())
}

/// Retires orphaned `BuildSchedule` and `BuildScheduleSlot` entries once an index has been
/// published, aborted, or dropped and no peer worker holds an unexpired lease.
pub(crate) async fn retire_orphaned_schedule<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    retry: RetryPolicy,
) -> Result<()> {
    if matches!(
        context.bulk_authority,
        Some(BulkAuthority::Scheduled { .. })
    ) {
        return Ok(());
    }
    let slot_range = LogicalRange::build_schedule_slots(id);
    loop {
        context.checkpoint()?;
        let budget = context.backend().admission_budget();
        let max_del = budget.max_mutations.saturating_sub(2).clamp(1, 256);

        let mut attempts = 0;
        let more_slots = loop {
            let step = {
                context.checkpoint()?;
                let backend = context.backend();
                let prepared = async {
                    let mut txn = WriteLogicalTxn::bootstrap(
                        backend.begin_write().await?,
                        backend.hard_limits(),
                        budget,
                    );
                    if let Some(val) = txn.get_for_update(LogicalKey::Manifest(id)).await? {
                        match val {
                            PersistentValue::IndexManifest(m)
                                if m.logical_index_id() == id
                                    && m.lifecycle() != IndexLifecycle::Building => {}
                            _ => return Err(Error::invalid_argument()),
                        }
                    }
                    let now = now_ms()?;
                    let has_schedule =
                        match txn.get_for_update(LogicalKey::BuildSchedule(id)).await? {
                            None => false,
                            Some(PersistentValue::BuildSchedule(s)) => {
                                if s.options.distributed_workers <= 1
                                    && s.owner != [0; 32]
                                    && s.expires_ms > now
                                {
                                    return Err(Error::new(ErrorKind::BulkBuildBusy));
                                }
                                true
                            }
                            Some(_) => return Err(bulk_worker::corrupt()),
                        };

                    let page = txn
                        .scan(
                            &slot_range,
                            None,
                            ScanLimits {
                                item_limit: 260,
                                byte_limit: 256 * 1024,
                            },
                        )
                        .await?;
                    let mut to_delete = Vec::new();
                    let mut has_remaining_slots = page.next_cursor().is_some();
                    for item in page.items() {
                        let LogicalKey::BuildScheduleSlot { slot, .. } = *item.key() else {
                            return Err(bulk_worker::corrupt());
                        };
                        let PersistentValue::BuildUnit(u) = item.value() else {
                            return Err(bulk_worker::corrupt());
                        };
                        let is_self = matches!(
                            context.bulk_authority,
                            Some(BulkAuthority::UnscheduledPeer { slot: s, owner })
                                if s == slot && owner == u.owner
                        );
                        if !is_self {
                            if u.owner != [0; 32] && u.expires_ms > now {
                                return Err(Error::new(ErrorKind::BulkBuildBusy));
                            }
                            if to_delete.len() < max_del {
                                to_delete.push(item.key().clone());
                            } else {
                                has_remaining_slots = true;
                            }
                        }
                    }

                    if to_delete.is_empty() && !has_schedule {
                        txn.rollback().await;
                        return Ok((None, false));
                    }
                    let mut batch = txn.mutations();
                    for skey in to_delete {
                        batch.delete(skey)?;
                    }
                    if !has_remaining_slots && has_schedule {
                        batch.delete(LogicalKey::BuildSchedule(id))?;
                    }
                    txn.apply(batch).await?;
                    Ok((Some(txn), has_remaining_slots))
                }
                .await;
                context.commit_prepared(prepared).await
            };
            if let Some(has_more) = step
                .resolve(Operation::CleanupBulkBuild, retry, &mut attempts)
                .await?
            {
                break has_more;
            }
        };

        if !more_slots {
            return Ok(());
        }
    }
}

pub(crate) async fn enqueue<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    name: crate::api::IndexName,
    options: BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<()> {
    let options = bulk_worker::normalize_options(context, options).await?;
    let mut attempts = 0;
    loop {
        let step = {
            context.checkpoint()?;
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                bulk_worker::building(&mut txn, &index).await?;
                let id = index.logical_index_id();
                if let Some(PersistentValue::BuildWorkspace(w)) =
                    txn.get_for_update(LogicalKey::BuildWorkspace(id)).await?
                {
                    if w.options != options {
                        return Err(Error::invalid_argument());
                    }
                    if let Some(kind) = w.failure {
                        return Err(Error::new(kind));
                    }
                }
                let key = LogicalKey::BuildSchedule(id);
                match txn.get_for_update(key.clone()).await? {
                    Some(PersistentValue::BuildSchedule(s)) if s.options == options => {
                        txn.rollback().await;
                        return Ok((None, ()));
                    }
                    Some(_) => return Err(Error::invalid_argument()),
                    None => {}
                }
                if options.distributed_workers > 1 {
                    let (active, _) = inspect_peer_slots(
                        &mut txn,
                        id,
                        options.distributed_workers as u32,
                        None,
                        now_ms()?,
                    )
                    .await?;
                    if active > 0 {
                        return Err(Error::new(ErrorKind::BulkBuildBusy));
                    }
                }
                txn.put(
                    key,
                    PersistentValue::BuildSchedule(BuildSchedule {
                        name: name.clone(),
                        options: options.clone(),
                        owner: [0; 32],
                        expires_ms: 0,
                    }),
                )
                .await?;
                Ok((Some(txn), ()))
            }
            .await;
            context.commit_prepared(prepared).await
        };

        if step
            .resolve(Operation::ScheduleBulkBuild, retry, &mut attempts)
            .await?
            .is_some()
        {
            return Ok(());
        }
    }
}

async fn claim<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    lease: Duration,
    retry: RetryPolicy,
) -> Result<Option<(BuildSchedule, u32, [u8; 32], u64)>> {
    let owner = non_zero_token()?;
    let mut attempts = 0;
    loop {
        let step = {
            context.checkpoint()?;
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                let key = LogicalKey::BuildSchedule(id);
                let Some(PersistentValue::BuildSchedule(mut s)) =
                    txn.get_for_update(key.clone()).await?
                else {
                    txn.rollback().await;
                    return Ok((None, None));
                };
                let now = now_ms()?;
                if s.options.distributed_workers <= 1 {
                    if s.owner == owner {
                        let exp = s.expires_ms;
                        txn.rollback().await;
                        return Ok((None, Some((s, 0, exp))));
                    }
                    if s.owner == [0; 32] || s.expires_ms <= now {
                        let exp = expiry(lease)?;
                        s.owner = owner;
                        s.expires_ms = exp;
                        txn.put(key, PersistentValue::BuildSchedule(s.clone()))
                            .await?;
                        return Ok((Some(txn), Some((s, 0, exp))));
                    }
                    txn.rollback().await;
                    return Ok((None, None));
                }

                let is_building = match txn.get(LogicalKey::Manifest(id)).await? {
                    Some(PersistentValue::IndexManifest(m)) => {
                        m.lifecycle() == IndexLifecycle::Building
                    }
                    None => false,
                    _ => return Err(bulk_worker::corrupt()),
                };

                let slots = s.options.distributed_workers as u32;
                let slot_keys: Vec<LogicalKey> = (0..slots)
                    .map(|slot| LogicalKey::BuildScheduleSlot { index: id, slot })
                    .collect();
                txn.warm_for_update(slot_keys).await?;

                match scan_and_claim_slot(&mut txn, id, slots, owner, lease, !is_building).await? {
                    SlotClaim::AlreadyOwned { slot, expires_ms } => {
                        txn.rollback().await;
                        Ok((None, Some((s, slot, expires_ms))))
                    }
                    SlotClaim::Claimed { slot, expires_ms } => {
                        Ok((Some(txn), Some((s, slot, expires_ms))))
                    }
                    SlotClaim::Busy => {
                        txn.rollback().await;
                        Ok((None, None))
                    }
                }
            }
            .await;
            context.commit_prepared(prepared).await
        };

        if let Some(claimed) = step
            .resolve(Operation::ScheduleBulkBuild, retry, &mut attempts)
            .await?
        {
            return Ok(claimed.map(|(s, slot, exp)| {
                context.bulk_authority = Some(BulkAuthority::Scheduled { slot, owner });
                (s, slot, owner, exp)
            }));
        }
    }
}

async fn cleanup_scheduled_job<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    collab: bool,
    retry: RetryPolicy,
) -> Result<()> {
    match bulk_worker::cleanup(context, id, retry).await {
        Err(e) if collab && e.kind() == ErrorKind::BulkBuildBusy => Ok(()),
        other => other,
    }
}

async fn drive<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    schedule: BuildSchedule,
    retry: RetryPolicy,
) -> Result<DriveOutcome> {
    let collab = schedule.options.distributed_workers > 1;
    let backend = context.backend();
    let (manifest_opt, descriptor_opt) = bulk_worker::retry_read(
        &context.options,
        Operation::ScheduleBulkBuild,
        retry,
        || {
            let backend = Arc::clone(&backend);
            async move {
                let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
                let m = match txn.get(LogicalKey::Manifest(id)).await? {
                    None => return Ok((None, None)),
                    Some(PersistentValue::IndexManifest(m)) => m,
                    _ => return Err(bulk_worker::corrupt()),
                };
                if m.lifecycle() != IndexLifecycle::Building {
                    return Ok((Some(m), None));
                }
                let d = match txn.get(LogicalKey::BuildDescriptor(id)).await? {
                    Some(PersistentValue::BuildDescriptor(d)) => d,
                    _ => return Err(bulk_worker::corrupt()),
                };
                Ok((Some(m), Some(d)))
            }
        },
    )
    .await?;

    let Some(manifest) = manifest_opt else {
        cleanup_scheduled_job(context, id, collab, retry).await?;
        return Ok(DriveOutcome::AbortedOrDropped);
    };
    if manifest.lifecycle() == IndexLifecycle::Dropping {
        super::lifecycle::drop_index_bound(context, schedule.name, retry, Some(id)).await?;
        cleanup_scheduled_job(context, id, collab, retry).await?;
        return Ok(DriveOutcome::AbortedOrDropped);
    }
    if manifest.lifecycle() == IndexLifecycle::Active {
        cleanup_scheduled_job(context, id, collab, retry).await?;
        return Ok(DriveOutcome::Published);
    }
    let descriptor = descriptor_opt.ok_or_else(bulk_worker::corrupt)?;
    match bulk_worker::complete(
        context,
        manifest,
        descriptor,
        schedule.options.clone(),
        retry,
    )
    .await
    {
        Ok(_) => Ok(DriveOutcome::Published),
        Err(e)
            if matches!(
                e.kind(),
                ErrorKind::IndexDropping | ErrorKind::IndexNotFound
            ) =>
        {
            if e.kind() == ErrorKind::IndexDropping {
                super::lifecycle::drop_index_bound(context, schedule.name, retry, Some(id)).await?;
            }
            cleanup_scheduled_job(context, id, collab, retry).await?;
            Ok(DriveOutcome::AbortedOrDropped)
        }
        Err(e) => Err(e),
    }
}

struct FinishState {
    failed: bool,
    needs_final_cleanup: bool,
}

async fn finish<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    outcome: Option<DriveOutcome>,
    delay: Duration,
    retry: RetryPolicy,
) -> Result<FinishState> {
    let mut attempts = 0;
    loop {
        let step = {
            context.checkpoint()?;
            let backend = context.backend();
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    backend.begin_write().await?,
                    backend.hard_limits(),
                    backend.admission_budget(),
                );
                let key = LogicalKey::BuildSchedule(id);
                let ws_opt = txn.get(LogicalKey::BuildWorkspace(id)).await?;
                let failed = matches!(
                    &ws_opt,
                    Some(PersistentValue::BuildWorkspace(w)) if w.failure.is_some()
                );
                let ws_reclaimed = ws_opt.is_none();
                let Some(PersistentValue::BuildSchedule(mut s)) =
                    txn.get_for_update(key.clone()).await?
                else {
                    txn.rollback().await;
                    return Ok((
                        None,
                        FinishState {
                            failed,
                            needs_final_cleanup: false,
                        },
                    ));
                };

                let collab = s.options.distributed_workers > 1;
                let (slot, owner) = match context.bulk_authority {
                    Some(BulkAuthority::Scheduled { slot, owner }) => (slot, owner),
                    _ => return Err(Error::new(ErrorKind::BulkBuildSuperseded)),
                };
                if !collab {
                    if slot != 0 || s.owner != owner {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                } else {
                    let slots = s.options.distributed_workers as u32;
                    let slot_keys: Vec<LogicalKey> = (0..slots)
                        .map(|idx| LogicalKey::BuildScheduleSlot {
                            index: id,
                            slot: idx,
                        })
                        .collect();
                    txn.warm_for_update(slot_keys).await?;
                    let skey = LogicalKey::BuildScheduleSlot { index: id, slot };
                    match txn.get_for_update(skey).await? {
                        Some(PersistentValue::BuildUnit(u)) if u.owner == owner => {}
                        _ => return Err(Error::new(ErrorKind::BulkBuildSuperseded)),
                    }
                }

                let terminal_done = outcome.is_some();
                if !collab {
                    if (terminal_done && ws_reclaimed) || failed {
                        txn.delete(key).await?;
                    } else {
                        s.owner = [0; 32];
                        s.expires_ms = expiry(delay)?;
                        txn.put(key, PersistentValue::BuildSchedule(s)).await?;
                    }
                    return Ok((
                        Some(txn),
                        FinishState {
                            failed,
                            needs_final_cleanup: false,
                        },
                    ));
                }

                let (active_peer_slots, expired_peer_slots) = inspect_peer_slots(
                    &mut txn,
                    id,
                    s.options.distributed_workers as u32,
                    Some((slot, owner)),
                    now_ms()?,
                )
                .await?;

                let mut needs_final_cleanup = false;
                if terminal_done || failed {
                    txn.delete(LogicalKey::BuildScheduleSlot { index: id, slot })
                        .await?;
                    if active_peer_slots == 0 {
                        for okey in expired_peer_slots {
                            txn.delete(okey).await?;
                        }
                        if ws_reclaimed || failed {
                            txn.delete(key).await?;
                        } else {
                            needs_final_cleanup = true;
                        }
                    }
                } else {
                    let mut u = BuildUnit::unclaimed(1);
                    u.expires_ms = expiry(delay)?;
                    txn.put(
                        LogicalKey::BuildScheduleSlot { index: id, slot },
                        PersistentValue::BuildUnit(u),
                    )
                    .await?;
                }
                Ok((
                    Some(txn),
                    FinishState {
                        failed,
                        needs_final_cleanup,
                    },
                ))
            }
            .await;
            context.commit_prepared(prepared).await
        };

        if let Some(state) = step
            .resolve(Operation::ScheduleBulkBuild, retry, &mut attempts)
            .await?
        {
            return Ok(state);
        }
    }
}

async fn execute<B: Backend>(
    mut context: OperationContext<B>,
    id: LogicalIndexId,
    settings: BulkSchedulerOptions,
    retry: RetryPolicy,
) -> Result<()> {
    let Some((s, slot, owner, initial_expires_ms)) =
        claim(&mut context, id, settings.lease_duration, retry).await?
    else {
        return Ok(());
    };
    let collab = s.options.distributed_workers > 1;
    let dist_workers = s.options.distributed_workers;
    let authority = BulkAuthority::Scheduled { slot, owner };
    let lease_lock = Arc::new(tokio::sync::RwLock::new(()));
    context.bulk_lease_lock = Some(Arc::clone(&lease_lock));
    let backend = context.backend();

    let (result, work_finished) = tokio::select! {
        res = drive(&mut context, id, s.clone(), retry) => (res, true),
        err = run_authority_lease_renewals(
            backend,
            lease_lock,
            id,
            authority,
            dist_workers,
            settings.lease_duration,
            initial_expires_ms,
        ) => (Err(err), false),
    };
    context.bulk_lease_lock = None;

    let drive_outcome = result.as_ref().ok().copied();
    let finish_state = match finish(
        &mut context,
        id,
        drive_outcome,
        settings.poll_interval,
        retry,
    )
    .await
    {
        Ok(state) => state,
        Err(e) if collab && e.kind() == ErrorKind::BulkBuildSuperseded && result.is_ok() => {
            FinishState {
                failed: false,
                needs_final_cleanup: false,
            }
        }
        Err(e) => return Err(e),
    };

    if finish_state.needs_final_cleanup {
        context.bulk_authority = None;
        let _ = bulk_worker::cleanup(&mut context, id, retry).await;
    }

    if work_finished
        && (finish_state.failed || drive_outcome == Some(DriveOutcome::AbortedOrDropped))
    {
        Ok(())
    } else {
        result.map(|_| ())
    }
}

async fn count_active_slots<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    id: LogicalIndexId,
    now: u64,
) -> Result<usize> {
    let range = LogicalRange::build_schedule_slots(id);
    let mut cursor = None;
    let mut active = 0_usize;
    loop {
        let page = txn
            .scan(
                &range,
                cursor.as_ref(),
                ScanLimits {
                    item_limit: 260,
                    byte_limit: 256 * 1024,
                },
            )
            .await?;
        for s_item in page.items() {
            let LogicalKey::BuildScheduleSlot { index, .. } = *s_item.key() else {
                return Err(bulk_worker::corrupt());
            };
            if index != id {
                return Err(bulk_worker::corrupt());
            }
            let PersistentValue::BuildUnit(u) = s_item.value() else {
                return Err(bulk_worker::corrupt());
            };
            if u.owner != [0; 32] && u.expires_ms > now {
                active += 1;
            }
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    Ok(active)
}

async fn discover<B: Backend>(
    context: OperationContext<B>,
    after: Option<LogicalIndexId>,
    limit: usize,
    active_local: BTreeSet<LogicalIndexId>,
) -> Result<(Vec<LogicalIndexId>, Option<LogicalIndexId>)> {
    context.checkpoint()?;
    let backend = context.backend();
    let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
    let mut remaining = 256;
    let mut page_limit = limit.max(1);
    let mut cursor = after;
    let now = now_ms()?;
    let mut ids = Vec::new();
    while remaining != 0 {
        let range = LogicalRange::build_schedules(cursor);
        let page = txn
            .scan(
                &range,
                None,
                ScanLimits {
                    item_limit: page_limit.min(remaining),
                    byte_limit: 1024 * 1024,
                },
            )
            .await?;
        let has_more = page.next_cursor().is_some();
        let items = page.into_items();
        let page_len = items.len();
        for item in items {
            let LogicalKey::BuildSchedule(id) = *item.key() else {
                return Err(bulk_worker::corrupt());
            };
            let PersistentValue::BuildSchedule(schedule) = item.into_value() else {
                return Err(bulk_worker::corrupt());
            };
            cursor = Some(id);
            if active_local.contains(&id) {
                continue;
            }
            let eligible = if schedule.options.distributed_workers <= 1 {
                schedule.owner == [0; 32] || schedule.expires_ms <= now
            } else {
                let active_slots = count_active_slots(&mut txn, id, now).await?;
                let is_building = match txn.get(LogicalKey::Manifest(id)).await? {
                    Some(PersistentValue::IndexManifest(m)) => {
                        m.lifecycle() == IndexLifecycle::Building
                    }
                    None => false,
                    _ => return Err(bulk_worker::corrupt()),
                };
                if is_building {
                    active_slots < schedule.options.distributed_workers
                } else {
                    active_slots == 0
                }
            };
            if eligible {
                ids.push(id);
                if ids.len() == limit {
                    return Ok((ids, cursor));
                }
            }
        }
        if !has_more || page_len == 0 {
            return Ok((ids, None));
        }
        remaining = remaining.saturating_sub(page_len);
        page_limit = (page_limit * 2).min(remaining.max(1));
        context.checkpoint()?;
    }
    Ok((ids, cursor))
}

pub(crate) async fn run<B: Backend>(
    runtime: &Runtime<B>,
    settings: BulkSchedulerOptions,
    control: OperationOptions,
) -> Result<()> {
    settings.validate()?;
    super::check_control(&control)?;
    let stop = CancellationToken::new();
    let _cancel_on_drop = stop.clone().drop_guard();
    let mut jobs = tokio::task::JoinSet::<Result<()>>::new();
    let mut active_tasks = BTreeMap::<tokio::task::Id, LogicalIndexId>::new();
    let mut active_ids = BTreeSet::<LogicalIndexId>::new();
    let mut cursor = None;
    let mut timer = tokio::time::interval(settings.poll_interval);
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let retry = RetryPolicy::from_config(runtime.config());
    let terminal = loop {
        tokio::select! {
            _ = runtime.handle.inner.maintenance_cancel.cancelled() => break Ok(()),
            _ = async { if let Some(token) = control.cancellation() { token.cancelled().await } else { std::future::pending().await } } => break Err(Error::new(ErrorKind::Cancelled)),
            _ = async { if let Some(deadline) = control.deadline() { tokio::time::sleep_until(deadline.into()).await } else { std::future::pending().await } } => break Err(Error::new(ErrorKind::DeadlineExceeded)),
            result = jobs.join_next_with_id(), if !jobs.is_empty() => {
                match result {
                    Some(Err(e)) => break Err(Error::with_source(ErrorKind::Other, e)),
                    Some(Ok((task_id, res))) => {
                        if let Some(done_id) = active_tasks.remove(&task_id) {
                            active_ids.remove(&done_id);
                        }
                        match res {
                            Err(e) if e.kind() == ErrorKind::RuntimeClosed && runtime.handle.inner.maintenance_cancel.is_cancelled() => break Ok(()),
                            Err(e) if !matches!(
                                e.kind(),
                                ErrorKind::RetryableAbort
                                    | ErrorKind::ContentionExhausted
                                    | ErrorKind::CommitOutcomeUnknown
                                    | ErrorKind::LimitExceeded
                                    | ErrorKind::BulkBuildSuperseded
                                    | ErrorKind::BulkBuildBusy
                                    | ErrorKind::IndexDropping
                                    | ErrorKind::IndexNotFound
                            ) => break Err(e),
                            _ => {}
                        }
                    }
                    None => {}
                }
            }
            _ = timer.tick(), if jobs.len() < settings.max_jobs => {
                let count = settings.max_jobs - jobs.len();
                let running_snapshot = active_ids.clone();
                let page = runtime.run_foreground(Operation::ScheduleBulkBuild, None, control.clone(), move |context| discover(context, cursor, count, running_snapshot)).await;
                let (ids, next) = match page {
                    Ok(page) => page,
                    Err(e) if matches!(e.kind(), ErrorKind::RetryableAbort | ErrorKind::LimitExceeded) => continue,
                    Err(e) if e.kind() == ErrorKind::RuntimeClosed && runtime.handle.inner.maintenance_cancel.is_cancelled() => break Ok(()),
                    Err(e) => break Err(e),
                };
                cursor = next;
                for id in ids {
                    if !active_ids.insert(id) {
                        continue;
                    }
                    let inner = Arc::clone(&runtime.handle.inner);
                    let options = OperationOptions::default().with_cancellation(stop.clone());
                    let ab = jobs.spawn(async move {
                        inner.run_foreground(Operation::ScheduleBulkBuild, Some(id), options, move |context| execute(context, id, settings, retry)).await
                    });
                    active_tasks.insert(ab.id(), id);
                }
            }
        }
    };
    stop.cancel();
    while jobs.join_next().await.is_some() {}
    terminal
}
