//! Namespace queue discovery, renewable claims, and identity-fenced execution.
use super::{OperationContext, Runtime, bulk_publish, bulk_worker, lifecycle::RetryPolicy};
use crate::api::{
    BulkSchedulerOptions, BulkWorkerOptions, Error, ErrorKind, LogicalIndexId, OperationOptions,
    Result,
};
use crate::observe::labels::Operation;
use crate::storage::backend::{Backend, ReadOps, ScanLimits, WriteTxn};
use crate::storage::keys::{self, KeyRange, LogicalKey};
use crate::storage::values::{
    BuildPhase, BuildSchedule, IndexLifecycle, IndexManifest, PersistentValue, ValueCodec,
};
use crate::storage::{ReadLogicalTxn, WriteLogicalTxn};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;

fn now_ms() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .ok_or_else(|| Error::new(ErrorKind::Other))
}
fn expiry(delay: Duration) -> Result<u64> {
    now_ms()?
        .checked_add(u64::try_from(delay.as_millis()).map_err(|_| Error::invalid_argument())?)
        .ok_or_else(Error::invalid_argument)
}
/// Every build mutation reads this token, including manual operations. Expiry
/// enables takeover; only a committed token change revokes write authority.
pub(crate) async fn authorize<B: Backend, T: WriteTxn>(
    context: &OperationContext<B>,
    txn: &mut WriteLogicalTxn<'_, T>,
    id: LogicalIndexId,
) -> Result<()> {
    match (
        context.bulk_authority,
        txn.get_for_update(LogicalKey::BuildSchedule(id)).await?,
    ) {
        (None, None) => Ok(()),
        (Some(owner), Some(PersistentValue::BuildSchedule(s))) if s.owner == owner => Ok(()),
        (None, Some(PersistentValue::BuildSchedule(_))) => {
            Err(Error::new(ErrorKind::BulkBuildBusy))
        }
        (Some(_), _) => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
        _ => Err(bulk_worker::corrupt()),
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
        context.checkpoint()?;
        let backend = context.backend();
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
            Some(PersistentValue::BuildSchedule(s)) if s.options == options => return Ok(()),
            Some(_) => return Err(Error::invalid_argument()),
            None => {}
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
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(()),
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                ) =>
            {
                retry
                    .wait_or_exhaust(Operation::ScheduleBulkBuild, &mut attempts)
                    .await?
            }
            Err(e) => return Err(e),
        }
    }
}

async fn claim<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    lease: Duration,
    retry: RetryPolicy,
) -> Result<Option<BuildSchedule>> {
    let mut owner = [0; 32];
    getrandom::fill(&mut owner).map_err(|e| Error::with_source(ErrorKind::Other, e))?;
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        let key = LogicalKey::BuildSchedule(id);
        let Some(PersistentValue::BuildSchedule(mut s)) = txn.get_for_update(key.clone()).await?
        else {
            return Ok(None);
        };
        // Resolve an unknown claim only against this exact random token.
        if s.owner == owner {
            context.bulk_authority = Some(owner);
            return Ok(Some(s));
        }
        if s.expires_ms > now_ms()? {
            return Ok(None);
        }
        s.owner = owner;
        s.expires_ms = expiry(lease)?;
        txn.put(key, PersistentValue::BuildSchedule(s.clone()))
            .await?;
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => {
                context.bulk_authority = Some(owner);
                return Ok(Some(s));
            }
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                ) =>
            {
                retry
                    .wait_or_exhaust(Operation::ScheduleBulkBuild, &mut attempts)
                    .await?
            }
            Err(e) => return Err(e),
        }
    }
}

// Heartbeats share the admitted build operation. They must not request a second
// foreground permit, which would deadlock a Runtime configured with capacity one.
async fn renew<B: Backend>(
    backend: &B,
    id: LogicalIndexId,
    owner: [u8; 32],
    lease: Duration,
) -> Result<()> {
    let mut txn = WriteLogicalTxn::bootstrap(
        backend.begin_write().await?,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    let key = LogicalKey::BuildSchedule(id);
    match txn.get_for_update(key.clone()).await? {
        Some(PersistentValue::BuildSchedule(mut s)) if s.owner == owner => {
            s.expires_ms = expiry(lease)?;
            txn.put(key, PersistentValue::BuildSchedule(s)).await?;
            txn.commit().await
        }
        _ => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
    }
}

async fn drive<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    schedule: BuildSchedule,
    retry: RetryPolicy,
) -> Result<()> {
    let backend = context.backend();
    let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
    let manifest = match txn.get(LogicalKey::Manifest(id)).await? {
        None => {
            drop(txn);
            return bulk_worker::cleanup(context, id, retry).await;
        }
        Some(PersistentValue::IndexManifest(m)) => m,
        _ => return Err(bulk_worker::corrupt()),
    };
    if manifest.lifecycle() == IndexLifecycle::Dropping {
        drop(txn);
        super::lifecycle::drop_index_bound(context, schedule.name, retry, Some(id)).await?;
        return bulk_worker::cleanup(context, id, retry).await;
    }
    if manifest.lifecycle() == IndexLifecycle::Active {
        drop(txn);
        return bulk_worker::cleanup(context, id, retry).await;
    }
    let descriptor = match txn.get(LogicalKey::BuildDescriptor(id)).await? {
        Some(PersistentValue::BuildDescriptor(d)) => d,
        _ => return Err(bulk_worker::corrupt()),
    };
    if let Some(PersistentValue::BuildWorkspace(w)) =
        txn.get(LogicalKey::BuildWorkspace(id)).await?
        && let Some(kind) = w.failure
    {
        return Err(Error::new(kind));
    }
    let sealed = matches!(txn.get(LogicalKey::BuildProgress(id)).await?, Some(PersistentValue::BuildProgress(p)) if matches!(p.phase, BuildPhase::Validating { .. } | BuildPhase::Validated));
    drop(txn);
    if !sealed {
        bulk_worker::run(
            context,
            manifest.clone(),
            descriptor.clone(),
            schedule.options,
            None,
            retry,
        )
        .await?;
    }
    bulk_publish::publish(context, manifest, descriptor, retry).await?;
    bulk_worker::cleanup(context, id, retry).await
}

/// Releases the schedule and reports whether the worker persisted a terminal failure.
async fn finish<B: Backend>(
    context: &mut OperationContext<B>,
    id: LogicalIndexId,
    success: bool,
    delay: Duration,
    retry: RetryPolicy,
) -> Result<bool> {
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        let key = LogicalKey::BuildSchedule(id);
        let failed = matches!(txn.get(LogicalKey::BuildWorkspace(id)).await?, Some(PersistentValue::BuildWorkspace(w)) if w.failure.is_some());
        let Some(PersistentValue::BuildSchedule(mut s)) = txn.get_for_update(key.clone()).await?
        else {
            return Ok(failed);
        };
        if Some(s.owner) != context.bulk_authority {
            return Err(Error::new(ErrorKind::BulkBuildSuperseded));
        }
        if success || failed {
            txn.delete(key).await?;
        } else {
            s.owner = [0; 32];
            s.expires_ms = expiry(delay)?;
            txn.put(key, PersistentValue::BuildSchedule(s)).await?;
        }
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(failed),
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                ) =>
            {
                retry
                    .wait_or_exhaust(Operation::ScheduleBulkBuild, &mut attempts)
                    .await?
            }
            Err(e) => return Err(e),
        }
    }
}

async fn execute<B: Backend>(
    mut context: OperationContext<B>,
    id: LogicalIndexId,
    settings: BulkSchedulerOptions,
    retry: RetryPolicy,
) -> Result<()> {
    let Some(s) = claim(&mut context, id, settings.lease_duration, retry).await? else {
        return Ok(());
    };
    let backend = context.backend();
    let (result, work_finished) = {
        let work = drive(&mut context, id, s.clone(), retry);
        let renewals = async {
            let mut heartbeat = tokio::time::interval(settings.lease_duration / 3);
            heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            heartbeat.tick().await;
            loop {
                heartbeat.tick().await;
                if let Err(error) =
                    renew(backend.as_ref(), id, s.owner, settings.lease_duration).await
                {
                    break Err(error);
                }
            }
        };
        // A pending renewal must not stop work that holds a backend transaction slot.
        tokio::select! {
            result = work => (result, true),
            result = renewals => (result, false),
        }
    };
    let failed = finish(
        &mut context,
        id,
        result.is_ok(),
        settings.poll_interval,
        retry,
    )
    .await?;
    // Only persisted worker failures are handled here. The scheduler classifies
    // transient errors; coordination failures must reach its caller.
    if work_finished && failed {
        Ok(())
    } else {
        result
    }
}

async fn discover<B: Backend>(
    context: OperationContext<B>,
    after: Option<LogicalIndexId>,
    limit: usize,
) -> Result<(Vec<LogicalIndexId>, Option<LogicalIndexId>)> {
    context.checkpoint()?;
    let backend = context.backend();
    let mut txn = backend.begin_read().await?;
    let start = after.map_or_else(
        || vec![0, 3],
        |id| {
            let mut key = keys::build_schedule_key(id);
            key.push(0);
            key
        },
    );
    let page = txn
        .scan(
            &KeyRange::new(start, vec![0, 4]),
            ScanLimits {
                item_limit: limit,
                byte_limit: 1024 * 1024,
            },
        )
        .await?;
    let now = now_ms()?;
    let mut ids = Vec::new();
    let mut last_scanned = None;
    for item in page.items() {
        let key = keys::decode_key(&[], item.key())?;
        let LogicalKey::BuildSchedule(id) = key else {
            return Err(bulk_worker::corrupt());
        };
        let PersistentValue::BuildSchedule(schedule) =
            ValueCodec::bootstrap().decode(&key, item.value().clone())?
        else {
            return Err(bulk_worker::corrupt());
        };
        last_scanned = Some(id);
        // This snapshot only avoids futile attempts; claim still establishes
        // authority transactionally before any work is started.
        if schedule.expires_ms <= now {
            ids.push(id);
        }
    }
    let next = if page.is_terminal() {
        None
    } else {
        last_scanned
    };
    Ok((ids, next))
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
    let mut cursor = None;
    let mut timer = tokio::time::interval(settings.poll_interval);
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let retry = RetryPolicy::from_config(runtime.config());
    let terminal = loop {
        tokio::select! {
            _ = runtime.handle.inner.maintenance_cancel.cancelled() => break Ok(()),
            _ = async { if let Some(token) = control.cancellation() { token.cancelled().await } else { std::future::pending().await } } => break Err(Error::new(ErrorKind::Cancelled)),
            _ = async { if let Some(deadline) = control.deadline() { tokio::time::sleep_until(deadline.into()).await } else { std::future::pending().await } } => break Err(Error::new(ErrorKind::DeadlineExceeded)),
            result = jobs.join_next(), if !jobs.is_empty() => {
                match result {
                    Some(Err(e)) => break Err(Error::with_source(ErrorKind::Other,e)),
                    Some(Ok(Err(e))) if e.kind() == ErrorKind::RuntimeClosed && runtime.handle.inner.maintenance_cancel.is_cancelled() => break Ok(()),
                    Some(Ok(Err(e))) if !matches!(e.kind(), ErrorKind::RetryableAbort | ErrorKind::ContentionExhausted | ErrorKind::CommitOutcomeUnknown | ErrorKind::LimitExceeded | ErrorKind::BulkBuildSuperseded) => break Err(e),
                    _ => {}
                }
            }
            _ = timer.tick(), if jobs.len() < settings.max_jobs => {
                let count = settings.max_jobs - jobs.len();
                let page = runtime.run_foreground(Operation::ScheduleBulkBuild, None, control.clone(), move |context| discover(context,cursor,count)).await;
                let (ids,next) = match page {
                    Ok(page) => page,
                    Err(e) if matches!(e.kind(), ErrorKind::RetryableAbort | ErrorKind::LimitExceeded) => continue,
                    Err(e) if e.kind() == ErrorKind::RuntimeClosed && runtime.handle.inner.maintenance_cancel.is_cancelled() => break Ok(()),
                    Err(e) => break Err(e),
                };
                cursor = next;
                for id in ids {
                    let inner = Arc::clone(&runtime.handle.inner);
                    let options = OperationOptions::default().with_cancellation(stop.clone());
                    jobs.spawn(async move { inner.run_foreground(Operation::ScheduleBulkBuild, Some(id), options, move |context| execute(context,id,settings,retry)).await });
                }
            }
        }
    };
    stop.cancel();
    while jobs.join_next().await.is_some() {}
    terminal
}
