//! Durable, attempt-fenced preparation and namespace-owned artifact reclamation.
use super::{OperationContext, lifecycle::RetryPolicy};
use crate::api::{BulkWorkerOptions, Error, ErrorKind, Result};
use crate::bulk::{ForestArtifact, ForestOptions, InputSnapshot, ServingArtifact, ServingOptions};
use crate::observe::labels::Operation;
use crate::storage::backend::Backend;
use crate::storage::keys::LogicalKey;
use crate::storage::values::{
    BuildDescriptor, BuildWorkspace, IndexLifecycle, IndexManifest, PersistentValue,
    PreparedArtifact,
};
use crate::storage::{ReadLogicalTxn, WriteLogicalTxn};
use std::{
    fs::{self, File, OpenOptions},
    path::Path,
    sync::Arc,
};

/// A shared root lock is retained by native work, even when its async waiter is
/// cancelled. Exclusive cleanup therefore cannot race a detached/stale writer.
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
pub(crate) async fn blocking<B: Backend, T: Send + 'static>(
    context: &OperationContext<B>,
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    context.checkpoint()?;
    let admission = context
        .cpu_admission
        .clone()
        .expect("bulk worker admission");
    let options = context.options.clone();
    tokio::task::spawn_blocking(move || {
        let _admission = admission;
        super::check_control(&options)?;
        work()
    })
    .await
    .map_err(|e| Error::with_source(ErrorKind::Other, e))?
}
pub(crate) async fn manifest<T: crate::storage::backend::WriteTxn>(
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
pub(crate) async fn building<T: crate::storage::backend::WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    expected: &IndexManifest,
) -> Result<()> {
    match manifest(txn, expected).await?.lifecycle() {
        IndexLifecycle::Building => Ok(()),
        IndexLifecycle::Dropping => Err(Error::new(ErrorKind::IndexDropping)),
        IndexLifecycle::Active => Err(Error::invalid_argument()),
    }
}
pub(crate) async fn workspace<B: Backend>(
    context: &OperationContext<B>,
    index: &IndexManifest,
) -> Result<BuildWorkspace> {
    let backend = context.backend();
    let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
    match txn
        .get(LogicalKey::BuildWorkspace(index.logical_index_id()))
        .await?
    {
        Some(PersistentValue::BuildWorkspace(w)) => Ok(w),
        None => Err(Error::invalid_argument()),
        _ => Err(corrupt()),
    }
}

pub(crate) async fn run<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    descriptor: BuildDescriptor,
    mut options: BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<()> {
    options.validate()?;
    let root = options.workspace.clone();
    options.workspace = blocking(context, move || fs::canonicalize(root).map_err(io)).await?;
    options.validate()?;
    // Lock before registration or any directory creation, so cleanup can remove
    // the ledger only when every potential creator has left this critical region.
    let root = options.workspace.clone();
    let lock = Arc::new(blocking(context, move || lock_root(&root, false)).await?);
    let mut state = claim(context, &index, &descriptor, options, retry).await?;
    let result = prepare_and_load(context, &index, &descriptor, &mut state, lock, retry).await;
    if let Err(error) = &result {
        record_failure(context, &index, &state, error.kind(), retry).await?;
    }
    result
}
async fn claim<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    options: BulkWorkerOptions,
    retry: RetryPolicy,
) -> Result<BuildWorkspace> {
    let mut token = [0; 32];
    getrandom::fill(&mut token).map_err(|_| Error::new(ErrorKind::Other))?;
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        building(&mut txn, index).await?;
        super::bulk_scheduler::authorize(context, &mut txn, index.logical_index_id()).await?;
        match txn
            .get(LogicalKey::BuildDescriptor(index.logical_index_id()))
            .await?
        {
            Some(PersistentValue::BuildDescriptor(d)) if d == *descriptor => {}
            _ => return Err(corrupt()),
        }
        let state = match txn
            .get_for_update(LogicalKey::BuildWorkspace(index.logical_index_id()))
            .await?
        {
            None => BuildWorkspace {
                options: options.clone(),
                hard_limits: backend.hard_limits(),
                token,
                epoch: 1,
                forest: None,
                serving: None,
                failure: None,
            },
            Some(PersistentValue::BuildWorkspace(mut w)) => {
                if w.options != options || w.hard_limits != backend.hard_limits() {
                    return Err(Error::invalid_argument());
                }
                if let Some(kind) = w.failure {
                    return Err(Error::new(kind));
                }
                w.epoch = w
                    .epoch
                    .checked_add(1)
                    .ok_or_else(|| Error::new(ErrorKind::IdExhausted))?;
                w
            }
            _ => return Err(corrupt()),
        };
        if let Some(PersistentValue::BuildLoad(load)) = txn
            .get(LogicalKey::BuildLoad(index.logical_index_id()))
            .await?
            && load.sealed
        {
            return Err(Error::invalid_argument());
        }
        txn.put(
            LogicalKey::BuildWorkspace(index.logical_index_id()),
            PersistentValue::BuildWorkspace(state.clone()),
        )
        .await?;
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(state),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?
            }
            Err(e) => return Err(e),
        }
    }
}
async fn prepare_and_load<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &mut BuildWorkspace,
    lock: Arc<File>,
    retry: RetryPolicy,
) -> Result<()> {
    if let Some(accepted) = &state.serving {
        let path = state.attempt(accepted.epoch).join("serving");
        let expected = accepted.manifest.clone();
        let manifest = index.clone();
        let descriptor_copy = descriptor.clone();
        let limits = state.hard_limits;
        let guard = lock.clone();
        let serving = blocking(context, move || {
            let _lock = guard;
            ServingArtifact::accepted(&path, expected, &manifest, &descriptor_copy, limits)
        })
        .await?;
        return super::bulk_load::load(
            context,
            index.clone(),
            descriptor.clone(),
            serving,
            state.options.load,
            retry,
        )
        .await;
    }
    let input = blocking(context, {
        let descriptor = descriptor.clone();
        let config = index.config().clone();
        let lock = lock.clone();
        move || {
            let _lock = lock;
            InputSnapshot::open(descriptor.source(), config, descriptor.input().clone())
        }
    })
    .await?;
    if state.serving.is_none() {
        let path = state.attempt(state.epoch);
        let lock = lock.clone();
        blocking(context, move || {
            let _lock = lock;
            fs::create_dir_all(path.parent().expect("owned directory")).map_err(io)?;
            fs::create_dir(&path).map_err(io)?;
            File::open(path.parent().expect("parent"))
                .and_then(|f| f.sync_all())
                .map_err(io)?;
            File::open(
                path.parent()
                    .and_then(Path::parent)
                    .expect("workspace root"),
            )
            .and_then(|f| f.sync_all())
            .map_err(io)
        })
        .await?;
    }
    let forest_options = ForestOptions {
        tree: descriptor.options(),
        sort_memory_bytes: state.options.sort_memory_bytes,
        sort_scratch_bytes: state.options.sort_scratch_bytes,
    };
    let forest = if let Some(accepted) = &state.forest {
        let path = state.attempt(accepted.epoch).join("forest");
        let expected = accepted.manifest.clone();
        let source = input.clone();
        let seed = *index.rotation_seed();
        let lock = lock.clone();
        blocking(context, move || {
            let _lock = lock;
            ForestArtifact::open(&path, expected, &source, seed, forest_options)
        })
        .await?
    } else {
        let path = state.attempt(state.epoch).join("forest");
        let source = input.clone();
        let seed = *index.rotation_seed();
        let quota = state.options.max_artifact_bytes;
        let lock = lock.clone();
        let forest = blocking(context, move || {
            let _lock = lock;
            ForestArtifact::build(&path, &source, seed, forest_options, quota)
                .map(|(artifact, _)| artifact)
        })
        .await?;
        let mut after = state.clone();
        after.forest = Some(PreparedArtifact {
            epoch: state.epoch,
            manifest: forest.manifest().clone(),
        });
        accept(context, index, state, &after, retry).await?;
        *state = after;
        forest
    };
    let serving_options = ServingOptions {
        memory_bytes: state.options.serving_memory_bytes,
        scratch_bytes: state.options.serving_scratch_bytes,
        hard_limits: state.hard_limits,
    };
    let serving = {
        let path = state.attempt(state.epoch).join("serving");
        let manifest = index.clone();
        let quota = state.options.max_artifact_bytes;
        let lock = lock.clone();
        let serving = blocking(context, move || {
            let _lock = lock;
            ServingArtifact::build(&path, &input, &forest, &manifest, serving_options, quota)
                .map(|(artifact, _)| artifact)
        })
        .await?;
        let mut after = state.clone();
        after.serving = Some(PreparedArtifact {
            epoch: state.epoch,
            manifest: serving.manifest().clone(),
        });
        accept(context, index, state, &after, retry).await?;
        *state = after;
        serving
    };
    // The shared root lock spans the entire async load; detached reader tasks
    // hold only open immutable file handles and cannot recreate deleted paths.
    super::bulk_load::load(
        context,
        index.clone(),
        descriptor.clone(),
        serving,
        state.options.load,
        retry,
    )
    .await
}
async fn accept<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    before: &BuildWorkspace,
    after: &BuildWorkspace,
    retry: RetryPolicy,
) -> Result<()> {
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        building(&mut txn, index).await?;
        super::bulk_scheduler::authorize(context, &mut txn, index.logical_index_id()).await?;
        let key = LogicalKey::BuildWorkspace(index.logical_index_id());
        let current = match txn.get_for_update(key.clone()).await? {
            Some(PersistentValue::BuildWorkspace(w)) => w,
            _ => return Err(corrupt()),
        };
        if current == *after {
            return Ok(());
        }
        if current != *before {
            return Err(Error::new(ErrorKind::BulkBuildSuperseded));
        }
        txn.put(key, PersistentValue::BuildWorkspace(after.clone()))
            .await?;
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(()),
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                ) =>
            {
                if let Err(exhausted) = retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await
                {
                    return Err(if e.kind() == ErrorKind::CommitOutcomeUnknown {
                        e
                    } else {
                        exhausted
                    });
                }
            }
            Err(e) => return Err(e),
        }
    }
}
pub(crate) async fn record_failure<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    before: &BuildWorkspace,
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
    match accept(context, index, before, &after, retry).await {
        Err(e)
            if matches!(
                e.kind(),
                ErrorKind::BulkBuildSuperseded
                    | ErrorKind::IndexDropping
                    | ErrorKind::IndexNotFound
                    | ErrorKind::Cancelled
                    | ErrorKind::DeadlineExceeded
            ) =>
        {
            Ok(())
        }
        result => result,
    }
}

/// Only generated child directories are removed. The existing root and its
/// coordination file stay in place so stale initializers cannot evade the lock.
pub(crate) async fn cleanup<B: Backend>(
    context: &mut OperationContext<B>,
    id: crate::api::LogicalIndexId,
    retry: RetryPolicy,
) -> Result<()> {
    let backend = context.backend();
    let mut snapshot = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
    let state = match snapshot.get(LogicalKey::BuildWorkspace(id)).await? {
        None => return Ok(()),
        Some(PersistentValue::BuildWorkspace(w)) => w,
        _ => return Err(corrupt()),
    };
    drop(snapshot);
    let root = state.options.workspace.clone();
    let lock = Arc::new(blocking(context, move || lock_root(&root, true)).await?);
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        if let Some(value) = txn.get_for_update(LogicalKey::Manifest(id)).await? {
            match value {
                PersistentValue::IndexManifest(m)
                    if m.logical_index_id() == id && m.lifecycle() != IndexLifecycle::Building => {}
                _ => return Err(Error::invalid_argument()),
            }
        }
        let key = LogicalKey::BuildWorkspace(id);
        match txn.get_for_update(key.clone()).await? {
            None => return Ok(()),
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
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(()),
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                ) =>
            {
                if let Err(exhausted) = retry
                    .wait_or_exhaust(Operation::CleanupBulkBuild, &mut attempts)
                    .await
                {
                    return Err(if e.kind() == ErrorKind::CommitOutcomeUnknown {
                        e
                    } else {
                        exhausted
                    });
                }
            }
            Err(e) => return Err(e),
        }
    }
}
pub(crate) fn io(e: std::io::Error) -> Error {
    Error::with_source(ErrorKind::Other, e)
}
pub(crate) fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// Discovers namespace cleanup records even after their original names and
/// manifests were dropped. One bounded page is visited per call.
pub(crate) async fn cleanup_pending<B: Backend>(
    context: &mut OperationContext<B>,
    maximum: usize,
    after: Option<crate::api::LogicalIndexId>,
    retry: RetryPolicy,
) -> Result<crate::api::BulkCleanupPage> {
    use crate::storage::backend::ReadOps;
    if maximum == 0 || maximum > 1000 {
        return Err(Error::invalid_argument());
    }
    let backend = context.backend();
    let mut raw = backend.begin_read().await?;
    let start = match after {
        None => vec![0, 2],
        Some(id) => {
            let mut key = crate::storage::keys::build_workspace_key(id);
            key.push(0);
            key
        }
    };
    let page = raw
        .scan(
            &crate::storage::keys::KeyRange::new(start, vec![0, 3]),
            crate::storage::backend::ScanLimits {
                item_limit: maximum,
                byte_limit: 1024 * 1024,
            },
        )
        .await?;
    let mut ids = Vec::new();
    for item in page.items() {
        match crate::storage::keys::decode_key(&[], item.key())? {
            LogicalKey::BuildWorkspace(id) => ids.push(id),
            _ => return Err(corrupt()),
        }
    }
    let next = if page.is_terminal() {
        None
    } else {
        ids.last().copied()
    };
    drop(raw);
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
