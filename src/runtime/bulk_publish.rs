//! Exact paged validation of frozen backend bytes and one-transaction activation.
use super::{
    OperationContext,
    bulk_load::read_progress,
    bulk_worker::{self, blocking, corrupt},
    lifecycle::RetryPolicy,
};
use crate::api::{Error, ErrorKind, Result};
use crate::bulk::{ServingArtifact, ServingReader};
use crate::observe::labels::Operation;
use crate::storage::backend::{Backend, ReadOps, ScanLimits};
use crate::storage::keys::{self, KeyRange, LogicalKey};
use crate::storage::values::{
    BuildDescriptor, BuildPhase, BuildProgress, BuildWorkspace, IndexLifecycle, IndexManifest,
    PersistentValue, ValueCodec,
};
use crate::storage::{ReadLogicalTxn, WriteLogicalTxn};
use bytes::Bytes;
use std::sync::Arc;

pub(crate) async fn publish<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    descriptor: BuildDescriptor,
    retry: RetryPolicy,
) -> Result<IndexManifest> {
    let result = publish_inner(context, index.clone(), descriptor, retry).await;
    // Another publisher may activate and reclaim the workspace while this
    // publisher is opening files or resuming a proof. The original identity is
    // authoritative even after the preparation files have been reclaimed.
    if result.is_err()
        && let Ok(Some(active)) = already_active(context, &index).await
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
    let state = {
        let backend = context.backend();
        let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
        if let Some(active) = read_active(&mut txn, &index).await? {
            return Ok(active);
        }
        match txn
            .get(LogicalKey::BuildWorkspace(index.logical_index_id()))
            .await?
        {
            Some(PersistentValue::BuildWorkspace(state)) => state,
            None => return Err(Error::invalid_argument()),
            _ => return Err(corrupt()),
        }
    };
    if let Some(kind) = state.failure {
        return Err(Error::new(kind));
    }
    let accepted = state.serving.as_ref().ok_or_else(Error::invalid_argument)?;
    let root = state.options.workspace.clone();
    let lock = Arc::new(blocking(context, move || bulk_worker::lock_root(&root, false)).await?);
    let path = state.attempt(accepted.epoch).join("serving");
    let expected = accepted.manifest.clone();
    let manifest = index.clone();
    let limits = state.hard_limits;
    let guard = lock.clone();
    let artifact = blocking(context, move || {
        let _lock = guard;
        ServingArtifact::accepted(&path, expected, &manifest, limits)
    })
    .await;
    let result = match artifact {
        Ok(artifact) => {
            validate_and_publish(context, &index, &descriptor, &state, artifact, lock, retry).await
        }
        Err(error) => Err(error),
    };
    if let Err(e) = &result {
        bulk_worker::record_failure(context, &index, &state, e.kind(), retry).await?;
    }
    result
}
async fn already_active<B: Backend>(
    context: &OperationContext<B>,
    expected: &IndexManifest,
) -> Result<Option<IndexManifest>> {
    context.checkpoint()?;
    let backend = context.backend();
    let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
    read_active(&mut txn, expected).await
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
async fn fence<T: crate::storage::backend::WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: &IndexManifest,
    state: &BuildWorkspace,
) -> Result<BuildProgress> {
    bulk_worker::building(txn, index).await?;
    match txn
        .get_for_update(LogicalKey::BuildWorkspace(index.logical_index_id()))
        .await?
    {
        Some(PersistentValue::BuildWorkspace(w))
            if w.token == state.token && w.serving == state.serving && w.failure.is_none() => {}
        _ => return Err(corrupt()),
    }
    match read_progress(txn, index).await? {
        Some(progress)
            if matches!(
                progress.phase,
                BuildPhase::Validating { .. } | BuildPhase::Validated
            ) && progress.artifact == state.serving.as_ref().ok_or_else(corrupt)?.manifest =>
        {
            Ok(progress)
        }
        _ => Err(corrupt()),
    }
}
async fn seal<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &BuildWorkspace,
    retry: RetryPolicy,
) -> Result<BuildProgress> {
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        bulk_worker::building(&mut txn, index).await?;
        super::bulk_scheduler::authorize(context, &mut txn, index.logical_index_id()).await?;
        match txn
            .get_for_update(LogicalKey::BuildWorkspace(index.logical_index_id()))
            .await?
        {
            Some(PersistentValue::BuildWorkspace(w)) if w == *state => {}
            _ => return Err(Error::new(ErrorKind::BulkBuildSuperseded)),
        }
        match txn
            .get(LogicalKey::BuildDescriptor(index.logical_index_id()))
            .await?
        {
            Some(PersistentValue::BuildDescriptor(d)) if d == *descriptor => {}
            _ => return Err(corrupt()),
        }
        let mut progress = read_progress(&mut txn, index)
            .await?
            .ok_or_else(|| Error::new(ErrorKind::BulkBuildBusy))?;
        // Preparation accepts the artifact before loading begins. A publisher
        // arriving early must leave this ordinary in-progress work resumable.
        if matches!(progress.phase, BuildPhase::Loading { .. }) {
            return Err(Error::new(ErrorKind::BulkBuildBusy));
        }
        if progress.artifact
            != state
                .serving
                .as_ref()
                .ok_or_else(Error::invalid_argument)?
                .manifest
        {
            return Err(Error::invalid_argument());
        }
        if matches!(
            progress.phase,
            BuildPhase::Validating { .. } | BuildPhase::Validated
        ) {
            return Ok(progress);
        }
        // Updating the same checkpoint key freezes all previously claimed loads.
        progress.phase = BuildPhase::Validating {
            cursor: Bytes::new(),
            entries: 0,
            prefix_sha256: progress.artifact.initial_sha256(),
        };
        txn.put(
            LogicalKey::BuildProgress(index.logical_index_id()),
            PersistentValue::BuildProgress(progress.clone()),
        )
        .await?;
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(progress),
            Err(e) => {
                retry
                    .after_commit_error(Operation::PublishBulkBuild, &mut attempts, e)
                    .await?
            }
        }
    }
}
async fn validate_and_publish<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &BuildWorkspace,
    artifact: ServingArtifact,
    lock: Arc<std::fs::File>,
    retry: RetryPolicy,
) -> Result<IndexManifest> {
    let mut proof = seal(context, index, descriptor, state, retry).await?;
    if !matches!(proof.phase, BuildPhase::Validated) {
        let guard = lock.clone();
        let mut reader = blocking(context, move || {
            let _lock = guard;
            artifact.reader()
        })
        .await?;
        let mut skipped = 0;
        // Reconstruct the exact source prefix in bounded, cancellation-aware
        // steps. The backend proof must name the identical prefix before reuse.
        while skipped < proof.entries() {
            let take = (proof.entries() - skipped).min(state.options.load.max_mutations as u64);
            let guard = lock.clone();
            let control = context.options.clone();
            reader = blocking(context, move || {
                let _lock = guard;
                for _ in 0..take {
                    super::check_control(&control)?;
                    reader.next().ok_or_else(corrupt)??;
                }
                Ok(reader)
            })
            .await?;
            skipped += take;
        }
        if reader.prefix_sha256() != *proof.prefix_sha256() {
            return Err(corrupt());
        }
        while !matches!(proof.phase, BuildPhase::Validated) {
            let (next_reader, next) =
                page(context, index, state, reader, &proof, lock.clone(), retry).await?;
            reader = next_reader;
            proof = next;
        }
    }
    activate(context, index, state, &proof, retry).await
}
async fn page<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    state: &BuildWorkspace,
    reader: ServingReader,
    before: &BuildProgress,
    lock: Arc<std::fs::File>,
    retry: RetryPolicy,
) -> Result<(ServingReader, BuildProgress)> {
    let BuildPhase::Validating { cursor, .. } = &before.phase else {
        return Err(corrupt());
    };
    let backend = context.backend();
    let mut attempts = 0;
    let mut reader = Some(reader);
    let mut after: Option<BuildProgress> = None;
    loop {
        context.checkpoint()?;
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        let current = fence(&mut txn, index, state).await?;
        super::bulk_scheduler::authorize(context, &mut txn, index.logical_index_id()).await?;
        if after.as_ref().is_some_and(|after| *after == current) {
            return Ok((reader.take().expect("reader"), current));
        }
        if current != *before {
            return Err(Error::new(ErrorKind::BulkBuildSuperseded));
        }
        if after.is_none() {
            let mut raw = txn.into_raw();
            let start = if cursor.is_empty() {
                Bytes::from(keys::index_range(index.logical_index_id()).start().to_vec())
            } else {
                cursor.clone()
            };
            let end = keys::index_range(index.logical_index_id()).end().to_vec();
            let page = raw
                .scan(
                    &KeyRange::new(start.to_vec(), end),
                    ScanLimits {
                        item_limit: state
                            .options
                            .load
                            .max_mutations
                            .min(backend.admission_budget().max_mutations),
                        byte_limit: state
                            .options
                            .load
                            .max_bytes
                            .min(backend.admission_budget().max_mutation_bytes),
                    },
                )
                .await?;
            let next = page.next_start().cloned();
            let terminal = next.is_none();
            let mut data = Vec::new();
            let (types, count) = index.tree_key_types();
            for item in page.items() {
                let key = item.key().clone();
                let value = item.value().clone();
                let logical = keys::decode_key(&types[..count], &key)?;
                if matches!(
                    logical,
                    LogicalKey::Manifest(_)
                        | LogicalKey::BuildDescriptor(_)
                        | LogicalKey::BuildProgress(_)
                ) {
                    ValueCodec::for_index(index).decode(&logical, value)?;
                } else {
                    data.push((key, value));
                }
            }
            let mut input = reader.take().expect("reader");
            let guard = lock.clone();
            let control = context.options.clone();
            let mut next_proof = before.clone();
            let mut compared = before.entries();
            let (input, verified) = blocking(context, move || {
                let _lock = guard;
                for (key, value) in data {
                    super::check_control(&control)?;
                    let expected = input.next().ok_or_else(corrupt)??;
                    if expected.key != key || expected.value != value {
                        return Err(corrupt());
                    }
                    compared = compared.checked_add(1).ok_or_else(corrupt)?;
                }
                if terminal && input.next().transpose()?.is_some() {
                    return Err(corrupt());
                }
                next_proof.phase = if terminal {
                    // Exhausting the reader above verified the full artifact.
                    BuildPhase::Validated
                } else {
                    BuildPhase::Validating {
                        cursor: next.expect("nonterminal scan cursor"),
                        entries: compared,
                        prefix_sha256: input.prefix_sha256(),
                    }
                };
                Ok((input, next_proof))
            })
            .await?;
            reader = Some(input);
            after = Some(verified);
            txn =
                WriteLogicalTxn::bootstrap(raw, backend.hard_limits(), backend.admission_budget());
        }
        txn.put(
            LogicalKey::BuildProgress(index.logical_index_id()),
            PersistentValue::BuildProgress(after.as_ref().expect("page checked").clone()),
        )
        .await?;
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok((reader.take().expect("reader"), after.take().expect("proof"))),
            Err(e) => {
                retry
                    .after_commit_error(Operation::PublishBulkBuild, &mut attempts, e)
                    .await?
            }
        }
    }
}
async fn activate<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    state: &BuildWorkspace,
    proof: &BuildProgress,
    retry: RetryPolicy,
) -> Result<IndexManifest> {
    let active = index.clone().with_lifecycle(IndexLifecycle::Active);
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let mut txn = WriteLogicalTxn::bootstrap(
            backend.begin_write().await?,
            backend.hard_limits(),
            backend.admission_budget(),
        );
        let current = bulk_worker::manifest(&mut txn, index).await?;
        if current.lifecycle() == IndexLifecycle::Active {
            return Ok(current);
        }
        let current = fence(&mut txn, index, state).await?;
        super::bulk_scheduler::authorize(context, &mut txn, index.logical_index_id()).await?;
        if !matches!(proof.phase, BuildPhase::Validated) || current != *proof {
            return Err(corrupt());
        }
        txn.put(
            LogicalKey::Manifest(index.logical_index_id()),
            PersistentValue::IndexManifest(active.clone()),
        )
        .await?;
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(active),
            Err(e) if e.kind() == ErrorKind::CommitOutcomeUnknown => {
                if let Some(active) = already_active(context, index).await? {
                    return Ok(active);
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
                retry
                    .wait_or_exhaust(Operation::PublishBulkBuild, &mut attempts)
                    .await?
            }
            Err(e) => return Err(e),
        }
    }
}
