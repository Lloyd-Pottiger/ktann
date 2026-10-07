//! A single fenced, resumable load task over a sealed serving artifact.

use super::{
    OperationContext,
    bulk_worker::{blocking, building},
    lifecycle::RetryPolicy,
};
use crate::api::{BulkBuildStatus, BulkLoadOptions, Error, ErrorKind, Result};
use crate::bulk::{ServingArtifact, ServingEntry, ServingReader};
use crate::observe::labels::Operation;
use crate::storage::backend::{AdmissionBudget, Backend};
use crate::storage::keys::LogicalKey;
use crate::storage::values::{BuildDescriptor, BuildLoad, IndexManifest, PersistentValue};
use crate::storage::{MutationBuilder, WriteLogicalTxn};

/// Maps a persistent checkpoint to the caller-visible loading phase.
pub(crate) fn status(load: &BuildLoad) -> BulkBuildStatus {
    if load.complete {
        BulkBuildStatus::Loaded {
            entries: load.entries,
        }
    } else {
        BulkBuildStatus::Loading {
            loaded_entries: load.entries,
            total_entries: load.artifact.items(),
        }
    }
}

pub(crate) async fn load<B: Backend>(
    context: &mut OperationContext<B>,
    index: IndexManifest,
    descriptor: BuildDescriptor,
    artifact: ServingArtifact,
    options: BulkLoadOptions,
    retry: RetryPolicy,
) -> Result<()> {
    if !artifact.matches_build(&index, &descriptor) {
        return Err(Error::invalid_argument());
    }
    let backend = context.backend();
    let mut budget = backend.admission_budget();
    budget.max_mutations = budget.max_mutations.min(options.max_mutations);
    budget.max_mutation_bytes = budget.max_mutation_bytes.min(options.max_bytes);
    if budget.max_mutations < 2 || budget.max_mutation_bytes == 0 {
        return Err(Error::invalid_argument());
    }
    // Registration has the same admission constraints as every checkpoint.
    let initial = BuildLoad {
        artifact: artifact.manifest().clone(),
        epoch: 1,
        entries: 0,
        prefix_sha256: artifact.manifest().initial_sha256(),
        complete: false,
        sealed: false,
    };
    let mut admission = MutationBuilder::for_index(&index, backend.hard_limits(), budget);
    admission.put(
        LogicalKey::BuildLoad(index.logical_index_id()),
        PersistentValue::BuildLoad(initial.clone()),
    )?;
    let checkpoint_bytes = admission.size().bytes();
    let mut load = claim(context, &index, &descriptor, initial, budget, retry).await?;
    if load.complete {
        return Ok(());
    }
    let mut stream = blocking(context, move || {
        artifact.reader().map(|reader| Stream {
            prefix_sha256: reader.prefix_sha256(),
            reader,
            pending: None,
            consumed: 0,
        })
    })
    .await?;
    loop {
        context.checkpoint()?;
        let committed = load.entries;
        let expected_prefix = load.prefix_sha256;
        let control = context.options.clone();
        let (next_stream, entries, complete) = blocking(context, move || {
            let (entries, complete) = stream.next_chunk(
                committed,
                expected_prefix,
                budget,
                checkpoint_bytes,
                &control,
            )?;
            Ok::<_, Error>((stream, entries, complete))
        })
        .await?;
        stream = next_stream;
        // Prefix replay validates identity without issuing redundant data writes.
        if stream.consumed < committed {
            continue;
        }
        if entries.is_empty() && !complete {
            continue;
        }
        let next = BuildLoad {
            entries: load
                .entries
                .checked_add(entries.len() as u64)
                .ok_or_else(corrupt)?,
            complete,
            prefix_sha256: stream.prefix_sha256,
            ..load.clone()
        };
        commit_chunk(context, &index, &load, &next, &entries, budget, retry).await?;
        load = next;
        if load.complete {
            return Ok(());
        }
    }
}

/// Reader state remains bounded by one chunk and one look-ahead entry. Recovery
/// replays the immutable prefix in bounded blocking steps to reconstruct SHA-256.
struct Stream {
    reader: ServingReader,
    pending: Option<(ServingEntry, [u8; 32])>,
    prefix_sha256: [u8; 32],
    consumed: u64,
}
impl Stream {
    fn next_chunk(
        &mut self,
        committed: u64,
        expected_prefix: [u8; 32],
        budget: AdmissionBudget,
        checkpoint_bytes: usize,
        control: &crate::api::OperationOptions,
    ) -> Result<(Vec<ServingEntry>, bool)> {
        if self.consumed == committed && self.prefix_sha256 != expected_prefix {
            return Err(corrupt());
        }
        let mut entries = Vec::new();
        let mut bytes = checkpoint_bytes;
        for _ in 1..budget.max_mutations {
            super::check_control(control)?;
            let (entry, digest) = match self.pending.take() {
                Some(pending) => pending,
                None => match self.reader.next() {
                    Some(entry) => (entry?, self.reader.prefix_sha256()),
                    None => {
                        return if self.consumed < committed {
                            Err(corrupt())
                        } else {
                            Ok((entries, true))
                        };
                    }
                },
            };
            if self.consumed < committed {
                self.consumed += 1;
                self.prefix_sha256 = digest;
                if self.consumed == committed && digest != expected_prefix {
                    return Err(corrupt());
                }
                continue;
            }
            let size = entry
                .key
                .len()
                .checked_add(entry.value.len())
                .and_then(|n| n.checked_add(budget.mutation_key_overhead_bytes))
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            let next_bytes = bytes
                .checked_add(size)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            if next_bytes > budget.max_mutation_bytes {
                if entries.is_empty() {
                    return Err(Error::new(ErrorKind::LimitExceeded));
                }
                self.pending = Some((entry, digest));
                return Ok((entries, false));
            }
            bytes = next_bytes;
            self.prefix_sha256 = digest;
            self.consumed += 1;
            entries.push(entry);
        }
        Ok((entries, false))
    }
}

async fn read_load<T: crate::storage::backend::WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: &IndexManifest,
) -> Result<Option<BuildLoad>> {
    match txn
        .get_for_update(LogicalKey::BuildLoad(index.logical_index_id()))
        .await?
    {
        Some(PersistentValue::BuildLoad(load)) => Ok(Some(load)),
        None => Ok(None),
        _ => Err(corrupt()),
    }
}

async fn claim<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    initial: BuildLoad,
    budget: AdmissionBudget,
    retry: RetryPolicy,
) -> Result<BuildLoad> {
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let mut txn = WriteLogicalTxn::for_index(
            backend.begin_write().await?,
            index,
            backend.hard_limits(),
            budget,
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
        let load = match read_load(&mut txn, index).await? {
            None => initial.clone(),
            Some(mut current) => {
                if current.artifact != initial.artifact {
                    return Err(Error::invalid_argument());
                }
                if current.sealed {
                    return Err(Error::invalid_argument());
                }
                if current.complete {
                    return Ok(current);
                }
                current.epoch = current
                    .epoch
                    .checked_add(1)
                    .ok_or_else(|| Error::new(ErrorKind::IdExhausted))?;
                current
            }
        };
        txn.put(
            LogicalKey::BuildLoad(index.logical_index_id()),
            PersistentValue::BuildLoad(load.clone()),
        )
        .await?;
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(load),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(Operation::LoadBulkBuild, &mut attempts)
                    .await?
            }
            // Unknown claims grant no authority to this invocation. Calling again
            // takes a newer epoch whether or not this claim actually committed.
            Err(e) => return Err(e),
        }
    }
}

async fn commit_chunk<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    before: &BuildLoad,
    after: &BuildLoad,
    entries: &[ServingEntry],
    budget: AdmissionBudget,
    retry: RetryPolicy,
) -> Result<()> {
    let backend = context.backend();
    let mut attempts = 0;
    loop {
        context.checkpoint()?;
        let mut txn = WriteLogicalTxn::for_index(
            backend.begin_write().await?,
            index,
            backend.hard_limits(),
            budget,
        );
        building(&mut txn, index).await?;
        super::bulk_scheduler::authorize(context, &mut txn, index.logical_index_id()).await?;
        let current = read_load(&mut txn, index).await?.ok_or_else(corrupt)?;
        if current == *after {
            return Ok(());
        }
        if current.artifact != before.artifact {
            return Err(corrupt());
        }
        if current.epoch != before.epoch {
            return Err(Error::new(ErrorKind::BulkBuildSuperseded));
        }
        if current != *before {
            return Err(corrupt());
        }
        // Rebuild the small mutation batch on retry; codecs do not perform IO.
        let mut batch = txn.mutations();
        batch.put(
            LogicalKey::BuildLoad(index.logical_index_id()),
            PersistentValue::BuildLoad(after.clone()),
        )?;
        for entry in entries {
            // The bound ServingReader already validated ownership and codecs.
            batch.put_encoded(entry.key.clone(), entry.value.clone())?;
        }
        txn.apply(batch).await?;
        match context.commit(|start| txn.commit_with(start)).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                retry
                    .after_commit_error(Operation::LoadBulkBuild, &mut attempts, e)
                    .await?
            }
        }
    }
}
fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}
