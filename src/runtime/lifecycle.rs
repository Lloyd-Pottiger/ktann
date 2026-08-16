//! Runtime-owned Logical Index lifecycle over typed storage operations.
//!
//! Create, open, and drop compose the storage module's typed transactions into
//! idempotent persistent state machines. Create reserves one never-reused
//! Logical Index ID and installs the Index Name mapping with an Active Index
//! Manifest. Open validates one named Manifest. Drop first persists the
//! Dropping lifecycle state, then removes the complete index-owned range with
//! transactional range clear when the backend advertises it, or with bounded
//! point-delete pages otherwise, and removes the Index Name mapping last.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use xxhash_rust::xxh3::{xxh3_64, xxh3_128_with_seed};

use crate::api::{Error, ErrorKind, IndexConfig, IndexName, LogicalIndexId, Result, RuntimeConfig};
use crate::storage::backend::{Backend, Capabilities, ScanLimits, WriteTxn};
use crate::storage::keys::LogicalKey;
use crate::storage::values::{
    BloomParameters, IndexIdAllocator, IndexLifecycle, IndexManifest, IndexNameEntry,
    PersistentValue,
};
use crate::storage::{LogicalRange, LogicalScanCursor, ReadLogicalTxn, WriteLogicalTxn};

use super::OperationContext;

const ROTATION_SEED_DOMAIN: u64 = 0x4b54_414e_4e01_b101;
const ROTATION_SEED_SECOND_DOMAIN: u64 = 0x4b54_414e_4e01_b102;

/// Process-local retry-jitter sequence; timing is not persistent protocol.
static RETRY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Whole-operation retry policy copied from one validated RuntimeConfig.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RetryPolicy {
    attempts: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl RetryPolicy {
    pub(crate) fn from_config(config: &RuntimeConfig) -> Self {
        Self {
            attempts: config.foreground_attempts(),
            initial_backoff: config.retry_initial_backoff(),
            max_backoff: config.retry_max_backoff(),
        }
    }

    fn would_exhaust(self, failed_attempts: u32) -> bool {
        failed_attempts.saturating_add(1) >= self.attempts
    }

    async fn wait(self, failed_attempts: u32) {
        let shift = failed_attempts.min(31);
        let current = self
            .initial_backoff
            .checked_mul(1_u32.checked_shl(shift).unwrap_or(u32::MAX))
            .map_or(self.max_backoff, |backoff| backoff.min(self.max_backoff));
        if !current.is_zero() {
            // Full jitter in the current interval. The process-local sequence is
            // mixed through the deterministic XXH3 word stream; no persistent
            // algorithm depends on the chosen value.
            let sequence = RETRY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let sample = xxh3_64(&sequence.to_le_bytes());
            let jittered_nanos = u128::from(sample)
                .checked_mul(current.as_nanos())
                .map_or(u128::MAX, |product| product / u128::from(u64::MAX));
            let jittered_nanos = u64::try_from(jittered_nanos).unwrap_or(u64::MAX);
            tokio::time::sleep(Duration::from_nanos(jittered_nanos)).await;
        }
    }
}

/// Creates one Active Logical Index idempotently for `name`.
///
/// A definite transaction abort restarts the complete create attempt from a
/// fresh snapshot. A commit of unknown outcome is never retried blindly:
/// recovery reads the Index Name once from a fresh snapshot and either returns
/// the current matching index, reports the current conflicting or Dropping
/// state, or — when the Name is absent — returns `CommitOutcomeUnknown`
/// without reserving another ID.
pub(crate) async fn create_index<B: Backend>(
    context: &mut OperationContext<B>,
    name: IndexName,
    config: IndexConfig,
    retry: RetryPolicy,
) -> Result<IndexManifest> {
    let mut failed_attempts = 0_u32;
    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let hard_limits = backend.hard_limits();
        let budget = backend.admission_budget();
        let raw = backend.begin_write().await?;
        let mut txn = WriteLogicalTxn::bootstrap(raw, hard_limits, budget);

        let allocator = match txn.get_for_update(LogicalKey::IndexIdAllocator).await? {
            None => IndexIdAllocator::new(0),
            Some(PersistentValue::IndexIdAllocator(allocator)) => allocator,
            Some(_) => return Err(corruption()),
        };
        let name_key = LogicalKey::IndexNameDirectory(name.clone());
        let existing = txn.get_for_update(name_key.clone()).await?;
        if let Some(existing) = existing {
            let PersistentValue::IndexNameEntry(entry) = existing else {
                return Err(corruption());
            };
            let manifest = read_manifest_for_update(&mut txn, entry.logical_index_id()).await?;
            txn.rollback().await;
            return classify_existing(manifest, &config);
        }

        let next_high_water = allocator
            .high_water()
            .checked_add(1)
            .ok_or_else(id_exhausted)?;
        let logical_index_id = LogicalIndexId::new(next_high_water).map_err(|_| id_exhausted())?;
        let manifest = IndexManifest::new(
            IndexLifecycle::Active,
            logical_index_id,
            config.clone(),
            derive_rotation_seed(logical_index_id),
            derive_bloom_parameters(&config)?,
        )?;

        let inserted = txn
            .insert(
                name_key,
                PersistentValue::IndexNameEntry(IndexNameEntry::new(logical_index_id)),
            )
            .await?;
        if inserted != crate::storage::backend::InsertOutcome::Inserted {
            return Err(Error::new(ErrorKind::Backend));
        }
        txn.put(
            LogicalKey::IndexIdAllocator,
            PersistentValue::IndexIdAllocator(IndexIdAllocator::new(next_high_water)),
        )
        .await?;
        txn.put(
            LogicalKey::Manifest(logical_index_id),
            PersistentValue::IndexManifest(manifest.clone()),
        )
        .await?;

        match context.commit(move |start| txn.commit_with(start)).await {
            Ok(()) => return Ok(manifest),
            Err(error) if error.kind() == ErrorKind::RetryableAbort => {
                if retry.would_exhaust(failed_attempts) {
                    return Err(Error::new(ErrorKind::ContentionExhausted));
                }
                retry.wait(failed_attempts).await;
                failed_attempts += 1;
            }
            Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => {
                return recover_create(backend.as_ref(), &name, &config).await;
            }
            Err(error) => return Err(error),
        }
    }
}

async fn recover_create<B: Backend>(
    backend: &B,
    name: &IndexName,
    config: &IndexConfig,
) -> Result<IndexManifest> {
    let raw = backend.begin_read().await?;
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    let name_key = LogicalKey::IndexNameDirectory(name.clone());
    let Some(existing) = txn.get(name_key).await? else {
        return Err(Error::new(ErrorKind::CommitOutcomeUnknown));
    };
    let PersistentValue::IndexNameEntry(entry) = existing else {
        return Err(corruption());
    };
    let manifest = read_manifest(&mut txn, entry.logical_index_id()).await?;
    classify_existing(manifest, config)
}

fn classify_existing(manifest: IndexManifest, config: &IndexConfig) -> Result<IndexManifest> {
    match manifest.lifecycle() {
        IndexLifecycle::Dropping => Err(Error::new(ErrorKind::IndexDropping)),
        IndexLifecycle::Active if manifest.config() == config => Ok(manifest),
        IndexLifecycle::Active => Err(Error::new(ErrorKind::IndexAlreadyExists)),
    }
}

/// Opens the current Active Manifest for one Index Name.
pub(crate) async fn open_index<B: Backend>(
    context: &mut OperationContext<B>,
    name: IndexName,
) -> Result<IndexManifest> {
    context.checkpoint()?;
    let backend = context.backend();
    let raw = backend.begin_read().await?;
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    let name_key = LogicalKey::IndexNameDirectory(name);
    let Some(existing) = txn.get(name_key).await? else {
        return Err(Error::new(ErrorKind::IndexNotFound));
    };
    let PersistentValue::IndexNameEntry(entry) = existing else {
        return Err(corruption());
    };
    let manifest = read_manifest(&mut txn, entry.logical_index_id()).await?;
    match manifest.lifecycle() {
        IndexLifecycle::Active => Ok(manifest),
        IndexLifecycle::Dropping => Err(Error::new(ErrorKind::IndexDropping)),
    }
}

/// Drops one Logical Index idempotently.
///
/// The first committed transaction changes Active to Dropping. A range-clear
/// backend then atomically clears the index-owned range and removes the Index
/// Name mapping. A point-delete backend repeatedly deletes one bounded scan
/// page, preserving the Dropping Manifest, and finally removes the empty
/// Manifest and Index Name mapping atomically. After a commit of unknown
/// outcome the next step starts from a fresh snapshot and, for point deletion,
/// restarts the remaining range from its prefix beginning because no page
/// cursor is persisted.
pub(crate) async fn drop_index<B: Backend>(
    context: &mut OperationContext<B>,
    name: IndexName,
    retry: RetryPolicy,
) -> Result<()> {
    let name_key = LogicalKey::IndexNameDirectory(name);
    let mut cursor: Option<LogicalScanCursor> = None;
    let mut failed_attempts = 0_u32;
    let mut unknown_attempts = 0_u32;

    loop {
        context.checkpoint()?;
        let backend = context.backend();
        let capabilities = backend.capabilities();
        let hard_limits = backend.hard_limits();
        let budget = backend.admission_budget();

        let raw = backend.begin_write().await?;
        let mut txn = WriteLogicalTxn::bootstrap(raw, hard_limits, budget);
        let Some(existing) = txn.get_for_update(name_key.clone()).await? else {
            txn.rollback().await;
            return Ok(());
        };
        let PersistentValue::IndexNameEntry(entry) = existing else {
            return Err(corruption());
        };
        let manifest = read_manifest_for_update(&mut txn, entry.logical_index_id()).await?;
        match manifest.lifecycle() {
            IndexLifecycle::Active => {
                let dropping = manifest.with_lifecycle(IndexLifecycle::Dropping)?;
                txn.put(
                    LogicalKey::Manifest(manifest.logical_index_id()),
                    PersistentValue::IndexManifest(dropping),
                )
                .await?;
                match commit_drop_step(context, txn, false).await {
                    CommitStep::Committed { complete } => {
                        debug_assert!(!complete);
                        continue;
                    }
                    CommitStep::RetryableAbort => {
                        if retry.would_exhaust(failed_attempts) {
                            return Err(Error::new(ErrorKind::ContentionExhausted));
                        }
                        retry.wait(failed_attempts).await;
                        failed_attempts += 1;
                        continue;
                    }
                    CommitStep::Unknown => {
                        cursor = None;
                        unknown_attempts += 1;
                        if unknown_attempts >= retry.attempts {
                            return recover_drop_after_unknown(backend.as_ref(), &name_key).await;
                        }
                        continue;
                    }
                    CommitStep::Error(error) => return Err(error),
                }
            }
            IndexLifecycle::Dropping => {
                let raw = txn.into_raw();
                txn = WriteLogicalTxn::for_drop(raw, &manifest, hard_limits, budget)?;
            }
        }

        let Some(existing) = txn.get_for_update(name_key.clone()).await? else {
            txn.rollback().await;
            return Ok(());
        };
        let PersistentValue::IndexNameEntry(entry) = existing else {
            return Err(corruption());
        };
        let current = read_manifest_for_update(&mut txn, entry.logical_index_id()).await?;
        if current.lifecycle() != IndexLifecycle::Dropping {
            return Err(corruption());
        }
        if current.logical_index_id() != manifest.logical_index_id() {
            return Err(corruption());
        }

        let step =
            prepare_delete_step(&mut txn, &name_key, &current, capabilities, cursor.as_ref())
                .await?;
        match step {
            DropStep::Advance { cursor: next } => {
                txn.rollback().await;
                cursor = Some(next);
            }
            DropStep::Commit {
                cursor: next,
                complete,
            } => match commit_drop_step(context, txn, complete).await {
                CommitStep::Committed { complete: true } => return Ok(()),
                CommitStep::Committed { complete: false } => {
                    cursor = next;
                }
                CommitStep::RetryableAbort => {
                    if retry.would_exhaust(failed_attempts) {
                        return Err(Error::new(ErrorKind::ContentionExhausted));
                    }
                    retry.wait(failed_attempts).await;
                    failed_attempts += 1;
                }
                CommitStep::Unknown => {
                    cursor = None;
                    unknown_attempts += 1;
                    if unknown_attempts >= retry.attempts {
                        return recover_drop_after_unknown(backend.as_ref(), &name_key).await;
                    }
                }
                CommitStep::Error(error) => return Err(error),
            },
        }
    }
}

enum CommitStep {
    Committed { complete: bool },
    RetryableAbort,
    Unknown,
    Error(Error),
}

async fn commit_drop_step<B: Backend>(
    context: &mut OperationContext<B>,
    txn: WriteLogicalTxn<'_, B::WriteTxn<'_>>,
    complete: bool,
) -> CommitStep {
    match context.commit(move |start| txn.commit_with(start)).await {
        Ok(()) => CommitStep::Committed { complete },
        Err(error) if error.kind() == ErrorKind::RetryableAbort => CommitStep::RetryableAbort,
        Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => CommitStep::Unknown,
        Err(error) => CommitStep::Error(error),
    }
}

enum DropStep {
    Advance {
        cursor: LogicalScanCursor,
    },
    Commit {
        cursor: Option<LogicalScanCursor>,
        complete: bool,
    },
}

async fn prepare_delete_step<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    name_key: &LogicalKey,
    manifest: &IndexManifest,
    capabilities: Capabilities,
    cursor: Option<&LogicalScanCursor>,
) -> Result<DropStep> {
    let range = LogicalRange::index(manifest);
    if capabilities.transactional_clear_range {
        txn.clear_range(&range).await?;
        txn.delete(name_key.clone()).await?;
        return Ok(DropStep::Commit {
            cursor: None,
            complete: true,
        });
    }

    let page = txn
        .scan(
            &range,
            cursor,
            ScanLimits {
                item_limit: scan_item_limit(txn),
                byte_limit: scan_byte_limit(txn),
            },
        )
        .await?;
    let next_cursor = page.next_cursor().cloned();
    let mut deleted = 0_usize;
    for item in page.items() {
        if matches!(item.key(), LogicalKey::Manifest(_)) {
            continue;
        }
        txn.delete(item.key().clone()).await?;
        deleted += 1;
    }

    if deleted == 0 {
        if let Some(next_cursor) = next_cursor {
            Ok(DropStep::Advance {
                cursor: next_cursor,
            })
        } else {
            txn.delete(LogicalKey::Manifest(manifest.logical_index_id()))
                .await?;
            txn.delete(name_key.clone()).await?;
            Ok(DropStep::Commit {
                cursor: None,
                complete: true,
            })
        }
    } else {
        Ok(DropStep::Commit {
            cursor: next_cursor,
            complete: false,
        })
    }
}

async fn recover_drop_after_unknown<B: Backend>(backend: &B, name_key: &LogicalKey) -> Result<()> {
    let raw = backend.begin_read().await?;
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    let Some(existing) = txn.get(name_key.clone()).await? else {
        return Ok(());
    };
    let PersistentValue::IndexNameEntry(entry) = existing else {
        return Err(corruption());
    };
    let _ = read_manifest(&mut txn, entry.logical_index_id()).await?;
    Err(Error::new(ErrorKind::CommitOutcomeUnknown))
}

fn scan_item_limit<T: WriteTxn>(_txn: &WriteLogicalTxn<'_, T>) -> usize {
    _txn.admission_budget().max_mutations.max(1)
}

fn scan_byte_limit<T: WriteTxn>(_txn: &WriteLogicalTxn<'_, T>) -> usize {
    _txn.admission_budget().max_mutation_bytes.max(1)
}

async fn read_manifest<T: crate::storage::backend::ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    logical_index_id: LogicalIndexId,
) -> Result<IndexManifest> {
    match txn.get(LogicalKey::Manifest(logical_index_id)).await? {
        Some(PersistentValue::IndexManifest(manifest)) => Ok(manifest),
        Some(_) => Err(corruption()),
        None => Err(corruption()),
    }
}

async fn read_manifest_for_update<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    logical_index_id: LogicalIndexId,
) -> Result<IndexManifest> {
    match txn
        .get_for_update(LogicalKey::Manifest(logical_index_id))
        .await?
    {
        Some(PersistentValue::IndexManifest(manifest)) => Ok(manifest),
        Some(_) => Err(corruption()),
        None => Err(corruption()),
    }
}

fn derive_bloom_parameters(config: &IndexConfig) -> Result<Vec<Option<BloomParameters>>> {
    config
        .fields()
        .iter()
        .map(|field| BloomParameters::derive(field.synopsis()))
        .collect()
}

fn derive_rotation_seed(logical_index_id: LogicalIndexId) -> [u8; 32] {
    let first = xxh3_128_with_seed(&logical_index_id.get().to_be_bytes(), ROTATION_SEED_DOMAIN);
    let second = xxh3_128_with_seed(
        &first.to_le_bytes(),
        ROTATION_SEED_SECOND_DOMAIN ^ (first as u64),
    );
    let mut seed = [0_u8; 32];
    seed[..16].copy_from_slice(&first.to_le_bytes());
    seed[16..].copy_from_slice(&second.to_le_bytes());
    seed
}

fn corruption() -> Error {
    Error::new(ErrorKind::Corruption)
}

fn id_exhausted() -> Error {
    Error::new(ErrorKind::IdExhausted)
}
