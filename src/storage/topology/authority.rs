//! Typed partition authority reads and Header/State agreement checks.
//! Shared by snapshot readers and update-protected topology steps.

use crate::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result};
use crate::storage::backend::WriteTxn;
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    PartitionHeader, PartitionTransition, PersistentValue, expect_header, expect_state,
};
use crate::storage::{LogicalReader, WriteLogicalTxn};

use super::corrupt;

/// Reads one partition Header that must exist.
///
/// This is the single home of the typed authority read; read-only and write
/// transactions share it through [`LogicalReader`]. A missing Header on a
/// referenced partition is Corruption.
pub(super) async fn read_header<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<PartitionHeader> {
    read_header_opt(reader, index, tree_key, partition)
        .await?
        .ok_or_else(corrupt)
}

/// Reads one partition Header, which may be absent after a completed split.
pub(super) async fn read_header_opt<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Option<PartitionHeader>> {
    expect_header(
        reader
            .get(LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition,
            })
            .await?,
    )
}

/// Reads one partition State, which may be absent after a completed split.
pub(super) async fn read_state<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Option<PartitionTransition>> {
    expect_state(
        reader
            .get(LogicalKey::State {
                index,
                tree_key: tree_key.clone(),
                partition,
            })
            .await?,
    )
}

/// Reads one partition's authority pair in one batched plain read,
/// classifying presence and agreement.
///
/// Both values present and in agreement is `Some`; both gone — a completed
/// split or merge removed them, or the partition never existed — is `None`; a
/// half-present or disagreeing pair is Corruption.
pub(crate) async fn read_authority_pair<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Option<(PartitionHeader, PartitionTransition)>> {
    let (header_key, state_key) = authority_keys(index, tree_key, partition);
    let mut values = reader
        .batch_get(vec![header_key, state_key])
        .await?
        .into_iter();
    decode_authority_pair(&mut values)
}

/// Reads one visited partition's Header and State in one batch, failing
/// closed when either is missing, of the wrong kind, or in disagreement.
///
/// Every reachable partition carries both authority values in every committed
/// state: creation installs them together, and completion removes them
/// together with the partition's last incoming reference.
pub(crate) async fn read_authority<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<(PartitionHeader, PartitionTransition)> {
    read_authority_pair(reader, index, tree_key, partition)
        .await?
        .ok_or_else(corrupt)
}

/// Reads the authority pairs for a distinct partition wave in one batched
/// plain read.
///
/// The caller owns the wave and must pass each Partition Key at most once.
/// Keeping the pairs aligned with that input lets a grouped foreground route
/// inspect every partition at one level without issuing one backend call per
/// partition. The read remains in the current transaction snapshot; it is a
/// routing optimization, not a shared cache.
pub(crate) async fn read_authority_batch<R: LogicalReader>(
    reader: &mut R,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partitions: &[PartitionKey],
) -> Result<Vec<(PartitionHeader, PartitionTransition)>> {
    if partitions.is_empty() {
        return Ok(Vec::new());
    }
    let mut keys = Vec::with_capacity(partitions.len().saturating_mul(2));
    for &partition in partitions {
        let (header_key, state_key) = authority_keys(index, tree_key, partition);
        keys.push(header_key);
        keys.push(state_key);
    }
    let mut values = reader.batch_get(keys).await?.into_iter();
    let mut authorities = Vec::with_capacity(partitions.len());
    for _ in partitions {
        authorities.push(decode_authority_pair(&mut values)?.ok_or_else(corrupt)?);
    }
    if values.next().is_some() {
        return Err(Error::new(ErrorKind::Backend));
    }
    Ok(authorities)
}

/// Decodes matching Header and State values. Only a wholly absent pair is missing.
fn decode_authority_pair(
    values: &mut impl Iterator<Item = Option<PersistentValue>>,
) -> Result<Option<(PartitionHeader, PartitionTransition)>> {
    let (Some(header), Some(state)) = (values.next(), values.next()) else {
        // The typed batch read returns exactly one value per input key.
        return Err(Error::new(ErrorKind::Backend));
    };
    match (expect_header(header)?, expect_state(state)?) {
        (Some(header), Some(state)) => {
            expect_agreement(header, state)?;
            Ok(Some((header, state)))
        }
        (None, None) => Ok(None),
        _ => Err(corrupt()),
    }
}

/// Reads one partition's Header and State with update protection in one
/// batch.
pub(crate) async fn authority_for_update<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Option<(PartitionHeader, PartitionTransition)>> {
    let (header, state) = authority_keys(index, tree_key, partition);
    authority_pair(txn, header, state).await
}

/// Reads one partition's authority pair with update protection, classifying
/// presence and agreement.
///
/// Both values present and in agreement is `Some`; both gone — a completed
/// split removed them, or the partition never existed — is `None`; a
/// half-present or disagreeing pair is Corruption.
pub(super) async fn authority_pair<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    header_key: LogicalKey,
    state_key: LogicalKey,
) -> Result<Option<(PartitionHeader, PartitionTransition)>> {
    let values = txn
        .batch_get_for_update(vec![header_key, state_key])
        .await?;
    decode_authority_pair(&mut values.into_iter())
}

/// Builds the Header and State keys of one partition.
pub(super) fn authority_keys(
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> (LogicalKey, LogicalKey) {
    (
        LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition,
        },
        LogicalKey::State {
            index,
            tree_key: tree_key.clone(),
            partition,
        },
    )
}

/// Verifies that a Header discriminator and a State value agree.
pub(super) fn expect_agreement(header: PartitionHeader, state: PartitionTransition) -> Result<()> {
    if header.state() != state.state() {
        return Err(corrupt());
    }
    Ok(())
}
