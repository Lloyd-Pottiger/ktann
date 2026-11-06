//! Atomic Leaf Entry and Child Entry moves with bounded transaction admission.
//! Membership, Record Locations, counts, epochs, and Synopses move together.

use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::storage::WriteLogicalTxn;
use crate::storage::backend::{AdmissionBudget, WriteTxn};
use crate::storage::keys::{self, LogicalKey, TreeKey};
use crate::storage::membership::{added_entry, expect_inserted, removed_entry};
use crate::storage::values::{
    ChildEntry, IndexManifest, LeafEntry, PartitionHeader, PartitionState, PersistentValue,
    RecordLocation, VectorRecord, expect_child_entry, expect_header, expect_leaf_entry,
    expect_location, expect_record, expect_synopsis, leaf_relocation_value_sizes,
};

use super::corrupt;

/// One verified leaf drain candidate: the source Leaf Entry and its current
/// Vector Record, read from the caller's transaction with the entry and its
/// Record Location update-protected.
///
/// The Vector Record is read without update protection on purpose: a
/// concurrent same-leaf replacement rewrites the entry in the same commit, so
/// the entry conflict covers the pair, and the consistent snapshot keeps the
/// decoded record in agreement with the entry.
pub struct LeafDrainEntry {
    source: PartitionKey,
    entry: LeafEntry,
    record: VectorRecord,
}

impl LeafDrainEntry {
    /// Returns the source Leaf Entry, copied verbatim to the target on a move.
    #[must_use]
    pub const fn entry(&self) -> &LeafEntry {
        &self.entry
    }

    /// Returns the current Vector Record used for the routing decision.
    #[must_use]
    pub const fn record(&self) -> &VectorRecord {
        &self.record
    }
}

impl std::fmt::Debug for LeafDrainEntry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LeafDrainEntry([REDACTED])")
    }
}

/// Re-reads one batch of leaf drain candidates inside the drain write
/// transaction.
///
/// Returns one slot per input (source, Record ID) candidate, in input order. A `None` slot means
/// the source entry is gone — exactly the committed trace of a concurrent
/// foreground mutation, and never an error. A remaining entry whose Record
/// Location or Vector Record is absent or inconsistent is Corruption. The
/// reads are three batches (entries and Locations update-protected, Records
/// plain), not per-entry round trips.
pub async fn read_leaf_drain_candidates<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    candidates: &[(PartitionKey, Bytes)],
) -> Result<Vec<Option<LeafDrainEntry>>> {
    let index = txn.require_manifest()?.logical_index_id();
    let entry_keys: Vec<LogicalKey> = candidates
        .iter()
        .map(|(source, id)| LogicalKey::LeafEntry {
            index,
            tree_key: tree_key.clone(),
            partition: *source,
            id: id.clone(),
        })
        .collect();
    let mut entries = txn.batch_get_for_update(entry_keys).await?.into_iter();

    // Validate present entries, then batch their Locations and Records.
    let mut present: Vec<(usize, LeafEntry)> = Vec::new();
    let mut slots: Vec<Option<LeafDrainEntry>> = Vec::with_capacity(candidates.len());
    for (position, (_, id)) in candidates.iter().enumerate() {
        let Some(entry_value) = entries.next() else {
            // The typed batch read returns exactly one value per input key.
            return Err(Error::new(ErrorKind::Backend));
        };
        match expect_leaf_entry(entry_value)? {
            Some(entry) if entry.record_id() == id => {
                present.push((position, entry));
                slots.push(None);
            }
            Some(_) => return Err(corrupt()),
            None => slots.push(None),
        }
    }

    let location_keys: Vec<LogicalKey> = present
        .iter()
        .map(|(_, entry)| LogicalKey::Location {
            index,
            id: entry.record_id().clone(),
        })
        .collect();
    let record_keys: Vec<LogicalKey> = present
        .iter()
        .map(|(_, entry)| LogicalKey::Record {
            index,
            id: entry.record_id().clone(),
        })
        .collect();
    let mut locations = txn.batch_get_for_update(location_keys).await?.into_iter();
    let mut records = txn.batch_get(record_keys).await?.into_iter();

    for (position, entry) in present {
        let (Some(location_value), Some(record_value)) = (locations.next(), records.next()) else {
            return Err(Error::new(ErrorKind::Backend));
        };
        let location = expect_location(location_value)?.ok_or_else(corrupt)?;
        if location.tree_key() != tree_key || location.leaf() != candidates[position].0 {
            return Err(corrupt());
        }
        let record = expect_record(record_value)?.ok_or_else(corrupt)?;
        if record.record_id() != entry.record_id() {
            return Err(corrupt());
        }
        slots[position] = Some(LeafDrainEntry {
            source: candidates[position].0,
            entry,
            record,
        });
    }
    Ok(slots)
}

/// The structural movement protocol one entry-relocation batch runs under.
///
/// Both state machines move entries with the same atomic insert/delete,
/// exact-count, cache-epoch, and Record Location rules; they differ only in
/// the source and target states a move is legal between.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Movement {
    /// A `DrainingSplit` source draining into its two persisted
    /// `ReceivingSplit` targets (ADR 0014).
    Split,
    /// A `Merging` source draining into per-batch reselected `Ready` targets
    /// (ADR 0008).
    Merge,
    /// Caller-exclusive refinement between existing `Ready` leaves.
    Refine,
}

impl Movement {
    /// The only source state a move under this protocol is legal from.
    fn source_state(self) -> PartitionState {
        match self {
            Self::Split => PartitionState::DrainingSplit,
            Self::Merge => PartitionState::Merging,
            Self::Refine => PartitionState::Ready,
        }
    }

    /// The only target state a move under this protocol is legal into.
    fn target_state(self) -> PartitionState {
        match self {
            Self::Split => PartitionState::ReceivingSplit,
            Self::Merge | Self::Refine => PartitionState::Ready,
        }
    }
}

/// Returns the largest safe single-source leaf relocation batch for one Backend budget.
///
/// The bound charges exact codec sizes for the current Manifest and Tree Key,
/// the Backend adapter's per-key physical overhead, and the worst target
/// distribution for the movement protocol. A split may touch both persisted
/// targets; a merge or refinement may route every entry to a different Ready target.
pub(crate) fn leaf_relocation_batch_limit(
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    movement: Movement,
    budget: AdmissionBudget,
) -> Result<usize> {
    let charge = LeafRelocationCharge::new(manifest, tree_key, budget.mutation_key_overhead_bytes)?;
    let mut high = budget.max_mutations.saturating_sub(1) / 3;
    if high == 0 || !charge.fits(1, movement, budget) {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }

    // The charge is monotonic in entry count, including the split's second
    // target at entry two, so binary search finds the exact joint count/byte
    // bound without iterating once per admissible entry.
    let mut low = 1_usize;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if charge.fits(middle, movement, budget) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

/// Worst-case physical mutation-byte charges for one leaf relocation.
struct LeafRelocationCharge {
    source_header: usize,
    entry: usize,
    target: usize,
}

impl LeafRelocationCharge {
    /// Precomputes conservative relocation costs, including adapter overhead for every physical mutation.
    fn new(
        manifest: &IndexManifest,
        tree_key: &TreeKey,
        mutation_key_overhead_bytes: usize,
    ) -> Result<Self> {
        let values = leaf_relocation_value_sizes(manifest, tree_key)?;
        let leaf_key = keys::maximum_leaf_entry_key_len(tree_key);
        let metadata_key = keys::partition_metadata_key_len(tree_key);
        let location_key = keys::maximum_location_key_len();
        let mutation = |key: usize, value: usize| {
            key.checked_add(value)
                .and_then(|bytes| bytes.checked_add(mutation_key_overhead_bytes))
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))
        };
        let source_header = mutation(metadata_key, values.partition_header)?;
        let location = mutation(location_key, values.record_location)?;
        let entry = mutation(leaf_key, values.leaf_entry)?
            .checked_add(mutation(leaf_key, 0)?)
            .and_then(|bytes| bytes.checked_add(location))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let target = mutation(metadata_key, values.partition_header)?
            .checked_add(mutation(metadata_key, values.partition_synopsis)?)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        Ok(Self {
            source_header,
            entry,
            target,
        })
    }

    fn size(&self, entries: usize, movement: Movement) -> Option<(usize, usize)> {
        let targets = match movement {
            Movement::Split => entries.min(2),
            Movement::Merge | Movement::Refine => entries,
        };
        let mutations = entries
            .checked_mul(3)
            .and_then(|count| targets.checked_mul(2)?.checked_add(count))
            .and_then(|count| count.checked_add(1))?;
        let bytes = entries
            .checked_mul(self.entry)
            .and_then(|bytes| targets.checked_mul(self.target)?.checked_add(bytes))
            .and_then(|bytes| bytes.checked_add(self.source_header))?;
        Some((mutations, bytes))
    }

    fn fits(&self, entries: usize, movement: Movement, budget: AdmissionBudget) -> bool {
        matches!(
            self.size(entries, movement),
            Some((mutations, bytes))
                if mutations <= budget.max_mutations && bytes <= budget.max_mutation_bytes
        )
    }
}

/// Atomically moves one batch of verified leaf entries to their chosen targets.
///
/// Each record appears at most once in a batch; candidates are read before moves.
///
/// Each move copies the Leaf Entry — including its absolute RaBitQ7 payload —
/// unchanged to the target, unique-inserts it, deletes the source entry, and
/// repoints the Record Location. The batch reads each touched Header and each
/// receiving target's Synopsis once, accumulates the exact counts,
/// cache epochs, and synopsis expansions in memory, and writes each authority
/// value back once, so a batch never pays per-entry authority round trips.
/// Every source and target must be a leaf in the states `movement` names:
/// callers provide candidates read in this transaction and preserve the
/// protocol, including caller exclusivity for refinement. Returns the number
/// of moved entries.
pub async fn relocate_leaf_entries<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    moves: Vec<(LeafDrainEntry, PartitionKey)>,
    movement: Movement,
) -> Result<usize> {
    if moves.is_empty() {
        return Ok(0);
    }
    let manifest = txn.require_manifest()?;
    let index = manifest.logical_index_id();

    // Source and target roles can overlap in refinement. Read each Header once,
    // validate every role, and apply count/epoch changes in caller move order.
    let sources: BTreeSet<_> = moves.iter().map(|(drain, _)| drain.source).collect();
    let targets: BTreeSet<_> = moves.iter().map(|(_, target)| *target).collect();
    if moves.iter().any(|(drain, target)| drain.source == *target) {
        return Err(corrupt());
    }
    let partitions: Vec<_> = sources.union(&targets).copied().collect();
    let header_keys = partitions
        .iter()
        .map(|partition| LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: *partition,
        })
        .collect();
    let mut values = txn.batch_get_for_update(header_keys).await?.into_iter();
    let mut headers = BTreeMap::new();
    for partition in partitions {
        let Some(value) = values.next() else {
            return Err(Error::new(ErrorKind::Backend));
        };
        let header = expect_header(value)?.ok_or_else(corrupt)?;
        if header.level() != 1
            || (sources.contains(&partition) && header.state() != movement.source_state())
            || (targets.contains(&partition) && header.state() != movement.target_state())
        {
            return Err(corrupt());
        }
        headers.insert(partition, header);
    }
    let synopsis_keys: Vec<LogicalKey> = targets
        .iter()
        .map(|target| LogicalKey::Synopsis {
            index,
            tree_key: tree_key.clone(),
            partition: *target,
        })
        .collect();
    let mut synopsis_values = txn.batch_get_for_update(synopsis_keys).await?.into_iter();
    let mut synopses = BTreeMap::new();
    for target in &targets {
        let Some(value) = synopsis_values.next() else {
            return Err(Error::new(ErrorKind::Backend));
        };
        synopses.insert(
            *target,
            (expect_synopsis(value)?.ok_or_else(corrupt)?, false),
        );
    }

    let moved = moves.len();
    for (drain, target) in &moves {
        let id = drain.entry.record_id().clone();
        expect_inserted(
            txn.insert(
                LogicalKey::LeafEntry {
                    index,
                    tree_key: tree_key.clone(),
                    partition: *target,
                    id: id.clone(),
                },
                PersistentValue::LeafEntry(drain.entry.clone()),
            )
            .await?,
        )?;
        let mut mutations = txn.mutations();
        mutations.delete(LogicalKey::LeafEntry {
            index,
            tree_key: tree_key.clone(),
            partition: drain.source,
            id: id.clone(),
        })?;
        mutations.put(
            LogicalKey::Location { index, id },
            PersistentValue::RecordLocation(RecordLocation::new(tree_key.clone(), *target)),
        )?;
        txn.apply(mutations).await?;
    }

    // Preserve move order even when a partition receives and then sends entries.
    for (drain, target) in &moves {
        let source_header = headers
            .get_mut(&drain.source)
            .expect("source collected from moves");
        *source_header = removed_entry(*source_header)?;
        let target_header = headers
            .get_mut(target)
            .expect("target collected from moves");
        *target_header = added_entry(*target_header)?;
        let (synopsis, changed) = synopses
            .get_mut(target)
            .expect("target collected from moves");
        *changed |= synopsis.expand(manifest, drain.entry.fields())?;
    }
    for (partition, header) in headers {
        txn.put(
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition,
            },
            PersistentValue::PartitionHeader(header),
        )
        .await?;
    }
    for (partition, (synopsis, changed)) in synopses {
        if changed {
            txn.put(
                LogicalKey::Synopsis {
                    index,
                    tree_key: tree_key.clone(),
                    partition,
                },
                PersistentValue::PartitionSynopsis(synopsis),
            )
            .await?;
        }
    }
    Ok(moved)
}

/// Re-reads one batch of internal drain candidates inside the drain write
/// transaction.
///
/// Returns one slot per input child Partition Key, in input order; a `None`
/// slot means the source Child Entry is gone — the committed trace of a
/// concurrent move — and never an error. A key/value identity mismatch is
/// Corruption.
pub async fn read_child_drain_candidates<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    children: &[PartitionKey],
) -> Result<Vec<Option<ChildEntry>>> {
    let index = txn.require_manifest()?.logical_index_id();
    let keys: Vec<LogicalKey> = children
        .iter()
        .map(|child| LogicalKey::ChildEntry {
            index,
            tree_key: tree_key.clone(),
            partition: source,
            child: *child,
        })
        .collect();
    let values = txn.batch_get_for_update(keys).await?;
    let mut slots = Vec::with_capacity(children.len());
    for (child, value) in children.iter().zip(values) {
        match expect_child_entry(value)? {
            Some(entry) if entry.child() == *child => slots.push(Some(entry)),
            Some(_) => return Err(corrupt()),
            None => slots.push(None),
        }
    }
    Ok(slots)
}

/// Atomically moves one batch of verified Child Entries from the draining
/// source to their chosen targets.
///
/// Internal movement applies the same exact source-delete/target-insert rule
/// as leaf movement and updates both exact Header counts and cache epochs —
/// read and written once per partition per batch — but touches neither Record
/// Location, Vector Record, nor Synopsis (ADR 0014): exact Child Entry
/// ownership is the coordination point between adjacent-level maintenance.
/// The source and every target must be in the states `movement` names, and
/// every target must sit at the source's level: exact Child Entry ownership
/// holds only while movement runs inside one structural protocol. Returns the
/// number of moved entries.
pub async fn relocate_child_entries<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
    moves: Vec<(ChildEntry, PartitionKey)>,
    movement: Movement,
) -> Result<usize> {
    if moves.is_empty() {
        return Ok(0);
    }
    let manifest = txn.require_manifest()?;
    let index = manifest.logical_index_id();

    let mut targets: Vec<PartitionKey> = moves.iter().map(|(_, target)| *target).collect();
    targets.sort_unstable();
    targets.dedup();
    let mut header_keys = Vec::with_capacity(targets.len() + 1);
    for target in &targets {
        header_keys.push(LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: *target,
        });
    }
    header_keys.push(LogicalKey::Header {
        index,
        tree_key: tree_key.clone(),
        partition: source,
    });
    let mut headers = txn.batch_get_for_update(header_keys).await?.into_iter();
    let mut target_headers: Vec<(PartitionKey, PartitionHeader)> = Vec::new();
    for target in &targets {
        let Some(value) = headers.next() else {
            return Err(Error::new(ErrorKind::Backend));
        };
        let header = expect_header(value)?.ok_or_else(corrupt)?;
        if header.state() != movement.target_state() {
            return Err(corrupt());
        }
        target_headers.push((*target, header));
    }
    let Some(source_value) = headers.next() else {
        return Err(Error::new(ErrorKind::Backend));
    };
    let source_header = expect_header(source_value)?.ok_or_else(corrupt)?;
    // Movement is legal only inside the named protocol: the source must be
    // draining into exactly these targets.
    if source_header.state() != movement.source_state() {
        return Err(corrupt());
    }
    for (_, target_header) in &target_headers {
        if target_header.level() != source_header.level() {
            return Err(corrupt());
        }
    }

    // Graph-aware drains rewrite the source and one destination together.
    // Their combined serialization bound depends on the small merge source and
    // the hard transient destination ceiling, not on a statistical graph size.
    if manifest.config().local_graph().is_some() {
        let ceiling = manifest.config().max_internal_partition_entries() + 2;
        let source_limit = match movement {
            Movement::Merge => manifest.config().min_internal_partition_entries() - 1,
            _ => ceiling,
        };
        if target_headers.len() != 1 || source_header.entry_count() > source_limit {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let target_count = target_headers[0].1.entry_count();
        if u64::from(target_count) + moves.len() as u64 > u64::from(ceiling)
            || (movement == Movement::Split
                && u64::from(source_header.entry_count()) + u64::from(target_count)
                    > u64::from(ceiling))
        {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
    }
    let moved = moves.len();
    for (entry, target) in &moves {
        expect_inserted(
            txn.insert(
                LogicalKey::ChildEntry {
                    index,
                    tree_key: tree_key.clone(),
                    partition: *target,
                    child: entry.child(),
                },
                PersistentValue::ChildEntry(entry.clone()),
            )
            .await?,
        )?;
        txn.delete(LogicalKey::ChildEntry {
            index,
            tree_key: tree_key.clone(),
            partition: source,
            child: entry.child(),
        })
        .await?;
    }

    let mut source_header = source_header;
    for _ in &moves {
        source_header = removed_entry(source_header)?;
    }
    txn.put(
        LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: source,
        },
        PersistentValue::PartitionHeader(source_header),
    )
    .await?;
    for (target, mut header) in target_headers {
        for (_, move_target) in &moves {
            if move_target == &target {
                header = added_entry(header)?;
            }
        }
        txn.put(
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: target,
            },
            PersistentValue::PartitionHeader(header),
        )
        .await?;
    }
    Ok(moved)
}

#[cfg(test)]
mod tests {
    use crate::api::{DataType, FieldSchema, IndexConfig, LogicalIndexId, Metric, SynopsisConfig};
    use crate::storage::backend::AdmissionBudget;
    use crate::storage::keys::TreeKey;
    use crate::storage::values::{BloomParameters, IndexLifecycle, IndexManifest};

    use super::{LeafRelocationCharge, Movement, leaf_relocation_batch_limit};

    fn manifest() -> IndexManifest {
        let config = IndexConfig::new(128, Metric::L2)
            .expect("valid dimension")
            .with_fields(vec![
                FieldSchema::new("first", DataType::I64).expect("field"),
                FieldSchema::new("second", DataType::String)
                    .expect("field")
                    .with_synopsis(SynopsisConfig::MinMaxBloom {
                        expected_distinct: std::num::NonZeroU32::new(128).expect("nonzero"),
                        false_positive_rate: 0.01,
                    })
                    .expect("bloom"),
            ])
            .expect("valid fields");
        let bloom_parameters = config
            .fields()
            .iter()
            .map(|field| BloomParameters::derive(field.synopsis()).expect("bloom parameters"))
            .collect();
        IndexManifest::new(
            IndexLifecycle::Active,
            LogicalIndexId::new(1).expect("nonzero"),
            config,
            [0; 32],
            bloom_parameters,
        )
        .expect("valid manifest")
    }

    fn tree_key() -> TreeKey {
        TreeKey::encode(&[], &[]).expect("empty Tree Key")
    }

    fn budget(mutations: usize, bytes: usize, overhead: usize) -> AdmissionBudget {
        AdmissionBudget {
            max_mutations: mutations,
            max_mutation_bytes: bytes,
            mutation_key_overhead_bytes: overhead,
        }
    }

    #[test]
    fn leaf_relocation_requires_room_for_one_complete_move() {
        let manifest = manifest();
        let tree_key = tree_key();
        let overhead = 17;
        for movement in [Movement::Split, Movement::Merge, Movement::Refine] {
            let charge = LeafRelocationCharge::new(&manifest, &tree_key, overhead).expect("charge");
            let (mutations, bytes) = charge.size(1, movement).expect("one move fits usize");
            assert_eq!(
                leaf_relocation_batch_limit(
                    &manifest,
                    &tree_key,
                    movement,
                    budget(mutations, bytes, overhead),
                )
                .expect("one move is admissible"),
                1
            );
            assert!(
                leaf_relocation_batch_limit(
                    &manifest,
                    &tree_key,
                    movement,
                    budget(mutations - 1, bytes, overhead),
                )
                .is_err()
            );
            assert!(
                leaf_relocation_batch_limit(
                    &manifest,
                    &tree_key,
                    movement,
                    budget(mutations, bytes - 1, overhead),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn leaf_relocation_uses_exact_count_and_byte_boundaries() {
        let manifest = manifest();
        let tree_key = tree_key();
        let overhead = 23;
        let entries = 37;
        for movement in [Movement::Split, Movement::Merge, Movement::Refine] {
            let charge = LeafRelocationCharge::new(&manifest, &tree_key, overhead).expect("charge");
            let (mutations, bytes) = charge
                .size(entries, movement)
                .expect("test batch fits usize");
            assert_eq!(
                leaf_relocation_batch_limit(
                    &manifest,
                    &tree_key,
                    movement,
                    budget(mutations, usize::MAX, overhead),
                )
                .expect("exact count budget"),
                entries
            );
            assert_eq!(
                leaf_relocation_batch_limit(
                    &manifest,
                    &tree_key,
                    movement,
                    budget(usize::MAX, bytes, overhead),
                )
                .expect("exact byte budget"),
                entries
            );
            assert_eq!(
                leaf_relocation_batch_limit(
                    &manifest,
                    &tree_key,
                    movement,
                    budget(mutations - 1, usize::MAX, overhead),
                )
                .expect("one fewer mutation"),
                entries - 1
            );
            assert_eq!(
                leaf_relocation_batch_limit(
                    &manifest,
                    &tree_key,
                    movement,
                    budget(usize::MAX, bytes - 1, overhead),
                )
                .expect("one fewer byte"),
                entries - 1
            );
        }
    }

    #[test]
    fn leaf_relocation_selects_the_largest_jointly_admissible_batch() {
        let manifest = manifest();
        let tree_key = tree_key();
        let budget = budget(10_000, 1 << 20, 263);
        for movement in [Movement::Split, Movement::Merge, Movement::Refine] {
            let limit = leaf_relocation_batch_limit(&manifest, &tree_key, movement, budget)
                .expect("production-sized budget admits a move");
            let charge =
                LeafRelocationCharge::new(&manifest, &tree_key, budget.mutation_key_overhead_bytes)
                    .expect("charge");
            assert!(charge.fits(limit, movement, budget));
            assert!(!charge.fits(limit + 1, movement, budget));
        }
    }
}
