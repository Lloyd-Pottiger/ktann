//! Exact incoming-edge and same-level discovery without reverse parent pointers.
//! Snapshot proofs retain the parent-level authority needed for write validation.

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::storage::backend::{ScanLimits, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    ChildEntry, IndexManifest, PartitionHeader, PartitionState, PartitionTransition,
    expect_child_entry, expect_child_entry_ref, expect_header,
};
use crate::storage::{LogicalRange, LogicalReader, WriteLogicalTxn};

use super::authority::{read_header, read_header_opt, read_state};
use super::{corrupt, root_partition};

/// Maximum concurrent discovery pages, independent of the tree's fanout.
/// Each leg retains the ordinary page bound; at most eight pages are live.
const DISCOVERY_BODY_BATCH: usize = 8;

/// The bound on one incoming-edge discovery or entry scan page.
///
/// Paging only shapes I/O: discovery walks whole tree levels and every page is
/// bounded, so work stays proportional to the scanned level's actual size.
const DISCOVERY_SCAN_LIMITS: ScanLimits = ScanLimits {
    item_limit: 128,
    byte_limit: 1_048_576,
};
/// Rediscovers one partition's unique incoming Child Entry by an exact
/// bounded root-down scan of every Child Entry at the required parent level
/// (ADR 0007).
///
/// A duplicate match is Corruption; no match returns `None`.
async fn find_incoming_edge<T: LogicalReader>(
    txn: &mut T,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    level: u32,
) -> Result<Option<(PartitionKey, ChildEntry)>> {
    let parent_level = level.checked_add(1).ok_or_else(corrupt)?;
    let bodies = level_bodies(txn, manifest, tree_key, parent_level).await?;
    find_incoming_edge_in(txn, manifest, tree_key, partition, &bodies).await
}

/// Resolves a unique incoming edge within an already discovered parent level.
async fn find_incoming_edge_in<T: LogicalReader>(
    txn: &mut T,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    bodies: &[PartitionKey],
) -> Result<Option<(PartitionKey, ChildEntry)>> {
    // Only the parent level's matches are collected.
    let mut found: Option<(PartitionKey, ChildEntry)> = None;
    scan_child_edges(txn, manifest, tree_key, bodies, &mut |body, entry| {
        if entry.child() == partition {
            if found.is_some() {
                return Err(corrupt());
            }
            found = Some((body, entry.clone()));
        }
        Ok(())
    })
    .await?;
    Ok(found)
}

/// Ephemeral proof of a complete incoming-edge discovery in one read snapshot.
/// Reuse requires the same reachable parent set and update-protected Headers.
/// Every Child Entry change advances its owner's epoch, including edge moves.
pub(crate) struct IncomingEdgeProof {
    level: u32,
    headers: Vec<(PartitionKey, PartitionHeader)>,
}

impl IncomingEdgeProof {
    pub(crate) async fn read<R: LogicalReader>(
        reader: &mut R,
        manifest: &IndexManifest,
        tree: &TreeKey,
        level: u32,
        working_bytes: usize,
    ) -> Result<Self> {
        let level = level.checked_add(1).ok_or_else(corrupt)?;
        let bodies = level_bodies(reader, manifest, tree, level).await?;
        let bytes = bodies
            .len()
            .checked_mul(256)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        if bytes > working_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let mut headers = Vec::with_capacity(bodies.len());
        for batch in bodies.chunks(DISCOVERY_BODY_BATCH) {
            let keys = batch
                .iter()
                .map(|&partition| LogicalKey::Header {
                    index: manifest.logical_index_id(),
                    tree_key: tree.clone(),
                    partition,
                })
                .collect();
            for (&partition, value) in batch.iter().zip(reader.batch_get(keys).await?) {
                headers.push((partition, expect_header(value)?.ok_or_else(corrupt)?));
            }
        }
        Ok(Self { level, headers })
    }

    /// Resolves the edge in this snapshot without rediscovering the parent level.
    pub(crate) async fn find<R: LogicalReader>(
        &self,
        reader: &mut R,
        manifest: &IndexManifest,
        tree: &TreeKey,
        partition: PartitionKey,
    ) -> Result<Option<(PartitionKey, ChildEntry)>> {
        let bodies: Vec<_> = self.bodies().collect();
        find_incoming_edge_in(reader, manifest, tree, partition, &bodies).await
    }

    /// Reuse existing ordered body guards to avoid preparing against a proof
    /// another local publisher is already changing. Storage fences remain final.
    pub(crate) fn bodies(&self) -> impl Iterator<Item = PartitionKey> + '_ {
        self.headers.iter().map(|(partition, _)| *partition)
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.headers.capacity() * 256
    }

    /// False means discard the private plan and rediscover; never use a stale hint.
    pub(crate) async fn validate<T: WriteTxn>(
        &self,
        txn: &mut WriteLogicalTxn<'_, T>,
        tree: &TreeKey,
    ) -> Result<bool> {
        let manifest = txn.require_manifest()?;
        if read_header(txn, manifest.logical_index_id(), tree, root_partition())
            .await?
            .level()
            < self.level
        {
            return Ok(false);
        }
        let bodies = level_bodies(txn, manifest, tree, self.level).await?;
        if !bodies
            .iter()
            .copied()
            .eq(self.headers.iter().map(|(key, _)| *key))
        {
            return Ok(false);
        }
        for batch in self.headers.chunks(DISCOVERY_BODY_BATCH) {
            let keys = batch
                .iter()
                .map(|&(partition, _)| LogicalKey::Header {
                    index: manifest.logical_index_id(),
                    tree_key: tree.clone(),
                    partition,
                })
                .collect();
            for ((_, expected), value) in batch.iter().zip(txn.batch_get_for_update(keys).await?) {
                if expect_header(value)? != Some(*expected) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// Rediscovers one non-root partition's unique incoming Child Entry by the
/// exact root-down scan and update-protects it, returning the parent's
/// Partition Key.
///
/// A missing or duplicate edge is Corruption; a concurrent edge move aborts
/// the commit through the update-protected read (ADR 0007).
pub(super) async fn lock_incoming_edge<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    partition: PartitionKey,
    level: u32,
    parent: Option<PartitionKey>,
) -> Result<PartitionKey> {
    let index = txn.require_manifest()?.logical_index_id();
    let parent = if let Some(parent) = parent {
        parent
    } else {
        find_incoming_edge(txn, txn.require_manifest()?, tree_key, partition, level)
            .await?
            .ok_or_else(corrupt)?
            .0
    };
    expect_child_entry(
        txn.get_for_update(LogicalKey::ChildEntry {
            index,
            tree_key: tree_key.clone(),
            partition: parent,
            child: partition,
        })
        .await?,
    )?
    .ok_or_else(corrupt)?;
    Ok(parent)
}

/// Scans complete Child Entry sets in bounded waves, with an independent
/// continuation for each body. Finished legs leave the next round immediately.
async fn scan_child_edges<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    bodies: &[PartitionKey],
    visit: &mut impl FnMut(PartitionKey, &ChildEntry) -> Result<()>,
) -> Result<()> {
    for chunk in bodies.chunks(DISCOVERY_BODY_BATCH) {
        let mut pending = chunk
            .iter()
            .map(|&body| {
                Ok((
                    body,
                    LogicalRange::child_entries(manifest, tree_key, body)?,
                    None,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        while !pending.is_empty() {
            let legs = pending
                .iter()
                .map(|(_, range, cursor)| (range, cursor.as_ref()))
                .collect::<Vec<_>>();
            let pages = reader.batch_scan(&legs, DISCOVERY_SCAN_LIMITS).await?;
            // Typed batch scans return exactly one page per leg in input order.
            for ((body, _, cursor), page) in pending.iter_mut().zip(pages) {
                for item in page.items() {
                    visit(*body, expect_child_entry_ref(item.value())?)?;
                }
                *cursor = page.into_next_cursor();
            }
            pending.retain(|(_, _, cursor)| cursor.is_some());
        }
    }
    Ok(())
}

/// Walks from the root down to `level`, returning every partition body at
/// that level, by the exact bounded root-down walk of ADR 0007.
///
/// Partitions persist no reverse parent pointers, so the walk reads every
/// partition Header at each level to prove the level invariant and collects
/// the next level from scanned Child Entries. While the root is `Splitting`
/// or `DrainingSplit` its targets have no incoming edge of their own, so the
/// exclusive target slots named by the root State are added to the root's
/// level. A level that fails to decrement and an unreachable level are
/// Corruption. Paging only shapes I/O: every page is bounded, so work stays
/// proportional to the scanned levels' actual size.
async fn level_bodies<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    level: u32,
) -> Result<Vec<PartitionKey>> {
    let index = manifest.logical_index_id();
    let root = root_partition();
    let root_header = read_header(reader, index, tree_key, root).await?;
    if root_header.level() < level {
        return Err(corrupt());
    }

    let mut current_level = root_header.level();
    let mut bodies = vec![root];
    loop {
        // While the root exposes its split targets only through its own
        // State, those targets sit at the root's level without an incoming
        // edge; add them exactly once. Exposure and the DrainingSplit
        // transition commit atomically (ADR 0014): a Splitting root's targets
        // cannot exist yet, so an absent target is skipped, while a
        // DrainingSplit root's targets exist by construction.
        if current_level == root_header.level()
            && let Some(root_state) = read_state(reader, index, tree_key, root).await?
        {
            let draining = matches!(root_state, PartitionTransition::DrainingSplit { .. });
            let targets = match root_state {
                PartitionTransition::Splitting { left, right, .. }
                | PartitionTransition::DrainingSplit { left, right, .. } => [left, right],
                _ => [root, root],
            };
            for target in targets {
                if target == root || bodies.contains(&target) {
                    continue;
                }
                let exists = read_header_opt(reader, index, tree_key, target)
                    .await?
                    .is_some();
                if !exists && !draining {
                    continue;
                }
                bodies.push(target);
            }
        }

        if current_level == level {
            return Ok(bodies);
        }

        // Descend: intermediate levels contribute only child Partition Keys.
        let mut next = Vec::new();
        for chunk in bodies.chunks(DISCOVERY_BODY_BATCH) {
            let keys = chunk
                .iter()
                .map(|&partition| LogicalKey::Header {
                    index,
                    tree_key: tree_key.clone(),
                    partition,
                })
                .collect();
            for value in reader.batch_get(keys).await? {
                let header = expect_header(value)?.ok_or_else(corrupt)?;
                if header.level() != current_level {
                    return Err(corrupt());
                }
            }
            scan_child_edges(reader, manifest, tree_key, chunk, &mut |_, entry| {
                next.push(entry.child());
                Ok(())
            })
            .await?;
        }
        if next.is_empty() {
            // A level above one that cannot be reached contradicts the
            // level-decrement invariant.
            return Err(corrupt());
        }
        bodies = next;
        current_level -= 1;
    }
}

/// One partition at a merge source's level, reached by its unique incoming
/// Child Entry during the exact root-down level walk: one candidate that
/// merge target reselection orders by routing distance (ADR 0008).
pub(crate) struct LevelCandidate {
    parent: PartitionKey,
    entry: ChildEntry,
    header: PartitionHeader,
}

impl LevelCandidate {
    /// Returns the candidate's Partition Key.
    pub(crate) const fn partition(&self) -> PartitionKey {
        self.entry.child()
    }

    /// Returns the partition holding the candidate's incoming Child Entry.
    pub(crate) const fn parent(&self) -> PartitionKey {
        self.parent
    }

    /// Returns the candidate's immutable full-f32 centroid projection.
    pub(crate) fn centroid(&self) -> &[f32] {
        self.entry.centroid()
    }

    /// Returns the candidate's decoded Header.
    pub(crate) const fn header(&self) -> &PartitionHeader {
        &self.header
    }

    /// Whether the candidate is a legal merge target for `source`: another
    /// partition currently `Ready` (ADR 0008).
    pub(crate) fn is_legal_merge_target(&self, source: PartitionKey) -> bool {
        self.partition() != source && self.header.state() == PartitionState::Ready
    }
}

/// Enumerates every partition at `level` with its incoming Child Entry and
/// decoded Header, in deterministic walk order.
///
/// This is the candidate set ADR 0008's same-level merge routing orders:
/// every non-root partition owns exactly one incoming Child Entry in one
/// level-`level + 1` body in every committed state, so the exact root-down
/// walk plus one bounded Child Entry scan per body is complete. Each
/// candidate's Header is batch-read in bounded chunks; a missing Header or a
/// level disagreement is Corruption. The caller filters states — the merge
/// protocol selects among `Ready` candidates only.
pub(crate) async fn same_level_candidates<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    level: u32,
) -> Result<Vec<LevelCandidate>> {
    let index = manifest.logical_index_id();
    let parent_level = level.checked_add(1).ok_or_else(corrupt)?;
    let bodies = level_bodies(reader, manifest, tree_key, parent_level).await?;

    let mut edges: Vec<(PartitionKey, ChildEntry)> = Vec::new();
    scan_child_edges(reader, manifest, tree_key, &bodies, &mut |body, entry| {
        edges.push((body, entry.clone()));
        Ok(())
    })
    .await?;

    // Candidate Headers are batch-read in bounded chunks.
    let mut headers = Vec::with_capacity(edges.len());
    for chunk in edges.chunks(DISCOVERY_SCAN_LIMITS.item_limit) {
        let keys: Vec<LogicalKey> = chunk
            .iter()
            .map(|(_, entry)| LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: entry.child(),
            })
            .collect();
        headers.extend(reader.batch_get(keys).await?);
    }

    let mut candidates = Vec::with_capacity(edges.len());
    for ((parent, entry), value) in edges.into_iter().zip(headers) {
        let header = expect_header(value)?.ok_or_else(corrupt)?;
        if header.level() != level {
            return Err(corrupt());
        }
        candidates.push(LevelCandidate {
            parent,
            entry,
            header,
        });
    }
    Ok(candidates)
}

#[cfg(test)]
#[path = "proof_tests.rs"]
mod proof_tests;
