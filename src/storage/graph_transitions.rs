//! Snapshot preparation and atomic graph publication for topology changes.
//!
//! The topology owner protects every source before editing headers or membership,
//! then includes graph records in the same commit. Native work runs only in
//! `prepare`, outside the transaction and under the caller's retained admission.

use super::backend::{ReadOps, ScanLimits, WriteTxn};
use super::keys::{LogicalKey, TreeKey};
use super::values::{
    ChildEntry, GraphChunk, GraphDescriptor, IndexLifecycle, IndexManifest, PartitionHeader,
    PersistentValue,
};
use super::{LogicalRange, MutationBuilder, ReadLogicalTxn, WriteLogicalTxn};
use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::local_hnsw::{
    Centroids,
    persistence::{GraphRecords, GraphSource, build_records, restore_graph, update_records},
};
use crate::search::numeric::VectorKernel;

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}
fn stale() -> Error {
    Error::new(ErrorKind::RetryableAbort)
}
fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}
// Covers bounded Manifest/schema clones, vector headers, descriptors, and plans.
const METADATA_BYTES: usize = 16 * 1024;
const PAGE: ScanLimits = ScanLimits {
    item_limit: 64,
    byte_limit: 256 * 1024,
};

/// A complete existing partition body, including graph-free empty/leaf bodies.
#[derive(Debug)]
pub struct GraphTransitionSource {
    manifest: IndexManifest,
    tree: TreeKey,
    partition: PartitionKey,
    header: Option<PartitionHeader>,
    descriptor: Option<GraphDescriptor>,
    children: Vec<ChildEntry>,
    chunks: Vec<(u32, GraphChunk)>,
}

impl GraphTransitionSource {
    /// Reads a bounded snapshot; an absent partition is a valid creation source.
    pub async fn read<T: ReadOps, C: Fn() -> Result<()> + ?Sized>(
        txn: &mut ReadLogicalTxn<'_, T>,
        tree: &TreeKey,
        partition: PartitionKey,
        working_bytes: usize,
        check: &C,
    ) -> Result<Self> {
        check()?;
        let manifest = txn.require_manifest()?.clone();
        if manifest.lifecycle() != IndexLifecycle::Active
            || manifest.config().local_graph().is_none()
        {
            return Err(stale());
        }
        let index = manifest.logical_index_id();
        let values = txn
            .batch_get(vec![
                LogicalKey::Manifest(index),
                LogicalKey::Header {
                    index,
                    tree_key: tree.clone(),
                    partition,
                },
                LogicalKey::GraphDescriptor {
                    index,
                    tree_key: tree.clone(),
                    partition,
                },
            ])
            .await?;
        let mut values = values.into_iter();
        if values.next().flatten() != Some(PersistentValue::IndexManifest(manifest.clone())) {
            return Err(stale());
        }
        let header = match values.next().flatten() {
            Some(PersistentValue::PartitionHeader(h)) => Some(h),
            None => None,
            _ => return Err(corrupt()),
        };
        let descriptor = match values.next().flatten() {
            Some(PersistentValue::GraphDescriptor(d)) => Some(d),
            None => None,
            _ => return Err(corrupt()),
        };
        let count = header
            .filter(|h| h.level() > 1)
            .map_or(0, |h| h.entry_count());
        if count > manifest.config().max_internal_partition_entries() + 2
            || (count > 0) != descriptor.is_some()
        {
            return Err(corrupt());
        }
        let entry_bytes = manifest
            .config()
            .dimension()
            .checked_mul(4)
            .and_then(|n| n.checked_add(std::mem::size_of::<ChildEntry>() + 64))
            .ok_or_else(limit)?;
        let retained = entry_bytes
            .checked_mul(count as usize)
            .and_then(|n| {
                n.checked_add(usize::try_from(descriptor.as_ref().map_or(0, |d| d.bytes)).ok()?)
            })
            .and_then(|n| n.checked_add(PAGE.byte_limit + tree.as_bytes().len() + METADATA_BYTES))
            .ok_or_else(limit)?;
        if retained > working_bytes {
            return Err(limit());
        }
        let mut children = Vec::new();
        children
            .try_reserve_exact(count as usize)
            .map_err(|_| limit())?;
        let mut cursor = None;
        let range = LogicalRange::child_entries(&manifest, tree, partition)?;
        loop {
            check()?;
            let (items, next) = txn.scan(&range, cursor.as_ref(), PAGE).await?.into_parts();
            for item in items {
                if children.len() == count as usize {
                    return Err(corrupt());
                }
                let PersistentValue::ChildEntry(child) = item.into_value() else {
                    return Err(corrupt());
                };
                children.push(child);
            }
            cursor = next;
            if cursor.is_none() {
                break;
            }
        }
        if children.len() != count as usize {
            return Err(corrupt());
        }
        let chunk_count = descriptor.as_ref().map_or(0, GraphDescriptor::chunks);
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(chunk_count as usize)
            .map_err(|_| limit())?;
        let range = LogicalRange::graph_chunks(&manifest, tree, partition)?;
        let mut cursor = None;
        loop {
            check()?;
            let (items, next) = txn.scan(&range, cursor.as_ref(), PAGE).await?.into_parts();
            for item in items {
                let LogicalKey::GraphChunk { chunk, .. } = item.key() else {
                    return Err(corrupt());
                };
                let ordinal = *chunk;
                if ordinal as usize != chunks.len() || ordinal >= chunk_count {
                    return Err(corrupt());
                }
                let PersistentValue::GraphChunk(chunk) = item.into_value() else {
                    return Err(corrupt());
                };
                chunks.push((ordinal, chunk));
            }
            cursor = next;
            if cursor.is_none() {
                break;
            }
        }
        if chunks.len() != chunk_count as usize {
            return Err(corrupt());
        }
        Ok(Self {
            manifest,
            tree: tree.clone(),
            partition,
            header,
            descriptor,
            children,
            chunks,
        })
    }

    /// Retained source bytes charged while preparing another partition.
    pub fn retained_bytes(&self) -> Result<usize> {
        self.projection_bytes(self.children.len())?
            .checked_add(self.descriptor.as_ref().map_or(0, |d| d.bytes as usize))
            .and_then(|n| {
                n.checked_add(self.chunks.len() * std::mem::size_of::<(u32, GraphChunk)>())
            })
            .and_then(|n| n.checked_add(self.tree.as_bytes().len() + METADATA_BYTES))
            .ok_or_else(limit)
    }

    fn projection_bytes(&self, count: usize) -> Result<usize> {
        self.manifest
            .config()
            .dimension()
            .checked_mul(4)
            .and_then(|n| n.checked_add(std::mem::size_of::<ChildEntry>() + 64))
            .and_then(|n| n.checked_mul(count))
            .ok_or_else(limit)
    }

    /// Prepares all participant graphs under one retained-memory admission.
    /// Pending sources/proposals and completed outputs reduce each native budget.
    pub fn prepare_batch(
        mut targets: Vec<(Self, Option<PartitionHeader>, Vec<ChildEntry>)>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<PreparedGraphTransition>> {
        let mut pending = targets
            .iter()
            .try_fold(0usize, |total, (source, _, next)| {
                total
                    .checked_add(source.retained_bytes()?)
                    .and_then(|n| n.checked_add(source.projection_bytes(next.len()).ok()?))
                    .ok_or_else(limit)
            })?;
        if pending > working_bytes {
            return Err(limit());
        }
        let mut retained = 0usize;
        let mut prepared = Vec::new();
        prepared
            .try_reserve_exact(targets.len())
            .map_err(|_| limit())?;
        while let Some((source, header, next)) = targets.pop() {
            check()?;
            let current = source
                .retained_bytes()?
                .checked_add(source.projection_bytes(next.len())?)
                .ok_or_else(limit)?;
            pending = pending.checked_sub(current).ok_or_else(limit)?;
            let available = working_bytes
                .checked_sub(pending)
                .and_then(|n| n.checked_sub(retained))
                .ok_or_else(limit)?;
            let output = source.prepare(header, next, available, check)?;
            retained = retained
                .checked_add(output.retained_bytes()?)
                .ok_or_else(limit)?;
            if pending.checked_add(retained).ok_or_else(limit)? > working_bytes {
                return Err(limit());
            }
            prepared.push(output);
        }
        prepared.reverse();
        Ok(prepared)
    }

    /// Partition whose body is replaced by this transition.
    pub(crate) fn partition(&self) -> PartitionKey {
        self.partition
    }

    /// Source header, absent for a newly allocated partition.
    pub fn header(&self) -> Option<PartitionHeader> {
        self.header
    }
    /// Ordered authoritative child projections in the source snapshot.
    pub fn children(&self) -> &[ChildEntry] {
        &self.children
    }

    /// Prepares the exact final internal body, or validates and removes an old
    /// graph when its owner becomes empty, leaf, or absent.
    pub fn prepare(
        self,
        next_header: Option<PartitionHeader>,
        next: Vec<ChildEntry>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedGraphTransition> {
        check()?;
        let config = self.manifest.config();
        let capacity = config.max_internal_partition_entries() as usize + 2;
        let kernel = VectorKernel::new(
            config.dimension(),
            config.metric(),
            *self.manifest.rotation_seed(),
        )?;
        let projection_bytes = config
            .dimension()
            .checked_mul(4)
            .and_then(|n| n.checked_add(std::mem::size_of::<ChildEntry>() + 64))
            .ok_or_else(limit)?;
        let old_bytes = projection_bytes
            .checked_mul(self.children.len())
            .ok_or_else(limit)?;
        let target_bytes = projection_bytes.checked_mul(next.len()).ok_or_else(limit)?;
        let available = working_bytes
            .checked_sub(
                kernel.heap_bytes()
                    + self.tree.as_bytes().len()
                    + self.chunks.len() * std::mem::size_of::<(u32, GraphChunk)>()
                    + METADATA_BYTES,
            )
            .ok_or_else(limit)?;
        let old = Centroids::new(&kernel, &self.children, capacity, check).map_err(|e| {
            if e.kind() == ErrorKind::InvalidArgument {
                corrupt()
            } else {
                e
            }
        })?;
        let target_count = next_header
            .filter(|h| h.level() > 1)
            .map_or(0, |h| h.entry_count()) as usize;
        if target_count != next.len()
            || next
                .iter()
                .any(|c| c.child() == self.partition || c.child().get() == 1)
        {
            return Err(Error::invalid_argument());
        }
        let target = Centroids::new(&kernel, &next, capacity, check)?;
        let parameters = config.local_graph().ok_or_else(corrupt)?.parameters();
        let unchanged = self
            .header
            .zip(next_header)
            .is_some_and(|(a, b)| a.level() == b.level() && a.cache_epoch() == b.cache_epoch())
            && self.children == next;
        if let Some((old_header, new_header)) = self.header.zip(next_header)
            && !unchanged
            && (old_header.level() > 1 || new_header.level() > 1)
            && new_header.cache_epoch() <= old_header.cache_epoch()
        {
            return Err(Error::invalid_argument());
        }
        let prepared = if let Some(descriptor) = &self.descriptor {
            let source = GraphSource {
                index: self.manifest.logical_index_id(),
                tree: &self.tree,
                parent: self.partition,
                header: self.header.ok_or_else(corrupt)?,
                parameters,
            };
            if target_count > 0
                && !unchanged
                && next_header.is_some_and(|header| header.level() == source.header.level())
            {
                Some(update_records(
                    old,
                    target,
                    source,
                    descriptor,
                    self.chunks,
                    next_header.ok_or_else(corrupt)?,
                    available,
                    check,
                )?)
            } else {
                drop(restore_graph(
                    old,
                    source,
                    descriptor,
                    self.chunks,
                    available.checked_sub(target_bytes).ok_or_else(limit)?,
                    check,
                )?);
                if target_count > 0 && !unchanged {
                    Some(build_records(
                        target,
                        GraphSource {
                            header: next_header.ok_or_else(corrupt)?,
                            ..source
                        },
                        available.checked_sub(old_bytes).ok_or_else(limit)?,
                        check,
                    )?)
                } else {
                    None
                }
            }
        } else if target_count > 0 {
            Some(build_records(
                target,
                GraphSource {
                    index: self.manifest.logical_index_id(),
                    tree: &self.tree,
                    parent: self.partition,
                    header: next_header.ok_or_else(corrupt)?,
                    parameters,
                },
                available.checked_sub(old_bytes).ok_or_else(limit)?,
                check,
            )?)
        } else {
            None
        };
        let publication = if let Some(prepared) = prepared {
            let body = crate::search::local_hnsw::body::InternalGraphBody::prepared(
                next,
                kernel,
                prepared.graph,
                prepared.identity,
                prepared.records.descriptor.clone(),
                target_bytes,
            );
            GraphPublication::Replace {
                header: next_header.ok_or_else(corrupt)?,
                records: prepared.records,
                body,
            }
        } else if unchanged {
            GraphPublication::Preserve {
                header: next_header.ok_or_else(corrupt)?,
            }
        } else {
            GraphPublication::Remove {
                header: next_header,
            }
        };
        Ok(PreparedGraphTransition {
            manifest: self.manifest,
            tree: self.tree,
            partition: self.partition,
            source_header: self.header,
            source_descriptor: self.descriptor,
            publication,
        })
    }
}

/// Complete graph action after source validation. A replacement owns both its
/// persistent records and the exact body that can enter the cache after commit.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "bounded per-body publication retains its replacement without another allocation"
)]
enum GraphPublication {
    Preserve {
        header: PartitionHeader,
    },
    Remove {
        header: Option<PartitionHeader>,
    },
    Replace {
        header: PartitionHeader,
        records: GraphRecords,
        body: crate::search::local_hnsw::body::InternalGraphBody,
    },
}

impl GraphPublication {
    fn header(&self) -> Option<PartitionHeader> {
        match self {
            Self::Remove { header } => *header,
            Self::Preserve { header } | Self::Replace { header, .. } => Some(*header),
        }
    }
}

/// Private graph publication; no persisted staging or recovery cursor exists.
#[derive(Debug)]
pub struct PreparedGraphTransition {
    manifest: IndexManifest,
    tree: TreeKey,
    partition: PartitionKey,
    source_header: Option<PartitionHeader>,
    source_descriptor: Option<GraphDescriptor>,
    publication: GraphPublication,
}

/// Protects every prepared source before the topology owner edits its bodies.
/// A changed source requires rediscovery instead of publishing the stale batch.
pub async fn protect_sources<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    graphs: &[PreparedGraphTransition],
) -> Result<bool> {
    for graph in graphs {
        if !graph.protect(txn).await? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Publishes graphs after source protection and the topology owner's edits.
/// Validates all final headers and admits the combined authority envelope before
/// appending payloads; the caller commits topology and graphs atomically.
pub async fn publish_batch<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    graphs: &[PreparedGraphTransition],
) -> Result<()> {
    let mut builder = txn.mutations();
    for graph in graphs {
        graph.validate_target(txn).await?;
        graph.append_authority(&mut builder)?;
    }
    if txn
        .affected_bytes()
        .checked_add(builder.affected_bytes()?)
        .ok_or_else(limit)?
        > 900_000
    {
        return Err(limit());
    }
    for graph in graphs {
        graph.append_payload(&mut builder)?;
    }
    txn.apply(builder).await
}

impl PreparedGraphTransition {
    /// Called only after the enclosing topology transaction is known committed.
    pub(crate) fn cache_committed(self, cache: &crate::search::cache::PartitionCache) {
        if let GraphPublication::Replace { body, header, .. } = self.publication {
            crate::search::cache::graph::install_prepared(
                cache,
                &self.manifest,
                self.tree,
                self.partition,
                header,
                body,
            );
        }
    }

    /// Retained encoded output and cache body charged while preparing another partition.
    pub fn retained_bytes(&self) -> Result<usize> {
        let replacement_bytes = match &self.publication {
            GraphPublication::Replace { records, body, .. } => (records.descriptor.bytes as usize)
                .checked_add(body.retained_bytes())
                .ok_or_else(limit)?,
            GraphPublication::Preserve { .. } | GraphPublication::Remove { .. } => 0,
        };
        replacement_bytes
            .checked_add(self.tree.as_bytes().len() + METADATA_BYTES)
            .ok_or_else(limit)
    }

    /// Establishes conflicts on the exact old body before topology edits.
    async fn protect<T: WriteTxn>(&self, txn: &mut WriteLogicalTxn<'_, T>) -> Result<bool> {
        let index = self.manifest.logical_index_id();
        let values = txn
            .batch_get_for_update(vec![
                LogicalKey::Manifest(index),
                LogicalKey::Header {
                    index,
                    tree_key: self.tree.clone(),
                    partition: self.partition,
                },
                LogicalKey::GraphDescriptor {
                    index,
                    tree_key: self.tree.clone(),
                    partition: self.partition,
                },
            ])
            .await?;
        let mut values = values.into_iter();
        if values.next().flatten() != Some(PersistentValue::IndexManifest(self.manifest.clone())) {
            return Err(stale());
        }
        let header = match values.next().flatten() {
            Some(PersistentValue::PartitionHeader(h)) => Some(h),
            None => None,
            _ => return Err(corrupt()),
        };
        let descriptor = match values.next().flatten() {
            Some(PersistentValue::GraphDescriptor(d)) => Some(d),
            None => None,
            _ => return Err(corrupt()),
        };
        Ok(header == self.source_header && descriptor == self.source_descriptor)
    }

    /// Verifies the topology owner's final header before shared graph writes.
    /// Call after `protect` and topology edits, before appending any graph records.
    /// The protected Manifest and descriptor remain unchanged by topology edits.
    async fn validate_target<T: WriteTxn>(&self, txn: &mut WriteLogicalTxn<'_, T>) -> Result<()> {
        let header = match txn
            .get_for_update(LogicalKey::Header {
                index: self.manifest.logical_index_id(),
                tree_key: self.tree.clone(),
                partition: self.partition,
            })
            .await?
        {
            Some(PersistentValue::PartitionHeader(header)) => Some(header),
            None => None,
            _ => return Err(corrupt()),
        };
        if header != self.publication.header() {
            return Err(stale());
        }
        Ok(())
    }

    /// Adds descriptor and obsolete-tail edits to a shared authority envelope.
    /// Batch publication checks the combined authority envelope before payloads.
    fn append_authority(&self, builder: &mut MutationBuilder<'_>) -> Result<()> {
        let records = match &self.publication {
            GraphPublication::Preserve { .. } => return Ok(()),
            GraphPublication::Remove { .. } => None,
            GraphPublication::Replace { records, .. } => Some(records),
        };
        let index = self.manifest.logical_index_id();
        let key = LogicalKey::GraphDescriptor {
            index,
            tree_key: self.tree.clone(),
            partition: self.partition,
        };
        if let Some(records) = records {
            builder.put(
                key,
                PersistentValue::GraphDescriptor(records.descriptor.clone()),
            )?;
        } else if self.source_descriptor.is_some() {
            builder.delete(key)?;
        }
        for chunk in records.map_or(0, |r| r.descriptor.chunks())
            ..self
                .source_descriptor
                .as_ref()
                .map_or(0, GraphDescriptor::chunks)
        {
            builder.delete(LogicalKey::GraphChunk {
                index,
                tree_key: self.tree.clone(),
                partition: self.partition,
                chunk,
            })?;
        }
        Ok(())
    }

    /// Adds prepared chunk payloads after the shared envelope has been checked.
    fn append_payload(&self, builder: &mut MutationBuilder<'_>) -> Result<()> {
        let index = self.manifest.logical_index_id();
        if let GraphPublication::Replace { records, .. } = &self.publication {
            for (chunk, value) in records.chunks() {
                builder.put(
                    LogicalKey::GraphChunk {
                        index,
                        tree_key: self.tree.clone(),
                        partition: self.partition,
                        chunk,
                    },
                    PersistentValue::GraphChunk(value),
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{IndexConfig, LocalGraphConfig, LogicalIndexId, Metric};
    const MEMORY: usize = 16 * 1024 * 1024;

    fn source() -> GraphTransitionSource {
        GraphTransitionSource {
            manifest: IndexManifest::new(
                IndexLifecycle::Active,
                LogicalIndexId::new(1).unwrap(),
                IndexConfig::new(2, Metric::L2)
                    .unwrap()
                    .with_internal_partition_entries(2, 16)
                    .unwrap()
                    .with_local_graph(LocalGraphConfig::new(4, 16, [3; 32]).unwrap()),
                [0; 32],
                vec![],
            )
            .unwrap(),
            tree: TreeKey::encode(&[], &[]).unwrap(),
            partition: PartitionKey::new(1).unwrap(),
            header: None,
            descriptor: None,
            children: vec![],
            chunks: vec![],
        }
    }

    fn header(
        count: u32,
        epoch: u64,
        state: super::super::values::PartitionState,
    ) -> PartitionHeader {
        PartitionHeader::new(2, count, epoch, state).unwrap()
    }

    fn rows() -> Vec<ChildEntry> {
        (2..6)
            .map(|n| ChildEntry::new(PartitionKey::new(n).unwrap(), vec![n as f32, 1.0]))
            .collect()
    }

    fn populated() -> GraphTransitionSource {
        let old = source();
        let next = rows();
        let prepared = old
            .prepare(
                Some(header(4, 5, super::super::values::PartitionState::Ready)),
                next.clone(),
                MEMORY,
                &|| Ok(()),
            )
            .unwrap();
        let GraphPublication::Replace {
            header, records, ..
        } = prepared.publication
        else {
            panic!("expected graph replacement");
        };
        let chunks = records.chunks().collect();
        GraphTransitionSource {
            header: Some(header),
            descriptor: Some(records.descriptor),
            children: next,
            chunks,
            ..source()
        }
    }

    #[test]
    fn committed_prepared_graph_matches_restore_and_respects_cache_epochs() {
        use super::super::values::PartitionState;
        use crate::search::cache::{BodyEntries, CacheKey, PartitionCache, PartitionKind};
        use crate::search::local_hnsw::body::InternalGraphBody;
        let manifest = source().manifest;
        let tree = source().tree;
        let parent = source().partition;
        let key = CacheKey::new(
            manifest.logical_index_id(),
            tree.clone(),
            parent,
            PartitionKind::Internal,
        );
        let prepare = |epoch| {
            source()
                .prepare(
                    Some(header(4, epoch, PartitionState::Ready)),
                    rows(),
                    MEMORY,
                    &|| Ok(()),
                )
                .unwrap()
        };
        let prepared = prepare(9);
        let GraphPublication::Replace { records, body, .. } = &prepared.publication else {
            panic!("expected graph replacement");
        };
        assert!(prepared.retained_bytes().unwrap() >= body.retained_bytes());
        let kernel =
            VectorKernel::new(2, manifest.config().metric(), *manifest.rotation_seed()).unwrap();
        let restored = InternalGraphBody::restore(
            rows().into_boxed_slice(),
            kernel,
            GraphSource {
                index: manifest.logical_index_id(),
                tree: &tree,
                parent,
                header: header(4, 9, PartitionState::Ready),
                parameters: manifest.config().local_graph().unwrap().parameters(),
            },
            records.descriptor.clone(),
            records.chunks().collect(),
            MEMORY,
            &|| Ok(()),
        )
        .unwrap();
        let cache = PartitionCache::new(MEMORY as u64);
        assert!(cache.lookup(&key, 9).is_none());
        // A discarded/failed transition has no publication side effect.
        drop(prepare(8));
        assert!(cache.lookup(&key, 8).is_none());
        prepared.cache_committed(&cache);
        let body = cache.lookup(&key, 9).unwrap();
        let BodyEntries::InternalGraph(graph) = body.entries() else {
            panic!("expected prepared native topology")
        };
        assert_eq!(graph.children(), restored.children());
        for query in [[2.0, 1.0], [7.0, 3.0]] {
            let mut a = ktann_usearch::Workspace::new(4, 4, MEMORY).unwrap();
            let mut b = ktann_usearch::Workspace::new(4, 4, MEMORY).unwrap();
            let (a, sa) = graph
                .search_batched(&query, 2, 4, MEMORY, &mut a, &|| Ok(()))
                .unwrap();
            let (b, sb) = restored
                .search_batched(&query, 2, 4, MEMORY, &mut b, &|| Ok(()))
                .unwrap();
            assert_eq!(a, b);
            assert_eq!(sa.distances, sb.distances);
        }
        prepare(8).cache_committed(&cache);
        assert!(cache.lookup(&key, 8).is_none());
        assert!(cache.lookup(&key, 9).is_some());
        let disabled = PartitionCache::new(0);
        prepare(9).cache_committed(&disabled);
        assert!(disabled.lookup(&key, 9).is_none());
    }

    #[test]
    fn creates_advances_many_epochs_and_removes_empty_graph() {
        let old = populated();
        let mut next = rows();
        next[0] = ChildEntry::new(next[0].child(), vec![9.0, 1.0]);
        let update = old
            .prepare(
                Some(header(4, 9, super::super::values::PartitionState::Ready)),
                next,
                MEMORY,
                &|| Ok(()),
            )
            .unwrap();
        let GraphPublication::Replace { records, .. } = update.publication else {
            panic!("expected graph replacement");
        };
        assert_eq!(records.descriptor.epoch, 9);
        let empty = populated()
            .prepare(
                Some(header(0, 10, super::super::values::PartitionState::Ready)),
                vec![],
                MEMORY,
                &|| Ok(()),
            )
            .unwrap();
        assert!(matches!(empty.publication, GraphPublication::Remove { .. }));
        assert!(empty.source_descriptor.is_some());
    }

    #[test]
    fn corrupt_graph_cannot_be_hidden_by_empty_or_deleted_target() {
        let mut old = populated();
        old.chunks[0].1 = GraphChunk(bytes::Bytes::from_static(b"corrupt"));
        assert_eq!(
            old.prepare(None, vec![], MEMORY, &|| Ok(()))
                .unwrap_err()
                .kind(),
            ErrorKind::Corruption
        );
    }

    #[test]
    fn state_only_change_preserves_exact_graph_body() {
        let old = populated();
        let update = old
            .prepare(
                Some(header(
                    4,
                    5,
                    super::super::values::PartitionState::DrainingSplit,
                )),
                rows(),
                MEMORY,
                &|| Ok(()),
            )
            .unwrap();
        assert!(matches!(
            update.publication,
            GraphPublication::Preserve { .. }
        ));
    }

    #[test]
    fn source_and_output_memory_are_admitted() {
        assert_eq!(
            populated()
                .prepare(None, vec![], 1, &|| Ok(()))
                .unwrap_err()
                .kind(),
            ErrorKind::LimitExceeded
        );
    }

    #[tokio::test]
    async fn snapshot_read_accepts_absent_and_empty_internal_bodies() {
        use super::super::test_support::MockReadTxn;
        use super::super::values::ValueCodec;
        let source = source();
        let codec = ValueCodec::for_index(&source.manifest);
        let index = source.manifest.logical_index_id();
        let manifest_key = LogicalKey::Manifest(index);
        let manifest_row = (
            super::super::keys::encode_key(&manifest_key).unwrap(),
            codec
                .encode_for_key(
                    &manifest_key,
                    &PersistentValue::IndexManifest(source.manifest.clone()),
                )
                .unwrap(),
        );
        let mut txn =
            ReadLogicalTxn::for_index(MockReadTxn::new([manifest_row.clone()]), &source.manifest);
        let absent =
            GraphTransitionSource::read(&mut txn, &source.tree, source.partition, MEMORY, &|| {
                Ok(())
            })
            .await
            .unwrap();
        assert!(absent.header().is_none());
        let empty_header = header(0, 3, super::super::values::PartitionState::ReceivingSplit);
        let header_key = LogicalKey::Header {
            index,
            tree_key: source.tree.clone(),
            partition: source.partition,
        };
        let header_row = (
            super::super::keys::encode_key(&header_key).unwrap(),
            codec
                .encode_for_key(&header_key, &PersistentValue::PartitionHeader(empty_header))
                .unwrap(),
        );
        let mut txn = ReadLogicalTxn::for_index(
            MockReadTxn::new([manifest_row, header_row]),
            &source.manifest,
        );
        let empty =
            GraphTransitionSource::read(&mut txn, &source.tree, source.partition, MEMORY, &|| {
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(empty.header(), Some(empty_header));
        assert!(empty.children().is_empty());
    }
}
