//! Snapshot preparation and atomic graph publication for topology changes.
//!
//! The topology owner protects every source before editing headers or membership,
//! then includes graph records in the same commit. Native work runs only in
//! `prepare`, outside the transaction and under the caller's retained admission.

use std::fmt;
use std::sync::Arc;

use super::backend::{ReadOps, ScanLimits, WriteTxn};
use super::keys::{LogicalKey, TreeKey};
use super::values::{
    ChildEntry, GraphChunk, GraphDescriptor, IndexLifecycle, IndexManifest, PartitionHeader,
    PartitionState, PersistentValue,
};
use super::{LogicalRange, MutationBuilder, ReadLogicalTxn, WriteLogicalTxn};
use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::cache::{BodyEntries, CacheKey, CachedBody, PartitionCache, PartitionKind};
use crate::search::local_hnsw::{
    Centroids,
    persistence::{
        GraphRecords, GraphSource, admit_cached_source, build_records, restore_graph,
        update_cached_records, update_records, validate_descriptor_envelope,
    },
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

enum SourceBody {
    Loaded {
        children: Vec<ChildEntry>,
        chunks: Vec<(u32, GraphChunk)>,
    },
    Cached(Arc<CachedBody>),
}

impl SourceBody {
    fn children(&self) -> &[ChildEntry] {
        match self {
            Self::Loaded { children, .. } => children,
            Self::Cached(body) => match body.entries() {
                BodyEntries::InternalGraph(graph) => graph.children(),
                _ => unreachable!("validated internal graph cache entry"),
            },
        }
    }

    fn chunk_count(&self, descriptor: Option<&GraphDescriptor>) -> usize {
        match self {
            Self::Loaded { chunks, .. } => chunks.len(),
            Self::Cached(_) => descriptor.map_or(0, GraphDescriptor::chunks) as usize,
        }
    }
}

impl fmt::Debug for SourceBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loaded { children, chunks } => formatter
                .debug_struct("Loaded")
                .field("children", &children.len())
                .field("chunks", &chunks.len())
                .finish(),
            Self::Cached(body) => formatter
                .debug_struct("Cached")
                .field("children", &self.children().len())
                .field("heap_bytes", &body.heap_bytes())
                .finish(),
        }
    }
}

async fn scan_graph_chunks<T: ReadOps, C: Fn() -> Result<()> + ?Sized>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree: &TreeKey,
    partition: PartitionKey,
    chunk_count: u32,
    check: &C,
) -> Result<Vec<(u32, GraphChunk)>> {
    let mut chunks = Vec::new();
    chunks
        .try_reserve_exact(chunk_count as usize)
        .map_err(|_| limit())?;
    let range = LogicalRange::graph_chunks(manifest, tree, partition)?;
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
    Ok(chunks)
}

/// A complete existing partition body, including graph-free empty/leaf bodies.
#[derive(Debug)]
pub struct GraphTransitionSource {
    manifest: IndexManifest,
    tree: TreeKey,
    partition: PartitionKey,
    header: Option<PartitionHeader>,
    descriptor: Option<GraphDescriptor>,
    body: SourceBody,
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
        Self::read_cached(txn, None, tree, partition, working_bytes, check).await
    }

    /// Reads a bounded snapshot, reusing a snapshot-validated cached internal
    /// body when its epoch and descriptor binding match `txn` without changing
    /// working-memory admission.
    pub(crate) async fn read_cached<T: ReadOps, C: Fn() -> Result<()> + ?Sized>(
        txn: &mut ReadLogicalTxn<'_, T>,
        cache: Option<&PartitionCache>,
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
        if count > 0
            && let Some(cache) = cache
        {
            let header_val = header.ok_or_else(corrupt)?;
            let descriptor_val = descriptor.as_ref().ok_or_else(corrupt)?;
            let parameters = manifest
                .config()
                .local_graph()
                .ok_or_else(corrupt)?
                .parameters();
            let source = GraphSource {
                index,
                tree,
                parent: partition,
                header: header_val,
                parameters,
            };
            validate_descriptor_envelope(source, count as usize, descriptor_val)?;
            let key = CacheKey::new(index, tree.clone(), partition, PartitionKind::Internal);
            if let Some(cached) = cache.lookup(&key, header_val.cache_epoch()) {
                match cached.entries() {
                    BodyEntries::InternalGraph(graph) => {
                        if graph.children().len() != count as usize
                            || !graph.matches(source, descriptor_val)
                        {
                            return Err(corrupt());
                        }
                        return Ok(Self {
                            manifest,
                            tree: tree.clone(),
                            partition,
                            header,
                            descriptor,
                            body: SourceBody::Cached(cached),
                        });
                    }
                    BodyEntries::Internal(cached_children) => {
                        if cached_children.len() != count as usize {
                            return Err(corrupt());
                        }
                        let mut children = Vec::new();
                        children
                            .try_reserve_exact(cached_children.len())
                            .map_err(|_| limit())?;
                        children.extend_from_slice(cached_children);
                        let chunks = scan_graph_chunks(
                            txn,
                            &manifest,
                            tree,
                            partition,
                            descriptor_val.chunks(),
                            check,
                        )
                        .await?;
                        return Ok(Self {
                            manifest,
                            tree: tree.clone(),
                            partition,
                            header,
                            descriptor,
                            body: SourceBody::Loaded { children, chunks },
                        });
                    }
                    BodyEntries::Leaf { .. } => return Err(corrupt()),
                }
            }
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
        let chunks = scan_graph_chunks(txn, &manifest, tree, partition, chunk_count, check).await?;
        Ok(Self {
            manifest,
            tree: tree.clone(),
            partition,
            header,
            descriptor,
            body: SourceBody::Loaded { children, chunks },
        })
    }

    /// Retained source bytes charged while preparing another partition.
    pub fn retained_bytes(&self) -> Result<usize> {
        let chunk_count = self.body.chunk_count(self.descriptor.as_ref());
        self.projection_bytes(self.children().len())?
            .checked_add(self.descriptor.as_ref().map_or(0, |d| d.bytes as usize))
            .and_then(|n| n.checked_add(chunk_count * std::mem::size_of::<(u32, GraphChunk)>()))
            .and_then(|n| n.checked_add(self.tree.as_bytes().len() + METADATA_BYTES))
            .ok_or_else(limit)
    }

    /// Admission charge for a retained proposal's centroid projections.
    pub(crate) fn projection_bytes(&self, count: usize) -> Result<usize> {
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
        self.body.children()
    }

    /// Prepares the exact final internal body, or validates and removes an old
    /// graph when its owner becomes empty, leaf, or absent.
    /// At the same level, a topology step only inserts children or removes them;
    /// surviving children keep their centroids. A level change rebuilds the graph.
    pub fn prepare(
        self,
        next_header: Option<PartitionHeader>,
        next: Vec<ChildEntry>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedGraphTransition> {
        self.prepare_inner(next_header, next, working_bytes, check, false)
    }

    /// Publishes new routing models after validating the old persisted graph.
    /// Coordinate changes rebuild topology; scalar-only changes rebind the existing
    /// geometric graph to the new parent epoch without retraining its edges.
    /// The caller admits the update copies and replacement Child Entries before allocation.
    pub(crate) fn prepare_centroid_update(
        self,
        updates: Vec<(PartitionKey, crate::storage::values::PartitionCentroid)>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedGraphTransition> {
        check()?;
        let mut header = self.header.ok_or_else(corrupt)?;
        if header.state() != PartitionState::Ready {
            return Err(Error::invalid_argument());
        }
        let mut next = self.children().to_vec();
        let mut geometry_changed = false;
        for (child, center) in updates {
            check()?;
            let position = next
                .binary_search_by_key(&child, ChildEntry::child)
                .map_err(|_| corrupt())?;
            if next[position].model() != &center {
                geometry_changed |= next[position].centroid() != center.components();
                header = super::membership::touched_entry(header)?;
                next[position] = ChildEntry::from_centroid(child, center);
            }
        }
        self.prepare_inner(Some(header), next, working_bytes, check, geometry_changed)
    }

    fn prepare_inner(
        self,
        next_header: Option<PartitionHeader>,
        next: Vec<ChildEntry>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
        refresh_centroids: bool,
    ) -> Result<PreparedGraphTransition> {
        check()?;
        let old_bytes = self.projection_bytes(self.children().len())?;
        let target_bytes = self.projection_bytes(next.len())?;
        let chunk_count = self.body.chunk_count(self.descriptor.as_ref());
        let Self {
            manifest,
            tree,
            partition,
            header,
            descriptor,
            body,
        } = self;
        let (loaded_children, loaded_chunks, cached_body) = match body {
            SourceBody::Loaded { children, chunks } => (children, chunks, None),
            SourceBody::Cached(cached) => (Vec::new(), Vec::new(), Some(cached)),
        };
        let cached_graph = match cached_body.as_ref().map(|body| body.entries()) {
            Some(BodyEntries::InternalGraph(graph)) => Some(graph),
            Some(_) => unreachable!("validated internal graph cache entry"),
            None => None,
        };
        let old_children = cached_graph.map_or(loaded_children.as_slice(), |g| g.children());
        let config = manifest.config();
        let capacity = config.max_internal_partition_entries() as usize + 2;
        let kernel = VectorKernel::new(
            config.dimension(),
            config.metric(),
            *manifest.rotation_seed(),
        )?;
        let available = working_bytes
            .checked_sub(
                kernel.heap_bytes()
                    + tree.as_bytes().len()
                    + chunk_count * std::mem::size_of::<(u32, GraphChunk)>()
                    + METADATA_BYTES,
            )
            .ok_or_else(limit)?;
        let old = Centroids::new(&kernel, old_children, capacity, check).map_err(|e| {
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
                .any(|c| c.child() == partition || c.child().get() == 1)
        {
            return Err(Error::invalid_argument());
        }
        let target = Centroids::new(&kernel, &next, capacity, check)?;
        let parameters = config.local_graph().ok_or_else(corrupt)?.parameters();
        let unchanged = header
            .zip(next_header)
            .is_some_and(|(a, b)| a.level() == b.level() && a.cache_epoch() == b.cache_epoch())
            && old_children == next.as_slice();
        if let Some((old_header, new_header)) = header.zip(next_header)
            && !unchanged
            && (old_header.level() > 1 || new_header.level() > 1)
            && new_header.cache_epoch() <= old_header.cache_epoch()
        {
            return Err(Error::invalid_argument());
        }
        let prepared = if let Some(descriptor) = &descriptor {
            let source = GraphSource {
                index: manifest.logical_index_id(),
                tree: &tree,
                parent: partition,
                header: header.ok_or_else(corrupt)?,
                parameters,
            };
            if target_count > 0
                && !unchanged
                && !refresh_centroids
                && next_header.is_some_and(|header| header.level() == source.header.level())
            {
                let next_header = next_header.ok_or_else(corrupt)?;
                let updated = match cached_graph {
                    Some(graph) => update_cached_records(
                        old,
                        target,
                        source,
                        descriptor,
                        graph,
                        next_header,
                        available,
                        check,
                    )?,
                    None => update_records(
                        old,
                        target,
                        source,
                        descriptor,
                        loaded_chunks,
                        next_header,
                        available,
                        check,
                    )?,
                };
                Some(updated)
            } else {
                let verify_budget = available.checked_sub(target_bytes).ok_or_else(limit)?;
                match cached_graph {
                    Some(graph) => {
                        admit_cached_source(old, source, descriptor, graph, verify_budget, check)?;
                    }
                    None => {
                        drop(restore_graph(
                            old,
                            source,
                            descriptor,
                            loaded_chunks,
                            verify_budget,
                            check,
                        )?);
                    }
                }
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
                    index: manifest.logical_index_id(),
                    tree: &tree,
                    parent: partition,
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
            manifest,
            tree,
            partition,
            source_header: header,
            source_descriptor: descriptor,
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
            body: SourceBody::Loaded {
                children: vec![],
                chunks: vec![],
            },
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
            body: SourceBody::Loaded {
                children: next,
                chunks,
            },
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
        let next = rows();
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
        let SourceBody::Loaded { chunks, .. } = &mut old.body else {
            panic!("expected encoded fixture");
        };
        chunks[0].1 = GraphChunk(bytes::Bytes::from_static(b"corrupt"));
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

    #[tokio::test]
    async fn warm_cache_hit_skips_body_scans_matches_cold_preparation_and_fails_closed() {
        use super::super::keys::encode_key;
        use super::super::test_support::MockReadTxn;
        use super::super::values::{PartitionState, ValueCodec};

        let base = source();
        let codec = ValueCodec::for_index(&base.manifest);
        let index = base.manifest.logical_index_id();
        let initial_rows = rows();
        let initial_header = header(4, 5, PartitionState::Ready);
        let initial = source()
            .prepare(Some(initial_header), initial_rows.clone(), MEMORY, &|| {
                Ok(())
            })
            .unwrap();
        let GraphPublication::Replace { records, .. } = &initial.publication else {
            panic!("expected initial graph replacement");
        };
        let descriptor = records.descriptor.clone();
        let mut kv_rows = vec![
            (
                LogicalKey::Manifest(index),
                PersistentValue::IndexManifest(base.manifest.clone()),
            ),
            (
                LogicalKey::Header {
                    index,
                    tree_key: base.tree.clone(),
                    partition: base.partition,
                },
                PersistentValue::PartitionHeader(initial_header),
            ),
            (
                LogicalKey::GraphDescriptor {
                    index,
                    tree_key: base.tree.clone(),
                    partition: base.partition,
                },
                PersistentValue::GraphDescriptor(descriptor.clone()),
            ),
        ];
        for child in &initial_rows {
            kv_rows.push((
                LogicalKey::ChildEntry {
                    index,
                    tree_key: base.tree.clone(),
                    partition: base.partition,
                    child: child.child(),
                },
                PersistentValue::ChildEntry(child.clone()),
            ));
        }
        for (ordinal, chunk) in records.chunks() {
            kv_rows.push((
                LogicalKey::GraphChunk {
                    index,
                    tree_key: base.tree.clone(),
                    partition: base.partition,
                    chunk: ordinal,
                },
                PersistentValue::GraphChunk(chunk),
            ));
        }
        let encode_rows = |items: &[(LogicalKey, PersistentValue)]| {
            items
                .iter()
                .map(|(k, v)| (encode_key(k).unwrap(), codec.encode_for_key(k, v).unwrap()))
                .collect::<Vec<_>>()
        };
        let encoded_rows = encode_rows(&kv_rows);

        let cache = PartitionCache::new(MEMORY as u64);
        initial.cache_committed(&cache);

        // Warm `InternalGraph` hit performs zero scans (`with_failing_scans` tripwire)
        // and produces identical insertion and deletion outputs as cold reads.
        let mut added = initial_rows.clone();
        added.push(ChildEntry::new(
            PartitionKey::new(6).unwrap(),
            vec![6.0, 1.0],
        ));
        added.push(ChildEntry::new(
            PartitionKey::new(7).unwrap(),
            vec![7.0, 1.0],
        ));
        let mut removed = initial_rows.clone();
        removed.remove(1);

        for (next_header, next_children) in [
            (header(6, 6, PartitionState::Ready), added),
            (header(3, 6, PartitionState::Ready), removed),
        ] {
            let mut warm_txn = ReadLogicalTxn::for_index(
                MockReadTxn::new(encoded_rows.clone()).with_failing_scans(),
                &base.manifest,
            );
            let warm = GraphTransitionSource::read_cached(
                &mut warm_txn,
                Some(&cache),
                &base.tree,
                base.partition,
                MEMORY,
                &|| Ok(()),
            )
            .await
            .unwrap();
            assert_eq!(warm_txn.into_raw().scans, 0);

            let mut cold_txn =
                ReadLogicalTxn::for_index(MockReadTxn::new(encoded_rows.clone()), &base.manifest);
            let cold = GraphTransitionSource::read(
                &mut cold_txn,
                &base.tree,
                base.partition,
                MEMORY,
                &|| Ok(()),
            )
            .await
            .unwrap();
            assert!(cold_txn.into_raw().scans >= 2);
            assert_eq!(
                warm.retained_bytes().unwrap(),
                cold.retained_bytes().unwrap()
            );

            let warm_out = warm
                .prepare(Some(next_header), next_children.clone(), MEMORY, &|| Ok(()))
                .unwrap();
            let cold_out = cold
                .prepare(Some(next_header), next_children, MEMORY, &|| Ok(()))
                .unwrap();
            let (
                GraphPublication::Replace {
                    records: warm_rec, ..
                },
                GraphPublication::Replace {
                    records: cold_rec, ..
                },
            ) = (&warm_out.publication, &cold_out.publication)
            else {
                panic!("expected replacement");
            };
            assert_eq!(warm_rec.descriptor, cold_rec.descriptor);
            let warm_chunks: Vec<_> = warm_rec.chunks().collect();
            let cold_chunks: Vec<_> = cold_rec.chunks().collect();
            assert_eq!(warm_chunks, cold_chunks);
        }

        // Admission limit is identical warm vs cold.
        let mut warm_txn = ReadLogicalTxn::for_index(
            MockReadTxn::new(encoded_rows.clone()).with_failing_scans(),
            &base.manifest,
        );
        let retained = GraphTransitionSource::read_cached(
            &mut warm_txn,
            Some(&cache),
            &base.tree,
            base.partition,
            MEMORY,
            &|| Ok(()),
        )
        .await
        .unwrap()
        .retained_bytes()
        .unwrap();
        let tight = retained
            - descriptor.chunks() as usize * std::mem::size_of::<(u32, GraphChunk)>()
            + PAGE.byte_limit
            - 1;
        let mut warm_tight = ReadLogicalTxn::for_index(
            MockReadTxn::new(encoded_rows.clone()).with_failing_scans(),
            &base.manifest,
        );
        let mut cold_tight =
            ReadLogicalTxn::for_index(MockReadTxn::new(encoded_rows.clone()), &base.manifest);
        assert_eq!(
            GraphTransitionSource::read_cached(
                &mut warm_tight,
                Some(&cache),
                &base.tree,
                base.partition,
                tight,
                &|| Ok(()),
            )
            .await
            .unwrap_err()
            .kind(),
            ErrorKind::LimitExceeded
        );
        assert_eq!(
            GraphTransitionSource::read(
                &mut cold_tight,
                &base.tree,
                base.partition,
                tight,
                &|| Ok(())
            )
            .await
            .unwrap_err()
            .kind(),
            ErrorKind::LimitExceeded
        );

        // Projections-only `Internal` cache hit skips ChildEntry scan and scans only GraphChunks.
        let projections_cache = PartitionCache::new(MEMORY as u64);
        projections_cache.install(
            CacheKey::new(
                index,
                base.tree.clone(),
                base.partition,
                PartitionKind::Internal,
            ),
            Arc::new(CachedBody::internal(5, initial_rows.clone())),
        );
        let mut partial_txn =
            ReadLogicalTxn::for_index(MockReadTxn::new(encoded_rows.clone()), &base.manifest);
        let partial = GraphTransitionSource::read_cached(
            &mut partial_txn,
            Some(&projections_cache),
            &base.tree,
            base.partition,
            MEMORY,
            &|| Ok(()),
        )
        .await
        .unwrap();
        assert_eq!(partial_txn.into_raw().scans, 1);
        assert_eq!(partial.children(), initial_rows.as_slice());

        // Mismatched snapshot GraphDescriptor binding fails closed even on matching epoch.
        let mut tampered = kv_rows;
        let mut bad_descriptor = descriptor;
        bad_descriptor.binding[0] ^= 0xff;
        tampered[2].1 = PersistentValue::GraphDescriptor(bad_descriptor);
        let mut bad_txn = ReadLogicalTxn::for_index(
            MockReadTxn::new(encode_rows(&tampered)).with_failing_scans(),
            &base.manifest,
        );
        assert_eq!(
            GraphTransitionSource::read_cached(
                &mut bad_txn,
                Some(&cache),
                &base.tree,
                base.partition,
                MEMORY,
                &|| Ok(()),
            )
            .await
            .unwrap_err()
            .kind(),
            ErrorKind::Corruption
        );
    }
}
