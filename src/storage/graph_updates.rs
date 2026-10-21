//! Preparation and fenced publication of ordinary Ready-partition graph changes.
//!
//! These storage primitives do not enable mutable graph serving. The topology
//! owner must compose child creation/removal or transfers with `publish` in the
//! same transaction. Callers roll back on any error, dispatch `prepare` to their
//! bounded CPU executor, and retain admission until the computation exits.

use super::backend::{ReadOps, ScanLimits, WriteTxn};
use super::keys::{LogicalKey, TreeKey};
use super::values::{
    ChildEntry, GraphChunk, GraphDescriptor, IndexLifecycle, IndexManifest, PartitionCentroid,
    PartitionHeader, PartitionState, PersistentValue,
};
use super::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};
use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::local_hnsw::{
    Centroids,
    persistence::{GraphRecords, GraphSource, update_records},
};
use crate::search::numeric::VectorKernel;

const PAGE: ScanLimits = ScanLimits {
    item_limit: 64,
    byte_limit: 256 * 1024,
};

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

fn stale() -> Error {
    Error::new(ErrorKind::RetryableAbort)
}

fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

/// Complete source from one backend snapshot, without retaining that transaction.
#[derive(Debug)]
pub struct GraphUpdateSource {
    manifest: IndexManifest,
    tree: TreeKey,
    parent: PartitionKey,
    header: PartitionHeader,
    descriptor: GraphDescriptor,
    children: Vec<ChildEntry>,
    chunks: Vec<(u32, GraphChunk)>,
}

impl GraphUpdateSource {
    /// Reads a bounded complete source. Native validation occurs during preparation.
    pub async fn read<T: ReadOps>(
        txn: &mut ReadLogicalTxn<'_, T>,
        tree: &TreeKey,
        parent: PartitionKey,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        check()?;
        let manifest = txn.require_manifest()?.clone();
        let index = manifest.logical_index_id();
        if manifest.lifecycle() != IndexLifecycle::Active
            || manifest.config().fixed_graph().is_none()
        {
            return Err(Error::invalid_argument());
        }
        let values = txn
            .batch_get(vec![
                LogicalKey::Manifest(index),
                LogicalKey::Header {
                    index,
                    tree_key: tree.clone(),
                    partition: parent,
                },
                LogicalKey::GraphDescriptor {
                    index,
                    tree_key: tree.clone(),
                    partition: parent,
                },
            ])
            .await?;
        let mut values = values.into_iter();
        if values.next().flatten() != Some(PersistentValue::IndexManifest(manifest.clone())) {
            return Err(stale());
        }
        let Some(PersistentValue::PartitionHeader(header)) = values.next().flatten() else {
            return Err(corrupt());
        };
        let Some(PersistentValue::GraphDescriptor(descriptor)) = values.next().flatten() else {
            return Err(corrupt());
        };
        if header.state() != PartitionState::Ready || header.level() <= 1 {
            return Err(stale());
        }
        if header.entry_count() == 0
            || header.entry_count() > manifest.config().max_internal_partition_entries() + 2
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
            .checked_mul(header.entry_count() as usize)
            .and_then(|n| n.checked_add(usize::try_from(descriptor.bytes).ok()?))
            .and_then(|n| n.checked_add(PAGE.byte_limit + 4096))
            .ok_or_else(limit)?;
        if retained > working_bytes {
            return Err(limit());
        }
        let mut children = Vec::new();
        children
            .try_reserve_exact(header.entry_count() as usize)
            .map_err(|_| limit())?;
        let range = LogicalRange::child_entries(&manifest, tree, parent)?;
        let mut cursor = None;
        loop {
            check()?;
            let (items, next) = txn.scan(&range, cursor.as_ref(), PAGE).await?.into_parts();
            for item in items {
                if children.len() == header.entry_count() as usize {
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
        if children.len() != header.entry_count() as usize {
            return Err(corrupt());
        }
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(descriptor.chunks() as usize)
            .map_err(|_| limit())?;
        let range = LogicalRange::graph_chunks(&manifest, tree, parent)?;
        let mut cursor = None;
        loop {
            check()?;
            let (items, next) = txn.scan(&range, cursor.as_ref(), PAGE).await?.into_parts();
            for item in items {
                let LogicalKey::GraphChunk { chunk, .. } = item.key() else {
                    return Err(corrupt());
                };
                let ordinal = *chunk;
                if ordinal as usize != chunks.len() || chunks.len() == descriptor.chunks() as usize
                {
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
        Ok(Self {
            manifest,
            tree: tree.clone(),
            parent,
            header,
            descriptor,
            children,
            chunks,
        })
    }

    /// Authoritative canonical Child Entries to transform into the proposed body.
    pub fn children(&self) -> &[ChildEntry] {
        &self.children
    }

    /// Prepares a canonical nonempty next body without IO or cache publication.
    /// `None` denotes an unchanged body and does not advance the epoch.
    pub fn prepare(
        self,
        next: Vec<ChildEntry>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Option<PreparedGraphUpdate>> {
        let config = self.manifest.config();
        let kernel = VectorKernel::new(
            config.dimension(),
            config.metric(),
            *self.manifest.rotation_seed(),
        )?;
        let capacity = config.max_internal_partition_entries() as usize + 2;
        let old_centroids =
            Centroids::new(&kernel, &self.children, capacity, check).map_err(|error| {
                if error.kind() == ErrorKind::InvalidArgument {
                    corrupt()
                } else {
                    error
                }
            })?;
        let next_centroids = Centroids::new(&kernel, &next, capacity, check)?;
        if next.is_empty()
            || next
                .iter()
                .any(|child| child.child() == self.parent || child.child().get() == 1)
        {
            return Err(Error::invalid_argument());
        }
        if self.children == next {
            return Ok(None);
        }
        let header = PartitionHeader::new(
            self.header.level(),
            next.len() as u32,
            self.header
                .cache_epoch()
                .checked_add(1)
                .ok_or_else(corrupt)?,
            PartitionState::Ready,
        )?;
        let source = GraphSource {
            index: self.manifest.logical_index_id(),
            tree: &self.tree,
            parent: self.parent,
            header: self.header,
            parameters: config.fixed_graph().ok_or_else(corrupt)?.parameters(),
        };
        let change_capacity = self
            .children
            .len()
            .checked_add(next.len())
            .ok_or_else(limit)?;
        let change_bytes = change_capacity
            .checked_mul(std::mem::size_of::<Change>())
            .ok_or_else(limit)?;
        let available = working_bytes
            .checked_sub(kernel.heap_bytes() + 4096 + change_bytes)
            .ok_or_else(limit)?;
        let records = update_records(
            old_centroids,
            next_centroids,
            source,
            &self.descriptor,
            self.chunks,
            header,
            available,
            check,
        )?;
        // Move changed projections into the delta rather than retaining a third
        // full centroid allocation alongside the source and proposed bodies.
        let mut changes = Vec::new();
        changes
            .try_reserve_exact(change_capacity)
            .map_err(|_| limit())?;
        let mut old = self.children.into_iter().peekable();
        let mut new = next.into_iter().peekable();
        while old.peek().is_some() || new.peek().is_some() {
            check()?;
            match (old.peek(), new.peek()) {
                (Some(a), Some(b)) if a.child() == b.child() => {
                    let a = old.next();
                    let b = new.next();
                    if a != b {
                        changes.push(Change { old: a, new: b });
                    }
                }
                (Some(a), Some(b)) if a.child() < b.child() => changes.push(Change {
                    old: old.next(),
                    new: None,
                }),
                (Some(_), None) => changes.push(Change {
                    old: old.next(),
                    new: None,
                }),
                _ => changes.push(Change {
                    old: None,
                    new: new.next(),
                }),
            }
        }
        Ok(Some(PreparedGraphUpdate {
            manifest: self.manifest,
            tree: self.tree,
            parent: self.parent,
            source_header: self.header,
            source: self.descriptor,
            header,
            records,
            changes,
        }))
    }
}

/// Exact authoritative delta retained after private graph preparation.
#[derive(Debug)]
struct Change {
    old: Option<ChildEntry>,
    new: Option<ChildEntry>,
}

impl Change {
    fn child(&self) -> PartitionKey {
        self.old
            .as_ref()
            .or(self.new.as_ref())
            .expect("nonempty change")
            .child()
    }
}

/// Private graph result bound to one source epoch and one authoritative delta.
#[derive(Debug)]
pub struct PreparedGraphUpdate {
    manifest: IndexManifest,
    tree: TreeKey,
    parent: PartitionKey,
    source_header: PartitionHeader,
    source: GraphDescriptor,
    header: PartitionHeader,
    records: GraphRecords,
    changes: Vec<Change>,
}

impl PreparedGraphUpdate {
    /// Checks exact durable publication before staging accompanying topology
    /// writes after an unknown commit result. Supersession is not success evidence.
    pub async fn published<T: WriteTxn>(&self, txn: &mut WriteLogicalTxn<'_, T>) -> Result<bool> {
        let (header, descriptor) = authority(txn, &self.tree, self.parent).await?;
        if header == self.header && descriptor == self.records.descriptor {
            return Ok(true);
        }
        if header != self.source_header || descriptor != self.source {
            return Err(stale());
        }
        Ok(false)
    }

    /// Atomically replaces the graph and its authoritative projections. The
    /// caller composes child creation/removal/reparenting in this transaction and
    /// commits it. False recognizes an already committed identical result.
    /// No durable staging exists: cancellation/rollback discards all writes.
    pub async fn publish<T: WriteTxn>(&self, txn: &mut WriteLogicalTxn<'_, T>) -> Result<bool> {
        if self.published(txn).await? {
            return Ok(false);
        }
        let header = self.source_header;
        let index = self.manifest.logical_index_id();
        for batch in self.changes.chunks(32) {
            let mut keys = Vec::with_capacity(batch.len() * 2);
            for change in batch {
                keys.push(LogicalKey::Header {
                    index,
                    tree_key: self.tree.clone(),
                    partition: change.child(),
                });
                keys.push(LogicalKey::Centroid {
                    index,
                    tree_key: self.tree.clone(),
                    partition: change.child(),
                });
            }
            let values = txn.batch_get_for_update(keys).await?;
            for (change, pair) in batch.iter().zip(values.as_chunks::<2>().0) {
                let Some(PersistentValue::PartitionHeader(child)) = &pair[0] else {
                    return Err(stale());
                };
                let Some(PersistentValue::PartitionCentroid(centroid)) = &pair[1] else {
                    return Err(stale());
                };
                let expected = change
                    .old
                    .as_ref()
                    .or(change.new.as_ref())
                    .expect("nonempty change");
                if child.state() != PartitionState::Ready
                    || child.level().checked_add(1) != Some(header.level())
                    || centroid.components() != expected.centroid()
                {
                    return Err(stale());
                }
            }
        }
        let mut builder = txn.mutations();
        self.queue_publication(
            &mut builder,
            txn.admission_budget().mutation_key_overhead_bytes,
        )?;
        txn.apply(builder).await?;
        Ok(true)
    }

    fn queue_publication(
        &self,
        builder: &mut super::MutationBuilder<'_>,
        key_overhead: usize,
    ) -> Result<()> {
        let index = self.manifest.logical_index_id();
        for change in &self.changes {
            let key = LogicalKey::ChildEntry {
                index,
                tree_key: self.tree.clone(),
                partition: self.parent,
                child: change.child(),
            };
            if let Some(new) = &change.new {
                builder.put(key, PersistentValue::ChildEntry(new.clone()))?;
                builder.put(
                    LogicalKey::Centroid {
                        index,
                        tree_key: self.tree.clone(),
                        partition: new.child(),
                    },
                    PersistentValue::PartitionCentroid(PartitionCentroid::new(new.centroid())),
                )?;
            } else {
                builder.delete(key)?;
            }
        }
        builder.put(
            LogicalKey::Header {
                index,
                tree_key: self.tree.clone(),
                partition: self.parent,
            },
            PersistentValue::PartitionHeader(self.header),
        )?;
        builder.put(
            LogicalKey::GraphDescriptor {
                index,
                tree_key: self.tree.clone(),
                partition: self.parent,
            },
            PersistentValue::GraphDescriptor(self.records.descriptor.clone()),
        )?;
        // Shorter replacements remove their old tail in the same commit.
        for chunk in self.records.descriptor.chunks()..self.source.chunks() {
            builder.delete(LogicalKey::GraphChunk {
                index,
                tree_key: self.tree.clone(),
                partition: self.parent,
                chunk,
            })?;
        }
        // Bound authority changes and obsolete-tail deletes independently of new
        // graph payloads. A topology coordinator includes its writes in the same 900 KB
        // envelope when composing multiple graph publications.
        let protected_key_bytes = self.tree.as_bytes().len() + 64;
        let read_cost = (self.changes.len() * 2 + 3)
            .checked_mul(2 * (protected_key_bytes + key_overhead) + 64)
            .ok_or_else(limit)?;
        if builder
            .affected_bytes()?
            .checked_add(read_cost)
            .ok_or_else(limit)?
            > 900_000
        {
            return Err(limit());
        }
        for (chunk, value) in self.records.chunks() {
            builder.put(
                LogicalKey::GraphChunk {
                    index,
                    tree_key: self.tree.clone(),
                    partition: self.parent,
                    chunk,
                },
                PersistentValue::GraphChunk(value),
            )?;
        }
        Ok(())
    }
}

/// Protects lifecycle and the exact source version without retaining a read
/// transaction across native work. Backend snapshots preserve overwritten bytes.
async fn authority<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree: &TreeKey,
    parent: PartitionKey,
) -> Result<(PartitionHeader, GraphDescriptor)> {
    let manifest = txn.require_manifest()?;
    if manifest.lifecycle() != IndexLifecycle::Active || manifest.config().fixed_graph().is_none() {
        return Err(stale());
    }
    let index = manifest.logical_index_id();
    let values = txn
        .batch_get_for_update(vec![
            LogicalKey::Manifest(index),
            LogicalKey::Header {
                index,
                tree_key: tree.clone(),
                partition: parent,
            },
            LogicalKey::GraphDescriptor {
                index,
                tree_key: tree.clone(),
                partition: parent,
            },
        ])
        .await?;
    let mut values = values.into_iter();
    if values.next().flatten() != Some(PersistentValue::IndexManifest(manifest.clone())) {
        return Err(stale());
    }
    let Some(PersistentValue::PartitionHeader(header)) = values.next().flatten() else {
        return Err(corrupt());
    };
    let Some(PersistentValue::GraphDescriptor(descriptor)) = values.next().flatten() else {
        return Err(corrupt());
    };
    if header.state() != PartitionState::Ready {
        return Err(stale());
    }
    if descriptor.epoch != header.cache_epoch()
        || descriptor.nodes != header.entry_count()
        || header.level() <= 1
        || header.entry_count() == 0
        || header.entry_count() > manifest.config().max_internal_partition_entries() + 2
    {
        return Err(corrupt());
    }
    Ok((header, descriptor))
}
