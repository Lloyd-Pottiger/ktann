//! Ordinary Ready-partition projection changes over the shared graph lifecycle.
//!
//! The topology owner composes child creation/removal with publication in the
//! same transaction. Native preparation runs outside it under retained admission.

use super::backend::{ReadOps, WriteTxn};
use super::graph_transitions::{GraphTransitionSource, PreparedGraphTransition};
use super::keys::{LogicalKey, TreeKey};
use super::values::{
    ChildEntry, IndexLifecycle, PartitionCentroid, PartitionHeader, PartitionState, PersistentValue,
};
use super::{ReadLogicalTxn, WriteLogicalTxn};
use crate::api::{Error, ErrorKind, PartitionKey, Result};

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}
fn stale() -> Error {
    Error::new(ErrorKind::RetryableAbort)
}
fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

/// Complete source for one ordinary nonempty Ready internal partition.
#[derive(Debug)]
pub struct GraphUpdateSource(GraphTransitionSource);

impl GraphUpdateSource {
    /// Reads the shared bounded graph snapshot, with ordinary-update admission.
    pub async fn read<T: ReadOps>(
        txn: &mut ReadLogicalTxn<'_, T>,
        tree: &TreeKey,
        parent: PartitionKey,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let manifest = txn.require_manifest()?;
        if manifest.lifecycle() != IndexLifecycle::Active
            || manifest.config().local_graph().is_none()
        {
            return Err(Error::invalid_argument());
        }
        let source = GraphTransitionSource::read(txn, tree, parent, working_bytes, check).await?;
        let header = source.header().ok_or_else(corrupt)?;
        if header.state() != PartitionState::Ready || header.level() <= 1 {
            return Err(stale());
        }
        if header.entry_count() == 0 {
            return Err(corrupt());
        }
        Ok(Self(source))
    }

    /// Canonical authoritative projections to transform into the proposed body.
    pub fn children(&self) -> &[ChildEntry] {
        self.0.children()
    }

    /// Prepares a canonical nonempty next body. `None` leaves the epoch unchanged.
    pub fn prepare(
        self,
        next: Vec<ChildEntry>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Option<PreparedGraphUpdate>> {
        check()?;
        if next.is_empty()
            || next
                .iter()
                .any(|child| child.child() == self.0.partition || child.child().get() == 1)
        {
            return Err(Error::invalid_argument());
        }
        if self.children() == next {
            return Ok(None);
        }
        let source_header = self.0.header().ok_or_else(corrupt)?;
        let header = PartitionHeader::new(
            source_header.level(),
            next.len() as u32,
            source_header
                .cache_epoch()
                .checked_add(1)
                .ok_or_else(corrupt)?,
            PartitionState::Ready,
        )?;
        let change_capacity = self
            .children()
            .len()
            .checked_add(next.len())
            .ok_or_else(limit)?;
        let change_bytes = change_capacity
            .checked_mul(std::mem::size_of::<Change>())
            .ok_or_else(limit)?;
        let (graph, old, next) = self.0.prepare_with_entries(
            Some(header),
            next,
            working_bytes.checked_sub(change_bytes).ok_or_else(limit)?,
            check,
        )?;
        // Move projections after native work, avoiding a third centroid copy at
        // the native allocation peak. Only changed entries survive preparation.
        let mut changes = Vec::new();
        changes
            .try_reserve_exact(change_capacity)
            .map_err(|_| limit())?;
        let mut old = old.into_iter().peekable();
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
        Ok(Some(PreparedGraphUpdate { graph, changes }))
    }
}

/// Changed projections, including their child authority expectations.
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

/// Exact graph replacement plus the ordinary projection and centroid delta.
#[derive(Debug)]
pub struct PreparedGraphUpdate {
    graph: PreparedGraphTransition,
    changes: Vec<Change>,
}

impl PreparedGraphUpdate {
    /// Resolves unknown commits only by the shared exact target identity.
    pub async fn published<T: WriteTxn>(&self, txn: &mut WriteLogicalTxn<'_, T>) -> Result<bool> {
        self.graph.published(txn).await
    }

    /// Atomically publishes graph, parent projections/Header and child centroids.
    /// False recognizes an identical result already committed. The caller owns
    /// accompanying child creation/removal and must roll back on any error.
    pub async fn publish<T: WriteTxn>(&self, txn: &mut WriteLogicalTxn<'_, T>) -> Result<bool> {
        if self.published(txn).await? {
            return Ok(false);
        }
        let header = self.graph.source_header.ok_or_else(corrupt)?;
        let index = self.graph.manifest.logical_index_id();
        let tree = &self.graph.tree;
        for batch in self.changes.chunks(32) {
            let mut keys = Vec::with_capacity(batch.len() * 2);
            for change in batch {
                keys.push(LogicalKey::Header {
                    index,
                    tree_key: tree.clone(),
                    partition: change.child(),
                });
                keys.push(LogicalKey::Centroid {
                    index,
                    tree_key: tree.clone(),
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
        for change in &self.changes {
            let key = LogicalKey::ChildEntry {
                index,
                tree_key: tree.clone(),
                partition: self.graph.partition,
                child: change.child(),
            };
            if let Some(new) = &change.new {
                builder.put(key, PersistentValue::ChildEntry(new.clone()))?;
                builder.put(
                    LogicalKey::Centroid {
                        index,
                        tree_key: tree.clone(),
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
                tree_key: tree.clone(),
                partition: self.graph.partition,
            },
            PersistentValue::PartitionHeader(self.graph.header.ok_or_else(corrupt)?),
        )?;
        self.graph.append_authority(&mut builder)?;
        if txn
            .affected_bytes()
            .checked_add(builder.affected_bytes()?)
            .ok_or_else(limit)?
            > 900_000
        {
            return Err(limit());
        }
        self.graph.append_payload(&mut builder)?;
        txn.apply(builder).await?;
        Ok(true)
    }
}
