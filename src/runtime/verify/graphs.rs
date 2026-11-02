//! Graph-body integrity joins over the audit's ordered snapshot stream.
//!
//! Only one graph is retained. Per-tree fingerprints join Child Entries to the
//! separately stored centroids; topology ownership remains in TopologyLedger.
use std::collections::{BTreeMap, btree_map::Entry};

use bytes::Bytes;
use sha2::{Digest, Sha256};

use super::Context;
use crate::api::{ErrorKind, PartitionKey, Result, VerifyIssueKind};
use crate::runtime::{OperationContext, check_control, compute::controlled_compute};
use crate::search::local_hnsw::{body::InternalGraphBody, persistence::GraphSource};
use crate::search::numeric::VectorKernel;
use crate::storage::backend::Backend;
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    ChildEntry, GraphChunk, GraphDescriptor, PartitionHeader, PersistentValue,
};

/// Centroid facts retained until the tree boundary, independent of key order.
#[derive(Default)]
struct Projection {
    header: bool,
    actual: Option<[u8; 32]>,
    expected: Option<([u8; 32], PartitionKey)>,
}

#[derive(Default)]
struct Body {
    header: Option<PartitionHeader>,
    children: Vec<ChildEntry>,
    descriptor: Option<GraphDescriptor>,
    chunks: Vec<(u32, GraphChunk)>,
    charged: u64,
}

/// Streaming audit state, never backed by the runtime cache.
#[derive(Default)]
pub(super) struct GraphLedger {
    tree: Option<TreeKey>,
    partition: Option<PartitionKey>,
    body: Body,
    projections: BTreeMap<PartitionKey, Projection>,
    tree_bytes: u64,
}

impl GraphLedger {
    pub(super) async fn observe<B: Backend>(
        &mut self,
        cx: &mut Context<'_>,
        key: &LogicalKey,
        value: &PersistentValue,
        context: &OperationContext<B>,
    ) -> Result<()> {
        let (Some(tree), Some(partition)) = (key.tree_key(), super::partition_key(key)) else {
            return Ok(());
        };
        if cx.manifest.config().local_graph().is_none() {
            if matches!(
                value,
                PersistentValue::GraphDescriptor(_) | PersistentValue::GraphChunk(_)
            ) {
                cx.issue(
                    VerifyIssueKind::Membership,
                    Some(tree),
                    Some(partition),
                    None,
                );
            }
            return Ok(());
        }
        if self.tree.as_ref() != Some(tree) || self.partition != Some(partition) {
            self.close_body(cx, context).await?;
            if self.tree.as_ref() != Some(tree) {
                self.close_tree(cx);
                cx.charge_memory(tree.as_bytes().len() as u64);
                if cx.truncated() {
                    return Ok(());
                }
                self.tree_bytes = tree.as_bytes().len() as u64;
                self.tree = Some(tree.clone());
            }
            self.partition = Some(partition);
        }
        if cx.truncated() {
            return Ok(());
        }
        match value {
            PersistentValue::PartitionHeader(header) => {
                self.body.header = Some(*header);
                if let Some(projection) = self.projection(cx, partition) {
                    projection.header = true;
                }
            }
            PersistentValue::PartitionCentroid(centroid) => {
                if let Some(projection) = self.projection(cx, partition) {
                    projection.actual = Some(fingerprint(centroid.components()));
                }
            }
            PersistentValue::ChildEntry(child) => {
                if let Some(projection) = self.projection(cx, child.child()) {
                    projection.expected = Some((fingerprint(child.centroid()), partition));
                }
                if reserve_entry(
                    cx,
                    &mut self.body.children,
                    &mut self.body.charged,
                    size_of_val(child.centroid()),
                ) {
                    self.body.children.push(child.clone());
                }
            }
            PersistentValue::GraphDescriptor(descriptor) => {
                self.body.descriptor = Some(descriptor.clone());
            }
            PersistentValue::GraphChunk(chunk) => {
                let LogicalKey::GraphChunk { chunk: ordinal, .. } = key else {
                    unreachable!()
                };
                if reserve_entry(
                    cx,
                    &mut self.body.chunks,
                    &mut self.body.charged,
                    chunk.bytes().len(),
                ) {
                    self.body
                        .chunks
                        .push((*ordinal, GraphChunk(Bytes::copy_from_slice(chunk.bytes()))));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn projection(
        &mut self,
        cx: &mut Context<'_>,
        partition: PartitionKey,
    ) -> Option<&mut Projection> {
        match self.projections.entry(partition) {
            Entry::Occupied(entry) => Some(entry.into_mut()),
            Entry::Vacant(entry) => {
                // Include B-tree nodes and allocation overhead, not just payloads.
                let bytes = (3 * size_of::<(PartitionKey, Projection)>() + 64) as u64;
                cx.charge_memory(bytes);
                if cx.truncated() {
                    return None;
                }
                self.tree_bytes += bytes;
                Some(entry.insert(Projection::default()))
            }
        }
    }

    /// Reuse the serving reader's complete native validation, including source
    /// binding, digest, ordinal coverage, adjacency and reachability. Admission
    /// uses only the audit's remaining memory, with retained input still charged.
    async fn close_body<B: Backend>(
        &mut self,
        cx: &mut Context<'_>,
        context: &OperationContext<B>,
    ) -> Result<()> {
        let body = std::mem::take(&mut self.body);
        let Some(partition) = self.partition.take() else {
            return Ok(());
        };
        if cx.truncated() {
            return Ok(());
        }
        context.checkpoint()?;
        let tree = self.tree.as_ref().unwrap();
        match body.header {
            Some(header) if header.level() > 1 && header.entry_count() == 0 => {
                if body.descriptor.is_some() || !body.chunks.is_empty() || !body.children.is_empty()
                {
                    cx.issue(
                        VerifyIssueKind::Membership,
                        Some(tree),
                        Some(partition),
                        None,
                    );
                }
            }
            Some(header) if header.level() > 1 => {
                if let Some(descriptor) = body.descriptor {
                    let dimension = cx.manifest.config().dimension();
                    let kernel_bytes =
                        VectorKernel::allocation_bound(dimension).unwrap_or(usize::MAX) as u64;
                    let kernel_bytes = kernel_bytes.saturating_add(tree.as_bytes().len() as u64);
                    cx.charge_memory(kernel_bytes);
                    if !cx.truncated() {
                        let index = cx.index();
                        let tree_owned = tree.clone();
                        let metric = cx.manifest.config().metric();
                        let seed = *cx.manifest.rotation_seed();
                        let parameters = cx.manifest.config().local_graph().unwrap().parameters();
                        let working_bytes = cx.limits.remaining_memory() as usize;
                        let result = controlled_compute(
                            context.options.clone(),
                            context
                                .cpu_admission
                                .clone()
                                .expect("verification admission"),
                            move |options| {
                                InternalGraphBody::restore(
                                    body.children.into_boxed_slice(),
                                    VectorKernel::new(dimension, metric, seed)?,
                                    GraphSource {
                                        index,
                                        tree: &tree_owned,
                                        parent: partition,
                                        header,
                                        parameters,
                                    },
                                    descriptor,
                                    body.chunks,
                                    working_bytes,
                                    &|| check_control(options),
                                )
                                .map(|_| ())
                            },
                        )
                        .await;
                        match result {
                            Ok(_) => {}
                            Err(error) if error.kind() == ErrorKind::Corruption => {
                                cx.issue(
                                    VerifyIssueKind::InvalidEncoding,
                                    Some(tree),
                                    Some(partition),
                                    None,
                                );
                            }
                            Err(error) if error.kind() == ErrorKind::LimitExceeded => {
                                cx.truncated = true
                            }
                            Err(error) => return Err(error),
                        }
                        cx.release_memory(kernel_bytes);
                    }
                } else {
                    cx.issue(
                        VerifyIssueKind::Reachability,
                        Some(tree),
                        Some(partition),
                        None,
                    );
                }
            }
            _ if body.descriptor.is_some() || !body.chunks.is_empty() => {
                cx.issue(
                    VerifyIssueKind::Membership,
                    Some(tree),
                    Some(partition),
                    None,
                );
            }
            _ => {}
        }
        cx.release_memory(body.charged);
        Ok(())
    }

    fn close_tree(&mut self, cx: &mut Context<'_>) {
        for (partition, projection) in std::mem::take(&mut self.projections) {
            if cx.truncated() {
                break;
            }
            // The root has no incoming projection and need not store a centroid.
            if projection.header
                && partition != crate::storage::topology::root_partition()
                && projection.actual.is_none()
            {
                cx.issue(
                    VerifyIssueKind::Reachability,
                    self.tree.as_ref(),
                    Some(partition),
                    None,
                );
            }
            if let Some((expected, parent)) = projection.expected
                && projection.actual != Some(expected)
            {
                cx.issue(
                    VerifyIssueKind::Membership,
                    self.tree.as_ref(),
                    Some(parent),
                    None,
                );
            }
        }
        cx.release_memory(self.tree_bytes);
        self.tree_bytes = 0;
        self.tree = None;
    }

    pub(super) async fn finish<B: Backend>(
        &mut self,
        cx: &mut Context<'_>,
        context: &OperationContext<B>,
    ) -> Result<()> {
        self.close_body(cx, context).await?;
        self.close_tree(cx);
        Ok(())
    }
}

/// Canonical f32 coordinates, including the persistent spelling of signed zero.
fn fingerprint(vector: &[f32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for &component in vector {
        hash.update(
            if component == 0.0 {
                0_u32
            } else {
                component.to_bits()
            }
            .to_le_bytes(),
        );
    }
    hash.finalize().into()
}

/// Admit geometric Vec growth before allocation, including transient relocation
/// and conversion to a boxed child slice. Payloads are detached from scan pages.
fn reserve_entry<T>(
    cx: &mut Context<'_>,
    entries: &mut Vec<T>,
    charged: &mut u64,
    payload: usize,
) -> bool {
    let growth = if entries.len() == entries.capacity() {
        entries.capacity().max(4)
    } else {
        0
    };
    let bytes = (2 * growth * size_of::<T>() + payload) as u64;
    cx.charge_memory(bytes);
    if cx.truncated() {
        return false;
    }
    if growth != 0 && entries.try_reserve_exact(growth).is_err() {
        cx.truncated = true;
        return false;
    }
    *charged += bytes;
    true
}
