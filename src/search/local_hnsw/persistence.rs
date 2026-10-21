//! Snapshot-bound graph envelopes; callers supply entries from one snapshot.

use super::*;
use crate::api::{LogicalIndexId, Metric};
use crate::storage::values::{
    GRAPH_CHUNK_BYTES, GraphChunk, GraphDescriptor, PartitionHeader, PartitionState,
};
use bytes::Bytes;

/// Snapshot source of a persisted internal body; parameters come from its Manifest.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphSource<'a> {
    pub(crate) index: LogicalIndexId,
    pub(crate) tree: &'a TreeKey,
    pub(crate) parent: PartitionKey,
    pub(crate) header: PartitionHeader,
    pub(crate) parameters: Parameters,
}

/// Encoded graph with cheap, format-sized slices for staged KV writes.
#[derive(Debug)]
pub(crate) struct GraphRecords {
    pub(crate) descriptor: GraphDescriptor,
    body: Bytes,
}

impl GraphRecords {
    /// Yields contiguous ordinals without copying or retaining a second body.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = (u32, GraphChunk)> + '_ {
        (0..self.descriptor.chunks()).map(|ordinal| {
            let start = ordinal as usize * GRAPH_CHUNK_BYTES;
            let end = (start + GRAPH_CHUNK_BYTES).min(self.body.len());
            (ordinal, GraphChunk(self.body.slice(start..end)))
        })
    }
}

/// Binds all source coordinates and configuration, never just the child count.
/// Signed zero uses the canonical persistent f32 spelling.
pub(super) fn identity(
    centroids: Centroids<'_>,
    parameters: Parameters,
    tree: &TreeKey,
    parent: PartitionKey,
    check: &dyn Fn() -> Result<()>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"ktann/local-hnsw/source/v1\0");
    hash.update((tree.as_bytes().len() as u64).to_be_bytes());
    hash.update(tree.as_bytes());
    hash.update(parent.get().to_be_bytes());
    hash.update((parameters.m as u64).to_be_bytes());
    hash.update((parameters.ef_construction as u64).to_be_bytes());
    hash.update(parameters.seed);
    hash.update([match centroids.kernel.metric() {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::InnerProduct => 2,
    }]);
    hash.update((centroids.kernel.dimension() as u64).to_be_bytes());
    hash.update((centroids.children.len() as u64).to_be_bytes());
    // Batch canonical coordinates without retaining a second centroid body.
    let mut coordinates = [0_u8; 1024];
    for child in centroids.children {
        check()?;
        hash.update(child.child().get().to_be_bytes());
        for chunk in child
            .centroid()
            .chunks(coordinates.len() / size_of::<f32>())
        {
            for (value, encoded) in chunk.iter().zip(coordinates.as_chunks_mut::<4>().0) {
                encoded.copy_from_slice(
                    &(if *value == 0.0 { 0 } else { value.to_bits() }).to_be_bytes(),
                );
            }
            hash.update(&coordinates[..std::mem::size_of_val(chunk)]);
        }
    }
    Ok(hash.finalize().into())
}

pub(super) fn binding(
    identity: [u8; 32],
    index: LogicalIndexId,
    header: PartitionHeader,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ktann/local-hnsw/snapshot/v1\0");
    hash.update(identity);
    hash.update(index.get().to_be_bytes());
    hash.update(header.level().to_be_bytes());
    hash.update(header.entry_count().to_be_bytes());
    hash.update(header.cache_epoch().to_be_bytes());
    hash.finalize().into()
}

fn header_matches(header: PartitionHeader, nodes: usize) -> bool {
    header.level() > 1
        && nodes > 0
        && header.entry_count() as usize == nodes
        && header.state() == PartitionState::Ready
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// Builds and encodes one final partition without exposing an intermediate graph.
/// Admission includes borrowed projections, native construction and encoding scratch.
pub(crate) fn build_records(
    centroids: Centroids<'_>,
    source: GraphSource<'_>,
    working_bytes: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<GraphRecords> {
    let (graph, identity) = build_graph(
        centroids,
        source.parameters,
        source.tree,
        source.parent,
        working_bytes,
        check,
    )?;
    ::metrics::histogram!("ktann.routing.native_build_peak_bytes")
        .record(graph.peak_bytes() as f64);
    persist_graph(
        centroids,
        &graph,
        identity,
        source.index,
        source.header,
        working_bytes,
        check,
    )
}

/// Prepares one complete replacement body. A deletion reuses the canonical
/// builder; insertion and centroid replacement consume a private restored graph.
/// Source corruption is never concealed by rebuilding.
#[expect(
    clippy::too_many_arguments,
    reason = "snapshot-bound source and proposed body with operation admission"
)]
pub(crate) fn update_records(
    old: Centroids<'_>,
    next: Centroids<'_>,
    source: GraphSource<'_>,
    descriptor: &GraphDescriptor,
    chunks: impl IntoIterator<Item = (u32, GraphChunk)>,
    next_header: PartitionHeader,
    working_bytes: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<GraphRecords> {
    if next_header.level() != source.header.level()
        || source.header.cache_epoch().checked_add(1) != Some(next_header.cache_epoch())
        || !header_matches(next_header, next.children.len())
    {
        return Err(Error::invalid_argument());
    }
    let mapping_bytes = old
        .children
        .len()
        .checked_add(next.children.len())
        .and_then(|count| count.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(128))
        .ok_or_else(limit)?;
    let restore_budget = working_bytes
        .checked_sub(next.bytes + mapping_bytes)
        .ok_or_else(limit)?;
    let (graph, _) = restore_graph(old, source, descriptor, chunks, restore_budget, check)?;
    let mut mapping = Vec::new();
    let mut changed = Vec::new();
    mapping
        .try_reserve_exact(next.children.len())
        .map_err(|_| limit())?;
    changed
        .try_reserve_exact(old.children.len())
        .map_err(|_| limit())?;
    for (slot, child) in old.children.iter().enumerate() {
        check()?;
        let Ok(target) = next
            .children
            .binary_search_by_key(&child.child(), ChildEntry::child)
        else {
            drop(graph);
            return build_records(
                next,
                GraphSource {
                    header: next_header,
                    ..source
                },
                working_bytes
                    .checked_sub(old.bytes + mapping_bytes)
                    .ok_or_else(limit)?,
                check,
            );
        };
        mapping.push(target as u32);
        if child.centroid() != next.children[target].centroid() {
            changed.push(slot as u32);
        }
    }
    for (target, child) in next.children.iter().enumerate() {
        check()?;
        if old
            .children
            .binary_search_by_key(&child.child(), ChildEntry::child)
            .is_err()
        {
            mapping.push(target as u32);
        }
    }
    let available = working_bytes
        .checked_sub(old.bytes + next.bytes + mapping_bytes)
        .ok_or_else(limit)?;
    let distance = |a: u32, b: u32| {
        next.distance(
            next.children[mapping[a as usize] as usize].centroid(),
            next.children[mapping[b as usize] as usize].centroid(),
        )
    };
    let graph = controlled(check, |proceed| {
        graph.evolve(
            &mapping,
            &changed,
            source.parameters.ef_construction,
            partition_seed(source.parameters.seed, source.tree, source.parent)
                ^ next_header.cache_epoch(),
            available,
            &distance,
            proceed,
        )
    })?;
    let identity = identity(next, source.parameters, source.tree, source.parent, check)?;
    persist_graph(
        next,
        &graph,
        identity,
        source.index,
        next_header,
        working_bytes
            .checked_sub(old.bytes + mapping_bytes)
            .ok_or_else(limit)?,
        check,
    )
}

pub(super) fn persist_graph(
    centroids: Centroids<'_>,
    graph: &ktann_usearch::Graph,
    identity: [u8; 32],
    index: LogicalIndexId,
    header: PartitionHeader,
    working_bytes: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<GraphRecords> {
    if !header_matches(header, centroids.children.len()) {
        return Err(Error::invalid_argument());
    }
    let available = working_bytes
        .checked_sub(centroids.bytes + 1024)
        .ok_or_else(limit)?;
    let body = Bytes::from(controlled(check, |proceed| {
        graph.encode(available, proceed)
    })?);
    let mut hash = Sha256::new();
    for chunk in body.chunks(GRAPH_CHUNK_BYTES) {
        check()?;
        hash.update(chunk);
    }
    let descriptor = GraphDescriptor {
        nodes: centroids.children.len() as u32,
        epoch: header.cache_epoch(),
        bytes: body.len() as u64,
        binding: binding(identity, index, header),
        sha256: hash.finalize().into(),
    };
    Ok(GraphRecords { descriptor, body })
}

/// Loads a complete ordered chunk stream. The caller must use the same
/// snapshot for Header, Child Entries, descriptor, and chunks, and scan the
/// entire chunk range so extra keys cannot be hidden by a point-read list.
///
/// Budget charges both retained chunk input and the assembled buffer, child
/// projections, validation scratch, and native restore allocations. Encoding
/// corruption is never repaired or routed through an exact fallback.
pub(super) fn restore_graph(
    centroids: Centroids<'_>,
    source: GraphSource<'_>,
    descriptor: &GraphDescriptor,
    chunks: impl IntoIterator<Item = (u32, GraphChunk)>,
    working_bytes: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<(ktann_usearch::Graph, [u8; 32])> {
    check()?;
    if !header_matches(source.header, centroids.children.len())
        || source.parameters.ef_construction < source.parameters.m
        || descriptor.nodes as usize != centroids.children.len()
        || descriptor.epoch != source.header.cache_epoch()
    {
        return Err(corrupt());
    }
    let length = usize::try_from(descriptor.bytes).map_err(|_| corrupt())?;
    if length < ktann_usearch::NATIVE_HEADER_BYTES
        || length
            > ktann_usearch::encoded_bound(centroids.children.len(), source.parameters.m)
                .map_err(|_| corrupt())?
    {
        return Err(corrupt());
    }
    let available = working_bytes
        .checked_sub(centroids.bytes)
        .and_then(|v| v.checked_sub(length + 1024))
        .ok_or_else(limit)?;
    if ktann_usearch::decode_bound(centroids.children.len(), source.parameters.m, length)
        .map_err(native_error)?
        > available
    {
        return Err(limit());
    }
    let identity = identity(
        centroids,
        source.parameters,
        source.tree,
        source.parent,
        check,
    )?;
    if descriptor.binding != binding(identity, source.index, source.header) {
        return Err(corrupt());
    }
    let mut body = Vec::new();
    body.try_reserve_exact(length).map_err(|_| limit())?;
    let mut hash = Sha256::new();
    for (count, (ordinal, chunk)) in chunks.into_iter().enumerate() {
        check()?;
        if ordinal as usize != count || count >= descriptor.chunks() as usize {
            return Err(corrupt());
        }
        let expected = (length - body.len()).min(GRAPH_CHUNK_BYTES);
        if chunk.bytes().len() != expected {
            return Err(corrupt());
        }
        hash.update(chunk.bytes());
        body.extend_from_slice(chunk.bytes());
    }
    if body.len() != length || <[u8; 32]>::from(hash.finalize()) != descriptor.sha256 {
        return Err(corrupt());
    }
    let graph = controlled(check, |proceed| {
        ktann_usearch::Graph::decode(
            &body,
            centroids.children.len(),
            source.parameters.m,
            available,
            proceed,
        )
    })?;
    Ok((graph, identity))
}
