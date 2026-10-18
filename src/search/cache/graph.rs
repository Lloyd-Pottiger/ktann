//! Snapshot-bound fixed-index bodies shared by exact and graph routing.

use super::*;
use crate::api::Metric;
use crate::search::local_hnsw::{
    Candidate, Parameters, Workspace, owned::OwnedGraph, persistence::GraphSource,
};
use crate::search::numeric::VectorKernel;
use crate::storage::keys::LogicalKey;
use crate::storage::values::{GRAPH_CHUNK_BYTES, GraphChunk, GraphDescriptor, PartitionState};

/// Runtime execution boundary: one snapshot, bounded owned CPU work and control.
/// Traversal owns candidate selection; the runtime owns admission and dispatch.
pub(crate) trait RoutingExecution {
    fn checkpoint(&self) -> Result<()>;
    async fn read<T: ReadOps>(
        &self,
        txn: &mut ReadLogicalTxn<'_, T>,
        manifest: &IndexManifest,
        request: BodyRequest<'_>,
        retained_bytes: usize,
    ) -> Result<RoutingGraph>;
    async fn candidates(
        &self,
        graph: RoutingGraph,
        query: Vec<f32>,
        beam: usize,
        mode: crate::api::RoutingMode,
    ) -> Result<Vec<(PartitionKey, f64)>>;
}

/// An admitted complete snapshot read, ready for CPU restoration outside the executor.
pub(crate) enum GraphRead {
    Cached(RoutingGraph),
    Load(Box<GraphLoad>),
}

/// Complete unpublished input. Dropping an interrupted load publishes nothing.
pub(crate) struct GraphLoad {
    key: CacheKey,
    header: PartitionHeader,
    parameters: Parameters,
    dimension: usize,
    metric: Metric,
    rotation_seed: [u8; 32],
    body: Arc<CachedBody>,
    descriptor: GraphDescriptor,
    chunks: Vec<(u32, GraphChunk)>,
    working_bytes: usize,
    restore_bytes: usize,
}

/// One parent's graph held by a query. It is charged even after cache eviction.
/// Consume it before loading another parent; other query-owned state is reserved
/// by the caller through `retained_bytes`, including candidate and identity sets.
pub(crate) struct RoutingGraph {
    body: Arc<CachedBody>,
    m: usize,
    working_bytes: usize,
}

impl RoutingGraph {
    /// Complete children, including those the local algorithm will not select.
    pub(crate) fn children(&self) -> &[ChildEntry] {
        let BodyEntries::Internal(entries) = &self.body.entries else {
            unreachable!("validated internal graph")
        };
        entries
    }

    /// Exact routing still requires the validated graph body, but ignores adjacency.
    pub(crate) fn exact(
        self,
        query: &[f32],
        beam: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<(PartitionKey, f64)>> {
        let hits = self
            .body
            .graph
            .as_ref()
            .expect("validated graph body")
            .exact(query, beam, self.working_bytes, check)?;
        self.map_candidates(hits, check)
    }

    /// Maps core ordinals without changing their scores or ordering.
    fn map_candidates(
        &self,
        hits: Vec<Candidate>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<(PartitionKey, f64)>> {
        let entries = self.children();
        let mut output = Vec::new();
        output.try_reserve_exact(hits.len()).map_err(|_| limit())?;
        for (ordinal, Candidate { node, distance }) in hits.into_iter().enumerate() {
            if ordinal % 64 == 0 {
                check()?;
            }
            output.push((entries[node as usize].child(), distance));
        }
        check()?;
        Ok(output)
    }

    /// Native query work; invoke only inside an admitted blocking task. Scratch is
    /// private to this query, while all concurrent queries share immutable topology.
    pub(crate) fn search(
        self,
        query: &[f32],
        beam: usize,
        ef: usize,
        check: &(dyn Fn() -> Result<()> + Sync),
    ) -> Result<(Vec<(PartitionKey, f64)>, ktann_usearch::Stats)> {
        check()?;
        let entries = self.children();
        let graph = self.body.graph.as_ref().expect("validated graph body");
        let available = self
            .working_bytes
            .checked_sub(self.body.bytes as usize)
            .ok_or_else(limit)?;
        let mut scratch = Workspace::new(entries.len(), self.m, available).map_err(native_error)?;
        let (hits, stats) =
            graph.search(query, beam, ef, self.working_bytes, &mut scratch, check)?;
        Ok((self.map_candidates(hits, check)?, stats))
    }
}

impl GraphLoad {
    /// Finishes validation and only then installs the complete body. This consumes
    /// the encoded chunks; no encoding buffer survives in a cache entry.
    pub(crate) fn restore(
        self: Box<Self>,
        cache: &PartitionCache,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<RoutingGraph> {
        check()?;
        let GraphLoad {
            key,
            header,
            parameters,
            dimension,
            metric,
            rotation_seed,
            body,
            descriptor,
            chunks,
            working_bytes,
            restore_bytes,
        } = *self;
        let BodyEntries::Internal(entries) = &body.entries else {
            unreachable!("internal source")
        };
        let kernel = VectorKernel::new(dimension, metric, rotation_seed)?;
        let graph = OwnedGraph::restore(
            Arc::clone(entries),
            kernel,
            GraphSource {
                index: key.index,
                tree: &key.tree_key,
                parent: key.partition,
                header,
                parameters,
            },
            descriptor,
            chunks,
            restore_bytes,
            check,
        )?;
        let bytes = body
            .bytes
            .checked_add(graph.additional_bytes() as u64)
            .ok_or_else(limit)?;
        if bytes > working_bytes as u64 {
            return Err(limit());
        }
        let body = Arc::new(CachedBody {
            epoch: header.cache_epoch(),
            bytes,
            entries: BodyEntries::Internal(Arc::clone(entries)),
            graph: Some(Box::new(graph)),
        });
        check()?;
        cache.install(key, Arc::clone(&body));
        Ok(RoutingGraph {
            body,
            m: parameters.m,
            working_bytes,
        })
    }
}

/// Reads authority and complete input from the caller's existing snapshot.
/// Cache warmth never affects admission; missing graph data is Corruption.
pub(crate) async fn read_graph<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    cache: &PartitionCache,
    manifest: &IndexManifest,
    request: BodyRequest<'_>,
    working_bytes: usize,
    retained_bytes: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<GraphRead> {
    check()?;
    let parameters = manifest
        .config()
        .fixed_graph()
        .ok_or_else(Error::invalid_argument)?
        .parameters();
    let index = manifest.logical_index_id();
    let BodyRequest {
        tree_key: tree,
        partition: parent,
        header,
    } = request;
    let header = *header;
    if header.level() <= 1
        || header.state() != PartitionState::Ready
        || header.entry_count() == 0
        || header.entry_count() > manifest.config().max_internal_partition_entries()
    {
        return Err(corrupt());
    }
    let n = header.entry_count() as usize;
    let d = manifest.config().dimension();
    let working_bytes = working_bytes
        .checked_sub(retained_bytes)
        .ok_or_else(limit)?;
    // Includes child allocations and conversion to shared storage, kernel construction,
    // private query scratch and both ordinal/mapped output arrays. Page buffers remain
    // bounded separately from complete bodies; no multi-graph prefetch is performed.
    let centroids = n
        .checked_mul(size_of::<ChildEntry>() + d * 4 + 64)
        .ok_or_else(limit)?;
    let overhead = VectorKernel::allocation_bound(d)
        .ok_or_else(limit)?
        .checked_add(n * (2 * size_of::<ChildEntry>() + 2 * size_of::<Candidate>()))
        .and_then(|v| v.checked_add(ktann_usearch::workspace_bound(n, parameters.m).ok()?))
        .and_then(|v| v.checked_add(2 * 1024 * 1024 + tree.as_bytes().len() * 4 + 4096 + d * 4))
        .ok_or_else(limit)?;
    if centroids
        .checked_add(overhead)
        .is_none_or(|bytes| bytes > working_bytes)
    {
        return Err(limit());
    }
    check()?;
    let Some(PersistentValue::GraphDescriptor(descriptor)) = txn
        .get(LogicalKey::GraphDescriptor {
            index,
            tree_key: tree.clone(),
            partition: parent,
        })
        .await?
    else {
        return Err(corrupt());
    };
    let length = usize::try_from(descriptor.bytes).map_err(|_| corrupt())?;
    if descriptor.nodes != header.entry_count()
        || descriptor.epoch != header.cache_epoch()
        || length > ktann_usearch::encoded_bound(n, parameters.m).map_err(|_| corrupt())?
    {
        return Err(corrupt());
    }
    let restore_bytes = centroids
        .checked_add(length)
        .and_then(|v| v.checked_add(1024))
        .and_then(|v| v.checked_add(ktann_usearch::decode_bound(n, parameters.m, length).ok()?))
        .ok_or_else(limit)?;
    let required = restore_bytes
        .checked_add(overhead)
        .and_then(|v| {
            v.checked_add(descriptor.chunks() as usize * (size_of::<(u32, GraphChunk)>() + 64))
        })
        .ok_or_else(limit)?;
    if required > working_bytes {
        return Err(limit());
    }
    let request = BodyRequest {
        tree_key: tree,
        partition: parent,
        header: &header,
    };
    let read = BodyRead::start(cache, manifest, &request)?;
    if let BodyRead::Cached(body) = &read {
        let BodyEntries::Internal(entries) = &body.entries else {
            return Err(corrupt());
        };
        if entries.len() != n {
            return Err(corrupt());
        }
        let graph = body
            .graph
            .as_ref()
            .expect("fixed internal cache body has a graph");
        if !graph.matches(
            GraphSource {
                index,
                tree,
                parent,
                header,
                parameters,
            },
            &descriptor,
        ) {
            return Err(corrupt());
        }
        return Ok(GraphRead::Cached(RoutingGraph {
            body: Arc::clone(body),
            m: parameters.m,
            working_bytes,
        }));
    }
    let body = read.read(txn, manifest, check).await?;
    let range = LogicalRange::graph_chunks(manifest, tree, parent)?;
    let mut cursor = None;
    let mut chunks = Vec::new();
    chunks
        .try_reserve_exact(descriptor.chunks() as usize)
        .map_err(|_| limit())?;
    loop {
        check()?;
        let page = txn.scan(&range, cursor.as_ref(), BODY_SCAN_LIMITS).await?;
        let (items, next) = page.into_parts();
        for item in items {
            let LogicalKey::GraphChunk { chunk, .. } = item.key() else {
                return Err(corrupt());
            };
            let ordinal = *chunk;
            let PersistentValue::GraphChunk(chunk) = item.into_value() else {
                return Err(corrupt());
            };
            let offset = chunks.len() * GRAPH_CHUNK_BYTES;
            if ordinal as usize != chunks.len()
                || chunks.len() >= descriptor.chunks() as usize
                || chunk.bytes().len() != (length - offset).min(GRAPH_CHUNK_BYTES)
            {
                return Err(corrupt());
            }
            // Detach from encoded page allocations before retaining the chunk.
            chunks.push((
                ordinal,
                GraphChunk(bytes::Bytes::copy_from_slice(chunk.bytes())),
            ));
        }
        cursor = next;
        if cursor.is_none() {
            break;
        }
    }
    if chunks.len() != descriptor.chunks() as usize {
        return Err(corrupt());
    }
    check()?;
    Ok(GraphRead::Load(Box::new(GraphLoad {
        key: CacheKey::new(index, tree.clone(), parent, PartitionKind::Internal),
        header,
        parameters,
        dimension: d,
        metric: manifest.config().metric(),
        rotation_seed: *manifest.rotation_seed(),
        body,
        descriptor,
        chunks,
        working_bytes,
        restore_bytes,
    })))
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

fn native_error(error: ktann_usearch::Error) -> Error {
    crate::search::local_hnsw::native_error(error)
}

#[cfg(test)]
pub(crate) mod tests;
