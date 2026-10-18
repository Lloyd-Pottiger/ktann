//! Owned immutable query graphs; the cache and graph share one centroid allocation.

use std::sync::Arc;

use super::persistence::{GraphSource, binding};
use super::*;
use crate::storage::values::{GraphChunk, GraphDescriptor};

/// A restored native graph and the exact projections used to validate its binding.
/// No self-reference or native vector copy is needed. Each query owns its scratch.
pub(crate) struct OwnedGraph {
    children: Arc<[ChildEntry]>,
    kernel: VectorKernel,
    graph: ktann_usearch::Graph,
    identity: [u8; 32],
    descriptor: GraphDescriptor,
    centroid_bytes: usize,
}

impl OwnedGraph {
    /// Restores persisted topology without invoking construction or changing quality parameters.
    pub(crate) fn restore(
        children: Arc<[ChildEntry]>,
        kernel: VectorKernel,
        source: GraphSource<'_>,
        descriptor: GraphDescriptor,
        chunks: Vec<(u32, GraphChunk)>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let centroids =
            Centroids::new(&kernel, &children, children.len(), check).map_err(|error| {
                if error.kind() == ErrorKind::InvalidArgument {
                    corrupt()
                } else {
                    error
                }
            })?;
        let centroid_bytes = centroids.bytes;
        let restored =
            LocalHnsw::restore(centroids, source, &descriptor, chunks, working_bytes, check)?;
        let LocalHnsw {
            graph, identity, ..
        } = restored;
        Ok(Self {
            children,
            kernel,
            graph,
            identity,
            descriptor,
            centroid_bytes,
        })
    }

    /// Validates the snapshot envelope even on a warm read. Identity includes index,
    /// tree, parent, configuration, metric, canonical children and centroids.
    pub(crate) fn matches(&self, source: GraphSource<'_>, descriptor: &GraphDescriptor) -> bool {
        self.descriptor == *descriptor
            && descriptor.binding == binding(self.identity, source.index, source.header)
    }

    /// Resident ownership in addition to the entries already charged by CachedBody.
    pub(crate) fn additional_bytes(&self) -> usize {
        size_of::<Self>() + 32 + self.kernel.heap_bytes() + self.graph.resident_bytes()
    }

    /// Exact baseline over the same validated centroid allocation.
    pub(crate) fn exact(
        &self,
        query: &[f32],
        beam: usize,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<Candidate>> {
        Centroids {
            kernel: &self.kernel,
            children: &self.children,
            bytes: self.centroid_bytes,
        }
        .exact(query, beam, working_bytes, check)
    }

    /// Admission by the owning reader includes the kernel, body and all query state.
    pub(crate) fn search(
        &self,
        query: &[f32],
        beam: usize,
        ef: usize,
        working_bytes: usize,
        scratch: &mut Workspace,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<(Vec<Candidate>, ktann_usearch::Stats)> {
        search_graph(
            Centroids {
                kernel: &self.kernel,
                children: &self.children,
                bytes: self.centroid_bytes,
            },
            &self.graph,
            query,
            beam,
            ef,
            working_bytes,
            scratch,
            check,
        )
    }
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}
