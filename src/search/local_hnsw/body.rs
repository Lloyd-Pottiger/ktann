//! Complete snapshot-bound internal graph partitions, shared as one cached body.

use super::persistence::{GraphSource, binding, restore_graph};
use super::*;
use crate::storage::values::{GraphChunk, GraphDescriptor};

/// One internal graph body: authoritative child projections, numeric kernel,
/// validated topology and source identity have a single owner.
/// No self-reference or native vector copy is needed. Each query owns its scratch.
pub(crate) struct InternalGraphBody {
    children: Box<[ChildEntry]>,
    kernel: VectorKernel,
    graph: ktann_usearch::Graph,
    identity: [u8; 32],
    descriptor: GraphDescriptor,
    centroid_bytes: usize,
}

impl InternalGraphBody {
    /// Restores persisted topology without invoking construction or changing quality parameters.
    pub(crate) fn restore(
        children: Box<[ChildEntry]>,
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
        let (graph, identity) =
            restore_graph(centroids, source, &descriptor, chunks, working_bytes, check)?;
        Ok(Self {
            children,
            kernel,
            graph,
            identity,
            descriptor,
            centroid_bytes,
        })
    }

    /// Authoritative projections used by both routing modes and edge validation.
    pub(crate) fn children(&self) -> &[ChildEntry] {
        &self.children
    }

    /// Validates the snapshot envelope even on a warm read. Identity includes index,
    /// tree, parent, configuration, metric, canonical children and centroids.
    pub(crate) fn matches(&self, source: GraphSource<'_>, descriptor: &GraphDescriptor) -> bool {
        self.descriptor == *descriptor
            && descriptor.binding == binding(self.identity, source.index, source.header)
    }

    /// Resident ownership in addition to the entries already charged by the reader.
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
