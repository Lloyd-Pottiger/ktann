//! One internal partition's immutable, canonical local routing graph.
//!
//! This module performs no IO, normalization, or production traversal. It
//! builds encoded records from borrowed Child Entries and restores complete
//! internal graph bodies. The existing numeric kernel supplies native distances.

#[cfg(feature = "benchmark")]
pub(crate) mod benchmark;
pub(crate) mod body;
pub(crate) mod persistence;

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;
use std::mem::size_of;

use sha2::{Digest, Sha256};

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::storage::keys::TreeKey;
use crate::storage::values::ChildEntry;

use super::numeric::{VectorKernel, compare_finite};

/// An ordinal into the canonical Child Entries and its retained routing score.
pub(crate) use ktann_usearch::Hit as Candidate;
/// Reusable native scratch; allocate once and reuse across parent partitions.
pub(crate) use ktann_usearch::Workspace;

/// Immutable construction quality and graph identity parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Parameters {
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    pub(crate) seed: [u8; 32],
}

/// Validated canonical child projections, borrowed without copying centroids.
#[derive(Clone, Copy)]
pub(crate) struct Centroids<'a> {
    kernel: &'a VectorKernel,
    children: &'a [ChildEntry],
    bytes: usize,
}

impl fmt::Debug for Centroids<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Centroids")
            .field("children", &self.children.len())
            .finish()
    }
}

impl<'a> Centroids<'a> {
    /// Validates finite coordinates, dimensions, capacity, and increasing keys.
    ///
    /// Inputs are already in routing space. Zero cosine centroids and ordinary
    /// inner-product magnitudes are valid; neither is normalized here.
    pub(crate) fn new(
        kernel: &'a VectorKernel,
        children: &'a [ChildEntry],
        capacity: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        if capacity == 0 || capacity > ktann_usearch::MAX_NODES || children.len() > capacity {
            return Err(Error::invalid_argument());
        }
        check()?;
        let mut previous = None;
        for child in children {
            check()?;
            if previous.is_some_and(|key| key >= child.child()) {
                return Err(Error::invalid_argument());
            }
            // Validate once at the borrowed-view boundary, never per graph edge.
            kernel.validate_routing(child.centroid())?;
            previous = Some(child.child());
        }
        let bytes = kernel
            .dimension()
            .checked_mul(size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(size_of::<ChildEntry>() + 64))
            .and_then(|bytes| bytes.checked_mul(children.len()))
            .ok_or_else(limit)?;
        Ok(Self {
            kernel,
            children,
            bytes,
        })
    }

    /// Maps a result ordinal to the authoritative child Partition Key.
    #[cfg(test)]
    pub(crate) fn child(self, candidate: Candidate) -> PartitionKey {
        self.children[candidate.node as usize].child()
    }

    /// Exact local top-k baseline, independent of graph construction.
    ///
    /// Uses O(min(n, beam)) retained candidates, including the exhaustive case.
    pub(crate) fn exact(
        self,
        query: &[f32],
        beam: usize,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<Candidate>> {
        self.validate_query(query, beam)?;
        let beam = beam.min(self.children.len());
        admit(self.bytes, query.len() * 4, 2 * beam, 0, working_bytes)?;
        check()?;
        let mut best = BinaryHeap::new();
        best.try_reserve_exact(beam).map_err(|_| limit())?;
        for (node, child) in self.children.iter().enumerate() {
            if node % 64 == 0 {
                check()?;
            }
            let hit = Ranked(Candidate {
                node: node as u32,
                distance: self.distance(query, child.centroid()),
            });
            if best.len() < beam {
                best.push(hit);
            } else if let Some(mut worst) = best.peek_mut()
                && hit < *worst
            {
                *worst = hit;
            }
        }
        let mut output = Vec::new();
        output.try_reserve_exact(beam).map_err(|_| limit())?;
        while let Some(Ranked(hit)) = best.pop() {
            if output.len() % 64 == 0 {
                check()?;
            }
            output.push(hit);
        }
        output.reverse();
        check()?;
        Ok(output)
    }

    fn validate_query(self, query: &[f32], beam: usize) -> Result<()> {
        if beam == 0 {
            return Err(Error::invalid_argument());
        }
        self.kernel.validate_routing(query)?;
        Ok(())
    }

    fn distance(self, a: &[f32], b: &[f32]) -> f64 {
        self.kernel.routing_distances_validated([a], b)[0]
    }
}

/// Builds topology over borrowed projections; no Rust reference survives native work.
pub(crate) fn build_graph(
    centroids: Centroids<'_>,
    parameters: Parameters,
    tree: &TreeKey,
    parent: PartitionKey,
    working_bytes: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<(ktann_usearch::Graph, [u8; 32])> {
    if parameters.ef_construction < parameters.m {
        return Err(Error::invalid_argument());
    }
    let bound =
        ktann_usearch::build_bound(centroids.children.len(), parameters.m).map_err(native_error)?;
    if centroids
        .bytes
        .checked_add(bound)
        .is_none_or(|bytes| bytes > working_bytes)
    {
        return Err(limit());
    }
    let seed = partition_seed(parameters.seed, tree, parent);
    let distance = |a: u32, b: u32| {
        centroids.distance(
            centroids.children[a as usize].centroid(),
            centroids.children[b as usize].centroid(),
        )
    };
    let graph = controlled(check, |proceed| {
        ktann_usearch::Graph::build(
            centroids.children.len(),
            parameters.m,
            parameters.ef_construction,
            seed,
            working_bytes - centroids.bytes,
            &distance,
            proceed,
        )
    })?;
    let identity = persistence::identity(centroids, parameters, tree, parent, check)?;
    Ok((graph, identity))
}

/// Executes graph routing against a validated borrowed view of a complete body.
#[expect(
    clippy::too_many_arguments,
    reason = "the existing search inputs plus the borrowed graph and centroids"
)]
pub(crate) fn search_graph(
    centroids: Centroids<'_>,
    graph: &ktann_usearch::Graph,
    query: &[f32],
    beam: usize,
    ef: usize,
    working_bytes: usize,
    scratch: &mut Workspace,
    check: &dyn Fn() -> Result<()>,
) -> Result<(Vec<Candidate>, ktann_usearch::Stats)> {
    search_graph_scoring(
        centroids,
        graph,
        query,
        beam,
        ef,
        working_bytes,
        scratch,
        check,
        false,
    )
}

/// Shares traversal admission between scalar queries and batched write routing.
#[expect(
    clippy::too_many_arguments,
    reason = "graph search inputs with caller-owned workspace and scoring mode"
)]
pub(crate) fn search_graph_scoring(
    centroids: Centroids<'_>,
    graph: &ktann_usearch::Graph,
    query: &[f32],
    beam: usize,
    ef: usize,
    working_bytes: usize,
    scratch: &mut Workspace,
    check: &dyn Fn() -> Result<()>,
    batched: bool,
) -> Result<(Vec<Candidate>, ktann_usearch::Stats)> {
    centroids.validate_query(query, beam)?;
    if ef == 0 {
        return Err(Error::invalid_argument());
    }
    let beam = beam.min(centroids.children.len());
    admit(
        centroids.bytes,
        query.len() * 4,
        beam,
        graph
            .resident_bytes()
            .checked_add(scratch.allocated_bytes())
            .ok_or_else(limit)?,
        working_bytes,
    )?;
    let mut hits = Vec::new();
    hits.try_reserve_exact(beam).map_err(|_| limit())?;
    hits.resize(beam, Candidate::default());
    let distance =
        |_: u32, node: u32| centroids.distance(query, centroids.children[node as usize].centroid());
    let distance_four = |_: u32, nodes: &[u32], output: &mut [f64]| match *nodes {
        [a, b] => output.copy_from_slice(&centroids.kernel.routing_distances_validated(
            [
                centroids.children[a as usize].centroid(),
                centroids.children[b as usize].centroid(),
            ],
            query,
        )),
        [a, b, c] => output.copy_from_slice(&centroids.kernel.routing_distances_validated(
            [
                centroids.children[a as usize].centroid(),
                centroids.children[b as usize].centroid(),
                centroids.children[c as usize].centroid(),
            ],
            query,
        )),
        [a, b, c, d] => output.copy_from_slice(&centroids.kernel.routing_distances_validated(
            [
                centroids.children[a as usize].centroid(),
                centroids.children[b as usize].centroid(),
                centroids.children[c as usize].centroid(),
                centroids.children[d as usize].centroid(),
            ],
            query,
        )),
        _ => unreachable!("native distance batch width"),
    };
    let stats = controlled(check, |proceed| {
        if batched {
            graph.search_batched(ef, &distance, &distance_four, proceed, scratch, &mut hits)
        } else {
            graph.search(ef, &distance, proceed, scratch, &mut hits)
        }
    })?;
    Ok((hits, stats))
}

/// Domain-separated seed framing uses the canonical Tree Key bytes and parent ID.
fn partition_seed(seed: [u8; 32], tree: &TreeKey, parent: PartitionKey) -> u64 {
    let mut hash = Sha256::new();
    hash.update(b"ktann/local-hnsw/v1\0");
    hash.update(seed);
    hash.update((tree.as_bytes().len() as u64).to_le_bytes());
    hash.update(tree.as_bytes());
    hash.update(parent.get().to_le_bytes());
    u64::from_le_bytes(
        hash.finalize()[..8]
            .try_into()
            .expect("eight-byte digest prefix"),
    )
}

fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

pub(crate) fn native_error(error: ktann_usearch::Error) -> Error {
    Error::new(match error {
        ktann_usearch::Error::InvalidArgument => ErrorKind::InvalidArgument,
        ktann_usearch::Error::LimitExceeded => ErrorKind::LimitExceeded,
        ktann_usearch::Error::Cancelled => ErrorKind::Cancelled,
        ktann_usearch::Error::Invariant => ErrorKind::Other,
        ktann_usearch::Error::InvalidEncoding => ErrorKind::Corruption,
    })
}

/// Preserve the caller's exact cancellation/deadline error across the C++ ABI.
fn controlled<T>(
    check: &dyn Fn() -> Result<()>,
    work: impl FnOnce(&dyn Fn() -> bool) -> std::result::Result<T, ktann_usearch::Error>,
) -> Result<T> {
    check()?;
    let error = Cell::new(None);
    let proceed = || match check() {
        Ok(()) => true,
        Err(failure) => {
            error.set(Some(failure));
            false
        }
    };
    let result = work(&proceed).map_err(native_error);
    match error.into_inner() {
        Some(error) => Err(error),
        None => result,
    }
}

/// Charge retained inputs, native allocations, and the caller's candidate buffers.
fn admit(
    centroids: usize,
    query: usize,
    candidate_slots: usize,
    native: usize,
    ceiling: usize,
) -> Result<()> {
    let bytes = candidate_slots
        .checked_mul(size_of::<Candidate>())
        .and_then(|bytes| bytes.checked_add(centroids))
        .and_then(|bytes| bytes.checked_add(query))
        .and_then(|bytes| bytes.checked_add(native))
        .and_then(|bytes| bytes.checked_add(1024));
    if bytes.is_none_or(|bytes| bytes > ceiling) {
        return Err(limit());
    }
    Ok(())
}

/// A max-heap entry using the same finite-distance/key ordering as traversal.
#[derive(Clone, Copy, Debug)]
struct Ranked(Candidate);

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_finite(self.0.distance, other.0.distance).then(self.0.node.cmp(&other.0.node))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}

#[cfg(test)]
pub(crate) mod tests;
