//! Private integration of USearch's graph-only C++ engine.
//!
//! No vectors, Rust references, or callbacks survive a call into C++. Node
//! ordinals are dense and immutable. Query scratch belongs to the caller.

use std::fmt;

#[allow(unsafe_code)] // CXX checks signatures; raw pointers stay inside its generated bridge.
#[cxx::bridge(namespace = "ktann_usearch")]
mod ffi {
    /// One canonical ordinal and the distance computed during graph search.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    struct Hit {
        /// Ordinal in the canonical Child Entry sequence.
        node: u32,
        /// Unmodified finite routing distance; smaller is better.
        distance: f64,
    }

    /// Work performed inside the local graph, independent of partition visits.
    #[derive(Clone, Copy, Debug, Default)]
    struct Stats {
        /// Number of calls to the borrowed distance kernel.
        distances: usize,
        /// Number of graph expansion iterations.
        visits: usize,
    }

    extern "Rust" {
        type MetricContext<'a>;
        fn measure(context: &MetricContext, a: u32, b: u32) -> f64;
        fn proceed(context: &MetricContext) -> bool;
    }

    unsafe extern "C++" {
        include!("native/graph.h");
        type NativeGraph;
        type NativeWorkspace;
        fn build(
            n: u32,
            m: u32,
            ef: u32,
            seed: u64,
            limit: usize,
            context: &MetricContext,
        ) -> Result<UniquePtr<NativeGraph>>;
        fn workspace(n: u32, m: u32, limit: usize) -> Result<UniquePtr<NativeWorkspace>>;
        fn search(
            self: &NativeGraph,
            ef: u32,
            context: &MetricContext,
            workspace: Pin<&mut NativeWorkspace>,
            output: &mut [Hit],
        ) -> Result<Stats>;
        fn resident_bytes(self: &NativeGraph) -> usize;
        fn peak_bytes(self: &NativeGraph) -> usize;
        fn entry(self: &NativeGraph) -> u32;
        fn level(self: &NativeGraph, node: u32) -> u32;
        fn neighbors(self: &NativeGraph, node: u32, level: u32, output: &mut [u32]) -> usize;
        fn repaired(self: &NativeGraph) -> bool;
        fn allocated_bytes(self: &NativeWorkspace) -> usize;
        fn allocation_count(self: &NativeWorkspace) -> usize;
    }
}

/// A directly returned graph distance and canonical node ordinal.
pub use ffi::Hit;
/// Logical graph work, separate from KTANN partition-visit budgets.
pub use ffi::Stats;

/// Fixed maximum node level (seventeen layers, including layer zero).
pub const MAX_LEVEL: u32 = 16;
/// Maximum partition size supported by this bridge.
pub const MAX_NODES: usize = 65_536;

/// The bridge's bounded failure categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// An invalid size, configuration, or non-finite callback distance.
    InvalidArgument,
    /// Allocation or admission failed.
    LimitExceeded,
    /// The operation-control callback rejected further work.
    Cancelled,
    /// USearch violated an internal graph invariant.
    Invariant,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidArgument => "invalid graph input",
            Self::LimitExceeded => "graph resource limit exceeded",
            Self::Cancelled => "graph operation cancelled",
            Self::Invariant => "graph invariant violated",
        })
    }
}

impl std::error::Error for Error {}

impl From<cxx::Exception> for Error {
    fn from(value: cxx::Exception) -> Self {
        match value.what() {
            "limit" | "Out of memory" | "Out of memory!" | "std::bad_alloc" => Self::LimitExceeded,
            "cancelled" => Self::Cancelled,
            "invalid" => Self::InvalidArgument,
            _ => Self::Invariant,
        }
    }
}

// Opaque borrowed callbacks live only for a synchronous native call. C++ never
// retains them, invokes them on other threads, or unwinds through a Rust frame.
struct MetricContext<'a> {
    distance: &'a dyn Fn(u32, u32) -> f64,
    check: &'a dyn Fn() -> bool,
}

fn measure(context: &MetricContext<'_>, a: u32, b: u32) -> f64 {
    (context.distance)(a, b)
}

fn proceed(context: &MetricContext<'_>) -> bool {
    (context.check)()
}

fn validate(n: usize, m: usize) -> Result<(), Error> {
    if n > MAX_NODES || !(2..=MAX_NODES / 2).contains(&m) {
        return Err(Error::InvalidArgument);
    }
    Ok(())
}

/// Conservative admission bound for reusable query scratch, including overhead.
pub fn workspace_bound(n: usize, m: usize) -> Result<usize, Error> {
    validate(n, m)?;
    n.checked_add(2 * m + 1)
        .and_then(|n| n.checked_mul(128))
        .and_then(|n| n.checked_add(8192))
        .ok_or(Error::LimitExceeded)
}

/// Conservative graph-plus-build-scratch bound; excludes caller-owned vectors.
pub fn build_bound(n: usize, m: usize) -> Result<usize, Error> {
    let scratch = workspace_bound(n, m)?;
    let node = 17 * (4 + 4 * m) + 4 * m + 192;
    n.checked_mul(node)
        .and_then(|v| v.checked_add(1024 * m + 32768))
        .and_then(|v| v.checked_add(scratch))
        .ok_or(Error::LimitExceeded)
}

/// Immutable graph topology. Centroid ownership stays with KTANN.
pub struct Graph {
    inner: cxx::UniquePtr<ffi::NativeGraph>,
    n: usize,
    m: usize,
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("nodes", &self.n)
            .field("m", &self.m)
            .finish()
    }
}

impl Graph {
    /// Builds in ordinal order with a fixed seed and borrowed distance callback.
    ///
    /// The callback receives only ordinals in `0..n`. It must return finite
    /// symmetric routing distances. Failure discards the unpublished graph.
    pub fn build(
        n: usize,
        m: usize,
        ef: usize,
        seed: u64,
        limit: usize,
        distance: &dyn Fn(u32, u32) -> f64,
        check: &dyn Fn() -> bool,
    ) -> Result<Self, Error> {
        if ef < m {
            return Err(Error::InvalidArgument);
        }
        if build_bound(n, m)? > limit {
            return Err(Error::LimitExceeded);
        }
        let context = MetricContext { distance, check };
        let inner = ffi::build(
            n as u32,
            m as u32,
            ef.min(n.max(1)) as u32,
            seed,
            limit,
            &context,
        )?;
        Ok(Self { inner, n, m })
    }

    /// Searches into caller-owned output without recomputing result distances.
    ///
    /// `n` is the query sentinel supplied as the first distance-callback
    /// argument; the second is always a node ordinal. Output length is the
    /// requested beam, already clipped to `n`. Scratch is reusable across graphs.
    pub fn search(
        &self,
        ef: usize,
        distance: &dyn Fn(u32, u32) -> f64,
        check: &dyn Fn() -> bool,
        scratch: &mut Workspace,
        output: &mut [Hit],
    ) -> Result<Stats, Error> {
        if ef == 0 || output.len() > self.n || scratch.n < self.n || scratch.m < self.m {
            return Err(Error::InvalidArgument);
        }
        let context = MetricContext { distance, check };
        Ok(self.inner.search(
            ef.max(output.len()).min(self.n) as u32,
            &context,
            scratch.inner.pin_mut(),
            output,
        )?)
    }

    /// Returns charged resident graph bytes, including allocation overhead.
    pub fn resident_bytes(&self) -> usize {
        self.inner.resident_bytes()
    }

    /// Returns peak charged construction bytes, excluding external vectors.
    pub fn peak_bytes(&self) -> usize {
        self.inner.peak_bytes()
    }

    /// Returns the entry ordinal, or `None` for an empty graph.
    pub fn entry(&self) -> Option<u32> {
        (self.n != 0).then(|| self.inner.entry())
    }

    /// Returns a node's highest layer, or `None` for an invalid ordinal.
    pub fn level(&self, node: u32) -> Option<u32> {
        ((node as usize) < self.n).then(|| self.inner.level(node))
    }

    /// Copies one adjacency list into a buffer of at least `2*M` ordinals.
    pub fn neighbors(&self, node: u32, level: u32, output: &mut [u32]) -> Result<usize, Error> {
        if self.level(node).is_none_or(|max| level > max) || output.len() < 2 * self.m {
            return Err(Error::InvalidArgument);
        }
        Ok(self.inner.neighbors(node, level, output))
    }

    /// Reports whether construction needed the deterministic connectivity repair.
    pub fn repaired(&self) -> bool {
        self.inner.repaired()
    }
}

/// Exclusive, reusable query workspace with no graph or vector ownership.
pub struct Workspace {
    inner: cxx::UniquePtr<ffi::NativeWorkspace>,
    n: usize,
    m: usize,
}

impl fmt::Debug for Workspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Workspace")
            .field("capacity", &self.n)
            .field("m", &self.m)
            .finish()
    }
}

impl Workspace {
    /// Allocates bounded scratch once, before queries begin.
    pub fn new(n: usize, m: usize, limit: usize) -> Result<Self, Error> {
        if workspace_bound(n, m)? > limit {
            return Err(Error::LimitExceeded);
        }
        Ok(Self {
            inner: ffi::workspace(n as u32, m as u32, limit)?,
            n,
            m,
        })
    }

    /// Returns charged allocation bytes, including native workspace metadata.
    pub fn allocated_bytes(&self) -> usize {
        self.inner.allocated_bytes()
    }

    /// Returns total successful native allocations, including freed allocations.
    pub fn allocation_count(&self) -> usize {
        self.inner.allocation_count()
    }
}

// SAFETY: immutable native graphs retain neither Rust pointers nor mutable
// query contexts. Searches only read graph nodes and use exclusive Workspace.
#[expect(
    unsafe_code,
    reason = "immutable native graph ownership is transferable"
)]
unsafe impl Send for Graph {}
// SAFETY: searches mutate only caller-exclusive scratch and native atomic locks.
#[expect(
    unsafe_code,
    reason = "immutable native graph with external query scratch"
)]
unsafe impl Sync for Graph {}

// SAFETY: scratch owns its allocations; all native mutation requires &mut self.
#[expect(
    unsafe_code,
    reason = "native scratch ownership is exclusive and transferable"
)]
unsafe impl Send for Workspace {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_bounds_cover_small_sparse_and_large_degree_allocations() {
        for (n, m) in [(0, 2), (1, 2), (9, 3), (128, 12), (16, 129), (1, 32768)] {
            let bound = build_bound(n, m).unwrap();
            let graph = Graph::build(
                n,
                m,
                m,
                123,
                bound,
                &|a, b| {
                    let difference = f64::from(a) - f64::from(b);
                    difference * difference
                },
                &|| true,
            )
            .unwrap();
            assert!(graph.peak_bytes() <= bound);
            assert!(graph.resident_bytes() <= graph.peak_bytes());
            let bound = workspace_bound(n, m).unwrap();
            let mut scratch = Workspace::new(n, m, bound).unwrap();
            assert!(scratch.allocated_bytes() <= bound);
            let allocations = scratch.allocation_count();
            let mut output = vec![Hit::default(); n];
            let stats = graph
                .search(
                    usize::MAX,
                    &|query, node| {
                        assert_eq!(query as usize, n);
                        assert!((node as usize) < n);
                        -f64::from(node)
                    },
                    &|| true,
                    &mut scratch,
                    &mut output,
                )
                .unwrap();
            assert_eq!(scratch.allocation_count(), allocations);
            assert_eq!(output.len(), n);
            assert_eq!(stats.distances, n);
            assert_eq!(stats.visits, 0);
            for (rank, hit) in output.iter().enumerate() {
                assert_eq!(hit.node as usize, n - rank - 1);
                assert_eq!(hit.distance, -f64::from(hit.node));
            }
        }
    }

    #[test]
    fn malformed_callbacks_cancellation_and_small_workspaces_fail_without_unwinding_rust() {
        let limit = build_bound(8, 2).unwrap();
        for distance in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                Graph::build(8, 2, 8, 0, limit, &|_, _| distance, &|| true),
                Err(Error::InvalidArgument)
            ));
        }
        assert!(matches!(
            Graph::build(8, 2, 8, 0, limit, &|_, _| 0.0, &|| false),
            Err(Error::Cancelled)
        ));
        let graph = Graph::build(8, 2, 8, 0, limit, &|_, _| 0.0, &|| true).unwrap();
        let mut scratch = Workspace::new(7, 2, workspace_bound(7, 2).unwrap()).unwrap();
        assert!(matches!(
            graph.search(
                8,
                &|_, _| 0.0,
                &|| true,
                &mut scratch,
                &mut [Hit::default()]
            ),
            Err(Error::InvalidArgument)
        ));
        let mut neighbors = [0; 4];
        assert!(matches!(
            graph.neighbors(8, 0, &mut neighbors),
            Err(Error::InvalidArgument)
        ));
        assert!(matches!(
            graph.neighbors(0, MAX_LEVEL + 1, &mut neighbors),
            Err(Error::InvalidArgument)
        ));
        assert!(matches!(Workspace::new(8, 2, 0), Err(Error::LimitExceeded)));
    }
}
