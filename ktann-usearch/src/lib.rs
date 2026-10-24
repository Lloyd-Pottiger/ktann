//! Private integration of USearch's graph-only C++ engine.
//!
//! No vectors, Rust references, or callbacks survive a call into C++. Node
//! ordinals are dense and immutable. Query scratch belongs to the caller.

use std::fmt;

mod persistence;
#[cfg(test)]
mod updates_tests;
pub use persistence::{NATIVE_ENCODING, NATIVE_HEADER_BYTES, decode_bound, encoded_bound};

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
        fn has_four(context: &MetricContext) -> bool;
        fn has_tails(context: &MetricContext) -> bool;
        fn measure_four(context: &MetricContext, a: u32, nodes: &[u32], output: &mut [f64]);
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
        fn restore(
            bytes: &[u8],
            n: u32,
            m: u32,
            limit: usize,
            context: &MetricContext,
        ) -> Result<UniquePtr<NativeGraph>>;
        fn evolve(
            graph: UniquePtr<NativeGraph>,
            mapping: &[u32],
            changed: &[u32],
            ef: u32,
            seed: u64,
            limit: usize,
            context: &MetricContext,
        ) -> Result<UniquePtr<NativeGraph>>;
        fn retain(
            graph: UniquePtr<NativeGraph>,
            survivors: &[u32],
            ef: u32,
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
        fn encoded_size(self: &NativeGraph) -> usize;
        fn save(self: &NativeGraph, output: &mut [u8], context: &MetricContext) -> Result<()>;
        fn resident_bytes(self: &NativeGraph) -> usize;
        fn peak_bytes(self: &NativeGraph) -> usize;
        fn entry(self: &NativeGraph) -> u32;
        fn level(self: &NativeGraph, node: u32) -> u32;
        fn neighbors(self: &NativeGraph, node: u32, level: u32, output: &mut [u32]) -> usize;
        fn allocated_bytes(self: &NativeWorkspace) -> usize;
        fn allocation_count(self: &NativeWorkspace) -> usize;
    }
}

/// A directly returned graph distance and canonical node ordinal.
pub use ffi::Hit;
/// Logical graph work, separate from KTANN partition-visit budgets.
pub use ffi::Stats;

/// Global node-level ceiling; the degree-specific limit is [`max_level`].
pub const MAX_LEVEL: u32 = 16;
/// Maximum partition size supported by this bridge.
pub const MAX_NODES: usize = 65_536;

/// The bridge's bounded failure categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// An invalid size, configuration, or non-finite callback distance.
    InvalidArgument,
    /// Malformed or structurally invalid topology bytes.
    InvalidEncoding,
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
            Self::InvalidEncoding => "invalid graph encoding",
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

/// Synchronous ordered scoring of two through four requested node ordinals.
pub type DistanceBatch<'a> = dyn Fn(u32, &[u32], &mut [f64]) + 'a;

// Opaque borrowed callbacks live only for a synchronous native call. C++ never
// retains them, invokes them on other threads, or unwinds through a Rust frame.
struct MetricContext<'a> {
    distance: &'a dyn Fn(u32, u32) -> f64,
    four: Option<&'a DistanceBatch<'a>>,
    tails: bool,
    check: &'a dyn Fn() -> bool,
}

fn measure(context: &MetricContext<'_>, a: u32, b: u32) -> f64 {
    (context.distance)(a, b)
}

/// Four ordered native ordinals; no callback or borrowed vector escapes this call.
fn has_four(context: &MetricContext<'_>) -> bool {
    context.four.is_some()
}

fn measure_four(context: &MetricContext<'_>, a: u32, nodes: &[u32], output: &mut [f64]) {
    assert!((2..=4).contains(&nodes.len()) && nodes.len() == output.len());
    context.four.expect("enabled native batch")(a, nodes, output);
}

fn has_tails(context: &MetricContext<'_>) -> bool {
    context.tails
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

/// Highest hierarchy level whose expected population at MAX_NODES is at least
/// one. Integer arithmetic matches the native sampler and bounds persisted size.
pub fn max_level(m: usize) -> u32 {
    assert!(m >= 2);
    let mut level = 0;
    let mut population = MAX_NODES;
    while population >= m {
        population /= m;
        level += 1;
    }
    level
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
    /// Consumes a private graph, replaces selected old slots, appends new slots,
    /// and permutes the result into canonical order. `mapping[slot]` is the final
    /// ordinal; distances use pre-permutation slots throughout this operation.
    /// Deletion is deliberately handled by the caller's existing full builder.
    #[expect(
        clippy::too_many_arguments,
        reason = "private topology transform with borrowed metric and operation controls"
    )]
    pub fn evolve(
        self,
        mapping: &[u32],
        changed: &[u32],
        ef: usize,
        seed: u64,
        limit: usize,
        distance: &dyn Fn(u32, u32) -> f64,
        check: &dyn Fn() -> bool,
    ) -> Result<Self, Error> {
        let n = mapping.len();
        if n < self.n || ef < self.m || n == 0 {
            return Err(Error::InvalidArgument);
        }
        let native_limit = limit.checked_sub(n + 128).ok_or(Error::LimitExceeded)?;
        if build_bound(n, self.m)? > native_limit {
            return Err(Error::LimitExceeded);
        }
        let mut seen = Vec::new();
        seen.try_reserve_exact(n)
            .map_err(|_| Error::LimitExceeded)?;
        seen.resize(n, false);
        for &target in mapping {
            if !check() {
                return Err(Error::Cancelled);
            }
            let target = target as usize;
            if target >= n || std::mem::replace(&mut seen[target], true) {
                return Err(Error::InvalidArgument);
            }
        }
        if changed.iter().any(|&slot| slot as usize >= self.n)
            || changed.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::InvalidArgument);
        }
        let context = MetricContext {
            distance,
            four: None,
            tails: false,
            check,
        };
        let inner = ffi::evolve(
            self.inner,
            mapping,
            changed,
            ef.min(n) as u32,
            seed,
            native_limit,
            &context,
        )?;
        Ok(Self {
            inner,
            n,
            m: self.m,
        })
    }

    /// Physically removes all but the sorted old `survivors` from a private
    /// graph. Distances address the compacted survivor order. Existing levels
    /// are retained; only neighborhoods touching removed nodes are refreshed.
    pub fn retain(
        self,
        survivors: &[u32],
        ef: usize,
        limit: usize,
        distance: &dyn Fn(u32, u32) -> f64,
        check: &dyn Fn() -> bool,
    ) -> Result<Self, Error> {
        let n = survivors.len();
        if n == 0
            || ef < self.m
            || survivors.iter().any(|&slot| slot as usize >= self.n)
            || survivors.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::InvalidArgument);
        }
        let context = MetricContext {
            distance,
            four: None,
            tails: false,
            check,
        };
        let inner = ffi::retain(self.inner, survivors, ef.min(n) as u32, limit, &context)?;
        Ok(Self {
            inner,
            n,
            m: self.m,
        })
    }

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
        let context = MetricContext {
            distance,
            four: None,
            tails: false,
            check,
        };
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
        let context = MetricContext {
            distance,
            four: None,
            tails: false,
            check,
        };
        Ok(self.inner.search(
            ef.max(output.len()).min(self.n) as u32,
            &context,
            scratch.inner.pin_mut(),
            output,
        )?)
    }

    /// Experimental four-lane scoring with identical traversal and scalar tails.
    #[allow(clippy::too_many_arguments)]
    pub fn search_four(
        &self,
        ef: usize,
        distance: &dyn Fn(u32, u32) -> f64,
        four: &DistanceBatch<'_>,
        tails: bool,
        check: &dyn Fn() -> bool,
        scratch: &mut Workspace,
        output: &mut [Hit],
    ) -> Result<Stats, Error> {
        if ef == 0 || output.len() > self.n || scratch.n < self.n || scratch.m < self.m {
            return Err(Error::InvalidArgument);
        }
        let context = MetricContext {
            distance,
            four: Some(four),
            tails,
            check,
        };
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

#[cfg(test)]
mod atomic_size_tests {
    use super::*;

    #[test]
    fn supported_internal_shapes_fit_one_fdb_publication() {
        // A split distributes at most threshold+2 nodes. A merge additionally
        // retains an underfull source (minimum<=128); only one target per batch.
        for capacity in [8192, 16384] {
            for m in 2..=16 {
                let nodes = capacity + 2 + 127;
                let graph_bytes = encoded_bound(nodes, m).unwrap() + NATIVE_HEADER_BYTES;
                // Two nonempty bodies can need one extra partial chunk.
                let chunks = graph_bytes.div_ceil(99_995) + 1;
                // Conservative maximum physical key length; charge mutation,
                // both conflict endpoints, framing and per-mutation overhead.
                let affected = graph_bytes + chunks * (3 * 10_000 + 64) + 900_000;
                assert!(
                    affected < 10_000_000,
                    "capacity={capacity} M={m}: {affected}"
                );
            }
        }
    }
}

#[cfg(test)]
mod quartet_tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// Batched callbacks preserve distance bits, request order and traversal work.
    #[test]
    fn quartets_match_scalar_traces_and_reuse() {
        for n in [17, 128, 374] {
            for dot in [false, true] {
                let rows: Vec<[f64; 8]> = (0..=n)
                    .map(|i| {
                        std::array::from_fn(|d| {
                            // Repeated rows exercise ordinal tie-breaking.
                            (((i / 2 + 1) * (d + 7) * 73) % 251) as f64 / 251.0
                        })
                    })
                    .collect();
                let distance = |a: u32, b: u32| {
                    let mut value = 0.0;
                    for (av, bv) in rows[a as usize].iter().zip(&rows[b as usize]) {
                        value += if dot {
                            *av * *bv
                        } else {
                            let delta = *av - *bv;
                            delta * delta
                        };
                    }
                    if dot { -value } else { value }
                };
                let graph = Graph::build(
                    n,
                    16,
                    64,
                    19,
                    build_bound(n, 16).unwrap(),
                    &distance,
                    &|| true,
                )
                .unwrap();
                let mut scratch = Workspace::new(n, 16, workspace_bound(n, 16).unwrap()).unwrap();
                for query in 0..n.min(32) as u32 {
                    for (beam, tails) in [1, 4, 64.min(n), n]
                        .into_iter()
                        .flat_map(|beam| [(beam, false), (beam, true)])
                    {
                        let scalar_trace = RefCell::new(Vec::new());
                        let batch_trace = RefCell::new(Vec::new());
                        let score = |node| distance(query, node);
                        let scalar = |a, b| {
                            scalar_trace.borrow_mut().push((a, b));
                            score(b)
                        };
                        let tail = |a, b| {
                            batch_trace.borrow_mut().push((a, b));
                            score(b)
                        };
                        let four = |a, nodes: &[u32], output: &mut [f64]| {
                            for (&b, value) in nodes.iter().zip(output) {
                                batch_trace.borrow_mut().push((a, b));
                                *value = score(b);
                            }
                        };
                        let mut expected = vec![Hit::default(); beam];
                        let mut actual = expected.clone();
                        let scalar_stats = graph
                            .search(64, &scalar, &|| true, &mut scratch, &mut expected)
                            .unwrap();
                        let batch_stats = graph
                            .search_four(
                                64,
                                &tail,
                                &four,
                                tails,
                                &|| true,
                                &mut scratch,
                                &mut actual,
                            )
                            .unwrap();
                        assert_eq!(scalar_trace.into_inner(), batch_trace.into_inner());
                        assert_eq!(scalar_stats.distances, batch_stats.distances);
                        assert_eq!(scalar_stats.visits, batch_stats.visits);
                        for (a, b) in expected.iter().zip(actual) {
                            assert_eq!(a.node, b.node);
                            assert_eq!(a.distance.to_bits(), b.distance.to_bits());
                        }
                    }
                }
            }
        }
    }

    /// Native cancellation and invalid batched scores return through CXX safely.
    #[test]
    fn quartets_reject_invalid_scores_and_cancel_then_reuse() {
        let n = 374;
        let distance = |a: u32, b: u32| (f64::from(a) - f64::from(b)).powi(2);
        let graph = Graph::build(
            n,
            16,
            64,
            7,
            build_bound(n, 16).unwrap(),
            &distance,
            &|| true,
        )
        .unwrap();
        let mut scratch = Workspace::new(n, 16, workspace_bound(n, 16).unwrap()).unwrap();
        let mut output = [Hit::default(); 4];
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                graph.search_four(
                    64,
                    &distance,
                    &|_, _, output| output.fill(invalid),
                    true,
                    &|| true,
                    &mut scratch,
                    &mut output
                ),
                Err(Error::InvalidArgument)
            ));
        }
        // Safe callbacks may leave output lanes untouched; reject those scores.
        for tails in [false, true] {
            for written in [0, 1] {
                assert!(matches!(
                    graph.search_four(
                        64,
                        &distance,
                        &|a, nodes, output| {
                            for i in 0..written {
                                output[i] = distance(a, nodes[i]);
                            }
                        },
                        tails,
                        &|| true,
                        &mut scratch,
                        &mut output,
                    ),
                    Err(Error::InvalidArgument)
                ));
            }
        }
        let calls = Cell::new(0);
        let four = |a, nodes: &[u32], output: &mut [f64]| {
            calls.set(calls.get() + 1);
            for (&b, value) in nodes.iter().zip(output) {
                *value = distance(a, b);
            }
        };
        assert!(matches!(
            graph.search_four(
                64,
                &distance,
                &four,
                true,
                &|| calls.get() == 0,
                &mut scratch,
                &mut output
            ),
            Err(Error::Cancelled)
        ));
        graph
            .search_four(
                64,
                &distance,
                &four,
                true,
                &|| true,
                &mut scratch,
                &mut output,
            )
            .unwrap();
    }
}
