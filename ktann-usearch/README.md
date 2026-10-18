# KTANN's internal USearch bridge

This unpublished workspace crate adapts USearch's graph-only C++ template.
It owns topology and reusable scratch, accepts synchronous borrowed distance
and operation-control callbacks, and retains no vectors or Rust references.
The Rust core in `src/search/local_hnsw.rs` owns canonical Child Entry validation,
partition seed derivation, routing distances, and operation-wide admission.

The algorithm contract belongs to
[partition-local-hnsw.md](../docs/design/partition-local-hnsw.md#41-local-core-algorithm-and-usearch-integration).
There is no graph KV format or production routing switch here.

## Dependency provenance and maintenance

- Upstream: [USearch 2.26.4](https://crates.io/crates/usearch/2.26.4), Apache-2.0.
- Source: `include/usearch/index.hpp` from the published `usearch-2.26.4.crate`.
- Original header SHA-256:
  `1afba43e687bcf91109e42c64b1056042a88051144145d95a6f89fe574d52b1f`.
- The original license is [vendor/LICENSE](vendor/LICENSE).
- The complete local delta is [vendor/ktann.patch](vendor/ktann.patch), applied
  to [vendor/usearch/index.hpp](vendor/usearch/index.hpp).

The patch changes level sampling, deterministic tie ordering, and cancellation
propagation along construction and approximate-search paths.
It exposes external scratch, topology inspection, and one bounded
successor-edge operation for connectivity repair. USearch retains ownership
of graph insertion, diversity selection, reverse-link pruning and graph search.
No upstream dense index, metric plugins, serialization, SIMD dispatch, or
vector storage is instantiated by this bridge.

Upper-layer greedy traversal can evaluate a node more than once; each move
strictly improves the distance/ordinal pair. Visited state deduplicates
best-first construction and base-layer query exploration. Full-coverage
queries scan and sort in the bridge; native exact search and dynamic update
are not exposed or adapted for KTANN callbacks.

Update the patched header and patch together. Verify the source hash when
rebasing, regenerate the delta against the pinned original, and rerun the graph
tests in both debug and release: native exception handling must work in release.
The native translation unit enables USearch assertions even if the build sets
`NDEBUG`; metric callback failures and cancellation unwind C++ frames, never
Rust frames. Do not enable fast-math for this integration.

## Allocation and concurrency boundary

The native allocator charges each payload, alignment/header and a 64-byte
allocator-overhead allowance to its graph or workspace. It returns null instead
of throwing inside USearch's allocation helpers. Admission bounds also include
wrapper metadata and construction's directed-reachability queue. They are
conservative allocation bounds, not OS/RSS bounds. Temporary exception objects,
Rust callback state and the caller's other allocations are outside the native
allocator; the core's operation admission includes fixed overhead and inputs.

Graph construction is serial. Once built, the graph releases construction
scratch and is immutable. Every concurrent query supplies a distinct mutable
workspace, allocated before query work. Successful graph searches do not
allocate or grow native containers; neighbor-pruning scratch belongs only to
construction, and query output is supplied by Rust. Graphs and
workspaces can be moved to another thread. Allocation headers preserve the
original account through destruction, independent of thread-local allocation
scope. The workspace is deliberately not `Sync`.

The CXX-generated boundary and the three documented `Send`/`Sync` implementations
are the only Rust unsafe boundary. The main KTANN crate retains `unsafe_code =
forbid`. Public safe bridge methods validate node/degree/count bounds before
native access. They accept only finite callback scores and keep callbacks alive
for each synchronous call. A callback must not panic; a panic crossing CXX's
Rust callback boundary aborts, as CXX documents.

## Checks

Run Cargo commands sequentially:

```sh
cargo test -p ktann-usearch
cargo test -p ktann --lib search::local_hnsw
cargo test --release -p ktann --lib search::local_hnsw
cargo clippy -p ktann -p ktann-usearch --all-targets -- -D warnings
```

The core tests include independent-process topology fingerprints, graph
invariants/reachability, all metrics against an independent oracle, duplicate
centroids/full beam, input and memory boundaries, cancellation/scratch reuse,
concurrent readers, and small independent-query quality fixtures. These fixtures
do not establish end-to-end recall, latency, or production parameter defaults.
