# KTANN's internal USearch bridge

This unpublished workspace crate adapts USearch's graph-only C++ template.
It owns topology and reusable scratch, accepts synchronous borrowed distance
and operation-control callbacks, and retains no vectors or Rust references.
The Rust core in `src/search/local_hnsw.rs` owns canonical Child Entry validation,
partition seed derivation, routing distances, and operation-wide admission.

The algorithm contract belongs to
[partition-local-hnsw.md](../docs/design/partition-local-hnsw.md#local-graph-algorithm).
Core storage owns graph KV envelopes and snapshot binding; this bridge owns
validated native stream persistence. Production routing remains
outside this crate.

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
It exposes external scratch, topology inspection, safe native loader cleanup, and one bounded
successor-edge operation for connectivity repair. USearch retains ownership
of graph insertion, diversity selection, reverse-link pruning and graph search.
No upstream dense index, metric plugins, SIMD dispatch, or
vector storage is instantiated by this bridge.

Upper-layer greedy traversal can evaluate a node more than once; each move
strictly improves the distance/ordinal pair. Visited state deduplicates
best-first construction and base-layer query exploration. Full-coverage
queries scan and sort in the bridge. Native exact search is not exposed;
ordinary private updates use the adapted native update operation. Pure deletion
compacts surviving node tapes and refreshes neighborhoods incident to removed
nodes. The compaction's ownership-transfer phase cannot throw; before that phase,
cancellation destroys the original tape owners safely. Both update and deletion
propagate metric cancellation through the native update search helper.

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

### Native AddressSanitizer on macOS

Set `CXX` to upstream LLVM Clang. Instrument the C++ bridge and vendored engine,
and link the ASan runtime shipped with the installed stable Rust toolchain:

```sh
KTANN_ASAN_LIB="$(rustc --print sysroot)/lib"
CXXFLAGS='-fsanitize=address -fno-omit-frame-pointer' \
RUSTFLAGS="-C link-arg=$KTANN_ASAN_LIB/librustc-stable_rt.asan.dylib -C link-arg=-Wl,-rpath,$KTANN_ASAN_LIB" \
ASAN_SYMBOLIZER_PATH="$(dirname "$CXX")/llvm-symbolizer" \
ASAN_OPTIONS=detect_leaks=1 \
cargo test -p ktann-usearch \
  --target "$(rustc -vV | sed -n 's/^host: //p')" --target-dir target/asan
```

The explicit target keeps sanitizer flags off Cargo's host build tools. Rust's
stable runtime avoids the LLVM 17 ASan initialization failure on macOS 26;
the compiler's default runtime can fail before tests start. The explicit runtime
and rpath affect only this test build; production linking is unchanged.
This command instruments native code; safe Rust remains normally compiled.
See the [upstream macOS initialization report](https://github.com/llvm/llvm-project/issues/200447).

## Persistence safety boundary

`Graph::encode` delegates to USearch's generic native `save_to_stream`.
`Graph::decode` validates the pinned native layout in safe Rust before calling
`load_from_stream` on the same immutable borrowed bytes. Validation covers
header/configuration, both level copies, ordinal keys, exact lengths,
degree/layer bounds, entry, duplicate/self edges, neighbor layers, and directed
base reachability. Inactive reserved adjacency slots are opaque and preserved;
they are never interpreted as edges. This is a version-pinned, native-endian
u64-key/u32-slot format, not a portable canonical topology encoding.

The loader patch checks node allocation, registers only fully initialized nodes,
and frees a partially read node using its allocated size. Both stream callbacks
copy in at most 64 KiB steps and return false on cancellation; they never throw
through upstream `noexcept` functions. The bridge reports cancellation after
native frames return. Partial restoration is never published. Loading calls no
metric and performs no construction or repair.

The [safe stream validator](src/persistence.rs) defines the supported native
grammar; the core [graph codec](../src/storage/values/graph.rs) defines KV envelopes
and the byte-order marker. The [local HNSW design](../docs/design/partition-local-hnsw.md#persistence-and-snapshot-binding)
owns source binding and loading constraints. Decode admission covers borrowed
encoded bytes, validation scratch, and native allocations; callers additionally charge their own retained
inputs. Native streams include full reserved adjacency capacity, accounted in
encoded-size bounds. Updating USearch requires rechecking the layout and the
validator together. Golden native fixtures, malformed inputs, allocation-failure
sweeps, partial-node cancellation and byte-preserving round trips cover this seam.

## Private ordinary updates

`Graph::insert` consumes a privately restored graph. It appends nodes through
USearch, then remaps ordinal keys and neighbor
slots to the canonical child order. The source persisted graph and shared search
handles remain immutable. `Graph::retain` removes a sorted subset of old slots
and repairs incident surviving neighborhoods. Existing centroids remain unchanged;
each update inserts or removes nodes. Both paths produce a complete validated replacement.

The pinned patch additionally releases and reinitializes thread scratch limits,
and provides a checked slot permutation without changing neighbor selection.
The bridge reapplies the existing directed-reachability repair, releases build
scratch, and serializes only after successful cancellation and memory checks.
Update tests cover permutation validation, source immutability, and cancellation;
the owning crate covers metric scores and atomic snapshot-consistent publication.

Graph levels are capped at `floor(log_M(65536))` after the original geometric
sampler finishes, preserving its RNG word consumption for subsequent nodes.
The safe parser and encoded-size bound enforce the same integer cap. The owning
KTANN configuration limits M to 16 and internal capacity to 16,384 plus two
transient slots; the native crate retains its broader standalone input limits.
