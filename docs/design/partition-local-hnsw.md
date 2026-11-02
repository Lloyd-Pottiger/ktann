# Partition-local HNSW

Partition-local HNSW reduces the centroid-selection work within wide internal
partitions while retaining KTANN's tree as the storage and ownership structure.
Each nonempty internal partition has a graph over its own Child Entries. The
partition is the unit of graph loading and maintenance; no global centroid graph
must remain resident.

This document owns the graph algorithm, source binding, native update policy and
combined resource bounds. Traversal and cache behavior belong to
[Search](search.md), atomic publication to [Storage](storage.md), topology
execution to [Maintenance](maintenance.md), and lifecycle operations to
[Bulk Build](bulk-build.md) and [Runtime and operations](runtime-operations.md).

## Model and configuration

The graph is a derived routing structure. Headers, Child Entries, Partition
Centroids and record membership remain authoritative. A valid graph contains
exactly one node per Child Entry in its owning partition. Edges never cross
partition boundaries. Leaf partitions and empty internal partitions have no
graph records.

A graph's validity depends on agreement with its Child Entries, including legal
non-Ready topology states. [API configuration](api.md#4-immutable-and-runtime-configuration)
owns immutable graph parameters and capacities; [search](search.md#6-search-algorithm)
owns per-query routing selection. Graph indexes support ordinary creation, Bulk
Build and writes. Refinement runs only during artifact preparation, before graphs
are constructed.

### Foreground write routing

Foreground writes use exact scoring for internal bodies with fewer than 1024
nodes and HNSW for bodies with at least 1024 nodes. The snapshot Header count
selects the path independently for each body. Both selectors retain the configured
write beam and feed the same deterministic global candidate merge; HNSW uses
`max(64, beam)` search effort bounded by the body size. Indexes without local graphs
retain exact write routing. Public query routing and budgets are independent.

Cohere lifecycle measurements found exact scoring cheaper for small bodies, while
mature topology benefited from graph traversal. The 1024-node threshold is a
conservative policy above the measured crossover, not a universal optimum or
persisted configuration. [PR #226](https://github.com/Lloyd-Pottiger/ktann/pull/226)
records the measurement scope and limits. Approximate routing can choose a
different leaf, so policy changes require fixed-budget quality and write-lifecycle
measurements covering maintenance, cache and storage costs.

## Local graph algorithm

The graph core uses USearch's generic C++ graph through the private
[`ktann-usearch`](../../ktann-usearch/README.md) bridge. It borrows KTANN's
already-preprocessed centroids and calls the existing Rust `VectorKernel` for
f64 routing scores. This preserves KTANN's numeric range, metric semantics and
cross-parent score comparability without storing another vector copy or
recomputing returned candidate distances. The upstream dense interface's f32
scores and normalization behavior do not satisfy that contract.

The algorithm has the following constraints:

- Child Entries are in strictly increasing Partition Key order. Dense ordinals
  follow that order; distance ties use Partition Key order.
- `2 <= M <= 16` and `ef_construction >= M`. Base-layer degree is at most `2*M`,
  upper-layer degree at most `M`, and neither exceeds the available neighbors.
- Serial construction and a seed derived from the configured seed, Tree Key and
  owning Partition Key produce deterministic output for the same inputs on the
  same build and host. Level sampling uses a fixed integer random stream and
  caps levels at `floor(log_M(65536))`. The cap also bounds serialization and
  restoration allocations.
- USearch owns insertion, neighborhood selection, pruning and graph search.
  KTANN checks directed base-layer reachability from the entry point. If needed,
  a deterministic successor-cycle repair replaces at most one outgoing edge per
  node without exceeding the degree bound. Repair happens during preparation,
  never during query loading.
- The native graph retains topology only. Query scratch is separate and bounded;
  concurrent queries use separate workspaces. No borrowed Rust reference or
  distance callback survives a native call.

The [core implementation](../../src/search/local_hnsw.rs) owns input validation,
metric integration and seed derivation. The native bridge's README owns dependency
provenance and patch maintenance; its
[stream validator](../../ktann-usearch/src/persistence.rs) defines the supported
native layout. Native bytes and floating-point reconstruction are not promised
to be portable across builds or host byte orders.

## Persistence and snapshot binding

Graph Descriptor and Graph Chunk records live under the owning
`(Logical Index ID, Tree Key, Partition Key)` prefix. The descriptor records the
encoding, node count, source epoch, payload size, source binding and payload
digest. Chunks contain native topology with local ordinals; they contain no
independent centroid projections. The query's Child Entries supply the vectors.

The source binding covers the owning index/tree/partition, graph construction
parameters, metric and dimension, canonical child identities and centroid bytes,
and the Header's level, count and cache epoch. The Header epoch is the sole
body-version authority. A body change advances it and publishes the corresponding
graph in the same transaction. State-only transitions preserve the epoch and graph
bytes when the body is unchanged.

The [graph value codec](../../src/storage/values/graph.rs) defines KV envelopes and
canonical chunk sizes. The
[persistence integration](../../src/search/local_hnsw/persistence.rs) defines
source hashing, graph validation and encoding. Each complete graph value is at
most 100,000 bytes. Chunk ordinals are contiguous; complete-range reads detect
extra chunks as well as missing ones. Reserved native adjacency capacity counts
toward encoded size even when its slots are unused.

Before native restoration, validate sizes and byte order, canonical node identity,
levels, entry point, degree limits, edge targets, duplicate/self edges and directed
reachability. Descriptor counts, source epoch/binding and payload digest must
match. A checksum is not a substitute for structural validation. Loading performs
no construction, repair or distance computation.

Missing mandatory graph records or malformed/mismatched persisted data are
`Corruption`. Selecting Exact routing does not bypass graph-body validation.
Empty internal bodies must have neither a descriptor nor chunks; this is a valid
maintenance state, not a missing-graph fallback.

## Initial construction

[Offline artifact preparation](bulk-artifacts.md#1-pipeline) finalizes membership
and centroids before constructing graphs from the exact Child Entry projections.
No membership or centroid edits follow graph construction within an artifact.
[Bulk Build](bulk-build.md#5-loading-and-exact-validation) loads and validates all
serving bytes before activation; accepted Serving Artifacts support recovery
without reconstructing graphs.

## Query execution and cache

For a parent with `n` children at level `L`, let
`B = beam_width(leaf_beam, L - 1)`. Local HNSW returns `min(n, B)` distinct children,
using `effective_ef = min(n, max(ef_search, B))`. It returns the kernel's scores
without recomputation. When `B >= n`, selection is exhaustive; the persisted graph
still must be valid. Graph exploration deduplicates ordinals without consuming
logical partition visits; selected child bodies consume visits when traversed.

[Search traversal](search.md#61-snapshot-and-traversal-order) owns the shared
snapshot, global candidate merge, child ownership checks and budget charging.
[Graph index bodies](search.md#graph-index-bodies) owns graph loading, descriptor
validation on cache hits, epoch-based reuse and working-set admission. Cache
warmth does not guarantee detection of out-of-band corruption without an epoch
change.

## Local graph transitions

[`storage::graph_transitions`](../../src/storage/graph_transitions.rs) prepares
replacements from complete source bodies and exact proposed Headers and Child
Entries. Absent and empty bodies are valid creation sources. Existing graph bytes
are validated even when the target removes the graph, so deletion cannot hide
corruption.

Topology maintenance changes child sets without changing surviving centroids:

- Insertion preserves every old child and centroid, inserts new native nodes,
  then restores canonical child ordinals.
- Pure deletion requires a strict subset with unchanged surviving centroids. It
  physically removes native nodes, compacts slots, refreshes surviving
  neighborhoods incident to removed nodes, and audits directed reachability
  before and after neighborhood repair. Surviving levels are retained and a
  highest-level entry is selected.
- An unchanged body preserves topology; an empty result removes graph records.
- Creation and root level conversion build a complete graph for the resulting
  internal body.

Mixed child replacement and surviving centroid changes are unsupported.
Ordinary leaf membership changes do not refresh parent centroids or rebuild
parent graphs. Native nodes have no persistent tombstones or slot directory.

[Graph-aware maintenance](maintenance.md#7-graph-aware-topology) owns participant
discovery, split/merge ordering, body ownership and stale-plan rediscovery.
[Storage operations](storage.md#7-typed-atomic-operations) owns source fencing,
target-Header validation and atomic graph/topology publication.
[ADR 0029](../adr/0029-atomic-local-graph-updates.md) records why graph replacements
share that transaction.

## Resource admission and failures

The atomic design constrains graph size, not just individual KV values. Internal
capacity is at most 16,384, internal minimum occupancy at most 128, and a transition
can temporarily hold two entries above its split threshold. Split exposure waits
for ancestors to make room instead of exceeding this allowance. Together with
the degree and level caps, these bounds limit a source/receiver graph rewrite;
raising them requires revisiting the joint publication bound.

Graph-aware internal drains target one receiver per transaction. Storage batch
publication checks all participants against one 900,000-byte authority envelope
for topology and graph non-payload changes and protected reads, including obsolete
tail deletes, before applying graph payloads. Payloads remain subject to the
backend's complete transaction admission and native commit limits. FoundationDB
counts conflict ranges as well as mutation bytes; fitting each chunk under the
value limit is insufficient.

The [native encoding bound](../../ktann-usearch/src/persistence.rs) is
`40 + n * (12 + 4 * (h + 1) + 4 * (2*M + h*M))` bytes, where
`h = floor(log_M(65536))`. Across the supported `2 <= M <= 16`, the maximum is
`40 + 416*n` at `M=16`, `h=4`. A split drain keeps the source/receiver combined
count at most 16,386. A merge drain can combine a source below minimum occupancy
(at most 127 nodes) and a receiver up to 16,386 nodes. The latter bounds both
replacement streams by `2*40 + (127 + 16386)*416 = 6,869,488` native payload bytes.
At 99,995 payload bytes per complete 100,000-byte chunk value this uses at most
70 chunks, including both streams. Graph metadata and obsolete tail deletes
remain inside the shared authority envelope.

The [backend transaction contract](storage.md#2-transaction-sizing-and-retry) owns
affected-data accounting. These bounds admit supported transitions; they do not
make an arbitrary topology plan or wider keyspace fit.

Construction, restoration and update preparation admit their full working set:
source and proposed centroids, kernel, native graph and scratch, validation,
encoding buffers, and retained results for other affected partitions. Search
charges graph bodies, decode buffers, query workspace, result/merge buffers and
query-wide ownership tracking. An evicted but still referenced body remains part
of the query's working set. Cache capacity alone therefore does not bound routing
memory, and these charged-byte bounds are not process RSS bounds.

[`GraphMaintenance`](../../src/runtime/graph_maintenance.rs) bounds native
preparations and holds admission until native CPU work actually exits, even if its
async waiter is dropped. Runtime workers also bound waiting plans. Low-level
callers must reuse an admission owner and bound their concurrent drivers.
The [maintenance ownership protocol](maintenance.md#7-graph-aware-topology)
serializes only overlapping graph bodies; independent bodies remain concurrent.
Split draining also owns the sibling whose Header guides balancing. The lock
registry retains weak references, so it does not grow with historical partition
count. Cancellation retains body ownership until detached native work exits;
storage source fencing remains the write authority across runtimes.

Operation control is checked within validation, native work and serialization;
limits never reduce `M`, construction ef or requested query width to obtain success.

[Operation control and errors](api.md#6-operation-control-and-errors) owns error
classification. Finite graph bounds constrain work but do not promise a latency
SLA for every accepted configuration.

## Verification and trade-offs

[Public verification](runtime-operations.md#6-verification) audits graph/source
agreement independently of the search cache and ordinary membership/topology
checks. [Bulk publication](bulk-build.md#5-loading-and-exact-validation) requires
the completed-load byte comparison rather than public verification alone.

Wider internal partitions can reduce upper-tree routing work but increase graph
loading, cache eviction and update units. A parent's centroid payload alone is
`4*n*dimension` bytes, before graph and query scratch. A global graph could avoid
parent-by-parent selection, but would create a
different residency and maintenance unit; it is outside this
design.

Choose operating points from end-to-end recall, latency and resource measurements,
not local graph speed alone. The [benchmark guide](../../benchmarks/README.md)
owns experiment commands, fixed-rerank comparison methods and measurement limits.
