# Partition-local HNSW

Partition-local HNSW reduces the centroid-selection work within wide internal
partitions while retaining KTANN's tree as the storage and ownership structure.
Each nonempty internal partition has a graph over its own Child Entries. The
partition is the unit of graph loading and maintenance; no global centroid graph
must remain resident.

This document owns the local graph model, candidate contract, source binding and
atomic maintenance design. [Search](search.md) owns traversal and reranking;
[Storage](storage.md) owns logical records and transactions;
[Maintenance](maintenance.md) owns topology transitions;
[Bulk Build](bulk-build.md) owns artifact preparation and publication;
[Runtime and operations](runtime-operations.md) owns admission and cancellation.
Domain terms and system invariants are defined in [CONTEXT](../../CONTEXT.md) and
the [design overview](overview.md).

## Model and configuration

The graph is a derived routing structure. Headers, Child Entries, Partition
Centroids and record membership remain authoritative. A valid graph contains
exactly one node per Child Entry in its owning partition. Edges never cross
partition boundaries. Leaf partitions and empty internal partitions have no
graph records.

Graph maintenance preserves the existing Record–Location–Leaf membership and
topology invariants. Every committed topology state remains searchable, including
legal split and merge states. A graph's validity does not require its partition
to be Ready; it requires the graph and Child Entries to describe the same body.

[`IndexConfig`](../../src/api/config.rs) stores `Option<LocalGraphConfig>`.
`with_local_graph` enables graph construction and maintenance; `local_graph`
returns its parameters. Graph indexes support ordinary creation, Bulk Build and
ordinary writes after activation. Graph configuration is immutable for a Logical
Index ID; changing construction parameters or partition capacities requires a
new index.

| Owner | Configuration |
| --- | --- |
| Index Configuration | `M`, `ef_construction`, graph seed, independent leaf/internal capacities, metric, dimension and rotation seed |
| Persistent format | Canonical ordinals, level and degree bounds, graph encoding and source-binding algorithms |
| Build options | Optional artifact refinement and bounded preparation, scratch, artifact and loading resources |
| Runtime Configuration | Cache capacity, operation admission and routing/maintenance working-byte limits |
| SearchOptions | Explicit `Exact` or `PartitionLocalHnsw { ef_search }`, leaf beam and Search Budgets |

Exact routing is the query default. HNSW selection requires graph configuration;
a leaf-only graph index accepts either routing choice without graph work.
Foreground writes retain the existing exact routing algorithm. Serving
`Index::refine` returns `InvalidArgument` for a graph index; optional artifact
refinement precedes initial graph construction.

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

Ordinary creation starts with no internal graph bodies. Graph-aware maintenance
constructs them when topology growth creates nonempty internal partitions.

Bulk Build imposes a stricter dependency order before publication:

1. Construct the forest with independent leaf and internal capacities.
2. Apply optional artifact refinement and finalize membership and centroids.
   Requested zero-round refinement still refreshes centroids; omitted refinement
   preserves construction output.
3. Accept the final Forest Artifact, derive its Child Entries and build graphs
   over those exact child identities and centroid bytes.
4. Seal one Serving Artifact containing the ordinary records and graph records,
   prove the source/membership/topology joins, and load and validate it under the
   existing Building lifecycle before activation.

No membership, centroid or topology edits follow graph construction within that
artifact. Refinement is preparation work, not a call to serving `Index::refine`.
A resource limit fails preparation rather than silently omitting refinement or
changing graph quality parameters.

Initial graph loading may span transactions because Building isolation hides
partial contents. Completed loading freezes the bytes for resumable exact
backend/artifact comparison; publication is the existing small activation
transaction. Public one-snapshot verification is not a substitute for that proof.

Before artifact acceptance, interrupted graph preparation can be recomputed from
the accepted forest. After Serving Artifact acceptance, recovery loads its bytes
without reconstructing graphs. The loader cannot rewrite serving bytes after
activation; subsequent changes use ordinary writes and atomic maintenance.
Workspace fencing, identity-bound recovery and cleanup follow
[Bulk Build](bulk-build.md).

## Query execution and cache

A query uses one backend snapshot for the Active Manifest, Headers, Child Entries,
graphs and reranking. On a cache miss it reads the complete internal body from that
snapshot, validates the source binding and restores one immutable body containing
both child projections and graph topology. It never loads a newer graph or uses a
workspace artifact to satisfy an older snapshot.

The [shared partition cache](search.md#graph-index-bodies) owns graph bodies under
the existing index/tree/partition/kind key. Reuse requires the snapshot Header
epoch. Older fills cannot replace newer bodies; an equal-epoch graph fill can
upgrade an ordinary internal projection without a later plain fill downgrading
it. Cache hits remain subject to working-set admission. Cache warmth does not
change candidate semantics or guarantee detection of out-of-band corruption
without an epoch change.

For a parent with `n` children at level `L`, let
`B = beam_width(leaf_beam, L - 1)`. Local HNSW returns `min(n, B)` distinct children,
using `effective_ef = min(n, max(ef_search, B))`. It returns the kernel's scores
without recomputation. It does not divide the beam among parents or silently
increase `ef_search`.

All admitted parents at a tree/depth feed the same deterministic global top-B
merge. Local outputs do not create independent per-parent beams. Exact routing
scores all children and can retain local top B before that merge without changing
global exact selection. When `B >= n`, both selectors use the exhaustive local
case; the persisted graph still must be valid.

Graph exploration deduplicates ordinals. This does not replace the traversal's
ownership checks for all Child Entries in an exposed body, including unselected
children. Graph vertices examined within a body do not consume additional logical
partition visits; selected child bodies do when visited. Tree Key fairness,
filtering, leaf scanning, RaBitQ selection and exact reranking remain owned by
[Search](search.md).

HNSW does not guarantee exact top-B or final recall. A valid connected graph must
nevertheless fill its requested local width unless the operation fails. Working
memory exhaustion or cancellation is an error, not a shorter successful candidate
list. Existing Search Budget exhaustion retains its established partial-outcome
semantics.

## Local graph transitions

[`maintenance::graph_topology`](../../src/maintenance/graph_topology.rs) owns the
combined topology plan. [`storage::graph_transitions`](../../src/storage/graph_transitions.rs)
owns graph snapshots, private preparation, source fencing and graph-record
publication. A graph replacement alone cannot establish valid tree membership.

Preparation reads complete source bodies from one snapshot and accepts an exact
proposed Header and canonical Child Entry body. Absent and empty bodies are valid
creation sources. Existing graph bytes are validated even when the target removes
the graph, so deletion cannot hide corruption.

Insertions and centroid replacements evolve a private restored USearch graph,
using proposed centroids for distance calculations. A representation-only
permutation restores canonical child ordinals afterward. A batch containing any
deletion rebuilds that parent's graph with the canonical builder. This avoids
tombstones and a second stable-slot directory, at the cost of whole-parent
construction for a deletion batch. Prepared results never enter the query cache.
Ordinary leaf membership changes do not refresh parent centroids or rebuild graphs.

| Topology change | Graph work in the same publication |
| --- | --- |
| Split exposure | Add the two target edges to the parent's graph; require parent headroom before exposure |
| Internal drain | Move children to one receiver per batch and replace both changed graph bodies |
| Merge receiver reselection | Plan against a fresh Ready receiver and its available headroom |
| Emptying or populating an internal body | Remove all graph records with the last child, or build a graph for the first child |
| Final source removal | Replace the parent's graph with the source edge removed |
| Root growth completion | Build the two-child internal root graph while converting the root in place |
| State-only transition | Preserve graph bytes and body epoch |

Each publication follows one sequence:

1. Prepare all affected graphs outside the write transaction, within a combined
   working-set admission.
2. Protect the Active Manifest and exact source Headers/descriptors in the write
   transaction. Discard a stale plan and replan under the existing retry policy.
3. Apply the topology owner's child/partition changes and validate each resulting
   Header against the prepared target.
4. Admit the combined topology mutations, graph descriptors, obsolete tail
   deletions and protected reads before adding graph payloads.
5. Commit the complete topology and graph change atomically.

Cancellation before commit discards the private work or transaction. An unknown
commit outcome is reported without blindly replaying the plan; the next
maintenance attempt rediscovers committed topology. Backend snapshots retain old
graph bytes for older readers. There is no separately published graph generation,
completion task or asynchronous graph reclamation protocol.

## Resource admission and failures

The atomic design constrains graph size, not just individual KV values. Internal
capacity is at most 16,384, internal minimum occupancy at most 128, and a transition
can temporarily hold two entries above its split threshold. Split exposure waits
for ancestors to make room instead of exceeding this allowance. Together with
the degree and level caps, these bounds limit a source/receiver graph rewrite;
raising them requires revisiting the joint publication bound.

Graph-aware internal drains target one receiver per transaction. All participants
share a 900,000-byte authority envelope for non-payload changes and protected
reads, including obsolete tail deletes. Payloads then remain subject to the
backend's complete transaction admission and native commit limits. In particular,
FoundationDB affected-data limits count conflict ranges as well as mutation
bytes. Fitting each chunk under the value limit is not sufficient. The
[affected-data tests](../../tests/graph_transaction_budget.rs) and
[backend contract](storage.md) define the accounting seam.

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
Operation control is checked within validation, native work and serialization;
limits never reduce `M`, construction ef or requested query width to obtain success.

Malformed caller configuration is `InvalidArgument`; persisted invariant
violations are `Corruption`; resource admission failure is `LimitExceeded`.
Backend-native transaction errors remain distinct and follow the existing retry
and commit-outcome contracts. Finite graph bounds constrain work but do not
promise a latency SLA for every accepted configuration.

## Verification and trade-offs

Graph validity and exact membership are independent obligations. Preparation and
public verification audit graph/source agreement, graph structure and reachability
in addition to ordinary record and topology checks. Bulk publication requires the
stronger completed-load comparison described above. The implementation's
[persistence tests](../../src/search/local_hnsw/persistence_tests.rs),
[snapshot/cache tests](../../src/search/cache/graph/tests.rs),
[publication tests](../../tests/bulk_serving/graphs.rs) and
[shared transition histories](../../tests/support/graph_history.rs) exercise these
seams, including failures and recovery.

Wider internal partitions can reduce upper-tree routing work but increase graph
loading, cache eviction and update units. A parent's centroid payload alone is
`4*n*dimension` bytes, before graph and query scratch. Deletion batches pay the
whole-parent rebuild cost. A global graph could avoid parent-by-parent selection,
but would create a different residency and maintenance unit; it is outside this
design.

Choose operating points from end-to-end recall, latency and resource measurements,
not local graph speed alone. The [benchmark guide](../../benchmarks/README.md)
owns experiment commands, fixed-rerank comparison methods and measurement limits.
The rationale for atomic updates is recorded in
[ADR 0029](../adr/0029-atomic-local-graph-updates.md).
