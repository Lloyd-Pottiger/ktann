# Partition-local HNSW: serving and maintenance contract

Status: Bulk Build preparation and publication are implemented, including optional
artifact refinement, final-centroid graph construction and fenced immutable
loading. Graph-enabled indexes support ordinary creation and writes after
publication. Snapshot-bound graph reads, shared cache ownership and production
HNSW traversal use explicit public routing selection and bounded per-tree/level
candidate merging. Offline verification audits graph bodies and centroid
projections. The shared adapter corpus covers durable loading, reopening,
publication and both routing modes for all three metrics. This document owns the
local graph contract of [issue #195](https://github.com/Lloyd-Pottiger/ktann/issues/195).
[Search](search.md), [Storage](storage.md), and [Bulk Build](bulk-build.md) retain
ownership of their shared contracts.

Graph maintenance connects graph publication to split, drain, merge, root
growth, and parent membership changes. Fault-injected Memory histories and
durable RocksDB/FoundationDB histories pass. Bounded update experiments cover SIFT at 16,384 records
for 32 rounds and 65,536 records for eight rounds, plus Cohere at 16,384 records
for eight rounds, with fixed rerank budgets and independent evaluation queries.
These establish the tested update behavior, not a general HNSW speedup, tuning
default, or large-scale SLA. The reproducible runner and measurement limits are
documented in [the benchmark guide](../../benchmarks/README.md#local-graph-update-history).

## 1. Scope and invariants

A graph-enabled Logical Index has one persisted HNSW graph per nonempty internal
partition, over exactly that partition's Child Entries. The graph selects routing
candidates; it does not own partitions, records or membership. The tree remains
the storage and ownership structure. Leaf and empty internal partitions have no
graph. Writes maintain graphs atomically with their authoritative projections.

Preserve exact Record–Location–Leaf membership, one incoming Child Entry per
non-root partition, existing Tree Key selection, per-tree level-scaled global
beam competition, leaf scans, exact filtering, RaBitQ selection, and exact
reranking. Routing distances retain the current metric and tie semantics. A
query uses one backend snapshot through traversal and reranking.

The contract includes empty input, leaf-only trees, multiple Tree Keys, multiple
internal levels, all three metrics, filters, cancellation, reopen and bounded
resource failure. Graph configuration is opt-in and immutable. There is no
in-place conversion of an Active index: changing construction parameters or
capacities requires a new Logical Index ID. Membership changes use ordinary
writes and maintenance; artifact refinement is available before initial graph
construction. Online serving refinement is unsupported for graph indexes.

Automatic query-mode selection, adaptive expansion and production tuning defaults
are outside this contract. Measured warm local graph speed alone does not prove
an end-to-end benefit.

## 2. Evidence and changes required

Current implementation boundaries:

| Existing owner | Verified behavior and implication |
| --- | --- |
| [`IndexConfig`](../../src/api/config.rs) | Independent leaf/internal min/max pairs are implemented; leaf maximum is capped at 65,536; internal maximum at 16,384 and internal minimum at 128. `with_local_graph` enables graph maintenance; `LocalGraphConfig` holds persisted construction parameters. |
| [`construct_tree`, `ConstructionOptions`](../../src/construction.rs) | Bottom-up construction selects capacities by level, including root-fit. Memory admission reserves sample, centroid and row buffers; caller-retained output is excluded. |
| [`local_hnsw`](../../src/search/local_hnsw.rs), [`ktann-usearch`](../../ktann-usearch/README.md) | Standalone graph construction, search, exact local top-k, structural inspection, and reusable bounded scratch are implemented. Bulk Build constructs and persists graphs; production queries explicitly select exact or graph routing. |
| [`ForestArtifact`](../../src/bulk/forest.rs), [`ServingArtifact`](../../src/bulk/serving.rs) | Forest plans bind source/configuration/rotation/sample size. Serving preparation proves exact joins and emits canonical values; optional artifact refinement completes before final forest acceptance; serving encoding builds graphs from final Child Entries. |
| [`refine::run`, `plan`](../../src/runtime/refine.rs) | Existing Offline Refinement reads an Active index, plans in memory, commits member moves in batches, then replaces non-root centroids. Even zero rounds recomputes centroids. Failure can leave a valid partially refined index. |
| [`update_centroid`](../../src/storage/topology.rs) | Atomically replaces the partition centroid and incoming Child Entry, advancing the parent's body cache epoch. It does not update a graph. |
| [`Traversal::visit_internal`, `select_next_frontier`](../../src/search/traverse.rs) | Exact and graph candidates feed the same bounded per-tree/level top-B selection. Every exposed Child Entry retains its duplicate-reference check. |
| [`BodyRead::finish`, `PartitionCache`](../../src/search/cache.rs) | Cold search loads the complete decoded body before cache insertion. Cache capacity bounds retained entries, not the complete temporary body or evicted bodies still held by queries. |
| [`bulk_load`](../../src/runtime/bulk_load.rs), [`bulk_publish`](../../src/runtime/bulk_publish.rs) | One accepted Serving Artifact, epoch-fenced chunk loading, immutable completed load, paged exact byte validation, then atomic activation. No graph-ready lifecycle state is needed. |

The relevant prior decisions are [exact internal membership](../adr/0007-exact-internal-membership-without-parent-pointers.md),
[snapshot cache validation](../adr/0010-snapshot-validated-partition-cache.md),
[committed routing cache reuse](../adr/0023-committed-internal-routing-cache.md),
[bounded search](../adr/0011-bounded-deterministic-approximate-search.md),
[Building reservations](../adr/0026-building-reservations.md), and
[workspace/publication ownership](../adr/0027-bulk-workspace-and-publication.md).
This stage extends those mechanisms; it does not replace their authority.

## 3. Allowed operations and enforcement

The immutable Index Configuration stores `Option<LocalGraphConfig>` in the
Index Manifest. It controls graph construction and maintenance independently of
query routing. Both ordinary creation and Bulk Build support graph-enabled
indexes; publication makes a built index available for ordinary writes.

| Operation | Building index | Active graph index |
| --- | --- | --- |
| Ordinary open, record/payload reads, search, public verify | `IndexBuilding` | Allowed; search chooses Exact or local HNSW explicitly |
| Insert, upsert, delete, batch mutation | `IndexBuilding` | Allowed with graph-aware maintenance |
| Public `refine` | `IndexBuilding` | `InvalidArgument`; use artifact refinement before graph construction |
| Build preparation, load, validation, publish, recovery | Existing job identity and fences | Identity-bound status/recovery; no loader rewrite of serving bytes |
| Automatic Structure Maintenance | Never admitted | Uses graph-aware preparation and atomic publication |
| Abort | Existing unpublished-identity rules | Rejected by existing abort rules |
| Drop | Existing lifecycle rules | Allowed; uses Dropping cleanup |

Building, Dropping and absent identities retain their existing lifecycle errors.
Normal request syntax validation may reject malformed input first. Core write
admission and fenced Building loader authority preserve lifecycle isolation;
private artifact computation has no backend mutation authority. Raw backend
writes remain outside the library API's integrity guarantee.

## 4. Configuration ownership

The table defines canonical ownership for persisted construction, runtime
resources, and per-query routing.
Keep one canonical owner for each value and include semantic values in artifact
bindings; copied bindings are validation evidence, not independently editable
configuration.

| Owner | Values and rules |
| --- | --- |
| Index Configuration / Manifest | Graph serving mode, leaf and internal min/max capacities, graph `M`, `ef_construction`, graph seed; existing dimension, metric and rotation seed. Immutable and compared by idempotent creation/open. |
| Persistent Format | Graph algorithm/version, canonical node order, level sampling and maximum level, neighbor-selection/tie rules, degree limits (base at most `2*M`, upper at most `M`), graph encoding and chunk layout. |
| Build Descriptor | Source identity, existing construction sampling, optional refinement rounds and its deterministic algorithm version. These affect the accepted forest/serving identities and cannot change on resume. |
| Build Workspace / worker options | Construction/refinement/graph working-memory and scratch ceilings, artifact limits, load chunk limits. Reuse the persisted worker-options equality and ownership checks. Resource budgets must not alter successful output bytes. |
| Runtime Configuration | Cache bytes, foreground admission, and a positive routing/maintenance working-byte ceiling. These bound residency and execution, not graph identity or persisted construction quality. |
| SearchOptions | Explicit `Exact` or `PartitionLocalHnsw`, plus `ef_search` for HNSW. Existing leaf beam and Search Budgets remain query controls. |

Do not copy mutable `ef_search` into graph metadata or persist a second mode in
the Build Workspace. This stage keeps exact routing as the API default and
requires explicit HNSW selection, avoiding a premature production default.
HNSW on an ordinary index without graphs is `InvalidArgument`; an irrelevant
HNSW-only parameter on exact routing is also rejected. Leaf-only graph indexes
accept either routing choice and perform no graph work.

Validate `M >= 2`, `ef_construction >= M`, positive `ef_search`, checked degree,
size and index arithmetic, and finite metric-correct centroid inputs. The codec
must have explicit finite level and allocation bounds; do not allocate from an
untrusted serialized count. Query exploration uses
`effective_ef = min(n, max(ef_search, B))`, where `n` is the parent's child count
and `B` the target next-level beam. Clip to `n` before allocating. No calibration
or automatic retry increases `ef_search` in this stage. Maximum node count comes
from the configured internal capacity, within the checked index format cap.

Use the existing `VectorKernel` routing scores for graph construction and
search: squared L2, negated dot product for cosine and inner product, with
Partition Key ties. The core borrows already-preprocessed vectors and does not
normalize them. Graph construction order and derived per-partition
seed are deterministic from the configured seed, canonical Tree Key and
Partition Key; rebuilding the same accepted inputs must produce identical bytes.
This reproducibility is tested across fresh processes on the same build/host;
neither native encoding nor floating-point reconstruction promises cross-platform
portability.
Construction may parallelize only if it preserves this property and its declared
resource bounds. The first implementation constructs one graph at a time.

### 4.1 Local core algorithm and USearch integration

The implementation reuses USearch 2.26.4's C++ `index_gt<double>`, through the
private `ktann-usearch` crate. The official Rust dense interface returns f32
distances, copies vectors, and does not expose seed or level-cap controls.
Finite accepted KTANN inputs near `1e20` can overflow its scores; subtracting
one from its `1-dot` representation does not fix overflow. Its cosine metric
also renormalizes inputs and treats two zero vectors differently from KTANN
routing. The chosen generic graph calls the existing Rust kernel directly and
retains the returned f64 scores. This preserves the supported input range and
cross-parent score comparability without candidate distance recomputation or
an additional f32-to-f64 reranking stage. The bridge does not enable fast-math.

The algorithm contract is:

- Require canonical strictly increasing Child Entry Partition Keys; reject
  duplicate or noncanonical input. Dense u32 ordinals follow that order. Build
  serially in ordinal order and retain no centroid copies in the native graph.
- Require `2 <= M <= 16` and `ef_construction >= M`. Layer zero has at most
  `2*M` outgoing edges; upper layers have at most `M`. Actual degree also cannot
  exceed the number of other nodes. Clip construction exploration to the node
  count, preserving the supplied quality parameters as identity inputs.
- Derive the per-partition u64 seed from the first eight bytes, little-endian,
  of SHA-256 over `b"ktann/local-hnsw/v1\0"`, the 32-byte configured seed,
  little-endian u64 canonical Tree Key byte length, those bytes, and the
  little-endian u64 parent Partition Key.
- Sample levels from a fixed SplitMix64 stream. For each node, start at zero;
  promote while the next word is below `floor(2^64/M)`, stopping at level 16.
  No further word is consumed at level 16. Cap the sampled result at
  `floor(log_M(65536))` without changing stream consumption. Integer arithmetic
  avoids platform-dependent random distributions and logarithms.
- Preserve USearch's best-first construction and diversity neighbor selection:
  consider candidates by distance/key; a candidate is discarded when an
  already-selected neighbor is strictly closer to it than the insertion
  distance. Reverse lists use the same bounded pruning. Do not force metric
  properties that inner product lacks. Candidate heaps, greedy improvements,
  and equal-distance admission use ordinal ties, equivalent to Partition Key
  ties; signed zeros compare equal.
- Audit directed layer-zero reachability from the entry point. Only if it is
  incomplete, add the canonical cycle `i -> (i+1) mod n`, preserving an existing
  successor edge and replacing the last edge when a list is full. This changes
  at most one outgoing edge per node, respects `2*M`, and ensures reachability.
  Native pruning alone failed this requirement on a repeated-centroid fixture.
  Report whether repair was needed; it is part of construction, never query
  fallback or adaptive expansion. Empty/singleton graphs need no cycle.
- Search uses the full requested local beam and clipped effective ef. Native
  query scratch is separate from immutable topology. It is reserved once and
  reusable across parents and queries; successful searches do not grow its
  allocations. Candidate output is O(min(n, B)). The exhaustive `B >= n` case
  scores each node once and sorts the output in O(n log n); exact local top-k
  is also independently callable without constructing a graph.

`Centroids` validates dimensions and finite values through the numeric kernel,
without computing a distance, and checks capacity and canonical keys once at
the borrowed-view boundary. Bulk construction consumes that view synchronously
and returns only encoded graph records; callers do not retain a borrowed graph
or sequence separate build and persistence operations. Restoration returns a
complete graph internal body that owns its projections, numeric kernel and
validated graph. `Graph` itself owns only topology and exposes entry, level and
bounded adjacency inspection; no Rust references or callbacks survive an FFI
call. Immutable graphs support concurrent queries with separate workspaces.

Before construction, admit a checked upper bound including borrowed centroids,
all nodes at the maximum level, adjacency, locks and construction scratch.
Native allocations enforce a separate charged-byte ceiling, including alignment
and an allocator-overhead allowance. The graph reports resident and peak charged
bytes; these are not process RSS measurements. Query admission includes the
borrowed body, retained graph, the complete reusable workspace, query vector and
bounded result buffers: one output buffer for graph search, and a heap plus
output for the independent exact selector. Neighbor-pruning scratch is allocated
only for construction. Workspace counters distinguish retained bytes from total
allocation events. Allocation failure returns `LimitExceeded`; it never changes
M or ef. Encoding and restoration include their buffers in operation admission;
initial publication IO remains owned by Bulk Build. Ordinary update publication
uses the atomic replacement protocol below.

Operation control is checked during input validation, between inserted nodes,
at most every 64 distance calls, and during reachability and result loops.
Cancellation unwinds native frames before returning the original KTANN error;
no partial candidate list is published. Visited state prevents repeated scoring
in best-first construction and base-layer query exploration. Upper-layer greedy
traversal can rescore a node, but each move strictly improves the distance/key
pair. Finite node, layer and degree limits bound work;
allocation admission does not promise a latency bound for extreme parameters.

## 5. Preparation, refinement, loading and publication

Use the existing lifecycle, with the following ordered preparation pipeline:

1. Reserve the Building Manifest and Build Descriptor. Register the token-owned
   workspace before producing files, with existing worker/scheduler fencing.
2. Validate source schema and global Record ID uniqueness; construct the forest
   with independent leaf/internal capacities and canonical Partition Keys.
3. Apply optional bounded artifact refinement before accepting the final Forest
   Artifact. Use the existing metric-correct assignment algorithm and leaf
   occupancy rules, then recompute final leaf centroids and internal centroids
   bottom-up. Preserve the existing unweighted child-centroid mean convention.
   Zero rounds still refreshes centroids if refinement is requested; omitted
   refinement preserves construction output. Refined assignments must still
   account for every source record exactly once in its original Tree Key.
4. Seal and accept the final Forest Artifact, binding refinement semantics.
   Produce final Child Entries and partition centroids from that same forest.
   Build each internal graph over those exact Child Entry centroid bytes and
   canonical IDs. No membership, centroid or topology edit follows this step.
5. Prove exact source/assignment/topology joins and graph invariants. Encode
   graph records alongside existing serving records into one sorted, unique-key
   Serving Artifact bound to the target Manifest and final Forest Artifact.
   Seal and accept it through the existing workspace mechanism.
6. Load that immutable stream through existing Build Load chunks. Child Entries,
   centroids and graph chunks need not fit one transaction: Building keeps every
   partial prefix invisible to ordinary operations.
7. Freeze at completed Build Load; perform resumable exact backend/artifact byte
   comparison, including every graph key. Only the completed proof and immutable
   load authorize the existing final Building-to-Active transaction.
8. Reopen/search the Active index and reclaim workspace files using existing
   cleanup. No post-publication refinement or graph completion task runs.

Artifact refinement is new pure preparation work, not a call to the current
`Index::refine`. Its numerical planning may be extracted for reuse, but its
persistence path must not reuse Active-index member moves. It is bounded: if a
tree cannot fit the configured refinement working set, preparation returns
`LimitExceeded`; it never silently skips refinement. Source, output and sort
buffers remain separately accounted. An external-memory refinement algorithm is
not required for this first contract.

The final Forest Artifact is the existing accepted forest slot. Unrefined plans
and unfinished graph files are attempt-owned scratch, not a second accepted
forest or a new durable phase authority. A crash before acceptance recomputes
that stage; a crash after forest acceptance can rebuild graphs from it. After
Serving Artifact acceptance, recovery loads its bytes without source/refinement
or graph reconstruction. Reopening a completed load never grants new writes.

Do not publish a graphless Active index and refine/rebuild it later. Do not
mutate loaded Building bytes after completing the load: the existing exact
validation proof relies on their immutability. A different refinement request
or seed conflicts with the original request; abort and start a new identity.
Existing terminal workspace failures stay terminal until abort; transient
failures and cancellation retain existing resumable semantics. Unknown load or
publication outcomes resolve against the original identity and durable
checkpoints, never a reused name. Cleanup failure after activation does not
make the index Building again.

## 6. Persisted graph and snapshot binding

Store graph records under the owning `(Logical Index ID, Tree Key, Partition
Key)` prefix, next to the internal partition's ordinary records. Graph node
ordinals follow sorted Child Entry Partition Keys. Adjacency contains local
ordinals only: no cross-partition edges, record IDs, reverse parent pointers,
or independent centroid copies. The query's decoded Child Entries supply the
vectors. This preserves one authoritative projection per parent/child edge.

The core stores the pinned USearch 2.26.4 generic native stream, with u64 keys
and u32 slots. Safe Rust validates the entire immutable byte slice before the
native loader reads it. The loader has a small allocation/cleanup patch: reject
failed node allocations, count only initialized nodes, and free a partially
read node using its known allocation size. It performs no distance calls,
construction, pruning, or repair. Stream callbacks copy at most 64 KiB between
cancellation checks and return failure through the upstream `noexcept` boundary;
the bridge reports cancellation after that call returns.

The descriptor's integers are big-endian. Its value is exactly 86 bytes:
`0x12 | encoding:u8 | nodes:u32 | source_epoch:u64 | payload_bytes:u64 |
source_binding:32 bytes | topology_sha256:32 bytes`. Encoding 1 identifies the
little-endian native layout below; encoding 2 identifies the big-endian layout.
A reader rejects the other host byte order. The supported layout and key/slot
widths are fixed by the vendored dependency; updates must revalidate the native
layout and validator together. The project has no released format to migrate. The descriptor
stores no duplicate entry point, levels, or graph configuration. Persisted graph node count is
in `1..=16386`; empty local cores are supported in memory but are not persistent
internal bodies. The source Header must be Ready, at level greater than one,
with the same exact count and epoch. Singleton internal partitions have the
ordinary descriptor. Leaf partitions have no graph keys; empty forests and leaf
roots need none.

The native payload uses host-endian integers:

- Header: five u64 fields: node count, M, base capacity (2*M), maximum level,
  entry ordinal (40 bytes total).
- Level table: one i16 per node in canonical Child Entry order.
- Node tape, in the same order: key:u64 equal to the node ordinal, another i16
  level equal to its table entry, then each layer from zero through that level.
  Each layer is degree:u32 followed by its full capacity of u32 slots: 2*M at
  the base, M above it. Only the first `degree` slots are edges. Inactive slots
  are opaque native state, preserved by save/load and never traversed.

Levels are in 0..=16. Base degree is at most `min(2*M,n-1)`, upper degree at most
`min(M,n-1)`. Neighbor order is retained exactly because it can influence
approximate search. No self, duplicate, out-of-range, or absent-target-layer edge
is accepted. The entry has the global maximum level, and every node is reachable
from it through directed base-layer edges. The standalone codec represents an
empty core with maximum level `u64::MAX`, entry zero and no node records. No
trailing bytes are accepted. Native bytes round-trip exactly, but are not a
canonical representation of abstract topology: unused capacity may differ.
Cross-host byte-order portability is not promised. Reserved slots increase
payload size compared with an active-edge-only codec; encoded-size admission and chunking include all of
them, including for a singleton with large M.

Chunk values are `0x13 | payload_length:u32 | payload`. Every nonfinal payload
has exactly 99,995 bytes, and the final payload has 1..=99,995 bytes, so the full
value is at most 100,000 bytes. Chunk keys end in contiguous u32 ordinals from
zero; count is `ceil(payload_bytes/99995)`. Boundaries are independent of backend,
budget, node, and adjacency boundaries, including an adjacency crossing chunks.
Missing, repeated, extra, reordered, wrong-size, or checksum-mismatched chunks
fail with Corruption. Complete chunk-range scanning is required; reading only
the declared point keys would conceal extras. The graph SHA-256 covers the
concatenated payload. Checksums detect corruption, not authenticity; even bytes
with a recomputed checksum undergo full structural validation before native use.

Source identity is SHA-256 over `b"ktann/local-hnsw/source/v1\0"`, u64 Tree Key
byte length and canonical bytes, u64 parent Partition Key, u64 M, u64
`ef_construction`, 32-byte configured seed, metric u8 (L2=0, cosine=1,
inner-product=2), u64 dimension, u64 child count, then each u64 child identity
and each canonical f32 coordinate's u32 bits (zero always positive). The stored
binding is SHA-256 over `b"ktann/local-hnsw/snapshot/v1\0"`, that identity,
u64 Logical Index ID, u32 Header level, u32 count, and u64 cache epoch.
Parameters are supplied by the owning preparation/snapshot integration; no
second parameter authority is stored in the descriptor. The Header cache epoch
remains the sole body-version authority.

Encoding admits borrowed Child Entries, resident graph, and the exact output
size. `GraphRecords` exposes zero-copy chunk slices.
Restoration admits retained chunk bytes, the assembled payload, borrowed Child
Entries, validation arrays/queue, and a conservative native allocation bound
before reserving the body. Validation checks layout and edges together, followed by local
reachability; no edge-sized Rust object graph or centroid copy is created.
Allocations are fallible and operation control is checked through hashing,
validation and native stream loading. Cancellation discards the unpublished graph.
The caller must charge other retained caller-owned state and release chunk
buffers when done; these bounds describe charged allocations, not RSS.

Before accepting the Serving Artifact, validate:

- Its graph node set is a bijection with the parent's Child Entries; all child
  identities are unique and descend exactly one level. Existing topology and
  centroid equality checks remain in force.
- Counts, levels, entry point, degree bounds, neighbor ranges, no duplicate/self
  edges, canonical encoding, and graph/source binding are valid. Level-zero
  traversal from the entry point reaches every node; construction must ensure
  this or fail before acceptance. Connectivity repair, if needed by the chosen
  builder, must respect degree bounds and deterministic output.
- Every serving partition is Ready, every expected graph is present, no leaf
  graph exists, and there are no extra graph keys. Exact membership and source
  equality remain independently proved; graph validity is not their substitute.

Public Index Verification gains graph auditing under its existing one-snapshot,
object/memory limits. A resource-limited incomplete audit is not a publication
proof. Bulk publication continues its frozen multi-transaction exact comparison;
it does not call public verify and assume a large FDB index fits one snapshot.

For search, read Active Manifest and Header in the query transaction. On a cache
miss load Child Entries, graph descriptor and chunks from that same transaction,
validate count, source epoch/digest and structural encoding, then publish one
immutable internal body. Do not fetch graph bytes from a newer transaction or
from a workspace file. Persisted malformed or mismatched data is `Corruption`;
missing mandatory graphs cannot become an exact-routing fallback. Public verify
and preparation own full cross-projection/reachability audits; cache misses need
not repeat those whole-tree audits.

Extend the existing internal `CachedBody` with the validated graph. The cache
key remains index/tree/partition/kind, and reuse requires the snapshot Header
epoch. Account both Child Entries and graph allocations. Historical snapshots
must not evict or replace newer entries. Equal/newer fills use the existing
race policy; no graph-specific cache, readiness registry, pinning rule or
singleflight state is introduced. A body change replaces/invalidates its graph and increments that same epoch
atomically.

Exact routing uses the same graph-body validation and ignores adjacency
when selecting candidates. Thus both query modes fail on a malformed graph
body; exact is an explicit algorithm baseline, not a corruption recovery path.
For serving-only timing this gives equal loading prerequisites. Also report
ordinary exact-index costs separately to expose the graph's storage/load tax.
A warm cache retains the existing integrity model: out-of-band byte corruption
without an epoch change need not be discovered on every cache hit.

## 7. Candidate contract

For a parent at level `L`, set `B = beam_width(leaf_beam, L - 1)`. Local HNSW
returns at most `min(n, B)` distinct child candidates. It seeks that full local
width with `effective_ef`; it must not divide `B` among admitted parents.
Deduplicate graph exploration by node ordinal before returning candidates.
This does not deduplicate tree ownership: retain the existing duplicate incoming
Child Entry check for all children in each exposed parent, including children
not selected by HNSW or local exact top-B. Its query-wide identity set counts
against routing working memory. This retains an O(exposed child count) check
even on warm graph hits; include its CPU/memory cost in end-to-end measurements.
Distances returned by the local core are retained from its routing-kernel
evaluations and ordered by distance then Partition Key, without recomputation.

Every admitted parent at a given tree/depth contributes before the next depth
selects its single global beam. Bound each tree/depth candidate container by `B`
using deterministic top-B merging; local outputs do not create independent
per-parent beams. Preserve fair progress across Tree Keys and the existing
logical partition budget. Graph vertices examined inside a body do not consume
additional partition visits; selected child bodies do when visited.

Exact routing scores all children and may retain local top B before that same
global merge. This is equivalent to full global exact selection under the total
tie order: a parent's child below local rank B cannot enter global top B.
Verify that equivalence against the current full-exposure traversal. HNSW has
no exact-top-B or final-recall guarantee. When `B >= n`, directly scoring all
children is a deterministic exhaustive case, independent of cache warmth; the
graph body still requires valid persisted graph data.

Exploration has explicit finite bounds from `n`, bounded levels/degrees and
`effective_ef`. Upper-layer greedy moves strictly improve distance/key; visited
state prevents repeated node evaluation during base-layer exploration. Heap
storage and stale entries must also be bounded. Check operation control
between bounded work batches. The first stage adds no silently truncating graph
budget: inability to allocate the admitted working set is `LimitExceeded`, and
cancellation/deadline is an error, not an unexplained shorter candidate list.
A valid connected graph and unexhausted operation must fill `min(n, B)` local
candidates; a builder/search implementation unable to do that does not satisfy
this contract. Existing Search Budget exhaustion still returns the established
partial Search Outcome. Keep graph work counters separate from that accounting.

## 8. Backend and memory feasibility

[`Backend::hard_limits` and `admission_budget`](../../src/storage/backend.rs)
are authoritative for writes. The current
[FoundationDB adapter](../../ktann-foundationdb/src/backend.rs) limits physical
keys to 10,000 bytes and values to 100,000 bytes, with conservative admission of
10,000 mutations and 9,000,000 mutation bytes per transaction. Its physical prefix
counts against limits. FDB native affected-data limits can still reject a
transaction below these policy budgets. RocksDB currently uses the same
transaction admission sizes despite larger hard value limits.

100,000-byte graph values fit the current FDB value ceiling; they are not permission
to exceed another backend's advertised limit. Serving encoding checks every
key/value, including chunk framing. Load planning accounts for graph keys,
physical prefix bytes, checkpoints and fencing overhead, using the existing
budgeted loader. If even one value plus required bookkeeping cannot fit a
transaction, fail boundedly; do not loop shrinking an indivisible item. Native
`TransactionTooLarge` remains distinct from declared-budget `LimitExceeded`.
Large graphs load over multiple transactions while hidden. Publication remains
a small Manifest/proof transition, never an all-graph transaction.

For one internal parent with `n` children and dimension `d`, centroid payload
alone is `4*n*d` bytes. At `n=32768, d=768` it is 96 MiB, before Child Entry
objects, graph adjacency, allocation overhead, queues or visited arrays. Thus
large internal capacity must not be admitted solely because each KV value fits.

The resource contract requires:

- Construction reserves a checked upper bound for centroid arrays, node/edge
  storage at bounded levels/degrees, neighbor-selection scratch, visited state,
  and encoding buffers before starting one graph. Spilled inputs, scratch and
  accepted artifacts also obey their existing separate disk quotas. No forest
  of graphs is retained in RAM. If one graph exceeds the worker budget, fail
  `LimitExceeded` before acceptance; budgets never reduce M or construction ef.
- Search has a configured per-query routing working-byte ceiling `W`. Before
  loading, bound the complete internal body and query workspace using trusted
  configuration/Header bounds, then validate descriptor sizes before allocation.
  Charge the retained body handle even on a cache hit; it may outlive eviction.
  Include centroids, graph, visited/heap state, decoding buffers, candidate merge state and the query-wide observed-child
  identity set (which survives individual parent visits). Release one parent's body before consuming the next. Existing
  bounded first-page prefetch remains separately accounted, never four complete
  graphs. Reject over-limit work equally on warm and cold paths.
- With cache capacity `C`, admitted concurrent operations `A`, and bounded
  prefetch/metadata overhead `P`, routing residency is conservatively bounded
  by `C + A*(W + P)`. Double-counting cached bodies is acceptable for this bound;
  omitting evicted but referenced bodies is not. This is a routing bound, not
  total process RSS: leaf data, backend/OS caches, build jobs and runtime state
  must also be measured and bounded by their owners.
- Cache capacity zero or an entry larger than `C` disables retention only. A
  body fitting `W` still loads and searches without retention. A miss/eviction
  never constructs a graph, changes ef, or selects a different routing policy.
  Concurrent misses may duplicate bounded loading under existing admission.

Snapshot expiry while reading many graph chunks fails the complete query under
existing backend semantics; no cross-snapshot continuation is introduced.
Emit build time/peak working bytes, graph stored/decoded bytes, cache fills and
miss bytes, routing distances/visits, per-level candidates and merge work.
Metrics retain existing privacy rules and do not label raw Tree Keys or vectors.

## 9. Acceptance and next implementation boundary

The following checks govern the complete feature, including integration steps:

1. **Lifecycle and writes:** graph-enabled indexes support ordinary creation and
   writes after Bulk Build publication. Building/drop/abort precedence is
   preserved. Mutation and maintenance keep graph and topology changes atomic;
   public serving refinement rejects graph indexes.
2. **Artifact order:** zero/nonzero refinement produces final centroids before
   graph construction. Source/assignment/Child Entry/graph joins agree for all
   metrics. Attempting post-load edits or mismatched resume options fails.
3. **Persistence:** canonical codec tests cover chunk boundaries, oversized
   values, count/epoch/digest mismatch, missing/extra nodes/chunks and malformed
   adjacency. Independent-process rebuilds of identical inputs match bytes.
4. **Recovery:** inject crashes/takeovers before acceptance, between graph load
   chunks, after completed load, during validation and after uncertain publish.
   Only the original fully validated identity becomes Active. Run shared load
   and publication suites on Memory, RocksDB and FoundationDB.
5. **Snapshot/cache:** cold, warm, disabled and turnover caches yield identical
   candidates for a fixed graph/query. Test historical/newer epoch races, drop
   and name reuse, concurrent misses, allocation ceilings and snapshot expiry.
   No miss invokes the builder; malformed data fails closed.
6. **Routing:** compare local exact top-B plus global merge with current exact
   output, including ties, uneven parent sizes, leaf-only/multilevel trees,
   Tree Key fan-out and budgets. HNSW returns only owned children, with full
   local width, bounded work and unchanged leaf/filter/rerank semantics.
7. **Experiment:** keep rerank budget fixed. Compare final recall@100 (including
   the >90% target), per-query quality distributions and matched-recall latency/
   QPS using independent queries and multiple graph seeds. Include routing CPU,
   distance evaluations, per-level visits, leaf entries, reads/decoded bytes,
   build/storage/peak-memory cost, cold loading, realistic cache sizes and
   concurrent loading. Report actual topology; deduplicate configurations that
   collapse to the same graph. Local coverage is not final recall evidence.

Capacity separation is implemented throughout IndexConfig, Manifest codecs,
construction and artifact bindings, root-fit/grouping, occupancy validation,
refinement leaf bounds, and mutable maintenance thresholds/drain/recovery age.
Level 1 selects leaf bounds; higher levels select internal bounds, including
roots, with the root minimum-occupancy exception. Both pairs require positive
minimum, `2*minimum <= maximum`, and maximum at most 16,384 for internal partitions (65,536 for leaves). Defaults remain
64/512 for both pairs. Exact routing already uses these independently sized
trees. All four values participate in Manifest and forest/serving identity;
these pre-release contracts change directly without a version upgrade.

Earlier capacity experiments included 32768; current supported operating points
include internal maxima 4096/8192/16384 and leaf maxima 64/128. These are
experimental inputs, not defaults. Independent sizing must preserve a fixed leaf
layout/centroids when comparing internal grouping; no rerank-budget sweeps.

The graph format and snapshot reader implement the source and resource contracts
above. [Search](search.md#graph-index-bodies) owns graph-cache admission,
complete graph-body ownership, and the explicit loading/query interfaces. Candidate
generation must supply the existing query snapshot, reserve query-wide retained
state, consume one parent at a time, and merge the returned child identities and
unmodified routing distances under the existing global beam and budgets. This
document does not claim an end-to-end performance or 100M/10B feasibility result.
Production parameter choices remain workload-dependent. Graph maintenance is
defined below. There is no stable release requiring compatibility machinery or
migration of old builds.


The step-9 experiment entrypoints and metric semantics are documented in
[benchmark routing experiments](../../benchmarks/README.md#frozen-layout-routing-experiments-issue-195-step-9).
Benchmark-only global routing and diagnostic replay are feature-gated and use the
production leaf/rerank pipeline. They do not add a production global index or
change mutation semantics. Matched-recall tuning is covered by step 10.

## Local graph transitions

`storage::graph_transitions` owns bounded graph snapshots, private preparation,
source fencing and graph-record publication. The topology owner composes its
Header, Child Entry and child creation/removal or movement in the same transaction.
A graph replacement alone does not establish valid tree membership. Foreground
write routing remains exact.

### Algorithm and preparation

`GraphTransitionSource::read` collects a complete partition body from one snapshot
with bounded scans. An absent partition is a creation source; empty internal and
leaf bodies have no graph. `prepare` consumes the source, exact proposed Header and
canonical proposed Child Entry body. It validates the old graph before modifying
or removing it, and checks finite coordinates, level/count/capacity. The topology
owner determines the final body epoch; state-only changes preserve it. No separate
Ready-only projection update protocol exists.

Insertion and centroid replacement consume a private restored USearch graph.
The distance callback uses proposed centroids throughout the update. Old slots
retain their identity while USearch replaces their neighborhoods; inserted nodes
are appended. A representation-only permutation then restores canonical child
ordinal order, including keys, adjacency and entry slot. USearch still owns
insertion, neighborhood selection and pruning. The existing directed-reachability
audit and bounded successor repair run on the complete private result. Candidate
distances are not recomputed.

A batch containing any deletion uses the same canonical builder as initial
construction. This removes nodes and their storage without tombstones or a
separate stable-slot directory. The accepted cost is whole-parent construction
for such a batch: the step-10 Cohere 128/16384 single-root case took 13.61 seconds
for build/serialization. That measurement excludes centroid reads and publication
and is not a latency promise for this update path. A small synthetic native probe
also found a substantial single-centroid update cost advantage, but does not
establish end-to-end maintenance throughput or long-run quality.

`prepare` is synchronous CPU work. Its caller must dispatch it to bounded blocking
execution and retain admission through actual exit. Its budget covers source and
proposed centroids, kernel, restoration, native scratch, ordinal mapping, encoded
output and all retained graph plans. Cancellation discards private state.
Prepared graphs never populate the shared query cache. Ordinary leaf membership
changes and cache misses do not cause preparation or parent-centroid changes.

### Bounded atomic publication

The production configuration supports internal split thresholds up to 16,384
(8,192 remains a supported operating point), internal minimum occupancy up to
128, and graph M in 2..=16. Leaves retain their independent capacity limits.
Internal graph preparation permits two transient entries above the configured
split threshold. Exposing a split's two targets waits when the parent is already
above its threshold: ancestors must make room first.

Native level sampling retains its original SplitMix64 word consumption and
geometric distribution, but caps the returned level at `floor(log_M(65536))`.
This is 4 for M=16; the expected population beyond that level is below one even
at the native maximum node count. The integer calculation is shared semantically
by sampling, safe decoding and serialization admission. For an accepted M, the
height cap preserves the configured degree, ef and metric. Tests compare sampling
before/after, including a seed with an original level 6 node.

A node at level L occupies `16 + 8*M + L*(4 + 4*M)` serialized bytes, plus a
40-byte header per graph. For M in 2..=16 the per-node bound is at most 416 bytes.
The backend mutation-byte admission is 9,000,000 bytes; FDB's separate affected-
data ceiling is 10,000,000 bytes and includes conflict ranges. Graph chunk values
are at most 100,000 bytes (99,995 payload bytes plus framing).

For graph-aware internal topology, a drain batch touches one source and one
receiver. A split distributes at most `threshold + 2` children; do not charge
both bodies as full independent copies of the original population. A merging
source has fewer than 128 children; its receiver has at most `threshold + 2`
after the batch. Thus at threshold 16,384, the two graph bodies hold at most
16,513 nodes and occupy at most 6,869,488 bytes. At most 70 chunks suffice. Even
with 10,000-byte physical keys, a conservative charge for framing, mutation keys,
write-conflict endpoints and a 900,000-byte authority envelope remains below
10,000,000 bytes. A native FDB 7.3 probe of this worst-case payload/key shape
committed at approximately 9,868,260 affected bytes.

`PreparedGraphTransition::protect` update-protects the Active Manifest, source
Header and descriptor before topology edits. The topology owner validates child
authority through its existing expose/drain/finalize operations, then checks each
final Header with `validate_target` before appending graph descriptor, obsolete
tail deletes and chunk payloads. The combined protected reads and non-payload
mutations share the authority envelope; native adapter admission and commit
checks remain authoritative. Preparation runs outside the transaction, and a
stale source discards the prepared plan before publication.

After an unknown commit, the graph-aware driver rediscovers persisted topology
on its next call. It does not infer success from a superseding epoch or replay
child creation/removal from an uncertain plan.
Cancellation before commit discards the transaction. There is no begin/load,
pending or retired generation, durable progress cursor, cancel operation or graph
GC. Existing backend snapshots retain overwritten graph bytes for old queries;
caches still validate the source epoch and descriptor before reuse.

### Split/merge integration boundary

The graph-aware storage drain validates a single receiver and the transient
node ceilings. Graph-aware maintenance composes the following transitions in
`maintenance::graph_topology`, using `storage::graph_transitions` for snapshot
preparation and publication:

- Split exposure inserts two parent edges only with the existing two-slot
  headroom. It prepares the changed parent graph before exposing either target.
- Internal drains move children to one receiver per transaction, publishing both
  changed bodies with the exact memberships. Merge reselects a Ready receiver
  from a fresh snapshot; receiver headroom bounds each batch.
- An empty internal body has neither descriptor nor chunks. The last drain
  removes its graph in the same transaction, making later source-prefix point
  cleanup complete without a graph rescan. A first incoming child builds a graph
  for the formerly empty receiver.
- Final source removal replaces the parent's graph. Root completion builds the
  two-child root graph while converting the root in place. Root contraction is
  not a supported topology operation.
- State-only transitions preserve the body epoch and graph bytes. Promoting
  receiving targets does not rebuild their graphs. Leaf membership movement
  changes neither parent centroids nor graphs.

Every preparation is private and outside the write transaction. A combined
working-set admission includes all source/proposed bodies and retained output.
Publication protects exact source Headers/descriptors before topology writes,
checks planned target Headers, and admits all topology changes, graph envelopes,
tail deletions and protected reads against one combined authority envelope before
adding graph payloads. Stale preparation replans from a new snapshot. Unknown
commit is returned without blind replay; subsequent maintenance rediscovers the
committed state. Backend snapshots retain overwritten graph bytes for older
queries; there is no staging generation or asynchronous reclamation protocol.

Graph configuration is immutable; graph contents evolve with ordinary writes and
maintenance. Serving `Index::refine` returns `InvalidArgument`; optional artifact
refinement precedes initial graph construction. Graph readers and verification
accept legal non-Ready bodies and graph-free empty internal bodies.

The serialization proof and storage probes establish atomic publication
feasibility, not throughput or long-running recall. Empirical opening evidence
must come from the mutation histories and update experiments, not the frozen-dataset
step-10 matrix alone.

Obsolete source-tail chunk deletions count against the same combined 900,000-byte
non-payload envelope, including their physical keys and conflict costs. They are
not additional to the bound. Ordinary publication includes those deletes before
its envelope preflight; topology composition enforces this envelope
across all participants. A smaller batch or narrower keys may be needed to fit.
