# Approximate Search and Partition Caching

This module owns vector numeric semantics, persistent RaBitQ7 codes, budgeted
Tree Key enumeration, bounded traversal, candidate selection, exact reranking,
and cache correctness.

Filter semantics, Tree Key range planning, and synopsis classification are
defined by [Filtering](filtering.md).

## 1. Vector and distance semantics

Stored and query vectors contain finite f32 components and have the Manifest
dimension. Routing preprocessing validates the vector, normalizes cosine in f64
and converts it to finite f32, leaves L2/inner-product unnormalized, then applies
the persisted rotation. Internal centroids are full-f32 routing vectors. Routing
ranks a partition's Child Entries by the f64 routing distance between the
preprocessed vector and each persisted centroid: squared Euclidean distance for
L2, negated dot product for inner product, and negated dot product for cosine.
Cosine records are normalized during preprocessing and cosine split centroids
are normalized after their arithmetic mean is trained; a zero centroid remains
zero and uses a neutral distance of zero. Distance ties resolve to the smaller
Partition Key. Routing distances use eight independent f64 sums for component
indices modulo eight, followed by a fixed balanced pairwise reduction. Batched
routing reuses the same single-distance kernel, shortening the dependency chain
without multiplying the live accumulators or adding heap allocations. Vectors
shorter than one eight-component block use direct scalar sums, interleaved for
batch calls. Exact reranking keeps its separate kernel:
it reads the unrotated original vector, accumulates in f64, and defines:

- L2 ranking: squared distance `sum((q_i - x_i)^2)`, with public SearchHit
  returning its Euclidean square root;
- inner product: `-sum(q_i * x_i)` so smaller remains better;
- cosine: `1 - dot(q, x)` after validating both norms are finite and nonzero.

Any non-finite result derived from caller input is `InvalidArgument`. Non-finite
or inconsistent decoded persistent numeric state is Corruption. Exact hits sort
by distance and then unsigned lexicographic Record ID bytes.

## 2. Absolute RaBitQ7 format

Each Leaf Entry stores an absolute, centroid-independent code. A fixed seeded
orthogonal Givens rotation is defined by the Persistent Format.
The storage design fixes its exact ChaCha8 permutation and three-round pair
protocol.

For each rotated component `x_i`, encoding stores one sign bit and a six-bit
magnitude `m_i` in `0..=63`. Define the signed integer code:

```text
c_i = -m_i  when the sign bit is negative
c_i =  m_i  otherwise
```

Zero magnitude always has positive sign. Initial quantization uses
`max_abs / 63`, nearest rounding with ties away from zero, and clamping to
`0..=63`. With the integer codes fixed, f64 computes the nonnegative
least-squares scale:

```text
scale = max(0, sum(x_i * c_i) / sum(c_i * c_i))
```

`code_norm_squared` is the exact integer denominator. Scale is rounded to finite
f32 for storage; reconstruction is `x_hat_i = stored_scale * c_i`.
Reconstruction error is then computed in f64 against that stored scale and
stored as the least finite f32 not below the exact result. A zero vector has an
all-zero header and streams.

The payload is a 12-byte little-endian header (`scale: f32`,
`code_norm_squared: u32`, `reconstruction_error_upper: f32`), followed by
`ceil(dimension/8)` LSB-first sign bits and `ceil(6*dimension/8)` LSB-first
magnitudes. Dimension comes from the Manifest. Wrong length, nonzero padding,
inconsistent norm, negative/non-finite metadata, or noncanonical zero is
Corruption.

## 3. Conservative approximate intervals

Production scores each stored code with one exact integer dot. The rotated
query is quantized once per search: `L=min(32767, floor(i32::MAX/(63*n)))`,
`s=max(abs(q_i))/L` (or 1 for a zero query), and
`a_i=clamp(round(q_i/s),-L,L)`, stored as i16. These components and scale
are query scratch; the persistent vector codec and exact reranking are unchanged.
For signed seven-bit stored components `c`, `S=sum(a_i*c_i)` is exact:
`n*L*63 <= i32::MAX`. The dimension-dependent limit uses the highest
i16 precision compatible with exact i32 accumulation, avoiding a wider dot
reduction. It uses one additional scratch byte per query dimension compared
with i8 screening. The dot estimate
is `D=(f64(S)*s)*code_scale`, with two f64 multiplications.

Query preparation computes an outward interval for every `q_i-s*a_i` using
directed multiplication and subtraction. Squaring the largest absolute endpoint,
summing upward and taking an upward square root gives `E_q >= ||q-s*a||`.
Let `Q >= ||q||` and `C >= code_scale*sqrt(sum(c_i*c_i))`. Cauchy–Schwarz
bounds query-quantization error by `E_q*C`. Also `||s*a|| <= Q+E_q`, so the
two final multiplications have rounding error at most
`2*f64::EPSILON*(Q+E_q)*C` (an upper bound on their gamma factor).
Thus the dot endpoints are `D ± (E_q + 2*f64::EPSILON*(Q+E_q))*C`,
computed outward. Finite f32 inputs and the public dimension limit keep the
nonzero scaled integer products within the normal f64 range.

For reconstruction `x_hat` and stored reconstruction-error bound `E_x`:

| Metric | rough distance | reconstruction-error radius |
| --- | --- | --- |
| Inner product | `-D` | `norm(q) E_x` |
| Cosine | `1-D` | `norm(q) E_x` |
| Squared L2 | `max(0, norm(q)^2 + norm(x_hat)^2 - 2D)` | `2 sqrt(d_upper) E_x + E_x^2` |

Directed norm and dot endpoints first bound distance to the reconstructed vector.
The reconstruction-error radius then expands those bounds to cover the original
vector; L2 lower endpoints clamp at zero. `d_upper` is the conservative upper
endpoint before adding reconstruction error. The rough score need not equal the
unquantized query/code dot; the interval accounts for both query and stored-vector
quantization. Final Search Hit distances still use original vectors.

### 3.1 Admission against the global rough-distance heap

Each eligible entry is scored once. A full heap rejects a rough score strictly
greater than its current worst rough score before constructing its interval.
Equality remains eligible for Record ID tie-breaking. Successful admissions can
only improve the cutoff; the next entry uses the updated value. This admission
rule is exactly the same rough-score top-budget policy as full scoring and
sorting, without a second dot, separate screening bound, or batch scratch.
Every eligible identity is still registered for duplicate-membership detection.

## 4. Tree Key enumeration

[Filtering](filtering.md#4-conservative-tree-key-planning) derives conservative
Tree Key ranges. Search enumerates forward directory pages in canonical order,
counting every decoded/checked Tree Key against one global budget. At most the
budgeted eligible keys are materialized, and enumeration finishes before
partition traversal. Empty trees and their Tree Manifests remain until index drop.

## 5. Filter integration

[Filtering](filtering.md#5-partition-synopsis-classification) defines exact
eligibility and conservative synopsis classification. Search uses `NoMatch` to
prune leaves and admits only entries that make the complete predicate TRUE,
before RaBitQ candidate selection. All pruning and evaluation use the same
snapshot as traversal and reranking.

## 6. Search algorithm

### 6.1 Snapshot and traversal order

One consistent snapshot performs:

1. Active Manifest validation and request compilation;
2. bounded Tree Key enumeration;
3. deterministic level-scaled beam traversal of every eligible tree under one
   global partition budget;
4. synopsis pruning and exact leaf predicate evaluation;
5. bounded RaBitQ candidate selection;
6. original Vector Record batch loading and exact reranking;
7. deterministic top-k ordering and budget report construction.

Every Leaf Entry in an admitted partition is considered and counted. Exact leaf
filtering precedes RaBitQ scoring, so rejected entries consume no scoring work.
The complete loaded body is still decoded and validated before filtering;
predicate rejection cannot hide malformed persistent entries.

Traversal is a level-scaled beam. The leaf-level base beam defaults to 128 and
SearchOptions may override it per request within the hard cap; moving one
level toward the root divides it by two with minimum one. All admitted parents
at one depth contribute before the next depth selects the globally nearest
beam for each tree. `SearchOptions::routing_mode` defaults to Exact. Graph-enabled
indexes can explicitly select PartitionLocalHnsw with positive `ef_search`;
indexes without graph configuration reject that choice. Each parent contributes up to the full
local next-level beam B, never a share divided among parents. Candidate scores
come directly from the graph's f64 routing kernel evaluations. Exact scores all
children. Both modes merge into a single bounded top-B heap per tree/level,
ordered by distance and Partition Key. Every exposed Child Entry is checked
for duplicate incoming references, including children omitted by local search. This prevents one wide parent from
consuming the next level's beam before other admitted parents contribute their
children. Eligible trees advance fairly. Ties use Tree Key, Partition Key,
then Record ID.
Partition and rerank bounds are charged
before corresponding work. No speculative read-ahead occurs beyond a budget.

### 6.2 Bounded IO and prefetching

Traversal batches snapshot Header reads for a funded prefix of up to 1024
non-root partitions, including leaves. Filtered leaf visits include their
Synopses in the same batch, with at most 2048 logical keys per request; adapters
chunk requests at their own point-read ceilings. Visits, body loads,
candidate selection, and budget accounting retain their original order. Roots
are read individually because their transition state can inject split targets
ahead of the queued work. Batching changes backend scheduling, not the selected
partitions or recall policy.

Within that funded prefix, traversal validates metadata and applies Synopsis
pruning before prefetching body pages. At most four first pages of cache misses
are read together, each under the existing 256-entry/1 MiB encoded scan limits
(including the backend's oversized-first-item rule). Cache hits do not scan.
Visits consume those pages in order and stream any continuation through one
body at a time; a complete body passes count validation before cache publication.
This bounds temporary prefetching independently of complete partition size.

### 6.3 Candidate selection

Every predicate-eligible entry of every visited leaf is scored. One global max
heap owns the exact-rerank capacity and retains at most that many entries, ordered by rough distance
and then unsigned Record ID bytes. There is no per-leaf quota. Before reranking, the retained pool's kth-smallest
upper endpoint excludes candidates whose lower endpoint is greater; with fewer
than k candidates no interval exclusion is applied. This bounded final pass avoids
loading clearly noncompetitive vectors, especially for small or selective queries.
Rejected or evicted heap candidates set exact-rerank budget exhaustion. This is an approximate ranking policy:
fixed-capacity retention does not guarantee exact top-k recall. The exact-rerank
stage loads every admitted candidate; it does not apply a second capacity limit.

Only admitted candidates acquire a shared reference to the leaf's immutable entry
array and their original entry ordinal. Filter fields are never copied. The final
rerank stage reads these original projections to verify them and the source Record
Location against authoritative records from the same snapshot. Expanded RaBitQ
codes are not retained by candidate references. Eviction immediately releases a
source array when its last candidate leaves the heap.

The pool scores eligible entries individually and admits them directly,
without per-leaf candidate or screened-ID buffers. Every eligible Record ID is
registered once in a separate compact list, including screened, rejected, and
evicted entries. Only admitted heap candidates require an additional ID reference.
After traversal, one borrowed-ID set is reserved from the actual list length to
validate duplicate membership before exact reads. Both the list and set are
released immediately after successful validation. Thus candidate count and projection ownership are bounded by the rerank
budget, while corruption-detection identity storage grows with the number of
predicate-eligible entries visited. Both retained IDs and transient duplicate-table space are charged to routing working memory;
shared source arrays are charged once, including on cache misses or eviction.

### 6.4 Intermediate topology

Intermediate traversal follows the persistent reference rules exactly:

- non-root Splitting/Draining sources and ReceivingSplit targets are visited
  only through current Child Entries;
- root Splitting searches the root body only;
- root DrainingSplit searches root and both target bodies;
- ReceivingSplit and Merging search like ordinary same-level partitions.

Physical bodies are deduplicated by `(TreeKey, PartitionKey)` and charged once.
Two distinct incoming Child Entries for one non-root partition are Corruption.

During split/merge, traversal follows every reference named by the persistent
state. Exact Record Location/Leaf Entry ownership prevents duplicate result
membership; defensive duplicate Record IDs encountered in one snapshot are
Corruption rather than silently deduplicated.

## 7. Search budgets and response

Leaf Entry scans have no independent cap. Beam and the partition budget limit
which partitions are admitted; actual partition sizes determine scan work.
Leaf Entry usage is a 64-bit counter, not an exhaustion dimension.

SearchOptions overrides nonzero bounds for Tree Keys and partitions within
hard caps, plus the leaf-level base beam width. Exact-rerank
sizing is owned by search: checked arithmetic computes
`max(64,k+ceil(k/4))` and bounds it by the
Runtime's exact-rerank ceiling. Defaults are process-local and benchmark-tunable.
One successful response reports usage and each dimension that prevented
pending work.

The API deliberately has no `complete` boolean: ANN search is approximate even
when no logical budget is exhausted. It also has no quality score or
continuation token. Callers needing more traversal work may resubmit with
higher Tree Key or partition budgets, or a wider beam. Exact-rerank
sizing changes only with `k` or the engine policy. RaBitQ estimate distances
likewise stay internal: there is deliberately no estimate-only response mode or
skip-rerank switch. Focused selection tests protect the candidate policy, while
search reports exact-rerank usage, exhaustion, and stage latency.

## 8. Partition cache

The Runtime shares byte-bounded caches of decoded internal and leaf search data.
Foreground preparation also reuses committed internal bodies before staging
internal changes, under the same snapshot/epoch rules
([ADR 0023](../adr/0023-committed-internal-routing-cache.md)). Graph-aware
maintenance may seed complete immutable internal graph bodies after known
successful commit. General write routing after staged changes and leaf mutation
do not fill this cache.
A key contains Logical Index ID, canonical Tree Key, Partition Key, and kind.
Every entry contains Header cache epoch. Search reads the Header from its own
snapshot and may reuse cached data only when epoch and kind match; otherwise it
loads and decodes the partition from that snapshot.

Leaf bodies expand validated RaBitQ7 components to signed bytes once on load.
Each leaf body owns one contiguous expanded-component buffer, and its entries
retain numeric headers and ranges into that buffer instead of packed payloads.
Initial reservation is capped at one scan page; larger bodies grow as validated
entries arrive. Storage validation and the subsequent expansion remain separate
passes, and complete body counts are checked before publication. Cached Record
IDs own their bytes so they cannot keep the old encoded buffer alive. Cache
accounting charges the shared expanded allocation once at the body owner, in
addition to entry-owned data and the body envelope. Scoring uses the integer query/code dot and conservative intervals described
above; the persistent RaBitQ7 format is unchanged.

Entries are immutable and never pinned. Concurrent misses may duplicate work and
race to publish an equal or newer epoch; there is no waiter/cancellation state.
Cache insertion may be skipped when an item exceeds capacity. Internal and leaf
byte shares may be configured, but the eviction policy is an internal
benchmark-driven detail, not a persistent or public compatibility contract.
Corruption is never cached.

### Graph index bodies

The graph reader uses the complete query's existing validated logical
transaction and the snapshot Header already read by traversal. It reads the
Graph Descriptor on every access, including hits. Internal kind, exact count, epoch, configured capacity and
the descriptor's source binding must agree. Missing or mismatched graphs are
Corruption; there is no exact fallback. Ordinary indexes do not request graph
records. Both routing modes on graph-enabled indexes validate complete
nonempty graph bodies. Graph indexes
allow every legal searchable state and skip graph loading for an exactly empty
internal body after applying root split traversal.

On a miss, the reader scans complete Child Entries and contiguous graph chunks
from that same transaction, then restores the persisted native graph. It never
builds a graph or changes its seed, construction parameters, adjacency or query
distances. A graph internal body owns its Child Entries, native graph, source
binding, and numeric kernel together. An internal cache entry may contain ordinary
projections or the complete graph body. Foreground exact routing reuses
projections from either representation. When it populated ordinary projections
first, the graph reader upgrades that same-epoch body after loading and validating
the persisted graph. A late same-epoch projections-only fill cannot downgrade a
complete graph body; a newer epoch may replace it normally.
Cold reads retain unpublished entries and chunks until restoration succeeds,
then publish one complete shared body. Older snapshots cannot replace newer cached
epochs. Eviction does not invalidate handles still held by queries.

`RuntimeConfig::routing_working_bytes` (default 256 MiB) bounds one query's graph
routing work independently of retained cache capacity. Before loading entries or
chunks, the reader admits checked bounds for the complete body, kernel creation,
chunk and assembly buffers, native validation/restoration, private query scratch,
and candidate output. The caller reserves other retained query state, including
global candidate merging and observed-child identities. Admission uses the same
bound on warm and cold reads. Zero/insufficient cache capacity disables retention,
not graph use; work exceeding the routing ceiling returns LimitExceeded.

Each parent handle is consumed before loading the next. Restore and query work
run on the blocking pool, retain foreground admission through actual task exit,
and honor cancellation/deadlines. Scratch is private to each query; native graphs
are immutable and shared. Snapshot expiry propagates to complete-query retry;
neither graph reads nor CPU restoration open a new transaction. Traversal passes
its already-read snapshot Header into the reader, avoiding a duplicate Header
read. Graph internal bodies are consumed serially; leaf
visits retain bounded metadata/body prefetch. Exact on a graph index loads and
validates the same mandatory graph body, then ignores adjacency; corruption
never triggers a routing fallback. All children still participate in the
query-wide incoming-edge integrity check. Its hash storage, live frontiers,
query inputs, and any leaf candidates retained by shallower trees are charged
before admitting the next graph. These checks and native scratch admission are
independent of cache warmth. Graph exploration consumes no logical partition
visits; each visited child body consumes its ordinary visit.

## 9. Validation

- Golden and property tests cover signed-code encoding, error rounding,
  conservative intervals, zero vectors, malformed payloads, and exact distance.
- Predicate properties compare compiled evaluation and synopsis truth sets with
  a simple SQL three-valued oracle over boundary values and NULLs.
- Deterministic tests cover Tree Key range widening, scan budgets, traversal
  ordering, every exhaustion dimension, and intermediate
  topology traversal.
- Cache histories prove snapshot epoch validation and safe concurrent fills
  without asserting a particular eviction algorithm.
- Recall and performance benchmarks compare brute force on fixed public and
  synthetic distributions and report all budget use; they establish baselines,
  not an unsupported SLA.
