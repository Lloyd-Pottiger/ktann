# Filtering and Approximate Search

Status: Implementation-ready

This module owns vector numeric semantics, persistent RaBitQ7 codes, Tree Key
query planning, predicate evaluation and synopses, bounded traversal, exact
reranking, and cache correctness.

## 1. Vector and distance semantics

Stored and query vectors contain finite f32 components and have the Manifest
dimension. Routing preprocessing validates the vector, normalizes cosine in f64
and converts it to finite f32, leaves L2/inner-product unnormalized, then applies
the persisted rotation. Internal centroids are full-f32 routing vectors. Exact
reranking reads the unrotated original vector, accumulates in f64, and defines:

- L2 ranking: squared distance `sum((q_i - x_i)^2)`, with public SearchHit
  returning its Euclidean square root;
- inner product: `-sum(q_i * x_i)` so smaller remains better;
- cosine: `1 - dot(q, x)` after validating both norms are finite and nonzero.

Any non-finite result derived from caller input is `InvalidArgument`. Non-finite
or inconsistent decoded persistent numeric state is Corruption. Exact hits sort
by distance and then unsigned lexicographic Record ID bytes.

## 2. Absolute RaBitQ7 format

Each Leaf Entry stores an absolute, centroid-independent code. A fixed seeded
orthogonal Givens rotation is part of persistent format v1. The storage design
fixes its exact ChaCha8 permutation and three-round pair protocol.

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

Production v1 uses the scalar f64 reference kernel for rotated query/code dot
products. This avoids an unaccounted f32/SIMD rounding term and makes the stored
reconstruction error sufficient for conservative intervals.

For query `q`, reconstruction `x_hat`, and error upper bound `E`:

| Metric | rough distance `d_hat` | radius `B` |
| --- | --- | --- |
| Inner product | `-q dot x_hat` | `norm(q) E` |
| Cosine | `1 - q dot x_hat` | `norm(q) E` |
| Squared L2 | `max(0, norm(q)^2 + norm(x_hat)^2 - 2 q dot x_hat)` | `2 sqrt(d_hat) E + E^2` |

The interval is `[d_hat - B, d_hat + B]`; only the squared-L2 lower endpoint is
clamped to zero. Cosine is not clamped. A future optimized kernel is permitted
only after its proven rounding bound is added to `B`; tests alone are not a
substitute for that bound.

## 4. Tree Key planning

Tree Key fields form an ordered memcomparable tuple. Each valid record maps to
one Tree Key and one tree. Empty Tree Key schema means one tree. Empty trees and
their Tree Manifests remain until index drop.

Predicate compilation derives exact field-prefix ranges when possible. Equality
and bounded `IN` on leading Tree Key fields create direct ranges; representable
ranges on the next field narrow them. If exact disjoint range expansion would
exceed 1,024 ranges, planning widens conservatively and relies on exact predicate
evaluation later.

Search enumerates forward directory pages in canonical key order and counts
every decoded/checked Tree Key against one global budget. At most the budgeted
eligible keys are materialized before traversal, so an unlimited number of
stored trees cannot cause unbounded memory. Enumeration finishes before
partition traversal to make budget use deterministic.

## 5. Predicates and Partition Synopses

One compiled evaluator implements SQL TRUE/FALSE/UNKNOWN. Only TRUE qualifies a
record. Leaf Entry fields are evaluated before approximate candidate admission.

Every field, including Tree Key fields, maintains its configured synopsis:

- MinMax stores exact min/max among non-NULL values, plus exact `has_null` and
  `has_non_null` flags;
- MinMaxBloom adds a fixed-size Bloom summary for equality/IN pruning.

The compiler evaluates a predicate over possible truth-value sets. A synopsis
result is `NoMatch` only when TRUE is impossible, and `AllMatch` only when TRUE
is the sole possibility. `Not` complements truth sets rather than inverting a
boolean pruning result.

Insertion incrementally widens MinMax, sets NULL-presence flags, and sets Bloom
bits. Delete and replacement never shrink a synopsis; stale history may weaken
pruning but cannot make it unsound. A new split target starts from the canonical
empty synopsis, and movement expands only the actual target. Synopsis and entry
changes commit atomically.

## 6. Search algorithm

One consistent snapshot performs:

1. Active Manifest validation and request compilation;
2. bounded Tree Key enumeration;
3. deterministic best-first traversal of every eligible tree under one global
   Partition budget;
4. synopsis pruning and exact leaf predicate evaluation;
5. bounded RaBitQ candidate selection;
6. original Vector Record batch loading and exact reranking;
7. deterministic top-k ordering and budget report construction.

Traversal is a level-scaled beam. The leaf-level base beam defaults to 32;
moving one level toward the root divides it by two with minimum one. Eligible
trees advance fairly. Ties use Tree Key, Partition Key, then Record ID.
Partition, Leaf Entry, rerank, and optional RaBitQ-overlap bounds are charged
before corresponding work. No speculative read-ahead occurs beyond a budget.

For a leaf with `n` eligible entries, checked arithmetic computes
`r = min(n, max(2*k, 64))`. The leaf keeps the rough top `r`, uses the r-th
smallest upper endpoint as its overlap threshold, includes entries whose lower
endpoint overlaps it, and caps the set at `min(4*r, remaining rerank budget)` by
lower bound, rough distance, and Record ID. After merging leaf sets, search uses
the global kth-smallest upper endpoint (positive infinity when fewer than `k`
exist), retains overlapping candidates, truncates by rough distance and Record
ID to the remaining rerank budget, then loads original vectors. Any local
overlap truncation sets `rabitq_overlap_truncated`.

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

SearchOptions overrides nonzero bounds for Tree Keys, partitions, leaf entries,
rerank candidates, and RaBitQ overlap candidates within hard caps. Defaults are
process-local and benchmark-tunable. One successful response reports usage and
each dimension that prevented pending work.

The API deliberately has no `complete` boolean: ANN search is approximate even
when no logical budget is exhausted. It also has no quality score or
continuation token. Callers needing more work resubmit with higher budgets.

## 8. Partition cache

The Runtime shares byte-bounded caches of decoded internal and leaf search data.
A key contains Logical Index ID, canonical Tree Key, Partition Key, and kind.
Every entry contains Header cache epoch. Search reads the Header from its own
snapshot and may reuse cached data only when epoch and kind match; otherwise it
loads and decodes the partition from that snapshot.

Entries are immutable and never pinned. Concurrent misses may duplicate work and
race to publish an equal or newer epoch; there is no waiter/cancellation state.
Cache insertion may be skipped when an item exceeds capacity. Internal and leaf
byte shares may be configured, but the eviction policy is an internal
benchmark-driven detail, not a persistent or public compatibility contract.
Corruption is never cached.

## 9. Validation

- Golden and property tests cover signed-code encoding, error rounding,
  conservative intervals, zero vectors, malformed payloads, and exact distance.
- Predicate properties compare compiled evaluation and synopsis truth sets with
  a simple SQL three-valued oracle over boundary values and NULLs.
- Deterministic tests cover Tree Key range widening, scan budgets, traversal
  ordering, every exhaustion dimension, overlap truncation, and intermediate
  topology traversal.
- Cache histories prove snapshot epoch validation and safe concurrent fills
  without asserting a particular eviction algorithm.
- Recall and performance benchmarks compare brute force on fixed public and
  synthetic distributions and report all budget use; they establish baselines,
  not an unsupported v1 SLA.
