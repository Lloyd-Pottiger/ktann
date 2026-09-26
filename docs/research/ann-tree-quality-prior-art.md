# ANN tree quality and fixed-recall efficiency: applicable prior art

Research date: 2026-09-26. Code inspected at
`3a62957c458a87b5b708af58ea1975c679e54ac2`. This note proposes experiments;
it does not establish the cause of Cohere1M's observed recall or claim a
measured speedup. Issue history and current benchmark reproduction are separate
inputs to the diagnosis.

Do not mix `k=10` and `k=100` evidence. The coordinating investigation reports
the current VectorDBBench fixture at `k=100`, beam 384, recall 0.9293; historical
native tree-hit figures at `k=10` answer a different question. Attribute losses
using the same `k` as the target workload before selecting a fix.

## Contract and current implementation

KTANN has one authoritative Leaf Entry per Vector Record, disjoint Tree Keys,
online searchable split states, and one consistent snapshot per search.
Those are constraints on the candidate techniques, not optional benchmark
settings. Relevant decisions are [exact membership](../adr/0001-exact-leaf-membership.md),
[sharded trees](../adr/0004-sharded-kmeans-forest.md),
[expose-then-drain](../adr/0014-expose-then-drain-splits.md), and
[incremental binary training](../adr/0015-incremental-binary-kmeans-tree.md).

Observed implementation facts:

- [Training](../../src/maintenance/training.rs) loads the complete source
  snapshot, including original Vector Records for a leaf. It uses deterministic
  farthest-pair initialization, at most ten Lloyd rounds, and exactly half of
  the entries on each side in each round. Cosine centroids are normalized.
- Only the two trained centroids survive into the split protocol. The balanced
  labels do not. [Drain placement](../../src/maintenance/split.rs) normally uses
  nearest centroid, reserving the last necessary entries to satisfy each
  target's minimum occupancy when possible. Foreground routing uses nearest
  centroids. Non-root centroids are immutable under the current decision.
- [Traversal](../../src/search/traverse.rs) selects a beam at each tree level;
  the leaf beam halves toward the root. Leaf candidate approximation and global
  exact reranking introduce separate recall losses. [ADR 0011](../adr/0011-bounded-deterministic-approximate-search.md)
  fixes exact-rerank sizing at `max(64, k + ceil(k/2))`, within runtime limits,
  and bounds RaBitQ overlap selection. A larger traversal beam therefore does
  not directly isolate tree quality.

The important mathematical distinction is between training and placement.
Let `delta(x) = distance(x, left) - distance(x, right)`. Half-assignment places
the lowest half of the deltas on the left; nearest-centroid assignment places
negative deltas on the left. These boundaries coincide only when the median
delta lies at zero, apart from ties. Recomputing centroids does not enforce that
condition. Minimum-occupancy reservation introduces another, smaller placement
constraint. Thus balanced training does **not** prove balanced final partitions
or that centroids represent the final membership. This is a hypothesis about
quality, not a correctness defect: membership remains authoritative.

## What transfers from published systems

### SPANN: balance alone is not its recall mechanism

SPANN combines hierarchical balanced clustering with boundary-vector
replication, query-aware posting-list pruning, and an in-memory centroid
index. Its high recall is not evidence that balanced binary clustering alone
will provide the same outcome. Boundary replication is incompatible with
KTANN's exact one-leaf membership. Query-dependent probing is transferable as
a hypothesis, but centroid distance by itself is not a safe lower bound on
every member's distance. [SPANN, NeurIPS 2021](https://proceedings.nips.cc/paper/2021/file/299dc35e747eb77177d9cea10a802da2-Paper.pdf).

KTANN-specific inference: spend additional probes on ambiguous sibling or
nearby partitions rather than copying records. Compare this with the present
fixed level beam at equal visited entries and equal recall. A valid experiment
must include clustered queries that strongly favor one branch and boundary
queries whose neighbors span branches. No early-stop rule should be described
as exact unless it has a valid bound under the configured metric.

### SPFresh: splitting can invalidate nearby assignments

SPFresh identifies nearest partition assignment as a desirable geometric
property. Splitting a partition changes the nearest centroid both for its own
vectors and for some vectors in nearby partitions. LIRE combines local splits
with nearby reassignment. Its implementation also uses replicas, version maps,
and garbage collection; these are not KTANN's transaction model.
[SPFresh, sections 3–4, SOSP 2023](https://arxiv.org/html/2410.14452v1).

KTANN-specific inference: immutable routing centroids and source-local split
movement can accumulate geometric errors across import. First measure this:
for sampled records, compare the assigned leaf with the nearest centroid among
all live leaves, then with a bounded neighboring-leaf set. Also measure centroid
drift against the current membership mean. This is an offline diagnostic, not
a proposed global foreground scan.

If this explains the recall gap, a bounded local reassignment experiment could
move each record atomically with its Record Location, Leaf Entry, counts and
synopses. Every committed state must remain searchable, including after retry,
crash, or overlapping split. Searchable topology alone is insufficient: moves
must also leave useful routing models. This is substantially larger than
changing the trainer and would require a design for neighbor discovery,
revalidation, bounded work, and maintenance convergence before implementation.
The paper's convergence arguments cannot simply be reused for KTANN's
different split/merge and transactional rules.

### Careful seeding: isolate initialization from assignment policy

The k-means++ paper proposes distance-weighted randomized seeding and analyzes
the squared-Euclidean clustering objective. Its guarantee is not a recall
guarantee, nor automatically a guarantee for normalized cosine clustering or
capacity-constrained assignments.
[Arthur and Vassilvitskii, SODA 2007](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf).

KTANN-specific inference: compare existing farthest-pair seeding with a small
number of reproducible seeds, selected by the objective of the **actual
placement rule**. Derive a seed from canonical source contents if deterministic
training remains required. Outlier-sensitive farthest-pair selection may be
poor on some distributions; that must be measured, not assumed. Start by
holding initialization constant while changing assignment, so a combined
variant does not hide the effective cause. Additional restarts increase import
CPU and may be a poor trade unless they reduce subsequent splits or search
work enough to justify the cost.

### ScaNN: separate partition, scoring, and reranking losses

ScaNN documents distinct partition selection, approximate scoring, and optional
rescoring stages, with independently tunable partition probes and rescore
counts. This is a useful decomposition, rather than a suggestion to copy its
parameter defaults. [Official algorithms and configuration](https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md).

Its anisotropic quantization research optimizes quantization error relevant to
inner-product ranking, instead of treating all reconstruction errors equally.
Changing KTANN's RaBitQ7 codec is a separate intervention with storage and
training costs, not the first response to an unexplained recall deficit.
[Guo et al., ICML 2020](https://proceedings.mlr.press/v119/guo20h.html).

KTANN-specific inference: run an offline oracle on sampled production queries:

1. Save the leaves selected by normal traversal; exact-score every record in
   those leaves. This measures the ceiling attainable without changing routing.
2. Run current approximate selection on exactly those same leaves and record
   which ground-truth neighbors survive leaf selection and global selection.
3. Compare final recall with the oracle and record overlap truncation and every
   exhausted budget. Sweep a diagnostic rerank count separately from the beam.

If the first stage already misses neighbors, larger reranking cannot recover
them. If its recall is high but final recall is low, improve selection before
paying for a larger tree beam. Increasing a diagnostic limit is not a proposal
to silently change the public resource contract.

### DiskANN: separate algorithmic breadth from I/O concurrency

DiskANN's SSD search batches a small number of neighborhood fetches to reduce
dependent I/O rounds, caches frequently visited nodes, and uses compressed
distances with full-precision reranking. Its search-list size and I/O beam serve
different roles. Its sector layout can fetch full vectors with neighborhoods;
that advantage does not automatically transfer to transactional KV reads.
[DiskANN, sections 3.3–3.5, NeurIPS 2019](https://suhasjs.github.io/files/diskann_neurips19.pdf).

KTANN-specific inference: keep candidate breadth fixed while measuring the
number of dependent KV round trips, batch sizes, CPU scoring, and exact-record
fetches. Sweep bounded storage concurrency independently of the recall beam.
Batch only reads whose dependencies are already satisfied, preserving the
same snapshot and deterministic logical admission. A warmer Partition Cache
still needs snapshot validation, so report validation reads as well as cache
hit rate. On a CPU-bound warm workload, more read concurrency may do nothing
or worsen scheduling. A graph rewrite would change the topology and mutation
contract and is outside this experiment set.

## Prioritized experiments

| Order | Experiment | Evidence needed to retain the idea |
| --- | --- | --- |
| 1 | Fixed saved topology: traversal oracle, current RaBitQ, and diagnostic rerank sweeps | Attribute lost true neighbors to routing versus candidate selection; identify budget truncation |
| 2 | Replay source snapshots: current balanced trainer versus nearest-centroid Lloyd, with identical seeds | Lower actual-placement distortion and label mismatch without unacceptable occupancy skew or split/merge churn |
| 3 | Full online import using the best trainer candidate | Better fixed-recall QPS and settled import time across input orders; acceptable maintenance work and peak memory |
| 4 | Same topology: alternative per-level beam allocation and bounded adaptive probing | Fewer visited entries and lower latency at the same recall, including difficult boundary queries |
| 5 | Same logical search work: bounded KV batching/concurrency and kernel profiling | Improved matched-recall latency/QPS with measured, reasonable memory and backend costs; cache-dependent gains are acceptable |
| 6 | Reassignment or centroid-refresh prototype, only if drift diagnostics justify it | Better recall that survives online mutations, with bounded additional maintenance and a coherent transaction protocol |

These are discriminating experiments, not claims that the project has never
tried related techniques. Existing local artifacts include
`diagnose-cohere-cosine-spherical-rerank1000.json`,
`diagnose-cohere-cosine-normalized-centroid-2026-08-31.json`,
`diagnose-cohere-same-beam-all-levels.json`, and
`diagnose-cohere-online-100k-rounds32-2026-09-01.json` under
`.benchmark-data/results/`. Reconcile their source revision, dirty changes,
`k`, budgets, and saturation before repeating them. Cosine normalization is
already present in current training; wider internal beams and additional Lloyd
rounds have no established new benefit in this note.

For experiment 2, unconstrained training is a diagnostic candidate rather than
an unconditional production replacement. Identical or near-identical vectors
can produce empty or tiny clusters, and skew can create cascaded splits.
The existing minimum-occupancy protection addresses a real convergence risk.
Test duplicate-heavy, highly skewed, and angularly concentrated sources in
addition to Cohere; measure post-drain assignments, not just the trainer's
objective. Any adopted change to fixed balanced training must explicitly
revise ADR 0015 and its tests.

Sampled or capped training is worth considering only after measuring full-source
loading and peak overshoot. A uniform bounded reservoir over the complete scan
reduces training memory/compute but does not by itself reduce scan I/O; sampling
the first page introduces key-order bias. Avoid promising import speedup from
sampling before locating its cost. Likewise, optimizing admission for raw
upserts/s is insufficient if it leaves more oversize sources and maintenance
work or worsens the resulting tree.

A trained empty-index bulk build is another useful **offline quality ceiling**:
compare a tree trained with knowledge of the full distribution against the
incremental tree at comparable occupancy. This does not make it an authorized
implementation shortcut. Production bulk-build publication would require an
explicit change to ADR 0015's no-bulk-generation choice, manifest lifecycle,
interrupted-build handling, and the visibility contract for concurrent readers
and mutations. Establish whether that ceiling closes a meaningful gap before
proposing those architectural changes.

## Evaluation requirements

Acceptance follows the user's clarified priorities: complete import, tree
quality, matched-recall QPS and latency, and reasonable, correctly used
resources. Modest memory growth and improvements that depend on an enabled,
warm cache are acceptable trade-offs. Characterize cold-cache and cache-disabled
behavior, including warm-up time and resource costs; lack of improvement in
those modes is not by itself a veto. Report the supported operating conditions
of a gain rather than claiming universal benefit. Correctness, snapshot
validation, complete import, and bounded resource use remain requirements;
unbounded growth or material reliability regressions are not authorized by
this allowance.

Use the same metric, preprocessing, `k`, queries, ground truth, adapter,
resource caps, build mode, and machine load. Split query tuning and held-out
evaluation. Keep one saved topology when isolating search changes; rebuild
when testing training. For build variants, include original and deterministic
shuffled import orders and record maintenance/intermediate-state counts at
measurement time. Report import admission time and time to the chosen settled
state separately; KTANN's demand-driven maintenance does not promise automatic
global convergence.

Report recall/latency/QPS curves, not a single beam number. At recall targets
0.90 and 0.95, include p50/p95/p99 latency, visited partitions and entries,
exact records, truncation, backend calls/bytes, CPU, cache memory, and peak RSS.
For import include conflicts/retries, records reloaded for training, split and
merge counts, drain moves, write bytes, and final occupancy/depth distributions.
Use the production backend for retained performance claims; an in-memory oracle
isolates algorithmic behavior but does not predict FoundationDB QPS.

No published performance number above is a KTANN target. The first decision
should be driven by where the present implementation loses neighbors and
spends time; the sources supply candidate mechanisms and failure modes.
