# Tree quality options for Cohere1M

Date: 2026-10-03. Scope: independent literature and contract assessment, not a benchmark result. Objective: improve measured QPS at recall@k >= 0.90; a smaller beam by itself is insufficient evidence. Resource growth and modestly slower import are acceptable. The implementation and experiments are owned by the main task.

The subsequent [current-code evaluation](tree-quality-evaluation-2026-10-03.md) records the measured disposition of these hypotheses. The initial recommendations below are research priorities, not the final implementation recommendation.

## Recommendation before experiments

Start with bounded nearest-assignment split training and deterministic multiple starts. They address a discoverable mismatch between training and placement while preserving exact membership, snapshot search, and the expose-then-drain protocol. Treat representative import ordering as a second diagnostic for early immutable routing decisions. Boundary replication remains an option if these fail, but requires a larger membership design and a routing-quality experiment before implementation. None is yet demonstrated to be the best solution for KTANN.

## Current KTANN contract and relevant mechanism

Sources: [CONTEXT](../../CONTEXT.md), [ADR 0001](../adr/0001-exact-leaf-membership.md), [ADR 0014](../adr/0014-expose-then-drain-splits.md), [ADR 0015](../adr/0015-incremental-binary-kmeans-tree.md), [maintenance design](../design/maintenance.md), [search design](../design/search.md), and [training implementation](../../src/maintenance/training.rs).

Each committed Vector Record has one Record Location and one corresponding Leaf Entry. Foreground Mutation changes these, exact counts, and Synopses atomically. Each search uses one backend snapshot. Training takes a complete consistent source snapshot, uses canonical ordering and farthest-pair initialization, and runs at most ten Lloyd rounds. Every training round assigns exactly floor(n/2) entries to one cluster. Non-root centroids are immutable; internal training operates on Child Entry centroids rather than the full descendant record population.

Actual drain placement uses the nearer trained centroid, with minimum-size reservations when needed. New inserts use nearest routing. Thus the balanced training assignment is not the final persisted assignment. A high balanced-training score alone need not predict final leaf compactness. Internal centroid training also represents child partitions equally despite potentially unequal descendant populations; this is a hypothesis for future diagnosis, not evidence that weighting would improve search.

Traversal admits partitions through a level-scaled beam. Missing a high-level branch can eliminate many relevant leaves. Leaf quality, routing quality, quantized candidate selection, and exact rerank truncation must be measured separately. More compact leaves cannot recover a branch that was never visited.

## What the primary papers establish

### SPANN

[SPANN paper, §§3.2.1–3.2.3 and 4.2](https://arxiv.org/pdf/2111.08566) combines hierarchical balanced clustering, closure assignment, and query-aware posting pruning. Its balance objective adds a tunable size-variance penalty to clustering error; it is not an exact 50/50 binary constraint. Posting centroids have an in-memory navigation index. Closure admits x to an additional centroid c when its distance is within (1 + epsilon) times the nearest-centroid distance. RNG then removes redundant nearby representatives. Experiments choose a maximum of eight replicas; this is not a reported average storage multiplier of eight. Figure 11 studies caps of 1, 4, 8, and 10. The paper’s construction sees the dataset globally, unlike KTANN’s first-insert creation and incremental local splits.

### SPFresh

[SPFresh paper, §§3.1–3.3, 4.2, and 5.1](https://arxiv.org/html/2410.14452v1) inherits SPANN’s boundary replication and SPTAG centroid navigation. Its central contribution is LIRE: split/merge plus nearby reassignment to maintain nearest-partition assignment as data changes. Its reassignment conditions assume Euclidean geometry and previously valid nearest assignment. The implementation uses an in-memory per-vector version map and disk versions to invalidate old replicas, then garbage-collects them. Evaluation updates already populated base indexes; it does not establish equivalent quality for KTANN’s empty-to-1M incremental construction. The paper does not establish a Cohere1M replication ratio or KTANN QPS benefit.

### Initialization evidence

[Arthur and Vassilvitskii, k-means++](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) establishes the importance of initialization and an expected approximation guarantee for its randomized seeding in the ordinary k-means objective. That guarantee does not automatically transfer to bounded balance, spherical/cosine training, a deterministic multi-start variant, or ANN recall. This is motivation for an ablation, not a predicted recall improvement.

## Applicability and limitations

The following assessments are inferences from the papers and KTANN contracts, not externally measured results.

| Option | Why it might help | Main limitation / cost |
| --- | --- | --- |
| Bounded nearest assignment, initially 25/75 | Train toward actual nearest placement while preventing degenerate children; remove forced 50/50 allocations far across a natural boundary | Relaxed balance can increase height and partition variance, accelerate repeat splits, and hurt tail latency/import. Minimum drain reservations still create some mismatch. A 25% bound is an experiment, not a paper-derived optimum. |
| Deterministic multiple starts | Farthest-pair seeding may be dominated by extremes; choose a better local routing model from several seeds | Extra CPU scales with starts and rounds; use the same loaded snapshot. Better objective does not guarantee better held-out query routing, especially at internal levels. Preserve deterministic tie-breaking. |
| More Lloyd rounds | Improve insufficiently converged splits | Cannot fix bad seed basins, a mismatched objective, or ancestor routing. Count convergence before paying this cost everywhere. |
| Smaller leaf threshold | Reduce vectors scanned per admitted leaf | Adds centroids, leaves, maintenance, and traversal work. More boundaries can reduce recall at the same beam. Evaluate equal scanned-vector budgets and equal recall, not equal beam alone. |
| Representative records imported first | Make early immutable centers see broader support | Changes order, not later immutable-center drift. Random samples may miss rare modes. Requires source access/reordering for a representative sample; an arbitrary prefix is insufficient. |
| Cross-leaf closure/RNG replication | A boundary record can survive admission of another representative leaf | Must locate useful cross-branch representatives. Extra memberships consume scan/candidate budgets, can trigger more splits, and require exact deletion/upsert/synopsis handling. Simple sibling duplication may repeatedly return the same candidates without recovering missing ancestors. |
| Split-time corrective migration | Repair globally misplaced records after local splits | SPFresh’s NPA reasoning is about nearest global postings; KTANN hierarchical routing does not guarantee that precondition. Finding useful neighboring leaves and atomically relocating records can dominate maintenance. Historical failure is evidence to inspect, not universal disproof. |
| Flat leaf-centroid navigation | Avoid errors compounded at ancestors | Material routing/index design expansion; centroid memory, updates, and snapshot validation must be solved. SPANN replication gains cannot be assumed without its navigation geometry. |

Replication would require distinguishing an authoritative Record Location from additional searchable memberships, or replacing it with a set. Foreground Mutation must atomically remove old projections and install new ones; structural moves, conservative Synopses, deduplication before rerank limits, and offline verification must all follow the new model. SPFresh’s asynchronous stale-version filtering cannot be transplanted while claiming the present exact-membership contract still holds. There is no compatibility requirement, but a deliberate contract change remains necessary.

The RNG paper formula and epsilon admission are geometric rules over candidate postings. KTANN cosine routing uses negated dot product; directly multiplying that signed score by (1 + epsilon) changes the intended criterion. Any replication experiment must specify a suitable nonnegative cosine/chord distance and test it separately from L2.

## Minimum warm-start experiment without a bulk builder

For the known benchmark source, deterministically select actual records across the complete dataset, import them through ordinary upserts, let their normal maintenance settle, then import every remaining record exactly once. For example, compare no prelude with a 64K representative prelude, retaining original IDs and excluding those IDs from the subsequent pass. Every committed record is immediately searchable and uses the ordinary atomic membership protocol. No unpublished topology, synthetic placeholders, whole-import transaction, or new bulk-generation API is required.

This measures whether early training support is important. The prelude still builds an incremental tree, so it is not SPANN-style global clustering. For a one-pass online source whose future records cannot be inspected, the same representative sample cannot be promised. A bounded buffered prefix can be reordered before commit, but delays visibility of uncommitted records and has no guarantee of representing future data; that production behavior needs an explicit import contract. Do not silently transfer benchmark-only random access to the online API.

## Two falsifiable experiments

1. **Training ablation on full Cohere1M.** Baseline; 25/75 bounded nearest assignment with the same seed; then the same bounded assignment with a small deterministic seed set, retaining the original seed as a candidate. Use the same source order, metric, backend, import settings, and held-out query set. Record the minimum measured beam reaching recall >= 0.90, QPS and tail latency there, scanned unique records, admitted/internal partitions, exact-rerank truncation, tree depth, leaf-size distribution, import time, maintenance backlog, and storage. Objective reduction without equal-recall QPS improvement rejects the practical hypothesis. If no beam reaches the target, report failure rather than extrapolate. Repeat only contenders to distinguish host noise from a material gain.

2. **Early-support and routing diagnosis.** Compare baseline import order against the representative prelude using the best training variant. Separately, as an offline diagnostic over the same committed tree, rank all leaf centroids and estimate true-neighbor coverage by selected leaves; compare with hierarchical beam coverage at matched scan budgets. Offline global routing is not a production QPS claim. A large coverage gap would prioritize navigation/ancestor modeling over copying within siblings. A small gap with many true neighbors just across selected leaf boundaries would justify a bounded cross-leaf replication prototype. Measure representative prelude import cost and end-state quality; reject it if only early partial-index recall improves.

Historical local reports indicate several centroid-refresh, dual-membership, RNG, and sparse-replication attempts did not yield sufficient benefit. Their raw data is incomplete, and this note does not claim to have reproduced them. The present experiments should publish current executable configurations and raw recall/latency curves before choosing a durable architecture.

## Follow-up: current evidence and raw means with squared L2

The main task reports the full-Cohere1M bounded-assignment comparison below. These are supplied experiment results; this research task did not reproduce them. The candidate was reverted because a material improvement was not established. Four-start balanced training remains under evaluation at the time of this addition.

| beam | variant | recall | entries | CPU/query |
| --- | --- | --- | --- | --- |
| 112 | baseline | 0.91157 | 40,999 | 27.45 ms |
| 112 | 25/75 bounded assignment | 0.91357 | 40,595 | 27.22 ms |

The next proposed independent candidate keeps cosine record preprocessing (normalize then rotate), retains arithmetic means without renormalizing centroids, and uses squared Euclidean routing throughout scalar/batched paths. Exact cosine results and RaBitQ coding are unchanged. This changes the approximate routing model; it is not an equivalence-preserving cosine refactor.

The main task confirms the concrete prototype removes the input-centroid normalization loop and the cluster-mean output normalization; both `routing_distance` and `interleaved_distances` use squared L2 for cosine routing. `preprocess` and `exact_distance` remain unchanged. Internal training retains each raw Child Entry centroid with equal weight; it does not add descendant-population weights. The experiment therefore measures that specific raw-representative model, not the direction-normalized hybrid discussed below.

[Faiss's first-party metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) gives the normalized-vector identity and distinguishes inner product from cosine when norms differ. For unit q and unit c, squared L2 is 2 - 2 q·c, so ranking agrees with cosine. More generally, for a fixed query the ranking agrees with negative dot only when candidate centroid norms are equal. An orthonormal rotation preserves these mathematical identities; finite-f32 preprocessing does not promise bit-for-bit norm equality.

For a raw mean m of unit records x_i, direct expansion gives:

```
||q - m||² = ||q||² + ||m||² - 2 q·m
mean cosine_distance(q, x_i) = 1 - q·m             [unit q, unit x_i]
mean ||q - x_i||² = ||q - m||² + mean ||x_i - m||²
mean ||x_i - m||² = 1 - ||m||²                    [unit x_i]
```

Thus raw-mean negative-dot ranks by average cosine distance to a cluster. Raw-mean L2 adds a centroid-norm offset, equivalently subtracting within-cluster variance from average squared distance. Low-norm, dispersed clusters can become more attractive. Example: q=(1,0), m_A=(0.8,0.6), m_B=(0.75,0). Negative-dot/average-cosine prefers A, but squared L2 prefers B (0.4 versus 0.0625). Both are realizable means of unit vectors. This demonstrates the exact boundary of the claimed equivalence and a possible failure mode. The zero centroid is valid for L2; it has no cosine direction.

For **strict balanced binary training**, comparing distances to fixed raw centers yields:

```
d_left(x) - d_right(x)
    = ||m_left||² - ||m_right||² - 2 x·(m_left - m_right)
```

The centroid-norm difference is constant across entries. Selecting exactly half by this difference therefore gives the same balanced assignment as raw-mean negative-dot for the same centers, ignoring floating-point/tie effects. The new norm offset changes the **nearest** threshold used by draining, inserts, and beam ranking. On unit leaf inputs, initial farthest choices relative to a fixed center also retain negative-dot ordering. Multi-start selection by the sum of nearest squared distances, unequal-norm internal inputs, or numerical tie handling can still make the trained model differ. Consequently historical raw-mean/negative-dot measurements do not settle raw-mean/L2 behavior, but some training paths may be identical.

The current training implementation normalizes all input entries for cosine, including Child Entry centroids, and normalizes each cluster mean. The candidate must make the intended internal model explicit. Keeping **raw child centers** as training inputs preserves their norms and trains ordinary Euclidean clustering of those representatives. Renormalizing those children first discards their concentration and trains directions, then uses raw output means for L2 routing; that is a different hybrid. Neither automatically represents every descendant record equally: an unweighted arithmetic mean of child centroids weights each child equally, whereas a descendant mean needs child-population weights. Retaining raw children alone cannot solve unequal descendant populations or immutable-center drift. All training, split placement, foreground routing, merge-target selection, and traversal calls must share the chosen routing metric; scalar/batched equality and finite-value validation remain necessary.

The raw-mean/L2 candidate is mathematically coherent as Euclidean clustering of normalized records. It can plausibly improve routing fit, or over-admit diffuse clusters; only equal-recall measurements can decide. Changing persistent centroid semantics also requires updating the corresponding design/protocol contracts, even though no released compatibility is needed.

## Follow-up: a bounded query-aware pruning experiment

[SPANN §3.2.3, equation (3), and Figure 12](https://arxiv.org/pdf/2111.08566) prune a query's candidate postings when centroid distance exceeds (1 + epsilon_2) times its nearest candidate-centroid distance. The paper reports an ablation with lower latency and preserved recall. This evidence concerns SPANN's complete design, including closure replication and its navigation index; it is not a KTANN guarantee.

If both multi-start and raw-mean/L2 fail to improve same-recall QPS, query-aware **leaf** pruning is a more bounded next experiment than permanent replication: it can reuse admitted leaf-centroid scores, leave exact membership untouched, avoid write amplification, and be tested from recorded query traces before modifying persistent formats. This is a judgment about experiment cost and isolation, not a claim that its performance potential is larger.

Apply the threshold after all admitted parents have contributed leaf candidates, using the closest candidate within each Tree Key. Keep ancestor traversal unchanged for this experiment. Every unpruned leaf must still use ordinary exact filtering and scoring. A root that is itself a leaf has no competing centroid and should be scanned normally. Use a wider fixed maximum beam if necessary, then adapt the final leaves per query; pruning the existing beam can only remove opportunities and cannot recover an excluded ancestor. Maintain a small minimum number of leaves for degenerate nearest distance, with an explicit policy rather than an accidental divide-by-zero behavior.

The present cosine score is negative dot, so multiplying it by 1 + epsilon is invalid. For normalized centroids, compute nonnegative chord distance from the actual query/centroid norms and dot product; special-case neither zero centroids nor f32 norm drift by pretending both norms equal one. For raw means, squared L2 is already nonnegative. If the implementation thresholds squared distances, an epsilon on Euclidean distance becomes a factor of (1 + epsilon)²; alternatively define a separate squared-distance threshold parameter explicitly. Reuse computed scores where practical and measure any extra centroid arithmetic. This is a heuristic pruning rule, not a certified lower bound on record distances.

First record per-query sorted leaf distances, sizes, true-neighbor leaf coverage, and candidate/rerank truncation at several maximum beams. Sweep distance-ratio thresholds offline, then benchmark only thresholds whose full query set still reaches recall >= 0.90. Report QPS, p95/p99, mean/p95 scanned entries, leaves skipped, routing overhead, and difficult-query recall. Dense high-dimensional centroid distances may be too similar for the threshold to remove useful work; sparse easy queries may permit large savings. Reject this experiment if achieving target recall leaves entry scans essentially unchanged, if difficult queries regress materially, or if routing overhead consumes the saved leaf work. Do not infer wall-clock QPS from offline saved entries.
