# Current ANN implementation audit

Date: 2026-09-26. Scope: source audit at HEAD `3a62957`; no implementation changes or new benchmark runs. This note identifies causal hypotheses, not measured performance improvements. See [the integrated report](ann-performance-research-2026-09-26.md) for issue history and benchmark attribution; historical k=10 evidence must not be substituted for the current k=100 workload. The user accepts modest memory growth and improvements limited to cached workloads: measure complementary costs without requiring every workload to improve.

## Main finding

There are two separable opportunities: improve which leaves contain and expose true neighbors, and reduce work/round trips for the same leaf and candidate set. Increasing beam can compensate for routing loss but increases partition metadata reads and leaf scoring. A high beam alone does not establish which of centroid age, split assignment, early-level pruning, or candidate truncation caused the loss.

## Source-backed behavior and hypotheses

### 1. Balanced training does not imply balanced persisted partitions

**Evidence.** Split training normalizes cosine inputs, uses deterministic farthest-pair seeds, and runs at most ten balanced Lloyd rounds. Every round partitions entries by distance difference and assigns exactly half the snapshot to the left cluster; only two means survive training ([training.rs](../../src/maintenance/training.rs#L260), [assignment](../../src/maintenance/training.rs#L338)). Draining independently picks the nearer of the two persisted centroids ([routing.rs](../../src/maintenance/routing.rs#L1155)). The balanced assignment mask and its separating distance-difference threshold are not the drain contract. Internal training weights each child centroid as one entry, without descendant cardinality ([loader](../../src/maintenance/training.rs#L210)).

**Inference.** A skewed source can produce balanced training means whose ordinary Voronoi assignment is imbalanced. This could create oversized targets, repeated splits, extra movement, and a poorer routing model. It is not a membership correctness bug: training deliberately produces a routing model, and concurrent writes need not match its snapshot.

**Disconfirmation.** For sampled large-data splits, compare final training assignment to nearest-centroid assignment on the identical snapshot: disagreement fraction, actual target sizes, angular distortion/SSE, and subsequent split count. Low disagreement and no relationship to low-recall query paths reject this as a primary cause. Compare balanced training, ordinary Lloyd, and threshold-aware routing offline before changing production topology. Persisting the balanced mask is insufficient under concurrent writes; any new rule must define new arrivals and search admission too.

### 2. Centroid immutability can make early online history matter

**Evidence.** Immutable non-root centroids are an explicit accepted design, not a missing incidental update ([ADR 0015](../adr/0015-incremental-binary-kmeans-tree.md), [domain definition](../../CONTEXT.md#L91)). The internal training loader reads immutable child projections. Writes use configurable beam routing (default eight), and choose one terminal leaf; batch routing is already grouped and preprocesses vectors once ([routing.rs](../../src/maintenance/routing.rs#L199), [mutation.rs](../../src/maintenance/mutation.rs#L14), [defaults](../../src/api/config.rs#L23)).

**Inference.** Early small-sample centers and later nonstationary import order may make current members poorly represented, especially when an internal centroid represents many changing descendants. Increasing write beam may find better current leaves but cannot by itself refresh their representation. This is a hypothesis, not proof that recentering wins overall.

**Disconfirmation.** Measure persisted-center versus current-mean drift, descendant distortion, record-to-best-leaf assignment regret, and true-neighbor ancestor survival. Compare fixed-order, seeded-shuffled, and representative-sample-first imports at equal work/settings. Run an offline centroid-refit search ablation on an immutable snapshot. Little recall change rejects recentering as the main remedy. A real refresh would change the accepted ADR and must atomically maintain incoming child projections, affected cache epochs, and search/write consistency; do not simply overwrite one centroid value.

### 3. Search beam is a per-level partition beam, not an arbitrary candidate count

**Evidence.** The default leaf beam is 128. It halves toward the root (minimum one), and each tree/level chooses a global next-level beam across parents; roots and required root-split target injections bypass pruning ([traversal contract](../../src/search/traverse.rs#L16), [promotion](../../src/search/traverse.rs#L750)). Beam pruning does not set a resource-exhaustion flag. Stable internal partitions can contain many children; this is not simply a binary decision tree at query time.

**Inference.** Early-level pruning can lose an ancestor even when a larger leaf beam would contain the desired leaf if it were reachable. Root depth, fanout, and ancestor survival are necessary to interpret the user's beam=384 result.

**Disconfirmation.** Trace exact ground-truth record locations through ancestors and report survival by level, then compare the existing schedule with independently varied internal/leaf widths. If loss occurs only inside admitted leaves, topology/beam is not the responsible stage. Charge partitions, centroid distance evaluations, leaf entries, latency, and memory for all alternatives.

### 4. Candidate caps can explain a separate recall ceiling

**Evidence.** Each leaf uses `r=min(n,max(2*k,64))`, keeps candidates whose lower interval endpoint overlaps the r-th upper endpoint, and caps survivors at `min(4*r, exact_rerank_budget)` ([selection.rs](../../src/search/rabitq/selection.rs#L84)). Global selection uses the k-th upper endpoint and caps by rough distance ([global selection](../../src/search/rabitq/selection.rs#L124)). Reranking fetches record groups in sequential chunks of 64, validates exact record/location/field correspondence, and computes exact distances ([rerank.rs](../../src/search/rerank.rs#L218)).

**Inference.** A small global exact-rerank budget can discard true neighbors even after successful traversal. A truncation flag means candidates were removed, not that a true top-k item was removed. Larger beam can add candidates without resolving this bottleneck. For k=100, a 150-candidate global cap deserves its own sweep; old k=10 traversal-only loss cannot answer it.

**Disconfirmation.** On the same immutable tree/query set, report recall after exact scoring all visited leaves, after local selection, after global selection, and final ranking. Vary cap independently of beam. Compare candidate survival and stage time; if exact-all-visited recall equals final recall, rerank budget is not causing loss. Preserve membership checks and report added record bytes/RPCs when increasing the cap.

### 5. Warm search still has sequential leaf metadata reads

**Evidence.** Non-root internal header reads are already batched under the remaining partition budget ([traverse.rs](../../src/search/traverse.rs#L398)). Leaf visits remain demand-driven, with one header read (and synopsis only when needed) before the body/cache lookup ([leaf visit](../../src/search/traverse.rs#L452)). Cache hits validate against the header's epoch; misses scan a whole body page by page ([cache.rs](../../src/search/cache.rs#L525)). FoundationDB `batch_get` already launches independent point reads concurrently ([backend.rs](../../ktann-foundationdb/src/backend.rs#L423)).

**Inference.** At hundreds of visited leaves, sequential metadata rounds may dominate warm-cache latency even though bodies are cached and FDB supports concurrent requests. A bounded leaf-header prefetch window is a plausible experiment, not an established win. Cold body scans, exact-record fetches, cache contention, or CPU scoring may instead dominate.

**Disconfirmation.** Separate warm/cold cache, one/multiple query concurrency, FDB latency/bytes, and per-stage CPU/wall time. Compare a bounded prefetch prototype with identical outputs and budget usage. Existing demand-driven semantics deliberately avoid reading later leaves after entry-budget exhaustion: preserving that physical-I/O property may require a restricted full-budget path or a deliberate contract decision. Do not hide speculative work behind unchanged logical counters.

### 6. Import has both online algorithm cost and admission-lifetime cost

**Evidence.** Import uses ordinary searchable batch mutation, not a bulk-built tree. Native import concurrency starts at one, grows on clean completions, and contracts on contention; configured concurrency is a ceiling ([import.rs](../../src/runtime/import.rs#L70)). Training reads the complete source snapshot, loads vectors in batches of 128, preprocesses them, and trains in memory; total source memory/CPU is not independently bounded ([training contract](../../src/maintenance/training.rs#L27), [load](../../src/maintenance/training.rs#L187)). Foreground grouped routing is already implemented; drains copy unchanged absolute RaBitQ payload rather than re-encoding it ([ADR 0014](../adr/0014-expose-then-drain-splits.md)).

**Inference.** Oversize source backlog amplifies training and drain cost. The [current bridge insert handler](../../benchmarks/src/bridge.rs#L452) creates/finishes a session per at-most-50-row request while holding its write state lock. Short-lived sessions reset admission learning, but the recorded `load_concurrency=1` workload also supplies only one outstanding batch. Merely retaining a session cannot create parallel work. Separate avoidable session setup from deliberate sequential acknowledgment and the maintenance scheduling effects before treating this as a production-library limitation.

**Disconfirmation.** Compare native persistent import-session throughput to the bridge with the same records and batch sizes. Record attained in-flight count, clean/contention outcomes, split backlog/source sizes, bytes moved, and time to fully settled searchable topology. Reusing a bridge session must preserve request acknowledgments, error ownership, shutdown/finish behavior, and visibility; removing a lock without replacing its ownership contract is not a valid optimization. Consider a bulk builder only after measuring how much time the required online topology protocol consumes.

## Already present: avoid proposing these as new optimizations

- Grouped mutation routing and shared vector preprocessing; configurable write beam.
- Concurrent FDB point reads within `batch_get`; batched internal search headers.
- Epoch-validated decoded body cache; no synopsis read for unfiltered leaf search.
- Interleaved centroid/routing distance lanes preserving scalar accumulation order ([numeric.rs](../../src/search/numeric.rs#L149)).
- Selection by upper-endpoint partitioning rather than sorting every candidate, plus no redundant rough-top-r union ([selection.rs](../../src/search/rabitq/selection.rs#L102)).
- Copying absolute RaBitQ payload on split movement, and no redundant source-emptiness scan at split completion.

## Recommended next evidence

First reproduce the current k=100 baseline with explicit import order, batch/session lifetime, settled topology, beam, rerank cap, cache state, and concurrency. On that tree collect stage-wise true-neighbor survival and physical I/O. Then prioritize one quality ablation (split-assignment mismatch/centroid drift) and one cost ablation (leaf-header round trips). Compare all final outcomes at equal recall, while also retaining fixed-work comparisons that reveal causal effects. No new performance result is claimed by this source-only note.
