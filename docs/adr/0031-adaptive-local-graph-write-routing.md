# Default adaptive local-graph write routing

For graph-enabled indexes, foreground grouped routing selects exact scoring for
internal bodies below 1024 nodes and HNSW for bodies at or above 1024 nodes.
The snapshot-authoritative Header count selects each body's path independently;
the index record count and cache warmth do not select the algorithm. Indexes
without graph configuration retain exact routing. Query behavior and budgets,
write beam, and HNSW search effort are unchanged.

This supersedes the exact-write-routing restriction in ADR 0029. Approximate
selection may change the selected leaf; exact membership, stable tie breaking,
complete child ownership validation, final write fences, cancellation, and
bounded working memory remain mandatory. Mixed waves share one candidate beam.

Cohere lifecycle experiments found graph traversal overhead dominates small
bodies, while mature 1M topology benefits substantially. The 1024 threshold is a
conservative policy above the measured crossover, not a universal optimal value.
See [PR #226](https://github.com/Lloyd-Pottiger/ktann/pull/226) for measurement
scope and limits.
The threshold is internal policy, not persisted configuration.

Prepared immutable graphs are published to the byte-bounded cache only after
known successful commit. Parent discovery collects the lock set before waiting
and scans incoming edges after guard acquisition and snapshot refresh; exact
source fences protect publication. These maintenance costs are part of the
write-lifecycle comparison. Behavioral experiment switches are removed; the
exact comparison baseline remains recoverable in version control.
