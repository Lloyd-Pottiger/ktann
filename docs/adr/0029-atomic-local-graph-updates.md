# Publish bounded local graph replacements atomically

Local graphs use internal split thresholds up to 16,384, minimum occupancy up to
128 and M<=16. Native hierarchy height is capped at floor(log_M(65536)), preserving
original RNG consumption. Graph-aware topology reserves two transient entries
and drains one receiving partition per transaction. These bounds cover the 8,192
and 16,384 operating points without retaining speculative larger configurations.

Prepare privately outside the storage transaction. Insertions and centroid changes
use native incremental operations; batches containing deletion rebuild locally.
The deletion cost was explicitly accepted. One transaction replaces graph chunks,
removes an obsolete tail and switches the graph descriptor, child projections,
changed child centroids and parent Header. The topology owner composes its child
creation/removal or movement in that same commit.

The supported two-graph worst case is 6,869,488 payload bytes. With 100,000-byte
chunk values, maximum physical key widths and a bounded combined authority
transaction, it fits FDB's 10,000,000-byte affected-data ceiling. The adapters admit
9,000,000 mutation bytes; native checks still account for actual conflict ranges.
The owning local-HNSW design records the full proof and transition obligations.

No graph generation, staging ownership, loading cursor or reclamation protocol is
needed. Backend snapshots preserve overwritten bytes. Unknown commits are resolved
only by exact target identity; a superseded source is not success evidence.
Fixed Index serving remains publicly read-only until graph-aware topology and
write routing are integrated and repeated-update quality is validated.

Obsolete source-tail chunk deletions count against the same combined 900,000-byte
non-payload envelope, including their physical keys and conflict costs. They are
not additional to the bound. Ordinary publication includes those deletes before
its envelope preflight; future topology composition must enforce this envelope
across all participants. A smaller batch or narrower keys may be needed to fit.

## Implementation clarification: mutable serving (2026-10-21)

Topology integration preserves Fixed Index read-only behavior and introduces a
separate mutable graph configuration identity. It retains the existing foreground
write routing; HNSW write routing is outside this decision. The graph-aware driver
composes private graph preparation with exact topology mutations, uses one
receiver per internal drain and removes the graph as a body becomes empty.
State-only transitions retain the body epoch. Both ordinary graph changes and
topology changes share source fencing and graph publication machinery.
