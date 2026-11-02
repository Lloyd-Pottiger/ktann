# Publish bounded local graph replacements atomically

## Context

A Local Graph Index must keep every committed split and merge state searchable.
Its graph is derived from the owning partition's Child Entries, but stale graph
ordinals or centroid projections can route a search through a different body.
Publishing graph bytes separately from topology would require another generation,
staging, recovery and reclamation protocol.

## Decision

Prepare complete graph replacements privately outside the write transaction.
The topology owner then protects the Active Manifest and exact source
Headers/descriptors, applies its child and partition changes, validates the
resulting Headers, and publishes the replacement descriptors and chunks in one
transaction. Obsolete tail chunks are deleted in that same transaction. The
Header cache epoch is the body-version authority; state-only transitions preserve
the epoch and graph bytes.

`maintenance::graph_topology` owns the combined topology plan.
`storage::graph_transitions` owns graph source validation, private preparation,
source fencing and graph-record mutations. A graph replacement alone does not
establish child ownership or valid tree membership. Empty internal bodies have
no graph records; removing their graph still validates the source bytes.

Bound atomic work through internal split thresholds of at most 16,384, minimum
occupancy of at most 128, `M <= 16`, the degree-specific native level cap, and two
transient entries above the split threshold. Internal drains change one receiver
per transaction. Split exposure waits for parent headroom and cannot add edges
to a ReceivingSplit parent whose source is still draining.

All participants share one 900,000-byte authority envelope for protected reads,
topology changes, graph metadata and obsolete tail deletes. Graph payloads also
pass the backend's complete transaction admission and native commit limits. The
supported maximum combined native payload is 6,869,488 bytes; the derivation and
adapter verification are part of the
[local graph resource contract](../design/partition-local-hnsw.md#resource-admission-and-failures).

A stale source discards the plan and triggers bounded replanning. An unknown
commit outcome returns to topology rediscovery; it does not authorize replaying
a private prepared mutation. Backend snapshots retain the old bytes for older
readers. Only known successful publication may seed the shared cache with a
prepared immutable body.

## Consequences

One atomic transaction preserves graph/body agreement in every searchable
committed topology state. No separate graph generation or asynchronous graph
reclamation is needed. Bulk Build may load graph bytes across transactions while
the Building lifecycle hides them, then uses its existing exact validation and
activation protocol.

Every graph update serializes a complete replacement, so a small topology edit
can still produce substantial storage work. The internal-capacity and combined
transaction bounds are correctness constraints; raising them requires a new
sizing argument. Private update algorithms and foreground routing policy can
evolve within this publication contract; their current choices and measurement
requirements belong to the [local graph design](../design/partition-local-hnsw.md).
