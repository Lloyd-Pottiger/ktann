# Repair private local graphs after pure deletion

This decision supersedes only ADR 0029's choice to rebuild every deletion batch.
The atomic replacement protocol and storage bounds remain unchanged.

Wide internal parents repeatedly remove an old child when a split finishes.
Rebuilding all surviving nodes makes this small topology change proportional to
full HNSW construction and repeats most distance calculations.

For a nonempty strict subset whose surviving centroids are unchanged, restore and
validate the source graph, physically remove deleted node tapes, and compact
surviving ordinals into canonical Child Entry order. Keep survivor levels, choose
a highest-level entry, and refresh the surviving endpoints of edges incident to
removed nodes with the existing native update operation. Audit directed base-layer
reachability before and after repair. All work is private and bounded by the
existing operation allocation account. Cancellation or allocation failure discards
the private graph; tape ownership transfer itself performs no throwing operations.

The existing source-fenced transaction publishes complete graph records, child
projections, centroids, and headers together. Snapshots retain their old graph.
Mixed insertion/deletion/centroid changes retain the builder path. Empty results
remove graph records after source validation. Source corruption must still fail
before either update path is chosen.

Batching sibling topology transitions would enlarge scheduling and transaction
authority. Tombstones would transfer work and state to queries and cleanup.
Private physical deletion needs neither and changes no persisted format. It still
scans adjacency and serializes the full replacement; it does not promise constant
cost, identical approximate geometry, or incremental storage writes. Fixed-budget
recall, repeated updates, allocation/cancellation safety, and end-to-end maintenance
must be verified alongside CPU savings.
