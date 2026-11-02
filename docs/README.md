# Documentation

Start with the [design overview](design/overview.md) for system boundaries,
invariants, and the end-to-end flow. The [domain glossary](../CONTEXT.md) defines
shared terminology. Module designs describe the current contracts; ADRs record
the decisions and rationale behind them. Later ADRs may explicitly supersede
parts of earlier decisions.

KTANN has no stable release. Check public types and defaults against the current
source; the [repository guide](../AGENTS.md) defines engineering and validation
requirements.

## Module designs

| Question | Document |
| --- | --- |
| What does the system guarantee, and which module owns each concern? | [Overview](design/overview.md) |
| How do callers configure, read, mutate, search, refine, and verify an index? | [Public API](design/api.md) |
| How do transactions, codecs, persistent identities, and deletion work? | [Storage](design/storage.md) |
| How do typed predicates, NULLs, Tree Key planning, and synopsis pruning work? | [Filtering](design/filtering.md) |
| How do traversal, ranking, budgets, and caches work? | [Search](design/search.md) |
| How do online mutations and incremental splits/merges preserve membership? | [Maintenance](design/maintenance.md) |
| How are admission, workers, recovery age, shutdown, and verification operated? | [Runtime and operations](design/runtime-operations.md) |
| How is a new index built, recovered, scheduled, and published? | [Bulk Build](design/bulk-build.md) |
| How are partition-local graphs built, updated, bound to their source, and sized? | [Partition-local HNSW](design/partition-local-hnsw.md) |
| How are immutable input, forest, and serving files prepared? | [Offline artifacts](design/bulk-artifacts.md) |

## Architectural decisions

Decision numbers are persistent identifiers and need not form a contiguous
sequence. These records explain design choices, while the module designs above
provide the consolidated current contract.

| ADR | Decision |
| --- | --- |
| 0001 | [Maintain exact leaf membership with Record Location](adr/0001-exact-leaf-membership.md) |
| 0002 | [Require one transactional KV interface](adr/0002-transactional-kv-interface.md) |
| 0003 | [Store an engine-owned Vector Record with the index](adr/0003-engine-owned-vector-record.md) |
| 0004 | [Use a Tree-Key-sharded K-means forest](adr/0004-sharded-kmeans-forest.md) |
| 0005 | [Keep exact filter projections and strong leaf synopses](adr/0005-strong-leaf-prefilter.md) |
| 0006 | [Maintain topology through searchable incremental states](adr/0006-incremental-searchable-topology.md) |
| 0007 | [Keep exact child membership without reverse parent pointers](adr/0007-exact-internal-membership-without-parent-pointers.md) |
| 0008 | [Reselect merge targets through same-level routing](adr/0008-reselect-merge-targets.md) |
| 0009 | [Use RaBitQ7 leaves with exact reranking](adr/0009-rabitq7-with-exact-reranking.md) |
| 0010 | [Cache decoded search bodies with snapshot validation](adr/0010-snapshot-validated-partition-cache.md) |
| 0011 | [Expose bounded approximate search](adr/0011-bounded-deterministic-approximate-search.md) |
| 0012 | [Retry whole mutations without record revisions](adr/0012-retry-whole-mutations-without-record-revisions.md) |
| 0013 | [Allocate stable partition identities in per-tree ranges](adr/0013-stable-partition-identity.md) |
| 0014 | [Expose split targets before draining their source](adr/0014-expose-then-drain-splits.md) |
| 0015 | [Grow an incremental binary K-means tree from an empty root](adr/0015-incremental-binary-kmeans-tree.md) |
| 0016 | [Store partition authority and entries as independent KV values](adr/0016-fine-grained-partition-storage.md) |
| 0017 | [Allocate non-reusable Logical Index IDs and govern them with manifests](adr/0017-manifest-governed-index-lifecycle.md) |
| 0018 | [Store the persistent format version in the Index Manifest](adr/0018-version-the-persistent-index-format.md) |
| 0019 | [Fail closed and verify invariants outside hot paths](adr/0019-fail-closed-and-verify-offline.md) |
| 0020 | [Use Tree Manifests as the forest directory](adr/0020-tree-manifest-directory.md) |
| 0021 | [Use one-probe Bloom sizing for an explicit false-positive bound](adr/0021-one-probe-bloom-bound.md) |
| 0023 | [Reuse committed internal bodies during foreground routing](adr/0023-committed-internal-routing-cache.md) |
| 0024 | [Support an ephemeral production Memory adapter](adr/0024-production-memory-adapter.md) |
| 0026 | [Reserve Bulk Builds through the Index lifecycle](adr/0026-building-reservations.md) |
| 0027 | [Durable workspace ownership and exact Bulk Build publication](adr/0027-bulk-workspace-and-publication.md) |
| 0028 | [Renewable automatic Bulk Build scheduling](adr/0028-automatic-bulk-scheduling.md) |
| 0029 | [Publish bounded local graph replacements atomically](adr/0029-atomic-local-graph-updates.md) |

## Development and validation

- [Project README](../README.md): setup, examples, and workspace crates.
- [Benchmark guide](../benchmarks/README.md): datasets, profiles, and comparable reports.
- [Repository guide](../AGENTS.md): required checks and test entry points.
