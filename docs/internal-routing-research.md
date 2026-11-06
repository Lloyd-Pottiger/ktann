# Internal partition routing: HNSW versus SPTAG

Research date: 2026-11-05. This note evaluates candidates; it changes no design
decision and reports no new benchmark results.

## Conclusion

Keep local HNSW as the working baseline. Prioritize measuring exact centroid
scoring against it across actual partition sizes. SPTAG-BKT is a useful challenger
for wide, high-dimensional, read-heavy partitions, but SPANN's use of SPTAG is
not evidence that it wins for KTANN's local routing workload. Vamana is another
plausible experimental graph core; neither replacement has been measured here.

## What is being indexed

KTANN indexes one internal partition's Child Entry centroids, not all records or
a global set of leaf representatives. The partition is the loading and maintenance
unit. The [local graph design](design/partition-local-hnsw.md) uses USearch's generic
graph with KTANN's f64 scoring, borrowed centroid storage, bounded workspaces and
snapshot-bound immutable graph bodies. The code's foreground write routing uses
exact scoring below 1024 children (`src/maintenance/routing.rs`); this is a write
policy, not a universal search crossover.

[ADR 0029](adr/0029-atomic-local-graph-updates.md) requires complete graph
replacements to publish atomically with their authoritative body. A different
search core alone does not remove serialization, transaction or cache costs.

## External evidence

- [SPTAG's official description](https://github.com/microsoft/SPTAG) defines
  SPTAG-BKT as balanced k-means trees plus a relative neighborhood graph. Trees
  supply search seeds and tree/graph searches alternate. It also provides KDT;
  its documentation positions BKT for very high-dimensional search accuracy.
  This is a compound search structure, not simply a different name for HNSW.
- [SPANN](https://arxiv.org/pdf/2111.08566) uses memory-resident posting-list
  representatives and disk-resident postings. Balanced clustering, closure-based
  posting expansion and query-aware posting pruning all contribute to the system.
  Its whole-system results do not isolate a local HNSW-versus-SPTAG comparison.
  Copying expanded record membership into KTANN would conflict with its exact
  single-Leaf-Entry membership invariant; replacing the routing core does not
  require that expansion.
- [DiskANN's original publication](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/)
  introduces Vamana and discusses in-memory graph use as well as SSD search.
  Its large-scale results do not establish a win for small local centroid sets.

## Candidate selection (engineering judgment)

| Candidate | Reason to measure | Cost or uncertainty |
| --- | --- | --- |
| Exact centroid scoring | Small partitions, large requested child fraction, no graph traversal | Linear distance work; preserve numeric semantics when optimizing |
| Existing HNSW | Wide partitions and a small routing candidate set | Loading, graph preparation and complete replacement costs |
| SPTAG-BKT | Very wide, high-dimensional partitions; possible better search seeding | Additional tree structure and update/serialization integration; benefit unproven locally |
| Vamana graph core | Read-heavy partitions; alternative connectivity/pruning | Build and update costs, graph bytes and local recall need measurement |

Do not assume an external library's insertion/deletion API supplies KTANN's
transactional snapshot semantics, deterministic tie handling or numeric contract.
These need to be preserved by any adapter.

## Smallest useful experiment

Correction after inspecting `benchmarks/README.md`: frozen-layout routing,
mutable-graph and VectorDBBench bridge entry points already exist. Reuse their
evidence and diagnostics; the comparison below is not a proposal to create a
new harness or repeat established Exact/HNSW crossover work.

Start with captured real centroid bodies and queries, keeping topology and child
selection width fixed. Compare exact scoring and HNSW at representative counts
(for example 128, 512, 1024, 4096 and configured larger capacities), actual
dimensions and query distributions. Measure centroid top-b recall, distance
evaluations, warm/cold latency, graph bytes and preparation cost. Only then add a
challenger if local routing is a material bottleneck.

Confirm end-to-end record recall versus p50/p95/p99 latency and QPS at comparable
recall, recording leaf/rerank budgets, backend reads and cache memory. Include
Bulk Build and a write/maintenance workload: changed centroid routing can change
leaf assignment and future topology. Report actual serialized bytes and commit
costs; a faster in-memory search alone is insufficient to choose a replacement.
