# Dense visited membership in local HNSW

Date: 2026-11-06. Baseline: `a131caf58bf2d250e04f40e26ad6e96ac70b49e3`,
including the previously committed graph-reference scheduling improvement.
Candidate: the accompanying native visited-bitmap change, which avoids writing
bits for nodes already visited.
[Individual measurements](hnsw-visited-2026-11-06.json).

## Decision and scope

Replace the bridge's visited hash table with a bitmap indexed by dense local
ordinals. Full-domain preparation, exclusive scratch ownership, bounded allocation,
cancellation, graph traversal order and numerical scoring remain intact. Native
workspace admission bounds remain conservative; no storage format changes.

Retain this change for its demonstrated **local-routing latency and scratch-memory
benefit** on wide partitions. End-to-end throughput improvement is **not
demonstrated**. In particular, the local latency improvement must not be reported
as a whole-query QPS improvement or a result against Milvus.

The private bridge supplies the visited-set type through a small template hook;
the upstream default remains available. Regenerating the vendor patch also captures
two batching changes already present in the baseline header. Applying the regenerated
patch to the SHA-verified USearch 2.26.4 source reproduces the checked-in header
exactly; this artifact synchronization adds no further header behavior.

## Local routing measurements

Release build, deterministic independent L2 queries and centroids; M=12,
construction ef=128, seed `[7; 32]`, beam=16, search ef=64. The production f64
kernel is used. Each process builds the graphs once, warms all 64 queries, then
times five rounds of those queries, allocating fresh scratch as production reads
do. Construction and result digests are outside query timing.

Each value below is the median of three process medians, alternating baseline and
candidate processes. All 24 case digests agree across versions and processes;
the digest includes returned ordinals, exact score bits, distance evaluations and
graph visits.

| Nodes / dimensions | Baseline μs/query | Candidate μs/query | Latency change | Baseline / candidate workspace bytes |
| --- | ---: | ---: | ---: | ---: |
| 512 / 128 | 43.651 | 41.708 | −4.5% | 21,256 / 17,288 |
| 4096 / 128 | 102.182 | 98.298 | −3.8% | 164,616 / 132,424 |
| 4096 / 768 | 434.984 | 434.526 | −0.1%; no meaningful gain | 164,616 / 132,424 |
| 16384 / 128 | 256.124 | 199.743 | −22.0% | 656,136 / 527,176 |

At 16,384 nodes, workspace allocation falls 19.7%. This is charged native query
scratch, not total process RAM. Visited membership uses one bit per ordinal
instead of clearing/probing a sparse table; candidate queues remain unchanged.
For the same four cases, median local graph build times are respectively
27.618/26.957, 707.188/685.749, 3382.442/3383.938 and 5623.305/5515.189 ms
(baseline/candidate). These are local builds, not complete Bulk Build jobs.

## Complete-query check and measurement limits

The existing frozen-layout RocksDB benchmark uses 32,768 clustered 16-dimensional
vectors, one 3,215-child root, beam 128, ef 200, k=10, eight clients, four Tokio
workers, 64 MiB Partition Cache and 8 MiB RocksDB block cache. Every process warms
1,000 queries and measures 20,000. Mean recall is 98.126% throughout.

Two alternating three-run windows of the **same final binaries** produced:

| Window | Baseline QPS range | Candidate QPS range | Interpretation |
| --- | --- | --- | --- |
| Initial | about 1,945–2,132 | about 1,688–2,797 | Strong variability; cannot establish benefit |
| Confirmation | about 2,783–2,867 | about 2,786–2,865 | Versions overlap |

In the confirmation window, median QPS is 2845.6/2849.8; P95 is
3.788/3.806 ms and process CPU time is 33.454/33.347 seconds. These differences
do not demonstrate an end-to-end improvement. A separate A/A control used the
identical baseline binary in both positions and ranged from about 2,672 to 2,849
QPS. The cause of the between-window variation was not established; the earlier
unfavorable window is retained in the data rather than discarded.

All compared reports have identical dataset identity, topology, leaf-layout
fingerprint, recall distributions, graph work, logical budgets and backend IO.
The original baseline binary and a fresh baseline build are byte-identical.
The reported local benefit is strongest for much wider partitions than this
complete-query fixture. Real-data, higher-dimensional VectorDBBench performance
and sensitivity under deployment load remain unverified.

## Reproduction and validation

For the local diagnostic, run on each source version:

```sh
cargo test --release -p ktann --lib local_routing_performance -- --ignored --nocapture
```

The diagnostic itself is new; apply only its test addition to the baseline when
reproducing. Use separate binaries and alternate process order. It prints build
time, workspace bytes, result/work digests, and all five query timings.

For the complete-query check, use the high-recall command and frozen-fixture
procedure in the [previous report](hnsw-query-execution-2026-11-05.md), with the
baseline revision above. Configuration and environment are retained in the JSON
summary. Raw logs and frozen fixture from this session are under the ignored
`.benchmark-data/hnsw-visited/` directory.

Validation of the final source:

- Formatting and Clippy for `ktann` and `ktann-usearch`, all targets/features.
- Native bridge tests: 15 passed in each of debug and release, including golden
  bytes, allocation-failure sweeps, updates, cancellation/reuse, and new slot-boundary
  coverage at 63/64/65, 127/128/129 and 4097 nodes.
- Core release tests with all features: 304 passed, two ignored, one excluded.
- Release `graph_updates`, `graph_split_interleaving`, `graph_transaction_budget`
  and `e2e`: seven tests passed, including history and commit-fault cases.
- Independent native ownership/admission review found no actionable issues.
- Upstream source SHA and vendor patch round-trip verified.

The excluded RaBitQ property test is the same baseline failure reproduced and
documented in the previous report. No FoundationDB integration or production-scale
Milvus comparison was run.
