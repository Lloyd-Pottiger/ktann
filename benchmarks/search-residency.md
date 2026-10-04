# Resident million-vector production search baseline

These runs establish stable, resident RocksDB search on both public million-vector
corpora, with approximately 7.8 of eight CPU cores busy at 16 clients. They do
not establish an optimization speedup or distinguish arithmetic from CPU-cache
and memory-bandwidth limits. The earlier [production sweep](search-production.md)
used different imported trees and warmup conditions and is not a causal baseline
for these runs.

## Conditions

- Apple M1 Pro, eight logical cores, 16 GiB RAM; Darwin 25.6.0.
- Rust 1.99.0 / LLVM 23.1.1, release, all Backend features;
  `RUSTFLAGS='-L native=/Users/lloyd/.local/lib'`.
- Production RocksDB adapter, RocksDB 10.4.2, eight native actors and eight Tokio
  workers; default import settings, no refinement, max partition entries 512.
- One retained, fully verified import per dataset. Both have one million records,
  three searchable levels, no transitional or actionable partitions. Cohere has
  2,767 partitions; SIFT has 2,737.
- Cohere: 768-dimensional Cosine, 4 GiB native block cache. SIFT: 128-dimensional
  L2, 1 GiB native block cache. Both: 1 GiB decoded Partition Cache, beam 32,
  k=100, the supplied 1,000 held-out queries repeated ten times per point.
- Ordered clients `16,4,1,4,16`. Every point has 10,000 untimed warmup searches
  followed by 10,000 measured searches. Setup, full audits, warmup, and the CPU
  sample are outside timing. CPU/physical IO include the post-search maintenance
  drain, which performs no logical writes here.

## Measurements

CPU cores means process user+system CPU seconds / timed wall seconds. CPU/query
is process CPU seconds / successful searches. Latencies are end-to-end public
API observations. Rows retain execution order rather than averaging repetitions.

| Dataset | Clients | QPS | CPU cores | CPU/query ms | p50 ms | p99 ms | Physical read bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cohere | 16 | 808.63 | 7.82 | 9.668 | 19.267 | 28.765 | 0 |
| Cohere | 4 | 489.43 | 3.98 | 8.137 | 8.149 | 8.617 | 4096 |
| Cohere | 1 | 131.87 | 1.00 | 7.604 | 7.546 | 8.333 | 0 |
| Cohere | 4 | 489.36 | 3.98 | 8.135 | 8.148 | 8.592 | 0 |
| Cohere | 16 | 809.21 | 7.85 | 9.695 | 19.224 | 29.423 | 0 |
| SIFT | 16 | 1687.59 | 7.85 | 4.654 | 9.099 | 14.370 | 0 |
| SIFT | 4 | 1020.96 | 3.99 | 3.904 | 3.905 | 4.219 | 0 |
| SIFT | 1 | 285.18 | 1.00 | 3.524 | 3.492 | 3.954 | 0 |
| SIFT | 4 | 1020.46 | 3.99 | 3.905 | 3.903 | 4.230 | 0 |
| SIFT | 16 | 1691.16 | 7.86 | 4.647 | 9.104 | 14.318 | 0 |

All searches succeeded; no rejections or operation errors. No timed physical
writes. SIFT physical reads were zero at every point. Cohere had one 4 KiB read
at the first four-client point; all other points were zero. This establishes
native/OS memory residency for the measured workload, rather than absence of
logical backend reads or proof that every vector fits simultaneously in the
native block cache.

Within each dataset, every point has identical logical IO, search budgets and
visited-leaf distributions. Each point performs 10,000 read transactions,
2,920,000 point-read keys and 10,000 scans. Logical bytes read are 3,956,329,560
for Cohere and 756,350,780 for SIFT. Recall is unchanged within each sweep:
Cohere 0.76819 and SIFT 0.84608.

Both decoded caches have only hit lookups and no measured installs. Accounted
bytes are 871,180,512 (Cohere) and 217,708,944 (SIFT). Final native cache usage is
4,294,787,632 / 4 GiB and 1,073,592,149 / 1 GiB respectively, with 87 pinned bytes
in each. These native usage observations occur **after the final full audit**;
they do not establish occupancy during individual timed points. Construction
peak RSS is 5,886,803,968 and 2,548,465,664 bytes; these process high-water marks
are not per-point resident memory or the OS file-cache footprint.

At 16 clients the native actor gate adds mean waits of approximately 9.84 ms
(Cohere) and 4.72 ms (SIFT), while four-client waits are effectively zero.
The eight actor slots already keep eight cores busy. Increasing waiting clients
adds queueing; it does not supply more execution capacity. Compared with one
client, CPU/query rises approximately 27% (Cohere) and 32% (SIFT) at saturation;
this does not isolate cache contention, scheduler effects or core heterogeneity.

## Reopened fixtures

Both full databases were closed and reopened in a fresh worker using
`--reuse-index true --query-concurrency 16,16` with the same capacities,
10,000-search warmup and 10,000-search timed regions. Full audits verified the
identical topology. Logical IO, budgets, visited-leaf distributions and recall
match each dataset's fresh run.

| Dataset | QPS | CPU cores | CPU/query ms | p50 ms | p99 ms | Physical read bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cohere | 809.80 | 7.77 | 9.601 | 19.300 | 28.075 | 4096 |
| Cohere | 809.82 | 7.77 | 9.593 | 19.275 | 27.996 | 4096 |
| SIFT | 2157.20 | 7.85 | 3.641 | 7.072 | 11.664 | 0 |
| SIFT | 2157.86 | 7.83 | 3.627 | 7.098 | 11.562 | 0 |

No timed physical writes. Post-final-audit native cache occupancy is
4,066,835,156 bytes for Cohere and 787,210,147 for SIFT, both below capacity.
Cohere is stable across fresh/reopened runs. SIFT's approximate stage stays near
2.78 ms, but exact reranking falls from approximately 1.84 ms to 0.85 ms after
reopening. The same tree and near-zero physical IO do not imply the same native
storage/cache lifecycle or CPU cost. This is not a code speedup; native cache
hit/miss and compaction/memtable effects have not been isolated. Use reopened
fixtures on both sides of future comparisons, with independent process
repetitions, rather than comparing fresh import measurements against reopened
candidate measurements.

## Profile and next experiment

A five-second `/usr/bin/sample` capture during **untimed single-client Cohere
warmup** shows active top-of-stack sample counts of 1,572 for
`approximate_distances<4>`, 401 for `routing_centroid_distances<4>`, 110 for
`finish_distance`, and 60 for exact `distance_validated`. No `pread` appears in
the collapsed top-of-stack entries with at least five samples. These are sample
counts, not percentages of process CPU; idle threads and inlining limit exact
attribution. The sample finished before the timed region.

The saturated stage means agree with this direction: Cohere approximate
selection is 7.11–7.17 ms and exact reranking 2.57–2.62 ms; SIFT is approximately
2.80 ms and 1.84 ms. Profile arithmetic and memory access in the batched rough
and routing kernels first. Preserve conservative interval bounds, deterministic
rough/tie ordering, exact membership, snapshot semantics, recall and budgets.
No new production scoring candidate is retained by this experiment.

Hardware cache-miss/bandwidth counters remain unmeasured. FoundationDB's separate
server CPU and RPC behavior remain unmeasured here; application physical IO
cannot describe server IO. The local FDB memory engine's approximately 1 GiB
storage capacity cannot hold Cohere's raw corpus as configured. Do not apply
this RocksDB result to FoundationDB.

## Reproduction and artifacts

```sh
export DYLD_LIBRARY_PATH=/Users/lloyd/.local/lib
export KTANN_BENCH_DATASET_CACHE=/Users/lloyd/projects/ktann/.benchmark-data/vectordb_bench/dataset
RUSTFLAGS='-L native=/Users/lloyd/.local/lib' cargo build --release \
  -p ktann-benchmarks --bin ktann-bench --all-features
target/release/ktann-bench run --backend rocksdb --profile large \
  --scenario quality-cohere-1m --worker-threads 8 \
  --query-concurrency 16,4,1,4,16 --leaf-beam-size 32 \
  --partition-cache-bytes 1073741824 --rocksdb-block-cache-bytes 4294967296 \
  --warmup-operations 10000 --rocksdb-path /absolute/new/cohere-fixture \
  --output cohere-fresh.json
```

For SIFT use `quality-sift-1m`, native capacity `1073741824`, and a separate fresh
fixture directory. Reopen with the same construction settings and
`--reuse-index true`; client count, warmup and read beam can change. The runner
checks the construction identity and fully verifies the identical topology.
Use reused fixtures for both sides of later code comparisons. The native cache
capacity is in the runtime identity, preventing an accidental code comparison
between different cache capacities.

Evidence directory:
`/Users/lloyd/projects/ktann/.benchmark-data/results/search-residency-20261005-31d4`.
It retains full schema-8 JSON/logs, both persisted databases and fixture manifests,
the exact measurement executable and SHA256, full and bounded fresh/reuse validation and
self-comparison, and the CPU sample. Measurement controls are implemented in
`ae54feb` (only CLI diagnostic text/tests changed after the executable build).
The executable SHA256 is
`4bd64d795d3078005805c7b37a75b01e4bde5c28ce8aa7cdfc3d4c59814b8d00`.

Controls passed 45 benchmark library tests, formatting, default/all-feature
Clippy, release build, bounded fixture import/reuse and changed-config rejection,
and independent review with no actionable findings.
