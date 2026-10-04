# Million-vector search residency diagnostics

Measured on 2026-10-05 with committed binary sources `8a8940c`, release/all features,
Apple M1 Pro (8 logical CPUs, 16 GiB RAM), 8 Tokio workers, RocksDB 10.4.2,
and the adapter default of 8 native transaction slots. Each dataset imports once,
converges and passes complete structural verification, then runs client counts
`1,4,16,64,16,4,1` on the same index. Each point has 1,000 warmup and 10,000
measured queries drawn repeatedly from 1,000 supplied queries, k=100, beam=32,
and unchanged search budgets. The first Cohere single-client point was CPU-profiled
for two seconds; exclude it from throughput conclusions. All other timed points
were unprofiled and builds/tests ran before measurement.

These are residency and saturation observations, **not a baseline/candidate speedup**.
Physical disk IO, hardware cache misses and memory bandwidth were not measured.
A busy CPU can also be stalled on memory; occupancy alone cannot distinguish that.

## SIFT1M / 128-dimensional L2

Partition Cache capacity: 1 GiB. Accounted decoded bodies: 207.7 MiB. Recall@100: 0.84647.

| Clients (run order) | QPS | CPU cores | CPU ms/query | p50 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 278 | 1.00 | 3.62 | 3.58 | 4.07 |
| 4 | 993 | 3.99 | 4.02 | 4.01 | 4.40 |
| 16 | 1633 | 7.81 | 4.79 | 9.37 | 15.20 |
| 64 | 1627 | 7.81 | 4.80 | 38.95 | 44.72 |
| 16 | 1575 | 7.51 | 4.77 | 9.66 | 15.92 |
| 4 | 990 | 3.99 | 4.03 | 4.02 | 4.38 |
| 1 | 277 | 1.00 | 3.62 | 3.59 | 4.02 |

All 10,000 queries succeeded at every point. Each point had 90,000 internal
and 320,000 leaf Partition Cache hits, no misses or installations. Logical
backend IO and every budget summary were identical across the seven points;
recall means agree within floating-point aggregation precision.

The two 16-client points occupied 7.5-7.8 cores: enough query concurrency exists
to saturate this host on the real corpus. 64 clients delivered similar QPS while
p50 rose from about 9.5 to 39 ms. Increasing concurrency further has no demonstrated
throughput benefit here. CPU/query rises from 3.62 ms at one client to 4.77-4.79 ms
at 16 clients; coordination and cache/memory effects remain profiling questions.
At the first 16-client point, approximate selection averaged 2.81 ms and exact
reranking 2.00 ms, excluding adapter admission. Mean native admission wait was
4.89 ms. These overlapping stage durations are not additive CPU measurements.
## Cohere1M / 768-dimensional Cosine

Partition Cache capacity: 4 GiB. Accounted decoded bodies: 831.3 MiB. Recall@100: 0.76671.

| Clients (run order) | QPS | CPU cores | CPU ms/query | p50 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (profiled) | 66 | 0.59 | 8.95 | 15.11 | 19.01 |
| 4 | 277 | 2.60 | 9.37 | 14.33 | 17.63 |
| 16 | 504 | 5.19 | 10.30 | 31.41 | 41.82 |
| 64 | 575 | 6.12 | 10.64 | 119.01 | 152.12 |
| 16 | 689 | 7.53 | 10.92 | 22.36 | 33.57 |
| 4 | 371 | 3.47 | 9.37 | 10.67 | 16.58 |
| 1 | 114 | 0.97 | 8.49 | 8.39 | 15.35 |

All 10,000 queries succeeded at every point. Each point had 90,000 internal
and 320,000 leaf Partition Cache hits, no misses or installations. Logical
backend IO and every budget summary were identical across the seven points;
recall means agree within floating-point aggregation precision.

The repeated 16-client QPS changed by 37%, from 504 to 689, with CPU occupancy
rising from 5.19 to 7.53 cores. Exact reranking fell from 9.69 to 4.28 ms while
approximate selection rose from 6.09 to 7.16 ms. Mean native admission wait fell
from 15.83 to 11.56 ms. The unprofiled final single-client point occupied 0.97 cores.
This is material drift despite fully resident decoded partitions and identical work;
the run does not establish steady raw-vector residency or a repeatable scaling curve.

## What the caches actually cover

The Partition Cache holds decoded Leaf/Child Entries, not original Vector Records.
Each timed query still reads 292 point keys plus one logical scan. Across 10,000
queries, logical returned bytes were 756,350,340 (SIFT) and 3,956,330,540 (Cohere).
These counters include bytes returned from memory and are not disk/RPC measurements.

The benchmark opens RocksDB with `Options::default()`. Its separate native block
cache is therefore only 8 MiB (rocksdb 0.24.0 `db_options.rs`, documented default).
An OS file cache may satisfy the remaining reads, but residency is not guaranteed.
The first Cohere single-client stack sample included both `pread` and RaBitQ/routing
scoring. A file-read syscall does not prove a physical disk read. Combined with
the rerank/occupancy drift, it identifies raw-record access and cache convergence
as the next discriminating measurement; it does not prove an arithmetic or bandwidth
bottleneck. No mutex-sharding or SIMD production candidate is retained on this evidence.

The configured local FoundationDB status reports the **memory** storage engine,
a 2 GiB process limit and 1 GiB storage capacity. Full Cohere raw vectors alone
require about 3 GiB. A resident full-Cohere FoundationDB run is infeasible under
that configuration; no cluster reconfiguration or full-corpus import was attempted.
FoundationDB service CPU/RPC scaling remains unmeasured in this investigation.

## Reproduce and continue

Build and launch sequentially, with the local native-library environment from
[README](README.md). For each scenario, use the same query corpus and all other
parameters when changing one variable:

```sh
cargo build --release -p ktann-benchmarks --bin ktann-bench --all-features
target/release/ktann-bench run --backend rocksdb --profile large \
  --scenario quality-sift-1m --worker-threads 8 \
  --query-concurrency 1,4,16,64,16,4,1 --leaf-beam-size 32 \
  --partition-cache-bytes 1073741824 --output sift-rocksdb-concurrency.json
```

For Cohere, use `quality-cohere-1m` and `4294967296` Partition Cache bytes.
The external evidence directory is
`/Users/lloyd/projects/ktann/.benchmark-data/results/search-production-20261005-31d4`:
full JSON reports/logs, bounded SIFT diagnostic/self-comparison, test/Clippy/build
logs, the source commit and binary hash, the CPU sample, and FoundationDB status.

Next: expose and record native RocksDB block-cache capacity/residency and timed
physical IO; give warmup a convergence criterion rather than assuming 1,000 queries
are enough. Reuse the same persisted index for baseline/candidate trials so topology
and recall remain controlled. Only after a steady baseline, profile and reduce the
dominant CPU/memory work, preserving budgets, exact membership and snapshot semantics.
