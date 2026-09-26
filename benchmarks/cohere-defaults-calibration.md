# Cohere1M search default calibration

The quality target is mean recall@100 above 0.9 on the complete Cohere1M cosine
dataset and its 1000 canonical queries. These measurements cover that workload;
they are not a recall guarantee for arbitrary datasets, filters or values of k.

## Same-index search curve — 2026-09-26

The optimized native harness built one million-vector index, verified it, then
warmed and measured all 1000 queries for each beam at concurrency 16. The index
used partition entries 32/128 and write beam 8. Search used public API default
budgets: 4096 Tree Keys, 1024 partitions, 65536 Leaf Entries, and the engine's
150 exact-rerank candidates for k=100. Only the beam varied between points.

| Beam | Recall@100 | QPS | p50 (ms) | p99 (ms) | CPU seconds | Backend bytes read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.62719 | 529.66 | 29.11 | 47.43 | 13.40 | 879303448 |
| 128 | 0.82525 | 208.79 | 73.93 | 117.00 | 34.26 | 2101290542 |
| 256 | 0.89762 | 124.41 | 125.29 | 177.00 | 60.26 | 3509379962 |
| 512 | 0.94838 | 67.08 | 223.95 | 796.09 | 108.63 | 6138641508 |
| 1024 | 0.96497 | 49.62 | 316.07 | 442.58 | 147.22 | 8117112002 |

Beam 32 was the original VectorDBBench adapter override; 128 was the original
public API default. Increasing the beam improves recall without changing exact
reranking. The mean number of visited Leaf Entries rises from 11894 at beam 128
to 47510 at beam 512. Beam 1024 reaches the default Leaf Entry budget on every
query; no smaller measured point exhausts a traversal budget. Every point uses
all 150 rerank candidates. Exhausting that candidate budget alone therefore
does not establish that reranking caused the original low recall.

The gain has a substantial cost: beam 512 uses about 2.9 times the Backend bytes
and 3.2 times the CPU of beam 128 in this run. These are measurements of an
increased quality setting, not a performance improvement. Tail latencies and
absolute throughput from a single short measurement should not be treated as
stable capacity estimates. No other benchmark ran during the search curve.

The initial complete artifact is
`.benchmark-data/results/cohere-defaults-calibration-2026-09-26.json`.
The tested binary SHA-256 was
`c19d1fded4aff6eb84f9d53df45fee9f2606fee6bc8987b03c515ec42ef6621c`.
The calibration scenario also includes beam 384 to examine the interval
between the measured 256 and 512 points:

```sh
KTANN_BENCH_DATASET_CACHE="$PWD/.benchmark-data/vectordb_bench/dataset" \
target/release/ktann-bench run --backend rocksdb --profile large \
  --scenario quality-defaults-cohere-1m --worker-threads 8 \
  --output "$PWD/.benchmark-data/results/cohere-defaults-calibration.json"
```

VectorDBBench now inherits the engine's search defaults when flags are omitted.
Its explicit `--leaf-beam` and `--leaf-budget` overrides remain available; the
companion report records resolved values rather than an implicit or duplicated
Python default.

## Canonical validation of beam 384

A separate complete VectorDBBench run on a newly built Cohere1M index measured
**recall@100 = 0.9293** with `--leaf-beam 384` and otherwise public Search Budget
defaults. All 1000 canonical queries were used; the dataset, ground truth,
k=100 and exact-rerank policy were unchanged. Serial p50/p95/p99 were
68.3/89.6/99.3 ms. The 5-second concurrent stages at concurrency 1 and 4 reported
6.7778 and 45.6125 QPS respectively. These timings include the benchmark bridge
and must not be directly compared with the native curve's concurrency-16
timings above.

All 1,000,000 records were confirmed and the final topology had no actionable
or transitional partitions. No query exhausted a traversal budget; mean
visited Leaf Entries were approximately 35426 across the run's 1266 searches.
The process exited cleanly, removed its socket and wrote all canonical and
companion artifacts under
`.benchmark-data/results/issue-128/cohere1m-beam384/`.
The bridge binary SHA-256 was
`3b79e18b5a6306a742aa4221649f8e615acb1b351ba5a3b37d88ff12e23f2acd`.
Functional test/build work overlapped parts of loading, but no other benchmark
or test ran during the search stages. This validates the quality setting, not
an isolated load-performance comparison.

The candidate configuration is beam 384 with unchanged public budgets and
reranking. It reaches the requested quality threshold with a smaller search
width than 512; the existing beam 128 default remains unchanged pending
acceptance of the measured quality/cost trade-off.

Verification of default inheritance and calibration preparation passed:
658 workspace tests (8 ignored), five real process/CLI tests on each backend,
all-targets/all-features Clippy, formatting and independent scoped review.
