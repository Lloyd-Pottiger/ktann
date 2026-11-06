# Local HNSW query execution: reference validation

Date: 2026-11-05. Baseline: `9aab14d9264497ac7c8007e386b8b469f8719c16`.
Candidate: the accompanying change to graph-enabled internal traversal.
[Individual measurements and environment](hnsw-query-execution-2026-11-05.json).

## Change and evidence

Move the complete Child Entry reference check into the existing admitted CPU task
that selects graph candidates. Transfer the query's reference set without copying
it, then return it with the candidates. This eliminates one async yield per 64
edges while preserving all duplicate-reference checks, including unselected edges.
There are no new caches, persistent state, graph algorithms or scoring rules.

The measured root has 3,215 children: the baseline yields 50 times during its
reference check. The candidate checks cancellation at bounded intervals inside
the CPU task instead. Every compared run has identical layout fingerprints,
query recall distributions, logical search budgets, graph distance/visit counts,
and backend IO counts/bytes. Native HNSW does the same search work.

Results demonstrate a benefit on this **expanded synthetic diagnostic**. They do
not establish a SIFT/Cohere 1M improvement or a comparison with Milvus. Although
the runner entry point is `quality-routing-smoke`, this experiment overrides its
default 256 records and short measurement window. It is not a production capacity
baseline or SLA.

## Results

Same frozen RocksDB index, 32,768 clustered 16-dimensional L2 vectors, 64 queries,
3,215 leaves and one internal root. M=12, construction ef=100, graph seed=7,
no refinement. The host reports eight logical CPUs, approximately 16 GiB RAM,
four Tokio workers; release build without RUSTFLAGS. Partition Cache is 64 MiB
except where disabled; RocksDB block cache is 8 MiB.

Values below are medians across independent processes, run in alternating
baseline/candidate order. QPS is end-to-end public-API throughput.

| Workload | Runs per version / queries per run | Recall@10 | Baseline QPS | Candidate QPS | Change | Baseline / candidate P95 ms |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Beam 4, ef 64, concurrency 1 | 3 / 4,000 | 27.00% | 841.6 | 919.6 | +9.3% | 1.469 / 1.281 |
| Beam 4, ef 64, concurrency 8 | 3 / 4,000 | 27.00% | 5,431.4 | 5,980.4 | +10.1% | 2.025 / 1.851 |
| Beam 128, ef 200, concurrency 8 | 3 / 20,000 | 98.13% | 2,627.9 | 2,827.7 | +7.6% | 4.077 / 3.877 |
| Partition Cache disabled, beam 4, ef 64, concurrency 8 | 2 / 1,000 | 26.94% | 644.9 | 672.3 | +4.2% | 16.562 / 15.688 |

In the longer high-recall case, median process CPU time for 20,000 queries falls
from 35.576 to 34.068 seconds (4.2%); P99 falls from 5.066 to 4.847 ms.
QPS ranges are 2,543–2,631 baseline and 2,762–2,840 candidate. Process-lifetime
peak RSS ranges overlap (approximately 102–108 MiB); there is no demonstrated
memory reduction. The reference set retains its existing allocation and admission.
Cache-disabled measurements disable KTANN retention, not RocksDB or OS caches.

Short high-recall serial runs were noisy and did not demonstrate a serial
improvement. Narrow-beam recall is intentionally reported: its larger relative
speedup must not be substituted for the high-recall result.

## Reproduction

Build `ktann-bench` in release at the baseline revision and with this patch, saving
the two binaries separately. Use the same fixture directory for both; construct
it once with `--reuse-index false`, then measure with `true`. Setup, verification,
oracle computation and diagnostics are outside the timed query region.

```sh
cargo build --release -p ktann-benchmarks --bin ktann-bench
target/release/ktann-bench run \
  --backend rocksdb --profile smoke --scenario quality-routing-smoke \
  --base-vectors 32768 --query-vectors 64 \
  --min-leaf-partition-entries 4 --max-leaf-partition-entries 16 \
  --min-internal-partition-entries 128 --max-internal-partition-entries 8192 \
  --leaf-beam-size 128 --refinement-rounds 0 \
  --partition-cache-bytes 67108864 --worker-threads 4 \
  --warmup-operations 1000 --measured-operations 20000 --query-concurrency 8 \
  --routing-experiment '{"mode":"partition-local-hnsw","m":12,"ef_construction":100,"ef_search":200,"graph_seed":7}' \
  --rocksdb-path .benchmark-data/hnsw-execution-fixture --reuse-index true \
  --output .benchmark-data/hnsw-execution-result.json
```

For narrow-beam runs use beam 4, ef 64, warmup 256, 4,000 measured operations,
and concurrency `1,8`. For the cache-disabled comparison use beam 4, ef 64,
cache bytes 0, warmup 64, 1,000 operations and concurrency 8.

## Validation and known limitation

- Formatting and `cargo clippy -p ktann --all-targets --all-features -- -D warnings`.
- Release core tests with all features: 304 passed, one ignored, one excluded
  pre-existing failing property test.
- Release integration suites `tree_routing`, `search`, `e2e`: 44 tests passed.
- Added coverage: an unselected duplicate child still produces Corruption;
  Partition Keys in different trees do not collide; reference validation checks
  cancellation after a bounded batch.
- Independent read-only review found no actionable issues.

The initial full core run failed
`search::rabitq::tests::integer_scores_cover_generated_fast_path_scores` while
validating its generated fixture. The same case fails on the unmodified baseline:
`entries = [(0.0, 0)], scale = 1.8106259e19`. Proptest regression seed:
`b1669c7bf6bef1af855f1b891cd5b72d1a32ef51392e36584d3007f3cd928a85`.
This change does not modify RaBitQ or that test; the final core run excludes only
that test. No FoundationDB integration run or real-dataset VectorDBBench/Milvus
comparison was performed.
