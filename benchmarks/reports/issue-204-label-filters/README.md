# Issue #204: official label workload validation

This change carries VectorDBBench scalar labels through the binary insert protocol
and applies native typed equality predicates. Numeric and unfiltered cases remain
supported. An ID-only index retains its original schema and rejects label queries.
The client and bridge must be updated together; the private protocol is unversioned.

## Verification

- Rust bridge codec/request tests: 3 passed.
- Existing Rust Bulk Build process regressions: 6 passed.
- Python client/filter/CLI unit tests: 3 passed.
- Release bridge process suite: RocksDB 11/11 and FoundationDB 11/11 passed.
- `cargo fmt --all -- --check`, affected-crate Clippy with all targets/features,
  and Ruff on the changed Python client modules passed.
- Independent review: no actionable findings, including the shutdown-test correction.

See the retained test logs. The initial sandbox run could not create Unix sockets;
real-process checks ran with local socket access. An initial concurrent-build CLI
test timed out at 90 seconds; its isolated rerun passed, and the complete release
suites passed without compilation contention. No timeout was increased.
The shutdown test now accepts both EOF and ECONNRESET: a minimal socketpair
reproducer confirmed the outcome depends on whether the peer consumed the frame
header. Successful exit, socket deletion, and scratch cleanup remain required.

## Workload and evidence

Only the RocksDB official label workload was performance-measured here.
FoundationDB was functionally tested, not performance-measured with the 1M dataset.
Existing numeric-filter results are not evidence for label-filter performance.

- Official `LabelFilterPerformanceCase`, `Medium Cohere (768dim, 1M)`.
- 1,000,000 768-dimensional vectors, cosine metric, shuffled input.
- Predicate `labels == "label_1p"`; 10,000 records match (1%).
- 1,000 official queries, `k=100`, ground truth `neighbors_labels_label_1p.parquet`.
- The official ground truth has 1,000 neighbors per query; the benchmark uses its
  first 100 for this run. All 1,000,000 ground-truth entries satisfy the label.
- Separate label rows were verified to be aligned with zero-based record IDs.
- `leaf_beam=256`, load concurrency 1, insert batch size 50, refinement rounds 2,
  refinement admission budget 12 GiB; other bridge defaults retained.
- Query concurrency 1 and 4, 30 seconds each. Canonical throughput tests run before
  serial recall/latency measurement; serial latency is not a cold-cache claim.
- 8 vCPUs, approximately 15 GiB RAM, no swap, virtiofs storage. Bulk scratch uses
  disk explicitly; `/tmp` is tmpfs on this host. No compilation or other benchmark
  was running during the workload. These are empirical host-specific results,
  not a capacity or SLA claim.

`dataset-provenance.json` records official source, file hashes, schemas, cardinality
and label checks. `environment.json` identifies source bases plus hashes of changed
source files, optimized binary, compiler and host; the implementations were
uncommitted working-tree changes at measurement time. The canonical result and
bridge/client reports retain effective settings and measured counters.

## Measured results

The canonical case completed successfully (`:)`). Source configuration and raw
metrics are retained in [result.json](result.json), the CLI trace in
[workload.log](workload.log), and per-process client evidence in [clients/](clients/).

| Metric | Result |
| --- | ---: |
| Recall@100 | 0.4277 |
| Serial latency P50 / P95 / P99 | 17.6 / 21.0 / 22.4 ms |
| Concurrency 1 QPS | 1.1245 |
| Concurrency 4 QPS | 107.0364 |
| Concurrency 1 P50 / P95 / P99 | 811.92 / 1571.38 / 1898.81 ms |
| Concurrency 4 P50 / P95 / P99 | 18.68 / 93.27 / 532.35 ms |
| Insert duration (canonical runner) | 556.4763 s |
| Optimize duration | 2055.8121 s |
| Total load duration | 2612.2883 s |

The large difference between concurrency runs is confounded by execution order
and cache warming: concurrency 1 runs first, then concurrency 4, then serial
measurement. It is **not evidence of concurrency scaling**. No repeated trials or
cache-controlled comparison were performed. Recall is only 42.77% with this search
configuration; passing this workload establishes end-to-end label support, not a
high-recall performance target. No numeric-filter comparison or performance
improvement claim is made.

The [bridge companion report](bridge.json) confirms 1,000,000 ready records,
13,131 partitions and 4,248 searches. All 4,248 searches exhausted the default
exact-rerank candidate budget (531,000 candidates total, 125 per query); none
exhausted the Tree Key or visited-partition budget. The configured leaf beam was
256. These are bounded approximate searches, not exact global top-k results.
Peak bridge RSS through report capture was 5,207,207,936 bytes (4.85 GiB).
The bridge exited successfully after dedicated-index cleanup; reported workload
I/O is captured before that cleanup.

## Resource-limited preliminary run

The preliminary run captured all 1M rows but failed Optimize with `LimitExceeded`
and executed no searches. Its refinement admission budget was 1 GiB. The current
forest code charges `1024 + 4 * id_bytes + 12 * dimension` per retained record:
`(1024 + 4 * 8 + 12 * 768) * 1_000_000 = 10_272_000_000` bytes (9.57 GiB), before
partition plans. Thus 1 GiB cannot admit this dataset. The rerun changed only that
budget to 12 GiB and used a fresh bridge/storage directory. This is conservative
admission accounting, not an RSS reservation. Separate tree/serving budgets still
apply. `failed-1g-result.json` and `failed-1g-bridge.json` retain the failed run;
its zero-valued search metrics are not valid measurements.

## Reproduce

Build the synchronized bridge/client sources, start a fresh RocksDB bridge with
disk-backed `TMPDIR`, and use the official S3 dataset source:

```sh
cargo build --release -p ktann-benchmarks --all-features --bin ktann-vdbbench-bridge
# Start bridge as described in benchmarks/README.md, using fresh socket/database/report paths.
vectordbbench ktann --socket-path /path/to/bridge.sock \
  --dataset-identity cohere-1m-labels --case-type LabelFilterPerformanceCase \
  --dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
  --label-percentage 0.01 --load-concurrency 1 --insert-batch-size 50 --k 100 \
  --refinement-rounds 2 --refinement-memory-bytes 12884901888 \
  --leaf-beam 256 --num-concurrency 1,4 --concurrency-duration 30 \
  --companion-dir /path/to/client-reports
```

Check the case's success marker and metrics in the canonical result: the top-level
CLI can return successfully even when an individual case fails. Shut down the
bridge cleanly to persist its companion report.
