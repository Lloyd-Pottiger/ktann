# Maintenance drain and training validation

This implements the measured drain and training portions of [issue #162](https://github.com/Lloyd-Pottiger/ktann/issues/162).
Balanced assignment selects the median boundary using the existing total order
(distance difference, then canonical ID position), and a stable assignment reuses
its existing centroids while preserving the confirming round. Canonical mean
accumulation order, centroid bits, errors, and round counts remain unchanged.

Relocation preserves native unique insertion of destination Entries. After those
inserts succeed, one MutationBuilder stages source deletes, Locations, and
aggregated Header/Synopsis updates against the remaining transaction budget.
The existing transaction still owns atomic commit, rollback, and recovery.

The retained change also adds bounded maintenance-stage, source-level, and
candidate-count metrics; adapter API-call counters; and a delete-driven merge
benchmark with post-delete recall against the surviving corpus. Report schema
version is 4. These measurements expose where subsequent optimization is justified.

## Method and scope

Measurements ran on 2026-09-12 local time on an Apple M1 Pro, 8 logical CPUs,
16 GiB RAM, Darwin 25.6.0, Rust 1.85.0, release mode, four Tokio worker threads.
Backends were RocksDB 10.4.2 through rust-rocksdb 0.24.0 and local single-node
FoundationDB server 7.3.69 with client library 7.3.55.

Benchmarks ran serially on an otherwise idle host. Overlapping exploratory
samples were excluded. Each measured scenario used a fresh worker
process and store/namespace. Baseline and candidate used the same harness and
live instrumentation; the baseline used full sorting and unconditional centroid
recomputation. Each final backend/scenario has three pairs, ordered B/C, C/B,
B/C. Configuration, seed, dataset checksum, and limits are recorded in every JSON.

- Import: checked-in SIFT small, 10,000 vectors, 128 dimensions, 100 queries,
  batch size 50, at most four in-flight batches, two maintenance workers,
  partition thresholds 32/128. Import wall time includes `finish`; complete
  lifecycle wall/CPU includes immediate search, convergence, and cold/warm search.
  Submit p95/p99 measures import submission/backpressure, not foreground mutation
  latency. Immediate-search p95/p99 measures the first query pass after import.
- Merge: 5,000 clustered 128-dimensional vectors, import and convergence outside
  measurement, then 3,750 distinct deletes interleaved with 3,750 searches at concurrency four.
  Foreground p95/p99 measures accepted delete/search operations. Timed wall/CPU
  includes draining maintenance; verification and 100 post-delete recall queries
  are outside that interval. RSS is the process high-water mark and includes setup.
- Numeric microbenchmark: production leaf `Bytes` IDs, L2, 20 training calls per
  sample, six alternating reference/candidate samples per input shape. Both
  paths pay for input cloning. A separate differential test covers all metrics,
  canonical ties, empty/singleton/odd/even inputs, zeros, and extreme magnitudes,
  comparing centroid bits, errors, and round counts against the original algorithm.

The targeted gains are lower training work and fewer adapter mutation calls. End-to-end
results below are complementary checks, not a claim that every maintenance
workload becomes faster. These small trees exercise leaf drains; deeper-tree
candidate discovery and the million-vector datasets remain unmeasured.

## Retained results

Training times are median milliseconds per 20 calls.

| Dimensions | Entries | Full sort/recompute | Retained | Change |
|---:|---:|---:|---:|---:|
| 16 | 128 | 0.580 | 0.499 | -14.0% |
| 16 | 1024 | 5.606 | 4.464 | -20.4% |
| 128 | 128 | 4.844 | 4.720 | -2.6% |
| 128 | 1024 | 29.569 | 28.833 | -2.5% |
| 768 | 128 | 23.083 | 22.204 | -3.8% |
| 768 | 1024 | 183.644 | 176.496 | -3.9% |

System wall times are median [min–max] seconds across three runs. CPU seconds
and peak RSS MiB are baseline → retained medians. Import CPU covers the whole
lifecycle; merge CPU covers only its measured workload and drain.

| Case | Baseline wall | Retained wall | CPU s | RSS MiB |
|---|---:|---:|---:|---:|
| RocksDB / import | 1.258 [1.245–1.265] | 1.021 [1.020–1.049] | 7.43 → 6.87 | 135.1 → 134.9 |
| RocksDB / merge | 7.868 [7.823–7.904] | 7.634 [7.261–7.646] | 32.02 → 31.01 | 129.3 → 129.1 |
| FoundationDB / import | 8.169 [8.125–8.913] | 8.140 [8.018–8.229] | 12.43 → 11.35 | 79.9 → 81.0 |
| FoundationDB / merge | 26.192 [26.172–26.476] | 25.907 [25.856–25.945] | 33.86 → 32.63 | 66.8 → 68.0 |

Complete import lifecycle wall time (baseline → retained median [min–max]):

- RocksDB / import: 6.532 [6.516–6.573] → 6.256 [6.240–6.301] s.
- FoundationDB / import: 28.113 [26.310–28.371] → 25.437 [25.415–27.860] s.

Tail latency is median milliseconds across runs, baseline → retained. The first
two columns are import submit latency for import and foreground delete latency
for merge. Full per-run distributions remain in the JSON reports.

| Case | Submit/delete p95 | Submit/delete p99 | Search p95 | Search p99 |
|---|---:|---:|---:|---:|
| RocksDB / import | 16.12 → 13.98 | 23.11 → 16.33 | 18.88 → 18.10 | 20.40 → 18.87 |
| RocksDB / merge | 0.24 → 0.24 | 0.27 → 0.28 | 11.24 → 10.82 | 11.55 → 11.13 |
| FoundationDB / import | 102.80 → 94.01 | 165.55 → 153.78 | 70.43 → 81.15 | 99.05 → 96.95 |
| FoundationDB / merge | 8.98 → 9.08 | 13.38 → 13.80 | 26.73 → 26.41 | 37.26 → 35.45 |

Adapter work is baseline → retained median. Calls are API invocations, not
physical RPCs. Relocation batches existing writes; variations in mutation bytes
and retries reflect concurrent maintenance/import execution.

| Case | Point-read calls | Mutation calls | Mutation MiB | Write retries |
|---|---:|---:|---:|---:|
| RocksDB / import | 92659 → 92532 | 53671 → 19837 | 14.806 → 14.788 | 97 → 98 |
| RocksDB / merge | 464516 → 464354 | 7592 → 5103 | 0.788 → 0.772 | 4 → 2 |
| FoundationDB / import | 91715 → 91082 | 49799 → 18751 | 14.288 → 14.501 | 146 → 137 |
| FoundationDB / merge | 465384 → 465355 | 8936 → 5496 | 0.906 → 0.899 | 33 → 36 |

Relocation stage cost normalized by committed moved entries (ms/entry, median
[min–max]) includes retry work in the numerator.

| Case | Baseline | Retained |
|---|---:|---:|
| RocksDB / import | 0.055 [0.054–0.059] | 0.030 [0.029–0.030] |
| RocksDB / merge | 0.095 [0.090–0.096] | 0.047 [0.046–0.052] |
| FoundationDB / import | 0.222 [0.218–0.240] | 0.222 [0.216–0.227] |
| FoundationDB / merge | 0.406 [0.404–0.430] | 0.394 [0.374–0.406] |

Stage totals are median attempted milliseconds, baseline → retained. Import
aggregates lifecycle phases; merge excludes setup. `training_load` includes
`training_preprocess`, so those columns must not be added together.

| Case | Training load | Preprocess | Training | Relocation | Discovery | Routing |
|---|---:|---:|---:|---:|---:|---:|
| RocksDB / import | 42.87 → 42.39 | 5.45 → 5.66 | 53.46 → 53.00 | 789.76 → 428.24 | — | — |
| RocksDB / merge | — | — | — | 96.75 → 48.41 | 14.11 → 14.28 | 12.28 → 11.46 |
| FoundationDB / import | 227.02 → 212.11 | 6.75 → 6.81 | 62.11 → 61.11 | 3276.15 → 3308.74 | — | — |
| FoundationDB / merge | — | — | — | 412.55 → 400.81 | 76.54 → 75.39 | 17.40 → 16.93 |

All measured final query passes had mean recall@10 1.0. Import query passes reported RaBitQ-overlap truncation on all 100 queries,
with no exact-rerank-cap exhaustion. Post-delete query passes reached the
configured exact-rerank cap of 64 on all 100 queries and reported no RaBitQ-overlap
truncation. Scanned-tree, visited-partition, and visited-leaf budgets were not
exhausted. These results matched between baseline and retained runs. Merge runs
committed 33 drain steps each and ended with no pending maintenance.
Trees reached level 2; discovered candidate counts ranged from
32 to 64. The training microbenchmark demonstrates lower training cost; whole-case
latency includes scheduling and backend noise, and is not presented as a general
end-to-end speedup.

## Interpretation and remaining limits

The RocksDB result is demonstrated for these workloads: relocation time per
committed moved entry fell 46% during import and 50% during deletion-driven
merge, with 63%/33% fewer mutation API calls and 19%/3% lower measured wall time.
The mechanism is fewer blocking adapter operations while retaining unique
insertion. Training additionally reduced the isolated numeric workload by
2.5%–20.4% across the measured shapes.

FoundationDB mutation calls fell 62%/39%, but import relocation time per moved
entry was unchanged. Import wall time was effectively flat, so fewer calls are
not presented as an RPC or import-throughput improvement. CPU decreased in both
scenarios; peak RSS medians increased by approximately 1.2 MiB, with overlapping
run ranges.

Tail results are mixed. FoundationDB immediate-search p95 increased from 70.43
to 81.15 ms (run ranges 66.41–70.69 versus 63.45–82.69 ms), while p99 decreased
from 99.05 to 96.95 ms. Its interleaved delete p99 increased from 13.38 to
13.80 ms; concurrent search p99 decreased from 37.26 to 35.45 ms. These short
three-pair runs do not establish universal tail-latency neutrality. The strict
all-tail non-regression portion of the broader issue remains unproven; the
retained implementation is a reviewable result with the measured costs above,
not a claim that every issue acceptance target has been satisfied.

## Rejected and deferred candidates

Batched destination absence checks plus one MutationBuilder for leaf/internal
relocation were implemented and passed correctness tests, including duplicate
inputs, read-your-writes, concurrent destination inserts, bounded drains, and
unknown-outcome recovery. The whole-system performance gate rejected them.
In the exclusive combined-candidate run, median FoundationDB import rose from
8.329 to 14.832 seconds and submit p99 from 173.3 to 303.5 ms, despite reducing
adapter call counts and relocation-stage duration. This is a material regression.

Two interleaved ablation repeats isolated the regression: relocation alone took
14.254 seconds versus 8.081 seconds for the baseline; its submit p99 was 274.7
versus 169.2 ms. Training alone did not reproduce that regression. The batched destination-read
implementation and its unused destination-validation
metric were removed. The final, smaller candidate preserves individual unique
inserts and batches only the remaining writes. The result establishes which
candidate causes the regression; it does
not establish the underlying FoundationDB scheduling/latency mechanism. Fewer
adapter calls must not be interpreted as fewer physical RPCs or faster imports.

Same-level discovery now records duration and candidate count. The measured
merge trees have only one internal parent body, so there is no demonstrated
benefit from batching parent-body scans here. Discovery, drain limits, workers,
transaction admission, and numeric preprocessing retain their existing behavior.
Large/deep-tree candidate-scan optimization remains a separate measured follow-up.

## Correctness and verification

The final version passed 577 core library and integration tests in release mode,
excluding the pre-existing timing-sensitive audit cancellation test described
below. All 21 verification tests passed in debug mode. The six-shape release
training microbenchmark was run explicitly. Formatting and workspace/all-targets/
all-features release Clippy with warnings denied passed.

The benchmark crate's 35 tests passed. Real RocksDB adapter, fault, and recall
tests and real FoundationDB adapter, fault, recall, and verification tests passed
on the final candidate. An independent reviewer found no actionable correctness or design issues in the
implemented training, metrics, harness, or final relocation candidate. Final benchmark
runs additionally fail on incomplete/corrupt post-run verification.

A subsequent simplification removed post-delete dataset/configuration clones and
reused the existing conflict-test setup, without changing the measured training
or relocation algorithms. The two focused relocation tests, all 35 benchmark
tests, FoundationDB delete-driven smoke validation, formatting, and full release
Clippy passed again. The smoke run validates behavior, not performance.

One pre-existing release-only test, `verify::cancellation_stops_a_long_audit`,
failed when the optimized 4,000-record audit completed before cancellation
took effect; the test uses a 10 ms timer. The same failure was reproduced with the original algorithms;
all 21 verification tests passed in debug mode. This is recorded as an existing
timing-test limitation, not silently counted as a passing release gate.

## Reproduction and artifacts

Use the local FoundationDB environment in the root `AGENTS.md`, then:

```sh
cargo build --release -p ktann-benchmarks --features rocksdb,foundationdb --bin ktann-bench
target/release/ktann-bench run --backend rocksdb --profile full \
  --scenario import-to-search-lifecycle --worker-threads 4 --output /tmp/import.json
target/release/ktann-bench run --backend foundationdb --profile full \
  --scenario delete-driven-merge --worker-threads 4 --output /tmp/merge.json
cargo test --release -p ktann --lib training_release_comparison -- --ignored --nocapture
```

Repeat each scenario on both backends. Run only one Cargo/test/benchmark process
at a time; each A/B benchmark invocation must run on an otherwise idle host.
For an algorithm baseline, restore full sorting in `assign`, unconditional
centroid recomputation in `train`, and individual source/location/metadata writes
in relocation while keeping unique inserts and the same harness/instrumentation.
The baseline source snapshots and immutable binary hashes are retained locally.

Raw artifacts live under `.benchmark-data/results/issue-162/` in the primary
checkout (ignored generated data):

- `writebatch/`: final three-pair JSON reports, exact command/time schedule,
  summary generator and output, tests, source patch, and source/binary hashes.
- `retained/`: intermediate training-only three-pair reports and an independent
  results summary, retained to distinguish training from relocation effects.
- `isolated/training-release.log`: six alternating samples for each training shape.
- `isolated/`: exclusive rejected combined-candidate measurements; the import
  comparison completed all three pairs. Merge repetition was stopped after
  rejection, so partial merge data is not an acceptance result.
- `ablation/`: two interleaved FoundationDB import comparisons for training-only,
  relocation-only, and baseline.
- `final/`: original/candidate source snapshots and baseline cancellation evidence.
  Earlier performance measurements here are exploratory and excluded from acceptance.

Base revision: `909b750f51a1b1524bca45caec9716632192b2da`.
