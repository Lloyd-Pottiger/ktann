# KTANN benchmarks

`ktann-bench` measures ANN quality and resource costs through the public
`Runtime` and `Index` APIs. It writes JSON reports for reproducible, same-host
comparisons. Results are empirical measurements, not an SLA.

## Run

Build and run with optimizations enabled:

```sh
cargo run --release -p ktann-benchmarks --bin ktann-bench -- \
  run --backend rocksdb --profile smoke --output rocksdb-smoke.json
```

Each scenario runs in a fresh subprocess to isolate metrics, Partition Cache,
and process peak RSS. `--scenario NAME` selects one scenario;
`--worker-threads N` sets each worker's Tokio thread count. Reports include the
resolved configuration and the full parent command needed to reproduce a run.

| Profile | Workloads | Purpose |
| --- | --- | --- |
| `smoke` | Deterministic synthetic ANN, cache-disabled ANN, 95/5 and hot 50/50 search/update, saturated backend admission, import-to-search lifecycle | Functional CI checks |
| `full` | Synthetic workloads plus SIFTsmall, Fashion-MNIST, clustered, skewed, and duplicate-heavy inputs | Performance comparisons on an otherwise idle host |
| `large` | Cohere 1M cosine and SIFT1M L2 quality curves | Recall versus search cost on supplied ground truth |

### FoundationDB

FoundationDB requires a reachable cluster, its native client library, and
`fdbcli` on `PATH`. Set `FDB_CLUSTER_FILE` when needed. The runner obtains the
server version through `fdbcli --exec status json` and clears each worker's
process-unique Backend Namespace before exit.

```sh
cargo run --release -p ktann-benchmarks \
  --no-default-features --features foundationdb --bin ktann-bench -- \
  run --backend foundationdb --profile smoke --output foundationdb-smoke.json
```

The `blocking_resource_limit` scenario setting applies only to RocksDB's native
blocking actor bound.

### Large datasets

Prepare the pinned files described in [datasets/README.md](datasets/README.md).
Use `KTANN_BENCH_DATASET_CACHE` for a persistent cache; the default is
`/tmp/vectordb_bench/dataset`.

```sh
cargo run --release -p ktann-benchmarks --bin ktann-bench -- \
  run --backend rocksdb --profile large --worker-threads 8 \
  --output rocksdb-large.json
```

Each curve uses `k=100`, 1,000 held-out queries, 1,000 warmups, 16 concurrent
clients, and 10,000 measured operations per beam. Repeated queries reduce timing
noise; they do not add independent recall samples. Import batches contain 50
records. The leaf beam sweep is
`1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 192, 256, 384`.

Runtime and Logical Index settings use library defaults, including Partition
Cache sizing, maintenance concurrency, retry limits, and partition occupancy.
Search Budget limits and the k-derived exact-rerank policy remain fixed across
the curve; the effective rerank limit is 125. Reports record resolved settings,
so check them before comparing runs. Each worker requires at least three
searchable topology levels and variation in both recall and visited Leaf Entry
work; shallow or saturated curves fail validation.

The [large ANN workflow](../.github/workflows/large-ann-quality.yml) runs weekly
and on manual dispatch, validates datasets, and uploads reports. Use a fixed,
otherwise idle host for performance comparisons.

### Diagnostic options

`--write-beam-size N` overrides the write routing beam. Large runs also accept
`--base-vectors N`, `--query-vectors N`, `--query-offset N`, and
`--max-partition-entries N`. Resolved overrides are recorded in each report.

These import options apply to `--profile large` or an explicitly selected
`--scenario import-to-search-lifecycle`:

| Option | Meaning |
| --- | --- |
| `--maintenance-workers N` | Import Runtime maintenance workers; zero is allowed |
| `--import-max-in-flight-batches N` | Positive Import Session concurrency ceiling |
| `--import-batch-size N` | Positive records per atomic batch |
| `--import-backlog-watermark N` | Positive process-local Fixup Backlog watermark |

After immediate search, the lifecycle runner reopens the Logical Index with
its configured convergence workers, allowing an import with maintenance disabled
to reach the stable search phases. Import Sessions adapt concurrency below the
configured ceiling; see [ADR 0022](../docs/adr/0022-feedback-controlled-import-admission.md).

## Measurements

Reports contain one tagged payload per scenario: `steady_state`, `lifecycle`,
or `quality_sweep`. Configuration includes backend mutation limits, physical
key-prefix charges, cache limits, Search Budgets, and beam overrides.

Steady-state measurements exclude dataset loading and validation, index creation,
batch import, invariant verification, oracle construction, request materialization,
cache warmup, and draining warmup's maintenance backlog. A second invariant audit
runs after measurement. Import diagnostics are written to stderr and consumed
before later measurement phases so their accounting remains disjoint.

Reports include:

- Attempted, accepted, rejected, and failed operations by search/write class,
  accepted-operation throughput, and accepted end-to-end latency distributions.
- Recall@k against metric-specific brute-force truth for immutable ordinary ANN
  scenarios, or supplied ground truth for large datasets.
- Search Budget usage and exhaustion, visited Leaf Entries, approximate-selection
  and exact-reranking latency, and Partition Cache activity.
- CPU, process peak RSS, backend logical reads/scans/items/bytes, transaction
  attempts and commit outcomes, and attempted mutation operations/bytes.
- Admission and blocking-resource waits, operation-attributed write attempts and
  native commit waits, Fixup steps, and entries moved by split/merge drains.
- Logical write amplification: attempted mutation operations and bytes per
  successful public write, including retries and failed work.

Foreground timing ends when the last public operation completes. Backend and
maintenance counters include the subsequent bounded drain. Mutation stage
latencies include failed/cancelled work and overlapping operations; they are
aggregate service/wait times, not percentages of wall time. Logical scans and
write amplification do not measure physical storage IO or engine amplification.

Search Budget configuration distinguishes the Runtime default, optional request
override, and effective limit for `scanned_tree_keys`, `visited_partitions`, and
`exact_rerank_candidates`. Exact reranking is engine-sized, with no request
budget override. `visited_leaf_entries` counts uncapped filtering and
approximate-selection work; it is separate from records loaded for exact
reranking. The leaf beam is also a separate traversal setting.

Peak RSS always covers the entire worker, including setup and warmup. It is a
process high-water mark, not a phase-local or additive measurement.

### Quality curves

A `quality_sweep` records the ordered beam sweep and one measurement point per
beam, plus Tree count, Partition Header count, maximum level, and partitions by
level from the invariant audit. Construction spans import through the first
complete maintenance-converged topology audit. Import and convergence resources
are reported separately; concurrent maintenance is already included in import.
Dataset/index creation, oracle preparation, metric rendering, and query warmup
are outside construction timing. Construction RSS still includes dataset loading.
Raw import metrics are retained through bounded convergence and released before
queries.

Compare actual operating points at equal or better recall. Interpolated curves
do not establish latency or throughput non-regression. Per-point peak RSS cannot
isolate individual beams within the worker and is excluded from comparison.

### Admission saturation

`backend-admission-saturated` submits fixed concurrent waves, draining each
before starting the next. A valid run requires at least 100 accepted searches
and 100 accepted writes, with 25–75% overall rejection. Smoke measures 640
operations; full measures 2,000. Runs outside this region fail validation.

### Import-to-search lifecycle

Each lifecycle worker creates a fresh Backend Namespace and Logical Index and
prepares its dataset, requests, and exact oracle before timing. The continuous
case starts before the first `ImportSession::submit` and ends after warmed search:

1. `import` ends after `ImportSession::finish` and includes concurrent maintenance.
   Finish is a batch-outcome barrier, not a topology-convergence barrier. The
   scenario fails unless every submitted record is accepted.
2. `immediate_search` runs the fixed queries before the runner drives convergence.
3. `convergence` uses bounded public `verify` and `search` calls, requires unchanged
   topology across three observations, drains the observed Fixup Backlog, and
   rechecks counts. `from_import_finish_seconds` includes immediate search.
4. `cache_reset` shuts down the Runtime and reopens the same Logical Index in a
   new Runtime, giving the cold pass an empty process-local Partition Cache.
5. `stable_cold_search` runs the queries once; `stable_warm_search` repeats them.

Case wall time, CPU, and backend IO include all phases and intervening harness
work. Unattributed overhead is derived by subtracting named phases. Evaluate
import throughput, failures, convergence, recall, latency, CPU, and IO together;
fewer retries alone do not establish an improvement.

## Compare

Capture baseline and candidate on the same otherwise idle host with matching
build settings, compiler, Rust flags, worker count, backend identity and limits,
scenario configuration, and dataset checksum:

```sh
cargo run --release -p ktann-benchmarks --bin ktann-bench -- \
  compare --baseline baseline.json --candidate candidate.json \
  --output comparison.json
```

The comparator rejects different schemas, scenario sets, inputs, or
hardware/runtime fingerprints. Default regression thresholds are:

| Measure | Threshold |
| --- | --- |
| p95 latency, throughput, CPU, peak RSS, logical write amplification | More than 20% relative regression |
| Mean recall | More than 0.02 absolute drop |
| Overall admission rejection rate | More than 0.05 absolute increase |

Latency also requires at least a 1 ms absolute p95 increase. Rejection is compared
across the fixed operation mix because permit allocation between searches and
writes can vary. Override thresholds with `--maximum-relative-regression`,
`--maximum-recall-drop`, and `--maximum-rejection-rate-increase`; all take fractions.

Quality sweeps compare corresponding beam points. Lifecycle comparisons include
import throughput, submit latency, failures, waits, finish-to-stable time, search
phases, and backend IO. Preserve the exact source revision and patch used for
local baselines; `git_revision` has a `-dirty` suffix when the workspace changes.

## VectorDBBench

`ktann-vdbbench-bridge` owns one Runtime, backend, and Logical Index across
VectorDBBench's loader, optimizer, and search workers. The Python adapter and
process tests live in the [VectorDBBench repository](https://github.com/Lloyd-Pottiger/VectorDBBench).

The bridge supports RocksDB and FoundationDB, unfiltered single-tenant IDs-only
L2/cosine search, and signed 64-bit record IDs. Search uses public API defaults;
overrides and native diagnostics appear in a companion report without changing
canonical VectorDBBench metrics.

Protocol version 1 uses length-prefixed JSON over a Unix socket: a four-byte
big-endian length, at most 8 MiB per frame, and at most 128 connections. Inserts
commit at most 50 records per batch and finish the Import Session before success.
Do not automatically replay unknown outcomes. Optimize verifies the exact record
count and waits for no actionable or transitional partitions within a deadline.

```sh
cargo build --release -p ktann-benchmarks --bin ktann-vdbbench-bridge
```

For FoundationDB, add `--all-features` and configure its native client and cluster.
Install the VectorDBBench checkout with `pip install -e .`; its
`vectordb_bench/backend/clients/ktann/README.md` documents startup and CLI usage.
Run its process tests from that checkout:

```sh
export KTANN_BRIDGE_BIN=/path/to/ktann/target/release/ktann-vdbbench-bridge
python -m unittest discover -s tests -p 'test_ktann_bridge.py' -v
KTANN_TEST_BACKEND=foundationdb python -m unittest discover \
  -s tests -p 'test_ktann_bridge.py' -v
```

Use a fresh bridge and dedicated backend location per case. Shutdown finishes
import, drops the Logical Index, and removes the socket. After a crash, confirm
the old process has exited before removing its stale socket; startup never
unlinks a preexisting socket.

### Offline refinement after import

Quality sweeps accept `--refinement-rounds 0..5`. Omit the flag for the ordinary
import baseline; `0` recomputes centroids without relocation rounds, and `2` or
`5` requests bounded local refinement. Every path uses ordinary import and settles
topology until maintenance is drained before refinement. The runner issues no
other index operations during refinement and starts measured queries afterward.
The existing tree's partition IDs and topology remain fixed.

Reports record the requested rounds, 32 neighbor centroids, the 32 GiB refinement
input limit, completed rounds and moves. The input limit covers loaded vectors,
IDs, centroids and topology representation; numerical workspace and move lists
add memory. Construction wall time, CPU and backend work include import,
maintenance convergence and refinement. The convergence phase includes refinement.
Compare the ordinary baseline and each refinement setting on identical import
and search parameters; zero rounds is a centroid-recomputation control. Existing
bulk-builder research measurements use a different initialization/publication
pipeline and do not establish quality or performance for this API. Archive the
executable and source/binary hashes before timing under the shared resource lock.
