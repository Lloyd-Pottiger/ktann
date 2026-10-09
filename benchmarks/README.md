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

The raw RocksDB block cache is independent of the decoded Partition Cache.
`--rocksdb-block-cache-bytes N` sets its capacity (default 8 MiB); the worker
prints occupied and pinned bytes to stderr after the final full audit; those
values include audit reads and do not describe individual timed points.
`--warmup-operations N` sets
per-point untimed warmup, including zero for diagnostics. Each timed point
records `physical_read_bytes` and `physical_write_bytes` from OS process
accounting (Darwin `proc_pid_rusage`, Linux `/proc/self/io`). These are physical
bytes charged to the application process, rather than logical backend reads;
for FoundationDB they exclude the separate server process. Zero physical reads
can also mean OS page-cache hits, so combine this evidence with cache occupancy
and repeated stable latency before describing a workload as resident.

For controlled large RocksDB comparisons, create a fixture with
`--rocksdb-path /absolute/new-directory` and one explicit `--scenario`.
The directory must not already exist. Subsequent runs use that same path with
`--reuse-index true`. The fixture checks the dataset checksum, index settings,
write beam and refinement inputs, then fully verifies the persisted topology
before queries. Reused reports set `construction: null`; comparison omits
construction costs and rejects mixing fresh and reused runs. Cache capacity,
client counts, warmup and read beam can
change without reconstructing the tree. Keep both comparison binaries and the
fixture manifest with the reports. Reports use schema version 8.

Large search diagnostics accept `--query-concurrency 1,4,16,4,1` and
`--leaf-beam-size 32`. The worker imports and verifies the index once, then
warms and measures each client count in order on that same index. Repeated counts
help detect drift; each point reports its own concurrency, latency, CPU, recall,
cache observations, backend logical IO, and admission waits. Without these
options the standard beam curve and 16 clients are unchanged. A single beam
checks complete search and recall samples without requiring a quality curve.
`--partition-cache-bytes N` sets the decoded partition cache capacity for any
profile; zero disables it. This cache excludes raw vectors read during reranking
and does not replace RocksDB's block cache or FoundationDB's server cache.

```sh
KTANN_BENCH_DATASET_CACHE=/path/to/dataset-cache \
  cargo run --release -p ktann-benchmarks --bin ktann-bench -- \
  run --backend rocksdb --profile large --scenario quality-sift-1m \
  --worker-threads 8 --query-concurrency 1,4,16,64,16,4,1 \
  --partition-cache-bytes 1073741824 --leaf-beam-size 32 \
  --output sift-search-concurrency.json
```

These import options apply to `--profile large` or an explicitly selected
`--scenario import-to-search-lifecycle`:

| Option | Meaning |
| --- | --- |
| `--maintenance-workers N` | Import Runtime maintenance workers; zero is allowed |
| `--import-batch-size N` | Positive records per atomic batch |

After immediate search, the lifecycle runner reopens the Logical Index with
its configured convergence workers, allowing an import with maintenance disabled
to reach the stable search phases. Online loads submit batches sequentially; see
[ADR 0025](../docs/adr/0025-caller-owned-online-batch-submission.md).

## Measurements

The preparation/construction gate has a separate streaming SIFT probe:

```sh
cargo run --release -p ktann-benchmarks --bin ktann-bulk-construct -- \
  "$KTANN_BENCH_DATASET_CACHE/sift1m/sift_base.fvecs" \
  /path/to/new-output-directory
```

An optional final argument limits records for a smoke run. The output directory
must be new. It contains sealed `input/` and `plan/` artifact directories and
`report.json`, including source SHA-256, options, separate snapshot/construction/
artifact-verification timings, peak scratch bytes and cumulative scratch writes.
The probe uses one Tree Key and streams source vectors; payloads and backend IO
are outside this probe's scope. Completed artifacts can be reopened against saved
manifests; interrupted sorting runs are recomputed. Source snapshots and final
artifacts have separate 32 GiB quotas in addition to the sort scratch quota.
These results cannot establish end-to-end speedup or serving
recall: loading, validation, publication and query measurements are still required.

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
case starts before the first direct `batch_mutate` call and ends after warmed search:

1. `import` ends after completion of all direct batch calls and includes concurrent maintenance.
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

The bridge supports RocksDB and FoundationDB, single-tenant IDs-only L2/cosine
search, and signed 64-bit record IDs. Each record stores its ID in an `i64` filter
field. Search accepts an optional inclusive `id_min` threshold, evaluated by the
native exact predicate before candidate selection. This supports VectorDBBench
Cohere 1M unfiltered, 1% excluded, and 99% excluded cases. The bridge uses the
public IndexConfig, RuntimeConfig, and backend adapter defaults, including
partition entries 64/512. Only an explicit client leaf-beam option overrides
search defaults. Native diagnostics read the effective index and Runtime
configuration without changing canonical VectorDBBench metrics. Bounded
readiness probes used by Optimize are separate from measured searches.

Protocol version 2 uses a four-byte big-endian frame length over a Unix socket,
at most 8 MiB per frame and at most 128 connections. Control/search requests and
responses are JSON. Inserts carry `KTI` plus byte version 2, big-endian u32 record
count and dimension, then little-endian i64 IDs and row-major little-endian f32
vectors. JSON inserts are rejected; client and bridge must be updated together.
Inserts commit at most 50 records per batch and wait for the atomic batch result
before success in online mode. Bulk receipt acknowledges capture of the batch;
it is not per-batch durability or a published index.
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

### Bulk Build through VectorDBBench

Pass `--bulk-workspace /absolute/new/directory` to `ktann-vdbbench-bridge` to
measure the core Bulk Build path. Use a fresh bridge, RocksDB directory and
workspace for each case. The Python adapter and canonical runner remain unchanged.
Insert requests stage bounded batches of original IDs/vectors; they do not write
serving data. Optimize seals an InputSnapshot, reserves the Building index, runs
`run_worker`, and calls `publish` for exact validation and atomic activation.
Search remains unavailable until publication and topology readiness succeed.

The native report identifies `build_mode: bulk` and records input staging,
snapshot creation, preparation/loading, and validation/publication/cleanup times.
`committed_import_seconds` is null in this mode: canonical load time measures
input receipt, while canonical load plus optimize/index time covers the build.
Bulk receipt writes canonical source frames and their hashes directly through
`InputSnapshotWriter`, without a raw staging file or EOF rewrite. `snapshot_seconds`
now measures only final flush, fsync and sealing; record encoding/hashing is
included in receipt. The source snapshot and report remain caller-owned;
successful publication reclaims core attempts. Forest construction still begins
after EOF seals the source; this is incremental input preparation, not streaming
final tree assignment.
Construction uses the index's min/max defaults, sample 256, 256 MiB tree memory,
and 64 GiB tree scratch; the report records worker limits. No online insertion or
post-build refinement is substituted into this path.

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

### Multi-tree Bulk Build preparation probe

The optional fourth argument to `ktann-bulk-construct` selects the forest path:
`ktann-bulk-construct INPUT.fvecs OUTPUT_DIRECTORY RECORD_LIMIT FOREST_TREES`.
A positive tree count assigns the original SIFT ordinal modulo that count to an
I64 Tree Key field; `0` exercises the empty Tree Key through the forest path.
Omitting this argument runs the same forest pipeline with one empty Tree Key.
Both modes report global-sort and per-tree resource budgets. Reports include
separate global-sort and per-tree scratch peaks/write totals. Their quotas add
while both stages retain files. The probe verifies sealed topology framing but
does not measure backend loading, exact serving validation, publication, or recall.

Append `--serving` after `FOREST_TREES` to also execute exact joins and encode a
serving artifact. This probe uses a synthetic Logical Index ID and explicit
10,000-byte key / 100,000-byte value limits; it does not reserve or write a real
backend. The nested `serving` report separates encoding and full-file verification
time, output bytes, and scratch IO. The sort budget charges actual allocated
payload capacities and row slots, allowing small and large projections to share
a byte ceiling without allocating the maximum value size for every row.


## Complete Bulk Build probe

`cargo run --release -p ktann-benchmarks --bin ktann-bulk-build -- <sift-directory> <new-output-directory> [record-limit]`

The SIFT directory contains `sift_base.fvecs`, `sift_query.fvecs`, and
`sift_groundtruth.ivecs`. Default count is 1,000,000. The probe streams a source
snapshot, reserves a RocksDB index, runs preparation/loading, publishes through
exact validation, checks point reads, and measures held-out top-10 queries.
Official recall is reported only for the full population. `report.json` separates
phase wall times and query latency; use `/usr/bin/time -l` for process resource
counts on macOS. Reports retain the source and RocksDB database, while successful
publication reclaims worker-owned artifacts. Run on an idle host; the result is
not directly comparable to the file-only `ktann-bulk-construct` probe and does not
establish distributed throughput or an online insertion speedup.
