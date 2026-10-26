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
`--max-leaf-partition-entries N` and `--max-internal-partition-entries N`. Resolved overrides are recorded in each report.

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
fixture manifest with the reports. Reports use schema version 9.

Large search diagnostics accept `--query-concurrency 1,4,16,4,1` and
`--leaf-beam-size 32`. The worker imports and verifies the index once, then
warms and measures each client count in order on that same index. Repeated counts
help detect drift; each point reports its own concurrency, latency, CPU, recall,
cache observations, backend logical IO, and admission waits. Without these
options the standard beam curve and 16 clients are unchanged. A single beam
checks complete search and recall samples without requiring a quality curve.
`--measured-operations N` overrides the positive timed operation count; omitting it
keeps the selected profile default. Use the same count for compared runs; for
quality scenarios, an integer multiple of the query window gives equal weight
to each query. Warmup and full quality diagnostics remain separate.
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
[the mutation contract](../docs/design/api.md).

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
import throughput, batch latency, failures, waits, load-completion-to-stable time, search
phases, and backend IO. Preserve the exact source revision and patch used for
local baselines; `git_revision` has a `-dirty` suffix when the workspace changes.

## VectorDBBench

`ktann-vdbbench-bridge` owns one Runtime, backend, and Logical Index across
VectorDBBench's loader, optimizer, and search workers. The Python adapter and
process tests live in the [VectorDBBench repository](https://github.com/Lloyd-Pottiger/VectorDBBench).

The bridge uses Bulk Build with refinement. It supports
RocksDB and FoundationDB, single-tenant IDs-only L2/cosine search, signed 64-bit
record IDs, and an optional inclusive `id_min` predicate. This supports Cohere
1M unfiltered, 1% excluded, and 99% excluded cases, plus custom ANN performance
cases. Incremental online and read/write workloads are not supported.

### Lifecycle and controls

Insert requests capture bounded batches through `PreparedInputWriter`, which
writes canonical source frames, hashes, and receipt-time sort runs. A successful
insert acknowledges capture, not per-batch durability or publication. No
`batch_mutate` path is used. Duplicate IDs can be detected when Optimize builds
the index rather than when a batch is received; unknown inserts must not be
replayed automatically.

Optimize checks the expected record count and seals the Input Snapshot. All
routing modes refine the forest before serving encoding, loading, exact validation
and atomic publication. Global modes then prepare their in-process router. Core artifact validation
owns topology correctness; the bridge admits search after query preparation. Repeated successful Optimize
returns the existing result without rebuilding or refining.

The bridge retains sealed input and the same core Bulk Build job for retries.
Cancellation and transient build failures resume that job; terminal preparation
errors such as insufficient memory remain failed and require a fresh case.
Publication can precede a cleanup or router error, so an unsuccessful Optimize
never implies that no index exists. Search stays closed until all preparation
succeeds. Shutdown cancels at operation checkpoints and waits for admitted work
before dropping the index; socket timeouts do not release exclusive access.

Reset requires `refinement_rounds` (0..5) and `refinement_memory_bytes` (positive).
The Python client defaults to two rounds and 32 GiB. Zero rounds refreshes
centroids without moving records. The memory ceiling accounts for retained
inputs, topology and refinement planning; it is not a process RSS limit, and
construction/sort/serving/graph budgets remain separate. Reset owns the refinement
budget in every mode, including routing experiments: their `build_memory_bytes`
continues to configure the other build stages. Reports record the effective
`bulk_build.refinement_rounds` and `bulk_build.refinement_memory_bytes`.

Construction uses IndexConfig defaults (64/512 partition entries), a sample cap
of 1024, 256 MiB tree memory and 64 GiB tree scratch. Only an explicit leaf-beam
option changes the search defaults. Optimize returns readiness and record count;
per-level Header statistics and an independent Header-slot limit are not part of
the bridge contract.

### Running

```sh
cargo build --release -p ktann-benchmarks --bin ktann-vdbbench-bridge
target/release/ktann-vdbbench-bridge --backend rocksdb \
  --socket /tmp/ktann-bench.sock --database /tmp/ktann-bench-db \
  --report /tmp/ktann-bridge.json
```

Use a fresh bridge and dedicated backend location for each case. The bridge
creates its own temporary workspace and removes it on normal exit. Set `TMPDIR`
to choose a disk with sufficient space. The workspace stays alive across Optimize
retries within the same process. For FoundationDB, build with `--all-features`
and configure its native client and
cluster; `--database` then specifies the backend namespace. Startup never unlinks
a preexisting socket. After a crash, confirm that the old process exited before
removing its stale socket. Reports remain caller-owned; temporary source snapshots
are removed on normal exit. A crash may leave temporary files behind; core-owned
preparation files are reclaimed after publication.

From the VectorDBBench checkout with its dependencies installed:

```sh
vectordbbench ktann --socket-path /tmp/ktann-bench.sock \
  --dataset-identity cohere-1m --case-type Performance768D1M \
  --load-concurrency 1 --insert-batch-size 50 --k 100 \
  --refinement-rounds 2 --refinement-memory-bytes 34359738368
```

The private protocol is v1, with a big-endian u32 frame length, an 8 MiB frame
limit and at most 128 connections. Control/search and responses are JSON.
Binary inserts carry `KTI` plus byte version 1, big-endian u32 count/dimension,
little-endian i64 IDs and row-major f32 values, with at most 50 records per batch.
The reset request carries `refinement_rounds` and `refinement_memory_bytes`.
JSON inserts are rejected; update the client and bridge together.

### Measurements and tests

Canonical insert time measures input capture. Optimize time includes Bulk Build,
refinement and router preparation; canonical total load time includes both phases.
Native reports identify `build_mode: bulk`, effective refinement controls,
`phases.input_staging_seconds`, total `phases.optimize_seconds`, and detailed Bulk
Build stages. Artifact refinement is included in `bulk_build.forest_seconds`;
there is no separately measured refinement duration. Core stage timings describe
the successful completion call (reused stages have zero duration); Optimize timers
accumulate failed attempts as well as successful ones. Successful repeated
Optimize calls leave those measurements unchanged. Compare identical datasets,
query sets, construction/refinement settings and backend budgets; do not compare
receipt-only timing with committed online insertion throughput.

Run Python process tests against the matching bridge:

```sh
export KTANN_BRIDGE_BIN=/path/to/ktann/target/release/ktann-vdbbench-bridge
python -m unittest discover -s tests -p 'test_ktann_bridge.py' -v
KTANN_TEST_BACKEND=foundationdb python -m unittest discover \
  -s tests -p 'test_ktann_bridge.py' -v
```

Keep local reports under `.benchmark-data/results/` with source revision, binary
hash, settings, dataset identity and raw evidence. These checks do not establish
large-dataset throughput, recall, peak memory or multi-host filesystem behavior.

### Offline refinement after import

Quality sweeps without `--routing-experiment` accept `--refinement-rounds 0..5`.
Omit the flag for the ordinary
import baseline; `0` recomputes centroids without relocation rounds, and `2` or
`5` requests bounded local refinement. These runs use ordinary import and settle
topology until maintenance is drained before refinement. The runner issues no
other index operations during refinement and starts measured queries afterward.
The existing tree's partition IDs and topology remain fixed.

Reports record the requested rounds, 32 neighbor centroids, the 32 GiB refinement
input limit, completed rounds and moves. The input limit covers loaded vectors,
IDs, centroids and topology representation; numerical workspace and move lists
add memory. Construction wall time, CPU and backend work include import,
maintenance convergence and refinement. The convergence phase includes refinement.
Compare the ordinary baseline and each refinement setting on identical import
and search parameters; zero rounds is a centroid-recomputation control.
Routing experiments use the Bulk Build artifact path described below and are
not interchangeable baselines for this import-and-refine path.

## Complete Bulk Build probe

`cargo run --release -p ktann-benchmarks --bin ktann-bulk-build -- <sift-directory> <new-output-directory> [record-limit]`

The SIFT directory contains `sift_base.fvecs`, `sift_query.fvecs`, and
`sift_groundtruth.ivecs`. Default count is 1,000,000. The probe streams a source
snapshot, reserves a RocksDB index, runs preparation/loading, publishes through
exact validation, checks point reads, and measures held-out top-10 queries.
Official recall is reported only for the full population. `report.json` separates
phase wall times, forest and serving scratch IO, construction seed, and query
latency; use `/usr/bin/time -l` for process resource
counts on macOS. Reports retain the source and RocksDB database, while successful
publication reclaims worker-owned artifacts. Run on an idle host. The result
does not establish distributed throughput or an online insertion speedup.

Bulk Build bridge reports include `serving_detail` wall times for source
sorting, topology sorting, exact joins/encoding, output merging, Synopsis
reduction, and final emission/sealing, plus total and peak serving scratch bytes.
`ready_peak_rss_bytes` captures the bridge process's resident-memory high-water
mark after successful publication, cleanup and router preparation, before queries.
The whole-run peak remains separate and may include later query cache growth.
A resumed worker reports no detail for an already accepted serving artifact.

Core and benchmark SHA-256 use sha2 0.11's default runtime CPU detection,
including AArch64 SHA-2 instructions with a software fallback. Artifact framing,
per-frame checksums, whole-file hashes and publication verification are unchanged.
Performance comparisons must record the CPU architecture and resolved dependency
versions; the gain from hardware hashing is platform dependent.

The four capacity overrides are independent. Unspecified values retain the
profile's configured values (the large profile uses IndexConfig defaults).
Lowering a maximum below twice its minimum requires explicitly lowering that
minimum as well; the CLI and bridge do not infer a different capacity pair. Reports record all four values and comparison
rejects mismatched configurations. The VectorDBBench bridge's
`reset` accepts optional `min_leaf_partition_entries`,
`max_leaf_partition_entries`, `min_internal_partition_entries`, and
`max_internal_partition_entries`; omitted fields use IndexConfig defaults.
Bridge reports record all four resolved values. Invalid pairs fail before reset
changes storage. Keep rerank budgets fixed when comparing capacity choices.


## Frozen-layout routing experiments (issue #195, step 9)

These experiments hold the dataset and leaf layout fixed during query comparisons.
The graph indexes they build support writes after publication; the frozen workload
is an experimental control, not an index capability.

`ktann-bench run --routing-experiment JSON` uses production Bulk Build and
publication, including optional artifact refinement **before** graphs are built.
It is supported by `large` quality scenarios and `quality-routing-smoke`. The
ordinary online scenarios retain their existing import path. JSON rejects unknown
fields; reports retain every resolved experiment setting. For example:

```sh
cargo build --release -p ktann-benchmarks --bin ktann-bench
mkdir -p .benchmark-data/results
B=target/release/ktann-bench
$B run --backend rocksdb --profile smoke --scenario quality-routing-smoke \
  --routing-experiment '{"mode":"exact","m":16,"ef_construction":200,"graph_seed":7,"ef_search":64}' \
  --rocksdb-path .benchmark-data/results/routing-fixture \
  --leaf-beam-size 4 --refinement-rounds 0 \
  --output .benchmark-data/results/routing-exact.json
$B run --backend rocksdb --profile smoke --scenario quality-routing-smoke \
  --routing-experiment '{"mode":"partition-local-hnsw","m":16,"ef_construction":200,"graph_seed":7,"ef_search":64}' \
  --rocksdb-path .benchmark-data/results/routing-fixture --reuse-index true \
  --leaf-beam-size 4 --refinement-rounds 0 \
  --output .benchmark-data/results/routing-local.json
$B summarize-routing --inputs .benchmark-data/results/routing-exact.json,.benchmark-data/results/routing-local.json
```

Use a **new** fixture directory for each construction configuration. Reuse checks
base-vector identity, capacities, graph parameters/seed, refinement and training
sample size. Query windows are independent of fixture identity, allowing held-out
validation with `--query-offset` and `--query-vectors`. The dataset checksum in each
report still includes the actual query window. Rotation seed and source checksum
are recorded on construction. `routing_setup.leaf_layout_sha256` streams final
leaf identities, centroids and membership bytes; compare this fingerprint when
claiming the same layout across routing modes or internal capacities.

Modes are `exact` (production exact tree routing),
`partition-local-hnsw` (production local graphs), `global-exact`, and
`global-hnsw`. Global modes are compiled only through the core `benchmark`
feature and keep a separate, non-persisted global baseline. They use final stored
centroids and the existing USearch engine. Selected leaves enter the production
snapshot/leaf-scan/overlap/rerank pipeline. Global mode setup is untimed; query
routing, private scratch allocation, dispatch and the production query are timed.
This path preprocesses the query for global routing and again for production leaf
scanning; its timings include that benchmark integration overhead.

The experiment JSON accepts:

| Field | Default | Meaning |
| --- | ---: | --- |
| `mode` | `exact` | Explicit routing policy |
| `m`, `ef_construction` | 16, 200 | Persisted graph construction parameters |
| `graph_seed` | 0 | u64 encoded little-endian into the first 8 of 32 seed bytes |
| `ef_search` | 200 | Requested expansion; actual per-parent effective values are recorded |
| `routing_working_bytes` | 268435456 | Production per-operation routing admission |
| `global_working_bytes` | 268435456 | Global centroids/graph plus one query/build working set |
| `sample_items` | 1024 | Existing construction training sample control |
| `build_memory_bytes` | 268435456 | Each existing construction/sort/serving/graph/refinement worker bound |
| `build_scratch_bytes` | 68719476736 | Existing scratch and per-artifact bound |

Existing independent `--min-leaf-partition-entries`, `--max-leaf-partition-entries`,
`--min-internal-partition-entries`, `--max-internal-partition-entries`, beam,
concurrency, refinement, Partition Cache and RocksDB block-cache controls still
apply. The rerank budget remains engine-owned and fixed for a given k. These are
experimental settings, not selected production defaults. Construction resources
are per existing worker phase, not a whole-process RSS limit.

Global baselines support one Tree Key, at most 65,536 leaves (the shared local
core/USearch limit), and 262,144 allocated Header slots. An explicitly selected
global mode fails with a reason if its bound cannot be met; it never changes data
size, mode or ef. A global exact reference used only for quality diagnostics may
instead be recorded as skipped with a reason. The streaming fingerprint has the
same Header-enumeration ceiling. Global retained bytes and setup time are reported;
private query work is bounded separately for each concurrent caller. A global
operation gate uses the Runtime foreground limit and retains its permit through
native work and the production search. Total
residency also includes runtime caches, backend/OS caches, dataset/oracle storage
and other process state.

### Measurement semantics and step 10 inputs

Every worker reports construction phases, per-graph build/encode durations,
native graph construction peaks (excluding Rust centroids/encoding), encoded graph
payload bytes, process peak RSS, topology, and measured query latency/QPS/CPU.
Routing counters distinguish distances, graph expansions, per-level parent visits
(levels above 32 share `33+`), global tree-beam merge candidates, restore counts,
encoded miss bytes, and accounted decoded body bytes. Repeated loads count again.
Compute/merge/restore durations are **wall time**, not thread CPU. Process CPU is
available over the complete timed interval. Logical read bytes are returned KV
bytes; process physical IO is OS-dependent and unavailable values stay null.
Graph payload size excludes KV keys and storage-engine amplification. Existing
leaf-entry scan and truncation metrics remain separate from graph work.

All timing/resource snapshots finish before diagnostic replay. For each unique
query, `quality_diagnostics` records true-neighbor leaf coverage, local RaBitQ
candidate coverage, rerank candidate coverage, final recall, and optional overlap
with global exact centroid top-beam. Per-truth `truth_leaves` rows also record the
persisted leaf identity and occupancy, its one-based global centroid rank, and
whether the leaf was scanned and the neighbor returned. Rank is null when the
bounded global reference is unavailable. These diagnostics require an immutable
fixture and run outside timing. These are distinct quantities. Timed queries
also retain individual recall values in completion order. Offline-refinement move
and completed-round counts are null when the artifact API does not expose them;
requested rounds remain in configuration. Setup and diagnostics are never called
cold-cache measurements. In particular, zero warmup does not flush RocksDB/OS
caches, and later beam points can inherit the worker's Partition Cache. Use fresh
workers and cache capacities (including zero) to measure loading and turnover;
no system cache purge is performed.

For step 10, use the existing `large` dataset cache and scenarios
`quality-cohere-1m` / `quality-sift-1m`, k=100, separate calibration/validation query
windows, multiple graph seeds, independent capacity pairs, beam/ef, cache budgets
and client counts. Keep the leaf-layout fingerprint and query checksum with each
result. `summarize-routing` emits TSV rows with mean/min final recall, QPS, p95,
logical bytes and process-lifetime peak RSS before diagnostics (not per-point
RSS); it does not declare a winner or compare unequal recall.
The existing `compare` command remains a strict identical-input regression check.
Select matched final recall rows (including the >90% target) before comparing
capacity/performance. No full 1M matrix or default selection is part of step 9.

The VectorDBBench bridge accepts the same `--routing-experiment JSON` at startup;
the Python client/protocol is unchanged. Its requested refinement runs in artifact
preparation before initial graph construction. The native runner owns the detailed per-query oracle
and routing metrics; bridge reports remain companion metrics, with VectorDBBench
owning canonical recall. Use native runner controls for cache matrices.

### Local-graph update history

`ktann-mutable-graph` exercises ordinary writes and real background split/merge
maintenance on a bounded prefix of a cached `sift-1m` or `cohere-1m` dataset:

```sh
KTANN_BENCH_DATASET_CACHE=/Users/lloyd/projects/ktann/.benchmark-data/vectordb_bench/dataset \
  cargo run --release -p ktann-benchmarks --bin ktann-mutable-graph -- \
  sift-1m .benchmark-data/results/mutable-sift-16384 16384 8
```

The output directory must be new. The default is 16,384 records and eight rounds.
An optional final `INTERNAL_MIN:MAX` argument overrides the default `32:128`;
representative wider-parent runs use `64:512`, for example append `64:512` after
the record and round counts.
Each round rotates its update/delete windows, updates 10% from reserved dataset
rows outside the initial prefix, deletes 10%, settles maintenance, reinserts
that many vectors with fresh IDs, and settles again. One hundred held-out queries
starting at query offset 100 are independent of the inserted vectors. The runner
uses each corpus's metric (SIFT L2, Cohere cosine), graph seed `[19; 32]`, M16,
efConstruction64, efSearch64, and fixed exact-rerank budget125. Before measured updates it selects the first
beam in 8/16/32/64/128/256 reaching mean recall92% on a separate 50-query
calibration window (offset0), then freezes that beam for all evaluated phases.

`report.json` records initial, per-round, and fresh-rebuild recall@100 against
current exact ground truth; per-query recall, minimum/p05/p50 recall, query
p50/p95 and CPU; process RSS high-water marks;
uncached graph/topology verification and record coverage; attempted logical KV
mutation bytes, commits and conflicts; maintenance step counts; and mutation
throughput including maintenance. Initial import timing runs through settlement.
Oracle computation, paired Exact routing, untimed true-neighbor leaf/candidate
coverage diagnostics, and invariant verification are outside HNSW query timing.
The large loader decodes all one million vectors before selecting a prefix, so
cumulative peak RSS includes that full decode, the reserved update pool and oracle
allocations; it is not graph working memory.
Logical KV bytes do not measure native physical writes. These experiments cover
the reported size and update history and do not establish a production SLA.

### Adaptive write routing

Graph-enabled indexes use exact scoring below 1024 internal nodes and HNSW at
or above 1024, independently for each snapshot-authoritative body. There are no
write-routing experiment switches. Query behavior and budgets are unchanged.
See [foreground write routing](../docs/design/partition-local-hnsw.md#foreground-write-routing)
for the policy, rationale and measurement evidence.
