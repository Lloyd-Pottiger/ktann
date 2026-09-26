# KTANN plus benchmark bridge

This is a **benchmark-only** VectorDBBench integration, not a KTANN service API.
The supported upstream revision is
[`1760db148b951363f2282261f30179dfd2ce3790`](https://github.com/Lloyd-Pottiger/VectorDBBench/tree/1760db148b951363f2282261f30179dfd2ce3790).
The overlay adds only a client, connection/case configuration, DB enum/registry
entries and CLI registration. It does not change runner workloads or metrics.

See [VALIDATION.md](VALIDATION.md) for measured integration runs and verification
coverage, including the limits on interpreting their performance numbers.

## Ownership and supported workload

A separately launched `ktann-vdbbench-bridge` (a non-published binary in
`ktann-benchmarks`) owns the Tokio executor, FoundationDB network when selected,
KV adapter, Runtime, Logical Index and Import Sessions. Python objects contain
configuration only when pickled. `init()` opens a connection in the current
worker and closes it when the context exits; pickling a live client omits its
socket and lock. Loader, optimizer and search workers share one bridge. Searches
hold a shared lifecycle lock and run concurrently; inserts and optimize are
exclusive. Threaded upstream callers use their normal per-thread client copies.

Only unfiltered, single-tenant, IDs-only L2 and cosine ANN are supported. The
initial cases are Cohere 1M cosine (768 dimensions) and SIFT1M L2 (128 dimensions).
SIFT is a canonical **custom dataset** case, not the upstream capacity case:
upstream SIFT 500K has no published ground truth. Filtering, payloads, streaming,
capacity and search-only reuse are deliberately rejected. Use one case and a
fresh bridge process per run; reset twice in one process is an error.

Record IDs are signed 64-bit integers, encoded as exactly eight big-endian bytes
and decoded inversely. Integer coercions from floating point are rejected. The
same fixed index configuration (partition entries 32/128, write beam 8), Runtime
(2 maintenance workers, 512 MiB partition cache, 128 foreground slots, import
backlog watermark 1), Search
Budgets and runner settings apply to both backends. Search inherits KTANN's
public API defaults; the adapter has no separate default beam or budget values.
`--leaf-beam` and `--leaf-budget` explicitly override the beam and Leaf Entry
budget. The companion report records effective values. Exact reranking uses
KTANN's k-dependent default. Always publish the measured recall.

Each wire insert has at most 50 records and runs through a bounded Import
Session. The bridge finishes the session **before** returning a committed count;
there are no deferred insert failures hidden in optimize. A backlog watermark
of one pauses import while local maintenance is pending or running; this keeps
bulk import from accumulating enormous transitional leaves. The resulting wait
is truthfully part of canonical insert duration. The client may split
one upstream batch into these atomic batches and returns a non-retryable
`PartialInsertError` with the confirmed prefix if a later batch fails. A lost
response or unknown commit never causes automatic replay. Optimize therefore
begins with all acknowledged import batches finished, then probes for
demand-driven maintenance and performs bounded topology Header snapshots until
no transitional/actionable/oversized partitions remain and the vector count is
exact. It fails on invalid snapshots or its 3,500-second deadline;
`finish()` alone is never called topology readiness. Each snapshot reads at most
262,144 allocated Header slots in 256-key batches and retains at most 32
centroid probes for partitions needing maintenance. Snapshots use core codecs
and one read transaction, check root presence and summed Leaf Entry counts, and
report states/counts by level. They are explicitly **not full record-integrity
audits**: `Index::verify` separately checks every record and can dominate large
runs. Header polling does not scan Vector Records; occasional rediscovery searches
perform ordinary search work. Search Budgets bound logical work, not physical IO. Rediscovery runs only
after five seconds without Header progress and at most once per 30 seconds,
using beam 1 and 128-partition/128-Leaf-Entry budgets. It does not repeatedly run
the quality workload against oversized leaves while maintenance is progressing. Native backend transaction limits still apply, including
FoundationDB snapshot lifetime; a failed snapshot fails the run rather than
claiming readiness.

## Build and install

Requires Unix-domain sockets (macOS/Linux), Rust per the repository MSRV,
a C++ toolchain/libclang for RocksDB, Python 3.11+, and the pinned upstream
Python dependencies. Use an otherwise idle host for publishable measurements.
Budget at least 16 GiB RAM and 30 GiB free disk for these cases, plus persistent
backend storage and the FoundationDB server. The adapter caps bridge frames at
8 MiB and connections at 128; these are bounds, not a whole-host memory limit.
VectorDBBench and FoundationDB also consume memory in separate processes.

```sh
# In KTANN; the FoundationDB feature additionally needs its native client library.
export PATH=/Users/lloyd/.local/foundationdb/7.3.69/bin:$PATH
export FDB_CLUSTER_FILE=/Users/lloyd/.local/foundationdb/7.3.69/etc/fdb.cluster
export DYLD_LIBRARY_PATH=/Users/lloyd/.local/lib
export RUSTFLAGS='-L native=/Users/lloyd/.local/lib'
cargo build --release -p ktann-benchmarks --bin ktann-vdbbench-bridge --all-features

python3.12 -m venv /tmp/ktann-vdbbench-venv
. /tmp/ktann-vdbbench-venv/bin/activate
python benchmarks/vectordbbench/install.py ~/projects/VectorDBBench
pip install -e ~/projects/VectorDBBench
export IR_DATASETS_HOME=/tmp/ktann-vdbbench-ir
export MPLCONFIGDIR=/tmp/ktann-vdbbench-mpl
export LOG_FILE=/tmp/ktann-vdbbench.log
vectordbbench ktann --help
```

The installer checks the exact upstream HEAD, changes only its three registration
sites and CLI imports, and copies the owned `ktann/` module. Re-running it updates
the overlay. It never changes the checkout revision or installs a monkey patch.
RocksDB-only builds may omit `--all-features` and the FoundationDB environment.

FoundationDB requires a running cluster. Follow the repository's local server
instructions and check `fdbcli --exec 'status minimal'`. The bridge records the
linked client and connected server identities separately; build with the 7.3 API.
The launcher generates a unique `ktann-vdbbench-*` Backend Namespace. Never point
another process at the same namespace or RocksDB directory during a run.

## Fixed data and reproducible optimized runs

Source URLs, object revisions, lengths and checksums are pinned in
[`../datasets/cohere-1m.json`](../datasets/cohere-1m.json) and
[`../datasets/sift-1m.json`](../datasets/sift-1m.json). Acquisition commands and
provenance are in [`../datasets/README.md`](../datasets/README.md).
The preparation command can acquire missing files with `--download`; it always
validates source lengths and pinned SHA-256/S3 multipart-MD5 checksums. SIFT
conversion is bounded by 10,000-row chunks, preserves every vector, query, ID
and supplied top-100 neighbor, and writes converted-file SHA-256 checksums.
It does not truncate the million-vector ground truth to a 500K subset. The
pinned upstream runner may also acquire `scalar_labels.parquet` during Cohere
preparation; the unfiltered KTANN workload does not use those labels.

```sh
cache="$PWD/.benchmark-data/vectordb_bench/dataset"
sift="$PWD/.benchmark-data/vdbbench-sift1m"
python benchmarks/vectordbbench/datasets.py cohere-1m --cache "$cache" \
  --output "$PWD/.benchmark-data/vdbbench-cohere-provenance"
python benchmarks/vectordbbench/datasets.py sift-1m --cache "$cache" --output "$sift"

# Run each combination separately on an idle host; do not run backends concurrently.
python benchmarks/vectordbbench/run.py \
  --checkout "$HOME/projects/VectorDBBench" \
  --bridge "$PWD/target/release/ktann-vdbbench-bridge" \
  --cache "$cache" --sift "$sift" \
  --case sift-1m --backend rocksdb \
  --output "$PWD/.benchmark-data/results/vdbbench-sift1m-rocksdb" \
  --concurrency 1,2,4,8,16 --duration 30 --leaf-beam 32
```

Repeat with `--case cohere-1m` and/or `--backend foundationdb`, using a distinct
output directory each time and identical concurrency, duration, k and tuning.
The launcher uses upstream `--k 100 --load-concurrency 1 --insert-batch-size 50`,
performs checksum validation, launches the ordinary CLI and shuts down the bridge
in a `finally` block. It rejects failed canonical result labels and missing
concurrency results. Optimizer timeouts remain bounded by the bridge even if
the upstream case grants a longer allowance. Small synthetic tests below verify
protocol correctness; they are not large-scale performance evidence.

## Artifacts and metric meanings

All public results must say **KTANN plus benchmark bridge**. Report the two
backends separately. JSON serialization, socket IO, Python overhead, scheduling
and queueing are included in upstream measured latencies and QPS. Upstream
latency also includes Python input coercion before the wire-request timer.

| Artifact | Meaning |
| --- | --- |
| `canonical/KTANN/result_*.json` | Unmodified upstream insert duration, optimize duration, their sum `load_duration`, serial p95/p99, mean recall@k, maximum-QPS headline and every per-concurrency QPS/latency list |
| `canonical-p50.json` | Source-linked copies of upstream serial/per-concurrency p50; this pinned revision already supplies these fields |
| `bridge.json` | Labelled companion: KTANN phase timing, Search Budget totals/exhaustion, topology, process CPU/peak RSS, Backend IO, backend identity/limits, wire bounds/version |
| `clients/client-*.json`, `client-summary.json` | Per-worker and aggregate wire round-trip, JSON encode/decode, KTANN elapsed time, overhead, bounded latency histograms, continuous first-insert-through-final-search time |
| `invocation.json`, `dataset.json` | Exact commands, upstream/KTANN revisions, binary SHA-256, host/Python facts and data identity/checksums |
| `runner.log`, `bridge.log` | Failure and lifecycle diagnostics |

The client continuous interval starts on entry to the first bridge insert
request, before JSON encoding, and ends after decoding the final successful
search response, using this host's shared monotonic clock. Initial input
coercion before that first request is outside this interval. It includes
process handoff, optimize, all search stages and intervening idle time. Bridge
phase sums exclude Python and socket work. Its continuous interval starts on
entry to import and ends after KTANN search, so use the **client** interval for
the continuous bridge-client case. The client overhead is round-trip minus KTANN search time:
JSON + IPC + queue/lock/scheduling wait; it is **not pure socket latency**. Decode
and encode seconds are also reported independently. Log2-nanosecond histogram
p50 values are labelled bucket upper bounds; use copied canonical p50 values for
published serial/per-concurrency latency comparisons. Histograms use constant
memory rather than retaining every query duration.

Search Budget arrays are ordered Tree Keys, partitions, Leaf Entries, exact
reranks; exhaustion arrays add RaBitQ overlap truncation last. Backend IO is
logical adapter-boundary IO, not physical disk IO. CPU/RSS describe the bridge
process, not the separate Python runner or FoundationDB server. Shutdown captures
the companion before cleanup, deletes the dedicated Logical Index, shuts down
Runtime and releases native ownership, then unlinks only its own socket. The
RocksDB directory is retained under the result directory; obsolete SST bytes
may remain until compaction. SIGINT uses the
same path. SIGKILL cannot clean up; after confirming the old process is gone,
remove its stale socket and launch a fresh bridge with the same dedicated
backend location to reset the old Logical Index. The bridge never unlinks a
preexisting socket automatically.

## Tests and upstream contribution

```sh
export KTANN_BRIDGE_BIN="$PWD/target/release/ktann-vdbbench-bridge"
export PYTHONPATH="$HOME/projects/VectorDBBench"
python -m unittest discover -s benchmarks/vectordbbench/tests -v
KTANN_TEST_BACKEND=foundationdb python -m unittest discover \
  -s benchmarks/vectordbbench/tests -v
cargo test -p ktann-benchmarks --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

The process tests also execute the official CLI and assert canonical metric
relationships and retention of both requested concurrency results. They cover
a live client's pickle boundary, loader/optimizer handoff,
concurrent spawned search workers, both backend selections, socket collision,
clean and crash restart, native cleanup, shutdown with an incomplete frame,
signed IDs, frame limits and non-retryable errors.
The CLI takes the same registry/configuration path as other upstream adapters.

For an upstream PR, contribute `ktann/` as
`vectordb_bench/backend/clients/ktann/`, the enum/config/case registry entries and
CLI registration produced by `install.py`, plus the process-contract tests.
Keep the Rust bridge and dataset provenance here. Document the pinned bridge
binary/hash and version-1 protocol in the upstream adapter README. Publish the
canonical result JSON alongside the labelled companion artifacts; do not add
KTANN-only definitions to upstream metric fields.
