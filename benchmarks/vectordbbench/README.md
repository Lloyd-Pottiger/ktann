# VectorDBBench bridge

`ktann-vdbbench-bridge` is a benchmark-only Rust binary, not a KTANN service API.
KTANN owns the bridge, its native tests and dataset provenance. The Python
client, CLI registration, run tools and process integration tests belong to
[VectorDBBench](https://github.com/Lloyd-Pottiger/VectorDBBench).
There is no Python overlay or installer in this repository.

See [VALIDATION.md](VALIDATION.md) for historical measurements and their limits.

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

## Build and run

Build an optimized bridge in the KTANN checkout:

```sh
cargo build --release -p ktann-benchmarks --bin ktann-vdbbench-bridge
# Add --all-features for FoundationDB; configure its native client and cluster
# as described in the parent benchmark README.
```

Use a VectorDBBench checkout containing the KTANN adapter and install it with
`pip install -e .`. Its `scripts/ktann/README.md` documents dataset preparation,
canonical CLI runs and companion reports. The launcher accepts the bridge path
and `--manifests /path/to/ktann/benchmarks/datasets`; it does not modify either
checkout. It records the VectorDBBench revision and bridge binary SHA-256.

The native bridge uses protocol version 1 over a Unix socket: four-byte
big-endian frame length followed by JSON, bounded to 8 MiB and 128 connections.
Use a dedicated backend location and a fresh bridge for each run. Shutdown
finishes import, drops the benchmark index and unlinks its socket. A killed
process can leave a stale socket; remove it only after confirming the process
has exited. The bridge never unlinks a preexisting socket automatically.

## Verification

Native tests remain in `benchmarks/src/bridge.rs` and `bridge/topology.rs`:

```sh
cargo test -p ktann-benchmarks --all-features
```

Run the process tests from the VectorDBBench checkout, with its dependencies
installed and the bridge built above:

```sh
export KTANN_BRIDGE_BIN=/path/to/ktann/target/release/ktann-vdbbench-bridge
python -m unittest discover -s tests -p 'test_ktann_bridge.py' -v
KTANN_TEST_BACKEND=foundationdb python -m unittest discover \
  -s tests -p 'test_ktann_bridge.py' -v
```

These tests cover spawned workers, pickling, concurrent search, official CLI
metrics, signed IDs, protocol errors, crash/restart and cleanup on both backends.
