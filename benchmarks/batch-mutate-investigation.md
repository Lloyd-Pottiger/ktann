# Batch mutation investigation (#160)

Baseline: `909b750f51a1b1524bca45caec9716632192b2da`.

## Delivered change

Retain one optimization: decode each existing Upsert Record/Location pair once,
drop the old vector immediately, and carry its validated location into the
internal replacement operation. Unique batch Record IDs and the same transaction's
update-protected reads establish that location's validity. The standalone storage
replacement operation still validates its own pair.

The checked per-input decode also fixes corruption attribution: a malformed old
Record or Location reports the original position in a mixed batch and the entire
transaction rolls back. The raw read cache and batched backend prefetch remain in
use. Atomicity, result order, duplicate-ID rejection, update conflicts, whole-batch
retry, unknown-commit handling, write beam, and scalar numeric order are preserved.

## Method and environment

Apple M1 Pro (8 logical CPUs), 16 GiB RAM, Darwin 25.6.0; Rust 1.85.0,
LLVM 19.1.7, release builds with both production adapter features. RocksDB uses
crate 0.24 / native 10.4.2. FoundationDB uses client library 7.3.55 and the persistent
local single-node 7.3.69 server. Each command uses four runtime worker threads.

The same benchmark harness is applied to the recoverable baseline and independent
candidate patches. Orders alternate over three repetitions. Baseline, candidate,
and delivered source patches, binary SHA-256 hashes, commands, per-run JSON latency
distributions, and drivers are stored under:

`/Users/lloyd/projects/ktann/.benchmark-data/results/issue-160/final`

`delivered.patch` is byte-for-byte equal to `decode.patch`; the delivered source
therefore matches the independently measured `ktann-160-decode-final` binary.
The complete exploratory matrix uses `*-matrix-*`; independent candidates use
`*-decode-*`, `*-kernel-*`, and `*-directory-*`. Final Upsert migration and hotspot
checks use `*-decodemigrate-*` and `*-decodehot-*`. The final FoundationDB mixed
workload is the baseline/decode subset of `*-memory-*`.

```sh
export DYLD_LIBRARY_PATH=/Users/lloyd/.local/lib
export FDB_CLUSTER_FILE=/Users/lloyd/.local/foundationdb/7.3.69/etc/fdb.cluster
export RUSTFLAGS='-L native=/Users/lloyd/.local/lib'
cargo run --release -p ktann-benchmarks --all-features -- \
  run --backend rocksdb --profile batch --worker-threads 4 --output batch.json
```

Use `--backend foundationdb` for the second adapter. JSON throughput is atomic
batches/s; multiply by `mutation_batch_size` for records/s. The batch profile covers
Insert, existing-record Upsert, eight-tree migration, and real Delete, with dimensions
128/1536, batches 1/32/128, and concurrency 1/4. Requests are materialized before
measurement; accepted outcome checks reject no-op work. Each scenario has warmup
and records its useful operation count, seed, and complete configuration.

This is an interactive macOS host. Timed workers run serially, coordinated with
issue #162. Files marked `overlap` are excluded, including one conservatively
excluded sample triggered by a lightweight analysis script. Final measurement
controllers distinguish executable names from analysis script arguments.

CPU is worker-process CPU; FoundationDB server CPU is outside that metric. Peak
RSS includes setup and allocator retention. Backend counters are logical keys,
bytes, writes and commits, not physical disk amplification. Allocation counts and
commit-only wait time are not separately instrumented. CPU and batch latency must
not be presented as direct measurements of either quantity.

## Results and disposition

The delivered decode optimization is **demonstrated on RocksDB** for the measured
high-dimensional Upsert workload: median records/s improves 9.7%, with CPU/record
down 6.4%; all three paired throughput changes are positive (+3.7% to +16.3%).
FoundationDB is **directionally supported**, with median records/s +19.8% and
CPU/record -9.3%, but paired throughput changes range from -14.1% to +24.9%.
Its median improvement is not a repeatable per-run guarantee. Tail latency is noisy:
FoundationDB
p95 improves, while its median p99 rises from 41.1 to 46.1 ms (128 measured batches); this is not a universal
tail-latency improvement. The correctness fix is independently justified.

Two performance-only candidates are **not delivered**:

- Directory prefetch replaced eight serial directory point reads with one backend
  batch call in a deterministic mechanism check. Root-authority reads and keys/bytes
  were unchanged; FoundationDB still submits per-key futures. Independent median
  throughput changed +1.8% on RocksDB and -0.4% on FoundationDB, with FoundationDB
  CPU/record +4.5%. End-to-end benefit was not demonstrated, so the private helper,
  integration, and dedicated mechanism test were removed.
- Per-handle kernel reuse improved RocksDB's 1536-dimensional single-record workload
  by 23.7%, while FoundationDB throughput changed -0.7%. However, combining it with
  decoding changes produced repeatable elevated FoundationDB mixed-workload RSS:
  the first three samples were 105.7/70.7/104.7 MiB versus baseline 69.1/71.0/68.9;
  a follow-up comparison produced 93.9/92.8/70.9 versus 69.6/71.2/72.6 MiB. Individual
  candidates did not reproduce the large increase. Its allocation-level cause is
  unresolved, so the combination failed the whole-system resource gate and kernel
  reuse was withdrawn. No cache, runtime flag, or dormant alternative remains.

An initial hotspot retry increase did not reproduce in subsequent comparisons.
Directory prefetch was inactive on that single-tree path, so its removal is not
claimed as the cause of the different hotspot result. Raw retry/write-amplification
counters are retained rather than treating one timing sample as a regression diagnosis.

## Complementary behavior and scope

The maintained mixed workload has 5,000 records and a two-level tree, with concurrent
searches and hot updates. The SIFTsmall lifecycle workload imports 10,000 records,
waits for verified stable topology, and checks cold/warm recall@10. All lifecycle
runs completed with recall 1.0 and no actionable or transitional partitions remaining.
The delivered patch does not change the pure-Insert, search, or Structure Maintenance
paths of this lifecycle; no large-import speedup is claimed. The lifecycle data
collected for the discarded kernel combination is supporting investigation evidence,
not a measurement of the delivered artifact.

Borrowed encoding, routing-distance validation removal, deeper cross-tree routing,
FoundationDB's bounded 1024-key chunks, and Import Admission scheduling are unchanged.
The pinned RocksDB 0.24 API has ordinary MultiGet and individual GetForUpdate but
no equivalent update-protected MultiGet; replacing the latter with ordinary reads
would weaken the contract. This investigation does not claim million-vector import
improvements, arbitrary-depth routing gains, or benefits for every batch size.

## Correctness verification

The corruption regression was reproduced before the fix: it returned no input
position instead of the original mixed-batch position. Coverage now checks malformed
Record and Location values, preservation of a preceding healthy Upsert, and rollback
of a preceding Insert.

Workspace all-feature tests and real RocksDB tests passed during implementation.
Explicit ignored FoundationDB adapter-contract and recall tests passed against the
local cluster. On the delivered source, formatting, workspace/all-target/all-feature
Clippy, and focused mutation/membership suites (37 tests) passed as the final checks. An independent
review of the implementation and harness found no actionable correctness issues.

## Delivered measurements

Each cell is the median of three independent runs. Batch latency columns are
p50 / p95 / p99 in milliseconds. `mixed` uses operations/s and CPU/operation;
other rows use records/s and CPU/record. RSS is the whole-worker peak.

| Backend / workload | Throughput (change) | Batch p50 / p95 / p99 ms | CPU µs | RSS MiB |
|---|---:|---|---:|---:|
| rocksdb / 1536d, Upsert 128, concurrency 1 | 18,643.2 → 20,455.3 (+9.7%) | 6.524 / 8.892 / 10.664 → 6.173 / 6.847 / 8.124 | 52.0 → 48.7 | 377.4 → 367.4 |
| foundationdb / 1536d, Upsert 128, concurrency 1 | 4,199.0 → 5,028.4 (+19.8%) | 29.979 / 37.902 / 41.113 → 23.972 / 35.782 / 46.148 | 100.6 → 91.2 | 171.7 → 171.4 |
| rocksdb / 1536d, migration 32, concurrency 1 | 15,342.4 → 15,829.8 (+3.2%) | 2.067 / 2.291 / 3.778 → 2.001 / 2.181 / 2.787 | 65.3 → 63.2 | 361.7 → 361.0 |
| foundationdb / 1536d, migration 32, concurrency 1 | 1,494.9 → 1,569.7 (+5.0%) | 20.094 / 32.445 / 68.310 → 17.964 / 35.294 / 67.812 | 236.1 → 224.6 | 163.8 → 164.0 |
| rocksdb / 1536d, Upsert 32, concurrency 4 | 15,801.5 → 16,161.9 (+2.3%) | 1.853 / 18.199 / 117.316 → 1.810 / 15.407 / 154.943 | 79.4 → 77.7 | 374.3 → 374.0 |
| foundationdb / 1536d, Upsert 32, concurrency 4 | 1,887.1 → 1,986.9 (+5.3%) | 17.931 / 292.327 / 848.391 → 15.875 / 271.799 / 845.902 | 322.9 → 297.0 | 172.8 → 172.4 |
| foundationdb / 128d, mixed search/hot update | 358.9 → 353.7 (-1.5%) | 7.310 / 14.474 / 23.839 → 7.265 / 15.057 / 24.814 | 6,389.1 → 6,716.3 | 70.4 → 70.3 |

The final FoundationDB mixed workload changes throughput -1.5%, CPU/operation
+5.1%, and RSS 70.4 → 70.3 MiB. No mixed-workload speedup is claimed. Migration
p95 is also mixed across adapters; the optimization is not a universal latency win.

### Backend work and contention

Totals cover the measured operations in the corresponding report. The 128-record
batch scenario measures 128 batches, the 32-record scenarios 512 batches, and the
mixed case 2,000 operations (1,000 writes). Read counts include the mixed case’s
searches. Logical write amplification is per successful batch/write.

| Backend / workload | Read keys | Read bytes | Commits | Retries | Logical mutations/write | Logical bytes/write |
|---|---:|---:|---:|---:|---:|---:|
| rocksdb / 1536d, Upsert 128, concurrency 1 | 49,920 → 49,920 | 125,559,680 → 125,559,680 | 128 → 128 | 0 → 0 | 386.0 → 386.0 | 978,033.0 → 978,033.0 |
| foundationdb / 1536d, Upsert 128, concurrency 1 | 49,920 → 49,920 | 125,559,680 → 125,559,680 | 128 → 128 | 0 → 0 | 386.0 → 386.0 | 978,033.0 → 978,033.0 |
| rocksdb / 1536d, migration 32, concurrency 1 | 79,872 → 79,872 | 126,201,088 → 126,201,088 | 512 → 512 | 0 → 0 | 173.5 → 173.5 | 247,559.0 → 247,559.0 |
| foundationdb / 1536d, migration 32, concurrency 1 | 79,872 → 79,872 | 126,201,088 → 126,201,088 | 512 → 512 | 0 → 0 | 173.5 → 173.5 | 247,559.0 → 247,559.0 |
| rocksdb / 1536d, Upsert 32, concurrency 4 | 80,682 → 83,232 | 194,173,889 → 200,310,864 | 512 → 512 | 279 → 304 | 150.9 → 155.7 | 377,845.1 → 389,785.7 |
| foundationdb / 1536d, Upsert 32, concurrency 4 | 126,174 → 120,768 | 303,657,523 → 290,647,136 | 512 → 512 | 725 → 672 | 235.6 → 225.5 | 590,860.7 → 565,545.3 |
| foundationdb / 128d, mixed search/hot update | 285,000 → 285,000 | 111,947,344 → 112,334,992 | 1,000 → 1,000 | 0 → 0 | 5.0 → 5.0 | 889.0 → 889.0 |

Every measured write was accepted. Non-contended batches keep the same backend
keys, bytes, commits, and write amplification: the saving is repeated decoding,
not fewer physical reads. Hotspot retries are stochastic. RocksDB baseline counts
were 344/279/266 versus 207/350/304; median retries therefore rise 279 → 304 despite
all three paired throughput improvements. FoundationDB medians fall 725 → 672.
No general retry or write-amplification reduction is claimed.

RocksDB hotspot p99 spans 106.8–131.1 ms in the baseline and 64.4–199.9 ms in
the candidate. This tail variability is a remaining measurement limit, not
evidence of a consistent tail improvement.
