# Write-time local HNSW: Cohere 1M findings

Status: opt-in experiment, with exact write routing still the production default.
This work concerns writes and their structural maintenance. Query behavior and
budgets are unchanged. The implementation is based on `partition-local-hnsw`
(`48b52914b133d1b96807c1e8fac2c4975e3e71ad`). Measurements below were collected
on 2026-10-24 with RocksDB; they are workload observations, not an SLA.

## Fixed workload

- Cohere 1M, 768-dimensional cosine vectors. Every run clones the same settled
  database: 13,130 leaves, three level-2 internal partitions and one level-3 root;
  all 13,134 partitions Ready, no actionable maintenance, largest internal body
  5,150 children. M16, efConstruction200, seed7, offline refinement2;
  leaf bounds 16:128 and internal bounds 64:8192.
- 8,192 vector-changing upserts, one foreground writer, batch32 (256 calls), four
  runtime threads. A seed7 xorshift permutation selects target IDs from its first
  8,192 ordinals and replacement vectors starting at ordinal 500,000. The record
  count remains 1M. Input construction is outside timing.
- Write beam4 and experimental HNSW ef64. Warmup queries 200..299 use
  beam128/ef64/k100/rerank125, outside write timing. No rerank sweep.
- Identical default caches: on the measured 16GiB host the partition cache budget
  is 4GiB; RocksDB uses its default block cache. No compilation or tests overlap
  timed runs. Full Verify before/after each run is outside timing.
- With two maintenance workers, total time includes bounded Header rediscovery,
  reoffering pending fixups, and drain to three quiet Ready rounds. This does
  **not** demonstrate autonomous queue convergence. Logical read bytes measure
  the adapter interface, not physical disk traffic. CPU includes maintenance.

## Results

Each table compares modes within one run series; do not mix medians across
series. Concurrent scheduling can change the number of completed maintenance
steps even with identical upserts.

### Batched write routing and committed graph reuse

Fixed warm topology, no maintenance: three-run median complete-write time is
17.584s exact versus 4.683s batched HNSW (**73.4% less**). A prior scalar HNSW
mechanism probe took 9.070s versus 4.756s batched, with identical 9,583,160 distance
calls; this was a single scalar probe, not a repeated throughput comparison.

With two workers, three-run medians:

| Mode | Foreground writes | Writes + drain | Total CPU | Logical reads |
|---|---:|---:|---:|---:|
| Exact | 22.313s | 25.130s | 63.400s | 30.999GB |
| Batched HNSW | 6.943s | 21.032s | 42.270s | 25.627GB |
| + committed prepared graph reuse | 5.254s | 20.791s | 38.988s | 24.434GB |

Publishing an already-prepared graph after known commit success reduces
foreground graph loads from 64–66 to 5–8. Median graph read/restore time drops from
1.864s to 0.211s. Foreground writes improve 24.3%, but writes plus drain improve
only 1.1%; the foreground gain is not the overall throughput gain.

When every graph reuse is forcibly bypassed, two-run medians are 25.442s exact
versus 20.248s HNSW. Each HNSW run loads 1,024 graphs: reading 9.151s and restoring
6.046s dominate actual graph search 2.716s. This is an adversarial control, not a
realistic invalidation frequency. Workspace allocation is only a few milliseconds.

### Avoid repeated parent-edge discovery during maintenance

The remaining dominant work was exact incoming-edge discovery, which scans
complete parent-level Child Entries and their 768-dimensional centroid payloads.
Preparing and committing maintenance could independently repeat that scan.

Three-run medians, **same executable**, all with batched HNSW and committed graph
reuse; only parent-proof reuse differs:

| Metric | Control | Parent-proof reuse | Change |
|---|---:|---:|---:|
| Foreground writes | 5.083s | 5.116s | +0.6% |
| Writes + drain | 19.777s | 18.406s | **-6.9%** |
| Total CPU | 37.749s | 27.292s | **-27.7%** |
| Logical reads | 24.301GB | 13.118GB | **-46.0%** |
| Full incoming-edge discoveries | 484 | 209 | -56.8% |
| Cumulative discovery time | 16.423s | 6.317s | -61.5% |

Control total-time range 19.738–20.478s; optimized 17.972–18.419s. Optimized runs
reuse 209 proofs each with zero stale-proof attempts. Retryable commits are 3/2/4
versus 2/0/0. Controls complete 96 splits each, optimized 98 each; all complete 13
merges. The improvement is not caused by fewer completed splits. Foreground
latency is effectively unchanged: the gain is in maintenance and resources.

Inclusive phase durations overlap across workers and nested phases; they cannot
be added to reconstruct wall-clock time. Native preparation remains approximately
6.87s versus 6.82s cumulative. Two mechanism probes established why avoiding the
second scan alone was insufficient: without local guards it caused 98 stale
plans and 300 native preparations; with guards but scanning before waiting, it
still performed 375 full discoveries. The final path defers the full scan until
after guard acquisition and snapshot refresh.

## Retained implementation and invariants

- Write HNSW always reuses per-body workspace and scores groups of two to four
  neighbors. Adjacency consumption, candidate ordering and distance-call trace
  are preserved; scalar production queries retain their existing path.
- Write routing observes operation cancellation/deadlines and the configured
  working-memory budget. CPU work stays within its admitted foreground task.
- Prepared graph publication uses the existing bounded cache, only after known
  commit success. Native graph/centroid/kernel retention is charged; old snapshots
  still validate descriptor, identity and epoch. Failed or stale plans do not
  publish. Offers are not necessarily cache admissions.
- Parent-proof reuse first reads the parent-body set and Headers, then takes the
  existing sorted body guards. Waiting refreshes the snapshot before full exact
  incoming-edge discovery. Proof and discovery use the same snapshot.
- Before editing, the commit rewalks the complete parent set and update-protects
  every captured Header, requires equality, and protects the actual incoming
  Child Entry in addition to existing graph-source fences. Any mismatch discards
  the plan. Child Entry changes advance their owner's epoch. No reverse pointer,
  second shared cache or persistent format change is introduced.
- Extra proof and guard metadata reduces working-memory admission. Broader local
  guard scope can affect concurrency; performance beyond two maintenance workers
  and across processes/backends remains unmeasured. These results do not justify
  enabling the experiment by default or claiming FoundationDB performance.

## Reproduction

Build and use [`ktann-write-routing`](README.md#isolated-write-routing-experiment).
The runner expects `OUTPUT/database`, namespace `ktann-benchmark`, index
`benchmark`. It mutates that copy. Exact reproduction requires the original
settled fixture; a fresh build with the same settings is a different topology.
The locally retained fixture is:

```text
.benchmark-data/results/cohere-1m-attribution-20261024/fixture/database
```

Example on macOS/APFS (use a normal independent copy on other filesystems):

```sh
cargo build --release -p ktann-benchmarks --bin ktann-write-routing
export KTANN_BENCH_DATASET_CACHE=/path/to/vectordb_bench/dataset
case_dir=.benchmark-data/results/write-routing-final-repeat
mkdir "$case_dir"
cp -cR .benchmark-data/results/cohere-1m-attribution-20261024/fixture/database "$case_dir/database"
env -u KTANN_FORCE_GRAPH_MISS \
  KTANN_EXPERIMENT_WRITE_HNSW=1 \
  KTANN_EXPERIMENT_PUBLISH_READY=1 \
  KTANN_EXPERIMENT_REUSE_PARENT_PROOF=1 \
  KTANN_EXPERIMENT_MAINTENANCE_TIMING=1 \
  target/release/ktann-write-routing "$case_dir" 2 8192
```

For the parent-proof control, unset `KTANN_EXPERIMENT_REUSE_PARENT_PROOF`; retain
the other switches. For the exact baseline, unset all three optimization switches.
Use a fresh fixture copy per run and alternate run order. All switches enable by
presence, so unset them rather than setting them to 0. Workers 0 isolates warm
routing; `KTANN_FORCE_GRAPH_MISS=1` bypasses reuse only in benchmark-feature builds.
The former workspace/scalar/quartet selection and audit-only switches were
removed during cleanup; they are not needed to reproduce the retained modes.

`report.json` contains foreground and total timing/CPU/IO, full Verify summaries,
maintenance steps, proof reuse/stale counts, graph loads and phase samples.
Local raw artifacts (not committed due to size):

| Series | Directory under `.benchmark-data/results/` |
|---|---|
| Initial complete-write and scalar probe | `write-routing-1m-20261024` |
| Prepared graph publication, 20 verified runs | `write-routing-publish-20261024` |
| Parent-proof attribution/comparison, 13 verified runs | `write-maintenance-phases-20261024` |

Each series retains reports, logs, driver invocations, source snapshots and
measured executables. The six final parent-proof comparisons used SHA-256
`9299ca3ec15c1d1027417e2989f5f967eddeb13c09d4330e66ae466cbfbe9df5`.
These figures predate removal of redundant experiment selectors; cleanup
validation is recorded below rather than relabeling historical measurements as
new runs.

## Correctness evidence

Core tests, native scalar/batched trace equivalence, snapshot pagination, graph
updates, split interleavings, commit faults/recovery, cache epoch publication,
and parent-proof rejection/conflict-read coverage pass. Parent-proof tests include
changes to an unselected parent, a changed parent set, root demotion and metadata
budget exhaustion. Independent reviews covered publication, proof validation,
ordered guards and discovery ordering. Complete-write runs verify the full 1M
records before and after; the final whole-patch review/checks accompany merge
preparation. No query optimization or rerank-budget change is included.

## Merge cleanup validation (2026-10-24)

After selector cleanup, the core benchmark-feature suite passed 751 tests; native
tests passed 17, including scalar/batched equivalence. The explicit prepared-graph
publication test and opt-in graph update, split-interleaving and transaction-budget
suites passed. Formatting, workspace Clippy (excluding FoundationDB), and the core
no-default-features check passed. An independent complete-patch review reported
no actionable findings. FoundationDB performance was not measured.

A final release-build smoke run used all three retained experiment switches,
two maintenance workers, batch 32 and 8,192 updates on a fresh copy of the same
Cohere 1M fixture. Full Verify before and after completed without issues; final
topology had no actionable partitions. Foreground time was 5.598 s and total time
including maintenance was 19.575 s. This single run verifies the cleaned entrypoint;
it is not a replacement for the repeated comparisons above. Its report, invocation,
binary hash and validation logs are under
`.benchmark-data/results/write-routing-merge-cleanup-20261024/`.
