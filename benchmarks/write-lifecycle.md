# Write-routing default-enablement validation

This experiment tests whether local HNSW write routing is suitable for default
use. It does not change production defaults or optimize queries. It complements
[the mature-topology write measurements](write-routing.md).

## Controlled workload

`ktann-write-lifecycle fixture|run OUTPUT SCENARIO.json` loads Cohere 1M.
`fixture` builds the first `records` vectors under namespace `ktann-benchmark`,
index `benchmark`, with leaf bounds 16:128, internal bounds 64:8192, graph M16,
efConstruction200, seed7 and two offline refinement rounds. `run` requires a
fresh copy of the fixture at `OUTPUT/database`. It verifies record count and
membership before writing. The existing full-scale fixture is reused for 1M.

Scenarios explicitly specify `records`, `rounds`, `updates_per_round`, `warm`,
`workers` and `queries`. Batch size remains 32. Query options remain beam128,
ef64, k100, rerank125, with distance/candidate budgets 65536/16384. Queries begin
at row100. There is no budget sweep, adaptive query calibration or query-path
change.

Each round shuffles record indices with deterministic xorshift seed `7 + round`
and swaps vectors between consecutive disjoint pairs. Both records of every
pair are written in the same batch. This changes routing destinations while
preserving the complete vector multiset. The harness maintains inverse maps
between vector identity and current Record ID and remaps the original nearest
neighbors accordingly. Full-corpus truth comes from the supplied dataset; the
16K subset uses scalar-f64 brute force. A focused overlapping-swap test checks
the remapped truth against the changed corpus. This workload does not represent
net growth, deletion or an out-of-distribution vector stream.

Cold means a fresh process with no graph-cache warmup. The initial Verify reads
storage, so this is **not** a cold OS/block-cache or cold-disk measurement. Warm
runs evaluate the fixed query set before writing. Every round measures foreground
writes and maintenance separately from verification and query quality. Subsequent
rounds reuse that process and its caches. Maintenance includes Header rediscovery
and three quiet observations, as in the previous complete-write harness; the
result is not a claim of autonomous maintenance convergence.

The exact control removes all three experimental variables from its environment.
These switches test presence: setting a variable to `0` still enables it.
Reports record their actual presence, which the driver checks against each mode. The candidate enables
`KTANN_EXPERIMENT_WRITE_HNSW`, `KTANN_EXPERIMENT_PUBLISH_READY` and
`KTANN_EXPERIMENT_REUSE_PARENT_PROOF`. Thus this comparison measures the combined
candidate for default use, not the marginal effect of each optimization.

## Predeclared matrix and decision criteria

- 16,384 records: warm and cold, four rounds of 8,192 updates, two maintenance
  workers, 100 held-out queries, three repetitions per mode.
- 1,000,000 records: initially cold, eight rounds of 32,768 updates, two maintenance
  workers, 100 held-out queries, three repetitions per mode. Later rounds measure
  sustained warm/recovery behavior. Each run performs 262,144 update operations;
  this is a finite lifecycle test, not unlimited-duration stability evidence.
- Each run starts from an independent APFS clone of the same size-specific fixture.
  Baseline/candidate execution order alternates between repetitions. No builds,
  tests or other benchmarks run concurrently with timings.

Report foreground/total time, CPU, bytes read, graph routing counters, maintenance
steps, verified topology, per-round recall and paired per-query recall differences.
Investigate repeatable time regressions over 5% and recall losses of 0.5 percentage
points or more; these thresholds flag concerns rather than authorize a loss of
correctness. Full Verify must complete without issues, and settled runs must have
no actionable partitions. Small differences within variation do not establish a
win or quality equivalence. Failure of any representative case prevents a blanket
default-on recommendation.

Raw reports, exact invocations, scenario files, measured binary hash and the serial
runner are retained under `.benchmark-data/results/write-lifecycle-20261024/`.

## Reproduction example

Build the harness with `cargo build --release -p ktann-benchmarks --bin ktann-write-lifecycle`. An example small warm scenario is:

```json
{"records":16384,"rounds":4,"updates_per_round":8192,"warm":true,"workers":2,"queries":100}
```

Set `KTANN_BENCH_DATASET_CACHE` as documented in the repository instructions.
Create a fixture with `ktann-write-lifecycle fixture NEW_FIXTURE scenario.json`.
For each repetition, copy `NEW_FIXTURE/database` to a new `OUTPUT/database`
(`cp -cR` on APFS), then run the same executable and scenario:

```sh
# Exact control: variables must be absent, not set to 0.
env -u KTANN_EXPERIMENT_WRITE_HNSW -u KTANN_EXPERIMENT_PUBLISH_READY \
  -u KTANN_EXPERIMENT_REUSE_PARENT_PROOF -u KTANN_FORCE_GRAPH_MISS \
  target/release/ktann-write-lifecycle run OUTPUT scenario.json

# Combined candidate, on a separate fresh fixture copy.
env -u KTANN_FORCE_GRAPH_MISS KTANN_EXPERIMENT_WRITE_HNSW=1 \
  KTANN_EXPERIMENT_PUBLISH_READY=1 KTANN_EXPERIMENT_REUSE_PARENT_PROOF=1 \
  target/release/ktann-write-lifecycle run OTHER_OUTPUT scenario.json
```

The measured matrix has no maintenance phase-timing switch enabled. Do not run
compilation, tests or another benchmark concurrently. Preserve `report.json`, the
scenario, binary hash and invocation after removing only the generated database
copy. The full-scale comparison uses the historical immutable mature fixture;
a freshly built 1M fixture is a different topology sample.

## Results (2026-10-24)

All numbers below use the valid, presence-checked controls: three repetitions per
mode and scenario, each from a fresh fixture copy. Times are medians of the sum
of per-round write plus KTANN maintenance intervals; Verify and quality queries
are excluded. CPU and logical read bytes cover those same intervals. They do not
measure physical disk IO, peak RSS, or full-process resource use during setup and
quality evaluation, and the harness does not explicitly drain RocksDB compaction.

| Scenario | Updates/run | Exact total | HNSW total | HNSW change | CPU change | Logical read change |
|---|---:|---:|---:|---:|---:|---:|
| 16K warm | 32,768 | 4.232 s | 5.270 s | +24.5% | +24.3% | -1.9% |
| 16K initially cold | 32,768 | 4.255 s | 5.339 s | +25.5% | +25.6% | -2.3% |
| 1M initially cold, eight rounds | 262,144 | 662.712 s | 195.850 s | -70.4% | -63.5% | -44.8% |

For 1M, the three exact totals were 658.477–665.580 s and HNSW totals were
192.497–202.803 s. Foreground time fell 71.1%. The first cold round alone had
median total times of 81.539 s exact versus 25.028 s HNSW (69.3% lower). CPU fell
from 725.125 to 264.956 s and logical reads from 101.518 to 55.990 GB over eight
rounds. The reduction is material and repeats beyond observed timing variation.

The small graph started with 199 children in its root and ended with 201 under
both modes. Exact total times ranged 4.222–4.349 s across the warm/cold runs;
HNSW ranged 5.248–5.366 s. CPU increases track the slowdown while logical reads
fall slightly. Both warm and cold cases regress, so initial restoration alone
cannot explain the small-graph cost.

### Quality and topology

The held-out query baseline was 99.62% recall@100 on 16K and 82.60% on 1M at the
unchanged beam128/ef64/rerank125 budget. On 16K every per-query recall matched
between modes and repetitions at each checkpoint; the final average was 99.57%.
On 1M the mean recall trajectory was:

| Completed round | Exact | HNSW |
|---|---:|---:|
| 1 | 82.270% | 82.313% |
| 2 | 82.380% | 82.393% |
| 3 | 82.350% | 82.380% |
| 4 | 82.370% | 82.400% |
| 5 | 82.450% | 82.460% |
| 6 | 82.470% | 82.490% |
| 7 | 82.540% | 82.550% |
| 8 | 82.630% | 82.630% |

The final averages match; individual queries can gain or lose one of their 100
neighbors. Resampling these 100 paired queries gives a final mean-difference
95% interval of approximately -0.05 to +0.05 percentage points, well within the
predeclared 0.5-point concern threshold for this sample. This is evidence against
a material average loss in this finite workload, not a claim about all query
populations or unlimited write duration. The initial dip also occurs under exact
routing and does not accumulate across rounds.

All 18 matrix runs, plus the separate 1M baseline probe, completed full Verify
before writes and after every round without issues. Every settled checkpoint had
zero actionable or transitional partitions. The 1M fixture started with 13,130
leaves; exact ended with 13,409 and HNSW with 13,407. Both retained three level-2
partitions and one level-3 root. Maximum leaf occupancy remained 128; maximum
level-2 occupancy was 5,264 exact and 5,262 HNSW.

### Default-enablement decision

Large-graph write benefit is **demonstrated** for the tested RocksDB workload,
including a cold graph cache and repeated updates with maintenance. Blanket
default enablement is **not supported**, because 16K has a repeatable 24–25%
regression even with warm graphs. Production defaults remain unchanged.

The next write-routing candidate should retain the cheap exact path for small
internal bodies and use HNSW where pruning repays traversal overhead. Select a
crossover using internal graph node count and measured costs, not total dataset
record count; these two sizes do not establish a threshold. No query-budget
increase or query optimization is needed to address the observed small-body cost.
The stationary swap workload does not establish quality under growth, deletion,
or distribution shift; those claims remain outside this evidence.

### Validation and provenance

The new harness and shared settlement helper pass focused tests (including an
overlapping-swap truth oracle and small-record-count settlement), formatting and
Clippy. Independent review covered the harness; settlement explicitly uses the scenario
record count. An initial driver error set exact switches to `0`;
those runs actually enabled the candidate and were invalidated. They remain
under `invalid-present-zero/` for diagnosis and are excluded from every table.
Current reports record actual switch presence, and the serial driver asserts the
expected mode. No production or query implementation changed.

The matrix executable is retained as `measured-write-lifecycle` with SHA-256
`f9e94cdc19cc956b993381c5be6db677708ddd04fed5f7d10e6c36283033e89f`.
Scenario files, `run.py`, `analyze.py`, individual reports and `analysis.json`
are in the raw artifact directory named above. A subsequent diagnostic-only
report extension exposes existing write phase timers and distance counts; it
adds no hot-path instrumentation and does not relabel the matrix measurements.

### Supplemental small-graph attribution

After the matrix, one additional warm 16K round per mode exposed existing phase
counters (8,192 updates, same fixture and fixed budgets). This diagnostic has its
own reports and binary hash and does not replace the repeated timing statistics.
Exact foreground time was 0.961 s and HNSW 1.287 s. HNSW performed 1,645,008
recorded distance evaluations; an ideal full scan of 199–201 children for 8,192
vectors is already 1,630,208–1,646,592 evaluations. This is a scale comparison,
not an instrumented exact-path count; retries can add work in either mode.

HNSW phase totals were search 559.877 ms, graph read 3.577 ms, graph restore
0.017 ms, workspace 0.213 ms, ownership 2.178 ms and merge 2.305 ms. The phase
samples are diagnostic durations, not a complete decomposition of foreground
wall time. Graph read/restore/workspace together are about 3.8 ms versus a
326.7 ms foreground gap. Together with the near-full-scan distance count and
warm/cold CPU regressions, this supports small-body traversal/scoring overhead
as the next target rather than graph recovery or workspace reuse.

The exact-path distance field is zero because this experimental counter only
instruments HNSW; it does not mean exact routing performs no distance work.
Supplemental artifacts are `diagnostic-exact/`, `diagnostic-hnsw/`,
`diagnostic.py`, `diagnostic.json` and `diagnostic.sha256` under the same directory.
