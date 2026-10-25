# Write-routing default-enablement validation

Graph-enabled indexes now default to exact write routing below 1024 nodes per
internal body and local HNSW at or above 1024 nodes. Indexes without local graphs
retain exact routing. Query behavior and budgets are unchanged. It complements
[the mature-topology write measurements](write-routing.md).

The earlier sections below preserve historical opt-in measurements (including
768-node experiments); they are not measurements of the final 1024 policy.
The final default validation is recorded at the end. The removed
`KTANN_EXPERIMENT_WRITE_HNSW`, `KTANN_EXPERIMENT_WRITE_HNSW_MIN_NODES`,
`KTANN_EXPERIMENT_PUBLISH_READY`, and `KTANN_EXPERIMENT_REUSE_PARENT_PROOF`
switches no longer affect behavior. Reproduce historical arms with their pinned
binaries/revisions; unsetting those switches in current code does not select exact.
Benchmark-only timing and forced-cache-miss diagnostics remain available.

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
neighbors accordingly. Full-corpus truth comes from the supplied dataset; every
partial-corpus fixture uses scalar-f64 brute force. A focused overlapping-swap test checks
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

### All-HNSW decision (2026-10-24)

Large-graph write benefit is **demonstrated** for the tested RocksDB workload,
including a cold graph cache and repeated updates with maintenance. Blanket
default enablement is **not supported**, because 16K has a repeatable 24–25%
regression even with warm graphs. Production defaults remain unchanged.

These findings motivated the internal-body threshold experiment below: retain
the cheap exact path for small bodies and use HNSW where pruning repays traversal
overhead. The two original sizes alone did not establish a threshold. The added
node-count cost curve and hybrid lifecycle validation keep query budgets fixed.
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

## Internal-body threshold experiment

`KTANN_EXPERIMENT_WRITE_HNSW_MIN_NODES=N` refines the opt-in write route:
nonempty internal bodies below `N` use the existing exact quartet scorer;
bodies with at least `N` nodes use HNSW. The node count is the current transaction's
Header entry count. It is not the index record count, a cached graph estimate, or
a query parameter. Omission (or `N=0`) retains the all-HNSW experimental arm.
Malformed values fail with `InvalidArgument` when write HNSW is enabled.

To test the selected policy, add `KTANN_EXPERIMENT_WRITE_HNSW_MIN_NODES=768`
to the combined-candidate invocation above. The exact control must still unset
all three experiment switches. Keep the same scenario, cloned fixture and binary
for both arms; setting a presence-based switch to `0` does not disable it.

Mixed waves retain one ownership map and one global candidate beam, including
split-family multiplicity and deterministic tie ordering. Empty bodies retain
uncached count validation. Small bodies in the same wave share the existing
lockstep scan and quartet scorer. The ordinary production exact route is unchanged.
Reports record `write_hnsw_min_nodes` and foreground `write_exact_bodies` /
`write_hnsw_bodies` counts to distinguish policy selection from merely setting a flag.
Counts include attempts/retries and are not unique partition counts.

The fixture command reserves an 8 GiB refinement envelope for crossover fixtures;
this is preparation-only admission, not allocated runtime memory or a query/write
budget. The original default 256 MiB worker envelope cannot retain the 32K
768-dimensional fixture's input and planning arrays. Larger fixtures can require
more than this envelope; the Cohere 1M comparisons reuse the existing immutable
fixture rather than rebuilding it.

### Node-count cost curve (2026-10-25)

Fresh size-specific Cohere fixtures, warm via the same 100 quality queries,
8192 stationary updates in batches of 32, two maintenance workers. Three paired
repetitions per arm, alternating order; timings include foreground and drain,
exclude fixture construction/loading, Verify and quality queries. Each size uses
an identical cloned fixture for both arms. All arms use the fixed query budget
above. These are all-HNSW versus exact measurements, before applying a threshold.

| Initial internal nodes | Records | Exact total median (s) | All-HNSW total median (s) | HNSW change |
|---:|---:|---:|---:|---:|
| 413 | 32,768 | 1.416135 | 1.448947 | +2.3% |
| 793 | 61,440 | 2.091852 | 1.624442 | -22.3% |
| 834 | 65,536 | 2.149647 | 1.701588 | -20.8% |
| 1652 | 131,072 | 3.494182 | 1.875728 | -46.3% |

All 24 runs completed full Verify without issues. Per-query post-write recall
was identical across both arms and all repetitions within each fixture.
At 413 nodes HNSW reduced distance work (about 2.23M distances, versus 3.38M for
one complete scan per update before retries) but did not improve end-to-end time.
At 834 and 1652 nodes total CPU also fell from 2.458 to 1.969 s and 4.265 to 2.694 s;
logical reads fell from 226.46 to 153.41 MB and 627.01 to 348.44 MB respectively.
The benefit therefore is not just time shifted into maintenance.

Select 768 nodes as a conservative candidate above the small-graph crossover;
the nearby 793-node sample already has 22.3% net benefit. This is an operating
point for the measured Cohere workload, not a uniquely optimal or universal
hardware/dimension-independent crossover. The following cold and sustained
hybrid validation tests this selected policy.

Pilot artifacts and binary hash: `.benchmark-data/results/write-routing-threshold-20261025/`.

### Hybrid lifecycle validation at 768 nodes

The same-binary exact/hybrid comparison includes the post-review cancellation
checkpoint in cache-hit exact scoring. All three candidate switches are enabled;
`write_hnsw_min_nodes` is 768. Ordinary exact controls unset all switches.
Each scenario uses three fresh fixture clones per arm with alternating order.

| Scenario | Exact total median (s) | Hybrid total median (s) | Hybrid change |
|---|---:|---:|---:|
| small-warm | 4.084102 | 4.062980 | -0.5% |
| small-cold | 4.097522 | 4.074323 | -0.6% |
| near-cold | 4.132713 | 3.222436 | -22.0% |
| large-cold | 621.184045 | 188.560261 | -69.6% |

Small scenarios repeat 4×8192 updates starting at 199 internal nodes, either
query-warmed or cold graph cache. Their times overlap normal variation: the
result is elimination of the prior 24–25% regression, not a demonstrated small-graph
speedup. Candidate counters show only exact bodies in every round.
Near-cold starts with 793 nodes and repeats 2×8192 updates; candidate counters show
HNSW bodies. Every per-query post-write recall matches exact across these 18 runs.
All complete Verify calls pass; no actionable or transitional partitions remain.

Boundary checks on the same 199-node body select HNSW at threshold 199
(`exact=0, hnsw=1`) and exact at threshold 200 (`exact=1, hnsw=0`). This checks
inclusive selection independently of wall-clock measurements.


The 1M scenario repeats 8×32768 updates (262144 per repetition), starting from
a cold graph cache. Exact totals range from 617.442 to 632.685 s; hybrid totals
range from 182.684 to 188.839 s. Median foreground time falls from 619.155 to
182.555 s (70.5%), measured total CPU from 710.527 to 263.560 s (62.9%), and
logical bytes read from 101.412 to 56.009 GB (44.8%). The cold first-round total
median falls from 78.440 to 24.867 s (68.3%). The faster foreground can leave more
maintenance to drain, which is included in these totals rather than hidden.
Candidate body counters show exact for the three-node root and HNSW for its
three large internal bodies throughout the run.

All six 1M runs complete every Verify without issues, with no actionable or
transitional partitions after settling. Final exact leaf count is 13409;
hybrid counts are 13407–13408. Both retain three level-2 bodies and one root,
maximum leaf occupancy 128, and maximum level-2 occupancy 5264 exact / 5262 hybrid.
Thus the time reduction does not come from materially reducing partition count
or leaving maintenance unfinished.

Final mean recall is 82.63% in every run, under the unchanged beam128/ef64/
rerank125 budget. Across rounds, paired mean hybrid-minus-exact differences are
+0.043,+0.010,+0.027,+0.027,+0.007,+0.017,+0.007,0.000 percentage points.
The final query-resampling 95% interval is [-0.05,+0.05] percentage points
(seed 7, 3000 resamples of the 100 per-query differences averaged across repetitions).
This describes the sampled stationary workload, not a guarantee for every query:
some individual queries lose one neighbor, while others gain one or two.
No mean-quality loss approaching the predeclared 0.5-point concern threshold was
observed. The cold, small and sustained-update concerns that motivated this
experiment are therefore addressed for this Cohere/RocksDB profile.

Two additional functional probes use the validation binary and unchanged budgets:

- A 1M threshold 5150 run routes one large body with HNSW while the other two
  level-2 bodies use exact in the same wave (plus an exact root): counters are
  `exact=3,hnsw=1`. Full Verify passes, and all 100 per-query recall values match
  the corresponding exact 32-update control.
- A threshold 200 run starts with 199 nodes and follows the same four small warm
  update rounds. Round 1 executes 184 exact and 75 HNSW bodies and ends at 200 nodes;
  rounds 2–4 each execute 256 HNSW bodies and end at 201 nodes. This proves selection
  follows changing snapshot Headers rather than a retained old graph size.
  Full Verify and per-query quality comparisons with exact pass in every round.
  These 200/5150 thresholds are correctness probes, not alternative performance
  recommendations.

### Disposition and validation

**The 768-node adaptive policy is demonstrated for the tested workload and is the
recommended default-enablement candidate.** It removes the small-body regression
and preserves the large-body write benefit without raising query budgets.
This commit retains the benchmark-only switch and leaves production defaults
unchanged; promotion should carry the adaptive policy together with the prepared
cache publication, parent-proof reuse, and existing resource/control invariants.
There is no reason in these measurements to retain unconditional HNSW for small
bodies. The evidence remains bounded to stationary swaps, this host/backend and
two maintenance workers; growth, deletion and distribution drift are not measured.

Both ordinary and threshold-enabled focused suites pass: tree routing 26 tests,
graph split interleaving 2, graph transaction budget 1, and graph updates 3
(one unrelated cost probe remains ignored). The lifecycle harness's 3 tests,
workspace formatting, core/benchmark all-target Clippy and core no-default-feature
check pass. Independent review found no actionable defects in the threshold
implementation; cache-hit exact scoring subsequently gained an operation-control
checkpoint, followed by the focused tests and the complete hybrid matrix above.

Artifacts include `pilot.py`, `near.py`, `boundary.py`, `validate.py`, `mixed.py`,
`crossing.py`, per-run reports, binary hashes, `analysis.json`, and source/compiler
metadata under `.benchmark-data/results/write-routing-threshold-20261025/`.
Measurements used arm64 macOS and Rust 1.99.0 (LLVM 23.1.1). `pilot-binary` preserves
the cost-curve executable; `validation-binary` includes the exact-scoring control
checkpoint. No compilation or tests ran concurrently with timing measurements.

## Default policy: 1024 nodes (2026-10-25)

The default policy is now exact below 1024 nodes and HNSW at or above 1024,
per snapshot-authoritative internal Header. The user selected 1024 as a
conservative boundary above the measured crossover. This intentionally leaves
834-node bodies on exact even though the earlier all-HNSW experiment benefited
there. Graph-free indexes and all query algorithms/configuration remain unchanged.
All-exact waves reuse the ordinary scanner without mixed-wave allocations or
HNSW working-set admission. Mixed waves retain shared beam ordering, complete
ownership validation, and final mutation fences.

The four behavioral experiment switches are removed. Prepared-graph cache
publication and guarded parent-proof reuse are always active. Phase timers,
body/distance counters, maintenance timing, and forced cache misses remain
benchmark-only diagnostics; production does not execute phase instrumentation.

### Verification

- Default core suite: 753 passed, two existing ignored probes.
- All-feature core suite: 751 non-doc tests passed; default doctests also passed.
- Adapter/native/harness suites: 112 passed, two existing ignored probes.
- FoundationDB: shared Bulk Build/graph update history and maximum atomic graph
  publication integration tests both passed against the configured 7.3 cluster.
- Clippy across core, native, Memory, RocksDB and benchmark targets with all
  features and warnings denied; formatting; no-default-features check; release
  build all passed.
- Fresh independent review found no actionable issues or material implementation
  concerns. The threshold predicate covers 1023/1024; lifecycle runs validate
  public default routing and mixed exact/HNSW topology. An exact-boundary public
  fixture is not part of the retained regression suite.

Initial expanded-test attempts were interrupted by missing FoundationDB dynamic
library search paths and sandbox-denied localhost binding. Both were rerun with
the documented library environment/local network access and passed; these were
setup failures, not waived assertions. The unrelated `mutable_graph.rs` working
tree changes were preserved and excluded from the default-enablement change.

New measurements use the pinned `f99e17f` validation binary with behavior flags
unset as the exact baseline, and the new default binary with no behavior flags.
Both binaries, source hashes, commands, reports and scripts are retained under
`.benchmark-data/results/write-routing-default-1024-20261025/`. Each run clones
an identical fixture. Small/medium comparisons alternate three repetitions;
the 1M comparison is one new eight-round confirmation pair, complemented by
(but not substituted with) the earlier three-pair 768-node study. No compilation,
tests or other benchmark jobs overlap timings. The temporary local FoundationDB
server was stopped before timing. Query beam128/ef64/rerank125 and write
batch32/beam4/ef64 remain fixed.

### Fresh default-1024 results

| Fixture / initial internal nodes | Updates per run | Exact total seconds | Default total seconds | Total change |
| --- | ---: | ---: | ---: | ---: |
| 16K warm / 199 | 4 × 8192 | 4.139 | 4.257 | +2.9% |
| 16K cold / 199 | 4 × 8192 | 4.169 | 4.162 | -0.2% |
| 64K cold / 834 | 2 × 8192 | 4.416 | 4.312 | -2.4% |
| 128K cold / 1652 | 2 × 8192 | 7.098 | 3.886 | -45.3% |
| 1M cold / root 3, three large bodies | 8 × 32768 | 651.351 | 193.196 | -70.3% |

Small/medium rows are medians of three runs per arm. The 1M row is one new
confirmation pair, not a three-run median. All 26 runs completed. Every round
passed full Verify with exact record/location counts, no issues, no actionable
partitions, and zero remaining split/merge transitional states.

The 199- and 834-node fixtures executed zero HNSW write bodies. The 834-node
case reduced logical reads 29.9% through the maintenance/cache optimization
bundle while retaining exact routing; this comparison does not isolate each
maintenance optimization's contribution. At 1652 nodes, CPU fell 38.5% and
logical reads fell 41.5%. Every small/medium per-query quality result matched
exact. Small warm elapsed time increased 2.9% (CPU +1.8%); cold was essentially
flat. The evidence does not establish zero small-graph overhead or a universal
speedup, but it avoids the earlier roughly 25% all-HNSW small-graph regression.

For 1M, foreground time fell from 649.032s to 187.357s (-71.1%). Total CPU fell
from 725.248s to 267.622s (-63.1%), and logical reads from 101.569GB to 55.992GB
(-44.9%). The first cold round fell from 83.043s to 25.114s (-69.8%). Default
routing executed 8,201 exact bodies and 24,603 HNSW bodies, demonstrating the
small-root/large-internal-body selection throughout ongoing maintenance.
Maintenance drain is included: it rose from 2.319s to 5.839s, but did not offset
the foreground reduction. These are empirical RocksDB measurements, not a
FoundationDB throughput claim.

Final mean recall is 82.63% for both arms at fixed beam128/ef64/rerank125.
Default-minus-exact mean differences by round are +0.04, +0.01, +0.03, +0.03,
+0.01, +0.02, +0.01 and 0.00 percentage points. The final paired query-bootstrap
95% interval is [-0.05, +0.05] percentage points (100 queries, 3,000 resamples,
seed7). Individual final queries can gain or lose one of 100 neighbors; mean
parity is not per-query equivalence. Final leaf counts are 13,409 exact and
13,407 default, with three level-2 bodies and one root in both. Maximum leaf
size remains 128; maximum level-2 body size is 5,264 exact and 5,262 default.

Disposition: retain default adaptive routing at 1024. Net write-lifecycle
benefit is demonstrated for the tested mature 1M workload and supported by the
prior repeated study with identical 1M path selection. Quality and topology
remain stable across these eight stationary corpus-permutation rounds. This is
bounded workload evidence, not a claim about indefinite duration, distribution
drift, every graph configuration, or every query. Query algorithms and budgets
were not optimized or expanded.
