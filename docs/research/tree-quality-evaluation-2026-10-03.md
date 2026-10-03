# Cohere1M tree-quality evaluation

Date: 2026-10-03. Baseline: `d0cf608`. Companion: [primary-source assessment](tree-quality-options-2026-10-03.md).

## Result

The strongest tested direction is **capacity-bounded post-import local
refinement with single membership**. On one fixed full Cohere1M RocksDB index,
five rounds reduce beam 100 to 80 at essentially matched recall
(0.90284 to 0.90297). Two native repetitions per fixture average 242.6 to 290.8
QPS (+19.9%), with 19.1% less query CPU and 15.0% fewer scanned entries.

This demonstrates a static-model query-efficiency improvement under the tested
conditions. It does **not** establish complete import overhead, crash-safe
publication, online maintenance behavior, another dataset, or FoundationDB
performance. The algorithm remains an archived isolated prototype; production
search/storage code is unchanged. The retained code adds precise benchmark
sweep controls needed to make this comparison reproducible.

## Objective and evidence standard

Improve actual search throughput at recall@100 >= 0.90, allowing additional
resources and modest import overhead. Beam size is a control, not the objective:
more entries per selected leaf or duplicate memberships can erase its benefit.
Do not change the exact-membership contract for an unmeasured benefit.

All native runs below use the complete shuffled Cohere1M dataset, cosine,
1,000 distinct queries, 10,000 measured operations per point, concurrency 16,
eight Tokio workers, RocksDB, and the existing default import/cache settings.
Each fresh build completes convergence and topology verification before and
after its sweep. Candidate training does not use query labels. QPS varies with
host conditions; entry counts and process CPU support or contradict the
proposed mechanism. These are local screens, not production SLAs or proof of
optimality across backends and datasets.

Raw JSON, logs, binaries, candidate patches, snapshot exports, and offline
scripts are under
`/Users/lloyd/projects/ktann/.benchmark-data/results/tree-quality-20261003`.
The clean baseline binary's provenance is recorded in `baseline-source.json`;
`git_revision` in some reports reflects the subsequently edited worktree rather
than the immutable executable. Use archived binaries/patches for attribution.

## Native training and routing screens

| Algorithm / beam | Recall@100 | QPS | Mean entries/query | CPU ms/query |
| --- | ---: | ---: | ---: | ---: |
| Original, fine build / 96 | 0.89815 | 285.4 | 35,135 | 24.41 |
| Original, fine build / 112 | 0.91157 | 248.6 | 40,999 | 27.45 |
| Original, threshold build / 96 | 0.89919 | 266.0 | 35,517 | 23.60 |
| Original, threshold build / 100 | 0.90284 | 291.6 | 37,003 | 24.65 |
| Original, threshold build / 104 | 0.90643 | 284.2 | 38,490 | 25.60 |
| Bounded 25/75 assignment / 96 | 0.89959 | 287.1 | 34,787 | 24.77 |
| Bounded 25/75 assignment / 112 | 0.91357 | 260.1 | 40,595 | 27.22 |
| Four deterministic starts / 96 | 0.90233 | 269.6 | 35,772 | 24.39 |
| Four deterministic starts / 112 | 0.91536 | 229.3 | 41,717 | 27.72 |
| Raw means + squared-L2 routing / 104 | 0.89750 | 268.0 | 38,239 | 26.42 |
| Raw means + squared-L2 routing / 112 | 0.90389 | 252.8 | 41,177 | 28.11 |

These are actual operating points, not identical-recall comparisons. The
nonmonotonic baseline QPS and differences between fresh builds make a small
wall-time-only improvement insufficient.

- **Bounded assignment:** keep nearest-centroid clusters between 25% and 75%
  instead of exactly half during Lloyd training. At beam 128 versus the initial
  coarse baseline, recall increases only 0.105 percentage points, scans fall
  1.55%, and CPU rises 1.89%. No material benefit demonstrated; reverted.
- **Multiple starts:** original farthest-pair seed plus three canonical-position
  seeds, choosing the lowest nearest-center training loss. A small quality
  signal exists, but beam 96 versus baseline 112 overstates it: baseline 100
  already reaches the target. Construction versus the fine baseline costs
  +9.2% wall and +14.7% CPU. Query CPU at the close operating points differs by
  about 1%, across different builds. No repeatable QPS benefit demonstrated;
  reverted.
- **Raw means with L2 routing:** preserve normalized record preprocessing and
  exact cosine ranking, retain raw arithmetic centroids, and change cosine
  routing to squared L2. This is distinct from historical raw-mean/negative-dot
  tests. Recall and scan/CPU efficiency worsen here; reverted.

Baseline construction wall times range from 451 to 592 seconds across these
runs, underlining why import wall-time alone cannot rank the algorithms.
The original tree has roughly 2,740–2,770 leaves and eight intermediate nodes.
For beam >=16, traversal covers every intermediate node. High-beam leaf-cache
lookups all hit: this fixture's main recall loss is leaf selection, not pruning
of ancestors or cache misses.

## Fixed-snapshot routing diagnosis

`baseline-routing.json` exports native-preprocessed queries, leaf centroids,
leaf counts, and exact-neighbor owners from one verified index snapshot.
Flat centroid ordering reproduces native mean entry counts and recall at beams
64, 96, 100, 104, and 112. The same equality holds at all nine measured points of
the raw-mean/L2 fixture. On these fixtures, approximate selection and exact
reranking therefore do not account for the observed recall loss.

A ratio cutoff on the baseline cosine loss, capped at 112 leaves, reduces held-out
mean scans from 38,130 to 36,083 (5.4%) while recall falls from 0.90424 to 0.90295.
The cutoff was selected on queries 0–199 for a 0.91 calibration target and
inspected on 200–999. This is an internal threshold holdout in a corpus already
used for architecture screens, not independent validation. The L2 fixture saves
only 1.5–2.5%. Neither is an end-to-end speedup; no cutoff was retained.

## Current-data reassignment and replication screens

`baseline-full-routing.*` also contains the native-preprocessed million vectors,
current owners, and exact-neighbor record ordinals. Each vector considers the
32 centers nearest its current owner center. Centers remain fixed; ranking the
queries is unchanged. Candidate placement does not inspect query labels.

| Frozen-center policy | Total memberships | First integer beam at >=0.90 coverage | Mean entries |
| --- | ---: | ---: | ---: |
| Original | 1,000,000 | 97 | 35,886 |
| Move to closest candidate | 1,000,000 | 90 | 33,519 |
| Keep original + closest alternative, ratio 1.00 | 1,187,047 | 81 | 36,446 |
| Keep original + closest alternative, ratio 1.05 | 1,542,184 | 69 | 41,634 |
| Keep original + closest alternative, ratio 1.10 | 1,750,491 | 63 | 43,323 |
| Keep original + closest alternative, ratio 1.20 | 1,916,832 | 59 | 44,163 |
| Reassign + up to three centers, ratio 1.05 | 1,740,898 | 66 | 46,414 |
| Reassign + up to three centers, ratio 1.20 | 2,741,630 | 47 | 52,286 |

The ratio is applied to `1 - dot(vector, center)`, relative to the nearest
candidate's loss. This is a simple closure-style screen, not a reproduction of
SPANN's global balanced clustering or RNG-filtered replication algorithm.
Duplicate entries count toward scanning; neighbor hits are deduplicated.

18.7% of vectors have a closer candidate center than their owner. Moving them
saves only 6.6% scans and creates 26 over-capacity leaves (maximum 607). Capacity
repair, additional splits, live-write revalidation, mutation bytes, and actual
QPS are unmeasured. Replication reduces beam but increases scanned entries, even
before its much larger capacity and membership-maintenance costs. Historical
migration results alone did not decide this: these are fresh current-data
screens. Neither result justifies changing exact membership.

## Leaf granularity and representative centers

Recursively split each existing leaf in memory using ten rounds of balanced
spherical two-means, without query labels. Compare virtual subleaf selection
with keeping the original physical leaf and ranking it by its nearest trained
representative. All numbers here are routing proxies with frozen original
membership boundaries; native online construction may differ.

| Offline policy | Beam at >=0.90 coverage | Mean entries |
| --- | ---: | ---: |
| Refresh one mean per leaf | 97 | 35,850 |
| Two virtual subleaves per leaf | 168 | 31,077 |
| Four virtual subleaves per leaf | 287 | 26,571 |
| Two representatives, original leaves | 96 | 35,498 |
| Four representatives, original leaves | 93 | 34,380 |

Finer granularity has a more promising scan reduction than multiple
representatives of the same physical leaf. It also requires more centroid
arithmetic and leaf visits. The native capacity-256 run reaches recall 0.90358 at beam 176, scanning
32,553 entries, but costs 27.78 CPU ms/query and yields 244.9 QPS. The close
baseline point (beam 100, recall 0.90284) scans 37,003 entries but costs only
24.65 CPU ms/query and yields 291.6 QPS. Scans fall 12.0% while CPU rises 12.7%.
Construction costs 526.16 wall / 642.22 CPU seconds. More centroid work and
partition visits erase the scan reduction. No capacity-default change is retained.


## Capacity-bounded post-import local refinement

Unlike one fixed-center migration, alternate nearest-candidate assignment and
recomputation of normalized means. Every vector considers 32 neighboring
centers. Admit proposed moves by descending reduction in cosine loss while
preserving source count >=16 and target count <=512. Recompute means from the
resulting members. Query labels never select moves, centers, or stopping points.
This is local spherical Lloyd refinement with greedy capacity bounds, not a
globally optimal balanced clustering solver.

| Rounds | Beam at >=0.90 coverage | Mean entries | Fraction moved from original |
| --- | ---: | ---: | ---: |
| 0 | 97 | 35,886 | 0% |
| 1 | 87 | 32,530 | 18.6% |
| 2 | 83 | 31,579 | 27.2% |
| 3 | 81 | 31,245 | 32.2% |
| 4 | 79 | 30,785 | 35.6% |
| 5 | 78 | 30,655 | 38.1% |

Every refined leaf remains within 16–512 entries. The five-round scan reduction
is 14.6%; two rounds already achieve 12.0%, so incremental import cost matters
when choosing an iteration budget. These are still routing proxies until the
native comparison below establishes their effect.

An isolated copy of the baseline RocksDB fixture receives the five-round model.
The diagnostic rewrites moved Leaf Entries and Record Locations, exact counts,
leaf Synopses, leaf centers, parent Child Entry projections, and cache epochs.
RaBitQ7 is absolute, so its bytes remain unchanged. Existing Vector Records,
payloads, partition IDs, internal centers, query processing, and budgets remain
unchanged. No runtime serves the fixture while it is rewritten. Full native
verification must pass before and after every query measurement.

The rewrite takes 56.9 seconds including dataset loading. This is not an
end-to-end import-overhead measurement: snapshot export, model training,
additional resident memory, publication/recovery, and production-adapter costs
must also be charged. The prototype is archived, not retained as a public API
or online maintenance operation.

## Repeated native comparison

Use the unchanged archived search executable and the original/refined copies
of the same index. Run candidate, baseline, baseline, candidate, with 10,000
operations at each of two beam values per run. Every run passes complete
verification before and after measurement. Construction fields in these reuse
reports are **not** import measurements.

| Run | Beam | Recall@100 | QPS | CPU ms/query | Mean entries | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 1 | 100 | 0.90284 | 257.7 | 24.78 | 37,003 | 84.72 |
| Baseline 2 | 100 | 0.90284 | 227.5 | 25.93 | 37,003 | 106.85 |
| Refined 1 | 80 | 0.90297 | 301.9 | 20.09 | 31,442 | 59.26 |
| Refined 2 | 80 | 0.90297 | 279.7 | 20.93 | 31,442 | 68.96 |

The arithmetic mean QPS rises 19.9%, CPU/query falls 19.1%, and deterministic
entry work falls 15.0%. QPS varies appreciably with the host, so the magnitude
is local empirical evidence, not a promised production uplift. Both candidate
runs improve throughput and CPU; the measured scan and leaf-visit reductions
support the mechanism. A lower operating point also works: original beam 97
has recall 0.90024 and 35,886 entries; refined beam 78 has recall 0.90061 and
30,655 entries. The higher pair above matches recall more closely.

Reproduce the four measurements with the archived `paired-refinement.py`.
`refined-native-{1,2}.json` and `paired-baseline-{1,2}.json` contain full stage,
IO, budget, latency, and cache counters. This repetition holds the trained model
fixed; it does not measure variation over independent refined builds.

## Production direction and remaining boundaries

The most promising tested direction is **capacity-bounded local refinement at
the end of bulk construction, preserving one membership per record**. It treats
the stabilized set of leaves as a clustering model, instead of making every
online split pay for quality repair. Consider a bounded iteration budget (the
screen uses five; two capture most of the scan reduction), 32 neighboring
centers per leaf, and the existing occupancy limits. Training loss, not query
labels, must drive assignment and termination.

Production publication needs a separate design: published centroids are
immutable today, and Import Session does not grant exclusive ownership of a
serving index. Build/refine/verify an unpublished index, then publish it
atomically. Failure should leave the current serving index intact and allow
an incomplete build to be discarded or resumed. Do not expose the diagnostic
raw rewrite as an online `optimize` operation. Foreground mutation, deletes,
filtered queries, snapshots, maintenance, and verification must retain their
present single-membership guarantees.

Before production retention, measure a complete construction lifecycle rather
than adding isolated phase times; include export-equivalent reads, training,
relocations, exact counters/Synopses, publication, recovery, memory, and write
amplification. A production implementation should avoid the diagnostic JSON
and vector export round trip. Compare a small iteration budget against five
rounds using native timing, and verify the winning policy on another dataset
and FoundationDB. The current fixture visits all intermediate nodes at the
measured beams; deeper trees or narrower ancestor beams can expose additional
routing loss, so these measurements do not validate arbitrary tree depths.

Boundary replication is not the preferred first change on this evidence. The
simple closure-style variants here spend more scan work for lower beam and
would expand the membership/update contract. This does not disprove SPANN:
its global clustering, navigation, and replication selection are a larger
combined design than the isolated policy screened here.


## Retained changes and validation

- Benchmark CLI: `--leaf-beam-sweep` accepts strictly increasing positive
  widths; `--measured-operations` controls repeated measurements without
  changing the query corpus or warmup. Parent/worker forwarding and resolved
  report configuration preserve the selected experiment.
- No production tree, membership, routing, storage format, or maintenance
  algorithm change is retained. Rejected variants and temporary probes are
  archived outside the source tree for reproduction.
- `cargo test -p ktann-benchmarks --lib cli::tests`: all nine pass.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: pass,
  with the configured FoundationDB native library path.
- `cargo fmt --all -- --check` and `git diff --check`: pass.
- Seven fresh full Cohere1M construction/sweep runs and four paired fixed-index
  runs complete native convergence/integrity verification. FoundationDB runtime
  and a second dataset were not benchmarked for this prototype.
