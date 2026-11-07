# HNSW refinement candidates: SIFT1M experiment

Measured 2026-11-07. The original controlled SIFT1M experiment showed lower
construction time and comparable recall/QPS; integration runs were less stable.
All results and limitations are retained below.

Current [refinement](../src/refinement/hnsw.rs) uses exact candidates below 1,024
leaves and HNSW otherwise, independently of serving routing mode. Bulk refinement
admission reserves native build/query memory before planning; offline refinement
reserves the largest graph across sequential trees against its input limit.
Zero rounds reserve no graph. The u32 ordinal domain and memory budget bound
candidate graphs independently of the 65,536-entry partition ceiling. Native
failures and cancellation propagate without algorithm fallback. Forest binding
is `artifact refinement v3`; persistent codecs and the fixed hierarchy policy
are unchanged.

## Method

Baseline: revision `86d948daec5740b633fe9b497b98bd2ac6989cfa` plus the working-tree
[tiled exact refinement optimization](refinement-performance.md). Candidate:
replace only leaf-neighbor generation with an ephemeral HNSW of leaf centroids,
rebuilt every refinement round. Use the existing exact routing-distance callback,
M=8, efConstruction=64, efSearch=64, seed=7; request 33 hits, remove self, retain
up to 32 neighbors. Vector scoring, positive-gain move selection, capacity checks,
and centroid refresh remain unchanged. Partition assignments may differ.

Dataset: pinned SIFT1M, 1,000,000 vectors, 128 dimensions, L2; 1,000 query vectors,
k=100. See [dataset provenance](datasets/README.md). Three refinement rounds,
11,834 final Leaf Partitions. Source hashes, rotation seed and initial index ID
match across builds. Each candidate layout fingerprint matches across its repeats,
including the controlled construction runs and the recall/QPS fixtures.

Host: x86_64, 8 logical CPUs, about 15 GiB RAM; Rust 1.99.1, RocksDB 10.4.2.
Optimized binaries; 8 worker threads and query concurrency 8; 1 GiB partition
cache and 64 MiB RocksDB block cache. Construction/refinement/serving memory limits
are 4 GiB. An initial 2 GiB build failed admission before publication and is
excluded. Process peak RSS is approximately 2.6 GB, including dataset loading;
tmpfs storage consumes host memory outside process RSS.

Final serving graphs are always M=16, efConstruction=200, efSearch=64, seed=7.
Changing the refinement graph does not change serving graph configuration.

## Total construction time

Seconds from source capture through completed Active publication, including load
and cleanup; excludes the separate final audit. Forest time includes refinement,
not just candidate search. Both algorithms in this table use the **same original
benchmark harness**. Each row is an independent fresh index. Temporary construction
files use tmpfs throughout; only the database location changes.

| Database filesystem | Algorithm / repeat | Total s | Forest s | Serving s | Load s | Publish s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Workspace (`fuseblk`) | Exact 1 | 85.951 | 34.220 | 11.307 | 14.801 | 24.399 |
| Workspace | Exact 2 | 88.693 | 34.989 | 11.559 | 16.592 | 24.213 |
| Workspace | HNSW 1 | 87.498 | 33.884 | 10.980 | 15.263 | 24.914 |
| Workspace | HNSW 2 | 80.295 | 26.440 | 10.467 | 16.760 | 25.412 |
| tmpfs | Exact 1 | 72.795 | 33.471 | 11.373 | 11.544 | 14.991 |
| tmpfs | Exact 2 | 73.464 | 34.153 | 10.769 | 11.136 | 16.109 |
| tmpfs | HNSW 1 | 64.905 | 26.910 | 10.427 | 10.197 | 16.067 |
| tmpfs | HNSW 2 | 64.112 | 26.087 | 10.465 | 10.526 | 15.820 |

Means: workspace 87.322 → 83.897 s (3.9% less); tmpfs 73.129 → 64.508 s
(11.8% less). Workspace ranges overlap: a stable disk-backed gain is not established.
The tmpfs forest stage averages 33.812 → 26.499 s (21.6% less). CPU time including
the separate audit averages 87.565 → 78.549 s there (10.3% less).

These are two repeats, not a confidence interval or capacity guarantee. Exact runs
preceded the final controlled HNSW runs; host drift is not eliminated. Earlier
exploratory light-HNSW builds used a changed diagnostic harness and showed noisier
totals (workspace 82.847/93.356 s; tmpfs 69.543/73.820 s). They are retained in raw
reports but excluded from the identical-harness comparison, not silently discarded.
A stronger M=16/efConstruction=200 candidate gave only a 2.3% single-run total gain
at 1M; at 100k it gave no meaningful gain (23.913 → 23.863 s).

## Final recall and QPS at comparable recall

The sweep tests leaf beams 1,2,4,6,8,12,16,24,32,48,64,128,192,256,384 with a fixed
rerank limit of 125. Both variants first reach the following targets at the same
tested beam; these are measured operating points, not interpolated optima.

| Recall target | Leaf beam | Exact recall | HNSW recall |
| --- | ---: | ---: | ---: |
| 90% | 128 | 92.461% | 92.482% |
| 95% | 192 | 95.807% | 95.828% |
| 98% | 384 | 98.871% | 98.879% |

For the 95% target, remeasure both persisted fixtures with the **same reader binary**,
1,000 warmup and 10,000 measured requests, in Exact/HNSW/HNSW/Exact order:

| Algorithm | QPS repeat 1 | QPS repeat 2 | Mean QPS |
| --- | ---: | ---: | ---: |
| Exact | 724.50 | 709.37 | 716.94 |
| HNSW | 721.28 | 716.28 | 718.78 |

The +0.26% mean difference is smaller than run variation: no QPS improvement is
established. The paired recall difference is +0.021 percentage points; bootstrap
95% interval [-0.014, +0.056] percentage points (1,000 queries, 2,000 resamples,
seed=7). This supports comparable recall here, not exact partition equivalence.
Only the 95% operating point received repeated same-reader throughput validation.

## Resource bounds and disposition

The light graph reported native peak 3,743,584 bytes plus 526,600 bytes of query
scratch. Typical per-round graph build took 0.52–0.56 s and neighbor queries
0.51–0.56 s in the exploratory run. The prototype had a 65,536-node ceiling and
native allocation bounds, but lacked combined refinement admission; the measured
runs had ample 4 GiB headroom. Current admission is described above.

End-to-end evidence is limited to SIFT/L2/RocksDB. Other metrics, datasets, memory
pressure and online persistence costs were not established. The policy is based
on the original controlled comparison; later integration measurements do not
establish a stable improvement. More representative disk-backed trials remain
useful; no search-QPS gain is claimed.

## Reproduce and inspect

Raw JSON, compact reports, logs, source snapshots, binaries, scripts and hashes are
under `.benchmark-data/refinement-hnsw/` (local, gitignored). Construction reports:
`exact-1m-r{1,2}`, `controlled-disk-light-r{1,2}`, `local-exact-1m-r{1,2}`,
`controlled-local-light-r{1,2}`. Recall curves: `exact-1m-r1`,
`hnsw-light-1m-r1`; repeated throughput: `qps-{exact,light}-r{1,2}`.

The current release binary uses the measured hybrid candidate policy automatically.
For an exact comparison, save a release binary before changing the implementation,
or use the retained `exact-bench` and `hnsw-light-control-bench` binaries. Baseline
source snapshots and binary hashes are in the raw evidence directory. Use identical
benchmark source for both variants.

Representative construction command (use a fresh database path for every run):

```sh
TMPDIR=/tmp/ktann-refinement-hnsw ./candidate-bench run \
  --backend rocksdb --profile large --scenario quality-sift-1m \
  --worker-threads 8 --query-concurrency 8 \
  --partition-cache-bytes 1073741824 --rocksdb-block-cache-bytes 67108864 \
  --base-vectors 1000000 --query-vectors 1 \
  --warmup-operations 1 --measured-operations 1 \
  --refinement-rounds 3 --leaf-beam-size 1 \
  --routing-experiment '{"mode":"partition-local-hnsw","m":16,"ef_construction":200,"graph_seed":7,"ef_search":64,"build_memory_bytes":4294967296}' \
  --rocksdb-path /tmp/ktann-refinement-hnsw/fresh-index --output result.json
```

Create TMPDIR first. Move only `--rocksdb-path` to the workspace for the other
filesystem comparison. For recall curves omit `--leaf-beam-size`, use 1,000 queries,
1,000 warmups and 3,000 measured operations. For repeated QPS use the retained
fixture with `--reuse-index true`, beam 192 and 10,000 measured operations
(see `reuse.sh` for the precise invocation).

The retained benchmark-only change resolves ground-truth memberships once after
all timing snapshots and reuses them across beam diagnostics on the frozen fixture.
It is not part of the claimed HNSW speedup. Full diagnostics, including memberships,
matched the original harness for all 1,000 queries at beam 192.

## Production integration verification

Both integrated SIFT1M builds have exactly the same leaf-layout fingerprint as the
measured light-HNSW prototype. Beam-192 recall is again 95.828% versus exact
95.807%. The additional integration measurements were **not stable enough to
establish a fresh performance win**:

| Workspace database run | Total build s | Forest s |
| --- | ---: | ---: |
| Integrated hybrid 1 | 115.651 | 50.365 |
| Archived exact control | 91.466 | 35.485 |
| Integrated hybrid 2 | 85.947 | 28.477 |

The hybrid mean of these two runs is 100.799 s; do not select only the faster
repeat to claim improvement. The control binary uses the original diagnostic
harness while integration uses cached diagnostics, so this is an integration
check, not a replacement for the identical-harness construction table above.

Same integrated reader, beam 192, 1,000 warmups and 10,000 measured requests,
chronological order:

| Fixture / repeat | QPS | Query CPU s |
| --- | ---: | ---: |
| Hybrid 1 | 678.99 | 69.564 |
| Exact 1 | 708.54 | 68.581 |
| Hybrid 2 | 337.65 | 97.991 |
| Exact 2 | 496.01 | 78.902 |

All four had zero measured physical read bytes and full partition-cache hits.
Both layouts slowed across repeats; the cause is not established. These numbers
neither reconfirm QPS parity nor isolate a causal candidate-policy regression.
The original controlled evidence remains the basis of the user's decision to
enable the hybrid policy; no stronger production performance guarantee is made.
Raw reports are `integrated-hybrid-1m{,-r2}`, `integrated-control-exact-1m`,
`integrated-hybrid-qps{,-r2}`, and `integrated-control-exact-qps{,-r2}` in the
same local evidence directory.

Validation: workspace/all-target/all-feature Clippy, formatting, independent
review, and 780 core/integration/doc tests passed (7 ignored diagnostics/optional
tests). One unchanged release-mode test, `cancellation_stops_a_long_audit`, failed
both in the full run and alone because the audit completed before its fixed
10 ms cancellation delay; the remaining suite passed with that test filtered.
The new tests cover all three metrics' candidate distance/order/determinism,
unique membership and occupancy after real moves, native cancellation, and
reservation boundaries. No adapter transaction semantics changed.

## Capacity separation validation

Reproducible optimized checks on the same host:

```sh
cargo test --release -p ktann-usearch large_graph_exceeds_partition_capacity_with_bounded_work -- --nocapture
cargo test --release -p ktann --lib candidate_graph_above_partition_capacity -- --ignored --nocapture
```

- Native ordered one-dimensional corpus: 65,537 nodes, M=8, efConstruction=64,
  efSearch=64, seed=7. Build 0.190 s / 12,726,743 distance evaluations; 129 queries
  spanning the whole ordinal range used 14,280 distance evaluations and returned
  the exact nearest point in all cases. Native peak 21,863,328 bytes, query scratch
  4,203,336 bytes. Serialization and reload preserved all bytes and query results.
  An admission limit one byte below the build bound failed before any distance
  callback ran.
- Full refinement candidate generation: 65,537 leaf centers on a two-dimensional
  grid embedded in three dimensions, 0.809 s. Checked every leaf's 32 candidates
  for exact routing distances, sorted order, uniqueness and self exclusion.
  Additional native reservation was 71,628,628 bytes; neighbor lists and retained
  inputs remain separately admitted by callers.

These are boundary/capability checks on synthetic inputs, not final ANN recall,
whole-index build-time or QPS evidence for datasets requiring over 65,536 leaves.
They verify that the larger domain is usable and avoids exhaustive query scans;
they do not claim that graphs approaching the u32 bound are practical in memory
or that the fixed hierarchy policy is optimal at all sizes.

Capacity-change gates passed: 17 native bridge tests; 396 selected core, storage,
Bulk Build, graph update and offline-refinement tests; the separately invoked
large-candidate test; workspace/all-target/all-feature Clippy; formatting and an
independent read-only review. Seven optional/diagnostic tests are ignored by the
normal selected suite, including the large-candidate test run explicitly above.
The default release benchmark binary was rebuilt. Verification logs are retained
locally under `.benchmark-data/graph-capacity/`.
