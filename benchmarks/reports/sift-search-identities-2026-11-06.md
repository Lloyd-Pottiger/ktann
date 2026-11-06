# SIFT1M search identity collection, 2026-11-06

The retained change reduces whole-query CPU consumption on this frozen SIFT1M
workload: **229.234 → 211.688 CPU seconds per 20,000 searches (−7.65%)**.
Recall@100 stays **97.914%**. P95 is essentially unchanged. Median QPS rises
4.10%, but process-to-process ranges overlap, so a stable throughput improvement
is **not demonstrated**.

The implementation stores IDs of at most 16 bytes inline in the query's observed
identity list. Longer IDs retain their existing shared allocation. Every eligible
ID, including screened, rejected and evicted candidates, still participates in
full-byte duplicate validation before record reads. Candidate ranking, corruption
semantics, cancellation checkpoints, persistent data and backend operations are
unchanged. Accounting covers the actual list element size, shared payloads and
the temporary borrowed-key hash table.

[Machine-readable measurements, provenance, patches and profiling script](sift-search-identities-2026-11-06.json)
include all 17 measurement points, not just the final comparison.

## Workload and setup

- Baseline: `fb4bdd872589c2093430e5a27556931939849067`, including the earlier HNSW
  optimizations and RaBitQ property-generator fix.
- Public SIFT1M: 1,000,000 base vectors, 128-dimensional L2, 1,000 held-out queries,
  supplied exact top-100 ground truth. All three files passed the pinned lengths
  and SHA-256 checks in `benchmarks/datasets/sift-1m.json`.
- Library-default partition capacities: leaf 16–128; internal 64–16,384.
  The built tree has one root with 11,834 children and 11,834 leaves, averaging
  about 85 records per leaf. This is a two-level graph-index experiment, not the
  non-graph large-profile three-level acceptance test.
- HNSW M=12, efConstruction=100, efSearch=200, graph seed=7; beam=384, k=100,
  engine-owned rerank ceiling=125. Effective graph expansion can rise to the beam.
  No refinement moves (`refinement_rounds=0`). Benchmark-generated IDs are 8 bytes.
- RocksDB, release build, no Rust target-CPU flags, eight virtual CPUs, four Tokio
  workers, eight concurrent queries. Runtime-default Partition Cache capacity
  was 4,198,857,728 bytes; accounted decoded cache data was 226,612,016 bytes.
  RocksDB block cache remained 8 MiB.
- Initial construction with a 1 GiB preparation budget failed with
  `LimitExceeded`. A separate fixture with a 4 GiB preparation budget completed.
  This establishes a working configuration, not the exact source of that limit.
  All comparisons use the same successfully built, verified fixture.

Each process verifies the reused fixture, warms 1,000 searches, measures 20,000
searches, performs untimed quality replay, and verifies again. No concurrent
Cargo build or second benchmark ran during measurement. Host frequency and
virtualization effects were not controlled. The reported environment is not a
production capacity claim.

## Selecting a meaningful recall point

Beam 32 achieved only 65.187% recall@100. Its initial eight-client throughput
of 1,462 QPS cannot represent high-recall performance. Beam 384 reached 97.914%
and became the comparison point.

An initial high-recall diagnostic with 5,000 operations at concurrency 1/8/1
reported 104.8 / 536.6 / 90.1 QPS. The changing single-client results motivated
longer, alternating-process comparisons. The selected workload visits 385
partitions and roughly 33,193 Leaf Entries per query. All measured cache lookups
hit; physical read bytes were zero. This still includes logical reads, OS
page-cache reads, decoding and native-worker scheduling.

## Final implementation: complete query measurements

Two runs per binary, in baseline/candidate/candidate/baseline order:

| Run | QPS | CPU seconds / 20k | P95 ms | Mean approximate selection ms |
| --- | ---: | ---: | ---: | ---: |
| Baseline 1 | 470.7 | 235.405 | 21.486 | 14.131 |
| Candidate 1 | 475.2 | 219.944 | 22.525 | 13.768 |
| Candidate 2 | 535.0 | 203.432 | 18.817 | 12.166 |
| Baseline 2 | 499.7 | 223.062 | 19.921 | 13.268 |
| Baseline median | 485.2 | 229.234 | 20.704 | 13.699 |
| Candidate median | 505.1 | 211.688 | 20.671 | 12.967 |

Whole-query CPU falls 6.57% and 8.80% in the two paired comparisons. CPU per
search falls from a median 11.462 ms to 10.584 ms. Approximate-selection elapsed
time falls 5.35% at the median. Individual latency and throughput results vary;
these small samples do not justify confidence bounds or a general QPS claim.

All four runs accepted every search without errors. Their complete per-query
quality-replay objects, sorted recall multisets, budget summaries, visited-entry
summaries, cache counters, backend IO and graph work counters agree. The JSON
contains the fingerprints. They do not encode every exact hit distance; core
candidate-selection tests provide the ordering/selection oracle coverage.

## Causal profile

GNU gprofng sampled only the measured search window. A wrapper starts collection
paused, signals the worker when its measurement starts, and pauses it when the
measurement completes. Dataset loading, fixture verification, warmup and quality
replay are excluded from CPU sampling. Profiled runs are separate from the table.

In the baseline, candidate admission accounted for 98.4% of samples attributed
to `Bytes::promotable_even_clone`; that function alone accounted for 8.98% of
collected CPU samples. Duplicate-table insertion, including hashing, accounted
for 16.71%; RaBitQ scoring accounted for 12.51%. These inclusive/exclusive figures
must not be indiscriminately added.

In the final candidate, the sampled clone function no longer has candidate
admission as a caller. Remaining samples come from reranking and storage paths.
That directly corroborates elimination of shared ID reference updates in the
short-ID scan path. Duplicate-table insertion accounts for 9.90% of candidate
samples; other costs, including libc helpers, remain. Sampling attribution is
not exact CPU accounting and does not explain every cycle saved.

The separate 10,000-operation profiles consumed 118.913 / 111.932 process CPU
seconds for baseline/candidate. Treat this as complementary evidence, not an
unprofiled latency or throughput comparison. The primary demonstrated benefit
is lower whole-search CPU on this workload, supported by the source mechanism,
two unprofiled pairs and the independently collected profiles.

## Long-ID check and rejected approach

The first real-data candidate copied all IDs into a contiguous byte buffer plus
end offsets. It improved high-recall median QPS 9.10% and CPU 10.42%, but its
256-byte membership diagnostic regressed about 15%. That approach was rejected.
The final representation avoids those long-payload copies.

The recorded version of the ignored `candidate_membership_performance` diagnostic
collects and validates
32,768 distinct IDs over 64 rounds. Leaf setup is outside timing; the test-only
precomputed-score batch construction remains inside timing in both binaries.
There is no vector scoring, routing or backend IO. Three separate processes per
binary were alternated; these are membership proxies, not complete queries.

| Representation | ID length | Median microseconds / scan | Accounted bytes |
| --- | ---: | ---: | ---: |
| Baseline, final diagnostic window | 8 | 1633.538 | 1,900,608 |
| Final inline/shared | 8 | 1226.154 | 1,638,464 |
| Baseline, final diagnostic window | 256 | 3262.926 | 10,027,072 |
| Final inline/shared | 256 | 3231.810 | 10,027,072 |

The short-ID proxy improves about 25%. Long-ID medians are within 1%; its
candidate range of 3092–3436 µs overlaps the baseline range of 3248–3290 µs.
No repeatable long-ID regression was observed in this diagnostic. The current
harness now borrows a reusable score batch and source location, excluding their
per-batch preparation from timing. The numbers above retain the original harness;
use the archived patch to reproduce them, or apply the same current harness to
both binaries for a new comparison. Accounted
bytes are conservative query working-memory charges, **not measured RSS** or
an assertion about total process memory. Long-ID end-to-end search remains
unmeasured.

## Validation and reproduction

- 308 core tests passed, 3 performance diagnostics ignored by the normal suite.
  No RaBitQ test was excluded.
- 50 integration tests passed: search, Tree Key routing, graph updates and e2e.
- Clippy passed for `ktann` and `ktann-usearch`, all targets/features, with
  warnings denied. Formatting and diff whitespace checks passed.
- Added coverage for binary prefixes, 16/17-byte representation boundaries,
  maximum-length IDs, duplicate membership and original leaf release.
- Independent read-only review found no actionable issues or material
  implementation concerns. The reviewed snapshot is preserved in the companion JSON.
  Subsequent cleanup changes only test input ownership and a cancellation assertion;
  the measured production implementation remains unchanged.

Use the pinned dataset setup in `benchmarks/datasets/README.md`. The companion
JSON contains the complete argument array, baseline revision, binary hashes,
reviewed patch, environment and measurement/profiler script. Keep baseline and
candidate executables separately and alternate them on one frozen fixture;
never compile concurrently with measurement. The recorded `git_revision` from
the runner is evaluated at execution time, so use the artifact hashes to identify
the copied binaries.

```sh
cargo build --release -p ktann-benchmarks --bin ktann-bench
cargo test --release -p ktann --all-features --lib candidate_membership_performance -- --ignored --nocapture
```

Raw files, verified datasets, fixture and executables remain locally under
`.benchmark-data/sift-profile/` and `.benchmark-data/datasets/`. Neither the
small synthetic workload nor this real-data experiment compares matched Milvus
settings. Filtered, cold-cache, mixed read/write, longer-ID complete-query,
Cohere/cosine and other-backend performance remain unmeasured. This result
supports optimizing candidate identity handling; it does not establish a need
to replace HNSW or an overall performance gap to another database.
