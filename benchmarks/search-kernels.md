# Resident search kernel trials

The routing-validation and query-widening candidates did not establish a
repeatable whole-search improvement and were restored. A subsequent safe
code-chunk traversal did improve both million-vector workloads and is retained.
The earlier exact-reranking optimization remains unchanged.

## Controlled workload

These trials use the same verified, reopened RocksDB million-vector fixtures as
the resident baseline, including the same supplied 1,000 queries repeated ten
times, k=100, beam 32, eight native actors and eight Tokio workers. Each new
process loads the dataset, audits the retained index, then runs 10,000 untimed
warmup searches and 10,000 measured searches at each ordered client count
`16,4`, followed by an untimed full audit. No imports or concurrent benchmark,
build, or test processes ran during measured regions.

The host is an eight-core Apple M1 Pro with 16 GiB RAM, Darwin 25.6.0, Rust
1.99.0 / LLVM 23.1.1. Builds use release, all backend features and
`RUSTFLAGS='-L native=/Users/lloyd/.local/lib'`. Cohere is 768-dimensional
Cosine with a 4 GiB native block cache; SIFT is 128-dimensional L2 with a
1 GiB native block cache. Both use a 1 GiB decoded Partition Cache. Native
occupancy reported after the final audit was identical between implementations:
4,066,835,156 bytes for Cohere and 787,210,147 for SIFT. This is not timed
per-point occupancy or proof that all records were simultaneously in native cache.

CPU/query is process user+system CPU seconds divided by successful searches.
Latency is the public Search API measurement. Two repetitions show drift; they
are not a statistical confidence interval.

## Remove redundant routing validation

The candidate replaced only the distance calls in
`Traversal::visit_internal` with the existing validated routing kernel.
Query preprocessing and canonical Child Entry decoding already establish
finite values and the Manifest dimension before immutable cached entries reach
this function. Existing routing tests compare scalar/interleaved and symmetric
operand-order results bit for bit across all metrics, dimensions and extreme
values. Other callers retained their validation. No arithmetic order, storage,
cache, concurrency, or budget changes were introduced.

Each dataset used independent process order B/C/C/B. B is the archived
`baseline-ae54feb` executable rebuilt for the resident experiment, rather than
the original fresh-import measurement executable. Search code and runner match
`e645f2f`; that commit changed only documentation.

| Dataset | Clients | Implementation | QPS, two runs | CPU/query ms, two runs | p50 ms, two runs | p99 ms, two runs |
| --- | ---: | --- | --- | --- | --- | --- |
| cohere | 16 | Baseline | 813.32, 833.56 | 9.423, 9.313 | 19.163, 18.710 | 27.868, 28.181 |
| cohere | 16 | Routing candidate | 855.42, 836.27 | 9.096, 9.367 | 18.234, 18.709 | 26.557, 26.847 |
| cohere | 4 | Baseline | 498.56, 504.90 | 7.971, 7.884 | 7.975, 7.900 | 8.523, 8.407 |
| cohere | 4 | Routing candidate | 515.01, 503.90 | 7.730, 7.904 | 7.742, 7.915 | 8.234, 8.421 |
| sift | 16 | Baseline | 2180.92, 2159.99 | 3.602, 3.632 | 7.017, 7.101 | 11.582, 11.633 |
| sift | 16 | Routing candidate | 2172.75, 2162.69 | 3.616, 3.625 | 7.037, 7.062 | 11.620, 11.576 |
| sift | 4 | Baseline | 1323.70, 1308.08 | 3.021, 3.057 | 2.990, 3.026 | 3.286, 3.337 |
| sift | 4 | Routing candidate | 1314.06, 1321.57 | 3.043, 3.027 | 3.014, 2.999 | 3.310, 3.271 |

Cohere mean QPS rises 2.7% at 16 clients and 1.5% at four; mean CPU/query
falls 1.5% and 1.4%, respectively. However the CPU/query repeat ranges overlap,
as do four-client QPS ranges. At 16 clients baseline CPU occupancy was
7.66/7.76 cores versus candidate 7.78/7.83, so increased core availability also
contributes to throughput. SIFT changes are within approximately 1% drift.
Cohere's approximate stage mean decreases around 4%, but exact reranking rises
around 2–3%. This is directional evidence for removing routing work, not a
repeatable whole-system gain across these runs. The candidate was rejected.

## Widen rough query components once

This separate candidate converted query f32 components exactly to an owned f64
slice once during `RaBitQQuery::new`, removing those conversions from every
four-lane rough-scoring group. Even/odd scalar sums, final scaling, interval
rounding, ordering and validation were unchanged. It added one bounded transient
allocation: 8 bytes per dimension (6 KiB for Cohere, 32 KiB at the maximum
supported dimension), released with the search. It changed neither the persistent
codec nor retained cache contents.

After restoring routing, Cohere ran B/C/C/B as `cohere-b3`, `cohere-r1`,
`cohere-r2`, `cohere-b4`; both sides reused the same fixture and settings.

| Dataset | Clients | Implementation | QPS, two runs | CPU/query ms, two runs | p50 ms, two runs | p99 ms, two runs |
| --- | ---: | --- | --- | --- | --- | --- |
| cohere | 16 | Baseline | 833.33, 808.47 | 9.424, 9.373 | 18.726, 19.002 | 27.396, 31.232 |
| cohere | 16 | Widened query | 800.80, 828.98 | 9.630, 9.471 | 19.265, 18.765 | 30.730, 27.995 |
| cohere | 4 | Baseline | 501.90, 504.62 | 7.928, 7.895 | 7.943, 7.907 | 8.447, 8.358 |
| cohere | 4 | Widened query | 493.35, 501.93 | 8.071, 7.931 | 8.083, 7.948 | 8.567, 8.408 |

This trial did not improve QPS, latency or CPU/query consistently. CPU/query
increases about 1.6% at 16 clients and 1.1% at four; approximate-stage means
also do not improve. The candidate was rejected without further SIFT trials.
The numerical tests passed, but removing conversions did not establish lower
overall cost. These measurements do not diagnose the compiler or memory subsystem.

## Correctness and remaining limits

For every point, dataset checksums, audited topology, logical backend IO,
cache hit counts/bytes, leaf-entry distributions, Search Budgets and error counts
match the corresponding baseline exactly. Recall means match within 1e-12
(the concurrent floating reduction can differ in its final bits): Cohere
0.76819, SIFT 0.84608. Each point accepted all 10,000 queries with no errors
and zero physical writes. Physical reads were zero or one 4 KiB read per
Cohere point, and zero for SIFT. These OS counters include native/OS cache
residency; logical IO remains about 3.96 GB of returned bytes per Cohere point.

Both candidates passed 234 library tests (one existing ignored training
benchmark), 43 codec tests, 15 search tests, the public e2e corpus, ground-truth
oracle, formatting and all-target/all-feature Clippy for `ktann`. Numeric tests
include independent rough-score/interval checks and bit-preserving scalar/lane
comparisons. After rejecting both, source files were restored exactly to HEAD;
no candidate-only tests or switches remain.

The resident baseline already establishes approximately 7.8 busy CPU cores at
16 clients on both million-vector datasets. This report establishes no additional
production optimization. Hardware cache misses and memory bandwidth remain
unmeasured; the local `xcrun --find xctrace` cannot find an installed developer
utility. Separate FoundationDB service CPU/RPC behavior remains outside these
RocksDB results; see the [configured FoundationDB limit](search-production.md).
Busy-core counts alone cannot distinguish arithmetic from memory limits.

## Evidence and reproduction

Evidence is under
`/Users/lloyd/projects/ktann/.benchmark-data/results/search-kernels-20261005-31d4`:
all 12 schema-8 reports and logs, exact candidate source diffs, executables,
validation logs, invariant checker and summary. Reports retain the complete
command. Their runtime git revision identifies the checkout when run, not the
binary's build source; use the archived binaries/diffs and hashes below.

| Executable | SHA256 |
| --- | --- |
| baseline | `752444b7332bf1aee95da114ce21084b40469db17d207fec583b53538349d2a9` |
| routing-candidate | `caace6530945284704d9fe009ba70c6f4814decace53dba151f2a072f457a513` |
| rough-candidate | `d70a8be49aefe0b5debf61c1752bf8efd36b97225a4146059b52aacb21fb3571` |

Reproduce with the [resident fixture command](search-residency.md#reproduction-and-artifacts),
adding `--reuse-index true --query-concurrency 16,4` and using each archived
executable in B/C/C/B order with separate output files. Continue comparisons
with reopened fixtures on both sides; a fresh import changes storage/cache
lifecycle and is not a causal counterpart.

## Traverse decoded codes in pairs

The retained candidate changes only `approximate_distances`: after validating
matching dimensions, each lane advances an iterator over two signed codes.
The query still supplies one even and one odd component per iteration, with
separate sums and the same final addition and scale. The trailing odd component,
interval calculation, errors, record loads, and cache representations are
unchanged. The iterators allocate nothing and add no per-search heap storage.
Existing bitwise tests cover both one and four lanes, odd dimensions, numeric
extremes, and all metrics.

The emitted arm64 four-lane baseline loop contained two shared code-index bounds
branches and four code-pointer reloads per pair. The candidate removes those
branches and keeps the pointers in registers outside the loop. Signed-code
conversion and arithmetic remain scalar. This establishes the instruction
mechanism without claiming hardware-cache or bandwidth attribution.

A separate B/C/C/B comparison used the controlled workload above, with the same
fixtures and options. Values below are arithmetic means of two repetitions;
latencies are means of the per-run percentiles, rather than pooled percentiles.

| Dataset | Clients | QPS baseline → candidate | CPU ms/query baseline → candidate | p50 ms baseline → candidate | p99 ms baseline → candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cohere | 16 | 831 → 902 (+8.6%) | 9.327 → 8.630 (−7.5%) | 18.735 → 17.259 | 27.948 → 26.085 |
| Cohere | 4 | 505 → 548 (+8.6%) | 7.887 → 7.251 (−8.1%) | 7.898 → 7.273 | 8.378 → 7.703 |
| SIFT | 16 | 2112 → 2222 (+5.2%) | 3.624 → 3.520 (−2.9%) | 7.194 → 6.916 | 11.884 → 10.962 |
| SIFT | 4 | 1305 → 1352 (+3.6%) | 3.040 → 2.955 (−2.8%) | 3.009 → 2.927 | 3.320 → 3.209 |

Both candidate repetitions outperform both baseline repetitions in QPS,
CPU/query, p50 and p99 at every point. Saturated CPU occupancy was 7.63–7.87
cores baseline and 7.77–7.80 candidate for Cohere, 7.46–7.85 baseline and
7.77–7.87 candidate for SIFT. Some saturated QPS benefit therefore includes
occupancy variation; CPU/query and four-client measurements also improve.
Cohere's four-client approximate-selection mean falls from 5.64–5.68 to
5.03–5.04 ms, while exact reranking stays near 2.16 ms; SIFT's corresponding
approximate mean falls from 2.24–2.28 to 2.16 ms.

All reports have identical dataset, topology, recall (within `1e-12` for
concurrent mean reduction), budgets, visited entries, decoded-cache statistics,
logical backend IO, successful search counts, and empty error counts. Each point
has zero physical writes, with 0 or 4 KiB physical reads for Cohere and zero for
SIFT. Final native cache occupancy matches the earlier trials. This demonstrates
a resident RocksDB search improvement on these corpora and this arm64 host;
it does not establish FoundationDB or other-architecture performance.

Evidence is retained under
`/Users/lloyd/projects/ktann/.benchmark-data/results/search-chunks-20261005-31d4`,
including eight JSON reports and logs, scripts, source diff, binaries, disassembly,
hashes, invariant comparison, and verification logs.

Verification passed: 234 library tests (one existing training benchmark ignored),
43 codec tests, 15 search tests, the public API corpus, and the independent ground
truth test. Formatting and `cargo clippy -p ktann --all-targets --all-features --
-D warnings` passed. An independent read-only review of the full change set found
no actionable issues.
