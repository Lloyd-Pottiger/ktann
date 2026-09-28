# Import routing cache and decoded search kernel

Measured 2026-09-28 against `1e71cf569484d62872ad8f66b534edbe9f32ef10`.
These are same-host RocksDB results, not a cross-platform or FoundationDB SLA.

## Retained changes

- Reuse committed internal partition bodies in foreground pre-mutation routing,
  validated against the current snapshot Header epoch. General routing after
  staged writes remains uncached. Existing membership conflict protection and
  cache capacity remain authoritative; fill buffers are bounded per wave.
- Accumulate decoded query/code products in two independent f64 sums and apply
  scale once. The existing conservative error bound still covers all rounding;
  see [the numerical contract](../docs/design/search.md#3-conservative-approximate-intervals).
  Quantization, exact reranking, budgets and persistent formats are unchanged.

## Construction

Cohere1M, 768d cosine, max partition size 512, write beam 4, 4 GiB cache,
8 executor/blocking workers. Each variant creates a fresh database. Complete
construction includes maintenance convergence and topology verification.

| Metric | Baseline | Routing cache |
|---|---:|---:|
| Complete construction | 516.116 s | 466.245 s (-9.66%) |
| Import phase | 498.182 s | 448.240 s (-10.03%) |
| Complete CPU | 618.241 s | 579.063 s (-6.34%) |
| Import logical reads | 127.941 GB | 44.491 GB (-65.23%) |
| Import scans | 692,221 | 397,213 (-42.62%) |
| Peak process RSS | 4,458.83 MiB | 4,590.70 MiB (+2.96%) |
| Recall@100, beam 136 | 0.92820 | 0.92858 |

Both trees have three levels and no actionable maintenance backlog. This is
one full-size pair, not a confidence interval. Earlier B/C/C/B 200k screening
repeated the mechanism: mean construction -4.28%, logical reads -38.61%,
scans -57.53%, CPU -2.55%. Parallel maintenance can change the final tree;
the routing cache does not claim a tree-quality improvement.

## Search at identical recall

Baseline/candidate/candidate/baseline, all reopening the same baseline 1M
fixture with an unchanged database hash. Each point uses 1,000 distinct queries,
1,000 warmups and 5,000 measured operations, k=100 and 16 concurrent clients.
Repeated operations improve timing stability, not the independent sample count.
Values are arithmetic means of the two runs per variant.

| Beam | Recall@100 (both) | QPS, baseline → candidate | P95 ms | P99 ms | CPU change |
|---|---:|---:|---:|---:|---:|
| 8 | 0.53743 | 1209.84 → 1349.59 (+11.55%) | 15.22 → 13.52 | 16.81 → 14.31 | -8.86% |
| 96 | 0.89830 | 255.23 → 321.01 (+25.77%) | 67.36 → 53.38 | 71.60 → 58.20 | -21.68% |
| 136 | 0.92820 | 184.37 → 244.62 (+32.68%) | 93.18 → 70.90 | 99.66 → 78.36 | -22.79% |

All measured backend IO counters and per-query digests match exactly. Digests
include returned IDs, exact-distance bits, budget usage, exhaustion and
interval-overlap truncation. The kernel adds only a fixed-size stack array,
with no heap allocation or retained state. Per-search-stage RSS was unavailable;
the construction RSS above is a separate process high-water measurement.

## Complementary paths

With the Cohere cache disabled, the single fixed-index pair has identical
query digests and backend IO. QPS is 302.88 → 320.66 at beam 8 and
33.20 → 34.62 at beam 136. This detects no regression but does not establish
a precise no-cache speedup.

SIFT1M (128d L2), also B/N/N/B on a fixed fixture, has identical query digests
and backend IO. At beam 64 / recall 0.93752, baseline QPS is 831.66–923.16,
versus 980.56–984.67 for the candidate. The first baseline followed construction
and was slower than the reopened baseline; treat this as complementary
screening. Both candidate runs beat the faster baseline at beams 8, 32 and 64.

## Tree-quality experiment

A deterministic second balanced-Lloyd start, selected by nearest-centroid
training cost, failed to repeatably improve held-out 200k recall and increased
CPU about 7–8%. It was removed. Better tree quality remains unresolved.

## Reproduction and scope

Host: Apple Silicon macOS, Rust 1.85.0, release build, RocksDB 10.4.2
(`rust-rocksdb` 0.24.0). No builds or tests overlapped performance measurements.
Cohere source/checksum and complete configuration are recorded in each JSON.

Local artifacts are under
`.benchmark-data/results/import-search-2026-09-28` in the main checkout:
`REPORT.md`, `run.py`, `diagnostic-harness.patch`, frozen binaries and hashes,
`b1m.json`, `c1m.json`, `search-{b1,n1,n2,b2}.json`, and per-run provenance.
Apply the diagnostic harness to each source variant in a separate checkout;
it adds selected beams/counts and persisted-index reuse with receipt/hash
validation. The production benchmark CLI has no new diagnostic flags.
The original issue #156/#170 results are historical context, not the baseline
for these comparisons.
