# Integration validation — 2026-09-25

These are functional acceptance runs of **KTANN plus benchmark bridge**, not
isolated performance comparisons. The host ran other validation work during
parts of these measurements, including overlapping Cohere/RocksDB and
SIFT/FoundationDB loads. For publishable measurements, use the idle-host
procedure in [README.md](README.md), longer concurrency stages, and calibrated
quality settings. The measured recall below is not a recall target or guarantee.

These historical runs explicitly used beam 32 and expanded traversal budgets.
See [Cohere default calibration](../cohere-defaults-calibration.md) for the
subsequent quality investigation and public-default budget measurements.

Environment: macOS 26.6.2 arm64, 8 logical CPUs, Python 3.12.9, optimized
all-features bridge, VectorDBBench revision
`1760db148b951363f2282261f30179dfd2ce3790`.
Bridge SHA-256:
`4531685a2f8885dd692dbb9172af9a370109f56fcc4c7bd77fd4e26b4ca5f6d3`.
RocksDB: rust-rocksdb 0.24.0 / RocksDB 10.4.2.
FoundationDB: client 7.3.55, server 7.3.69, API 730.

Each large run uses all 1,000,000 vectors, IDs-only results, k=100,
leaf beam 32, concurrency `[1, 4]`, 5 seconds per concurrency, one loader,
50-record batches, and the same fixed index/Runtime settings documented in the
README. Source and converted-data checksums are retained in each `dataset.json`.

| Dataset / backend | Insert (s) | Optimize (s) | Canonical max QPS | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| SIFT1M L2 / RocksDB | 411.3293 | 0.0415 | 964.8058 | 0.6946 |
| Cohere1M cosine / RocksDB | 1642.4039 | 0.2704 | 104.4039 | 0.6279 |
| SIFT1M L2 / FoundationDB | 1774.2843 | 0.6546 | 95.4739 | 0.6940 |

All three runs finished with 1,000,000 confirmed records, matching leaf
header counts, zero actionable or transitional partitions, canonical success,
both concurrency results retained, companion reports written, and clean bridge
exit. SIFT uses all 10,000 supplied queries; Cohere uses its canonical 1,000
queries. Canonical load duration includes insert plus optimize; maintenance
waiting is included in insert duration.
The full Cohere/FoundationDB combination was not measured; cosine behavior on
FoundationDB is covered by the process tests. This is not a complete benchmark
matrix or quality calibration.

Local complete artifacts live under `.benchmark-data/results/issue-128/`:
`sift1m-rocksdb-final/`, `cohere1m-rocksdb/`, and `sift1m-foundationdb/`.
Each directory includes the exact
invocation, binary hash, dataset provenance, unchanged canonical JSON, client
timings, and native topology/resource/IO/budget reports. These generated files
are intentionally outside source control.

Automated checks passed:

- 38 benchmark-crate Rust tests with all features, including a readiness snapshot
  checked against full index verification and an oversized-root regression.
- Five real bridge/process/official-CLI tests per backend: pickle and spawned
  process handoff, concurrent search, canonical metric relationships, invalid
  requests and size bounds, socket ownership, clean/crash restart, and shutdown
  while a client holds an incomplete frame.
- Workspace all-targets/all-features Clippy with warnings denied, Rust formatting,
  Python compilation, and diff whitespace checks.
- Independent code review and focused verification of maintenance rediscovery
  scheduling; no remaining actionable findings.
