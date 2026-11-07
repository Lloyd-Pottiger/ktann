# Test support

The [validation matrix](../docs/design/runtime-operations.md#8-whole-system-validation)
defines the guarantees; [AGENTS.md](../AGENTS.md#testing) describes their checks.
Tests belong at the smallest seam that can observe a failure. Public-operation
wiring, native backend behavior, and individual fault outcomes remain distinct
coverage even when their setup looks similar.

| Shared responsibility | Owner |
| --- | --- |
| Backend contract cases and byte-redacted comparisons | `support/backend_contract.rs`, used unchanged by every adapter |
| Typed IDs, Tree Keys, transactions, initial trees | `support/builders.rs` |
| Committed topology reads, paged membership, maintenance candidate ordering, drained-branch fixtures | `support/topology_probe.rs` |
| Core unit-test read instrumentation and always-failing transactions | `src/storage/test_support.rs` |
| Independent numeric reference for core search tests | `src/search/test_support.rs` |
| Temporary-directory ownership and cleanup | `tempfile::TempDir` |

Corpus loaders and telemetry capture are declared only by their consumers;
the shared RNG does not depend on dataset loading.

Keep special configurations, transaction boundaries, and fault placement visible
in their tests. Share repeated domain operations, not a configurable framework
that hides those differences. Use standard assertions for scalar results; byte
comparisons must retain redacted diagnostics. Deterministic backend cases need
case labels; generated histories also need their actual replay seed.

Constructors that create their own directories receive an absent child of a
`TempDir`, and its owner stays alive until the artifact is no longer used.
Lifecycle smoke tests use small multi-batch fixtures. Real-vector recall keeps
its full datasets and thresholds, reusing loaded data across compatible phases.

Before removing coverage, identify the retained assertion or compiler constraint
that detects the same regression. Boundary, corruption, unknown-outcome,
committed-topology, and native durability coverage is not replaced by a passing
happy-path test. Changes to shared backend cases require Memory, RocksDB, and
the documented FoundationDB integration tests.
