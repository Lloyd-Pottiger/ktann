# Repository Guidelines

## Project Sources of Truth

KTANN is in active implementation with no stable release. Do not add backward
compatibility machinery unless the task explicitly requires it. Read `README.md`
for project status; verify available APIs and behavior in code because parts of
the design are not yet implemented.

- `CONTEXT.md`: canonical domain language and system-wide invariants.
- `docs/design/overview.md`: product boundary, authoritative invariants, target
  architecture, end-to-end behavior, and implementation order.
- `docs/design/`: detailed contracts for the public API, storage, search,
  maintenance, and runtime/operations modules.
- `docs/adr/`: accepted architectural decisions and their rationale. Add an ADR
  only for a new hard-to-reverse decision; do not rewrite an accepted decision
  silently in code.

Any local `refwiki/` material is background reading, not an authoritative KTANN
contract. Do not edit or depend on it unless the task explicitly requires it.

## Commands

- Build: `cargo build --workspace`
- Test: `cargo test --workspace`
- Focused test: `cargo test -p <crate> <test_name>`
- Check: `cargo check --workspace --all-targets`
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Format: check with `cargo fmt --all -- --check`, apply with `cargo fmt --all`

Select checks by the changed contract. For Rust changes, run formatting, Clippy,
and relevant tests; use workspace-wide checks for shared contracts or changes
across crates. For documentation-only changes, check accuracy, links, and the
diff; Cargo checks are unnecessary. After relevant checks pass, broaden testing
only for unresolved risks or failures. Report what ran and any verification gaps.
Never run Cargo commands concurrently: they contend on Cargo and target-directory
locks. Use Rust Edition 2024, MSRV 1.85, and stable CI; no nightly-only production
features.

## Workflow Principles

- Before a non-trivial change, read `CONTEXT.md`, the relevant sections of
  `docs/design/overview.md`, the owning module design, and every directly
  relevant ADR. Trace callers, persistent state, backend behavior, and runtime
  effects before editing.
- Use the domain terms from `CONTEXT.md` exactly. Do not introduce synonyms
  such as "table" for Logical Index or "partition key" for Tree Key.
- Implement in the dependency order in the overview unless the task establishes
  a smaller self-contained vertical slice. Do not add placeholder abstractions
  for later stages.
- Keep each responsibility at its documented owner: logical codecs and atomic
  index operations in core storage; backend limits and error classification in
  the adapters; lifecycle and admission behavior in runtime/operations.
- Treat a discrepancy among code, design, and ADRs as a decision to resolve,
  not permission to choose whichever is easiest. Preserve behavior outside the
  requested change. Update affected designs and `CONTEXT.md` when changing a
  contract or domain language; record hard-to-reverse decisions in ADRs.
- Keep commits and diffs scoped to one coherent outcome. Do not mix formatting,
  dependency churn, or unrelated cleanup into a behavioral change.
- Continue authorized work through verification. Resolve routine choices from
  context; ask only when missing information materially affects the outcome or
  an action needs authorization not already given. If an instruction blocks
  completion, cite its file and exact requirement and explain why it applies.
- Keep updates and final reports concise: outcome, meaningful verification,
  and unresolved decisions.

## Correctness and Storage Rules

- Preserve the exact-membership invariant: every committed Vector Record has
  exactly one Record Location and one corresponding Leaf Entry.
- A Foreground Mutation atomically updates the record, location, leaf
  membership, exact counts, and affected synopses. Never split this contract
  across best-effort writes or asynchronous repair.
- Every committed split or merge state must remain searchable. Structure
  Maintenance may be delayed or lost from process-local queues; correctness
  cannot depend on a durable worker, lease, or coordinator.
- Persistent Logical Index IDs and Partition Keys are never reused. Gaps are
  valid. Follow `docs/design/storage.md` for explicit format changes and
  independent key/value codec versioning.
- Use canonical, deterministic codecs. Reject malformed and noncanonical
  bytes; do not silently normalize persistent data.
- Fail closed: invalid persistent encoding or invariant mismatches are
  `Corruption`; invalid caller input and non-finite caller-derived arithmetic
  are `InvalidArgument`. Do not skip, repair, or hide corruption on hot paths.
- Preserve transaction semantics across the deterministic test backend,
  FoundationDB, and RocksDB. Expose real backend capability differences
  explicitly; never weaken the shared contract to accommodate an adapter.
- Unknown commit outcomes must follow the documented idempotency/recovery
  protocol. Never report success, retry a partial mutation, or allocate a new
  identity based on an uncertain outcome.

## Search, Concurrency, and Performance

- One search uses one consistent backend snapshot for manifest validation,
  Tree Key enumeration, traversal, filtering, record loads, and exact
  reranking.
- Filter predicates are exact. Partition Synopses are conservative pruning
  aids: `NoMatch` must prove impossibility and `AllMatch` must prove every
  entry matches.
- Search is bounded and deterministic: stable traversal and tie-breaking
  order, every budget dimension accounted for, truncation exposed rather than
  claiming exact global top-k or guaranteed `k` results.
- Keep all queues, retries, scans, transactions, caches, concurrency, and
  memory bounded. Respect backend transaction and blocking-resource limits.
- Avoid per-vector allocation, repeated decoding or conversion, unnecessary
  copies, unbounded fan-out, coarse locks, and blocking work on async executor
  threads. Make performance claims only with reproducible benchmarks.
- Cache entries are only hints. Reuse decoded partition data only after
  validating its epoch and kind in the search snapshot; never cache
  corruption.

## Rust Style

- Prefer small, explicit interfaces and typed domain operations over generic
  KV access, boolean flags, or stringly typed state.
- Make illegal states hard to represent, but do not mirror persistent state
  with redundant in-memory authorities.
- Use checked arithmetic and explicit conversions for IDs, counts, sizes, and
  budget accounting. Handle floating-point edge cases according to the numeric
  contract.
- Return structured errors with useful context while keeping vectors,
  payloads, filter values, and raw Tree Keys out of logs and error messages.
- Avoid `unwrap`, `expect`, and `panic!` in production paths unless an
  invariant is statically guaranteed and documented. Production libraries
  forbid `unsafe`; keep any test-only FFI exceptions narrowly scoped.
- Document public APIs in backend-neutral terms. Use `rustfmt` defaults and
  keep Clippy clean under the repository command above.

## Testing

Choose the relevant coverage below; this is not a checklist for every change.
Test observable contracts with deterministic, replayable inputs, using the
owning design's evidence matrix to select the layer without duplicating coverage.

- Backend semantics: run the shared contract suite unchanged on affected
  backends; shared-contract changes cover the deterministic backend, FoundationDB,
  and RocksDB. Cover conflicts, snapshots, read-your-writes, pagination, limits,
  rollback, commit outcomes, durability, and declared capabilities.
- Persistent formats: golden bytes, ordering properties, malformed/noncanonical
  inputs, and cross-process determinism.
- Mutation and maintenance: model/history tests and fault injection for exact
  membership, retries, unknown outcomes, crashes, and committed topology states.
- Search: SQL three-valued-logic and brute-force numeric oracles for predicates,
  synopsis pruning, and reranking; cover changed resource limits and truncation.
- Performance: reproducible recall, latency, contention, memory, and write
  amplification measurements, without freezing tunable implementation details.

Use these entry points for harness details and commands:

| Concern | Reference |
| --- | --- |
| Public API corpus and expectation regeneration | `tests/e2e.rs`, `tests/datadriven/*.kddt`; run `KTANN_REWRITE=1 cargo test --test e2e` only for intended expectation changes and review the diff |
| Real-dataset provenance and ground truth | `tests/datadriven/data/README.md`, `tests/oracle_groundtruth.rs` |
| Metric labels/counts and telemetry privacy | `tests/metrics.rs`, `tests/observability.rs` |
| Seeded crash/recovery replay and expanded profile | `tests/model_history.rs`, `.github/workflows/nightly.yml` |
| Production-adapter recall parity | `ktann-rocksdb/tests/rocksdb_recall.rs`, `ktann-foundationdb/tests/foundationdb_recall.rs`, shared `tests/support/adapter_recall.rs` |
| Benchmark profiles, FoundationDB setup, and report comparability | `benchmarks/README.md`; run optimized `full` profiles on an otherwise idle host |

FoundationDB integration tests need a local cluster; follow the test's documented
invocation, including `--ignored` where required. Benchmark results are empirical,
not an SLA.
