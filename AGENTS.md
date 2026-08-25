# Repository Guidelines

## Project Sources of Truth

Read `README.md` for the project status, goals, and architecture overview.
Parts of the design are not yet implemented; do not infer available API or
behavior from design documents alone.

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

Run focused checks while iterating; before completing work, run formatting,
clippy, and the relevant workspace tests. Never run Cargo commands
concurrently: they contend on Cargo and target-directory locks. The toolchain
is Rust Edition 2024 with MSRV 1.85 and current stable CI; production code
must not require nightly features.

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
- For substantial changes, define the private module layout before
  implementation; do not accumulate multiple responsibilities in one file.
- Keep each responsibility at its documented owner: logical codecs and atomic
  index operations in core storage; backend limits and error classification in
  the adapters; lifecycle and admission behavior in runtime/operations.
- Treat a discrepancy among code, design, and ADRs as a decision to resolve,
  not permission to choose whichever is easiest. Preserve shipped behavior,
  and when intentionally changing a contract or domain language, update the
  relevant design, ADR, and `CONTEXT.md` in the same change.
- Keep commits and diffs scoped to one coherent outcome. Do not mix formatting,
  dependency churn, or unrelated cleanup into a behavioral change.

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
  valid. Persistent format changes must be explicit and versioned as one whole.
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
  invariant is statically guaranteed and documented. Do not use `unsafe`
  without a narrow, reviewed justification and dedicated tests.
- Document public APIs in backend-neutral terms. Use `rustfmt` defaults and
  keep Clippy clean under the repository command above.

## Testing

- Test externally observable contracts, not implementation shape. Prefer
  deterministic tests with replayable seeds.
- Add focused regression coverage for each changed guarantee, at the layer the
  module designs' evidence matrixes prescribe; do not duplicate tests at every
  layer.
- Run the shared backend contract suite unchanged against the deterministic
  test backend and each production adapter, covering conflicts, snapshot
  consistency, read-your-writes, pagination and limits, rollback, commit
  outcomes, durability, and declared capabilities.
- Protect persistent formats with golden bytes, ordering properties,
  malformed/noncanonical corpora, and cross-process deterministic vectors.
- Use model/history tests and fault injection for exact membership, retries,
  unknown outcomes, crashes, and every committed topology transition.
- Check predicate evaluation and synopsis pruning against a SQL
  three-valued-logic oracle, and exact reranking against a brute-force numeric
  oracle.
- Test every resource boundary and truncation reason. Benchmarks report
  recall, latency, contention, memory, and write amplification without
  freezing benchmark-tunable internals such as cache eviction or task layout.
- The data-driven integration corpus lives in `tests/datadriven/*.kddt`,
  executed by `tests/e2e.rs` against the public API on the deterministic
  backend with seeded synthetic datasets (`tests/support/dataset.rs`), a
  brute-force oracle (`tests/support/oracle.rs`), and the persistent-state
  audit (`tests/support/audit.rs`). Regenerate expectations with
  `KTANN_REWRITE=1 cargo test --test e2e` and review the diff like any other
  change. Real-dataset fixtures (siftsmall, fashion-mnist; see
  `tests/datadriven/data/README.md` for provenance) are checked in under
  `tests/datadriven/data/` and loaded via `file:NAME[:N]` dataset specs (the
  optional `:N` takes the fixture's first N vectors); the oracle is
  cross-checked against published siftsmall ground truth in
  `tests/oracle_groundtruth.rs`.
- Metric recording is asserted in `tests/metrics.rs`: the documented `ktann.*`
  series fire with the expected labels and counts as the public API drives
  work. Telemetry privacy (no caller data in metrics or traces) is audited in
  `tests/observability.rs`.
- Reproducible ANN and whole-system baselines live in the non-published
  `ktann-benchmarks` workspace crate. Run the fast production-adapter matrix
  with `cargo run -p ktann-benchmarks --bin ktann-bench -- run --backend rocksdb
  --profile smoke`; run optimized `full` profiles only on an otherwise idle
  host. `benchmarks/README.md` defines timing boundaries, logical write
  amplification, FoundationDB setup, report comparability, and why these
  empirical results are not a v1 SLA.
- The replayable crash-history and model-validation harness lives in
  `tests/model_history.rs` (issue #37): one seeded, fully pre-generated script
  drives the public API through lifecycle transitions, atomic Foreground
  Mutations (some armed with commit faults), manually advanced split/merge
  transitions, queue loss via crash/reopen, unknown commit outcomes,
  cancellation, and shutdown, asserting exact membership, Partition Key
  non-reuse, and Logical Index ID non-reuse after every step. Determinism
  comes from zero maintenance workers with manually driven bounded advances
  plus a script drawn from one seed before any async work. A failure prints
  the step trace and a replay command; reproduce a seed with
  `KTANN_MODEL_SEED=<seed> cargo test --test model_history model_history_replay`
  (optionally `KTANN_MODEL_STEPS=<n>`), and run the expanded deterministic
  profile (24 seeds × 400 steps) with `KTANN_MODEL_PROFILE=expanded`. The
  nightly workflow (`.github/workflows/nightly.yml`) runs the expanded profile
  daily and on manual dispatch.
- API-level recall parity on the production adapters lives in
  `ktann-rocksdb/tests/rocksdb_recall.rs` (embedded, runs in CI) and
  `ktann-foundationdb/tests/foundationdb_recall.rs` (requires a local
  cluster; the FoundationDB CI job runs it). Both share the scenario in
  `tests/support/adapter_recall.rs` and the fixture loaders in
  `tests/support/fixtures.rs`.
