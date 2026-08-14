# Repository Guidelines

## Project Sources of Truth

Read `README.md` for the project status, goals, and architecture overview. The
current Rust code is a workspace and module skeleton; do not infer unimplemented
API or behavior from empty modules.

- `CONTEXT.md`: canonical domain language and system-wide invariants.
- `docs/design/overview.md`: product boundary, authoritative invariants, target
  architecture, end-to-end behavior, and implementation order.
- `docs/design/`: detailed contracts for the public API, storage, search,
  maintenance, and runtime/operations modules.
- `docs/adr/`: accepted architectural decisions and their rationale. Add an ADR
  only for a new hard-to-reverse decision; do not rewrite an accepted decision
  silently in code.
- `README.md`: project purpose, status, high-level architecture, and influences.

Any local `refwiki/` material is background reading, not an authoritative KTANN
contract. Do not edit or depend on it unless the task explicitly requires it.

## Commands

- Build: `cargo build --workspace`
- Test everything currently implemented: `cargo test --workspace`
- Run a focused test: `cargo test -p <crate> <test_name>`
- Check without producing binaries: `cargo check --workspace --all-targets`
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Check formatting: `cargo fmt --all -- --check`
- Apply formatting: `cargo fmt --all`

Run focused checks while iterating, then the relevant workspace-wide checks
before completion. Do not run multiple Cargo commands concurrently because they
contend on Cargo and target-directory locks. The target toolchain is Rust
Edition 2024 with MSRV 1.85 and current stable CI; production code must not
require nightly features.

## Workflow Principles

- Before a non-trivial change, read `CONTEXT.md`, the relevant sections of
  `docs/design/overview.md`, the owning module design, and every directly
  relevant ADR. Trace callers, persistent state, backend behavior, and runtime
  effects before editing.
- Use the domain terms from `CONTEXT.md` exactly. Do not introduce synonyms such
  as “table” for Logical Index or “partition key” for Tree Key.
- Implement in the dependency order in the overview unless the task establishes
  a smaller self-contained vertical slice. Do not add placeholder abstractions
  for later stages.
- For substantial changes, define the private module layout before
  implementation; do not accumulate multiple responsibilities in one file.
- Keep each responsibility at its documented owner. In particular, logical
  codecs and atomic index operations belong in core storage; backend-specific
  limits and error classification belong in the adapters; lifecycle and
  admission behavior belong in runtime/operations.
- Treat a discrepancy among code, design, and ADRs as a decision to resolve, not
  permission to choose whichever is easiest. Preserve shipped behavior, but
  update the relevant design and add or supersede an ADR when intentionally
  changing an architectural contract.
- Update `CONTEXT.md` when domain language or relationships change, and update
  the relevant design/ADR in the same change when its contract or rationale
  changes.
- Keep commits and diffs scoped to one coherent outcome. Do not mix formatting,
  dependency churn, or unrelated cleanup into a behavioral change.

## Correctness and Storage Rules

- Preserve the exact-membership invariant: every committed Vector Record has
  exactly one Record Location and one corresponding Leaf Entry.
- A Foreground Mutation must atomically update the record, location, leaf
  membership, exact counts, and affected synopses. Never split this contract
  across best-effort writes or asynchronous repair.
- Every committed split or merge state must remain searchable. Structure
  Maintenance may be delayed or lost from process-local queues; correctness
  cannot depend on a durable worker, lease, or coordinator.
- Persistent Logical Index IDs and Partition Keys are never reused. Gaps are
  valid. Persistent format changes must be explicit and versioned as one whole.
- Use canonical, deterministic codecs. Reject malformed and noncanonical bytes;
  do not silently normalize persistent data.
- Fail closed: invalid persistent encoding or invariant mismatches are
  `Corruption`. Invalid caller input and non-finite caller-derived arithmetic are
  `InvalidArgument`. Do not skip, repair, or hide corruption on hot paths.
- Preserve transaction semantics across the deterministic test backend,
  FoundationDB, and RocksDB. Do not weaken the shared contract to accommodate
  an adapter; expose real backend capability differences explicitly.
- Unknown commit outcomes must follow the documented idempotency/recovery
  protocol. Never report success, retry a partial mutation, or allocate a new
  identity based on an uncertain outcome.

## Search, Concurrency, and Performance

- One search uses one consistent backend snapshot for manifest validation, Tree
  Key enumeration, traversal, filtering, record loads, and exact reranking.
- Filter predicates are exact. Partition Synopses are conservative pruning aids:
  `NoMatch` must prove impossibility and `AllMatch` must prove every entry
  matches.
- Search is bounded and deterministic. Preserve stable traversal/tie-breaking
  order, account for every budget dimension, and expose truncation rather than
  claiming exact global top-k or guaranteed `k` results.
- Keep all queues, retries, scans, transactions, caches, concurrency, and memory
  bounded. Respect backend transaction and blocking-resource limits.
- Avoid per-vector allocation, repeated decoding/conversion, unnecessary copies,
  unbounded fan-out, coarse locks, and blocking work on async executor threads.
  Support material performance claims with reproducible benchmarks; do not turn
  an unmeasured target into an API guarantee.
- Cache entries are only hints. Reuse decoded partition data only after
  validating its epoch and kind in the search snapshot; never cache corruption.

## Rust Style

- Prefer small, explicit interfaces and typed domain operations over generic KV
  access, boolean flags, or stringly typed state.
- Make illegal states hard to represent, but do not mirror persistent state with
  redundant in-memory authorities.
- Use checked arithmetic and explicit conversions for IDs, counts, sizes, and
  budget accounting. Handle floating-point edge cases according to the numeric
  contract.
- Return structured errors with useful context while keeping vectors, payloads,
  filter values, and raw Tree Keys out of logs and error messages.
- Avoid `unwrap`, `expect`, and `panic!` in production paths unless an invariant
  is statically guaranteed and documented. Do not use `unsafe` without a narrow,
  reviewed justification and dedicated tests.
- Keep public APIs documented and backend-neutral. Use `rustfmt` defaults and
  keep Clippy clean under the repository command above.

## Testing

- Test externally observable contracts, not the current task layout or internal
  implementation shape. Prefer deterministic tests with replayable seeds.
- Add focused regression coverage for each changed guarantee. Use the evidence
  matrixes in the module designs to choose the correct layer rather than adding
  duplicate tests at every layer.
- Run the shared backend contract suite unchanged against the deterministic test
  backend and each production adapter. Cover conflicts, snapshot consistency,
  read-your-writes, pagination/limits, rollback, commit outcomes, durability,
  and declared capabilities.
- Protect persistent formats with golden bytes, ordering properties,
  malformed/noncanonical corpora, and cross-process deterministic vectors.
- Use model/history tests and fault injection for exact membership, retries,
  unknown outcomes, crashes, and every committed topology transition.
- Compare predicate evaluation and synopsis pruning with a simple SQL
  three-valued-logic oracle. Compare exact reranking with a brute-force numeric
  oracle.
- Test every resource boundary and truncation reason. Benchmarks should report
  recall, latency, contention, memory, and write amplification without freezing
  benchmark-tunable internals such as cache eviction or task layout.
