# KTANN memory adapter

`ktann-memory` runs KTANN entirely in process memory, without files, native
libraries, or an external service. Use it for temporary or rebuildable indexes,
embedded applications, and local development. It implements the same snapshot
and atomic transaction contract as the persistent adapters.

## Quick start

Add `ktann`, `ktann-memory`, `bytes`, and Tokio (with `macros` and
`rt-multi-thread`) to your application. From this checkout, the KTANN crates
can be used as path dependencies.

```rust
use bytes::Bytes;
use ktann::api::{IndexConfig, Metric, Record, RuntimeConfig, SearchRequest};
use ktann::runtime::Runtime;
use ktann_memory::MemoryBackend;

#[tokio::main]
async fn main() -> ktann::api::Result<()> {
    let runtime = Runtime::new(MemoryBackend::new(), RuntimeConfig::default())?;
    let index = runtime.create_index("vectors", IndexConfig::new(3, Metric::L2)?).await?;
    index.insert(Record::new(
        Bytes::from_static(b"first"),
        vec![1.0, 0.0, 0.0],
        vec![],
    )?).await?;
    let result = index.search(SearchRequest::new(vec![1.0, 0.0, 0.0], 1)?).await?;
    assert_eq!(result.hits[0].id(), &Bytes::from_static(b"first"));
    runtime.shutdown().await?;
    Ok(())
}
```

## Lifetime and transactions

- `MemoryBackend::new()` and `Default` create isolated Backend Namespaces.
  `clone()` shares the same keyspace, allowing another Runtime to reopen its
  indexes while a backend handle remains alive. There is no namespace prefix.
- Commits are atomic and immediately visible to new transactions. Existing
  snapshots remain unchanged. A write transaction sees its own staged writes.
- Protected point reads and unique insertions detect concurrent changes,
  including insert/delete ABA changes on absent keys. Conflicts return
  `RetryableAbort`; ordinary reads and scans establish no conflicts.
- There is no persistence, export format, or eviction. Dropping all backend
  handles loses the keyspace; outstanding read snapshots retain their own data
  until dropped. Process restart always starts empty.
- Transactional range clear is unsupported. KTANN drops indexes through its
  existing bounded point-delete path. Shutdown needs no native cleanup.

## Resource behavior

The ordered map uses `imbl` structural sharing. Opening a snapshot shares its
root instead of copying the whole keyspace; mutations copy affected tree paths.
Keys and values use reference-counted `Bytes`. Commits apply only the final
changed keys to the latest root, preserving unrelated concurrent commits.

One mutex serializes snapshot registration and commits. In-memory operations
execute on the calling thread, with no IO or worker threads. Long-lived snapshots
retain old tree nodes and values. Keep transactions short. Dataset size and
the number of live snapshots are application-managed; there is no total-memory
quota or automatic eviction. Runtime admission limits bound Runtime operations.

The adapter enforces 10,000-byte keys, 100,000-byte values, and transaction
budgets of 10,000 mutations and 1 MiB of mutated key/value bytes (including
repeated writes). A rejected batch stages none of its mutations. Scan pages
respect item/byte limits, with the contract's single oversized first-item
exception.

Conflict history is reclaimed as writers finish and capped at 100,000 changed
key references and 8 MiB of key bytes, excluding container overhead. A commit
may transiently add one transaction before trimming. A write snapshot older
than retained history returns `RetryableAbort` if it has protected reads, even
if those keys were unchanged. Read-only snapshots do not expire.

## Optional test controls

Enable `test-support` in a test dependency to exercise the same transaction
implementation with deterministic controls:

```toml
[dev-dependencies]
ktann-memory = { path = "../ktann-memory", features = ["test-support"] }
```

`MemoryBackend::with_test_config(TestConfig)` accepts simulated hard limits,
admission budgets, scan/batch ceilings, database capacity, and range-clear
capability. `MemoryBackend::new()` keeps the production limits even when the
feature is enabled. The `test_support` module exports the configuration and
fault/history types.

`push_fault` and `set_fault_plan` resolve commits as normal, definitely aborted,
unknown-but-applied, or unknown-and-not-applied. An unknown-but-applied step
still validates conflicts and capacity before publishing. Operation counts and
bounded redacted commit history support work-amplification checks and replay.
`reopen()` simulates ephemeral or durable restart; the durable mode copies a
snapshot handle into a fresh instance and never writes to disk.

Core functional tests, failure tests, and adapter tests all use `MemoryBackend`.
There is no second in-memory transaction engine. Without `test-support`, fault
plans, control APIs, counters, diagnostics, and simulated capabilities are
compiled out. Cargo features are additive: enable this feature only for tests.

## Validation

```sh
cargo test -p ktann-memory
cargo test -p ktann --test backend_contract
cargo clippy -p ktann-memory --all-targets -- -D warnings
```

Tests reuse the shared backend contract and real-vector recall suite, with
additional coverage for concurrent writers, admission, isolation, and conflict
history reclamation. The quick-start example runs as a documentation test.
