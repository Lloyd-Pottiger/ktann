# KTANN RocksDB adapter

This crate maps KTANN's backend-neutral transactional KV interface onto
RocksDB's `OptimisticTransactionDB`. It uses explicit snapshots, point-key
optimistic conflicts, WAL-backed writes, and synchronous commits. Each live
snapshot or transaction is owned by one permit-bounded native thread actor;
all native calls and cleanup for that handle run serially on that actor.

The caller opens an `OptimisticTransactionDB` and passes it, or a shared `Arc`
containing it, to `RocksDbBackend`. Each adapter instance adds a versioned,
RocksDB-specific physical prefix containing its caller-selected Backend
Namespace; logical codecs and index algorithms remain in `ktann`.

## Physical key format

Every RocksDB key has this exact physical prefix before its opaque KTANN
logical key:

```text
00 6b 74 61 6e 6e 2d 72 6f 63 6b 73 64 62 01
<namespace-length:u8> <namespace-bytes> <logical-key>
```

The marker is deliberately RocksDB-specific. It versions this adapter's
physical format without claiming that persisted indexes are portable to
FoundationDB. Backend Namespaces are limited to 255 bytes. Length delimiting
keeps adjacent namespace values disjoint without escaping or rewriting the
logical key.

## Database configuration

The database must use a comparator whose ordering and equality exactly match
lexicographic byte ordering. A comparator with different semantics is
unsupported: it can make distinct physical namespaces compare equal or break
the ordered ranges required by KTANN scans. Prefix extractors and hash-based
memtables are supported; the adapter forces total-order seeks for every
contractual range scan.

The adapter keeps WAL enabled and sets `sync=true` for every transaction. It
uses conservative defaults of 10,000 mutations, 1 MiB of physical mutation
bytes, and 80 KiB per scan page. RocksDB v1 reports transactional range clear
as unsupported, so higher layers use bounded point deletes.

The adapter requires a Tokio runtime; current-thread runtimes and `LocalSet`
callers are supported because async tasks never execute RocksDB directly.
`RocksDbConfig::blocking_resource_limit` bounds live native transaction actors
and defaults to the host's available parallelism. Each actor has a one-command
queue and retains one permit from transaction admission through native cleanup.
Existing transactions reuse their actor for ordinary calls, commit, rollback,
and destruction, so retaining the configured maximum cannot deadlock those
transactions; only another transaction open waits asynchronously.

Cancelling before admission removes the semaphore waiter and creates no actor.
Dropping an ordinary operation may discard a native call that already started;
if the actor observes a cancelled call it abandons that transaction. This
applies to reads issued through a write transaction as well as mutations: the
write actor retires on any closed response, so a cancelled read-through-write
does not leave the transaction in an ambiguous partially-mutated state.
Dropping a commit future before its actor claims commit ownership abandons the
transaction. After the claim, commit runs to completion and the usual
unknown-outcome rule applies to the dropped caller. Handle `Drop` only closes
the bounded actor channel. Snapshot or transaction destruction therefore never
waits synchronously on an async executor thread, including from a `LocalSet`,
task cancellation, panic unwinding, or Tokio runtime shutdown.

Call `RocksDbBackend::shutdown().await` before immediately reopening or
destroying the underlying database when deterministic cleanup completion is
required by a direct adapter user. KTANN Runtime invokes the backend cleanup
hook automatically after foreground drain, so successful `Runtime::shutdown`
already includes this barrier. Native actor threads are detached from Tokio and
may outlive an ungraceful runtime drop.

## Local tests

RocksDB is built statically by `rust-rocksdb`. A local C++17 compiler, Clang
with a loadable `libclang` shared library, and the platform tools required by
the `cc` crate must be installed. No external RocksDB server or system RocksDB
installation is needed.

Run the focused adapter tests with:

```sh
cargo test -p ktann-rocksdb
```

The tests use temporary databases and cover namespace isolation, snapshot
consistency, read-your-writes, point conflicts, item/byte-bounded ordered scan
pagination, unique insertion, rollback, unsupported range clear, admission
limits, cancellation and slow-cleanup scheduling, runtime shutdown, panic
cleanup, and visibility after orderly adapter shutdown and database reopening.

Each test owns a `tempfile` directory that is removed when the test finishes
and binds a dedicated Backend Namespace, so databases are isolated from each
other and from any caller-owned database; keys and values are bounded and
small. The adapter test runs every in-process case from the unchanged shared
backend suite and adds adapter-specific checks: namespace isolation across one
database, the 80 KiB scan-page ceiling, admission-budget rejection, a real
optimistic unique-insert conflict mapping to `RetryableAbort`, one open
snapshot paginating consistently across concurrent commits, and the blocking
resource limit delaying a new transaction until a live one releases its slot.

The fault test flushes one value into an on-disk SST, truncates the file, and
requires both a point read and a range scan to surface `Corruption`, proving
the adapter's error classification against a real RocksDB failure. RocksDB
cannot deterministically stage a controlled commit outcome, so the shared suite
declares controlled fault injection unavailable for this adapter. The
deterministic backend remains the exhaustive evidence for both applied and
unapplied unknown outcomes, and the commit-error mapping table is unit-tested.

Durability uses a separate two-phase test because only a fresh process proves
that an acknowledged WAL-synced commit is recovered on open. Run the write
phase, then the verify phase from a new process against the same directory:

```sh
KTANN_ROCKSDB_DURABILITY_PHASE=write \
  KTANN_ROCKSDB_DURABILITY_PATH=/tmp/ktann-rocksdb-durability \
  cargo test -p ktann-rocksdb --test rocksdb_durability -- --ignored

KTANN_ROCKSDB_DURABILITY_PHASE=verify \
  KTANN_ROCKSDB_DURABILITY_PATH=/tmp/ktann-rocksdb-durability \
  cargo test -p ktann-rocksdb --test rocksdb_durability -- --ignored
```

The write phase clears any previous database at the path before writing two
bounded keys; the verify phase destroys the database directory when it
finishes. CI runs both phases explicitly. The in-process shared harness
therefore declares controlled fault injection and backend restart unsupported;
the fault and durability binaries provide the adapter-specific evidence instead
of silently weakening those shared cases.
