# KTANN RocksDB adapter

This crate maps KTANN's backend-neutral transactional KV interface onto
RocksDB's `OptimisticTransactionDB`. It uses explicit snapshots, point-key
optimistic conflicts, WAL-backed writes, and synchronous commits. Every native
call runs in a Tokio `block_in_place` section while its snapshot or transaction
holds one bounded adapter resource slot.

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

The adapter requires a Tokio multi-thread runtime and rejects calls polled from
a `LocalSet` because Tokio forbids `block_in_place` there. `RocksDbConfig` bounds
live native snapshot/transaction resource slots and defaults to the host's
available parallelism. Each transaction retains its slot through cleanup, so
native call concurrency is bounded without making synchronous Drop wait for
admission. Waiting transaction opens remain ordinary caller-owned futures, so
cancelling one removes its semaphore waiter without scheduling blocking work.
A call already inside `block_in_place`, including commit, runs to
completion; KTANN Runtime ownership retains an admitted Foreground Mutation's
outcome when its caller drops the future. Snapshot and transaction destruction
retains the same resource slot and uses a blocking section, including
cancellation and rollback cleanup. If a handle is moved into a `LocalSet`,
operations are rejected and destruction uses a scoped native thread because
Tokio forbids `block_in_place` in that context.

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
limits, and visibility after reopening the database.
