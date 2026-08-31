# KTANN Vector Index Design Overview

Status: Implementation-ready

This document is the system-level design for KTANN. It defines the product
boundary, authoritative invariants, module ownership, and end-to-end behavior.
Detailed contracts live in the module designs linked below. Domain terms are
defined in [`CONTEXT.md`](../../CONTEXT.md); rationale for hard-to-reverse choices
is recorded in [`docs/adr`](../adr/).

## 1. Purpose

KTANN is an asynchronous Rust library that stores source vectors and their
searchable approximate-nearest-neighbor index atomically in a transactional KV
backend. One Logical Index is a Tree-Key-sharded forest of incrementally
maintained binary K-means trees. Search uses conservative predicate pruning,
RaBitQ7 approximate ranking, and exact reranking over the original vectors.

The first stable release supports FoundationDB and RocksDB through one logical
storage contract. The adapters share Rust interfaces and logical codecs, but
their physical keyspaces are neither portable nor mutually compatible.

## 2. Goals

- Atomically commit every foreground Vector Record change with its exact leaf
  membership and affected metadata.
- Support insert, replacement upsert, delete, and atomic mutation batches while
  the index remains searchable.
- Maintain split and merge topology asynchronously without a durable work queue,
  leader, lease, or unsearchable committed state.
- Support Tree Key routing and SQL `WHERE`-style typed predicates. Every returned
  hit satisfies the predicate; conservative synopses may only avoid work.
- Bound query work, transaction work, background concurrency, queues, retries,
  and memory.
- Return original-vector exact distances for the bounded candidate set and make
  every source of budget truncation visible.
- Treat naturally encountered invariant violations as corruption rather than
  hiding, skipping, or repairing them.

## 3. Non-goals

- Exact global top-k, a guarantee to return `k` hits, or a fixed recall/latency
  SLA without benchmark evidence.
- Replicated randomized recall trees. “Forest” means disjoint Tree Key shards;
  each record belongs to exactly one tree.
- Caller-owned source encodings, callbacks into host business rows, or a
  cross-backend persistent-data interchange format.
- Online migration of schema, metric, dimension, Tree Key, quantizer, or
  persistent format.
- Redis, a production in-memory backend, durable maintenance jobs, repair on
  read, automatic repair, bulk-build generations, or staging indexes.
- Compatibility with any implementation predating the first stable format.

## 4. Authoritative invariants

1. Every Vector Record has exactly one Record Location and one corresponding
   Leaf Entry in each committed state.
2. A foreground mutation atomically changes the Vector Record, Record Location,
   Leaf Entry, exact Partition Header counts, and affected Partition Synopses.
3. Every ordinary non-root partition has exactly one incoming Child Entry. The
   only exception is a root-split target exclusively referenced by the root
   transition state before exposure.
4. Moving an entry atomically inserts the target, removes the source, updates
   exact counts and cache epochs, and, for a leaf entry, changes Record Location.
5. Partition Header count is exact. A zero count is sufficient to complete
   structural removal; completion does not rescan to prove emptiness.
6. Every committed topology state is searchable. A cold intermediate state may
   remain until a later relevant access rediscovers it.
7. A Partition Synopsis is conservative. `NoMatch` proves no entry can satisfy
   the predicate; `AllMatch` proves every entry does. Every schema field,
   including a Tree Key field, maintains the configured leaf synopsis.
8. One search uses one backend snapshot for Manifest validation, Tree Key
   enumeration, traversal, filtering, Vector Record loading, and exact rerank.
9. Persistent Logical Index IDs and Partition Keys are never reused. Gaps are
   valid.
10. Invalid persistent encoding or an invariant mismatch is `Corruption`.
    Invalid caller input or non-finite caller-derived arithmetic is
    `InvalidArgument`.

## 5. Architecture and ownership

The Rust workspace contains three production crates:

```text
ktann/                  public API, algorithms, logical storage and codecs
ktann-foundationdb/     FoundationDB transaction and physical-key adapter
ktann-rocksdb/          RocksDB OptimisticTransactionDB adapter
```

Rust Edition 2024 and MSRV 1.85 are required. CI checks MSRV and current stable;
production code uses no nightly features.

| Module design | Sole owner of |
| --- | --- |
| [Public API](api.md) | caller-visible types, validation, configuration, errors, lifecycle calls |
| [Storage](storage.md) | backend transaction contract, logical keys/codecs, typed atomic operations, manifests |
| [Search](search.md) | numeric semantics, RaBitQ7, predicates, Tree Key planning, traversal, rerank, cache correctness |
| [Maintenance](maintenance.md) | foreground routing and mutation protocol, tree shape, split/merge state machines |
| [Runtime and operations](runtime-operations.md) | admission, retry scheduling, shutdown, import, observability, verification and validation |

Backend crates own transaction mechanics, physical prefixes, backend limits,
error classification, and declared capabilities. They do not specialize index
algorithms. The core storage module owns all logical values and invariants.

## 6. End-to-end behavior

### 6.1 Create and open

Create validates the complete immutable configuration against logical and
backend limits, reserves a never-reused Logical Index ID, then atomically
installs the Index Name mapping and Active Index Manifest. Retrying the same
create after an unknown commit outcome returns the matching index; a different
configuration conflicts. Open validates the persistent format and immutable
configuration before returning an Index handle.

### 6.2 Mutation

The caller submits one mutation or an atomic batch. The engine validates the
whole request before storage work, routes each record through its Tree Key and
current searchable topology, locks the small set of authoritative state needed
for the transition, and commits record, location, leaf membership, counts,
synopses, and cache epochs together. Retryable conflicts restart the complete
attempt from a fresh transaction. `CommitOutcomeUnknown` is returned rather
than guessed unless the operation has a documented idempotent recovery check.

Committed mutations may enqueue bounded process-local maintenance. Queue loss
does not affect correctness because persistent transition states stay
searchable and later accesses can rediscover them.

### 6.3 Search

Search validates the request and opens one consistent read snapshot. It plans
Tree Key ranges, enumerates at most the Tree Key budget, traverses candidate
partitions with a global deterministic budget, and uses leaf synopses only for
safe pruning. Exact typed evaluation precedes candidate admission. RaBitQ7
selects a bounded candidate set; original vectors are loaded and reranked with
f64 exact distance. Hits sort by `(distance, RecordId)`.

Success may contain fewer than `k` hits. The response reports actual budget
usage, every exhausted budget dimension, and RaBitQ overlap truncation. It does
not claim completeness, expose a quality score, or provide continuation state;
callers may retry the same request with larger caller-controlled traversal
budgets or a wider beam. Exact-rerank sizing remains engine-owned for `k`.

### 6.4 Structure maintenance

Split and merge are persistent, incremental state machines. Each short
transaction preserves all invariants and leaves a traversal rule that covers
both source and target membership. Any process that encounters an intermediate
state may attempt a bounded fixup; no process owns the state. Completion removes
obsolete data using transactional range clear when supported, otherwise using
paged point deletion before the final topology switch.

### 6.5 Shutdown and import

Runtime shutdown stops new admission and waits for already admitted foreground
operations and commits. Those operations retain their real results, including
`CommitOutcomeUnknown`; `RuntimeClosed` never overwrites an admitted result.
Only calls admitted after shutdown begins fail with `RuntimeClosed`.

After foreground drain, Runtime invokes the backend shutdown hook before
releasing it. For RocksDB this is the completion barrier for native actor
cleanup, so successful Runtime shutdown permits immediate database reopen or
teardown. Direct adapter users consume `RocksDbBackend` with its asynchronous
shutdown for the same guarantee; transaction-handle Drop remains nonblocking.

An Import Session accepts ordinary atomic mutation batches under adaptive,
bounded concurrency and maintenance backpressure. It learns useful concurrency
from actual retryable contention rather than scanning tree topology. `submit` returns a process-local
Batch Token after admission. `finish` waits for accepted work and returns batch
results in submission order. No import state is persistent and no whole import
is atomic.

### 6.6 Drop and verify

Drop first marks the Manifest `Dropping`, making open and data operations fail
closed. It then deletes only the Logical Index's owned range and finally removes
the Manifest and Index Name mapping. Retries recover from persistent lifecycle
state.

Verify is a bounded, read-only audit of one consistent snapshot. It never
repairs data. Large FoundationDB audits run against a caller-provided offline
copy or another backend instance that can keep the required snapshot; the
common Backend API does not invent a long-lived snapshot facility.

## 7. Performance and operational constraints

- Logical budgets are checked before work that would exceed them; backend
  adapters also enforce conservative key, value, transaction, and scan budgets.
- Tree Key enumeration materializes only keys counted within the same bounded
  Tree Key budget before traversal, so tree count cannot create unbounded query
  memory.
- RocksDB transactions are admitted by a bounded semaphore and owned by
  dedicated native thread actors through cleanup. Capacity-one channels serialize
  calls without blocking async executor threads; existing transactions never
  reacquire admission, and backend shutdown awaits detached cleanup.
- Caches are process-local and byte bounded. Persistent epochs make cached
  decoded partition data safe; a particular eviction policy is an internal,
  benchmark-driven choice rather than a stable contract.
- Logs and metrics never contain raw Index Names, Tree Keys, Record IDs, field
  values, vectors, or payloads. Traces may use the explicitly allowed stable
  identifiers and hashes defined by the operations design.

## 8. Delivery and validation

Implementation proceeds in coherent increments:

1. Workspace, domain types, exact codecs, deterministic test backend, and
   backend contract suite.
2. Typed storage, FoundationDB adapter, lifecycle, mutation, and point reads.
3. Predicate evaluation, synopses, Tree Key directory, and initial routing.
4. Search, scalar-f64 RaBitQ7, exact reranking, and epoch-safe cache.
5. Split and merge state machines plus the maintenance runtime.
6. RocksDB adapter and blocking-resource admission.
7. Import, verification, observability, crash histories, and benchmarks.

Tests protect current contracts: backend transaction semantics; codec golden
vectors; model-based mutation and crash histories; synopsis and numeric
properties; deterministic search budgets; lifecycle recovery; and focused
recall, latency, contention, memory, and write-amplification benchmarks. The
detailed evidence matrix is in the module designs. No compatibility suite for
an unshipped format and no duplicated implementation-path tests are required.
