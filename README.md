# KTANN

**K-means Tree Approximate Nearest Neighbor**

KTANN is a Rust library for building an approximate nearest-neighbor index on
top of transactional key-value storage. It keeps vectors and their index
membership atomically consistent while maintaining a dynamic K-means tree
asynchronously.

KTANN separates its index algorithms from the storage engine so the same
logical index can run on multiple KV backends.

## Status

KTANN is in active implementation; there is no stable release and the
persistent format may still change. The Logical Index lifecycle, point and
batch reads, atomic foreground mutations, bounded approximate search with
exact filtering and reranking, and bounded import sessions are implemented
over both the FoundationDB and RocksDB adapters. The split and merge state
machines are implemented, with bounded demand-driven maintenance scheduling
in the Runtime.

## Features

- Embeddable library, not a standalone database or service.
- Each vector, its location, and its leaf membership commit in one atomic
  foreground transaction.
- Dynamic inserts, updates, and deletes without rebuilding the index; every
  committed split and merge state remains searchable.
- The same logical index runs over multiple transactional KV stores;
  FoundationDB and RocksDB are the initial backends.
- Typed metadata filtering with conservative partition pruning.
- Bounded, deterministic approximate candidate selection followed by exact
  reranking over the original vectors.

## Design overview

Records are divided into a sharded forest by caller-declared **Tree Key**
fields. Each Tree Key selects one incremental binary K-means tree. Internal
partitions route searches by centroid distance; leaf partitions use RaBitQ7
for compact approximate ranking and retain access to the original vector for
exact reranking.

```text
Application
    |
    v
KTANN public API
    |
    +-- mutation and lifecycle
    +-- K-means tree search and maintenance
    +-- logical storage, codecs, and invariants
    |
    v
Transactional KV contract
    |
    +-- FoundationDB adapter
    +-- RocksDB adapter
    +-- additional adapters that satisfy the contract
```

Foreground mutations update authoritative record and leaf state atomically.
Partition splits and merges run as bounded, retryable state transitions. No
durable maintenance queue, leader, or lease is required for correctness: a
lost maintenance task leaves a searchable intermediate state that a later
relevant access can resume.

Structure Maintenance is demand-driven: a cold partition can remain in a
searchable intermediate topology state indefinitely, and there is no
time-bound on index-wide convergence. An index that is bulk-loaded and then
rarely accessed does not become compact on its own; the next relevant access
resumes maintenance from the durable intermediate state.

A search uses one consistent backend snapshot for index validation, Tree Key
selection, tree traversal, filtering, record loading, and exact reranking. It
returns valid, exactly reranked hits from a bounded candidate set; it does not
claim exact global top-k or guarantee that every request returns `k` hits.

## Crate layout

| Crate | Responsibility |
| --- | --- |
| `ktann` | Public API, K-means tree algorithms, search, maintenance, logical storage, and persistent codecs |
| `ktann-foundationdb` | FoundationDB transactions, physical keyspace, limits, and error mapping |
| `ktann-rocksdb` | RocksDB `OptimisticTransactionDB` integration, admission, limits, and error mapping |

Backend adapters share one logical transaction contract, but their physical
keyspaces are backend-specific and are not portable between storage engines.

## Documentation

- [`CONTEXT.md`](CONTEXT.md) defines the domain language and system-wide
  invariants.
- [`docs/design/overview.md`](docs/design/overview.md) describes the product
  boundary, architecture, and end-to-end behavior.
- [`docs/design/`](docs/design/) contains the detailed API, storage, search,
  maintenance, and runtime contracts.
- [`docs/adr/`](docs/adr/) records accepted architectural decisions.
- [`AGENTS.md`](AGENTS.md) lists development commands and engineering rules.

These documents, rather than external implementations or papers, define
KTANN's contract.

## Influences

KTANN is informed by:

- CockroachDB's [C-SPANN vector index](https://github.com/cockroachdb/cockroach/tree/master/pkg/sql/vecindex/cspann),
  a dynamically maintained K-means tree integrated with transactional KV
  storage.
- Xu et al., [*SPFresh: Incremental In-Place Update for Billion-Scale Vector
  Search*](https://doi.org/10.1145/3600006.3613166), SOSP 2023.

KTANN adapts these ideas around its own backend-neutral transaction contract,
persistent format, filtering model, and operational invariants.
