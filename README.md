# KTANN

**K-means Tree Approximate Nearest Neighbor**

KTANN is a Rust library for building an approximate nearest-neighbor index on
top of transactional key-value storage. It is designed to keep vectors and
their index membership atomically consistent while maintaining a dynamic
K-means tree asynchronously.

KTANN separates its index algorithms from the storage engine so the same logical
index can run on multiple KV backends.

> [!IMPORTANT]
> KTANN is currently in the design and early implementation stage. The system
> design is implementation-ready, and the workspace now includes the public
> domain types, canonical logical key and value codecs, backend transaction
> seam, and FoundationDB and RocksDB adapters. The higher-level vector-index
> lifecycle and algorithms are not yet available for use.

## Goals

- Provide an embeddable vector-index library rather than a standalone database
  or service.
- Store each vector, its location, and its leaf membership in one atomic
  foreground transaction.
- Support dynamic inserts, updates, and deletes without periodically rebuilding
  the whole index.
- Keep every committed split and merge state searchable while asynchronous
  maintenance progresses.
- Run the same logical index over multiple transactional KV stores. FoundationDB
  and RocksDB are the initial planned backends.
- Offer typed metadata filtering with conservative partition pruning.
- Bound search work, transaction size, concurrency, queues, retries, caches, and
  memory explicitly.
- Use approximate candidate selection followed by exact reranking over the
  original vectors.

## Design overview

Records are divided into a sharded forest by caller-declared **Tree Key**
fields. Each Tree Key selects one incremental binary K-means tree. Internal
partitions route searches by centroid distance; leaf partitions use RaBitQ7 for
compact approximate ranking and retain access to the original vector for exact
reranking.

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

Backend adapters share the same logical transaction contract, but their
physical keyspaces are backend-specific and are not intended to be portable
between storage engines.

## Documentation

- [`CONTEXT.md`](CONTEXT.md) defines the domain language and system-wide
  invariants.
- [`docs/design/overview.md`](docs/design/overview.md) describes the product
  boundary, architecture, and end-to-end behavior.
- [`docs/design/`](docs/design/) contains the detailed API, storage, search,
  maintenance, and runtime contracts.
- [`docs/adr/`](docs/adr/) records accepted architectural decisions.

These documents, rather than external implementations or papers, define
KTANN's contract.

## Development

The repository currently contains the Rust workspace and module skeleton. See
[`AGENTS.md`](AGENTS.md) for development commands, engineering constraints, and
verification guidelines.

## Influences

KTANN is informed by:

- CockroachDB's [C-SPANN vector index](https://github.com/cockroachdb/cockroach/tree/master/pkg/sql/vecindex/cspann),
  a dynamically maintained K-means tree integrated with transactional KV
  storage.
- Xu et al., [*SPFresh: Incremental In-Place Update for Billion-Scale Vector
  Search*](https://doi.org/10.1145/3600006.3613166), SOSP 2023.

KTANN adapts these ideas around its own backend-neutral transaction contract,
persistent format, filtering model, and operational invariants.
