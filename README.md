# KTANN

**Vector search that stays in step with your data.**

KTANN (**K-means Tree Approximate Nearest Neighbor**) is an embeddable Rust
library for transactional vector search. It commits vectors, metadata, and
index membership together, and keeps its K-means trees searchable as they grow
and rebalance. Build semantic search, recommendations, and retrieval into your
application with the storage backend that fits your deployment.

## Why KTANN?

- **One atomic write, a consistent index.** Each insert, update, or delete
  commits the record, its location, its leaf membership, counts, and affected
  metadata together. Atomic batches use the same contract.
- **Built for changing data.** Online inserts, updates, and deletes do not
  require rebuilding the index. Background splits and merges preserve a
  searchable topology at every committed step.
- **In-memory, embedded, or distributed storage.** Use Memory for ephemeral
  indexes, RocksDB for embedded persistence, or FoundationDB for transactional
  distributed storage. All adapters share
  the same index algorithms and logical transaction contract.
- **Filters that mean what they say.** Typed metadata predicates use SQL
  `WHERE` semantics. Partition summaries prune unnecessary work; every returned
  hit passes exact filtering.
- **Fast candidate selection, precise distances.** Centroid routing and
  RaBitQ7 quantization narrow the search, then original vectors provide exact
  reranking. L2, cosine, and inner-product metrics are supported.
- **Control over work and resources.** Search budgets bound tree enumeration,
  partition visits, and exact reranking. Results expose budget truncation;
  runtime admission, caches, queues, retries, and maintenance concurrency are
  bounded too.
- **Operations are part of the design.** Adaptive bulk-import admission,
  read-only index verification, metrics, tracing, and graceful shutdown support
  the full lifecycle of an index.

The core library, all three storage adapters, search, online maintenance, import,
and verification are implemented. KTANN is pre-1.0: APIs and persistent formats
may change, and there is no stable release yet.

## How it works

```text
Your application
       |
       v
KTANN Runtime + Index
  Atomic mutations | Filtered ANN search | Background maintenance
       |
       v
Backend-neutral transaction contract
       |
       +-- Memory          ephemeral in-process storage
       +-- RocksDB         embedded storage
       +-- FoundationDB    distributed storage
```

Caller-declared **Tree Key** fields divide an index into disjoint shards, each
with an incremental binary K-means tree. This lets applications route searches
by fields such as tenant or collection. Internal partitions route by centroid
distance; leaf partitions apply exact predicates and rank compact RaBitQ7
representations before reranking over the original vectors.

Each search uses a single consistent backend snapshot. Approximate candidate
selection can miss true nearest neighbors, even when it returns `k` hits;
exact reranking guarantees the distances of selected candidates, not global
exact top-k recall.

KTANN is a library: your application owns the service interface and storage
lifecycle. Backend physical keyspaces are not portable between RocksDB and
FoundationDB.

## Get started

Use the latest stable Rust toolchain. The [Memory adapter](ktann-memory/README.md)
needs no native libraries or external services. RocksDB builds require a C++ toolchain
and Clang/libclang; FoundationDB additionally requires its 7.3 native client
library. See the
[RocksDB](ktann-rocksdb/README.md) and
[FoundationDB](ktann-foundationdb/README.md) adapter guides for integration.

From a checkout:

```sh
make build       # Core, Memory and RocksDB adapters, and benchmark tools
make test        # Tests that do not need a FoundationDB installation
make doc         # Generate API documentation in target/doc
make bench       # Run the optimized RocksDB smoke benchmark
cargo test -p ktann-memory  # Memory adapter, no native dependencies
```

For embedding, use the `ktann` crate together with a storage adapter. Construct
a `Runtime` from the adapter and a `RuntimeConfig`, create or open an `Index`,
then call `insert`, `upsert`, `delete`, `batch_mutate`, and `search` on it.
Call `Runtime::shutdown` before shutting down the backend. The
[public API guide](docs/design/api.md) documents configuration, typed records,
filters, request controls, and result semantics; the
[adapter recall test](tests/support/adapter_recall.rs) exercises this flow
against both real storage engines.

## Performance you can measure

KTANN combines compact candidate ranking, byte-bounded partition caches, and
incremental maintenance to keep search and updates practical as data changes.
Its benchmark suite measures the costs together: recall, latency distributions,
throughput, CPU, memory, backend IO, contention, and write amplification.

Reproduce measurements on your hardware:

```sh
make bench PROFILE=full
make bench PROFILE=full BENCH_ARGS="--scenario import-to-search-lifecycle"
# Requires the FoundationDB native client and a running local cluster:
make bench-fdb PROFILE=full
```

Reports go to `.benchmark-data/results/` by default. Use a distinct
`REPORT_DIR` to preserve runs for comparison. The suite also provides **Cohere
1M and SIFT1M** quality profiles and a **VectorDBBench bridge**. See the
[benchmark guide](benchmarks/README.md) for dataset setup, measurement boundaries,
recall curves, and report comparison. Smoke profiles check the harness; use
optimized full profiles on an otherwise idle host for performance analysis.

## Development

Run `make help` for all targets. `make verify` runs formatting checks, Clippy,
and tests without FoundationDB. After configuring its native client,
`make verify-all` checks all workspace packages and features;
`make test-fdb` runs the cluster-dependent integration tests. Durability tests
require separate process or server restarts as described in the adapter guides.

| Crate | Responsibility |
| --- | --- |
| `ktann` | Public API, tree algorithms, search, maintenance, logical storage, and codecs |
| `ktann-rocksdb` | RocksDB transactions, resource admission, and error mapping |
| `ktann-foundationdb` | FoundationDB transactions, keyspace, limits, and error mapping |
| `ktann-benchmarks` | ANN quality, lifecycle benchmarks, and VectorDBBench integration |

## Documentation

- [Architecture](docs/design/overview.md): boundaries, invariants, and module ownership.
- [API](docs/design/api.md): lifecycle, mutations, filters, and search requests.
- [Search](docs/design/search.md): routing, ranking, caches, and query budgets.
- [Operations](docs/design/runtime-operations.md): import, verification, telemetry, and shutdown.
- [Storage](docs/design/storage.md) and [maintenance](docs/design/maintenance.md): transaction and topology contracts.
- [Domain glossary](CONTEXT.md) and [architectural decisions](docs/adr/): terminology and rationale.
- [Benchmarks](benchmarks/README.md): reproducible quality and performance evaluation.
- [Contributor guidance](AGENTS.md): engineering rules and validation commands.

## Influences

KTANN draws on CockroachDB's
[C-SPANN vector index](https://github.com/cockroachdb/cockroach/tree/master/pkg/sql/vecindex/cspann)
and Xu et al., [*SPFresh: Incremental In-Place Update for Billion-Scale Vector
Search*](https://doi.org/10.1145/3600006.3613166), SOSP 2023. It adapts these ideas
around its own backend-neutral transaction contract, filtering model, persistent
format, and operational invariants.

## License

Licensed under the [MIT License](LICENSE).
