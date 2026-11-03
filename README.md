# KTANN

KTANN is an embeddable Rust library for approximate nearest-neighbor search over
vectors and typed metadata. It stores records and their index entries together
in a transactional key-value backend, so inserts, updates, and deletes become
searchable atomically.

Use KTANN to add filtered vector search to an application that owns its storage
and service layer. It supports in-memory, embedded, and distributed storage;
it does not provide a standalone database server or generate embeddings.

**Status:** active development, with no stable release. APIs and persistent
formats may change without backward compatibility. The core search engine,
three storage adapters, online updates, and resumable initial builds are
implemented. The scalability goals below are not claims of validated capacity.

## Capabilities

- **Filtered search.** L2, cosine, and inner-product search with typed metadata
  predicates using SQL `WHERE` semantics. Every returned hit satisfies the filter.
- **Continuous updates.** Insert, upsert, delete, and atomic batches. Background
  splits and merges keep the index searchable as its structure changes.
- **Consistent reads.** Each query uses one backend snapshot, including filtering
  and exact distance calculation over the selected candidates.
- **Initial construction.** Build an index from an immutable input snapshot,
  resume interrupted work, and publish it when loading and validation finish.
- **Bounded query work.** Search budgets limit traversal and reranking; results
  report budget exhaustion. Approximate search can miss neighbors or return fewer
  than the requested number of hits.

## Storage backends

Choose a backend for the deployment. All adapters share the logical transaction
contract and core index algorithms.

| Backend | Use | Setup |
| --- | --- | --- |
| [Memory](ktann-memory/README.md) | Ephemeral indexes and local development | No external service; data is lost on restart |
| [RocksDB](ktann-rocksdb/README.md) | Persistent embedded storage | C++ toolchain and Clang/libclang for builds |
| [FoundationDB](ktann-foundationdb/README.md) | Distributed transactional storage | FoundationDB 7.3 client library and a running cluster |

Your application owns backend initialization and lifetime. Shut down the KTANN
runtime before the backend. Physical index data is not portable between backends.

## Try it

From a checkout, using the latest stable Rust toolchain:

```sh
cargo run --example basic
```

The [complete example](examples/basic.rs) creates an in-memory index, inserts a
three-dimensional vector, searches it, and shuts down the runtime. Its main
operations are:

```rust
let runtime = Runtime::new(MemoryBackend::new(), RuntimeConfig::default())?;
let index = runtime
    .create_index("vectors", IndexConfig::new(3, Metric::L2)?)
    .await?;

index
    .insert(Record::new(
        Bytes::from_static(b"first"),
        vec![1.0, 0.0, 0.0],
        vec![],
    )?)
    .await?;

let result = index
    .search(SearchRequest::new(vec![1.0, 0.0, 0.0], 1)?)
    .await?;
```

To embed KTANN, use the `ktann` crate and a storage adapter from this checkout as
path dependencies. Start with the [Memory adapter guide](ktann-memory/README.md),
then see the [API guide](docs/design/api.md) for schemas, filters, updates, and
search options. For initial datasets, see [Bulk Build](docs/design/bulk-build.md).

## Project direction

Strong scalability, query and filtering capabilities, continuous updates, and
ease of use define the product requirements. KTANN targets both large datasets
that need distributed storage and execution, and small datasets that should run
efficiently on a single machine. These goals are not tied to particular backends.
High recall, fast construction, high query throughput, and low latency matter
across all backends.

Distributed Bulk Build is a requirement; distributed incremental writes are also
a desired capability. The current bulk scheduler distributes whole jobs across
workers with shared-filesystem requirements. It does not distribute the
construction of a single index across workers.

Strict read/write separation means queries do not mutate the index or perform
structure maintenance. Read capacity should scale independently of writes;
distributed storage alone does not establish this scalability.

Within the required capabilities and correctness guarantees, continuously pursue
better whole-system performance at comparable recall. Additional complexity must
provide a required capability or a meaningful, measured benefit; when outcomes
are comparable, choose the simpler design. Observability supports validation and
operation. [AGENTS.md](AGENTS.md) records the engineering rules.

## How it works

KTANN stands for **K-means Tree Approximate Nearest Neighbor**. It organizes vectors
in centroid-routing trees. Optional metadata fields called **Tree Keys** separate
records into trees, allowing queries to route by values such as tenant or
collection. Within a tree, centroid routing and conservative metadata summaries
narrow the search. Compact RaBitQ7 representations rank candidates before exact
reranking over the original vectors.

Exact reranking gives exact distances for selected candidates; it does not make
the overall search exact. See the [architecture overview](docs/design/overview.md)
and [search design](docs/design/search.md) for the full contracts.

## Development and benchmarks

```sh
make verify      # Format, lint, and test without FoundationDB
make bench       # Run the RocksDB smoke benchmark
make help        # List targets, including FoundationDB checks
```

Smoke benchmarks check the harness, not performance. Use the
[benchmark guide](benchmarks/README.md) for full runs, datasets, recall comparisons,
and VectorDBBench integration. Performance claims should include workload,
recall, latency, throughput, and resource costs.

The [documentation guide](docs/README.md) links the module designs and architectural
decisions. [CONTEXT.md](CONTEXT.md) defines domain terminology and system invariants.

## Influences

KTANN draws on CockroachDB's
[C-SPANN](https://github.com/cockroachdb/cockroach/tree/master/pkg/sql/vecindex/cspann)
and [SPFresh](https://doi.org/10.1145/3600006.3613166), SOSP 2023.

## License

[MIT](LICENSE).
