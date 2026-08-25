# KTANN performance baselines

`ktann-bench` measures ANN quality and whole-system costs through KTANN's public
`Runtime` and `Index` APIs. It produces versioned JSON intended for same-host,
same-input comparisons. These results are empirical baselines, not a v1 SLA.

## Running a suite

Build and run benchmarks with optimizations enabled:

```sh
cargo run --release -p ktann-benchmarks --bin ktann-bench -- \
  run --backend rocksdb --profile smoke --output rocksdb-smoke.json
```

`--scenario NAME` selects one scenario, and `--worker-threads N` fixes the
Tokio executor size. The emitted `reproduction_command` records the complete
parent invocation. Each scenario runs in a fresh subprocess so its metrics
recorder, Partition Cache, and peak RSS do not contain another scenario's
state.

The `smoke` profile uses a small deterministic synthetic dataset and exercises
warm-cache ANN, cache-disabled ANN, 95/5 search/update, hot 50/50
search/update, and saturated Backend admission. It is a functional CI signal,
not a stable performance sample. The `full` profile adds checked-in SIFTsmall
and Fashion-MNIST inputs plus clustered, skewed, and duplicate-heavy synthetic
distributions at representative sizes. Full runs are intended for optimized,
otherwise idle hosts.

FoundationDB requires a reachable local cluster, the client library, and
`fdbcli`. The runner queries `fdbcli --exec status json` so the connected
server version becomes part of the comparable runtime identity:

```sh
cargo run --release -p ktann-benchmarks \
  --no-default-features --features foundationdb --bin ktann-bench -- \
  run --backend foundationdb --profile smoke --output foundationdb-smoke.json
```

Set `FDB_CLUSTER_FILE` when the default cluster file is not appropriate. The
runner assigns every FoundationDB worker a process-unique namespace and clears
that namespace before exit. RocksDB's `blocking_resource_limit` scenario input
has no FoundationDB equivalent; it is recorded as configuration and only
changes RocksDB's native blocking actor bound.

## Measurement contract

Setup is excluded from the wall-clock, CPU, latency, throughput, metric, and
Backend-IO deltas. Setup includes dataset loading, index creation and batch
load, demand-driven topology convergence, verification, brute-force oracle
construction, operation materialization, cache warmup, and draining warmup's
Structure Maintenance backlog. A second invariant audit runs after measurement.
Peak RSS is different: the operating-system high-water mark necessarily covers
the entire isolated worker, including setup and warmup.

The timed workload reports:

- successful-operation throughput and end-to-end latency distributions;
- exact recall@k against a precomputed brute-force L2 oracle for immutable ANN
  scenarios;
- every Search Budget dimension and separate `approximate_selection` and
  `exact_reranking` stage latency;
- Partition Cache hits, misses, stale misses, installs, and accounted bytes;
- blocking-resource wait/held time and Import admission wait where emitted;
- Backend-boundary logical reads, scans, returned items/bytes, transaction
  attempts, commit outcomes, and attempted mutations/bytes;
- logical write amplification as attempted mutation operations and bytes per
  successful public write, including retry attempts.

`configuration.search_budgets` exposes the same four dimension names used by
`measurements.search_budgets`:
`scanned_tree_keys`, `visited_partitions`, `visited_leaf_entries`, and
`exact_rerank_candidates`. Each configuration entry distinguishes the Runtime
`runtime_default`, an optional per-request `request_override`, and the concrete
`effective_limit`. The measurement entry can therefore compare its usage and
exhausted-search count directly with the governing limit. `visited_leaf_entries`
counts derived Leaf Entries considered during filtering and approximate
selection; `exact_rerank_candidates` counts original Vector Records loaded and
exactly reranked.

The configuration also records `leaf_beam_size_override` because this
per-request traversal input affects recall and partition work even though it is
not a Search Budget dimension.

Foreground `wall_seconds`, throughput, and operation latency stop when the last
public operation completes. `maintenance_drain_seconds`, CPU, and Backend IO
continue until the pending-plus-running Fixup backlog returns to zero, so
maintenance causally triggered by measured writes is attributed to that run
without inflating foreground latency. Recall calculation happens after all
resource sampling.

Mixed update/search scenarios omit recall because the immutable pre-run oracle
would be stale after concurrent updates. Failed operations are counted by
stable error category and excluded from successful latency and throughput;
they still contribute any Backend work attempted before failure. The logical
write-amplification definition is backend-neutral and deliberately distinct
from RocksDB engine-level physical write amplification.

## Comparing results

Capture baseline and candidate suites on the same otherwise idle host, with
the same build profile, feature set, compiler and Rust flags, worker count,
Backend client/server identity and limits, scenario configuration, and dataset
checksum. Then run:

```sh
cargo run --release -p ktann-benchmarks --bin ktann-bench -- \
  compare --baseline baseline.json --candidate candidate.json \
  --output comparison.json
```

The comparator refuses different report schemas, scenario sets, inputs, or
hardware/runtime fingerprints. By default it flags more than 20% regression in
p95 latency, throughput, CPU, peak RSS, or logical write amplification, and an
absolute mean recall drop greater than 0.02. Override these materiality bounds
with `--maximum-relative-regression` and `--maximum-recall-drop`. Both accept
fractions: for example, `--maximum-relative-regression 0.10` means 10%, while
`--maximum-recall-drop 0.01` means one percentage point of absolute recall.
Thresholds absorb ordinary measurement noise; changing them is benchmark
policy, not a public KTANN guarantee.

`git_revision` receives a `-dirty` suffix when tracked or untracked workspace
changes are present. Commit or otherwise preserve the exact patch before using
such a local report as a durable baseline.
