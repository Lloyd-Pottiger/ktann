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

For controlled `import-to-search-lifecycle` diagnostics,
`--maintenance-workers N` overrides the import Runtime's Structure Maintenance
worker count (including zero), and `--import-in-flight-batches N` overrides the
positive Import Session in-flight limit. After immediate search, the runner
reopens the same index with the profile's
`convergence_maintenance_workers` count so an import-only run can still reach
the stable search phases. These options require explicitly selecting that
lifecycle scenario so ordinary suites retain their fixed configuration.

The `smoke` profile uses a small deterministic synthetic dataset and exercises
warm-cache ANN, cache-disabled ANN, 95/5 search/update, hot 50/50
search/update, saturated Backend admission, and one bounded import-to-search
lifecycle. It is a functional CI signal, not a stable performance sample. The
`full` profile adds checked-in SIFTsmall
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

Report schema v1 stores exactly one tagged `measurements` payload per scenario:
`steady_state` for the existing workload cases or `lifecycle` for the
`import-to-search-lifecycle` case. The lifecycle case does not change the setup
exclusions or workload semantics of the steady-state scenarios described below.

Setup is excluded from the wall-clock, CPU, latency, throughput, metric, and
Backend-IO deltas. Setup includes dataset loading, index creation and batch
load, demand-driven topology convergence, verification, brute-force oracle
construction, operation materialization, cache warmup, and draining warmup's
Structure Maintenance backlog. A second invariant audit runs after measurement.
Peak RSS is different: the operating-system high-water mark necessarily covers
the entire isolated worker, including setup and warmup.

The timed workload reports:

- accepted-operation throughput plus attempted, accepted, admission-rejected,
  and stable error-category counts by search/write class;
- accepted end-to-end latency distributions by search/write class; rejection
  rates are derived from the reported outcome counts;
- exact recall@k against a precomputed brute-force L2 oracle for immutable ANN
  scenarios;
- every Search Budget dimension and separate `approximate_selection` and
  `exact_reranking` stage latency;
- Partition Cache hits, misses, stale misses, installs, and accounted bytes;
- blocking-resource wait/held time and Import admission wait where emitted;
- Backend-boundary logical reads, scans, returned items/bytes, transaction
  attempts, commit outcomes, and attempted mutations/bytes;
- phase-local whole write attempts, retries, logical mutations/bytes, and native
  commit wait attributed by `batch_mutate`, `split_fixup`, and `merge_fixup`;
- Fixup state-machine advance results and entries moved per committed split or
  merge drain step;
- logical write amplification as attempted mutation operations and bytes per
  successful public write, including retry attempts.

`configuration.search_budgets` exposes the same four dimension names used by
the steady-state measurement payload's `search_budgets`:
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

The `backend-admission-saturated` scenario submits fixed concurrent waves. A
wave drains before the next begins, preventing fast `LimitExceeded` results
from recursively generating nearly all remaining attempts while accepted work
is still in flight. Runtime admission limits are unchanged. Both profiles
require at least 100 accepted searches and 100 accepted writes, with an overall
rejection rate from 25% through 75%. A run outside
that declared operating region fails instead of emitting a statistically weak
baseline. The smoke scenario measures 640 operations; the full scenario
measures 2,000 operations.

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
absolute mean recall drop greater than 0.02. Admission rejection is compared
across the fixed operation mix with a five-percentage-point absolute increase
threshold. Per-class outcomes remain in the report, while the aggregate avoids
mistaking scheduler-dependent permit allocation between searches and writes
for an admission regression. Override
these materiality bounds with `--maximum-relative-regression`,
`--maximum-recall-drop`, and `--maximum-rejection-rate-increase`. All accept
fractions: for example, `0.10` means ten percentage points for the absolute
thresholds and 10% for the relative threshold. Thresholds absorb ordinary
measurement noise; changing them is benchmark policy, not a public KTANN
guarantee. Latency distributions additionally require at least a 1 ms absolute
p95 increase before a relative increase is material, avoiding large ratios on
sub-millisecond samples.

`git_revision` receives a `-dirty` suffix when tracked or untracked workspace
changes are present. Commit or otherwise preserve the exact patch before using
such a local report as a durable baseline.

## Import-to-search lifecycle

Every lifecycle worker opens a fresh isolated Backend Namespace and creates a
fresh Logical Index through `Runtime::create_index` before timing. Dataset
loading, record/request materialization, and exact-oracle construction are also
complete before the continuous case timer. The timer begins immediately before
the first `ImportSession::submit` and ends when the final warmed search returns.

The report keeps these boundaries distinct:

1. `import` begins before the first submit and ends after
   `ImportSession::finish`. It reports accepted batches and records, throughput,
   submit percentiles, gate waits, batch failures, CPU, Backend IO, peak RSS,
   operation-attributed write work, Fixup steps, drain batch sizes, and
   concurrent Structure Maintenance. `finish` remains only an accepted
   batch-outcome barrier. The case fails if any submitted record is not
   accepted, so subsequent recall always measures the complete fixed corpus.
2. `immediate_search` is the first fixed query pass after finish, before the
   runner drives convergence. It reports first-query and p50/p95/p99 latency,
   throughput, mean/minimum recall@k, truncation, Search Budget use, cache
   behavior, CPU, and Backend IO.
3. `convergence` uses bounded public `verify` and `search` calls until a
   structured topology is verified unchanged across three observations, drains
   the observed Fixup backlog, and verifies that the counts remain unchanged.
   Its active phase costs are separate, while `from_import_finish_seconds` includes
   the preceding immediate-search pass so time to verified readiness is not
   understated.
4. `cache_reset` shuts down the first Runtime, creates a new Runtime over the
   same Backend Namespace, and reopens the Index. This gives the stable cold
   pass a deterministic empty process-local Partition Cache without changing
   persistent topology.
5. `stable_cold_search` runs the fixed query set once; `stable_warm_search`
   repeats it immediately to expose warmed steady-state behavior.

`case_wall_seconds`, `case_cpu_seconds`, and `case_backend_io` cover the
continuous case. Unattributed harness overhead is derived by subtracting the
named phases from those totals, so the report does not store a second accounting
authority. Peak RSS is the operating-system process high-water mark at each
boundary and is not additive; the import value can therefore include the
pre-timed dataset and oracle resident in the isolated worker.

The comparator requires identical lifecycle bounds and inputs. In addition to
the warmed steady-state rules, it compares import throughput and submit
latency, batch failures and gate waits, finish-to-stable time, each query
stage's latency/throughput/recall/cache/budget results, and phase Backend IO.
