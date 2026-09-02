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

The large profile accepts `--write-beam-size N` for import diagnostics. The
write beam is applied globally at each tree level, like the search beam; the
final foreground mutation still assigns each record to exactly one leaf. The
default is eight, so the option is explicit when measuring another import beam
and its quality effect.

For controlled `import-to-search-lifecycle` diagnostics,
`--maintenance-workers N` overrides the import Runtime's Structure Maintenance
worker count (including zero), `--import-max-in-flight-batches N` overrides the
positive Import Session concurrency ceiling, `--import-batch-size N` overrides the
positive records per atomic batch, and `--import-backlog-watermark N` overrides
the positive process-local Fixup backlog gate. After immediate search, the runner
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

The separate `large` profile is an optimized scheduled/manual quality run and
never runs in smoke CI. It loads the fixed external inputs described in
[`datasets/README.md`](datasets/README.md), creates one converged index per
dataset, and sweeps leaf beam `1, 4, 8, 16, 32` while holding the four Search
Budgets, k-derived exact-rerank policy, `k`, Runtime limits, Index configuration,
concurrency, dataset, and Backend fixed. The curves use Cohere 1M with cosine
and SIFT1M with L2, each with 1,000 held-out queries and supplied ground truth.

```sh
cargo run --release -p ktann-benchmarks --bin ktann-bench -- \
  run --backend rocksdb --profile large --worker-threads 8 \
  --output rocksdb-large.json
```

Each large worker verifies at least three searchable topology levels and
requires both recall and visited-Leaf-Entry work to move across its curve. A
saturated or structurally shallow run fails instead of publishing a
non-discriminating artifact.

`.github/workflows/large-ann-quality.yml` runs weekly and on manual dispatch
on a dedicated self-hosted runner labeled `ktann-benchmark`. Its concurrency
group serializes runs so the host is otherwise idle; it retains the validated
dataset cache and uploads the schema-versioned JSON artifact for 90 days.

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

Each report stores exactly one tagged `measurements` payload per scenario and
records the adapter's physical key-prefix charge alongside its mutation-count
and mutation-byte ceilings:
`steady_state` for the existing workload cases, `lifecycle` for the
`import-to-search-lifecycle` case, or `quality_sweep` for an ordered large ANN
curve. The lifecycle case does not change the setup
exclusions or workload semantics of the steady-state scenarios described below.

Setup is excluded from the wall-clock, CPU, latency, throughput, metric, and
Backend-IO deltas. Setup includes dataset loading, index creation and batch
load, demand-driven topology convergence, verification, brute-force oracle
construction for ordinary scenarios, supplied ground-truth validation for
large scenarios, operation materialization, cache warmup, and draining
warmup's Structure Maintenance backlog. A second invariant audit runs after
measurement.
Peak RSS is different: the operating-system high-water mark necessarily covers
the entire isolated worker, including setup and warmup.

The timed workload reports:

- accepted-operation throughput plus attempted, accepted, admission-rejected,
  and stable error-category counts by search/write class;
- accepted end-to-end latency distributions by search/write class; rejection
  rates are derived from the reported outcome counts;
- exact recall@k against metric-specific brute-force truth for ordinary
  immutable ANN scenarios; large quality scenarios use the dataset's supplied
  exact-neighbor truth (`cosine` for Cohere and `L2` for SIFT);
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
`effective_limit`; exact reranking is engine-sized, so that dimension's request
override is always absent. The measurement entry can therefore compare its
usage and exhausted-search count directly with the governing limit. `visited_leaf_entries`
counts derived Leaf Entries considered during filtering and approximate
selection; `exact_rerank_candidates` counts original Vector Records loaded and
exactly reranked.

The configuration also records `leaf_beam_size_override` because this
per-request traversal input affects recall and partition work even though it is
not a Search Budget dimension.

Large reports record the ordered `leaf_beam_sweep` once and emit one point per
beam. Every point contains mean/min recall@k, search-latency p50/p95/p99,
throughput, CPU, process peak RSS, all Search Budget usage and exhaustion,
approximate/rerank stage latency, Partition Cache behavior, and Backend IO. The
topology records Tree count, true Partition Header count, maximum level, and
Partition counts by level from the same successful invariant audit.

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

For `quality_sweep` reports, comparison pairs identical beam points and applies
the existing recall, latency, throughput, CPU, cache, Search Budget, and
Backend-IO policies to every point. Per-point RSS is unavailable because the
operating-system high-water mark cannot distinguish multiple beam points in one
worker. Compare only artifacts from the same otherwise idle host and fixed
dataset cache; beam values and other tuning inputs are experiment coordinates,
not a production SLA.

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

### Actionable maintenance validation

An otherwise-idle Apple M1 Pro run on 2026-08-28 compared revision `da57d81`
with actionable, batch-coalesced maintenance discovery. Both runs used the
`full` `import-to-search-lifecycle` scenario: SIFTsmall, 10,000 records,
200 batches, one in-flight batch, and two maintenance workers.

| Backend | Implementation | Fixup admissions | Fixup executions | Import read transactions | Import wall seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| RocksDB | `da57d81` | 10,000 | 5,145 | 14,115 | 3.147 |
| RocksDB | actionable discovery | 109 | 109 | 4,041 | 3.515 |
| FoundationDB | `da57d81` | 10,000 | 5,145 | 14,113 | 27.205 |
| FoundationDB | actionable discovery | 109 | 109 | 4,041 | 27.721 |

The current implementation removed all 5,036 merge-idle and 5,038/5,036
split-idle steps reported during RocksDB/FoundationDB import. Both sides still
reported 2,602 successful commits, 109 split begin/completion steps, 1,857
split drain steps, complete convergence, recall@10 of 1.0 in the immediate,
stable-cold, and stable-warm passes, and no Search Budget truncation. This
single paired run proves the work reduction, not a latency improvement; wall
time and retry variation require repeated sampling before drawing a latency or
contention conclusion.

The RocksDB reports were produced with:

```sh
cargo run --release -p ktann-benchmarks --bin ktann-bench -- \
  run --backend rocksdb --profile full \
  --scenario import-to-search-lifecycle --output REPORT.json
```

The FoundationDB reports used the documented local environment and:

```sh
cargo run --release -p ktann-benchmarks \
  --no-default-features --features foundationdb --bin ktann-bench -- \
  run --backend foundationdb --profile full \
  --scenario import-to-search-lifecycle --output REPORT.json
```

### Adaptive leaf-drain validation

Three independent same-host runs on 2026-08-28 compared revision `ccb4090`
with the adaptive leaf relocation batch. Both sides used the `full`
`import-to-search-lifecycle` scenario, SIFTsmall, 10,000 records, 200 batches,
one in-flight batch, and two maintenance workers. Each timing is the arithmetic
mean with the sample coefficient of variation in parentheses.

| Backend | Measurement | `ccb4090` | Adaptive drain | Change |
| --- | --- | ---: | ---: | ---: |
| RocksDB | Case wall seconds | 6.311 (3.12%) | 4.880 (0.27%) | -22.7% |
| RocksDB | Import wall seconds | 3.997 (4.78%) | 2.781 (1.00%) | -30.4% |
| RocksDB | Case CPU seconds | 8.454 (1.44%) | 6.439 (0.84%) | -23.8% |
| RocksDB | Import retryable commits | 310.7 (0.74%) | 250.0 (1.74%) | -19.5% |
| RocksDB | Import mutation operations | 147,496 (0.21%) | 131,759 (0.60%) | -10.7% |
| FoundationDB | Case wall seconds | 34.522 (0.54%) | 23.405 (0.44%) | -32.2% |
| FoundationDB | Import wall seconds | 29.035 (0.68%) | 18.467 (0.50%) | -36.4% |
| FoundationDB | Case CPU seconds | 12.956 (0.44%) | 8.887 (0.75%) | -31.4% |
| FoundationDB | Import retryable commits | 464.3 (1.11%) | 293.3 (1.57%) | -36.8% |
| FoundationDB | Import mutation operations | 173,590 (0.53%) | 134,572 (0.30%) | -22.5% |

The workload's 128-entry partition limit selects a 32-entry contention cap.
On both adapters, successful import commits fell from 2,602 to 1,290, read
transactions from 4,041 to 1,417, and split drain steps from 1,857 to 545.
Every run accepted all 10,000 records, converged completely, reported
recall@10 of 1.0 in immediate, stable-cold, and stable-warm search, and had no
Search Budget truncation. The three-run result therefore validates lower
whole-system work and latency without trading away the scenario's correctness
or recall contract.

### Adaptive import admission validation

Adaptive import admission uses observed write contention rather than raw
partition count: an atomic batch may update many leaves, skew can keep one leaf
hot in a large tree, and concurrent Structure Maintenance can conflict while
the process-local queue is nearly empty. `ImportSession` treats its configured
maximum in-flight value as a ceiling. It starts at one, cautiously probes higher
concurrency after saturated clean completion windows, and contracts before
retrying a contended batch. The default backlog watermark is two, allowing one
pending or running Fixup to coexist with import before new admission pauses.
Submitted batches remain indivisible atomic operations.

The complete batch-size, concurrency-ceiling, maintenance-worker, and backlog
watermark matrix, including three-run variance, logical Backend IO, Fixup work,
convergence, and rejected configurations, is recorded in
[`import-admission-calibration.md`](import-admission-calibration.md).

Three interleaved same-host full SIFTsmall runs on 2026-08-29 compared revision
`ca5b00b` with fixed concurrency one and backlog watermark 512 against adaptive
admission with a ceiling of four and backlog watermark two. Both sides used
50-record batches and two maintenance workers. Every run accepted 10,000 of
10,000 records, converged to 442 partitions, shut down cleanly, and reported
mean and minimum recall@10 of 1.0 in immediate, stable-cold, and stable-warm
search without Search Budget exhaustion. Reproduce the adaptive side with the
RocksDB and FoundationDB commands above plus these arguments, using a distinct
output path for each run:

```text
--import-max-in-flight-batches 4 --import-batch-size 50 \
  --import-backlog-watermark 2 --output REPORT.json
```

| Backend | Measurement | Fixed mean | Adaptive mean | Change |
| --- | --- | ---: | ---: | ---: |
| RocksDB | Case wall seconds | 4.900 | 4.470 | -8.8% |
| RocksDB | Import wall seconds | 2.773 | 2.418 | -12.8% |
| RocksDB | Import CPU seconds | 4.677 | 3.798 | -18.8% |
| RocksDB | Retryable commits | 217.3 | 165.7 | -23.8% |
| RocksDB | Mutation operations | 126,443 | 117,060 | -7.4% |
| RocksDB | p95 submit latency ms | 34.8 | 28.2 | -18.8% |
| FoundationDB | Case wall seconds | 24.780 | 24.333 | -1.8% |
| FoundationDB | Import wall seconds | 19.584 | 19.317 | -1.4% |
| FoundationDB | Import CPU seconds | 7.078 | 6.510 | -8.0% |
| FoundationDB | Retryable commits | 289.3 | 231.0 | -20.2% |
| FoundationDB | Mutation operations | 134,606 | 125,245 | -7.0% |
| FoundationDB | p95 submit latency ms | 210.1 | 205.8 | -2.0% |

The single-Tree-Key corpus supported one probe of concurrency two, observed
contention, and returned to one. Adaptive admission therefore reduced CPU,
retryable commits, and mutation work without changing recall or forcing unsafe
concurrency. Wall time and p95 submit latency remained close to the fixed
baseline rather than improving uniformly. Callers choose batch size from their
transaction, latency, and atomicity requirements rather than from tree size.
