# Runtime, Import, Observability, and Verification

Status: Implementation-ready

This module owns process-local admission and lifecycle, maintenance scheduling,
Import Sessions, observability/privacy, offline verification behavior, and the
whole-system validation matrix.

## 1. Runtime ownership

`RuntimeInner` owns backend access, configuration, cache, bounded maintenance
queue, workers, retry/backoff policy, shutdown state, and foreground in-flight
tracking. Index handles retain `Arc<RuntimeInner>` and immutable Logical Index
identity.

Admission acquires the relevant bounded permit before starting work. A
foreground in-flight guard is registered before an operation can begin commit
and is held by the owned completion path until the backend commit finishes,
even if the caller drops its future. This is required because RocksDB commits
cannot be interrupted and any backend may produce an unknown outcome.
At most the configured foreground operation limit may run and the same number
may wait for admission; further calls fail with `LimitExceeded` rather than
creating an unbounded process-local queue.

## 2. Shutdown

Shutdown is idempotent:

1. atomically stop new foreground, import, and maintenance admission;
2. cancel queued work that has not begun;
3. wait for admitted foreground operations and detached commit completions;
4. stop workers and release process-local resources.

Admitted operations return their actual result. `RuntimeClosed` applies only to
new admission and never masks success, failure, or CommitOutcomeUnknown from an
operation already admitted. Dropping the final public handle initiates the same
resource cleanup but cannot synchronously return failures; callers that need a
known outcome call shutdown explicitly.

## 3. Maintenance scheduling

Relevant mutation and search paths may offer a Fixup key to a bounded,
deduplicating process-local queue. Admission bounds per-index and global
concurrency. Queue full, duplicate admission, or worker loss is observable but
does not affect correctness.

A fixup retries a bounded number of whole state-machine steps with capped
jittered backoff, then retires. Later relevant access may enqueue it again.
There is no durable scan, queue, leader, lease, or claim that one Runtime knows
cluster-wide backlog.

Wall clock writes Unix-epoch nanosecond diagnostic timestamps; Tokio monotonic
time controls deadlines and backoff. Invalid wall time prevents a state
transition. Future persistent timestamps are not stalled.

## 4. Import Session

An Import Session is a non-cloneable process-local coordinator bound to one
Index. `submit` applies ordinary batch validation, waits for an in-flight slot
and backlog gate, admits exactly one ordinary atomic mutation operation,
and returns a Batch Token. Tokens are monotonically increasing within the
session and have no persistent or transaction identity.

Accepted batches may execute concurrently. Their atomicity, retry, error, and
maintenance behavior is identical to normal mutate. `finish` closes admission,
waits for all accepted tokens, and returns every batch result in submission
order; a failed batch does not discard other known results. The caller can
select the first failure if desired without losing outcome information.

Dropping a session cancels batches not yet admitted to commit. Started commits
remain owned by Runtime in-flight guards and finish without an Import Session
observer. Import never claims a cluster-wide maintenance barrier or an atomic
whole-import result.

`finish` does not wait for the Runtime's maintenance backlog: batch outcomes are
its complete contract, while process-local or cluster-wide topology convergence
remains demand-driven and separately observable.

## 5. Metrics, tracing, and privacy

KTANN emits through the `metrics` and `tracing` facades. Metric labels are
bounded enums only: backend, operation, outcome, partition level/state, fixup
kind, cache level/result, and budget dimension. Raw Index Name, IDs, Tree Key,
Record ID, field values, vector, and payload are forbidden labels.

Tracing may include Logical Index ID, Partition Key, and a stable Tree Key hash.
It never includes raw Index Name, Tree Key, Record ID, fields, vector, or
payload. Errors use the same redaction policy.

Required observations cover operation latency/outcome; conflicts/retries and
commit unknown; logical budget use; cache bytes/results; maintenance admission,
backlog, retries, state age, and completion; Bloom saturation; RocksDB semaphore
wait/blocking duration; and import backpressure. Names use one `ktann.*`
namespace but individual metric names and span nesting are not public API.

## 6. Verification

`Index::verify` performs one read-only audit using one Backend ReadTxn. It first
validates Active Manifest and then checks, within explicit object, issue, memory,
deadline, and cancellation bounds:

- decodability and canonical encodings;
- Tree Manifest/root reachability and exactly one incoming child reference;
- exact Header counts and legal state references;
- one Record Location and Leaf Entry per Vector Record and no dangling entries;
- Leaf Entry field/code agreement with the Vector Record;
- conservative synopsis contents recomputed from leaf entries;
- allocator high-water marks and ownership ranges.

The report is conclusive only when `complete` is true. A reached limit returns a
successful incomplete report with collected issues; cancellation, deadline, or
snapshot failure returns an error and no cross-snapshot conclusion. Issues use
coarse stable kinds and safe identifiers. Verification never writes, repairs,
spills state into the index, continues from a token, or samples.

FoundationDB's ordinary snapshot lifetime may be too short for a large audit.
Such an audit runs against a caller-provided offline copy or separately opened
backend instance suitable for the workload. The common API does not expose a
fictional renewable/native long-lived snapshot.

## 7. Whole-system validation

| Contract | Evidence |
| --- | --- |
| Backend semantics | shared adapter contract suite plus backend durability/fault tests |
| Persistent bytes | golden vectors, malformed corpus, ordering and deterministic cross-process tests |
| Exact membership | model-based transactional histories and crash injection |
| Searchability | traversal tests at every split/merge state and queue-loss histories |
| Predicate safety | SQL truth oracle and synopsis property tests |
| Numeric safety | RaBitQ signed-code properties, conservative interval proof cases, exact rerank oracle |
| Bounded resources | boundary tests for every logical/backend budget and queue/permit cap |
| Lifecycle | create/drop unknown-outcome histories, shutdown/future-drop races, redaction audits |
| ANN behavior | reproducible recall/latency/contention/memory/write-amplification benchmarks |

CI uses small deterministic seed sets and focused integration services. Nightly
runs expand seeds, crash histories, and benchmarks. Failures print replayable
seeds. Tests do not freeze internal cache eviction, task layout, or other
benchmark-tunable implementation details.

## 8. Operational limitations

- Demand-driven maintenance offers no time-bound cluster-wide convergence.
- RocksDB handle Drop starts nonblocking actor cleanup. Runtime shutdown invokes
  the backend cleanup hook after foreground drain and waits before releasing the
  adapter, so successful shutdown permits native database reopen or teardown.
  Direct adapter users call its consuming asynchronous shutdown. Dedicated
  native actors may outlive an ungraceful Tokio runtime drop.
- Verify has no repair mode and may require an offline copy for large
  FoundationDB indexes.
- Search quality depends on explicit budgets and data distribution; v1 publishes
  measured baselines rather than an unsupported SLA.
- Rollback means continuing to use an older separate Logical Index until a new
  one is validated and traffic is switched externally. The v1 format is never
  mutated backward in place.
