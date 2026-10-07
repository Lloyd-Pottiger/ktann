# Runtime, Observability, and Verification

This module owns process-local admission and lifecycle, maintenance scheduling,
online loading, observability/privacy, offline verification behavior, and the
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

1. atomically stop new foreground and maintenance admission;
2. cancel queued work that has not begun;
3. wait for admitted foreground operations and detached commit completions;
4. stop workers and release process-local resources.

Admitted operations return their actual result. `RuntimeClosed` applies only to
new admission and never masks success, failure, or CommitOutcomeUnknown from an
operation already admitted. Dropping the final public handle initiates the same
resource cleanup but cannot synchronously return failures; callers that need a
known outcome call shutdown explicitly.

## 3. Maintenance scheduling

Relevant mutation and search paths may offer an actionable Fixup key to a
bounded, deduplicating process-local queue. Eligibility is derived from the
committed Header: threshold-crossing Ready partitions and durable split/merge
source states are actionable, while healthy Ready partitions and
ReceivingSplit targets are not. One committed mutation batch coalesces each
partition before offering it. Admission bounds per-index and global
concurrency. Queue full, duplicate admission, or worker loss is observable but
does not affect correctness.

A fixup reads one Header/State pair and dispatches directly to the owning split
or merge state machine, then runs a bounded number of whole state-machine steps
with capped jittered backoff. An execution that consumes its step budget through
successful progress returns the same admission to the queue tail; this keeps
each execution bounded and fair without requiring another foreground access.
Errors and cancellation retire the admission. Completing a split replaces the
source's queue slot with best-effort offers for both newly Ready targets and the
updated parent in one process-local queue transition, then publishes the final
backlog; this catches immediately actionable follow-on work without exposing a
false quiescent interval or making the split transaction or one worker
execution unbounded. Normal queue capacity, deduplication, and loss rules still
apply. Later relevant access may enqueue dropped or retired work again.
There is no durable scan, queue, leader, lease, or claim that one Runtime knows
cluster-wide backlog.

Runtime construction starts the configured maintenance workers immediately.
Zero workers disables background scheduling: that Runtime's offers are
dropped, and topology changes advance only when driven outside it, which
stays correct because every committed intermediate state remains searchable.

The Runtime supplies unsigned 64-bit Unix-epoch millisecond timestamps for
partition state transitions. A wall-clock sample before the epoch or outside
the `u64` millisecond range is represented by zero, denoting an unavailable
timestamp. `ktann.fixup.state_age` records a sample when both timestamps are
nonzero and the state started at or before the current time. The nonnegative
millisecond difference is reported in seconds. Tokio monotonic time controls deadlines and retry backoff.

`stalled_timeout` is the minimum age of a persisted `Splitting`,
`DrainingSplit`, or `Merging` state before an independently rediscovered Fixup
may assist it. Its default is
`max(1 ms, 1 s * max_partition_entries / 128)`; overrides must be positive.
The worker checks `now - started >= timeout` in its existing authority read,
after process-local queue deduplication. Ready threshold crossings start
immediately. A progressing worker, including one yielded to the queue tail,
continues without an age delay. An unavailable timestamp permits recovery;
a known future timestamp defers recovery until the age threshold is met.
The timestamp measures time in the state, rather than time since the last
progress: this is an assistance threshold, not task expiry or a lease.
Later relevant access rediscovers deferred work; no timer schedules it.


## 4. Online loading and refinement

Callers load records with ordinary `Index::batch_mutate` operations and bound
concurrency explicitly. Repository consumers use sequential submission. Every
batch retains ordinary admission, atomicity, bounded retry, cancellation, and
unknown-commit behavior. There is no session-level scheduling, token, completion
barrier, or Fixup Backlog gate; see [ADR 0025](../adr/0025-caller-owned-online-batch-submission.md).

Offline `Index::refine` requires caller-exclusive access and settled Ready topology
as specified in [the API contract](api.md). Admission rejects local queued or
running fixups. Numerical planning runs off the async executor, remains
cooperatively cancellable, and retains foreground admission until CPU work ends.
Apply uses ordinary bounded write attempts; cancellation or an unknown commit
outcome can leave a valid, partially refined index.

## 5. Metrics, tracing, and privacy

KTANN emits through the `metrics` and `tracing` facades. Metric labels use
only fixed categories: backend, operation, outcome, partition level/state, fixup
kind, cache level/result, budget dimension, search/mutation stage, and
verification issue kind. Raw Index Name, IDs, Tree Key, Record ID, field
values, vector, and payload are forbidden labels.

Tracing may include Logical Index ID, Partition Key, and a stable Tree Key hash.
It never includes raw Index Name, Tree Key, Record ID, fields, vector, or
payload. Errors use the same redaction policy.

Required observations cover operation latency/outcome and foreground admission
rejection; whole write attempts, exact logical mutation work, commit wait,
conflicts/retries, and commit unknown; logical budget use; cache bytes/results;
maintenance admission, backlog, state-machine steps, drain batch sizes, retries,
state age, and completion; Bloom saturation; RocksDB semaphore wait/blocking
duration. Names use one `ktann.*`
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

The complete report also exposes audited Tree Manifest count, Partition Header
count, maximum tree level, Partition Header and exact entry counts by level,
maximum entries by level, persistent partition-state counts, and the number of
partitions whose committed Header can advance Structure Maintenance. These
facts come from the same snapshot as the invariant audit and make topology
quiescence independently auditable without treating a temporarily unchanged
partition count as convergence or exposing raw Tree Keys or Partition Keys.

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


Bulk Build reservation, worker, load, publication, status, abort and cleanup use
foreground operation control and bounded retry policies. Native construction and
file IO run on blocking tasks with shared admission retained until actual exit,
even when the awaiting future is cancelled. Running native stages are not
preempted; subsequent admissions and checkpoints observe cancellation.

Worker epochs fence accepted outputs; load epochs fence atomic KV checkpoints.
Publication seals writers, resumes bounded exact validation, then atomically
activates the original identity. Terminal failures persist; transient failures
and cancellation remain resumable. Root advisory locks order native IO against
reclamation. Busy cleanup is deferred; namespace cleanup pages discover orphaned
workspaces after ordinary index deletion. See [Bulk Build](bulk-build.md).


Automatic Bulk Build scheduling uses `schedule_bulk_build` for queue discovery,
lease coordination and scheduled attempts. Its bounded JoinSet admits at most
`max_jobs` per scheduler, additionally limited by shared foreground admission.
Lease renewal runs inside the admitted attempt, without acquiring another
foreground permit. Dropping the scheduler cancels child tasks; shutdown stops
discovery and cancels active attempts. Native IO may outlive cancellation while
retaining admission and file locks. Expired queue owners are replaceable by any
participating Runtime; each build mutation transaction verifies the owner token.
A failed job is durably inspectable and does not stop unrelated jobs. Invalid
coordination state is surfaced rather than retried indefinitely.
