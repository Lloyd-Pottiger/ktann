# ADR 0028: Renewable automatic Bulk Build scheduling

Status: Accepted; supersedes the automatic-scheduling deferral in ADR 0027.

## Context

Explicit worker invocation already provides resumable preparation, fenced load,
exact validation and publication. Operators also need processes to discover
pending jobs and recover automatically when a worker disappears. A local task
registry cannot establish distributed ownership, and lease expiry alone cannot
prevent a paused old worker from committing later.

## Decision

`BulkBuildJob::schedule` durably queues an identity-bound job with immutable
worker options. `Runtime::run_bulk_scheduler` runs a bounded discovery loop until
cancelled or Runtime shutdown. Run it on each participating process sharing the
same Backend Namespace and durable source/workspace mounts.

Store the queue at namespace key `[0,3] || LogicalIndexId`. Each record carries
the original name, worker options, random claim token, and UTC expiry millis.
Claim transactionally when eligible, resolving unknown claims only against the
same attempted token. Renew every one-third lease duration. Heartbeats share the
admitted build operation, so one foreground permit is sufficient. Polling,
parallel jobs and lease duration have explicit per-process bounds.

Every preparation acceptance, load claim/chunk, proof page and publication reads
the queue token with update protection. A new claim conflicts with stale writes;
manual workers reject queued jobs. Expiry permits takeover but does not itself
revoke authority: a committed token change does. Clock skew can cause redundant
work or delayed takeover, but cannot authorize both owners to commit. Use
reasonably synchronized clocks for predictable recovery latency.

The owner runs preparation/load or resumes an already sealed proof, publishes,
and retries owned-file cleanup before removing the queue entry. Terminal job
failure remains in Build Workspace and is removed from scheduling; abort is
still explicit. Transient work failures release ownership with a retry delay.
Queue/lease corruption is returned by the scheduler rather than silently retried.
A stopped scheduler cancels its local tasks; another process recovers after the
persisted lease expires. No process-local state is publication authority.

## Consequences

Different jobs distribute across Runtime processes without a central service.
One job remains a coarse unit; this does not parallelize its tree construction.
There is one bounded discovery scan per available-capacity poll, one lease
transaction per active job per heartbeat, and one additional owner read in each
build mutation transaction. Heartbeats can conflict with in-flight chunks,
which use existing bounded retries. Total workers and resources across processes
are operator-controlled, not enforced by per-process limits.

FoundationDB supports workers in separate processes. RocksDB scheduling remains
inside the database-owning process. The existing advisory-lock/rename/fsync
filesystem contract remains required; local process-kill tests do not qualify a
network filesystem. No automatic abort or deletion of a published index is added.
