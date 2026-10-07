# ADR 0027: Durable workspace ownership and exact Bulk Build publication

Status: Accepted. The automatic-scheduling deferral is superseded by
[ADR 0028](0028-automatic-bulk-scheduling.md).

## Context

An initial Bulk Build must survive worker replacement and uncertain commits,
prove exact source membership before visibility, and reclaim files after the
index prefix has disappeared. Filesystem work cannot commit atomically with KV.

## Decision

Use one epoch-fenced preparation task, one fenced load task and a paged proof.
The accepted immutable Serving Artifact seals the complete finite inventory.
Core preparation proves exact joins; frozen backend validation compares every
serving key/value against that artifact before one Active Manifest commit.
This avoids independent task-registration and proof-reduction state machines.

Persist workspace ownership in a namespace record outside the index prefix.
Register a random token-owned directory before file creation. All native IO
holds a shared root advisory lock; cleanup takes that lock exclusively, deletes
the owned directory durably, then deletes its record. This orders reclamation
against stale writers, including detached blocking work. Caller-owned roots and
source snapshots are never reclaimed by the job.

Retain constant-size descriptor/load/proof records in Active indexes for
identity-bound recovery. Reclaim workspace records/files separately. Explicit
worker invocation/takeover is supported; automatic scheduling and intra-job
parallel construction are deferred. No stable format compatibility is required.

## Consequences

Publication is small and atomic; recovery and cleanup need no live coordinator.
A busy root can delay cleanup of other jobs sharing that root. Reliable advisory
locks, rename, and fsync are deployment requirements. Multi-host filesystem
qualification is separate from transactional adapter tests. Repeated prefix
reads and exact validation are deliberate IO costs for recovery and integrity.
