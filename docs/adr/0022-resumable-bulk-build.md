# ADR 0022: Resumable Bulk Build and distributed scheduling

Status: Accepted

## Context

Initial construction must handle inputs larger than memory without exposing
partial indexes. Jobs must survive process loss and uncertain commits, support
automatic distributed execution, and reclaim owned files safely. Backend
transactions cannot atomically commit filesystem operations.

## Decision

Online loading uses ordinary atomic `Index::batch_mutate` operations. Callers
own batch size, bounded concurrency and each result. Every batch retains ordinary
validation, atomicity, bounded retries, cancellation and unknown-commit semantics.
Runtime admission and Structure Maintenance remain independently bounded. Initial
offline construction uses the separate Bulk Build lifecycle described below.

Reserve an immutable source and construction configuration under a never-reused
Logical Index ID. Reserve the name and hidden Building Manifest atomically;
ordinary index operations require Active. Recovery remains bound to that ID,
including after name reuse or an unknown commit outcome.

Use one coarse build job with bounded preparation, loading and validation.
An immutable Serving Artifact fixes its complete inventory. Preparation proves
exact source membership; frozen backend validation compares all serving bytes
against the artifact before a single transaction activates the Manifest.
Durable progress resumes incomplete work; counts alone never prove completion.
Direct completion and automatic scheduling share this execution path. Individual
stages and artifact manipulation are internal.

Discover scheduled jobs from a durable namespace queue. Transactional claims
and renewable leases distribute jobs among processes. Every build mutation
checks the current owner token with update protection. Expiry permits takeover;
only a committed owner change revokes the previous worker's authority. This
prevents a paused worker from committing after replacement. Unknown outcomes
are resolved using the same attempted identity, never a fresh claim.

Persist workspace ownership outside the index prefix so cleanup survives index
removal. Register token-owned directories before creating files. Native IO holds
a shared root lock; cleanup holds it exclusively, durably removes owned files,
then removes the ownership record. Caller-owned input snapshots and workspace
roots are retained. Abort removes only unpublished identities; published indexes
require an explicit normal drop.

Retain the Build Descriptor and Build Progress after activation for identity-bound
recovery. Reclaim workspace files and bookkeeping separately. The API, record
layouts, checkpoints and resource limits are specified in
[Bulk Build](../design/bulk-build.md), [API](../design/api.md#bulk-build), and
[Storage](../design/storage.md).

## Consequences

Publication is atomic and recovery needs no live coordinator. Durable claims
provide distributed authority that a process-local task registry cannot; lease
expiry alone is insufficient to fence stale writes. One job remains a coarse
unit of work, with bounded resources per process and no intra-job distributed
construction. Aggregate concurrency and disk use remain operator controlled.

The cost is durable source/artifact storage, repeated scans, exact validation,
lease renewal and reclamation. Heartbeats may conflict with build transactions;
existing bounded retries handle them. Clock skew can delay takeover or duplicate
work without granting two owners commit authority.

Source/workspace mounts require reliable advisory locks, atomic rename and fsync.
A busy shared root can delay cleanup. FoundationDB permits workers in separate
processes; RocksDB workers remain in the database-owning process. Transactional
adapter tests do not qualify a multi-host filesystem.
