# Resumable Bulk Build

Bulk Build constructs one new Logical Index from a finite immutable Input
Snapshot. It does not replace an existing Active index. The synchronous
[artifact pipeline](bulk-artifacts.md) owns source, forest, and serving bytes;
the runtime owns reservation, fenced loading, validation, and publication.

## Manual completion

Create an Input Snapshot and call `Runtime::start_bulk_build(name, &input,
construction_options)`. Construction partition bounds must match the Index
Configuration. Reservation atomically allocates an ID, reserves its name, and
persists a Building Manifest and Build Descriptor. Repeating the same request
recovers that identity. `open_bulk_build(name)` recovers without source IO.
Ordinary open, reads, and mutations reject Building indexes.

Create an existing durable workspace and pass `BulkWorkerOptions::new(root)`
to `job.complete(options)`. It prepares, loads, validates, publishes, and attempts
cleanup. `complete_with_control` applies cancellation and deadline control across
the entire operation. No automatic scheduler is started. Explicit `run_worker`,
`load_serving`, and `publish` APIs remain available for phase control.

## Durable ownership and recovery

The Build Descriptor binds the source's absolute path, trusted Artifact Manifest,
and construction options. The namespace-owned Build Workspace binds a canonical
workspace path, resource limits, random ownership token, attempt epoch, accepted
artifacts, and terminal failure. Workers register ownership before creating files.
Accepted Forest and Serving Artifacts survive worker replacement. Accepted
Serving bytes permit loading and validation without reopening the source path.
Before acceptance, the immutable source must remain available at its stored path.
After publication, completion requires neither source nor workspace artifacts.

The workspace filesystem must provide reliable advisory locks, atomic rename,
and fsync. Native IO retains shared root locks and admission until it actually
exits, including when its async waiter is cancelled. Cleanup takes an exclusive
root lock. Workspace options must match the persisted options when resuming
preparation. Multi-host filesystem qualification is a deployment responsibility.

## Loading and exact validation

Load registration fixes one Serving Artifact identity. Each invocation takes a
new load epoch; every transaction checks the original index identity and epoch,
and commits serving entries with its prefix count and digest checkpoint. Adapter
limits and caller budgets bound mutation count and physical bytes. Unknown
outcomes reread durable state. Recovery reconstructs the artifact prefix in
bounded memory; it does not rewrite the committed prefix.

Publication seals the loader before validating. Preparation has proved exact
source/forest joins; validation then compares every serving backend key/value
against the accepted immutable Serving Artifact in bounded pages. Missing, extra,
or different bytes fail closed. Lifecycle and build metadata are excluded from
this comparison. Persisted proof cursors and digests make validation resumable.
No serving writer can change the frozen Building index. One final transaction
checks the completed proof and original identity and changes its Manifest to
Active. There is no partially visible serving state.

`complete` skips preparation after sealing and resolves an already Active
identity before opening files. Concurrent invocations use bounded retries for
superseded attempts; a peer's successful publication resolves to the same index.
Unknown publication outcomes likewise resolve through the original ID. Cancellation
and transient failures remain resumable; terminal failures persist until abort.

## Abort and cleanup

`abort` removes only the handle's unpublished identity through Dropping cleanup;
it cannot follow name reuse or abort an Active index. Caller-owned source files
and workspace roots are preserved. The namespace workspace ledger survives index
prefix deletion. Cleanup deletes only token-owned attempt files, fsyncs removal,
and then removes the ledger. An uncertain cleanup commit is safe to retry.

Completion, publication, and abort attempt cleanup. A busy lock defers reclamation;
other cleanup failures may be returned after the lifecycle commit. Inspect status
or repeat the operation. `job.cleanup()` and bounded
`Runtime::cleanup_bulk_builds(maximum, after)` finish deferred or orphan cleanup.
An in-progress job is retained, not reclaimed.

## Verification

`tests/bulk_serving.rs` includes the load and completion suites: checkpoints,
unknown outcomes, cancellation, epoch takeover, path recovery, concurrent
completion, exact corruption checks, atomic visibility, abort, and cleanup.
The shared `tests/support/bulk_load_adapter.rs` exercises durable reopen, exact
loaded bytes, complete/publication, verification, and cleanup on RocksDB and
FoundationDB. Filesystem qualification across hosts is outside these tests.

See [ADR 0026](../adr/0026-building-reservations.md) and
[ADR 0027](../adr/0027-bulk-workspace-and-publication.md) for persistence decisions.
