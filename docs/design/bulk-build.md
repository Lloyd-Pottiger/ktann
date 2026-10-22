# Resumable Bulk Build

Bulk Build constructs one new Logical Index from a finite immutable Input
Snapshot. It does not replace an existing Active index. The synchronous
[artifact pipeline](bulk-artifacts.md) owns source, forest, and serving bytes;
the runtime owns reservation, fenced loading, validation, and publication.

Persistent keys and encodings belong to [Storage](storage.md#6-persistent-values).
The rationale is recorded in [ADR 0026](../adr/0026-building-reservations.md),
[ADR 0027](../adr/0027-bulk-workspace-and-publication.md), and
[ADR 0028](../adr/0028-automatic-bulk-scheduling.md).

## 1. Manual completion

Create an Input Snapshot and call `Runtime::start_bulk_build(name, &input,
construction_options)`. ConstructionOptions contains the training sample size and construction budgets.
Pass `BulkBuildOptions { construction, refinement_rounds }` to request optional
artifact refinement; a plain ConstructionOptions selects no refinement. The
Build Descriptor binds `None` separately from `Some(0)` (centroid refresh without
moves), and accepts at most five rounds. These semantics cannot change on resume. Dimension, metric and capacities come from
the Input Snapshot's Index Configuration and are persisted in the Manifest;
the Build Descriptor does not store a second copy. Reservation atomically allocates an ID, reserves its name, and
persists a Building Manifest and Build Descriptor. Repeating the same request
recovers that identity. `open_bulk_build(name)` recovers without source IO.
Ordinary open, reads, and mutations reject Building indexes. The descriptor
exposes source identity and construction parameters with redacted Debug output.

Create an existing durable workspace and pass `BulkWorkerOptions::new(root)`
to `job.complete(options)`. It prepares, loads, validates, publishes, and attempts
cleanup. `complete_with_control` applies cancellation and deadline control across
the entire operation. No automatic scheduler is started. Explicit `run_worker`,
`load_serving`, and `publish` APIs remain available for phase control.

For a configuration with `LocalGraphConfig`, preparation refines
membership and recomputes centroids before accepting the final Forest Artifact,
then builds each internal graph exactly once from the final Child Entries while
encoding Serving bytes. Omitted refinement preserves construction centroids.
`refinement_memory_bytes` admits retained per-tree input, topology and numerical
planning; `graph_memory_bytes` bounds one graph's centroids, native construction
and serialization. Both are persisted worker resource limits, not semantic inputs.
Failure exceeds neither budget by falling back or omitting work. Construction and
refinement memory ceilings add while retained inputs feed construction.

After publication, graph-enabled indexes support ordinary mutations and graph-aware
maintenance. Public serving refinement returns `InvalidArgument`; optional artifact
refinement must finish before initial graph construction. Query HNSW routing is
selected independently of the construction path.

### 1.1 Completion reports

Receipt-time preparation can be passed with `complete_with_report`.
It is a volatile optimization: its source identity, configuration, and options
are checked before use; recovery recomputes it from the sealed Input Snapshot.

`complete_with_report(options, prepared_input, control)` completes the same
protocol while returning the Active Index and a `BulkBuildReport` containing
stage timing and the persisted rotation seed. Reused preparation and loading
stages report zero work. These measurements are
per-invocation observations, not throughput or latency guarantees.

### 1.2 Explicit phase control

`job.run_worker` prepares accepted Forest and Serving Artifacts and loads the
backend in bounded transactions. Repeating the same options resumes
accepted work and takes over old worker epochs. A worker arriving after validation
has begun returns `BulkBuildSuperseded`; `complete` retries publication.

`job.publish()` requires the immutable completed load,
performs resumable exact paged validation, and returns the ordinary Active Index.
Publishing before loading completes returns `BulkBuildBusy`; finish or resume
loading and retry. Both operations and `status`/`abort` have
`_with_control` variants. A failed job reports `Failed { kind }` and requires
abort/rebuild; cancellation is resumable.

`load_serving` remains a lower-level fenced load primitive. It fixes one immutable
artifact and commits data plus checkpoints atomically under `BulkLoadOptions` and
adapter admission limits. A completed load remains hidden; publication also
requires core worker acceptance of that artifact.

## 2. Automatic scheduling

Automatic scheduling is opt-in: after reservation call
`job.schedule(BulkWorkerOptions::new(shared_workspace)).await?`, then run
`runtime.run_bulk_scheduler(BulkSchedulerOptions::default(), control).await`
on each participating process. `schedule` persists the worker options. Queued
jobs reject manual mutation attempts with `BulkBuildBusy`.

The scheduler discovers jobs, renews leases, takes over expired workers,
publishes, and retries cleanup. Each build mutation transaction verifies the
owner token, fencing worker, load, and publication writes against takeover.
A failed job remains observable through `status` and does not stop unrelated
jobs. Invalid coordination state is surfaced rather than retried indefinitely.

Each scheduler admits at most `max_jobs` local tasks, additionally limited by
shared foreground admission. Discovery starts with a page sized to the available
slots and widens across leased entries, examining at most 256 queue records per
poll. Its cursor follows the last consumed record, so filling the available
slots never skips remaining queued work. Lease renewal runs inside the admitted
attempt without acquiring another foreground permit.

The scheduler future runs until `OperationOptions` cancellation/deadline or
Runtime shutdown. Dropping it cancels child tasks; shutdown stops discovery and
cancels active attempts. Native IO may outlive cancellation while retaining
admission and file locks. Stopping a scheduler does not abort durable jobs;
another participating Runtime can take over after the lease expires.

## 3. Durable ownership and recovery

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

## 4. Admission and cancellation

Bulk Build reservation, worker, load, publication, status, abort and cleanup use
foreground operation control and bounded retry policies. Native construction and
file IO run on blocking tasks with shared admission retained until actual exit,
even when the awaiting future is cancelled. Running native stages are not
preempted; subsequent admissions and checkpoints observe cancellation.

## 5. Loading and exact validation

Load registration fixes one Serving Artifact identity. Each invocation takes a
new load epoch; every transaction checks the original index identity and epoch,
and commits serving entries with its prefix count and digest checkpoint. Adapter
limits and caller budgets bound mutation count and physical bytes. Unknown
outcomes reread durable state. Recovery reconstructs the artifact prefix in
bounded memory; it does not rewrite the committed prefix.

Completed loading permanently fences data writes: new claims return the same
checkpoint and stale chunks cannot match it as their predecessor. Publication
registers a validation proof only after loading is complete; this proof owns the
validation phase without an additional sealing flag or checkpoint update.
Preparation has proved exact
source/forest joins; validation then compares every serving backend key/value
against the accepted immutable Serving Artifact in bounded pages. Missing, extra,
or different bytes fail closed. Lifecycle and build metadata are excluded from
this comparison. Persisted proof cursors and digests make validation resumable.
No serving writer can change the frozen Building index. One final transaction
checks the completed proof and original identity and changes its Manifest to
Active. There is no partially visible serving state.

`complete` first attempts publication and runs preparation/loading only when it
is not ready. It resumes Loaded builds without taking over a worker, and resolves
an already Active identity before opening files. A completed validation proof
can activate the index without reopening the accepted artifact files. Concurrent
invocations use bounded retries for
superseded attempts; a peer's successful publication resolves to the same index.
Unknown publication outcomes likewise resolve through the original ID. Cancellation
and transient failures remain resumable; terminal failures persist until abort.

## 6. Abort and cleanup

`abort` removes only the handle's unpublished identity through Dropping cleanup;
it cannot follow name reuse or abort an Active index. It is idempotent when the
original identity is absent. Caller-owned source files, workspace roots, and
coordination lock files are preserved. The namespace workspace ledger survives index
prefix deletion. Cleanup deletes only token-owned attempt files, fsyncs removal,
and then removes the ledger. An uncertain cleanup commit is safe to retry.

Completion, publication, and abort attempt cleanup. A busy lock defers reclamation;
other cleanup failures may be returned after the lifecycle commit. Inspect status
or repeat the operation. `job.cleanup()` and bounded
`Runtime::cleanup_bulk_builds(maximum, after)` finish deferred or orphan cleanup.
An in-progress job is retained, not reclaimed.

## 7. Verification

`tests/bulk_serving.rs` includes the load and completion suites: checkpoints,
unknown outcomes, cancellation, epoch takeover, path recovery, concurrent
completion, exact corruption checks, atomic visibility, abort, and cleanup.
The shared `tests/support/bulk_load_adapter.rs` exercises durable reopen, exact
loaded bytes, complete/publication, verification, and cleanup on RocksDB and
FoundationDB. Filesystem qualification across hosts is outside these tests.

See [ADR 0026](../adr/0026-building-reservations.md) and
[ADR 0027](../adr/0027-bulk-workspace-and-publication.md) for persistence decisions.
