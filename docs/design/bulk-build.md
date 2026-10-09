# Resumable Bulk Build

Status: **Implemented for a complete, resumable initial build.**

Throughput and search-quality limitations, with a proposed redesign, are tracked
in [Bulk Build Throughput Redesign](bulk-build-performance.md). The redesign is
a draft and does not change the implemented contract below.

`start_bulk_build` reserves a hidden Logical Index; `run_worker` prepares and
loads it; `publish` validates the frozen backend and atomically makes it Active.
Import Sessions were removed under [ADR 0025](../adr/0025-caller-owned-online-batch-submission.md).
[ADR 0027](../adr/0027-bulk-workspace-and-publication.md) records the bounded
coarse-task protocol and workspace ownership. This implementation supports
explicit worker invocation and automatic distributed scheduling under
[ADR 0028](../adr/0028-automatic-bulk-scheduling.md). A single job remains the
unit of work; its tree construction is not parallelized. Multi-host shared-filesystem qualification
remains a deployment requirement; it is not established by local adapter tests.

### Implemented reservation contract

`Runtime::start_bulk_build` atomically reserves the name and a never-reused
Logical Index ID with a Building Manifest and an immutable Build Descriptor.
The descriptor persists the absolute UTF-8 source directory, expected input
manifest, construction algorithm version, options, and resource ceilings.
Identical requests reopen the reservation; conflicting requests fail. An unknown
reservation commit is resolved against its attempted ID, never a replacement.
`open_bulk_build` recovers the request without reading source files. Ordinary
create/open and serving paths reject Building with `IndexBuilding`.

`BulkBuildJob::status` follows the original ID. `abort` transitions Building to
Dropping and uses bounded existing backend cleanup; an old handle cannot remove
a replacement under the same name, and abort rejects a published index. Input
files remain caller owned. Runtime operation controls and admission apply to
all job operations. No construction workers are launched by reservation yet.
Worker-owned files require a namespace cleanup ledger before creation;
reservation alone creates no filesystem artifacts. Dropping jobs can be cleaned
through `drop_index` or an existing job handle; reopening Dropping returns
`IndexDropping` because cleanup may already have removed the descriptor.

### Implemented multi-tree construction contract

`ForestArtifact::build` consumes the complete finite source and externally sorts
Record ID projections before constructing any tree. IDs duplicated across any
Tree Keys fail the entire build. A second bounded external sort groups original
vectors by canonical Tree Key; each group streams through the existing
single-tree constructor. Full Records and payloads remain in the source snapshot.
The implementation never retains the whole source or a directory of all trees.

Global sort runs charge allocated payload capacities and row slots, and use
binary carry merges. One
quota charges all live runs, including retained inputs while writing merged
outputs. Global sort scratch and per-tree construction scratch may overlap, so
their disk ceilings add; durable output has a separate quota. Sort and training
memory are used sequentially, with a bounded projection reader retained during
training. Source decoding and final frame encoding remain separately codec-bounded
caller IO. Successful construction removes all temporary runs; failed output
stays unsealed in its caller-owned directory. No job-owned worker IO is added.

Forest artifact kind `2` uses the existing independently versioned framing.
Each body is a u32 Tree Key byte length, canonical Tree Key, then the existing
partition-plan body. Its binding includes the complete source identity, ordered
Tree Key field selection, metric, persisted seed, construction version, and all
resource options. Trees are in canonical order; partitions are child-before-parent
with each root last. Reopen/verification checks local allocation, occupancy,
ordered tree closure, and total leaf assignments as well as all file checksums.
This does not replace the later exact membership and serving-value validation.
Empty input emits no partitions or synthetic Tree Manifest.

### Implemented serving-data artifact contract

`ServingArtifact::build` binds the source, forest and immutable Index Manifest,
including its Logical Index ID, persisted rotation seed and Bloom parameters.
External joins match every source Record to exactly one leaf assignment and
check its actual Tree Key. A second join matches every non-root partition to
exactly one incoming reference one level above; roots have no incoming reference.
With the forest's unique root and local allocation checks, decreasing levels and
one parent per non-root prove reachability without retaining the graph in memory.

The encoder reuses core codecs and online preprocessing: original Records and
payload presence are preserved, Leaf Entries use absolute RaBitQ7 codes, and
Child Entries copy finalized child centroids exactly. Headers have exact counts,
cache epoch 1, and Ready state; transition timestamps are 0 (unavailable). Non-root
centroids and leaf-only Synopses follow existing serving layout. Synopses are
reduced from exact leaf fields, including persisted Bloom parameters. Tree
Manifest high-water marks cover each complete plan.

Artifact kind `3` contains strictly ordered unique canonical logical KV pairs;
each framed body is a u32 key length, key bytes and value bytes. No Manifest,
name mapping, namespace allocator or build metadata is included. Reader validation
checks index ownership, canonical key/value codecs, key order and declared adapter
limits, as well as complete file identity. These checks do not grant write or
publication authority. Target limits and sort options are part of the artifact
binding; admission/chunk budgets are enforced by the fenced loader. Source and
value decoding and one Synopsis remain separately codec-bounded IO allocations;
sort payload capacities, slots and merge buffers have an explicit memory ceiling.
All simultaneous scratch runs share one quota and failed directories stay caller
owned and unsealed. The artifact itself is a prepared input to later loading,
not proof that a backend contains that complete data.

The ordinary Active verifier validates any retained Build Descriptor's codec but
excludes it from membership/topology ledgers; malformed build metadata remains an
InvalidEncoding issue. This allows bookkeeping reclamation after publication.

### Implemented fenced serving load contract

`BulkBuildJob::load_serving(artifact, BulkLoadOptions)` explicitly drives one
coarse, sequential load task. It does not launch construction or distributed
workers. The serving handle must match the reserved source, construction options,
and immutable Index Manifest. The first claim atomically fixes the complete
artifact manifest; later calls may resume only that exact identity. The complete
artifact is the sealed inventory for this single task, so no paged task
registration is needed.

Index-owned key kind `0x06` stores Build Progress (value tag `0x0e`): one sized
89-byte artifact manifest, a positive u64 load epoch, and a canonical phase.
Loading carries committed entries and the artifact-prefix SHA-256; Loaded carries
no checkpoint fields. Completion is recorded only after artifact EOF verification,
not inferred from counts. Each unfinished load invocation takes over by increasing
the epoch; repeating a Loaded invocation is an idempotent no-op. Epoch overflow
fails without resetting ownership. Every chunk update-protects the Building
Manifest and Build Progress, checks its epoch, and atomically writes serving
entries plus the next progress state. It does not rewrite the shared Manifest.
An old prepared transaction conflicts with takeover, sealing, or drop and then
observes the durable replacement state.

A claim whose commit outcome is unknown grants no worker authority; the caller
may retry, taking a new epoch. Unknown chunk commits are resolved by reading the
same epoch's checkpoint before replaying deterministic bytes. Each transaction
charges both data and checkpoint mutations, including adapter namespace overhead,
against the smaller of caller and adapter admission budgets. Canonical KV bytes
from the validating serving reader are passed through without re-encoding.

File reading runs on bounded blocking tasks retaining foreground admission.
Recovery replays the committed prefix without rewriting it, reconstructing the
whole-file SHA-256 state in bounded chunks. Before skipping that prefix, its
digest must match the checkpoint committed with the backend bytes. A changed
committed prefix fails closed and requires abort/rebuild, including when validly
framed damage was only detected by the earlier whole-file check. Memory is bounded by the chunk byte
budget, one look-ahead frame, codec buffers, and adapter transaction buffering;
recovery IO/time remains proportional to the committed prefix. Full reader
exhaustion, including final digest validation, is required before atomically
marking Loaded, even when the cursor already equals the declared item count.
Cancellation and errors can leave a committed prefix. Files remain caller owned;
restoring the identical immutable artifact permits resumption only when its
committed-prefix digest still matches.

Status distinguishes Preparing, Loading (committed and total entry counts), and
Loaded while the Manifest remains Building. Loaded grants no publication
permission and performs no backend validation. The worker/publication APIs add persistent failure reporting, owned-file
cleanup and sealed backend validation as described below. The current loader does not
require record groups to fit one transaction: all intermediate data is hidden
and cannot be read by serving operations.

### Implemented preparation file contract

`InputSnapshot::create` streams complete original Records (IDs, vectors, typed
fields, and optional payloads) into a new, exclusively owned directory.
Absent payload and present-empty payload remain distinct. Schema validation and
canonical encoding reuse storage's existing record primitives without inventing
an Index Manifest before reservation. Source order is preserved; duplicate-ID
detection belongs to the later global preparation sort. Source shape binds the
dimension and ordered field names/types/nullability. Metric, Tree Key selection,
partition sizes, and synopsis policy remain consumer choices.

`ForestArtifact::build` consumes and verifies a snapshot, groups by Tree Key,
invokes the existing constructor, and seals child-before-parent partition plans.
Without Tree Key fields it builds a single tree through the same pipeline. Its descriptor binds the
complete source manifest, metric, persisted rotation seed supplied by the caller,
construction version, and all construction options. Full Records remain in the
source for the exact assignment join and serving-value encoder.

The independently versioned file format uses an 89-byte manifest: eight magic/
version bytes, a one-byte artifact kind, a 32-byte binding, two big-endian u64s
(item count and exact data-file bytes), and a 32-byte SHA-256. Data starts with
the same magic/kind/binding, then bounded frames containing a big-endian u32
length, body, and body SHA-256. The manifest hashes the entire data file, so
reordering otherwise valid frames also fails verification. Input bodies reuse
the canonical record/vector/field wire primitives and append payload presence
and optional bounded payload bytes. Plan bodies contain key, level, count,
full-f32 centroid, and canonical length-prefixed entry identities.

Sealing flushes and syncs data, renames it, syncs its directory, then syncs and
renames the manifest completion marker and syncs the directory and its parent.
Reopen requires the expected manifest saved by the owner, rather than trusting
the directory's current descriptor. Opening checks metadata/header identity;
successful reader exhaustion checks every frame, canonical body, exact count,
EOF, and whole-file hash. Dropping a reader early proves only its consumed
frames. Input verification must finish before a dependent plan can be sealed.
The job worker fences authority before accepting a sealed file.

Source storage, temporary sort storage, and final artifact storage have separate
explicit quotas. Preparation's total disk allowance must cover their sum;
the current pure-construction memory budget excludes the caller's source record
and artifact encoding buffers. Failure leaves an unsealed attempt directory for
its owner to reclaim. No API overwrites or recursively cleans an existing
directory. Crash recovery can reopen completed artifacts against their saved
descriptors; interrupted construction attempts are recomputed in a new directory.

## 1. Problem and scope

An initial load currently builds an index by repeatedly executing ordinary
Foreground Mutations and Structure Maintenance. Early writes share a small
number of Leaf Partitions, update their exact counts and Synopses, and compete
with split exposure and draining. Records can be moved repeatedly as the tree
grows. More import processes do not create independent leaf write capacity.

The reported baseline is approximately ten minutes for one million records.
This is an observation to reproduce, not a measured attribution of time to
conflicts. The new path targets 10M, 100M, and eventually 1B records without
requiring them to fit in memory or holding a transaction for the build duration.

The first release constructs a **new Logical Index from a finite immutable input
snapshot**, then makes it queryable in one publication step. During construction,
ordinary reads and writes to that index are rejected. Existing unrelated indexes
remain available. This is the implemented product scope.

Non-goals are appending a build to an Active Logical Index, accepting concurrent
mutations into the build, rebuilding behind an existing Index Name, online name
swapping, CDC catch-up, changing search semantics, and promising linear speedup.
These require separate contracts rather than implicit extensions of this path.

Success means lower end-to-end time to a searchable, structurally ready index,
without sacrificing exact membership, recovery, bounded resources, or recall and
latency at the same Search Budget. Upload, preparation, validation, and publication
are part of the reported time; moving work out of the foreground is not a saving
by itself.

## 2. Evidence and changes to existing decisions

| Current fact | Evidence | Consequence |
| --- | --- | --- |
| Import was process-local adaptive admission for ordinary atomic batches | [ADR 0022](../adr/0022-feedback-controlled-import-admission.md), superseded by [ADR 0025](../adr/0025-caller-owned-online-batch-submission.md) | Removed; online callers use ordinary mutation APIs directly. |
| Writes route and update exact membership, counts, and Synopses | [`mutation`](../../src/maintenance/mutation.rs), [`membership`](../../src/storage/membership.rs) | Independent final outputs avoid shared incremental counters. |
| Splits train from complete snapshots and expose then drain | [`training`](../../src/maintenance/training.rs), [ADR 0014](../adr/0014-expose-then-drain-splits.md), [ADR 0015](../adr/0015-incremental-binary-kmeans-tree.md) | Reusing the online split orchestrator would preserve its scaling costs. |
| Lifecycle creation immediately makes an Active index | [`create_index`](../../src/runtime/lifecycle.rs), [`IndexManifest`](../../src/storage/values/manifest.rs), [ADR 0017](../adr/0017-manifest-governed-index-lifecycle.md) | Construction needs an explicit non-serving lifecycle. |
| Tree Key is user-schema-derived, and partition allocation is tree-local | [ADR 0004](../adr/0004-sharded-kmeans-forest.md), [ADR 0020](../adr/0020-tree-manifest-directory.md) | Worker sharding must also work inside a single Tree Key. |
| Verification currently uses one validated Active snapshot and in-memory ledgers | [`verify`](../../src/runtime/verify.rs) and its `records`/`topology` modules | Publication needs sealed, paged verification, not an unbounded call to `Index::verify`. |
| Ordinary non-root partitions have exact incoming references | [ADR 0007](../adr/0007-exact-internal-membership-without-parent-pointers.md) | Offline clustering must emit the actual serving topology and projections. |

The [overview](overview.md) explicitly excludes staging indexes and bulk-build
generations. [ADR 0001](../adr/0001-exact-leaf-membership.md) requires exact
membership in every committed state, and ADR 0015 selects one uniform online
construction path. This proposal changes those decisions; it is not merely an
implementation optimization under the current contract. On acceptance, record a
new ADR superseding those portions and update the owning designs and glossary.
Do not silently rewrite accepted ADRs.

## 3. Public boundary and Import Session removal

Remove `ImportSession` rather than rename it or introduce a replacement session.
Online callers use `insert`, `upsert`, `delete`, and `batch_mutate` directly.
Full initial construction uses `BulkBuildJob`.

Remove `Index::import_session`, `ImportOptions`, `ImportBatchResult`,
`ImportCoordinator`, and the session-only `BatchToken`, together with their
exports, modules, configuration, telemetry, and documentation. Remove the
session's adaptive concurrency and Fixup Backlog admission gate. Do not migrate
these policies into Runtime or introduce shared foreground-write backpressure
as part of this change. Reconsider backpressure separately only when workload
evidence justifies it. Existing unrelated runtime bounds and retry policies stay.

Callers own batch submission, bounded concurrency, and result handling. Each
`batch_mutate` retains its current atomicity, retry, cancellation, and unknown
commit contract; there is no session-wide ordering or completion contract.
Removing admission can increase contention if a caller submits too much work;
consumer migration must choose an explicit bounded execution loop, with
sequential submission as the simplest default. It must not launch unbounded
batch tasks or claim to preserve the removed adaptive throughput policy.

No deprecated aliases are needed: KTANN has no stable release. Existing CLI
`import` commands may retain their user-facing name but must identify whether
execution uses direct online batches or Bulk Build. Preserve their record and
error reporting requirements when replacing Session-based implementations.

A **Bulk Build Job** is persistent construction work for one newly allocated
Logical Index. Its identity is that Logical Index ID; do not add an equivalent
second allocator. A **Build Task** is a durable, independently retryable unit of
work. These are distinct from maintenance Fixups and are not transaction identities.
Process-local Batch Tokens were removed with Import Sessions (ADR 0025).

The proposed public operations, with signatures to be finalized during the API
slice, are:

- `Runtime::start_bulk_build(name, config, input_manifest, options)` reserves the
  name and returns a `BulkBuildJob` handle. Retrying an identical request resumes
  the same build; a different descriptor at that name fails.
- `Runtime::open_bulk_build(name)` recovers a handle after process loss.
- `BulkBuildJob::status()` returns durable phase, entry progress and failures. `run_worker(worker_options)` drives bounded local work;
  multiple processes can call it against the same distributed backend.
- `BulkBuildJob::publish()` completes sealed validation and attempts publication;
  it returns the ordinary `Index` only after Active is known to have committed.
- `BulkBuildJob::abort()` fences work and transitions through normal drop cleanup.

Dropping a handle or stopping a worker stops local execution, not the job. A
terminal input or integrity failure leaves the index non-serving with inspectable
failure information until explicitly aborted. Ordinary create/open report
`IndexBuilding` for a reserved building name; they never attach to partial data.
An existing Active name makes a new start fail; recovery through a known job
identity can report that its publication already succeeded. Every handle binds
the never-reused Logical Index ID, so name reuse cannot redirect a retry to a
different index. Abort rejects Active: deleting a published index requires the
ordinary explicit drop operation.

## 4. Ownership and deployment

Core storage owns build lifecycle records, task claims, checkpoints, publication,
canonical keys, and invariant validation. A pure construction module owns
clustering and final partition descriptions. Runtime owns bounded workers,
retry policy, cancellation, and resource admission. Adapters retain ownership of
transaction size limits, commit classification, and physical encoding.

Input decoding and immutable scratch artifacts are a narrow build IO boundary.
The initial implementation uses a filesystem directory, with atomic finalization
of immutable artifacts. Multi-host workers require that directory on shared
durable storage with the same semantics; node-local paths are not distributed
artifacts. An object-storage implementation is deferred, not silently assumed.
The source descriptor records schema, format, immutable file identities, byte
lengths, digests, and deterministic record ordinals. Mutable sources are rejected
or first materialized as immutable snapshots. Credentials are runtime inputs,
never persisted in descriptors or logs.

FoundationDB is the initial multi-process/multi-host target. RocksDB supports
workers in its owning process; this design does not make an embedded database
safe for arbitrary multi-process access. Memory supports ephemeral construction
and tests, not crash-durable jobs. Shared storage bandwidth and the final KV
backend remain throughput limits even with more compute workers.

## 5. Construction algorithm

### Normalize and establish unique identity

Validate the same schema, dimensions, finite numeric values, Tree Key encoding,
and backend size limits as online mutations. Preserve original Vector Records
and payloads; use the existing metric normalization and persisted rotation for
routing and quantization. Stream normalization to immutable scratch files.

Input capture may use `InputSnapshotWriter` to validate and encode records and
update integrity hashes as batches arrive. Only `seal` exposes an immutable
Input Snapshot; append failure consumes the writer and leaves caller-owned,
unsealed files. No raw input rewrite is required at EOF. Job allocation and
forest construction still require the sealed input identity.

Externally sort only Record IDs across the whole Logical Index. During the
same source pass, sequentially spool the Tree Key, Record ID, and vector
projection under the shared sort scratch quota. This avoids carrying vectors
through duplicate-detection merges or decoding source payloads twice. Reject
duplicate IDs, including duplicates in different Tree Keys, before sorting
the projection by canonical Tree Key or emitting partitions. There is no
last-writer-wins rule or dependence on worker completion order. ID sorting and
Tree Key sorting share one memory budget and run sequentially; projection IO
buffers are included in the existing sort IO reservation. If the complete ID
set fits that sort budget, uniqueness is checked directly in the sorted buffer
without writing an intermediate ID run.
Carry record count and content digests through the task manifests. Empty input
produces an empty Active index without synthetic Tree Manifests.

### Form bounded leaf groups

For each Tree Key, recursively divide groups larger than `max_partition_entries`.
Large groups use a bounded deterministic sample: select the lowest seeded hashes
of canonical item IDs with ID tie-breaking. The persisted build algorithm and
sample size determine training; the same sealed input and configuration must
produce identical task outputs. Train two centroids on the sample using the
existing metric rules and balanced two-cluster numeric procedure.

Assign the complete group by externally sorting
`(distance_to_left - distance_to_right, canonical_id)` and cutting at
`floor(n / 2)`. This guarantees progress for duplicates, equal distances, and
skew; nearest-centroid assignment alone does not. Recurse until each group fits
the maximum. Since `2 * minimum <= maximum`, splitting only oversized groups
also respects the minimum for every non-root final group. A tree whose entire
population is below the minimum remains a single leaf root.

Each partition's final centroid is computed from its complete assigned group,
with canonical ID accumulation order and the current metric-specific treatment.
Root inputs are ordered by the initial duplicate preflight or by ascending
allocated parent keys. When the shortest encoded vector row is at least twice
the 20-byte split key, sort only `(distance_difference, input_ordinal)` to find
the exact boundary, then sequentially rescan and stably scatter the original
rows. Canonical ID order is preserved through each child, so ordinals resolve
ties identically to IDs and final centroids need no additional ID sort. For
smaller rows, sort complete rows and restore ID order only in final groups.
This row-width choice bounds the retained-input cost without changing grouping,
centroid bytes, algorithm identity or the configured memory ceiling.
Sample centroids guide grouping; they are not used as substitutes for final
centroids. Leaf groups are final assignments, not a requirement that every record
would subsequently follow greedy nearest-centroid insertion to that same leaf.
Search quality must be measured under the resulting topology.

The artifact pipeline's generic external sorter uses two to eight merge inputs,
selected from its existing memory budget and maximum encoded row size. All row
heads and IO buffers are reserved before forming runs. Numeric run IDs bound
pending metadata; final consolidation merges smaller runs first to avoid
repeatedly rewriting a large run. There is no new public tuning parameter.

This explicitly uses external sorting and repeated data passes. It is not a
single-pass algorithm and does not promise cheap construction at 1B scale. A
large single Tree Key initially needs parallel scan/sort work even before child
groups exist; external sort runs and merge tasks therefore have bounded fan-in
and durable outputs. Subgroup construction becomes parallel after each division.

### Assemble the serving topology bottom-up

Binary splitting describes the training operation, not a requirement that every
serving internal partition has exactly two Child Entries. Do not publish the
recursive grouping history as the serving tree.

Create level-1 leaves from the final record groups. Cluster their full-f32
centroids into bounded parent groups using the same grouping procedure, then
repeat at each level until a single root can contain the remaining entries.
Every parent references only children one level below. Root Partition Key is 1;
non-root Partition Keys are assigned deterministically in a sealed tree plan.
The Tree Manifest allocator high-water mark covers every assigned key before
publication. Parallel tasks consume their assigned keys, never a shared allocator
per record.

Compute a non-root internal centroid from its assigned Child Entry centroids
using current internal-training semantics. Generate incoming Child Entry
projections only after child centroids are final. All partitions are Ready,
counts are exact, and Synopses are conservative under existing storage rules.
Do not publish ReceivingSplit, DrainingSplit, or other maintenance transitions.

### Encode and load

Freeze the complete topology before generating dependent persistent bytes. Use
existing logical codecs, metric normalization, persisted rotation, and Bloom
parameters. Encode each Leaf Entry with the absolute, centroid-independent
RaBitQ7 code of the preprocessed record vector, exactly as online mutations do.
Each Child Entry stores its child's full-f32 centroid. Derive Header counts and
Synopses from final assignments once; loading records does not increment shared
partition counters or enqueue Fixups.

Load Vector Record/payload/Record Location/Leaf Entry groups with bounded
transactions; where one atomic group is required, validate it fits before load.
Metadata has separate disjoint task ownership. Deterministic chunk manifests
identify exact keys and values. Every write transaction validates its current
build authority and commits its chunk checkpoint in the same transaction.
Backend-specific direct file ingestion is outside this initial protocol.

## 6. Durable preparation, fencing, and retries

The first implementation has one preparation task, one load task, and one paged
validation proof per job. The single accepted Serving Artifact is the complete
load inventory. No per-record task queue, paginated task registration, lease
service, or hierarchical proof reduction is needed for this finite inventory.
Workers are explicitly invoked or claimed by the automatic scheduler; each
invocation takes a new preparation epoch.
This trades intra-job parallelism for a small, recoverable protocol. Different
jobs can run concurrently within Runtime admission and backend limits.

`run_worker(BulkWorkerOptions)` requires an existing absolute workspace root.
Before creating any job-owned files, it holds a shared advisory lock on
`<root>/.ktann-build-lock` and registers a namespace-scoped Build Workspace.
The record fixes the options, backend hard limits, random ownership token, epoch,
and accepted Forest/Serving Artifact descriptors. Job files live under
`<root>/<token>/attempt-<epoch>/`. Parent directories are fsynced. Only an output
accepted transactionally under the current epoch and Building Manifest can
become an input. Interrupted or superseded attempts remain owned cleanup work.

Recovery reuses accepted predecessors. Once Serving is accepted, it does not
reopen Source or Forest. Otherwise the immutable source is checked against its
reserved manifest. An unknown claim grants no authority; callers retry with a
new epoch. Unknown acceptance is resolved against the exact before/after record.
Loading uses the independent load epoch in Build Progress and bounded atomic checkpoint
protocol above. Terminal preparation/validation errors persist in Build Workspace
and surface as `Failed { kind }`; abort/rebuild is required. Cancellation and
transient failures leave resumable work.

### Automatic distributed scheduling

`job.schedule(worker_options)` persists an idempotent request for that original
Logical Index ID. `runtime.run_bulk_scheduler(settings, control)` is the
long-running process entry point: it polls bounded namespace pages, competes for
expired/unowned jobs, renews ownership, runs preparation/loading, resumes sealed
validation, and publishes automatically. Configure `max_jobs` (1..=64),
`poll_interval` (at least 1 ms), and `lease_duration` (at least three poll
intervals). Defaults are one job, one-second polling, and a 30-second lease.

Queue records survive process loss and index-prefix removal. They persist the
name/options, random owner token and expiry. All build mutation transactions
update-protect the token, including acceptance, load chunks, proof pages and
activation. A superseded worker cannot commit even if its native IO finishes
later. Queued jobs reject manual worker/load/publish mutation attempts with
`BulkBuildBusy`; explicit abort remains available. Existing job status reports
Preparing until progress begins, then Loading/Loaded/Validating/Published or
Failed. No automatic scheduler is created merely by reserving a job.

Unknown queue/claim commits resolve by reading the same identity/token. Work
failures remain resumable or persist Failed according to the existing worker
contract; one failed job does not stop discovery. Corrupt coordination metadata
surfaces as a scheduler error. Successful publication/cleanup retires the queue;
cleanup busy errors retain it for retry. Interrupted Dropping jobs resume the
original identity's drop cleanup, never a replacement name. Failed jobs retire
from scheduling but retain their workspace until explicitly aborted.

Stopping/dropping the scheduler cancels local work. Native IO retains its old
admission/lock guards until actual exit; another process can reclaim the queue
after expiry. Token changes, not elapsed wall time, enforce correctness. Clock
skew affects recovery latency and redundant IO. Each process must expose the
same immutable source and workspace paths; filesystem qualification is separate.

## 7. Sealing, exact validation, publication, and reclamation

`publish` requires the Serving Artifact accepted by core preparation. A manually
loaded artifact alone does not authorize publication. The core encoder's exact
source/membership/topology joins establish the artifact's correctness; validation
then proves that the backend contains exactly that artifact, byte for byte.
Together these establish exact membership, rather than trusting counts or an
aggregate checksum alone.

Publication attempted before loading completes returns resumable `BulkBuildBusy`,
without recording a terminal failure. Sealing atomically update-protects the
Building Manifest and Build Progress, then changes Loaded to Validating with an
empty scan cursor, zero compared entries, and the artifact-header digest. Updating
the same progress key conflicts with
in-flight writers and rejects subsequent claims. The lifecycle remains Building;
`status` reports Validating with verified/total entries. A validator scans the
entire index prefix in bounded pages, decodes known control keys, and compares
all remaining keys and values one-to-one with the accepted sorted artifact.
Missing, extra, or changed data fails closed. Each Validating checkpoint atomically
persists scan cursor, compared entry count, and artifact-prefix SHA-256. Recovery
replays and verifies that file prefix before continuing. Transition to Validated requires the complete backend scan, artifact EOF
and the full accepted artifact digest. Counts alone never authorize this transition. Frozen data permits safe page retries.

Publication checks the Validated progress, workspace identity, and
Building Manifest in one small transaction, then changes only the Manifest to
Active. Unknown outcomes and concurrent publishers resolve against the original
Logical Index ID. Ordinary operations become available only after that commit;
no prefix swap or data copy occurs. Abort and publication serialize on the same
Manifest. An old handle can never publish a replacement using the same name.

The namespace Build Workspace survives index-prefix deletion. `publish` and
`abort` attempt reclamation; `cleanup` retries one job, and
`Runtime::cleanup_bulk_builds(maximum, after)` discovers orphaned records in
bounded pages. Building jobs are retained. Cleanup takes the root's exclusive
advisory lock, removes only the token-owned directory, fsyncs its parent, then
removes the durable record. Active native IO retains the shared lock even if its
async waiter is cancelled. A busy root defers reclamation without invalidating a
successful publish/abort; other cleanup errors are reported and can be retried.
Publication may therefore already be committed when cleanup reports an error;
`status`/idempotent `publish` resolve it. All jobs using one root share this IO
barrier. The caller owns the root, coordination file, and source snapshot.

Constant-size Build Descriptor and Validated Build Progress remain
inside the Active index to support original-identity status and publication
retries. They are removed with the index. Preparation files and namespace ledger
are reclaimed independently. Crash recovery does not require an in-memory job
registry or the original worker process.

## 8. Resource and operational contract

Construction, sort, artifact, load mutation/byte, and scan-page ceilings are
explicit. Memory limits bound the relevant algorithm buffers, not total process
RSS, adapter caches, allocator overhead, or all concurrent jobs. Native file and
construction work uses bounded blocking admission retained until the closure
exits. Cancellation stops admission/checkpoints; a running native stage may
finish before releasing resources. Operators bound aggregate concurrent jobs.

Load admission takes the smaller of configured and backend limits and charges
checkpoint/namespace overhead. It is deterministic bounded admission, not an
adaptive latency controller. No new online Fixup Backlog gate is introduced.
The source and accepted artifacts consume durable disk in addition to temporary
sort scratch; per-artifact ceilings are not a global filesystem quota. Provision
aggregate space and throughput for the chosen number of jobs.

The filesystem must provide reliable advisory locks, atomic rename and fsync.
Local Memory tests establish protocol behavior, RocksDB establishes local disk
recovery, and FoundationDB establishes transactional adapter behavior. These do
not qualify an arbitrary network filesystem or provide a multi-host SLA.

## 9. Alternatives and trade-offs

| Alternative | Decision |
| --- | --- |
| Increase online batch concurrency | Does not remove hot leaves or movement; callers use bounded direct mutations, without a Session API. |
| Pre-create sampled leaves, then use ordinary batches | Smaller change, but shared counters and skew-driven maintenance remain; useful as a benchmark comparator, not the chosen final path. |
| Disable maintenance while loading one root | Reject: transfers cost to huge partitions, later training, and read quality. |
| Train and publish the full recursive binary grouping tree | Reject: does not preserve serving occupancy and level structure merely by using k=2. |
| Direct backend SST/file ingestion | Defer: adapter-specific and cannot replace portable lifecycle and exact validation. |
| Build a replacement generation behind an Active name | Defer: requires write capture, cutover, and old-generation reclamation absent from the initial-load contract. |

Selected costs are temporary storage, repeated sorting/scanning, durable build
metadata, delayed query availability, and a second construction algorithm to
validate. The benefit is eliminating online split contention and relocation from
initial construction while exposing independent tasks. No speedup or recall
claim is accepted before measurement.

## 10. Implementation and validation

The implemented path spans `construction`, `bulk`, runtime lifecycle/worker/load/
publish modules, persistent codecs, and all adapters' shared transaction contract.
The public APIs and repository callers use direct online batches or the explicit
Bulk Build workflow; no Import Session compatibility layer remains.

Tests cover exact artifact joins, all metrics' online encoding parity, bounded
load admission, stale worker fencing, unknown commits, cancellation, terminal
failure, empty input, backend omissions/extras, paged proof recovery, concurrent
publication, abort races, reopen, ordinary reads/mutations, and orphan cleanup.
RocksDB and FoundationDB share a full prepare/load/reopen/publish/verify test.
Persistent encodings fail closed on malformed bytes.

`ktann-bulk-build` measures complete SIFT construction through RocksDB publication
and cleanup, followed by point-read checks and held-out query recall/latency.
Its report distinguishes preparation/loading and validation/publication time.
Measurements are empirical, not an SLA or a comparison to online insertion.
Cohere, larger populations, concurrent-serving impact, multi-host filesystem
failure injection, and distributed scaling require separate qualification. The
implementation does not claim a 1B SLA or a
recall/speedup threshold that has not been measured against an agreed baseline.
