# Resumable Bulk Build

Bulk Build constructs a new Logical Index from a finite immutable input snapshot.
Its Building Manifest hides partial data until exact validation and one atomic
publication commit make it Active. Existing unrelated indexes remain available.
Appending to an Active index, online name swaps, CDC catch-up and replacement
generations are outside this contract.

[ADR 0022](../adr/0022-resumable-bulk-build.md) records the architectural choices.
The [API contract](api.md#bulk-build) owns caller operations and error behavior;
[Storage](storage.md#10-bulk-build-records) owns persistent encodings. This document
owns the construction algorithm, stage transitions and recovery protocol.
See the [benchmark guide](../../benchmarks/README.md#bulk-build-through-vectordbbench)
for performance measurement and reproduction.

## Ownership and execution

The storage layer owns identity allocation, lifecycle records, canonical codecs,
checkpoints and transactional validation. The pure construction module produces
partition plans without backend IO. Runtime owns bounded blocking work, retries,
cancellation and admission. Adapters own transaction limits, commit outcomes and
physical encoding.

Direct completion and automatic scheduling use the same internal
`bulk_worker::complete` path. A job has one preparation task, one load task and
paged exact validation. Its accepted Serving Artifact is the complete load
inventory; there is no per-record queue or independent per-partition loading.
Trees and recursive subgroups execute sequentially. Independent jobs can run
concurrently within Runtime and backend limits.

The caller owns the immutable source and workspace root. Job-owned directories
require reliable advisory locks, atomic rename and fsync. Multi-host workers
need source and workspace mounts with those semantics; node-local files are
not distributed artifacts, and object storage is not supported. FoundationDB
permits workers in separate processes; RocksDB workers remain in the owning
process. Memory supports ephemeral jobs and protocol tests, not crash durability.

## Construction algorithm

### Capture and establish unique identity

Validate schema, dimensions, finite vector values and canonical Record IDs as
for online mutations. Preserve complete original records, field values and
payload presence in the source. Preparation groups original vectors by Tree Key;
metric normalization and persisted rotation are applied by construction and
serving encoding, not by changing the source snapshot.

Input capture validates and hashes records as they arrive. A consuming append
that fails cannot later seal a partial batch. EOF flushes, syncs and seals the
source. Captured batches are not published index mutations or independently
durable receipts.

Optional receipt-time preparation sorts compact Record ID and Tree Key/vector
projections while retaining original source order. Two sorters share one IO
reservation, memory ceiling and scratch quota; their IO is serial. Callers apply
backpressure between batches. No normalization, rotation or partition training
runs before EOF. The resulting one-use preparation is bound to the source,
configuration and options; it avoids repeating the source scan and sort.
It is caller-owned, volatile work, never a durable recovery checkpoint. If it is
lost, workers reconstruct it from the sealed source.

Ordinary preparation externally sorts global Record IDs while spooling Tree Key,
ID and vector projections in the same source pass. It rejects duplicates across
all Tree Keys before emitting partitions, then sorts projections by canonical
Tree Key and Record ID. Full records and payloads stay in the source; there is no all-record
map or directory of every tree in memory. ID sorting and Tree Key sorting share
one memory budget. An ID set that fits is checked directly in its sorted buffer.
Empty input produces an Active empty index without synthetic Tree Manifests.

### Form bounded leaf groups

The forest stage supplies each single-tree constructor with globally unique
Record IDs in canonical order. The constructor consumes that order directly
for grouping, accumulation, sampling, and tie-breaking.

For each Tree Key, recursively divide groups larger than `max_partition_entries`.
Large groups use a bounded deterministic sample: select the lowest seeded hashes
of canonical item IDs with ID tie-breaking. The persisted build algorithm and
sample size determine training; the same sealed input and configuration must
produce identical task outputs. Train two centroids on the sample using the
existing metric rules and balanced two-cluster numeric procedure.

Assign the complete group by the total order
`(distance_to_left - distance_to_right, canonical_id)`, cutting at
`floor(n / 2)`. Groups larger than the reserved row buffer use external
sorting; groups that fit use in-place selection over one loaded buffer. This
guarantees progress for duplicates, equal distances, and skew; nearest-centroid assignment alone does not. Recurse until each group fits
the maximum. Since `2 * minimum <= maximum`, splitting only oversized groups
also respects the minimum for every non-root final group. A tree whose entire
population is below the minimum remains a single leaf root.

Each partition's final centroid is computed from its complete assigned group,
with canonical ID accumulation order and the current metric-specific treatment.
Root inputs follow canonical Record ID order or ascending allocated parent keys. A group that fits the construction row allowance is
loaded once and recursively divided into disjoint slices of that buffer.
Training clones only the bounded selected sample, and releases it and its
centroids before descending. Each terminal slice is sorted by canonical ID
before consuming its rows through the same partition emitter as external
construction. The row allowance already reserves memory for training, terminal
entries and IO; it does not increase with total source size. No descendants of
a resident group are materialized as scratch runs. Both resident and spill paths
preserve sampling, grouping, centroid bytes, partition allocation and algorithm
identity.

For external groups, when the shortest encoded vector row is at least twice
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
repeatedly rewriting a large run. Fan-in is derived from the sort budget.

Construction uses external sorting and repeated data passes. Sort buffers and
training buffers are used sequentially, with a bounded projection reader retained
during training. Global sort scratch and per-tree construction scratch can
overlap, so their disk ceilings add; durable artifact outputs have separate
quotas. One job processes its trees and recursive subgroups sequentially.
Concurrency comes from independent jobs under Runtime and backend admission.

### Assemble the forest

After assigning leaf membership, group the finalized leaf centroids into bounded
parents with the same grouping procedure. Repeat until one root contains the
remaining entries. Each parent references children exactly one level below.
Root Partition Key is 1; non-root keys are allocated deterministically, and the
Tree Manifest high-water mark covers the complete plan.

Internal centroids use the existing internal-training semantics. Incoming Child
Entries are generated only after child centroids are final. The sealed Forest
Artifact orders trees canonically and partitions child-before-parent, with each
root last. Its reader checks allocation, occupancy, tree closure and total leaf
assignments. These structural checks do not replace exact membership validation.

### Establish exact membership and encode serving data

External joins match every source Record to exactly one leaf assignment with
the same Tree Key. A second join matches every non-root partition to exactly one
incoming reference one level above; roots have none. Unique roots, valid local
allocation, decreasing levels and one parent per non-root prove reachability
without retaining the graph in memory.

Encode existing serving values with core codecs, metric normalization, persisted
rotation and Bloom parameters. Leaf Entries use the same absolute RaBitQ7 codes
as online mutations. Child Entries copy finalized full-f32 centroids. Headers
are Ready, with exact counts, cache epoch 1 and transition timestamp 0
(unavailable). Leaf Synopses are reduced from exact fields. No split/merge
transitions or per-record shared-counter updates are emitted.

Record groups stream into the artifact in final key order. Tree rows are sorted
and merged with an ordered Synopsis run. The Serving Artifact contains strictly
ordered unique canonical KV pairs; lifecycle, namespace and build bookkeeping
are excluded. Readers validate index ownership, key order, value codecs and
adapter limits. Source decoding, value decoding and one Synopsis are separately
codec-bounded IO allocations; sort payloads, slots and merge buffers are charged
to explicit memory and shared scratch ceilings.

## Artifact sealing and workspace ownership

Artifacts bind their source, configuration and construction identity. The exact
[file layout](storage.md#11-bulk-build-files) is independent of the backend format.
Writers create new directories and never overwrite an existing attempt. Sealing
flushes and syncs data, renames it and syncs the directory, then syncs and renames
the manifest completion marker and syncs the directory and parent.

Reopen uses the expected manifest held by the owner, not a descriptor read from
the candidate directory. Opening checks metadata/header identity; successful
reader exhaustion checks every frame, canonical body, exact count, EOF and the
whole-file hash. Early reader termination proves only the consumed prefix.
Source verification must finish before dependent output can be sealed. A sealed
file grants no authority until accepted transactionally by its current worker.

Before creating files, a worker holds a shared advisory lock on
`<root>/.ktann-build-lock` and registers namespace-scoped Build Workspace. This
record fixes resource options, backend limits, ownership token, preparation
epoch and accepted artifact identities. Files live under
`<root>/<token>/attempt-<epoch>/`; parent directories are fsynced.

A preparation invocation takes a new epoch. Only the current epoch under a
Building Manifest can accept its output. Recovery reuses accepted predecessors:
with Serving accepted it need not reopen Source or Forest; otherwise it checks
the source against the reserved identity. Interrupted or superseded attempts
remain owned cleanup work. Unknown preparation claims confer no authority;
retries claim a new epoch. Unknown acceptance resolves against the exact
before/after record.

Completion resumes Validating or Validated progress without another preparation
claim. An Active identity only needs any remaining cleanup. Terminal preparation
or validation errors persist in Build Workspace; cancellation and classified
transient errors leave resumable work. Recomputing unaccepted attempts uses a
new directory and cannot overwrite another worker's files.

## Fenced loading

The loader accepts only Serving data matching the reserved source, construction
options and immutable Index Manifest. Its first claim fixes the artifact
identity; retries must use that identity. Build Progress owns the load epoch,
phase and checkpoint. Files remain immutable throughout loading and validation.

Each chunk transaction update-protects the Building Manifest and current epoch,
writes deterministic KV bytes, and advances the checkpoint atomically. It also
checks workspace identity and any scheduler owner. Stale workers cannot commit.
Unknown claims grant no authority; unknown chunk outcomes resolve against the
same epoch's checkpoint before replay. Admission charges data and checkpoint
mutations plus adapter namespace overhead against the smaller of configured and
backend limits. Canonical validated KV bytes pass through without re-encoding.

A record and its related serving entries may span load transactions: Building
data is hidden from serving operations. Atomic visibility is established by
publication after exact validation, not by atomic record-group loading.

File reads run on bounded blocking tasks retaining admission. Recovery replays
the committed prefix without rewriting it, rebuilding whole-file SHA-256 state
in bounded chunks. Its digest must match the checkpoint before skipping that
prefix. Changed bytes fail closed; restoring identical immutable bytes permits
resumption only when the committed-prefix digest matches. Memory is bounded by
the chunk budget, one look-ahead frame, codec buffers and adapter buffering.
Recovery IO/time is proportional to the committed prefix.

Full reader exhaustion and final digest validation are required before atomically
marking Loaded, even when the checkpoint count equals the declared total. Errors
or cancellation may leave a committed prefix. Loaded is still hidden and grants
no publication authority.

## Automatic distributed scheduling

A durable namespace queue identifies each scheduled job and its immutable worker
options. A bounded discovery loop claims eligible unowned or expired jobs,
renews ownership and drives the same completion path. Queue records survive
process loss and index-prefix removal.

Every build mutation update-protects the random owner token, including artifact
acceptance, load chunks, validation pages and activation. Expiry permits takeover;
only a committed token change revokes the old owner. Native IO finishing after
that change cannot authorize stale writes. Unknown queue/claim commits resolve
against the same attempted identity/token. Manual completion rejects queued jobs;
explicit abort remains available.

Renewal and build futures advance concurrently, including while renewal waits
for a backend transaction slot. Terminal persisted worker failures retire from
the queue; other errors reach the scheduler, which retries its classified
transient errors. Corrupt coordination metadata is surfaced. Successful
publication and cleanup retire the entry; busy cleanup retains it for retry.
Interrupted Dropping jobs finish cleanup for their original ID.

Cancellation, deadline or Runtime shutdown stops local tasks without aborting
durable jobs. A dropped scheduler cannot renew indefinitely; another process
can recover after expiry. Polling, active jobs and leases are bounded per
process, not globally. Clock skew may delay takeover or cause redundant work,
but transactionally checked owner tokens preserve write exclusion. Predictable
recovery latency requires reasonably synchronized clocks.

## Exact validation, publication and reclamation

Sealing changes Loaded to Validating under update protection on the same Build
Progress key checked by all load writes. Validation and loading therefore cannot
advance concurrently. Bounded pages scan all backend serving keys and compare
them byte-for-byte with the accepted artifact. Bookkeeping is excluded from
membership ledgers, but malformed build metadata remains corruption.

Each page update-protects the Manifest, workspace, progress and scheduler
identity and commits its proof checkpoint atomically. Restart verifies and
skips the already-proven artifact prefix. Validated requires complete backend
scan, artifact EOF and the accepted digest; counts alone are insufficient.
Missing, extra or mismatched backend entries persist a terminal failure.

Publication checks Validated progress and all remaining ownership fences in one
small transaction, then changes only the Manifest to Active. No prefix swap or
data copy occurs. Unknown outcomes and concurrent publishers resolve against the
original Logical Index ID. Abort and publication serialize on the same Manifest;
an old handle cannot publish a replacement under a reused name.

Build Workspace survives index-prefix removal. Reclamation takes the root's
exclusive advisory lock, removes only the token-owned directory, fsyncs its
parent and then removes the durable record. Native IO retains the shared lock
even when its async waiter is cancelled. All jobs using one root share this
barrier. Caller-owned source, root and coordination file remain untouched.

Completion can report a cleanup error, including Busy, after publication. Retry
under the same identity to finish cleanup. Abort tolerates Busy and leaves the
ownership ledger for later reclamation. Namespace cleanup retains Building jobs
and reclaims orphaned work in bounded pages. Build Descriptor and Validated Build
Progress remain inside an Active index for original-identity status and recovery;
normal index drop removes them. No live coordinator or original worker is needed.

## Resource limits and validation

Construction, sorting, artifacts, load transactions and scan pages have explicit
ceilings. Algorithm budgets are not process-RSS limits: source decoding, output
encoding, adapter caches, allocator reservation and OS residency also consume
memory. Source, sort scratch and accepted artifacts have separate disk limits;
operators provision their aggregate cost and concurrent-job throughput.
Construction remains bounded independently of dataset size, including spill
paths. Native work retains blocking admission until its closure exits.

Tests cover all metric encodings, exact membership and topology joins, empty and
skewed inputs, deterministic spill/resident output, quotas, adapter admission,
stale workers, unknown commits, cancellation, backend omissions/extras, paged
validation, abort races, publication retries and orphan cleanup. See
`tests/bulk_artifacts.rs`, `tests/bulk_serving.rs` and the shared adapter contract
in `tests/support/bulk_load_adapter.rs`. Core implementation owners are
`src/construction.rs`, `src/bulk/` and `src/runtime/bulk_{worker,load,publish,scheduler}.rs`.

Local protocol and adapter tests do not qualify arbitrary multi-host filesystems
or terabyte-scale performance. Benchmark complete receive-to-Ready time and
read cost at matched recall, including CPU, IO, scratch and RSS; an isolated
stage improvement is not a whole-system performance guarantee.
