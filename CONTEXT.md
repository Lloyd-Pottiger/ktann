# KTANN

KTANN is a KV-backed approximate nearest-neighbor index whose foreground data changes are atomic while its partition topology is maintained asynchronously.

## Language

**Logical Index**:
An independently configured vector search space whose records are divided among a sharded forest by declared Tree Key fields.
_Avoid_: Table, tree

**Index Name**:
A caller-chosen stable UTF-8 name used to create, find, and idempotently recover one current Logical Index within a Backend Namespace.
_Avoid_: Logical Index ID, Tree Key

**Backend Namespace**:
The caller-selected storage scope within which Index Names are unique and Logical Index IDs are allocated.
For the Memory adapter, a newly created backend supplies this scope; clones
share it, and its contents are ephemeral.
_Avoid_: Logical Index, Tree Key

**Index Manifest**:
The authoritative format, configuration, and lifecycle state of one existing Logical Index.
_Avoid_: Tree root, process configuration

**Persistent Format**:
The canonical encoding contract for Logical Keys, stored values, adapter physical keys, and algorithms that determine persisted bytes. An Index Manifest identifies its format version.

**Sharded Forest**:
A set of disjoint K-means trees in one Logical Index where each Vector Record belongs to exactly one tree.
_Avoid_: Randomized-tree ensemble, replicated forest

**Tree Key**:
The encoded values of declared Vector Record fields that select one tree in a Sharded Forest.
_Avoid_: Partition key, record ID

**Tree Manifest**:
The directory entry and Partition Key allocation state for one Tree Key.
_Avoid_: Index Manifest, root partition

**Tree Fan-out**:
A search over predicate-selected or budget-bounded enumerated Tree Keys governed by one global Search Budget.
_Avoid_: Replicated-tree search, unbounded parallelism

**Partition Key**:
The stable identity of one partition within a Tree Key.
_Avoid_: Tree Key, record ID

**KV Backend**:
A storage implementation that satisfies KTANN's common transactional and ordered-key contract.
_Avoid_: Database driver, storage engine

**Foreground Mutation**:
A user-visible data change whose source data and required leaf-index changes commit atomically.
_Avoid_: Incremental fixup, background update

**Vector Record**:
The KTANN-defined canonical source entity containing a record ID, original vector, declared nullable filter values, and optional opaque payload.
_Avoid_: Business row, index entry

**Record Location**:
The authoritative mapping from a Vector Record to its one current Tree Key and Leaf Partition.
_Avoid_: Approximate route, membership set

**Opaque Payload**:
Optional non-indexed business bytes associated with one Vector Record.
_Avoid_: Filter field, stored vector

**Structure Maintenance**:
An asynchronous topology change such as splitting or merging partitions that preserves searchability throughout every durable intermediate state.
_Avoid_: Foreground mutation, rebuild

**Split Threshold**:
The configured entry count for a partition's level (leaf or internal) that makes a Ready partition actionable for split without synchronously limiting its size.
_Avoid_: Hard partition limit, target partition size

**Balanced Binary K-means Split**:
A deterministic k=2 partition split that repeatedly assigns exactly half the entries to each cluster by their relative distance to the two centroids.
_Avoid_: Nearest-centroid split, random K-means initialization

**Split Target**:
A partition in ReceivingSplit state that is searchable and may receive redirected or migrated entries from its declared source, but cannot itself split or merge until that source is drained.
_Avoid_: Ready partition, staging-only partition

**Leaf Partition**:
A lowest-level partition containing derived index entries for a bounded region of vector space.
_Avoid_: Shard, bucket

**Leaf Entry**:
A searchable derived projection of one Vector Record within a Leaf Partition.
_Avoid_: Vector Record, search result

**Local Graph Index**:
A Logical Index with partition-local HNSW graphs maintained atomically with
internal membership changes. Foreground writes score internal bodies below 1024
nodes exactly and use local HNSW at or above 1024 nodes, retaining the configured
write beam and exact membership/fencing guarantees.
Empty internal bodies have no graph; nonempty bodies in every searchable
topology state have one snapshot-bound graph.

**Child Entry**:
An internal-partition entry containing one child Partition Key and its centroid projection. Replacing the projection updates the child centroid atomically, together with the parent graph when configured.
_Avoid_: Parent pointer, Leaf Entry

**Filter Predicate**:
An arbitrarily nested boolean expression over typed fields that every returned Vector Record must satisfy under SQL WHERE-style NULL semantics.
_Avoid_: Partition pruning hint, post-filter

**Search Budget**:
A hard per-query bound on Tree Key enumeration and ANN work under which the requested result count is an upper limit rather than a fill guarantee.
_Avoid_: Exact top-k guarantee, timeout

**Search Hit**:
A nearest-neighbor result containing only a Record ID and its exact distance.
_Avoid_: Vector Record, payload projection

**Search Outcome**:
A successful search response containing Search Hits, actual budget usage, and which budget dimensions prevented pending work.
_Avoid_: Exact top-k result, timeout error

**Partition Synopsis**:
A conservative summary that propagates possible SQL truth values for a Filter Predicate and can prove that a Leaf Partition has no matches or that all its entries match.
_Avoid_: Exact filter, statistics hint

**Partition Header**:
Small mutable operational metadata associated with one partition.
_Avoid_: Weak count hint, topology state

**Partition Cache**:
A runtime-owned shared cache of decoded partition search data.
_Avoid_: Persistent cache, pinned-root map

**Fixup Worker**:
A worker that advances Structure Maintenance.
_Avoid_: Maintenance owner, leader

**Demand-Driven Maintenance**:
Structure Maintenance rediscovered by relevant index access rather than by durable scheduled work.
_Avoid_: Periodic repair scan, durable job queue

**Index Verification**:
A bounded read-only audit of one Logical Index's persistent invariants that either completes within one backend snapshot or returns no cross-snapshot conclusion.
_Avoid_: Search-time validation, automatic repair

**Input Snapshot**:
A finite, immutable, caller-owned directory of segmented canonical source Records (`seg-{i:08x}.bin`) and an **Input Manifest**, bound to an Index Configuration and per-segment SHA-256 digests for Bulk Build preparation. It is not a backend transaction snapshot or a serving Logical Index.

**Coarse Shard**:
A deterministic build-time partition of routing work produced by `ShardPlan` from a bounded prefix sample (`<= 32,768` records). Heavy Tree Keys split across multiple vector-space coarse centers; light Tree Keys hash into shared coarse shards. Coarse Shards exist only during Bulk Build and are not serving partitions.
_Avoid_: Leaf Partition, Tree Key

**Offline Refinement**:
A caller-exclusive preparation operation on a settled Active Logical Index before serving. It improves leaf assignments and centroids under capacity constraints while preserving Partition Keys, topology and exact membership. Errors can leave a valid, partially refined index.
_Avoid_: Bulk Construction, Building Index, atomic whole-operation refinement

**Fixup Backlog**:
The process-local count of pending plus running actionable Fixups used to observe bounded Structure Maintenance work.
_Avoid_: Partition count, convergence state

## Relationships

- A **Backend Hard Limit** is a stable storage-engine fact; a **Backend Admission Budget** is conservative adapter policy used to bound KTANN work early. Staying below a budget is not proof that FoundationDB's affected-data accounting will accept a transaction.

- A **Logical Index** contains exactly one **Sharded Forest**
- A **Backend Namespace** contains zero or more named **Logical Indexes**
- An **Index Name** identifies at most one current **Logical Index** and may identify a new one after drop completes
- A **Logical Index** has exactly one **Index Manifest** until its drop completes
- A **Sharded Forest** contains zero or more trees identified by distinct **Tree Keys**
- Each existing **Tree Key** has exactly one **Tree Manifest**
- A **Logical Index** exclusively owns **Vector Records** of exactly one vector dimension
- A **Logical Index** uses exactly one **KV Backend**
- A **Foreground Mutation** changes a **Vector Record** and its required **Leaf Partition** entries atomically
- A **Vector Record** has zero or one **Opaque Payload** committed in the same Foreground Mutation
- A **Vector Record** has exactly one **Record Location** and exactly one corresponding **Leaf Entry**
- A **Vector Record** belongs to exactly one tree selected by its **Tree Key**
- A query may use **Tree Fan-out** without a separate tree-count limit, but Tree Key enumeration and tree search both consume one **Search Budget**
- A **Search Outcome** contains zero or more **Search Hits** governed by one **Search Budget**
- A **Leaf Partition** has zero or more **Partition Synopses**
- A **Leaf Partition** contains zero or more **Leaf Entries**
- Every **Leaf Partition** is at tree level 1; partitions above level 1 contain **Child Entries**
- Every non-root partition has exactly one corresponding **Child Entry** in one parent partition
- A **Fixup Worker** may advance **Structure Maintenance** for any **Logical Index**
- **Demand-Driven Maintenance** may leave a cold partition in a searchable intermediate topology state indefinitely
- **Index Verification** may run concurrently with Foreground Mutations and never changes persistent data
- A **Fixup Backlog** is neither a durable maintenance queue nor evidence that topology has converged
- A **Leaf Partition** crossing its **Split Threshold** makes Structure Maintenance actionable without making the partition unsearchable
- A **Search Outcome** may vary with searchable topology shape and Search Budget even when exact membership is unchanged
- A filtered search returns at most its requested number of **Vector Records** within its **Search Budget**
- With Exact routing and a predicate, the leaf beam targets leaves containing at least one exact match per **Tree Key**; empty filtered leaves are replaced through retained upper branches, under one shared **Search Budget**. Unfiltered and explicitly selected HNSW routing retain their level-scaled top-B policy.

## Contract clarifications

- Foreground Mutations change records atomically; Structure Maintenance changes
  topology asynchronously. Online loaders submit ordinary bounded atomic batches.
- The Split Threshold triggers maintenance; it is not a synchronous hard bound
  on partition size. A mutation can retry after a topology conflict.
- Every committed intermediate topology of an Active Logical Index remains searchable.
  After a worker stops, maintenance resumes when a relevant access discovers pending work.
- Online callers own bounded batch concurrency and result handling; there is no
  process-local session scheduler or whole-load atomicity contract.
- A Vector Record is owned and encoded by KTANN, not a host application's
  business row. An Opaque Payload is separate from searchable fields.
- Exact Filter Predicates determine which hits qualify. Conservative Partition
  Synopses only prune work and never replace exact filtering.
- A Sharded Forest contains disjoint records, not replicated recall trees.
- Structure Maintenance requires no global owner or lease.

## Bulk Build coordination

- An **Input Snapshot** is a finite, immutable, caller-owned segmented directory of full Vector Records in receipt order, bound to an Index Configuration via an **Input Manifest** containing segment counts, byte lengths, and SHA-256 digests. The caller retains the trusted expected `InputManifest`.
- Bulk Build emits canonical serving KV records directly into the hidden `Building` Logical Index without intermediate whole-index topology or serving files. In ample memory, coarse shards remain resident with zero workspace scratch writes; under tight memory budgets or multi-worker collaboration, workers exchange bounded indexed coarse-shard files in the Build Workspace.

### Bulk Build Reservation

A durable name and Logical Index ID held by a Building Manifest, with a separate
immutable Build Descriptor identifying the caller-owned Input Snapshot and
construction parameters. Ordinary operations reject Building. Job handles bind
the never-reused ID; abort cannot follow a reused name or remove a published index.
Reservation itself launches no work. `complete` drives sharded construction, loading,
validation, activation, and owned-file reclamation. Scheduled jobs execute the
same phases under renewable single-worker or collaborative multi-slot ownership.

**Build Progress**: The index-scoped authoritative state record (`LogicalKey::BuildProgress`) for one `Building` Logical Index. It tracks the current `BuildPhase` (`Preparing`, `Loading`, `Loaded`, `Validating`), attempt epoch, topology counts, expected 512-bit homomorphic multiset digest (`LtHash`, wrapping `8 x u64` SHA-512 words), and parallel validation split keys.

**Build Unit**: One stage-scoped unit of collaborative work, writer-lane fencing, or validation slice (`LogicalKey::BuildUnit { index, stage, unit }`). Each unit binds the build epoch, a renewable worker owner token and expiry, completion status, item counts, and 512-bit `LtHash` multiset digests for Record ID deduplication, exact `Location == LeafEntry` membership, `ChildEntry == Centroid` closure, and emitted serving KV inventory.

**Build Workspace**: Namespace-scoped durable ownership of one job's token-named
preparation directory, worker options, epoch, and terminal failure.
It survives removal of the index prefix until exclusive-lock cleanup completes.

**Build Schedule**: Namespace-scoped durable request for automatic execution of
one Bulk Build Job. It fixes worker options (including `distributed_workers`) and the original name/ID and coordinates
renewable peer slot leases (`LogicalKey::BuildScheduleSlot` when `distributed_workers > 1`). A committed token change
fences superseded writers; time alone is not write authority. It is separate
from scratch-directory ownership in Build Workspace and retires after workspace reclamation
or a persisted terminal job failure.
