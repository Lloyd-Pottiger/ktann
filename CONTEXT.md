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
The configured Leaf Partition entry count that makes a Ready partition actionable for split without synchronously limiting its size.
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

**Child Entry**:
An internal-partition entry containing one child Partition Key and its centroid projection. The projection is immutable while serving; offline refinement replaces it atomically with the child centroid.
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
A process-shared cache of decoded partition search data.
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
A finite immutable source of complete Records for Bulk Build preparation, identified by a schema-bound, checksummed manifest. It is not a backend transaction snapshot or a serving Logical Index.

**Construction Artifact**:
An immutable, sealed output of construction work, identified independently of its filesystem location. A single-tree topology artifact contains partition plans; it does not by itself establish serving membership or authorize publication.

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

### Bulk Build Reservation

A durable name and Logical Index ID held by a Building Manifest, with a separate
immutable Build Descriptor identifying the caller-owned input snapshot and
construction parameters. Ordinary operations reject Building. Job handles bind
the never-reused ID; abort cannot follow a reused name or remove a published index.
Reservation itself launches no work; `run_worker` prepares/loads and `publish`
validates and activates the index.

**Forest Artifact**: A sealed, caller-owned topology artifact for every nonempty
Tree Key in one finite Input Snapshot. Preparation rejects globally duplicate
Record IDs before constructing trees. It carries no backend loading or publication
authority; exact serving-membership validation is still required.

**Serving Artifact**: A sealed, sorted file of canonical serving KV pairs for a
specific Logical Index identity, produced by exact source/assignment and topology
joins. It preserves source records and payloads and uses the online projection
codecs. It excludes lifecycle bookkeeping and conveys no loading or publication
authority; sealed backend validation is still required.

**Build Load**: The single fenced task that loads one immutable Serving Artifact
into a hidden Building index. Its epoch rejects superseded workers, and its
entry cursor commits atomically with each data chunk. Loaded means the artifact
stream finished; it is not sealed backend validation or permission to publish.


**Build Workspace**: Namespace-scoped durable ownership of one job's token-named
preparation directory, options, epoch, accepted artifacts and terminal failure.
It survives removal of the index prefix until exclusive-lock cleanup completes.

**Build Validation**: A paged exact comparison proof between a core-accepted
Serving Artifact and frozen backend serving KV bytes. The completed proof and
sealed Build Load authorize one atomic transition from Building to Active.
