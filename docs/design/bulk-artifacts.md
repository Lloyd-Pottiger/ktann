# Bounded Offline Artifacts

`bulk/` prepares caller-owned immutable files without modifying a KV Backend.
It does not reserve Logical Index IDs, schedule jobs, load data, or publish an
index. [Bulk Build](bulk-build.md) owns those lifecycle operations. Existing
serving lifecycle and exact-membership contracts are unchanged.

## 1. Pipeline

1. `InputSnapshotWriter` captures full Vector Records, including optional Opaque
   Payloads, in arrival order. Configuration validation and canonical source
   encoding happen during append. Plain snapshots retain duplicate IDs;
   forest preparation rejects them globally, including across Tree Keys.
2. `ForestArtifact::build` sorts IDs for global uniqueness, groups by Tree Key,
   and delegates each tree to the existing deterministic construction algorithm.
   Plans are emitted in
   canonical tree order, child-before-parent with root Partition Key 1 last.
   Optional bounded refinement runs on one tree before emission, reusing
   the ordinary metric-correct assignment algorithm. Zero rounds refreshes leaf
   and bottom-up internal centroids; omission preserves construction output.
   Refinement round controls participate in the forest binding.
   Each round retains up to 32 candidate leaf centroids ordered by exact routing
   distance, then leaf ordinal. Trees with at least 1,024 leaves use an ephemeral
   centroid HNSW (M=8, efConstruction=64, efSearch=64, seed=7), rebuilt each round;
   33 hits are requested to exclude self. Smaller trees use exact symmetric pairs,
   tiled on both leaf axes with cancellation between tiles. Candidate graph capacity
   is independent of the 65,536-entry partition ceiling: graphs are admitted against
   the refinement memory budget and the native u32 ordinal domain. An unsupported
   count or insufficient budget fails with LimitExceeded, without exact fallback. Candidate selection is independent of serving
   graph configuration. Both paths use the same metric-correct move scoring,
   positive-gain ordering and occupancy enforcement. HNSW may change membership.
   Native build and query scratch are reserved against refinement admission before
   planning; cancellation and native failures propagate without algorithm fallback.
   A contiguous fixed-size neighbor list per leaf uses
   512 bytes on 64-bit hosts, included in refinement admission; it is released
   before applying moves. Validated routing vectors and refreshed centroids reuse
   the same distance kernel without repeated component validation. The reader validates
   allocation, occupancy, levels, tree closure, and counts. Construction takes all
   four capacities from the Input Snapshot configuration. Leaf and internal
   grouping use their respective bounds; roots retain the minimum-occupancy
   exception. Changing only internal bounds preserves leaf membership, IDs, and
   centroid bits across resident and spilled paths. Working-memory admission
   accounts for the larger maximum capacity. `ForestOptions` supplies sampling,
   refinement, and resource budgets; standalone `construct_tree` likewise takes
   capacities from `IndexConfig` and sampling/resources from `ConstructionOptions`.
3. `ServingArtifact::build` uses bounded external joins to prove exact source/leaf
   membership and parent/child relationships. Core storage codecs encode records,
   locations, payloads, trees, partition metadata, entries, and synopses. Output
   keys are unique and sorted, and adapter key/value limits are checked. For Local Graph
   Indexes, final canonical internal bodies feed one graph at a time; descriptor
   and chunk keys follow that body's Child Entries. Graph encoding completes
   before sealing and no serving projection changes afterward. It excludes
   the Index Manifest, name directory, and allocator. Its binding includes the
   exact target Index Manifest and source and forest identities. Build budgets and
   adapter limits do not affect content identity.

A sealed forest can reopen with the retained source manifest, configuration,
seed, sample size, and refinement semantics; serving data can reopen with the retained source and forest
manifests and target Index Manifest. Neither operation requires upstream files or
construction budgets. Serving readers apply the consumer's current adapter limits;
exceeding them is `LimitExceeded`, not evidence of corrupt data.

## 2. Files, trust, and failure

The forest binding includes all four capacities. The serving binding includes
the target Manifest and forest identity. Capacity is not part of the source
record schema.

Index storage and bulk artifacts share `storage::FORMAT_VERSION`. Both the
data header and Artifact Manifest begin with `KTANNBF`, the big-endian u16
format version, the kind byte, and the 32-byte configuration binding. Unsupported
versions return `UnsupportedFormat`; type and integrity failures return `Corruption`.
Frames carry bounded lengths and SHA-256 checksums. The 90-byte Artifact Manifest
records the kind, binding, item count, total data length, and whole-file SHA-256.
The caller must persist a trusted expected manifest separately; checksums alone
are not authentication against replacement of both the data and its identity.

Directories are created exclusively. Successful sealing syncs data and writes
and syncs the manifest; failed operations leave caller-owned partial files and
must not be treated as complete artifacts. Reopening verifies identity and
framing; consumers must exhaust readers or call the provided verification
operation to validate the entire stream. Changed, truncated, extended, reordered,
noncanonical, and oversized input is rejected rather than repaired.

Invalid caller records and options are `InvalidArgument`; violated resource
ceilings are `LimitExceeded`; malformed persistent bytes and inconsistent
artifact relationships are `Corruption`. IO errors retain the library error
classification used by the file helpers.

## 3. Resource contract

All operations are synchronous. Async callers must provide blocking execution
and backpressure. Output quotas include framing and the manifest. Sort memory
includes bounded row buffers and merge readers; numeric run IDs bound pending
run metadata. Spilled uniqueness checks consume the final merge directly, avoiding
an intermediate final run; ordinary sorted outputs still materialize their runs.
When the runtime builds from the source snapshot with no Tree Key fields,
it sorts the verified full frames once by Record ID under the serving budgets.
The default tree consumes their original vectors in that order. Serving then
reads the same run for its assignment join, including fields and payloads.
The run remains in the attempt workspace and counts against serving scratch
until the join finishes. That space can overlap per-tree construction scratch.
An attempt resuming an accepted forest rebuilds the serving sort from the
verified source.
When refinement is selected, construction keeps its canonical-ID run of
preprocessed routing vectors until the tree is refined. Its bytes remain charged
to tree scratch throughout construction. After admitting topology, membership,
move and candidate workspace, refinement caches as many routing vectors as fit.
Scoring moves and refreshing centers scan the same Record ID order at every
budget. Only the uncached suffix is reread from scratch. Candidate discovery, move arbitration and f64 centroid arithmetic are shared
with offline refinement.
The working budget must still cover metadata for one tree, including its leaf
centers and graph workspace. It is independent of the process-wide RSS.
Forest sort scratch and per-tree construction scratch may overlap, so their
ceilings add. Codec-bounded decoding buffers and a synopsis are additional to
sort memory; caller-owned input and retained output are outside these budgets.
Multi-tree preparation sorts IDs and tree projections in sequence using the full row budget.
Successful stages remove scratch. Callers own cleanup after failure or cancellation.

## 4. Verification

`tests/bulk_artifacts.rs` covers restart identity, canonical full records, global
duplicate rejection, spills, resource ceilings, deterministic
forest equivalence, and corrupt files. `tests/bulk_serving.rs` checks exact joins,
resource failures, relocation/reopen identity checks, escaped ID ordering, and all metrics with and without Tree
Keys. Its fixture loads into Memory solely to run ordinary index verification
and compare online leaf bytes; this is not a production loading protocol.
