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
2. `PreparedInputWriter` optionally performs bounded ID and Tree Key sorting
   while receiving records. Local duplicates fail during append; duplicates
   across spilled runs fail when forest construction consolidates the ID runs. Its one-use `PreparedInput` avoids
   repeating that work. The original-order source remains the durable authority;
   lost preparation can be recomputed using `ForestArtifact::build`.
3. `ForestArtifact::build` sorts IDs for global uniqueness, groups by Tree Key,
   and delegates each tree to the existing deterministic construction algorithm.
   `build_prepared` consumes receipt-time preparation. Plans are emitted in
   canonical tree order, child-before-parent with root Partition Key 1 last.
   The reader validates allocation, occupancy, levels, tree closure, and counts.
4. `ServingArtifact::build` uses bounded external joins to prove exact source/leaf
   membership and parent/child relationships. Core storage codecs encode records,
   locations, payloads, trees, partition metadata, entries, and synopses. Output
   keys are unique and sorted, and adapter key/value limits are checked. It excludes
   the Index Manifest, name directory, and allocator. Its binding includes the
   exact target Index Manifest and source and forest identities. Build budgets and
   adapter limits do not affect content identity. Forest partition bounds come
   from the source configuration; `ForestOptions` supplies only the training
   sample size and resource budgets. Standalone `construct_tree` also takes an
   `IndexConfig`; `ConstructionOptions` carries only sampling and resources.
   Leaf grouping and leaf-root fit use leaf
   bounds; all higher grouping levels and internal-root fit use internal bounds.
   Non-root occupancy checks select the same pair; roots retain the minimum
   exception. Changing only internal bounds preserves leaf membership, IDs and
   centroid bits, including across resident and spilled construction paths.
   Memory admission reserves for the larger of the two maxima; larger internal
   capacities may therefore require a larger working-memory budget.

A sealed forest can reopen with the retained source manifest, configuration,
seed, and sample size; serving data can reopen with the retained source and forest
manifests and target Index Manifest. Neither operation requires upstream files or
construction budgets. Serving readers apply the consumer's current adapter limits;
exceeding them is `LimitExceeded`, not evidence of corrupt data.

## 2. Files, trust, and failure

The forest binding includes all four capacities. The serving binding includes
the target Manifest and forest identity. Capacity is not part of the source
record schema. During pre-release development these contracts change directly;
this capacity split does not introduce format versions or migration machinery.

A versioned header binds each data file to its kind and configuration identity.
Frames carry bounded lengths and SHA-256 checksums. The 89-byte Artifact Manifest
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
Receipt-time sorters share their declared memory and scratch budget.
Forest sort scratch and per-tree construction scratch may overlap, so their
ceilings add. Codec-bounded decoding buffers and a synopsis are additional to
sort memory; caller-owned input and retained output are outside these budgets.
Ordinary preparation serializes ID and tree sorting to use the full row budget;
receipt preparation shares it between two live sorters. These schedules preserve
different minimum memory budgets and spill behavior. See the option fields for
each stage's precise accounting. Successful stages
remove scratch; callers own cleanup after failure or cancellation.

## 4. Verification

`tests/bulk_artifacts.rs` covers restart identity, canonical full records, receipt
preparation, global duplicate rejection, spills, resource ceilings, deterministic
forest equivalence, and corrupt files. `tests/bulk_serving.rs` checks exact joins,
resource failures, relocation/reopen identity checks, escaped ID ordering, and all metrics with and without Tree
Keys. Its fixture loads into Memory solely to run ordinary index verification
and compare online leaf bytes; this is not a production loading protocol.
