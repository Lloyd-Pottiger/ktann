# ADR 0030: Sharded, direct-loading, and collaborative Bulk Build

Status: Accepted; supersedes the monolithic `ForestArtifact`/`ServingArtifact`
files and single-lane sequential load/validation cursors in
[ADR 0027](0027-bulk-workspace-and-publication.md), and extends
[ADR 0028](0028-automatic-bulk-scheduling.md) with multi-worker collaboration
within a single Bulk Build job.

## Context

The initial Bulk Build implementation (ADRs 0026–0028) serialized every dataset
through two whole-index offline files (`ForestArtifact` and `ServingArtifact`),
held an entire tree's leaf centroids and candidate HNSW graph in memory during
refinement, re-decoded and re-encoded partition bodies when building local HNSW
graphs, loaded serving KVs sequentially through a single `1 MiB` checkpointed
cursor (`~54,000` serial write transactions on Cohere 1M), and re-read the
entire serving artifact sequentially against backend pages before activation.
On Cohere 1M (`768D`), this required `431 s` of optimize time, `17.25 GiB` of
local scratch writes plus a `3.45 GiB` serving artifact, and `> 9.5 GiB` of
admitted refinement memory, while preventing multiple workers from collaborating
on one large build (`10M–10B` vectors).

## Decision

1. **Remove public monolithic offline artifacts (`ForestArtifact`, `ServingArtifact`,
   `load_serving`) and segment `InputSnapshot`.** `InputSnapshotWriter` rolls
   immutable checksummed segment files (`seg-{i:08x}.bin`, `16,384` items or
   `64 MiB`) indexed by `segments.bin` and sealed by the 90-byte `InputManifest`.
   `storage::FORMAT_VERSION` remains `1` (KTANN has no prior stable release).
2. **Partition work with a deterministic `ShardPlan`.** A bounded uniform strided
   full-segment sample (`<= 32,768` items) defines quantile Record-ID buckets
   (`<= 4,096`) and Coarse Shards (`<= 65,536`, with a two-level super-centroid
   index and 3-nearest candidate lists when a heavy tree has `>= 32` coarse
   shards).
3. **Support both zero-disk resident execution and bounded disk/collaborative
   `BuildUnit` execution.** Single-worker builds whose compact routing state
   fits in `memory_bytes` stream segment waves, construct/refine shards, and emit
   serving KVs directly from memory with `0` workspace scratch writes. Tight
   memory budgets and multi-worker builds (`distributed_workers > 1`) persist
   stage-scoped `BuildUnit`s (`LogicalKey::BuildUnit { index, stage, unit }`,
   including `Propose / Extract / Apply` refinement stages) and namespace-scoped
   `BuildScheduleSlot` peer leases with renewable KV ownership (`unit_lease`) and
   exchange atomic consolidated/indexed coarse-shard files in the `BuildWorkspace`
   directory.
4. **Bound refinement neighborhoods (`max_halo_leaves <= 32,768`) and defer
   `RaBitQ7` quantization to leaf emission.** No worker holds whole-tree leaf
   metadata above `32,768` leaves (`~100 MiB`), and scatter/refinement buffers
   avoid carrying `RaBitQ7` codes. Upper-tree internal partitions build local
   HNSW graphs directly from in-memory `&[ChildEntry]` slices.
5. **Load across 16 pipelined writer tasks fenced on `BuildWorkspace.writer_token`
   and validate frozen slices in parallel with 512-bit SHA-512 `LtHash` + full
   `ValueCodec` decoding.** Serving writes batch up to `4 MiB` / `5,000`
   mutations (capped at `8 MiB` estimated FoundationDB affected bytes) across
   16 concurrent writer tasks whose read-conflict-protected `BuildWorkspace`
   check conflicts only when epoch ownership changes or validation freezes
   writers. Three 512-bit homomorphic multiset equations verify global ID
   uniqueness, `RecordLocation == LeafEntry` equality, and
   `Centroid == ChildEntry` closure; `publish` freezes writers by clearing
   `BuildWorkspace.writer_token`, scans `1..=256` keyspace slices
   collaboratively via short-lived `ReadTxn`s, decodes every serving key/value
   through `ValueCodec`, and checks exact count and `expected_kv_lthash` before
   flipping `Building -> Active` and reclaiming transient `[0x07..0x08)` build
   unit metadata.

## Consequences

- Single-machine ample-memory builds eliminate workspace disk I/O and cut KV
  write transactions by `~50x`; `512 MiB` constrained builds complete Cohere 1M
  within bounded RSS and `< 3.6 GiB` scratch writes while producing bit-identical
  layouts when `ShardPlan` selects the same shard count.
- Multiple threads and multiple `Runtime` processes (`distributed_workers > 1`)
  collaborate on one `BulkBuildJob` and recover from worker crashes via `unit_lease`
  takeover without a central coordinator.
- Exact membership, tree closure, fail-closed frozen validation (`Corruption` on
  any missing, extra, or altered backend byte), atomic visibility, and
  namespace-owned workspace reclamation are preserved across Memory, RocksDB,
  and FoundationDB.
