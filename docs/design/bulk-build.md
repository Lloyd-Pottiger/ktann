# Sharded and Collaborative Bulk Build

Bulk Build constructs one new Logical Index from a finite immutable segmented
`InputSnapshot`. It does not replace an existing Active index. Serving records,
locations, partitions, synopses, and optional partition-local HNSW graphs are
encoded and loaded directly into a hidden `Building` Logical Index across
fenced writer lanes, then verified against frozen backend bytes and 512-bit
homomorphic multiset digests before one atomic `Building -> Active` transition.

Persistent keys and encodings belong to [Storage](storage.md#6-persistent-values).
Architectural rationale is recorded in [ADR 0026](../adr/0026-building-reservations.md),
[ADR 0027](../adr/0027-bulk-workspace-and-publication.md),
[ADR 0028](../adr/0028-automatic-bulk-scheduling.md), and
[ADR 0030](../adr/0030-sharded-collaborative-bulk-build.md).

## 1. Segmented Input Snapshot

1. `InputSnapshotWriter` streams full `Record`s, including optional `OpaquePayload`s,
   into a newly created caller-owned directory. Configuration validation and
   canonical source framing happen during `append`.
2. Segment files (`seg-{i:08x}.bin`, active unsealed segment `data.partial`) roll
   at `16,384` items or `64 MiB` of framed data so downstream workers can sample
   and ingest segments in parallel without a sequential scan bottleneck.
3. Each segment begins with `KTANNBF`, big-endian `storage::FORMAT_VERSION`, kind
   byte `0`, and a 32-byte schema binding over dimension and ordered field schemas.
   Every record frame carries a big-endian `u32` byte length and a 32-byte SHA-256
   frame digest.
4. `seal` syncs the final segment, writes `segments.bin` (kind `1`, binding every
   segment's item count, byte length, and whole-segment SHA-256), and atomically
   writes the 90-byte `manifest.bin` (`InputManifest`). Plain snapshots preserve
   arrival order and retain duplicate Record IDs; Bulk Build ingestion rejects
   duplicates globally across all segments and Tree Keys.
5. `InputSnapshot::open(directory, config, expected_manifest)` verifies schema
   binding, `manifest.bin`, `segments.bin`, and segment lengths before returning
   the handle; exhausting `InputReader` or calling `verify()` checks every frame
   and whole-segment SHA-256. Unsupported format versions return `UnsupportedFormat`;
   altered, truncated, extended, or reordered bytes return `Corruption`.

## 2. Reservation and Worker Options

Call `Runtime::start_bulk_build(name, &input, options)` with
`BulkBuildOptions { sample_items, refinement_rounds }` (`sample_items >= 2`,
`refinement_rounds` in `0..=5`; `None` preserves constructed centroids and
`Some(0)` refreshes centroids without moves). Reservation atomically
allocates a never-reused Logical Index ID, reserves `name`, and persists a
`Building` `IndexManifest` and `BuildDescriptor` (`BuildProgress` is written
when a worker claims an execution epoch).
Repeating an identical request recovers that `BulkBuildJob`; `open_bulk_build(name)`
recovers without source I/O. Ordinary `open_index`, reads, and mutations reject
`Building` indexes with `IndexBuilding`.

`BulkWorkerOptions::new(workspace)` configures execution:

| Setting | Default | Meaning |
| --- | --- | --- |
| `workspace` | caller path | Existing directory for token-owned build state (`{workspace}/{hex_token}`) |
| `parallelism` | `available_parallelism` (`1..=256`, default capped at `32`) | Local CPU and async unit concurrency per worker |
| `distributed_workers` | `1` | Maximum concurrent worker processes collaborating on this job (`1..=256`) |
| `unit_lease` | `30 s` | Renewable KV lease duration per `BuildUnit` (`10 ms ..= 365 days`, persisted losslessly as `(secs, nanos)`) |
| `memory_bytes` | `512 MiB` | Per-worker working-memory budget (`> 0`, bounded per row/leaf at runtime) |
| `graph_memory_bytes` | `256 MiB` | Worker-wide concurrent local HNSW build memory ceiling (`> 0`) |
| `scratch_bytes` | `64 GiB` | Per-build simultaneous workspace disk ceiling (`> 0`) |
| `load` | `5,000` mutations / `4 MiB` | Max KV mutations and logical KV bytes per write transaction |

`job.complete(options)` (and `complete_with_report`) executes preparation/loading
when needed, runs frozen validation and activation, and attempts workspace and
transient metadata reclamation. `job.run_worker(options)`, `job.publish()`, and
`job.cleanup()` expose explicit phase control.

## 3. Pipeline Architecture

### 3.1 Deterministic `ShardPlan`

From a uniform strided full-segment sample across all segments (respecting
`sample_items` and scaling up to `131,072` rows for large shard counts within
`memory_bytes / 2`, selected deterministically by `xxh3_128(id)` min-sketch),
`ShardPlan` computes:

- **Quantile Record-ID splitters (`id_bounds`)** partitioning the ID space into
  non-overlapping buckets (`~16,384` records/bucket, up to `4,096` buckets) for
  global duplicate detection and sorted `RecordLocation` emission.
- **Coarse Shards (`total_shards <= 65,536`, default target `16,384` rows/shard, lowered when `memory_bytes` requires smaller shards)**:
  - *Heavy Tree Keys* (`estimated_rows > target_shard_rows`): split across `k`
    deterministic K-means coarse centers in serving routing space, globally
    bounded so all heavy and light shards fit within `65,536`. When `k >= 32`, a
    two-level coarse super-centroid index (`ceil(sqrt(k))` super-centers with
    3-nearest candidate lists) keeps routing per vector at $O(\sqrt{k} \cdot D)$
    instead of $O(k \cdot D)$.
  - *Light Tree Keys*: hashed by `xxh3_64(tree_key)` into shared light Coarse
    Shards (scaling proportionally up to the remaining `65,536` shard budget) so
    every light tree's records land in one shard without creating one file per
    tenant.

In disk/collaborative mode, the first worker persists `plan-e{epoch}.bin`
atomically under `ScratchQuota` in the token-owned workspace so all workers share
bit-identical bucket and shard boundaries.

### 3.2 Unified Pipeline over `StageStore` (`Memory` vs. `Disk`)

One unified stage pipeline (`src/runtime/bulk_pipeline.rs`) executes across both
ample-memory single-worker builds (`StageStore::Memory`, `in_memory_mode = true`,
`0 B` workspace scratch writes and zero intermediate byte encoding) and
constrained-memory or multi-worker builds (`StageStore::Disk`,
`in_memory_mode = false`) over a typed `StageStore`:

1. **`STAGE_SCATTER (1)`**: Reads immutable `InputSnapshot` segments, validates
   adapter key/value limits, projects `TreeKey`, routes rotated `f32` vectors to
   Coarse Shards, and partitions `(id, SourceFrameRef)` offsets into Record-ID
   buckets, storing indexed `ids-*` and `vec-*` (with per-shard row counts in the
   index header) entries in `StageStore`.
2. **`STAGE_DEDUP (2)`**: Each Record-ID bucket sorts by canonical escaped ID,
   rejects any duplicate across segments or Tree Keys with `RecordAlreadyExists`,
   stores `idref-*` in `StageStore`, and reclaims scatter ID runs.
3. **`STAGE_LEAVES (3)`**: Underfull heavy-tree shards (`0 < rows < min_leaf`)
   remap deterministically into their nearest sibling center (`compute_shard_remap`),
   each Coarse Shard builds recursive multi-center leaves (`build_shard_leaves`)
   and refines light-tree shards locally (`refine_single_light_shard`), stores
   self-contained `shard-e{epoch}-r0-*`, `centers-e{epoch}-r0-*`, and bucketed
   `(id, local_leaf)` indices (`loc-e{epoch}-r0-*`) in `StageStore`, and reclaims
   scatter vector runs.
4. **Optional Halo Refinement (`Propose = 10+3r`, `Extract = 11+3r`, `Apply = 12+3r`)**:
   `Some(0)` refreshes constructed leaf centroids without moves. For `1..=5`
   rounds on heavy trees, each shard scores its bounded halo neighborhood
   (`score_halo_shard`, capped deterministically at
   `max_halo_leaves = min(32,768, (memory_bytes / 2) / per_leaf)`)
   in canonical ascending shard order; `arbitrate_tree_proposals` selects
   positive-gain capacity-safe moves, `STAGE_REFINE_EXTRACT` packs cross-shard
   outbound rows into indexed `out-e{epoch}-r{r}-*` slices so recipients never
   re-read whole donor shards, and `apply_round_shard` consumes round `r` shards,
   applies inbound slices, compacts rows, and recomputes dirty `f64` centroids
   into round `r + 1` before reclaiming round `r`.
5. **Serving KV Emission (`STAGE_EMIT_RECORDS = 30` -> `STAGE_EMIT_TREES = 31` -> `STAGE_EMIT_LEAVES = 32`)**:
   - Deterministic `PartitionKey`s are assigned per tree (`1` for a single-leaf
     root; otherwise `2 ..= 1 + total_leaves` for leaves and `1` for the upper
     root, persisted to `leafkeys-e{epoch}.bin` in `Disk` mode before reclaiming
     early-converged proposal files).
   - `STAGE_EMIT_RECORDS` emits `[index, 0x01, escaped_id]` (`VectorRecord`,
     `RecordLocation`, and `OpaquePayload`) first in strict ascending ID-bucket
     and Record-ID order, reading verified `InputSnapshot` frames at exact frame
     offsets in per-segment order via `SegmentFileReader` and immediately
     reclaiming `idref-*` and `loc-*`.
   - `STAGE_EMIT_TREES` reads non-root leaf centroids directly from `StageStore`
     (`centers-*` on disk, opening only shards with multi-leaf trees in the target
     bucket), groups internal levels bottom-up, builds partition-local HNSW graphs
     within the worker-wide `graph_memory_bytes` ceiling, and streams `[0x03]`
     `TreeManifest` + `[0x04]` internal partition KVs in bounded node batches.
   - `STAGE_EMIT_LEAVES` expands `PartitionSynopsis` inline, emits `[0x04]` leaf
     `Header`, `Synopsis`, `State`, `Centroid`, and `LeafEntry` KVs in ascending
     `(tree_key, leaf_pk)` order, and reclaims final shard state immediately.

When `distributed_workers > 1`, each stage persists `LogicalKey::BuildUnit { index, stage, unit }`
with renewable `unit_lease` ownership (`unit_lease / 3` heartbeats) so peer
processes claim disjoint units concurrently and take over expired units cleanly;
when `distributed_workers == 1`, construction and emission stages run in-process
with zero intermediate `BuildUnit` KV writes.

### 3.3 Transaction Sizing and Pipelined Writer Fencing

`BulkLoadOptions::default()` allows up to `5,000` mutations and `4 MiB` of
logical KV bytes per write transaction, additionally capped at `8 MiB` of
estimated FoundationDB affected bytes (`3 * key_bytes + value_bytes + 64` per
mutation plus conflict ranges) and the adapter's `AdmissionBudget`. Mutations
are pipelined across up to 8 concurrent writer tasks per worker that read-protect
`BuildWorkspace.writer_token` with read-before-write warming so concurrent async
writers overlap CPU encoding with backend commits without conflicting with one
another or serializing on a single progress key.

## 4. Homomorphic Multiset Verification and Publication

Bulk Build enforces three algebraic 512-bit homomorphic multiset equations
(`LtHash`, wrapping `8 x u64` SHA-512 words) before sealing `Loaded`:

1. **Global Record Inventory**: $\sum_{\text{scatter}} \text{LtHash}_{\text{id}} = \sum_{\text{dedup}} \text{LtHash}_{\text{id}}$ and $\text{count}_{\text{scatter}} = \text{count}_{\text{location}}$.
2. **Exact Leaf Membership (`RecordLocation == LeafEntry`)**:
   $\sum_{\text{leaves}} \text{LtHash}(tk, pk, id, \text{fields\_bytes}) = \sum_{\text{locations}} \text{LtHash}(tk, pk, id, \text{fields\_bytes})$.
3. **Exact Tree Closure (`Centroid == ChildEntry`)**:
   $\sum_{\text{non-root definitions}} \text{LtHash}(tk, pk, \text{centroid\_bytes}) = \sum_{\text{parent references}} \text{LtHash}(tk, child\_pk, \text{centroid\_bytes})$.

`seal_loaded` transitions `BuildProgress` to `BuildPhase::Loaded`, storing the
expected serving `total_entries`, `expected_kv_lthash` (sum of `LtHash::from_kv`
over every emitted serving KV pair), and `validation_split_keys` (`0..=255`
splitters defining `1..=256` disjoint slices across `[0x01 RecordGroup]` and all
heavy-tree partition ranges).

`job.publish()` executes frozen backend validation (`src/runtime/bulk_publish.rs`):

1. **Freeze writers**: `begin_validation` atomically transitions
   `BuildProgress` to `Validating` and clears `BuildWorkspace.writer_token`
   (permanently fencing any late worker transaction), then `validate_all_slices`
   initializes `STAGE_VALIDATE (40)` slice `BuildUnit`s within the effective
   transaction budget.
2. **Collaborative slice verification**: Workers claim incomplete
   `STAGE_VALIDATE (40)` slices (taking over expired slice leases cleanly), scan
   disjoint `[split_i, split_{i+1})` ranges spanning the complete
   `index_range(id)` (`[0x00..=0xFF]`) via short-lived `ReadTxn`s released before
   CPU decoding, decode every index-scoped metadata and serving key/value through
   `keys::decode_key` and `ValueCodec::for_index` (verifying canonical format,
   counts, and bounds), and commit serving entry counts and `LtHash::from_kv` on
   each `BuildUnit` without contending on `BuildProgress`. Any malformed, stray,
   missing, extra, or modified byte anywhere in `index_range(id)` fails closed
   with `ErrorKind::Corruption`.
3. **Atomic activation**: Once $\sum_{\text{slices}} \text{count} = \text{total\_entries}$
   and $\sum_{\text{slices}} \text{kv\_lthash} = \text{expected\_kv\_lthash}$,
   one final transaction verifies the `Building` Manifest and `BuildDescriptor`
   identities and flips the Manifest lifecycle to `Active`.

## 5. Automatic and Collaborative Scheduling

After reservation, call `job.schedule(worker_options).await?` and run
`runtime.run_bulk_scheduler(BulkSchedulerOptions::default(), control).await` on
each participating process:

- Scheduled and collaborative peer workers claim renewable
  `LogicalKey::BuildScheduleSlot` leases (`slot = 0..distributed_workers`):
  when `distributed_workers == 1`, one scheduler claims slot `0` and runs the job
  exclusively, taking over if the owner expires; when `distributed_workers > 1`
  (requires a shared filesystem / object-store mount for `InputSnapshot` and
  `workspace`), up to `distributed_workers` scheduler instances claim distinct
  slots in parallel for the **same** `LogicalIndexId`, collaboratively drain all
  `BuildUnit` stages and validation slices, and converge on the single published
  `Active` index before slot `0` completes workspace and transient-metadata
  reclamation and retires `BuildSchedule`.

## 6. Abort and Cleanup

`abort` transitions an unpublished `Building` Manifest to `Dropping` (conflict-aborting
in-flight writers through their update-protected Manifest read), clears the
index prefix and any `BuildSchedule` / `BuildScheduleSlot`s, and attempts workspace reclamation.
Active indexes are never aborted. Post-publication `cleanup` first acquires the
exclusive advisory root lock, verifies no peer worker slots remain active, deletes
`{workspace}/{token}` and `BuildWorkspace`, retires any `BuildSchedule` /
`BuildScheduleSlot` keys, and then reclaims transient `[0x07..0x08)` (`BuildUnit`s)
while preserving `BuildDescriptor` (`0x05`) and compact `BuildProgress` (`0x06`,
with split keys cleared on activation) (`job.cleanup()` /
`Runtime::cleanup_bulk_builds`).

## 7. 10-Billion-Vector Scale Resource and Communication Model

For a `10^10`-record, `768`-dimensional `f32` dataset (`8`-byte IDs, one scalar
filter field, cosine metric, local HNSW enabled):

| Dimension | Bound / Formula | `10B` (`768D`) Value |
| --- | --- | --- |
| **Input Snapshot** | `610,352` segments (`16,384` rows, `~49 MiB` each); `segments.bin` = `29.3 MiB` | `~30.1 TiB` immutable segments; `O(1)`-FD segment reads |
| **`ShardPlan` memory & I/O** | Uniform strided full-segment sample scaling up to `131,072` rows; `65,536` Coarse Shards (`~152,588` rows/shard) with `256` super-centers (`3` candidate lists, `~512` distance evals/vector) + `4,096` ID buckets | `~410 MiB` max sample RAM; `plan.bin` = `201 MiB` (read once per worker) |
| **Per-worker RAM (`W` workers, `P` threads/worker)** | Scatter/Leaf/Emit hold at most `P` Coarse Shards (`152,588 * ~3,200 B ≈ 488 MiB`/shard); Halo refinement caps at `max_halo_leaves = 32,768` (`~26` shards, `~105 MiB` centroids + HNSW) + `1` active shard (`~488 MiB`) | `2–4 GiB` RAM per worker (`O(1)` in dataset size $N$) |
| **Intermediate coarse-shard storage** | Scatter (`vec-*` + `ids-*`): `~31.4 TiB`, consolidated into `65,536` self-contained `shard-*`, `centers-*`, outbound donor slices `out-*`, and final-round compact `(id, local_leaf)` `loc-*` files (`vec-*` immediately deleted after `STAGE_LEAVES`) | Peak simultaneous scratch `~32–34 TiB` on shared object/file store (`~1.05x–1.15x` raw vectors) |
| **Inter-worker coordination & KV metadata** | Workers share zero direct RPCs; coordination uses KV `BuildUnit` claims (`610k` scatter + `4k` dedup + `65.5k` leaf + `65.5k * 3R` refine + `70k` emit + `256` validate $\approx 1.14 \times 10^6$ units for `R = 2`, `< 330 MiB` transient KV metadata reclaimed on cleanup) | `< 350` lease/claim txns/sec across `64` workers (`30 s` lease) |
| **Serving KV write transactions** | `~2.08 \times 10^{10}` serving KV records (`~64 TiB` logical bytes) batched at `4 MiB` / `5,000` mutations across up to 8 pipelined writer lanes per worker | `~1.6 \times 10^7` write txns total (`~250k` txns/worker across `64` workers), zero whole-index serial bottleneck |
| **Upper-tree & HNSW construction** | `~1.0 \times 10^8` leaf centroids (`~307 GiB` across trees in `centers-*`, partitioned across `64` `tree_buckets`); for a single `10B` tree (`level 2`: `~15.6k` internal nodes of `~6,400` children; `level 3`: `1` root of `~15.6k` children), each internal node's local HNSW builds independently in `< 64 MiB` | Parallelized across `tree_buckets` and worker threads without global graph residency |

## 8. Verification

- `tests/bulk_artifacts.rs`: segmented `InputSnapshot` rollover, restart reopen,
  quota enforcement, frame/segment digest tamper rejection, and format version checks.
- `tests/bulk_serving.rs` (`load`, `complete`, `graphs`, `routing`, `scheduler`):
  zero-disk resident mode vs. `512 MiB` constrained disk mode bit-exact layout
  parity, multi-worker collaborative `BuildUnit` execution, crashed-worker lease
  takeover mid-stage, unknown commit outcomes, frozen backend byte tamper
  detection (`extra`, `missing`, `different`), cancellation, abort, and cleanup.
- `tests/support/bulk_load_adapter.rs`: shared RocksDB and FoundationDB durable
  reopen, collaborative completion, `Index::verify`, and workspace cleanup.
