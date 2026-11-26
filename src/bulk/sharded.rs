//! Sharded segment ingestion, in-memory leaf construction, halo refinement,
//! direct serving KV encoding, and bounded disk spill I/O.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use xxhash_rust::xxh3::{Xxh3, xxh3_64, xxh3_128_with_seed};

use crate::api::{
    DataType, Error, ErrorKind, IndexConfig, LogicalIndexId, MAX_RECORD_ID_BYTES, PartitionKey,
    Result, Value,
};
use crate::maintenance::routing::kernel_for;
use crate::refinement::{
    LeafCenter, LeafNeighbors, accept_moves, best_move, candidate_scratch_bytes, leaf_neighbors,
    leaf_neighbors_parallel, mean,
};
use crate::search::numeric::VectorKernel;
use crate::search::rabitq::RaBitQ7;
use crate::storage::backend::HardLimits;
use crate::storage::keys::{self, LogicalKey, MAX_TREE_KEY_BYTES, TreeKey, tree_key_hash};
use crate::storage::values::{
    ChildEntry, IndexManifest, LeafEntry, MAX_VALUE_BYTES, PartitionCentroid, PartitionHeader,
    PartitionState, PartitionSynopsis, PartitionTransition, PersistentValue, RecordLocation,
    TreeManifest, ValueCodec, source,
};

use super::digest::LtHash;
use super::files::{SegmentFileReader, corrupt, io_error, read_at_exact, write_atomic};
use super::graphs::build_internal_graph;
use super::input::InputSnapshot;
use super::plan::{MAX_TOTAL_SHARDS, SAMPLE_SEED, ShardPlan, nearest_center, train_centers};

pub(crate) const MAX_HALO_LEAVES_CAP: usize = 32_768;
const LEAF_KEYS_MAGIC: &[u8; 8] = b"KTANNLK1";
const INDEX_ENTRY_BYTES: u64 = 24; // `(offset: u64, len: u64, xxh3_64: u64)`

/// Runs `f` over `workers`-bounded immutable slice chunks using scoped OS threads.
pub(crate) fn parallel_chunks<T, R, F>(items: &[T], workers: usize, f: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &[T]) -> Result<R> + Sync,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = items.len().div_ceil(workers.max(1)).max(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_idx, chunk) in items.chunks(chunk_size).enumerate() {
            let base = chunk_idx * chunk_size;
            let f_ref = &f;
            handles.push(scope.spawn(move || f_ref(base, chunk)));
        }
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
        }
        Ok(out)
    })
}

/// Runs `f` over `workers`-bounded mutable slice chunks using scoped OS threads.
pub(crate) fn parallel_chunks_mut<T, R, F>(items: &mut [T], workers: usize, f: F) -> Result<Vec<R>>
where
    T: Send,
    R: Send,
    F: Fn(usize, &mut [T]) -> Result<R> + Sync,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = items.len().div_ceil(workers.max(1)).max(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_idx, chunk) in items.chunks_mut(chunk_size).enumerate() {
            let base = chunk_idx * chunk_size;
            let f_ref = &f;
            handles.push(scope.spawn(move || f_ref(base, chunk)));
        }
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
        }
        Ok(out)
    })
}

#[derive(Default)]
struct QuotaState {
    live: u64,
    peak: u64,
    written: u64,
    files: BTreeMap<PathBuf, u64>,
    dir: Option<PathBuf>,
}

/// Tracks simultaneous and cumulative workspace disk bytes against `scratch_bytes`,
/// reserving exact temporary-file bytes before writing and reconciling on `fs::rename`
/// so failed writes never spike sibling threads or unlink already-published canonical paths.
#[derive(Default)]
pub(crate) struct ScratchQuota {
    limit: u64,
    state: Mutex<QuotaState>,
}

impl ScratchQuota {
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            limit,
            state: Mutex::new(QuotaState::default()),
        }
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, QuotaState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Scans `dir` for non-hidden regular files, ignoring `NotFound` races from concurrent renames.
    fn scan_dir_files(dir: &Path) -> Result<BTreeMap<PathBuf, u64>> {
        let mut map = BTreeMap::new();
        let entries = match fs::read_dir(dir) {
            Ok(it) => it,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(map),
            Err(e) => return Err(io_error(e)),
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(io_error(e)),
            };
            if entry
                .file_name()
                .to_str()
                .is_some_and(|s| s.starts_with('.'))
            {
                continue;
            }
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(io_error(e)),
            };
            if meta.is_file() {
                map.insert(entry.path(), meta.len());
            }
        }
        Ok(map)
    }

    /// Seeds live quota accounting from existing current-epoch files in `dir`.
    pub(crate) fn seed_existing(&self, dir: &Path) -> Result<()> {
        let disk_files = Self::scan_dir_files(dir)?;
        let disk_sum = disk_files
            .values()
            .try_fold(0_u64, |acc, &len| acc.checked_add(len))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        if disk_sum > self.limit {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let mut st = self.lock_state();
        st.dir = Some(dir.to_path_buf());
        st.files = disk_files;
        st.live = disk_sum;
        st.peak = st.peak.max(disk_sum);
        Ok(())
    }

    fn refresh_locked(&self, st: &mut QuotaState, fallback_dir: Option<&Path>) -> Result<()> {
        let Some(dir) = st
            .dir
            .clone()
            .or_else(|| fallback_dir.map(Path::to_path_buf))
        else {
            return Ok(());
        };
        let tracked_sum: u64 = st.files.values().copied().sum();
        let in_flight_temp = st.live.saturating_sub(tracked_sum);
        let disk_files = Self::scan_dir_files(&dir)?;
        let disk_sum = disk_files
            .values()
            .try_fold(0_u64, |acc, &len| acc.checked_add(len))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        st.files = disk_files;
        st.live = disk_sum
            .checked_add(in_flight_temp)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        st.peak = st.peak.max(st.live);
        Ok(())
    }

    fn reserve_temp_bytes(&self, delta: u64, dir: Option<&Path>) -> Result<()> {
        if delta == 0 {
            return Ok(());
        }
        let mut st = self.lock_state();
        if st.live.checked_add(delta).is_none_or(|n| n > self.limit) {
            self.refresh_locked(&mut st, dir)?;
        }
        let next = st
            .live
            .checked_add(delta)
            .filter(|&n| n <= self.limit)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        st.live = next;
        st.peak = st.peak.max(next);
        Ok(())
    }

    /// Verifies upfront that `total_new_bytes` fits inside `limit` (accounting for tracked
    /// sizes at `overwritten_paths`), preventing partial multi-file writes.
    pub(crate) fn ensure_headroom(
        &self,
        total_new_bytes: u64,
        overwritten_paths: &[&Path],
    ) -> Result<()> {
        let mut st = self.lock_state();
        let credit = |s: &QuotaState| -> u64 {
            overwritten_paths
                .iter()
                .filter_map(|p| s.files.get(*p).copied())
                .sum()
        };
        if st
            .live
            .saturating_sub(credit(&st))
            .checked_add(total_new_bytes)
            .is_none_or(|n| n > self.limit)
        {
            let dir = overwritten_paths.first().and_then(|p| p.parent());
            self.refresh_locked(&mut st, dir)?;
            if st
                .live
                .saturating_sub(credit(&st))
                .checked_add(total_new_bytes)
                .is_none_or(|n| n > self.limit)
            {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
        }
        Ok(())
    }

    /// Reserves `expected_bytes` upfront, writes `path` atomically, and reconciles live quota.
    pub(crate) fn write_exact(
        &self,
        path: &Path,
        expected_bytes: u64,
        write_fn: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
    ) -> Result<u64> {
        self.reserve_temp_bytes(expected_bytes, path.parent())?;
        match write_atomic(path, write_fn) {
            Ok(actual) => {
                let mut st = self.lock_state();
                let old_bytes = st.files.insert(path.to_path_buf(), actual).unwrap_or(0);
                st.live = st
                    .live
                    .saturating_sub(expected_bytes)
                    .saturating_sub(old_bytes)
                    .saturating_add(actual);
                st.peak = st.peak.max(st.live);
                st.written = st.written.saturating_add(actual);
                Ok(actual)
            }
            Err(err) => {
                let mut st = self.lock_state();
                st.live = st.live.saturating_sub(expected_bytes);
                Err(err)
            }
        }
    }

    /// Releases quota tracked for `path` after removing the file.
    pub(crate) fn release_path(&self, path: &Path, fallback_bytes: u64) {
        let mut st = self.lock_state();
        let bytes = st.files.remove(path).unwrap_or(fallback_bytes);
        st.live = st.live.saturating_sub(bytes);
    }

    pub(crate) fn peak(&self) -> u64 {
        self.lock_state().peak
    }

    pub(crate) fn written(&self) -> u64 {
        self.lock_state().written
    }
}

/// Metadata for one compact routing row; its vector components live in `ShardRows::vectors` or `slab`.
#[derive(Clone)]
pub(crate) struct CompactMeta {
    pub tree_key: TreeKey,
    pub id: Bytes,
    pub fields: Box<[Value]>,
    /// Pre-quantized RaBitQ7 bytes when materialized before leaf emission.
    pub rabitq: Option<Bytes>,
    pub vec_slot: usize,
}

/// Contiguous column-style storage for one Coarse Shard's routing rows.
///
/// In-memory single-worker builds share a single contiguous `slab` across all shards
/// (`vectors` stays empty), avoiding per-shard `Vec` doubling and cross-shard vector copies.
#[derive(Clone)]
pub(crate) struct ShardRows {
    pub dim: usize,
    pub rows: Vec<CompactMeta>,
    pub vectors: Vec<f32>,
    pub slab: Option<Arc<Vec<f32>>>,
}

impl ShardRows {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            rows: Vec::new(),
            vectors: Vec::new(),
            slab: None,
        }
    }

    pub(crate) fn with_slab(dim: usize, slab: Arc<Vec<f32>>) -> Self {
        Self {
            dim,
            rows: Vec::new(),
            vectors: Vec::new(),
            slab: Some(slab),
        }
    }

    pub(crate) fn push(
        &mut self,
        tree_key: TreeKey,
        id: Bytes,
        fields: Box<[Value]>,
        rabitq: Option<Bytes>,
        rot_vec: &[f32],
    ) {
        let vec_slot = self.vectors.len() / self.dim;
        self.vectors.extend_from_slice(rot_vec);
        self.rows.push(CompactMeta {
            tree_key,
            id,
            fields,
            rabitq,
            vec_slot,
        });
    }

    #[must_use]
    pub(crate) fn vector(&self, vec_slot: usize) -> &[f32] {
        let start = vec_slot * self.dim;
        match &self.slab {
            Some(pool) => &pool[start..start + self.dim],
            None => &self.vectors[start..start + self.dim],
        }
    }
}

/// Exact byte location of one canonical record frame inside an immutable `InputSnapshot` segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SourceFrameRef {
    pub seg_idx: u32,
    pub frame_offset: u64,
    pub body_len: u32,
}

/// Verified Record-ID bucket entry pairing a Record ID with its immutable source frame span.
pub(crate) type IdFrameEntry = (Bytes, SourceFrameRef);

/// Output of ingesting one `InputSnapshot` segment range into compact routing shards and ID buckets.
pub(crate) struct SegmentIngest {
    pub records: u64,
    pub id_lthash: LtHash,
    pub ids_by_bucket: Vec<Vec<IdFrameEntry>>,
    pub rows_by_shard: Vec<ShardRows>,
}

/// Output of ingesting one segment directly into a pre-allocated global vector slab.
pub(crate) struct SlabSegmentIngest {
    pub records: u64,
    pub id_lthash: LtHash,
    pub ids_by_bucket: Vec<Vec<IdFrameEntry>>,
    pub metas_by_shard: Vec<Vec<CompactMeta>>,
}

/// Shared single-pass segment ingestion loop used by both resident slab ingestion and disk scatter.
fn ingest_segment_common(
    snapshot: &InputSnapshot,
    seg_idx: usize,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    hard_limits: HardLimits,
    check: &(dyn Fn() -> Result<()> + Sync),
    mut on_row: impl FnMut(usize, usize, TreeKey, Bytes, Box<[Value]>, &[f32]) -> Result<()>,
) -> Result<(u64, LtHash, Vec<Vec<IdFrameEntry>>)> {
    let config = manifest.config();
    let dim = config.dimension();
    let index_id = manifest.logical_index_id();
    let kernel = kernel_for(manifest)?;

    let num_buckets = plan.id_bucket_count() as usize;
    let mut ids_by_bucket = vec![Vec::new(); num_buckets];
    let mut id_lthash = LtHash::ZERO;
    let mut records = 0_u64;
    let seg_u32 = u32::try_from(seg_idx).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let empty_tree_key = if config.tree_key_fields().is_empty() {
        Some(TreeKey::encode(&[], &[])?)
    } else {
        None
    };
    let mut vec_buf = vec![0.0_f32; dim];
    let mut rot_buf = vec![0.0_f32; dim];

    let mut reader = snapshot.encoded_segment(seg_idx)?;
    while let Some(frame) = reader.next_frame_with_span() {
        if records.is_multiple_of(256) {
            check()?;
        }
        let (frame_offset, body_len, frame_bytes) = frame?;
        let (id, fields, payload_len) = source::decode_into_buf(config, frame_bytes, &mut vec_buf)?;

        // Check adapter key/value bounds up front before building routing shards.
        let rec_val_len = match payload_len {
            None => (body_len as usize).saturating_sub(1),
            Some(plen) => (body_len as usize).saturating_sub(5 + plen),
        };
        let worst_key_len = 16 + id.len() * 2;
        if rec_val_len > hard_limits.max_value_bytes
            || (worst_key_len > hard_limits.max_key_bytes
                && keys::record_key(index_id, &id)?.len() > hard_limits.max_key_bytes)
        {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        if let Some(plen) = payload_len
            && (plen.saturating_add(5) > hard_limits.max_value_bytes
                || (worst_key_len > hard_limits.max_key_bytes
                    && keys::payload_key(index_id, &id)?.len() > hard_limits.max_key_bytes))
        {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }

        let tree_key = match &empty_tree_key {
            Some(tk) => tk.clone(),
            None => source::project_tree_key(manifest, &fields)?,
        };

        id_lthash.add_assign(LtHash::from_id(&id));
        let bucket = plan.id_bucket(&id) as usize;
        ids_by_bucket[bucket].push((
            id.clone(),
            SourceFrameRef {
                seg_idx: seg_u32,
                frame_offset,
                body_len,
            },
        ));

        let local_idx =
            usize::try_from(records).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        kernel.preprocess_into(&vec_buf, &mut rot_buf)?;
        let shard = plan.route_shard(&kernel, &tree_key, &rot_buf) as usize;
        on_row(local_idx, shard, tree_key, id, fields, &rot_buf)?;
        records += 1;
    }

    Ok((records, id_lthash, ids_by_bucket))
}

/// Decodes a contiguous range of input segments into one combined scatter run so large builds
/// bound scatter file count (`<= 2,048` runs instead of `610k` per-segment files).
pub(crate) fn ingest_segment_range(
    snapshot: &InputSnapshot,
    seg_range: std::ops::Range<usize>,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    hard_limits: HardLimits,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<SegmentIngest> {
    let dim = manifest.config().dimension();
    let num_shards = plan.total_shards as usize;
    let num_buckets = plan.id_bucket_count() as usize;
    let mut rows_by_shard: Vec<ShardRows> = (0..num_shards).map(|_| ShardRows::new(dim)).collect();
    let mut ids_by_bucket = vec![Vec::new(); num_buckets];
    let mut id_lthash = LtHash::ZERO;
    let mut records = 0_u64;
    for seg_idx in seg_range {
        let (seg_records, seg_lt, seg_ids) = ingest_segment_common(
            snapshot,
            seg_idx,
            manifest,
            plan,
            hard_limits,
            check,
            |_, shard, tree_key, id, fields, rot_vec| {
                rows_by_shard[shard].push(tree_key, id, fields, None, rot_vec);
                Ok(())
            },
        )?;
        records = records
            .checked_add(seg_records)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        id_lthash.add_assign(seg_lt);
        for (dst, src) in ids_by_bucket.iter_mut().zip(seg_ids) {
            dst.extend(src);
        }
    }
    Ok(SegmentIngest {
        records,
        id_lthash,
        ids_by_bucket,
        rows_by_shard,
    })
}

/// Decodes one input segment and writes preprocessed vectors directly in-place into `slab_slice`
/// starting at `base_vec_slot`, eliminating intermediate vector allocations and copies.
#[expect(
    clippy::too_many_arguments,
    reason = "segment ingestion writes directly into its disjoint slab slice with explicit limits and cancellation"
)]
pub(crate) fn ingest_segment_into_slab(
    snapshot: &InputSnapshot,
    seg_idx: usize,
    base_vec_slot: usize,
    slab_slice: &mut [f32],
    manifest: &IndexManifest,
    plan: &ShardPlan,
    hard_limits: HardLimits,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<SlabSegmentIngest> {
    let dim = manifest.config().dimension();
    let num_shards = plan.total_shards as usize;
    let mut metas_by_shard = vec![Vec::new(); num_shards];
    let (records, id_lthash, ids_by_bucket) = ingest_segment_common(
        snapshot,
        seg_idx,
        manifest,
        plan,
        hard_limits,
        check,
        |local_idx, shard, tree_key, id, fields, rot_vec| {
            let dst_start = local_idx.checked_mul(dim).ok_or_else(corrupt)?;
            let dst_end = dst_start.checked_add(dim).ok_or_else(corrupt)?;
            let dst = slab_slice.get_mut(dst_start..dst_end).ok_or_else(corrupt)?;
            dst.copy_from_slice(rot_vec);
            metas_by_shard[shard].push(CompactMeta {
                tree_key,
                id,
                fields,
                rabitq: None,
                vec_slot: base_vec_slot + local_idx,
            });
            Ok(())
        },
    )?;
    Ok(SlabSegmentIngest {
        records,
        id_lthash,
        ids_by_bucket,
        metas_by_shard,
    })
}

/// Checks strict global Record ID uniqueness within one disjoint ID-range bucket.
pub(crate) fn verify_id_bucket(
    mut ids: Vec<IdFrameEntry>,
) -> Result<(u64, LtHash, Vec<IdFrameEntry>)> {
    ids.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    if ids.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(Error::new(ErrorKind::RecordAlreadyExists));
    }
    let mut hash = LtHash::ZERO;
    for (id, _) in &ids {
        hash.add_assign(LtHash::from_id(id));
    }
    Ok((ids.len() as u64, hash, ids))
}

/// Remaps any underfull heavy-tree coarse shards (`0 < count < min_leaf`) into
/// their nearest sibling shard so every non-root shard satisfies `[min_leaf, ..]`.
pub(crate) fn compute_shard_remap(
    kernel: &VectorKernel,
    plan: &ShardPlan,
    counts: &[u64],
    min_leaf: u64,
    max_leaf: u64,
) -> Vec<u32> {
    let mut remap: Vec<u32> = (0..plan.total_shards).collect();
    for heavy in &plan.heavy_trees {
        let start = heavy.shard_start as usize;
        let end = start + heavy.shard_count as usize;
        let mut active: Vec<u64> = counts[start..end].to_vec();
        let tree_total: u64 = active.iter().sum();
        if tree_total == 0 {
            continue;
        }
        if tree_total <= max_leaf {
            // Entire tree fits in a single root leaf: collapse into its first nonempty shard.
            let target = active.iter().position(|&c| c > 0).unwrap_or(0);
            for local in 0..heavy.shard_count as usize {
                remap[start + local] = (start + target) as u32;
            }
            continue;
        }
        while let Some(under) = active.iter().position(|&c| c > 0 && c < min_leaf) {
            let origin = &heavy.centers[under];
            let best_target = (0..active.len())
                .filter(|&other| other != under && active[other] > 0)
                .min_by(|&a, &b| {
                    let da = kernel.routing_distances_validated([&heavy.centers[a]], origin)[0];
                    let db = kernel.routing_distances_validated([&heavy.centers[b]], origin)[0];
                    da.total_cmp(&db).then_with(|| a.cmp(&b))
                });
            let Some(target) = best_target else {
                break;
            };
            active[target] += active[under];
            active[under] = 0;
            let dst_global = (start + target) as u32;
            let src_global = (start + under) as u32;
            for entry in &mut remap[start..end] {
                if *entry == src_global {
                    *entry = dst_global;
                }
            }
        }
    }
    remap
}

/// One constructed Leaf Partition inside a Coarse Shard.
#[derive(Clone)]
pub(crate) struct ShardLeaf {
    pub tree_key: TreeKey,
    pub centroid: PartitionCentroid,
    /// Indices into the owning `ShardState::rows`, strictly sorted by `Record ID`.
    pub members: Vec<usize>,
}

impl LeafCenter for ShardLeaf {
    fn center(&self) -> &[f32] {
        self.centroid.components()
    }
}

/// Complete leaf state of one Coarse Shard.
#[derive(Clone)]
pub(crate) struct ShardState {
    pub rows: ShardRows,
    pub leaves: Vec<ShardLeaf>,
}

trait PartitionItem {
    fn sample_key(&self) -> Bytes;

    fn vec(&self) -> &[f32];

    fn cmp_tie(&self, other: &Self) -> Ordering;
}

#[derive(Clone, Copy)]
struct LeafMember<'a> {
    rows: &'a ShardRows,
    row_idx: usize,
}

impl PartitionItem for LeafMember<'_> {
    fn sample_key(&self) -> Bytes {
        self.rows.rows[self.row_idx].id.clone()
    }

    fn vec(&self) -> &[f32] {
        self.rows.vector(self.rows.rows[self.row_idx].vec_slot)
    }

    fn cmp_tie(&self, other: &Self) -> Ordering {
        self.rows.rows[self.row_idx]
            .id
            .cmp(&other.rows.rows[other.row_idx].id)
    }
}

impl PartitionItem for (PartitionKey, PartitionCentroid) {
    fn sample_key(&self) -> Bytes {
        Bytes::copy_from_slice(&self.0.get().to_be_bytes())
    }

    fn vec(&self) -> &[f32] {
        self.1.components()
    }

    fn cmp_tie(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// Recursive multi-center K-means partitioner shared by initial leaf construction and
/// bottom-up internal partition grouping.
fn partition_recursively<T: PartitionItem>(
    kernel: &VectorKernel,
    sample_items: usize,
    min_cap: u64,
    max_cap: usize,
    items: &mut [(usize, T)],
    emit_group: &mut impl FnMut(&mut [(usize, T)]) -> Result<()>,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    if items.len() <= max_cap {
        items.sort_unstable_by(|a, b| a.1.cmp_tie(&b.1));
        return emit_group(items);
    }

    check()?;
    let cap = sample_items.max(2);
    let mut heap = BinaryHeap::new();
    for (pos, (_, it)) in items.iter().enumerate() {
        let id_bytes = it.sample_key();
        let cand = (xxh3_128_with_seed(&id_bytes, SAMPLE_SEED), id_bytes, pos);
        if heap.len() < cap {
            heap.push(cand);
        } else if heap.peek().is_some_and(|max| &cand < max) {
            heap.pop();
            heap.push(cand);
        }
    }
    let sample: Vec<(Bytes, Box<[f32]>)> = heap
        .into_iter()
        .map(|(_, id_bytes, pos)| (id_bytes, Box::from(items[pos].1.vec())))
        .collect();

    let desired = (items.len() as u64).div_ceil(max_cap as u64).min(8) as usize;
    let k = sample.len().min(desired).max(2);
    let centers = train_centers(kernel, sample, k, 10, check)?;
    let mut counts = vec![0_u64; centers.len()];
    for (label, it) in items.iter_mut() {
        *label = nearest_center(kernel, it.vec(), &centers);
        counts[*label] += 1;
    }

    let sizes = group_sizes(&counts, min_cap);
    let cmp = |a: &(usize, T), b: &(usize, T)| a.0.cmp(&b.0).then_with(|| a.1.cmp_tie(&b.1));
    let mut rest = items;
    for size in sizes {
        let size = size as usize;
        if size < rest.len() {
            rest.select_nth_unstable_by(size, cmp);
        }
        let (child, tail) = rest.split_at_mut(size);
        partition_recursively(
            kernel,
            sample_items,
            min_cap,
            max_cap,
            child,
            emit_group,
            check,
        )?;
        rest = tail;
    }
    Ok(())
}

/// Constructs initial leaves for one Coarse Shard in memory.
pub(crate) fn build_shard_leaves(
    manifest: &IndexManifest,
    sample_items: usize,
    rows: ShardRows,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<ShardState> {
    check()?;
    let config = manifest.config();
    let kernel = VectorKernel::new(
        config.dimension(),
        config.metric(),
        *manifest.rotation_seed(),
    )?;

    let mut by_tree: BTreeMap<Bytes, (TreeKey, Vec<usize>)> = BTreeMap::new();
    for (idx, meta) in rows.rows.iter().enumerate() {
        by_tree
            .entry(Bytes::copy_from_slice(meta.tree_key.as_bytes()))
            .or_insert_with(|| (meta.tree_key.clone(), Vec::new()))
            .1
            .push(idx);
    }

    let min_leaf = u64::from(config.min_leaf_partition_entries());
    let max_leaf = config.max_leaf_partition_entries() as usize;
    let mut leaves = Vec::new();
    {
        let rows_ref = &rows;
        for (_, (tree_key, mut indices)) in by_tree {
            check()?;
            indices.sort_unstable_by(|&a, &b| rows_ref.rows[a].id.cmp(&rows_ref.rows[b].id));
            if indices
                .windows(2)
                .any(|w| rows_ref.rows[w[0]].id == rows_ref.rows[w[1]].id)
            {
                return Err(Error::new(ErrorKind::RecordAlreadyExists));
            }
            let mut tagged: Vec<(usize, LeafMember<'_>)> = indices
                .into_iter()
                .map(|row_idx| {
                    (
                        0,
                        LeafMember {
                            rows: rows_ref,
                            row_idx,
                        },
                    )
                })
                .collect();
            partition_recursively(
                &kernel,
                sample_items,
                min_leaf,
                max_leaf,
                &mut tagged,
                &mut |group| {
                    let members: Vec<usize> = group.iter().map(|&(_, m)| m.row_idx).collect();
                    let centroid = compute_leaf_centroid(&kernel, rows_ref, &members)?;
                    leaves.push(ShardLeaf {
                        tree_key: tree_key.clone(),
                        centroid,
                        members,
                    });
                    Ok(())
                },
                check,
            )?;
        }
    }

    Ok(ShardState { rows, leaves })
}

/// Computes a canonical level-1 `PartitionCentroid` over preprocessed `ShardRows` members,
/// delegating directly to [`crate::refinement::mean`] for bit-exact parity with `Index::refine`.
pub(crate) fn compute_leaf_centroid(
    kernel: &VectorKernel,
    rows: &ShardRows,
    members: &[usize],
) -> Result<PartitionCentroid> {
    mean(
        kernel,
        members
            .iter()
            .map(|&idx| rows.vector(rows.rows[idx].vec_slot)),
        1,
    )
}

fn group_sizes(counts: &[u64], min: u64) -> Vec<u64> {
    let n: u64 = counts.iter().sum();
    let max = n - n.div_ceil(4);
    let mut natural_end = 0;
    let mut previous = 0;
    let mut sizes = Vec::with_capacity(counts.len());
    for (index, count) in counts.iter().enumerate() {
        natural_end += count;
        let remaining = (counts.len() - index - 1) as u64;
        let lower = (previous + min).max(n.saturating_sub(remaining.saturating_mul(max)));
        let upper = previous.saturating_add(max).min(n - remaining * min);
        let end = natural_end.clamp(lower, upper);
        sizes.push(end - previous);
        previous = end;
    }
    sizes
}

/// Lightweight centroid wrapper for candidate graph discovery and scoring.
#[derive(Clone)]
pub(crate) struct CentroidRef {
    pub centroid: PartitionCentroid,
}

impl LeafCenter for CentroidRef {
    fn center(&self) -> &[f32] {
        self.centroid.components()
    }
}

/// One positive-gain move proposal emitted during parallel halo scoring.
#[derive(Clone)]
pub(crate) struct MoveProposal {
    pub gain: f64,
    pub id: Bytes,
    pub src_shard: u32,
    pub src_local_leaf: u32,
    pub src_row_idx: u32,
    pub dst_shard: u32,
    pub dst_local_leaf: u32,
}

/// Per-leaf byte reservation during refinement candidate scoring.
pub(crate) const fn refinement_per_leaf_bytes(dim: usize, rounds: u8) -> usize {
    let base = dim.saturating_mul(4).saturating_add(64);
    if rounds > 0 {
        base.saturating_add(std::mem::size_of::<LeafNeighbors>())
    } else {
        base
    }
}

/// Refines (or refreshes when `rounds == 0`) one light-tree Coarse Shard independently,
/// enforcing `memory_budget_bytes` without indexing into a full-plan shard array.
pub(crate) fn refine_single_light_shard(
    manifest: &IndexManifest,
    shard: &mut ShardState,
    rounds: u8,
    memory_budget_bytes: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    let config = manifest.config();
    let dim = config.dimension();
    let per_leaf = refinement_per_leaf_bytes(dim, rounds);
    if memory_budget_bytes < per_leaf.saturating_mul(2) {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    if shard.leaves.is_empty() {
        return Ok(());
    }
    let kernel = VectorKernel::new(dim, config.metric(), *manifest.rotation_seed())?;
    let min_leaf = config.min_leaf_partition_entries() as usize;
    let max_leaf = config.max_leaf_partition_entries() as usize;
    refine_single_shard(
        &kernel,
        shard,
        rounds,
        min_leaf,
        max_leaf,
        per_leaf,
        memory_budget_bytes,
        check,
    )
}

/// Runs `rounds` of parallel halo-bounded refinement across `shards` in memory
/// (or refreshes centroids and validates memory admission when `rounds == 0`).
pub(crate) fn refine_shards_parallel(
    manifest: &IndexManifest,
    plan: &ShardPlan,
    shards: &mut [ShardState],
    rounds: u8,
    memory_budget_bytes: usize,
    parallelism: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    let config = manifest.config();
    let dim = config.dimension();
    let per_leaf = refinement_per_leaf_bytes(dim, rounds);
    let admitted_leaves = memory_budget_bytes / per_leaf.max(1);
    if admitted_leaves < 2 {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    if shards.is_empty() {
        return Ok(());
    }

    let min_leaf = config.min_leaf_partition_entries() as usize;
    let max_leaf = config.max_leaf_partition_entries() as usize;
    let kernel = VectorKernel::new(dim, config.metric(), *manifest.rotation_seed())?;
    let max_halo_leaves = admitted_leaves.min(MAX_HALO_LEAVES_CAP);

    let light_start = plan.light_shard_start as usize;
    let light_end = light_start + plan.light_shard_count as usize;
    if light_end > light_start {
        refine_light_shards(
            &kernel,
            &mut shards[light_start..light_end],
            rounds,
            min_leaf,
            max_leaf,
            per_leaf,
            memory_budget_bytes,
            parallelism,
            check,
        )?;
    }

    for heavy in &plan.heavy_trees {
        let h_start = heavy.shard_start as usize;
        let h_end = h_start + heavy.shard_count as usize;
        let total_leaves: usize = shards[h_start..h_end].iter().map(|s| s.leaves.len()).sum();
        if rounds == 0 {
            let needed = total_leaves
                .min(max_halo_leaves)
                .checked_mul(per_leaf)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            if needed > memory_budget_bytes {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
            for shard in &mut shards[h_start..h_end] {
                check()?;
                for leaf in &mut shard.leaves {
                    if !leaf.members.is_empty() {
                        leaf.centroid = compute_leaf_centroid(&kernel, &shard.rows, &leaf.members)?;
                    }
                }
            }
            continue;
        }
        if total_leaves < 2 {
            continue;
        }

        let global_scratch =
            candidate_scratch_bytes(total_leaves, rounds as usize).unwrap_or(usize::MAX);
        let can_use_global = total_leaves <= max_halo_leaves
            && total_leaves
                .checked_mul(per_leaf)
                .and_then(|b| b.checked_add(global_scratch))
                .is_some_and(|needed| needed <= memory_budget_bytes);

        for _round in 0..rounds {
            check()?;
            let proposals = if can_use_global {
                propose_global_tree_parallel(
                    &kernel,
                    &shards[h_start..h_end],
                    heavy.shard_start,
                    parallelism,
                    memory_budget_bytes,
                    check,
                )?
            } else {
                propose_halo_tree_parallel(
                    &kernel,
                    plan,
                    shards,
                    heavy.shard_start,
                    heavy.shard_count,
                    max_halo_leaves,
                    per_leaf,
                    memory_budget_bytes,
                    parallelism,
                    check,
                )?
            };

            if proposals.is_empty() {
                break;
            }
            let accepted = apply_tree_proposals(
                &kernel,
                &mut shards[h_start..h_end],
                heavy.shard_start,
                proposals,
                min_leaf,
                max_leaf,
                parallelism,
                check,
            )?;
            if accepted == 0 {
                break;
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "light-shard worker pool threads occupancy, per-leaf byte cost, and memory ceiling"
)]
fn refine_light_shards(
    kernel: &VectorKernel,
    light_shards: &mut [ShardState],
    rounds: u8,
    min_leaf: usize,
    max_leaf: usize,
    per_leaf: usize,
    memory_budget_bytes: usize,
    parallelism: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    let max_workers = (memory_budget_bytes / per_leaf.saturating_mul(2).max(1)).max(1);
    let workers = parallelism
        .max(1)
        .min(light_shards.len().max(1))
        .min(max_workers);
    let per_worker_budget = memory_budget_bytes / workers;
    parallel_chunks_mut(light_shards, workers, |_, chunk| {
        for shard in chunk {
            refine_single_shard(
                kernel,
                shard,
                rounds,
                min_leaf,
                max_leaf,
                per_leaf,
                per_worker_budget,
                check,
            )?;
        }
        Ok(())
    })?;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "single-shard refinement validates per-tree memory admission alongside occupancy bounds"
)]
fn refine_single_shard(
    kernel: &VectorKernel,
    shard: &mut ShardState,
    rounds: u8,
    min_leaf: usize,
    max_leaf: usize,
    per_leaf: usize,
    memory_budget_bytes: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    let mut tree_ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = 0;
    while start < shard.leaves.len() {
        let tk = shard.leaves[start].tree_key.as_bytes();
        let mut end = start + 1;
        while end < shard.leaves.len() && shard.leaves[end].tree_key.as_bytes() == tk {
            end += 1;
        }
        tree_ranges.push(start..end);
        start = end;
    }
    for range in tree_ranges {
        let leaves = &mut shard.leaves[range];
        let count = leaves.len();
        let needed = count
            .checked_mul(per_leaf)
            .and_then(|b| b.checked_add(candidate_scratch_bytes(count, rounds as usize).ok()?))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        if needed > memory_budget_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        if rounds == 0 {
            for leaf in leaves.iter_mut() {
                check()?;
                if !leaf.members.is_empty() {
                    leaf.centroid = compute_leaf_centroid(kernel, &shard.rows, &leaf.members)?;
                }
            }
            continue;
        }
        if count < 2 {
            continue;
        }
        for _ in 0..rounds {
            check()?;
            let neighbors = leaf_neighbors(&check, kernel, leaves)?;
            let mut moves: Vec<(f64, Bytes, usize, usize, usize)> = Vec::new();
            for (src, leaf) in leaves.iter().enumerate() {
                for &row_idx in &leaf.members {
                    let vec = shard.rows.vector(shard.rows.rows[row_idx].vec_slot);
                    if let Some((gain, dst)) = best_move(kernel, leaves, src, vec, &neighbors[src])
                    {
                        moves.push((gain, shard.rows.rows[row_idx].id.clone(), row_idx, src, dst));
                    }
                }
            }
            if moves.is_empty() {
                break;
            }
            moves.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let indexed_moves: Vec<(f64, usize, usize, usize)> = moves
                .iter()
                .enumerate()
                .map(|(rank, m)| (m.0, rank, m.3, m.4))
                .collect();
            let mut counts: Vec<usize> = leaves.iter().map(|l| l.members.len()).collect();
            let mut moved_row_to_dst: BTreeMap<usize, usize> = BTreeMap::new();
            let accepted = accept_moves(
                indexed_moves,
                &mut counts,
                min_leaf,
                max_leaf,
                |rank, _src, dst| {
                    moved_row_to_dst.insert(moves[rank].2, dst);
                },
            );
            if accepted == 0 {
                break;
            }
            let mut next_members = vec![Vec::new(); leaves.len()];
            for (src, leaf) in leaves.iter().enumerate() {
                for &row_idx in &leaf.members {
                    let dst = moved_row_to_dst.get(&row_idx).copied().unwrap_or(src);
                    next_members[dst].push(row_idx);
                }
            }
            for (leaf, mut mem) in leaves.iter_mut().zip(next_members) {
                mem.sort_unstable_by(|&a, &b| shard.rows.rows[a].id.cmp(&shard.rows.rows[b].id));
                leaf.centroid = compute_leaf_centroid(kernel, &shard.rows, &mem)?;
                leaf.members = mem;
            }
        }
    }
    Ok(())
}

fn propose_global_tree_parallel(
    kernel: &VectorKernel,
    tree_shards: &[ShardState],
    shard_start: u32,
    parallelism: usize,
    memory_budget_bytes: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<Vec<MoveProposal>> {
    let mut leaf_offsets = Vec::with_capacity(tree_shards.len() + 1);
    let mut all_centers = Vec::new();
    let mut global_to_shard_leaf = Vec::new();
    leaf_offsets.push(0_usize);
    for (local_s, shard) in tree_shards.iter().enumerate() {
        for (local_l, leaf) in shard.leaves.iter().enumerate() {
            all_centers.push(CentroidRef {
                centroid: leaf.centroid.clone(),
            });
            global_to_shard_leaf.push((shard_start + local_s as u32, local_l as u32));
        }
        leaf_offsets.push(all_centers.len());
    }
    let centers_bytes = all_centers
        .len()
        .saturating_mul(kernel.dimension().saturating_mul(4).saturating_add(64));
    let neighbor_budget = memory_budget_bytes.saturating_sub(centers_bytes);
    let neighbors =
        leaf_neighbors_parallel(&check, kernel, &all_centers, parallelism, neighbor_budget)?;

    let batches = parallel_chunks(tree_shards, parallelism, |base_s, chunk| {
        let mut out = Vec::new();
        for (offset_s, shard) in chunk.iter().enumerate() {
            check()?;
            let local_s = base_s + offset_s;
            let src_shard = shard_start + local_s as u32;
            let g_base = leaf_offsets[local_s];
            for (local_l, leaf) in shard.leaves.iter().enumerate() {
                let g_src = g_base + local_l;
                for &row_idx in &leaf.members {
                    let vec = shard.rows.vector(shard.rows.rows[row_idx].vec_slot);
                    if let Some((gain, g_dst)) =
                        best_move(kernel, &all_centers, g_src, vec, &neighbors[g_src])
                    {
                        let (dst_shard, dst_local_leaf) = global_to_shard_leaf[g_dst];
                        out.push(MoveProposal {
                            gain,
                            id: shard.rows.rows[row_idx].id.clone(),
                            src_shard,
                            src_local_leaf: local_l as u32,
                            src_row_idx: row_idx as u32,
                            dst_shard,
                            dst_local_leaf,
                        });
                    }
                }
            }
        }
        Ok(out)
    })?;
    Ok(batches.into_iter().flatten().collect())
}

/// Shared single-shard halo neighborhood admission, canonical ordering, and move scoring
/// used identically by resident `propose_halo_tree_parallel` and disk `propose_disk_shard`.
#[expect(
    clippy::too_many_arguments,
    reason = "shared halo scorer binds kernel, plan, source shard state, leaf/memory bounds, cancellation, and neighbor loader"
)]
pub(crate) fn score_halo_shard(
    kernel: &VectorKernel,
    plan: &ShardPlan,
    src_shard: u32,
    src_state: &ShardState,
    max_halo_leaves: usize,
    per_leaf: usize,
    per_worker_budget: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
    mut load_nbr_centers: impl FnMut(u32) -> Result<Vec<PartitionCentroid>>,
) -> Result<Vec<MoveProposal>> {
    check()?;
    if src_state.leaves.is_empty() {
        return Ok(Vec::new());
    }
    let src_leaves = src_state.leaves.len();
    let src_needed = src_leaves
        .checked_mul(per_leaf)
        .and_then(|b| b.checked_add(candidate_scratch_bytes(src_leaves, 1).ok()?))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    if src_needed > per_worker_budget {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }

    let worker_halo_cap = (per_worker_budget / per_leaf.max(1)).min(max_halo_leaves);
    let ranked_nbrs = plan.halo_shards(kernel, src_shard);
    let src_centers: Vec<PartitionCentroid> = src_state
        .leaves
        .iter()
        .map(|l| l.centroid.clone())
        .collect();
    let mut chosen_leaves = src_leaves;
    let mut chosen_shards = vec![(src_shard, src_centers)];

    for &nbr_shard in ranked_nbrs.iter().skip(1) {
        if chosen_leaves >= worker_halo_cap {
            break;
        }
        check()?;
        let nbr_centers = load_nbr_centers(nbr_shard)?;
        if nbr_centers.is_empty() {
            continue;
        }
        let next_leaves = chosen_leaves + nbr_centers.len();
        if next_leaves > worker_halo_cap {
            break;
        }
        let next_bytes = next_leaves
            .checked_mul(per_leaf)
            .and_then(|b| b.checked_add(candidate_scratch_bytes(next_leaves, 1).ok()?));
        if next_bytes.is_none_or(|b| b > per_worker_budget) {
            break;
        }
        chosen_shards.push((nbr_shard, nbr_centers));
        chosen_leaves = next_leaves;
    }

    // Canonicalize selected shards in ascending `shard_id` order so candidate tie-breaking
    // and HNSW insertion order are identical across resident and disk modes.
    chosen_shards.sort_unstable_by_key(|(s, _)| *s);

    let mut halo_centers = Vec::with_capacity(chosen_leaves);
    let mut halo_map = Vec::with_capacity(chosen_leaves);
    let mut src_leaf_base = 0_usize;
    for (s_id, centers) in chosen_shards {
        if s_id == src_shard {
            src_leaf_base = halo_centers.len();
        }
        for (ll, centroid) in centers.into_iter().enumerate() {
            halo_centers.push(CentroidRef { centroid });
            halo_map.push((s_id, ll as u32));
        }
    }
    if halo_centers.len() < 2 {
        return Ok(Vec::new());
    }

    let neighbors = leaf_neighbors(&check, kernel, &halo_centers)?;
    let mut out = Vec::new();
    for (local_l, leaf) in src_state.leaves.iter().enumerate() {
        check()?;
        let h_src = src_leaf_base + local_l;
        for &row_idx in &leaf.members {
            let vec = src_state.rows.vector(src_state.rows.rows[row_idx].vec_slot);
            if let Some((gain, halo_dst)) =
                best_move(kernel, &halo_centers, h_src, vec, &neighbors[h_src])
            {
                let (dst_shard, dst_local_leaf) = halo_map[halo_dst];
                out.push(MoveProposal {
                    gain,
                    id: src_state.rows.rows[row_idx].id.clone(),
                    src_shard,
                    src_local_leaf: local_l as u32,
                    src_row_idx: row_idx as u32,
                    dst_shard,
                    dst_local_leaf,
                });
            }
        }
    }
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "halo proposal takes immutable shard slice, tree range, per-worker memory ceiling, and thread bound"
)]
fn propose_halo_tree_parallel(
    kernel: &VectorKernel,
    plan: &ShardPlan,
    all_shards: &[ShardState],
    shard_start: u32,
    shard_count: u32,
    max_halo_leaves: usize,
    per_leaf: usize,
    memory_budget_bytes: usize,
    parallelism: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<Vec<MoveProposal>> {
    let shard_ids: Vec<u32> = (shard_start..shard_start + shard_count).collect();
    let max_src_leaves = shard_ids
        .iter()
        .map(|&s| all_shards[s as usize].leaves.len())
        .max()
        .unwrap_or(2)
        .max(2);
    let min_worker_bytes = max_src_leaves.saturating_mul(per_leaf).max(1);
    let max_workers = (memory_budget_bytes / min_worker_bytes).max(1);
    let workers = parallelism
        .max(1)
        .min(shard_ids.len().max(1))
        .min(max_workers);
    let per_worker_budget = memory_budget_bytes / workers;
    if per_worker_budget < per_leaf.saturating_mul(2) {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let batches = parallel_chunks(&shard_ids, workers, |_, chunk| {
        let mut out = Vec::new();
        for &src_shard in chunk {
            out.extend(score_halo_shard(
                kernel,
                plan,
                src_shard,
                &all_shards[src_shard as usize],
                max_halo_leaves,
                per_leaf,
                per_worker_budget,
                check,
                |nbr_shard| {
                    Ok(all_shards[nbr_shard as usize]
                        .leaves
                        .iter()
                        .map(|l| l.centroid.clone())
                        .collect())
                },
            )?);
        }
        Ok(out)
    })?;
    Ok(batches.into_iter().flatten().collect())
}

#[expect(
    clippy::too_many_arguments,
    reason = "move application coordinates counts, cross-shard row transfers, and parallel centroid updates"
)]
fn apply_tree_proposals(
    kernel: &VectorKernel,
    tree_shards: &mut [ShardState],
    shard_start: u32,
    mut proposals: Vec<MoveProposal>,
    min_leaf: usize,
    max_leaf: usize,
    parallelism: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<usize> {
    check()?;
    proposals.sort_unstable_by(|a, b| b.gain.total_cmp(&a.gain).then_with(|| a.id.cmp(&b.id)));

    let mut leaf_offsets = Vec::with_capacity(tree_shards.len() + 1);
    let mut counts = Vec::new();
    leaf_offsets.push(0_usize);
    for s in tree_shards.iter() {
        for l in &s.leaves {
            counts.push(l.members.len());
        }
        leaf_offsets.push(counts.len());
    }

    let raw_moves: Vec<(f64, usize, usize, usize)> = proposals
        .iter()
        .enumerate()
        .map(|(rank, p)| {
            let g_src =
                leaf_offsets[(p.src_shard - shard_start) as usize] + p.src_local_leaf as usize;
            let g_dst =
                leaf_offsets[(p.dst_shard - shard_start) as usize] + p.dst_local_leaf as usize;
            (p.gain, rank, g_src, g_dst)
        })
        .collect();

    let mut chosen_ranks = Vec::new();
    let accepted = accept_moves(
        raw_moves,
        &mut counts,
        min_leaf,
        max_leaf,
        |rank, _src, _dst| {
            chosen_ranks.push(rank);
        },
    );
    if accepted == 0 {
        return Ok(0);
    }

    let mut outbound: Vec<BTreeMap<usize, (u32, u32)>> = vec![BTreeMap::new(); tree_shards.len()];
    let mut inbound: Vec<Vec<(u32, CompactMeta, Vec<f32>)>> = vec![Vec::new(); tree_shards.len()];

    for rank in chosen_ranks {
        let p = &proposals[rank];
        let src_s = (p.src_shard - shard_start) as usize;
        let dst_s = (p.dst_shard - shard_start) as usize;
        outbound[src_s].insert(p.src_row_idx as usize, (p.dst_shard, p.dst_local_leaf));
        if src_s != dst_s {
            let src_rows = &tree_shards[src_s].rows;
            let meta = src_rows.rows[p.src_row_idx as usize].clone();
            let vec = if src_rows.slab.is_some() {
                Vec::new()
            } else {
                src_rows.vector(meta.vec_slot).to_vec()
            };
            inbound[dst_s].push((p.dst_local_leaf, meta, vec));
        }
    }

    let outbound_ref = &outbound;
    let inbound_ref = &inbound;
    parallel_chunks_mut(tree_shards, parallelism, |base_s, chunk| {
        for (offset_s, shard) in chunk.iter_mut().enumerate() {
            check()?;
            let local_s = base_s + offset_s;
            let moved_out = &outbound_ref[local_s];
            let moved_in = &inbound_ref[local_s];
            if moved_out.is_empty() && moved_in.is_empty() {
                continue;
            }
            let this_shard_id = shard_start + local_s as u32;
            let mut next_members = vec![Vec::new(); shard.leaves.len()];
            let mut dirty = vec![false; shard.leaves.len()];
            for (local_l, leaf) in shard.leaves.iter().enumerate() {
                for &row_idx in &leaf.members {
                    if let Some(&(dst_s, dst_l)) = moved_out.get(&row_idx) {
                        dirty[local_l] = true;
                        if dst_s == this_shard_id {
                            next_members[dst_l as usize].push(row_idx);
                            dirty[dst_l as usize] = true;
                        }
                    } else {
                        next_members[local_l].push(row_idx);
                    }
                }
            }

            let active_total = next_members.iter().map(Vec::len).sum::<usize>() + moved_in.len();
            let dim = shard.rows.dim;
            let shared_slab = shard.rows.slab.clone();
            let mut compacted = ShardRows {
                dim,
                rows: Vec::with_capacity(active_total),
                vectors: if shared_slab.is_some() {
                    Vec::new()
                } else {
                    Vec::with_capacity(active_total * dim)
                },
                slab: shared_slab,
            };
            let old_base = shard.rows.rows.len();
            for (in_idx, (dst_l, _, _)) in moved_in.iter().enumerate() {
                next_members[*dst_l as usize].push(old_base + in_idx);
                dirty[*dst_l as usize] = true;
            }

            for ((leaf, mut mem), changed) in shard.leaves.iter_mut().zip(next_members).zip(dirty) {
                if changed {
                    mem.sort_unstable_by(|&a, &b| {
                        let id_a = if a < old_base {
                            &shard.rows.rows[a].id
                        } else {
                            &moved_in[a - old_base].1.id
                        };
                        let id_b = if b < old_base {
                            &shard.rows.rows[b].id
                        } else {
                            &moved_in[b - old_base].1.id
                        };
                        id_a.cmp(id_b)
                    });
                }
                let mut remapped = Vec::with_capacity(mem.len());
                for old_idx in mem {
                    let new_idx = compacted.rows.len();
                    let (r, v) = if old_idx < old_base {
                        let r = &shard.rows.rows[old_idx];
                        let v = if compacted.slab.is_some() {
                            &[][..]
                        } else {
                            shard.rows.vector(r.vec_slot)
                        };
                        (r, v)
                    } else {
                        let (_, r, v) = &moved_in[old_idx - old_base];
                        (r, v.as_slice())
                    };
                    if compacted.slab.is_some() {
                        compacted.rows.push(r.clone());
                    } else {
                        compacted.push(
                            r.tree_key.clone(),
                            r.id.clone(),
                            r.fields.clone(),
                            r.rabitq.clone(),
                            v,
                        );
                    }
                    remapped.push(new_idx);
                }
                if changed {
                    leaf.centroid = compute_leaf_centroid(kernel, &compacted, &remapped)?;
                }
                leaf.members = remapped;
            }
            shard.rows = compacted;
        }
        Ok(())
    })?;

    Ok(accepted)
}

/// Global leaf PartitionKey assignment for one shard's leaves.
#[derive(Clone, Copy)]
pub(crate) struct AssignedLeafKey {
    pub partition_key: PartitionKey,
    pub is_single_leaf_root: bool,
}

/// Per-tree leaf partition counts keyed by encoded Tree Key bytes.
pub(crate) type TreeLeafTotals = BTreeMap<Bytes, (TreeKey, u64)>;

/// Encoded RecordLocation KV batch together with its KV and membership `LtHash` digests.
pub(crate) type EncodedLocationBatch = (Vec<(Bytes, Bytes)>, LtHash, LtHash);

/// Bucketed `(id, tree_key, leaf_pk)` tuples partitioned by `ShardPlan::id_bucket`.
pub(crate) type BucketLocations = Vec<Vec<(Bytes, TreeKey, PartitionKey)>>;

type InternalNodePlan = (
    PartitionKey,
    u32,
    PartitionCentroid,
    Vec<(PartitionKey, PartitionCentroid)>,
);

type EncodedNodeBatch = (Vec<(Bytes, Bytes)>, LtHash, LtHash, LtHash);

/// Assigns deterministic per-tree PartitionKeys across per-shard `TreeKey` iterators (`1` if
/// single-leaf root, else `2 ..= 1 + total_leaves` per tree).
pub(crate) fn assign_leaf_partition_keys<'a, S, I>(
    shards: &'a [S],
    leaves_of: impl Fn(&'a S) -> I,
) -> Result<(Vec<Vec<AssignedLeafKey>>, TreeLeafTotals)>
where
    I: IntoIterator<Item = &'a TreeKey>,
{
    let mut tree_leaf_totals: TreeLeafTotals = BTreeMap::new();
    for shard in shards {
        for tk in leaves_of(shard) {
            let entry = tree_leaf_totals
                .entry(Bytes::copy_from_slice(tk.as_bytes()))
                .or_insert_with(|| (tk.clone(), 0));
            entry.1 = entry
                .1
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        }
    }
    let mut tree_cursors: BTreeMap<Bytes, u64> = tree_leaf_totals
        .iter()
        .map(|(k, (_, total))| (k.clone(), if *total == 1 { 1 } else { 2 }))
        .collect();

    let mut assigned = Vec::with_capacity(shards.len());
    for shard in shards {
        let mut shard_keys = Vec::new();
        for tk in leaves_of(shard) {
            let raw_tk = tk.as_bytes();
            let (_, total) = &tree_leaf_totals[raw_tk];
            let cursor = tree_cursors.get_mut(raw_tk).expect("cursor");
            let pk = PartitionKey::new(*cursor)?;
            *cursor = cursor
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            shard_keys.push(AssignedLeafKey {
                partition_key: pk,
                is_single_leaf_root: *total == 1,
            });
        }
        assigned.push(shard_keys);
    }
    Ok((assigned, tree_leaf_totals))
}

/// Encoded outputs from one Coarse Shard's final leaves.
pub(crate) struct EncodedLeafShard {
    pub kvs: Vec<(Bytes, Bytes)>,
    pub kv_lthash: LtHash,
    pub membership_lthash: LtHash,
    pub edge_def_lthash: LtHash,
    pub locations_by_bucket: Vec<Vec<(Bytes, TreeKey, PartitionKey)>>,
    pub child_centroids: Vec<(TreeKey, PartitionKey, PartitionCentroid)>,
}

/// Batch-quantizes all active shard rows into a single contiguous `RaBitQ7` slab,
/// immediately releases the `f32` vector slab (`munmap`), and extracts bucketed
/// `RecordLocation` tuples so `[0x01 RecordGroup]` can emit before `[0x03..0x04 Partitions]`.
pub(crate) fn quantize_shards_and_extract_locations(
    manifest: &IndexManifest,
    plan: &ShardPlan,
    shards: &mut [ShardState],
    assigned: &[Vec<AssignedLeafKey>],
    parallelism: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<BucketLocations> {
    let code_len = RaBitQ7::encoded_len(manifest.config().dimension())?;
    let mut shard_row_offsets = Vec::with_capacity(shards.len() + 1);
    let mut total_rows = 0_usize;
    shard_row_offsets.push(0_usize);
    for s in shards.iter() {
        total_rows = total_rows
            .checked_add(s.rows.rows.len())
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        shard_row_offsets.push(total_rows);
    }
    let total_bytes = total_rows
        .checked_mul(code_len)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let mut rabitq_buf = vec![0_u8; total_bytes];

    let chunk_size = shards.len().div_ceil(parallelism.max(1)).max(1);
    let mut work = Vec::new();
    let mut rest_buf: &mut [u8] = rabitq_buf.as_mut_slice();
    for (chunk_idx, shard_chunk) in shards.chunks(chunk_size).enumerate() {
        let start_s = chunk_idx * chunk_size;
        let end_s = start_s + shard_chunk.len();
        let bytes_in_chunk = (shard_row_offsets[end_s] - shard_row_offsets[start_s]) * code_len;
        let (head, tail) = rest_buf.split_at_mut(bytes_in_chunk);
        rest_buf = tail;
        work.push((shard_chunk, head));
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(work.len());
        for (shard_chunk, dst_bytes) in work {
            handles.push(scope.spawn(move || -> Result<()> {
                let mut offset = 0_usize;
                for shard in shard_chunk {
                    check()?;
                    for (idx, row) in shard.rows.rows.iter().enumerate() {
                        if idx.is_multiple_of(512) {
                            check()?;
                        }
                        let q = RaBitQ7::quantize(shard.rows.vector(row.vec_slot))?;
                        dst_bytes[offset..offset + code_len].copy_from_slice(&q);
                        offset += code_len;
                    }
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().map_err(|_| Error::new(ErrorKind::Other))??;
        }
        Ok::<_, Error>(())
    })?;

    let rabitq_slab = Bytes::from(rabitq_buf);
    let num_buckets = plan.id_bucket_count() as usize;
    let mut locations_by_bucket: BucketLocations = vec![Vec::new(); num_buckets];
    let mut byte_cursor = 0_usize;

    for (shard, keys) in shards.iter_mut().zip(assigned) {
        shard.rows.slab = None;
        shard.rows.vectors = Vec::new();
        for row in &mut shard.rows.rows {
            row.rabitq = Some(rabitq_slab.slice(byte_cursor..byte_cursor + code_len));
            byte_cursor += code_len;
        }
        for (leaf, ak) in shard.leaves.iter().zip(keys) {
            let pk = ak.partition_key;
            let tk = &leaf.tree_key;
            for &row_idx in &leaf.members {
                let id = &shard.rows.rows[row_idx].id;
                let bucket = plan.id_bucket(id) as usize;
                locations_by_bucket[bucket].push((id.clone(), tk.clone(), pk));
            }
        }
    }

    Ok(locations_by_bucket)
}

/// Encodes all Leaf Partitions of one Coarse Shard into canonical KV bytes, expanding
/// `PartitionSynopsis` inline and slicing `fields_bytes` directly from encoded `LeafEntry` bytes.
pub(crate) fn encode_leaf_shard(
    manifest: &IndexManifest,
    plan: &ShardPlan,
    shard: &ShardState,
    assigned: &[AssignedLeafKey],
    hard_limits: HardLimits,
    collect_locations: bool,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<EncodedLeafShard> {
    let index_id = manifest.logical_index_id();
    let codec = ValueCodec::for_index(manifest);
    let num_buckets = if collect_locations {
        plan.id_bucket_count() as usize
    } else {
        0
    };
    let mut locations_by_bucket = vec![Vec::new(); num_buckets];
    let mut child_centroids = Vec::new();
    let mut kvs = Vec::new();
    let mut kv_lthash = LtHash::ZERO;
    let mut membership_lthash = LtHash::ZERO;
    let mut edge_def_lthash = LtHash::ZERO;

    for (leaf, &ak) in shard.leaves.iter().zip(assigned) {
        check()?;
        let pk = ak.partition_key;
        let tk = &leaf.tree_key;
        let count =
            u32::try_from(leaf.members.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;

        let mut synopsis = PartitionSynopsis::empty(manifest);
        let mut leaf_entry_kvs = Vec::with_capacity(leaf.members.len());
        for &row_idx in &leaf.members {
            let row = &shard.rows.rows[row_idx];
            synopsis.expand(manifest, &row.fields)?;
            let rabitq = match &row.rabitq {
                Some(bytes) => bytes.clone(),
                None => RaBitQ7::quantize(shard.rows.vector(row.vec_slot))?,
            };
            let rabitq_len = rabitq.len();
            let kv = encode_kv(
                codec,
                hard_limits,
                LogicalKey::LeafEntry {
                    index: index_id,
                    tree_key: tk.clone(),
                    partition: pk,
                    id: row.id.clone(),
                },
                PersistentValue::LeafEntry(LeafEntry::new(
                    row.id.clone(),
                    row.fields.clone(),
                    rabitq,
                )),
            )?;
            // `LeafEntry` wire layout: `[tag: 1B][u16 id_len: 2B][id][fields_bytes][u32 rq_len: 4B][rq]`.
            let fields_start = 3 + row.id.len();
            let fields_end = kv.1.len().checked_sub(4 + rabitq_len).ok_or_else(corrupt)?;
            let fields_bytes = kv.1.get(fields_start..fields_end).ok_or_else(corrupt)?;
            membership_lthash.add_assign(LtHash::from_membership(
                tk.as_bytes(),
                pk,
                &row.id,
                fields_bytes,
            ));
            leaf_entry_kvs.push(kv);
            if collect_locations {
                let bucket = plan.id_bucket(&row.id) as usize;
                locations_by_bucket[bucket].push((row.id.clone(), tk.clone(), pk));
            }
        }

        let header_kv = encode_kv(
            codec,
            hard_limits,
            LogicalKey::Header {
                index: index_id,
                tree_key: tk.clone(),
                partition: pk,
            },
            PersistentValue::PartitionHeader(PartitionHeader::new(
                1,
                count,
                1,
                PartitionState::Ready,
            )?),
        )?;
        let synopsis_kv = encode_kv(
            codec,
            hard_limits,
            LogicalKey::Synopsis {
                index: index_id,
                tree_key: tk.clone(),
                partition: pk,
            },
            PersistentValue::PartitionSynopsis(synopsis),
        )?;
        let state_kv = encode_kv(
            codec,
            hard_limits,
            LogicalKey::State {
                index: index_id,
                tree_key: tk.clone(),
                partition: pk,
            },
            PersistentValue::PartitionState(PartitionTransition::Ready {
                started_at_unix_millis: 0,
            }),
        )?;

        for kv in [header_kv, synopsis_kv, state_kv] {
            kv_lthash.add_assign(LtHash::from_kv(&kv.0, &kv.1));
            kvs.push(kv);
        }

        if !ak.is_single_leaf_root {
            let centroid_kv = encode_kv(
                codec,
                hard_limits,
                LogicalKey::Centroid {
                    index: index_id,
                    tree_key: tk.clone(),
                    partition: pk,
                },
                PersistentValue::PartitionCentroid(leaf.centroid.clone()),
            )?;
            // `PartitionCentroid` wire layout: `[tag: 1B][sized<VectorComponent>]`.
            edge_def_lthash.add_assign(LtHash::from_edge(tk.as_bytes(), pk, &centroid_kv.1[1..]));
            kv_lthash.add_assign(LtHash::from_kv(&centroid_kv.0, &centroid_kv.1));
            kvs.push(centroid_kv);
            child_centroids.push((tk.clone(), pk, leaf.centroid.clone()));
        }

        for kv in leaf_entry_kvs {
            kv_lthash.add_assign(LtHash::from_kv(&kv.0, &kv.1));
            kvs.push(kv);
        }
    }

    kvs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(EncodedLeafShard {
        kvs,
        kv_lthash,
        membership_lthash,
        edge_def_lthash,
        locations_by_bucket,
        child_centroids,
    })
}

/// Encodes all `Record`, `Payload`, and `RecordLocation` entries in one disjoint Record-ID
/// bucket together in strict ascending `[index, 0x01, escaped_id, subkind]` key order,
/// verifying exact 1-to-1 `(Record ID, TreeKey, fields)` set equality in `O(N)`.
pub(crate) fn encode_record_group_bucket(
    manifest: &IndexManifest,
    seg_reader: &mut SegmentFileReader<'_>,
    mut id_refs: Vec<IdFrameEntry>,
    mut locations: Vec<(Bytes, TreeKey, PartitionKey)>,
    hard_limits: HardLimits,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<EncodedLocationBatch> {
    if id_refs.len() != locations.len() {
        return Err(corrupt());
    }
    if !id_refs.windows(2).all(|w| w[0].0 < w[1].0) {
        id_refs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        if id_refs.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(Error::new(ErrorKind::RecordAlreadyExists));
        }
    }
    locations.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    if locations.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(corrupt());
    }
    for ((id_a, _), (id_b, _, _)) in id_refs.iter().zip(locations.iter()) {
        if id_a != id_b {
            return Err(corrupt());
        }
    }

    let n = locations.len();
    let index_id = manifest.logical_index_id();
    let codec = ValueCodec::for_index(manifest);
    let mut kv_lthash = LtHash::ZERO;
    let mut location_lthash = LtHash::ZERO;

    let mut read_order: Vec<(usize, SourceFrameRef)> = id_refs
        .into_iter()
        .enumerate()
        .map(|(rank, (_, fref))| (rank, fref))
        .collect();
    read_order.sort_unstable_by(|a, b| {
        a.1.seg_idx
            .cmp(&b.1.seg_idx)
            .then_with(|| a.1.frame_offset.cmp(&b.1.frame_offset))
    });

    let mut rec_slots: Vec<Option<(Bytes, Option<Bytes>)>> = vec![None; n];
    for (idx, (rank, fref)) in read_order.into_iter().enumerate() {
        if idx.is_multiple_of(512) {
            check()?;
        }
        let raw = seg_reader.read_frame(fref.seg_idx, fref.frame_offset, fref.body_len)?;
        let (actual_id, actual_tk, fields_bytes, rec_val, pay_val) =
            source::split_serving_bytes(manifest, raw)?;
        let (expected_id, expected_tk, leaf_pk) = &locations[rank];
        if actual_id != *expected_id || &actual_tk != expected_tk {
            return Err(corrupt());
        }
        location_lthash.add_assign(LtHash::from_membership(
            expected_tk.as_bytes(),
            *leaf_pk,
            expected_id,
            &fields_bytes,
        ));
        rec_slots[rank] = Some((rec_val, pay_val));
    }

    let mut kvs = Vec::with_capacity(n * 3);
    for (rank, (id, tree_key, leaf)) in locations.into_iter().enumerate() {
        if rank.is_multiple_of(512) {
            check()?;
        }
        let (rec_val, pay_val) = rec_slots[rank].take().ok_or_else(corrupt)?;
        let rec_key = Bytes::from(keys::record_key(index_id, &id)?);
        if rec_key.len() > hard_limits.max_key_bytes || rec_val.len() > hard_limits.max_value_bytes
        {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        kv_lthash.add_assign(LtHash::from_kv(&rec_key, &rec_val));
        kvs.push((rec_key, rec_val));

        let loc_kv = encode_kv(
            codec,
            hard_limits,
            LogicalKey::Location {
                index: index_id,
                id: id.clone(),
            },
            PersistentValue::RecordLocation(RecordLocation::new(tree_key, leaf)),
        )?;
        kv_lthash.add_assign(LtHash::from_kv(&loc_kv.0, &loc_kv.1));
        kvs.push(loc_kv);

        if let Some(pay_bytes) = pay_val {
            let pay_key = Bytes::from(keys::payload_key(index_id, &id)?);
            if pay_key.len() > hard_limits.max_key_bytes
                || pay_bytes.len() > hard_limits.max_value_bytes
            {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
            kv_lthash.add_assign(LtHash::from_kv(&pay_key, &pay_bytes));
            kvs.push((pay_key, pay_bytes));
        }
    }
    debug_assert!(kvs.windows(2).all(|w| w[0].0 < w[1].0));
    Ok((kvs, kv_lthash, location_lthash))
}

/// Encoded upper-tree output (`TreeManifest` + internal partitions + local HNSW graphs).
pub(crate) struct EncodedUpperTree {
    pub partitions: u64,
    pub kvs: Vec<(Bytes, Bytes)>,
    pub kv_lthash: LtHash,
    pub edge_def_lthash: LtHash,
    pub edge_ref_lthash: LtHash,
}

/// Builds and encodes internal levels (`level >= 2`) and `TreeManifest` for one tree,
/// constructing partition-local HNSW graphs in parallel across internal nodes.
#[expect(
    clippy::too_many_arguments,
    reason = "upper-tree builder passes manifest, tree identity, children, sampling/graph/backend limits, and cancellation"
)]
pub(crate) fn build_and_encode_upper_tree(
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    total_leaves: u64,
    mut children: Vec<(PartitionKey, PartitionCentroid)>,
    sample_items: usize,
    graph_memory_bytes: usize,
    hard_limits: HardLimits,
    parallelism: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<EncodedUpperTree> {
    check()?;
    let index_id = manifest.logical_index_id();
    let codec = ValueCodec::for_index(manifest);

    if total_leaves == 1 {
        let tm = encode_kv(
            codec,
            hard_limits,
            LogicalKey::TreeManifest {
                index: index_id,
                tree_key: tree_key.clone(),
            },
            PersistentValue::TreeManifest(TreeManifest::new(
                PartitionKey::new(1)?,
                PartitionKey::new(1)?,
            )?),
        )?;
        let kv_lthash = LtHash::from_kv(&tm.0, &tm.1);
        return Ok(EncodedUpperTree {
            partitions: 0,
            kvs: vec![tm],
            kv_lthash,
            edge_def_lthash: LtHash::ZERO,
            edge_ref_lthash: LtHash::ZERO,
        });
    }

    let config = manifest.config();
    let kernel = VectorKernel::new(
        config.dimension(),
        config.metric(),
        *manifest.rotation_seed(),
    )?;
    let max_internal = config.max_internal_partition_entries() as usize;
    let min_internal = u64::from(config.min_internal_partition_entries());

    children.sort_unstable_by_key(|(pk, _)| *pk);
    let mut high_water = 1 + total_leaves;
    let mut level = 2_u32;
    let mut internal_nodes: Vec<InternalNodePlan> = Vec::new();

    loop {
        check()?;
        if children.len() <= max_internal {
            let centroid = compute_internal_centroid(&kernel, &children)?;
            internal_nodes.push((PartitionKey::new(1)?, level, centroid, children));
            break;
        }
        let mut next_level_children = Vec::new();
        let mut tagged: Vec<(usize, (PartitionKey, PartitionCentroid))> =
            children.into_iter().map(|item| (0, item)).collect();
        partition_recursively(
            &kernel,
            sample_items,
            min_internal,
            max_internal,
            &mut tagged,
            &mut |group| {
                high_water = high_water
                    .checked_add(1)
                    .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
                let pk = PartitionKey::new(high_water)?;
                let owned: Vec<_> = group.iter().map(|(_, item)| item.clone()).collect();
                let centroid = compute_internal_centroid(&kernel, &owned)?;
                next_level_children.push((pk, centroid.clone()));
                internal_nodes.push((pk, level, centroid, owned));
                Ok(())
            },
            check,
        )?;
        next_level_children.sort_unstable_by_key(|(pk, _)| *pk);
        children = next_level_children;
        level = level
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    }

    let node_results: Vec<EncodedNodeBatch> =
        parallel_chunks(&internal_nodes, parallelism, |_, chunk| {
            let mut kvs = Vec::new();
            let mut kv_lt = LtHash::ZERO;
            let mut def_lt = LtHash::ZERO;
            let mut ref_lt = LtHash::ZERO;

            for (pk, node_level, centroid, node_children) in chunk {
                check()?;
                let count = u32::try_from(node_children.len())
                    .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
                let header = PartitionHeader::new(*node_level, count, 1, PartitionState::Ready)?;
                let header_kv = encode_kv(
                    codec,
                    hard_limits,
                    LogicalKey::Header {
                        index: index_id,
                        tree_key: tree_key.clone(),
                        partition: *pk,
                    },
                    PersistentValue::PartitionHeader(header),
                )?;
                let state_kv = encode_kv(
                    codec,
                    hard_limits,
                    LogicalKey::State {
                        index: index_id,
                        tree_key: tree_key.clone(),
                        partition: *pk,
                    },
                    PersistentValue::PartitionState(PartitionTransition::Ready {
                        started_at_unix_millis: 0,
                    }),
                )?;
                for kv in [header_kv, state_kv] {
                    kv_lt.add_assign(LtHash::from_kv(&kv.0, &kv.1));
                    kvs.push(kv);
                }
                if pk.get() != 1 {
                    let c_kv = encode_kv(
                        codec,
                        hard_limits,
                        LogicalKey::Centroid {
                            index: index_id,
                            tree_key: tree_key.clone(),
                            partition: *pk,
                        },
                        PersistentValue::PartitionCentroid(centroid.clone()),
                    )?;
                    def_lt.add_assign(LtHash::from_edge(tree_key.as_bytes(), *pk, &c_kv.1[1..]));
                    kv_lt.add_assign(LtHash::from_kv(&c_kv.0, &c_kv.1));
                    kvs.push(c_kv);
                }

                let mut typed_children = Vec::with_capacity(node_children.len());
                for (child_pk, child_centroid) in node_children {
                    let ce = ChildEntry::from_centroid(*child_pk, child_centroid.clone());
                    let ce_kv = encode_kv(
                        codec,
                        hard_limits,
                        LogicalKey::ChildEntry {
                            index: index_id,
                            tree_key: tree_key.clone(),
                            partition: *pk,
                            child: *child_pk,
                        },
                        PersistentValue::ChildEntry(ce.clone()),
                    )?;
                    // `ChildEntry` wire layout: `[tag: 1B][u64 child: 8B][sized<VectorComponent>]`.
                    ref_lt.add_assign(LtHash::from_edge(
                        tree_key.as_bytes(),
                        *child_pk,
                        &ce_kv.1[9..],
                    ));
                    kv_lt.add_assign(LtHash::from_kv(&ce_kv.0, &ce_kv.1));
                    kvs.push(ce_kv);
                    typed_children.push(ce);
                }

                let graph_items = build_internal_graph(
                    manifest,
                    &kernel,
                    tree_key,
                    *pk,
                    header,
                    &typed_children,
                    graph_memory_bytes,
                    check,
                )?;
                for (gk, gv) in graph_items {
                    let gkv = encode_kv(codec, hard_limits, gk, gv)?;
                    kv_lt.add_assign(LtHash::from_kv(&gkv.0, &gkv.1));
                    kvs.push(gkv);
                }
            }
            Ok((kvs, kv_lt, def_lt, ref_lt))
        })?;

    let tm_kv = encode_kv(
        codec,
        hard_limits,
        LogicalKey::TreeManifest {
            index: index_id,
            tree_key: tree_key.clone(),
        },
        PersistentValue::TreeManifest(TreeManifest::new(
            PartitionKey::new(1)?,
            PartitionKey::new(high_water)?,
        )?),
    )?;
    let mut kv_lthash = LtHash::from_kv(&tm_kv.0, &tm_kv.1);
    let mut edge_def_lthash = LtHash::ZERO;
    let mut edge_ref_lthash = LtHash::ZERO;
    let mut kvs = vec![tm_kv];

    for (part_kvs, kv_lt, def_lt, ref_lt) in node_results {
        kv_lthash.add_assign(kv_lt);
        edge_def_lthash.add_assign(def_lt);
        edge_ref_lthash.add_assign(ref_lt);
        kvs.extend(part_kvs);
    }
    kvs.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    Ok(EncodedUpperTree {
        partitions: internal_nodes.len() as u64,
        kvs,
        kv_lthash,
        edge_def_lthash,
        edge_ref_lthash,
    })
}

fn compute_internal_centroid(
    kernel: &VectorKernel,
    children: &[(PartitionKey, PartitionCentroid)],
) -> Result<PartitionCentroid> {
    mean(kernel, children.iter().map(|(_, c)| c.components()), 2)
}

pub(crate) fn encode_kv(
    codec: ValueCodec<'_>,
    limits: HardLimits,
    key: LogicalKey,
    value: PersistentValue,
) -> Result<(Bytes, Bytes)> {
    let k = keys::encode_key(&key)?;
    let v = codec.encode_for_key(&key, &value)?;
    if k.len() > limits.max_key_bytes || v.len() > limits.max_value_bytes {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    Ok((Bytes::from(k), Bytes::from(v)))
}

// ============================================================================
// Checksummed Disk Spill / Multi-Worker Indexed Run Files (`workspace/{token}/`)
// ============================================================================

/// Zero-copy big-endian cursor over verified in-memory scratch buffers.
pub(crate) struct ScratchCursor<'a> {
    buf: &'a [u8],
}

impl<'a> ScratchCursor<'a> {
    pub(crate) const fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.buf.len() < len {
            return Err(corrupt());
        }
        let (head, tail) = self.buf.split_at(len);
        self.buf = tail;
        Ok(head)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().expect("u32")))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().expect("u64")))
    }

    pub(crate) fn bytes(&mut self, max_len: usize) -> Result<Bytes> {
        let len = self.u32()? as usize;
        if len > max_len {
            return Err(corrupt());
        }
        Ok(Bytes::copy_from_slice(self.take(len)?))
    }

    pub(crate) fn non_empty_bytes(&mut self, max_len: usize) -> Result<Bytes> {
        let len = self.u32()? as usize;
        if len == 0 || len > max_len {
            return Err(corrupt());
        }
        Ok(Bytes::copy_from_slice(self.take(len)?))
    }

    pub(crate) fn tree_key(&mut self, tree_types: &[DataType]) -> Result<TreeKey> {
        TreeKey::from_encoded(tree_types, self.bytes(MAX_TREE_KEY_BYTES)?)
    }

    pub(crate) fn centroid(
        &mut self,
        codec: ValueCodec<'_>,
        index_id: LogicalIndexId,
        tk: &TreeKey,
        pk: PartitionKey,
    ) -> Result<PartitionCentroid> {
        let c_bytes = self.non_empty_bytes(MAX_VALUE_BYTES)?;
        let c_key = LogicalKey::Centroid {
            index: index_id,
            tree_key: tk.clone(),
            partition: pk,
        };
        match codec.decode(&c_key, c_bytes)? {
            PersistentValue::PartitionCentroid(c) => Ok(c),
            _ => Err(corrupt()),
        }
    }

    pub(crate) fn f32s_into(&mut self, dst: &mut [f32]) -> Result<()> {
        let raw = self.take(dst.len().checked_mul(4).ok_or_else(corrupt)?)?;
        for (slot, chunk) in dst.iter_mut().zip(raw.as_chunks::<4>().0.iter()) {
            let bits = u32::from_be_bytes(*chunk);
            let val = f32::from_bits(bits);
            if !val.is_finite() || bits == 0x8000_0000 {
                return Err(corrupt());
            }
            *slot = val;
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<()> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(corrupt())
        }
    }
}

pub(crate) fn push_sized(buf: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    let len = u32::try_from(bytes.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

pub(crate) fn push_f32s(buf: &mut Vec<u8>, vec: &[f32]) -> Result<()> {
    for &v in vec {
        if !v.is_finite() {
            return Err(corrupt());
        }
        let bits = if v == 0.0 { 0_u32 } else { v.to_bits() };
        buf.extend_from_slice(&bits.to_be_bytes());
    }
    Ok(())
}

/// Writes a sequential payload followed by an 8-byte big-endian `xxh3_64` trailer.
pub(crate) fn write_checksummed_bytes(
    path: &Path,
    body: &[u8],
    quota: &ScratchQuota,
) -> Result<u64> {
    let checksum = xxh3_64(body);
    let total_len = (body.len() as u64)
        .checked_add(8)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    quota.write_exact(path, total_len, |w| {
        w.write_all(body).map_err(io_error)?;
        w.write_all(&checksum.to_be_bytes()).map_err(io_error)
    })
}

/// Reads an entire scratch file and verifies its trailing 8-byte `xxh3_64` trailer.
pub(crate) fn read_checksummed_bytes(path: &Path) -> Result<Vec<u8>> {
    let mut raw = fs::read(path).map_err(io_error)?;
    if raw.len() < 8 {
        return Err(corrupt());
    }
    let body_len = raw.len() - 8;
    let expected = u64::from_be_bytes(raw[body_len..].try_into().expect("trailer"));
    if xxh3_64(&raw[..body_len]) != expected {
        return Err(corrupt());
    }
    raw.truncate(body_len);
    Ok(raw)
}

/// Shared `[count: u32][(offset: u64, len: u64, xxh3_64: u64); count][hdr_xxh3: u64][bodies...]`
/// container writer. Empty slices store `len = 0` so readers skip body I/O immediately.
fn write_indexed_slices(path: &Path, slices: &[Vec<u8>], quota: &ScratchQuota) -> Result<u64> {
    let count = u32::try_from(slices.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let header_bytes = 4_u64
        .checked_add((slices.len() as u64).saturating_mul(INDEX_ENTRY_BYTES))
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let body_bytes = slices
        .iter()
        .try_fold(0_u64, |acc, s| acc.checked_add(s.len() as u64))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let total_bytes = header_bytes
        .checked_add(body_bytes)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;

    quota.write_exact(path, total_bytes, |w| {
        let mut header_hash = Xxh3::new();
        let count_b = count.to_be_bytes();
        header_hash.update(&count_b);
        w.write_all(&count_b).map_err(io_error)?;
        let mut cursor = header_bytes;
        for slice in slices {
            let len = slice.len() as u64;
            let csum = if len == 0 { 0_u64 } else { xxh3_64(slice) };
            for b in [cursor.to_be_bytes(), len.to_be_bytes(), csum.to_be_bytes()] {
                header_hash.update(&b);
                w.write_all(&b).map_err(io_error)?;
            }
            cursor += len;
        }
        w.write_all(&header_hash.digest().to_be_bytes())
            .map_err(io_error)?;
        for slice in slices {
            if !slice.is_empty() {
                w.write_all(slice).map_err(io_error)?;
            }
        }
        Ok(())
    })
}

/// Reads and checksum-verifies one indexed slice via `pread` without buffering the whole file.
fn read_indexed_slice(path: &Path, slice_idx: usize, max_slices: usize) -> Result<Option<Vec<u8>>> {
    let file = File::open(path).map_err(io_error)?;
    let file_len = file.metadata().map_err(io_error)?.len();
    if file_len < 12 {
        return Err(corrupt());
    }
    let mut count_b = [0_u8; 4];
    read_at_exact(&file, &mut count_b, 0)?;
    let count = u32::from_be_bytes(count_b) as usize;
    let body_min_offset = 4_u64
        .checked_add(
            (count as u64)
                .checked_mul(INDEX_ENTRY_BYTES)
                .ok_or_else(corrupt)?,
        )
        .and_then(|n| n.checked_add(8))
        .ok_or_else(corrupt)?;
    if slice_idx >= count || count > max_slices || body_min_offset > file_len {
        return Err(corrupt());
    }

    let mut meta = [0_u8; 24];
    read_at_exact(&file, &mut meta, 4 + (slice_idx as u64) * INDEX_ENTRY_BYTES)?;
    let offset = u64::from_be_bytes(meta[0..8].try_into().expect("offset"));
    let len = u64::from_be_bytes(meta[8..16].try_into().expect("len"));
    let expected_hash = u64::from_be_bytes(meta[16..24].try_into().expect("hash"));
    if len == 0 {
        return if expected_hash == 0 {
            Ok(None)
        } else {
            Err(corrupt())
        };
    }
    if offset < body_min_offset || offset.checked_add(len).is_none_or(|end| end > file_len) {
        return Err(corrupt());
    }
    let len_usize = usize::try_from(len).map_err(|_| corrupt())?;
    let mut buf = vec![0_u8; len_usize];
    read_at_exact(&file, &mut buf, offset)?;
    if xxh3_64(&buf) != expected_hash {
        return Err(corrupt());
    }
    Ok(Some(buf))
}

fn encode_compact_row(
    buf: &mut Vec<u8>,
    config: &IndexConfig,
    row: &CompactMeta,
    vec: &[f32],
) -> Result<()> {
    push_sized(buf, row.tree_key.as_bytes())?;
    let meta_bytes = source::encode_row_meta(config, &row.id, &row.fields)?;
    push_sized(buf, &meta_bytes)?;
    push_f32s(buf, vec)
}

fn decode_compact_row(
    r: &mut ScratchCursor<'_>,
    config: &IndexConfig,
    tree_types: &[DataType],
    vec_f32: &mut [f32],
) -> Result<(TreeKey, Bytes, Box<[Value]>)> {
    let tree_key = r.tree_key(tree_types)?;
    let meta_bytes = r.non_empty_bytes(MAX_VALUE_BYTES)?;
    let (id, fields) = source::decode_row_meta(config, meta_bytes)?;
    r.f32s_into(vec_f32)?;
    Ok((tree_key, id, fields))
}

fn push_id_frame_ref(buf: &mut Vec<u8>, id: &Bytes, fref: SourceFrameRef) -> Result<()> {
    push_sized(buf, id)?;
    buf.extend_from_slice(&fref.seg_idx.to_be_bytes());
    buf.extend_from_slice(&fref.frame_offset.to_be_bytes());
    buf.extend_from_slice(&fref.body_len.to_be_bytes());
    Ok(())
}

fn decode_id_frame_ref(r: &mut ScratchCursor<'_>) -> Result<IdFrameEntry> {
    let id = r.non_empty_bytes(MAX_RECORD_ID_BYTES)?;
    let fref = SourceFrameRef {
        seg_idx: r.u32()?,
        frame_offset: r.u64()?,
        body_len: r.u32()?,
    };
    Ok((id, fref))
}

/// Writes one scatter run's bucketed `(id, SourceFrameRef)` entries (`ids-e{ep}-{run:06}.bin`).
pub(crate) fn write_indexed_ids(
    path: &Path,
    buckets: &[Vec<IdFrameEntry>],
    quota: &ScratchQuota,
) -> Result<u64> {
    let mut slices = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let mut buf = Vec::new();
        for &(ref id, fref) in bucket {
            push_id_frame_ref(&mut buf, id, fref)?;
        }
        slices.push(buf);
    }
    write_indexed_slices(path, &slices, quota)
}

/// Reads a single bucket `bucket_idx` from an indexed ID run file.
pub(crate) fn read_indexed_ids_slice(
    path: &Path,
    bucket_idx: usize,
    out: &mut Vec<IdFrameEntry>,
) -> Result<()> {
    let Some(raw) = read_indexed_slice(path, bucket_idx, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let mut r = ScratchCursor::new(&raw);
    while !r.is_empty() {
        out.push(decode_id_frame_ref(&mut r)?);
    }
    r.finish()
}

/// Writes one deduplicated bucket's `(id, SourceFrameRef)` list (`idref-e{ep}-{b:06}.bin`).
pub(crate) fn write_bucket_id_refs(
    path: &Path,
    items: &[IdFrameEntry],
    quota: &ScratchQuota,
) -> Result<u64> {
    let count = u32::try_from(items.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let mut buf = Vec::with_capacity(4 + items.len() * 28);
    buf.extend_from_slice(&count.to_be_bytes());
    for &(ref id, fref) in items {
        push_id_frame_ref(&mut buf, id, fref)?;
    }
    write_checksummed_bytes(path, &buf, quota)
}

/// Reads a deduplicated bucket's `(id, SourceFrameRef)` list.
pub(crate) fn read_bucket_id_refs(path: &Path) -> Result<Vec<IdFrameEntry>> {
    let raw = read_checksummed_bytes(path)?;
    let mut r = ScratchCursor::new(&raw);
    let count = r.u32()? as usize;
    if count > raw.len().saturating_sub(4) / 21 {
        return Err(corrupt());
    }
    let mut out = Vec::with_capacity(count.min(65_536));
    for _ in 0..count {
        out.push(decode_id_frame_ref(&mut r)?);
    }
    r.finish()?;
    Ok(out)
}

/// Writes one scatter run's Coarse-Shard rows (`vec-*`) + per-shard row counts (`cnt-*`).
pub(crate) fn write_indexed_shards(
    vec_path: &Path,
    cnt_path: &Path,
    manifest: &IndexManifest,
    shards: &[ShardRows],
    quota: &ScratchQuota,
) -> Result<u64> {
    let shard_count =
        u32::try_from(shards.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let config = manifest.config();
    let mut encoded_slices: Vec<Vec<u8>> = Vec::with_capacity(shards.len());
    let mut cnt_buf = Vec::with_capacity(4 + shards.len() * 8);
    cnt_buf.extend_from_slice(&shard_count.to_be_bytes());
    let mut body_bytes = 0_u64;

    for shard in shards {
        let row_count =
            u32::try_from(shard.rows.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        cnt_buf.extend_from_slice(&(shard.rows.len() as u64).to_be_bytes());
        let mut buf = Vec::new();
        if row_count > 0 {
            buf.extend_from_slice(&row_count.to_be_bytes());
            for row in &shard.rows {
                encode_compact_row(&mut buf, config, row, shard.vector(row.vec_slot))?;
            }
        }
        body_bytes += buf.len() as u64;
        encoded_slices.push(buf);
    }

    let total_needed = 12_u64
        .saturating_add((shards.len() as u64).saturating_mul(INDEX_ENTRY_BYTES))
        .saturating_add(body_bytes)
        .saturating_add(cnt_buf.len() as u64 + 8);
    quota.ensure_headroom(total_needed, &[vec_path, cnt_path])?;

    let b1 = write_indexed_slices(vec_path, &encoded_slices, quota)?;
    let b2 = write_checksummed_bytes(cnt_path, &cnt_buf, quota)?;
    Ok(b1 + b2)
}

/// Reads the per-shard row counts written for one scatter run.
pub(crate) fn read_shard_counts(cnt_path: &Path, expected_shards: usize) -> Result<Vec<u64>> {
    let raw = read_checksummed_bytes(cnt_path)?;
    let mut r = ScratchCursor::new(&raw);
    let count = r.u32()? as usize;
    if count != expected_shards || count > MAX_TOTAL_SHARDS || raw.len() != 4 + count * 8 {
        return Err(corrupt());
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(r.u64()?);
    }
    r.finish()?;
    Ok(out)
}

fn append_decoded_rows(raw: &[u8], manifest: &IndexManifest, dst: &mut ShardRows) -> Result<()> {
    let mut r = ScratchCursor::new(raw);
    let rows = r.u32()? as usize;
    if rows == 0 {
        return r.finish();
    }
    let config = manifest.config();
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let dim = config.dimension();
    let min_row_bytes = (8 + dim * 4) as u64;
    if (rows as u64).saturating_mul(min_row_bytes) > (raw.len().saturating_sub(4) as u64) {
        return Err(corrupt());
    }
    dst.rows.reserve(rows.min(65_536));
    dst.vectors.reserve(rows.min(65_536).saturating_mul(dim));
    let mut vec_f32 = vec![0.0_f32; dim];
    for _ in 0..rows {
        let (tk, id, fields) = decode_compact_row(&mut r, config, tree_types, &mut vec_f32)?;
        dst.push(tk, id, fields, None, &vec_f32);
    }
    r.finish()
}

/// Appends one Coarse-Shard slice `shard_idx` from `vec-{run:06}.bin` into `dst`.
pub(crate) fn read_indexed_shard_slice(
    vec_path: &Path,
    shard_idx: usize,
    manifest: &IndexManifest,
    dst: &mut ShardRows,
) -> Result<()> {
    let Some(raw) = read_indexed_slice(vec_path, shard_idx, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    append_decoded_rows(&raw, manifest, dst)
}

/// Consolidates one Coarse Shard's immutable base rows (`0..count`) into `base-e{ep}-{s:06}.bin`.
pub(crate) fn write_base_shard_rows(
    path: &Path,
    manifest: &IndexManifest,
    rows: &ShardRows,
    count: usize,
    quota: &ScratchQuota,
) -> Result<u64> {
    if count == 0 {
        return Ok(0);
    }
    let row_count = u32::try_from(count).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let config = manifest.config();
    let mut buf = Vec::new();
    buf.extend_from_slice(&row_count.to_be_bytes());
    for row in rows.rows.get(..count).ok_or_else(corrupt)? {
        encode_compact_row(&mut buf, config, row, rows.vector(row.vec_slot))?;
    }
    write_checksummed_bytes(path, &buf, quota)
}

/// Persists a shard's delta state (`mem-*`, `centers-*`, and non-empty `inb-*`) without
/// duplicating `(TreeKey, PartitionCentroid)` in `mem-*` or creating empty `inb-*` files.
pub(crate) fn write_compact_shard_state(
    mem_path: &Path,
    centers_path: &Path,
    inb_path: &Path,
    manifest: &IndexManifest,
    state: &ShardState,
    base_row_count: usize,
    quota: &ScratchQuota,
) -> Result<(u64, u64, u64)> {
    let base_u32 =
        u32::try_from(base_row_count).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let inb_rows = state.rows.rows.get(base_row_count..).unwrap_or(&[]);
    let inb_u32 =
        u32::try_from(inb_rows.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let leaf_count =
        u32::try_from(state.leaves.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let config = manifest.config();
    let codec = ValueCodec::for_index(manifest);

    let mut mem_buf = Vec::new();
    mem_buf.extend_from_slice(&base_u32.to_be_bytes());
    mem_buf.extend_from_slice(&inb_u32.to_be_bytes());
    mem_buf.extend_from_slice(&leaf_count.to_be_bytes());

    let mut centers_buf = Vec::new();
    centers_buf.extend_from_slice(&leaf_count.to_be_bytes());

    for leaf in &state.leaves {
        let mem_len =
            u32::try_from(leaf.members.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let c_bytes = codec.encode(&PersistentValue::PartitionCentroid(leaf.centroid.clone()))?;
        push_sized(&mut centers_buf, leaf.tree_key.as_bytes())?;
        centers_buf.extend_from_slice(&mem_len.to_be_bytes());
        push_sized(&mut centers_buf, &c_bytes)?;

        mem_buf.extend_from_slice(&mem_len.to_be_bytes());
        for &row_idx in &leaf.members {
            let row_u32 =
                u32::try_from(row_idx).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            mem_buf.extend_from_slice(&row_u32.to_be_bytes());
        }
    }

    let mut inb_buf = Vec::new();
    if inb_u32 > 0 {
        inb_buf.extend_from_slice(&inb_u32.to_be_bytes());
        for row in inb_rows {
            encode_compact_row(&mut inb_buf, config, row, state.rows.vector(row.vec_slot))?;
        }
    }

    let inb_needed = if inb_u32 > 0 {
        inb_buf.len() as u64 + 8
    } else {
        0
    };
    let total_needed = (mem_buf.len() as u64 + 8)
        .saturating_add(centers_buf.len() as u64 + 8)
        .saturating_add(inb_needed);
    quota.ensure_headroom(total_needed, &[mem_path, centers_path, inb_path])?;

    let b1 = write_checksummed_bytes(mem_path, &mem_buf, quota)?;
    let b2 = write_checksummed_bytes(centers_path, &centers_buf, quota)?;
    let b3 = if inb_u32 > 0 {
        write_checksummed_bytes(inb_path, &inb_buf, quota)?
    } else {
        0
    };
    Ok((b1, b2, b3))
}

/// Reconstructs a full `ShardState` (and `base_row_count`) from `base-*`, `centers-*`, `mem-*`, and `inb-*`.
pub(crate) fn read_compact_shard_state(
    base_path: &Path,
    centers_path: &Path,
    mem_path: &Path,
    inb_path: &Path,
    manifest: &IndexManifest,
) -> Result<(ShardState, usize)> {
    let mem_raw = read_checksummed_bytes(mem_path)?;
    let mut r_mem = ScratchCursor::new(&mem_raw);
    let recorded_base = r_mem.u32()? as usize;
    let recorded_inb = r_mem.u32()? as usize;
    let leaf_count = r_mem.u32()? as usize;

    let mut rows = ShardRows::new(manifest.config().dimension());
    if recorded_base > 0 {
        let base_raw = read_checksummed_bytes(base_path)?;
        append_decoded_rows(&base_raw, manifest, &mut rows)?;
        if rows.rows.len() != recorded_base {
            return Err(corrupt());
        }
    }
    if recorded_inb > 0 {
        let inb_raw = read_checksummed_bytes(inb_path)?;
        append_decoded_rows(&inb_raw, manifest, &mut rows)?;
        if rows.rows.len() != recorded_base + recorded_inb {
            return Err(corrupt());
        }
    }

    let centers = read_shard_centers(centers_path, manifest)?;
    if centers.len() != leaf_count {
        return Err(corrupt());
    }

    let mut leaves = Vec::with_capacity(leaf_count);
    let mut seen_members = 0_usize;
    for (tk, expected_count, centroid) in centers {
        let m_count = r_mem.u32()? as usize;
        if m_count != expected_count {
            return Err(corrupt());
        }
        seen_members = seen_members.checked_add(m_count).ok_or_else(corrupt)?;
        if seen_members > rows.rows.len() {
            return Err(corrupt());
        }
        let mut members = Vec::with_capacity(m_count);
        for _ in 0..m_count {
            let row_idx = r_mem.u32()? as usize;
            if row_idx >= rows.rows.len() {
                return Err(corrupt());
            }
            members.push(row_idx);
        }
        leaves.push(ShardLeaf {
            tree_key: tk,
            centroid,
            members,
        });
    }
    r_mem.finish()?;
    Ok((ShardState { rows, leaves }, recorded_base))
}

/// Writes outbound cross-shard moved rows (`outb-e{ep}-r{r}-{src_s:06}.bin`) indexed by `dst_shard`.
pub(crate) fn write_indexed_outbound_rows(
    path: &Path,
    manifest: &IndexManifest,
    state: &ShardState,
    by_dst_shard: &[Vec<(usize, u32)>],
    quota: &ScratchQuota,
) -> Result<u64> {
    let config = manifest.config();
    let mut slices = Vec::with_capacity(by_dst_shard.len());
    for items in by_dst_shard {
        let mut buf = Vec::new();
        if !items.is_empty() {
            let item_count =
                u32::try_from(items.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            buf.extend_from_slice(&item_count.to_be_bytes());
            for &(src_row_idx, dst_local_leaf) in items {
                let row = state.rows.rows.get(src_row_idx).ok_or_else(corrupt)?;
                buf.extend_from_slice(&dst_local_leaf.to_be_bytes());
                encode_compact_row(&mut buf, config, row, state.rows.vector(row.vec_slot))?;
            }
        }
        slices.push(buf);
    }
    write_indexed_slices(path, &slices, quota)
}

/// Appends inbound moved rows destined for `dst_shard` from `outb-e{ep}-r{r}-{src_s:06}.bin`.
pub(crate) fn read_indexed_outbound_slice(
    path: &Path,
    dst_shard: usize,
    manifest: &IndexManifest,
    rows: &mut ShardRows,
    inbound_leaf_targets: &mut Vec<(u32, usize)>,
) -> Result<()> {
    let Some(raw) = read_indexed_slice(path, dst_shard, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let mut r = ScratchCursor::new(&raw);
    let item_count = r.u32()? as usize;
    let config = manifest.config();
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let dim = config.dimension();
    let min_item_bytes = (12 + dim * 4) as u64;
    if (item_count as u64).saturating_mul(min_item_bytes) > (raw.len().saturating_sub(4) as u64) {
        return Err(corrupt());
    }
    let mut vec_f32 = vec![0.0_f32; dim];
    for _ in 0..item_count {
        let dst_local_leaf = r.u32()?;
        let (tk, id, fields) = decode_compact_row(&mut r, config, tree_types, &mut vec_f32)?;
        let new_idx = rows.rows.len();
        rows.push(tk, id, fields, None, &vec_f32);
        inbound_leaf_targets.push((dst_local_leaf, new_idx));
    }
    r.finish()
}

/// Reads the compact `(TreeKey, member_count, PartitionCentroid)` list of a shard.
pub(crate) fn read_shard_centers(
    centers_path: &Path,
    manifest: &IndexManifest,
) -> Result<Vec<(TreeKey, usize, PartitionCentroid)>> {
    let raw = read_checksummed_bytes(centers_path)?;
    let mut r = ScratchCursor::new(&raw);
    let count = r.u32()? as usize;
    if (count as u64).saturating_mul(12) > (raw.len().saturating_sub(4) as u64) {
        return Err(corrupt());
    }
    let codec = ValueCodec::for_index(manifest);
    let index_id = manifest.logical_index_id();
    let dummy_pk = PartitionKey::new(2).map_err(|_| corrupt())?;
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let mut out = Vec::with_capacity(count.min(65_536));
    for _ in 0..count {
        let tk = r.tree_key(tree_types)?;
        let member_count = r.u32()? as usize;
        let centroid = r.centroid(codec, index_id, &tk, dummy_pk)?;
        out.push((tk, member_count, centroid));
    }
    r.finish()?;
    Ok(out)
}

/// Reads only the `TreeKey` per leaf from `centers-*` without allocating decoded centroids.
pub(crate) fn read_shard_leaf_trees(
    centers_path: &Path,
    manifest: &IndexManifest,
) -> Result<Vec<TreeKey>> {
    let raw = read_checksummed_bytes(centers_path)?;
    let mut r = ScratchCursor::new(&raw);
    let count = r.u32()? as usize;
    if (count as u64).saturating_mul(12) > (raw.len().saturating_sub(4) as u64) {
        return Err(corrupt());
    }
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let mut out = Vec::with_capacity(count.min(65_536));
    for _ in 0..count {
        let tk = r.tree_key(tree_types)?;
        let _member_count = r.u32()?;
        let c_len = r.u32()? as usize;
        if c_len == 0 || c_len > MAX_VALUE_BYTES {
            return Err(corrupt());
        }
        r.take(c_len)?;
        out.push(tk);
    }
    r.finish()?;
    Ok(out)
}

/// Persists `(scatter_records, final_round, assigned, tree_totals)` to `leafkeys-e{ep}.bin`
/// before `STAGE_EMIT_LEAVES` so late or recovering workers remain decoupled from reclaimed
/// construction/refinement scratch files.
pub(crate) fn write_leaf_keys_plan(
    path: &Path,
    scatter_records: u64,
    final_round: u8,
    assigned: &[Vec<AssignedLeafKey>],
    tree_totals: &TreeLeafTotals,
    quota: &ScratchQuota,
) -> Result<u64> {
    let shard_count =
        u32::try_from(assigned.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let tree_count =
        u32::try_from(tree_totals.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let mut buf = Vec::new();
    buf.extend_from_slice(LEAF_KEYS_MAGIC);
    buf.extend_from_slice(&scatter_records.to_be_bytes());
    buf.push(final_round);
    buf.extend_from_slice(&shard_count.to_be_bytes());
    for keys in assigned {
        let k_len = u32::try_from(keys.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        buf.extend_from_slice(&k_len.to_be_bytes());
        for ak in keys {
            buf.extend_from_slice(&ak.partition_key.get().to_be_bytes());
            buf.push(u8::from(ak.is_single_leaf_root));
        }
    }
    buf.extend_from_slice(&tree_count.to_be_bytes());
    for (tk, total_leaves) in tree_totals.values() {
        push_sized(&mut buf, tk.as_bytes())?;
        buf.extend_from_slice(&total_leaves.to_be_bytes());
    }
    write_checksummed_bytes(path, &buf, quota)
}

pub(crate) type LoadedLeafKeysPlan = (u64, u8, Vec<Vec<AssignedLeafKey>>, TreeLeafTotals);

/// Loads `(scatter_records, final_round, assigned, tree_totals)` from `leafkeys-e{ep}.bin`.
pub(crate) fn read_leaf_keys_plan(
    path: &Path,
    expected_shards: u32,
    manifest: &IndexManifest,
) -> Result<LoadedLeafKeysPlan> {
    let raw = read_checksummed_bytes(path)?;
    let mut r = ScratchCursor::new(&raw);
    if r.take(8)? != LEAF_KEYS_MAGIC {
        return Err(corrupt());
    }
    let scatter_records = r.u64()?;
    let final_round = r.u8()?;
    let shard_count = r.u32()?;
    if shard_count != expected_shards || (shard_count as usize) > MAX_TOTAL_SHARDS {
        return Err(corrupt());
    }
    let mut assigned = Vec::with_capacity(shard_count as usize);
    for _ in 0..shard_count {
        let k_len = r.u32()? as usize;
        if (k_len as u64).saturating_mul(9) > (raw.len() as u64) {
            return Err(corrupt());
        }
        let mut keys = Vec::with_capacity(k_len);
        for _ in 0..k_len {
            let pk = PartitionKey::new(r.u64()?).map_err(|_| corrupt())?;
            let is_single_leaf_root = match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err(corrupt()),
            };
            keys.push(AssignedLeafKey {
                partition_key: pk,
                is_single_leaf_root,
            });
        }
        assigned.push(keys);
    }
    let tree_count = r.u32()? as usize;
    if (tree_count as u64).saturating_mul(12) > (raw.len() as u64) {
        return Err(corrupt());
    }
    let (types, tcount) = manifest.tree_key_types();
    let tree_types = &types[..tcount];
    let mut tree_totals = BTreeMap::new();
    for _ in 0..tree_count {
        let tk = r.tree_key(tree_types)?;
        let total_leaves = r.u64()?;
        if total_leaves == 0 {
            return Err(corrupt());
        }
        if tree_totals
            .insert(Bytes::copy_from_slice(tk.as_bytes()), (tk, total_leaves))
            .is_some()
        {
            return Err(corrupt());
        }
    }
    r.finish()?;
    Ok((scatter_records, final_round, assigned, tree_totals))
}

/// Writes indexed location buckets (`loc-e{ep}-{shard:06}.bin`) emitted by one Coarse Shard.
pub(crate) fn write_indexed_locations(
    path: &Path,
    buckets: &[Vec<(Bytes, TreeKey, PartitionKey)>],
    quota: &ScratchQuota,
) -> Result<u64> {
    let mut slices = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let mut buf = Vec::new();
        for (id, tk, pk) in bucket {
            push_sized(&mut buf, id)?;
            push_sized(&mut buf, tk.as_bytes())?;
            buf.extend_from_slice(&pk.get().to_be_bytes());
        }
        slices.push(buf);
    }
    write_indexed_slices(path, &slices, quota)
}

/// Reads bucket `bucket_idx` from an indexed location file `loc-e{ep}-{shard:06}.bin`.
pub(crate) fn read_indexed_locations_slice(
    path: &Path,
    bucket_idx: usize,
    manifest: &IndexManifest,
    out: &mut Vec<(Bytes, TreeKey, PartitionKey)>,
) -> Result<()> {
    let Some(raw) = read_indexed_slice(path, bucket_idx, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let mut r = ScratchCursor::new(&raw);
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    while !r.is_empty() {
        let id = r.non_empty_bytes(MAX_RECORD_ID_BYTES)?;
        let tk = r.tree_key(tree_types)?;
        let pk = PartitionKey::new(r.u64()?).map_err(|_| corrupt())?;
        out.push((id, tk, pk));
    }
    r.finish()
}

/// Writes `MoveProposal`s produced by one shard during disk-backed Halo Refinement.
pub(crate) fn write_proposals(
    path: &Path,
    proposals: &[MoveProposal],
    quota: &ScratchQuota,
) -> Result<u64> {
    let count = u32::try_from(proposals.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let mut buf = Vec::with_capacity(4 + proposals.len() * 40);
    buf.extend_from_slice(&count.to_be_bytes());
    for p in proposals {
        buf.extend_from_slice(&p.gain.to_bits().to_be_bytes());
        push_sized(&mut buf, &p.id)?;
        buf.extend_from_slice(&p.src_shard.to_be_bytes());
        buf.extend_from_slice(&p.src_local_leaf.to_be_bytes());
        buf.extend_from_slice(&p.src_row_idx.to_be_bytes());
        buf.extend_from_slice(&p.dst_shard.to_be_bytes());
        buf.extend_from_slice(&p.dst_local_leaf.to_be_bytes());
    }
    write_checksummed_bytes(path, &buf, quota)
}

/// Reads `MoveProposal`s written by [`write_proposals`].
pub(crate) fn read_proposals(path: &Path) -> Result<Vec<MoveProposal>> {
    let raw = read_checksummed_bytes(path)?;
    let mut r = ScratchCursor::new(&raw);
    let count = r.u32()? as usize;
    if count > raw.len().saturating_sub(4) / 33 {
        return Err(corrupt());
    }
    let mut out = Vec::with_capacity(count.min(65_536));
    for _ in 0..count {
        let gain = f64::from_bits(r.u64()?);
        if !gain.is_finite() || gain <= 0.0 {
            return Err(corrupt());
        }
        let id = r.non_empty_bytes(MAX_RECORD_ID_BYTES)?;
        out.push(MoveProposal {
            gain,
            id,
            src_shard: r.u32()?,
            src_local_leaf: r.u32()?,
            src_row_idx: r.u32()?,
            dst_shard: r.u32()?,
            dst_local_leaf: r.u32()?,
        });
    }
    r.finish()?;
    Ok(out)
}

/// Writes upper-tree child summaries (`uchild-e{ep}-{shard:06}.bin`) indexed by `tree_bucket`
/// (`0..tree_buckets`) so `STAGE_EMIT_TREES` seeks only to its owned bucket slice.
pub(crate) fn write_indexed_upper_children(
    path: &Path,
    manifest: &IndexManifest,
    tree_buckets: u32,
    children: &[(TreeKey, PartitionKey, PartitionCentroid)],
    quota: &ScratchQuota,
) -> Result<u64> {
    let codec = ValueCodec::for_index(manifest);
    let nbuckets = tree_buckets.max(1) as usize;
    let mut slices = vec![Vec::new(); nbuckets];
    for (tk, pk, centroid) in children {
        let tk_hash = u64::from_be_bytes(tree_key_hash(tk)[..8].try_into().expect("sha256 prefix"));
        let bucket = (tk_hash % (nbuckets as u64)) as usize;
        let dst = &mut slices[bucket];
        let c_b = codec.encode(&PersistentValue::PartitionCentroid(centroid.clone()))?;
        push_sized(dst, tk.as_bytes())?;
        dst.extend_from_slice(&pk.get().to_be_bytes());
        push_sized(dst, &c_b)?;
    }
    write_indexed_slices(path, &slices, quota)
}

/// Reads bucket `bucket_idx` from an indexed upper-child summary file `uchild-e{ep}-{shard:06}.bin`.
pub(crate) fn read_indexed_upper_children_slice(
    path: &Path,
    bucket_idx: usize,
    manifest: &IndexManifest,
    out: &mut Vec<(TreeKey, PartitionKey, PartitionCentroid)>,
) -> Result<()> {
    let Some(raw) = read_indexed_slice(path, bucket_idx, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let mut r = ScratchCursor::new(&raw);
    let codec = ValueCodec::for_index(manifest);
    let index_id = manifest.logical_index_id();
    let (types, tcount) = manifest.tree_key_types();
    let tree_types = &types[..tcount];
    while !r.is_empty() {
        let tk = r.tree_key(tree_types)?;
        let pk = PartitionKey::new(r.u64()?).map_err(|_| corrupt())?;
        let centroid = r.centroid(codec, index_id, &tk, pk)?;
        out.push((tk, pk, centroid));
    }
    r.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries_preserve_legal_counts_and_bound_degenerate_groups() {
        for min in [1_u64, 3, 64] {
            for k in 2..=8 {
                let n = k * min * 2 + 1;
                let max = n - n.div_ceil(4);
                let mut cases = vec![vec![n / k; k as usize]];
                cases[0][0] += n % k;
                for center in 0..k as usize {
                    let mut counts = vec![0; k as usize];
                    counts[center] = n;
                    cases.push(counts);
                }
                for first in 0..=n {
                    let mut counts = vec![0; k as usize];
                    counts[0] = first;
                    counts[k as usize - 1] = n - first;
                    cases.push(counts);
                }
                for counts in cases {
                    let sizes = group_sizes(&counts, min);
                    assert_eq!(sizes.iter().sum::<u64>(), n);
                    assert!(sizes.iter().all(|&size| size >= min && size <= max));
                    if counts.iter().all(|&count| count >= min && count <= max) {
                        assert_eq!(sizes, counts);
                    }
                }
            }
        }
        let n = u64::MAX;
        let sizes = group_sizes(&[n, 0], 1);
        assert_eq!(sizes, [n - n.div_ceil(4), n.div_ceil(4)]);
    }
}
