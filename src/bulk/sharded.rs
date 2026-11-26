//! Sharded segment ingestion, in-memory leaf construction, halo refinement,
//! direct serving KV encoding, and bounded disk spill I/O.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use bytes::Bytes;
use xxhash_rust::xxh3::{Xxh3, xxh3_64, xxh3_128_with_seed};

use crate::api::{Error, ErrorKind, IndexConfig, MAX_RECORD_ID_BYTES, PartitionKey, Result, Value};
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
use super::files::{SegmentFileReader, corrupt, io_error, write_atomic};
use super::graphs::build_internal_graph;
use super::input::InputSnapshot;
use super::plan::{MAX_TOTAL_SHARDS, SAMPLE_SEED, ShardPlan, nearest_center, train_centers};

pub(crate) const MAX_HALO_LEAVES_CAP: usize = 32_768;

#[derive(Default)]
struct QuotaState {
    live: u64,
    peak: u64,
    written: u64,
    files: BTreeMap<PathBuf, u64>,
    dir: Option<PathBuf>,
}

/// Tracks simultaneous and cumulative workspace disk bytes against `scratch_bytes`,
/// accounting exact bytes written to temporary files *before* `fs::rename` so failed writes
/// never spike sibling threads or unlink already-published canonical paths.
#[derive(Default)]
pub(crate) struct ScratchQuota {
    limit: u64,
    state: Mutex<QuotaState>,
}

pub(crate) struct QuotaWriter<'a, W> {
    inner: W,
    quota: &'a ScratchQuota,
    dir: Option<PathBuf>,
    charged: u64,
    unflushed: u64,
    written: u64,
}

impl<W: Write> QuotaWriter<'_, W> {
    fn reserve_more(&mut self, delta: u64) -> std::io::Result<()> {
        self.quota
            .reserve_temp_bytes(delta, self.dir.as_deref())
            .map_err(|_| std::io::Error::other("scratch quota exceeded"))?;
        self.charged += delta;
        Ok(())
    }
}

impl<W: Write> Write for QuotaWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = buf.len() as u64;
        let next_unflushed = self
            .unflushed
            .checked_add(len)
            .ok_or_else(|| std::io::Error::other("scratch quota overflow"))?;
        if next_unflushed > 64 * 1024 {
            self.reserve_more(next_unflushed)?;
            self.unflushed = 0;
        } else {
            self.unflushed = next_unflushed;
        }
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.unflushed > 0 {
            let rem = self.unflushed;
            self.reserve_more(rem)?;
            self.unflushed = 0;
        }
        self.inner.flush()
    }
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
            let name = entry.file_name();
            if name.to_str().is_some_and(|s| s.starts_with('.')) {
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

    /// Verifies upfront that `total_new_bytes` will fit inside `limit` (accounting for any
    /// existing tracked sizes at `overwritten_paths`), preventing partial multi-file writes.
    pub(crate) fn ensure_headroom(
        &self,
        total_new_bytes: u64,
        overwritten_paths: &[&Path],
    ) -> Result<()> {
        let mut st = self.lock_state();
        let credit: u64 = overwritten_paths
            .iter()
            .filter_map(|p| st.files.get(*p).copied())
            .sum();
        if st
            .live
            .saturating_sub(credit)
            .checked_add(total_new_bytes)
            .is_none_or(|n| n > self.limit)
        {
            let dir = overwritten_paths.first().and_then(|p| p.parent());
            self.refresh_locked(&mut st, dir)?;
            let refreshed_credit: u64 = overwritten_paths
                .iter()
                .filter_map(|p| st.files.get(*p).copied())
                .sum();
            if st
                .live
                .saturating_sub(refreshed_credit)
                .checked_add(total_new_bytes)
                .is_none_or(|n| n > self.limit)
            {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
        }
        Ok(())
    }

    /// Atomically writes `path` under exact pre-rename quota tracking.
    pub(crate) fn write_tracked(
        &self,
        path: &Path,
        write_fn: impl FnOnce(&mut QuotaWriter<'_, &mut BufWriter<File>>) -> Result<()>,
    ) -> Result<u64> {
        let charged = std::cell::Cell::new(0_u64);
        let quota_exceeded = std::cell::Cell::new(false);
        let parent_dir = path.parent().map(Path::to_path_buf);
        let res = write_atomic(path, |w| {
            let mut qw = QuotaWriter {
                inner: w,
                quota: self,
                dir: parent_dir,
                charged: 0,
                unflushed: 0,
                written: 0,
            };
            let body_res = write_fn(&mut qw);
            let flush_res = if body_res.is_ok() {
                qw.flush().map_err(|e| {
                    if e.to_string().contains("scratch quota") {
                        quota_exceeded.set(true);
                        Error::new(ErrorKind::LimitExceeded)
                    } else {
                        io_error(e)
                    }
                })
            } else {
                Ok(())
            };
            charged.set(qw.charged);
            if let Err(err) = body_res {
                if err.kind() == ErrorKind::Backend {
                    // Check if `QuotaWriter` triggered the underlying io::Error::other.
                    if qw.unflushed > 0
                        && self
                            .reserve_temp_bytes(qw.unflushed, qw.dir.as_deref())
                            .is_err()
                    {
                        return Err(Error::new(ErrorKind::LimitExceeded));
                    }
                }
                return Err(err);
            }
            flush_res
        });

        let reserved = charged.get();
        match res {
            Ok(actual) => {
                let mut st = self.lock_state();
                let old_bytes = st.files.insert(path.to_path_buf(), actual).unwrap_or(0);
                // Reconcile `reserved` -> `actual` and subtract `old_bytes` replaced by `rename`.
                st.live = st
                    .live
                    .saturating_sub(reserved)
                    .saturating_sub(old_bytes)
                    .saturating_add(actual);
                st.peak = st.peak.max(st.live);
                st.written = st.written.saturating_add(actual);
                Ok(actual)
            }
            Err(err) => {
                let mut st = self.lock_state();
                st.live = st.live.saturating_sub(reserved);
                if quota_exceeded.get() {
                    Err(Error::new(ErrorKind::LimitExceeded))
                } else {
                    Err(err)
                }
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
    pub slab: Option<std::sync::Arc<Vec<f32>>>,
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

    pub(crate) fn with_slab(dim: usize, slab: std::sync::Arc<Vec<f32>>) -> Self {
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

/// Output of ingesting one `InputSnapshot` segment into compact routing shards and ID buckets.
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

#[derive(Clone, Copy)]
struct IndexedItem {
    label: usize,
    row_idx: usize,
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

    let mut leaves = Vec::new();
    for (_, (tree_key, mut indices)) in by_tree {
        check()?;
        indices.sort_unstable_by(|&a, &b| rows.rows[a].id.cmp(&rows.rows[b].id));
        if indices
            .windows(2)
            .any(|w| rows.rows[w[0]].id == rows.rows[w[1]].id)
        {
            return Err(Error::new(ErrorKind::RecordAlreadyExists));
        }
        let mut items: Vec<IndexedItem> = indices
            .into_iter()
            .map(|row_idx| IndexedItem { label: 0, row_idx })
            .collect();
        group_into_leaves(
            &kernel,
            config,
            sample_items,
            &rows,
            &tree_key,
            &mut items,
            &mut leaves,
            check,
        )?;
    }

    Ok(ShardState { rows, leaves })
}

#[expect(
    clippy::too_many_arguments,
    reason = "recursive slice grouping passes immutable kernel/config/rows alongside output buffer"
)]
fn group_into_leaves(
    kernel: &VectorKernel,
    config: &IndexConfig,
    sample_items: usize,
    rows: &ShardRows,
    tree_key: &TreeKey,
    items: &mut [IndexedItem],
    out: &mut Vec<ShardLeaf>,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    let max_leaf = config.max_leaf_partition_entries() as usize;
    if items.len() <= max_leaf {
        items.sort_unstable_by(|a, b| rows.rows[a.row_idx].id.cmp(&rows.rows[b.row_idx].id));
        let members: Vec<usize> = items.iter().map(|it| it.row_idx).collect();
        let centroid = compute_leaf_centroid(kernel, rows, &members)?;
        out.push(ShardLeaf {
            tree_key: tree_key.clone(),
            centroid,
            members,
        });
        return Ok(());
    }

    check()?;
    let cap = sample_items.max(2);
    let mut heap = BinaryHeap::new();
    for (pos, it) in items.iter().enumerate() {
        let id = &rows.rows[it.row_idx].id;
        let cand = (xxh3_128_with_seed(id, SAMPLE_SEED), id, pos);
        if heap.len() < cap {
            heap.push(cand);
        } else if heap.peek().is_some_and(|max| &cand < max) {
            heap.pop();
            heap.push(cand);
        }
    }
    let sample: Vec<(Bytes, Box<[f32]>)> = heap
        .into_iter()
        .map(|(_, id, pos)| {
            let r = &rows.rows[items[pos].row_idx];
            (id.clone(), Box::from(rows.vector(r.vec_slot)))
        })
        .collect();

    let desired = (items.len() as u64).div_ceil(max_leaf as u64).min(8) as usize;
    let k = sample.len().min(desired).max(2);
    let centers = train_centers(kernel, sample, k, 10, check)?;
    let mut counts = vec![0_u64; centers.len()];
    for it in items.iter_mut() {
        let vec = rows.vector(rows.rows[it.row_idx].vec_slot);
        it.label = nearest_center(kernel, vec, &centers);
        counts[it.label] += 1;
    }
    let sizes = group_sizes(&counts, u64::from(config.min_leaf_partition_entries()));
    let cmp = |a: &IndexedItem, b: &IndexedItem| -> Ordering {
        a.label
            .cmp(&b.label)
            .then_with(|| rows.rows[a.row_idx].id.cmp(&rows.rows[b.row_idx].id))
    };
    let mut rest = items;
    for size in sizes {
        let size = size as usize;
        if size < rest.len() {
            rest.select_nth_unstable_by(size, cmp);
        }
        let (child, tail) = rest.split_at_mut(size);
        group_into_leaves(
            kernel,
            config,
            sample_items,
            rows,
            tree_key,
            child,
            out,
            check,
        )?;
        rest = tail;
    }
    Ok(())
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

    // Refine light-tree shards independently in parallel (each light tree lives in 1 shard).
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

    // Refine each heavy tree across its Coarse Shards.
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
    let chunk_size = light_shards.len().div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in light_shards.chunks_mut(chunk_size) {
            handles.push(scope.spawn(move || -> Result<()> {
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
            }));
        }
        for h in handles {
            h.join().map_err(|_| Error::new(ErrorKind::Other))??;
        }
        Ok(())
    })
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
    let neighbors_ref = &neighbors;
    let centers_ref = &all_centers;
    let map_ref = &global_to_shard_leaf;
    let offsets_ref = &leaf_offsets;

    let chunk_size = tree_shards.len().div_ceil(parallelism.max(1)).max(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_idx, chunk) in tree_shards.chunks(chunk_size).enumerate() {
            let base_s = chunk_idx * chunk_size;
            handles.push(scope.spawn(move || -> Result<Vec<MoveProposal>> {
                let mut out = Vec::new();
                for (offset_s, shard) in chunk.iter().enumerate() {
                    check()?;
                    let local_s = base_s + offset_s;
                    let src_shard = shard_start + local_s as u32;
                    let g_base = offsets_ref[local_s];
                    for (local_l, leaf) in shard.leaves.iter().enumerate() {
                        let g_src = g_base + local_l;
                        for &row_idx in &leaf.members {
                            let vec = shard.rows.vector(shard.rows.rows[row_idx].vec_slot);
                            if let Some((gain, g_dst)) =
                                best_move(kernel, centers_ref, g_src, vec, &neighbors_ref[g_src])
                            {
                                let (dst_shard, dst_local_leaf) = map_ref[g_dst];
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
            }));
        }
        let mut merged = Vec::new();
        for h in handles {
            merged.extend(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
        }
        Ok(merged)
    })
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
    let chunk_size = shard_ids.len().div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in shard_ids.chunks(chunk_size) {
            handles.push(scope.spawn(move || -> Result<Vec<MoveProposal>> {
                let mut out = Vec::new();
                for &src_shard in chunk {
                    let shard_props = score_halo_shard(
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
                    )?;
                    out.extend(shard_props);
                }
                Ok(out)
            }));
        }
        let mut merged = Vec::new();
        for h in handles {
            merged.extend(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
        }
        Ok(merged)
    })
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

    let chunk_size = tree_shards.len().div_ceil(parallelism.max(1)).max(1);
    let mut out_iter = outbound.into_iter();
    let mut in_iter = inbound.into_iter();
    let mut work_batches = Vec::new();
    for chunk in tree_shards.chunks_mut(chunk_size) {
        let mut sub_out = Vec::with_capacity(chunk.len());
        let mut sub_in = Vec::with_capacity(chunk.len());
        for _ in 0..chunk.len() {
            sub_out.push(out_iter.next().expect("aligned outbound"));
            sub_in.push(in_iter.next().expect("aligned inbound"));
        }
        work_batches.push((chunk, sub_out, sub_in));
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_idx, (chunk, sub_out, sub_in)) in work_batches.into_iter().enumerate() {
            let base_s = chunk_idx * chunk_size;
            handles.push(scope.spawn(move || -> Result<()> {
                for (offset_s, ((shard, moved_out), moved_in)) in
                    chunk.iter_mut().zip(sub_out).zip(sub_in).enumerate()
                {
                    check()?;
                    if moved_out.is_empty() && moved_in.is_empty() {
                        continue;
                    }
                    let this_shard_id = shard_start + (base_s + offset_s) as u32;
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
                    // Compact `shard.rows` to retain only active members, freeing moved-out slots.
                    let active_total: usize = next_members.iter().map(Vec::len).sum();
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
                    let mut inbound_rows: Vec<(CompactMeta, Vec<f32>)> =
                        Vec::with_capacity(moved_in.len());
                    let old_base = shard.rows.rows.len();
                    for (dst_l, meta, vec) in moved_in {
                        let temp_idx = old_base + inbound_rows.len();
                        inbound_rows.push((meta, vec));
                        next_members[dst_l as usize].push(temp_idx);
                        dirty[dst_l as usize] = true;
                    }
                    for ((leaf, mut mem), changed) in
                        shard.leaves.iter_mut().zip(next_members).zip(dirty)
                    {
                        if changed {
                            mem.sort_unstable_by(|&a, &b| {
                                let id_a = if a < old_base {
                                    &shard.rows.rows[a].id
                                } else {
                                    &inbound_rows[a - old_base].0.id
                                };
                                let id_b = if b < old_base {
                                    &shard.rows.rows[b].id
                                } else {
                                    &inbound_rows[b - old_base].0.id
                                };
                                id_a.cmp(id_b)
                            });
                        }
                        let mut remapped = Vec::with_capacity(mem.len());
                        for old_idx in mem {
                            let new_idx = compacted.rows.len();
                            if compacted.slab.is_some() {
                                let r = if old_idx < old_base {
                                    &shard.rows.rows[old_idx]
                                } else {
                                    &inbound_rows[old_idx - old_base].0
                                };
                                compacted.rows.push(r.clone());
                            } else if old_idx < old_base {
                                let r = &shard.rows.rows[old_idx];
                                let v = shard.rows.vector(r.vec_slot);
                                compacted.push(
                                    r.tree_key.clone(),
                                    r.id.clone(),
                                    r.fields.clone(),
                                    r.rabitq.clone(),
                                    v,
                                );
                            } else {
                                let (r, v) = &inbound_rows[old_idx - old_base];
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
            }));
        }
        for h in handles {
            h.join().map_err(|_| Error::new(ErrorKind::Other))??;
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

/// Assigns deterministic per-tree PartitionKeys across per-shard `TreeKey` lists (`1` if
/// single-leaf root, else `2 ..= 1 + total_leaves` per tree).
pub(crate) fn assign_leaf_partition_keys_from_trees<'a, I>(
    shard_trees: &'a [I],
) -> Result<(Vec<Vec<AssignedLeafKey>>, TreeLeafTotals)>
where
    for<'b> &'b I: IntoIterator<Item = &'b TreeKey>,
{
    let mut tree_leaf_totals: BTreeMap<Bytes, (TreeKey, u64)> = BTreeMap::new();
    for leaves in shard_trees {
        for tk in leaves {
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

    let mut assigned = Vec::with_capacity(shard_trees.len());
    for leaves in shard_trees {
        let mut shard_keys = Vec::new();
        for tk in leaves {
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

/// Assigns deterministic per-tree PartitionKeys across resident `ShardState`s.
pub(crate) fn assign_leaf_partition_keys(
    shards: &[ShardState],
) -> Result<(Vec<Vec<AssignedLeafKey>>, TreeLeafTotals)> {
    let shard_trees: Vec<Vec<TreeKey>> = shards
        .iter()
        .map(|s| s.leaves.iter().map(|l| l.tree_key.clone()).collect())
        .collect();
    assign_leaf_partition_keys_from_trees(&shard_trees)
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
        let rows_in_chunk = shard_row_offsets[end_s] - shard_row_offsets[start_s];
        let bytes_in_chunk = rows_in_chunk * code_len;
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
        // Release shared f32 slab reference immediately once RaBitQ7 codes are materialized.
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

/// Encodes all Leaf Partitions of one Coarse Shard directly into canonical KV bytes,
/// computing `PartitionSynopsis` inline without an extra pass.
pub(crate) fn encode_leaf_shard(
    manifest: &IndexManifest,
    plan: &ShardPlan,
    shard: &ShardState,
    assigned: &[AssignedLeafKey],
    hard_limits: HardLimits,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<EncodedLeafShard> {
    encode_leaf_shard_opts(manifest, plan, shard, assigned, hard_limits, true, check)
}

/// Encodes all Leaf Partitions of one Coarse Shard with optional location collection.
pub(crate) fn encode_leaf_shard_opts(
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
            let fields_bytes = source::encode_fields_bytes(manifest.config(), &row.fields)?;
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
            leaf_entry_kvs.push(kv);
            membership_lthash.add_assign(LtHash::from_membership(
                tk.as_bytes(),
                pk,
                &row.id,
                &fields_bytes,
            ));
            if collect_locations {
                let bucket = plan.id_bucket(&row.id) as usize;
                locations_by_bucket[bucket].push((row.id.clone(), tk.clone(), pk));
            }
        }

        // Emit partition keys in canonical subkind order: Header(0), Synopsis(1), State(2), Centroid(3), LeafEntry(4).
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
            edge_def_lthash.add_assign(LtHash::from_edge(tk.as_bytes(), pk, &centroid_kv.1));
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

    // Sort `(rank, SourceFrameRef)` by `(seg_idx, frame_offset)` for monotonic single-FD reads
    // while placing decoded `(record_bytes, payload_bytes)` directly into rank-aligned slots.
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

    // Tuple-escaped `Record ID` ordering matches raw `id` ordering (`0x00 -> 0x00 0xFF`,
    // terminated by `0x00 0x00`), and subkinds follow `Record(0x00) < Location(0x01) < Payload(0x02)`.
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
            // Root internal node receives reserved PartitionKey(1).
            let centroid = compute_internal_centroid(&kernel, &children)?;
            internal_nodes.push((PartitionKey::new(1)?, level, centroid, children));
            break;
        }
        let mut next_level_children = Vec::new();
        group_internal_nodes(
            &kernel,
            sample_items,
            min_internal,
            max_internal,
            level,
            &mut high_water,
            &mut children,
            &mut internal_nodes,
            &mut next_level_children,
            check,
        )?;
        next_level_children.sort_unstable_by_key(|(pk, _)| *pk);
        children = next_level_children;
        level = level
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    }

    // Build partition-local HNSW graphs and encode internal partitions in parallel.
    let chunk_size = internal_nodes.len().div_ceil(parallelism.max(1)).max(1);
    let node_results: Vec<EncodedNodeBatch> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in internal_nodes.chunks(chunk_size) {
            let kernel_ref = &kernel;
            handles.push(scope.spawn(move || -> Result<EncodedNodeBatch> {
                let mut kvs = Vec::new();
                let mut kv_lt = LtHash::ZERO;
                let mut def_lt = LtHash::ZERO;
                let mut ref_lt = LtHash::ZERO;

                for (pk, node_level, centroid, node_children) in chunk {
                    check()?;
                    let count = u32::try_from(node_children.len())
                        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
                    let header =
                        PartitionHeader::new(*node_level, count, 1, PartitionState::Ready)?;
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
                        def_lt.add_assign(LtHash::from_edge(tree_key.as_bytes(), *pk, &c_kv.1));
                        kv_lt.add_assign(LtHash::from_kv(&c_kv.0, &c_kv.1));
                        kvs.push(c_kv);
                    }

                    let mut typed_children = Vec::with_capacity(node_children.len());
                    for (child_pk, child_centroid) in node_children {
                        let centroid_bytes = codec
                            .encode(&PersistentValue::PartitionCentroid(child_centroid.clone()))?;
                        ref_lt.add_assign(LtHash::from_edge(
                            tree_key.as_bytes(),
                            *child_pk,
                            &centroid_bytes,
                        ));
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
                        kv_lt.add_assign(LtHash::from_kv(&ce_kv.0, &ce_kv.1));
                        kvs.push(ce_kv);
                        typed_children.push(ce);
                    }

                    let graph_items = build_internal_graph(
                        manifest,
                        kernel_ref,
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
            }));
        }
        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            results.push(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
        }
        Ok::<_, Error>(results)
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

#[expect(
    clippy::too_many_arguments,
    reason = "recursive internal grouping threads immutable bounds and mutable output collectors"
)]
fn group_internal_nodes(
    kernel: &VectorKernel,
    sample_items: usize,
    min_internal: u64,
    max_internal: usize,
    level: u32,
    high_water: &mut u64,
    items: &mut [(PartitionKey, PartitionCentroid)],
    out_nodes: &mut Vec<InternalNodePlan>,
    out_parents: &mut Vec<(PartitionKey, PartitionCentroid)>,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    if items.len() <= max_internal {
        items.sort_unstable_by_key(|(pk, _)| *pk);
        *high_water = high_water
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let pk = PartitionKey::new(*high_water)?;
        let owned: Vec<_> = items.to_vec();
        let centroid = compute_internal_centroid(kernel, &owned)?;
        out_parents.push((pk, centroid.clone()));
        out_nodes.push((pk, level, centroid, owned));
        return Ok(());
    }

    check()?;
    let cap = sample_items.max(2);
    let mut heap = BinaryHeap::new();
    for (pos, (pk, _)) in items.iter().enumerate() {
        let id_bytes = pk.get().to_be_bytes();
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
        .map(|(_, id_bytes, pos)| {
            (
                Bytes::copy_from_slice(&id_bytes),
                Box::from(items[pos].1.components()),
            )
        })
        .collect();
    let desired = (items.len() as u64).div_ceil(max_internal as u64).min(8) as usize;
    let k = sample.len().min(desired).max(2);
    let centers = train_centers(kernel, sample, k, 10, check)?;

    let mut tagged: Vec<(usize, (PartitionKey, PartitionCentroid))> = items
        .iter()
        .cloned()
        .map(|(pk, c)| {
            let lbl = nearest_center(kernel, c.components(), &centers);
            (lbl, (pk, c))
        })
        .collect();
    let mut counts = vec![0_u64; centers.len()];
    for (lbl, _) in &tagged {
        counts[*lbl] += 1;
    }
    let sizes = group_sizes(&counts, min_internal);
    let cmp = |a: &(usize, (PartitionKey, PartitionCentroid)),
               b: &(usize, (PartitionKey, PartitionCentroid))| {
        a.0.cmp(&b.0).then_with(|| a.1.0.cmp(&b.1.0))
    };
    let mut rest = &mut tagged[..];
    for size in sizes {
        let size = size as usize;
        if size < rest.len() {
            rest.select_nth_unstable_by(size, cmp);
        }
        let (child_slice, tail) = rest.split_at_mut(size);
        let mut child_vec: Vec<_> = child_slice.iter().map(|(_, item)| item.clone()).collect();
        group_internal_nodes(
            kernel,
            sample_items,
            min_internal,
            max_internal,
            level,
            high_water,
            &mut child_vec,
            out_nodes,
            out_parents,
            check,
        )?;
        rest = tail;
    }
    Ok(())
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

const INDEX_ENTRY_BYTES: u64 = 24; // `(offset: u64, len: u64, xxh3_64: u64)`

fn check_slice_bounds(file_len: u64, min_offset: u64, offset: u64, len: u64) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    if offset < min_offset || offset.checked_add(len).is_none_or(|end| end > file_len) {
        return Err(corrupt());
    }
    Ok(())
}

fn read_eof_checked(r: &mut impl Read) -> Result<()> {
    let mut tail = [0_u8; 1];
    if r.read(&mut tail).map_err(io_error)? != 0 {
        return Err(corrupt());
    }
    Ok(())
}

/// Writes a sequential payload followed by an 8-byte big-endian `xxh3_64` trailer.
fn write_checksummed_bytes(path: &Path, body: &[u8], quota: &ScratchQuota) -> Result<u64> {
    let checksum = xxh3_64(body);
    quota.write_tracked(path, |w| {
        w.write_all(body).map_err(io_error)?;
        w.write_all(&checksum.to_be_bytes()).map_err(io_error)
    })
}

/// Streaming reader that verifies a trailing 8-byte `xxh3_64` checksum on `finish()`.
struct ChecksummedFileReader {
    inner: std::io::Take<BufReader<File>>,
    hasher: Xxh3,
    body_len: u64,
}

impl ChecksummedFileReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(io_error)?;
        let file_len = file.metadata().map_err(io_error)?.len();
        if file_len < 8 {
            return Err(corrupt());
        }
        let body_len = file_len - 8;
        Ok(Self {
            inner: BufReader::new(file).take(body_len),
            hasher: Xxh3::new(),
            body_len,
        })
    }

    fn finish(mut self) -> Result<()> {
        read_eof_checked(&mut self)?;
        let actual = self.hasher.digest();
        let mut raw = self.inner.into_inner();
        let mut exp_b = [0_u8; 8];
        raw.read_exact(&mut exp_b).map_err(io_error)?;
        if u64::from_be_bytes(exp_b) != actual {
            return Err(corrupt());
        }
        read_eof_checked(&mut raw)
    }
}

impl Read for ChecksummedFileReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

/// Shared `[count: u32][(offset: u64, len: u64, xxh3_64: u64); count][non-empty bodies...]`
/// container writer. Empty slices store `len = 0` so readers skip I/O immediately.
fn write_indexed_slices(path: &Path, slices: &[Vec<u8>], quota: &ScratchQuota) -> Result<u64> {
    let count = u32::try_from(slices.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let header_end = 4_u64
        .checked_add((slices.len() as u64).saturating_mul(INDEX_ENTRY_BYTES))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    quota.write_tracked(path, |w| {
        let mut header_hash = Xxh3::new();
        let count_b = count.to_be_bytes();
        header_hash.update(&count_b);
        w.write_all(&count_b).map_err(io_error)?;
        let mut cursor = header_end + 8; // 8-byte header checksum right after index directory
        for slice in slices {
            let len = slice.len() as u64;
            let off = cursor;
            let csum = if len == 0 { 0_u64 } else { xxh3_64(slice) };
            for b in [off.to_be_bytes(), len.to_be_bytes(), csum.to_be_bytes()] {
                header_hash.update(&b);
                w.write_all(&b).map_err(io_error)?;
            }
            cursor = cursor
                .checked_add(len)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
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

struct SliceReader {
    inner: std::io::Take<BufReader<File>>,
    hasher: Xxh3,
    expected_hash: u64,
    len: u64,
}

impl SliceReader {
    fn open(path: &Path, slice_idx: usize, max_slices: usize) -> Result<Option<Self>> {
        let mut file = File::open(path).map_err(io_error)?;
        let file_len = file.metadata().map_err(io_error)?.len();
        if file_len < 12 {
            return Err(corrupt());
        }
        let mut count_b = [0_u8; 4];
        file.read_exact(&mut count_b).map_err(io_error)?;
        let count = u32::from_be_bytes(count_b) as usize;
        let dir_bytes = (count as u64)
            .checked_mul(INDEX_ENTRY_BYTES)
            .ok_or_else(corrupt)?;
        let body_min_offset = 4_u64
            .checked_add(dir_bytes)
            .and_then(|n| n.checked_add(8))
            .ok_or_else(corrupt)?;
        if slice_idx >= count || count > max_slices || body_min_offset > file_len {
            return Err(corrupt());
        }
        file.seek(SeekFrom::Start(4 + (slice_idx as u64) * INDEX_ENTRY_BYTES))
            .map_err(io_error)?;
        let mut meta = [0_u8; 24];
        file.read_exact(&mut meta).map_err(io_error)?;
        let offset = u64::from_be_bytes(meta[0..8].try_into().expect("offset"));
        let len = u64::from_be_bytes(meta[8..16].try_into().expect("len"));
        let expected_hash = u64::from_be_bytes(meta[16..24].try_into().expect("hash"));
        check_slice_bounds(file_len, body_min_offset, offset, len)?;
        if len == 0 {
            if expected_hash != 0 {
                return Err(corrupt());
            }
            return Ok(None);
        }
        file.seek(SeekFrom::Start(offset)).map_err(io_error)?;
        Ok(Some(Self {
            inner: BufReader::new(file).take(len),
            hasher: Xxh3::new(),
            expected_hash,
            len,
        }))
    }

    fn finish(mut self) -> Result<()> {
        read_eof_checked(&mut self)?;
        if self.hasher.digest() != self.expected_hash {
            return Err(corrupt());
        }
        Ok(())
    }
}

impl Read for SliceReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

fn encode_compact_row(
    buf: &mut Vec<u8>,
    config: &IndexConfig,
    row: &CompactMeta,
    vec: &[f32],
) -> Result<()> {
    let tk = row.tree_key.as_bytes();
    let tk_len = u32::try_from(tk.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    buf.extend_from_slice(&tk_len.to_be_bytes());
    buf.extend_from_slice(tk);
    let meta_bytes = source::encode_row_meta(config, &row.id, &row.fields)?;
    let meta_len =
        u32::try_from(meta_bytes.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    buf.extend_from_slice(&meta_len.to_be_bytes());
    buf.extend_from_slice(&meta_bytes);
    for &v in vec {
        if !v.is_finite() {
            return Err(corrupt());
        }
        let bits = if v == 0.0 { 0_u32 } else { v.to_bits() };
        buf.extend_from_slice(&bits.to_be_bytes());
    }
    Ok(())
}

fn decode_row_from_reader(
    r: &mut impl Read,
    config: &IndexConfig,
    tree_types: &[crate::api::DataType],
    vec_raw: &mut [u8],
    vec_f32: &mut [f32],
) -> Result<(TreeKey, Bytes, Box<[Value]>)> {
    let mut tk_len_b = [0_u8; 4];
    r.read_exact(&mut tk_len_b).map_err(io_error)?;
    let tk_len = u32::from_be_bytes(tk_len_b) as usize;
    if tk_len > MAX_TREE_KEY_BYTES {
        return Err(corrupt());
    }
    let mut tk_buf = vec![0_u8; tk_len];
    r.read_exact(&mut tk_buf).map_err(io_error)?;
    let tree_key = TreeKey::from_encoded(tree_types, Bytes::from(tk_buf))?;

    let mut meta_len_b = [0_u8; 4];
    r.read_exact(&mut meta_len_b).map_err(io_error)?;
    let meta_len = u32::from_be_bytes(meta_len_b) as usize;
    if meta_len == 0 || meta_len > MAX_VALUE_BYTES {
        return Err(corrupt());
    }
    let mut meta_buf = vec![0_u8; meta_len];
    r.read_exact(&mut meta_buf).map_err(io_error)?;
    let (id, fields) = source::decode_row_meta(config, Bytes::from(meta_buf))?;

    r.read_exact(vec_raw).map_err(io_error)?;
    for (dst_v, bytes) in vec_f32.iter_mut().zip(vec_raw.as_chunks::<4>().0.iter()) {
        let bits = u32::from_be_bytes(*bytes);
        let val = f32::from_bits(bits);
        if !val.is_finite() || bits == 0x8000_0000 {
            return Err(corrupt());
        }
        *dst_v = val;
    }
    Ok((tree_key, id, fields))
}

fn push_id_frame_ref(buf: &mut Vec<u8>, id: &Bytes, fref: SourceFrameRef) -> Result<()> {
    let id_len = u32::try_from(id.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    buf.extend_from_slice(&id_len.to_be_bytes());
    buf.extend_from_slice(id);
    buf.extend_from_slice(&fref.seg_idx.to_be_bytes());
    buf.extend_from_slice(&fref.frame_offset.to_be_bytes());
    buf.extend_from_slice(&fref.body_len.to_be_bytes());
    Ok(())
}

fn read_id_frame_ref(r: &mut impl Read) -> Result<(IdFrameEntry, u64)> {
    let mut l_b = [0_u8; 4];
    r.read_exact(&mut l_b).map_err(io_error)?;
    let id_len = u32::from_be_bytes(l_b) as usize;
    if id_len == 0 || id_len > MAX_RECORD_ID_BYTES {
        return Err(corrupt());
    }
    let mut id = vec![0_u8; id_len];
    r.read_exact(&mut id).map_err(io_error)?;
    let mut ref_b = [0_u8; 16];
    r.read_exact(&mut ref_b).map_err(io_error)?;
    let fref = SourceFrameRef {
        seg_idx: u32::from_be_bytes(ref_b[0..4].try_into().expect("seg")),
        frame_offset: u64::from_be_bytes(ref_b[4..12].try_into().expect("off")),
        body_len: u32::from_be_bytes(ref_b[12..16].try_into().expect("len")),
    };
    Ok(((Bytes::from(id), fref), 20 + id_len as u64))
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
    let Some(mut r) = SliceReader::open(path, bucket_idx, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let len = r.len;
    let mut read_bytes = 0_u64;
    while read_bytes < len {
        let (entry, step) = read_id_frame_ref(&mut r)?;
        read_bytes = read_bytes.checked_add(step).ok_or_else(corrupt)?;
        if read_bytes > len {
            return Err(corrupt());
        }
        out.push(entry);
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
    let mut r = ChecksummedFileReader::open(path)?;
    let mut n_b = [0_u8; 4];
    r.read_exact(&mut n_b).map_err(io_error)?;
    let count = u32::from_be_bytes(n_b) as usize;
    let max_possible = (r.body_len.saturating_sub(4) / 21) as usize;
    if count > max_possible {
        return Err(corrupt());
    }
    let mut out = Vec::with_capacity(count.min(65_536));
    for _ in 0..count {
        let (entry, _) = read_id_frame_ref(&mut r)?;
        out.push(entry);
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
    let mut counts = Vec::with_capacity(shards.len());
    let mut body_bytes = 0_u64;

    for shard in shards {
        let row_count =
            u32::try_from(shard.rows.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        counts.push(shard.rows.len() as u64);
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

    let mut cnt_buf = Vec::with_capacity(4 + counts.len() * 8);
    cnt_buf.extend_from_slice(&shard_count.to_be_bytes());
    for &c in &counts {
        cnt_buf.extend_from_slice(&c.to_be_bytes());
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
    let mut r = ChecksummedFileReader::open(cnt_path)?;
    let mut len_b = [0_u8; 4];
    r.read_exact(&mut len_b).map_err(io_error)?;
    let count = u32::from_be_bytes(len_b) as usize;
    if count != expected_shards || count > MAX_TOTAL_SHARDS || r.body_len != 4 + (count as u64) * 8
    {
        return Err(corrupt());
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut b = [0_u8; 8];
        r.read_exact(&mut b).map_err(io_error)?;
        out.push(u64::from_be_bytes(b));
    }
    r.finish()?;
    Ok(out)
}

/// Appends one Coarse-Shard slice `shard_idx` from `vec-{run:06}.bin` into `dst`.
pub(crate) fn read_indexed_shard_slice(
    vec_path: &Path,
    shard_idx: usize,
    manifest: &IndexManifest,
    dst: &mut ShardRows,
) -> Result<()> {
    let Some(mut r) = SliceReader::open(vec_path, shard_idx, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let len = r.len;
    let mut n_b = [0_u8; 4];
    r.read_exact(&mut n_b).map_err(io_error)?;
    let rows = u32::from_be_bytes(n_b) as usize;
    if rows == 0 {
        return r.finish();
    }
    let config = manifest.config();
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let dim = config.dimension();
    let min_row_bytes = (8 + dim * 4) as u64;
    if (rows as u64).saturating_mul(min_row_bytes) > len.saturating_sub(4) {
        return Err(corrupt());
    }
    let mut vec_raw = vec![0_u8; dim * 4];
    let mut vec_f32 = vec![0.0_f32; dim];

    for _ in 0..rows {
        let (tree_key, id, fields) =
            decode_row_from_reader(&mut r, config, tree_types, &mut vec_raw, &mut vec_f32)?;
        dst.push(tree_key, id, fields, None, &vec_f32);
    }
    r.finish()
}

/// Consolidates one Coarse Shard's immutable base rows (`0..count`) into a single checksummed
/// file (`base-e{ep}-{s:06}.bin`) during `STAGE_LEAVES`.
pub(crate) fn write_base_shard_rows(
    path: &Path,
    manifest: &IndexManifest,
    rows: &ShardRows,
    count: usize,
    quota: &ScratchQuota,
) -> Result<u64> {
    let row_count = u32::try_from(count).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let config = manifest.config();
    let mut buf = Vec::new();
    buf.extend_from_slice(&row_count.to_be_bytes());
    for row in rows.rows.get(..count).ok_or_else(corrupt)? {
        encode_compact_row(&mut buf, config, row, rows.vector(row.vec_slot))?;
    }
    write_checksummed_bytes(path, &buf, quota)
}

/// Reads consolidated base rows (`base-e{ep}-{s:06}.bin`) into `dst`.
pub(crate) fn read_base_shard_rows(
    path: &Path,
    manifest: &IndexManifest,
    dst: &mut ShardRows,
) -> Result<()> {
    let mut r = ChecksummedFileReader::open(path)?;
    let mut n_b = [0_u8; 4];
    r.read_exact(&mut n_b).map_err(io_error)?;
    let rows = u32::from_be_bytes(n_b) as usize;
    if rows == 0 {
        return r.finish();
    }
    let config = manifest.config();
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let dim = config.dimension();
    let min_row_bytes = (8 + dim * 4) as u64;
    if (rows as u64).saturating_mul(min_row_bytes) > r.body_len.saturating_sub(4) {
        return Err(corrupt());
    }
    dst.rows.reserve(rows.min(65_536));
    dst.vectors.reserve(rows.min(65_536).saturating_mul(dim));
    let mut vec_raw = vec![0_u8; dim * 4];
    let mut vec_f32 = vec![0.0_f32; dim];
    for _ in 0..rows {
        let (tk, id, fields) =
            decode_row_from_reader(&mut r, config, tree_types, &mut vec_raw, &mut vec_f32)?;
        dst.push(tk, id, fields, None, &vec_f32);
    }
    r.finish()
}

/// Persists a shard's delta state (`mem-*`, `centers-*`, `inb-*`) without rewriting its
/// immutable base vectors in `base-*` (`0..base_row_count`).
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
    let leaf_count =
        u32::try_from(state.leaves.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let config = manifest.config();
    let codec = ValueCodec::for_index(manifest);

    let mut mem_buf = Vec::new();
    mem_buf.extend_from_slice(&base_u32.to_be_bytes());
    mem_buf.extend_from_slice(&leaf_count.to_be_bytes());

    let mut centers_buf = Vec::new();
    centers_buf.extend_from_slice(&leaf_count.to_be_bytes());

    for leaf in &state.leaves {
        let tk = leaf.tree_key.as_bytes();
        let tk_len = u32::try_from(tk.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let mem_len =
            u32::try_from(leaf.members.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let c_bytes = codec.encode(&PersistentValue::PartitionCentroid(leaf.centroid.clone()))?;
        let c_len =
            u32::try_from(c_bytes.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;

        centers_buf.extend_from_slice(&tk_len.to_be_bytes());
        centers_buf.extend_from_slice(tk);
        centers_buf.extend_from_slice(&mem_len.to_be_bytes());
        centers_buf.extend_from_slice(&c_len.to_be_bytes());
        centers_buf.extend_from_slice(&c_bytes);

        mem_buf.extend_from_slice(&tk_len.to_be_bytes());
        mem_buf.extend_from_slice(tk);
        mem_buf.extend_from_slice(&c_len.to_be_bytes());
        mem_buf.extend_from_slice(&c_bytes);
        mem_buf.extend_from_slice(&mem_len.to_be_bytes());
        for &row_idx in &leaf.members {
            let row_u32 =
                u32::try_from(row_idx).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            mem_buf.extend_from_slice(&row_u32.to_be_bytes());
        }
    }

    let inb_rows = state.rows.rows.get(base_row_count..).unwrap_or(&[]);
    let inb_count =
        u32::try_from(inb_rows.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let mut inb_buf = Vec::new();
    inb_buf.extend_from_slice(&inb_count.to_be_bytes());
    for row in inb_rows {
        encode_compact_row(&mut inb_buf, config, row, state.rows.vector(row.vec_slot))?;
    }

    let total_needed = (mem_buf.len() as u64 + 8)
        .saturating_add(centers_buf.len() as u64 + 8)
        .saturating_add(inb_buf.len() as u64 + 8);
    quota.ensure_headroom(total_needed, &[mem_path, centers_path, inb_path])?;

    let b1 = write_checksummed_bytes(mem_path, &mem_buf, quota)?;
    let b2 = write_checksummed_bytes(centers_path, &centers_buf, quota)?;
    let b3 = write_checksummed_bytes(inb_path, &inb_buf, quota)?;
    Ok((b1, b2, b3))
}

/// Reconstructs a full `ShardState` (and its `base_row_count`) from `base-*` + `inb-*` + `mem-*`.
pub(crate) fn read_compact_shard_state(
    base_path: &Path,
    mem_path: &Path,
    inb_path: &Path,
    manifest: &IndexManifest,
) -> Result<(ShardState, usize)> {
    let config = manifest.config();
    let dim = config.dimension();
    let mut rows = ShardRows::new(dim);
    read_base_shard_rows(base_path, manifest, &mut rows)?;
    let actual_base = rows.rows.len();

    let mut r_mem = ChecksummedFileReader::open(mem_path)?;
    let mut base_b = [0_u8; 4];
    r_mem.read_exact(&mut base_b).map_err(io_error)?;
    let recorded_base = u32::from_be_bytes(base_b) as usize;
    if recorded_base != actual_base {
        return Err(corrupt());
    }
    let mut l_b = [0_u8; 4];
    r_mem.read_exact(&mut l_b).map_err(io_error)?;
    let leaf_count = u32::from_be_bytes(l_b) as usize;
    if (leaf_count as u64).saturating_mul(12) > r_mem.body_len.saturating_sub(8) {
        return Err(corrupt());
    }

    // Append inbound delta rows (`base_row_count ..`).
    read_base_shard_rows(inb_path, manifest, &mut rows)?;

    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let codec = ValueCodec::for_index(manifest);
    let index_id = manifest.logical_index_id();
    let dummy_pk = PartitionKey::new(2).map_err(|_| corrupt())?;
    let mut leaves = Vec::with_capacity(leaf_count.min(65_536));
    let mut seen_members = 0_usize;

    for _ in 0..leaf_count {
        let tk = decode_len_tree_key(&mut r_mem, tree_types)?;
        let centroid = decode_len_centroid(&mut r_mem, codec, index_id, &tk, dummy_pk)?;

        let mut m_len_b = [0_u8; 4];
        r_mem.read_exact(&mut m_len_b).map_err(io_error)?;
        let m_count = u32::from_be_bytes(m_len_b) as usize;
        seen_members = seen_members.checked_add(m_count).ok_or_else(corrupt)?;
        if seen_members > rows.rows.len() {
            return Err(corrupt());
        }
        let mut members = Vec::with_capacity(m_count);
        for _ in 0..m_count {
            let mut idx_b = [0_u8; 4];
            r_mem.read_exact(&mut idx_b).map_err(io_error)?;
            let row_idx = u32::from_be_bytes(idx_b) as usize;
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
    Ok((ShardState { rows, leaves }, actual_base))
}

fn decode_len_tree_key(r: &mut impl Read, tree_types: &[crate::api::DataType]) -> Result<TreeKey> {
    let mut tk_len_b = [0_u8; 4];
    r.read_exact(&mut tk_len_b).map_err(io_error)?;
    let tk_len = u32::from_be_bytes(tk_len_b) as usize;
    if tk_len > MAX_TREE_KEY_BYTES {
        return Err(corrupt());
    }
    let mut tk_buf = vec![0_u8; tk_len];
    r.read_exact(&mut tk_buf).map_err(io_error)?;
    TreeKey::from_encoded(tree_types, Bytes::from(tk_buf))
}

fn decode_len_centroid(
    r: &mut impl Read,
    codec: ValueCodec<'_>,
    index_id: crate::api::LogicalIndexId,
    tk: &TreeKey,
    pk: PartitionKey,
) -> Result<PartitionCentroid> {
    let mut c_len_b = [0_u8; 4];
    r.read_exact(&mut c_len_b).map_err(io_error)?;
    let c_len = u32::from_be_bytes(c_len_b) as usize;
    if c_len == 0 || c_len > MAX_VALUE_BYTES {
        return Err(corrupt());
    }
    let mut c_buf = vec![0_u8; c_len];
    r.read_exact(&mut c_buf).map_err(io_error)?;
    let c_key = LogicalKey::Centroid {
        index: index_id,
        tree_key: tk.clone(),
        partition: pk,
    };
    match codec.decode(&c_key, Bytes::from(c_buf))? {
        PersistentValue::PartitionCentroid(c) => Ok(c),
        _ => Err(corrupt()),
    }
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
    let Some(mut r) = SliceReader::open(path, dst_shard, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let len = r.len;
    let mut n_b = [0_u8; 4];
    r.read_exact(&mut n_b).map_err(io_error)?;
    let item_count = u32::from_be_bytes(n_b) as usize;

    let config = manifest.config();
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let dim = config.dimension();
    let min_item_bytes = (12 + dim * 4) as u64;
    if (item_count as u64).saturating_mul(min_item_bytes) > len.saturating_sub(4) {
        return Err(corrupt());
    }
    let mut vec_raw = vec![0_u8; dim * 4];
    let mut vec_f32 = vec![0.0_f32; dim];

    for _ in 0..item_count {
        let mut dl_b = [0_u8; 4];
        r.read_exact(&mut dl_b).map_err(io_error)?;
        let dst_local_leaf = u32::from_be_bytes(dl_b);
        let (tk, id, fields) =
            decode_row_from_reader(&mut r, config, tree_types, &mut vec_raw, &mut vec_f32)?;
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
    let mut r = ChecksummedFileReader::open(centers_path)?;
    let mut n_b = [0_u8; 4];
    r.read_exact(&mut n_b).map_err(io_error)?;
    let count = u32::from_be_bytes(n_b) as usize;
    if (count as u64).saturating_mul(12) > r.body_len.saturating_sub(4) {
        return Err(corrupt());
    }
    let codec = ValueCodec::for_index(manifest);
    let index_id = manifest.logical_index_id();
    let dummy_pk = PartitionKey::new(2).map_err(|_| corrupt())?;
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let mut out = Vec::with_capacity(count.min(65_536));
    for _ in 0..count {
        let tk = decode_len_tree_key(&mut r, tree_types)?;
        let mut mc_b = [0_u8; 4];
        r.read_exact(&mut mc_b).map_err(io_error)?;
        let member_count = u32::from_be_bytes(mc_b) as usize;
        let centroid = decode_len_centroid(&mut r, codec, index_id, &tk, dummy_pk)?;
        out.push((tk, member_count, centroid));
    }
    r.finish()?;
    Ok(out)
}

/// Reads only the `TreeKey` per leaf from `centers-*` while verifying its `xxh3_64` trailer.
pub(crate) fn read_shard_leaf_trees(
    centers_path: &Path,
    manifest: &IndexManifest,
) -> Result<Vec<TreeKey>> {
    let mut r = ChecksummedFileReader::open(centers_path)?;
    let mut n_b = [0_u8; 4];
    r.read_exact(&mut n_b).map_err(io_error)?;
    let count = u32::from_be_bytes(n_b) as usize;
    if (count as u64).saturating_mul(12) > r.body_len.saturating_sub(4) {
        return Err(corrupt());
    }
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let mut out = Vec::with_capacity(count.min(65_536));
    let mut skip_buf = Vec::new();
    for _ in 0..count {
        let tk = decode_len_tree_key(&mut r, tree_types)?;
        let mut skip_hdr = [0_u8; 8];
        r.read_exact(&mut skip_hdr).map_err(io_error)?;
        let c_len = u32::from_be_bytes(skip_hdr[4..8].try_into().expect("c_len")) as usize;
        if c_len == 0 || c_len > MAX_VALUE_BYTES {
            return Err(corrupt());
        }
        skip_buf.resize(c_len, 0);
        r.read_exact(&mut skip_buf).map_err(io_error)?;
        out.push(tk);
    }
    r.finish()?;
    Ok(out)
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
            let id_len =
                u32::try_from(id.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            let tk_b = tk.as_bytes();
            let tk_len =
                u32::try_from(tk_b.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            buf.extend_from_slice(&id_len.to_be_bytes());
            buf.extend_from_slice(id);
            buf.extend_from_slice(&tk_len.to_be_bytes());
            buf.extend_from_slice(tk_b);
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
    let Some(mut r) = SliceReader::open(path, bucket_idx, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let len = r.len;
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let mut read_bytes = 0_u64;
    while read_bytes < len {
        let mut id_l = [0_u8; 4];
        r.read_exact(&mut id_l).map_err(io_error)?;
        let id_len = u32::from_be_bytes(id_l) as usize;
        if id_len == 0 || id_len > MAX_RECORD_ID_BYTES {
            return Err(corrupt());
        }
        let mut id = vec![0_u8; id_len];
        r.read_exact(&mut id).map_err(io_error)?;

        let tk = decode_len_tree_key(&mut r, tree_types)?;
        let tk_len = tk.as_bytes().len();
        let step = 16_u64 + id_len as u64 + tk_len as u64;
        if read_bytes.checked_add(step).is_none_or(|end| end > len) {
            return Err(corrupt());
        }

        let mut pk_b = [0_u8; 8];
        r.read_exact(&mut pk_b).map_err(io_error)?;
        let pk = PartitionKey::new(u64::from_be_bytes(pk_b)).map_err(|_| corrupt())?;

        read_bytes += step;
        out.push((Bytes::from(id), tk, pk));
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
        let id_len = u32::try_from(p.id.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        buf.extend_from_slice(&p.gain.to_bits().to_be_bytes());
        buf.extend_from_slice(&id_len.to_be_bytes());
        buf.extend_from_slice(&p.id);
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
    let mut r = ChecksummedFileReader::open(path)?;
    let mut n_b = [0_u8; 4];
    r.read_exact(&mut n_b).map_err(io_error)?;
    let count = u32::from_be_bytes(n_b) as usize;
    let max_possible = (r.body_len.saturating_sub(4) / 33) as usize;
    if count > max_possible {
        return Err(corrupt());
    }
    let mut out = Vec::with_capacity(count.min(65_536));
    for _ in 0..count {
        let mut g_b = [0_u8; 8];
        r.read_exact(&mut g_b).map_err(io_error)?;
        let gain = f64::from_bits(u64::from_be_bytes(g_b));
        if !gain.is_finite() || gain <= 0.0 {
            return Err(corrupt());
        }

        let mut id_l = [0_u8; 4];
        r.read_exact(&mut id_l).map_err(io_error)?;
        let id_len = u32::from_be_bytes(id_l) as usize;
        if id_len == 0 || id_len > MAX_RECORD_ID_BYTES {
            return Err(corrupt());
        }
        let mut id = vec![0_u8; id_len];
        r.read_exact(&mut id).map_err(io_error)?;

        let mut fields = [0_u8; 20];
        r.read_exact(&mut fields).map_err(io_error)?;
        out.push(MoveProposal {
            gain,
            id: Bytes::from(id),
            src_shard: u32::from_be_bytes(fields[0..4].try_into().expect("src_s")),
            src_local_leaf: u32::from_be_bytes(fields[4..8].try_into().expect("src_l")),
            src_row_idx: u32::from_be_bytes(fields[8..12].try_into().expect("src_r")),
            dst_shard: u32::from_be_bytes(fields[12..16].try_into().expect("dst_s")),
            dst_local_leaf: u32::from_be_bytes(fields[16..20].try_into().expect("dst_l")),
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
        let tk_b = tk.as_bytes();
        let tk_len = u32::try_from(tk_b.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let c_b = codec.encode(&PersistentValue::PartitionCentroid(centroid.clone()))?;
        let c_len = u32::try_from(c_b.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        dst.extend_from_slice(&tk_len.to_be_bytes());
        dst.extend_from_slice(tk_b);
        dst.extend_from_slice(&pk.get().to_be_bytes());
        dst.extend_from_slice(&c_len.to_be_bytes());
        dst.extend_from_slice(&c_b);
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
    let Some(mut r) = SliceReader::open(path, bucket_idx, MAX_TOTAL_SHARDS)? else {
        return Ok(());
    };
    let len = r.len;
    let codec = ValueCodec::for_index(manifest);
    let index_id = manifest.logical_index_id();
    let (types, tcount) = manifest.tree_key_types();
    let tree_types = &types[..tcount];
    let mut read_bytes = 0_u64;

    while read_bytes < len {
        let tk = decode_len_tree_key(&mut r, tree_types)?;
        let tk_len = tk.as_bytes().len();

        let mut pk_b = [0_u8; 8];
        r.read_exact(&mut pk_b).map_err(io_error)?;
        let pk = PartitionKey::new(u64::from_be_bytes(pk_b)).map_err(|_| corrupt())?;

        let mut c_l = [0_u8; 4];
        r.read_exact(&mut c_l).map_err(io_error)?;
        let c_len = u32::from_be_bytes(c_l) as usize;
        if c_len == 0 || c_len > MAX_VALUE_BYTES {
            return Err(corrupt());
        }
        let step = 16_u64 + tk_len as u64 + c_len as u64;
        if read_bytes.checked_add(step).is_none_or(|end| end > len) {
            return Err(corrupt());
        }
        let mut c_b = vec![0_u8; c_len];
        r.read_exact(&mut c_b).map_err(io_error)?;
        let c_key = LogicalKey::Centroid {
            index: index_id,
            tree_key: tk.clone(),
            partition: pk,
        };
        let PersistentValue::PartitionCentroid(centroid) =
            codec.decode(&c_key, Bytes::from(c_b))?
        else {
            return Err(corrupt());
        };
        read_bytes += step;
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
