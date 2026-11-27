//! Sharded segment ingestion, leaf construction, bounded halo refinement,
//! direct serving KV encoding, and checksummed scratch codecs.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use bytes::Bytes;
use xxhash_rust::xxh3::{xxh3_64, xxh3_128_with_seed};

use crate::api::{
    DataType, Error, ErrorKind, IndexConfig, MAX_RECORD_ID_BYTES, PartitionKey, Result, Value,
};
use crate::maintenance::routing::kernel_for;
use crate::refinement::{
    LeafCenter, LeafNeighbors, accept_moves, best_move, candidate_scratch_bytes, leaf_neighbors,
    mean,
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
pub(crate) const INDEX_ENTRY_BYTES: u64 = 32; // `(offset: u64, len: u64, xxh3_64: u64, count: u64)`
const LEAF_KEYS_MAGIC: &[u8; 8] = b"KTANNLK2";

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

#[derive(Default)]
struct QuotaState {
    live: u64,
    peak: u64,
    written: u64,
    files: BTreeMap<PathBuf, u64>,
    dir: Option<PathBuf>,
}

/// Tracks simultaneous and cumulative workspace disk bytes against `scratch_bytes`,
/// reserving exact temporary-file bytes before writing and reconciling on `fs::rename`.
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

/// Metadata for one compact routing row; its vector components live in `ShardRows::vectors`.
#[derive(Clone)]
pub(crate) struct CompactMeta {
    pub tree_key: TreeKey,
    pub id: Bytes,
    pub fields: Box<[Value]>,
    pub vec_slot: usize,
}

/// Contiguous column-style storage for one Coarse Shard's routing rows.
#[derive(Clone)]
pub(crate) struct ShardRows {
    pub dim: usize,
    pub rows: Vec<CompactMeta>,
    pub vectors: Vec<f32>,
}

impl ShardRows {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            rows: Vec::new(),
            vectors: Vec::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        tree_key: TreeKey,
        id: Bytes,
        fields: Box<[Value]>,
        rot_vec: &[f32],
    ) {
        let vec_slot = self.vectors.len() / self.dim;
        self.vectors.extend_from_slice(rot_vec);
        self.rows.push(CompactMeta {
            tree_key,
            id,
            fields,
            vec_slot,
        });
    }

    #[must_use]
    pub(crate) fn vector(&self, vec_slot: usize) -> &[f32] {
        let start = vec_slot * self.dim;
        &self.vectors[start..start + self.dim]
    }

    pub(crate) fn append(&mut self, mut other: Self) {
        if other.rows.is_empty() {
            return;
        }
        if self.rows.is_empty() {
            *self = other;
            return;
        }
        let slot_offset = self.vectors.len() / self.dim;
        self.vectors.extend_from_slice(&other.vectors);
        self.rows.reserve(other.rows.len());
        for row in other.rows.drain(..) {
            self.rows.push(CompactMeta {
                tree_key: row.tree_key,
                id: row.id,
                fields: row.fields,
                vec_slot: row.vec_slot + slot_offset,
            });
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

/// Decodes a contiguous range of input segments into one scatter run (`<= 2,048` runs total).
pub(crate) fn ingest_segment_range(
    snapshot: &InputSnapshot,
    seg_range: std::ops::Range<usize>,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    hard_limits: HardLimits,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<SegmentIngest> {
    let config = manifest.config();
    let dim = config.dimension();
    let index_id = manifest.logical_index_id();
    let kernel = kernel_for(manifest)?;
    let num_shards = plan.total_shards as usize;
    let num_buckets = plan.id_bucket_count() as usize;
    let empty_tree_key = if config.tree_key_fields().is_empty() {
        Some(TreeKey::encode(&[], &[])?)
    } else {
        None
    };

    let mut rows_by_shard: Vec<ShardRows> = (0..num_shards).map(|_| ShardRows::new(dim)).collect();
    let mut ids_by_bucket = vec![Vec::new(); num_buckets];
    let mut id_lthash = LtHash::ZERO;
    let mut records = 0_u64;
    let mut vec_buf = vec![0.0_f32; dim];
    let mut rot_buf = vec![0.0_f32; dim];

    for seg_idx in seg_range {
        let seg_u32 = u32::try_from(seg_idx).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let mut reader = snapshot.encoded_segment(seg_idx)?;
        while let Some(frame) = reader.next_frame_with_span() {
            if records.is_multiple_of(256) {
                check()?;
            }
            let (frame_offset, body_len, frame_bytes) = frame?;
            let (id, fields, payload_len) =
                source::decode_into_buf(config, frame_bytes, &mut vec_buf)?;

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
            ids_by_bucket[plan.id_bucket(&id) as usize].push((
                id.clone(),
                SourceFrameRef {
                    seg_idx: seg_u32,
                    frame_offset,
                    body_len,
                },
            ));

            kernel.preprocess_into(&vec_buf, &mut rot_buf)?;
            let shard = plan.route_shard(&kernel, &tree_key, &rot_buf) as usize;
            rows_by_shard[shard].push(tree_key, id, fields, &rot_buf);
            records = records
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        }
    }
    Ok(SegmentIngest {
        records,
        id_lthash,
        ids_by_bucket,
        rows_by_shard,
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
    let kernel = kernel_for(manifest)?;

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

#[derive(Clone)]
struct CentroidRef {
    centroid: PartitionCentroid,
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

/// Refines (or refreshes when `rounds == 0`) one light-tree Coarse Shard independently.
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
    let kernel = kernel_for(manifest)?;
    let min_leaf = config.min_leaf_partition_entries() as usize;
    let max_leaf = config.max_leaf_partition_entries() as usize;

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
                    leaf.centroid = compute_leaf_centroid(&kernel, &shard.rows, &leaf.members)?;
                }
            }
            continue;
        }
        if count < 2 {
            continue;
        }
        for _ in 0..rounds {
            check()?;
            let neighbors = leaf_neighbors(&check, &kernel, leaves)?;
            let mut moves: Vec<(f64, usize, usize, usize)> = Vec::new();
            for (src, leaf) in leaves.iter().enumerate() {
                for &row_idx in &leaf.members {
                    let vec = shard.rows.vector(shard.rows.rows[row_idx].vec_slot);
                    if let Some((gain, dst)) = best_move(&kernel, leaves, src, vec, &neighbors[src])
                    {
                        moves.push((gain, row_idx, src, dst));
                    }
                }
            }
            if moves.is_empty() {
                break;
            }
            moves.sort_unstable_by(|a, b| {
                b.0.total_cmp(&a.0)
                    .then_with(|| shard.rows.rows[a.1].id.cmp(&shard.rows.rows[b.1].id))
            });
            let row_ids: Vec<usize> = moves.iter().map(|m| m.1).collect();
            let indexed_moves: Vec<(f64, usize, usize, usize)> = moves
                .into_iter()
                .enumerate()
                .map(|(rank, (gain, _, src, dst))| (gain, rank, src, dst))
                .collect();
            let mut counts: Vec<usize> = leaves.iter().map(|l| l.members.len()).collect();
            let mut moved_row_to_dst: BTreeMap<usize, usize> = BTreeMap::new();
            let accepted = accept_moves(
                indexed_moves,
                &mut counts,
                min_leaf,
                max_leaf,
                |rank, _src, dst| {
                    moved_row_to_dst.insert(row_ids[rank], dst);
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
                leaf.centroid = compute_leaf_centroid(&kernel, &shard.rows, &mem)?;
                leaf.members = mem;
            }
        }
    }
    Ok(())
}

/// Bounded-halo neighborhood admission, canonical shard ordering, and move-proposal scoring.
#[expect(
    clippy::too_many_arguments,
    reason = "halo scorer binds kernel, plan, source shard state, leaf/memory bounds, cancellation, and neighbor loader"
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
        let g_src = src_leaf_base + local_l;
        for &row_idx in &leaf.members {
            let vec = src_state.rows.vector(src_state.rows.rows[row_idx].vec_slot);
            if let Some((gain, g_dst)) =
                best_move(kernel, &halo_centers, g_src, vec, &neighbors[g_src])
            {
                let (dst_shard, dst_local_leaf) = halo_map[g_dst];
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

/// Arbitrates all `MoveProposal`s for one heavy tree (`[shard_start, shard_start + shard_count)`),
/// returning the accepted proposals in deterministic `(gain DESC, id ASC)` order.
pub(crate) fn arbitrate_tree_proposals(
    shard_start: u32,
    shard_leaf_member_counts: &[Vec<usize>],
    mut proposals: Vec<MoveProposal>,
    min_leaf: usize,
    max_leaf: usize,
) -> Result<Vec<MoveProposal>> {
    let shard_count = shard_leaf_member_counts.len() as u32;
    let end_shard = shard_start.checked_add(shard_count).ok_or_else(corrupt)?;
    let mut leaf_offsets = Vec::with_capacity(shard_leaf_member_counts.len() + 1);
    let mut counts = Vec::new();
    leaf_offsets.push(0_usize);
    for shard_counts in shard_leaf_member_counts {
        counts.extend_from_slice(shard_counts);
        leaf_offsets.push(counts.len());
    }
    if proposals.is_empty() || counts.len() < 2 {
        return Ok(Vec::new());
    }
    proposals.sort_unstable_by(|a, b| b.gain.total_cmp(&a.gain).then_with(|| a.id.cmp(&b.id)));

    let mut raw_moves = Vec::with_capacity(proposals.len());
    let mut seen_src_rows = BTreeSet::new();
    for (rank, p) in proposals.iter().enumerate() {
        if p.src_shard < shard_start
            || p.src_shard >= end_shard
            || p.dst_shard < shard_start
            || p.dst_shard >= end_shard
            || !seen_src_rows.insert((p.src_shard, p.src_row_idx))
        {
            return Err(corrupt());
        }
        let local_src = (p.src_shard - shard_start) as usize;
        let local_dst = (p.dst_shard - shard_start) as usize;
        if (p.src_local_leaf as usize) >= shard_leaf_member_counts[local_src].len()
            || (p.dst_local_leaf as usize) >= shard_leaf_member_counts[local_dst].len()
        {
            return Err(corrupt());
        }
        let g_src = leaf_offsets[local_src] + p.src_local_leaf as usize;
        let g_dst = leaf_offsets[local_dst] + p.dst_local_leaf as usize;
        raw_moves.push((p.gain, rank, g_src, g_dst));
    }

    let mut chosen = Vec::new();
    accept_moves(raw_moves, &mut counts, min_leaf, max_leaf, |rank, _, _| {
        chosen.push(rank);
    });
    Ok(chosen
        .into_iter()
        .map(|rank| proposals[rank].clone())
        .collect())
}

/// Applies accepted intra-shard and inbound cross-shard moves to `shard`, compacting its rows
/// and recomputing dirty leaf centroids.
pub(crate) fn apply_moves_to_shard(
    kernel: &VectorKernel,
    shard_id: u32,
    shard: &mut ShardState,
    moved_out: &BTreeMap<usize, (u32, u32)>,
    moved_in: Vec<(u32, CompactMeta, Vec<f32>)>,
) -> Result<()> {
    if moved_out.is_empty() && moved_in.is_empty() {
        return Ok(());
    }
    let mut next_members = vec![Vec::new(); shard.leaves.len()];
    let mut dirty = vec![false; shard.leaves.len()];
    for (local_l, leaf) in shard.leaves.iter().enumerate() {
        for &row_idx in &leaf.members {
            if let Some(&(dst_s, dst_l)) = moved_out.get(&row_idx) {
                dirty[local_l] = true;
                if dst_s == shard_id {
                    next_members
                        .get_mut(dst_l as usize)
                        .ok_or_else(corrupt)?
                        .push(row_idx);
                    *dirty.get_mut(dst_l as usize).ok_or_else(corrupt)? = true;
                }
            } else {
                next_members[local_l].push(row_idx);
            }
        }
    }

    let active_total = next_members.iter().map(Vec::len).sum::<usize>() + moved_in.len();
    let dim = shard.rows.dim;
    let mut compacted = ShardRows {
        dim,
        rows: Vec::with_capacity(active_total),
        vectors: Vec::with_capacity(active_total * dim),
    };
    let old_base = shard.rows.rows.len();
    for (in_idx, (dst_l, _, _)) in moved_in.iter().enumerate() {
        next_members
            .get_mut(*dst_l as usize)
            .ok_or_else(corrupt)?
            .push(old_base + in_idx);
        *dirty.get_mut(*dst_l as usize).ok_or_else(corrupt)? = true;
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
                let r = shard.rows.rows.get(old_idx).ok_or_else(corrupt)?;
                (r, shard.rows.vector(r.vec_slot))
            } else {
                let (_, r, v) = &moved_in[old_idx - old_base];
                (r, v.as_slice())
            };
            compacted.push(r.tree_key.clone(), r.id.clone(), r.fields.clone(), v);
            remapped.push(new_idx);
        }
        if changed {
            leaf.centroid = compute_leaf_centroid(kernel, &compacted, &remapped)?;
        }
        leaf.members = remapped;
    }
    shard.rows = compacted;
    Ok(())
}

/// Summary of one unique Tree Key across all final leaves of a Bulk Build.
#[derive(Clone)]
pub(crate) struct TreePlanEntry {
    pub tree_key: TreeKey,
    pub total_leaves: u64,
    pub tree_bucket: u32,
}

/// Global leaf PartitionKey assignment for one shard's leaves.
#[derive(Clone, Copy)]
pub(crate) struct AssignedLeafKey {
    pub tree_id: u32,
    pub partition_key: PartitionKey,
    pub is_single_leaf_root: bool,
}

/// Finalized leaf-key assignment and per-tree leaf totals across all Coarse Shards.
#[derive(Clone)]
pub(crate) struct LeafPlan {
    pub scatter_records: u64,
    pub final_round: u8,
    pub assigned: Vec<Vec<AssignedLeafKey>>,
    pub trees: Vec<TreePlanEntry>,
}

/// Encoded RecordLocation KV batch together with its KV and membership `LtHash` digests.
pub(crate) type EncodedLocationBatch = (Vec<(Bytes, Bytes)>, LtHash, LtHash);

type InternalNodePlan = (
    PartitionKey,
    u32,
    PartitionCentroid,
    Vec<(PartitionKey, PartitionCentroid)>,
);

type EncodedNodeBatch = (Vec<(Bytes, Bytes)>, LtHash, LtHash, LtHash);

fn tree_bucket_for(tk: &TreeKey, tree_buckets: u32) -> u32 {
    let tk_hash = u64::from_be_bytes(tree_key_hash(tk)[..8].try_into().expect("sha256 prefix"));
    (tk_hash % u64::from(tree_buckets.max(1))) as u32
}

/// Assigns deterministic per-tree PartitionKeys across per-shard `TreeKey` iterators (`1` if
/// single-leaf root, else `2 ..= 1 + total_leaves` per tree).
pub(crate) fn assign_leaf_partition_keys<'a, S, I>(
    shards: &'a [S],
    tree_buckets: u32,
    leaves_of: impl Fn(&'a S) -> I,
) -> Result<(Vec<Vec<AssignedLeafKey>>, Vec<TreePlanEntry>)>
where
    I: IntoIterator<Item = &'a TreeKey>,
{
    let mut tree_leaf_totals: BTreeMap<Bytes, (TreeKey, u64)> = BTreeMap::new();
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

    let mut trees = Vec::with_capacity(tree_leaf_totals.len());
    let mut tree_state: BTreeMap<Bytes, (u32, u64, u64)> = BTreeMap::new();
    for (idx, (raw, (tk, total))) in tree_leaf_totals.into_iter().enumerate() {
        let tree_id = u32::try_from(idx).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let cursor = if total == 1 { 1 } else { 2 };
        trees.push(TreePlanEntry {
            tree_bucket: tree_bucket_for(&tk, tree_buckets),
            tree_key: tk,
            total_leaves: total,
        });
        tree_state.insert(raw, (tree_id, total, cursor));
    }

    let mut assigned = Vec::with_capacity(shards.len());
    for shard in shards {
        let mut shard_keys = Vec::new();
        for tk in leaves_of(shard) {
            let (tree_id, total, cursor) = tree_state.get_mut(tk.as_bytes()).expect("tree cursor");
            let pk = PartitionKey::new(*cursor)?;
            *cursor = cursor
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            shard_keys.push(AssignedLeafKey {
                tree_id: *tree_id,
                partition_key: pk,
                is_single_leaf_root: *total == 1,
            });
        }
        assigned.push(shard_keys);
    }
    Ok((assigned, trees))
}

/// Encoded outputs from one Coarse Shard's final leaves.
pub(crate) struct EncodedLeafShard {
    pub records: u64,
    pub kvs: Vec<(Bytes, Bytes)>,
    pub kv_lthash: LtHash,
    pub membership_lthash: LtHash,
    pub edge_def_lthash: LtHash,
}

/// Encodes all Leaf Partitions of one Coarse Shard into canonical KV bytes, expanding
/// `PartitionSynopsis` inline and computing homomorphic KV, membership, and centroid digests.
pub(crate) fn encode_leaf_shard(
    manifest: &IndexManifest,
    shard: &ShardState,
    assigned: &[AssignedLeafKey],
    hard_limits: HardLimits,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<EncodedLeafShard> {
    let index_id = manifest.logical_index_id();
    let codec = ValueCodec::for_index(manifest);
    let mut records = 0_u64;
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
        records += u64::from(count);

        let mut synopsis = PartitionSynopsis::empty(manifest);
        let mut leaf_entry_kvs = Vec::with_capacity(leaf.members.len());
        for &row_idx in &leaf.members {
            let row = &shard.rows.rows[row_idx];
            synopsis.expand(manifest, &row.fields)?;
            let rabitq = RaBitQ7::quantize(shard.rows.vector(row.vec_slot))?;
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
            edge_def_lthash.add_assign(LtHash::from_edge(tk.as_bytes(), pk, &centroid_kv.1[1..]));
            kv_lthash.add_assign(LtHash::from_kv(&centroid_kv.0, &centroid_kv.1));
            kvs.push(centroid_kv);
        }

        for kv in leaf_entry_kvs {
            kv_lthash.add_assign(LtHash::from_kv(&kv.0, &kv.1));
            kvs.push(kv);
        }
    }

    kvs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(EncodedLeafShard {
        records,
        kvs,
        kv_lthash,
        membership_lthash,
        edge_def_lthash,
    })
}

/// Encodes all `Record`, `Payload`, and `RecordLocation` entries in one disjoint Record-ID
/// bucket in strict ascending `[index, 0x01, escaped_id, subkind]` key order.
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

/// Builds and encodes internal levels (`level >= 2`) and `TreeManifest` for one tree.
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
    let kernel = kernel_for(manifest)?;
    let max_internal = config.max_internal_partition_entries() as usize;
    let min_internal = u64::from(config.min_internal_partition_entries());

    children.sort_unstable_by_key(|(pk, _)| *pk);
    let mut high_water = 1 + total_leaves;
    let mut level = 2_u32;
    let mut internal_nodes: Vec<InternalNodePlan> = Vec::new();

    loop {
        check()?;
        if children.len() <= max_internal {
            let centroid = mean(&kernel, children.iter().map(|(_, c)| c.components()), 2)?;
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
                let centroid = mean(&kernel, owned.iter().map(|(_, c)| c.components()), 2)?;
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
// Checksummed Disk Containers & Shared StageStore Byte/Slice Codecs
// ============================================================================

/// Zero-copy big-endian cursor over verified scratch buffers.
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

/// Writes `[count: u32][(offset: u64, len: u64, xxh3_64: u64, item_count: u64); count][bodies...]`.
pub(crate) fn write_indexed_slices(
    path: &Path,
    slices: &[Vec<u8>],
    item_counts: Option<&[u64]>,
    quota: &ScratchQuota,
) -> Result<u64> {
    if item_counts.is_some_and(|c| c.len() != slices.len()) {
        return Err(Error::invalid_argument());
    }
    let count = u32::try_from(slices.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let header_bytes = 4_u64
        .checked_add((slices.len() as u64).saturating_mul(INDEX_ENTRY_BYTES))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let body_bytes = slices
        .iter()
        .try_fold(0_u64, |acc, s| acc.checked_add(s.len() as u64))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let total_bytes = header_bytes
        .checked_add(body_bytes)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;

    quota.write_exact(path, total_bytes, |w| {
        w.write_all(&count.to_be_bytes()).map_err(io_error)?;
        let mut cursor = header_bytes;
        for (idx, slice) in slices.iter().enumerate() {
            let len = slice.len() as u64;
            let csum = if len == 0 { 0_u64 } else { xxh3_64(slice) };
            let items = item_counts.map_or(0, |c| c[idx]);
            for b in [
                cursor.to_be_bytes(),
                len.to_be_bytes(),
                csum.to_be_bytes(),
                items.to_be_bytes(),
            ] {
                w.write_all(&b).map_err(io_error)?;
            }
            cursor += len;
        }
        for slice in slices {
            if !slice.is_empty() {
                w.write_all(slice).map_err(io_error)?;
            }
        }
        Ok(())
    })
}

/// Reads per-slice item counts from an indexed slice file header in a single sequential read.
pub(crate) fn read_indexed_counts(path: &Path, expected_slices: usize) -> Result<Vec<u64>> {
    let file = File::open(path).map_err(io_error)?;
    let file_len = file.metadata().map_err(io_error)?.len();
    if expected_slices > MAX_TOTAL_SHARDS || file_len < 4 {
        return Err(corrupt());
    }
    let mut count_b = [0_u8; 4];
    read_at_exact(&file, &mut count_b, 0)?;
    let count = u32::from_be_bytes(count_b) as usize;
    let table_bytes = (count as u64)
        .checked_mul(INDEX_ENTRY_BYTES)
        .ok_or_else(corrupt)?;
    if count != expected_slices || 4_u64.checked_add(table_bytes).is_none_or(|n| n > file_len) {
        return Err(corrupt());
    }
    let mut table = vec![0_u8; table_bytes as usize];
    read_at_exact(&file, &mut table, 4)?;
    let mut out = Vec::with_capacity(count);
    let (entries, remainder) = table.as_chunks::<{ INDEX_ENTRY_BYTES as usize }>();
    if !remainder.is_empty() {
        return Err(corrupt());
    }
    for entry in entries {
        let len = u64::from_be_bytes(entry[8..16].try_into().expect("len"));
        let items = u64::from_be_bytes(entry[24..32].try_into().expect("items"));
        if (len == 0) != (items == 0) {
            return Err(corrupt());
        }
        out.push(items);
    }
    Ok(out)
}

/// Reads and checksum-verifies one indexed slice via `pread` without buffering the whole file.
pub(crate) fn read_indexed_slice(
    path: &Path,
    slice_idx: usize,
    max_slices: usize,
) -> Result<Option<Vec<u8>>> {
    let file = File::open(path).map_err(io_error)?;
    let file_len = file.metadata().map_err(io_error)?.len();
    if file_len < 4 {
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
    push_sized(buf, &source::encode_row_meta(config, &row.id, &row.fields)?)?;
    push_f32s(buf, vec)
}

fn push_id_frame_ref(buf: &mut Vec<u8>, id: &Bytes, fref: SourceFrameRef) -> Result<()> {
    push_sized(buf, id)?;
    buf.extend_from_slice(&fref.seg_idx.to_be_bytes());
    buf.extend_from_slice(&fref.frame_offset.to_be_bytes());
    buf.extend_from_slice(&fref.body_len.to_be_bytes());
    Ok(())
}

fn decode_id_frame_ref(r: &mut ScratchCursor<'_>) -> Result<IdFrameEntry> {
    Ok((
        r.non_empty_bytes(MAX_RECORD_ID_BYTES)?,
        SourceFrameRef {
            seg_idx: r.u32()?,
            frame_offset: r.u64()?,
            body_len: r.u32()?,
        },
    ))
}

pub(crate) fn encode_id_buckets(buckets: &[Vec<IdFrameEntry>]) -> Result<Vec<Vec<u8>>> {
    let mut slices = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let mut buf = Vec::with_capacity(bucket.len() * 28);
        for &(ref id, fref) in bucket {
            push_id_frame_ref(&mut buf, id, fref)?;
        }
        slices.push(buf);
    }
    Ok(slices)
}

pub(crate) fn decode_id_slice(raw: &[u8], out: &mut Vec<IdFrameEntry>) -> Result<()> {
    let mut r = ScratchCursor::new(raw);
    while !r.is_empty() {
        out.push(decode_id_frame_ref(&mut r)?);
    }
    r.finish()
}

pub(crate) fn encode_id_refs(items: &[IdFrameEntry]) -> Result<Vec<u8>> {
    let count = u32::try_from(items.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let mut buf = Vec::with_capacity(4 + items.len() * 28);
    buf.extend_from_slice(&count.to_be_bytes());
    for &(ref id, fref) in items {
        push_id_frame_ref(&mut buf, id, fref)?;
    }
    Ok(buf)
}

pub(crate) fn decode_id_refs(raw: &[u8]) -> Result<Vec<IdFrameEntry>> {
    let mut r = ScratchCursor::new(raw);
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

pub(crate) fn encode_shard_slices(
    manifest: &IndexManifest,
    shards: &[ShardRows],
) -> Result<(Vec<Vec<u8>>, Vec<u64>)> {
    let _ = u32::try_from(shards.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let config = manifest.config();
    let mut slices = Vec::with_capacity(shards.len());
    let mut counts = Vec::with_capacity(shards.len());
    for shard in shards {
        let row_count =
            u32::try_from(shard.rows.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        counts.push(u64::from(row_count));
        let mut buf = Vec::new();
        if row_count > 0 {
            buf.extend_from_slice(&row_count.to_be_bytes());
            for row in &shard.rows {
                encode_compact_row(&mut buf, config, row, shard.vector(row.vec_slot))?;
            }
        }
        slices.push(buf);
    }
    Ok((slices, counts))
}

fn decode_rows_from_cursor(
    r: &mut ScratchCursor<'_>,
    manifest: &IndexManifest,
    dst: &mut ShardRows,
) -> Result<()> {
    let rows = r.u32()? as usize;
    if rows == 0 {
        return Ok(());
    }
    let config = manifest.config();
    let (types, type_count) = manifest.tree_key_types();
    let tree_types = &types[..type_count];
    let dim = config.dimension();
    if (rows as u64).saturating_mul((8 + dim * 4) as u64) > (r.buf.len() as u64) {
        return Err(corrupt());
    }
    dst.rows.reserve(rows.min(65_536));
    dst.vectors.reserve(rows.min(65_536).saturating_mul(dim));
    let mut vec_f32 = vec![0.0_f32; dim];
    for _ in 0..rows {
        let tree_key = r.tree_key(tree_types)?;
        let (id, fields) = source::decode_row_meta(config, r.non_empty_bytes(MAX_VALUE_BYTES)?)?;
        r.f32s_into(&mut vec_f32)?;
        dst.push(tree_key, id, fields, &vec_f32);
    }
    Ok(())
}

pub(crate) fn decode_shard_slice(
    raw: &[u8],
    manifest: &IndexManifest,
    dst: &mut ShardRows,
) -> Result<()> {
    let mut r = ScratchCursor::new(raw);
    decode_rows_from_cursor(&mut r, manifest, dst)?;
    r.finish()
}

pub(crate) type EncodedShardState = (Vec<u8>, Option<(Vec<u8>, Option<Vec<Vec<u8>>>)>);

pub(crate) fn encode_shard_locations(plan: &ShardPlan, state: &ShardState) -> Result<Vec<Vec<u8>>> {
    let mut loc_slices = vec![Vec::new(); plan.id_bucket_count() as usize];
    for (local_l, leaf) in state.leaves.iter().enumerate() {
        let local_l_u32 = local_l as u32;
        for &row_idx in &leaf.members {
            let id = &state.rows.rows[row_idx].id;
            let dst = &mut loc_slices[plan.id_bucket(id) as usize];
            push_sized(dst, id)?;
            dst.extend_from_slice(&local_l_u32.to_be_bytes());
        }
    }
    Ok(loc_slices)
}

pub(crate) fn encode_shard_state(
    manifest: &IndexManifest,
    plan: &ShardPlan,
    state: &ShardState,
    emit_locs: bool,
) -> Result<EncodedShardState> {
    let leaf_count =
        u32::try_from(state.leaves.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let codec = ValueCodec::for_index(manifest);
    let mut centers_buf = Vec::new();
    centers_buf.extend_from_slice(&leaf_count.to_be_bytes());
    for leaf in &state.leaves {
        let mem_len =
            u32::try_from(leaf.members.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let c_bytes = codec.encode(&PersistentValue::PartitionCentroid(leaf.centroid.clone()))?;
        push_sized(&mut centers_buf, leaf.tree_key.as_bytes())?;
        centers_buf.extend_from_slice(&mem_len.to_be_bytes());
        push_sized(&mut centers_buf, &c_bytes)?;
    }
    if state.leaves.is_empty() {
        return Ok((centers_buf, None));
    }

    let row_count =
        u32::try_from(state.rows.rows.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let config = manifest.config();
    let mut shard_buf = Vec::new();
    shard_buf.extend_from_slice(&row_count.to_be_bytes());
    for row in &state.rows.rows {
        encode_compact_row(&mut shard_buf, config, row, state.rows.vector(row.vec_slot))?;
    }
    shard_buf.extend_from_slice(&leaf_count.to_be_bytes());
    for leaf in &state.leaves {
        let mem_len =
            u32::try_from(leaf.members.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        shard_buf.extend_from_slice(&mem_len.to_be_bytes());
        for &row_idx in &leaf.members {
            let row_u32 =
                u32::try_from(row_idx).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            shard_buf.extend_from_slice(&row_u32.to_be_bytes());
        }
    }
    let loc_slices = if emit_locs {
        Some(encode_shard_locations(plan, state)?)
    } else {
        None
    };
    Ok((centers_buf, Some((shard_buf, loc_slices))))
}

pub(crate) fn decode_shard_centers(
    raw: &[u8],
    manifest: &IndexManifest,
) -> Result<Vec<(TreeKey, usize, PartitionCentroid)>> {
    let mut r = ScratchCursor::new(raw);
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
        let c_bytes = r.non_empty_bytes(MAX_VALUE_BYTES)?;
        let c_key = LogicalKey::Centroid {
            index: index_id,
            tree_key: tk.clone(),
            partition: dummy_pk,
        };
        let PersistentValue::PartitionCentroid(centroid) = codec.decode(&c_key, c_bytes)? else {
            return Err(corrupt());
        };
        out.push((tk, member_count, centroid));
    }
    r.finish()?;
    Ok(out)
}

pub(crate) fn decode_shard_state(
    centers_raw: &[u8],
    shard_raw: Option<&[u8]>,
    manifest: &IndexManifest,
) -> Result<ShardState> {
    let centers = decode_shard_centers(centers_raw, manifest)?;
    let mut rows = ShardRows::new(manifest.config().dimension());
    if centers.is_empty() {
        return Ok(ShardState {
            rows,
            leaves: Vec::new(),
        });
    }
    let raw = shard_raw.ok_or_else(corrupt)?;
    let mut r = ScratchCursor::new(raw);
    decode_rows_from_cursor(&mut r, manifest, &mut rows)?;
    if r.u32()? as usize != centers.len() {
        return Err(corrupt());
    }
    let mut leaves = Vec::with_capacity(centers.len());
    let mut seen_members = 0_usize;
    for (tk, expected_count, centroid) in centers {
        let m_count = r.u32()? as usize;
        if m_count != expected_count {
            return Err(corrupt());
        }
        seen_members = seen_members.checked_add(m_count).ok_or_else(corrupt)?;
        if seen_members > rows.rows.len() {
            return Err(corrupt());
        }
        let mut members = Vec::with_capacity(m_count);
        for _ in 0..m_count {
            let row_idx = r.u32()? as usize;
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
    r.finish()?;
    Ok(ShardState { rows, leaves })
}

pub(crate) fn decode_locations_slice(
    raw: &[u8],
    shard_assigned: &[AssignedLeafKey],
    trees: &[TreePlanEntry],
    out: &mut Vec<(Bytes, TreeKey, PartitionKey)>,
) -> Result<()> {
    let mut r = ScratchCursor::new(raw);
    while !r.is_empty() {
        let id = r.non_empty_bytes(MAX_RECORD_ID_BYTES)?;
        let local_leaf = r.u32()? as usize;
        let ak = shard_assigned.get(local_leaf).ok_or_else(corrupt)?;
        let tk = &trees.get(ak.tree_id as usize).ok_or_else(corrupt)?.tree_key;
        out.push((id, tk.clone(), ak.partition_key));
    }
    r.finish()
}

pub(crate) fn encode_leaf_keys_plan(leaf_plan: &LeafPlan, tree_buckets: u32) -> Result<Vec<u8>> {
    let shard_count = u32::try_from(leaf_plan.assigned.len())
        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let tree_count =
        u32::try_from(leaf_plan.trees.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let mut buf = Vec::new();
    buf.extend_from_slice(LEAF_KEYS_MAGIC);
    buf.extend_from_slice(&leaf_plan.scatter_records.to_be_bytes());
    buf.push(leaf_plan.final_round);
    buf.extend_from_slice(&tree_buckets.to_be_bytes());
    buf.extend_from_slice(&tree_count.to_be_bytes());
    for t in &leaf_plan.trees {
        push_sized(&mut buf, t.tree_key.as_bytes())?;
        buf.extend_from_slice(&t.total_leaves.to_be_bytes());
    }
    buf.extend_from_slice(&shard_count.to_be_bytes());
    for keys in &leaf_plan.assigned {
        let k_len = u32::try_from(keys.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        buf.extend_from_slice(&k_len.to_be_bytes());
        for ak in keys {
            buf.extend_from_slice(&ak.tree_id.to_be_bytes());
            buf.extend_from_slice(&ak.partition_key.get().to_be_bytes());
            buf.push(u8::from(ak.is_single_leaf_root));
        }
    }
    Ok(buf)
}

pub(crate) fn decode_leaf_keys_plan(
    raw: &[u8],
    expected_shards: u32,
    expected_tree_buckets: u32,
    manifest: &IndexManifest,
) -> Result<LeafPlan> {
    let mut r = ScratchCursor::new(raw);
    if r.take(8)? != LEAF_KEYS_MAGIC {
        return Err(corrupt());
    }
    let scatter_records = r.u64()?;
    let final_round = r.u8()?;
    let tree_buckets = r.u32()?;
    if tree_buckets != expected_tree_buckets || tree_buckets == 0 {
        return Err(corrupt());
    }
    let tree_count = r.u32()? as usize;
    if (tree_count as u64).saturating_mul(12) > (raw.len() as u64) {
        return Err(corrupt());
    }
    let (types, tcount) = manifest.tree_key_types();
    let tree_types = &types[..tcount];
    let mut trees: Vec<TreePlanEntry> = Vec::with_capacity(tree_count.min(65_536));
    for _ in 0..tree_count {
        let tk = r.tree_key(tree_types)?;
        let total_leaves = r.u64()?;
        if total_leaves == 0
            || trees
                .last()
                .is_some_and(|prev| prev.tree_key.as_bytes() >= tk.as_bytes())
        {
            return Err(corrupt());
        }
        trees.push(TreePlanEntry {
            tree_bucket: tree_bucket_for(&tk, tree_buckets),
            tree_key: tk,
            total_leaves,
        });
    }
    let shard_count = r.u32()?;
    if shard_count != expected_shards || (shard_count as usize) > MAX_TOTAL_SHARDS {
        return Err(corrupt());
    }
    let mut assigned = Vec::with_capacity(shard_count as usize);
    for _ in 0..shard_count {
        let k_len = r.u32()? as usize;
        if (k_len as u64).saturating_mul(13) > (raw.len() as u64) {
            return Err(corrupt());
        }
        let mut keys = Vec::with_capacity(k_len);
        for _ in 0..k_len {
            let tree_id = r.u32()?;
            if (tree_id as usize) >= trees.len() {
                return Err(corrupt());
            }
            let pk = PartitionKey::new(r.u64()?).map_err(|_| corrupt())?;
            let is_single_leaf_root = match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err(corrupt()),
            };
            keys.push(AssignedLeafKey {
                tree_id,
                partition_key: pk,
                is_single_leaf_root,
            });
        }
        assigned.push(keys);
    }
    r.finish()?;
    Ok(LeafPlan {
        scatter_records,
        final_round,
        assigned,
        trees,
    })
}

pub(crate) fn encode_proposals(proposals: &[MoveProposal]) -> Result<Vec<u8>> {
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
    Ok(buf)
}

pub(crate) fn decode_proposals(raw: &[u8]) -> Result<Vec<MoveProposal>> {
    let mut r = ScratchCursor::new(raw);
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
