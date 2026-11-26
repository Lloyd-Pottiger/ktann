//! Deterministic coarse-shard and Record-ID-bucket planning from a bounded sample.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use bytes::Bytes;
use xxhash_rust::xxh3::{Xxh3, xxh3_64, xxh3_128_with_seed};

use crate::api::{Error, ErrorKind, MAX_RECORD_ID_BYTES, Result};
use crate::construction::ConstructionOptions;
use crate::maintenance::routing::kernel_for;
use crate::search::numeric::VectorKernel;
use crate::storage::keys::{MAX_TREE_KEY_BYTES, TreeKey};
use crate::storage::values::{IndexManifest, source};

use super::files::{corrupt, io_error};
use super::input::InputSnapshot;
use super::sharded::ScratchQuota;

pub(super) const SAMPLE_SEED: u64 = 0x6b74_616e_6e62_756c;
const PLAN_MAGIC: &[u8; 8] = b"KTANNPL2";
const MAX_SAMPLE_CAP: usize = 32_768;
const MAX_ID_BUCKETS: usize = 4_096;
const TARGET_ID_BUCKET_ROWS: u64 = 16_384;
const DEFAULT_SHARD_TARGET_ROWS: u64 = 16_384;
pub(crate) const MAX_TOTAL_SHARDS: usize = 65_536;
const MAX_LIGHT_SHARDS: u32 = 1_024;

type TreeSampleMap = BTreeMap<Bytes, (TreeKey, Vec<(Bytes, Box<[f32]>)>)>;
type SuperIndex = (Vec<Box<[f32]>>, Vec<Vec<u32>>);

/// Coarse routing centroids owned by one large Tree Key.
#[derive(Clone)]
pub(crate) struct HeavyTreePlan {
    pub tree_key: TreeKey,
    pub shard_start: u32,
    pub shard_count: u32,
    pub centers: Vec<Box<[f32]>>,
    /// Optional first-stage coarse centroids when `centers.len() >= 32`.
    pub super_centers: Vec<Box<[f32]>>,
    pub super_children: Vec<Vec<u32>>,
}

impl fmt::Debug for HeavyTreePlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeavyTreePlan")
            .field("tree_key", &"[REDACTED]")
            .field("shard_start", &self.shard_start)
            .field("shard_count", &self.shard_count)
            .field("centers", &self.centers.len())
            .finish_non_exhaustive()
    }
}

/// Complete deterministic shard layout for one Bulk Build.
#[derive(Clone)]
pub(crate) struct ShardPlan {
    /// Strict ascending Record ID splitters defining `id_bounds.len() + 1` buckets.
    pub id_bounds: Vec<Bytes>,
    /// Heavy trees sorted by `tree_key.as_bytes()` for binary search.
    pub heavy_trees: Vec<HeavyTreePlan>,
    pub light_shard_start: u32,
    pub light_shard_count: u32,
    pub total_shards: u32,
    pub target_shard_rows: usize,
}

impl fmt::Debug for ShardPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShardPlan")
            .field("id_bounds", &self.id_bounds.len())
            .field("heavy_trees", &self.heavy_trees.len())
            .field("light_shard_start", &self.light_shard_start)
            .field("light_shard_count", &self.light_shard_count)
            .field("total_shards", &self.total_shards)
            .field("target_shard_rows", &self.target_shard_rows)
            .finish()
    }
}

struct SampledRow {
    hash: u128,
    id: Bytes,
    tree_key: TreeKey,
    rot_vec: Box<[f32]>,
}

impl PartialEq for SampledRow {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for SampledRow {}

impl PartialOrd for SampledRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SampledRow {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hash
            .cmp(&other.hash)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl ShardPlan {
    /// Samples uniformly across segments and constructs a deterministic coarse shard plan.
    ///
    /// All workers of one build share identical `BulkWorkerOptions` and `BuildDescriptor`,
    /// so `effective_mem = worker_memory_bytes.min(construction.memory_bytes)` deterministically
    /// lowers `target_shard_rows` when working memory is constrained.
    pub(crate) fn build(
        snapshot: &InputSnapshot,
        manifest: &IndexManifest,
        construction: ConstructionOptions,
        worker_memory_bytes: usize,
        check: &(dyn Fn() -> Result<()> + Sync),
    ) -> Result<Self> {
        check()?;
        let config = manifest.config();
        let row_bytes = config.dimension().saturating_mul(4).saturating_add(128);
        let effective_mem = worker_memory_bytes.min(construction.memory_bytes);
        if effective_mem < row_bytes.saturating_mul(16) {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }

        let total_items = snapshot.manifest().items();
        if total_items == 0 {
            return Ok(Self {
                id_bounds: Vec::new(),
                heavy_trees: Vec::new(),
                light_shard_start: 0,
                light_shard_count: 0,
                total_shards: 0,
                target_shard_rows: DEFAULT_SHARD_TARGET_ROWS as usize,
            });
        }

        let kernel = kernel_for(manifest)?;

        let min_shard_rows = (config.max_leaf_partition_entries() as usize)
            .saturating_mul(4)
            .max(64);
        let scale_floor = total_items.div_ceil(MAX_TOTAL_SHARDS as u64) as usize;
        let mem_shard_rows = (effective_mem / 4 / row_bytes.max(1)).max(min_shard_rows);
        let target_shard_rows = mem_shard_rows
            .min(DEFAULT_SHARD_TARGET_ROWS as usize)
            .max(scale_floor)
            .max(min_shard_rows);

        // Bound sample capacity by construction policy and available working memory.
        let mem_sample_cap = (effective_mem / 2 / row_bytes.max(1)).max(2);
        let sample_cap = construction
            .sample_items
            .clamp(4_096, MAX_SAMPLE_CAP)
            .min(mem_sample_cap);

        let segments = snapshot.segments();
        let mut heap: BinaryHeap<SampledRow> = BinaryHeap::new();
        let mut vec_buf = vec![0.0_f32; config.dimension()];

        let seg_stride = if segments.len() > sample_cap {
            (segments.len() / sample_cap.max(1)).max(1)
        } else {
            1
        };
        let visited_segs = segments.len().div_ceil(seg_stride).max(1);
        for seg_idx in (0..segments.len()).step_by(seg_stride) {
            check()?;
            let seg_meta = &segments[seg_idx];
            let stride = if total_items <= sample_cap as u64 {
                1
            } else {
                let per_seg_target =
                    ((sample_cap as u128 * 2) / visited_segs as u128).max(1) as u64;
                (seg_meta.items / per_seg_target).max(1)
            };
            let reader = if stride > 8 {
                snapshot.encoded_segment_strided(seg_idx)?
            } else {
                snapshot.encoded_segment(seg_idx)?
            };
            let mut visited = 0_u64;
            reader.visit_strided_frames(stride, |frame_bytes| {
                if visited.is_multiple_of(64) {
                    check()?;
                }
                visited += 1;
                let (id, fields, _) = source::decode_into_buf(config, frame_bytes, &mut vec_buf)?;
                let tree_key = source::project_tree_key(manifest, &fields)?;
                let rot_vec = kernel.preprocess(&vec_buf)?;
                let item = SampledRow {
                    hash: xxh3_128_with_seed(&id, SAMPLE_SEED),
                    id,
                    tree_key,
                    rot_vec,
                };
                if heap.len() < sample_cap {
                    heap.push(item);
                } else if heap.peek().is_some_and(|max| &item < max) {
                    heap.pop();
                    heap.push(item);
                }
                Ok(())
            })?;
        }

        let mut samples: Vec<SampledRow> = heap.into_vec();
        samples.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        let sample_len = samples.len().max(1);

        // Quantile Record ID bounds for balanced, non-overlapping ID buckets.
        let desired_buckets = total_items
            .div_ceil(TARGET_ID_BUCKET_ROWS)
            .clamp(1, MAX_ID_BUCKETS as u64) as usize;
        let mut unique_ids: Vec<&Bytes> = Vec::with_capacity(samples.len());
        for s in &samples {
            if unique_ids.last().is_none_or(|prev| *prev != &s.id) {
                unique_ids.push(&s.id);
            }
        }
        let bucket_count = desired_buckets.min(unique_ids.len().max(1));
        let mut id_bounds = Vec::with_capacity(bucket_count.saturating_sub(1));
        for b in 1..bucket_count {
            let idx = (b * unique_ids.len()) / bucket_count;
            let candidate = unique_ids[idx].clone();
            if id_bounds.last().is_none_or(|prev| prev < &candidate) {
                id_bounds.push(candidate);
            }
        }

        // Group sample by Tree Key to distinguish heavy trees from light trees.
        let mut by_tree: TreeSampleMap = BTreeMap::new();
        for s in samples {
            by_tree
                .entry(Bytes::copy_from_slice(s.tree_key.as_bytes()))
                .or_insert_with(|| (s.tree_key, Vec::new()))
                .1
                .push((s.id, s.rot_vec));
        }

        let reserved_light = if config.tree_key_fields().is_empty() {
            1_usize
        } else {
            MAX_LIGHT_SHARDS as usize
        };
        let max_heavy_budget = MAX_TOTAL_SHARDS - reserved_light;

        let mut heavy_candidates = Vec::new();
        let mut light_est_rows = 0_u64;
        for (_, (tree_key, tree_samples)) in by_tree {
            let est_rows =
                ((tree_samples.len() as u128 * total_items as u128) / sample_len as u128) as u64;
            if est_rows > target_shard_rows as u64 && tree_samples.len() >= 2 {
                let desired = est_rows
                    .div_ceil(target_shard_rows as u64)
                    .clamp(2, max_heavy_budget as u64) as usize;
                let k = desired.min(tree_samples.len()).max(2);
                heavy_candidates.push((tree_key, tree_samples, est_rows, k));
            } else {
                light_est_rows = light_est_rows.saturating_add(est_rows.max(1));
            }
        }

        // Enforce global `MAX_TOTAL_SHARDS` ceiling across all heavy trees combined.
        let sum_heavy_k: usize = heavy_candidates.iter().map(|c| c.3).sum();
        if sum_heavy_k > max_heavy_budget {
            // Keep the largest heavy trees if there are more than `max_heavy_budget / 2` candidates.
            if heavy_candidates.len() * 2 > max_heavy_budget {
                heavy_candidates.sort_by(|a, b| {
                    b.2.cmp(&a.2)
                        .then_with(|| a.0.as_bytes().cmp(b.0.as_bytes()))
                });
                for (_, _, est, _) in heavy_candidates.drain(max_heavy_budget / 2..) {
                    light_est_rows = light_est_rows.saturating_add(est.max(1));
                }
                heavy_candidates.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            }
            let remaining_sum: usize = heavy_candidates.iter().map(|c| c.3).sum();
            if remaining_sum > max_heavy_budget {
                let extra_budget = max_heavy_budget.saturating_sub(heavy_candidates.len() * 2);
                let extra_desired = remaining_sum
                    .saturating_sub(heavy_candidates.len() * 2)
                    .max(1);
                for cand in &mut heavy_candidates {
                    let over = cand.3.saturating_sub(2);
                    cand.3 = 2 + (over * extra_budget) / extra_desired;
                }
            }
        }

        let mut heavy_trees = Vec::with_capacity(heavy_candidates.len());
        let mut next_shard = 0_u32;
        for (tree_key, tree_samples, _, k) in heavy_candidates {
            check()?;
            let train_cap = construction
                .sample_items
                .max(k * 16)
                .min(tree_samples.len());
            let stride = (tree_samples.len() / train_cap.max(1)).max(1);
            let train_samples: Vec<_> = tree_samples
                .into_iter()
                .step_by(stride)
                .take(train_cap)
                .collect();
            let centers = train_centers_hierarchical(&kernel, train_samples, k, 8, check)?;
            let (super_centers, super_children) = build_super_index(&kernel, &centers, check)?;
            let shard_count =
                u32::try_from(centers.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            let shard_start = next_shard;
            next_shard = next_shard
                .checked_add(shard_count)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            heavy_trees.push(HeavyTreePlan {
                tree_key,
                shard_start,
                shard_count,
                centers,
                super_centers,
                super_children,
            });
        }

        // Always keep at least one light shard if the schema has Tree Key fields
        // (so unseen Tree Keys have a deterministic home) or when no heavy trees exist.
        let needs_light = !config.tree_key_fields().is_empty() || heavy_trees.is_empty();
        let remaining_shards = (MAX_TOTAL_SHARDS as u32).saturating_sub(next_shard);
        let light_shard_count = if needs_light {
            (light_est_rows
                .max(1)
                .div_ceil(target_shard_rows as u64)
                .clamp(1, u64::from(MAX_LIGHT_SHARDS)) as u32)
                .min(remaining_shards.max(1))
        } else {
            0
        };
        let light_shard_start = next_shard;
        let total_shards = next_shard
            .checked_add(light_shard_count)
            .filter(|&tot| (tot as usize) <= MAX_TOTAL_SHARDS)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;

        Ok(Self {
            id_bounds,
            heavy_trees,
            light_shard_start,
            light_shard_count,
            total_shards,
            target_shard_rows,
        })
    }

    #[must_use]
    pub(crate) fn id_bucket_count(&self) -> u32 {
        if self.total_shards == 0 {
            0
        } else {
            (self.id_bounds.len() + 1) as u32
        }
    }

    #[must_use]
    pub(crate) fn id_bucket(&self, id: &[u8]) -> u32 {
        self.id_bounds.partition_point(|bound| bound.as_ref() <= id) as u32
    }

    #[must_use]
    pub(crate) fn route_shard(
        &self,
        kernel: &VectorKernel,
        tree_key: &TreeKey,
        rot_vec: &[f32],
    ) -> u32 {
        if let Ok(idx) = self
            .heavy_trees
            .binary_search_by(|h| h.tree_key.as_bytes().cmp(tree_key.as_bytes()))
        {
            let heavy = &self.heavy_trees[idx];
            let local = if heavy.super_centers.is_empty() {
                nearest_center(kernel, rot_vec, &heavy.centers) as u32
            } else {
                let sc = nearest_center(kernel, rot_vec, &heavy.super_centers);
                nearest_indexed_center(kernel, rot_vec, &heavy.centers, &heavy.super_children[sc])
            };
            return heavy.shard_start + local;
        }
        let h = xxh3_64(tree_key.as_bytes());
        self.light_shard_start + (h % u64::from(self.light_shard_count.max(1))) as u32
    }

    /// Returns whether `shard` is one of several coarse shards of a heavy tree.
    #[must_use]
    pub(crate) fn heavy_tree_for_shard(&self, shard: u32) -> Option<&HeavyTreePlan> {
        self.heavy_trees
            .iter()
            .find(|h| shard >= h.shard_start && shard < h.shard_start + h.shard_count)
    }

    /// Ranks `shard` followed by its nearest sibling coarse shards in routing distance order.
    #[must_use]
    pub(crate) fn halo_shards(&self, kernel: &VectorKernel, shard: u32) -> Vec<u32> {
        let Some(heavy) = self.heavy_tree_for_shard(shard) else {
            return vec![shard];
        };
        let local = (shard - heavy.shard_start) as usize;
        let origin = &heavy.centers[local];
        let ranked: Vec<(f64, u32)> = if heavy.super_centers.len() >= 8 {
            let mut sc_ranked: Vec<(f64, usize)> = heavy
                .super_centers
                .iter()
                .enumerate()
                .map(|(sc_idx, sc)| (kernel.routing_distances_validated([sc], origin)[0], sc_idx))
                .collect();
            sc_ranked.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let mut cands = Vec::new();
            for &(_, sc_idx) in sc_ranked.iter().take(6) {
                cands.extend_from_slice(&heavy.super_children[sc_idx]);
            }
            cands.sort_unstable();
            cands.dedup();
            let mut sub: Vec<(f64, u32)> = cands
                .into_iter()
                .filter(|&c| c as usize != local)
                .map(|c| {
                    let dist =
                        kernel.routing_distances_validated([&heavy.centers[c as usize]], origin)[0];
                    (dist, heavy.shard_start + c)
                })
                .collect();
            sub.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            sub
        } else {
            let mut sub: Vec<(f64, u32)> = (0..heavy.shard_count)
                .filter(|&c| c as usize != local)
                .map(|c| {
                    let dist =
                        kernel.routing_distances_validated([&heavy.centers[c as usize]], origin)[0];
                    (dist, heavy.shard_start + c)
                })
                .collect();
            sub.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            sub
        };
        let mut out = Vec::with_capacity(1 + ranked.len());
        out.push(shard);
        out.extend(ranked.into_iter().map(|(_, s)| s));
        out
    }

    /// Persists the deterministic plan to `workspace/{token}/plan-e{epoch}.bin` under `quota`.
    pub(crate) fn save(&self, path: &Path, quota: &ScratchQuota) -> Result<u64> {
        let id_bounds_len = u32::try_from(self.id_bounds.len())
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let heavy_len = u32::try_from(self.heavy_trees.len())
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        quota.write_tracked(path, |w| {
            let mut hasher = Xxh3::new();
            let mut write_checked = |bytes: &[u8]| -> Result<()> {
                hasher.update(bytes);
                w.write_all(bytes).map_err(io_error)
            };
            write_checked(PLAN_MAGIC)?;
            write_checked(&(self.target_shard_rows as u64).to_be_bytes())?;
            write_checked(&self.light_shard_start.to_be_bytes())?;
            write_checked(&self.light_shard_count.to_be_bytes())?;
            write_checked(&self.total_shards.to_be_bytes())?;
            write_checked(&id_bounds_len.to_be_bytes())?;
            for bound in &self.id_bounds {
                let len =
                    u32::try_from(bound.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
                write_checked(&len.to_be_bytes())?;
                write_checked(bound)?;
            }
            write_checked(&heavy_len.to_be_bytes())?;
            for heavy in &self.heavy_trees {
                let tk = heavy.tree_key.as_bytes();
                let tk_len =
                    u32::try_from(tk.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
                let centers_len = u32::try_from(heavy.centers.len())
                    .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
                write_checked(&tk_len.to_be_bytes())?;
                write_checked(tk)?;
                write_checked(&heavy.shard_start.to_be_bytes())?;
                write_checked(&heavy.shard_count.to_be_bytes())?;
                write_checked(&centers_len.to_be_bytes())?;
                for center in &heavy.centers {
                    for &v in center.iter() {
                        if !v.is_finite() {
                            return Err(corrupt());
                        }
                        let bits = if v == 0.0 { 0_u32 } else { v.to_bits() };
                        write_checked(&bits.to_be_bytes())?;
                    }
                }
            }
            let checksum = hasher.digest();
            w.write_all(&checksum.to_be_bytes()).map_err(io_error)
        })
    }

    /// Loads and validates a persisted plan from `workspace/{token}/plan-e{epoch}.bin`.
    pub(crate) fn load(
        path: &Path,
        manifest: &IndexManifest,
        check: &(dyn Fn() -> Result<()> + Sync),
    ) -> Result<Self> {
        check()?;
        let file = File::open(path).map_err(io_error)?;
        let file_len = file.metadata().map_err(io_error)?.len();
        if file_len < 16 {
            return Err(corrupt());
        }
        let body_len = file_len - 8;
        let mut r = HashingReader {
            inner: BufReader::new(file).take(body_len),
            hasher: Xxh3::new(),
        };
        let mut magic = [0_u8; 8];
        r.read_exact(&mut magic).map_err(io_error)?;
        if &magic != PLAN_MAGIC {
            return Err(corrupt());
        }
        let target_shard_rows = usize::try_from(read_u64(&mut r)?).map_err(|_| corrupt())?;
        if target_shard_rows == 0 {
            return Err(corrupt());
        }
        let light_shard_start = read_u32(&mut r)?;
        let light_shard_count = read_u32(&mut r)?;
        let total_shards = read_u32(&mut r)?;
        if (total_shards as usize) > MAX_TOTAL_SHARDS
            || light_shard_count > MAX_LIGHT_SHARDS
            || light_shard_start.checked_add(light_shard_count) != Some(total_shards)
        {
            return Err(corrupt());
        }

        let id_bounds_len = read_u32(&mut r)? as usize;
        if id_bounds_len > MAX_ID_BUCKETS {
            return Err(corrupt());
        }
        let mut id_bounds = Vec::with_capacity(id_bounds_len);
        for _ in 0..id_bounds_len {
            let len = read_u32(&mut r)? as usize;
            if len == 0 || len > MAX_RECORD_ID_BYTES {
                return Err(corrupt());
            }
            let mut buf = vec![0_u8; len];
            r.read_exact(&mut buf).map_err(io_error)?;
            id_bounds.push(Bytes::from(buf));
        }
        if id_bounds.windows(2).any(|w| w[0] >= w[1]) {
            return Err(corrupt());
        }
        if total_shards == 0 && !id_bounds.is_empty() {
            return Err(corrupt());
        }

        let heavy_len = read_u32(&mut r)? as usize;
        if heavy_len > MAX_TOTAL_SHARDS / 2 {
            return Err(corrupt());
        }
        let (types, type_count) = manifest.tree_key_types();
        let tree_types = &types[..type_count];
        let config = manifest.config();
        let dim = config.dimension();
        let center_bytes = dim.checked_mul(4).ok_or_else(corrupt)?;
        let kernel = kernel_for(manifest)?;
        let mut heavy_trees: Vec<HeavyTreePlan> = Vec::with_capacity(heavy_len.min(1024));
        let mut expected_shard_start = 0_u32;

        for _ in 0..heavy_len {
            check()?;
            let tk_len = read_u32(&mut r)? as usize;
            if tk_len > MAX_TREE_KEY_BYTES {
                return Err(corrupt());
            }
            let mut tk_buf = vec![0_u8; tk_len];
            r.read_exact(&mut tk_buf).map_err(io_error)?;
            let tree_key = TreeKey::from_encoded(tree_types, Bytes::from(tk_buf))?;
            if let Some(prev) = heavy_trees.last()
                && prev.tree_key.as_bytes() >= tree_key.as_bytes()
            {
                return Err(corrupt());
            }

            let shard_start = read_u32(&mut r)?;
            let shard_count = read_u32(&mut r)?;
            let centers_len = read_u32(&mut r)? as usize;
            if shard_start != expected_shard_start
                || shard_count < 2
                || centers_len != shard_count as usize
                || (centers_len as u64).saturating_mul(center_bytes as u64) > file_len
            {
                return Err(corrupt());
            }
            expected_shard_start = expected_shard_start
                .checked_add(shard_count)
                .ok_or_else(corrupt)?;
            if expected_shard_start > light_shard_start {
                return Err(corrupt());
            }

            let mut centers = Vec::with_capacity(centers_len);
            let mut raw = vec![0_u8; center_bytes];
            for c_idx in 0..centers_len {
                if c_idx.is_multiple_of(256) {
                    check()?;
                }
                r.read_exact(&mut raw).map_err(io_error)?;
                let mut vec = Vec::with_capacity(dim);
                for b in raw.as_chunks::<4>().0 {
                    let bits = u32::from_be_bytes(*b);
                    let val = f32::from_bits(bits);
                    if !val.is_finite() || bits == 0x8000_0000 {
                        return Err(corrupt());
                    }
                    vec.push(val);
                }
                centers.push(vec.into_boxed_slice());
            }
            let (super_centers, super_children) = build_super_index(&kernel, &centers, check)?;
            heavy_trees.push(HeavyTreePlan {
                tree_key,
                shard_start,
                shard_count,
                centers,
                super_centers,
                super_children,
            });
        }
        if expected_shard_start != light_shard_start {
            return Err(corrupt());
        }
        if config.tree_key_fields().is_empty() && heavy_trees.len() > 1 {
            return Err(corrupt());
        }
        let mut extra = [0_u8; 1];
        if r.read(&mut extra).map_err(io_error)? != 0 {
            return Err(corrupt());
        }
        let actual_digest = r.hasher.digest();
        let mut inner = r.inner.into_inner();
        let mut expected_b = [0_u8; 8];
        inner.read_exact(&mut expected_b).map_err(io_error)?;
        if u64::from_be_bytes(expected_b) != actual_digest
            || inner.read(&mut extra).map_err(io_error)? != 0
        {
            return Err(corrupt());
        }
        Ok(Self {
            id_bounds,
            heavy_trees,
            light_shard_start,
            light_shard_count,
            total_shards,
            target_shard_rows,
        })
    }
}

struct HashingReader<R> {
    inner: R,
    hasher: Xxh3,
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0_u8; 4];
    r.read_exact(&mut b).map_err(io_error)?;
    Ok(u32::from_be_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0_u8; 8];
    r.read_exact(&mut b).map_err(io_error)?;
    Ok(u64::from_be_bytes(b))
}

pub(crate) fn nearest_center(
    kernel: &VectorKernel,
    vector: &[f32],
    centers: &[Box<[f32]>],
) -> usize {
    let mut best = 0;
    let mut best_dist = f64::INFINITY;
    for (chunk_idx, chunk) in centers.chunks(4).enumerate() {
        let dists = match chunk {
            [a, b, c, d] => kernel.routing_distances_validated([a, b, c, d], vector),
            [a, b, c] => {
                let d = kernel.routing_distances_validated([a, b, c], vector);
                [d[0], d[1], d[2], f64::INFINITY]
            }
            [a, b] => {
                let d = kernel.routing_distances_validated([a, b], vector);
                [d[0], d[1], f64::INFINITY, f64::INFINITY]
            }
            [a] => [
                kernel.routing_distances_validated([vector], a)[0],
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
            ],
            _ => unreachable!("nonempty chunk"),
        };
        for (lane, candidate) in dists.into_iter().take(chunk.len()).enumerate() {
            if candidate.total_cmp(&best_dist).is_lt() {
                best = chunk_idx * 4 + lane;
                best_dist = candidate;
            }
        }
    }
    best
}

fn nearest_indexed_center(
    kernel: &VectorKernel,
    vector: &[f32],
    centers: &[Box<[f32]>],
    indices: &[u32],
) -> u32 {
    let mut best = indices[0];
    let mut best_dist = f64::INFINITY;
    for chunk in indices.chunks(4) {
        let dists = match *chunk {
            [a, b, c, d] => kernel.routing_distances_validated(
                [
                    &centers[a as usize],
                    &centers[b as usize],
                    &centers[c as usize],
                    &centers[d as usize],
                ],
                vector,
            ),
            [a, b, c] => {
                let d = kernel.routing_distances_validated(
                    [
                        &centers[a as usize],
                        &centers[b as usize],
                        &centers[c as usize],
                    ],
                    vector,
                );
                [d[0], d[1], d[2], f64::INFINITY]
            }
            [a, b] => {
                let d = kernel.routing_distances_validated(
                    [&centers[a as usize], &centers[b as usize]],
                    vector,
                );
                [d[0], d[1], f64::INFINITY, f64::INFINITY]
            }
            [a] => [
                kernel.routing_distances_validated([vector], &centers[a as usize])[0],
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
            ],
            _ => unreachable!("nonempty chunk"),
        };
        for (&idx, candidate) in chunk.iter().zip(dists) {
            if candidate.total_cmp(&best_dist).is_lt()
                || (candidate.total_cmp(&best_dist).is_eq() && idx < best)
            {
                best = idx;
                best_dist = candidate;
            }
        }
    }
    best
}

/// Two-level hierarchical K-means when `k > 64` so large shard counts (`k <= 65,535`)
/// train in `O(N * sqrt(k))` distance evaluations instead of `O(N * k)`.
fn train_centers_hierarchical(
    kernel: &VectorKernel,
    entries: Vec<(Bytes, Box<[f32]>)>,
    k: usize,
    max_iters: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<Vec<Box<[f32]>>> {
    if k <= 64 || entries.len() < k.saturating_mul(2) {
        return train_centers(kernel, entries, k, max_iters, check);
    }
    let super_k = (k as f64).sqrt().ceil().clamp(2.0, 256.0) as usize;
    let stride = (entries.len() / (super_k * 32).max(1)).max(1);
    let super_sample: Vec<_> = entries.iter().step_by(stride).cloned().collect();
    let super_centers = train_centers(kernel, super_sample, super_k, max_iters, check)?;

    let mut buckets: Vec<Vec<(Bytes, Box<[f32]>)>> = vec![Vec::new(); super_centers.len()];
    for (idx, item) in entries.into_iter().enumerate() {
        if idx.is_multiple_of(256) {
            check()?;
        }
        let b = nearest_center(kernel, &item.1, &super_centers);
        buckets[b].push(item);
    }
    buckets.retain(|b| !b.is_empty());
    let total_entries: usize = buckets.iter().map(Vec::len).sum();
    if buckets.is_empty() || total_entries == 0 {
        return Ok(super_centers);
    }

    // Allocate sub-cluster counts `>= 1` proportionally so `sum(sub_k) == k`.
    let mut sub_ks: Vec<usize> = buckets
        .iter()
        .map(|b| ((b.len() * k) / total_entries).clamp(1, b.len()))
        .collect();
    let mut current_sum: usize = sub_ks.iter().sum();
    while current_sum > k {
        if let Some((idx, _)) = sub_ks
            .iter()
            .enumerate()
            .filter(|&(_, &sk)| sk > 1)
            .max_by_key(|&(i, &sk)| (sk, buckets[i].len()))
        {
            sub_ks[idx] -= 1;
            current_sum -= 1;
        } else {
            break;
        }
    }
    while current_sum < k {
        if let Some((idx, _)) = sub_ks
            .iter()
            .enumerate()
            .filter(|&(i, &sk)| sk < buckets[i].len())
            .max_by_key(|&(i, &sk)| buckets[i].len() / (sk + 1))
        {
            sub_ks[idx] += 1;
            current_sum += 1;
        } else {
            break;
        }
    }

    let mut all_centers = Vec::with_capacity(current_sum);
    for (bucket, sub_k) in buckets.into_iter().zip(sub_ks) {
        check()?;
        if sub_k == 1 {
            let mut mean = vec![0.0_f64; kernel.dimension()];
            for (_, v) in &bucket {
                for (s, x) in mean.iter_mut().zip(v.iter()) {
                    *s += f64::from(*x);
                }
            }
            let inv = 1.0 / bucket.len() as f64;
            let m: Vec<f32> = mean.into_iter().map(|s| (s * inv) as f32).collect();
            all_centers.push(kernel.normalize_centroid(&m)?);
        } else {
            all_centers.extend(train_centers(kernel, bucket, sub_k, max_iters, check)?);
        }
    }
    Ok(all_centers)
}

/// Deterministic farthest-first + Lloyd K-means center training shared by coarse shard
/// planning, leaf construction, and bottom-up internal partition grouping.
pub(crate) fn train_centers(
    kernel: &VectorKernel,
    mut entries: Vec<(Bytes, Box<[f32]>)>,
    k: usize,
    max_iters: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<Vec<Box<[f32]>>> {
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    if kernel.is_cosine() {
        for (_, vec) in &mut entries {
            *vec = kernel.normalize_centroid(vec)?;
        }
    }
    let dim = kernel.dimension();
    let mut mean = vec![0.0_f64; dim];
    for (_, vec) in &entries {
        for (s, v) in mean.iter_mut().zip(vec.iter()) {
            *s += f64::from(*v);
        }
    }
    let mean: Vec<f32> = mean
        .into_iter()
        .map(|v| (v / entries.len() as f64) as f32)
        .collect();

    let mut seeds: Vec<Box<[f32]>> = Vec::with_capacity(k);
    let mut min_dists = vec![f64::INFINITY; entries.len()];
    for seed_idx in 0..k {
        if seed_idx.is_multiple_of(16) {
            check()?;
        }
        let mut farthest = 0;
        let mut max_d = f64::NEG_INFINITY;
        let prev = if seed_idx == 0 {
            &mean[..]
        } else {
            &seeds[seed_idx - 1][..]
        };
        for (i, (_, vec)) in entries.iter().enumerate() {
            if i.is_multiple_of(4096) {
                check()?;
            }
            let d = kernel.routing_distances_validated([vec], prev)[0];
            min_dists[i] = if seed_idx <= 1 {
                d
            } else {
                min_dists[i].min(d)
            };
            if min_dists[i].total_cmp(&max_d).is_gt() {
                max_d = min_dists[i];
                farthest = i;
            }
        }
        seeds.push(entries[farthest].1.clone());
    }

    let mut previous = vec![usize::MAX; entries.len()];
    let mut sums = vec![vec![0.0_f64; dim]; k];
    let mut counts = vec![0_u64; k];
    for _ in 0..max_iters {
        check()?;
        for sum in &mut sums {
            sum.fill(0.0);
        }
        counts.fill(0);
        let mut changed = false;
        for (i, (_, vec)) in entries.iter().enumerate() {
            if i.is_multiple_of(256) {
                check()?;
            }
            let label = nearest_center(kernel, vec, &seeds);
            changed |= previous[i] != label;
            previous[i] = label;
            counts[label] += 1;
            for (s, v) in sums[label].iter_mut().zip(vec.iter()) {
                *s += f64::from(*v);
            }
        }
        if !changed {
            break;
        }
        for (label, sum) in sums.iter().enumerate() {
            if counts[label] != 0 {
                let m: Vec<f32> = sum
                    .iter()
                    .map(|v| (v / counts[label] as f64) as f32)
                    .collect();
                seeds[label] = kernel.normalize_centroid(&m)?;
            }
        }
    }
    Ok(seeds)
}

fn build_super_index(
    kernel: &VectorKernel,
    centers: &[Box<[f32]>],
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<SuperIndex> {
    if centers.len() < 32 {
        return Ok((Vec::new(), Vec::new()));
    }
    let super_k = (centers.len() as f64).sqrt().ceil() as usize;
    let entries: Vec<(Bytes, Box<[f32]>)> = centers
        .iter()
        .enumerate()
        .map(|(i, c)| (Bytes::copy_from_slice(&(i as u32).to_be_bytes()), c.clone()))
        .collect();
    let super_centers = train_centers_hierarchical(kernel, entries, super_k, 8, check)?;
    let mut super_children = vec![Vec::new(); super_centers.len()];
    for (idx, c) in centers.iter().enumerate() {
        if idx.is_multiple_of(256) {
            check()?;
        }
        // Assign each coarse center to its 3 nearest super-centers to mitigate Voronoi boundary loss.
        let mut best1 = (f64::INFINITY, 0_usize);
        let mut best2 = (f64::INFINITY, 0_usize);
        let mut best3 = (f64::INFINITY, 0_usize);
        for (sc_idx, sc) in super_centers.iter().enumerate() {
            let d = kernel.routing_distances_validated([c], sc)[0];
            if d.total_cmp(&best1.0).is_lt() {
                best3 = best2;
                best2 = best1;
                best1 = (d, sc_idx);
            } else if d.total_cmp(&best2.0).is_lt() {
                best3 = best2;
                best2 = (d, sc_idx);
            } else if d.total_cmp(&best3.0).is_lt() {
                best3 = (d, sc_idx);
            }
        }
        super_children[best1.1].push(idx as u32);
        if super_centers.len() > 1 && best2.1 != best1.1 {
            super_children[best2.1].push(idx as u32);
        }
        if super_centers.len() > 2 && best3.1 != best1.1 && best3.1 != best2.1 {
            super_children[best3.1].push(idx as u32);
        }
    }
    for list in &mut super_children {
        if list.is_empty() {
            list.push(0);
        }
    }
    Ok((super_centers, super_children))
}
