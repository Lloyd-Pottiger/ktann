//! Disk-spilling and multi-worker collaborative `BuildUnit` execution.
//!
//! Each stage partitions work into bounded, deterministic units. In collaborative mode
//! (`distributed_workers > 1`), units are leased and renewed in the KV backend so worker
//! processes claim disjoint units concurrently and reclaim expired units after stalls.
//! In single-worker disk mode (`distributed_workers == 1`), units execute directly via an
//! in-process queue with writer-lane fencing and zero intermediate `BuildUnit` KV traffic.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;

use crate::api::{BulkBuildReport, Error, ErrorKind, OperationOptions, PartitionKey, Result};
use crate::bulk::files::io_error;
use crate::bulk::sharded::{
    AssignedLeafKey, MoveProposal, ScratchQuota, ShardRows, ShardState, TreeLeafTotals,
    assign_leaf_partition_keys_from_trees, build_and_encode_upper_tree, build_shard_leaves,
    compute_leaf_centroid, compute_shard_remap, encode_leaf_shard, encode_record_group_bucket,
    ingest_segment_range, read_bucket_id_refs, read_compact_shard_state, read_indexed_ids_slice,
    read_indexed_locations_slice, read_indexed_outbound_slice, read_indexed_shard_slice,
    read_indexed_upper_children_slice, read_proposals, read_shard_centers, read_shard_counts,
    read_shard_leaf_trees, refine_single_light_shard, refinement_per_leaf_bytes, score_halo_shard,
    verify_id_bucket, write_base_shard_rows, write_bucket_id_refs, write_compact_shard_state,
    write_indexed_ids, write_indexed_locations, write_indexed_outbound_rows, write_indexed_shards,
    write_indexed_upper_children, write_proposals,
};
use crate::bulk::{InputSnapshot, LtHash, ShardPlan};
use crate::observe::labels::Operation;
use crate::refinement::accept_moves;
use crate::runtime::bulk_scheduler::{BulkAuthority, authorize_raw, expiry, now_ms};
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{AdmissionBudget, Backend, ScanLimits};
use crate::storage::keys::{LogicalKey, TreeKey, tree_key_hash};
use crate::storage::values::{
    BuildDescriptor, BuildPhase, BuildUnit, BuildWorkspace, IndexLifecycle, IndexManifest,
    PartitionCentroid, PersistentValue,
};
use crate::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

use super::{
    OperationContext, RetryPolicy, UnitLease, WriteFence, build_validation_split_keys,
    building_in_flight, check_lane_fence, clear_stale_serving_data, corrupt, drop_guarded_options,
    effective_budget, seal_loaded,
};

const STAGE_SCATTER: u8 = 1;
const STAGE_DEDUP: u8 = 2;
const STAGE_LEAVES: u8 = 3;
const STAGE_REFINE_BASE: u8 = 10; // 10 + 3*r = Propose, 11 + 3*r = Extract, 12 + 3*r = Apply
const STAGE_EMIT_LEAVES: u8 = 30;
const STAGE_EMIT_RECORDS: u8 = 31;
const STAGE_EMIT_TREES: u8 = 32;

type UpperTreeBucket = (TreeKey, u64, Vec<(PartitionKey, PartitionCentroid)>);
type NonEmptyRunSources = Vec<(u32, Vec<usize>)>;

#[derive(Default)]
struct UnitOutput {
    records: u64,
    kv_count: u64,
    aux_count: u64,
    kv_lthash: LtHash,
    aux_lthash: LtHash,
    aux2_lthash: LtHash,
}

struct StageEnv<B: Backend> {
    backend: Arc<B>,
    control: OperationOptions,
    bulk_authority: Option<BulkAuthority>,
    sched_lease_lock: Option<Arc<tokio::sync::RwLock<()>>>,
    distributed_workers: usize,
    admission: Option<Arc<dyn Any + Send + Sync>>,
    manifest: IndexManifest,
    fence: WriteFence,
    unit_lease: Duration,
    budget: AdmissionBudget,
    retry: RetryPolicy,
}

impl<B: Backend> Clone for StageEnv<B> {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            control: self.control.clone(),
            bulk_authority: self.bulk_authority,
            sched_lease_lock: self.sched_lease_lock.clone(),
            distributed_workers: self.distributed_workers,
            admission: self.admission.clone(),
            manifest: self.manifest.clone(),
            fence: self.fence,
            unit_lease: self.unit_lease,
            budget: self.budget,
            retry: self.retry,
        }
    }
}

impl<B: Backend> StageEnv<B> {
    async fn spawn_cpu<T: Send + 'static>(
        &self,
        work: impl FnOnce(OperationOptions) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        crate::runtime::check_control(&self.control)?;
        let adm = self.admission.clone();
        let (ctrl, _drop_guard) = drop_guarded_options(&self.control);
        tokio::task::spawn_blocking(move || {
            let _adm = adm;
            crate::runtime::check_control(&ctrl)?;
            work(ctrl)
        })
        .await
        .map_err(|e| Error::with_source(ErrorKind::Other, e))?
    }

    async fn commit_unit_kvs(
        &self,
        lease: Option<&UnitLease>,
        lane_hint: u32,
        kvs: Vec<(Bytes, Bytes)>,
    ) -> Result<()> {
        if kvs.is_empty() {
            return Ok(());
        }
        let chunks = super::split_into_chunks(&kvs, self.budget, lease.is_some())?;
        let lane = lane_hint % super::FENCE_LANES;
        for chunk in chunks {
            super::commit_one_chunk(
                Arc::clone(&self.backend),
                &self.control,
                self.bulk_authority,
                self.sched_lease_lock.as_deref(),
                &self.manifest,
                self.fence,
                lane,
                lease,
                chunk,
                self.budget,
                self.retry,
            )
            .await?;
        }
        Ok(())
    }
}

/// Removes leftover temporary files and prior-epoch scratch files when a single-worker
/// build starts a fresh epoch in an existing workspace child directory.
fn prune_stale_epoch_files(dir: &Path, epoch: u64) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_error(e)),
    };
    let epoch_tag = format!("-e{epoch}-");
    let plan_name = format!("plan-e{epoch}.bin");
    for entry in entries {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name == ".ktann-build-lock" || name == plan_name || name.contains(&epoch_tag) {
            continue;
        }
        let _ = fs::remove_file(path);
    }
    Ok(())
}

pub(crate) async fn run_disk_units<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &BuildWorkspace,
    fence: WriteFence,
    lock: Arc<File>,
    retry: RetryPolicy,
) -> Result<BulkBuildReport> {
    let mut report = BulkBuildReport {
        in_memory_mode: false,
        rotation_seed: *index.rotation_seed(),
        ..BulkBuildReport::default()
    };
    let parallelism = state.options.parallelism.max(1);
    let budget = effective_budget(context.backend().admission_budget(), state.options.load)?;
    let effective_mem = state
        .options
        .memory_bytes
        .min(descriptor.options().memory_bytes);
    let quota = Arc::new(ScratchQuota::new(
        state
            .options
            .scratch_bytes
            .min(descriptor.options().scratch_bytes),
    ));
    let work_dir = state.directory();
    let hard_limits = state.hard_limits;
    let graph_memory_bytes = state.options.graph_memory_bytes;

    if fence.epoch > 1 && state.options.distributed_workers == 1 {
        clear_stale_serving_data(context, index, fence, budget, retry).await?;
    }

    // 1. Open InputSnapshot, seed ScratchQuota, and load or create deterministic `plan-e{epoch}.bin`.
    let plan_start = Instant::now();
    let (snapshot, plan) = {
        let desc = descriptor.clone();
        let manifest = index.clone();
        let dir = work_dir.clone();
        let single_worker = state.options.distributed_workers == 1;
        let guard = lock.clone();
        let quota_c = Arc::clone(&quota);
        context
            .compute(move |control| {
                let _lock = guard;
                let check = || crate::runtime::check_control(control);
                if single_worker {
                    prune_stale_epoch_files(&dir, fence.epoch)?;
                }
                quota_c.seed_existing(&dir)?;
                let snap = InputSnapshot::open(
                    desc.source(),
                    manifest.config().clone(),
                    desc.input().clone(),
                )?;
                let plan_path = dir.join(format!("plan-e{}.bin", fence.epoch));
                let plan = if plan_path.exists() {
                    let row_bytes = manifest
                        .config()
                        .dimension()
                        .saturating_mul(4)
                        .saturating_add(128);
                    if effective_mem < row_bytes.saturating_mul(16) {
                        return Err(Error::new(ErrorKind::LimitExceeded));
                    }
                    ShardPlan::load(&plan_path, &manifest, &check)?
                } else {
                    let built =
                        ShardPlan::build(&snap, &manifest, desc.options(), effective_mem, &check)?;
                    built.save(&plan_path, &quota_c)?;
                    built
                };
                Ok((snap, plan))
            })
            .await?
    };
    report.plan = plan_start.elapsed();

    if let Some(rounds) = descriptor.refinement_rounds() {
        let per_leaf = refinement_per_leaf_bytes(index.config().dimension(), rounds);
        if effective_mem < per_leaf.saturating_mul(2) {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
    }

    if snapshot.manifest().items() == 0 {
        seal_loaded(
            context,
            index,
            fence,
            0,
            0,
            0,
            0,
            LtHash::ZERO,
            Vec::new(),
            retry,
        )
        .await?;
        report.scratch_written_bytes = quota.written();
        report.peak_scratch_bytes = quota.peak();
        return Ok(report);
    }

    let seg_mem_est = (16_384_usize)
        .saturating_mul(index.config().dimension() * 4 + 256)
        .max(1);
    let scatter_conc = (effective_mem / seg_mem_est).clamp(1, parallelism.min(8));
    let seg_count = snapshot.segments().len();
    let max_segs_by_mem = (effective_mem / scatter_conc.max(1) / seg_mem_est).clamp(1, 32);
    let segs_per_run = seg_count.div_ceil(2_048).max(1).min(max_segs_by_mem);
    let scatter_units = u32::try_from(seg_count.div_ceil(segs_per_run))
        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;

    let id_buckets = plan.id_bucket_count();
    let coarse_shards = plan.total_shards;
    let tree_buckets = id_buckets.clamp(1, 64);

    let env = StageEnv {
        backend: context.backend(),
        control: context.options.clone(),
        bulk_authority: context.bulk_authority,
        sched_lease_lock: context.bulk_lease_lock.clone(),
        distributed_workers: state.options.distributed_workers,
        admission: context.cpu_keep_alive(),
        manifest: index.clone(),
        fence,
        unit_lease: state.options.unit_lease,
        budget,
        retry,
    };

    let leaf_conc = (effective_mem / (96 * 1024 * 1024)).clamp(1, parallelism.min(8));
    let emit_conc = if state.options.distributed_workers == 1 {
        if context.backend().capabilities().transactional_clear_range {
            (effective_mem / (160 * 1024 * 1024)).clamp(1, 2)
        } else {
            1
        }
    } else {
        parallelism.min(8)
    };

    // 2. Stage 1: Scatter grouped segment runs into indexed Coarse-Shard `vec-*` and ID-bucket `ids-*` files.
    let ingest_start = Instant::now();
    init_stage_units(&env, STAGE_SCATTER, scatter_units).await?;
    let scatter_done = drive_stage(&env, STAGE_SCATTER, scatter_units, scatter_conc, {
        let env = env.clone();
        let snapshot = snapshot.clone();
        let plan = plan.clone();
        let work_dir = work_dir.clone();
        let quota = Arc::clone(&quota);
        let lock = Arc::clone(&lock);
        move |unit, _lease| {
            let env_c = env.clone();
            let snap_c = snapshot.clone();
            let plan_c = plan.clone();
            let dir = work_dir.clone();
            let quota_c = Arc::clone(&quota);
            let guard = lock.clone();
            async move {
                let (records, id_lt) = env_c
                    .spawn_cpu({
                        let manifest = env_c.manifest.clone();
                        move |ctrl| {
                            let _lock = guard;
                            let check = || crate::runtime::check_control(&ctrl);
                            let seg_start = (unit as usize) * segs_per_run;
                            let seg_end = (seg_start + segs_per_run).min(seg_count);
                            let seg = ingest_segment_range(
                                &snap_c,
                                seg_start..seg_end,
                                &manifest,
                                &plan_c,
                                hard_limits,
                                &check,
                            )?;
                            let ep = fence.epoch;
                            write_indexed_ids(
                                &dir.join(format!("ids-e{ep}-{unit:06}.bin")),
                                &seg.ids_by_bucket,
                                &quota_c,
                            )?;
                            write_indexed_shards(
                                &dir.join(format!("vec-e{ep}-{unit:06}.bin")),
                                &dir.join(format!("cnt-e{ep}-{unit:06}.bin")),
                                &manifest,
                                &seg.rows_by_shard,
                                &quota_c,
                            )?;
                            Ok((seg.records, seg.id_lthash))
                        }
                    })
                    .await?;
                Ok(UnitOutput {
                    records,
                    aux_lthash: id_lt,
                    ..UnitOutput::default()
                })
            }
        }
    })
    .await?;
    report.ingest = ingest_start.elapsed();

    // 3. Stage 2 (Global ID Dedup -> `idref-*`) & Stage 3 (Sharded Leaf Construction -> `base-*` + `mem-r0-*`).
    let construct_start = Instant::now();
    init_stage_units(&env, STAGE_DEDUP, id_buckets).await?;
    let dedup_done = drive_stage(&env, STAGE_DEDUP, id_buckets, parallelism, {
        let env = env.clone();
        let work_dir = work_dir.clone();
        let quota = Arc::clone(&quota);
        let lock = Arc::clone(&lock);
        move |b, _lease| {
            let env_c = env.clone();
            let dir = work_dir.clone();
            let quota_c = Arc::clone(&quota);
            let guard = lock.clone();
            async move {
                let (count, id_lt) = env_c
                    .spawn_cpu(move |ctrl| {
                        let _lock = guard;
                        let ep = fence.epoch;
                        let mut ids = Vec::new();
                        for run in 0..scatter_units {
                            if run.is_multiple_of(32) {
                                crate::runtime::check_control(&ctrl)?;
                            }
                            read_indexed_ids_slice(
                                &dir.join(format!("ids-e{ep}-{run:06}.bin")),
                                b as usize,
                                &mut ids,
                            )?;
                        }
                        let (count, hash, sorted) = verify_id_bucket(ids)?;
                        write_bucket_id_refs(
                            &dir.join(format!("idref-e{ep}-{b:06}.bin")),
                            &sorted,
                            &quota_c,
                        )?;
                        Ok((count, hash))
                    })
                    .await?;
                Ok(UnitOutput {
                    records: count,
                    aux_lthash: id_lt,
                    ..UnitOutput::default()
                })
            }
        }
    })
    .await?;

    let scatter_records: u64 = scatter_done.iter().map(|u| u.records).sum();
    let dedup_records: u64 = dedup_done.iter().map(|u| u.records).sum();
    let mut scatter_id_lt = LtHash::ZERO;
    for u in &scatter_done {
        scatter_id_lt.add_assign(u.aux_lthash);
    }
    let mut dedup_id_lt = LtHash::ZERO;
    for u in &dedup_done {
        dedup_id_lt.add_assign(u.aux_lthash);
    }
    if scatter_records != dedup_records || scatter_id_lt != dedup_id_lt {
        return Err(corrupt());
    }

    reclaim_files(
        &env,
        &work_dir,
        &quota,
        &lock,
        (0..scatter_units)
            .map(|run| format!("ids-e{}-{run:06}.bin", fence.epoch))
            .collect(),
    )
    .await?;

    // Compute deterministic shard remap and exact non-empty `(run, Vec<orig_shard>)` sources from `cnt-*`.
    let non_empty_runs_by_shard: Arc<Vec<NonEmptyRunSources>> = Arc::new(
        env.spawn_cpu({
            let dir = work_dir.clone();
            let manifest = index.clone();
            let plan_c = plan.clone();
            let guard = lock.clone();
            move |ctrl| {
                let _lock = guard;
                let ep = fence.epoch;
                let mut totals = vec![0_u64; coarse_shards as usize];
                let mut per_run_non_empty: Vec<Vec<usize>> =
                    Vec::with_capacity(scatter_units as usize);
                for run in 0..scatter_units {
                    if run.is_multiple_of(64) {
                        crate::runtime::check_control(&ctrl)?;
                    }
                    let c = read_shard_counts(
                        &dir.join(format!("cnt-e{ep}-{run:06}.bin")),
                        coarse_shards as usize,
                    )?;
                    let mut run_origins = Vec::new();
                    for (orig, v) in c.into_iter().enumerate() {
                        if v > 0 {
                            totals[orig] = totals[orig].checked_add(v).ok_or_else(corrupt)?;
                            run_origins.push(orig);
                        }
                    }
                    per_run_non_empty.push(run_origins);
                }
                let kernel = VectorKernel::new(
                    manifest.config().dimension(),
                    manifest.config().metric(),
                    *manifest.rotation_seed(),
                )?;
                let remap = compute_shard_remap(
                    &kernel,
                    &plan_c,
                    &totals,
                    u64::from(manifest.config().min_leaf_partition_entries()),
                    u64::from(manifest.config().max_leaf_partition_entries()),
                );
                let mut by_dst: Vec<NonEmptyRunSources> = vec![Vec::new(); coarse_shards as usize];
                for (run_idx, origins) in per_run_non_empty.into_iter().enumerate() {
                    let run_u32 = run_idx as u32;
                    for orig in origins {
                        let dst = remap[orig] as usize;
                        match by_dst[dst].last_mut() {
                            Some((last_run, list)) if *last_run == run_u32 => list.push(orig),
                            _ => by_dst[dst].push((run_u32, vec![orig])),
                        }
                    }
                }
                Ok(by_dst)
            }
        })
        .await?,
    );

    let refinement_rounds = descriptor.refinement_rounds();
    let sample_items = descriptor.options().sample_items;
    let light_per_worker_mem = effective_mem / leaf_conc.max(1);

    init_stage_units(&env, STAGE_LEAVES, coarse_shards).await?;
    drive_stage(&env, STAGE_LEAVES, coarse_shards, leaf_conc, {
        let env = env.clone();
        let work_dir = work_dir.clone();
        let run_sources = Arc::clone(&non_empty_runs_by_shard);
        let plan = plan.clone();
        let quota = Arc::clone(&quota);
        let lock = Arc::clone(&lock);
        move |s, _lease| {
            let env_c = env.clone();
            let dir = work_dir.clone();
            let sources_c = Arc::clone(&run_sources);
            let plan_c = plan.clone();
            let quota_c = Arc::clone(&quota);
            let guard = lock.clone();
            async move {
                let (row_count, leaf_count) = env_c
                    .spawn_cpu({
                        let manifest = env_c.manifest.clone();
                        move |ctrl| {
                            let _lock = guard;
                            let check = || crate::runtime::check_control(&ctrl);
                            let ep = fence.epoch;
                            let run_list = &sources_c[s as usize];
                            let mut rows = ShardRows::new(manifest.config().dimension());
                            for (idx, (run, src_shards)) in run_list.iter().enumerate() {
                                if idx.is_multiple_of(32) {
                                    check()?;
                                }
                                let vpath = dir.join(format!("vec-e{ep}-{run:06}.bin"));
                                for &src_s in src_shards {
                                    read_indexed_shard_slice(&vpath, src_s, &manifest, &mut rows)?;
                                }
                            }
                            let base_count = rows.rows.len();
                            let mut shard_state = if base_count == 0 {
                                ShardState {
                                    rows,
                                    leaves: Vec::new(),
                                }
                            } else {
                                build_shard_leaves(&manifest, sample_items, rows, &check)?
                            };
                            if let Some(rounds) = refinement_rounds
                                && s >= plan_c.light_shard_start
                            {
                                refine_single_light_shard(
                                    &manifest,
                                    &mut shard_state,
                                    rounds,
                                    light_per_worker_mem,
                                    &check,
                                )?;
                            }
                            write_base_shard_rows(
                                &dir.join(format!("base-e{ep}-{s:06}.bin")),
                                &manifest,
                                &shard_state.rows,
                                base_count,
                                &quota_c,
                            )?;
                            write_compact_shard_state(
                                &dir.join(format!("mem-e{ep}-r0-{s:06}.bin")),
                                &dir.join(format!("centers-e{ep}-r0-{s:06}.bin")),
                                &dir.join(format!("inb-e{ep}-r0-{s:06}.bin")),
                                &manifest,
                                &shard_state,
                                base_count,
                                &quota_c,
                            )?;
                            Ok((base_count as u64, shard_state.leaves.len() as u64))
                        }
                    })
                    .await?;
                Ok(UnitOutput {
                    records: row_count,
                    aux_count: leaf_count,
                    ..UnitOutput::default()
                })
            }
        }
    })
    .await?;
    drop(non_empty_runs_by_shard);

    // Reclaim scatter runs (`vec-*` + `cnt-*`) immediately after the `STAGE_LEAVES` barrier!
    {
        let ep = fence.epoch;
        let mut obsolete = Vec::with_capacity((scatter_units as usize) * 2);
        for run in 0..scatter_units {
            obsolete.push(format!("vec-e{ep}-{run:06}.bin"));
            obsolete.push(format!("cnt-e{ep}-{run:06}.bin"));
        }
        reclaim_files(&env, &work_dir, &quota, &lock, obsolete).await?;
    }
    report.construct = construct_start.elapsed();

    // 4. Optional Bounded-Halo Delta Refinement across heavy-tree shards (`0..plan.light_shard_start`).
    let mut final_round = 0_u8;
    let heavy_shards = plan.light_shard_start;
    let rounds = refinement_rounds.unwrap_or(0);
    if refinement_rounds.is_some() && heavy_shards > 0 {
        let refine_start = Instant::now();
        let per_leaf = refinement_per_leaf_bytes(index.config().dimension(), rounds);
        let halo_conc = (effective_mem / (192 * 1024 * 1024)).clamp(1, parallelism);
        let per_worker_budget = effective_mem / halo_conc.max(1);
        let admitted_leaves = per_worker_budget / per_leaf.max(1);
        if admitted_leaves < 2 {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let max_halo_leaves = admitted_leaves.min(32_768);

        for r in 0..rounds {
            let stage_prop = STAGE_REFINE_BASE + 3 * r;
            let stage_ext = STAGE_REFINE_BASE + 3 * r + 1;
            let stage_app = STAGE_REFINE_BASE + 3 * r + 2;

            init_stage_units(&env, stage_prop, heavy_shards).await?;
            let prop_done = drive_stage(&env, stage_prop, heavy_shards, halo_conc, {
                let env = env.clone();
                let work_dir = work_dir.clone();
                let plan = plan.clone();
                let quota = Arc::clone(&quota);
                let lock = Arc::clone(&lock);
                move |s, _lease| {
                    let env_c = env.clone();
                    let dir = work_dir.clone();
                    let plan_c = plan.clone();
                    let quota_c = Arc::clone(&quota);
                    let guard = lock.clone();
                    async move {
                        let prop_count = env_c
                            .spawn_cpu({
                                let manifest = env_c.manifest.clone();
                                move |ctrl| {
                                    let _lock = guard;
                                    let check = || crate::runtime::check_control(&ctrl);
                                    propose_disk_shard(
                                        &dir,
                                        fence.epoch,
                                        r,
                                        s,
                                        &manifest,
                                        &plan_c,
                                        max_halo_leaves,
                                        per_leaf,
                                        per_worker_budget,
                                        &quota_c,
                                        &check,
                                    )
                                }
                            })
                            .await?;
                        Ok(UnitOutput {
                            aux_count: prop_count,
                            ..UnitOutput::default()
                        })
                    }
                }
            })
            .await?;

            let total_proposals: u64 = prop_done.iter().map(|u| u.aux_count).sum();
            if total_proposals == 0 {
                break;
            }

            let accepted_moves = Arc::new(
                env.spawn_cpu({
                    let dir = work_dir.clone();
                    let manifest = index.clone();
                    let plan_c = plan.clone();
                    let guard = lock.clone();
                    move |ctrl| {
                        let _lock = guard;
                        let check = || crate::runtime::check_control(&ctrl);
                        arbitrate_disk_round(
                            &dir,
                            fence.epoch,
                            r,
                            &manifest,
                            &plan_c,
                            effective_mem,
                            &check,
                        )
                    }
                })
                .await?,
            );
            if accepted_moves.total_accepted == 0 {
                break;
            }

            init_stage_units(&env, stage_ext, heavy_shards).await?;
            drive_stage(&env, stage_ext, heavy_shards, leaf_conc, {
                let env = env.clone();
                let work_dir = work_dir.clone();
                let accepted_moves = Arc::clone(&accepted_moves);
                let quota = Arc::clone(&quota);
                let lock = Arc::clone(&lock);
                move |s, _lease| {
                    let env_c = env.clone();
                    let dir = work_dir.clone();
                    let acc_c = Arc::clone(&accepted_moves);
                    let quota_c = Arc::clone(&quota);
                    let guard = lock.clone();
                    async move {
                        env_c
                            .spawn_cpu({
                                let manifest = env_c.manifest.clone();
                                move |ctrl| {
                                    let _lock = guard;
                                    let check = || crate::runtime::check_control(&ctrl);
                                    extract_disk_outbound(
                                        &dir,
                                        fence.epoch,
                                        r,
                                        s,
                                        heavy_shards as usize,
                                        &manifest,
                                        &acc_c,
                                        &quota_c,
                                        &check,
                                    )
                                }
                            })
                            .await?;
                        Ok(UnitOutput::default())
                    }
                }
            })
            .await?;

            init_stage_units(&env, stage_app, heavy_shards).await?;
            drive_stage(&env, stage_app, heavy_shards, leaf_conc, {
                let env = env.clone();
                let work_dir = work_dir.clone();
                let accepted_moves = Arc::clone(&accepted_moves);
                let quota = Arc::clone(&quota);
                let lock = Arc::clone(&lock);
                move |s, _lease| {
                    let env_c = env.clone();
                    let dir = work_dir.clone();
                    let acc_c = Arc::clone(&accepted_moves);
                    let quota_c = Arc::clone(&quota);
                    let guard = lock.clone();
                    async move {
                        env_c
                            .spawn_cpu({
                                let manifest = env_c.manifest.clone();
                                move |ctrl| {
                                    let _lock = guard;
                                    let check = || crate::runtime::check_control(&ctrl);
                                    apply_disk_shard(
                                        &dir,
                                        fence.epoch,
                                        r,
                                        s,
                                        &manifest,
                                        &acc_c,
                                        &quota_c,
                                        &check,
                                    )
                                }
                            })
                            .await?;
                        Ok(UnitOutput::default())
                    }
                }
            })
            .await?;

            {
                let ep = fence.epoch;
                let mut obsolete = Vec::with_capacity((heavy_shards as usize) * 5);
                for s in 0..heavy_shards {
                    for kind in ["mem", "centers", "inb", "prop", "outb"] {
                        obsolete.push(format!("{kind}-e{ep}-r{r}-{s:06}.bin"));
                    }
                }
                reclaim_files(&env, &work_dir, &quota, &lock, obsolete).await?;
            }
            final_round = r + 1;
        }
        report.refine = refine_start.elapsed();
    }

    // 5. Assign global deterministic PartitionKeys from compact `centers-*` TreeKey headers.
    let emit_start = Instant::now();
    let (assigned, tree_totals) = env
        .spawn_cpu({
            let dir = work_dir.clone();
            let manifest = index.clone();
            let plan_c = plan.clone();
            let guard = lock.clone();
            move |ctrl| {
                let _lock = guard;
                let check = || crate::runtime::check_control(&ctrl);
                assign_disk_leaf_keys(
                    &dir,
                    fence.epoch,
                    final_round,
                    &plan_c,
                    coarse_shards,
                    &manifest,
                    &check,
                )
            }
        })
        .await?;
    let assigned = Arc::new(assigned);
    let leaf_partitions: u64 = tree_totals.values().map(|(_, c)| *c).sum();

    // Stage 30: Emit Leaf Partitions (`[0x04]`) in shard order + write indexed `loc-*` & `uchild-*`.
    init_stage_units(&env, STAGE_EMIT_LEAVES, coarse_shards).await?;
    let leaf_emit_done = drive_stage(&env, STAGE_EMIT_LEAVES, coarse_shards, emit_conc, {
        let env = env.clone();
        let work_dir = work_dir.clone();
        let plan = plan.clone();
        let assigned = Arc::clone(&assigned);
        let quota = Arc::clone(&quota);
        let lock = Arc::clone(&lock);
        move |s, u_lease| {
            let env_c = env.clone();
            let dir = work_dir.clone();
            let plan_c = plan.clone();
            let assigned_c = Arc::clone(&assigned);
            let quota_c = Arc::clone(&quota);
            let guard = lock.clone();
            async move {
                let res = env_c
                    .spawn_cpu({
                        let manifest = env_c.manifest.clone();
                        move |ctrl| {
                            let _lock = guard;
                            let check = || crate::runtime::check_control(&ctrl);
                            let ep = fence.epoch;
                            let r_use = if s < plan_c.light_shard_start {
                                final_round
                            } else {
                                0
                            };
                            let (shard_state, _) = read_compact_shard_state(
                                &dir.join(format!("base-e{ep}-{s:06}.bin")),
                                &dir.join(format!("mem-e{ep}-r{r_use}-{s:06}.bin")),
                                &dir.join(format!("inb-e{ep}-r{r_use}-{s:06}.bin")),
                                &manifest,
                            )?;
                            if shard_state.leaves.is_empty() {
                                write_indexed_locations(
                                    &dir.join(format!("loc-e{ep}-{s:06}.bin")),
                                    &vec![Vec::new(); plan_c.id_bucket_count() as usize],
                                    &quota_c,
                                )?;
                                write_indexed_upper_children(
                                    &dir.join(format!("uchild-e{ep}-{s:06}.bin")),
                                    &manifest,
                                    tree_buckets,
                                    &[],
                                    &quota_c,
                                )?;
                                return Ok((
                                    Vec::new(),
                                    LtHash::ZERO,
                                    LtHash::ZERO,
                                    LtHash::ZERO,
                                    0_u64,
                                ));
                            }
                            let enc = encode_leaf_shard(
                                &manifest,
                                &plan_c,
                                &shard_state,
                                &assigned_c[s as usize],
                                hard_limits,
                                &check,
                            )?;
                            write_indexed_locations(
                                &dir.join(format!("loc-e{ep}-{s:06}.bin")),
                                &enc.locations_by_bucket,
                                &quota_c,
                            )?;
                            write_indexed_upper_children(
                                &dir.join(format!("uchild-e{ep}-{s:06}.bin")),
                                &manifest,
                                tree_buckets,
                                &enc.child_centroids,
                                &quota_c,
                            )?;
                            Ok((
                                enc.kvs,
                                enc.kv_lthash,
                                enc.membership_lthash,
                                enc.edge_def_lthash,
                                shard_state.leaves.len() as u64,
                            ))
                        }
                    })
                    .await?;
                let (kvs, kv_lt, mem_lt, def_lt, leaves_n) = res;
                let kv_count = kvs.len() as u64;
                env_c.commit_unit_kvs(u_lease.as_ref(), s, kvs).await?;
                Ok(UnitOutput {
                    records: 0,
                    kv_count,
                    aux_count: leaves_n,
                    kv_lthash: kv_lt,
                    aux_lthash: mem_lt,
                    aux2_lthash: def_lt,
                })
            }
        }
    })
    .await?;

    {
        let ep = fence.epoch;
        let mut obsolete = Vec::with_capacity((coarse_shards as usize) * 4);
        for s in 0..coarse_shards {
            obsolete.push(format!("base-e{ep}-{s:06}.bin"));
            let r_use = if s < plan.light_shard_start {
                final_round
            } else {
                0
            };
            for kind in ["mem", "centers", "inb"] {
                obsolete.push(format!("{kind}-e{ep}-r{r_use}-{s:06}.bin"));
            }
        }
        reclaim_files(&env, &work_dir, &quota, &lock, obsolete).await?;
    }

    // Stage 32: Build and emit Upper Trees + Partition-Local HNSW graphs (`[0x03]` + `[0x04]`).
    let tree_totals_arc = Arc::new(tree_totals);
    init_stage_units(&env, STAGE_EMIT_TREES, tree_buckets).await?;
    let tree_emit_done = drive_stage(&env, STAGE_EMIT_TREES, tree_buckets, parallelism.min(4), {
        let env = env.clone();
        let work_dir = work_dir.clone();
        let tree_totals_arc = Arc::clone(&tree_totals_arc);
        let lock = Arc::clone(&lock);
        move |tb, u_lease| {
            let env_c = env.clone();
            let dir = work_dir.clone();
            let totals_c = Arc::clone(&tree_totals_arc);
            let guard = lock.clone();
            async move {
                let (trees_n, internal_parts, kvs, kv_lt, def_lt, ref_lt) = env_c
                    .spawn_cpu({
                        let manifest = env_c.manifest.clone();
                        move |ctrl| {
                            let _lock = guard;
                            let check = || crate::runtime::check_control(&ctrl);
                            let ep = fence.epoch;
                            let mut bucket_trees: BTreeMap<Bytes, UpperTreeBucket> =
                                BTreeMap::new();
                            for (raw, (tk, total_l)) in totals_c.iter() {
                                let tk_hash = u64::from_be_bytes(
                                    tree_key_hash(tk)[..8].try_into().expect("sha256 prefix"),
                                );
                                if (tk_hash % u64::from(tree_buckets)) as u32 == tb {
                                    bucket_trees
                                        .insert(raw.clone(), (tk.clone(), *total_l, Vec::new()));
                                }
                            }
                            if bucket_trees.is_empty() {
                                return Ok((
                                    0_u64,
                                    0_u64,
                                    Vec::new(),
                                    LtHash::ZERO,
                                    LtHash::ZERO,
                                    LtHash::ZERO,
                                ));
                            }
                            let mut children_buf = Vec::new();
                            for s in 0..coarse_shards {
                                if s.is_multiple_of(64) {
                                    check()?;
                                }
                                children_buf.clear();
                                read_indexed_upper_children_slice(
                                    &dir.join(format!("uchild-e{ep}-{s:06}.bin")),
                                    tb as usize,
                                    &manifest,
                                    &mut children_buf,
                                )?;
                                for (tk, pk, c) in children_buf.drain(..) {
                                    if let Some(entry) = bucket_trees.get_mut(tk.as_bytes()) {
                                        entry.2.push((pk, c));
                                    }
                                }
                            }
                            let mut all_kvs = Vec::new();
                            let mut kv_lt = LtHash::ZERO;
                            let mut def_lt = LtHash::ZERO;
                            let mut ref_lt = LtHash::ZERO;
                            let mut internal_parts = 0_u64;
                            let trees_n = bucket_trees.len() as u64;
                            let graph_threads = (parallelism / 2).max(1);
                            for (_, (tk, total_l, children)) in bucket_trees {
                                let enc = build_and_encode_upper_tree(
                                    &manifest,
                                    &tk,
                                    total_l,
                                    children,
                                    sample_items,
                                    graph_memory_bytes,
                                    hard_limits,
                                    graph_threads,
                                    &check,
                                )?;
                                internal_parts += enc.partitions;
                                kv_lt.add_assign(enc.kv_lthash);
                                def_lt.add_assign(enc.edge_def_lthash);
                                ref_lt.add_assign(enc.edge_ref_lthash);
                                all_kvs.extend(enc.kvs);
                            }
                            all_kvs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                            Ok((trees_n, internal_parts, all_kvs, kv_lt, def_lt, ref_lt))
                        }
                    })
                    .await?;
                let kv_count = kvs.len() as u64;
                env_c.commit_unit_kvs(u_lease.as_ref(), tb, kvs).await?;
                Ok(UnitOutput {
                    records: trees_n,
                    kv_count,
                    aux_count: internal_parts,
                    kv_lthash: kv_lt,
                    aux_lthash: def_lt,
                    aux2_lthash: ref_lt,
                })
            }
        }
    })
    .await?;

    reclaim_files(
        &env,
        &work_dir,
        &quota,
        &lock,
        (0..coarse_shards)
            .map(|s| format!("uchild-e{}-{s:06}.bin", fence.epoch))
            .collect(),
    )
    .await?;

    // Stage 31: Emit `[0x01 Record + Payload + Location]` in bounded sub-batches per `id_bucket`.
    let max_records_per_subbatch = (effective_mem / 4096).clamp(64, 4096);
    init_stage_units(&env, STAGE_EMIT_RECORDS, id_buckets).await?;
    let rec_emit_done = drive_stage(&env, STAGE_EMIT_RECORDS, id_buckets, emit_conc, {
        let env = env.clone();
        let work_dir = work_dir.clone();
        let snapshot = snapshot.clone();
        let lock = Arc::clone(&lock);
        move |b, u_lease| {
            let env_c = env.clone();
            let dir = work_dir.clone();
            let snap_c = snapshot.clone();
            let guard = lock.clone();
            async move {
                let (mut id_refs, mut locs) = env_c
                    .spawn_cpu({
                        let manifest = env_c.manifest.clone();
                        let guard_init = guard.clone();
                        move |ctrl| {
                            let _lock = guard_init;
                            let check = || crate::runtime::check_control(&ctrl);
                            let ep = fence.epoch;
                            let id_refs =
                                read_bucket_id_refs(&dir.join(format!("idref-e{ep}-{b:06}.bin")))?;
                            let mut locs = Vec::with_capacity(id_refs.len());
                            for s in 0..coarse_shards {
                                if s.is_multiple_of(64) {
                                    check()?;
                                }
                                read_indexed_locations_slice(
                                    &dir.join(format!("loc-e{ep}-{s:06}.bin")),
                                    b as usize,
                                    &manifest,
                                    &mut locs,
                                )?;
                            }
                            if id_refs.len() != locs.len() {
                                return Err(corrupt());
                            }
                            locs.sort_unstable_by(|x, y| x.0.cmp(&y.0));
                            Ok((id_refs, locs))
                        }
                    })
                    .await?;

                let loc_count = locs.len() as u64;
                let mut total_kv_count = 0_u64;
                let mut total_kv_lt = LtHash::ZERO;
                let mut total_loc_lt = LtHash::ZERO;

                while !id_refs.is_empty() {
                    let take_n = id_refs.len().min(max_records_per_subbatch);
                    let sub_refs: Vec<_> = id_refs.drain(..take_n).collect();
                    let sub_locs: Vec<_> = locs.drain(..take_n).collect();
                    let snap_sub = snap_c.clone();
                    let guard_sub = guard.clone();
                    let manifest_sub = env_c.manifest.clone();
                    let (kvs, kv_lt, loc_lt) = env_c
                        .spawn_cpu(move |ctrl| {
                            let _lock = guard_sub;
                            let check = || crate::runtime::check_control(&ctrl);
                            let mut seg_reader = snap_sub.frame_reader();
                            encode_record_group_bucket(
                                &manifest_sub,
                                &mut seg_reader,
                                sub_refs,
                                sub_locs,
                                hard_limits,
                                &check,
                            )
                        })
                        .await?;
                    total_kv_count += kvs.len() as u64;
                    total_kv_lt.add_assign(kv_lt);
                    total_loc_lt.add_assign(loc_lt);
                    env_c.commit_unit_kvs(u_lease.as_ref(), b, kvs).await?;
                }

                Ok(UnitOutput {
                    records: loc_count,
                    kv_count: total_kv_count,
                    aux_count: 0,
                    kv_lthash: total_kv_lt,
                    aux_lthash: total_loc_lt,
                    aux2_lthash: LtHash::ZERO,
                })
            }
        }
    })
    .await?;
    report.emit = emit_start.elapsed();

    // 6. Verify global 512-bit homomorphic multiset equations across all stages and seal `Loaded`.
    let loc_records: u64 = rec_emit_done.iter().map(|u| u.records).sum();
    let mut leaf_mem_lt = LtHash::ZERO;
    let mut edge_def_lt = LtHash::ZERO;
    for u in &leaf_emit_done {
        leaf_mem_lt.add_assign(u.aux_lthash);
        edge_def_lt.add_assign(u.aux2_lthash);
    }
    let mut loc_mem_lt = LtHash::ZERO;
    for u in &rec_emit_done {
        loc_mem_lt.add_assign(u.aux_lthash);
    }
    let mut edge_ref_lt = LtHash::ZERO;
    let mut internal_parts = 0_u64;
    let mut total_trees = 0_u64;
    for u in &tree_emit_done {
        total_trees += u.records;
        internal_parts += u.aux_count;
        edge_def_lt.add_assign(u.aux_lthash);
        edge_ref_lt.add_assign(u.aux2_lthash);
    }

    if loc_records != scatter_records || leaf_mem_lt != loc_mem_lt || edge_def_lt != edge_ref_lt {
        return Err(corrupt());
    }

    let mut total_entries = 0_u64;
    let mut expected_kv_lthash = LtHash::ZERO;
    for list in [&leaf_emit_done, &tree_emit_done, &rec_emit_done] {
        for u in list {
            total_entries = total_entries.checked_add(u.kv_count).ok_or_else(corrupt)?;
            expected_kv_lthash.add_assign(u.kv_lthash);
        }
    }

    let total_partitions = leaf_partitions + internal_parts;
    let split_keys =
        build_validation_split_keys(index.logical_index_id(), &plan, total_entries, hard_limits)?;
    seal_loaded(
        context,
        index,
        fence,
        scatter_records,
        total_trees,
        total_partitions,
        total_entries,
        expected_kv_lthash,
        split_keys,
        retry,
    )
    .await?;

    report.records = scatter_records;
    report.trees = total_trees;
    report.partitions = total_partitions;
    report.entries = total_entries;
    report.scratch_written_bytes = quota.written();
    report.peak_scratch_bytes = quota.peak();
    Ok(report)
}

async fn reclaim_files<B: Backend>(
    env: &StageEnv<B>,
    work_dir: &Path,
    quota: &Arc<ScratchQuota>,
    lock: &Arc<File>,
    names: Vec<String>,
) -> Result<()> {
    if names.is_empty() {
        return Ok(());
    }
    let dir = work_dir.to_path_buf();
    let quota_c = Arc::clone(quota);
    let guard = Arc::clone(lock);
    env.spawn_cpu(move |_ctrl| {
        let _lock = guard;
        for name in names {
            let p = dir.join(name);
            let len = fs::metadata(&p).map_or(0, |m| m.len());
            match fs::remove_file(&p) {
                Ok(()) => quota_c.release_path(&p, len),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    quota_c.release_path(&p, 0);
                }
                Err(_) => {}
            }
        }
        Ok(())
    })
    .await
}

async fn mark_loading_started<B: Backend>(env: &StageEnv<B>) -> Result<()> {
    let id = env.manifest.logical_index_id();
    let mut attempts = 0;
    loop {
        crate::runtime::check_control(&env.control)?;
        let _sched_guard = match &env.sched_lease_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
        let prepared = async {
            let mut txn = WriteLogicalTxn::bootstrap(
                env.backend.begin_write().await?,
                env.backend.hard_limits(),
                env.backend.admission_budget(),
            );
            building_in_flight(&mut txn, &env.manifest).await?;
            authorize_raw(&mut txn, id, env.bulk_authority).await?;
            check_lane_fence(&mut txn, &env.manifest, env.fence, 0).await?;
            if let Some(PersistentValue::BuildProgress(mut p)) =
                txn.get_for_update(LogicalKey::BuildProgress(id)).await?
                && p.epoch == env.fence.epoch
                && p.phase == BuildPhase::Preparing
            {
                p.phase = BuildPhase::Loading;
                txn.put(
                    LogicalKey::BuildProgress(id),
                    PersistentValue::BuildProgress(p),
                )
                .await?;
                return Ok(Some(txn));
            }
            txn.rollback().await;
            Ok::<_, Error>(None)
        }
        .await;
        let txn_opt = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                env.retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        let Some(txn) = txn_opt else {
            return Ok(());
        };
        match txn.commit().await {
            Ok(()) => return Ok(()),
            Err(e) => {
                drop(_sched_guard);
                env.retry
                    .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                    .await?;
            }
        }
    }
}

async fn bump_single_worker_progress<B: Backend>(env: &StageEnv<B>, delta: u64) -> Result<()> {
    if delta == 0 {
        return Ok(());
    }
    let _sched_guard = match &env.sched_lease_lock {
        Some(l) => Some(l.read().await),
        None => None,
    };
    let id = env.manifest.logical_index_id();
    let mut txn = WriteLogicalTxn::bootstrap(
        env.backend.begin_write().await?,
        env.backend.hard_limits(),
        env.backend.admission_budget(),
    );
    if building_in_flight(&mut txn, &env.manifest).await.is_err()
        || authorize_raw(&mut txn, id, env.bulk_authority)
            .await
            .is_err()
        || check_lane_fence(&mut txn, &env.manifest, env.fence, 0)
            .await
            .is_err()
    {
        txn.rollback().await;
        return Ok(());
    }
    if let Some(PersistentValue::BuildProgress(mut p)) =
        txn.get_for_update(LogicalKey::BuildProgress(id)).await?
        && p.epoch == env.fence.epoch
        && matches!(p.phase, BuildPhase::Preparing | BuildPhase::Loading)
    {
        p.phase = BuildPhase::Loading;
        p.loaded_entries = p.loaded_entries.saturating_add(delta);
        if p.loaded_entries > p.total_entries {
            p.total_entries = p.loaded_entries;
        }
        if txn
            .put(
                LogicalKey::BuildProgress(id),
                PersistentValue::BuildProgress(p),
            )
            .await
            .is_ok()
        {
            let _ = txn.commit().await;
            return Ok(());
        }
    }
    txn.rollback().await;
    Ok(())
}

async fn init_stage_units<B: Backend>(env: &StageEnv<B>, stage: u8, unit_count: u32) -> Result<()> {
    if unit_count == 0 {
        return Ok(());
    }
    if env.distributed_workers == 1 {
        if stage >= STAGE_EMIT_LEAVES {
            mark_loading_started(env).await?;
        }
        return Ok(());
    }
    let id = env.manifest.logical_index_id();
    let max_per_txn = env.budget.max_mutations.saturating_sub(4).clamp(1, 512) as u32;
    for batch_start in (0..unit_count).step_by(max_per_txn as usize) {
        let batch_end = (batch_start + max_per_txn).min(unit_count);
        let mut attempts = 0;
        loop {
            crate::runtime::check_control(&env.control)?;
            let _sched_guard = match &env.sched_lease_lock {
                Some(l) => Some(l.read().await),
                None => None,
            };
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    env.backend.begin_write().await?,
                    env.backend.hard_limits(),
                    env.backend.admission_budget(),
                );
                let mut warm = Vec::with_capacity((batch_end - batch_start) as usize + 5);
                warm.push(LogicalKey::Manifest(id));
                warm.push(LogicalKey::BuildUnit {
                    index: id,
                    stage: super::STAGE_FENCE,
                    unit: 0,
                });
                warm.push(LogicalKey::BuildSchedule(id));
                if let Some(auth) = env.bulk_authority {
                    warm.push(auth.slot_key(id));
                }
                if batch_start == 0 && stage >= STAGE_EMIT_LEAVES {
                    warm.push(LogicalKey::BuildProgress(id));
                }
                for u in batch_start..batch_end {
                    warm.push(LogicalKey::BuildUnit {
                        index: id,
                        stage,
                        unit: u,
                    });
                }
                txn.warm_for_update(warm).await?;

                building_in_flight(&mut txn, &env.manifest).await?;
                authorize_raw(&mut txn, id, env.bulk_authority).await?;
                check_lane_fence(&mut txn, &env.manifest, env.fence, 0).await?;

                let mut any_written = false;
                if batch_start == 0
                    && stage >= STAGE_EMIT_LEAVES
                    && let Some(PersistentValue::BuildProgress(mut p)) =
                        txn.get_for_update(LogicalKey::BuildProgress(id)).await?
                    && p.epoch == env.fence.epoch
                    && p.phase == BuildPhase::Preparing
                {
                    p.phase = BuildPhase::Loading;
                    txn.put(
                        LogicalKey::BuildProgress(id),
                        PersistentValue::BuildProgress(p),
                    )
                    .await?;
                    any_written = true;
                }
                for u in batch_start..batch_end {
                    let ukey = LogicalKey::BuildUnit {
                        index: id,
                        stage,
                        unit: u,
                    };
                    match txn.get_for_update(ukey.clone()).await? {
                        Some(PersistentValue::BuildUnit(existing))
                            if existing.epoch == env.fence.epoch => {}
                        _ => {
                            txn.put(
                                ukey,
                                PersistentValue::BuildUnit(BuildUnit::unclaimed(env.fence.epoch)),
                            )
                            .await?;
                            any_written = true;
                        }
                    }
                }
                if !any_written {
                    txn.rollback().await;
                    return Ok::<_, Error>(None);
                }
                Ok(Some(txn))
            }
            .await;

            let txn_opt = match prepared {
                Ok(v) => v,
                Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                    drop(_sched_guard);
                    env.retry
                        .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                        .await?;
                    continue;
                }
                Err(e) => return Err(e),
            };
            let Some(txn) = txn_opt else {
                break;
            };
            match txn.commit().await {
                Ok(()) => break,
                Err(e) => {
                    drop(_sched_guard);
                    env.retry
                        .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                        .await?;
                }
            }
        }
    }
    Ok(())
}

async fn drive_stage<B, F, Fut>(
    env: &StageEnv<B>,
    stage: u8,
    unit_count: u32,
    concurrency: usize,
    work: F,
) -> Result<Vec<BuildUnit>>
where
    B: Backend,
    F: Fn(u32, Option<UnitLease>) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<UnitOutput>> + Send + 'static,
{
    if unit_count == 0 {
        return Ok(Vec::new());
    }
    let work = Arc::new(work);
    let worker_conc = concurrency.max(1).min(unit_count as usize);

    // Single-worker fast path: process units in strict ascending order via an in-process
    // atomic cursor with writer-lane fencing and zero `BuildUnit` KV transactions.
    if env.distributed_workers == 1 {
        let next_unit = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let mut tasks = tokio::task::JoinSet::<Result<Vec<(u32, BuildUnit)>>>::new();
        for _ in 0..worker_conc {
            let counter = Arc::clone(&next_unit);
            let env_c = env.clone();
            let work_c = Arc::clone(&work);
            tasks.spawn(async move {
                let mut local_done = Vec::new();
                loop {
                    crate::runtime::check_control(&env_c.control)?;
                    let u = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if u >= unit_count {
                        return Ok(local_done);
                    }
                    let out = work_c(u, None).await?;
                    if out.kv_count > 0 && env_c.unit_lease >= Duration::from_millis(50) {
                        bump_single_worker_progress(&env_c, out.kv_count).await?;
                    }
                    let mut bu = BuildUnit::unclaimed(env_c.fence.epoch);
                    bu.complete = true;
                    bu.records = out.records;
                    bu.kv_count = out.kv_count;
                    bu.aux_count = out.aux_count;
                    bu.kv_lthash = out.kv_lthash;
                    bu.aux_lthash = out.aux_lthash;
                    bu.aux2_lthash = out.aux2_lthash;
                    local_done.push((u, bu));
                }
            });
        }
        let mut results = Vec::with_capacity(unit_count as usize);
        while let Some(res) = tasks.join_next().await {
            results.extend(res.map_err(|e| Error::with_source(ErrorKind::Other, e))??);
        }
        results.sort_unstable_by_key(|(u, _)| *u);
        return Ok(results.into_iter().map(|(_, bu)| bu).collect());
    }

    let mut known_complete = BTreeSet::<u32>::new();
    loop {
        crate::runtime::check_control(&env.control)?;
        let (claimable, completed_all) =
            scan_stage_status(env, stage, unit_count, &mut known_complete).await?;
        if let Some(done) = completed_all {
            return Ok(done);
        }
        if claimable.is_empty() {
            tokio::time::sleep(Duration::from_millis(25)).await;
            continue;
        }

        let mut jitter = [0_u8; 4];
        let _ = getrandom::fill(&mut jitter);
        let offset = (u32::from_le_bytes(jitter) as usize) % claimable.len();
        let mut ordered = Vec::with_capacity(claimable.len());
        ordered.extend_from_slice(&claimable[offset..]);
        ordered.extend_from_slice(&claimable[..offset]);

        let claimable_len = ordered.len();
        let queue = Arc::new(std::sync::Mutex::new(ordered.into_iter()));
        let mut tasks = tokio::task::JoinSet::<Result<()>>::new();
        for _ in 0..worker_conc.min(claimable_len) {
            let q = Arc::clone(&queue);
            let env_c = env.clone();
            let work_c = Arc::clone(&work);
            tasks.spawn(async move {
                let mut task_token = [0_u8; 32];
                getrandom::fill(&mut task_token).map_err(|_| Error::new(ErrorKind::Other))?;
                loop {
                    crate::runtime::check_control(&env_c.control)?;
                    let next_u = q.lock().unwrap_or_else(|e| e.into_inner()).next();
                    let Some(unit) = next_u else {
                        return Ok(());
                    };
                    let Some(initial_exp) = try_claim_unit(&env_c, stage, unit, task_token).await?
                    else {
                        continue;
                    };
                    let unit_lease = UnitLease {
                        stage,
                        unit,
                        owner: task_token,
                        duration: env_c.unit_lease,
                        lock: Arc::new(tokio::sync::Mutex::new(())),
                    };
                    let outcome = async {
                        let out = run_with_unit_heartbeat(
                            &env_c,
                            &unit_lease,
                            initial_exp,
                            work_c(unit, Some(unit_lease.clone())),
                        )
                        .await?;
                        complete_unit(&env_c, &unit_lease, out).await
                    }
                    .await;
                    match outcome {
                        Ok(()) => {}
                        Err(e) if e.kind() == ErrorKind::BulkBuildSuperseded => {
                            check_epoch_still_active(&env_c).await?;
                        }
                        Err(e) => {
                            // If a peer reclaimed our expired unit lease, completed this unit,
                            // and deleted prior-stage scratch files, ignore the stale local error.
                            if is_unit_completed_by_peer(&env_c, stage, unit).await? {
                                continue;
                            }
                            return Err(e);
                        }
                    }
                }
            });
        }
        while let Some(res) = tasks.join_next().await {
            res.map_err(|e| Error::with_source(ErrorKind::Other, e))??;
        }
    }
}

async fn is_unit_completed_by_peer<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit: u32,
) -> Result<bool> {
    check_epoch_still_active(env).await?;
    let id = env.manifest.logical_index_id();
    let mut rtxn = ReadLogicalTxn::bootstrap(env.backend.begin_read().await?);
    if let Some(PersistentValue::BuildUnit(u)) = rtxn
        .get(LogicalKey::BuildUnit {
            index: id,
            stage,
            unit,
        })
        .await?
        && u.epoch == env.fence.epoch
        && u.complete
    {
        return Ok(true);
    }
    Ok(false)
}

async fn check_epoch_still_active<B: Backend>(env: &StageEnv<B>) -> Result<()> {
    let mut attempts = 0;
    loop {
        crate::runtime::check_control(&env.control)?;
        let _sched_guard = match &env.sched_lease_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
        let res = async {
            let mut txn = WriteLogicalTxn::bootstrap(
                env.backend.begin_write().await?,
                env.backend.hard_limits(),
                env.backend.admission_budget(),
            );
            check_lane_fence(&mut txn, &env.manifest, env.fence, 0).await?;
            authorize_raw(
                &mut txn,
                env.manifest.logical_index_id(),
                env.bulk_authority,
            )
            .await?;
            txn.rollback().await;
            Ok::<_, Error>(())
        }
        .await;
        match res {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                env.retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn scan_stage_status<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit_count: u32,
    known_complete: &mut BTreeSet<u32>,
) -> Result<(Vec<u32>, Option<Vec<BuildUnit>>)> {
    let id = env.manifest.logical_index_id();
    let range = LogicalRange::build_units(id, stage);
    let mut attempts = 0;

    loop {
        crate::runtime::check_control(&env.control)?;
        let res = async {
            let now = now_ms()?;
            let mut rtxn = ReadLogicalTxn::bootstrap(env.backend.begin_read().await?);
            match rtxn.get(LogicalKey::Manifest(id)).await? {
                None => return Err(Error::new(ErrorKind::IndexNotFound)),
                Some(PersistentValue::IndexManifest(m))
                    if m.has_same_immutable_identity(&env.manifest) =>
                {
                    match m.lifecycle() {
                        IndexLifecycle::Building => {}
                        IndexLifecycle::Dropping => {
                            return Err(Error::new(ErrorKind::IndexDropping));
                        }
                        IndexLifecycle::Active => {
                            return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                        }
                    }
                }
                _ => return Err(corrupt()),
            }
            if let Some(PersistentValue::BuildWorkspace(w)) =
                rtxn.get(LogicalKey::BuildWorkspace(id)).await?
                && let Some(kind) = w.failure
            {
                return Err(Error::new(kind));
            }

            let mut cursor = None;
            let mut units = Vec::with_capacity(unit_count as usize);
            let mut claimable = Vec::new();
            let mut pages_on_txn = 0_usize;
            loop {
                if pages_on_txn >= 8 {
                    rtxn = ReadLogicalTxn::bootstrap(env.backend.begin_read().await?);
                    pages_on_txn = 0;
                }
                let page = rtxn
                    .scan(
                        &range,
                        cursor.as_ref(),
                        ScanLimits {
                            item_limit: 1024,
                            byte_limit: 512 * 1024,
                        },
                    )
                    .await?;
                pages_on_txn += 1;
                for item in page.items() {
                    let LogicalKey::BuildUnit { unit, .. } = *item.key() else {
                        return Err(corrupt());
                    };
                    let PersistentValue::BuildUnit(u) = item.value().clone() else {
                        return Err(corrupt());
                    };
                    if u.epoch != env.fence.epoch {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    if u.complete {
                        known_complete.insert(unit);
                    } else if u.owner == [0; 32] || u.expires_ms <= now {
                        claimable.push(unit);
                    }
                    units.push(u);
                }
                cursor = page.into_next_cursor();
                if cursor.is_none() {
                    break;
                }
            }
            if units.len() != unit_count as usize {
                return Err(corrupt());
            }
            if known_complete.len() == unit_count as usize && units.iter().all(|u| u.complete) {
                return Ok((Vec::new(), Some(units)));
            }
            Ok((claimable, None))
        }
        .await;

        match res {
            Ok(v) => return Ok(v),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                env.retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn try_claim_unit<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit: u32,
    owner: [u8; 32],
) -> Result<Option<u64>> {
    let id = env.manifest.logical_index_id();
    let ukey = LogicalKey::BuildUnit {
        index: id,
        stage,
        unit,
    };
    let lane = unit % super::FENCE_LANES;
    let mut attempts = 0;
    loop {
        crate::runtime::check_control(&env.control)?;
        let _sched_guard = match &env.sched_lease_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
        let prepared = async {
            let mut txn = WriteLogicalTxn::bootstrap(
                env.backend.begin_write().await?,
                env.backend.hard_limits(),
                env.backend.admission_budget(),
            );
            let mut warm = vec![
                LogicalKey::Manifest(id),
                LogicalKey::BuildUnit {
                    index: id,
                    stage: super::STAGE_FENCE,
                    unit: lane,
                },
                LogicalKey::BuildSchedule(id),
                ukey.clone(),
            ];
            if let Some(auth) = env.bulk_authority {
                warm.push(auth.slot_key(id));
            }
            txn.warm_for_update(warm).await?;

            check_lane_fence(&mut txn, &env.manifest, env.fence, lane).await?;
            authorize_raw(&mut txn, id, env.bulk_authority).await?;
            let now = now_ms()?;
            let mut u = match txn.get_for_update(ukey.clone()).await? {
                Some(PersistentValue::BuildUnit(u)) if u.epoch == env.fence.epoch => u,
                Some(PersistentValue::BuildUnit(_)) => {
                    return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                }
                _ => return Err(corrupt()),
            };
            if u.complete {
                txn.rollback().await;
                return Ok((None, None));
            }
            if u.owner == owner {
                let exp = u.expires_ms;
                txn.rollback().await;
                return Ok((None, Some(exp)));
            }
            if u.owner != [0; 32] && u.expires_ms > now {
                txn.rollback().await;
                return Ok((None, None));
            }
            let exp = expiry(env.unit_lease)?;
            u.owner = owner;
            u.expires_ms = exp;
            txn.put(ukey.clone(), PersistentValue::BuildUnit(u)).await?;
            Ok((Some(txn), Some(exp)))
        }
        .await;

        let (txn_opt, exp_opt) = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                if env.retry.would_exhaust(attempts) {
                    return Ok(None);
                }
                env.retry.wait(attempts).await;
                attempts += 1;
                continue;
            }
            Err(e) => return Err(e),
        };
        let Some(txn) = txn_opt else {
            return Ok(exp_opt);
        };
        match txn.commit().await {
            Ok(()) => return Ok(exp_opt),
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_sched_guard);
                if env.retry.would_exhaust(attempts) {
                    return Ok(None);
                }
                env.retry.wait(attempts).await;
                attempts += 1;
            }
            Err(e) => {
                drop(_sched_guard);
                env.retry
                    .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                    .await?;
            }
        }
    }
}

async fn run_with_unit_heartbeat<B: Backend, Fut>(
    env: &StageEnv<B>,
    lease: &UnitLease,
    initial_expires_ms: u64,
    work: Fut,
) -> Result<UnitOutput>
where
    Fut: std::future::Future<Output = Result<UnitOutput>>,
{
    let tick = (env.unit_lease / 3).max(Duration::from_millis(5));
    let renewals = async {
        let mut confirmed_expires_ms = initial_expires_ms;
        let mut timer = tokio::time::interval(tick);
        timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        timer.tick().await;
        loop {
            timer.tick().await;
            if let Err(e) = renew_unit(env, lease, &mut confirmed_expires_ms).await {
                break Err(e);
            }
        }
    };
    tokio::select! {
        res = work => res,
        res = renewals => res,
    }
}

async fn renew_unit<B: Backend>(
    env: &StageEnv<B>,
    lease: &UnitLease,
    confirmed_expires_ms: &mut u64,
) -> Result<()> {
    let Ok(_guard) = lease.lock.try_lock() else {
        return Ok(());
    };
    let _sched_guard = match &env.sched_lease_lock {
        Some(l) => Some(l.read().await),
        None => None,
    };
    let id = env.manifest.logical_index_id();
    let lane = lease.unit % super::FENCE_LANES;
    let ukey = LogicalKey::BuildUnit {
        index: id,
        stage: lease.stage,
        unit: lease.unit,
    };
    let outcome = async {
        let mut txn = WriteLogicalTxn::bootstrap(
            env.backend.begin_write().await?,
            env.backend.hard_limits(),
            env.backend.admission_budget(),
        );
        let mut warm = vec![
            LogicalKey::Manifest(id),
            LogicalKey::BuildUnit {
                index: id,
                stage: super::STAGE_FENCE,
                unit: lane,
            },
            LogicalKey::BuildSchedule(id),
            ukey.clone(),
        ];
        if let Some(auth) = env.bulk_authority {
            warm.push(auth.slot_key(id));
        }
        txn.warm_for_update(warm).await?;
        check_lane_fence(&mut txn, &env.manifest, env.fence, lane).await?;
        authorize_raw(&mut txn, id, env.bulk_authority).await?;
        match txn.get_for_update(ukey.clone()).await? {
            Some(PersistentValue::BuildUnit(mut u))
                if u.epoch == env.fence.epoch && u.owner == lease.owner && !u.complete =>
            {
                let next_exp = u.expires_ms.max(expiry(env.unit_lease)?);
                u.expires_ms = next_exp;
                txn.put(ukey, PersistentValue::BuildUnit(u)).await?;
                txn.commit().await?;
                Ok(next_exp)
            }
            _ => Err(Error::new(ErrorKind::BulkBuildSuperseded)),
        }
    }
    .await;

    match outcome {
        Ok(next_exp) => {
            *confirmed_expires_ms = (*confirmed_expires_ms).max(next_exp);
            Ok(())
        }
        Err(e)
            if matches!(
                e.kind(),
                ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
            ) =>
        {
            if now_ms()? < *confirmed_expires_ms {
                Ok(())
            } else {
                Err(e)
            }
        }
        Err(e) => Err(e),
    }
}

async fn complete_unit<B: Backend>(
    env: &StageEnv<B>,
    lease: &UnitLease,
    out: UnitOutput,
) -> Result<()> {
    let id = env.manifest.logical_index_id();
    let lane = lease.unit % super::FENCE_LANES;
    let ukey = LogicalKey::BuildUnit {
        index: id,
        stage: lease.stage,
        unit: lease.unit,
    };
    let mut attempts = 0;
    loop {
        crate::runtime::check_control(&env.control)?;
        let _sched_guard = match &env.sched_lease_lock {
            Some(l) => Some(l.read().await),
            None => None,
        };
        let _guard = lease.lock.lock().await;
        let prepared = async {
            let mut txn = WriteLogicalTxn::bootstrap(
                env.backend.begin_write().await?,
                env.backend.hard_limits(),
                env.backend.admission_budget(),
            );
            let mut warm = vec![
                LogicalKey::Manifest(id),
                LogicalKey::BuildUnit {
                    index: id,
                    stage: super::STAGE_FENCE,
                    unit: lane,
                },
                LogicalKey::BuildSchedule(id),
                ukey.clone(),
            ];
            if let Some(auth) = env.bulk_authority {
                warm.push(auth.slot_key(id));
            }
            txn.warm_for_update(warm).await?;

            check_lane_fence(&mut txn, &env.manifest, env.fence, lane).await?;
            authorize_raw(&mut txn, id, env.bulk_authority).await?;
            let mut u = match txn.get_for_update(ukey.clone()).await? {
                Some(PersistentValue::BuildUnit(u)) if u.epoch == env.fence.epoch => u,
                _ => return Err(Error::new(ErrorKind::BulkBuildSuperseded)),
            };
            if u.complete && u.owner == lease.owner {
                txn.rollback().await;
                return Ok(None);
            }
            if u.complete || u.owner != lease.owner {
                return Err(Error::new(ErrorKind::BulkBuildSuperseded));
            }
            u.complete = true;
            u.records = out.records;
            u.kv_count = out.kv_count;
            u.aux_count = out.aux_count;
            u.kv_lthash = out.kv_lthash;
            u.aux_lthash = out.aux_lthash;
            u.aux2_lthash = out.aux2_lthash;
            txn.put(ukey.clone(), PersistentValue::BuildUnit(u)).await?;
            Ok(Some(txn))
        }
        .await;

        let txn_opt = match prepared {
            Ok(v) => v,
            Err(e) if e.kind() == ErrorKind::RetryableAbort => {
                drop(_guard);
                drop(_sched_guard);
                env.retry
                    .wait_or_exhaust(Operation::RunBulkBuild, &mut attempts)
                    .await?;
                continue;
            }
            Err(e) => return Err(e),
        };
        let Some(txn) = txn_opt else {
            return Ok(());
        };
        match txn.commit().await {
            Ok(()) => return Ok(()),
            Err(e) => {
                drop(_guard);
                drop(_sched_guard);
                env.retry
                    .after_commit_error(Operation::RunBulkBuild, &mut attempts, e)
                    .await?;
            }
        }
    }
}

// ============================================================================
// Disk Halo Delta Refinement & Leaf PartitionKey Assignment Helpers
// ============================================================================

#[expect(
    clippy::too_many_arguments,
    reason = "scores one shard's vectors against its bounded halo neighborhood on disk"
)]
fn propose_disk_shard(
    dir: &Path,
    epoch: u64,
    round: u8,
    src_shard: u32,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    max_halo_leaves: usize,
    per_leaf: usize,
    per_worker_budget: usize,
    quota: &ScratchQuota,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<u64> {
    check()?;
    let (src_state, _) = read_compact_shard_state(
        &dir.join(format!("base-e{epoch}-{src_shard:06}.bin")),
        &dir.join(format!("mem-e{epoch}-r{round}-{src_shard:06}.bin")),
        &dir.join(format!("inb-e{epoch}-r{round}-{src_shard:06}.bin")),
        manifest,
    )?;
    if src_state.leaves.is_empty() {
        write_proposals(
            &dir.join(format!("prop-e{epoch}-r{round}-{src_shard:06}.bin")),
            &[],
            quota,
        )?;
        return Ok(0);
    }

    let kernel = VectorKernel::new(
        manifest.config().dimension(),
        manifest.config().metric(),
        *manifest.rotation_seed(),
    )?;
    let proposals: Vec<MoveProposal> = score_halo_shard(
        &kernel,
        plan,
        src_shard,
        &src_state,
        max_halo_leaves,
        per_leaf,
        per_worker_budget,
        check,
        |nbr| {
            let centers = read_shard_centers(
                &dir.join(format!("centers-e{epoch}-r{round}-{nbr:06}.bin")),
                manifest,
            )?;
            Ok(centers.into_iter().map(|(_, _, c)| c).collect())
        },
    )?;

    write_proposals(
        &dir.join(format!("prop-e{epoch}-r{round}-{src_shard:06}.bin")),
        &proposals,
        quota,
    )?;
    Ok(proposals.len() as u64)
}

struct DiskAcceptedMoves {
    total_accepted: usize,
    outbound: Vec<BTreeMap<usize, (u32, u32)>>,
    cross_outbound_by_src: Vec<BTreeMap<u32, Vec<(usize, u32)>>>,
    donors_by_dst: Vec<Vec<u32>>,
}

fn arbitrate_disk_round(
    dir: &Path,
    epoch: u64,
    round: u8,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    memory_budget_bytes: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<DiskAcceptedMoves> {
    let heavy_shards = plan.light_shard_start as usize;
    let mut outbound = vec![BTreeMap::new(); heavy_shards];
    let mut cross_outbound_by_src: Vec<BTreeMap<u32, Vec<(usize, u32)>>> =
        vec![BTreeMap::new(); heavy_shards];
    let mut donors_by_dst: Vec<Vec<u32>> = vec![Vec::new(); heavy_shards];
    let mut total_accepted = 0_usize;
    let min_leaf = manifest.config().min_leaf_partition_entries() as usize;
    let max_leaf = manifest.config().max_leaf_partition_entries() as usize;

    for heavy in &plan.heavy_trees {
        check()?;
        let start = heavy.shard_start as usize;
        let count = heavy.shard_count as usize;
        let end_shard = heavy
            .shard_start
            .checked_add(heavy.shard_count)
            .ok_or_else(corrupt)?;
        let mut leaf_offsets = Vec::with_capacity(count + 1);
        let mut shard_leaf_counts = Vec::with_capacity(count);
        let mut counts = Vec::new();
        let mut proposals = Vec::new();
        let mut used_mem = 0_usize;
        leaf_offsets.push(0_usize);
        for local_s in 0..count {
            if local_s.is_multiple_of(32) {
                check()?;
            }
            let s = (start + local_s) as u32;
            let centers = read_shard_centers(
                &dir.join(format!("centers-e{epoch}-r{round}-{s:06}.bin")),
                manifest,
            )?;
            shard_leaf_counts.push(centers.len());
            for (_, mc, _) in centers {
                counts.push(mc);
            }
            leaf_offsets.push(counts.len());
            let shard_props =
                read_proposals(&dir.join(format!("prop-e{epoch}-r{round}-{s:06}.bin")))?;
            used_mem = used_mem
                .saturating_add(shard_props.len().saturating_mul(96))
                .saturating_add(shard_leaf_counts[local_s].saturating_mul(16));
            if used_mem > memory_budget_bytes.max(64 * 1024) {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
            proposals.extend(shard_props);
        }
        if proposals.is_empty() || counts.len() < 2 {
            continue;
        }
        proposals.sort_unstable_by(|a, b| b.gain.total_cmp(&a.gain).then_with(|| a.id.cmp(&b.id)));

        let mut raw_moves = Vec::with_capacity(proposals.len());
        let mut seen_src_rows = BTreeSet::new();
        for (rank, p) in proposals.iter().enumerate() {
            if p.src_shard < heavy.shard_start
                || p.src_shard >= end_shard
                || p.dst_shard < heavy.shard_start
                || p.dst_shard >= end_shard
                || !seen_src_rows.insert((p.src_shard, p.src_row_idx))
            {
                return Err(corrupt());
            }
            let local_src = (p.src_shard - heavy.shard_start) as usize;
            let local_dst = (p.dst_shard - heavy.shard_start) as usize;
            if (p.src_local_leaf as usize) >= shard_leaf_counts[local_src]
                || (p.dst_local_leaf as usize) >= shard_leaf_counts[local_dst]
            {
                return Err(corrupt());
            }
            let g_src = leaf_offsets[local_src] + p.src_local_leaf as usize;
            let g_dst = leaf_offsets[local_dst] + p.dst_local_leaf as usize;
            raw_moves.push((p.gain, rank, g_src, g_dst));
        }

        let mut chosen = Vec::new();
        let acc = accept_moves(raw_moves, &mut counts, min_leaf, max_leaf, |rank, _, _| {
            chosen.push(rank);
        });
        total_accepted += acc;
        for rank in chosen {
            let p = &proposals[rank];
            let src_idx = p.src_shard as usize;
            let dst_idx = p.dst_shard as usize;
            if outbound
                .get_mut(src_idx)
                .ok_or_else(corrupt)?
                .insert(p.src_row_idx as usize, (p.dst_shard, p.dst_local_leaf))
                .is_some()
            {
                return Err(corrupt());
            }
            if p.src_shard != p.dst_shard {
                cross_outbound_by_src
                    .get_mut(src_idx)
                    .ok_or_else(corrupt)?
                    .entry(p.dst_shard)
                    .or_default()
                    .push((p.src_row_idx as usize, p.dst_local_leaf));
                donors_by_dst
                    .get_mut(dst_idx)
                    .ok_or_else(corrupt)?
                    .push(p.src_shard);
            }
        }
    }
    for donors in &mut donors_by_dst {
        donors.sort_unstable();
        donors.dedup();
    }
    Ok(DiskAcceptedMoves {
        total_accepted,
        outbound,
        cross_outbound_by_src,
        donors_by_dst,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "extracts cross-shard outbound rows once per donor shard"
)]
fn extract_disk_outbound(
    dir: &Path,
    epoch: u64,
    round: u8,
    src_shard: u32,
    heavy_shards: usize,
    manifest: &IndexManifest,
    acc: &DiskAcceptedMoves,
    quota: &ScratchQuota,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    check()?;
    let cross_out = acc
        .cross_outbound_by_src
        .get(src_shard as usize)
        .ok_or_else(corrupt)?;
    if cross_out.is_empty() {
        return Ok(());
    }
    let (state, _) = read_compact_shard_state(
        &dir.join(format!("base-e{epoch}-{src_shard:06}.bin")),
        &dir.join(format!("mem-e{epoch}-r{round}-{src_shard:06}.bin")),
        &dir.join(format!("inb-e{epoch}-r{round}-{src_shard:06}.bin")),
        manifest,
    )?;
    check()?;
    let mut by_dst = vec![Vec::new(); heavy_shards];
    for (&dst_s, list) in cross_out {
        *by_dst.get_mut(dst_s as usize).ok_or_else(corrupt)? = list.clone();
    }
    write_indexed_outbound_rows(
        &dir.join(format!("outb-e{epoch}-r{round}-{src_shard:06}.bin")),
        manifest,
        &state,
        &by_dst,
        quota,
    )?;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "applies outbound/inbound moves for one shard and writes next round compact state"
)]
fn apply_disk_shard(
    dir: &Path,
    epoch: u64,
    round: u8,
    shard_id: u32,
    manifest: &IndexManifest,
    acc: &DiskAcceptedMoves,
    quota: &ScratchQuota,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    check()?;
    let (mut shard, base_count) = read_compact_shard_state(
        &dir.join(format!("base-e{epoch}-{shard_id:06}.bin")),
        &dir.join(format!("mem-e{epoch}-r{round}-{shard_id:06}.bin")),
        &dir.join(format!("inb-e{epoch}-r{round}-{shard_id:06}.bin")),
        manifest,
    )?;
    let moved_out = acc.outbound.get(shard_id as usize).ok_or_else(corrupt)?;
    let donors = acc
        .donors_by_dst
        .get(shard_id as usize)
        .ok_or_else(corrupt)?;

    if !moved_out.is_empty() || !donors.is_empty() {
        let kernel = VectorKernel::new(
            manifest.config().dimension(),
            manifest.config().metric(),
            *manifest.rotation_seed(),
        )?;
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

        if shard.rows.rows.len() > base_count && !moved_out.is_empty() {
            let dim = shard.rows.dim;
            let mut remap_inb = vec![usize::MAX; shard.rows.rows.len() - base_count];
            let mut kept_meta = Vec::new();
            let mut kept_vecs = Vec::new();
            for mem in &mut next_members {
                for slot in mem.iter_mut() {
                    if *slot >= base_count {
                        let rel = *slot - base_count;
                        let mapped = *remap_inb.get(rel).ok_or_else(corrupt)?;
                        if mapped == usize::MAX {
                            let new_idx = base_count + kept_meta.len();
                            let old_meta = shard.rows.rows.get(*slot).ok_or_else(corrupt)?;
                            let old_vec = shard.rows.vector(old_meta.vec_slot);
                            let new_vec_slot = base_count + kept_vecs.len() / dim;
                            kept_vecs.extend_from_slice(old_vec);
                            let mut cloned = old_meta.clone();
                            cloned.vec_slot = new_vec_slot;
                            kept_meta.push(cloned);
                            remap_inb[rel] = new_idx;
                        }
                        *slot = remap_inb[rel];
                    }
                }
            }
            shard.rows.rows.truncate(base_count);
            shard.rows.vectors.truncate(base_count * dim);
            shard.rows.rows.extend(kept_meta);
            shard.rows.vectors.extend(kept_vecs);
        }

        let mut inbound_targets = Vec::new();
        for &donor_s in donors {
            check()?;
            read_indexed_outbound_slice(
                &dir.join(format!("outb-e{epoch}-r{round}-{donor_s:06}.bin")),
                shard_id as usize,
                manifest,
                &mut shard.rows,
                &mut inbound_targets,
            )?;
        }
        for (dst_l, new_idx) in inbound_targets {
            next_members
                .get_mut(dst_l as usize)
                .ok_or_else(corrupt)?
                .push(new_idx);
            *dirty.get_mut(dst_l as usize).ok_or_else(corrupt)? = true;
        }
        for ((leaf, mut mem), changed) in shard.leaves.iter_mut().zip(next_members).zip(dirty) {
            if changed {
                mem.sort_unstable_by(|&a, &b| shard.rows.rows[a].id.cmp(&shard.rows.rows[b].id));
                leaf.centroid = compute_leaf_centroid(&kernel, &shard.rows, &mem)?;
            }
            leaf.members = mem;
        }
    }

    let next_r = round + 1;
    write_compact_shard_state(
        &dir.join(format!("mem-e{epoch}-r{next_r}-{shard_id:06}.bin")),
        &dir.join(format!("centers-e{epoch}-r{next_r}-{shard_id:06}.bin")),
        &dir.join(format!("inb-e{epoch}-r{next_r}-{shard_id:06}.bin")),
        manifest,
        &shard,
        base_count,
        quota,
    )?;
    Ok(())
}

fn assign_disk_leaf_keys(
    dir: &Path,
    epoch: u64,
    final_round: u8,
    plan: &ShardPlan,
    coarse_shards: u32,
    manifest: &IndexManifest,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<(Vec<Vec<AssignedLeafKey>>, TreeLeafTotals)> {
    let mut shard_tks: Vec<Vec<TreeKey>> = Vec::with_capacity(coarse_shards as usize);
    for s in 0..coarse_shards {
        if s.is_multiple_of(64) {
            check()?;
        }
        let r_use = if s < plan.light_shard_start {
            final_round
        } else {
            0
        };
        let tks = read_shard_leaf_trees(
            &dir.join(format!("centers-e{epoch}-r{r_use}-{s:06}.bin")),
            manifest,
        )?;
        shard_tks.push(tks);
    }
    assign_leaf_partition_keys_from_trees(&shard_tks)
}
