//! Unified sharded Bulk Build pipeline across resident memory, bounded disk spill,
//! and multi-worker collaborative `BuildUnit` execution.
//!
//! One stage graph (`Plan -> Scatter -> Dedup -> Leaves -> [Halo Refine] -> Emit`)
//! runs over `StageStore`: `StageStore::Memory` keeps intermediate buckets and shards
//! in process memory (`0 B` scratch disk I/O, shared `f32` vector slab), while
//! `StageStore::Disk` spills checksummed per-unit files under `ScratchQuota` and
//! coordinates multi-worker unit leases through the KV backend when `distributed_workers > 1`.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::api::{
    BulkBuildReport, Error, ErrorKind, LogicalIndexId, OperationOptions, PartitionKey, Result,
};
use crate::bulk::files::io_error;
use crate::bulk::sharded::{
    CompactMeta, IdFrameEntry, LeafPlan, MAX_HALO_LEAVES_CAP, MoveProposal, PrecomputedTreeHalo,
    ScratchQuota, ShardRows, ShardState, apply_moves_to_shard, arbitrate_tree_proposals,
    assign_leaf_partition_keys, build_and_encode_upper_tree, build_shard_leaves, build_tree_halo,
    compute_shard_remap, encode_leaf_shard, encode_record_group_bucket, ingest_segment_into_slab,
    ingest_segment_range, parallel_chunks, prepare_in_memory_for_emit, read_bucket_id_refs,
    read_indexed_ids_slice, read_indexed_locations_slice, read_indexed_outbound_slice,
    read_indexed_shard_slice, read_leaf_keys_plan, read_proposals, read_shard_centers,
    read_shard_counts, read_shard_state_file, refine_single_light_shard, refinement_per_leaf_bytes,
    score_halo_shard, score_shard_with_tree_halo, verify_id_bucket, write_bucket_id_refs,
    write_indexed_ids, write_indexed_outbound_rows, write_indexed_shards, write_leaf_keys_plan,
    write_proposals, write_shard_state_files,
};
use crate::bulk::{InputSnapshot, LtHash, ShardPlan};
use crate::observe::labels::Operation;
use crate::refinement::candidate_scratch_bytes;
use crate::runtime::bulk_scheduler::{
    BulkAuthority, authorize_raw, expiry, non_zero_token, now_ms,
};
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{AdmissionBudget, Backend, HardLimits, ScanLimits, WriteTxn};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    BuildDescriptor, BuildPhase, BuildProgress, BuildUnit, BuildWorkspace, IndexLifecycle,
    IndexManifest, PartitionCentroid, PersistentValue,
};
use crate::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

use super::{
    CommitStep, MAX_WRITER_LANES, OperationContext, RetryPolicy, UnitLease, WriteFence,
    advance_loading_progress, building_in_flight, check_writer_fence, clear_stale_serving_data,
    commit_raw_prepared, corrupt, drop_guarded_options, effective_budget, finalize_and_seal_loaded,
    lock_sched_read, retry_raw_write, retry_read, seal_loaded, validate_split_keys_in_index,
};

const STAGE_SCATTER: u8 = 1;
const STAGE_DEDUP: u8 = 2;
const STAGE_LEAVES: u8 = 3;
const STAGE_REFINE_BASE: u8 = 10; // 10 + 3*r = Propose, 11 + 3*r = Extract, 12 + 3*r = Apply
const STAGE_EMIT_RECORDS: u8 = 30;
const STAGE_EMIT_TREES: u8 = 31;
const STAGE_EMIT_LEAVES: u8 = 32;

type UpperTreeBucket = (TreeKey, u64, Vec<(PartitionKey, PartitionCentroid)>);
type RecordLocationEntry = (Bytes, TreeKey, PartitionKey);
type NonEmptyRunSources = Vec<(u32, Vec<usize>)>;
type InboundMoveRow = (u32, CompactMeta, Vec<f32>);

fn shard_final_round(shard: u32, final_round: u8, plan: &ShardPlan) -> u8 {
    if shard < plan.light_shard_start {
        final_round
    } else {
        0
    }
}

/// Fencing mode shared across writer `BuildUnit` stages and frozen `STAGE_VALIDATE` slices.
#[derive(Clone, Copy)]
pub(crate) enum StageFence {
    Write(WriteFence),
    Validate { ws_token: [u8; 32], epoch: u64 },
}

impl StageFence {
    #[must_use]
    pub(crate) const fn epoch(self) -> u64 {
        match self {
            Self::Write(wf) => wf.epoch,
            Self::Validate { epoch, .. } => epoch,
        }
    }

    #[must_use]
    pub(crate) const fn operation(self) -> Operation {
        match self {
            Self::Write(_) => Operation::RunBulkBuild,
            Self::Validate { .. } => Operation::PublishBulkBuild,
        }
    }

    fn warm_keys(
        self,
        id: LogicalIndexId,
        bulk_authority: Option<BulkAuthority>,
        ukey: Option<LogicalKey>,
    ) -> Vec<LogicalKey> {
        let mut warm = Vec::with_capacity(6);
        warm.push(LogicalKey::Manifest(id));
        warm.push(LogicalKey::BuildWorkspace(id));
        if matches!(self, Self::Validate { .. }) {
            warm.push(LogicalKey::BuildProgress(id));
        }
        warm.push(LogicalKey::BuildSchedule(id));
        if let Some(auth) = bulk_authority {
            warm.push(auth.slot_key(id));
        }
        if let Some(uk) = ukey {
            warm.push(uk);
        }
        warm
    }

    pub(crate) async fn check<T: WriteTxn>(
        self,
        txn: &mut WriteLogicalTxn<'_, T>,
        manifest: &IndexManifest,
    ) -> Result<()> {
        match self {
            Self::Write(wf) => check_writer_fence(txn, manifest, wf).await,
            Self::Validate { ws_token, epoch } => {
                fence_validation(txn, manifest, ws_token, epoch).await?;
                Ok(())
            }
        }
    }
}

/// Verifies in a write transaction that `manifest` remains in frozen `(Building, Validating)`
/// at `(ws_token, expected_epoch)` with writers revoked (`writer_token == [0; 32]`).
pub(crate) async fn fence_validation<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    index: &IndexManifest,
    ws_token: [u8; 32],
    expected_epoch: u64,
) -> Result<BuildProgress> {
    building_in_flight(txn, index).await?;
    let id = index.logical_index_id();
    let ws = txn.get_for_update(LogicalKey::BuildWorkspace(id)).await?;
    let prog = match txn.get(LogicalKey::BuildProgress(id)).await? {
        Some(PersistentValue::BuildProgress(p))
            if p.phase == BuildPhase::Validating && p.epoch == expected_epoch =>
        {
            validate_split_keys_in_index(id, &p.validation_split_keys)?;
            p
        }
        Some(PersistentValue::BuildProgress(_)) => {
            return Err(Error::new(ErrorKind::BulkBuildSuperseded));
        }
        _ => return Err(corrupt()),
    };
    match ws {
        Some(PersistentValue::BuildWorkspace(w)) => {
            if let Some(kind) = w.failure {
                return Err(Error::new(kind));
            }
            if w.token != ws_token || w.epoch != expected_epoch || w.writer_token != [0; 32] {
                return Err(Error::new(ErrorKind::BulkBuildSuperseded));
            }
        }
        _ => return Err(corrupt()),
    }
    Ok(prog)
}

#[derive(Default)]
pub(crate) struct UnitOutput {
    pub records: u64,
    pub kv_count: u64,
    pub aux_count: u64,
    pub kv_lthash: LtHash,
    pub aux_lthash: LtHash,
    pub aux2_lthash: LtHash,
}

pub(crate) struct StageEnv<B: Backend> {
    pub backend: Arc<B>,
    pub control: OperationOptions,
    pub bulk_authority: Option<BulkAuthority>,
    pub sched_lease_lock: Option<Arc<tokio::sync::RwLock<()>>>,
    pub distributed_workers: usize,
    pub persist_single_worker: bool,
    pub in_memory: bool,
    pub writer_lanes: usize,
    pub commit_sem: Arc<Semaphore>,
    pub admission: Option<Arc<dyn Any + Send + Sync>>,
    pub manifest: IndexManifest,
    pub fence: StageFence,
    pub unit_lease: Duration,
    pub budget: AdmissionBudget,
    pub retry: RetryPolicy,
}

impl<B: Backend> Clone for StageEnv<B> {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            control: self.control.clone(),
            bulk_authority: self.bulk_authority,
            sched_lease_lock: self.sched_lease_lock.clone(),
            distributed_workers: self.distributed_workers,
            persist_single_worker: self.persist_single_worker,
            in_memory: self.in_memory,
            writer_lanes: self.writer_lanes,
            commit_sem: Arc::clone(&self.commit_sem),
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
    pub(crate) async fn spawn_cpu<T: Send + 'static>(
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
        kvs: Vec<(Bytes, Bytes)>,
    ) -> Result<()> {
        if kvs.is_empty() {
            return Ok(());
        }
        let StageFence::Write(wf) = self.fence else {
            return Err(corrupt());
        };
        let chunks = super::split_into_chunks(&kvs, self.budget, lease.is_some())?;
        let fanout = if lease.is_none() {
            self.writer_lanes
                .clamp(1, MAX_WRITER_LANES)
                .min(chunks.len())
        } else {
            1
        };
        if fanout <= 1 {
            for chunk in chunks {
                let _permit = self
                    .commit_sem
                    .acquire()
                    .await
                    .map_err(|_| Error::new(ErrorKind::Other))?;
                super::commit_one_chunk(
                    Arc::clone(&self.backend),
                    &self.control,
                    self.bulk_authority,
                    self.sched_lease_lock.as_deref(),
                    &self.manifest,
                    wf,
                    lease,
                    chunk,
                    self.budget,
                    self.retry,
                )
                .await?;
            }
            return Ok(());
        }

        let queue = Arc::new(Mutex::new(chunks.into_iter()));
        let mut workers = tokio::task::JoinSet::<Result<()>>::new();
        for _ in 0..fanout {
            let q = Arc::clone(&queue);
            let env_c = self.clone();
            workers.spawn(async move {
                loop {
                    crate::runtime::check_control(&env_c.control)?;
                    let Some(chunk) = lock_mutex(&q).next() else {
                        return Ok(());
                    };
                    let _permit = env_c
                        .commit_sem
                        .acquire()
                        .await
                        .map_err(|_| Error::new(ErrorKind::Other))?;
                    super::commit_one_chunk(
                        Arc::clone(&env_c.backend),
                        &env_c.control,
                        env_c.bulk_authority,
                        env_c.sched_lease_lock.as_deref(),
                        &env_c.manifest,
                        wf,
                        None,
                        chunk,
                        env_c.budget,
                        env_c.retry,
                    )
                    .await?;
                }
            });
        }
        while let Some(res) = workers.join_next().await {
            res.map_err(|e| Error::with_source(ErrorKind::Other, e))??;
        }
        Ok(())
    }
}

fn vec_mutexes<T>(n: u32) -> Vec<Mutex<Vec<T>>> {
    (0..n).map(|_| Mutex::new(Vec::new())).collect()
}

fn lock_mutex<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct MemStore {
    scatter_ids: Mutex<Vec<Vec<IdFrameEntry>>>,
    scatter_shards: Mutex<Vec<ShardRows>>,
    dedup_ids: Vec<Mutex<Vec<IdFrameEntry>>>,
    shards: Vec<Mutex<Option<ShardState>>>,
    proposals: Vec<Mutex<Vec<MoveProposal>>>,
    outbound: Vec<Mutex<Vec<InboundMoveRow>>>,
    locations: Vec<Mutex<Vec<(Bytes, TreeKey, PartitionKey)>>>,
}

impl MemStore {
    fn new(id_buckets: u32, coarse_shards: u32, heavy_shards: u32) -> Self {
        Self {
            scatter_ids: Mutex::new(Vec::new()),
            scatter_shards: Mutex::new(Vec::new()),
            dedup_ids: vec_mutexes(id_buckets),
            shards: (0..coarse_shards).map(|_| Mutex::new(None)).collect(),
            proposals: vec_mutexes(heavy_shards),
            outbound: vec_mutexes(heavy_shards),
            locations: vec_mutexes(id_buckets),
        }
    }
}

struct DiskStore {
    dir: PathBuf,
    epoch: u64,
    quota: Arc<ScratchQuota>,
}

impl DiskStore {
    fn shard_paths(&self, round: u8, s: u32) -> (PathBuf, PathBuf, PathBuf) {
        let ep = self.epoch;
        (
            self.dir.join(format!("shard-e{ep}-r{round}-{s:06}.bin")),
            self.dir.join(format!("centers-e{ep}-r{round}-{s:06}.bin")),
            self.dir.join(format!("loc-e{ep}-r{round}-{s:06}.bin")),
        )
    }

    fn leafkeys_path(&self) -> PathBuf {
        self.dir.join(format!("leafkeys-e{}.bin", self.epoch))
    }
}

/// Pluggable intermediate state backing the single sharded Bulk Build pipeline.
enum StageStore {
    Memory(MemStore),
    Disk(DiskStore),
}

enum LeafInputSources {
    Memory,
    Disk(Vec<NonEmptyRunSources>),
}

impl StageStore {
    async fn load_cached_leaf_plan<B: Backend>(
        &self,
        env: &StageEnv<B>,
        lock: &Arc<File>,
        coarse_shards: u32,
        tree_buckets: u32,
    ) -> Result<Option<LeafPlan>> {
        let Self::Disk(disk) = self else {
            return Ok(None);
        };
        let path = disk.leafkeys_path();
        let manifest = env.manifest.clone();
        let guard = Arc::clone(lock);
        env.spawn_cpu(move |_ctrl| {
            let _lock = guard;
            if path.exists() {
                Ok(Some(read_leaf_keys_plan(
                    &path,
                    coarse_shards,
                    tree_buckets,
                    &manifest,
                )?))
            } else {
                Ok(None)
            }
        })
        .await
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "scatter dispatches segment ingestion across either the resident vector slab or disk run units"
    )]
    async fn run_scatter<B: Backend>(
        self: &Arc<Self>,
        env: &StageEnv<B>,
        lock: &Arc<File>,
        snapshot: &InputSnapshot,
        plan: &ShardPlan,
        hard_limits: HardLimits,
        segs_per_run: usize,
        scatter_units: u32,
        scatter_conc: usize,
        parallelism: usize,
    ) -> Result<(u64, LtHash)> {
        match self.as_ref() {
            Self::Memory(_) => {
                let snap_c = snapshot.clone();
                let manifest_c = env.manifest.clone();
                let plan_c = plan.clone();
                let store_c = Arc::clone(self);
                let guard = Arc::clone(lock);
                env.spawn_cpu(move |ctrl| {
                    let _lock = guard;
                    let check = || crate::runtime::check_control(&ctrl);
                    let Self::Memory(mem) = store_c.as_ref() else {
                        return Err(corrupt());
                    };
                    ingest_all_into_memory_slab(
                        &snap_c,
                        &manifest_c,
                        &plan_c,
                        hard_limits,
                        parallelism,
                        mem,
                        &check,
                    )
                })
                .await
            }
            Self::Disk(_) => {
                let seg_count = snapshot.segments().len();
                let store_c = Arc::clone(self);
                let snap_c = snapshot.clone();
                let plan_c = plan.clone();
                let scatter_done =
                    drive_cpu_stage(env, lock, STAGE_SCATTER, scatter_units, scatter_conc, {
                        move |unit, manifest, check| {
                            let Self::Disk(disk) = store_c.as_ref() else {
                                return Err(corrupt());
                            };
                            let seg_start = (unit as usize) * segs_per_run;
                            let seg_end = (seg_start + segs_per_run).min(seg_count);
                            let seg = ingest_segment_range(
                                &snap_c,
                                seg_start..seg_end,
                                manifest,
                                &plan_c,
                                hard_limits,
                                check,
                            )?;
                            let ep = disk.epoch;
                            write_indexed_ids(
                                &disk.dir.join(format!("ids-e{ep}-{unit:06}.bin")),
                                &seg.ids_by_bucket,
                                &disk.quota,
                            )?;
                            write_indexed_shards(
                                &disk.dir.join(format!("vec-e{ep}-{unit:06}.bin")),
                                &disk.dir.join(format!("cnt-e{ep}-{unit:06}.bin")),
                                manifest,
                                &seg.rows_by_shard,
                                &disk.quota,
                            )?;
                            Ok(UnitOutput {
                                records: seg.records,
                                aux_lthash: seg.id_lthash,
                                ..UnitOutput::default()
                            })
                        }
                    })
                    .await?;
                let mut lt = LtHash::ZERO;
                let mut recs = 0_u64;
                for u in &scatter_done {
                    recs += u.records;
                    lt.add_assign(u.aux_lthash);
                }
                Ok((recs, lt))
            }
        }
    }

    fn take_scatter_ids(
        &self,
        bucket: u32,
        scatter_units: u32,
        check: &(dyn Fn() -> Result<()> + Sync),
    ) -> Result<Vec<IdFrameEntry>> {
        match self {
            Self::Memory(mem) => {
                let mut guard = lock_mutex(&mem.scatter_ids);
                Ok(std::mem::take(
                    guard.get_mut(bucket as usize).ok_or_else(corrupt)?,
                ))
            }
            Self::Disk(disk) => {
                let mut ids = Vec::new();
                for run in 0..scatter_units {
                    if run.is_multiple_of(32) {
                        check()?;
                    }
                    read_indexed_ids_slice(
                        &disk.dir.join(format!("ids-e{}-{run:06}.bin", disk.epoch)),
                        bucket as usize,
                        &mut ids,
                    )?;
                }
                Ok(ids)
            }
        }
    }

    fn save_dedup_ids(&self, bucket: u32, sorted: Vec<IdFrameEntry>) -> Result<()> {
        match self {
            Self::Memory(mem) => {
                *lock_mutex(mem.dedup_ids.get(bucket as usize).ok_or_else(corrupt)?) = sorted;
                Ok(())
            }
            Self::Disk(disk) => {
                write_bucket_id_refs(
                    &disk
                        .dir
                        .join(format!("idref-e{}-{bucket:06}.bin", disk.epoch)),
                    &sorted,
                    &disk.quota,
                )?;
                Ok(())
            }
        }
    }

    async fn reclaim_scatter_ids<B: Backend>(
        &self,
        env: &StageEnv<B>,
        scatter_units: u32,
        lock: &Arc<File>,
    ) -> Result<()> {
        match self {
            Self::Memory(mem) => {
                *lock_mutex(&mem.scatter_ids) = Vec::new();
                Ok(())
            }
            Self::Disk(disk) => {
                reclaim_files(
                    env,
                    disk,
                    lock,
                    (0..scatter_units)
                        .map(|run| format!("ids-e{}-{run:06}.bin", disk.epoch))
                        .collect(),
                )
                .await
            }
        }
    }

    async fn prepare_leaf_inputs<B: Backend>(
        self: &Arc<Self>,
        env: &StageEnv<B>,
        lock: &Arc<File>,
        scatter_units: u32,
        coarse_shards: u32,
        plan: &ShardPlan,
    ) -> Result<Option<Arc<LeafInputSources>>> {
        match self.as_ref() {
            Self::Memory(_) => {
                let manifest_c = env.manifest.clone();
                let plan_c = plan.clone();
                let store_c = Arc::clone(self);
                env.spawn_cpu(move |_ctrl| {
                    let Self::Memory(mem) = store_c.as_ref() else {
                        return Err(corrupt());
                    };
                    remap_memory_scatter_shards(mem, &manifest_c, &plan_c)
                })
                .await?;
                Ok(Some(Arc::new(LeafInputSources::Memory)))
            }
            Self::Disk(disk) => {
                let dir = disk.dir.clone();
                let epoch = disk.epoch;
                let manifest = env.manifest.clone();
                let plan_c = plan.clone();
                let guard = Arc::clone(lock);
                let run_res = env
                    .spawn_cpu(move |ctrl| {
                        let _lock = guard;
                        compute_non_empty_runs_by_shard(
                            &dir,
                            epoch,
                            scatter_units,
                            coarse_shards,
                            &manifest,
                            &plan_c,
                            &ctrl,
                        )
                    })
                    .await;
                match run_res {
                    Ok(v) => Ok(Some(Arc::new(LeafInputSources::Disk(v)))),
                    Err(_)
                        if env.distributed_workers > 1
                            && is_stage_complete(env, STAGE_LEAVES, coarse_shards).await? =>
                    {
                        Ok(None)
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn take_scatter_shard(
        &self,
        shard: u32,
        manifest: &IndexManifest,
        sources: &LeafInputSources,
        check: &(dyn Fn() -> Result<()> + Sync),
    ) -> Result<ShardRows> {
        let dim = manifest.config().dimension();
        match (self, sources) {
            (Self::Memory(mem), LeafInputSources::Memory) => {
                let mut guard = lock_mutex(&mem.scatter_shards);
                let slot = guard.get_mut(shard as usize).ok_or_else(corrupt)?;
                let replacement = match slot.slab.clone() {
                    Some(pool) => ShardRows::with_slab(dim, pool),
                    None => ShardRows::new(dim),
                };
                Ok(std::mem::replace(slot, replacement))
            }
            (Self::Disk(disk), LeafInputSources::Disk(runs)) => {
                let mut rows = ShardRows::new(dim);
                for (idx, (run, src_shards)) in runs[shard as usize].iter().enumerate() {
                    if idx.is_multiple_of(32) {
                        check()?;
                    }
                    let vpath = disk.dir.join(format!("vec-e{}-{run:06}.bin", disk.epoch));
                    for &src_s in src_shards {
                        read_indexed_shard_slice(&vpath, src_s, manifest, &mut rows)?;
                    }
                }
                Ok(rows)
            }
            _ => Err(corrupt()),
        }
    }

    async fn reclaim_scatter_shards<B: Backend>(
        &self,
        env: &StageEnv<B>,
        scatter_units: u32,
        lock: &Arc<File>,
    ) -> Result<()> {
        match self {
            Self::Memory(mem) => {
                *lock_mutex(&mem.scatter_shards) = Vec::new();
                Ok(())
            }
            Self::Disk(disk) => {
                let mut obsolete = Vec::with_capacity((scatter_units as usize) * 2);
                for run in 0..scatter_units {
                    obsolete.push(format!("vec-e{}-{run:06}.bin", disk.epoch));
                    obsolete.push(format!("cnt-e{}-{run:06}.bin", disk.epoch));
                }
                reclaim_files(env, disk, lock, obsolete).await
            }
        }
    }

    fn save_shard_state(
        &self,
        round: u8,
        shard: u32,
        state: ShardState,
        plan: &ShardPlan,
        manifest: &IndexManifest,
    ) -> Result<()> {
        match self {
            Self::Memory(mem) => {
                *lock_mutex(mem.shards.get(shard as usize).ok_or_else(corrupt)?) = Some(state);
                Ok(())
            }
            Self::Disk(disk) => {
                let (s_path, c_path, loc_path) = disk.shard_paths(round, shard);
                write_shard_state_files(
                    &s_path,
                    &c_path,
                    &loc_path,
                    manifest,
                    plan,
                    &state,
                    &disk.quota,
                )
            }
        }
    }

    fn load_shard_centers(
        &self,
        round: u8,
        shard: u32,
        manifest: &IndexManifest,
    ) -> Result<Vec<(TreeKey, usize, PartitionCentroid)>> {
        match self {
            Self::Memory(mem) => {
                let guard = lock_mutex(mem.shards.get(shard as usize).ok_or_else(corrupt)?);
                let st = guard.as_ref().ok_or_else(corrupt)?;
                Ok(st
                    .leaves
                    .iter()
                    .map(|l| (l.tree_key.clone(), l.members.len(), l.centroid.clone()))
                    .collect())
            }
            Self::Disk(disk) => {
                let (_, c_path, _) = disk.shard_paths(round, shard);
                read_shard_centers(&c_path, manifest)
            }
        }
    }

    fn with_shard_state<R>(
        &self,
        round: u8,
        shard: u32,
        manifest: &IndexManifest,
        f: impl FnOnce(&ShardState) -> Result<R>,
    ) -> Result<R> {
        match self {
            Self::Memory(mem) => {
                let guard = lock_mutex(mem.shards.get(shard as usize).ok_or_else(corrupt)?);
                f(guard.as_ref().ok_or_else(corrupt)?)
            }
            Self::Disk(disk) => {
                let (s_path, c_path, _) = disk.shard_paths(round, shard);
                let state = read_shard_state_file(&s_path, &c_path, manifest)?;
                f(&state)
            }
        }
    }

    fn save_proposals(&self, round: u8, shard: u32, proposals: Vec<MoveProposal>) -> Result<u64> {
        let count = proposals.len() as u64;
        match self {
            Self::Memory(mem) => {
                *lock_mutex(mem.proposals.get(shard as usize).ok_or_else(corrupt)?) = proposals;
            }
            Self::Disk(disk) => {
                if !proposals.is_empty() {
                    write_proposals(
                        &disk
                            .dir
                            .join(format!("prop-e{}-r{round}-{shard:06}.bin", disk.epoch)),
                        &proposals,
                        &disk.quota,
                    )?;
                }
            }
        }
        Ok(count)
    }

    fn take_proposals(&self, round: u8, shard: u32, expected: u64) -> Result<Vec<MoveProposal>> {
        if expected == 0 {
            return Ok(Vec::new());
        }
        let props = match self {
            Self::Memory(mem) => std::mem::take(&mut *lock_mutex(
                mem.proposals.get(shard as usize).ok_or_else(corrupt)?,
            )),
            Self::Disk(disk) => read_proposals(
                &disk
                    .dir
                    .join(format!("prop-e{}-r{round}-{shard:06}.bin", disk.epoch)),
            )?,
        };
        if props.len() as u64 != expected {
            return Err(corrupt());
        }
        Ok(props)
    }

    async fn reclaim_proposals<B: Backend>(
        &self,
        env: &StageEnv<B>,
        round: u8,
        prop_counts: &[u64],
        lock: &Arc<File>,
    ) -> Result<()> {
        let Self::Disk(disk) = self else {
            return Ok(());
        };
        let obsolete: Vec<String> = prop_counts
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(s, _)| format!("prop-e{}-r{round}-{s:06}.bin", disk.epoch))
            .collect();
        reclaim_files(env, disk, lock, obsolete).await
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "outbound move extraction drives either zero-copy memory routing or disk stage units"
    )]
    async fn extract_outbound<B: Backend>(
        self: &Arc<Self>,
        env: &StageEnv<B>,
        lock: &Arc<File>,
        stage_ext: u8,
        round: u8,
        heavy_shards: u32,
        conc: usize,
        accepted: &Arc<AcceptedRoundMoves>,
    ) -> Result<()> {
        match self.as_ref() {
            Self::Memory(_) => {
                let store_c = Arc::clone(self);
                let acc = Arc::clone(accepted);
                env.spawn_cpu(move |_ctrl| {
                    let Self::Memory(mem) = store_c.as_ref() else {
                        return Err(corrupt());
                    };
                    extract_memory_outbound(mem, &acc)
                })
                .await
            }
            Self::Disk(_) => {
                let store_c = Arc::clone(self);
                let acc = Arc::clone(accepted);
                drive_cpu_stage(
                    env,
                    lock,
                    stage_ext,
                    heavy_shards,
                    conc,
                    move |s, manifest, check| {
                        let Self::Disk(disk) = store_c.as_ref() else {
                            return Err(corrupt());
                        };
                        extract_disk_outbound(
                            disk,
                            round,
                            s,
                            heavy_shards as usize,
                            manifest,
                            &acc,
                            check,
                        )?;
                        Ok(UnitOutput::default())
                    },
                )
                .await?;
                Ok(())
            }
        }
    }

    async fn reclaim_refine_round<B: Backend>(
        &self,
        env: &StageEnv<B>,
        round: u8,
        heavy_shards: u32,
        lock: &Arc<File>,
    ) -> Result<()> {
        let Self::Disk(disk) = self else {
            return Ok(());
        };
        let ep = disk.epoch;
        let mut obsolete = Vec::with_capacity((heavy_shards as usize) * 5);
        for s in 0..heavy_shards {
            for kind in ["shard", "centers", "loc", "prop", "outb"] {
                obsolete.push(format!("{kind}-e{ep}-r{round}-{s:06}.bin"));
            }
        }
        reclaim_files(env, disk, lock, obsolete).await
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "finalizes global leaf keys, quantizes/drops resident slab in Memory mode, or persists leafkeys-e{ep}.bin on Disk"
    )]
    async fn finalize_leaf_plan<B: Backend>(
        self: &Arc<Self>,
        env: &StageEnv<B>,
        lock: &Arc<File>,
        scatter_records: u64,
        final_round: u8,
        plan: &ShardPlan,
        tree_buckets: u32,
        parallelism: usize,
    ) -> Result<LeafPlan> {
        let coarse_shards = plan.total_shards;
        match self.as_ref() {
            Self::Memory(_) => {
                let store_c = Arc::clone(self);
                let plan_c = plan.clone();
                let dim = env.manifest.config().dimension();
                env.spawn_cpu(move |ctrl| {
                    let check = || crate::runtime::check_control(&ctrl);
                    let Self::Memory(mem) = store_c.as_ref() else {
                        return Err(corrupt());
                    };
                    let mut all_shards: Vec<Option<ShardState>> =
                        mem.shards.iter().map(|m| lock_mutex(m).take()).collect();
                    let shard_tks: Vec<Vec<TreeKey>> = all_shards
                        .iter()
                        .map(|opt| {
                            opt.as_ref().map_or_else(Vec::new, |st| {
                                st.leaves.iter().map(|l| l.tree_key.clone()).collect()
                            })
                        })
                        .collect();
                    let (assigned, trees) =
                        assign_leaf_partition_keys(&shard_tks, tree_buckets, |tks| tks.iter())?;
                    let loc_buckets = prepare_in_memory_for_emit(
                        &plan_c,
                        dim,
                        &mut all_shards,
                        &assigned,
                        parallelism,
                        &check,
                    )?;
                    for (slot_m, locs) in mem.locations.iter().zip(loc_buckets) {
                        *lock_mutex(slot_m) = locs;
                    }
                    for (slot_m, st_opt) in mem.shards.iter().zip(all_shards) {
                        *lock_mutex(slot_m) = st_opt;
                    }
                    Ok(LeafPlan {
                        scatter_records,
                        final_round,
                        assigned,
                        trees,
                    })
                })
                .await
            }
            Self::Disk(disk) => {
                let path = disk.leafkeys_path();
                let store_c = Arc::clone(self);
                let manifest = env.manifest.clone();
                let plan_c = plan.clone();
                let quota_c = Arc::clone(&disk.quota);
                let guard = Arc::clone(lock);
                let leaf_plan_res = env
                    .spawn_cpu(move |ctrl| {
                        let _lock = guard;
                        if path.exists() {
                            return read_leaf_keys_plan(
                                &path,
                                coarse_shards,
                                tree_buckets,
                                &manifest,
                            );
                        }
                        let check = || crate::runtime::check_control(&ctrl);
                        let mut shard_tks = Vec::with_capacity(coarse_shards as usize);
                        for s in 0..coarse_shards {
                            if s.is_multiple_of(64) {
                                check()?;
                            }
                            let r_use = shard_final_round(s, final_round, &plan_c);
                            let centers = store_c.load_shard_centers(r_use, s, &manifest)?;
                            shard_tks
                                .push(centers.into_iter().map(|(tk, _, _)| tk).collect::<Vec<_>>());
                        }
                        let (assigned, trees) =
                            assign_leaf_partition_keys(&shard_tks, tree_buckets, |tks| tks.iter())?;
                        let leaf_plan = LeafPlan {
                            scatter_records,
                            final_round,
                            assigned,
                            trees,
                        };
                        write_leaf_keys_plan(&path, &leaf_plan, tree_buckets, &quota_c)?;
                        Ok(leaf_plan)
                    })
                    .await;

                match leaf_plan_res {
                    Ok(v) => Ok(v),
                    Err(e) if env.distributed_workers > 1 => {
                        match self
                            .load_cached_leaf_plan(env, lock, coarse_shards, tree_buckets)
                            .await?
                        {
                            Some(v) => Ok(v),
                            None => Err(e),
                        }
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn take_record_bucket_inputs(
        &self,
        bucket: u32,
        leaf_plan: &LeafPlan,
        plan: &ShardPlan,
        check: &(dyn Fn() -> Result<()> + Sync),
    ) -> Result<(Vec<IdFrameEntry>, Vec<RecordLocationEntry>)> {
        let (id_refs, mut locs) = match self {
            Self::Memory(mem) => (
                std::mem::take(&mut *lock_mutex(
                    mem.dedup_ids.get(bucket as usize).ok_or_else(corrupt)?,
                )),
                std::mem::take(&mut *lock_mutex(
                    mem.locations.get(bucket as usize).ok_or_else(corrupt)?,
                )),
            ),
            Self::Disk(disk) => {
                let id_refs = read_bucket_id_refs(
                    &disk
                        .dir
                        .join(format!("idref-e{}-{bucket:06}.bin", disk.epoch)),
                )?;
                let mut locs = Vec::with_capacity(id_refs.len());
                for (s, shard_keys) in leaf_plan.assigned.iter().enumerate() {
                    if shard_keys.is_empty() {
                        continue;
                    }
                    if s.is_multiple_of(64) {
                        check()?;
                    }
                    let r_use = shard_final_round(s as u32, leaf_plan.final_round, plan);
                    let (_, _, loc_path) = disk.shard_paths(r_use, s as u32);
                    read_indexed_locations_slice(
                        &loc_path,
                        bucket as usize,
                        shard_keys,
                        &leaf_plan.trees,
                        &mut locs,
                    )?;
                }
                (id_refs, locs)
            }
        };
        if id_refs.len() != locs.len() {
            return Err(corrupt());
        }
        locs.sort_unstable_by(|x, y| x.0.cmp(&y.0));
        Ok((id_refs, locs))
    }

    async fn reclaim_record_inputs<B: Backend>(
        &self,
        env: &StageEnv<B>,
        id_buckets: u32,
        leaf_plan: &LeafPlan,
        plan: &ShardPlan,
        lock: &Arc<File>,
    ) -> Result<()> {
        let Self::Disk(disk) = self else {
            return Ok(());
        };
        let ep = disk.epoch;
        let mut obsolete = Vec::with_capacity(id_buckets as usize + leaf_plan.assigned.len());
        for b in 0..id_buckets {
            obsolete.push(format!("idref-e{ep}-{b:06}.bin"));
        }
        for (s, keys) in leaf_plan.assigned.iter().enumerate() {
            if !keys.is_empty() {
                let r_use = shard_final_round(s as u32, leaf_plan.final_round, plan);
                obsolete.push(format!("loc-e{ep}-r{r_use}-{s:06}.bin"));
            }
        }
        reclaim_files(env, disk, lock, obsolete).await
    }

    fn load_tree_bucket_children(
        &self,
        tb: u32,
        leaf_plan: &LeafPlan,
        plan: &ShardPlan,
        manifest: &IndexManifest,
        bucket_trees: &mut BTreeMap<Bytes, UpperTreeBucket>,
        check: &(dyn Fn() -> Result<()> + Sync),
    ) -> Result<()> {
        for (s, shard_keys) in leaf_plan.assigned.iter().enumerate() {
            if !shard_keys.iter().any(|ak| {
                !ak.is_single_leaf_root && leaf_plan.trees[ak.tree_id as usize].tree_bucket == tb
            }) {
                continue;
            }
            check()?;
            match self {
                Self::Memory(mem) => {
                    let guard = lock_mutex(mem.shards.get(s).ok_or_else(corrupt)?);
                    let st = guard.as_ref().ok_or_else(corrupt)?;
                    if st.leaves.len() != shard_keys.len() {
                        return Err(corrupt());
                    }
                    for (leaf, ak) in st.leaves.iter().zip(shard_keys) {
                        if !ak.is_single_leaf_root
                            && leaf_plan.trees[ak.tree_id as usize].tree_bucket == tb
                        {
                            bucket_trees
                                .get_mut(leaf.tree_key.as_bytes())
                                .ok_or_else(corrupt)?
                                .2
                                .push((ak.partition_key, leaf.centroid.clone()));
                        }
                    }
                }
                Self::Disk(disk) => {
                    let r_use = shard_final_round(s as u32, leaf_plan.final_round, plan);
                    let (_, c_path, _) = disk.shard_paths(r_use, s as u32);
                    let centers = read_shard_centers(&c_path, manifest)?;
                    if centers.len() != shard_keys.len() {
                        return Err(corrupt());
                    }
                    for ((tk, _, centroid), ak) in centers.into_iter().zip(shard_keys) {
                        if !ak.is_single_leaf_root
                            && leaf_plan.trees[ak.tree_id as usize].tree_bucket == tb
                        {
                            bucket_trees
                                .get_mut(tk.as_bytes())
                                .ok_or_else(corrupt)?
                                .2
                                .push((ak.partition_key, centroid));
                        }
                    }
                }
            }
        }
        for (_, total_l, children) in bucket_trees.values() {
            if *total_l > 1 && children.len() as u64 != *total_l {
                return Err(corrupt());
            }
        }
        Ok(())
    }

    fn take_final_shard_state(
        &self,
        final_round: u8,
        shard: u32,
        plan: &ShardPlan,
        manifest: &IndexManifest,
    ) -> Result<ShardState> {
        let r_use = shard_final_round(shard, final_round, plan);
        match self {
            Self::Memory(mem) => lock_mutex(mem.shards.get(shard as usize).ok_or_else(corrupt)?)
                .take()
                .ok_or_else(corrupt),
            Self::Disk(disk) => {
                let (s_path, c_path, _) = disk.shard_paths(r_use, shard);
                read_shard_state_file(&s_path, &c_path, manifest)
            }
        }
    }

    async fn reclaim_final_shards<B: Backend>(
        &self,
        env: &StageEnv<B>,
        final_round: u8,
        coarse_shards: u32,
        plan: &ShardPlan,
        lock: &Arc<File>,
    ) -> Result<()> {
        let Self::Disk(disk) = self else {
            return Ok(());
        };
        let ep = disk.epoch;
        let mut obsolete = Vec::with_capacity((coarse_shards as usize) * 2);
        for s in 0..coarse_shards {
            let r_use = shard_final_round(s, final_round, plan);
            obsolete.push(format!("shard-e{ep}-r{r_use}-{s:06}.bin"));
            obsolete.push(format!("centers-e{ep}-r{r_use}-{s:06}.bin"));
        }
        reclaim_files(env, disk, lock, obsolete).await
    }
}

/// Initializes and drives a pure CPU/workspace `BuildUnit` stage while holding the shared workspace lock.
async fn drive_cpu_stage<B, F>(
    env: &StageEnv<B>,
    lock: &Arc<File>,
    stage: u8,
    unit_count: u32,
    concurrency: usize,
    work: F,
) -> Result<Vec<BuildUnit>>
where
    B: Backend,
    F: Fn(u32, &IndexManifest, &(dyn Fn() -> Result<()> + Sync)) -> Result<UnitOutput>
        + Send
        + Sync
        + 'static,
{
    if unit_count == 0 {
        return Ok(Vec::new());
    }
    if env.distributed_workers == 1 && !env.persist_single_worker {
        let manifest = env.manifest.clone();
        let guard = Arc::clone(lock);
        let epoch = env.fence.epoch();
        let workers = concurrency.max(1).min(unit_count as usize);
        return env
            .spawn_cpu(move |ctrl| {
                let _lock = guard;
                let check = || crate::runtime::check_control(&ctrl);
                let units: Vec<u32> = (0..unit_count).collect();
                let batches = parallel_chunks(&units, workers, |_, chunk| {
                    let mut out = Vec::with_capacity(chunk.len());
                    for &u in chunk {
                        check()?;
                        let uo = work(u, &manifest, &check)?;
                        let mut bu = BuildUnit::unclaimed(epoch);
                        bu.complete = true;
                        bu.records = uo.records;
                        bu.kv_count = uo.kv_count;
                        bu.aux_count = uo.aux_count;
                        bu.kv_lthash = uo.kv_lthash;
                        bu.aux_lthash = uo.aux_lthash;
                        bu.aux2_lthash = uo.aux2_lthash;
                        out.push(bu);
                    }
                    Ok(out)
                })?;
                Ok(batches.into_iter().flatten().collect())
            })
            .await;
    }

    init_stage_units(env, stage, unit_count).await?;
    let work = Arc::new(work);
    let env_outer = env.clone();
    let lock_outer = Arc::clone(lock);
    drive_stage(env, stage, unit_count, concurrency, move |unit, _lease| {
        let env_c = env_outer.clone();
        let guard = Arc::clone(&lock_outer);
        let work_c = Arc::clone(&work);
        async move {
            let manifest = env_c.manifest.clone();
            env_c
                .spawn_cpu(move |ctrl| {
                    let _lock = guard;
                    let check = || crate::runtime::check_control(&ctrl);
                    work_c(unit, &manifest, &check)
                })
                .await
        }
    })
    .await
}

fn prune_stale_epoch_files(dir: &Path, epoch: u64) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_error(e)),
    };
    let epoch_tag = format!("-e{epoch}-");
    let plan_name = format!("plan-e{epoch}.bin");
    let leafkeys_name = format!("leafkeys-e{epoch}.bin");
    for entry in entries {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name == ".ktann-build-lock"
            || name == plan_name
            || name == leafkeys_name
            || name.contains(&epoch_tag)
        {
            continue;
        }
        let _ = fs::remove_file(path);
    }
    Ok(())
}

/// Executes the unified sharded Bulk Build pipeline over either `StageStore::Memory` or `StageStore::Disk`.
#[expect(
    clippy::too_many_arguments,
    reason = "pipeline entry binds context, manifest, descriptor, workspace state, writer fence, storage mode, root lock, and retry policy"
)]
pub(crate) async fn run_pipeline<B: Backend>(
    context: &mut OperationContext<B>,
    index: &IndexManifest,
    descriptor: &BuildDescriptor,
    state: &BuildWorkspace,
    fence: WriteFence,
    in_memory: bool,
    lock: Arc<File>,
    retry: RetryPolicy,
) -> Result<BulkBuildReport> {
    let mut report = BulkBuildReport {
        in_memory_mode: in_memory,
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

    // 1. Open InputSnapshot and build or load deterministic `ShardPlan`.
    let plan_start = Instant::now();
    let (snapshot, plan) = {
        let desc = descriptor.clone();
        let manifest = index.clone();
        let dir = work_dir.clone();
        let single_worker = state.options.distributed_workers == 1;
        let guard = Arc::clone(&lock);
        let quota_c = Arc::clone(&quota);
        context
            .compute(move |control| {
                let _lock = guard;
                let check = || crate::runtime::check_control(control);
                if !in_memory {
                    if single_worker {
                        prune_stale_epoch_files(&dir, fence.epoch)?;
                    }
                    quota_c.seed_existing(&dir)?;
                }
                let snap = InputSnapshot::open(
                    desc.source(),
                    manifest.config().clone(),
                    desc.input().clone(),
                )?;
                let plan = if in_memory {
                    ShardPlan::build(&snap, &manifest, desc.options(), effective_mem, &check)?
                } else {
                    let plan_path = dir.join(format!("plan-e{}.bin", fence.epoch));
                    if plan_path.exists() {
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
                        let built = ShardPlan::build(
                            &snap,
                            &manifest,
                            desc.options(),
                            effective_mem,
                            &check,
                        )?;
                        built.save(&plan_path, &quota_c)?;
                        built
                    }
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

    let id_buckets = plan.id_bucket_count();
    let coarse_shards = plan.total_shards;
    let heavy_shards = plan.light_shard_start;
    let tree_buckets = id_buckets.clamp(1, 64);
    let refinement_rounds = descriptor.refinement_rounds();
    let sample_items = descriptor.options().sample_items;

    let store = Arc::new(if in_memory {
        StageStore::Memory(MemStore::new(id_buckets, coarse_shards, heavy_shards))
    } else {
        StageStore::Disk(DiskStore {
            dir: work_dir,
            epoch: fence.epoch,
            quota: Arc::clone(&quota),
        })
    });

    let writer_lanes = if state.options.distributed_workers > 1
        || context.backend().capabilities().transactional_clear_range
    {
        parallelism.clamp(1, 8)
    } else {
        1
    };

    let env = StageEnv {
        backend: context.backend(),
        control: context.options.clone(),
        bulk_authority: context.bulk_authority,
        sched_lease_lock: context.bulk_lease_lock.clone(),
        distributed_workers: state.options.distributed_workers,
        persist_single_worker: false,
        in_memory,
        writer_lanes,
        commit_sem: Arc::new(Semaphore::new(writer_lanes)),
        admission: context.cpu_keep_alive(),
        manifest: index.clone(),
        fence: StageFence::Write(fence),
        unit_lease: state.options.unit_lease,
        budget,
        retry,
    };

    let seg_mem_est = (16_384_usize)
        .saturating_mul(index.config().dimension() * 4 + 256)
        .max(1);
    let scatter_conc = (effective_mem / seg_mem_est).clamp(1, parallelism.min(8));
    let seg_count = snapshot.segments().len();
    let max_segs_by_mem = (effective_mem / scatter_conc.max(1) / seg_mem_est).clamp(1, 32);
    let segs_per_run = seg_count.div_ceil(2_048).max(1).min(max_segs_by_mem);
    let scatter_units = u32::try_from(seg_count.div_ceil(segs_per_run))
        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;

    let leaf_conc = if in_memory {
        parallelism
    } else {
        (effective_mem / (96 * 1024 * 1024)).clamp(1, parallelism.min(8))
    };
    let emit_conc = if in_memory || state.options.distributed_workers > 1 {
        parallelism.min(8)
    } else {
        (effective_mem / (64 * 1024 * 1024)).clamp(1, parallelism.min(8))
    };

    let leaf_plan = if let Some(loaded) = store
        .load_cached_leaf_plan(&env, &lock, coarse_shards, tree_buckets)
        .await?
    {
        loaded
    } else {
        // 2. Stage 1: Scatter input segments into Coarse Shards and Record-ID buckets.
        let ingest_start = Instant::now();
        let (scatter_records, scatter_id_lt) = store
            .run_scatter(
                &env,
                &lock,
                &snapshot,
                &plan,
                hard_limits,
                segs_per_run,
                scatter_units,
                scatter_conc,
                parallelism,
            )
            .await?;
        report.ingest = ingest_start.elapsed();

        // 3. Stage 2 (Global ID Dedup) & Stage 3 (Sharded Leaf Construction + Light-Shard Refinement).
        let construct_start = Instant::now();
        let dedup_done = drive_cpu_stage(&env, &lock, STAGE_DEDUP, id_buckets, parallelism, {
            let store_c = Arc::clone(&store);
            move |b, _manifest, check| {
                let ids = store_c.take_scatter_ids(b, scatter_units, check)?;
                let (count, hash, sorted) = verify_id_bucket(ids)?;
                store_c.save_dedup_ids(b, sorted)?;
                Ok(UnitOutput {
                    records: count,
                    aux_lthash: hash,
                    ..UnitOutput::default()
                })
            }
        })
        .await?;

        let dedup_records: u64 = dedup_done.iter().map(|u| u.records).sum();
        let mut dedup_id_lt = LtHash::ZERO;
        for u in &dedup_done {
            dedup_id_lt.add_assign(u.aux_lthash);
        }
        if scatter_records != dedup_records || scatter_id_lt != dedup_id_lt {
            return Err(corrupt());
        }
        store
            .reclaim_scatter_ids(&env, scatter_units, &lock)
            .await?;

        init_stage_units(&env, STAGE_LEAVES, coarse_shards).await?;
        if !is_stage_complete(&env, STAGE_LEAVES, coarse_shards).await?
            && let Some(sources) = store
                .prepare_leaf_inputs(&env, &lock, scatter_units, coarse_shards, &plan)
                .await?
        {
            let light_per_worker_mem = effective_mem / leaf_conc.max(1);
            drive_cpu_stage(&env, &lock, STAGE_LEAVES, coarse_shards, leaf_conc, {
                let store_c = Arc::clone(&store);
                let plan_c = plan.clone();
                move |s, manifest, check| {
                    let rows = store_c.take_scatter_shard(s, manifest, &sources, check)?;
                    let base_count = rows.rows.len() as u64;
                    let mut shard_state = if base_count == 0 {
                        ShardState {
                            rows,
                            leaves: Vec::new(),
                        }
                    } else {
                        build_shard_leaves(manifest, sample_items, rows, check)?
                    };
                    if let Some(rounds) = refinement_rounds
                        && s >= plan_c.light_shard_start
                    {
                        refine_single_light_shard(
                            manifest,
                            &mut shard_state,
                            rounds,
                            light_per_worker_mem,
                            check,
                        )?;
                    }
                    let leaf_count = shard_state.leaves.len() as u64;
                    store_c.save_shard_state(0, s, shard_state, &plan_c, manifest)?;
                    Ok(UnitOutput {
                        records: base_count,
                        aux_count: leaf_count,
                        ..UnitOutput::default()
                    })
                }
            })
            .await?;
            store
                .reclaim_scatter_shards(&env, scatter_units, &lock)
                .await?;
        }
        report.construct = construct_start.elapsed();

        // 4. Optional Bounded-Halo Refinement across heavy-tree shards (`0..heavy_shards`).
        let mut final_round = 0_u8;
        let rounds = refinement_rounds.unwrap_or(0);
        if refinement_rounds.is_some() && heavy_shards > 0 {
            let refine_start = Instant::now();
            final_round = run_halo_refinement(
                &env,
                &store,
                &lock,
                &plan,
                rounds,
                effective_mem,
                parallelism,
                leaf_conc,
            )
            .await?;
            report.refine = refine_start.elapsed();
        }

        store
            .finalize_leaf_plan(
                &env,
                &lock,
                scatter_records,
                final_round,
                &plan,
                tree_buckets,
                parallelism,
            )
            .await?
    };

    // 5. Emit serving KVs in ascending keyspace order:
    //    `STAGE_EMIT_RECORDS ([0x01])` -> `STAGE_EMIT_TREES ([0x03/0x04])` -> `STAGE_EMIT_LEAVES ([0x04])`.
    let emit_start = Instant::now();
    let leaf_partitions: u64 = leaf_plan.trees.iter().map(|t| t.total_leaves).sum();
    let leaf_plan = Arc::new(leaf_plan);

    // Stage 30: Emit `[0x01 Record + Location + Payload]` in ascending `id_bucket` and Record ID order.
    let max_records_per_subbatch = if in_memory {
        32_768
    } else {
        (effective_mem / 4096).clamp(64, 4096)
    };
    init_stage_units(&env, STAGE_EMIT_RECORDS, id_buckets).await?;
    let rec_emit_done = drive_stage(&env, STAGE_EMIT_RECORDS, id_buckets, emit_conc, {
        let env = env.clone();
        let store_c = Arc::clone(&store);
        let snapshot = snapshot.clone();
        let leaf_plan_c = Arc::clone(&leaf_plan);
        let plan_c = plan.clone();
        let lock = Arc::clone(&lock);
        move |b, u_lease| {
            let env_c = env.clone();
            let store_u = Arc::clone(&store_c);
            let snap_c = snapshot.clone();
            let lp_c = Arc::clone(&leaf_plan_c);
            let plan_u = plan_c.clone();
            let guard = Arc::clone(&lock);
            async move {
                let (mut id_refs, mut locs) = env_c
                    .spawn_cpu({
                        let guard_init = Arc::clone(&guard);
                        move |ctrl| {
                            let _lock = guard_init;
                            let check = || crate::runtime::check_control(&ctrl);
                            store_u.take_record_bucket_inputs(b, &lp_c, &plan_u, &check)
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
                    let guard_sub = Arc::clone(&guard);
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
                    env_c.commit_unit_kvs(u_lease.as_ref(), kvs).await?;
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
    store
        .reclaim_record_inputs(&env, id_buckets, &leaf_plan, &plan, &lock)
        .await?;

    // Stage 31: Build and emit Upper Trees + Partition-Local HNSW graphs (`[0x03]` + `[0x04]`).
    init_stage_units(&env, STAGE_EMIT_TREES, tree_buckets).await?;
    let tree_emit_done = drive_stage(&env, STAGE_EMIT_TREES, tree_buckets, parallelism.min(4), {
        let env = env.clone();
        let store_c = Arc::clone(&store);
        let leaf_plan_c = Arc::clone(&leaf_plan);
        let plan_c = plan.clone();
        let lock = Arc::clone(&lock);
        move |tb, u_lease| {
            let env_c = env.clone();
            let store_u = Arc::clone(&store_c);
            let lp_c = Arc::clone(&leaf_plan_c);
            let plan_u = plan_c.clone();
            let guard = Arc::clone(&lock);
            async move {
                let (trees_n, internal_parts, kvs, kv_lt, def_lt, ref_lt) = env_c
                    .spawn_cpu({
                        let manifest = env_c.manifest.clone();
                        move |ctrl| {
                            let _lock = guard;
                            let check = || crate::runtime::check_control(&ctrl);
                            let mut bucket_trees: BTreeMap<Bytes, UpperTreeBucket> =
                                BTreeMap::new();
                            for t in &lp_c.trees {
                                if t.tree_bucket == tb {
                                    bucket_trees.insert(
                                        Bytes::copy_from_slice(t.tree_key.as_bytes()),
                                        (t.tree_key.clone(), t.total_leaves, Vec::new()),
                                    );
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
                            store_u.load_tree_bucket_children(
                                tb,
                                &lp_c,
                                &plan_u,
                                &manifest,
                                &mut bucket_trees,
                                &check,
                            )?;
                            let mut all_kvs = Vec::new();
                            let mut kv_lt = LtHash::ZERO;
                            let mut def_lt = LtHash::ZERO;
                            let mut ref_lt = LtHash::ZERO;
                            let mut internal_parts = 0_u64;
                            let trees_n = bucket_trees.len() as u64;
                            let graph_threads = if in_memory {
                                parallelism
                            } else {
                                (parallelism / 2).max(1)
                            };
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
                env_c.commit_unit_kvs(u_lease.as_ref(), kvs).await?;
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

    // Stage 32: Emit `[0x04 Leaf Partitions]` and reclaim final shard state.
    init_stage_units(&env, STAGE_EMIT_LEAVES, coarse_shards).await?;
    let leaf_emit_done = drive_stage(&env, STAGE_EMIT_LEAVES, coarse_shards, emit_conc, {
        let env = env.clone();
        let store_c = Arc::clone(&store);
        let plan_c = plan.clone();
        let leaf_plan_c = Arc::clone(&leaf_plan);
        let lock = Arc::clone(&lock);
        move |s, u_lease| {
            let env_c = env.clone();
            let store_u = Arc::clone(&store_c);
            let plan_u = plan_c.clone();
            let lp_c = Arc::clone(&leaf_plan_c);
            let guard = Arc::clone(&lock);
            async move {
                let enc = env_c
                    .spawn_cpu({
                        let manifest = env_c.manifest.clone();
                        move |ctrl| {
                            let _lock = guard;
                            let check = || crate::runtime::check_control(&ctrl);
                            let shard_state = store_u.take_final_shard_state(
                                lp_c.final_round,
                                s,
                                &plan_u,
                                &manifest,
                            )?;
                            if shard_state.leaves.is_empty() {
                                return Ok(None);
                            }
                            Ok(Some(encode_leaf_shard(
                                &manifest,
                                &shard_state,
                                &lp_c.assigned[s as usize],
                                hard_limits,
                                &check,
                            )?))
                        }
                    })
                    .await?;
                let Some(enc) = enc else {
                    return Ok(UnitOutput::default());
                };
                let kv_count = enc.kvs.len() as u64;
                env_c.commit_unit_kvs(u_lease.as_ref(), enc.kvs).await?;
                Ok(UnitOutput {
                    records: enc.records,
                    kv_count,
                    aux_count: 0,
                    kv_lthash: enc.kv_lthash,
                    aux_lthash: enc.membership_lthash,
                    aux2_lthash: enc.edge_def_lthash,
                })
            }
        }
    })
    .await?;
    store
        .reclaim_final_shards(&env, leaf_plan.final_round, coarse_shards, &plan, &lock)
        .await?;
    report.emit = emit_start.elapsed();

    // 6. Verify global 512-bit homomorphic multiset equations across all stages and seal `Loaded`.
    let loc_records: u64 = rec_emit_done.iter().map(|u| u.records).sum();
    let leaf_records: u64 = leaf_emit_done.iter().map(|u| u.records).sum();
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

    if loc_records != leaf_plan.scatter_records
        || leaf_records != leaf_plan.scatter_records
        || leaf_mem_lt != loc_mem_lt
        || edge_def_lt != edge_ref_lt
    {
        return Err(corrupt());
    }

    let mut total_entries = 0_u64;
    let mut expected_kv_lthash = LtHash::ZERO;
    for list in [&rec_emit_done, &tree_emit_done, &leaf_emit_done] {
        for u in list {
            total_entries = total_entries.checked_add(u.kv_count).ok_or_else(corrupt)?;
            expected_kv_lthash.add_assign(u.kv_lthash);
        }
    }

    finalize_and_seal_loaded(
        context,
        index,
        fence,
        &plan,
        hard_limits,
        leaf_plan.scatter_records,
        total_trees,
        leaf_partitions + internal_parts,
        total_entries,
        expected_kv_lthash,
        retry,
        &mut report,
    )
    .await?;
    report.scratch_written_bytes = quota.written();
    report.peak_scratch_bytes = quota.peak();
    Ok(report)
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded-halo refinement coordinates proposal scoring, arbitration, outbound extraction, and shard compaction across rounds"
)]
async fn run_halo_refinement<B: Backend>(
    env: &StageEnv<B>,
    store: &Arc<StageStore>,
    lock: &Arc<File>,
    plan: &ShardPlan,
    rounds: u8,
    effective_mem: usize,
    parallelism: usize,
    leaf_conc: usize,
) -> Result<u8> {
    let heavy_shards = plan.light_shard_start;
    let per_leaf = refinement_per_leaf_bytes(env.manifest.config().dimension(), rounds);
    let halo_conc = if env.in_memory {
        parallelism
    } else {
        (effective_mem / (192 * 1024 * 1024)).clamp(1, parallelism)
    };
    let per_worker_budget = effective_mem / halo_conc.max(1);
    let admitted_leaves =
        (effective_mem / per_leaf.max(1)).max(per_worker_budget / per_leaf.max(1));
    if admitted_leaves < 2 {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let max_halo_leaves = (effective_mem / per_leaf.max(1)).min(MAX_HALO_LEAVES_CAP);

    if rounds == 0 && env.distributed_workers == 1 {
        let store_c = Arc::clone(store);
        let plan_c = plan.clone();
        let manifest_c = env.manifest.clone();
        let guard = Arc::clone(lock);
        env.spawn_cpu(move |_ctrl| {
            let _lock = guard;
            for heavy in &plan_c.heavy_trees {
                let mut total_leaves = 0_usize;
                for s in heavy.shard_start..heavy.shard_start + heavy.shard_count {
                    total_leaves += store_c.load_shard_centers(0, s, &manifest_c)?.len();
                }
                let needed = total_leaves
                    .min(max_halo_leaves)
                    .checked_mul(per_leaf)
                    .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
                if needed > effective_mem {
                    return Err(Error::new(ErrorKind::LimitExceeded));
                }
            }
            Ok(())
        })
        .await?;
        return Ok(0);
    }

    let mut final_round = 0_u8;
    for r in 0..rounds {
        let stage_prop = STAGE_REFINE_BASE + 3 * r;
        let stage_ext = STAGE_REFINE_BASE + 3 * r + 1;
        let stage_app = STAGE_REFINE_BASE + 3 * r + 2;

        let precomputed_halos: Arc<BTreeMap<u32, Arc<PrecomputedTreeHalo>>> =
            if env.distributed_workers == 1 {
                let store_c = Arc::clone(store);
                let plan_c = plan.clone();
                let manifest_c = env.manifest.clone();
                let guard = Arc::clone(lock);
                Arc::new(
                    env.spawn_cpu(move |ctrl| {
                        let _lock = guard;
                        let check = || crate::runtime::check_control(&ctrl);
                        precompute_fitting_tree_halos(
                            &store_c,
                            r,
                            &manifest_c,
                            &plan_c,
                            per_leaf,
                            max_halo_leaves,
                            effective_mem,
                            parallelism,
                            &check,
                        )
                    })
                    .await?,
                )
            } else {
                Arc::new(BTreeMap::new())
            };

        let prop_done = drive_cpu_stage(env, lock, stage_prop, heavy_shards, halo_conc, {
            let store_c = Arc::clone(store);
            let plan_c = plan.clone();
            let halos = Arc::clone(&precomputed_halos);
            move |s, manifest, check| {
                let proposals = propose_shard_moves(
                    &store_c,
                    r,
                    s,
                    manifest,
                    &plan_c,
                    &halos,
                    max_halo_leaves,
                    per_leaf,
                    per_worker_budget.max(per_leaf.saturating_mul(2)),
                    check,
                )?;
                let aux_count = store_c.save_proposals(r, s, proposals)?;
                Ok(UnitOutput {
                    aux_count,
                    ..UnitOutput::default()
                })
            }
        })
        .await?;
        drop(precomputed_halos);

        let prop_counts: Vec<u64> = prop_done.iter().map(|u| u.aux_count).collect();
        if prop_counts.iter().sum::<u64>() == 0 {
            break;
        }
        if is_stage_complete(env, stage_app, heavy_shards).await? {
            final_round = r + 1;
            continue;
        }

        let arb_res = env
            .spawn_cpu({
                let store_c = Arc::clone(store);
                let manifest = env.manifest.clone();
                let plan_c = plan.clone();
                let prop_counts_c = prop_counts.clone();
                let guard = Arc::clone(lock);
                move |ctrl| {
                    let _lock = guard;
                    let check = || crate::runtime::check_control(&ctrl);
                    arbitrate_round(
                        &store_c,
                        r,
                        &prop_counts_c,
                        &manifest,
                        &plan_c,
                        effective_mem,
                        &check,
                    )
                }
            })
            .await;

        let accepted_moves = match arb_res {
            Ok(v) => Arc::new(v),
            Err(_)
                if env.distributed_workers > 1
                    && is_stage_complete(env, stage_app, heavy_shards).await? =>
            {
                final_round = r + 1;
                continue;
            }
            Err(e) => return Err(e),
        };

        if accepted_moves.total_accepted == 0 {
            store.reclaim_proposals(env, r, &prop_counts, lock).await?;
            break;
        }

        store
            .extract_outbound(
                env,
                lock,
                stage_ext,
                r,
                heavy_shards,
                leaf_conc,
                &accepted_moves,
            )
            .await?;

        drive_cpu_stage(env, lock, stage_app, heavy_shards, leaf_conc, {
            let store_c = Arc::clone(store);
            let plan_c = plan.clone();
            let acc = Arc::clone(&accepted_moves);
            move |s, manifest, check| {
                apply_round_shard(&store_c, r, s, manifest, &plan_c, &acc, check)?;
                Ok(UnitOutput::default())
            }
        })
        .await?;

        store
            .reclaim_refine_round(env, r, heavy_shards, lock)
            .await?;
        final_round = r + 1;
    }
    Ok(final_round)
}

fn ingest_all_into_memory_slab(
    snapshot: &InputSnapshot,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    limits: HardLimits,
    parallelism: usize,
    mem: &MemStore,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<(u64, LtHash)> {
    let dim = manifest.config().dimension();
    let num_buckets = plan.id_bucket_count() as usize;
    let num_shards = plan.total_shards as usize;
    let total_items_usize = usize::try_from(snapshot.manifest().items())
        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let slab_floats = total_items_usize
        .checked_mul(dim)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;

    let mut raw_slab = vec![0.0_f32; slab_floats];
    let segs = snapshot.segments();
    let seg_count = segs.len();
    let mut all_ids: Vec<Vec<IdFrameEntry>> = vec![Vec::new(); num_buckets];
    let mut shard_metas: Vec<Vec<CompactMeta>> = vec![Vec::new(); num_shards];
    let mut total_records = 0_u64;
    let mut input_id_lthash = LtHash::ZERO;

    let wave_size = parallelism.clamp(1, 16);
    let mut slab_rest: &mut [f32] = raw_slab.as_mut_slice();
    let mut base_row = 0_usize;

    for wave_start in (0..seg_count).step_by(wave_size) {
        let wave_end = (wave_start + wave_size).min(seg_count);
        let mut seg_slices = Vec::with_capacity(wave_end - wave_start);
        for (seg_idx, seg_meta) in segs[wave_start..wave_end].iter().enumerate() {
            let items = usize::try_from(seg_meta.items).map_err(|_| corrupt())?;
            let floats = items.checked_mul(dim).ok_or_else(corrupt)?;
            if floats > slab_rest.len() {
                return Err(corrupt());
            }
            let (head, tail) = slab_rest.split_at_mut(floats);
            slab_rest = tail;
            seg_slices.push((wave_start + seg_idx, base_row, head));
            base_row += items;
        }

        let ingested = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(seg_slices.len());
            for (seg_idx, seg_base, slice) in seg_slices {
                handles.push(scope.spawn(move || {
                    ingest_segment_into_slab(
                        snapshot, seg_idx, seg_base, slice, manifest, plan, limits, check,
                    )
                }));
            }
            let mut out = Vec::with_capacity(handles.len());
            for h in handles {
                out.push(h.join().map_err(|_| Error::new(ErrorKind::Other))??);
            }
            Ok::<_, Error>(out)
        })?;

        for seg in ingested {
            total_records += seg.records;
            input_id_lthash.add_assign(seg.id_lthash);
            for (b, ids) in seg.ids_by_bucket.into_iter().enumerate() {
                all_ids[b].extend(ids);
            }
            for (s, metas) in seg.metas_by_shard.into_iter().enumerate() {
                shard_metas[s].extend(metas);
            }
        }
    }
    if !slab_rest.is_empty() {
        return Err(corrupt());
    }

    let slab = Arc::new(raw_slab);
    let rows_by_shard: Vec<ShardRows> = shard_metas
        .into_iter()
        .map(|rows| {
            let mut sr = ShardRows::with_slab(dim, Arc::clone(&slab));
            sr.rows = rows;
            sr
        })
        .collect();
    *lock_mutex(&mem.scatter_ids) = all_ids;
    *lock_mutex(&mem.scatter_shards) = rows_by_shard;
    Ok((total_records, input_id_lthash))
}

fn remap_memory_scatter_shards(
    mem: &MemStore,
    manifest: &IndexManifest,
    plan: &ShardPlan,
) -> Result<()> {
    let mut shards = lock_mutex(&mem.scatter_shards);
    let counts: Vec<u64> = shards.iter().map(|s| s.rows.len() as u64).collect();
    let kernel = VectorKernel::new(
        manifest.config().dimension(),
        manifest.config().metric(),
        *manifest.rotation_seed(),
    )?;
    let remap = compute_shard_remap(
        &kernel,
        plan,
        &counts,
        u64::from(manifest.config().min_leaf_partition_entries()),
        u64::from(manifest.config().max_leaf_partition_entries()),
    );
    for src in 0..shards.len() {
        let dst = remap[src] as usize;
        if dst != src && !shards[src].rows.is_empty() {
            let mut moved = std::mem::take(&mut shards[src].rows);
            shards[dst].rows.append(&mut moved);
        }
    }
    Ok(())
}

fn compute_non_empty_runs_by_shard(
    dir: &Path,
    ep: u64,
    scatter_units: u32,
    coarse_shards: u32,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    ctrl: &OperationOptions,
) -> Result<Vec<NonEmptyRunSources>> {
    let mut totals = vec![0_u64; coarse_shards as usize];
    let mut per_run_non_empty: Vec<Vec<usize>> = Vec::with_capacity(scatter_units as usize);
    for run in 0..scatter_units {
        if run.is_multiple_of(64) {
            crate::runtime::check_control(ctrl)?;
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
        plan,
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

async fn reclaim_files<B: Backend>(
    env: &StageEnv<B>,
    disk: &DiskStore,
    lock: &Arc<File>,
    names: Vec<String>,
) -> Result<()> {
    if names.is_empty() {
        return Ok(());
    }
    let dir = disk.dir.clone();
    let quota_c = Arc::clone(&disk.quota);
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

/// Checks whether a collaborative `BuildUnit` stage has already been 100% completed in KV
/// so late or racing workers skip reading intermediate files reclaimed by ahead peers.
async fn is_stage_complete<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit_count: u32,
) -> Result<bool> {
    if unit_count == 0 {
        return Ok(true);
    }
    if env.distributed_workers <= 1 {
        return Ok(false);
    }
    let id = env.manifest.logical_index_id();
    let epoch = env.fence.epoch();
    let range = LogicalRange::build_units(id, stage);
    retry_read(&env.control, env.fence.operation(), env.retry, || {
        let backend = Arc::clone(&env.backend);
        let range_ref = &range;
        async move {
            let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
            let mut cursor = None;
            let mut seen_complete = 0_u32;
            loop {
                let page = rtxn
                    .scan(
                        range_ref,
                        cursor.as_ref(),
                        ScanLimits {
                            item_limit: 1024,
                            byte_limit: 512 * 1024,
                        },
                    )
                    .await?;
                for item in page.items() {
                    let PersistentValue::BuildUnit(u) = item.value() else {
                        return Err(corrupt());
                    };
                    if u.epoch != epoch || !u.complete {
                        return Ok(false);
                    }
                    seen_complete += 1;
                }
                cursor = page.into_next_cursor();
                if cursor.is_none() {
                    break;
                }
            }
            Ok(seen_complete == unit_count)
        }
    })
    .await
}

pub(crate) async fn init_stage_units<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit_count: u32,
) -> Result<()> {
    if unit_count == 0 {
        return Ok(());
    }
    if env.distributed_workers == 1 && !env.persist_single_worker {
        if let StageFence::Write(wf) = env.fence
            && stage == STAGE_EMIT_RECORDS
            && !env.in_memory
        {
            advance_loading_progress(
                env.backend.as_ref(),
                &env.control,
                env.sched_lease_lock.as_deref(),
                env.bulk_authority,
                &env.manifest,
                wf,
                0,
                Some(env.retry),
            )
            .await?;
        }
        return Ok(());
    }
    let id = env.manifest.logical_index_id();
    let epoch = env.fence.epoch();
    let op = env.fence.operation();
    let max_per_txn = env.budget.max_mutations.saturating_sub(4).clamp(1, 512) as u32;
    for batch_start in (0..unit_count).step_by(max_per_txn as usize) {
        let batch_end = (batch_start + max_per_txn).min(unit_count);
        retry_raw_write(
            &env.control,
            env.sched_lease_lock.as_deref(),
            op,
            env.retry,
            || async {
                let prepared = async {
                    let mut txn = WriteLogicalTxn::bootstrap(
                        env.backend.begin_write().await?,
                        env.backend.hard_limits(),
                        env.backend.admission_budget(),
                    );
                    let mut warm = env.fence.warm_keys(id, env.bulk_authority, None);
                    if batch_start == 0
                        && matches!(env.fence, StageFence::Write(_))
                        && stage >= STAGE_EMIT_RECORDS
                    {
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

                    env.fence.check(&mut txn, &env.manifest).await?;
                    authorize_raw(&mut txn, id, env.bulk_authority).await?;

                    let mut any_written = false;
                    if batch_start == 0
                        && matches!(env.fence, StageFence::Write(_))
                        && stage >= STAGE_EMIT_RECORDS
                        && let Some(PersistentValue::BuildProgress(mut p)) =
                            txn.get_for_update(LogicalKey::BuildProgress(id)).await?
                        && p.epoch == epoch
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
                                if existing.epoch == epoch => {}
                            _ => {
                                txn.put(
                                    ukey,
                                    PersistentValue::BuildUnit(BuildUnit::unclaimed(epoch)),
                                )
                                .await?;
                                any_written = true;
                            }
                        }
                    }
                    if !any_written {
                        txn.rollback().await;
                        return Ok::<_, Error>((None, ()));
                    }
                    Ok((Some(txn), ()))
                }
                .await;
                commit_raw_prepared(prepared).await
            },
        )
        .await?;
    }
    Ok(())
}

pub(crate) async fn drive_stage<B, F, Fut>(
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

    if env.distributed_workers == 1 {
        let next_unit = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let mut tasks = tokio::task::JoinSet::<Result<Vec<(u32, BuildUnit)>>>::new();
        for _ in 0..worker_conc {
            let counter = Arc::clone(&next_unit);
            let env_c = env.clone();
            let work_c = Arc::clone(&work);
            tasks.spawn(async move {
                let task_token = non_zero_token()?;
                let mut local_done = Vec::new();
                loop {
                    crate::runtime::check_control(&env_c.control)?;
                    let u = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if u >= unit_count {
                        return Ok(local_done);
                    }
                    if env_c.persist_single_worker
                        && let Some(done) = read_completed_unit(&env_c, stage, u).await?
                    {
                        local_done.push((u, done));
                        continue;
                    }
                    let out = work_c(u, None).await?;
                    if let StageFence::Write(wf) = env_c.fence
                        && out.kv_count > 0
                        && env_c.unit_lease >= Duration::from_millis(50)
                    {
                        let _permit = env_c
                            .commit_sem
                            .acquire()
                            .await
                            .map_err(|_| Error::new(ErrorKind::Other))?;
                        advance_loading_progress(
                            env_c.backend.as_ref(),
                            &env_c.control,
                            env_c.sched_lease_lock.as_deref(),
                            env_c.bulk_authority,
                            &env_c.manifest,
                            wf,
                            out.kv_count,
                            None,
                        )
                        .await?;
                    }
                    let mut bu = BuildUnit::unclaimed(env_c.fence.epoch());
                    bu.owner = if env_c.persist_single_worker {
                        task_token
                    } else {
                        [0; 32]
                    };
                    bu.complete = true;
                    bu.records = out.records;
                    bu.kv_count = out.kv_count;
                    bu.aux_count = out.aux_count;
                    bu.kv_lthash = out.kv_lthash;
                    bu.aux_lthash = out.aux_lthash;
                    bu.aux2_lthash = out.aux2_lthash;
                    if env_c.persist_single_worker {
                        let unit_lease = UnitLease {
                            stage,
                            unit: u,
                            owner: task_token,
                            duration: env_c.unit_lease,
                            lock: Arc::new(tokio::sync::Mutex::new(())),
                        };
                        complete_unit(&env_c, &unit_lease, out, false).await?;
                    }
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
        let queue = Arc::new(Mutex::new(ordered.into_iter()));
        let mut tasks = tokio::task::JoinSet::<Result<()>>::new();
        for _ in 0..worker_conc.min(claimable_len) {
            let q = Arc::clone(&queue);
            let env_c = env.clone();
            let work_c = Arc::clone(&work);
            tasks.spawn(async move {
                let task_token = non_zero_token()?;
                loop {
                    crate::runtime::check_control(&env_c.control)?;
                    let Some(unit) = lock_mutex(&q).next() else {
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
                        complete_unit(&env_c, &unit_lease, out, true).await
                    }
                    .await;
                    match outcome {
                        Ok(()) => {}
                        Err(e) if e.kind() == ErrorKind::BulkBuildSuperseded => {
                            check_epoch_still_active(&env_c).await?;
                        }
                        Err(e) => {
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

async fn read_completed_unit<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit: u32,
) -> Result<Option<BuildUnit>> {
    let id = env.manifest.logical_index_id();
    let epoch = env.fence.epoch();
    retry_read(&env.control, env.fence.operation(), env.retry, || {
        let backend = Arc::clone(&env.backend);
        async move {
            let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await?);
            match rtxn
                .get(LogicalKey::BuildUnit {
                    index: id,
                    stage,
                    unit,
                })
                .await?
            {
                Some(PersistentValue::BuildUnit(u)) if u.complete && u.epoch == epoch => {
                    Ok(Some(u))
                }
                _ => Ok(None),
            }
        }
    })
    .await
}

async fn is_unit_completed_by_peer<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit: u32,
) -> Result<bool> {
    check_epoch_still_active(env).await?;
    Ok(read_completed_unit(env, stage, unit).await?.is_some())
}

async fn check_epoch_still_active<B: Backend>(env: &StageEnv<B>) -> Result<()> {
    retry_raw_write(
        &env.control,
        env.sched_lease_lock.as_deref(),
        env.fence.operation(),
        env.retry,
        || async {
            let res: Result<()> = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    env.backend.begin_write().await?,
                    env.backend.hard_limits(),
                    env.backend.admission_budget(),
                );
                env.fence.check(&mut txn, &env.manifest).await?;
                authorize_raw(
                    &mut txn,
                    env.manifest.logical_index_id(),
                    env.bulk_authority,
                )
                .await?;
                txn.rollback().await;
                Ok(())
            }
            .await;
            match res {
                Ok(()) => CommitStep::Done(()),
                Err(e) if e.kind() == ErrorKind::RetryableAbort => CommitStep::PrepareAbort,
                Err(e) => CommitStep::Fatal(e),
            }
        },
    )
    .await
}

async fn scan_stage_status<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit_count: u32,
    known_complete: &mut BTreeSet<u32>,
) -> Result<(Vec<u32>, Option<Vec<BuildUnit>>)> {
    let id = env.manifest.logical_index_id();
    let epoch = env.fence.epoch();
    let range = LogicalRange::build_units(id, stage);
    retry_read(&env.control, env.fence.operation(), env.retry, || {
        let range_ref = &range;
        let mut seen = known_complete.clone();
        async move {
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
                        range_ref,
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
                    if u.epoch != epoch {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    if u.complete {
                        seen.insert(unit);
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
            let done = if seen.len() == unit_count as usize && units.iter().all(|u| u.complete) {
                Some(units)
            } else {
                None
            };
            Ok((seen, claimable, done))
        }
    })
    .await
    .map(|(seen, claimable, done)| {
        *known_complete = seen;
        (claimable, done)
    })
}

async fn try_claim_unit<B: Backend>(
    env: &StageEnv<B>,
    stage: u8,
    unit: u32,
    owner: [u8; 32],
) -> Result<Option<u64>> {
    let id = env.manifest.logical_index_id();
    let epoch = env.fence.epoch();
    let op = env.fence.operation();
    let ukey = LogicalKey::BuildUnit {
        index: id,
        stage,
        unit,
    };
    let mut attempts = 0;
    loop {
        crate::runtime::check_control(&env.control)?;
        let _sched_guard = lock_sched_read(&env.sched_lease_lock).await;
        let prepared = async {
            let mut txn = WriteLogicalTxn::bootstrap(
                env.backend.begin_write().await?,
                env.backend.hard_limits(),
                env.backend.admission_budget(),
            );
            txn.warm_for_update(
                env.fence
                    .warm_keys(id, env.bulk_authority, Some(ukey.clone())),
            )
            .await?;

            env.fence.check(&mut txn, &env.manifest).await?;
            authorize_raw(&mut txn, id, env.bulk_authority).await?;
            let now = now_ms()?;
            let mut u = match txn.get_for_update(ukey.clone()).await? {
                Some(PersistentValue::BuildUnit(u)) if u.epoch == epoch => u,
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
                env.retry.after_commit_error(op, &mut attempts, e).await?;
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
    let _sched_guard = lock_sched_read(&env.sched_lease_lock).await;
    let id = env.manifest.logical_index_id();
    let epoch = env.fence.epoch();
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
        txn.warm_for_update(
            env.fence
                .warm_keys(id, env.bulk_authority, Some(ukey.clone())),
        )
        .await?;
        env.fence.check(&mut txn, &env.manifest).await?;
        authorize_raw(&mut txn, id, env.bulk_authority).await?;
        match txn.get_for_update(ukey.clone()).await? {
            Some(PersistentValue::BuildUnit(mut u))
                if u.epoch == epoch && u.owner == lease.owner && !u.complete =>
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
    require_owner: bool,
) -> Result<()> {
    let id = env.manifest.logical_index_id();
    let epoch = env.fence.epoch();
    let ukey = LogicalKey::BuildUnit {
        index: id,
        stage: lease.stage,
        unit: lease.unit,
    };
    retry_raw_write(
        &env.control,
        env.sched_lease_lock.as_deref(),
        env.fence.operation(),
        env.retry,
        || async {
            let _lease_guard = lease.lock.lock().await;
            let prepared = async {
                let mut txn = WriteLogicalTxn::bootstrap(
                    env.backend.begin_write().await?,
                    env.backend.hard_limits(),
                    env.backend.admission_budget(),
                );
                let mut warm = env
                    .fence
                    .warm_keys(id, env.bulk_authority, Some(ukey.clone()));
                if matches!(env.fence, StageFence::Write(_)) && out.kv_count > 0 {
                    warm.push(LogicalKey::BuildProgress(id));
                }
                txn.warm_for_update(warm).await?;

                env.fence.check(&mut txn, &env.manifest).await?;
                authorize_raw(&mut txn, id, env.bulk_authority).await?;
                match txn.get_for_update(ukey.clone()).await? {
                    Some(PersistentValue::BuildUnit(u)) if u.epoch == epoch && u.complete => {
                        txn.rollback().await;
                        return Ok((None, ()));
                    }
                    Some(PersistentValue::BuildUnit(mut u))
                        if u.epoch == epoch && (!require_owner || u.owner == lease.owner) =>
                    {
                        u.owner = lease.owner;
                        u.complete = true;
                        u.records = out.records;
                        u.kv_count = out.kv_count;
                        u.aux_count = out.aux_count;
                        u.kv_lthash = out.kv_lthash;
                        u.aux_lthash = out.aux_lthash;
                        u.aux2_lthash = out.aux2_lthash;
                        txn.put(ukey.clone(), PersistentValue::BuildUnit(u)).await?;
                    }
                    None if !require_owner => {
                        let mut u = BuildUnit::unclaimed(epoch);
                        u.owner = lease.owner;
                        u.complete = true;
                        u.records = out.records;
                        u.kv_count = out.kv_count;
                        u.aux_count = out.aux_count;
                        u.kv_lthash = out.kv_lthash;
                        u.aux_lthash = out.aux_lthash;
                        u.aux2_lthash = out.aux2_lthash;
                        txn.put(ukey.clone(), PersistentValue::BuildUnit(u)).await?;
                    }
                    Some(PersistentValue::BuildUnit(_)) | None => {
                        return Err(Error::new(ErrorKind::BulkBuildSuperseded));
                    }
                    _ => return Err(corrupt()),
                }

                if matches!(env.fence, StageFence::Write(_))
                    && out.kv_count > 0
                    && let Some(PersistentValue::BuildProgress(mut p)) =
                        txn.get_for_update(LogicalKey::BuildProgress(id)).await?
                    && p.epoch == epoch
                    && matches!(p.phase, BuildPhase::Preparing | BuildPhase::Loading)
                {
                    p.phase = BuildPhase::Loading;
                    p.loaded_entries = p.loaded_entries.saturating_add(out.kv_count);
                    if p.loaded_entries > p.total_entries {
                        p.total_entries = p.loaded_entries;
                    }
                    txn.put(
                        LogicalKey::BuildProgress(id),
                        PersistentValue::BuildProgress(p),
                    )
                    .await?;
                }
                Ok((Some(txn), ()))
            }
            .await;
            commit_raw_prepared(prepared).await
        },
    )
    .await
}

// ============================================================================
// Unified Halo Refinement Helpers (`StageStore::Memory` & `StageStore::Disk`)
// ============================================================================

#[expect(
    clippy::too_many_arguments,
    reason = "whole-tree halo precomputation checks per-tree leaf capacity and memory admission"
)]
fn precompute_fitting_tree_halos(
    store: &StageStore,
    round: u8,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    per_leaf: usize,
    max_halo_leaves: usize,
    memory_budget_bytes: usize,
    parallelism: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<BTreeMap<u32, Arc<PrecomputedTreeHalo>>> {
    let kernel = VectorKernel::new(
        manifest.config().dimension(),
        manifest.config().metric(),
        *manifest.rotation_seed(),
    )?;
    let mut out = BTreeMap::new();
    for heavy in &plan.heavy_trees {
        check()?;
        let mut shard_centers = Vec::with_capacity(heavy.shard_count as usize);
        let mut total_leaves = 0_usize;
        for s in heavy.shard_start..heavy.shard_start + heavy.shard_count {
            let centers = store.load_shard_centers(round, s, manifest)?;
            total_leaves += centers.len();
            shard_centers.push(centers.into_iter().map(|(_, _, c)| c).collect::<Vec<_>>());
        }
        if total_leaves < 2 || total_leaves > max_halo_leaves {
            continue;
        }
        let scratch = candidate_scratch_bytes(total_leaves, 1).unwrap_or(usize::MAX);
        if total_leaves
            .checked_mul(per_leaf)
            .and_then(|b| b.checked_add(scratch))
            .is_some_and(|needed| needed <= memory_budget_bytes)
        {
            let halo = build_tree_halo(
                &kernel,
                heavy.shard_start,
                shard_centers,
                parallelism,
                memory_budget_bytes,
                check,
            )?;
            out.insert(heavy.shard_start, Arc::new(halo));
        }
    }
    Ok(out)
}

#[expect(
    clippy::too_many_arguments,
    reason = "proposal scoring binds store, round, shard id, manifest, plan, optional precomputed tree halo, and memory bounds"
)]
fn propose_shard_moves(
    store: &StageStore,
    round: u8,
    src_shard: u32,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    halos: &BTreeMap<u32, Arc<PrecomputedTreeHalo>>,
    max_halo_leaves: usize,
    per_leaf: usize,
    per_worker_budget: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<Vec<MoveProposal>> {
    check()?;
    let kernel = VectorKernel::new(
        manifest.config().dimension(),
        manifest.config().metric(),
        *manifest.rotation_seed(),
    )?;
    let precomputed = plan
        .heavy_tree_for_shard(src_shard)
        .and_then(|h| halos.get(&h.shard_start));

    store.with_shard_state(round, src_shard, manifest, |src_state| {
        if src_state.leaves.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(halo) = precomputed {
            return score_shard_with_tree_halo(&kernel, halo, src_shard, src_state, check);
        }
        score_halo_shard(
            &kernel,
            plan,
            src_shard,
            src_state,
            max_halo_leaves,
            per_leaf,
            per_worker_budget,
            check,
            |nbr| {
                if nbr == src_shard {
                    return Ok(src_state
                        .leaves
                        .iter()
                        .map(|l| l.centroid.clone())
                        .collect());
                }
                Ok(store
                    .load_shard_centers(round, nbr, manifest)?
                    .into_iter()
                    .map(|(_, _, c)| c)
                    .collect())
            },
        )
    })
}

struct AcceptedRoundMoves {
    total_accepted: usize,
    outbound: Vec<BTreeMap<usize, (u32, u32)>>,
    cross_outbound_by_src: Vec<BTreeMap<u32, Vec<(usize, u32)>>>,
    donors_by_dst: Vec<Vec<u32>>,
}

fn arbitrate_round(
    store: &StageStore,
    round: u8,
    prop_counts: &[u64],
    manifest: &IndexManifest,
    plan: &ShardPlan,
    memory_budget_bytes: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<AcceptedRoundMoves> {
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
        if prop_counts[start..start + count].iter().all(|&c| c == 0) {
            continue;
        }
        let mut shard_leaf_counts = Vec::with_capacity(count);
        let mut proposals = Vec::new();
        let mut used_mem = 0_usize;
        for local_s in 0..count {
            if local_s.is_multiple_of(32) {
                check()?;
            }
            let s_usize = start + local_s;
            let s = s_usize as u32;
            let centers = store.load_shard_centers(round, s, manifest)?;
            let l_counts: Vec<usize> = centers.into_iter().map(|(_, mc, _)| mc).collect();
            used_mem = used_mem.saturating_add(l_counts.len().saturating_mul(16));
            shard_leaf_counts.push(l_counts);

            let expected_props = *prop_counts.get(s_usize).ok_or_else(corrupt)?;
            let shard_props = store.take_proposals(round, s, expected_props)?;
            used_mem = used_mem.saturating_add(shard_props.len().saturating_mul(96));
            proposals.extend(shard_props);
            if matches!(store, StageStore::Disk(_)) && used_mem > memory_budget_bytes.max(64 * 1024)
            {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
        }

        let accepted = arbitrate_tree_proposals(
            heavy.shard_start,
            &shard_leaf_counts,
            proposals,
            min_leaf,
            max_leaf,
        )?;
        total_accepted += accepted.len();
        for p in accepted {
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
    Ok(AcceptedRoundMoves {
        total_accepted,
        outbound,
        cross_outbound_by_src,
        donors_by_dst,
    })
}

fn extract_memory_outbound(mem: &MemStore, acc: &AcceptedRoundMoves) -> Result<()> {
    for (src_s, cross_out) in acc.cross_outbound_by_src.iter().enumerate() {
        if cross_out.is_empty() {
            continue;
        }
        let src_guard = lock_mutex(mem.shards.get(src_s).ok_or_else(corrupt)?);
        let src_state = src_guard.as_ref().ok_or_else(corrupt)?;
        for (&dst_s, list) in cross_out {
            let mut dst_inb = lock_mutex(mem.outbound.get(dst_s as usize).ok_or_else(corrupt)?);
            for &(src_row_idx, dst_local_leaf) in list {
                let meta = src_state
                    .rows
                    .rows
                    .get(src_row_idx)
                    .ok_or_else(corrupt)?
                    .clone();
                let vec = if src_state.rows.slab.is_some() {
                    Vec::new()
                } else {
                    src_state.rows.vector(meta.vec_slot).to_vec()
                };
                dst_inb.push((dst_local_leaf, meta, vec));
            }
        }
    }
    Ok(())
}

fn extract_disk_outbound(
    disk: &DiskStore,
    round: u8,
    src_shard: u32,
    heavy_shards: usize,
    manifest: &IndexManifest,
    acc: &AcceptedRoundMoves,
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
    let (s_path, c_path, _) = disk.shard_paths(round, src_shard);
    let state = read_shard_state_file(&s_path, &c_path, manifest)?;
    check()?;
    let mut by_dst = vec![Vec::new(); heavy_shards];
    for (&dst_s, list) in cross_out {
        *by_dst.get_mut(dst_s as usize).ok_or_else(corrupt)? = list.clone();
    }
    write_indexed_outbound_rows(
        &disk
            .dir
            .join(format!("outb-e{}-r{round}-{src_shard:06}.bin", disk.epoch)),
        manifest,
        &state,
        &by_dst,
        &disk.quota,
    )?;
    Ok(())
}

fn apply_round_shard(
    store: &StageStore,
    round: u8,
    shard_id: u32,
    manifest: &IndexManifest,
    plan: &ShardPlan,
    acc: &AcceptedRoundMoves,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    check()?;
    let moved_out = acc.outbound.get(shard_id as usize).ok_or_else(corrupt)?;
    let donors = acc
        .donors_by_dst
        .get(shard_id as usize)
        .ok_or_else(corrupt)?;

    match store {
        StageStore::Memory(mem) => {
            if moved_out.is_empty() && donors.is_empty() {
                return Ok(());
            }
            let moved_in = std::mem::take(&mut *lock_mutex(
                mem.outbound.get(shard_id as usize).ok_or_else(corrupt)?,
            ));
            let kernel = VectorKernel::new(
                manifest.config().dimension(),
                manifest.config().metric(),
                *manifest.rotation_seed(),
            )?;
            let mut guard = lock_mutex(mem.shards.get(shard_id as usize).ok_or_else(corrupt)?);
            let shard = guard.as_mut().ok_or_else(corrupt)?;
            apply_moves_to_shard(&kernel, shard_id, shard, moved_out, moved_in)
        }
        StageStore::Disk(disk) => {
            let (s_path, c_path, _) = disk.shard_paths(round, shard_id);
            let mut shard = read_shard_state_file(&s_path, &c_path, manifest)?;
            if !moved_out.is_empty() || !donors.is_empty() {
                let kernel = VectorKernel::new(
                    manifest.config().dimension(),
                    manifest.config().metric(),
                    *manifest.rotation_seed(),
                )?;
                let mut moved_in = Vec::new();
                for &donor_s in donors {
                    check()?;
                    read_indexed_outbound_slice(
                        &disk
                            .dir
                            .join(format!("outb-e{}-r{round}-{donor_s:06}.bin", disk.epoch)),
                        shard_id as usize,
                        manifest,
                        &mut moved_in,
                    )?;
                }
                apply_moves_to_shard(&kernel, shard_id, &mut shard, moved_out, moved_in)?;
            }
            let (next_s, next_c, next_loc) = disk.shard_paths(round + 1, shard_id);
            write_shard_state_files(
                &next_s,
                &next_c,
                &next_loc,
                manifest,
                plan,
                &shard,
                &disk.quota,
            )
        }
    }
}
