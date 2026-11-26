//! Identity-bound handles for durable Bulk Build reservations.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::observe::labels::Operation;
use crate::runtime::{RuntimeHandle, lifecycle};
use crate::storage::backend::Backend;
use crate::storage::values::{BuildDescriptor, IndexManifest};

use super::{IndexName, LogicalIndexId, OperationOptions, Result};

/// Immutable Bulk Build preparation semantics, persisted with the source identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BulkBuildOptions {
    /// Partition construction controls.
    pub construction: crate::construction::ConstructionOptions,
    /// None preserves constructed centroids; Some(0) refreshes them without moves.
    pub refinement_rounds: Option<u8>,
}

impl From<crate::construction::ConstructionOptions> for BulkBuildOptions {
    fn from(construction: crate::construction::ConstructionOptions) -> Self {
        Self {
            construction,
            refinement_rounds: None,
        }
    }
}

/// Persisted lifecycle of the Logical Index owned by a Bulk Build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BulkBuildStatus {
    /// The name and immutable input identity are reserved; the index is hidden.
    Preparing,
    /// Parallel slice verification of sealed backend data is in progress.
    Validating {
        /// Serving entries already verified against expected inventory digests.
        verified_entries: u64,
        /// Total expected serving entries.
        total_entries: u64,
    },
    /// A persistent input/integrity/resource failure requires abort and rebuild.
    Failed {
        /// Redacted, bounded error category.
        kind: super::ErrorKind,
    },
    /// Serving bytes are being emitted into the hidden index.
    Loading {
        /// Serving entries committed so far.
        loaded_entries: u64,
        /// Total expected serving entries.
        total_entries: u64,
    },
    /// All serving bytes are loaded; backend validation and publication remain.
    Loaded {
        /// Number of committed serving entries.
        entries: u64,
    },
    /// The completed index is serving.
    Published,
    /// Index-owned data is being removed in bounded transactions.
    Dropping,
    /// The original Logical Index has been removed.
    Aborted,
}

/// Per-transaction limits for Bulk Build backend writes and validation scans.
/// Adapter admission may impose smaller limits. Physical namespace overhead is
/// included in the byte budget. At least two mutations must fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BulkLoadOptions {
    /// Maximum data plus fence/checkpoint mutations in one transaction.
    pub max_mutations: usize,
    /// Maximum physical key plus value bytes in one transaction.
    pub max_bytes: usize,
}

impl Default for BulkLoadOptions {
    fn default() -> Self {
        Self {
            max_mutations: 5_000,
            max_bytes: 4 * 1024 * 1024,
        }
    }
}

/// One bounded namespace cleanup pass; pending jobs retain durable ownership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BulkCleanupPage {
    /// Workspace records reclaimed in this page.
    pub reclaimed: usize,
    /// Still-building jobs or workspaces held by active IO.
    pub pending: usize,
    /// Pass this identity to continue scanning; None ends this pass.
    pub next: Option<LogicalIndexId>,
}

/// Stage timings and resource metrics performed while completing one build.
/// Reused stages have zero duration; a retry returns only work done by that call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BulkBuildReport {
    /// Coarse-shard and Record-ID-bucket planning duration.
    pub plan: Duration,
    /// Parallel segment ingestion and coarse-shard scatter duration.
    pub ingest: Duration,
    /// Shard deduplication and bottom-up leaf construction duration.
    pub construct: Duration,
    /// Bounded halo refinement duration.
    pub refine: Duration,
    /// RaBitQ7 quantization and serving Record, Location, Tree Manifest, and Partition emission duration.
    pub emit: Duration,
    /// Parallel backend validation, atomic publication, and workspace cleanup duration.
    pub publish: Duration,
    /// Number of Vector Records built.
    pub records: u64,
    /// Number of Tree Keys built.
    pub trees: u64,
    /// Number of serving partitions built across all trees.
    pub partitions: u64,
    /// Total serving key-value entries committed to the backend.
    pub entries: u64,
    /// Total workspace scratch bytes written across the build.
    pub scratch_written_bytes: u64,
    /// Peak simultaneous workspace scratch bytes held on disk.
    pub peak_scratch_bytes: u64,
    /// Whether execution completed in the zero-disk in-memory fast path.
    pub in_memory_mode: bool,
    /// Persisted construction seed for reproducing the completed index.
    pub rotation_seed: [u8; 32],
}

/// Resource bounds and an existing durable scratch workspace for Bulk Build workers.
/// Single-machine builds use a local directory (writing zero scratch files when
/// resident in memory); multi-worker builds (`distributed_workers > 1`) share
/// coarse shards over a shared directory or object-store mount while coordinating
/// unit leases and validation through the KV backend.
#[derive(Clone, Eq, PartialEq)]
pub struct BulkWorkerOptions {
    /// Existing absolute workspace root, separate from the source snapshot.
    pub workspace: std::path::PathBuf,
    /// Per-worker CPU and backend-write concurrency (1..=256).
    pub parallelism: usize,
    /// Maximum concurrent worker processes collaborating on one build (1..=256).
    pub distributed_workers: usize,
    /// Renewable lease duration for individual collaborative BuildUnits (>= 10ms).
    pub unit_lease: Duration,
    /// Per-worker memory budget for sharded construction and halo refinement.
    pub memory_bytes: usize,
    /// Working memory ceiling for one internal partition's local graph build.
    pub graph_memory_bytes: usize,
    /// Workspace scratch-file ceiling in bytes.
    pub scratch_bytes: u64,
    /// Bounded backend write and validation transaction limits.
    pub load: BulkLoadOptions,
}

impl BulkWorkerOptions {
    /// Uses bounded defaults with the supplied workspace root.
    #[must_use]
    pub fn new(workspace: impl Into<std::path::PathBuf>) -> Self {
        let parallelism = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .clamp(1, 32);
        Self {
            workspace: workspace.into(),
            parallelism,
            distributed_workers: 1,
            unit_lease: Duration::from_secs(30),
            memory_bytes: 512 * 1024 * 1024,
            graph_memory_bytes: 256 * 1024 * 1024,
            scratch_bytes: 64 * 1024 * 1024 * 1024,
            load: BulkLoadOptions::default(),
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.workspace.is_absolute()
            || self
                .workspace
                .to_str()
                .is_none_or(|p| p.len() > 3900 || p.contains('\0'))
            || self.parallelism == 0
            || self.parallelism > 256
            || self.distributed_workers == 0
            || self.distributed_workers > 256
            || self.unit_lease < Duration::from_millis(10)
            || self.unit_lease > Duration::from_secs(86_400 * 365)
            || self.memory_bytes == 0
            || self.graph_memory_bytes == 0
            || self.scratch_bytes == 0
            || self.load.max_mutations < 2
            || self.load.max_bytes == 0
        {
            return Err(super::Error::invalid_argument());
        }
        Ok(())
    }
}

impl fmt::Debug for BulkWorkerOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BulkWorkerOptions([REDACTED])")
    }
}

/// Per-process automatic Bulk Build scheduling bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BulkSchedulerOptions {
    /// Maximum jobs admitted concurrently by this scheduler (1..=64).
    pub max_jobs: usize,
    /// Queue polling and retry delay, at least one millisecond.
    pub poll_interval: Duration,
    /// Renewable lease duration; at least three poll intervals.
    pub lease_duration: Duration,
}

impl Default for BulkSchedulerOptions {
    fn default() -> Self {
        Self {
            max_jobs: 1,
            poll_interval: Duration::from_secs(1),
            lease_duration: Duration::from_secs(30),
        }
    }
}

impl BulkSchedulerOptions {
    pub(crate) fn validate(self) -> Result<()> {
        if self.max_jobs == 0
            || self.max_jobs > 64
            || self.poll_interval.as_millis() == 0
            || self
                .poll_interval
                .checked_mul(3)
                .is_none_or(|minimum| self.lease_duration < minimum)
            || self.lease_duration > Duration::from_secs(86_400 * 365)
        {
            return Err(super::Error::invalid_argument());
        }
        Ok(())
    }
}

/// A recoverable reservation for one new Logical Index.
///
/// The handle retains the owning Runtime until explicit shutdown or the last
/// public Runtime, Index, or BulkBuildJob handle is dropped.
/// This handle never follows a reused Index Name. Reservation launches no work;
/// callers explicitly run workers and publish after loading, or schedule the job.
/// Source snapshots remain caller owned, including after abort.
pub struct BulkBuildJob<B: Backend> {
    runtime: Arc<RuntimeHandle<B>>,
    name: IndexName,
    manifest: IndexManifest,
    descriptor: BuildDescriptor,
}

impl<B: Backend> BulkBuildJob<B> {
    pub(crate) fn new(
        runtime: Arc<RuntimeHandle<B>>,
        name: IndexName,
        manifest: IndexManifest,
        descriptor: BuildDescriptor,
    ) -> Self {
        Self {
            runtime,
            name,
            manifest,
            descriptor,
        }
    }

    /// The reserved name; it is not the authority for subsequent operations.
    #[must_use]
    pub fn name(&self) -> &IndexName {
        &self.name
    }

    /// The never-reused identity bound to this handle.
    #[must_use]
    pub fn logical_index_id(&self) -> LogicalIndexId {
        self.manifest.logical_index_id()
    }

    /// Immutable index identity, configuration, and persisted construction seed.
    /// The lifecycle here is the handle's opening snapshot; use `status` for
    /// current lifecycle. Reading it confers no backend write authority.
    #[must_use]
    pub fn index_manifest(&self) -> &IndexManifest {
        &self.manifest
    }

    /// Immutable input locator, checksum, and construction parameters.
    #[must_use]
    pub fn descriptor(&self) -> &BuildDescriptor {
        &self.descriptor
    }

    /// Reads the original Logical Index's current lifecycle.
    pub async fn status(&self) -> Result<BulkBuildStatus> {
        self.status_with_control(OperationOptions::default()).await
    }

    /// Reads status with explicit cancellation and deadline control.
    pub async fn status_with_control(&self, options: OperationOptions) -> Result<BulkBuildStatus> {
        let manifest = self.manifest.clone();
        let descriptor = self.descriptor.clone();
        self.runtime
            .inner
            .run_foreground(
                Operation::BulkBuildStatus,
                Some(self.logical_index_id()),
                options,
                move |mut context| async move {
                    lifecycle::build_status(&mut context, manifest, descriptor).await
                },
            )
            .await
    }

    /// Durably queues this job for automatic construction, loading, validation, and publication.
    /// Repeating identical options is idempotent. Run a Runtime scheduler on each
    /// participating process; all must have access to the shared source/workspace.
    pub async fn schedule(&self, options: BulkWorkerOptions) -> Result<()> {
        self.schedule_with_control(options, OperationOptions::default())
            .await
    }

    /// Queues work with explicit cancellation and deadline control.
    pub async fn schedule_with_control(
        &self,
        options: BulkWorkerOptions,
        control: OperationOptions,
    ) -> Result<()> {
        let manifest = self.manifest.clone();
        let name = self.name.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.inner.config());
        self.runtime
            .inner
            .run_foreground(
                Operation::ScheduleBulkBuild,
                Some(self.logical_index_id()),
                control,
                move |mut context| async move {
                    crate::runtime::bulk_scheduler::enqueue(
                        &mut context,
                        manifest,
                        name,
                        options,
                        retry,
                    )
                    .await
                },
            )
            .await
    }

    /// Completes sharded construction, loading, validation, and atomic publication.
    /// Retries resume the durable phase, including after an unknown commit or
    /// publication by another caller. Owned scratch files are reclaimed when no IO holds
    /// the workspace lock; namespace cleanup can finish a busy reclamation later.
    pub async fn complete(&self, options: BulkWorkerOptions) -> Result<super::Index<B>> {
        self.complete_with_control(options, OperationOptions::default())
            .await
    }

    /// Completes the build under one cancellation, deadline, and admission scope.
    pub async fn complete_with_control(
        &self,
        options: BulkWorkerOptions,
        control: OperationOptions,
    ) -> Result<super::Index<B>> {
        self.complete_with_report(options, control)
            .await
            .map(|(index, _)| index)
    }

    /// Completes sharded construction, loading, validation, publication, and workspace cleanup.
    pub async fn complete_with_report(
        &self,
        options: BulkWorkerOptions,
        control: OperationOptions,
    ) -> Result<(super::Index<B>, BulkBuildReport)> {
        let manifest = self.manifest.clone();
        let descriptor = self.descriptor.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.inner.config());
        let (active, report) = self
            .runtime
            .inner
            .run_foreground(
                Operation::RunBulkBuild,
                Some(self.logical_index_id()),
                control,
                move |mut context| async move {
                    crate::runtime::bulk_worker::complete(
                        &mut context,
                        manifest,
                        descriptor,
                        options,
                        retry,
                    )
                    .await
                },
            )
            .await?;
        let index = super::Index::new(Arc::clone(&self.runtime), self.name.clone(), active)?;
        Ok((index, report))
    }

    /// Executes sharded construction and backend emission, leaving the index hidden in `Loaded`.
    /// Call `publish` to validate and activate it.
    pub async fn run_worker(&self, options: BulkWorkerOptions) -> Result<()> {
        self.run_worker_with_control(options, OperationOptions::default())
            .await
    }

    /// Runs a worker with explicit cancellation/deadline control.
    pub async fn run_worker_with_control(
        &self,
        options: BulkWorkerOptions,
        control: OperationOptions,
    ) -> Result<()> {
        let manifest = self.manifest.clone();
        let descriptor = self.descriptor.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.inner.config());
        self.runtime
            .inner
            .run_foreground(
                Operation::RunBulkBuild,
                Some(self.logical_index_id()),
                control,
                move |mut context| async move {
                    crate::runtime::bulk_worker::run(
                        &mut context,
                        manifest,
                        descriptor,
                        options,
                        retry,
                    )
                    .await
                },
            )
            .await
            .map(|_| ())
    }

    /// Requires completed loading, runs parallel slice validation, and atomically publishes.
    /// Unknown publication is resolved by the original Logical Index identity.
    pub async fn publish(&self) -> Result<super::Index<B>> {
        self.publish_with_control(OperationOptions::default()).await
    }

    /// Publishes with cancellation and deadline control.
    pub async fn publish_with_control(&self, control: OperationOptions) -> Result<super::Index<B>> {
        let manifest = self.manifest.clone();
        let descriptor = self.descriptor.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.inner.config());
        let cleanup_control = control.clone();
        let active = self
            .runtime
            .inner
            .run_foreground(
                Operation::PublishBulkBuild,
                Some(self.logical_index_id()),
                control,
                move |mut context| async move {
                    crate::runtime::bulk_publish::publish(&mut context, manifest, descriptor, retry)
                        .await
                },
            )
            .await?;
        match self.cleanup_with_control(cleanup_control).await {
            Err(e) if e.kind() == super::ErrorKind::BulkBuildBusy => {}
            result => result?,
        }
        super::Index::new(Arc::clone(&self.runtime), self.name.clone(), active)
    }

    /// Reclaims owned scratch files after publication or abort. This is safe to
    /// retry after process loss, including after index data has been dropped.
    /// Busy means workspace IO still holds a shared lock; retry after it exits.
    pub async fn cleanup(&self) -> Result<()> {
        self.cleanup_with_control(OperationOptions::default()).await
    }

    /// Reclaims owned scratch files with explicit cancellation and deadline control.
    pub async fn cleanup_with_control(&self, options: OperationOptions) -> Result<()> {
        let index = self.manifest.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.inner.config());
        self.runtime
            .inner
            .run_foreground(
                Operation::CleanupBulkBuild,
                Some(self.logical_index_id()),
                options,
                move |mut context| async move {
                    crate::runtime::bulk_worker::cleanup(
                        &mut context,
                        index.logical_index_id(),
                        retry,
                    )
                    .await
                },
            )
            .await
    }

    /// Removes an unpublished reservation and its backend data idempotently.
    ///
    /// Caller-owned input files are retained. A published index cannot be
    /// aborted; use the normal Runtime drop operation to remove it explicitly.
    pub async fn abort(&self) -> Result<()> {
        self.abort_with_control(OperationOptions::default()).await
    }

    /// Aborts with explicit cancellation and deadline control.
    pub async fn abort_with_control(&self, options: OperationOptions) -> Result<()> {
        let name = self.name.clone();
        let id = self.logical_index_id();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.inner.config());
        let cleanup_options = options.clone();
        self.runtime
            .inner
            .run_foreground(
                Operation::AbortBulkBuild,
                Some(id),
                options,
                move |mut context| async move {
                    lifecycle::drop_index_bound(&mut context, name, retry, Some(id)).await
                },
            )
            .await?;
        match self.cleanup_with_control(cleanup_options).await {
            Err(e) if e.kind() == super::ErrorKind::BulkBuildBusy => Ok(()),
            result => result,
        }
    }
}

impl<B: Backend> Clone for BulkBuildJob<B> {
    fn clone(&self) -> Self {
        Self::new(
            Arc::clone(&self.runtime),
            self.name.clone(),
            self.manifest.clone(),
            self.descriptor.clone(),
        )
    }
}

impl<B: Backend> fmt::Debug for BulkBuildJob<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BulkBuildJob")
            .field("logical_index_id", &self.logical_index_id())
            .finish_non_exhaustive()
    }
}
