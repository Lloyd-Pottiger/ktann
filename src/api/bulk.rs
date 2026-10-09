//! Identity-bound handles for durable Bulk Build reservations.

use std::fmt;
use std::sync::Arc;

use crate::observe::labels::Operation;
use crate::runtime::{RuntimeInner, lifecycle};
use crate::storage::backend::Backend;
use crate::storage::values::{BuildDescriptor, IndexManifest};

use super::{IndexName, LogicalIndexId, OperationOptions, Result};

/// Persisted lifecycle of the Logical Index owned by a Bulk Build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BulkBuildStatus {
    /// The name and immutable input identity are reserved; the index is hidden.
    Preparing,
    /// A bounded exact comparison of sealed backend data is in progress.
    Validating {
        /// Serving entries already checked against accepted bytes.
        verified_entries: u64,
        /// Total expected serving entries.
        total_entries: u64,
    },
    /// A persistent input/integrity/resource failure requires abort and rebuild.
    Failed {
        /// Redacted, bounded error category.
        kind: super::ErrorKind,
    },
    /// Serving bytes are partially loaded and remain hidden.
    Loading {
        /// Entries committed atomically with the checkpoint.
        loaded_entries: u64,
        /// Total entries in the immutable serving artifact.
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

/// Per-transaction limits for serving artifact loading, including its checkpoint.
/// Adapter admission may impose smaller limits. Physical namespace overhead is
/// included in the byte budget. At least two mutations must fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BulkLoadOptions {
    /// Maximum data plus checkpoint mutations in one transaction.
    pub max_mutations: usize,
    /// Maximum physical key plus value bytes in one transaction.
    pub max_bytes: usize,
}
impl Default for BulkLoadOptions {
    fn default() -> Self {
        Self {
            max_mutations: 128,
            max_bytes: 1024 * 1024,
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

/// Wall times of work performed by one worker invocation, excluding receipt.
/// Reused artifact stages have zero duration; publication is timed separately.
#[derive(Clone, Copy, Debug, Default)]
pub struct BulkWorkerReport {
    /// Forest sorting/training and immutable plan emission.
    pub forest: std::time::Duration,
    /// Preparation IO counts, including receipt-time sorting when supplied.
    /// Absent when an accepted forest is reused.
    pub forest_report: Option<crate::bulk::ForestReport>,
    /// Exact joins and serving encoding.
    pub serving: std::time::Duration,
    /// Backend artifact loading, including checkpoint transactions.
    pub load: std::time::Duration,
}

/// Resource bounds and an existing durable, shared preparation workspace.
/// Workers require reliable advisory file locks, atomic rename, and fsync on
/// this filesystem. The root and its coordination lock remain caller owned.
#[derive(Clone, Eq, PartialEq)]
pub struct BulkWorkerOptions {
    /// Existing absolute workspace root, separate from the source snapshot.
    pub workspace: std::path::PathBuf,
    /// Allocated buffers for forest sorting.
    pub sort_memory_bytes: usize,
    /// Forest sort scratch ceiling.
    pub sort_scratch_bytes: u64,
    /// Allocated buffers for serving preparation.
    pub serving_memory_bytes: usize,
    /// Serving scratch ceiling.
    pub serving_scratch_bytes: u64,
    /// Maximum size of each accepted artifact.
    pub max_artifact_bytes: u64,
    /// Bounded data and validation pages.
    pub load: BulkLoadOptions,
}
impl BulkWorkerOptions {
    /// Uses bounded defaults with the supplied durable workspace root.
    #[must_use]
    pub fn new(workspace: impl Into<std::path::PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            sort_memory_bytes: 64 * 1024 * 1024,
            sort_scratch_bytes: 64 * 1024 * 1024 * 1024,
            serving_memory_bytes: 128 * 1024 * 1024,
            serving_scratch_bytes: 64 * 1024 * 1024 * 1024,
            max_artifact_bytes: 64 * 1024 * 1024 * 1024,
            load: BulkLoadOptions::default(),
        }
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.workspace.is_absolute()
            || self
                .workspace
                .to_str()
                .is_none_or(|p| p.len() > 3900 || p.contains('\0'))
            || self.sort_memory_bytes == 0
            || self.sort_scratch_bytes == 0
            || self.serving_memory_bytes == 0
            || self.serving_scratch_bytes == 0
            || self.max_artifact_bytes == 0
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
    pub poll_interval: std::time::Duration,
    /// Renewable lease duration; at least three poll intervals.
    pub lease_duration: std::time::Duration,
}
impl Default for BulkSchedulerOptions {
    fn default() -> Self {
        Self {
            max_jobs: 1,
            poll_interval: std::time::Duration::from_secs(1),
            lease_duration: std::time::Duration::from_secs(30),
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
            || self.lease_duration.as_millis() > u128::from(u64::MAX)
        {
            return Err(super::Error::invalid_argument());
        }
        Ok(())
    }
}

/// A recoverable reservation for one new Logical Index.
///
/// This handle never follows a reused Index Name. Reservation launches no work;
/// callers explicitly run workers and publish after loading. Preparation, loading,
/// and validation are resumable. Source snapshots remain caller owned, including
/// after abort, and are reverified when consumed.
pub struct BulkBuildJob<B: Backend> {
    runtime: Arc<RuntimeInner<B>>,
    name: IndexName,
    manifest: IndexManifest,
    descriptor: BuildDescriptor,
}

impl<B: Backend> BulkBuildJob<B> {
    pub(crate) fn new(
        runtime: Arc<RuntimeInner<B>>,
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

    /// Loads a sealed serving artifact into this hidden Building index.
    ///
    /// The first call fixes the artifact identity. Subsequent calls resume it
    /// and take over from earlier invocations through a persisted epoch. A
    /// superseded call cannot commit further chunks. Any failure may leave a
    /// committed prefix; retry with the same artifact or abort the build.
    /// Completion does not authorize publication. Files remain caller owned.
    pub async fn load_serving(
        &self,
        artifact: &crate::bulk::ServingArtifact,
        options: BulkLoadOptions,
    ) -> Result<()> {
        self.load_serving_with_control(artifact, options, OperationOptions::default())
            .await
    }

    /// Loads with explicit cancellation and deadline control.
    pub async fn load_serving_with_control(
        &self,
        artifact: &crate::bulk::ServingArtifact,
        options: BulkLoadOptions,
        control: OperationOptions,
    ) -> Result<()> {
        let artifact = artifact.clone();
        let manifest = self.manifest.clone();
        let descriptor = self.descriptor.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.config());
        self.runtime
            .run_foreground(
                Operation::LoadBulkBuild,
                Some(self.logical_index_id()),
                control,
                move |mut context| async move {
                    crate::runtime::bulk_load::load(
                        &mut context,
                        manifest,
                        descriptor,
                        artifact,
                        options,
                        retry,
                    )
                    .await
                },
            )
            .await
    }

    /// Durably queues this job for automatic preparation, loading and publication.
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
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.config());
        self.runtime
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

    /// Resumes preparation and loading, reusing previously accepted artifacts.
    /// The new invocation takes a durable attempt epoch. Completion leaves the
    /// index hidden; call `publish` to validate and activate it.
    pub async fn run_worker(&self, options: BulkWorkerOptions) -> Result<BulkWorkerReport> {
        self.run_worker_with_control(options, OperationOptions::default())
            .await
    }
    /// Runs a worker with explicit cancellation/deadline control.
    pub async fn run_worker_with_control(
        &self,
        options: BulkWorkerOptions,
        control: OperationOptions,
    ) -> Result<BulkWorkerReport> {
        self.run_worker_input(options, None, control).await
    }

    /// Reuses bounded receipt-time sorting. Loss of this volatile preparation
    /// never prevents ordinary `run_worker` recovery from the sealed source.
    pub async fn run_worker_with_prepared_input(
        &self,
        options: BulkWorkerOptions,
        prepared: crate::bulk::PreparedInput,
        control: OperationOptions,
    ) -> Result<BulkWorkerReport> {
        self.run_worker_input(options, Some(prepared), control)
            .await
    }

    async fn run_worker_input(
        &self,
        options: BulkWorkerOptions,
        prepared: Option<crate::bulk::PreparedInput>,
        control: OperationOptions,
    ) -> Result<BulkWorkerReport> {
        let manifest = self.manifest.clone();
        let descriptor = self.descriptor.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.config());
        self.runtime
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
                        prepared,
                        retry,
                    )
                    .await
                },
            )
            .await
    }
    /// Freezes writes, resumes exact paged backend validation, and publishes.
    /// Unknown publication is resolved by the original Logical Index identity.
    pub async fn publish(&self) -> Result<super::Index<B>> {
        self.publish_with_control(OperationOptions::default()).await
    }
    /// Publishes with cancellation and deadline control.
    pub async fn publish_with_control(&self, control: OperationOptions) -> Result<super::Index<B>> {
        let manifest = self.manifest.clone();
        let descriptor = self.descriptor.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.config());
        let active = self
            .runtime
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
        match self.cleanup().await {
            Err(e) if e.kind() == super::ErrorKind::BulkBuildBusy => {}
            result => result?,
        }
        super::Index::new(Arc::clone(&self.runtime), self.name.clone(), active)
    }
    /// Reclaims owned attempt files after publication or abort. This is safe to
    /// retry after process loss, including after index data has been dropped.
    /// Busy means workspace IO still holds a shared lock; retry after it exits.
    pub async fn cleanup(&self) -> Result<()> {
        let index = self.manifest.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.config());
        self.runtime
            .run_foreground(
                Operation::CleanupBulkBuild,
                Some(self.logical_index_id()),
                OperationOptions::default(),
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
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.config());
        self.runtime
            .run_foreground(
                Operation::AbortBulkBuild,
                Some(id),
                options,
                move |mut context| async move {
                    lifecycle::drop_index_bound(&mut context, name, retry, Some(id)).await
                },
            )
            .await?;
        match self.cleanup().await {
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
