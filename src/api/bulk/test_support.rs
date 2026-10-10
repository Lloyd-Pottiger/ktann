//! Stage controls for transactional fault-injection tests.

use super::*;

#[doc(hidden)]
impl<B: Backend> BulkBuildJob<B> {
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

    /// Injects an artifact directly to exercise transactional loading faults.
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

    /// Stops after loading so tests can establish a validation or recovery boundary.
    pub async fn prepare_and_load(&self, options: BulkWorkerOptions) -> Result<BulkBuildReport> {
        let manifest = self.manifest.clone();
        let descriptor = self.descriptor.clone();
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.config());
        self.runtime
            .run_foreground(
                Operation::RunBulkBuild,
                Some(self.logical_index_id()),
                OperationOptions::default(),
                move |mut context| async move {
                    crate::runtime::bulk_worker::run(
                        &mut context,
                        manifest,
                        descriptor,
                        options,
                        None,
                        retry,
                    )
                    .await
                },
            )
            .await
    }
}
