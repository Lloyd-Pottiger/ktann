//! The public handle to one Active Logical Index.

use std::fmt;
use std::sync::Arc;

use bytes::Bytes;

use crate::runtime::RuntimeInner;
use crate::runtime::reads;
use crate::storage::backend::Backend;
use crate::storage::values::{IndexLifecycle, IndexManifest};

use super::{
    Error, GetOptions, IndexConfig, IndexName, LogicalIndexId, OperationOptions, Result,
    StoredRecord, validate_id, validate_ids,
};

/// A cheap cloneable handle to one Active Logical Index.
///
/// The handle retains the owning Runtime and the exact Index Manifest opened
/// by [`Runtime::create_index`](crate::runtime::Runtime::create_index) or
/// [`Runtime::open_index`](crate::runtime::Runtime::open_index). It never
/// retargets to a newer Logical Index created under the same Index Name.
pub struct Index<B: Backend> {
    runtime: Arc<RuntimeInner<B>>,
    name: IndexName,
    manifest: Arc<IndexManifest>,
}

impl<B: Backend> Index<B> {
    pub(crate) fn new(
        runtime: Arc<RuntimeInner<B>>,
        name: IndexName,
        manifest: IndexManifest,
    ) -> Result<Self> {
        if manifest.lifecycle() != IndexLifecycle::Active {
            return Err(Error::new(super::ErrorKind::IndexDropping));
        }
        Ok(Self {
            runtime,
            name,
            manifest: Arc::new(manifest),
        })
    }

    /// Returns the caller-chosen Index Name used to open this handle.
    #[must_use]
    pub fn name(&self) -> &IndexName {
        &self.name
    }

    /// Returns the never-reused Logical Index ID.
    #[must_use]
    pub fn logical_index_id(&self) -> LogicalIndexId {
        self.manifest.logical_index_id()
    }

    /// Returns the immutable Index Manifest configuration.
    #[must_use]
    pub fn config(&self) -> &IndexConfig {
        self.manifest.config()
    }

    /// Reads one Vector Record by Record ID.
    ///
    /// The read validates the persisted Active Manifest and the requested
    /// Record Group from one consistent backend snapshot. It returns
    /// `Ok(None)` when the Record ID is absent. A partial Record Group or a
    /// Manifest that no longer matches this handle fails with
    /// [`ErrorKind::Corruption`]; a Dropping Logical Index fails with
    /// [`ErrorKind::IndexDropping`] and a dropped one with
    /// [`ErrorKind::IndexNotFound`]. The Opaque Payload projection is closed
    /// by [`GetOptions`].
    pub async fn get(&self, id: Bytes, options: GetOptions) -> Result<Option<StoredRecord>> {
        self.get_with_control(id, options, OperationOptions::default())
            .await
    }

    /// Reads one Vector Record with explicit operation control.
    pub async fn get_with_control(
        &self,
        id: Bytes,
        options: GetOptions,
        operation_options: OperationOptions,
    ) -> Result<Option<StoredRecord>> {
        validate_id(&id)?;
        let manifest = Arc::clone(&self.manifest);
        self.runtime
            .run_foreground(operation_options, move |mut context| async move {
                reads::get_record(&mut context, &manifest, id, options.includes_payload()).await
            })
            .await
    }

    /// Reads Vector Records by Record ID in one bounded batch.
    ///
    /// The result is same-order and same-length with the input: `Ok(None)`
    /// marks an absent Record ID and duplicate Record IDs repeat their value.
    /// An empty batch succeeds with an empty result. All Record Groups and the
    /// validated Manifest are read from one consistent backend snapshot, and
    /// the backend's batch and key limits, cancellation, and deadline apply to
    /// the whole operation.
    pub async fn batch_get(
        &self,
        ids: Vec<Bytes>,
        options: GetOptions,
    ) -> Result<Vec<Option<StoredRecord>>> {
        self.batch_get_with_control(ids, options, OperationOptions::default())
            .await
    }

    /// Reads Vector Records with explicit operation control.
    pub async fn batch_get_with_control(
        &self,
        ids: Vec<Bytes>,
        options: GetOptions,
        operation_options: OperationOptions,
    ) -> Result<Vec<Option<StoredRecord>>> {
        validate_ids(&ids)?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let manifest = Arc::clone(&self.manifest);
        self.runtime
            .run_foreground(operation_options, move |mut context| async move {
                reads::batch_get_records(&mut context, &manifest, ids, options.includes_payload())
                    .await
            })
            .await
    }
}

impl<B: Backend> Clone for Index<B> {
    fn clone(&self) -> Self {
        Self {
            runtime: Arc::clone(&self.runtime),
            name: self.name.clone(),
            manifest: Arc::clone(&self.manifest),
        }
    }
}

impl<B: Backend> fmt::Debug for Index<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Index")
            .field("name", &"[REDACTED]")
            .field("logical_index_id", &self.manifest.logical_index_id())
            .field("lifecycle", &self.manifest.lifecycle())
            .finish()
    }
}
