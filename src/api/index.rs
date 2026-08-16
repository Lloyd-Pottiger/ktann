//! The public handle to one Active Logical Index.

use std::fmt;
use std::sync::Arc;

use crate::runtime::RuntimeInner;
use crate::storage::backend::Backend;
use crate::storage::values::{IndexLifecycle, IndexManifest};

use super::{Error, IndexConfig, IndexName, LogicalIndexId, Result};

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
