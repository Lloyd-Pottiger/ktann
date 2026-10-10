//! Durable input capture and resource options for resumable Bulk Builds.
//!
//! Source snapshots remain caller owned. Reserve a job through the Runtime,
//! then schedule it or complete it directly. Optional receipt-time preparation
//! avoids sorting the source again without becoming recovery authority.

mod files;
pub(crate) mod forest;
mod input;
mod plan;
pub(crate) mod serving;
mod sort;

pub use files::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest};
pub use input::{
    InputReader, InputSnapshot, InputSnapshotWriter, PreparedInput, PreparedInputWriter,
};

pub use crate::construction::ConstructionOptions;
pub use forest::{ForestOptions, ForestReport};
pub use serving::ServingReport;

pub(crate) use forest::ForestArtifact;
pub(crate) use serving::{ServingArtifact, ServingEntry, ServingOptions, ServingReader};
