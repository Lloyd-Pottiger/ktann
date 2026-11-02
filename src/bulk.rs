//! Bounded offline input capture, forest construction, and serving artifacts.
//!
//! Files remain caller owned. These synchronous operations prepare and validate
//! immutable artifacts; they do not load or publish a Logical Index.

mod files;
mod forest;
mod graphs;
mod input;
mod plan;
mod serving;
mod sort;

pub use files::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest};
pub use forest::{ForestArtifact, ForestOptions, ForestPartition, ForestReader, ForestReport};
pub use input::{
    InputReader, InputSnapshot, InputSnapshotWriter, PreparedInput, PreparedInputWriter,
};
pub use serving::{ServingArtifact, ServingEntry, ServingOptions, ServingReader, ServingReport};
