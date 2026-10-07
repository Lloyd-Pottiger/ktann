//! Durable, bounded files used by Bulk Build preparation.
//!
//! These APIs prepare immutable input snapshots, tree plans and serving KV files. They do
//! not reserve a Logical Index, load backend data, or make an index queryable.
//! A coordinator must persist each returned manifest before assigning dependent
//! work, and reopen files against that expected manifest after a restart.

mod files;
mod forest;
mod input;
mod plan;
mod serving;
mod sort;

pub use files::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest};
pub use input::{InputReader, InputSnapshot};

pub use forest::{ForestArtifact, ForestOptions, ForestPartition, ForestReader, ForestReport};

pub use serving::{ServingArtifact, ServingEntry, ServingOptions, ServingReader, ServingReport};
