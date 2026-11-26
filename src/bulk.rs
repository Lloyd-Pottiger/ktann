//! Segmented input snapshots, deterministic shard planning, and bounded bulk construction.
//!
//! Input snapshots remain caller owned. Bulk Build execution streams input segments
//! directly into sharded leaf construction, bounded halo refinement, and backend KV batches.

pub(crate) mod digest;
pub(crate) mod files;
pub(crate) mod graphs;
mod input;
pub(crate) mod plan;
pub(crate) mod sharded;

pub(crate) use digest::LtHash;
pub use files::{INPUT_MANIFEST_BYTES, InputManifest};
pub use input::{InputReader, InputSnapshot, InputSnapshotWriter};
pub(crate) use plan::ShardPlan;
