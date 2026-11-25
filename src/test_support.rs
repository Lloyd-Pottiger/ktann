//! Low-level controls for testing storage, maintenance, and recovery invariants.
//!
//! This module is excluded from production builds unless `test-support` is enabled.

pub use crate::bulk::{ForestArtifact, ForestPartition, ForestReader};
pub use crate::bulk::{ServingArtifact, ServingEntry, ServingOptions, ServingReader};

/// Pure tree-construction primitives used by algorithm tests.
pub mod construction {
    pub use crate::construction::*;
}

/// Held process-local graph body lock for interleaving tests.
pub struct GraphBodyLock {
    _guards: crate::runtime::graph_maintenance::GraphBodyGuards,
}

/// Acquires one process-local graph body lock for interleaving tests.
pub async fn lock_graph_body(
    executor: &crate::runtime::GraphMaintenance,
    index: crate::api::LogicalIndexId,
    tree: &crate::storage::keys::TreeKey,
    partition: crate::api::PartitionKey,
) -> crate::api::Result<GraphBodyLock> {
    executor
        .lock_bodies(index, tree, vec![partition])
        .await
        .map(|guards| GraphBodyLock { _guards: guards })
}
