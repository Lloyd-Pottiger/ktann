//! Low-level Bulk Build controls for testing storage and recovery invariants.
//!
//! This module is excluded from production builds unless `test-support` is enabled.

pub use crate::bulk::forest::{ForestArtifact, ForestPartition, ForestReader};
pub use crate::bulk::serving::{ServingArtifact, ServingEntry, ServingOptions, ServingReader};

/// Pure tree-construction primitives used by algorithm tests.
pub mod construction {
    pub use crate::construction::*;
}
