//! Foreground routing and searchable split and merge state machines.
//!
//! This module owns record routing, exact foreground membership changes,
//! binary K-means tree shape, and the persistent split/merge state machines
//! (design `maintenance.md`). `routing` implements the initial contract:
//! stable-topology descent from a tree root to one Leaf Partition with
//! observed-parent validation, plus lazy empty-root tree creation. `mutation`
//! orchestrates public insert, replacement upsert, delete, and atomic batches
//! over the storage module's typed membership operations with whole-operation
//! retries. `training` implements the deterministic binary K-means split
//! training (ADR 0015) whose target centroids the split state machine (#10)
//! publishes. The split/merge state machines (#10, #31) extend the same
//! routing rules to the intermediate topology states.

pub(crate) mod mutation;
pub mod routing;
pub mod training;
