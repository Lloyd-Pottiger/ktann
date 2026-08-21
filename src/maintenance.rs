//! Foreground routing and searchable split and merge state machines.
//!
//! This module owns record routing, exact foreground membership changes,
//! binary K-means tree shape, and the persistent split/merge state machines
//! (design `maintenance.md`). `routing` implements state-aware descent from a
//! tree root to one write-accepting Leaf Partition with observed-parent and
//! root-slot validation, plus lazy empty-root tree creation. `mutation`
//! orchestrates public insert, replacement upsert, delete, and atomic batches
//! over the storage module's typed membership operations with whole-operation
//! retries. `training` implements the deterministic binary K-means split
//! training (ADR 0015) whose target centroids the split state machine
//! publishes. `split` drives the bounded expose-then-drain split state machine
//! (ADR 0014) that `routing` routes through; the merge state machine (#31)
//! extends the same rules to `Merging`.

pub(crate) mod mutation;
pub mod routing;
pub mod split;
pub mod training;
