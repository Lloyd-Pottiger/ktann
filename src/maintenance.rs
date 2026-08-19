//! Foreground routing and searchable split and merge state machines.
//!
//! This module owns record routing, exact foreground membership changes,
//! binary K-means tree shape, and the persistent split/merge state machines
//! (design `maintenance.md`). `routing` implements the initial contract:
//! stable-topology descent from a tree root to one Leaf Partition with
//! observed-parent validation, plus lazy empty-root tree creation. The
//! split/merge state machines (#10, #31) extend the same routing rules to the
//! intermediate topology states.

pub mod routing;
