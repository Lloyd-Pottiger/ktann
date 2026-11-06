//! Atomic topology steps for expose-then-drain splits (ADR 0014) and
//! target-reselecting merges (ADR 0008).
//!
//! Maintenance drives these bounded transactions across snapshots. Each step
//! uses update-protected authority reads and recognizes previously committed
//! progress, allowing recovery from unknown commit outcomes.
//!
//! - Split exposure atomically creates both `ReceivingSplit` targets with
//!   centroids and advances the source from `Splitting` to `DrainingSplit`.
//! - Every non-root target has one Child Entry installed with its creation;
//!   root targets occupy exclusive slots in Partition Key 1's split State.
//!   Stale source or parent authority aborts before writes.
//! - Drain batches atomically insert target entries, delete source entries,
//!   update Record Locations for leaves, and update exact counts, cache epochs,
//!   and target Synopses once per partition. Entries already removed by a
//!   completed concurrent mutation are skipped; other mismatches are Corruption.
//! - The exact source Header count alone determines completion. At zero, only
//!   four metadata keys remain; cleanup uses a transactional range clear or
//!   bounded point deletes according to backend capabilities.
//!
//! Authority, level, state, reference, and count mismatches fail with
//! [`ErrorKind::Corruption`]. Callers must roll back on any error.

// The private modules own authority reads, discovery, entry moves, and state
// transitions. Callers use this module's typed operations; transaction ownership
// and commit/retry policy stay with the caller.

mod authority;
mod discovery;
mod relocation;
mod transitions;

pub(crate) use authority::{
    authority_for_update, read_authority, read_authority_batch, read_authority_opt,
    read_authority_pair,
};
pub(crate) use discovery::{IncomingEdgeProof, LevelCandidate, same_level_candidates};
pub(crate) use relocation::leaf_relocation_batch_limit;
pub use relocation::{
    LeafDrainEntry, Movement, read_child_drain_candidates, read_leaf_drain_candidates,
    relocate_child_entries, relocate_leaf_entries,
};
pub use transitions::{
    MergeCompletion, MergeStart, SourceRemoval, SplitCompletion, SplitExposure, SplitStart,
    begin_merge, begin_split, expose_split_targets, finalize_merge, finalize_split,
};
pub(crate) use transitions::{
    expose_split_targets_with_parent, finalize_merge_with_parent, finalize_split_with_parent,
    update_centroid,
};

use crate::api::{Error, ErrorKind, PartitionKey};

/// The stable root Partition Key of every tree.
///
/// The Tree Manifest constructor rejects any root other than Partition Key 1,
/// so construction cannot fail.
pub(crate) fn root_partition() -> PartitionKey {
    match PartitionKey::new(1) {
        Ok(root) => root,
        Err(_) => unreachable!("Partition Key 1 is nonzero"),
    }
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}
