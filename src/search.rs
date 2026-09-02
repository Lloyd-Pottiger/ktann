//! Numeric semantics, predicates, tree traversal, reranking, and caching.
//!
//! `numeric`, `predicate`, and `rabitq` own the format-v1 numeric, predicate,
//! and RaBitQ7 contracts. `plan` owns Tree Key range planning and bounded
//! directory enumeration. `traverse` owns deterministic bounded best-first
//! traversal across the enumerated trees, including intermediate topology
//! states, synopsis pruning, per-leaf candidate selection, and traversal
//! budget accounting. `rerank` owns exact Leaf Entry filtering, bounded
//! Vector Record loading, and exact reranking. `cache` owns the
//! snapshot-validated Partition Cache of decoded search bodies. The RaBitQ7
//! codec is consumed by storage's Leaf Entry encoding; the runtime's search
//! operation consumes the planner, cache, traversal, and rerank stages.

pub(crate) mod cache;
pub(crate) mod numeric;
pub(crate) mod plan;
pub(crate) mod predicate;
pub(crate) mod rabitq;
pub(crate) mod rerank;
pub(crate) mod traverse;

/// Returns the level-scaled beam width shared by search and foreground writes.
/// The configured leaf-level width halves toward the root and never drops
/// below one.
pub(crate) fn beam_width(leaf_beam: u32, level: u32) -> u32 {
    leaf_beam
        .checked_shr(level.saturating_sub(1))
        .unwrap_or(0)
        .max(1)
}
