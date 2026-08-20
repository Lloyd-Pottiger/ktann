//! Numeric semantics, predicates, tree traversal, reranking, and caching.
//!
//! `numeric`, `predicate`, and `rabitq` own the format-v1 numeric, predicate,
//! and RaBitQ7 contracts. `plan` owns Tree Key range planning and bounded
//! directory enumeration. `traverse` owns deterministic bounded best-first
//! traversal across the enumerated trees, including intermediate topology
//! states, synopsis pruning, per-leaf candidate selection, and traversal
//! budget accounting. `rerank` owns exact Leaf Entry filtering, bounded
//! Vector Record loading, and exact reranking. The RaBitQ7 codec is consumed
//! by storage's Leaf Entry encoding; the remaining planner, traversal, and
//! candidate-selection items are consumed by the public search operation
//! tracked in #30.

pub(crate) mod numeric;
pub(crate) mod plan;
pub(crate) mod predicate;
pub(crate) mod rabitq;
pub(crate) mod rerank;
pub(crate) mod traverse;
