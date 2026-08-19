//! Numeric semantics, predicates, tree traversal, reranking, and caching.
//!
//! `numeric`, `predicate`, and `rabitq` own the format-v1 numeric, predicate,
//! and RaBitQ7 contracts. `plan` owns Tree Key range planning and bounded
//! directory enumeration. `rerank` owns exact Leaf Entry filtering, bounded
//! Vector Record loading, and exact reranking. `cache` owns the
//! snapshot-validated Partition Cache of decoded search bodies. The RaBitQ7
//! codec is consumed by storage's Leaf Entry encoding; the cache, the rerank
//! stage, and the remaining numeric, predicate, and candidate-selection items
//! are consumed by the search pipeline tracked in #9 and #30.

pub(crate) mod cache;
pub(crate) mod numeric;
pub(crate) mod plan;
pub(crate) mod predicate;
pub(crate) mod rabitq;
pub(crate) mod rerank;
