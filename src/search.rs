//! Numeric semantics, predicates, tree traversal, reranking, and caching.
//!
//! `numeric`, `predicate`, and `rabitq` own the format-v1 numeric, predicate,
//! and RaBitQ7 contracts. `plan` owns Tree Key range planning and bounded
//! directory enumeration. The RaBitQ7 codec is consumed by storage's Leaf Entry
//! encoding; the remaining numeric, predicate, and candidate-selection items are
//! consumed by the search pipeline tracked in #9, #28, and #30.

pub(crate) mod numeric;
pub(crate) mod plan;
pub(crate) mod predicate;
pub(crate) mod rabitq;
