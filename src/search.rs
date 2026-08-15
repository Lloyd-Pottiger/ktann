//! Numeric semantics, predicates, tree traversal, reranking, and caching.
//!
//! Numeric implementation details live behind the stable crate-visible seam
//! re-exported here. Traversal, filtering, storage, and runtime concerns remain
//! outside that deep module.
mod numeric;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the search pipeline consumes the compiled predicate evaluator"
    )
)]
mod predicate;
mod rabitq;

#[expect(
    unused_imports,
    reason = "this preserves the caller seam for the future search pipeline"
)]
pub(crate) use numeric::{ExactDistance, VectorKernel};
#[expect(
    unused_imports,
    reason = "these preserve the RaBitQ7 seams for mutation and search pipelines"
)]
pub(crate) use rabitq::{
    ApproximateCandidate, ApproximateDistance, OverlapSelection, RaBitQ7, RaBitQQuery,
    select_global_overlap, select_leaf_overlap,
};
