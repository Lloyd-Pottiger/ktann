//! Numeric semantics, exact predicates, bounded traversal, and reranking.
//!
//! Search stages share one snapshot. The Partition Cache validates bodies
//! against snapshot Headers; traversal accounts for budgets and topology;
//! RaBitQ selects approximate candidates before exact Vector Record reranking.

pub(crate) mod cache;
pub(crate) mod candidates;
pub(crate) mod local_hnsw;
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

/// Gives ready-only backends and cached CPU loops an explicit scheduling boundary.
/// This is execution fairness, not a search budget: callers still process every
/// admitted entry in order. Check both sides so cancellation cannot start a new
/// batch after another task runs.
pub(crate) async fn cooperate(
    check: &(impl Fn() -> crate::api::Result<()> + Sync),
) -> crate::api::Result<()> {
    check()?;
    tokio::task::yield_now().await;
    check()
}
