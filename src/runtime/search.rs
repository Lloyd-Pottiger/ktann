//! The public bounded approximate-search operation.
//!
//! One search runs over one consistent backend snapshot (design `search.md`
//! section 6 and ADR 0011): the persisted Active Manifest is validated and
//! bound, Tree Keys are enumerated under the scanned-key budget, every
//! eligible tree is traversed under the partition and Leaf Entry budgets with
//! snapshot-validated cache reads, the merged candidates pass global overlap
//! selection, and the survivors' Vector Records are batch-loaded and exactly
//! reranked — all from the same snapshot.
//!
//! Request validation, query preprocessing, predicate compilation, and Tree
//! Key planning are pure functions of the handle's immutable Manifest and run
//! before foreground admission in [`PreparedSearch`]. The response reports the
//! actual budget usage, every exhausted dimension, and per-leaf overlap
//! truncation. Success deliberately makes no exact-global-top-k, completeness,
//! continuation, or monotonic-across-budgets guarantee.

use crate::api::{
    Error, ErrorKind, PartitionKey, Result, SearchBudgetExhaustion, SearchBudgetUsage,
    SearchBudgets, SearchOutcome, SearchRequest,
};
use crate::observe::metrics;
use crate::search::cache::PartitionCache;
use crate::search::numeric::VectorKernel;
use crate::search::plan::{TreeKeyPlan, enumerate_tree_keys, plan_tree_keys};
use crate::search::predicate::CompiledPredicate;
use crate::search::rabitq::{ApproximateCandidate, select_global_overlap};
use crate::search::rerank::{LeafCandidate, exact_rerank};
use crate::search::traverse::{DEFAULT_LEAF_BEAM, TraversalRequest, traverse};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::{Backend, ScanLimits};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::IndexManifest;

use super::OperationContext;
use super::reads::opened_manifest;

/// The page bounds for Tree Manifest directory enumeration.
///
/// The scanned Tree Key budget bounds the whole enumeration; these bounds
/// only keep one directory page small. Manifest keys and values are fixed
/// small records, so a 1 MiB page never truncates a legal item.
const DIRECTORY_SCAN_LIMITS: ScanLimits = ScanLimits {
    item_limit: 256,
    byte_limit: 1_024 * 1_024,
};

/// A validated search request prepared for snapshot execution.
///
/// Construction validates the request against the handle's immutable Manifest,
/// resolves the effective Search Budgets, builds the numeric kernel, and
/// compiles the routing vector, the Filter Predicate, and the Tree Key plan.
/// Every captured input is immutable, so the prepared artifacts stay valid for
/// the snapshot that later revalidates the same Manifest identity.
pub(crate) struct PreparedSearch {
    request: SearchRequest,
    budgets: SearchBudgets,
    leaf_beam: u32,
    kernel: VectorKernel,
    routing: Box<[f32]>,
    predicate: Option<CompiledPredicate>,
    plan: TreeKeyPlan,
}

impl PreparedSearch {
    /// Validates and compiles one search request for `manifest`.
    pub(crate) fn new(
        manifest: &IndexManifest,
        defaults: SearchBudgets,
        range_limit: u32,
        mut request: SearchRequest,
    ) -> Result<Self> {
        let config = manifest.config();
        let budgets = request.validate(config.dimension(), config.fields(), defaults)?;
        let leaf_beam = request
            .options()
            .leaf_beam_size()
            .unwrap_or(DEFAULT_LEAF_BEAM);
        let kernel = VectorKernel::new(
            config.dimension(),
            config.metric(),
            *manifest.rotation_seed(),
        )?;
        let routing = kernel.preprocess(request.vector())?;
        let predicate = request
            .predicate()
            .cloned()
            .map(|predicate| CompiledPredicate::compile(predicate, config.fields()))
            .transpose()?;
        let plan = plan_tree_keys(manifest, request.predicate(), range_limit)?;
        Ok(Self {
            request,
            budgets,
            leaf_beam,
            kernel,
            routing,
            predicate,
            plan,
        })
    }
}

/// Runs one prepared search against one consistent backend snapshot.
///
/// Every stage — Manifest validation, Tree Key enumeration, traversal with
/// synopsis pruning and candidate selection, and exact Vector Record loading
/// and reranking — reads from the same snapshot. Cancellation and deadline
/// are checked between stages; a search never commits, so a cancelled or
/// expired operation is simply an error.
///
/// The second return value is the traversal's demand-driven maintenance
/// discovery: visited partitions in a split or merge state and
/// threshold-crossing `Ready` partitions, which the caller offers to the
/// Runtime's Fixup queue after success.
pub(crate) async fn search<B: Backend>(
    context: &mut OperationContext<B>,
    cache: &PartitionCache,
    handle_manifest: &IndexManifest,
    prepared: PreparedSearch,
) -> Result<(SearchOutcome, Vec<(TreeKey, PartitionKey)>)> {
    context.checkpoint()?;
    let backend = context.backend();
    let raw = backend.begin_read().await?;
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    let current = opened_manifest(
        txn.get(LogicalKey::Manifest(handle_manifest.logical_index_id()))
            .await?,
        handle_manifest,
    )?;
    let raw = txn.into_raw();
    let mut txn = ReadLogicalTxn::for_index(raw, &current)?;
    context.checkpoint()?;

    // Bounded Tree Key enumeration finishes before traversal so budget use is
    // deterministic.
    let enumeration = enumerate_tree_keys(
        &mut txn,
        &current,
        &prepared.plan,
        prepared.budgets.scanned_tree_keys(),
        DIRECTORY_SCAN_LIMITS,
    )
    .await?;
    context.checkpoint()?;

    let mut traversal = traverse(
        &mut txn,
        cache,
        &prepared.kernel,
        TraversalRequest::new(
            &prepared.routing,
            enumeration.trees(),
            prepared.predicate.as_ref(),
            prepared.request.k(),
            prepared.budgets,
            prepared.leaf_beam,
        )?,
    )
    .await?;
    context.checkpoint()?;

    let maintenance = traversal.take_maintenance();
    let visited_partitions = traversal.visited_partitions();
    let visited_leaf_entries = traversal.visited_leaf_entries();
    let partition_budget_exhausted = traversal.partition_budget_exhausted();
    let leaf_entry_budget_exhausted = traversal.leaf_entry_budget_exhausted();
    let rabitq_overlap_truncated = traversal.rabitq_overlap_truncated();

    // Global overlap selection retains every candidate whose conservative
    // interval reaches the kth-smallest upper endpoint, then truncates the
    // survivors to the rerank budget in deterministic rough order. That
    // truncation is eligible rerank work the depleted budget prevented.
    let rerank_budget = usize::try_from(prepared.budgets.exact_rerank_candidates())
        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let pool: Vec<ApproximateCandidate<LeafCandidate>> = traversal
        .into_candidates()
        .into_iter()
        .map(ApproximateCandidate::from)
        .collect();
    let selection = select_global_overlap(pool, prepared.request.k(), rerank_budget)?;
    let rerank_budget_exhausted = selection.truncated();
    let selected: Vec<LeafCandidate> = selection.into_values();

    let rerank = exact_rerank(
        &mut txn,
        &prepared.kernel,
        prepared.request.vector(),
        selected,
        prepared.request.k(),
        prepared.budgets.exact_rerank_candidates(),
    )
    .await?;
    context.checkpoint()?;

    let exact_rerank_candidates = rerank.exact_rerank_candidates();
    let rerank_exhausted = rerank_budget_exhausted || rerank.exact_rerank_budget_exhausted();
    let outcome = SearchOutcome {
        hits: rerank.into_hits(),
        usage: SearchBudgetUsage {
            scanned_tree_keys: enumeration.scanned_tree_keys(),
            visited_partitions,
            visited_leaf_entries,
            exact_rerank_candidates,
        },
        exhausted: SearchBudgetExhaustion {
            scanned_tree_keys: enumeration.scanned_tree_key_budget_exhausted(),
            visited_partitions: partition_budget_exhausted,
            visited_leaf_entries: leaf_entry_budget_exhausted,
            exact_rerank_candidates: rerank_exhausted,
        },
        rabitq_overlap_truncated,
    };
    metrics::search_budget(&outcome.usage, &outcome.exhausted);
    Ok((outcome, maintenance))
}
