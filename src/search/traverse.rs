//! Deterministic, bounded best-first traversal across selected Tree Keys.
//!
//! Roots are always admitted. Filtered Exact searches retain branches until
//! each Tree Key reaches its effective-leaf beam; other searches use top-B.
//! Each distinct partition is charged before visiting its complete body.
//! Beam pruning is not budget exhaustion; exhaustion means eligible work was
//! prevented by a depleted budget (ADR 0011).
//!
//! Current Child Entries reach non-root partitions in every committed state.
//! A Splitting root exposes only its body; DrainingSplit also injects both
//! persisted targets at the root level. ReceivingSplit or Merging roots,
//! missing authority, inconsistent levels or counts, duplicate references,
//! and duplicate candidate Record IDs are Corruption.
//!
//! Synopses prune only proven NoMatch leaves. Exact predicates admit entries;
//! RaBitQ rough distances feed a global bounded candidate heap.

use std::fmt;

use crate::api::{Error, ErrorKind, PartitionKey, Result, SearchBudgets};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::ReadOps;
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{IndexManifest, RecordLocation, expect_header, expect_synopsis};

use super::cache::graph::RoutingExecution;
use super::cache::{
    BODY_PREFETCH_BATCH, BodyEntries, BodyRequest, CachedBody, PartitionCache, load_body,
    prefetch_bodies,
};
use super::candidates::GlobalCandidatePool;
#[cfg(test)]
use super::candidates::LeafCandidate;
use super::numeric::VectorKernel;
use super::plan::EnumeratedTree;
use super::predicate::CompiledPredicate;
use super::rabitq::RaBitQQuery;
use crate::api::RoutingMode;

mod frontier;
mod partition;

use frontier::{Frontier, FrontierEntry, VisitBatch};
use partition::{PartitionReader, PreparedVisit};

/// The default leaf-level base beam (design `search.md` section 6).
pub(crate) const DEFAULT_LEAF_BEAM: u32 = 128;

/// Bounds metadata prefetch; adapters independently chunk point reads.
const PARTITION_METADATA_BATCH: usize = 1_024;

/// One bounded traversal request over the enumerated trees of one snapshot.
///
/// `routing` is the validated, metric-preprocessed, rotated query vector.
/// `trees` are the materialized eligible trees in canonical Tree Key order.
/// `budgets` bound visited partitions and the global candidate pool
/// through the exact-rerank budget. All entries in an admitted leaf are
/// considered; their count is observational. `leaf_beam` is
/// the leaf-level base beam; [`DEFAULT_LEAF_BEAM`] is the production default.
pub(crate) struct TraversalRequest<'a> {
    routing: &'a [f32],
    trees: &'a [EnumeratedTree],
    predicate: Option<&'a CompiledPredicate>,
    budgets: SearchBudgets,
    leaf_beam: u32,
    routing_mode: RoutingMode,
    #[cfg(feature = "benchmark")]
    diagnostics: Option<&'a std::sync::Mutex<crate::benchmark::SearchDiagnostics>>,
}

impl<'a> TraversalRequest<'a> {
    #[cfg(feature = "benchmark")]
    pub(crate) fn with_diagnostics(
        mut self,
        diagnostics: Option<&'a std::sync::Mutex<crate::benchmark::SearchDiagnostics>>,
    ) -> Self {
        self.diagnostics = diagnostics;
        self
    }

    /// Uses the already validated public routing choice.
    pub(crate) fn with_routing_mode(mut self, mode: RoutingMode) -> Self {
        self.routing_mode = mode;
        self
    }

    /// Creates one traversal request, rejecting a zero beam.
    pub(crate) fn new(
        routing: &'a [f32],
        trees: &'a [EnumeratedTree],
        predicate: Option<&'a CompiledPredicate>,
        budgets: SearchBudgets,
        leaf_beam: u32,
    ) -> Result<Self> {
        if leaf_beam == 0 {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            routing,
            trees,
            predicate,
            budgets,
            leaf_beam,
            routing_mode: RoutingMode::Exact,
            #[cfg(feature = "benchmark")]
            diagnostics: None,
        })
    }
}

/// The bounded, deterministic result of forest traversal.
///
/// Candidates are merged across every visited leaf for global overlap
/// selection, which owns their final rough-distance/Record-ID ordering.
/// Reports visited-partition and Leaf Entry usage together with partition
/// budget exhaustion.
pub(crate) struct TraversalOutcome {
    candidates: GlobalCandidatePool,
    /// Visited partitions worth offering to the Runtime's demand-driven Fixup
    /// queue: split or merge states and threshold-crossing `Ready` partitions.
    maintenance: Vec<(TreeKey, PartitionKey)>,
    visited_partitions: u32,
    visited_leaf_entries: u64,
    partition_budget_exhausted: bool,
}

impl TraversalOutcome {
    /// Returns the merged candidate pool before global overlap selection.
    #[cfg(test)]
    pub(crate) fn candidates(&self) -> &[LeafCandidate] {
        self.candidates.candidates()
    }

    /// Consumes the outcome and returns the merged candidates.
    #[must_use]
    pub(crate) fn into_candidates(self) -> GlobalCandidatePool {
        self.candidates
    }

    /// Takes the visited partitions worth offering to maintenance — split or
    /// merge states and threshold-crossing `Ready` partitions — leaving an
    /// empty list behind.
    pub(crate) fn take_maintenance(&mut self) -> Vec<(TreeKey, PartitionKey)> {
        std::mem::take(&mut self.maintenance)
    }

    /// Returns the distinct partition bodies logically visited.
    #[must_use]
    pub(crate) const fn visited_partitions(&self) -> u32 {
        self.visited_partitions
    }

    /// Returns the Leaf Entries read and considered under the exact predicate.
    #[must_use]
    pub(crate) const fn visited_leaf_entries(&self) -> u64 {
        self.visited_leaf_entries
    }

    /// Returns whether the depleted Partition budget prevented eligible work.
    #[must_use]
    pub(crate) const fn partition_budget_exhausted(&self) -> bool {
        self.partition_budget_exhausted
    }
}

impl fmt::Debug for TraversalOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Counts and flags are safe; candidate Record IDs stay redacted.
        formatter
            .debug_struct("TraversalOutcome")
            .field("candidates", &self.candidates.candidates().len())
            .field("visited_partitions", &self.visited_partitions)
            .field("visited_leaf_entries", &self.visited_leaf_entries)
            .field(
                "partition_budget_exhausted",
                &self.partition_budget_exhausted,
            )
            .finish()
    }
}

/// Traverses every requested tree under one consistent snapshot.
///
/// Returns the global candidate pool and the traversal-owned
/// budget accounting. Partition bodies come from the snapshot-validated
/// `cache`: a hit serves the decoded body without backend reads, and a miss
/// scans, validates, and publishes the body from `txn`'s snapshot. The result
/// is a deterministic function of the snapshot, the request, and the budgets,
/// independent of cache warmth.
pub(crate) async fn traverse<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    cache: &PartitionCache,
    kernel: &VectorKernel,
    request: TraversalRequest<'_>,
    execution: &impl RoutingExecution,
) -> Result<TraversalOutcome> {
    let manifest = txn.require_manifest()?;
    let query = RaBitQQuery::new(request.routing, manifest.config().metric())?;
    let filtered_exact = request.predicate.is_some() && request.routing_mode == RoutingMode::Exact;
    let input_bytes = if manifest.config().local_graph().is_some() || filtered_exact {
        request.trees.iter().try_fold(
            kernel
                .heap_bytes()
                .checked_add(request.routing.len() * 8)
                .and_then(|bytes| bytes.checked_add(query.heap_bytes()))
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?,
            |bytes, tree| {
                bytes
                    .checked_add(
                        2 * size_of::<EnumeratedTree>() + tree.tree_key().as_bytes().len() + 64,
                    )
                    .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))
            },
        )?
    } else {
        0
    };
    let context = VisitContext {
        input_bytes,
        execution,
        manifest,
        cache,
        kernel,
        query: &query,
        request,
    };
    let capacity = usize::try_from(context.request.budgets.exact_rerank_candidates())
        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
    let mut state = Traversal::seed(
        context.request.trees,
        capacity,
        manifest.config().local_graph().is_some() || filtered_exact,
    )?;
    if filtered_exact {
        state.frontier.enable_continuation(
            context.request.leaf_beam,
            context.request.trees.len(),
            context.request.budgets.visited_partitions(),
        )?;
        state
            .frontier
            .check_retained(&context, 0, state.candidates.heap_bytes()?, 0)?;
    }
    state.run(txn, &context).await?;
    state
        .finish(context.request.budgets.visited_partitions(), &|| {
            execution.checkpoint()
        })
        .await
}

/// The manifest-bound inputs shared by every partition visit.
struct VisitContext<'a, R> {
    input_bytes: usize,
    execution: &'a R,
    manifest: &'a IndexManifest,
    cache: &'a PartitionCache,
    kernel: &'a VectorKernel,
    query: &'a RaBitQQuery,
    request: TraversalRequest<'a>,
}

/// Coordinates snapshot reads, internal routing and complete leaf scans.
struct Traversal {
    frontier: Frontier,
    partitions: PartitionReader,
    candidates: GlobalCandidatePool,
    visited_leaf_entries: u64,
}

impl Traversal {
    /// Seeds all eligible roots without charging visits before metadata is read.
    fn seed(trees: &[EnumeratedTree], capacity: usize, account_memory: bool) -> Result<Self> {
        Ok(Self {
            frontier: Frontier::seed(trees)?,
            partitions: PartitionReader::default(),
            candidates: GlobalCandidatePool::new(capacity, account_memory),
            visited_leaf_entries: 0,
        })
    }

    /// Advances funded work until the forest drains or the partition budget stops it.
    async fn run<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'_, impl RoutingExecution>,
    ) -> Result<()> {
        while let Some(batch) = self.frontier.next_batch(
            context.request.budgets.visited_partitions() - self.partitions.visited_partitions(),
            context.manifest.config().local_graph().is_some(),
        ) {
            context.execution.checkpoint()?;
            // One small root may finish synchronously. Further visits must yield
            // even when each cached body is below the CPU batch size.
            if self.partitions.visited_partitions() > 0 {
                super::cooperate(&|| context.execution.checkpoint()).await?;
            }
            match batch {
                VisitBatch::Single(entry) => self.visit_partition(txn, context, entry).await?,
                VisitBatch::Prefetch(entries) => self.visit_beam(txn, context, &entries).await?,
            }
        }
        Ok(())
    }

    /// Batches a funded prefix without changing its visitation order.
    ///
    /// Non-root entries do not inject root-split targets into the
    /// active frontier. Every fetched Header will therefore be visited unless
    /// the whole operation fails. Roots retain demand-driven reads because a
    /// root transition can add targets ahead of the queued partitions.
    /// On Local Graph Indexes, stop before an internal partition so it retains
    /// serial graph validation, routing and working-resource admission.
    async fn visit_beam<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'_, impl RoutingExecution>,
        entries: &[FrontierEntry],
    ) -> Result<()> {
        let mut keys = Vec::with_capacity(entries.len() * 2);
        for entry in entries {
            let index = context.manifest.logical_index_id();
            let tree_key = context.request.trees[entry.tree as usize].tree_key();
            keys.push(LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: entry.partition,
            });
            if context.request.predicate.is_some() && entry.expected_level == Some(1) {
                keys.push(LogicalKey::Synopsis {
                    index,
                    tree_key: tree_key.clone(),
                    partition: entry.partition,
                });
            }
        }
        let mut values = txn.batch_get(keys).await?.into_iter();
        for entries in entries.chunks(BODY_PREFETCH_BATCH) {
            // Small cached bodies may each finish without yielding internally.
            // Keep a wide funded beam cooperative between prefetch groups too.
            super::cooperate(&|| context.execution.checkpoint()).await?;
            let mut visits = Vec::with_capacity(entries.len());
            for &entry in entries {
                let header = expect_header(values.next().flatten())?
                    .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
                let synopsis =
                    if context.request.predicate.is_some() && entry.expected_level == Some(1) {
                        Some(
                            expect_synopsis(values.next().flatten())?
                                .ok_or_else(|| Error::new(ErrorKind::Corruption))?,
                        )
                    } else {
                        None
                    };
                if let Some(visit) = self
                    .partitions
                    .prepare(
                        txn,
                        context,
                        &mut self.frontier,
                        entry,
                        Some((header, synopsis)),
                    )
                    .await?
                {
                    visits.push(visit);
                }
            }
            let requests: Vec<_> = visits
                .iter()
                .map(|visit| BodyRequest {
                    tree_key: context.request.trees[visit.entry.tree as usize].tree_key(),
                    partition: visit.entry.partition,
                    header: &visit.header,
                })
                .collect();
            let reads = prefetch_bodies(txn, context.cache, context.manifest, &requests).await?;
            for (visit, read) in visits.into_iter().zip(reads) {
                let body = read
                    .finish(txn, context.cache, context.manifest, &|| {
                        context.execution.checkpoint()
                    })
                    .await?;
                self.visit_body(context, &visit, &body).await?;
            }
        }
        Ok(())
    }

    /// Loads a single visit when roots or budget boundaries prevent batching.
    async fn visit_partition<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'_, impl RoutingExecution>,
        entry: FrontierEntry,
    ) -> Result<()> {
        let Some(visit) = self
            .partitions
            .prepare(txn, context, &mut self.frontier, entry, None)
            .await?
        else {
            return Ok(());
        };
        let tree_key = context.request.trees[entry.tree as usize].tree_key();
        if visit.header.level() > 1 && context.manifest.config().local_graph().is_some() {
            return self
                .frontier
                .visit_graph(
                    txn,
                    context,
                    visit.entry,
                    &visit.header,
                    self.candidates.heap_bytes()?,
                )
                .await;
        }
        let body = load_body(
            txn,
            context.cache,
            context.manifest,
            tree_key,
            entry.partition,
            &visit.header,
            &|| context.execution.checkpoint(),
        )
        .await?;
        self.visit_body(context, &visit, &body).await
    }

    /// Consumes complete bodies in the original deterministic traversal order.
    async fn visit_body(
        &mut self,
        context: &VisitContext<'_, impl RoutingExecution>,
        visit: &PreparedVisit<'_>,
        body: &CachedBody,
    ) -> Result<()> {
        if visit.header.level() > 1 {
            let candidate_bytes = if self.frontier.retains_branches() {
                self.candidates.heap_bytes()?
            } else {
                0
            };
            self.frontier
                .visit_internal(
                    context,
                    visit.entry.tree,
                    &visit.header,
                    body,
                    candidate_bytes,
                )
                .await
        } else {
            self.scan_leaf(
                context,
                visit.entry.tree,
                context.request.trees[visit.entry.tree as usize].tree_key(),
                visit.entry.partition,
                body,
                visit.predicate,
            )
            .await
        }
    }

    /// Considers every entry of an admitted Leaf Partition, filters exactly,
    /// and admits scores to the global bounded candidate heap. The snapshot-validated cache
    /// supplies the whole decoded body; scanning has no independent entry cap.
    async fn scan_leaf(
        &mut self,
        context: &VisitContext<'_, impl RoutingExecution>,
        tree: u32,
        tree_key: &TreeKey,
        partition: PartitionKey,
        body: &CachedBody,
        predicate: Option<&CompiledPredicate>,
    ) -> Result<()> {
        let BodyEntries::Leaf { entries, codes } = body.entries() else {
            return Err(Error::new(ErrorKind::Corruption));
        };
        #[cfg(feature = "benchmark")]
        if let Some(diagnostics) = context.request.diagnostics {
            let mut diagnostics = diagnostics
                .lock()
                .map_err(|_| Error::new(ErrorKind::Other))?;
            diagnostics.leaves.push(partition);
            diagnostics.neighbors_in_leaves += entries
                .iter()
                .filter(|entry| diagnostics.truth.contains(entry.record_id()))
                .count();
        }
        self.visited_leaf_entries = self
            .visited_leaf_entries
            .checked_add(entries.len() as u64)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        // Filtering retains only original ordinals; the pool owns all scoring.
        let eligible = if let Some(predicate) = predicate {
            let mut ordinals = Vec::new();
            for (ordinal, entry) in entries.iter().enumerate() {
                if ordinal > 0 && ordinal % 64 == 0 {
                    super::cooperate(&|| context.execution.checkpoint()).await?;
                }
                if predicate.matches(entry.fields())? {
                    ordinals.push(ordinal);
                }
            }
            Some(ordinals)
        } else {
            None
        };
        if eligible
            .as_ref()
            .map_or(!entries.is_empty(), |ordinals| !ordinals.is_empty())
        {
            // Count exact eligibility, not admission to the bounded rough pool.
            // A far leaf with matches still satisfies one unit of search effort.
            self.frontier.record_effective_leaf(tree);
        }
        if self.frontier.retains_branches() {
            let count = eligible.as_ref().map_or(entries.len(), Vec::len);
            self.frontier.check_retained(
                context,
                0,
                self.candidates.heap_bytes()?,
                self.candidates
                    .leaf_growth_bytes(count, body.heap_bytes())?,
            )?;
        }
        self.candidates
            .push_leaf(
                entries,
                codes,
                RecordLocation::new(tree_key.clone(), partition),
                context.query,
                eligible.as_deref(),
                &|| context.execution.checkpoint(),
            )
            .await?;
        Ok(())
    }

    /// Reports the bounded candidate pool after membership validation.
    async fn finish(
        mut self,
        partition_budget: u32,
        check: &(impl Fn() -> Result<()> + Sync),
    ) -> Result<TraversalOutcome> {
        self.candidates.validate_membership(check).await?;
        Ok(TraversalOutcome {
            candidates: self.candidates,
            visited_partitions: self.partitions.visited_partitions(),
            visited_leaf_entries: self.visited_leaf_entries,
            // A depleted budget is exhaustion only when eligible work remains.
            partition_budget_exhausted: self.partitions.visited_partitions() == partition_budget
                && self.frontier.has_pending(),
            maintenance: self.partitions.into_maintenance(),
        })
    }
}

#[cfg(test)]
mod tests;
