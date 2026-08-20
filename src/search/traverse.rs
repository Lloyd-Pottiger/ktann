//! Deterministic bounded best-first traversal across Tree Key-selected trees.
//!
//! This module owns the search pipeline's traversal stage (design
//! `search.md` step 3 and the per-leaf half of steps 4 and 5). Given the Tree
//! Keys materialized by bounded directory enumeration, it advances every
//! eligible tree fairly through one global best-first frontier, expands
//! internal partitions through their Child Entries, prunes Leaf Partitions
//! through their conservative synopses, and reduces each visited leaf to its
//! bounded RaBitQ overlap candidates. Partition bodies are loaded through the
//! snapshot-validated Partition Cache (ADR 0010); cache warmth never changes
//! the logical budget accounting below. Global overlap selection, exact Vector
//! Record loading and reranking, and Search Outcome assembly stay with the
//! rerank stage and the public search operation (#30).
//!
//! # Contract
//!
//! - **Deterministic best-first order.** The frontier is one priority queue
//!   ordered by `(routing distance, Tree Key enumeration order, Partition
//!   Key)`. Every tree's root is queued up front, so identical snapshots,
//!   requests, and budgets pop identical visit sequences and no tree is
//!   starved by enumeration order alone.
//! - **Level-scaled beam.** Per tree and level, at most `beam(level)`
//!   Child Entry-referenced partitions enter the frontier: the leaf-level
//!   base beam defaults to [`DEFAULT_LEAF_BEAM`] and halves toward the root
//!   with a minimum of one. Since internal fanout is exactly two, the halving
//!   funds full best-first descent down to the leaf beam and prunes only
//!   transient-fanout or plateau-level surplus. Root-split target injection
//!   is topology-mandated membership coverage, not beam admission, and is
//!   never pruned. Beam pruning is not budget exhaustion and is not reported
//!   as one.
//! - **Bounded work, charged before it starts.** Each distinct
//!   `{Tree Key, Partition Key}` body is visited and charged to the Partition
//!   budget at most once. Decoded bodies arrive whole from the
//!   snapshot-validated cache, but Leaf Entries are charged and considered
//!   only while the Leaf Entry budget funds them, in canonical body order, so
//!   cache warmth never changes the logical accounting. A budget dimension is
//!   reported exhausted only when eligible pending work was actually prevented
//!   by its depletion — never merely because natural completion landed
//!   exactly on the limit (ADR 0011).
//! - **Intermediate topology.** Non-root partitions in every committed state
//!   are reached only through their current Child Entries and searched as
//!   ordinary same-level bodies. A Splitting root exposes no targets, so only
//!   its body is searched; a DrainingSplit root additionally injects both
//!   persisted targets into its own level's frontier, each consuming
//!   Partition budget. A root claiming ReceivingSplit or Merging is
//!   Corruption.
//! - **Fail closed.** A missing Header, Synopsis, or referenced State, a
//!   wrong-kind value, a level that fails to descend exactly one level per
//!   hop, a body whose decoded entries disagree with the Header's exact
//!   count, a second incoming reference to one partition, or a duplicate
//!   Record ID among the admitted candidates is Corruption.
//! - **Approximate bounds are never exact filters.** Synopses prune a leaf
//!   only when they prove `NoMatch`; exact predicate evaluation admits
//!   entries; RaBitQ intervals only order candidates and drive the bounded
//!   conservative overlap selection.
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;

use crate::api::{Error, ErrorKind, PartitionKey, Result, SearchBudgets};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::ReadOps;
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    IndexManifest, PartitionHeader, PartitionState, PartitionSynopsis, PartitionTransition,
    PersistentValue, RecordLocation,
};

use super::cache::{BodyEntries, PartitionCache, load_body};
use super::numeric::{VectorKernel, compare_finite};
use super::plan::EnumeratedTree;
use super::predicate::{CompiledPredicate, SynopsisClassification};
use super::rabitq::{ApproximateCandidate, RaBitQ7, RaBitQQuery, select_leaf_overlap};
use super::rerank::{LeafCandidate, filter_candidates};

/// The default leaf-level base beam (design `search.md` section 6).
pub(crate) const DEFAULT_LEAF_BEAM: u32 = 32;

/// One bounded traversal request over the enumerated trees of one snapshot.
///
/// `routing` is the validated, metric-preprocessed, rotated query vector.
/// `trees` are the materialized eligible trees in canonical Tree Key order.
/// `budgets` bound visited partitions and Leaf Entries; the exact-rerank
/// budget additionally caps every per-leaf overlap selection. `leaf_beam` is
/// the leaf-level base beam; [`DEFAULT_LEAF_BEAM`] is the production default.
pub(crate) struct TraversalRequest<'a> {
    routing: &'a [f32],
    trees: &'a [EnumeratedTree],
    predicate: Option<&'a CompiledPredicate>,
    k: usize,
    budgets: SearchBudgets,
    leaf_beam: u32,
}

impl<'a> TraversalRequest<'a> {
    /// Creates one traversal request, rejecting a zero result limit or beam.
    pub(crate) fn new(
        routing: &'a [f32],
        trees: &'a [EnumeratedTree],
        predicate: Option<&'a CompiledPredicate>,
        k: usize,
        budgets: SearchBudgets,
        leaf_beam: u32,
    ) -> Result<Self> {
        if k == 0 || leaf_beam == 0 {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            routing,
            trees,
            predicate,
            k,
            budgets,
            leaf_beam,
        })
    }
}

/// The bounded, deterministic result of forest traversal.
///
/// Candidates are merged across every visited leaf and ordered by rough
/// distance then unsigned lexicographic Record ID bytes, ready for global
/// overlap selection and exact reranking. The usage counters and exhaustion
/// flags fold into the Search Outcome's `visited_partitions` and
/// `visited_leaf_entries` dimensions.
pub(crate) struct TraversalOutcome {
    candidates: Vec<LeafCandidate>,
    visited_partitions: u32,
    visited_leaf_entries: u32,
    partition_budget_exhausted: bool,
    leaf_entry_budget_exhausted: bool,
    rabitq_overlap_truncated: bool,
}

impl TraversalOutcome {
    /// Returns the merged candidates in rough-distance/Record-ID order.
    #[cfg(test)]
    pub(crate) fn candidates(&self) -> &[LeafCandidate] {
        &self.candidates
    }

    /// Consumes the outcome and returns the merged candidates.
    #[must_use]
    pub(crate) fn into_candidates(self) -> Vec<LeafCandidate> {
        self.candidates
    }

    /// Returns the distinct partition bodies logically visited.
    #[must_use]
    pub(crate) const fn visited_partitions(&self) -> u32 {
        self.visited_partitions
    }

    /// Returns the Leaf Entries read and considered under the exact predicate.
    #[must_use]
    pub(crate) const fn visited_leaf_entries(&self) -> u32 {
        self.visited_leaf_entries
    }

    /// Returns whether the depleted Partition budget prevented eligible work.
    #[must_use]
    pub(crate) const fn partition_budget_exhausted(&self) -> bool {
        self.partition_budget_exhausted
    }

    /// Returns whether the depleted Leaf Entry budget prevented eligible work.
    #[must_use]
    pub(crate) const fn leaf_entry_budget_exhausted(&self) -> bool {
        self.leaf_entry_budget_exhausted
    }

    /// Returns whether any per-leaf overlap cap discarded a qualifying overlap.
    #[must_use]
    pub(crate) const fn rabitq_overlap_truncated(&self) -> bool {
        self.rabitq_overlap_truncated
    }
}

impl fmt::Debug for TraversalOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Counts and flags are safe; candidate Record IDs stay redacted.
        formatter
            .debug_struct("TraversalOutcome")
            .field("candidates", &self.candidates.len())
            .field("visited_partitions", &self.visited_partitions)
            .field("visited_leaf_entries", &self.visited_leaf_entries)
            .field(
                "partition_budget_exhausted",
                &self.partition_budget_exhausted,
            )
            .field(
                "leaf_entry_budget_exhausted",
                &self.leaf_entry_budget_exhausted,
            )
            .field("rabitq_overlap_truncated", &self.rabitq_overlap_truncated)
            .finish()
    }
}

/// Traverses every requested tree under one consistent snapshot.
///
/// Returns the merged per-leaf candidate selections and the traversal-owned
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
) -> Result<TraversalOutcome> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let query = RaBitQQuery::new(request.routing, manifest.config().metric())?;
    let context = VisitContext {
        manifest,
        cache,
        kernel,
        query: &query,
        rerank_cap: usize::try_from(request.budgets.exact_rerank_candidates())
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?,
        request,
    };
    let mut state = Traversal::seed(context.request.trees)?;
    state.run(txn, &context).await?;
    state.finish()
}

/// The manifest-bound inputs shared by every partition visit.
struct VisitContext<'a> {
    manifest: &'a IndexManifest,
    cache: &'a PartitionCache,
    kernel: &'a VectorKernel,
    query: &'a RaBitQQuery<'a>,
    /// The exact-rerank budget converted for the per-leaf overlap caps.
    rerank_cap: usize,
    request: TraversalRequest<'a>,
}

/// One queued frontier partition.
///
/// The best-first order key is `(routing distance, Tree Key enumeration
/// order, Partition Key)`; roots and injected root-split targets carry
/// distance zero. `expected_level` is the level the referencing edge
/// established — `None` for a tree root — and the visited Header must agree.
#[derive(Clone, Copy, Debug)]
struct FrontierEntry {
    distance: f64,
    tree: u32,
    partition: PartitionKey,
    expected_level: Option<u32>,
}

impl Ord for FrontierEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Every queued distance is validated finite, so IEEE comparison with
        // the -0.0/+0.0 tie falling through to the stable keys is total.
        compare_finite(self.distance, other.distance)
            .then_with(|| self.tree.cmp(&other.tree))
            .then_with(|| self.partition.cmp(&other.partition))
    }
}

impl PartialOrd for FrontierEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for FrontierEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for FrontierEntry {}

/// The mutable state of one forest traversal.
struct Traversal {
    frontier: BinaryHeap<Reverse<FrontierEntry>>,
    /// Every referenced `{tree, partition}`: roots, injected root-split
    /// targets, and admitted Child Entries. A second reference to one body is
    /// Corruption.
    referenced: HashSet<(u32, PartitionKey)>,
    /// Beam admissions per `{tree, level}`.
    admitted: HashMap<(u32, u32), u32>,
    candidates: Vec<LeafCandidate>,
    visited_partitions: u32,
    visited_leaf_entries: u32,
    partition_budget_exhausted: bool,
    leaf_entry_budget_exhausted: bool,
    rabitq_overlap_truncated: bool,
}

impl Traversal {
    /// Seeds the frontier with every eligible tree's root.
    fn seed(trees: &[EnumeratedTree]) -> Result<Self> {
        let mut state = Self {
            frontier: BinaryHeap::new(),
            referenced: HashSet::new(),
            admitted: HashMap::new(),
            candidates: Vec::new(),
            visited_partitions: 0,
            visited_leaf_entries: 0,
            partition_budget_exhausted: false,
            leaf_entry_budget_exhausted: false,
            rabitq_overlap_truncated: false,
        };
        for (ordinal, tree) in trees.iter().enumerate() {
            let ordinal =
                u32::try_from(ordinal).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            let root = tree.manifest().root();
            // Tree ordinals are unique within one enumeration, so the root
            // reference is always new.
            state.referenced.insert((ordinal, root));
            state.frontier.push(Reverse(FrontierEntry {
                distance: 0.0,
                tree: ordinal,
                partition: root,
                expected_level: None,
            }));
        }
        Ok(state)
    }

    /// Advances the best-first frontier until it drains or a budget stops it.
    async fn run<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'_>,
    ) -> Result<()> {
        while let Some(&Reverse(entry)) = self.frontier.peek() {
            // Stop before unfunded work: the peeked entry is beam-admitted or
            // topology-mandated, so it is eligible work the depleted Partition
            // budget would prevent.
            if self.visited_partitions == context.request.budgets.visited_partitions() {
                break;
            }
            self.frontier.pop();
            self.visit_partition(txn, context, entry).await?;
            // A depleted Leaf Entry budget stops the whole traversal: every
            // remaining leaf visit consumes it, and expanding internal nodes
            // could only queue more unfundable leaf work.
            if self.leaf_entry_budget_exhausted {
                break;
            }
        }
        // Every still-queued entry is eligible work, prevented exactly when
        // the Partition budget is fully spent; several dimensions may be
        // exhausted together, while natural completion on the limit (an empty
        // frontier) is not exhaustion.
        self.partition_budget_exhausted = !self.frontier.is_empty()
            && self.visited_partitions == context.request.budgets.visited_partitions();
        Ok(())
    }

    /// Visits one queued partition body and accounts for it.
    async fn visit_partition<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'_>,
        entry: FrontierEntry,
    ) -> Result<()> {
        self.visited_partitions = self
            .visited_partitions
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let index = context.manifest.logical_index_id();
        let tree_key = context.request.trees[entry.tree as usize].tree_key();

        // Leaf children dominate the frontier; batch the exact Synopsis read
        // with the Header when the referencing edge already proves a leaf.
        let header_key = LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: entry.partition,
        };
        let (header, synopsis) =
            if context.request.predicate.is_some() && entry.expected_level == Some(1) {
                let mut values = txn
                    .batch_get(vec![
                        header_key,
                        LogicalKey::Synopsis {
                            index,
                            tree_key: tree_key.clone(),
                            partition: entry.partition,
                        },
                    ])
                    .await?;
                let synopsis = expect_synopsis(values.pop().flatten())?;
                let header = expect_header(values.pop().flatten())?;
                (header, Some(synopsis))
            } else {
                (expect_header(txn.get(header_key).await?)?, None)
            };

        // Every Child Entry descends exactly one level, so a referenced
        // partition's Header must agree with the level its edge established;
        // the same check rejects cycles before they can repeat a partition.
        if let Some(expected) = entry.expected_level {
            if header.level() != expected {
                return Err(Error::new(ErrorKind::Corruption));
            }
        }
        let level = header.level();

        if entry.expected_level.is_none() {
            self.apply_root_state(txn, index, tree_key, entry.tree, entry.partition, &header)
                .await?;
        }

        if level > 1 {
            self.visit_internal(txn, context, tree_key, entry.tree, entry.partition, level)
                .await
        } else {
            let effective_predicate = match context.request.predicate {
                Some(predicate) => {
                    let synopsis = match synopsis {
                        Some(synopsis) => synopsis,
                        None => expect_synopsis(
                            txn.get(LogicalKey::Synopsis {
                                index,
                                tree_key: tree_key.clone(),
                                partition: entry.partition,
                            })
                            .await?,
                        )?,
                    };
                    match predicate.classify(context.manifest, &synopsis, header.entry_count())? {
                        // The synopsis proves no entry can satisfy the
                        // predicate: prune the leaf without charging entries.
                        SynopsisClassification::NoMatch => return Ok(()),
                        // Every entry provably matches: skip per-entry
                        // evaluation but still charge every entry read.
                        SynopsisClassification::AllMatch => None,
                        SynopsisClassification::MayMatch => Some(predicate),
                    }
                }
                None => None,
            };
            // The exact Header count authoritatively proves emptiness.
            if header.entry_count() > 0 {
                self.scan_leaf(txn, context, tree_key, entry.partition, effective_predicate)
                    .await?;
            }
            Ok(())
        }
    }

    /// Applies the root-only intermediate topology rules.
    ///
    /// A Splitting root's targets are unexposed, so only the root body is
    /// searched. A DrainingSplit root's targets have no incoming Child Entry
    /// yet, so both persisted targets join the root's own level's frontier
    /// explicitly. A root can never be a split target or merge source.
    async fn apply_root_state<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        index: crate::api::LogicalIndexId,
        tree_key: &TreeKey,
        tree: u32,
        partition: PartitionKey,
        header: &PartitionHeader,
    ) -> Result<()> {
        match header.state() {
            PartitionState::Ready | PartitionState::Splitting => Ok(()),
            PartitionState::DrainingSplit => {
                let transition = txn
                    .get(LogicalKey::State {
                        index,
                        tree_key: tree_key.clone(),
                        partition,
                    })
                    .await?;
                let (left, right) =
                    match transition {
                        Some(PersistentValue::PartitionState(
                            PartitionTransition::DrainingSplit { left, right, .. },
                        )) => (left, right),
                        _ => return Err(Error::new(ErrorKind::Corruption)),
                    };
                self.inject(tree, left, header.level())?;
                self.inject(tree, right, header.level())?;
                Ok(())
            }
            PartitionState::ReceivingSplit | PartitionState::Merging => {
                Err(Error::new(ErrorKind::Corruption))
            }
        }
    }

    /// Expands one internal partition's Child Entries into the frontier.
    ///
    /// The decoded body comes from the snapshot-validated cache. Children
    /// enter the frontier best-first within the parent. Per tree and level at
    /// most `beam(level)` children are admitted; surplus children — reachable
    /// only through transient split fanout or the minimum-one plateau — are
    /// pruned deterministically, never reported as exhaustion.
    async fn visit_internal<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'_>,
        tree_key: &TreeKey,
        tree: u32,
        partition: PartitionKey,
        level: u32,
    ) -> Result<()> {
        let body = load_body(txn, context.cache, context.manifest, tree_key, partition).await?;
        let BodyEntries::Internal(entries) = body.entries() else {
            return Err(Error::new(ErrorKind::Corruption));
        };
        let mut children: Vec<(f64, PartitionKey)> = Vec::with_capacity(entries.len());
        for child in entries {
            let distance = context
                .kernel
                .routing_distance(context.request.routing, child.centroid())?;
            // Every non-root partition has exactly one incoming Child
            // Entry; a second reference is Corruption, not a duplicate to
            // deduplicate.
            if !self.referenced.insert((tree, child.child())) {
                return Err(Error::new(ErrorKind::Corruption));
            }
            children.push((distance, child.child()));
        }
        children.sort_unstable_by(|left, right| {
            compare_finite(left.0, right.0).then_with(|| left.1.cmp(&right.1))
        });
        let child_level = level - 1;
        for (distance, child) in children {
            if self.admit(tree, child_level, context.request.leaf_beam) {
                self.frontier.push(Reverse(FrontierEntry {
                    distance,
                    tree,
                    partition: child,
                    expected_level: Some(child_level),
                }));
            }
        }
        Ok(())
    }

    /// Considers one Leaf Partition's entries under the Leaf Entry budget,
    /// filters them exactly, and merges the bounded overlap selection.
    ///
    /// The caller guarantees the leaf is non-empty and not synopsis-pruned, so
    /// a budget that funds no entry is provably pending work: the leaf is
    /// reported exhausted without spending a body load. Otherwise the decoded
    /// body arrives whole from the snapshot-validated cache and entries are
    /// charged and considered in canonical body order only while the remaining
    /// Leaf Entry budget funds them — a depleted budget with unconsidered
    /// entries is exhaustion and stops the whole traversal, regardless of
    /// cache warmth.
    async fn scan_leaf<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'_>,
        tree_key: &TreeKey,
        partition: PartitionKey,
        predicate: Option<&CompiledPredicate>,
    ) -> Result<()> {
        let remaining = usize::try_from(
            context
                .request
                .budgets
                .visited_leaf_entries()
                .checked_sub(self.visited_leaf_entries)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?,
        )
        .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        if remaining == 0 {
            self.leaf_entry_budget_exhausted = true;
            return Ok(());
        }
        let body = load_body(txn, context.cache, context.manifest, tree_key, partition).await?;
        let BodyEntries::Leaf(entries) = body.entries() else {
            return Err(Error::new(ErrorKind::Corruption));
        };
        let dimension = context.manifest.config().dimension();
        let funded = entries.len().min(remaining);
        // Unconsidered entries are eligible work the depleted budget prevents.
        // The already-funded entries below stay materialized and still flow
        // through selection.
        if funded < entries.len() {
            self.leaf_entry_budget_exhausted = true;
        }
        let mut batch = Vec::with_capacity(funded);
        for entry in &entries[..funded] {
            let code = RaBitQ7::decode(entry.rabitq7(), dimension)?;
            // The query and the decoded code are validated finite, so a
            // non-conservative distance here means corrupted state.
            let distance = code
                .approximate_distance(context.query)
                .map_err(|_| Error::new(ErrorKind::Corruption))?;
            batch.push(LeafCandidate::new(
                entry.record_id().clone(),
                entry.fields().into(),
                distance,
                RecordLocation::new(tree_key.clone(), partition),
            ));
        }
        let filtered = filter_candidates(batch, predicate, &mut self.visited_leaf_entries)?;
        let pool: Vec<ApproximateCandidate<LeafCandidate>> = filtered
            .into_iter()
            .map(ApproximateCandidate::from)
            .collect();
        let selection = select_leaf_overlap(pool, context.request.k, context.rerank_cap)?;
        if selection.truncated() {
            self.rabitq_overlap_truncated = true;
        }
        self.candidates.extend(selection.into_values());
        Ok(())
    }

    /// Admits one Child Entry-referenced partition to the level-scaled beam.
    fn admit(&mut self, tree: u32, level: u32, leaf_beam: u32) -> bool {
        let width = beam_width(leaf_beam, level);
        let admitted = self.admitted.entry((tree, level)).or_insert(0);
        if *admitted >= width {
            return false;
        }
        *admitted += 1;
        true
    }

    /// Injects a root-split target into the frontier, bypassing the beam.
    fn inject(&mut self, tree: u32, partition: PartitionKey, level: u32) -> Result<()> {
        // A split target is exclusively referenced by the root transition
        // state before exposure; any second reference is Corruption.
        if !self.referenced.insert((tree, partition)) {
            return Err(Error::new(ErrorKind::Corruption));
        }
        self.frontier.push(Reverse(FrontierEntry {
            distance: 0.0,
            tree,
            partition,
            expected_level: Some(level),
        }));
        Ok(())
    }

    /// Deduplicates defensively, orders the merged candidates, and reports.
    fn finish(mut self) -> Result<TraversalOutcome> {
        // Exact membership admits at most one Leaf Entry per Vector Record in
        // one snapshot, even across draining split bodies; a duplicate Record
        // ID is Corruption rather than silently deduplicated.
        let mut seen = HashSet::new();
        for candidate in &self.candidates {
            if !seen.insert(candidate.record_id().as_ref()) {
                return Err(Error::new(ErrorKind::Corruption));
            }
        }
        self.candidates.sort_unstable_by(|left, right| {
            compare_finite(left.distance().rough(), right.distance().rough())
                .then_with(|| left.record_id().cmp(right.record_id()))
        });
        Ok(TraversalOutcome {
            candidates: self.candidates,
            visited_partitions: self.visited_partitions,
            visited_leaf_entries: self.visited_leaf_entries,
            partition_budget_exhausted: self.partition_budget_exhausted,
            leaf_entry_budget_exhausted: self.leaf_entry_budget_exhausted,
            rabitq_overlap_truncated: self.rabitq_overlap_truncated,
        })
    }
}

/// The level-scaled beam width: `leaf_beam` at the leaf level, halved per
/// level toward the root, with a minimum of one.
fn beam_width(leaf_beam: u32, level: u32) -> u32 {
    leaf_beam
        .checked_shr(level.saturating_sub(1))
        .unwrap_or(0)
        .max(1)
}

/// Extracts a partition Header from a typed read, failing closed.
fn expect_header(value: Option<PersistentValue>) -> Result<PartitionHeader> {
    match value {
        Some(PersistentValue::PartitionHeader(header)) => Ok(header),
        _ => Err(Error::new(ErrorKind::Corruption)),
    }
}

/// Extracts a partition Synopsis from a typed read, failing closed.
fn expect_synopsis(value: Option<PersistentValue>) -> Result<PartitionSynopsis> {
    match value {
        Some(PersistentValue::PartitionSynopsis(synopsis)) => Ok(synopsis),
        _ => Err(Error::new(ErrorKind::Corruption)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ops::Bound;

    use bytes::Bytes;

    use crate::api::{
        CompareOp, DataType, Error, ErrorKind, FieldId, FieldSchema, IndexConfig, LogicalIndexId,
        Metric, PartitionKey, Predicate, Result, SearchBudgets, Value,
    };
    use crate::storage::ReadLogicalTxn;
    use crate::storage::backend::{ReadOps, ScanItem, ScanLimits, ScanPage};
    use crate::storage::keys::{self, KeyRange, TreeKey};
    use crate::storage::values::{
        BloomParameters, ChildEntry, IndexLifecycle, IndexManifest, LeafEntry, PartitionHeader,
        PartitionState, PartitionSynopsis, PartitionTransition, PersistentValue, TreeManifest,
        ValueCodec,
    };

    use super::super::cache::PartitionCache;
    use super::super::numeric::VectorKernel;
    use super::super::plan::EnumeratedTree;
    use super::super::predicate::CompiledPredicate;
    use super::super::rabitq::RaBitQ7;
    use super::{DEFAULT_LEAF_BEAM, TraversalOutcome, TraversalRequest, beam_width, traverse};

    const DIMENSION: usize = 2;
    const SEED: [u8; 32] = [9; 32];
    /// The test query: at the L2 origin a stored vector's rough distance is
    /// approximately its squared norm, so well-separated norms give a known
    /// deterministic candidate order.
    const QUERY: [f32; DIMENSION] = [0.0, 0.0];

    fn id(value: u64) -> LogicalIndexId {
        LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
    }

    fn pk(value: u64) -> PartitionKey {
        PartitionKey::new(value).expect("test Partition Key is nonzero")
    }

    fn manifest() -> IndexManifest {
        let fields = vec![
            FieldSchema::new("k", DataType::I64).expect("valid field"),
            FieldSchema::new("x", DataType::I64).expect("valid field"),
        ];
        let bloom = fields
            .iter()
            .map(|field| BloomParameters::derive(field.synopsis()).expect("valid synopsis"))
            .collect();
        let config = IndexConfig::new(DIMENSION, Metric::L2)
            .expect("valid config")
            .with_fields(fields)
            .expect("valid fields")
            .with_tree_key_fields(vec![FieldId(0)])
            .expect("valid tree key fields");
        IndexManifest::new(IndexLifecycle::Active, id(1), config, SEED, bloom)
            .expect("valid manifest")
    }

    fn tree_key(value: i64) -> TreeKey {
        TreeKey::encode(&[DataType::I64], &[Value::I64(value)]).expect("canonical tree key")
    }

    /// The Tree Manifest every fixture tree uses: stable root pk1 with ample
    /// reserved Partition Keys.
    fn test_tree_manifest() -> TreeManifest {
        TreeManifest::new(pk(1), pk(1_000)).expect("valid tree manifest")
    }

    fn tree_ref(key: &TreeKey) -> EnumeratedTree {
        EnumeratedTree::new(key.clone(), test_tree_manifest())
    }

    fn budgets(partitions: u32, leaf_entries: u32, rerank: u32) -> SearchBudgets {
        SearchBudgets::new(1_024, partitions, leaf_entries, rerank).expect("valid budgets")
    }

    fn encode(manifest: &IndexManifest, value: &PersistentValue) -> Vec<u8> {
        ValueCodec::for_index(manifest)
            .encode(value)
            .expect("encode value")
    }

    /// Accumulates encoded fixture state for one Logical Index.
    struct Fixture<'a> {
        manifest: &'a IndexManifest,
        items: Vec<(Vec<u8>, Vec<u8>)>,
    }

    impl<'a> Fixture<'a> {
        fn new(manifest: &'a IndexManifest) -> Self {
            Self {
                manifest,
                items: Vec::new(),
            }
        }

        fn push(&mut self, key: Vec<u8>, value: PersistentValue) {
            self.items.push((key, encode(self.manifest, &value)));
        }

        /// Installs the Tree Manifest directory entry and returns the key.
        fn tree(&mut self, value: i64) -> TreeKey {
            let key = tree_key(value);
            self.push(
                keys::tree_manifest_key(self.manifest.logical_index_id(), &key),
                PersistentValue::TreeManifest(test_tree_manifest()),
            );
            key
        }

        fn header(
            &mut self,
            tree: &TreeKey,
            partition: u64,
            level: u32,
            count: u32,
            state: PartitionState,
        ) {
            self.push(
                keys::header_key(self.manifest.logical_index_id(), tree, pk(partition)),
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(level, count, 0, state).expect("valid header"),
                ),
            );
        }

        fn state(&mut self, tree: &TreeKey, partition: u64, transition: PartitionTransition) {
            self.push(
                keys::state_key(self.manifest.logical_index_id(), tree, pk(partition)),
                PersistentValue::PartitionState(transition),
            );
        }

        /// Installs the exact incremental Synopsis for rows `[tree_value, x]`.
        fn synopsis(&mut self, tree: &TreeKey, tree_value: i64, partition: u64, xs: &[i64]) {
            let mut synopsis = PartitionSynopsis::empty(self.manifest);
            for &x in xs {
                synopsis
                    .expand(self.manifest, &[Value::I64(tree_value), Value::I64(x)])
                    .expect("expand synopsis");
            }
            self.push(
                keys::synopsis_key(self.manifest.logical_index_id(), tree, pk(partition)),
                PersistentValue::PartitionSynopsis(synopsis),
            );
        }

        fn entry(
            &mut self,
            tree: &TreeKey,
            tree_value: i64,
            partition: u64,
            record_id: &str,
            x: i64,
            vector: [f32; DIMENSION],
        ) {
            let record_id = Bytes::copy_from_slice(record_id.as_bytes());
            let code = RaBitQ7::quantize(&vector).expect("quantize");
            let entry = LeafEntry::new(
                record_id.clone(),
                vec![Value::I64(tree_value), Value::I64(x)],
                code,
            );
            self.push(
                keys::leaf_entry_key(
                    self.manifest.logical_index_id(),
                    tree,
                    pk(partition),
                    &record_id,
                )
                .expect("leaf entry key"),
                PersistentValue::LeafEntry(entry),
            );
        }

        fn child(&mut self, tree: &TreeKey, parent: u64, child: u64, centroid: [f32; DIMENSION]) {
            self.push(
                keys::child_entry_key(
                    self.manifest.logical_index_id(),
                    tree,
                    pk(parent),
                    pk(child),
                ),
                PersistentValue::ChildEntry(ChildEntry::new(pk(child), centroid)),
            );
        }

        fn raw_entry(&mut self, tree: &TreeKey, partition: u64, record_id: &str, value: Vec<u8>) {
            let record_id = Bytes::copy_from_slice(record_id.as_bytes());
            self.items.push((
                keys::leaf_entry_key(
                    self.manifest.logical_index_id(),
                    tree,
                    pk(partition),
                    &record_id,
                )
                .expect("leaf entry key"),
                value,
            ));
        }

        fn raw_child(&mut self, tree: &TreeKey, parent: u64, child: u64, value: Vec<u8>) {
            self.items.push((
                keys::child_entry_key(
                    self.manifest.logical_index_id(),
                    tree,
                    pk(parent),
                    pk(child),
                ),
                value,
            ));
        }
    }

    /// A snapshot read mock over committed key-value bytes.
    struct MockReadTxn {
        data: BTreeMap<Vec<u8>, Vec<u8>>,
    }

    impl MockReadTxn {
        fn new(items: Vec<(Vec<u8>, Vec<u8>)>) -> Self {
            Self {
                data: items.into_iter().collect(),
            }
        }
    }

    impl ReadOps for MockReadTxn {
        async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
            Ok(self.data.get(key.as_ref()).cloned().map(Bytes::from))
        }

        async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
            Ok(keys
                .iter()
                .map(|key| self.data.get(key.as_ref()).cloned().map(Bytes::from))
                .collect())
        }

        async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
            if limits.item_limit == 0 || limits.byte_limit == 0 {
                return Err(Error::invalid_argument());
            }
            let mut iter = self
                .data
                .range::<[u8], _>((Bound::Included(range.start()), Bound::Excluded(range.end())))
                .peekable();
            let mut items = Vec::new();
            let mut bytes = 0_usize;
            while let Some((key, value)) = iter.peek() {
                let size = key.len() + value.len();
                if items.is_empty() && size > limits.byte_limit {
                    let (key, value) = iter.next().expect("peeked item exists");
                    items.push(ScanItem::new(
                        Bytes::copy_from_slice(key),
                        Bytes::copy_from_slice(value),
                    ));
                    break;
                }
                if items.len() >= limits.item_limit || bytes + size > limits.byte_limit {
                    break;
                }
                let (key, value) = iter.next().expect("peeked item exists");
                items.push(ScanItem::new(
                    Bytes::copy_from_slice(key),
                    Bytes::copy_from_slice(value),
                ));
                bytes += size;
            }
            if iter.peek().is_some() {
                ScanPage::continued(items, keys::MAX_TREE_KEY_BYTES + 64)
            } else {
                Ok(ScanPage::terminal(items))
            }
        }
    }

    async fn run(
        items: Vec<(Vec<u8>, Vec<u8>)>,
        manifest: &IndexManifest,
        trees: &[EnumeratedTree],
        predicate: Option<Predicate>,
        k: usize,
        budgets: SearchBudgets,
        beam: u32,
    ) -> Result<TraversalOutcome> {
        let mut txn =
            ReadLogicalTxn::for_index(MockReadTxn::new(items), manifest).expect("bind manifest");
        let kernel = VectorKernel::new(DIMENSION, Metric::L2, SEED).expect("valid kernel");
        let compiled = predicate
            .map(|predicate| CompiledPredicate::compile(predicate, manifest.config().fields()))
            .transpose()
            .expect("compile predicate");
        let cache = PartitionCache::new(1 << 20);
        traverse(
            &mut txn,
            &cache,
            &kernel,
            TraversalRequest::new(&QUERY, trees, compiled.as_ref(), k, budgets, beam)
                .expect("valid request"),
        )
        .await
    }

    fn candidate_ids(outcome: &TraversalOutcome) -> Vec<&[u8]> {
        outcome
            .candidates()
            .iter()
            .map(|candidate| candidate.record_id().as_ref())
            .collect()
    }

    fn assert_corruption(result: Result<TraversalOutcome>) {
        assert_eq!(
            result.err().map(|error| error.kind()),
            Some(ErrorKind::Corruption)
        );
    }

    /// Runs the default corruption traversal: no predicate, k = 8, generous
    /// budgets, and the default beam.
    async fn assert_corrupt(
        items: Vec<(Vec<u8>, Vec<u8>)>,
        manifest: &IndexManifest,
        trees: &[EnumeratedTree],
    ) {
        assert_corruption(
            run(
                items,
                manifest,
                trees,
                None,
                8,
                budgets(16, 16, 64),
                DEFAULT_LEAF_BEAM,
            )
            .await,
        );
    }

    #[test]
    fn beam_width_halves_toward_the_root_with_minimum_one() {
        assert_eq!(beam_width(32, 1), 32);
        assert_eq!(beam_width(32, 2), 16);
        assert_eq!(beam_width(32, 3), 8);
        assert_eq!(beam_width(32, 4), 4);
        assert_eq!(beam_width(32, 5), 2);
        assert_eq!(beam_width(32, 6), 1);
        assert_eq!(beam_width(32, 40), 1);
        assert_eq!(beam_width(1, 1), 1);
        assert_eq!(beam_width(3, 2), 1);
    }

    #[test]
    fn request_rejects_zero_k_or_beam() {
        let trees = vec![];
        let budget = budgets(8, 8, 8);
        assert!(TraversalRequest::new(&QUERY, &trees, None, 0, budget, 1).is_err());
        assert!(TraversalRequest::new(&QUERY, &trees, None, 1, budget, 0).is_err());
    }

    #[tokio::test]
    async fn empty_tree_visits_only_the_root() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 0, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 1, &[]);

        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            4,
            budgets(8, 8, 8),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(outcome.visited_partitions(), 1);
        assert_eq!(outcome.visited_leaf_entries(), 0);
        assert!(!outcome.partition_budget_exhausted());
        assert!(!outcome.leaf_entry_budget_exhausted());
        assert!(!outcome.rabitq_overlap_truncated());
        assert!(outcome.into_candidates().is_empty());
    }

    #[tokio::test]
    async fn leaf_candidates_are_rough_ordered_and_deterministic() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 3, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 1, &[0, 0, 0]);
        // Scan order is Record ID order; rough order follows the norm.
        fixture.entry(&tree, 1, 1, "far", 0, [3.0, 0.0]);
        fixture.entry(&tree, 1, 1, "mid", 0, [2.0, 0.0]);
        fixture.entry(&tree, 1, 1, "near", 0, [1.0, 0.0]);
        let trees = vec![tree_ref(&tree)];

        let outcome = run(
            fixture.items.clone(),
            &manifest,
            &trees,
            None,
            4,
            budgets(8, 8, 8),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"near".as_slice(), b"mid".as_slice(), b"far".as_slice()]
        );
        assert_eq!(outcome.visited_partitions(), 1);
        assert_eq!(outcome.visited_leaf_entries(), 3);
        assert!(!outcome.partition_budget_exhausted());
        assert!(!outcome.leaf_entry_budget_exhausted());

        // An identical snapshot, request, and budgets select identically.
        let rerun = run(
            fixture.items,
            &manifest,
            &trees,
            None,
            4,
            budgets(8, 8, 8),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(candidate_ids(&rerun), candidate_ids(&outcome));
        assert_eq!(rerun.visited_partitions(), outcome.visited_partitions());
        assert_eq!(rerun.visited_leaf_entries(), outcome.visited_leaf_entries());
    }

    #[tokio::test]
    async fn identical_codes_tie_break_to_the_record_id() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 2, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 1, &[0, 0]);
        fixture.entry(&tree, 1, 1, "b", 0, [1.0, 0.0]);
        fixture.entry(&tree, 1, 1, "a", 0, [1.0, 0.0]);

        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            4,
            budgets(8, 8, 8),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"a".as_slice(), b"b".as_slice()]
        );
    }

    #[tokio::test]
    async fn missing_root_header_is_corruption() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);

        assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
    }

    #[tokio::test]
    async fn trees_advance_fairly_under_one_shared_budget() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let first = fixture.tree(1);
        fixture.header(&first, 1, 1, 2, PartitionState::Ready);
        fixture.synopsis(&first, 1, 1, &[0, 0]);
        fixture.entry(&first, 1, 1, "a4", 0, [2.0, 0.0]);
        fixture.entry(&first, 1, 1, "a16", 0, [4.0, 0.0]);
        let second = fixture.tree(2);
        fixture.header(&second, 1, 1, 2, PartitionState::Ready);
        fixture.synopsis(&second, 2, 1, &[0, 0]);
        fixture.entry(&second, 2, 1, "b1", 0, [1.0, 0.0]);
        fixture.entry(&second, 2, 1, "b9", 0, [3.0, 0.0]);
        let trees = vec![tree_ref(&first), tree_ref(&second)];

        // Both roots are seeded up front and every leaf is visited under a
        // generous budget; merged candidates order by rough distance.
        let outcome = run(
            fixture.items.clone(),
            &manifest,
            &trees,
            None,
            8,
            budgets(16, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![
                b"b1".as_slice(),
                b"a4".as_slice(),
                b"b9".as_slice(),
                b"a16".as_slice()
            ]
        );
        assert_eq!(outcome.visited_partitions(), 2);
        assert!(!outcome.partition_budget_exhausted());

        // Roots tie at distance zero, so enumeration order decides which tree
        // advances when the Partition budget funds only one visit.
        let outcome = run(
            fixture.items,
            &manifest,
            &trees,
            None,
            8,
            budgets(1, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"a4".as_slice(), b"a16".as_slice()]
        );
        assert_eq!(outcome.visited_partitions(), 1);
        assert!(outcome.partition_budget_exhausted());
        assert!(!outcome.leaf_entry_budget_exhausted());
    }

    #[tokio::test]
    async fn internal_distance_ties_break_to_the_smaller_partition_key() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        // A depth-2 tree whose two leaf children share one centroid, so the
        // expansion tie is decided by Partition Key alone.
        fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
        fixture.header(&tree, 2, 1, 1, PartitionState::Ready);
        fixture.header(&tree, 3, 1, 1, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 2, &[0]);
        fixture.synopsis(&tree, 1, 3, &[0]);
        fixture.child(&tree, 1, 2, [1.0, 0.0]);
        fixture.child(&tree, 1, 3, [1.0, 0.0]);
        fixture.entry(&tree, 1, 2, "p2", 0, [1.0, 0.0]);
        fixture.entry(&tree, 1, 3, "p3", 0, [2.0, 0.0]);
        let trees = vec![tree_ref(&tree)];

        // The exact natural partition count completes without exhaustion.
        let outcome = run(
            fixture.items.clone(),
            &manifest,
            &trees,
            None,
            8,
            budgets(3, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"p2".as_slice(), b"p3".as_slice()]
        );
        assert_eq!(outcome.visited_partitions(), 3);
        assert!(!outcome.partition_budget_exhausted());

        // One visit fewer admits the tied children in Partition Key order.
        let outcome = run(
            fixture.items,
            &manifest,
            &trees,
            None,
            8,
            budgets(2, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(candidate_ids(&outcome), vec![b"p2".as_slice()]);
        assert_eq!(outcome.visited_partitions(), 2);
        assert!(outcome.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn empty_leaf_consumes_no_leaf_entry_budget() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
        fixture.header(&tree, 2, 1, 0, PartitionState::Ready);
        fixture.header(&tree, 3, 1, 1, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 2, &[]);
        fixture.synopsis(&tree, 1, 3, &[0]);
        fixture.child(&tree, 1, 2, [0.0, 0.0]);
        fixture.child(&tree, 1, 3, [2.0, 0.0]);
        fixture.entry(&tree, 1, 3, "x", 0, [1.0, 0.0]);

        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            8,
            budgets(16, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(candidate_ids(&outcome), vec![b"x".as_slice()]);
        assert_eq!(outcome.visited_partitions(), 3);
        assert_eq!(outcome.visited_leaf_entries(), 1);
        assert!(!outcome.leaf_entry_budget_exhausted());
    }

    #[tokio::test]
    async fn beam_prunes_leaf_visits_without_budget_exhaustion() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        // A depth-3 tree: root pk1; internals pk2, pk3; leaves pk4..pk7.
        fixture.header(&tree, 1, 3, 2, PartitionState::Ready);
        fixture.header(&tree, 2, 2, 2, PartitionState::Ready);
        fixture.header(&tree, 3, 2, 2, PartitionState::Ready);
        for leaf in [4_u64, 5, 6, 7] {
            fixture.header(&tree, leaf, 1, 1, PartitionState::Ready);
            fixture.synopsis(&tree, 1, leaf, &[0]);
        }
        fixture.child(&tree, 1, 2, [0.0, 0.0]);
        fixture.child(&tree, 1, 3, [10.0, 0.0]);
        fixture.child(&tree, 2, 4, [0.0, 0.0]);
        fixture.child(&tree, 2, 5, [1.0, 0.0]);
        fixture.child(&tree, 3, 6, [10.0, 0.0]);
        fixture.child(&tree, 3, 7, [11.0, 0.0]);
        fixture.entry(&tree, 1, 4, "e4", 0, [1.0, 0.0]);
        fixture.entry(&tree, 1, 5, "e5", 0, [2.0, 0.0]);
        fixture.entry(&tree, 1, 6, "e6", 0, [3.0, 0.0]);
        fixture.entry(&tree, 1, 7, "e7", 0, [4.0, 0.0]);
        let trees = vec![tree_ref(&tree)];

        // A wide beam admits every body.
        let outcome = run(
            fixture.items.clone(),
            &manifest,
            &trees,
            None,
            8,
            budgets(16, 16, 64),
            4,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![
                b"e4".as_slice(),
                b"e5".as_slice(),
                b"e6".as_slice(),
                b"e7".as_slice()
            ]
        );
        assert_eq!(outcome.visited_partitions(), 7);
        assert!(!outcome.partition_budget_exhausted());

        // Beam 2 admits one level-2 internal and two leaves: pk3's subtree is
        // pruned by the beam, not by any budget.
        let outcome = run(
            fixture.items.clone(),
            &manifest,
            &trees,
            None,
            8,
            budgets(16, 16, 64),
            2,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"e4".as_slice(), b"e5".as_slice()]
        );
        assert_eq!(outcome.visited_partitions(), 4);
        assert!(!outcome.partition_budget_exhausted());
        assert!(!outcome.leaf_entry_budget_exhausted());

        // Beam 1 is a greedy single path to the nearest leaf.
        let outcome = run(
            fixture.items,
            &manifest,
            &trees,
            None,
            8,
            budgets(16, 16, 64),
            1,
        )
        .await
        .expect("traverse");
        assert_eq!(candidate_ids(&outcome), vec![b"e4".as_slice()]);
        assert_eq!(outcome.visited_partitions(), 3);
        assert!(!outcome.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn synopsis_pruning_skips_provably_nonmatching_leaves() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
        fixture.header(&tree, 2, 1, 2, PartitionState::Ready);
        fixture.header(&tree, 3, 1, 2, PartitionState::Ready);
        fixture.child(&tree, 1, 2, [0.0, 0.0]);
        fixture.child(&tree, 1, 3, [10.0, 0.0]);
        fixture.synopsis(&tree, 1, 2, &[1, 2]);
        fixture.synopsis(&tree, 1, 3, &[100, 101]);
        fixture.entry(&tree, 1, 2, "p", 1, [1.0, 0.0]);
        fixture.entry(&tree, 1, 2, "q", 2, [2.0, 0.0]);
        fixture.entry(&tree, 1, 3, "r", 100, [3.0, 0.0]);
        fixture.entry(&tree, 1, 3, "s", 101, [4.0, 0.0]);
        let trees = vec![tree_ref(&tree)];

        // pk2's synopsis proves no entry can equal 100: its entries are never
        // read nor charged. pk3 is MayMatch and filters exactly.
        let predicate = Predicate::Compare {
            field: FieldId(1),
            op: CompareOp::Eq,
            value: Value::I64(100),
        };
        let outcome = run(
            fixture.items.clone(),
            &manifest,
            &trees,
            Some(predicate),
            8,
            budgets(16, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(candidate_ids(&outcome), vec![b"r".as_slice()]);
        assert_eq!(outcome.visited_partitions(), 3);
        assert_eq!(outcome.visited_leaf_entries(), 2);
        assert!(!outcome.partition_budget_exhausted());
        assert!(!outcome.leaf_entry_budget_exhausted());

        // A predicate that always holds over pk2's synopsis is AllMatch:
        // evaluation is skipped but every entry is still charged and admitted.
        let always = Predicate::And(vec![
            Predicate::Compare {
                field: FieldId(1),
                op: CompareOp::GreaterOrEqual,
                value: Value::I64(1),
            },
            Predicate::Compare {
                field: FieldId(1),
                op: CompareOp::LessOrEqual,
                value: Value::I64(2),
            },
        ]);
        let outcome = run(
            fixture.items,
            &manifest,
            &trees,
            Some(always),
            8,
            budgets(16, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"p".as_slice(), b"q".as_slice()]
        );
        assert_eq!(outcome.visited_leaf_entries(), 2);
    }

    #[tokio::test]
    async fn leaf_budget_stops_mid_leaf_precisely() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 5, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 1, &[0, 0, 0, 0, 0]);
        for (index, record_id) in ["e0", "e1", "e2", "e3", "e4"].iter().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "tiny fixture ordinals")]
            fixture.entry(&tree, 1, 1, record_id, 0, [index as f32 + 1.0, 0.0]);
        }

        // The scan proceeds in Record ID order; the budget funds only e0..e2.
        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            8,
            budgets(16, 3, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"e0".as_slice(), b"e1".as_slice(), b"e2".as_slice()]
        );
        assert_eq!(outcome.visited_leaf_entries(), 3);
        assert!(outcome.leaf_entry_budget_exhausted());
        assert!(!outcome.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn leaf_budget_exact_completion_then_pending_leaf_exhausts() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let first = fixture.tree(1);
        fixture.header(&first, 1, 1, 2, PartitionState::Ready);
        fixture.synopsis(&first, 1, 1, &[0, 0]);
        fixture.entry(&first, 1, 1, "a1", 0, [1.0, 0.0]);
        fixture.entry(&first, 1, 1, "a2", 0, [2.0, 0.0]);
        let second = fixture.tree(2);
        fixture.header(&second, 1, 1, 1, PartitionState::Ready);
        fixture.synopsis(&second, 2, 1, &[0]);
        fixture.entry(&second, 2, 1, "b1", 0, [3.0, 0.0]);
        let trees = vec![tree_ref(&first), tree_ref(&second)];

        // The budget exactly covers the first tree's entries: natural
        // completion is not exhaustion, but the pending non-empty second leaf
        // is prevented work.
        let outcome = run(
            fixture.items.clone(),
            &manifest,
            &trees,
            None,
            8,
            budgets(16, 2, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"a1".as_slice(), b"a2".as_slice()]
        );
        assert_eq!(outcome.visited_partitions(), 2);
        assert_eq!(outcome.visited_leaf_entries(), 2);
        assert!(outcome.leaf_entry_budget_exhausted());
        assert!(!outcome.partition_budget_exhausted());

        // Funding every entry exactly empties the frontier: no exhaustion.
        let outcome = run(
            fixture.items,
            &manifest,
            &trees,
            None,
            8,
            budgets(16, 3, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(outcome.visited_leaf_entries(), 3);
        assert!(!outcome.leaf_entry_budget_exhausted());
        assert!(!outcome.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn per_leaf_overlap_cap_reports_truncation() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 3, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 1, &[0, 0, 0]);
        fixture.entry(&tree, 1, 1, "a", 0, [1.0, 0.0]);
        fixture.entry(&tree, 1, 1, "b", 0, [2.0, 0.0]);
        fixture.entry(&tree, 1, 1, "c", 0, [3.0, 0.0]);

        // k = 1 with a single-candidate rerank budget caps the leaf's overlap
        // set at min(4 * 3, 1) and reports the truncation.
        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            1,
            budgets(16, 16, 1),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(candidate_ids(&outcome), vec![b"a".as_slice()]);
        assert!(outcome.rabitq_overlap_truncated());
        assert_eq!(outcome.visited_leaf_entries(), 3);
        assert!(!outcome.leaf_entry_budget_exhausted());
    }

    #[tokio::test]
    async fn non_root_split_family_searches_source_and_targets() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        // Transient fanout three: the draining source and both receiving
        // targets are reachable through the parent's current Child Entries.
        fixture.header(&tree, 1, 2, 3, PartitionState::Ready);
        fixture.child(&tree, 1, 2, [2.0, 0.0]);
        fixture.child(&tree, 1, 3, [0.0, 0.0]);
        fixture.child(&tree, 1, 4, [4.0, 0.0]);
        fixture.header(&tree, 2, 1, 1, PartitionState::DrainingSplit);
        fixture.state(
            &tree,
            2,
            PartitionTransition::DrainingSplit {
                left: pk(3),
                right: pk(4),
                started_at_unix_millis: 0,
            },
        );
        fixture.header(&tree, 3, 1, 1, PartitionState::ReceivingSplit);
        fixture.state(
            &tree,
            3,
            PartitionTransition::ReceivingSplit {
                source: pk(2),
                started_at_unix_millis: 0,
            },
        );
        fixture.header(&tree, 4, 1, 1, PartitionState::ReceivingSplit);
        fixture.state(
            &tree,
            4,
            PartitionTransition::ReceivingSplit {
                source: pk(2),
                started_at_unix_millis: 0,
            },
        );
        for leaf in [2_u64, 3, 4] {
            fixture.synopsis(&tree, 1, leaf, &[0]);
        }
        fixture.entry(&tree, 1, 2, "b", 0, [2.0, 0.0]);
        fixture.entry(&tree, 1, 3, "a", 0, [1.0, 0.0]);
        fixture.entry(&tree, 1, 4, "c", 0, [3.0, 0.0]);

        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            8,
            budgets(16, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]
        );
        assert_eq!(outcome.visited_partitions(), 4);
        assert_eq!(outcome.visited_leaf_entries(), 3);
        assert!(!outcome.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn splitting_root_searches_only_the_root_body() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 1, PartitionState::Splitting);
        fixture.state(
            &tree,
            1,
            PartitionTransition::Splitting {
                left: pk(2),
                right: pk(3),
                started_at_unix_millis: 0,
            },
        );
        fixture.synopsis(&tree, 1, 1, &[0]);
        fixture.entry(&tree, 1, 1, "root", 0, [1.0, 0.0]);
        // Unexposed targets exist physically but stay unreachable to search.
        fixture.header(&tree, 2, 1, 1, PartitionState::ReceivingSplit);
        fixture.synopsis(&tree, 1, 2, &[0]);
        fixture.entry(&tree, 1, 2, "hidden", 0, [0.5, 0.0]);

        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            8,
            budgets(16, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(candidate_ids(&outcome), vec![b"root".as_slice()]);
        assert_eq!(outcome.visited_partitions(), 1);
        assert_eq!(outcome.visited_leaf_entries(), 1);
    }

    #[tokio::test]
    async fn draining_root_searches_root_and_both_targets() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 1, PartitionState::DrainingSplit);
        fixture.state(
            &tree,
            1,
            PartitionTransition::DrainingSplit {
                left: pk(2),
                right: pk(3),
                started_at_unix_millis: 0,
            },
        );
        for (target, record_id, vector) in [(2_u64, "a", [1.0, 0.0]), (3_u64, "c", [3.0, 0.0])] {
            fixture.header(&tree, target, 1, 1, PartitionState::ReceivingSplit);
            fixture.state(
                &tree,
                target,
                PartitionTransition::ReceivingSplit {
                    source: pk(1),
                    started_at_unix_millis: 0,
                },
            );
            fixture.synopsis(&tree, 1, target, &[0]);
            fixture.entry(&tree, 1, target, record_id, 0, vector);
        }
        fixture.synopsis(&tree, 1, 1, &[0]);
        fixture.entry(&tree, 1, 1, "b", 0, [2.0, 0.0]);

        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            8,
            budgets(16, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]
        );
        assert_eq!(outcome.visited_partitions(), 3);
        assert_eq!(outcome.visited_leaf_entries(), 3);
        assert!(!outcome.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn merging_source_is_searched_like_an_ordinary_partition() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
        fixture.child(&tree, 1, 2, [0.0, 0.0]);
        fixture.child(&tree, 1, 3, [2.0, 0.0]);
        fixture.header(&tree, 2, 1, 1, PartitionState::Merging);
        fixture.state(
            &tree,
            2,
            PartitionTransition::Merging {
                started_at_unix_millis: 0,
            },
        );
        fixture.synopsis(&tree, 1, 2, &[0]);
        fixture.entry(&tree, 1, 2, "a", 0, [1.0, 0.0]);
        fixture.header(&tree, 3, 1, 1, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 3, &[0]);
        fixture.entry(&tree, 1, 3, "b", 0, [2.0, 0.0]);

        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            8,
            budgets(16, 16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await
        .expect("traverse");
        assert_eq!(
            candidate_ids(&outcome),
            vec![b"a".as_slice(), b"b".as_slice()]
        );
        assert_eq!(outcome.visited_partitions(), 3);
        assert_eq!(outcome.visited_leaf_entries(), 2);
    }

    #[tokio::test]
    async fn missing_child_header_is_corruption() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 1, PartitionState::Ready);
        fixture.child(&tree, 1, 2, [0.0, 0.0]);

        assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
    }

    #[tokio::test]
    async fn child_level_mismatch_is_corruption() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 1, PartitionState::Ready);
        fixture.child(&tree, 1, 2, [0.0, 0.0]);
        // The child of a level-2 partition must be a level-1 leaf.
        fixture.header(&tree, 2, 2, 0, PartitionState::Ready);

        assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
    }

    #[tokio::test]
    async fn second_incoming_child_entry_is_corruption() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 3, 2, PartitionState::Ready);
        fixture.header(&tree, 2, 2, 1, PartitionState::Ready);
        fixture.header(&tree, 3, 2, 1, PartitionState::Ready);
        fixture.header(&tree, 4, 1, 0, PartitionState::Ready);
        fixture.child(&tree, 1, 2, [0.0, 0.0]);
        fixture.child(&tree, 1, 3, [1.0, 0.0]);
        fixture.child(&tree, 2, 4, [0.0, 0.0]);
        // A second incoming Child Entry for one non-root partition.
        fixture.child(&tree, 3, 4, [0.0, 0.0]);

        assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
    }

    #[tokio::test]
    async fn root_target_or_merge_states_are_corruption() {
        for state in [PartitionState::ReceivingSplit, PartitionState::Merging] {
            let manifest = manifest();
            let mut fixture = Fixture::new(&manifest);
            let tree = fixture.tree(1);
            fixture.header(&tree, 1, 1, 0, state);

            assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
        }
    }

    #[tokio::test]
    async fn draining_root_requires_its_matching_state() {
        // The State value is missing entirely.
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 0, PartitionState::DrainingSplit);
        assert_corrupt(fixture.items.clone(), &manifest, &[tree_ref(&tree)]).await;

        // The State value disagrees with the Header discriminator.
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 0, PartitionState::DrainingSplit);
        fixture.state(
            &tree,
            1,
            PartitionTransition::Ready {
                started_at_unix_millis: 0,
            },
        );
        assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
    }

    #[tokio::test]
    async fn missing_synopsis_with_predicate_is_corruption() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 1, PartitionState::Ready);
        fixture.child(&tree, 1, 2, [0.0, 0.0]);
        fixture.header(&tree, 2, 1, 1, PartitionState::Ready);
        fixture.entry(&tree, 1, 2, "x", 0, [1.0, 0.0]);

        let predicate = Predicate::Compare {
            field: FieldId(1),
            op: CompareOp::Eq,
            value: Value::I64(0),
        };
        assert_corruption(
            run(
                fixture.items,
                &manifest,
                &[tree_ref(&tree)],
                Some(predicate),
                8,
                budgets(16, 16, 64),
                DEFAULT_LEAF_BEAM,
            )
            .await,
        );
    }

    #[tokio::test]
    async fn malformed_entry_values_are_corruption() {
        // A truncated canonical Leaf Entry payload breaks the codec frame.
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 1, PartitionState::Ready);
        fixture.synopsis(&tree, 1, 1, &[0]);
        let mut value = encode(
            &manifest,
            &PersistentValue::LeafEntry(LeafEntry::new(
                Bytes::from_static(b"x"),
                vec![Value::I64(1), Value::I64(0)],
                RaBitQ7::quantize(&[1.0, 0.0]).expect("quantize"),
            )),
        );
        value.pop();
        fixture.raw_entry(&tree, 1, "x", value);
        assert_corrupt(fixture.items.clone(), &manifest, &[tree_ref(&tree)]).await;

        // A truncated canonical Child Entry payload likewise fails closed.
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 1, PartitionState::Ready);
        let mut value = encode(
            &manifest,
            &PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.0, 0.0])),
        );
        value.pop();
        fixture.raw_child(&tree, 1, 2, value);
        assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
    }

    #[tokio::test]
    async fn duplicate_record_ids_across_leaves_are_corruption() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let first = fixture.tree(1);
        fixture.header(&first, 1, 1, 1, PartitionState::Ready);
        fixture.synopsis(&first, 1, 1, &[0]);
        fixture.entry(&first, 1, 1, "dup", 0, [1.0, 0.0]);
        let second = fixture.tree(2);
        fixture.header(&second, 1, 1, 1, PartitionState::Ready);
        fixture.synopsis(&second, 2, 1, &[0]);
        fixture.entry(&second, 2, 1, "dup", 0, [2.0, 0.0]);

        assert_corrupt(
            fixture.items,
            &manifest,
            &[tree_ref(&first), tree_ref(&second)],
        )
        .await;
    }
}
