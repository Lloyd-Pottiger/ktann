//! Owns deterministic beam selection, incoming-edge uniqueness and internal routing.
//!
//! Ordinary searches use bounded beams; filtered Exact searches retain branches
//! until each tree reaches its effective-leaf target. Every exposed edge is
//! validated, including edges excluded by graph candidate selection.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::beam_width;
use crate::search::cache::graph::{PartitionReferences, RoutingExecution};
use crate::search::cache::{BodyEntries, BodyRequest, CachedBody};
use crate::search::numeric::compare_finite;
use crate::search::plan::EnumeratedTree;
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::ReadOps;
use crate::storage::values::PartitionHeader;

use super::VisitContext;

mod continuation;
use continuation::Continuation;

/// One queued frontier partition.
///
/// The active frontier order key is `(routing distance, tree, partition)`.
///
/// `expected_level` is the level the referencing edge established — `None`
/// for a tree root — and the visited Header must agree. The active frontier
/// contains only the selected beam for one depth. Roots and injected
/// root-split targets bypass the beam.
#[derive(Clone, Copy, Debug)]
pub(super) struct FrontierEntry {
    pub(super) distance: f64,
    pub(super) tree: u32,
    pub(super) partition: PartitionKey,
    pub(super) expected_level: Option<u32>,
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

/// Roots and graph bodies are read serially; eligible non-root bodies may be prefetched.
pub(super) enum VisitBatch {
    Single(FrontierEntry),
    Prefetch(Vec<FrontierEntry>),
}

/// Pending traversal work and the identities of every exposed partition.
pub(super) struct Frontier {
    continuation: Option<Continuation>,
    /// The already-selected beam at the current tree depth.
    frontier: BinaryHeap<Reverse<FrontierEntry>>,
    /// Children exposed by the current depth, grouped for next-level beam
    /// selection by tree and persistent level.
    next_frontier: HashMap<(u32, u32), BinaryHeap<FrontierEntry>>,
    /// Every referenced `{tree, partition}`: roots, injected root-split
    /// targets, and discovered Child Entries. A second reference to one body is
    /// Corruption.
    referenced: PartitionReferences,
}

impl Frontier {
    /// Seeds the frontier with every eligible tree's root.
    pub(super) fn seed(trees: &[EnumeratedTree]) -> Result<Self> {
        let mut state = Self {
            continuation: None,
            frontier: BinaryHeap::new(),
            next_frontier: HashMap::new(),
            referenced: HashSet::new(),
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

    /// Selects the next funded batch in deterministic order.
    ///
    /// Each admitted batch completes before more pending work is selected. Roots
    /// can inject earlier work, and internal graph bodies require serial admission;
    /// neither may be crossed by a metadata prefetch batch.
    pub(super) fn next_batch(&mut self, remaining: u32, local_graph: bool) -> Option<VisitBatch> {
        if remaining == 0 {
            return None;
        }
        if self.frontier.is_empty() {
            if let Some(continuation) = &mut self.continuation {
                continuation.select(&mut self.frontier);
            } else {
                for (_, entries) in self.next_frontier.drain() {
                    self.frontier.extend(entries.into_iter().map(Reverse));
                }
            }
        }
        let first = self.frontier.peek()?.0;
        let can_prefetch = |entry: &FrontierEntry| {
            entry.expected_level.is_some() && (!local_graph || entry.expected_level == Some(1))
        };
        if remaining == 1 || self.frontier.len() == 1 || !can_prefetch(&first) {
            return self
                .frontier
                .pop()
                .map(|Reverse(entry)| VisitBatch::Single(entry));
        }
        let limit = super::PARTITION_METADATA_BATCH.min(remaining as usize);
        let mut entries = Vec::with_capacity(limit.min(self.frontier.len()));
        while entries.len() < limit
            && self
                .frontier
                .peek()
                .is_some_and(|Reverse(entry)| can_prefetch(entry))
        {
            // The peek above establishes that the frontier is nonempty.
            entries.push(self.frontier.pop().expect("frontier checked non-empty").0);
        }
        Some(if entries.len() == 1 {
            VisitBatch::Single(entries[0])
        } else {
            VisitBatch::Prefetch(entries)
        })
    }

    /// Includes selected children not yet promoted to the active depth.
    pub(super) fn has_pending(&self) -> bool {
        if self
            .continuation
            .as_ref()
            .is_some_and(Continuation::has_work)
        {
            return true;
        }
        !self.frontier.is_empty() || !self.next_frontier.is_empty()
    }

    /// Enables predicate-aware branch retention before the first visit.
    pub(super) fn enable_continuation(
        &mut self,
        leaf_beam: u32,
        trees: usize,
        visits: u32,
    ) -> Result<()> {
        self.continuation = Some(Continuation::new(leaf_beam, trees, visits)?);
        Ok(())
    }

    /// Whether leaf eligibility controls continued exploration for each tree.
    pub(super) fn retains_branches(&self) -> bool {
        self.continuation.is_some()
    }

    /// Counts an admitted leaf with exact predicate matches, before rough scoring.
    pub(super) fn record_effective_leaf(&mut self, tree: u32) {
        if let Some(continuation) = &mut self.continuation {
            continuation.record_effective_leaf(tree);
        }
    }

    /// Checks unique incoming edges and reserves retained identity storage.
    fn reference(&mut self, tree: u32, partition: PartitionKey) -> Result<()> {
        if self.continuation.is_some() {
            self.referenced
                .try_reserve(1)
                .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        }
        if !self.referenced.insert((tree, partition)) {
            return Err(Error::new(ErrorKind::Corruption));
        }
        Ok(())
    }

    /// Bounds retained routing/candidate state independently of cache retention.
    pub(super) fn check_retained(
        &self,
        context: &VisitContext<'_, impl RoutingExecution>,
        children: usize,
        candidate_bytes: usize,
        extra: usize,
    ) -> Result<()> {
        if self.continuation.is_some() {
            let bytes = self
                .routing_bytes(children, 0)?
                .checked_add(candidate_bytes)
                .and_then(|bytes| bytes.checked_add(context.input_bytes))
                .and_then(|bytes| bytes.checked_add(extra))
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            if bytes > context.execution.working_byte_limit() {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
        }
        Ok(())
    }

    /// Injects a root-split target into the frontier, bypassing the beam.
    pub(super) fn inject(&mut self, tree: u32, partition: PartitionKey, level: u32) -> Result<()> {
        // A split target is exclusively referenced by the root transition
        // state before exposure; any second reference is Corruption.
        self.reference(tree, partition)?;
        self.frontier.push(Reverse(FrontierEntry {
            distance: 0.0,
            tree,
            partition,
            expected_level: Some(level),
        }));
        Ok(())
    }

    /// Exposes one internal partition's children to the next depth's beam.
    ///
    /// Reads and consumes one complete graph body through the runtime CPU boundary.
    pub(super) async fn visit_graph<T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'_, impl RoutingExecution>,
        entry: FrontierEntry,
        header: &PartitionHeader,
        candidate_bytes: usize,
    ) -> Result<()> {
        let level = header.level() - 1;
        let beam = if self.continuation.is_some() {
            header
                .entry_count()
                .min(context.request.budgets.visited_partitions()) as usize
        } else {
            beam_width(context.request.leaf_beam, level) as usize
        };
        let retained = self
            .routing_bytes(header.entry_count() as usize, beam)?
            .checked_add(candidate_bytes)
            .and_then(|bytes| bytes.checked_add(context.input_bytes))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let graph = context
            .execution
            .read(
                txn,
                context.manifest,
                BodyRequest {
                    tree_key: context.request.trees[entry.tree as usize].tree_key(),
                    partition: entry.partition,
                    header,
                },
                retained,
            )
            .await?;
        let mut query = Vec::new();
        query
            .try_reserve_exact(context.request.routing.len())
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        query.extend_from_slice(context.request.routing);
        let (candidates, references) = context
            .execution
            .candidates(
                graph,
                query,
                beam,
                context.request.routing_mode,
                entry.tree,
                std::mem::take(&mut self.referenced),
            )
            .await?;
        self.referenced = references;
        ::metrics::counter!("ktann.routing.parent_visits", "level" => routing_level_label(header.level())).increment(1);
        ::metrics::counter!("ktann.routing.merge_candidates").increment(candidates.len() as u64);
        let merge_started = std::time::Instant::now();
        for (ordinal, (child, distance)) in candidates.into_iter().enumerate() {
            if ordinal > 0 && ordinal % 64 == 0 {
                crate::search::cooperate(&|| context.execution.checkpoint()).await?;
            }
            self.merge_child(
                entry.tree,
                level,
                child,
                distance,
                context.request.leaf_beam,
            )?;
        }
        ::metrics::histogram!("ktann.routing.merge_seconds")
            .record(merge_started.elapsed().as_secs_f64());
        Ok(())
    }

    /// Conservative charged allocation bound, including simultaneous old/new hash
    /// storage during growth.
    fn routing_bytes(&self, children: usize, beam: usize) -> Result<usize> {
        let limit = || Error::new(ErrorKind::LimitExceeded);
        let identities = self
            .referenced
            .len()
            .checked_add(children)
            .ok_or_else(limit)?
            .max(self.referenced.capacity())
            .checked_next_power_of_two()
            .ok_or_else(limit)?;
        let mut bytes = identities
            .checked_mul(4 * (size_of::<(u32, PartitionKey)>() + 16))
            .ok_or_else(limit)?;
        let frontier = self
            .next_frontier
            .values()
            .try_fold(self.frontier.capacity(), |n, heap| {
                n.checked_add(heap.capacity())
            })
            .and_then(|n| n.checked_add(beam.min(children).max(4).min(beam)))
            .and_then(|n| n.checked_mul(4 * size_of::<FrontierEntry>()))
            .ok_or_else(limit)?;
        bytes = bytes.checked_add(frontier).ok_or_else(limit)?;
        let groups = self
            .next_frontier
            .capacity()
            .checked_add(1)
            .and_then(usize::checked_next_power_of_two)
            .and_then(|n| {
                n.checked_mul(4 * (size_of::<((u32, u32), BinaryHeap<FrontierEntry>)>() + 16))
            })
            .ok_or_else(limit)?;
        bytes = bytes
            .checked_add(match &self.continuation {
                Some(continuation) => continuation.heap_bytes(children)?,
                None => 0,
            })
            .ok_or_else(limit)?;
        bytes.checked_add(groups).ok_or_else(limit)
    }

    /// Retains a child under the effective-leaf or ordinary beam policy.
    fn merge_child(
        &mut self,
        tree: u32,
        level: u32,
        partition: PartitionKey,
        distance: f64,
        leaf_beam: u32,
    ) -> Result<()> {
        if let Some(continuation) = &mut self.continuation {
            continuation.push(
                FrontierEntry {
                    distance,
                    tree,
                    partition,
                    expected_level: Some(level),
                },
                level,
            );
            return Ok(());
        }
        let width = beam_width(leaf_beam, level) as usize;
        if !self.next_frontier.contains_key(&(tree, level)) {
            self.next_frontier
                .try_reserve(1)
                .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        }
        let heap = self.next_frontier.entry((tree, level)).or_default();
        let entry = FrontierEntry {
            distance,
            tree,
            partition,
            expected_level: Some(level),
        };
        if heap.len() < width {
            if heap.len() == heap.capacity() {
                let capacity = width.min(heap.capacity().saturating_mul(2).max(4));
                heap.try_reserve_exact(capacity - heap.len())
                    .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            }
            heap.push(entry);
        } else if let Some(mut worst) = heap.peek_mut()
            && entry < *worst
        {
            *worst = entry;
        }
        Ok(())
    }

    /// Exact routing baseline; all exposed children compete in one
    /// bounded tree/level heap under the existing total tie order.
    pub(super) async fn visit_internal(
        &mut self,
        context: &VisitContext<'_, impl RoutingExecution>,
        tree: u32,
        header: &PartitionHeader,
        body: &CachedBody,
        candidate_bytes: usize,
    ) -> Result<()> {
        let level = header.level();
        let BodyEntries::Internal(entries) = body.entries() else {
            return Err(Error::new(ErrorKind::Corruption));
        };
        self.check_retained(context, entries.len(), candidate_bytes, 0)?;
        for (batch, children) in entries.chunks(4).enumerate() {
            if batch > 0 && batch % 16 == 0 {
                crate::search::cooperate(&|| context.execution.checkpoint()).await?;
            }
            let distances = if children.len() == 4 {
                context.kernel.routing_centroid_distances(
                    context.request.routing,
                    std::array::from_fn::<_, 4, _>(|lane| children[lane].centroid()),
                )?
            } else {
                let mut distances = [0.0; 4];
                for (child, distance) in children.iter().zip(&mut distances) {
                    *distance = context
                        .kernel
                        .routing_distance(context.request.routing, child.centroid())?;
                }
                distances
            };
            for (child, distance) in children.iter().zip(distances) {
                // Every non-root partition has exactly one incoming Child
                // Entry; a second reference is Corruption, not a duplicate to
                // deduplicate.
                self.reference(tree, child.child())?;
                self.merge_child(
                    tree,
                    level - 1,
                    child.child(),
                    distance,
                    context.request.leaf_beam,
                )?;
            }
        }
        Ok(())
    }
}

/// Keeps metrics cardinality bounded even for unusually deep valid trees.
fn routing_level_label(level: u32) -> &'static str {
    const LEVELS: [&str; 32] = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
        "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
        "32",
    ];
    level
        .checked_sub(1)
        .and_then(|level| LEVELS.get(level as usize))
        .copied()
        .unwrap_or("33+")
}

#[cfg(test)]
mod tests {
    use crate::storage::test_support::pk;

    #[test]
    fn bounded_merge_matches_full_exposure_without_full_beam_allocation() {
        let mut state = super::Frontier::seed(&[]).unwrap();
        // A very wide request over many tiny trees must grow with actual entries.
        for tree in 0..4096 {
            state.merge_child(tree, 1, pk(1), 0.0, 16_384).unwrap();
        }
        assert_eq!(
            state
                .next_frontier
                .values()
                .map(|heap| heap.len())
                .sum::<usize>(),
            4096
        );
        assert!(
            state
                .next_frontier
                .values()
                .all(|heap| heap.capacity() <= 4)
        );

        for width in [1, 4, 17, 128] {
            let mut state = super::Frontier::seed(&[]).unwrap();
            let mut all = Vec::new();
            // Deliberately interleave parents and ties; the independent oracle
            // materializes every candidate and sorts once after full exposure.
            for ordinal in 1_u32..1000 {
                let distance = f64::from((ordinal * 37) % 53);
                let partition = pk(u64::from(ordinal));
                state.merge_child(0, 1, partition, distance, width).unwrap();
                all.push((distance, partition));
                assert!(state.next_frontier[&(0, 1)].len() <= width as usize);
                assert!(state.next_frontier[&(0, 1)].capacity() <= width as usize);
            }
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            all.truncate(width as usize);
            let actual: Vec<_> = state
                .next_frontier
                .remove(&(0, 1))
                .unwrap()
                .into_sorted_vec()
                .into_iter()
                .map(|entry| (entry.distance, entry.partition))
                .collect();
            assert_eq!(actual, all);
        }
    }
}
