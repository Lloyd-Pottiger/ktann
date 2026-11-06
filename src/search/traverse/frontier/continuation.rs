//! Retained tree branches for bounded, predicate-aware leaf coverage.

use std::collections::{BTreeMap, BTreeSet};

use super::{BinaryHeap, Error, ErrorKind, FrontierEntry, Result, Reverse, beam_width};

/// Pending branches by persistent level and Tree Key ordinal, bounded by visits.
///
/// Each tree owns its effective-leaf allowance. The owner charges every visit to
/// one shared budget and admits retained state against routing working bytes.
/// Completed trees cannot consume another tree's allowance or cause false budget
/// exhaustion. Root transition targets already in the active frontier bypass
/// these selection limits.
pub(super) struct Continuation {
    effective: Vec<u32>,
    leaf_beam: u32,
    visit_limit: usize,
    pending: BTreeMap<(u32, u32), BTreeSet<FrontierEntry>>,
}

impl Continuation {
    pub(super) fn new(leaf_beam: u32, trees: usize, visit_limit: u32) -> Result<Self> {
        let mut effective = Vec::new();
        effective.try_reserve_exact(trees).map_err(|_| limit())?;
        effective.resize(trees, 0);
        Ok(Self {
            leaf_beam,
            effective,
            visit_limit: visit_limit as usize,
            pending: BTreeMap::new(),
        })
    }

    /// Retains the closest V candidates per tree/level, where V is the query's
    /// total visit budget. Reaching an omitted candidate requires visiting V
    /// nearer pending candidates plus their already-visited parent, so omission
    /// cannot affect any funded traversal prefix. This bound grows with the
    /// budget; unlike an independent retention cap it does not hide reachable
    /// work or permit false natural exhaustion within that budget.
    pub(super) fn push(&mut self, entry: FrontierEntry, level: u32) {
        let queue = self.pending.entry((level, entry.tree)).or_default();
        if queue.len() == self.visit_limit {
            if queue.last().is_some_and(|worst| entry >= *worst) {
                return;
            }
            queue.pop_last();
        }
        queue.insert(entry);
    }

    /// Exact eligibility before rough-pool retention defines a useful leaf.
    pub(super) fn record_effective_leaf(&mut self, tree: u32) {
        self.effective[tree as usize] += 1;
    }

    fn tree_complete(&self, tree: u32) -> bool {
        self.effective[tree as usize] >= self.leaf_beam
    }

    /// Promotes one deepest level across unfinished trees, then restores global
    /// distance/tree/Partition Key ordering in the owner's active frontier.
    pub(super) fn select(&mut self, frontier: &mut BinaryHeap<Reverse<FrontierEntry>>) {
        let Some(level) = self
            .pending
            .keys()
            .find(|(_, tree)| !self.tree_complete(*tree))
            .map(|(level, _)| *level)
        else {
            return;
        };
        for ((_, tree), heap) in self.pending.range_mut((level, 0)..=(level, u32::MAX)) {
            if self.effective[*tree as usize] >= self.leaf_beam {
                continue;
            }
            let width = if level == 1 {
                self.leaf_beam - self.effective[*tree as usize]
            } else {
                beam_width(self.leaf_beam, level)
            };
            for _ in 0..width {
                let Some(entry) = heap.pop_first() else {
                    break;
                };
                frontier.push(Reverse(entry));
            }
        }
        self.pending.retain(|_, heap| !heap.is_empty());
    }

    pub(super) fn has_work(&self) -> bool {
        self.pending
            .keys()
            .any(|(_, tree)| !self.tree_complete(*tree))
    }

    /// Conservative admission before a parent's children are added. At most V
    /// new candidates can survive in its one tree/level group. A 512-byte charge
    /// per set entry or map group covers even a sparsely populated B-tree node;
    /// this is a working-state bound, not an allocator/process RSS estimate.
    pub(super) fn heap_bytes(&self, children: usize) -> Result<usize> {
        let slots = self
            .pending
            .values()
            .try_fold(children.min(self.visit_limit), |n, heap| {
                n.checked_add(heap.len())
            })
            .ok_or_else(limit)?;
        slots
            .checked_add(self.pending.len())
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(512))
            .and_then(|n| n.checked_add(self.effective.capacity().checked_mul(size_of::<u32>())?))
            .ok_or_else(limit)
    }
}

fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::PartitionKey;

    #[test]
    fn visit_bound_preserves_every_funded_candidate() {
        for budget in 1..=16 {
            for beam in [1, 3, 20] {
                let mut state = Continuation::new(beam, 1, budget).unwrap();
                let mut all = Vec::new();
                for ordinal in 1..=64 {
                    let entry = FrontierEntry {
                        tree: 0,
                        expected_level: Some(1),
                        partition: PartitionKey::new(ordinal).unwrap(),
                        distance: ((ordinal * 37) % 23) as f64,
                    };
                    state.push(entry, 1);
                    all.push(entry);
                }
                all.sort();
                let mut frontier = BinaryHeap::new();
                // The parent has already consumed one visit. All leaves here
                // reject the predicate, so the effective target never stops us.
                for expected in all.iter().take(budget as usize - 1) {
                    if frontier.is_empty() {
                        state.select(&mut frontier);
                    }
                    assert_eq!(frontier.pop().unwrap().0, *expected);
                }
                // If entries were omitted, eligible retained work still exposes
                // budget exhaustion; the queue cannot falsely claim completion.
                assert!(!frontier.is_empty() || state.has_work());
            }
        }
    }

    #[test]
    fn later_parents_preserve_the_remaining_funded_prefix() {
        let budget = 20;
        let mut state = Continuation::new(1, 1, budget).unwrap();
        let mut reference = BTreeSet::new();
        for round in 0..2 {
            // A parent visit precedes discovery. The second parent is expanded
            // after some of the first parent's candidates have been consumed.
            for ordinal in 1..=64 {
                let entry = FrontierEntry {
                    tree: 0,
                    expected_level: Some(1),
                    partition: PartitionKey::new(round * 64 + ordinal).unwrap(),
                    distance: ((ordinal * 37 + round * 13) % 71) as f64,
                };
                state.push(entry, 1);
                reference.insert(entry);
            }
            let leaves = if round == 0 { 5 } else { budget - 2 - 5 };
            for _ in 0..leaves {
                let mut frontier = BinaryHeap::new();
                state.select(&mut frontier);
                assert_eq!(frontier.pop().unwrap().0, reference.pop_first().unwrap());
            }
        }
        assert!(state.has_work());
    }
}
