//! Tree Manifest, topology, exact-count, State-reference, and Synopsis
//! checks.
//!
//! Partition bodies sort by Tree Key, Partition Key, then subkind — Header,
//! Synopsis, State, Centroid, Leaf Entries, Child Entries — so the ordered
//! scan accumulates at most one partition and one tree at a time. Closing a
//! partition checks its Header/State pair, exact entry count, and — for a
//! leaf — the conservative Synopsis recomputed streaming from its Leaf
//! Entries. Closing a tree checks the allocator high-water mark, the unique
//! incoming Child Entry of every non-root partition (a root being split owns
//! its two targets without edges), legal State references, and root-down
//! reachability.
//!
//! The invariants mirror the writer contract: only leaves carry a Synopsis
//! (an internal root's obsolete Synopsis is deleted at promotion), a
//! `Splitting` target may legitimately be unexposed, and a `DrainingSplit`
//! target is published and must exist.

use std::cmp::Ordering;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;

use crate::api::{PartitionKey, VerifyIssueKind, typed_order};
use crate::storage::keys::TreeKey;
use crate::storage::topology::root_partition;
use crate::storage::values::{
    FieldSynopsis, IndexManifest, LeafEntry, PartitionHeader, PartitionSynopsis,
    PartitionTransition, TreeManifest,
};

use super::Context;
use super::records::RecordLedger;

/// One decoded Leaf Entry with its key context.
pub(super) struct LeafEntryItem<'a> {
    pub(super) tree_key: &'a TreeKey,
    pub(super) partition: PartitionKey,
    pub(super) id: &'a Bytes,
    pub(super) entry: &'a LeafEntry,
    /// The canonical scanned bytes, fingerprinted for the projection check.
    pub(super) raw_value: &'a Bytes,
}

/// The accumulated bodies of the partition currently being scanned.
struct PartitionFacts {
    partition: PartitionKey,
    header: Option<PartitionHeader>,
    transition: Option<PartitionTransition>,
    /// The stored Synopsis, retained for the leaf conservatism check.
    synopsis: Option<PartitionSynopsis>,
    /// The conservative lower bound expanded per scanned Leaf Entry.
    recomputed: PartitionSynopsis,
    leaf_entries: u64,
    child_entries: u64,
    /// The bytes this accumulator charges the memory limit.
    charged: u64,
}

impl PartitionFacts {
    fn new(partition: PartitionKey, manifest: &IndexManifest, synopsis_estimate: u64) -> Self {
        Self {
            partition,
            header: None,
            transition: None,
            synopsis: None,
            recomputed: PartitionSynopsis::empty(manifest),
            leaf_entries: 0,
            child_entries: 0,
            charged: (size_of::<PartitionFacts>() as u64).saturating_add(synopsis_estimate),
        }
    }
}

/// The topology facts one closed partition retains for the tree-level
/// checks.
#[derive(Clone, Copy)]
struct PartitionSummary {
    /// The level from the Header; `None` when the Header is missing.
    level: Option<u32>,
    /// The persisted State; `None` when the State is missing.
    transition: Option<PartitionTransition>,
}

/// The accumulated topology of the tree currently being scanned.
struct TreeWalk {
    tree_key: TreeKey,
    manifest: Option<TreeManifest>,
    /// One summary per partition holding any body, in Partition Key order.
    partitions: BTreeMap<PartitionKey, PartitionSummary>,
    /// The recorded Child Entry edges as `(parent, child)`.
    edges: Vec<(PartitionKey, PartitionKey)>,
    current: Option<PartitionFacts>,
    /// The resident estimate of one in-memory Partition Synopsis.
    synopsis_estimate: u64,
    /// The bytes this walk charges the memory limit.
    charged: u64,
}

/// The ledger of Tree Manifests plus the one tree currently accumulating.
pub(super) struct TopologyLedger {
    /// Tree Manifests not yet claimed by a scanned partition, in Tree Key
    /// order; partition bodies sort after the whole directory.
    trees: BTreeMap<TreeKey, TreeManifest>,
    walk: Option<TreeWalk>,
    /// The resident estimate of one in-memory Partition Synopsis, computed
    /// once from the fixed Manifest.
    synopsis_estimate: u64,
}

impl TopologyLedger {
    pub(super) fn new(manifest: &IndexManifest) -> Self {
        Self {
            trees: BTreeMap::new(),
            walk: None,
            synopsis_estimate: synopsis_estimate(manifest),
        }
    }

    /// Notes one decoded Tree Manifest directory entry.
    pub(super) fn note_tree_manifest(
        &mut self,
        cx: &mut Context<'_>,
        tree_key: &TreeKey,
        manifest: TreeManifest,
    ) {
        cx.charge_memory(tree_manifest_bytes(tree_key));
        if cx.truncated() {
            return;
        }
        self.trees.insert(tree_key.clone(), manifest);
    }

    /// Notes one decoded partition Header.
    pub(super) fn absorb_header(
        &mut self,
        cx: &mut Context<'_>,
        tree_key: &TreeKey,
        partition: PartitionKey,
        header: PartitionHeader,
    ) {
        self.open_partition(cx, tree_key, partition);
        if let Some(facts) = self.current_facts() {
            facts.header = Some(header);
        }
    }

    /// Notes one decoded partition Synopsis.
    pub(super) fn absorb_synopsis(
        &mut self,
        cx: &mut Context<'_>,
        tree_key: &TreeKey,
        partition: PartitionKey,
        synopsis: PartitionSynopsis,
    ) {
        self.open_partition(cx, tree_key, partition);
        let estimate = self.synopsis_estimate;
        cx.charge_memory(estimate);
        if cx.truncated() {
            return;
        }
        if let Some(facts) = self.current_facts() {
            facts.charged = facts.charged.saturating_add(estimate);
            facts.synopsis = Some(synopsis);
        }
    }

    /// Notes one decoded partition State.
    pub(super) fn absorb_state(
        &mut self,
        cx: &mut Context<'_>,
        tree_key: &TreeKey,
        partition: PartitionKey,
        transition: PartitionTransition,
    ) {
        self.open_partition(cx, tree_key, partition);
        if let Some(facts) = self.current_facts() {
            facts.transition = Some(transition);
        }
    }

    /// Notes one decoded partition centroid; the audit checks no centroid
    /// invariant, but the body still registers its partition's existence.
    pub(super) fn absorb_centroid(
        &mut self,
        cx: &mut Context<'_>,
        tree_key: &TreeKey,
        partition: PartitionKey,
    ) {
        self.open_partition(cx, tree_key, partition);
    }

    /// Notes one decoded Leaf Entry: counts it, expands the recomputed
    /// Synopsis lower bound, and joins it against the record ledger.
    pub(super) fn absorb_leaf_entry(
        &mut self,
        cx: &mut Context<'_>,
        records: &mut RecordLedger,
        item: LeafEntryItem<'_>,
    ) {
        let (tree_key, partition, id) = (item.tree_key, item.partition, item.id);
        self.open_partition(cx, tree_key, partition);
        if cx.truncated() {
            return;
        }
        if let Some(facts) = self.current_facts() {
            facts.leaf_entries = facts.leaf_entries.saturating_add(1);
            if facts.header.is_some_and(|header| header.level() != 1) {
                cx.issue(
                    VerifyIssueKind::Membership,
                    Some(tree_key),
                    Some(partition),
                    Some(id.clone()),
                );
            } else if facts
                .recomputed
                .expand(cx.manifest, item.entry.fields())
                .is_err()
            {
                // A decoded entry re-validates against the schema; reaching
                // this arm means the decoded state disagreed with itself.
                cx.issue(
                    VerifyIssueKind::InvalidEncoding,
                    Some(tree_key),
                    Some(partition),
                    Some(id.clone()),
                );
            }
        }
        records.join_leaf_entry(cx, tree_key, partition, id, item.raw_value);
    }

    /// Notes one decoded Child Entry: counts it and records its edge.
    pub(super) fn absorb_child_entry(
        &mut self,
        cx: &mut Context<'_>,
        tree_key: &TreeKey,
        partition: PartitionKey,
        child: PartitionKey,
    ) {
        self.open_partition(cx, tree_key, partition);
        if cx.truncated() {
            return;
        }
        let Some(walk) = self.walk.as_mut() else {
            return;
        };
        let Some(facts) = walk.current.as_mut() else {
            return;
        };
        facts.child_entries = facts.child_entries.saturating_add(1);
        if facts.header.is_some_and(|header| header.level() == 1) || child == partition {
            // A leaf holds a Child Entry, or a partition parents itself.
            cx.issue(
                VerifyIssueKind::Membership,
                Some(tree_key),
                Some(partition),
                None,
            );
            return;
        }
        cx.charge_memory(edge_bytes());
        if cx.truncated() {
            return;
        }
        walk.edges.push((partition, child));
        walk.charged = walk.charged.saturating_add(edge_bytes());
    }

    /// Closes the open tree and reports every unclaimed Tree Manifest: its
    /// tree has no scanned partition bodies, so its root is missing.
    pub(super) fn finish(&mut self, cx: &mut Context<'_>) {
        self.close_walk(cx);
        for (tree_key, _) in std::mem::take(&mut self.trees) {
            if cx.truncated() {
                break;
            }
            cx.issue(
                VerifyIssueKind::Reachability,
                Some(&tree_key),
                Some(root_partition()),
                None,
            );
            cx.release_memory(tree_manifest_bytes(&tree_key));
        }
    }

    /// Ensures the walk accumulates `partition` of `tree_key`, closing the
    /// previous partition or tree at its key-order boundary.
    fn open_partition(
        &mut self,
        cx: &mut Context<'_>,
        tree_key: &TreeKey,
        partition: PartitionKey,
    ) {
        if self
            .walk
            .as_ref()
            .is_none_or(|walk| walk.tree_key != *tree_key)
        {
            self.close_walk(cx);
            let manifest = self.trees.remove(tree_key);
            if manifest.is_some() {
                cx.release_memory(tree_manifest_bytes(tree_key));
            }
            let estimate = self.synopsis_estimate;
            self.walk = Some(TreeWalk::new(tree_key.clone(), manifest, estimate, cx));
        }
        if let Some(walk) = self.walk.as_mut() {
            walk.open_partition(cx, partition);
        }
    }

    /// The bodies of the partition currently accumulating.
    fn current_facts(&mut self) -> Option<&mut PartitionFacts> {
        self.walk.as_mut().and_then(|walk| walk.current.as_mut())
    }

    fn close_walk(&mut self, cx: &mut Context<'_>) {
        if let Some(walk) = self.walk.take() {
            walk.finish(cx);
        }
    }
}

impl TreeWalk {
    fn new(
        tree_key: TreeKey,
        manifest: Option<TreeManifest>,
        synopsis_estimate: u64,
        cx: &mut Context<'_>,
    ) -> Self {
        let charged = size_of::<TreeWalk>() as u64 + tree_key.as_bytes().len() as u64;
        cx.charge_memory(charged);
        Self {
            tree_key,
            manifest,
            partitions: BTreeMap::new(),
            edges: Vec::new(),
            current: None,
            synopsis_estimate,
            charged,
        }
    }

    /// Ensures a facts accumulator exists for `partition`, closing the
    /// previous one at its key-order boundary.
    fn open_partition(&mut self, cx: &mut Context<'_>, partition: PartitionKey) {
        if self
            .current
            .as_ref()
            .is_some_and(|current| current.partition == partition)
        {
            return;
        }
        self.close_partition(cx);
        let facts = PartitionFacts::new(partition, cx.manifest, self.synopsis_estimate);
        cx.charge_memory(facts.charged);
        if cx.truncated() {
            return;
        }
        self.current = Some(facts);
    }

    /// Closes the open partition: checks its Header/State pair, exact entry
    /// count, and Synopsis rules, then retains its summary for the
    /// tree-level checks.
    fn close_partition(&mut self, cx: &mut Context<'_>) {
        let Some(facts) = self.current.take() else {
            return;
        };
        let partition = facts.partition;
        // A partition requires its Header and State, and the pair must agree
        // on the State discriminator.
        let pair_consistent = matches!(
            (facts.header, facts.transition),
            (Some(header), Some(transition)) if transition.state() == header.state()
        );
        if !pair_consistent {
            cx.issue(
                VerifyIssueKind::Reachability,
                Some(&self.tree_key),
                Some(partition),
                None,
            );
        }
        if let Some(header) = facts.header {
            cx.note_actionable_partition(partition, header);
            let expected = u64::from(header.entry_count());
            if header.level() == 1 {
                if facts.leaf_entries != expected {
                    cx.issue(
                        VerifyIssueKind::CountMismatch,
                        Some(&self.tree_key),
                        Some(partition),
                        None,
                    );
                }
                match &facts.synopsis {
                    // A leaf requires its Synopsis.
                    None => {
                        cx.issue(
                            VerifyIssueKind::Reachability,
                            Some(&self.tree_key),
                            Some(partition),
                            None,
                        );
                    }
                    Some(stored) if !conservative(stored, &facts.recomputed, cx.manifest) => {
                        cx.issue(
                            VerifyIssueKind::SynopsisNotConservative,
                            Some(&self.tree_key),
                            Some(partition),
                            None,
                        );
                    }
                    Some(_) => {}
                }
            } else {
                if facts.child_entries != expected {
                    cx.issue(
                        VerifyIssueKind::CountMismatch,
                        Some(&self.tree_key),
                        Some(partition),
                        None,
                    );
                }
                if facts.synopsis.is_some() {
                    // An internal partition holds a Synopsis.
                    cx.issue(
                        VerifyIssueKind::Reachability,
                        Some(&self.tree_key),
                        Some(partition),
                        None,
                    );
                }
            }
        }
        cx.release_memory(facts.charged);
        let charged = summary_bytes();
        cx.charge_memory(charged);
        if cx.truncated() {
            return;
        }
        self.charged = self.charged.saturating_add(charged);
        self.partitions.insert(
            partition,
            PartitionSummary {
                level: facts.header.map(|header| header.level()),
                transition: facts.transition,
            },
        );
    }

    /// Closes the last partition, charges the transient tree-check state,
    /// runs every tree-level check, and releases the walk's resident state.
    fn finish(mut self, cx: &mut Context<'_>) {
        self.close_partition(cx);
        let transients = (self.partitions.len() as u64)
            .saturating_add(self.edges.len() as u64)
            .saturating_mul(size_of::<(PartitionKey, PartitionKey)>() as u64);
        cx.charge_memory(transients);
        self.charged = self.charged.saturating_add(transients);
        if !cx.truncated() {
            self.check_tree(cx);
        }
        cx.release_memory(self.charged);
    }

    /// Runs the tree-level checks: allocator high-water mark, root
    /// existence, unique incoming references, legal State references, and
    /// root-down reachability.
    fn check_tree(&self, cx: &mut Context<'_>) {
        let tree_key = &self.tree_key;
        let Some(manifest) = self.manifest else {
            // Partitions of a tree with no Tree Manifest.
            for partition in self.partitions.keys() {
                if cx.truncated() {
                    return;
                }
                cx.issue(
                    VerifyIssueKind::Reachability,
                    Some(tree_key),
                    Some(*partition),
                    None,
                );
            }
            return;
        };
        let root = manifest.root();
        let high_water = manifest.partition_key_high_water();
        for partition in self.partitions.keys() {
            if cx.truncated() {
                return;
            }
            if *partition > high_water {
                cx.issue(
                    VerifyIssueKind::CountMismatch,
                    Some(tree_key),
                    Some(*partition),
                    None,
                );
            }
        }
        if !self.partitions.contains_key(&root) {
            cx.issue(
                VerifyIssueKind::Reachability,
                Some(tree_key),
                Some(root),
                None,
            );
        }

        // Every edge: the target must exist, the parent must sit exactly one
        // level above, and no child may have two incoming references.
        let mut children: BTreeMap<PartitionKey, Vec<PartitionKey>> = BTreeMap::new();
        let mut incoming: BTreeMap<PartitionKey, PartitionKey> = BTreeMap::new();
        for (parent, child) in &self.edges {
            if cx.truncated() {
                return;
            }
            children.entry(*parent).or_default().push(*child);
            if !self.partitions.contains_key(child) {
                cx.issue(
                    VerifyIssueKind::Reachability,
                    Some(tree_key),
                    Some(*parent),
                    None,
                );
            }
            if let (Some(parent_summary), Some(child_summary)) =
                (self.partitions.get(parent), self.partitions.get(child))
            {
                if let (Some(parent_level), Some(child_level)) =
                    (parent_summary.level, child_summary.level)
                {
                    if parent_level != child_level.saturating_add(1) {
                        cx.issue(
                            VerifyIssueKind::Membership,
                            Some(tree_key),
                            Some(*parent),
                            None,
                        );
                    }
                }
            }
            match incoming.entry(*child) {
                Entry::Occupied(_) => {
                    cx.issue(
                        VerifyIssueKind::Membership,
                        Some(tree_key),
                        Some(*child),
                        None,
                    );
                }
                Entry::Vacant(slot) => {
                    slot.insert(*parent);
                }
            }
        }

        // Unique incoming reference: every non-root partition has exactly
        // one, except the two targets of a root split in flight, which the
        // root's State owns without edges.
        let root_targets = self
            .partitions
            .get(&root)
            .and_then(|summary| summary.transition)
            .and_then(split_targets);
        for partition in self.partitions.keys() {
            if cx.truncated() {
                return;
            }
            let edgeless = *partition == root
                || root_targets
                    .is_some_and(|(left, right)| left == *partition || right == *partition);
            if edgeless == incoming.contains_key(partition) {
                cx.issue(
                    VerifyIssueKind::Membership,
                    Some(tree_key),
                    Some(*partition),
                    None,
                );
            }
        }

        // Legal State references.
        for (partition, summary) in &self.partitions {
            if cx.truncated() {
                return;
            }
            match summary.transition {
                Some(PartitionTransition::ReceivingSplit { source, .. }) => {
                    let names_back = self
                        .partitions
                        .get(&source)
                        .and_then(|summary| summary.transition)
                        .and_then(split_targets)
                        .is_some_and(|(left, right)| left == *partition || right == *partition);
                    if !names_back {
                        cx.issue(
                            VerifyIssueKind::Reachability,
                            Some(tree_key),
                            Some(*partition),
                            None,
                        );
                    }
                }
                Some(PartitionTransition::Splitting { left, right, .. })
                | Some(PartitionTransition::DrainingSplit { left, right, .. }) => {
                    let draining = matches!(
                        summary.transition,
                        Some(PartitionTransition::DrainingSplit { .. })
                    );
                    for target in [left, right] {
                        if cx.truncated() {
                            return;
                        }
                        if target > high_water {
                            cx.issue(
                                VerifyIssueKind::CountMismatch,
                                Some(tree_key),
                                Some(*partition),
                                None,
                            );
                        }
                        if target == root {
                            cx.issue(
                                VerifyIssueKind::Reachability,
                                Some(tree_key),
                                Some(*partition),
                                None,
                            );
                            continue;
                        }
                        match self.partitions.get(&target) {
                            Some(target_summary) => {
                                let names_back = matches!(
                                    target_summary.transition,
                                    Some(PartitionTransition::ReceivingSplit { source, .. })
                                        if source == *partition
                                );
                                if !names_back {
                                    cx.issue(
                                        VerifyIssueKind::Reachability,
                                        Some(tree_key),
                                        Some(*partition),
                                        None,
                                    );
                                }
                                if let (Some(target_level), Some(source_level)) =
                                    (target_summary.level, summary.level)
                                {
                                    if target_level != source_level {
                                        cx.issue(
                                            VerifyIssueKind::Reachability,
                                            Some(tree_key),
                                            Some(*partition),
                                            None,
                                        );
                                    }
                                }
                            }
                            // A Splitting target may be reserved but not yet
                            // exposed; a DrainingSplit target is published.
                            None if draining => {
                                cx.issue(
                                    VerifyIssueKind::Reachability,
                                    Some(tree_key),
                                    Some(*partition),
                                    None,
                                );
                            }
                            None => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // Root-down reachability over Child Entry edges plus the drain links
        // of a split in flight.
        let mut visited = BTreeSet::new();
        let mut stack = vec![root];
        while let Some(partition) = stack.pop() {
            if !visited.insert(partition) {
                continue;
            }
            if let Some(children) = children.get(&partition) {
                stack.extend(children);
            }
            if let Some(transition) = self
                .partitions
                .get(&partition)
                .and_then(|summary| summary.transition)
            {
                if let Some((left, right)) = split_targets(transition) {
                    if self.partitions.contains_key(&left) {
                        stack.push(left);
                    }
                    if self.partitions.contains_key(&right) {
                        stack.push(right);
                    }
                }
            }
        }
        for partition in self.partitions.keys() {
            if cx.truncated() {
                return;
            }
            if !visited.contains(partition) {
                cx.issue(
                    VerifyIssueKind::Reachability,
                    Some(tree_key),
                    Some(*partition),
                    None,
                );
            }
        }
    }
}

/// The split targets named by one transition, when it is a split in flight.
fn split_targets(transition: PartitionTransition) -> Option<(PartitionKey, PartitionKey)> {
    match transition {
        PartitionTransition::Splitting { left, right, .. }
        | PartitionTransition::DrainingSplit { left, right, .. } => Some((left, right)),
        _ => None,
    }
}

/// Whether the stored leaf Synopsis covers the synopsis rebuilt from the
/// scanned Leaf Entries: stored NULL flag, extrema, and Bloom bits must be
/// a superset, because synopses only ever expand.
fn conservative(
    stored: &PartitionSynopsis,
    recomputed: &PartitionSynopsis,
    manifest: &IndexManifest,
) -> bool {
    if !stored.has_shape_for(manifest) || !recomputed.has_shape_for(manifest) {
        return false;
    }
    for (stored, recomputed) in stored.fields().iter().zip(recomputed.fields()) {
        if recomputed.has_null() && !stored.has_null() {
            return false;
        }
        if let Some(minimum) = recomputed.minimum() {
            let covered = stored.minimum().is_some_and(|stored_minimum| {
                typed_order(stored_minimum, minimum).is_some_and(|order| order != Ordering::Greater)
            });
            if !covered {
                return false;
            }
        }
        if let Some(maximum) = recomputed.maximum() {
            let covered = stored.maximum().is_some_and(|stored_maximum| {
                typed_order(stored_maximum, maximum).is_some_and(|order| order != Ordering::Less)
            });
            if !covered {
                return false;
            }
        }
        if let (Some(stored_bloom), Some(recomputed_bloom)) = (stored.bloom(), recomputed.bloom()) {
            if stored_bloom.len() != recomputed_bloom.len()
                || stored_bloom
                    .iter()
                    .zip(recomputed_bloom.iter())
                    .any(|(stored, recomputed)| *recomputed & !*stored != 0)
            {
                return false;
            }
        }
    }
    true
}

/// The resident estimate of one in-memory Partition Synopsis: the fixed
/// struct, one field struct per schema field, and the exact Bloom bytes.
fn synopsis_estimate(manifest: &IndexManifest) -> u64 {
    let fields = manifest.config().fields().len() as u64;
    let bloom: u64 = manifest
        .bloom_parameters()
        .iter()
        .map(|parameters| {
            parameters.map_or(0, |parameters| {
                u64::from(parameters.bit_count()).div_ceil(8) + 1
            })
        })
        .sum();
    (size_of::<PartitionSynopsis>() as u64)
        .saturating_add(fields.saturating_mul(size_of::<FieldSynopsis>() as u64))
        .saturating_add(bloom)
}

/// The resident estimate of one Tree Manifest map entry.
fn tree_manifest_bytes(tree_key: &TreeKey) -> u64 {
    tree_key.as_bytes().len() as u64 + size_of::<TreeManifest>() as u64
}

/// The resident estimate of one retained partition summary map entry.
fn summary_bytes() -> u64 {
    2 * size_of::<(PartitionKey, PartitionSummary)>() as u64
}

/// The resident estimate of one recorded Child Entry edge.
fn edge_bytes() -> u64 {
    size_of::<(PartitionKey, PartitionKey)>() as u64
}
