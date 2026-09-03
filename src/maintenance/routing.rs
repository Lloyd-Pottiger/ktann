//! Tree routing through stable and split-intermediate topology states.
//!
//! Routing descends one tree of a Logical Index from its stable root
//! Partition Key 1 to the Leaf Partition nearest to a preprocessed routing
//! vector. The expose-then-drain split protocol (ADR 0014) keeps every
//! committed state routable: a `Splitting` source still holds its complete
//! entry set, each `ReceivingSplit` target participates through its own
//! incoming reference, and a `DrainingSplit` source is covered by the exact
//! union of its body and its two persisted targets. The write path routes a
//! whole batch at once: one grouped descent per Tree Key shares the Tree
//! Manifest read and every visited internal partition's reads across all of
//! the batch's records, and one level's Child Entry bodies are scanned in
//! batched lockstep rounds rather than one serialized page stream per body.
//!
//! # Contract
//!
//! - **Empty-root growth.** A foreground write route lazily installs the
//!   tree's Tree Manifest and initial leaf root through
//!   [`tree_manifest::create_tree`]; concurrent first inserts race through the
//!   unique directory insertion, and the loser retries and observes the
//!   winner's tree. Reads never create a tree and route an absent tree to
//!   `None`.
//! - **Observed parents, no reverse pointers.** The descent carries the
//!   parent it observed for the routed leaf. Partitions persist no reverse
//!   parent pointer (ADR 0007), so the write route validates the carried
//!   observation by update-protecting the leaf Header and the incoming Child
//!   Entry before the mutation applies. A concurrent topology change to
//!   either key aborts the commit; the retried attempt reroutes from a fresh
//!   snapshot, which is deterministic because descent is a pure function of
//!   the snapshot and the canonical tie-breakers.
//! - **Split states.** A write-accepting leaf is `Ready`, `Splitting`, or
//!   `ReceivingSplit`. A `DrainingSplit` leaf accepts no writes: the route
//!   redirects to the nearer of the two persisted target centroids with the
//!   Partition Key tie-break, and the source's `DrainingSplit` state slot
//!   naming the target is update-protected instead of any parent edge —
//!   adjacent-level maintenance may hold the two edges in different parents,
//!   and a root split's targets have no parent edge until completion.
//!   A `DrainingSplit` internal partition is descended through the union of
//!   its own remaining Child Entries and both targets', whose exact ownership
//!   is disjoint and complete. An internal `ReceivingSplit` partition is
//!   never descended alone: its source still owns the unmigrated children,
//!   so routing resolves the split family through the target's persisted
//!   source reference.
//! - **Merge state.** A `Merging` partition never accepts a write descent:
//!   the route reselects the nearest `Ready` same-level candidate with the
//!   Partition Key tie-break — the merge drain's own target rule (ADR 0008) —
//!   and redirects a leaf descent or re-descends an internal hop there, so no
//!   insert enters the source and an upsert whose Record Location names the
//!   source relocates atomically. The reselected partition's own incoming
//!   edge at its observed parent is the validated reference. When no `Ready`
//!   candidate exists the grouped write descent reports it, and the
//!   operation retries under its bounded policy before surfacing
//!   `ContentionExhausted`. Search traversal instead visits the `Merging`
//!   source as an ordinary body (ADR 0006).
//! - **Bounded depth and work.** The root Header's level bounds the descent:
//!   every hop must descend exactly one level and a leaf is exactly level 1
//!   (ADR 0006), so hop count is bounded by the persisted root level and a
//!   level that fails to decrement — including any cycle — is Corruption.
//!   Each internal body scanned contributes its exact Header count of Child
//!   Entries in bounded pages; a count mismatch is Corruption. A merge
//!   reroute hop adds one bounded same-level candidate enumeration,
//!   proportional to that level's partition count and paged like the
//!   incoming-edge rediscovery (ADR 0007).
//! - **Fail closed.** A missing Header, State, or centroid, a wrong-kind
//!   value, a level mismatch, a header/state disagreement, an incomplete
//!   split family, or a malformed centroid is Corruption; malformed caller
//!   vectors are InvalidArgument.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::beam_width;
use crate::search::numeric::{VectorKernel, compare_finite};
use crate::storage::backend::{ReadOps, ScanLimits, WriteTxn};
use crate::storage::keys::{LogicalKey, MAX_TREE_KEY_BYTES, TreeKey};
use crate::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionTransition, PersistentValue, TreeManifest, expect_centroid, expect_child_entry_ref,
    expect_header,
};
use crate::storage::{
    LogicalRange, LogicalReader, LogicalScanCursor, ReadLogicalTxn, WriteLogicalTxn, topology,
    tree_manifest,
};

/// The number of Child Entry candidates scanned per page during descent.
const CHILD_SCAN_PAGE: usize = 64;

/// The stable-topology route of one routing vector through one tree.
///
/// A Route is the observed descent outcome: the Leaf Partition the vector
/// routes to, its observed Header, and the observed parent whose Child Entry
/// is the leaf's one incoming topology reference. The carried parent
/// substitutes for a persistent reverse pointer (ADR 0007).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Route {
    leaf: PartitionKey,
    leaf_header: PartitionHeader,
    parent: Option<PartitionKey>,
    incoming: Incoming,
}

/// The incoming topology reference a write route update-protects.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Incoming {
    /// The leaf's Child Entry at the observed parent.
    ParentEdge,
    /// The draining source's persisted `DrainingSplit` state slot naming this
    /// leaf as a target.
    ///
    /// A drain redirect must not pin the target's Child Entry at a specific
    /// parent: adjacent-level maintenance may have moved the source edge and
    /// the target edge to different parents (ADR 0014). While the source
    /// drains, its state is the authoritative family reference, and
    /// completion deletes or converts it — aborting the write and forcing a
    /// fresh reroute. This also covers a root split, whose targets have no
    /// parent edge at all until root completion.
    SourceSlot(PartitionKey),
}

impl Route {
    /// Returns the Leaf Partition the vector routes to.
    #[must_use]
    pub const fn leaf(self) -> PartitionKey {
        self.leaf
    }

    /// Returns the observed leaf Header.
    #[must_use]
    pub const fn leaf_header(self) -> PartitionHeader {
        self.leaf_header
    }

    /// Returns the observed parent of the leaf, or `None` when the leaf is
    /// the tree root.
    #[must_use]
    pub const fn parent(self) -> Option<PartitionKey> {
        self.parent
    }

    /// Returns the draining split source whose state slot is this route's
    /// incoming reference, when the descent redirected around one.
    ///
    /// The mutation path offers the source to demand-driven maintenance: a
    /// write stream rerouted around a `DrainingSplit` leaf is exactly the
    /// relevant access that resumes its drain.
    #[must_use]
    pub(crate) const fn draining_source(self) -> Option<PartitionKey> {
        match self.incoming {
            Incoming::SourceSlot(source) => Some(source),
            Incoming::ParentEdge => None,
        }
    }
}

/// Routes one caller vector through one tree on a read snapshot.
///
/// Returns `None` when the Tree Key has no tree yet; reads never create one.
/// The vector is validated and preprocessed (metric normalization and the
/// persisted rotation) exactly as the write path preprocesses it, so both
/// paths descend identically on the same snapshot. A descent rerouted by a
/// `Merging` partition fails with `ContentionExhausted` when no `Ready`
/// same-level target exists — the single-vector shape has no retry layer,
/// unlike the grouped write path.
pub async fn route_leaf<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    vector: &[f32],
) -> Result<Option<Route>> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let kernel = kernel_for(manifest)?;
    let routing = kernel.preprocess(vector)?;
    let Some(tree) = tree_manifest::read_tree_manifest(txn, tree_key).await? else {
        return Ok(None);
    };
    Ok(Some(
        descend(txn, manifest, &kernel, tree_key, tree.root(), &routing).await?,
    ))
}

/// Routes one caller vector through one tree for a foreground write.
///
/// The tree is created lazily when absent (empty-root growth). The returned
/// Route is validated for the mutation: the leaf Header and, for a non-root
/// leaf, the carried parent edge are update-protected, so a concurrent
/// topology change makes the commit fail with a retryable conflict and the
/// whole attempt reroutes from a fresh snapshot. A descent rerouted by a
/// `Merging` partition fails with `ContentionExhausted` when no `Ready`
/// same-level target exists — the single-vector shape has no retry layer,
/// unlike the grouped write path.
///
/// `started_at_unix_millis` seeds the persisted state-start time when this
/// route creates the tree; it is unused when the tree already exists.
pub async fn route_leaf_for_write<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    vector: &[f32],
    started_at_unix_millis: u64,
) -> Result<Route> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let kernel = kernel_for(manifest)?;
    let routing = kernel.preprocess(vector)?;
    let root = ensure_tree(txn, tree_key, started_at_unix_millis).await?;
    let route = descend(txn, manifest, &kernel, tree_key, root, &routing).await?;
    validate_for_write(txn, manifest, tree_key, std::slice::from_ref(&route)).await?;
    Ok(route)
}

/// Routes one caller vector with an explicit write beam.
///
/// A beam wider than one explores several child paths at every internal level
/// and chooses the nearest terminal leaf among those paths. The vector is
/// still assigned to one leaf, and the same observed Header and incoming-edge
/// validation as [`route_leaf_for_write`] applies.
pub async fn route_leaf_for_write_with_beam<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    vector: &[f32],
    started_at_unix_millis: u64,
    write_beam_size: u32,
) -> Result<Route> {
    if write_beam_size == 0 {
        return Err(Error::invalid_argument());
    }
    if write_beam_size == 1 {
        return route_leaf_for_write(txn, tree_key, vector, started_at_unix_millis).await;
    }
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let kernel = kernel_for(manifest)?;
    let routing = kernel.preprocess(vector)?;
    let root = ensure_tree(txn, tree_key, started_at_unix_millis).await?;
    let descent = descend_grouped_with_beam(
        txn,
        manifest,
        &kernel,
        tree_key,
        root,
        &[&routing],
        write_beam_size,
    )
    .await?;
    match descent {
        GroupedDescent::Routed(routes) => {
            validate_for_write(txn, manifest, tree_key, &routes).await?;
            routes
                .into_iter()
                .next()
                .ok_or_else(|| Error::new(ErrorKind::Backend))
        }
        GroupedDescent::NoReadyMergeTarget => Err(Error::new(ErrorKind::ContentionExhausted)),
    }
}

/// The outcome of a grouped write descent.
pub(crate) enum GroupedDescent {
    /// Every vector routed to a write-accepting leaf, in input order.
    Routed(Vec<Route>),
    /// A `Merging` partition blocked the descent and no `Ready` same-level
    /// target exists; the operation retries under its bounded policy and
    /// surfaces `ContentionExhausted` on exhaustion (ADR 0008).
    NoReadyMergeTarget,
}

/// Routes one batch of preprocessed routing vectors through one tree for a
/// foreground write.
///
/// The batch shares one Tree Manifest read and one authority batch per visited
/// tree level instead of re-descending per record. On [`GroupedDescent::Routed`]
/// the returned Routes correspond to the input vectors by index, and every
/// *distinct* routed leaf is validated once in one batched update-protected
/// read: its Header and incoming reference are update-protected, so a
/// concurrent topology change aborts the commit and the whole attempt
/// reroutes from a fresh snapshot. [`GroupedDescent::NoReadyMergeTarget`]
/// carries no routes and validates nothing.
///
/// Every routing vector must be the exact output of
/// [`VectorKernel::preprocess`] under `kernel` for the bound Logical Index.
pub(crate) async fn route_leaves_for_write_preprocessed<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    kernel: &VectorKernel,
    routings: &[&[f32]],
    started_at_unix_millis: u64,
    write_beam_size: u32,
) -> Result<GroupedDescent> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    if write_beam_size == 0 {
        return Err(Error::invalid_argument());
    }
    if routings
        .iter()
        .any(|routing| routing.len() != manifest.config().dimension())
    {
        return Err(Error::invalid_argument());
    }
    if routings.is_empty() {
        return Ok(GroupedDescent::Routed(Vec::new()));
    }
    let root = ensure_tree(txn, tree_key, started_at_unix_millis).await?;
    let descent = if write_beam_size == 1 {
        descend_grouped(txn, manifest, kernel, tree_key, root, routings).await?
    } else {
        descend_grouped_with_beam(
            txn,
            manifest,
            kernel,
            tree_key,
            root,
            routings,
            write_beam_size,
        )
        .await?
    };
    if let GroupedDescent::Routed(routes) = &descent {
        validate_for_write(txn, manifest, tree_key, routes).await?;
    }
    Ok(descent)
}

/// Returns the root of `tree_key`'s tree, lazily installing the tree when the
/// first foreground write discovers its absence.
async fn ensure_tree<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    started_at_unix_millis: u64,
) -> Result<PartitionKey> {
    if let Some(tree) = read_tree_manifest_plain(txn, tree_key).await? {
        return Ok(tree.root());
    }
    tree_manifest::create_tree(txn, tree_key, started_at_unix_millis).await?;
    // Read-your-writes makes the just-installed directory entry visible. Its
    // absence would mean the transaction snapshot contradicts the unique
    // insertion this transaction just performed, which is a backend-contract
    // violation rather than a routing outcome.
    read_tree_manifest_plain(txn, tree_key)
        .await?
        .map(|tree| tree.root())
        .ok_or_else(|| Error::new(ErrorKind::Corruption))
}

/// Reads one Tree Manifest from a write transaction without establishing a
/// conflict on the directory key.
///
/// Foreground routes must not update-protect the allocator key: reserving
/// Partition Keys for a split update-protects the same key, and locking it on
/// every mutation would serialize unrelated foreground writes against
/// maintenance. Tree identity is instead protected by the leaf Header and
/// incoming-edge conflicts established during validation.
async fn read_tree_manifest_plain<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
) -> Result<Option<TreeManifest>> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let key = LogicalKey::TreeManifest {
        index: manifest.logical_index_id(),
        tree_key: tree_key.clone(),
    };
    match txn.get(key).await? {
        Some(PersistentValue::TreeManifest(tree)) => Ok(Some(tree)),
        // The codec decodes a directory key only as a Tree Manifest, so a
        // different value kind is unreachable but must stay fail-closed.
        Some(_) => Err(Error::new(ErrorKind::Corruption)),
        None => Ok(None),
    }
}

/// Validates one visited partition's level progression.
///
/// Every hop must descend exactly one level, so a level that fails to
/// decrement — including any cycle — is Corruption.
fn check_level(header: &PartitionHeader, expected_level: Option<u32>) -> Result<()> {
    if let Some(expected) = expected_level {
        if header.level() != expected {
            return Err(Error::new(ErrorKind::Corruption));
        }
    }
    Ok(())
}

/// Validates that the root partition carries a legal root state.
///
/// The root is the one searchable entry point: it is never a split target
/// and never merges, matching the search traversal contract.
fn check_root_state(
    root: PartitionKey,
    partition: PartitionKey,
    state: PartitionTransition,
) -> Result<()> {
    if partition == root
        && matches!(
            state,
            PartitionTransition::ReceivingSplit { .. } | PartitionTransition::Merging { .. }
        )
    {
        return Err(Error::new(ErrorKind::Corruption));
    }
    Ok(())
}

/// Descends from `root` to the write-accepting leaf nearest to `routing`,
/// carrying the observed parent.
///
/// The single-vector shape of the shared split-aware descent: each hop is
/// resolved by [`resolve_hop`] and the nearest child is selected by a
/// streaming argmin that never materializes a Child Entry.
async fn descend<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    root: PartitionKey,
    routing: &[f32],
) -> Result<Route> {
    let mut partition = root;
    let mut parent = None;
    let mut expected_level = None;
    loop {
        let authority =
            topology::read_authority(reader, manifest.logical_index_id(), tree_key, partition)
                .await?;
        match resolve_hop(
            reader,
            manifest,
            tree_key,
            root,
            partition,
            expected_level,
            authority,
        )
        .await?
        {
            Hop::Leaf(header) => {
                return Ok(Route {
                    leaf: partition,
                    leaf_header: header,
                    parent,
                    incoming: Incoming::ParentEdge,
                });
            }
            Hop::DrainRedirect {
                source,
                left,
                right,
            } => {
                let target = nearer_redirect_target(kernel, routing, &left, &right)?;
                return Ok(Route {
                    leaf: target.partition,
                    leaf_header: target.header,
                    parent,
                    incoming: Incoming::SourceSlot(source),
                });
            }
            Hop::Children { bodies, next_level } => {
                let mut nearest = NearestChild::default();
                scan_bodies(reader, manifest, tree_key, &bodies, &mut |index, entry| {
                    nearest.consider(kernel, routing, entry, bodies[index].0)
                })
                .await?;
                let (child, owner) = nearest.finish()?;
                parent = Some(owner);
                partition = child;
                expected_level = Some(next_level);
            }
            Hop::Sideways(source) => {
                // The carried parent is informational here: the next descent
                // hop below the internal source replaces it.
                partition = source;
            }
            Hop::MergeReroute { level, candidates } => {
                let Some(candidate) =
                    nearest_ready_candidate(kernel, routing, partition, &candidates)?
                else {
                    // No Ready same-level target exists while the source is
                    // Merging; the stall is transient and the caller may
                    // retry (ADR 0008).
                    return Err(Error::new(ErrorKind::ContentionExhausted));
                };
                if level == 1 {
                    // The reselected Ready leaf ends the descent; its own
                    // incoming edge at its observed parent is the reference
                    // write validation protects.
                    return Ok(Route {
                        leaf: candidate.partition(),
                        leaf_header: *candidate.header(),
                        parent: Some(candidate.parent()),
                        incoming: Incoming::ParentEdge,
                    });
                }
                // Re-descend from the reselected same-level body at the same
                // level; the next hop replaces the carried parent.
                partition = candidate.partition();
                parent = Some(candidate.parent());
            }
        }
    }
}

/// Descends from `root` with the whole group, reading each visited partition
/// once and returning one Route per input vector in input order. The same
/// shared state-machine body ([`resolve_hop`]) and fail-closed level and
/// state contract as [`descend`] apply. A descent blocked by a `Merging`
/// partition with no `Ready` same-level target reports
/// [`GroupedDescent::NoReadyMergeTarget`] instead of routes.
async fn descend_grouped<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    root: PartitionKey,
    routings: &[&[f32]],
) -> Result<GroupedDescent> {
    descend_grouped_with_beam(reader, manifest, kernel, tree_key, root, routings, 1).await
}

/// One member of a write beam currently waiting at one partition.
///
/// `distance` is the distance of the edge that reached this partition. It is
/// used only to choose between the final candidate leaves; every intermediate
/// level replaces it with the distance of the newly selected Child Entry.
#[derive(Clone, Copy)]
struct PendingBeamMember {
    member: usize,
    parent: Option<PartitionKey>,
    distance: f64,
}

/// Descends a batch while retaining several candidate paths per vector.
///
/// The grouped shape is important for imports: all vectors that currently
/// share a partition scan its Child Entry bodies once, while each vector keeps
/// its own top-N candidates. A vector still receives exactly one final Route;
/// the beam changes which leaves are eligible for that choice, not the
/// foreground membership invariant.
async fn descend_grouped_with_beam<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    root: PartitionKey,
    routings: &[&[f32]],
    write_beam_size: u32,
) -> Result<GroupedDescent> {
    let mut routes: Vec<Option<(f64, Route)>> = vec![None; routings.len()];
    let mut pending: BTreeMap<(PartitionKey, Option<u32>), Vec<PendingBeamMember>> =
        BTreeMap::from([(
            (root, None),
            (0..routings.len())
                .map(|member| PendingBeamMember {
                    member,
                    parent: None,
                    distance: 0.0,
                })
                .collect(),
        )]);
    // Track the owner of every child reference observed during the descent.
    // The same edge may be revisited through a split-family sideways hop, but
    // a child referenced by two different bodies violates the tree invariant.
    let mut incoming_owners = HashMap::from([(root, root)]);
    let mut leaf_batch_fallback = false;

    while !pending.is_empty() {
        // A normal leaf wave can finish directly after its one batch read. A
        // transitional leaf must re-enter the state-aware resolver because it
        // can redirect or expose a different body before accepting the write.
        let mut seeded_authorities: Option<
            HashMap<PartitionKey, (PartitionHeader, PartitionTransition)>,
        > = None;
        if !leaf_batch_fallback
            && pending
                .keys()
                .all(|(_, expected_level)| *expected_level == Some(1))
        {
            let leaf_pending = std::mem::take(&mut pending);
            let partitions: Vec<PartitionKey> = leaf_pending
                .keys()
                .map(|(partition, _)| *partition)
                .collect();
            let authorities = topology::read_authority_batch(
                reader,
                manifest.logical_index_id(),
                tree_key,
                &partitions,
            )
            .await?;
            let mut fast_path = true;
            let mut headers = Vec::with_capacity(authorities.len());
            for (&partition, &(header, state)) in partitions.iter().zip(&authorities) {
                check_level(&header, Some(1))?;
                check_root_state(root, partition, state)?;
                if !header.state().accepts_writes() {
                    fast_path = false;
                }
                headers.push(header);
            }
            if fast_path {
                for (((partition, _), members), header) in leaf_pending.into_iter().zip(headers) {
                    for member in members {
                        consider_write_route(
                            &mut routes,
                            member.member,
                            member.distance,
                            Route {
                                leaf: partition,
                                leaf_header: header,
                                parent: member.parent,
                                incoming: Incoming::ParentEdge,
                            },
                        );
                    }
                }
                continue;
            }
            leaf_batch_fallback = true;
            pending = leaf_pending;
            // The fallback wave re-enters the same partitions through the
            // state-aware resolver; seed its batch from this read.
            seeded_authorities = Some(partitions.into_iter().zip(authorities).collect());
        }

        // Process one level as a wave. The read traversal applies the beam
        // globally to all parents at a level; doing the same here keeps a
        // configured write beam equal to the number of live paths, rather
        // than multiplying it once per parent.
        let expected_level = pending
            .keys()
            .next()
            .map(|(_, level)| *level)
            .ok_or_else(|| Error::new(ErrorKind::Backend))?;
        let mut wave = BTreeMap::new();
        let mut rest = BTreeMap::new();
        for (key, members) in std::mem::take(&mut pending) {
            if key.1 == expected_level {
                wave.insert(key, members);
            } else {
                rest.insert(key, members);
            }
        }
        pending = rest;

        let mut nearest = None;
        let mut child_level = None;
        // Every Child Entry body the wave's resolved hops ask for, collected
        // during resolution and then scanned in batched lockstep rounds, so
        // one level's bodies share each backend round trip. `scan_members`
        // holds the waiting beam members of one resolved hop per slot, and
        // `body_slots` maps each collected body back to its hop's slot.
        let mut bodies: Vec<(PartitionKey, PartitionHeader)> = Vec::new();
        let mut body_slots: Vec<usize> = Vec::new();
        let mut scan_members: Vec<Vec<PendingBeamMember>> = Vec::new();
        // Read the complete current wave before doing any routing work. This
        // keeps authority reads proportional to tree depth, while the
        // transaction-local cache handles a partition revisited by a
        // split-family sideways hop.
        let partitions: Vec<PartitionKey> = wave.keys().map(|(partition, _)| *partition).collect();
        let mut authorities: HashMap<PartitionKey, (PartitionHeader, PartitionTransition)> =
            match seeded_authorities {
                Some(seeded) => seeded,
                None => topology::read_authority_batch(
                    reader,
                    manifest.logical_index_id(),
                    tree_key,
                    &partitions,
                )
                .await?
                .into_iter()
                .zip(&partitions)
                .map(|(authority, partition)| (*partition, authority))
                .collect(),
            };
        while let Some(((partition, current_level), members)) = wave.pop_first() {
            // A partition re-added by a sideways or merge hop during this wave
            // was not in its batch and is read on demand.
            let (header, state) = match authorities.remove(&partition) {
                Some(authority) => authority,
                None => {
                    topology::read_authority(
                        reader,
                        manifest.logical_index_id(),
                        tree_key,
                        partition,
                    )
                    .await?
                }
            };
            match resolve_hop(
                reader,
                manifest,
                tree_key,
                root,
                partition,
                current_level,
                (header, state),
            )
            .await?
            {
                Hop::Leaf(header) => {
                    for member in members {
                        consider_write_route(
                            &mut routes,
                            member.member,
                            member.distance,
                            Route {
                                leaf: partition,
                                leaf_header: header,
                                parent: member.parent,
                                incoming: Incoming::ParentEdge,
                            },
                        );
                    }
                }
                Hop::DrainRedirect {
                    source,
                    left,
                    right,
                } => {
                    for member in members {
                        let target =
                            nearer_redirect_target(kernel, routings[member.member], &left, &right)?;
                        let distance = kernel.routing_distance(
                            routings[member.member],
                            target.centroid.components(),
                        )?;
                        consider_write_route(
                            &mut routes,
                            member.member,
                            distance,
                            Route {
                                leaf: target.partition,
                                leaf_header: target.header,
                                parent: member.parent,
                                incoming: Incoming::SourceSlot(source),
                            },
                        );
                    }
                }
                Hop::Children {
                    bodies: hop_bodies,
                    next_level,
                } => {
                    if child_level.is_some_and(|level| level != next_level) {
                        return Err(Error::new(ErrorKind::Corruption));
                    }
                    child_level = Some(next_level);
                    nearest.get_or_insert_with(|| {
                        vec![
                            NearestChildren::new(
                                usize::try_from(beam_width(write_beam_size, next_level))
                                    .unwrap_or(usize::MAX),
                            );
                            routings.len()
                        ]
                    });
                    // Scanning is deferred until the wave is fully resolved;
                    // the nearest candidates are then accumulated across every
                    // collected body of the level in batched lockstep rounds.
                    let slot = scan_members.len();
                    scan_members.push(members);
                    body_slots.extend(std::iter::repeat_n(slot, hop_bodies.len()));
                    bodies.extend(hop_bodies);
                    debug_assert!(
                        expected_level.is_none_or(|level| level > next_level),
                        "child level must descend"
                    );
                }
                Hop::Sideways(source) => {
                    wave.entry((source, current_level))
                        .or_default()
                        .extend(members);
                }
                Hop::MergeReroute { level, candidates } => {
                    for member in members {
                        let Some(candidate) = nearest_ready_candidate(
                            kernel,
                            routings[member.member],
                            partition,
                            &candidates,
                        )?
                        else {
                            return Ok(GroupedDescent::NoReadyMergeTarget);
                        };
                        let distance = kernel
                            .routing_distance(routings[member.member], candidate.centroid())?;
                        if level == 1 {
                            consider_write_route(
                                &mut routes,
                                member.member,
                                distance,
                                Route {
                                    leaf: candidate.partition(),
                                    leaf_header: *candidate.header(),
                                    parent: Some(candidate.parent()),
                                    incoming: Incoming::ParentEdge,
                                },
                            );
                        } else {
                            wave.entry((candidate.partition(), Some(level)))
                                .or_default()
                                .push(PendingBeamMember {
                                    member: member.member,
                                    parent: Some(candidate.parent()),
                                    distance,
                                });
                        }
                    }
                }
            }
        }

        // The wave's hops are resolved; now every collected Child Entry body
        // is scanned in batched lockstep rounds, feeding the shared per-vector
        // beams and the dual-ownership check exactly as a sequential scan did.
        if !bodies.is_empty() {
            let nearest = nearest
                .as_mut()
                .ok_or_else(|| Error::new(ErrorKind::Backend))?;
            scan_bodies(reader, manifest, tree_key, &bodies, &mut |index, entry| {
                let (body, _) = bodies[index];
                if incoming_owners
                    .insert(entry.child(), body)
                    .is_some_and(|owner| owner != body)
                {
                    return Err(Error::new(ErrorKind::Corruption));
                }
                for member in &scan_members[body_slots[index]] {
                    nearest[member.member].consider(
                        kernel,
                        routings[member.member],
                        entry,
                        body,
                    )?;
                }
                Ok(())
            })
            .await?;
        }

        // Child candidates from this wave become the next wave only after the
        // global per-vector beam has been selected.
        if let Some(next_level) = child_level {
            let nearest = nearest.ok_or_else(|| Error::new(ErrorKind::Backend))?;
            for (member, slot) in nearest.into_iter().enumerate() {
                for (distance, child, owner) in slot.into_best() {
                    pending
                        .entry((child, Some(next_level)))
                        .or_default()
                        .push(PendingBeamMember {
                            member,
                            parent: Some(owner),
                            distance,
                        });
                }
            }
        }
    }

    Ok(GroupedDescent::Routed(
        routes
            .into_iter()
            .map(|route| {
                route
                    .map(|(_, route)| route)
                    .ok_or_else(|| Error::new(ErrorKind::Backend))
            })
            .collect::<Result<Vec<_>>>()?,
    ))
}

/// Returns the write beam at one child level, halving toward the root like the
/// read traversal. `write_beam_size` is the leaf-level width; this keeps a
/// configured beam comparable between write routing and search routing.
/// Keeps the best terminal leaf for one member of a write beam.
fn consider_write_route(
    routes: &mut [Option<(f64, Route)>],
    member: usize,
    distance: f64,
    route: Route,
) {
    let replace = routes[member].is_none_or(|(best_distance, best_route)| {
        beats(distance, route.leaf(), (best_distance, best_route.leaf()))
    });
    if replace {
        routes[member] = Some((distance, route));
    }
}

/// The nearest `beam` Child Entries for one vector during grouped descent.
#[derive(Clone)]
struct NearestChildren {
    beam: usize,
    best: std::collections::BinaryHeap<WriteChildCandidate>,
}

impl NearestChildren {
    fn new(beam: usize) -> Self {
        Self {
            beam,
            best: std::collections::BinaryHeap::with_capacity(beam.min(8)),
        }
    }

    fn consider(
        &mut self,
        kernel: &VectorKernel,
        routing: &[f32],
        entry: &ChildEntry,
        owner: PartitionKey,
    ) -> Result<()> {
        let distance = kernel.routing_distance(routing, entry.centroid())?;
        let candidate = WriteChildCandidate {
            distance,
            child: entry.child(),
            owner,
        };
        if self.best.len() < self.beam {
            self.best.push(candidate);
        } else if self.best.peek().is_some_and(|worst| candidate < *worst) {
            self.best.pop();
            self.best.push(candidate);
        }
        Ok(())
    }

    fn into_best(self) -> impl Iterator<Item = (f64, PartitionKey, PartitionKey)> {
        self.best
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| (candidate.distance, candidate.child, candidate.owner))
    }
}

/// One candidate retained by the bounded write beam. The heap's greatest
/// element is the worst candidate, so a better candidate can replace it in
/// logarithmic time without shifting a sorted vector on every scan.
#[derive(Clone, Copy, Debug)]
struct WriteChildCandidate {
    distance: f64,
    child: PartitionKey,
    owner: PartitionKey,
}

impl Ord for WriteChildCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        compare_finite(self.distance, other.distance)
            .then_with(|| self.child.cmp(&other.child))
            .then_with(|| self.owner.cmp(&other.owner))
    }
}

impl PartialOrd for WriteChildCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for WriteChildCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for WriteChildCandidate {}

/// One resolved descent hop: the split-aware state machine's whole per-hop
/// decision for one visited partition, shared by both descent shapes.
enum Hop {
    /// A write-accepting leaf — `Ready`, `Splitting`, or `ReceivingSplit` —
    /// ends the descent here.
    Leaf(PartitionHeader),
    /// A draining leaf accepts no writes: each vector redirects to its nearer
    /// persisted target, sharing the source's observed parent or root target
    /// slot. Both targets' authority values were read in one batch.
    DrainRedirect {
        /// The draining source whose state slot is the redirect's incoming
        /// reference.
        source: PartitionKey,
        /// The left target's validated authority values.
        left: DrainTarget,
        /// The right target's validated authority values.
        right: DrainTarget,
    },
    /// Descend one level across the exact Child Entry union of these
    /// candidate bodies: the partition itself for stable or splitting
    /// descent, or its whole draining split family, source first.
    Children {
        /// The bodies whose Child Entry union covers this hop.
        bodies: Vec<(PartitionKey, PartitionHeader)>,
        /// The level every child of this hop must sit at.
        next_level: u32,
    },
    /// An internal `ReceivingSplit` target is never descended alone while its
    /// source still holds every entry: re-descend from the source at the same
    /// level.
    Sideways(PartitionKey),
    /// A `Merging` partition is never descended into for writes: each vector
    /// reselects the nearest `Ready` same-level candidate — the merge drain's
    /// own target rule (ADR 0008) — and redirects (leaf) or re-descends
    /// (internal) there. Carries every same-level candidate, including the
    /// source itself and non-Ready partitions, both excluded at selection.
    MergeReroute {
        /// The level the merging partition and its candidates sit at.
        level: u32,
        /// Every same-level candidate under the descent's snapshot.
        candidates: Vec<topology::LevelCandidate>,
    },
}

/// One draining leaf's redirect target with its validated authority values.
struct DrainTarget {
    partition: PartitionKey,
    header: PartitionHeader,
    centroid: PartitionCentroid,
}

/// Resolves one split-aware hop from an authority pair already read by the
/// caller. This state-machine body serves both the single-vector read descent
/// and the grouped write descent; batching changes only how the snapshot pair
/// arrives.
/// Split-family resolution stays within one level: a `ReceivingSplit`
/// internal partition hops sideways to its source's family at most once,
/// because targets cannot themselves split until their source drains.
async fn resolve_hop<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    root: PartitionKey,
    partition: PartitionKey,
    expected_level: Option<u32>,
    authority: (PartitionHeader, PartitionTransition),
) -> Result<Hop> {
    let (header, state) = authority;
    let index = manifest.logical_index_id();
    check_level(&header, expected_level)?;
    check_root_state(root, partition, state)?;
    // A write-accepting leaf — Ready, Splitting, or ReceivingSplit — is the
    // descent's end.
    if header.level() == 1 && header.state().accepts_writes() {
        return Ok(Hop::Leaf(header));
    }
    match state {
        PartitionTransition::Ready { .. } | PartitionTransition::Splitting { .. } => {
            // A Splitting source still holds its complete child set; its
            // empty targets are correctly skipped until draining starts.
            Ok(Hop::Children {
                bodies: vec![(partition, header)],
                next_level: header.level() - 1,
            })
        }
        PartitionTransition::ReceivingSplit { source, .. } => {
            // An internal target is never descended alone: its source still
            // owns the unmigrated children, so the split family is resolved
            // through the persisted source reference.
            let (source_header, source_state) =
                topology::read_authority(reader, index, tree_key, source).await?;
            match source_state {
                PartitionTransition::Splitting { left, right, .. }
                    if partition == left || partition == right =>
                {
                    // The source still holds every entry; the target is empty
                    // and contributes nothing.
                    Ok(Hop::Sideways(source))
                }
                source_state @ PartitionTransition::DrainingSplit { .. } => Ok(Hop::Children {
                    bodies: topology::split_family_bodies(
                        reader,
                        index,
                        tree_key,
                        source,
                        source_header,
                        source_state,
                        header.level(),
                    )
                    .await?,
                    next_level: header.level() - 1,
                }),
                _ => Err(Error::new(ErrorKind::Corruption)),
            }
        }
        state @ PartitionTransition::DrainingSplit { left, right, .. } => {
            if header.level() == 1 {
                return drain_redirect(reader, manifest, tree_key, partition, left, right).await;
            }
            // The exact union of the source body and both targets covers the
            // source's original children.
            Ok(Hop::Children {
                bodies: topology::split_family_bodies(
                    reader,
                    index,
                    tree_key,
                    partition,
                    header,
                    state,
                    header.level(),
                )
                .await?,
                next_level: header.level() - 1,
            })
        }
        PartitionTransition::Merging { .. } => {
            // A Merging partition accepts no write descent: its entries are
            // leaving for reselected Ready targets, so the route reselects
            // under current topology — the merge drain's own rule (ADR 0008).
            let candidates =
                topology::same_level_candidates(reader, manifest, tree_key, header.level()).await?;
            Ok(Hop::MergeReroute {
                level: header.level(),
                candidates,
            })
        }
    }
}

/// Resolves one draining leaf's redirect, reading both targets' Centroids and
/// Headers in one batched call and validating the complete split family.
async fn drain_redirect<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    left: PartitionKey,
    right: PartitionKey,
) -> Result<Hop> {
    let index = manifest.logical_index_id();
    let mut values = reader
        .batch_get(vec![
            LogicalKey::Centroid {
                index,
                tree_key: tree_key.clone(),
                partition: left,
            },
            LogicalKey::Centroid {
                index,
                tree_key: tree_key.clone(),
                partition: right,
            },
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: left,
            },
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: right,
            },
        ])
        .await?
        .into_iter();
    let (Some(left_centroid), Some(right_centroid), Some(left_header), Some(right_header)) =
        (values.next(), values.next(), values.next(), values.next())
    else {
        // The typed batch read returns exactly one value per input key.
        return Err(Error::new(ErrorKind::Backend));
    };
    let read_target = |partition, centroid, header| -> Result<DrainTarget> {
        let centroid =
            expect_centroid(centroid)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        let header = expect_header(header)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        if header.level() != 1 || header.state() != PartitionState::ReceivingSplit {
            return Err(Error::new(ErrorKind::Corruption));
        }
        Ok(DrainTarget {
            partition,
            header,
            centroid,
        })
    };
    Ok(Hop::DrainRedirect {
        source,
        left: read_target(left, left_centroid, left_header)?,
        right: read_target(right, right_centroid, right_header)?,
    })
}

/// Chooses the nearer of one draining leaf's two persisted targets.
///
/// The persisted target centroids are routing models learned at exposure;
/// exact movement, not centroid freshness, preserves membership (ADR 0014).
fn nearer_redirect_target<'a>(
    kernel: &VectorKernel,
    routing: &[f32],
    left: &'a DrainTarget,
    right: &'a DrainTarget,
) -> Result<&'a DrainTarget> {
    let left_distance = kernel.routing_distance(routing, left.centroid.components())?;
    let right_distance = kernel.routing_distance(routing, right.centroid.components())?;
    Ok(
        if nearer_of_two(
            left.partition,
            left_distance,
            right.partition,
            right_distance,
        ) == left.partition
        {
            left
        } else {
            right
        },
    )
}

/// Scans every listed body's complete Child Entry set in batched lockstep
/// rounds, visiting each entry by reference with its body's index in `bodies`.
///
/// One round reads the current page of every unfinished body in a single
/// `batch_scan`, so a group of bodies costs the largest body's page count in
/// backend round trips instead of the sum of every body's pages. The Header
/// count is exact in every committed state, so each body's scan must see
/// precisely that many Child Entries; any mismatch is Corruption. An empty
/// result is legal only for a not-yet-filled split target or a fully drained
/// source.
async fn scan_bodies<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    bodies: &[(PartitionKey, PartitionHeader)],
    visit: &mut impl FnMut(usize, &ChildEntry) -> Result<()>,
) -> Result<()> {
    /// One body's in-progress scan: its exact range, continuation, and the
    /// number of Child Entries seen so far.
    struct BodyScan {
        range: LogicalRange,
        cursor: Option<LogicalScanCursor>,
        seen: usize,
        done: bool,
    }

    let limits = child_scan_limits(manifest.config().dimension())?;
    let mut scans = bodies
        .iter()
        .map(|(partition, _)| {
            Ok(BodyScan {
                range: LogicalRange::child_entries(manifest, tree_key, *partition)?,
                cursor: None,
                seen: 0,
                done: false,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    loop {
        let mut legs = Vec::new();
        for scan in &scans {
            if !scan.done {
                legs.push((&scan.range, scan.cursor.as_ref()));
            }
        }
        if legs.is_empty() {
            break;
        }
        let mut pages = reader.batch_scan(&legs, limits).await?.into_iter();
        for (index, scan) in scans.iter_mut().enumerate() {
            if scan.done {
                continue;
            }
            let Some(page) = pages.next() else {
                // The typed batch scan returns exactly one page per leg.
                return Err(Error::new(ErrorKind::Backend));
            };
            for item in page.items() {
                visit(index, expect_child_entry_ref(item.value())?)?;
                scan.seen += 1;
            }
            scan.cursor = page.into_next_cursor();
            scan.done = scan.cursor.is_none();
        }
    }
    for (scan, (_, header)) in scans.iter().zip(bodies) {
        if scan.seen != header.entry_count() as usize {
            return Err(Error::new(ErrorKind::Corruption));
        }
    }
    Ok(())
}

/// One vector's nearest-child accumulator over a streamed Child Entry scan:
/// the canonical argmin shared by the single-vector and grouped descents.
#[derive(Clone, Copy, Default)]
struct NearestChild {
    best: Option<(f64, PartitionKey, PartitionKey)>,
}

impl NearestChild {
    /// Considers one scanned Child Entry owned by `owner`.
    fn consider(
        &mut self,
        kernel: &VectorKernel,
        routing: &[f32],
        entry: &ChildEntry,
        owner: PartitionKey,
    ) -> Result<()> {
        let distance = kernel.routing_distance(routing, entry.centroid())?;
        if self.best.is_none_or(|(best_distance, best_child, _)| {
            beats(distance, entry.child(), (best_distance, best_child))
        }) {
            self.best = Some((distance, entry.child(), owner));
        }
        Ok(())
    }

    /// Returns the nearest child and its owning body, failing closed when the
    /// scanned union was empty.
    fn finish(self) -> Result<(PartitionKey, PartitionKey)> {
        self.best
            .map(|(_, child, owner)| (child, owner))
            .ok_or_else(|| Error::new(ErrorKind::Corruption))
    }
}

/// Whether `(distance, child)` displaces the current best under the canonical
/// routing order: nearer distance wins and ties resolve to the smaller
/// Partition Key. IEEE equality makes -0.0 and +0.0 a distance tie, so the
/// Partition Key decides; the comparison is total because every distance is
/// validated finite.
fn beats(distance: f64, child: PartitionKey, best: (f64, PartitionKey)) -> bool {
    distance < best.0 || (distance == best.0 && child < best.1)
}

/// The canonical two-target split decision shared by drain placement and
/// descent redirect: the nearer routing distance wins and ties resolve to the
/// smaller Partition Key, which is stable across workers because
/// `begin_split` reserves the left target first.
pub(crate) fn nearer_of_two(
    left: PartitionKey,
    left_distance: f64,
    right: PartitionKey,
    right_distance: f64,
) -> PartitionKey {
    if beats(right_distance, right, (left_distance, left)) {
        right
    } else {
        left
    }
}

/// Selects the nearest legal merge target for one routing-space vector: a
/// `Ready` candidate other than the source, nearer routing distance first
/// with the Partition Key tie-break — ADR 0008's canonical reselection rule,
/// shared by the merge drain and the write descent's merge redirect.
///
/// Persisted candidate centroids are routing models, so kernel errors here
/// are fail-closed Corruption rather than caller error.
pub(crate) fn nearest_ready_candidate<'c>(
    kernel: &VectorKernel,
    routing: &[f32],
    source: PartitionKey,
    candidates: &'c [topology::LevelCandidate],
) -> Result<Option<&'c topology::LevelCandidate>> {
    let mut best: Option<(f64, PartitionKey, &topology::LevelCandidate)> = None;
    for candidate in candidates {
        if !candidate.is_legal_merge_target(source) {
            continue;
        }
        let distance = kernel
            .routing_distance(routing, candidate.centroid())
            .map_err(|_| Error::new(ErrorKind::Corruption))?;
        if best
            .as_ref()
            .is_none_or(|&(best_distance, best_partition, _)| {
                beats(
                    distance,
                    candidate.partition(),
                    (best_distance, best_partition),
                )
            })
        {
            best = Some((distance, candidate.partition(), candidate));
        }
    }
    Ok(best.map(|(_, _, candidate)| candidate))
}

/// Builds the page bounds for a Child Entry scan during descent.
///
/// The byte limit covers one page of worst-case entries at the index
/// dimension — 28 fixed key bytes plus the canonical Tree Key, and 8 + 4
/// payload bytes plus value framing per entry — so a legal page is never
/// byte-truncated before its item limit.
fn child_scan_limits(dimension: usize) -> Result<ScanLimits> {
    let vector_bytes = dimension
        .checked_mul(4)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let per_item = 28_usize
        .checked_add(MAX_TREE_KEY_BYTES)
        .and_then(|bytes| bytes.checked_add(32))
        .and_then(|bytes| bytes.checked_add(vector_bytes))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let byte_limit = per_item
        .checked_mul(CHILD_SCAN_PAGE)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    Ok(ScanLimits {
        item_limit: CHILD_SCAN_PAGE,
        byte_limit,
    })
}

/// Validates every distinct carried route observation for the mutation that
/// applies next, in one batched update-protected read.
///
/// The update-protected reads establish exactly the conflicts that keep the
/// observed membership legal: each leaf Header, whose count and epoch the
/// mutation changes and whose state must still accept writes (`Ready`,
/// `Splitting`, or `ReceivingSplit`), and the leaf's incoming reference. A
/// directly routed leaf protects its Child Entry at the observed parent (ADR
/// 0007). A drain-redirected leaf instead protects the source's
/// `DrainingSplit` state slot naming it: adjacent-level maintenance may have
/// moved the source edge and the target edge to different parents, so no
/// single parent edge is authoritative for the redirect (ADR 0014). Either
/// way, a concurrent topology change aborts the commit and the retried
/// attempt reroutes from a fresh snapshot. The values still match the carried
/// observations because both were read from this transaction's snapshot; the
/// fail-closed checks guard that contract.
///
/// One snapshot gives each leaf exactly one incoming reference, but a split
/// target is reachable both through its own parent edge and through a
/// draining source's redirect; each distinct reference is validated once, in
/// first-appearance route order.
async fn validate_for_write<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    routes: &[Route],
) -> Result<()> {
    let index = manifest.logical_index_id();
    let mut validated = BTreeSet::new();
    let mut distinct = Vec::new();
    for route in routes {
        if validated.insert((route.leaf(), route.incoming)) {
            distinct.push(route);
        }
    }
    let mut keys = Vec::with_capacity(2 * distinct.len());
    for route in &distinct {
        keys.push(LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: route.leaf(),
        });
        match route.incoming {
            Incoming::ParentEdge => {
                // The tree root has no incoming edge.
                if let Some(parent) = route.parent {
                    keys.push(LogicalKey::ChildEntry {
                        index,
                        tree_key: tree_key.clone(),
                        partition: parent,
                        child: route.leaf(),
                    });
                }
            }
            Incoming::SourceSlot(source) => keys.push(LogicalKey::State {
                index,
                tree_key: tree_key.clone(),
                partition: source,
            }),
        }
    }
    let mut values = txn.batch_get_for_update(keys).await?.into_iter();
    for route in distinct {
        let Some(header_value) = values.next() else {
            // The typed batch read returns exactly one value per input key.
            return Err(Error::new(ErrorKind::Backend));
        };
        match header_value {
            Some(PersistentValue::PartitionHeader(header))
                if header.level() == 1 && header.state().accepts_writes() => {}
            _ => return Err(Error::new(ErrorKind::Corruption)),
        }
        match route.incoming {
            Incoming::ParentEdge => {
                // The tree root has no incoming edge.
                if route.parent.is_some() {
                    let Some(edge_value) = values.next() else {
                        // The typed batch read returns one value per input key.
                        return Err(Error::new(ErrorKind::Backend));
                    };
                    match edge_value {
                        Some(PersistentValue::ChildEntry(entry))
                            if entry.child() == route.leaf() => {}
                        _ => return Err(Error::new(ErrorKind::Corruption)),
                    }
                }
            }
            Incoming::SourceSlot(_) => {
                let Some(state_value) = values.next() else {
                    // The typed batch read returns one value per input key.
                    return Err(Error::new(ErrorKind::Backend));
                };
                match state_value {
                    Some(PersistentValue::PartitionState(PartitionTransition::DrainingSplit {
                        left,
                        right,
                        ..
                    })) if route.leaf() == left || route.leaf() == right => {}
                    _ => return Err(Error::new(ErrorKind::Corruption)),
                }
            }
        }
    }
    Ok(())
}

/// Builds the format-v1 vector kernel for the bound Logical Index.
pub(crate) fn kernel_for(manifest: &IndexManifest) -> Result<VectorKernel> {
    VectorKernel::new(
        manifest.config().dimension(),
        manifest.config().metric(),
        *manifest.rotation_seed(),
    )
}
