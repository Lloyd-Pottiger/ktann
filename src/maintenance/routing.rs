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
//! the batch's records.
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
//! - **Bounded depth and work.** The root Header's level bounds the descent:
//!   every hop must descend exactly one level and a leaf is exactly level 1
//!   (ADR 0006), so hop count is bounded by the persisted root level and a
//!   level that fails to decrement — including any cycle — is Corruption.
//!   Each internal body scanned contributes its exact Header count of Child
//!   Entries in bounded pages; a count mismatch is Corruption.
//! - **Fail closed.** A missing Header, State, or centroid, a wrong-kind
//!   value, a level mismatch, a header/state disagreement, an incomplete
//!   split family, or a malformed centroid is Corruption; a `Merging`
//!   partition is unreachable before the merge state machine (#31) and is
//!   likewise rejected; malformed caller vectors are InvalidArgument.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{ReadOps, ScanLimits, WriteTxn};
use crate::storage::keys::{LogicalKey, MAX_TREE_KEY_BYTES, TreeKey};
use crate::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionTransition, PersistentValue, TreeManifest, expect_centroid, expect_header,
    expect_state,
};
use crate::storage::{
    LogicalRange, LogicalScanCursor, LogicalScanPage, ReadLogicalTxn, WriteLogicalTxn,
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
}

/// Routes one caller vector through one tree on a read snapshot.
///
/// Returns `None` when the Tree Key has no tree yet; reads never create one.
/// The vector is validated and preprocessed (metric normalization and the
/// persisted rotation) exactly as the write path preprocesses it, so both
/// paths descend identically on the same snapshot.
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
/// whole attempt reroutes from a fresh snapshot.
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
    let mut routes = route_leaves_for_write_preprocessed(
        txn,
        tree_key,
        &kernel,
        &[&routing],
        started_at_unix_millis,
    )
    .await?;
    // The batched contract yields exactly one route per input vector.
    routes.pop().ok_or_else(|| Error::new(ErrorKind::Backend))
}

/// Routes one batch of preprocessed routing vectors through one tree for a
/// foreground write.
///
/// The batch shares one Tree Manifest read and one read per visited internal
/// partition instead of re-descending per record. The returned Routes
/// correspond to the input vectors by index, and every *distinct* routed leaf
/// is validated once: its Header and incoming reference are update-protected,
/// so a concurrent topology change aborts the commit and the whole attempt
/// reroutes from a fresh snapshot.
///
/// Every routing vector must be the exact output of
/// [`VectorKernel::preprocess`] under `kernel` for the bound Logical Index.
pub(crate) async fn route_leaves_for_write_preprocessed<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    kernel: &VectorKernel,
    routings: &[&[f32]],
    started_at_unix_millis: u64,
) -> Result<Vec<Route>> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    if routings
        .iter()
        .any(|routing| routing.len() != manifest.config().dimension())
    {
        return Err(Error::invalid_argument());
    }
    if routings.is_empty() {
        return Ok(Vec::new());
    }
    let root = ensure_tree(txn, tree_key, started_at_unix_millis).await?;
    let routes = descend_grouped(txn, manifest, kernel, tree_key, root, routings).await?;
    let mut validated = BTreeSet::new();
    for route in &routes {
        // One snapshot gives each leaf exactly one incoming reference, but a
        // split target is reachable both through its own parent edge and
        // through a draining source's redirect; each distinct reference is
        // validated once.
        if validated.insert((route.leaf(), route.incoming)) {
            validate_for_write(txn, manifest, tree_key, route).await?;
        }
    }
    Ok(routes)
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

/// Descends from `root` to the write-accepting leaf nearest to `routing`,
/// carrying the observed parent.
///
/// Split-family resolution stays within one level: a `ReceivingSplit`
/// internal partition hops sideways to its source's family at most once,
/// because targets cannot themselves split until their source drains.
async fn descend<R: RoutingReader>(
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
        let (header, state) = read_authority(reader, manifest, tree_key, partition).await?;
        check_level(&header, expected_level)?;
        // The root is the one searchable entry point: it is never a split
        // target and never merges, matching the search traversal contract.
        if partition == root
            && matches!(
                state,
                PartitionTransition::ReceivingSplit { .. } | PartitionTransition::Merging { .. }
            )
        {
            return Err(Error::new(ErrorKind::Corruption));
        }
        // A write-accepting leaf — Ready, Splitting, or ReceivingSplit — is
        // the descent's end.
        if header.level() == 1 && header.state().accepts_writes() {
            return Ok(Route {
                leaf: partition,
                leaf_header: header,
                parent,
                incoming: Incoming::ParentEdge,
            });
        }
        match state {
            PartitionTransition::Ready { .. } | PartitionTransition::Splitting { .. } => {
                // A Splitting source still holds its complete child set; its
                // empty targets are correctly skipped until draining starts.
                expected_level = Some(header.level() - 1);
                let child = nearest_child(
                    reader, manifest, kernel, tree_key, partition, header, routing,
                )
                .await?;
                parent = Some(partition);
                partition = child;
            }
            PartitionTransition::ReceivingSplit { source, .. } => {
                // An internal target is never descended alone: its source
                // still owns the unmigrated children, so the split family is
                // resolved through the persisted source reference.
                match read_state(reader, manifest, tree_key, source).await? {
                    PartitionTransition::Splitting { .. } => {
                        // The source still holds every entry; the target is
                        // empty and contributes nothing. The carried parent is
                        // informational here: the next descent hop below the
                        // internal source replaces it.
                        partition = source;
                    }
                    PartitionTransition::DrainingSplit { left, right, .. } => {
                        let source_header = read_header(reader, manifest, tree_key, source).await?;
                        expected_level = Some(header.level() - 1);
                        let family = SplitFamily {
                            source,
                            source_header,
                            targets: [left, right],
                        };
                        let (child, owner) = nearest_child_across(
                            reader,
                            manifest,
                            kernel,
                            tree_key,
                            family,
                            header.level(),
                            routing,
                        )
                        .await?;
                        parent = Some(owner);
                        partition = child;
                    }
                    _ => return Err(Error::new(ErrorKind::Corruption)),
                }
            }
            PartitionTransition::DrainingSplit { left, right, .. } => {
                if header.level() == 1 {
                    // A draining leaf accepts no writes: redirect to the
                    // nearer persisted target. The target shares the source's
                    // observed parent — or, for a root split, the root target
                    // slot — so the carried parent observation stays valid.
                    let target = nearer_target_leaf(
                        reader, manifest, kernel, tree_key, routing, left, right,
                    )
                    .await?;
                    let target_header = read_header(reader, manifest, tree_key, target).await?;
                    if target_header.level() != 1
                        || target_header.state() != PartitionState::ReceivingSplit
                    {
                        return Err(Error::new(ErrorKind::Corruption));
                    }
                    return Ok(Route {
                        leaf: target,
                        leaf_header: target_header,
                        parent,
                        incoming: Incoming::SourceSlot(partition),
                    });
                }
                // The exact union of the source body and both targets covers
                // the source's original children.
                expected_level = Some(header.level() - 1);
                let family = SplitFamily {
                    source: partition,
                    source_header: header,
                    targets: [left, right],
                };
                let (child, owner) = nearest_child_across(
                    reader,
                    manifest,
                    kernel,
                    tree_key,
                    family,
                    header.level(),
                    routing,
                )
                .await?;
                parent = Some(owner);
                partition = child;
            }
            // Merging is unreachable before the merge state machine (#31) and
            // stays fail-closed.
            _ => return Err(Error::new(ErrorKind::Corruption)),
        }
    }
}

/// Descends from `root` with the whole group, reading each visited partition
/// once and returning one Route per input vector in input order. The same
/// fail-closed level and state contract as [`descend`] applies.
async fn descend_grouped<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    root: PartitionKey,
    routings: &[&[f32]],
) -> Result<Vec<Route>> {
    let mut routes: Vec<Option<Route>> = vec![None; routings.len()];
    // (partition, observed parent, expected level, member indexes).
    let mut pending = vec![(
        root,
        None,
        None,
        (0..routings.len()).collect::<Vec<usize>>(),
    )];
    while let Some((partition, parent, expected_level, members)) = pending.pop() {
        let (header, state) = read_authority(txn, manifest, tree_key, partition).await?;
        check_level(&header, expected_level)?;
        // The root is the one searchable entry point: it is never a split
        // target and never merges, matching the search traversal contract.
        if partition == root
            && matches!(
                state,
                PartitionTransition::ReceivingSplit { .. } | PartitionTransition::Merging { .. }
            )
        {
            return Err(Error::new(ErrorKind::Corruption));
        }
        if header.level() == 1 && header.state().accepts_writes() {
            for member in members {
                routes[member] = Some(Route {
                    leaf: partition,
                    leaf_header: header,
                    parent,
                    incoming: Incoming::ParentEdge,
                });
            }
            continue;
        }
        // The candidate child bodies for this hop: the partition itself for
        // stable or splitting descent, or its whole draining split family.
        let mut bodies: Vec<(PartitionKey, PartitionHeader)> = Vec::new();
        match state {
            PartitionTransition::Ready { .. } | PartitionTransition::Splitting { .. } => {
                bodies.push((partition, header));
            }
            PartitionTransition::ReceivingSplit { source, .. } => {
                match read_state(txn, manifest, tree_key, source).await? {
                    PartitionTransition::Splitting { .. } => {
                        // The source still holds every entry; requeue the
                        // group at it. The source shares the target's level;
                        // the carried parent is informational — the next hop
                        // below the internal source replaces it.
                        pending.push((source, parent, expected_level, members));
                        continue;
                    }
                    PartitionTransition::DrainingSplit { left, right, .. } => {
                        let source_header = read_header(txn, manifest, tree_key, source).await?;
                        bodies = split_family_bodies(
                            txn,
                            manifest,
                            tree_key,
                            source,
                            source_header,
                            [left, right],
                            header.level(),
                        )
                        .await?;
                    }
                    _ => return Err(Error::new(ErrorKind::Corruption)),
                }
            }
            PartitionTransition::DrainingSplit { left, right, .. } => {
                if header.level() == 1 {
                    // A draining leaf accepts no writes: each member redirects
                    // to its nearer persisted target, sharing the source's
                    // observed parent or root target slot.
                    let left_centroid = read_centroid(txn, manifest, tree_key, left).await?;
                    let right_centroid = read_centroid(txn, manifest, tree_key, right).await?;
                    let left_header = read_header(txn, manifest, tree_key, left).await?;
                    let right_header = read_header(txn, manifest, tree_key, right).await?;
                    for target_header in [left_header, right_header] {
                        if target_header.level() != 1
                            || target_header.state() != PartitionState::ReceivingSplit
                        {
                            return Err(Error::new(ErrorKind::Corruption));
                        }
                    }
                    for member in members {
                        let left_distance = kernel
                            .routing_distance(routings[member], left_centroid.components())?;
                        let right_distance = kernel
                            .routing_distance(routings[member], right_centroid.components())?;
                        let target = nearer_of_two(left, left_distance, right, right_distance);
                        let target_header = if target == left {
                            left_header
                        } else {
                            right_header
                        };
                        routes[member] = Some(Route {
                            leaf: target,
                            leaf_header: target_header,
                            parent,
                            incoming: Incoming::SourceSlot(partition),
                        });
                    }
                    continue;
                }
                bodies = split_family_bodies(
                    txn,
                    manifest,
                    tree_key,
                    partition,
                    header,
                    [left, right],
                    header.level(),
                )
                .await?;
            }
            // Merging is unreachable before the merge state machine (#31) and
            // stays fail-closed.
            _ => return Err(Error::new(ErrorKind::Corruption)),
        }

        // Gather the exact Child Entry union of the candidate bodies; a
        // draining family's union is never empty, and a stable or splitting
        // internal partition never has zero children.
        let mut children: Vec<(PartitionKey, PartitionKey, ChildEntry)> = Vec::new();
        for (body, body_header) in &bodies {
            for entry in scan_children(txn, manifest, tree_key, *body, *body_header).await? {
                children.push((entry.child(), *body, entry));
            }
        }
        if children.is_empty() {
            return Err(Error::new(ErrorKind::Corruption));
        }

        // Partition the members by nearest child, then descend per group.
        let next_level = Some(header.level() - 1);
        let mut grouped: BTreeMap<PartitionKey, (PartitionKey, Vec<usize>)> = BTreeMap::new();
        for member in members {
            let mut best: Option<(f64, PartitionKey, PartitionKey)> = None;
            for (child, owner, entry) in &children {
                let distance = kernel.routing_distance(routings[member], entry.centroid())?;
                let nearer = match best {
                    None => true,
                    Some((best_distance, best_child, _)) => {
                        distance < best_distance
                            || (distance == best_distance && *child < best_child)
                    }
                };
                if nearer {
                    best = Some((distance, *child, *owner));
                }
            }
            let (_, child, owner) = best.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
            grouped
                .entry(child)
                .or_insert_with(|| (owner, Vec::new()))
                .1
                .push(member);
        }
        for (child, (owner, members)) in grouped {
            pending.push((child, Some(owner), next_level, members));
        }
    }
    // Every member is resolved exactly once: each partition visit either
    // assigns its members at a write-accepting leaf or redirects them to a
    // target leaf, requeues them at a splitting source, or splits them across
    // the children of the candidate bodies.
    routes
        .into_iter()
        .map(|route| route.ok_or_else(|| Error::new(ErrorKind::Backend)))
        .collect()
}

/// Reads one partition Header, failing closed when a routed-to partition has
/// no Header.
async fn read_header<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<PartitionHeader> {
    let key = LogicalKey::Header {
        index: manifest.logical_index_id(),
        tree_key: tree_key.clone(),
        partition,
    };
    // A Header key decodes only as a Partition Header, so another value kind
    // is unreachable; a missing Header on a referenced partition is Corruption
    // either way.
    expect_header(reader.get(key).await?)?.ok_or_else(|| Error::new(ErrorKind::Corruption))
}

/// Reads one visited partition's Header and State in one batch, failing
/// closed when either is missing, of the wrong kind, or in disagreement.
///
/// Every reachable partition carries both authority values in every committed
/// state: creation installs them together, and completion removes them
/// together with the partition's last incoming reference.
async fn read_authority<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<(PartitionHeader, PartitionTransition)> {
    let index = manifest.logical_index_id();
    let mut values = reader
        .batch_get(vec![
            LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition,
            },
            LogicalKey::State {
                index,
                tree_key: tree_key.clone(),
                partition,
            },
        ])
        .await?
        .into_iter();
    let (Some(header_value), Some(state_value)) = (values.next(), values.next()) else {
        // The typed batch read returns exactly one value per input key.
        return Err(Error::new(ErrorKind::Backend));
    };
    let header = expect_header(header_value)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    let state = expect_state(state_value)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
    if header.state() != state.state() {
        return Err(Error::new(ErrorKind::Corruption));
    }
    Ok((header, state))
}

/// Reads one partition State, failing closed when a routed-to partition has
/// no State or a wrong-kind value.
async fn read_state<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<PartitionTransition> {
    expect_state(
        reader
            .get(LogicalKey::State {
                index: manifest.logical_index_id(),
                tree_key: tree_key.clone(),
                partition,
            })
            .await?,
    )?
    .ok_or_else(|| Error::new(ErrorKind::Corruption))
}

/// Reads one partition's persisted centroid, failing closed when absent.
async fn read_centroid<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<PartitionCentroid> {
    expect_centroid(
        reader
            .get(LogicalKey::Centroid {
                index: manifest.logical_index_id(),
                tree_key: tree_key.clone(),
                partition,
            })
            .await?,
    )?
    .ok_or_else(|| Error::new(ErrorKind::Corruption))
}

/// One draining split family: the source with its already-read Header and
/// the two persisted targets.
struct SplitFamily {
    source: PartitionKey,
    source_header: PartitionHeader,
    targets: [PartitionKey; 2],
}

/// Validates and lists one draining split family's bodies: the source first,
/// then its two targets, each proven at the family's level with its expected
/// state.
async fn split_family_bodies<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    source_header: PartitionHeader,
    targets: [PartitionKey; 2],
    level: u32,
) -> Result<Vec<(PartitionKey, PartitionHeader)>> {
    if source_header.level() != level || source_header.state() != PartitionState::DrainingSplit {
        return Err(Error::new(ErrorKind::Corruption));
    }
    let mut bodies = vec![(source, source_header)];
    for target in targets {
        let header = read_header(reader, manifest, tree_key, target).await?;
        if header.level() != level || header.state() != PartitionState::ReceivingSplit {
            return Err(Error::new(ErrorKind::Corruption));
        }
        bodies.push((target, header));
    }
    Ok(bodies)
}

/// Scans one internal body's complete Child Entry set in bounded pages.
///
/// The Header count is exact in every committed state, so the scan must see
/// precisely that many Child Entries; any mismatch is Corruption. An empty
/// result is legal only for a not-yet-filled split target or a fully drained
/// source.
async fn scan_children<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    header: PartitionHeader,
) -> Result<Vec<ChildEntry>> {
    let range = LogicalRange::child_entries(manifest, tree_key, partition)?;
    let limits = child_scan_limits(manifest.config().dimension())?;
    let mut entries = Vec::with_capacity(header.entry_count().min(1_024) as usize);
    let mut cursor = None;
    loop {
        let page = reader.scan(&range, cursor.as_ref(), limits).await?;
        for item in page.items() {
            let PersistentValue::ChildEntry(entry) = item.value() else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            entries.push(entry.clone());
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    if entries.len() != header.entry_count() as usize {
        return Err(Error::new(ErrorKind::Corruption));
    }
    Ok(entries)
}

/// Selects the nearest child of one stable or splitting internal partition.
///
/// Distance ties resolve to the smaller child Partition Key, the canonical
/// routing tie-breaker. A `Ready` or `Splitting` internal partition never has
/// an empty child set.
async fn nearest_child<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    partition: PartitionKey,
    header: PartitionHeader,
    routing: &[f32],
) -> Result<PartitionKey> {
    let mut best: Option<(f64, PartitionKey)> = None;
    for entry in scan_children(reader, manifest, tree_key, partition, header).await? {
        let distance = kernel.routing_distance(routing, entry.centroid())?;
        let nearer = match best {
            None => true,
            Some((best_distance, best_child)) => {
                // IEEE equality makes -0.0 and +0.0 a distance tie, so the
                // Partition Key decides. The comparison is total because
                // every distance is validated finite.
                distance < best_distance
                    || (distance == best_distance && entry.child() < best_child)
            }
        };
        if nearer {
            best = Some((distance, entry.child()));
        }
    }
    best.map(|(_, child)| child)
        .ok_or_else(|| Error::new(ErrorKind::Corruption))
}

/// Selects the nearest child across one draining split family's exact Child
/// Entry union, returning the child and the body that owns it.
async fn nearest_child_across<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    family: SplitFamily,
    level: u32,
    routing: &[f32],
) -> Result<(PartitionKey, PartitionKey)> {
    let SplitFamily {
        source,
        source_header,
        targets,
    } = family;
    let bodies = split_family_bodies(
        reader,
        manifest,
        tree_key,
        source,
        source_header,
        targets,
        level,
    )
    .await?;
    let mut best: Option<(f64, PartitionKey, PartitionKey)> = None;
    for (body, header) in bodies {
        for entry in scan_children(reader, manifest, tree_key, body, header).await? {
            let distance = kernel.routing_distance(routing, entry.centroid())?;
            let nearer = match best {
                None => true,
                Some((best_distance, best_child, _)) => {
                    distance < best_distance
                        || (distance == best_distance && entry.child() < best_child)
                }
            };
            if nearer {
                best = Some((distance, entry.child(), body));
            }
        }
    }
    best.map(|(_, child, owner)| (child, owner))
        .ok_or_else(|| Error::new(ErrorKind::Corruption))
}

/// Chooses the nearer of one draining leaf's two persisted targets.
///
/// The persisted target centroids are routing models learned at exposure;
/// exact movement, not centroid freshness, preserves membership (ADR 0014).
async fn nearer_target_leaf<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    routing: &[f32],
    left: PartitionKey,
    right: PartitionKey,
) -> Result<PartitionKey> {
    let left_centroid = read_centroid(reader, manifest, tree_key, left).await?;
    let right_centroid = read_centroid(reader, manifest, tree_key, right).await?;
    let left_distance = kernel.routing_distance(routing, left_centroid.components())?;
    let right_distance = kernel.routing_distance(routing, right_centroid.components())?;
    Ok(nearer_of_two(left, left_distance, right, right_distance))
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
    if right_distance < left_distance || (right_distance == left_distance && right < left) {
        right
    } else {
        left
    }
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

/// Validates a carried route observation for the mutation that applies next.
///
/// The update-protected reads establish exactly the conflicts that keep the
/// observed membership legal: the leaf Header, whose count and epoch the
/// mutation changes and whose state must still accept writes (`Ready`,
/// `Splitting`, or `ReceivingSplit`), and the leaf's incoming reference. A
/// directly routed leaf protects its Child Entry at the observed parent (ADR
/// 0007). A drain-redirected leaf instead protects the source's
/// `DrainingSplit` state slot naming it: adjacent-level maintenance may have
/// moved the source edge and the target edge to different parents, so no
/// single parent edge is authoritative for the redirect (ADR 0014). Either
/// way, a concurrent topology change aborts the commit and the retried
/// attempt reroutes from a fresh snapshot. The values still match the carried
/// observation because both were read from this transaction's snapshot; the
/// fail-closed checks guard that contract.
async fn validate_for_write<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    route: &Route,
) -> Result<()> {
    let index = manifest.logical_index_id();
    let header = txn
        .get_for_update(LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition: route.leaf,
        })
        .await?;
    match header {
        Some(PersistentValue::PartitionHeader(header))
            if header.level() == 1 && header.state().accepts_writes() => {}
        _ => return Err(Error::new(ErrorKind::Corruption)),
    }
    match route.incoming {
        Incoming::ParentEdge => {
            let Some(parent) = route.parent else {
                // The tree root has no incoming edge.
                return Ok(());
            };
            let edge = txn
                .get_for_update(LogicalKey::ChildEntry {
                    index,
                    tree_key: tree_key.clone(),
                    partition: parent,
                    child: route.leaf,
                })
                .await?;
            match edge {
                Some(PersistentValue::ChildEntry(entry)) if entry.child() == route.leaf => {}
                _ => return Err(Error::new(ErrorKind::Corruption)),
            }
            Ok(())
        }
        Incoming::SourceSlot(source) => match txn
            .get_for_update(LogicalKey::State {
                index,
                tree_key: tree_key.clone(),
                partition: source,
            })
            .await?
        {
            Some(PersistentValue::PartitionState(PartitionTransition::DrainingSplit {
                left,
                right,
                ..
            })) if route.leaf == left || route.leaf == right => Ok(()),
            _ => Err(Error::new(ErrorKind::Corruption)),
        },
    }
}

/// Builds the format-v1 vector kernel for the bound Logical Index.
pub(crate) fn kernel_for(manifest: &IndexManifest) -> Result<VectorKernel> {
    VectorKernel::new(
        manifest.config().dimension(),
        manifest.config().metric(),
        *manifest.rotation_seed(),
    )
}

/// The read surface a routing descent needs from either transaction kind.
///
/// Both logical transaction wrappers expose identical typed `get`/`scan`
/// operations; this trait lets one descent implementation serve the read-only
/// and the write path without duplicating the topology logic.
trait RoutingReader {
    fn get(
        &mut self,
        key: LogicalKey,
    ) -> impl Future<Output = Result<Option<PersistentValue>>> + Send;
    fn batch_get(
        &mut self,
        keys: Vec<LogicalKey>,
    ) -> impl Future<Output = Result<Vec<Option<PersistentValue>>>> + Send;
    fn scan(
        &mut self,
        range: &LogicalRange,
        cursor: Option<&LogicalScanCursor>,
        limits: ScanLimits,
    ) -> impl Future<Output = Result<LogicalScanPage>> + Send;
}

impl<T: ReadOps> RoutingReader for ReadLogicalTxn<'_, T> {
    fn get(
        &mut self,
        key: LogicalKey,
    ) -> impl Future<Output = Result<Option<PersistentValue>>> + Send {
        ReadLogicalTxn::get(self, key)
    }

    fn batch_get(
        &mut self,
        keys: Vec<LogicalKey>,
    ) -> impl Future<Output = Result<Vec<Option<PersistentValue>>>> + Send {
        ReadLogicalTxn::batch_get(self, keys)
    }

    fn scan(
        &mut self,
        range: &LogicalRange,
        cursor: Option<&LogicalScanCursor>,
        limits: ScanLimits,
    ) -> impl Future<Output = Result<LogicalScanPage>> + Send {
        ReadLogicalTxn::scan(self, range, cursor, limits)
    }
}

impl<T: WriteTxn> RoutingReader for WriteLogicalTxn<'_, T> {
    fn get(
        &mut self,
        key: LogicalKey,
    ) -> impl Future<Output = Result<Option<PersistentValue>>> + Send {
        WriteLogicalTxn::get(self, key)
    }

    fn batch_get(
        &mut self,
        keys: Vec<LogicalKey>,
    ) -> impl Future<Output = Result<Vec<Option<PersistentValue>>>> + Send {
        WriteLogicalTxn::batch_get(self, keys)
    }

    fn scan(
        &mut self,
        range: &LogicalRange,
        cursor: Option<&LogicalScanCursor>,
        limits: ScanLimits,
    ) -> impl Future<Output = Result<LogicalScanPage>> + Send {
        WriteLogicalTxn::scan(self, range, cursor, limits)
    }
}
