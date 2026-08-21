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

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{ReadOps, ScanLimits, WriteTxn};
use crate::storage::keys::{LogicalKey, MAX_TREE_KEY_BYTES, TreeKey};
use crate::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionTransition, PersistentValue, TreeManifest, expect_centroid, expect_child_entry_ref,
    expect_header,
};
use crate::storage::{
    LogicalRange, LogicalReader, ReadLogicalTxn, WriteLogicalTxn, topology, tree_manifest,
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
    let root = ensure_tree(txn, tree_key, started_at_unix_millis).await?;
    let route = descend(txn, manifest, &kernel, tree_key, root, &routing).await?;
    validate_for_write(txn, manifest, tree_key, std::slice::from_ref(&route)).await?;
    Ok(route)
}

/// Routes one batch of preprocessed routing vectors through one tree for a
/// foreground write.
///
/// The batch shares one Tree Manifest read and one read per visited internal
/// partition instead of re-descending per record. The returned Routes
/// correspond to the input vectors by index, and every *distinct* routed leaf
/// is validated once in one batched update-protected read: its Header and
/// incoming reference are update-protected, so a concurrent topology change
/// aborts the commit and the whole attempt reroutes from a fresh snapshot.
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
    validate_for_write(txn, manifest, tree_key, &routes).await?;
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
        match resolve_hop(reader, manifest, tree_key, root, partition, expected_level).await? {
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
                for &(body, header) in &bodies {
                    scan_children_with(reader, manifest, tree_key, body, header, &mut |entry| {
                        nearest.consider(kernel, routing, entry, body)
                    })
                    .await?;
                }
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
        }
    }
}

/// Descends from `root` with the whole group, reading each visited partition
/// once and returning one Route per input vector in input order. The same
/// shared state-machine body ([`resolve_hop`]) and fail-closed level and
/// state contract as [`descend`] apply.
async fn descend_grouped<R: LogicalReader>(
    reader: &mut R,
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
        match resolve_hop(reader, manifest, tree_key, root, partition, expected_level).await? {
            Hop::Leaf(header) => {
                for member in members {
                    routes[member] = Some(Route {
                        leaf: partition,
                        leaf_header: header,
                        parent,
                        incoming: Incoming::ParentEdge,
                    });
                }
            }
            Hop::DrainRedirect {
                source,
                left,
                right,
            } => {
                for member in members {
                    let target = nearer_redirect_target(kernel, routings[member], &left, &right)?;
                    routes[member] = Some(Route {
                        leaf: target.partition,
                        leaf_header: target.header,
                        parent,
                        incoming: Incoming::SourceSlot(source),
                    });
                }
            }
            Hop::Children { bodies, next_level } => {
                // Partition the members by nearest child across the bodies'
                // exact Child Entry union — streamed per body, never
                // materialized — then descend per group. A draining family's
                // union is never empty, and a stable or splitting internal
                // partition never has zero children, so every member resolves.
                let mut nearest = vec![NearestChild::default(); members.len()];
                for &(body, header) in &bodies {
                    scan_children_with(reader, manifest, tree_key, body, header, &mut |entry| {
                        for (slot, &member) in nearest.iter_mut().zip(&members) {
                            slot.consider(kernel, routings[member], entry, body)?;
                        }
                        Ok(())
                    })
                    .await?;
                }
                let mut grouped: BTreeMap<PartitionKey, (PartitionKey, Vec<usize>)> =
                    BTreeMap::new();
                for (slot, member) in nearest.into_iter().zip(members) {
                    let (child, owner) = slot.finish()?;
                    grouped
                        .entry(child)
                        .or_insert_with(|| (owner, Vec::new()))
                        .1
                        .push(member);
                }
                for (child, (owner, members)) in grouped {
                    pending.push((child, Some(owner), Some(next_level), members));
                }
            }
            Hop::Sideways(source) => {
                // The source shares the target's level; the carried parent is
                // informational — the next hop below the internal source
                // replaces it.
                pending.push((source, parent, expected_level, members));
            }
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
}

/// One draining leaf's redirect target with its validated authority values.
struct DrainTarget {
    partition: PartitionKey,
    header: PartitionHeader,
    centroid: PartitionCentroid,
}

/// Resolves one visited partition's descent hop: the single split-aware
/// state-machine body serving both the single-vector read descent and the
/// grouped write descent. The merge state machine (#31) extends exactly this
/// match.
///
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
) -> Result<Hop> {
    let index = manifest.logical_index_id();
    let (header, state) = topology::read_authority(reader, index, tree_key, partition).await?;
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
                PartitionTransition::Splitting { .. } => {
                    // The source still holds every entry; the target is empty
                    // and contributes nothing.
                    Ok(Hop::Sideways(source))
                }
                PartitionTransition::DrainingSplit { left, right, .. } => Ok(Hop::Children {
                    bodies: split_family_bodies(
                        reader,
                        manifest,
                        tree_key,
                        source,
                        source_header,
                        [left, right],
                        header.level(),
                    )
                    .await?,
                    next_level: header.level() - 1,
                }),
                _ => Err(Error::new(ErrorKind::Corruption)),
            }
        }
        PartitionTransition::DrainingSplit { left, right, .. } => {
            if header.level() == 1 {
                return drain_redirect(reader, manifest, tree_key, partition, left, right).await;
            }
            // The exact union of the source body and both targets covers the
            // source's original children.
            Ok(Hop::Children {
                bodies: split_family_bodies(
                    reader,
                    manifest,
                    tree_key,
                    partition,
                    header,
                    [left, right],
                    header.level(),
                )
                .await?,
                next_level: header.level() - 1,
            })
        }
        // Merging is unreachable before the merge state machine (#31) and
        // stays fail-closed.
        _ => Err(Error::new(ErrorKind::Corruption)),
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

/// Validates and lists one draining split family's bodies: the source first,
/// then its two targets, each proven at the family's level with its expected
/// state. Both target Headers are read in one batch.
async fn split_family_bodies<R: LogicalReader>(
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
    let index = manifest.logical_index_id();
    let mut values = reader
        .batch_get(
            targets
                .map(|target| LogicalKey::Header {
                    index,
                    tree_key: tree_key.clone(),
                    partition: target,
                })
                .into(),
        )
        .await?
        .into_iter();
    let mut bodies = Vec::with_capacity(3);
    bodies.push((source, source_header));
    for target in targets {
        let Some(value) = values.next() else {
            // The typed batch read returns exactly one value per input key.
            return Err(Error::new(ErrorKind::Backend));
        };
        let header = expect_header(value)?.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
        if header.level() != level || header.state() != PartitionState::ReceivingSplit {
            return Err(Error::new(ErrorKind::Corruption));
        }
        bodies.push((target, header));
    }
    Ok(bodies)
}

/// Scans one internal body's complete Child Entry set in bounded pages,
/// visiting each entry by reference without materializing it.
///
/// The Header count is exact in every committed state, so the scan must see
/// precisely that many Child Entries; any mismatch is Corruption. An empty
/// result is legal only for a not-yet-filled split target or a fully drained
/// source.
async fn scan_children_with<R: LogicalReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    header: PartitionHeader,
    visit: &mut impl FnMut(&ChildEntry) -> Result<()>,
) -> Result<()> {
    let range = LogicalRange::child_entries(manifest, tree_key, partition)?;
    let limits = child_scan_limits(manifest.config().dimension())?;
    let mut seen = 0_usize;
    let mut cursor = None;
    loop {
        let page = reader.scan(&range, cursor.as_ref(), limits).await?;
        for item in page.items() {
            visit(expect_child_entry_ref(item.value())?)?;
            seen += 1;
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    if seen != header.entry_count() as usize {
        return Err(Error::new(ErrorKind::Corruption));
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
