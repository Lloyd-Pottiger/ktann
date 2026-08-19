//! Initial tree routing through stable internal and leaf partitions.
//!
//! Routing descends one tree of a Logical Index from its stable root
//! Partition Key 1 to the Leaf Partition nearest to a preprocessed routing
//! vector. Only the stable `Ready` topology exists before the split/merge
//! state machines (#10, #31); every other persisted state is unreachable and
//! therefore rejected as Corruption rather than routed through.
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
//! - **Bounded depth and work.** The root Header's level bounds the descent:
//!   every hop must descend exactly one level and a leaf is exactly level 1
//!   (ADR 0006), so hop count is bounded by the persisted root level and a
//!   level that fails to decrement — including any cycle — is Corruption.
//!   Each hop reads one Header and scans at most one extra Child Entry beyond
//!   the binary fanout to prove the fanout invariant.
//! - **Fail closed.** A missing Header, a wrong-kind value, a fanout other
//!   than two, a level mismatch, a non-Ready state, or a malformed centroid
//!   is Corruption; malformed caller vectors are InvalidArgument.

use std::future::Future;

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{ReadOps, ScanLimits, WriteTxn};
use crate::storage::keys::{LogicalKey, MAX_TREE_KEY_BYTES, TreeKey};
use crate::storage::values::{
    IndexManifest, PartitionHeader, PartitionState, PersistentValue, TreeManifest,
};
use crate::storage::{
    LogicalRange, LogicalScanCursor, LogicalScanPage, ReadLogicalTxn, WriteLogicalTxn,
    tree_manifest,
};

/// The exact Child Entry count of a stable internal partition; binary fanout
/// is a fixed format-v1 protocol choice.
const INTERNAL_FANOUT: usize = 2;

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
    route_leaf_for_write_preprocessed(txn, tree_key, &kernel, &routing, started_at_unix_millis)
        .await
}

/// Routes an already preprocessed routing vector through one tree for a
/// foreground write.
///
/// `routing` must be the exact output of [`VectorKernel::preprocess`] under
/// `kernel` for the bound Logical Index: the mutation pipeline preprocesses
/// and quantizes each caller vector once and reuses the same routing vector
/// here. The same empty-root growth and write validation contract as
/// [`route_leaf_for_write`] applies.
pub(crate) async fn route_leaf_for_write_preprocessed<T: WriteTxn>(
    txn: &mut WriteLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    kernel: &VectorKernel,
    routing: &[f32],
    started_at_unix_millis: u64,
) -> Result<Route> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    if routing.len() != manifest.config().dimension() {
        return Err(Error::invalid_argument());
    }
    let root = ensure_tree(txn, tree_key, started_at_unix_millis).await?;
    let route = descend(txn, manifest, kernel, tree_key, root, routing).await?;
    validate_for_write(txn, manifest, tree_key, &route).await?;
    Ok(route)
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

/// Descends from `root` to the leaf nearest to `routing`, carrying the
/// observed parent.
///
/// The hop count is bounded by the persisted root level: every hop must
/// descend exactly one level, so a level that fails to decrement — including
/// any cycle — is Corruption before the descent can repeat a partition.
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
        let header = read_header(reader, manifest, tree_key, partition).await?;
        if let Some(expected) = expected_level {
            if header.level() != expected {
                return Err(Error::new(ErrorKind::Corruption));
            }
        }
        if header.state() != PartitionState::Ready {
            return Err(Error::new(ErrorKind::Corruption));
        }
        if header.level() == 1 {
            return Ok(Route {
                leaf: partition,
                leaf_header: header,
                parent,
            });
        }
        expected_level = Some(header.level() - 1);
        let child = nearest_child(reader, manifest, kernel, tree_key, partition, routing).await?;
        parent = Some(partition);
        partition = child;
    }
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
    match reader.get(key).await? {
        Some(PersistentValue::PartitionHeader(header)) => Ok(header),
        // A Header key decodes only as a Partition Header, so another value
        // kind is unreachable; a missing Header on a referenced partition is
        // Corruption either way.
        _ => Err(Error::new(ErrorKind::Corruption)),
    }
}

/// Selects the nearest child of one stable internal partition for `routing`.
///
/// A stable internal partition holds exactly [`INTERNAL_FANOUT`] Child
/// Entries; the scan admits one extra entry solely to detect a fanout
/// violation, and any other count is Corruption. Distance ties resolve to the
/// smaller child Partition Key, the canonical routing tie-breaker.
async fn nearest_child<R: RoutingReader>(
    reader: &mut R,
    manifest: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    partition: PartitionKey,
    routing: &[f32],
) -> Result<PartitionKey> {
    let range = LogicalRange::child_entries(manifest, tree_key, partition)?;
    let limits = child_scan_limits(manifest.config().dimension())?;
    let mut count = 0_usize;
    let mut best: Option<(f64, PartitionKey)> = None;
    let mut cursor = None;
    loop {
        let page = reader.scan(&range, cursor.as_ref(), limits).await?;
        for item in page.items() {
            let PersistentValue::ChildEntry(entry) = item.value() else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
            if count > INTERNAL_FANOUT {
                return Err(Error::new(ErrorKind::Corruption));
            }
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
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    if count != INTERNAL_FANOUT {
        return Err(Error::new(ErrorKind::Corruption));
    }
    best.map(|(_, child)| child)
        .ok_or_else(|| Error::new(ErrorKind::Corruption))
}

/// Builds the scan bounds that prove the internal fanout invariant.
///
/// The item limit admits one extra Child Entry beyond the binary fanout. The
/// byte limit covers that many worst-case entries at the index dimension —
/// 28 fixed key bytes plus the canonical Tree Key, and 8 + 4 payload bytes
/// plus value framing per entry — so a legal page is never byte-truncated and
/// the fanout check always sees every Child Entry.
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
        .checked_mul(INTERNAL_FANOUT + 1)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    Ok(ScanLimits {
        item_limit: INTERNAL_FANOUT + 1,
        byte_limit,
    })
}

/// Validates a carried route observation for the mutation that applies next.
///
/// The update-protected reads establish exactly the conflicts that keep the
/// observed membership legal: the leaf Header, whose count and epoch the
/// mutation changes, and the leaf's incoming Child Entry, which adjacent-level
/// maintenance must move or remove to change the leaf's parent (ADR 0007). The
/// values still match the carried observation because both were read from this
/// transaction's snapshot; the fail-closed checks guard that contract.
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
            if header.level() == 1 && header.state() == PartitionState::Ready => {}
        _ => return Err(Error::new(ErrorKind::Corruption)),
    }
    if let Some(parent) = route.parent {
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

    fn scan(
        &mut self,
        range: &LogicalRange,
        cursor: Option<&LogicalScanCursor>,
        limits: ScanLimits,
    ) -> impl Future<Output = Result<LogicalScanPage>> + Send {
        WriteLogicalTxn::scan(self, range, cursor, limits)
    }
}
