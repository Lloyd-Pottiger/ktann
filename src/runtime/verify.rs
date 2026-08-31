//! Bounded read-only verification of one consistent backend snapshot.
//!
//! [`verify`] implements the `Index::verify` audit of ADR 0019 and
//! `docs/design/runtime-operations.md` §6. Exactly one backend read
//! transaction pins the snapshot: the persisted Active Manifest validates
//! first, and every later check — canonical encodings, tree reachability and
//! unique incoming references, exact Header counts and legal State
//! references, Record–Location–Leaf membership, Leaf Entry projection
//! agreement, conservative Synopses, and allocator high-water marks — reads
//! that single snapshot. Transaction expiry, deadline, and cancellation
//! return errors rather than any cross-snapshot conclusion, so a large audit
//! on a backend with a short snapshot lifetime (FoundationDB) runs against a
//! caller-provided offline copy instead.
//!
//! One ordered scan of the index-owned key space is an ordered merge:
//! Record Groups sort before Tree Manifests and partition bodies, and one
//! partition's bodies sort Header, Synopsis, State, Centroid, then Leaf and
//! Child Entries. The cross-range join state is the per-record facts map
//! plus the current tree's topology accumulator; every buffer, set, and
//! transient sort state charges the resident-memory limit. Reaching the
//! issue, object, or memory limit stops the audit and returns the collected
//! issues with `complete: false`; finalization never runs on a truncated
//! pass, so a partial walk cannot invent findings. The audit never writes,
//! repairs, spills into the index, samples, or continues from a token.
//!
//! Issue kinds are deliberately coarse; the stable mapping is:
//!
//! - [`VerifyIssueKind::InvalidEncoding`]: a key or value fails canonical
//!   decoding.
//! - [`VerifyIssueKind::Reachability`]: a required object is missing (a
//!   partition Header or State, a leaf's Synopsis, a tree root, an edge
//!   target, the allocator) or an unowned object exists (a partition without
//!   a Tree Manifest, an unreachable partition, an internal partition's
//!   Synopsis, an illegal State reference).
//! - [`VerifyIssueKind::Membership`]: a Child Entry edge or Record Group
//!   membership is invalid or duplicated.
//! - [`VerifyIssueKind::CountMismatch`]: a stored exact count or high-water
//!   mark disagrees with its logical contents.
//! - [`VerifyIssueKind::RecordProjectionMismatch`]: a Leaf Entry's fields or
//!   RaBitQ7 code disagree with its Vector Record.
//! - [`VerifyIssueKind::SynopsisNotConservative`]: a stored leaf Synopsis
//!   does not cover the contents recomputed from the Leaf Entries.
//!
//! Issues carry only allowlisted identifiers: the Logical Index ID, a
//! domain-separated hash of the Tree Key, the Partition Key, and an optional
//! Record ID. Findings without Tree Key or partition context — namespace and
//! record-group findings — use the zero hash and Partition Key 1 sentinels.

use bytes::Bytes;
use xxhash_rust::xxh3::xxh3_128_with_seed;

use crate::api::{
    DataType, LogicalIndexId, PartitionKey, Result, VerifyIssue, VerifyIssueKind,
    VerifyObjectCounts, VerifyOptions, VerifyReport, VerifyTopology,
};
use crate::maintenance::fixup;
use crate::observe::metrics;
use crate::storage::backend::{Backend, ReadOps, ScanItem, ScanLimits};
use crate::storage::keys::{self, KeyRange, LogicalKey, TreeKey, tree_key_hash};
use crate::storage::topology::root_partition;
use crate::storage::values::{IndexManifest, PersistentValue, ValueCodec};

use super::OperationContext;
use super::reads::open_validated_read;

mod limits;
mod records;
mod topology;

use limits::Limits;
use records::RecordLedger;
use topology::{LeafEntryItem, TopologyLedger};

/// Domain-separates the Leaf Entry projection fingerprint.
const ENTRY_FINGERPRINT_DOMAIN: u64 = 0x4b54_414e_4e01_b1a2;

/// The page bound of one verification scan step; paging loops until the
/// index-owned range is exhausted or a limit stops the audit.
const VERIFY_SCAN: ScanLimits = ScanLimits {
    item_limit: 256,
    byte_limit: 1 << 20,
};

/// Runs the bounded read-only audit of one Logical Index in one snapshot.
pub(crate) async fn verify<B: Backend>(
    context: &mut OperationContext<B>,
    manifest: &IndexManifest,
    options: VerifyOptions,
) -> Result<VerifyReport> {
    context.checkpoint()?;
    let backend = context.backend();
    let txn = open_validated_read(backend.as_ref(), manifest).await?;
    let mut raw = txn.into_raw();

    let mut cx = Context::new(manifest, &options);
    check_allocator(&mut cx, &mut raw).await?;
    scan_index(&mut cx, context, &mut raw).await?;
    let report = cx.finish();
    metrics::verify_report(&report);
    Ok(report)
}

/// The shared per-audit state: resource limits, the issue sink, and the
/// decoded-object counters every ledger reports through.
pub(super) struct Context<'m> {
    /// The opened handle's validated Manifest.
    pub(super) manifest: &'m IndexManifest,
    /// The canonical value decoder bound to the Manifest.
    codec: ValueCodec<'m>,
    /// The ordered Tree Key field types required by key decoding.
    tree_key_types: Box<[DataType]>,
    limits: Limits,
    issues: Vec<VerifyIssue>,
    counts: VerifyObjectCounts,
    topology: VerifyTopology,
    truncated: bool,
}

impl<'m> Context<'m> {
    fn new(manifest: &'m IndexManifest, options: &VerifyOptions) -> Self {
        let (types, type_count) = manifest.tree_key_types();
        Self {
            manifest,
            codec: ValueCodec::for_index(manifest),
            tree_key_types: types[..type_count].into(),
            limits: Limits::new(options),
            issues: Vec::new(),
            counts: VerifyObjectCounts::default(),
            topology: VerifyTopology::default(),
            truncated: false,
        }
    }

    fn index(&self) -> LogicalIndexId {
        self.manifest.logical_index_id()
    }

    fn tree_key_types(&self) -> &[DataType] {
        &self.tree_key_types
    }

    /// Whether a reached limit has already stopped the audit.
    pub(super) fn truncated(&self) -> bool {
        self.truncated
    }

    /// Charges one visited key–value pair, including a malformed one: the
    /// object limit bounds total work while the report counts only decoded
    /// objects.
    fn charge_object(&mut self) {
        if self.truncated || !self.limits.charge_object() {
            self.truncated = true;
        }
    }

    /// Reserves resident buffer bytes, stopping the audit at the memory
    /// limit. Estimates follow the Partition Cache idiom: fixed struct sizes
    /// plus exact byte lengths where the type exposes them.
    pub(super) fn charge_memory(&mut self, bytes: u64) {
        if self.truncated || !self.limits.charge_memory(bytes) {
            self.truncated = true;
        }
    }

    /// Releases previously reserved resident buffer bytes.
    pub(super) fn release_memory(&mut self, bytes: u64) {
        self.limits.release_memory(bytes);
    }

    /// Appends one redacted issue, stopping the audit at the issue limit.
    pub(super) fn issue(
        &mut self,
        kind: VerifyIssueKind,
        tree_key: Option<&TreeKey>,
        partition: Option<PartitionKey>,
        record_id: Option<Bytes>,
    ) {
        if self.truncated {
            return;
        }
        let bytes =
            size_of::<VerifyIssue>() as u64 + record_id.as_ref().map_or(0, |id| id.len() as u64);
        if !self.limits.charge_memory(bytes) || !self.limits.take_issue_slot() {
            self.truncated = true;
            return;
        }
        self.issues.push(VerifyIssue {
            kind,
            logical_index_id: self.index(),
            tree_key_hash: tree_key.map(tree_key_hash).unwrap_or([0; 32]),
            partition_key: partition.unwrap_or_else(root_partition),
            record_id,
        });
    }

    /// Counts one successfully decoded object by category.
    fn count(&mut self, value: &PersistentValue) {
        self.counts.total = self.counts.total.saturating_add(1);
        match value {
            PersistentValue::VectorRecord(_) => {
                self.counts.vector_records = self.counts.vector_records.saturating_add(1);
            }
            PersistentValue::RecordLocation(_) => {
                self.counts.record_locations = self.counts.record_locations.saturating_add(1);
            }
            PersistentValue::TreeManifest(_) => {
                self.topology.trees = self.topology.trees.saturating_add(1);
            }
            PersistentValue::PartitionHeader(header) => {
                self.counts.partitions = self.counts.partitions.saturating_add(1);
                self.topology.partitions = self.topology.partitions.saturating_add(1);
                self.topology.max_level = Some(
                    self.topology
                        .max_level
                        .map_or(header.level(), |level| level.max(header.level())),
                );
                let count = self
                    .topology
                    .partitions_by_level
                    .entry(header.level())
                    .or_default();
                *count = count.saturating_add(1);
                let entries = self
                    .topology
                    .entries_by_level
                    .entry(header.level())
                    .or_default();
                *entries = entries.saturating_add(u64::from(header.entry_count()));
                let maximum = self
                    .topology
                    .max_entries_by_level
                    .entry(header.level())
                    .or_default();
                *maximum = (*maximum).max(header.entry_count());
                let states = &mut self.topology.partition_states;
                let count = match header.state() {
                    crate::storage::values::PartitionState::Ready => &mut states.ready,
                    crate::storage::values::PartitionState::Splitting => &mut states.splitting,
                    crate::storage::values::PartitionState::ReceivingSplit => {
                        &mut states.receiving_split
                    }
                    crate::storage::values::PartitionState::DrainingSplit => {
                        &mut states.draining_split
                    }
                    crate::storage::values::PartitionState::Merging => &mut states.merging,
                };
                *count = count.saturating_add(1);
            }
            PersistentValue::PartitionSynopsis(_)
            | PersistentValue::PartitionState(_)
            | PersistentValue::PartitionCentroid(_) => {
                self.counts.partitions = self.counts.partitions.saturating_add(1);
            }
            PersistentValue::LeafEntry(_) | PersistentValue::ChildEntry(_) => {
                self.counts.entries = self.counts.entries.saturating_add(1);
            }
            _ => {}
        }
    }

    /// Counts a Header whose durable state can advance Structure Maintenance.
    pub(super) fn note_actionable_partition(
        &mut self,
        partition: PartitionKey,
        header: crate::storage::values::PartitionHeader,
    ) {
        if fixup::is_actionable(self.manifest.config(), partition, header) {
            self.topology.actionable_partitions =
                self.topology.actionable_partitions.saturating_add(1);
        }
    }

    fn finish(self) -> VerifyReport {
        VerifyReport {
            complete: !self.truncated,
            issues: self.issues,
            objects: self.counts,
            topology: self.topology,
        }
    }
}

/// The canonical Leaf Entry projection fingerprint compared across the
/// Record–Location–Leaf join. Decode accepts only canonical bytes, so the
/// scanned bytes of a decoded Leaf Entry equal the canonical encoding of the
/// expected entry.
pub(super) fn entry_fingerprint(canonical: &[u8]) -> [u8; 16] {
    xxh3_128_with_seed(canonical, ENTRY_FINGERPRINT_DOMAIN).to_le_bytes()
}

/// Checks the namespace allocator high-water mark against this Logical
/// Index ID, reading the same snapshot as the walk.
async fn check_allocator<T: ReadOps>(cx: &mut Context<'_>, raw: &mut T) -> Result<()> {
    cx.charge_object();
    if cx.truncated() {
        return Ok(());
    }
    let bytes = raw.get(Bytes::from(keys::index_id_allocator_key())).await?;
    match bytes {
        Some(bytes) => {
            let decoded = ValueCodec::bootstrap().decode(&LogicalKey::IndexIdAllocator, bytes);
            match decoded {
                Ok(PersistentValue::IndexIdAllocator(allocator)) => {
                    cx.count(&PersistentValue::IndexIdAllocator(allocator));
                    if cx.index().get() > allocator.high_water() {
                        cx.issue(VerifyIssueKind::CountMismatch, None, None, None);
                    }
                }
                // A typed decode derives the value family from the key, so a
                // mismatched family is unreachable; malformed bytes fail.
                Ok(_) | Err(_) => cx.issue(VerifyIssueKind::InvalidEncoding, None, None, None),
            }
        }
        None => cx.issue(VerifyIssueKind::Reachability, None, None, None),
    }
    Ok(())
}

/// Scans the index-owned key space in one ordered pass, dispatching every
/// decoded object to its ledger, then runs the cross-range finalization.
async fn scan_index<B: Backend, T: ReadOps>(
    cx: &mut Context<'_>,
    context: &OperationContext<B>,
    raw: &mut T,
) -> Result<()> {
    let mut records = RecordLedger::new(cx.manifest)?;
    let mut topology = TopologyLedger::new(cx.manifest);

    let range = keys::index_range(cx.index());
    let end = range.end().to_vec();
    let mut start = range.start().to_vec();
    while !cx.truncated() {
        context.checkpoint()?;
        let page = raw
            .scan(&KeyRange::new(start, end.clone()), VERIFY_SCAN)
            .await?;
        let page_bytes = page.items().iter().map(page_item_bytes).sum();
        cx.charge_memory(page_bytes);
        for item in page.items() {
            cx.charge_object();
            if cx.truncated() {
                break;
            }
            process_item(cx, &mut records, &mut topology, item);
        }
        cx.release_memory(page_bytes);
        match page.next_start() {
            Some(next) => start = next.to_vec(),
            None => break,
        }
    }
    // A truncated pass stops where it is: finalizing partial accumulators
    // would report findings the unvisited remainder could contradict.
    if cx.truncated() {
        return Ok(());
    }
    records.finalize_groups(cx);
    topology.finish(cx);
    if !cx.truncated() {
        records.finish(cx);
    }
    Ok(())
}

/// Decodes and dispatches one scanned item to the ledgers.
fn process_item(
    cx: &mut Context<'_>,
    records: &mut RecordLedger,
    topology: &mut TopologyLedger,
    item: &ScanItem,
) {
    let key = match keys::decode_key(cx.tree_key_types(), item.key()) {
        Ok(key) => key,
        Err(_) => {
            cx.issue(VerifyIssueKind::InvalidEncoding, None, None, None);
            return;
        }
    };
    let value = match cx.codec.decode(&key, item.value().clone()) {
        Ok(value) => value,
        Err(_) => {
            let tree_key = key.tree_key();
            let (partition, record_id) = key_context(&key);
            cx.issue(
                VerifyIssueKind::InvalidEncoding,
                tree_key,
                partition,
                record_id,
            );
            return;
        }
    };
    cx.count(&value);
    // Record Groups sort after the Manifest and before Tree Manifests and
    // partition bodies; their group checks finalize when the first
    // tree-scoped kind arrives (idempotent).
    if matches!(
        key,
        LogicalKey::TreeManifest { .. }
            | LogicalKey::Header { .. }
            | LogicalKey::Synopsis { .. }
            | LogicalKey::State { .. }
            | LogicalKey::Centroid { .. }
            | LogicalKey::LeafEntry { .. }
            | LogicalKey::ChildEntry { .. }
    ) {
        records.finalize_groups(cx);
    }
    match (&key, value) {
        (LogicalKey::Record { id, .. }, PersistentValue::VectorRecord(record)) => {
            records.note_record(cx, id, &record);
        }
        (LogicalKey::Location { id, .. }, PersistentValue::RecordLocation(location)) => {
            records.note_location(cx, id, location);
        }
        (LogicalKey::Payload { id, .. }, PersistentValue::OpaquePayload(_)) => {
            records.note_payload(cx, id);
        }
        (
            LogicalKey::TreeManifest { tree_key, .. },
            PersistentValue::TreeManifest(tree_manifest),
        ) => {
            topology.note_tree_manifest(cx, tree_key, tree_manifest);
        }
        (
            LogicalKey::Header {
                tree_key,
                partition,
                ..
            },
            PersistentValue::PartitionHeader(header),
        ) => {
            topology.absorb_header(cx, tree_key, *partition, header);
        }
        (
            LogicalKey::Synopsis {
                tree_key,
                partition,
                ..
            },
            PersistentValue::PartitionSynopsis(synopsis),
        ) => {
            topology.absorb_synopsis(cx, tree_key, *partition, synopsis);
        }
        (
            LogicalKey::State {
                tree_key,
                partition,
                ..
            },
            PersistentValue::PartitionState(transition),
        ) => {
            topology.absorb_state(cx, tree_key, *partition, transition);
        }
        (
            LogicalKey::Centroid {
                tree_key,
                partition,
                ..
            },
            PersistentValue::PartitionCentroid(_),
        ) => {
            topology.absorb_centroid(cx, tree_key, *partition);
        }
        (
            LogicalKey::LeafEntry {
                tree_key,
                partition,
                id,
                ..
            },
            PersistentValue::LeafEntry(entry),
        ) => {
            topology.absorb_leaf_entry(
                cx,
                records,
                LeafEntryItem {
                    tree_key,
                    partition: *partition,
                    id,
                    entry: &entry,
                    raw_value: item.value(),
                },
            );
        }
        (
            LogicalKey::ChildEntry {
                tree_key,
                partition,
                child,
                ..
            },
            PersistentValue::ChildEntry(_),
        ) => {
            topology.absorb_child_entry(cx, tree_key, *partition, *child);
        }
        // The Manifest was validated before the scan; namespace keys cannot
        // sort inside the index-owned range, and a successful typed decode
        // always yields the key's value family.
        (LogicalKey::Manifest(_), _)
        | (LogicalKey::IndexIdAllocator, _)
        | (LogicalKey::IndexNameDirectory(_), _) => {}
        _ => {
            let (partition, record_id) = key_context(&key);
            cx.issue(
                VerifyIssueKind::InvalidEncoding,
                key.tree_key(),
                partition,
                record_id,
            );
        }
    }
}

/// Extracts the safe partition and Record ID context of a decoded key.
fn key_context(key: &LogicalKey) -> (Option<PartitionKey>, Option<Bytes>) {
    let partition = match key {
        LogicalKey::Header { partition, .. }
        | LogicalKey::Synopsis { partition, .. }
        | LogicalKey::State { partition, .. }
        | LogicalKey::Centroid { partition, .. }
        | LogicalKey::LeafEntry { partition, .. }
        | LogicalKey::ChildEntry { partition, .. } => Some(*partition),
        _ => None,
    };
    let record_id = match key {
        LogicalKey::Record { id, .. }
        | LogicalKey::Location { id, .. }
        | LogicalKey::Payload { id, .. }
        | LogicalKey::LeafEntry { id, .. } => Some(id.clone()),
        _ => None,
    };
    (partition, record_id)
}

/// Charges one scanned page's actual encoded bytes.
fn page_item_bytes(item: &ScanItem) -> u64 {
    (item.key().len() as u64).saturating_add(item.value().len() as u64)
}
