//! Record Group membership and Leaf Entry projection agreement.
//!
//! Record Groups sort before every partition body in the index-owned key
//! order, so the audit first accumulates one [`RecordFacts`] entry per
//! Record ID — the record's canonical Leaf Entry fingerprint and the
//! authoritative location — then joins each scanned Leaf Entry against it in
//! constant state per entry. Matched facts are dropped immediately, which
//! both frees memory and turns a second entry for the same Record ID into a
//! dangling-membership finding. Entries left over after the scan name Leaf
//! Entries that do not exist.
//!
//! Group-internal findings (a Location or Opaque Payload without a Vector
//! Record, a Vector Record without a Location) finalize when the
//! record-group key range ends; the groups they name are dropped from the
//! join so one root cause does not cascade.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use bytes::Bytes;

use crate::api::{Error, ErrorKind, PartitionKey, Result, VerifyIssueKind};
use crate::search::numeric::VectorKernel;
use crate::search::rabitq::RaBitQ7;
use crate::storage::keys::TreeKey;
use crate::storage::values::{
    IndexManifest, LeafEntry, PersistentValue, RecordLocation, VectorRecord,
};

use super::{Context, entry_fingerprint};

/// The per-record join facts retained for the Leaf Entry cross-check.
struct RecordFacts {
    /// The canonical Leaf Entry fingerprint of the Vector Record; `None`
    /// when the record cannot be re-quantized (already reported).
    expected: Option<[u8; 16]>,
    /// The authoritative location, when a Record Location exists.
    location: Option<(TreeKey, PartitionKey)>,
    has_record: bool,
    has_payload: bool,
    /// The bytes this entry charges the memory limit.
    charged: u64,
}

impl RecordFacts {
    /// The resident estimate of one map entry holding no location.
    fn base_bytes(id: &Bytes) -> u64 {
        size_of::<(Bytes, RecordFacts)>() as u64 + id.len() as u64
    }
}

/// The bounded Record–Location–Leaf join state of one audit.
pub(super) struct RecordLedger {
    kernel: VectorKernel,
    facts: BTreeMap<Bytes, RecordFacts>,
    /// Whether the record-group key range has been finalized already.
    groups_finalized: bool,
}

impl RecordLedger {
    /// Creates the ledger, deriving the index's quantization kernel. A
    /// Manifest whose config cannot build its kernel is `Corruption`.
    pub(super) fn new(manifest: &IndexManifest) -> Result<Self> {
        let config = manifest.config();
        let kernel = VectorKernel::new(
            config.dimension(),
            config.metric(),
            *manifest.rotation_seed(),
        )
        .map_err(|_| Error::new(ErrorKind::Corruption))?;
        Ok(Self {
            kernel,
            facts: BTreeMap::new(),
            groups_finalized: false,
        })
    }

    /// Returns the mutable facts of one Record ID, inserting and charging an
    /// empty entry on first sight; `None` when the memory limit stopped the
    /// audit. The Record ID clone is a reference-count bump only.
    fn facts_of<'a>(&'a mut self, cx: &mut Context<'_>, id: &Bytes) -> Option<&'a mut RecordFacts> {
        match self.facts.entry(id.clone()) {
            Entry::Occupied(entry) => Some(entry.into_mut()),
            Entry::Vacant(entry) => {
                let charged = RecordFacts::base_bytes(id);
                cx.charge_memory(charged);
                if cx.truncated() {
                    return None;
                }
                Some(entry.insert(RecordFacts {
                    expected: None,
                    location: None,
                    has_record: false,
                    has_payload: false,
                    charged,
                }))
            }
        }
    }

    /// Notes one decoded Vector Record and its expected Leaf Entry
    /// fingerprint.
    pub(super) fn note_record(&mut self, cx: &mut Context<'_>, id: &Bytes, record: &VectorRecord) {
        let expected = self.expected_entry(cx, id, record);
        if let Some(facts) = self.facts_of(cx, id) {
            facts.has_record = true;
            facts.expected = expected;
        }
    }

    /// Notes one decoded Record Location.
    pub(super) fn note_location(
        &mut self,
        cx: &mut Context<'_>,
        id: &Bytes,
        location: RecordLocation,
    ) {
        let (tree_key, leaf) = location.into_parts();
        let Some(facts) = self.facts_of(cx, id) else {
            return;
        };
        if facts.location.is_none() {
            let delta = tree_key.as_bytes().len() as u64;
            cx.charge_memory(delta);
            if cx.truncated() {
                return;
            }
            facts.charged = facts.charged.saturating_add(delta);
        }
        facts.location = Some((tree_key, leaf));
    }

    /// Notes one decoded Opaque Payload.
    pub(super) fn note_payload(&mut self, cx: &mut Context<'_>, id: &Bytes) {
        if let Some(facts) = self.facts_of(cx, id) {
            facts.has_payload = true;
        }
    }

    /// The canonical fingerprint of the Leaf Entry the mutation path would
    /// have written for this record, recomputed from the stored body. Keep
    /// the projection formula in sync with `PreparedRecord::new`
    /// (`src/maintenance/mutation.rs`).
    fn expected_entry(
        &self,
        cx: &mut Context<'_>,
        id: &Bytes,
        record: &VectorRecord,
    ) -> Option<[u8; 16]> {
        let expected = || -> Result<Vec<u8>> {
            let routing = self.kernel.preprocess(record.vector())?;
            let rabitq7 = RaBitQ7::quantize(&routing)?;
            let entry = LeafEntry::new(id.clone(), record.fields().to_vec(), rabitq7);
            cx.codec.encode(&PersistentValue::LeafEntry(entry))
        }();
        let canonical = match expected {
            Ok(canonical) => canonical,
            // A stored record that cannot be re-quantized could never have
            // committed through the mutation path; no Leaf Entry can agree
            // with it.
            Err(_) => {
                cx.issue(
                    VerifyIssueKind::RecordProjectionMismatch,
                    None,
                    None,
                    Some(id.clone()),
                );
                return None;
            }
        };
        // The single bounded scratch buffer counts toward the limit too.
        let bytes = canonical.len() as u64;
        cx.charge_memory(bytes);
        if cx.truncated() {
            return None;
        }
        let fingerprint = entry_fingerprint(&canonical);
        cx.release_memory(bytes);
        Some(fingerprint)
    }

    /// Reports group-internal membership findings once the record-group key
    /// range ends, and drops every group that cannot join cleanly so one
    /// root cause does not cascade into the Leaf Entry cross-check.
    /// Idempotent: the key-order trigger and the end-of-scan fallback race.
    pub(super) fn finalize_groups(&mut self, cx: &mut Context<'_>) {
        if self.groups_finalized {
            return;
        }
        self.groups_finalized = true;
        self.facts.retain(|id, facts| {
            if cx.truncated() {
                return true;
            }
            if !facts.has_record {
                if let Some((tree_key, leaf)) = &facts.location {
                    cx.issue(
                        VerifyIssueKind::Membership,
                        Some(tree_key),
                        Some(*leaf),
                        Some(id.clone()),
                    );
                }
                if facts.has_payload {
                    cx.issue(VerifyIssueKind::Membership, None, None, Some(id.clone()));
                }
            } else if facts.location.is_none() {
                cx.issue(VerifyIssueKind::Membership, None, None, Some(id.clone()));
            }
            let healthy = facts.has_record && facts.location.is_some();
            if !healthy {
                cx.release_memory(facts.charged);
            }
            healthy
        });
    }

    /// Joins one scanned Leaf Entry against the retained record facts.
    pub(super) fn join_leaf_entry(
        &mut self,
        cx: &mut Context<'_>,
        tree_key: &TreeKey,
        partition: PartitionKey,
        id: &Bytes,
        raw_value: &Bytes,
    ) {
        let Some(facts) = self.facts.get(id) else {
            // No Vector Record names this entry (a partial group was already
            // reported when the record range finalized).
            cx.issue(
                VerifyIssueKind::Membership,
                Some(tree_key),
                Some(partition),
                Some(id.clone()),
            );
            return;
        };
        let named = facts
            .location
            .as_ref()
            .is_some_and(|(key, leaf)| key == tree_key && *leaf == partition);
        if !named {
            // The Record Location names another position for this Record ID.
            cx.issue(
                VerifyIssueKind::Membership,
                Some(tree_key),
                Some(partition),
                Some(id.clone()),
            );
            return;
        }
        if facts
            .expected
            .is_some_and(|expected| entry_fingerprint(raw_value) != expected)
        {
            cx.issue(
                VerifyIssueKind::RecordProjectionMismatch,
                Some(tree_key),
                Some(partition),
                Some(id.clone()),
            );
        }
        // Matched: drop the join state, freeing memory and turning any
        // second entry for this Record ID into a dangling finding.
        if let Some(facts) = self.facts.remove(id) {
            cx.release_memory(facts.charged);
        }
    }

    /// Reports every retained facts entry: a Vector Record and its Record
    /// Location whose Leaf Entry never appeared in the scan.
    pub(super) fn finish(self, cx: &mut Context<'_>) {
        for (id, facts) in self.facts {
            if cx.truncated() {
                break;
            }
            if let Some((tree_key, leaf)) = &facts.location {
                cx.issue(
                    VerifyIssueKind::Membership,
                    Some(tree_key),
                    Some(*leaf),
                    Some(id),
                );
            }
        }
    }
}
