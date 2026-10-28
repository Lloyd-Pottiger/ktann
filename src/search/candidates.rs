//! Deferred field ownership across local and global candidate selection.
//!
//! Keep one shared entry array per admitted leaf, without its expanded RaBitQ
//! codes. Candidates carry only array indices until global selection completes.

use std::sync::Arc;

use crate::api::{Error, ErrorKind, Result};
use crate::storage::values::RecordLocation;

use super::cache::{CachedLeafEntry, leaf_entry_bytes};
use super::rabitq::{ApproximateCandidate, select_global_overlap};
use super::rerank::LeafCandidate;

/// An entry identity within the retained snapshot projections.
pub(crate) struct LeafReference {
    leaf: usize,
    entry: usize,
}

/// One admitted leaf's immutable projections and authoritative source location.
struct LeafSource {
    entries: Arc<[CachedLeafEntry]>,
    location: RecordLocation,
}

/// Owns the complete locally selected pool until global selection has finished.
#[derive(Default)]
pub(crate) struct LeafCandidatePool {
    candidates: Vec<ApproximateCandidate<LeafReference>>,
    leaves: Vec<LeafSource>,
    retained_bytes: usize,
}

impl LeafCandidatePool {
    /// Exposes candidate identities without materializing their filter fields.
    pub(crate) fn candidates(&self) -> &[ApproximateCandidate<LeafReference>] {
        &self.candidates
    }

    /// Retains every local survivor and one shared projection array for its leaf.
    pub(crate) fn push_leaf(
        &mut self,
        entries: &Arc<[CachedLeafEntry]>,
        location: RecordLocation,
        selected: Vec<ApproximateCandidate<usize>>,
        account_memory: bool,
    ) -> Result<()> {
        if selected.is_empty() {
            return Ok(());
        }
        if account_memory {
            let bytes = entries.iter().try_fold(64_u64, |bytes, entry| {
                bytes.checked_add(leaf_entry_bytes(entry))
            });
            self.retained_bytes = bytes
                .and_then(|bytes| bytes.checked_add(location.tree_key().as_bytes().len() as u64))
                .and_then(|bytes| usize::try_from(bytes).ok())
                .and_then(|bytes| self.retained_bytes.checked_add(bytes))
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        }
        self.candidates
            .try_reserve(selected.len())
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        self.leaves
            .try_reserve(1)
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let leaf = self.leaves.len();
        self.leaves.push(LeafSource {
            entries: Arc::clone(entries),
            location,
        });
        self.candidates
            .extend(selected.into_iter().map(|candidate| {
                let (id, distance, entry) = candidate.into_parts();
                ApproximateCandidate::new(id, distance, LeafReference { leaf, entry })
            }));
        Ok(())
    }

    /// Charges retained projection arrays once per leaf, including uncached ones.
    pub(crate) fn heap_bytes(&self) -> Result<usize> {
        self.candidates
            .capacity()
            .checked_mul(size_of::<ApproximateCandidate<LeafReference>>())
            .and_then(|bytes| {
                bytes.checked_add(
                    self.leaves
                        .capacity()
                        .checked_mul(size_of::<LeafSource>())?,
                )
            })
            .and_then(|bytes| bytes.checked_add(self.retained_bytes))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))
    }

    /// Applies the unchanged global selector, then copies only surviving fields.
    pub(crate) fn select(
        self,
        k: usize,
        rerank_budget: usize,
    ) -> Result<(Vec<LeafCandidate>, bool)> {
        let selection = select_global_overlap(self.candidates, k, rerank_budget)?;
        let truncated = selection.truncated();
        let selected = selection
            .into_candidates()
            .into_iter()
            .map(|candidate| {
                let (id, distance, reference) = candidate.into_parts();
                let source = &self.leaves[reference.leaf];
                LeafCandidate::from_leaf(
                    id,
                    source.entries[reference.entry].fields().into(),
                    distance,
                    source.location.clone(),
                )
            })
            .collect();
        Ok((selected, truncated))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use crate::api::{DataType, PartitionKey, Value};
    use crate::storage::keys::TreeKey;
    use crate::storage::values::{LeafEntry, RecordLocation};

    use super::super::cache::CachedLeafEntry;
    use super::super::rabitq::{
        ApproximateCandidate, RaBitQ7, select_global_overlap, test_approximate_distance,
    };
    use super::super::rerank::LeafCandidate;
    use super::LeafCandidatePool;

    #[test]
    fn deferred_fields_match_eager_selection_after_source_release() {
        for budget in [1, 7, 100] {
            let mut pool = LeafCandidatePool::default();
            let mut eager = Vec::new();
            let mut sources = Vec::new();
            for leaf in 1..=3 {
                let mut codes = Vec::new();
                let entries: Arc<[CachedLeafEntry]> = (0..12)
                    .map(|i| {
                        CachedLeafEntry::from_leaf(
                            LeafEntry::new(
                                Bytes::copy_from_slice(&[leaf as u8, i]),
                                vec![Value::String(format!("leaf-{leaf}-entry-{i}"))],
                                RaBitQ7::quantize(&[f32::from(i); 4]).expect("code"),
                            ),
                            4,
                            &mut codes,
                        )
                    })
                    .collect::<Vec<_>>()
                    .into();
                let location = RecordLocation::new(
                    TreeKey::encode(&[DataType::I64], &[Value::I64(leaf as i64)])
                        .expect("tree key"),
                    PartitionKey::new(leaf).expect("partition"),
                );
                // Non-contiguous original ordinals model exact predicate filtering.
                let selected: Vec<_> = [1, 4, 7, 10]
                    .into_iter()
                    .map(|i| {
                        let entry = &entries[i];
                        let distance = test_approximate_distance((i % 3) as f64);
                        eager.push(LeafCandidate::from_leaf(
                            entry.record_id().clone(),
                            entry.fields().into(),
                            distance,
                            location.clone(),
                        ));
                        ApproximateCandidate::new(entry.record_id().clone(), distance, i)
                    })
                    .collect();
                pool.push_leaf(&entries, location, selected, true)
                    .expect("admit every local survivor");
                assert_eq!(Arc::strong_count(&entries), 2, "one owner per leaf");
                sources.push(Arc::downgrade(&entries));
                drop(entries);
                drop(codes);
            }
            assert_eq!(pool.candidates().len(), 12, "no early global pruning");
            assert!(pool.heap_bytes().expect("memory charge") > 0);
            let expected = select_global_overlap(eager, 5, budget).expect("eager oracle");
            let (actual, truncated) = pool.select(5, budget).expect("deferred selection");
            assert_eq!(truncated, expected.truncated());
            let expected = expected.into_candidates();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.record_id(), expected.record_id());
                assert_eq!(actual.distance(), expected.distance());
                assert_eq!(actual.fields(), expected.fields());
                assert_eq!(actual.location(), expected.location());
            }
            assert!(sources.iter().all(|source| source.upgrade().is_none()));
        }
    }
}
