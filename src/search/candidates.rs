//! Global rough-distance top-budget selection across every visited leaf.
//!
//! A max heap bounds projection ownership independently of leaf count. Record
//! identities move into a separate list to detect corrupt duplicate membership,
//! including duplicates of candidates already evicted from the heap.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::{Arc, Weak};

use bytes::Bytes;

use crate::api::{Error, ErrorKind, Result};
use crate::storage::values::RecordLocation;

use super::cache::{CachedLeafEntry, leaf_entry_bytes};
use super::numeric::compare_finite;
use super::rabitq::ApproximateCandidate;
use super::rerank::{CandidateLeaf, LeafCandidate, LeafProjection};

/// Retains only globally competitive candidates and their snapshot projections.
#[derive(Default)]
pub(crate) struct LeafCandidatePool {
    candidates: Vec<LeafCandidate>,
    ids: Vec<Bytes>,
    retained_bytes: usize,
    truncated: bool,
    id_bytes: usize,
}

/// The worst candidate is the heap root; ties follow unsigned Record ID order.
fn compare<T, U>(a: &ApproximateCandidate<T>, b: &ApproximateCandidate<U>) -> Ordering {
    compare_finite(a.distance().rough(), b.distance().rough())
        .then_with(|| a.record_id().cmp(b.record_id()))
}

impl LeafCandidatePool {
    #[cfg(test)]
    pub(crate) fn truncated(&self) -> bool {
        self.truncated
    }

    /// Exposes the bounded pool for diagnostics, without guaranteeing order.
    pub(crate) fn candidates(&self) -> &[LeafCandidate] {
        &self.candidates
    }

    /// Scores are admitted globally, without a separate per-leaf cutoff.
    pub(crate) fn push_leaf(
        &mut self,
        entries: &Arc<[CachedLeafEntry]>,
        location: RecordLocation,
        scored: Vec<ApproximateCandidate<usize>>,
        budget: usize,
        account_memory: bool,
    ) -> Result<()> {
        // Grow only the compact ID buffer while traversing. The duplicate table
        // is allocated once, from the actual count, after traversal finishes.
        self.ids
            .try_reserve(scored.len())
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let mut source: Weak<CandidateLeaf> = Weak::new();
        for candidate in scored {
            self.id_bytes = self
                .id_bytes
                .checked_add(candidate.record_id().len())
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            if self.candidates.len() == budget {
                self.truncated = true;
                if budget == 0 || compare(&candidate, &self.candidates[0]) != Ordering::Less {
                    let (id, _, _) = candidate.into_parts();
                    self.ids.push(id);
                    continue;
                }
            }
            // Only an admitted candidate acquires a projection reference. Weak
            // ownership lets eviction release a leaf even during its own scan.
            let source = match source.upgrade() {
                Some(source) => source,
                None => {
                    let bytes = if account_memory {
                        entries
                            .iter()
                            .try_fold(size_of::<CandidateLeaf>() + 32, |bytes, entry| {
                                usize::try_from(leaf_entry_bytes(entry))
                                    .ok()
                                    .and_then(|size| bytes.checked_add(size))
                            })
                            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?
                    } else {
                        0
                    };
                    self.retained_bytes = self
                        .retained_bytes
                        .checked_add(bytes)
                        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
                    let shared = Arc::new(CandidateLeaf {
                        entries: Arc::clone(entries),
                        location: location.clone(),
                        bytes,
                    });
                    source = Arc::downgrade(&shared);
                    shared
                }
            };
            let (id, distance, entry) = candidate.into_parts();
            self.ids.push(id.clone());
            let candidate =
                ApproximateCandidate::new(id, distance, LeafProjection::Shared { source, entry });
            if self.candidates.len() < budget {
                self.candidates
                    .try_reserve(1)
                    .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
                self.candidates.push(candidate);
                let mut child = self.candidates.len() - 1;
                while child > 0 {
                    let parent = (child - 1) / 2;
                    if compare(&self.candidates[child], &self.candidates[parent])
                        != Ordering::Greater
                    {
                        break;
                    }
                    self.candidates.swap(child, parent);
                    child = parent;
                }
            } else {
                let old = std::mem::replace(&mut self.candidates[0], candidate);
                match old.value() {
                    LeafProjection::Shared { source, .. } => {
                        if Arc::strong_count(source) == 1 {
                            self.retained_bytes -= source.bytes;
                        }
                    }
                    #[cfg(test)]
                    LeafProjection::Owned { .. } => unreachable!("pool admits shared projections"),
                }
                drop(old);
                let mut parent = 0;
                loop {
                    let mut child = parent * 2 + 1;
                    if child >= self.candidates.len() {
                        break;
                    }
                    if child + 1 < self.candidates.len()
                        && compare(&self.candidates[child + 1], &self.candidates[child])
                            == Ordering::Greater
                    {
                        child += 1;
                    }
                    if compare(&self.candidates[parent], &self.candidates[child]) != Ordering::Less
                    {
                        break;
                    }
                    self.candidates.swap(parent, child);
                    parent = child;
                }
            }
        }
        Ok(())
    }

    /// Checks every visited eligible identity before exact reads, then releases
    /// the scratch storage. Borrowed keys require no additional ID references.
    pub(crate) fn validate_membership(&mut self) -> Result<()> {
        let mut seen = HashSet::new();
        seen.try_reserve(self.ids.len())
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        for id in &self.ids {
            if !seen.insert(id) {
                return Err(Error::new(ErrorKind::Corruption));
            }
        }
        drop(seen);
        self.ids = Vec::new();
        self.id_bytes = 0;
        Ok(())
    }

    /// Charges retained arrays, ID ownership, and the transient duplicate table.
    /// Two slots per ID, rounded up, conservatively cover table load-factor
    /// slack; the extra cache line covers control bytes for small tables.
    pub(crate) fn heap_bytes(&self) -> Result<usize> {
        let table_bytes = if self.ids.is_empty() {
            Some(0)
        } else {
            self.ids
                .len()
                .checked_mul(2)
                .and_then(usize::checked_next_power_of_two)
                .and_then(|slots| slots.checked_mul(size_of::<&Bytes>() + 1))
                .and_then(|bytes| bytes.checked_add(64))
        };
        self.candidates
            .capacity()
            .checked_mul(size_of::<LeafCandidate>())
            .and_then(|bytes| bytes.checked_add(self.retained_bytes))
            .and_then(|bytes| {
                bytes.checked_add(self.ids.capacity().checked_mul(size_of::<Bytes>())?)
            })
            .and_then(|bytes| bytes.checked_add(self.id_bytes))
            .and_then(|bytes| bytes.checked_add(table_bytes?))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))
    }

    /// Excludes nonoverlapping retained candidates, then orders exact rerank work.
    pub(crate) fn select(mut self, k: usize) -> Result<(Vec<LeafCandidate>, bool)> {
        if k == 0 {
            return Err(Error::invalid_argument());
        }
        if self.candidates.len() >= k {
            let (_, threshold, _) = self.candidates.select_nth_unstable_by(k - 1, |a, b| {
                a.distance().upper().total_cmp(&b.distance().upper())
            });
            let upper = threshold.distance().upper();
            self.candidates
                .retain(|candidate| candidate.distance().lower() <= upper);
        }
        self.candidates.sort_unstable_by(compare);
        Ok((self.candidates, self.truncated))
    }
}

#[cfg(test)]
mod tests {
    use super::super::rabitq::{RaBitQ7, test_approximate_distance};
    use super::*;
    use crate::api::{PartitionKey, Value};
    use crate::storage::keys::TreeKey;
    use crate::storage::values::LeafEntry;

    fn leaf(n: u8) -> (Arc<[CachedLeafEntry]>, RecordLocation) {
        let mut codes = Vec::new();
        let entries = (0..20)
            .map(|i| {
                CachedLeafEntry::from_leaf(
                    LeafEntry::new(
                        Bytes::copy_from_slice(&[n, i]),
                        vec![Value::String(format!("{n}-{i}"))],
                        RaBitQ7::quantize(&[f32::from(i); 4]).unwrap(),
                    ),
                    4,
                    &mut codes,
                )
            })
            .collect::<Vec<_>>()
            .into();
        (
            entries,
            RecordLocation::new(
                TreeKey::encode(&[], &[]).unwrap(),
                PartitionKey::new(u64::from(n) + 1).unwrap(),
            ),
        )
    }

    #[test]
    fn bounded_heap_matches_global_sort_and_shares_original_fields() {
        for budget in [0, 1, 7, 125, 1000] {
            for reverse in [false, true] {
                let mut pool = LeafCandidatePool::default();
                let mut expected = Vec::new();
                let mut sources = Vec::new();
                for n in if reverse {
                    (0..30).rev().collect::<Vec<_>>()
                } else {
                    (0..30).collect()
                } {
                    let (entries, location) = leaf(n);
                    let scored = [1, 4, 7, 10, 13, 19]
                        .into_iter()
                        .map(|i| {
                            let distance = test_approximate_distance(f64::from((n + i as u8) % 9));
                            expected.push((
                                distance.rough(),
                                entries[i].record_id().clone(),
                                entries[i].fields().as_ptr(),
                                location.clone(),
                            ));
                            ApproximateCandidate::new(entries[i].record_id().clone(), distance, i)
                        })
                        .collect();
                    pool.push_leaf(&entries, location, scored, budget, true)
                        .unwrap();
                    assert!(pool.candidates.len() <= budget);
                    assert!(pool.heap_bytes().is_ok());
                    sources.push(Arc::downgrade(&entries));
                }
                expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let truncated = expected.len() > budget;
                expected.truncate(budget);
                pool.validate_membership().unwrap();
                let (actual, was_truncated) = pool.select(1000).unwrap();
                assert_eq!(was_truncated, truncated);
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_eq!(actual.record_id(), &expected.1);
                    assert_eq!(
                        actual.fields().as_ptr(),
                        expected.2,
                        "projection is shared, not copied"
                    );
                    assert_eq!(actual.location(), &expected.3);
                }
                assert_eq!(actual.len(), expected.len());
                drop(actual);
                assert!(sources.iter().all(|source| source.upgrade().is_none()));
            }
        }
    }

    #[test]
    fn signed_zero_distances_use_record_id_ties() {
        let mut pool = LeafCandidatePool::default();
        let (entries, location) = leaf(0);
        let scored = vec![
            ApproximateCandidate::new(
                entries[0].record_id().clone(),
                test_approximate_distance(0.0),
                0,
            ),
            ApproximateCandidate::new(
                entries[1].record_id().clone(),
                test_approximate_distance(-0.0),
                1,
            ),
        ];
        pool.push_leaf(&entries, location, scored, 1, true).unwrap();
        let (selected, truncated) = pool.select(1).unwrap();
        assert!(truncated);
        assert_eq!(selected[0].record_id(), entries[0].record_id());
    }

    #[test]
    fn final_interval_filter_avoids_unnecessary_exact_reads() {
        let mut pool = LeafCandidatePool::default();
        let (entries, location) = leaf(0);
        let scored = (0..20)
            .map(|i| {
                ApproximateCandidate::new(
                    entries[i].record_id().clone(),
                    test_approximate_distance(i as f64),
                    i,
                )
            })
            .collect();
        pool.push_leaf(&entries, location, scored, 64, true)
            .unwrap();
        let (selected, truncated) = pool.select(3).unwrap();
        assert_eq!(selected.len(), 3);
        assert!(!truncated);
        assert_eq!(selected[2].distance().rough(), 2.0);
    }

    #[test]
    fn eviction_releases_arrays_but_remembers_duplicate_ids() {
        let mut pool = LeafCandidatePool::default();
        let (entries, location) = leaf(0);
        let weak = Arc::downgrade(&entries);
        pool.push_leaf(
            &entries,
            location.clone(),
            vec![ApproximateCandidate::new(
                entries[0].record_id().clone(),
                test_approximate_distance(9.0),
                0,
            )],
            1,
            true,
        )
        .unwrap();
        drop(entries);
        let (better, better_location) = leaf(1);
        pool.push_leaf(
            &better,
            better_location,
            vec![ApproximateCandidate::new(
                better[0].record_id().clone(),
                test_approximate_distance(1.0),
                0,
            )],
            1,
            true,
        )
        .unwrap();
        assert!(weak.upgrade().is_none());
        let (duplicate, _) = leaf(0);
        pool.push_leaf(
            &duplicate,
            location,
            vec![ApproximateCandidate::new(
                duplicate[0].record_id().clone(),
                test_approximate_distance(10.0),
                0,
            )],
            1,
            true,
        )
        .unwrap();
        let error = pool.validate_membership().err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Corruption);
    }

    #[test]
    fn duplicate_rejected_ids_are_checked_even_with_no_heap_capacity() {
        for budget in [0, 1, 8] {
            let (entries, location) = leaf(0);
            let mut pool = LeafCandidatePool::default();
            let scored = [0, 1, 0]
                .into_iter()
                .map(|i| {
                    ApproximateCandidate::new(
                        entries[i].record_id().clone(),
                        test_approximate_distance(i as f64),
                        i,
                    )
                })
                .collect();
            pool.push_leaf(&entries, location, scored, budget, true)
                .unwrap();
            assert_eq!(
                pool.validate_membership().err().unwrap().kind(),
                ErrorKind::Corruption
            );
        }
    }
}
