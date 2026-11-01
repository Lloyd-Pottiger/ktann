//! Global rough-distance top-budget selection across every visited leaf.
//!
//! A max heap bounds projection ownership independently of leaf count. Record
//! identities move into a separate list to detect corrupt duplicate membership,
//! including duplicates of candidates already evicted from the heap.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::{Arc, Weak};

use bytes::Bytes;

use crate::api::{Error, ErrorKind, Result, Value};
use crate::storage::values::RecordLocation;

use super::cache::{CachedLeafEntry, leaf_entry_bytes};
use super::numeric::compare_finite;
use super::rabitq::{ApproximateDistance, DecodedRaBitQ7, RaBitQQuery};

/// One globally retained candidate with its original snapshot projection.
///
/// The pool constructs candidates from Leaf Entries: the Record ID, the exact
/// typed filter projection, the rough distance with its conservative interval,
/// and the source location the authoritative Record Location must match.
pub(crate) struct LeafCandidate {
    record_id: Bytes,
    distance: ApproximateDistance,
    projection: LeafProjection,
}

/// Snapshot projections shared only by candidates still eligible for reranking.
struct CandidateLeaf {
    entries: Arc<[CachedLeafEntry]>,
    location: RecordLocation,
    bytes: usize,
}

/// A reference to a snapshot projection; fields are never copied during search.
enum LeafProjection {
    Shared {
        source: Arc<CandidateLeaf>,
        entry: usize,
    },
    #[cfg(test)]
    Owned {
        fields: Box<[Value]>,
        location: RecordLocation,
    },
}

impl LeafCandidate {
    pub(crate) fn record_id(&self) -> &Bytes {
        &self.record_id
    }

    pub(crate) fn distance(&self) -> ApproximateDistance {
        self.distance
    }

    /// Constructs a standalone projection for rerank contract tests.
    #[cfg(test)]
    pub(crate) fn from_leaf(
        record_id: Bytes,
        fields: Box<[Value]>,
        distance: ApproximateDistance,
        location: RecordLocation,
    ) -> Self {
        Self {
            record_id,
            distance,
            projection: LeafProjection::Owned { fields, location },
        }
    }

    /// Returns the original exact typed projection without copying it.
    pub(crate) fn fields(&self) -> &[Value] {
        match &self.projection {
            LeafProjection::Shared { source, entry } => source.entries[*entry].fields(),
            #[cfg(test)]
            LeafProjection::Owned { fields, .. } => fields,
        }
    }

    /// Returns the source location checked against authoritative membership.
    pub(crate) fn location(&self) -> &RecordLocation {
        match &self.projection {
            LeafProjection::Shared { source, .. } => &source.location,
            #[cfg(test)]
            LeafProjection::Owned { location, .. } => location,
        }
    }
}

impl std::fmt::Debug for LeafCandidate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LeafCandidate([REDACTED])")
    }
}

/// Retains only globally competitive candidates and their snapshot projections.
pub(crate) struct GlobalCandidatePool {
    capacity: usize,
    account_memory: bool,
    candidates: Vec<LeafCandidate>,
    ids: Vec<Bytes>,
    retained_bytes: usize,
    truncated: bool,
    id_bytes: usize,
}

/// The worst candidate is the heap root; ties follow unsigned Record ID order.
fn compare(a: &LeafCandidate, b: &LeafCandidate) -> Ordering {
    compare_finite(a.distance().rough(), b.distance().rough())
        .then_with(|| a.record_id().cmp(b.record_id()))
}

impl GlobalCandidatePool {
    /// Owns the global retention limit and projection accounting for one search.
    pub(crate) fn new(capacity: usize, account_memory: bool) -> Self {
        Self {
            capacity,
            account_memory,
            candidates: Vec::new(),
            ids: Vec::new(),
            retained_bytes: 0,
            truncated: false,
            id_bytes: 0,
        }
    }

    /// Exposes the bounded pool for diagnostics, without guaranteeing order.
    pub(crate) fn candidates(&self) -> &[LeafCandidate] {
        &self.candidates
    }

    /// An upper cutoff for screening; a partially filled pool cannot reject.
    fn rough_cutoff(&self) -> Option<f64> {
        if self.candidates.len() == self.capacity {
            self.candidates
                .first()
                .map(|candidate| candidate.distance().rough())
        } else {
            None
        }
    }

    /// Scores eligible ordinals and admits only globally competitive entries.
    /// None means the whole leaf; predicate matches preserve original ordinals.
    pub(crate) fn push_leaf(
        &mut self,
        entries: &Arc<[CachedLeafEntry]>,
        components: &[i8],
        location: RecordLocation,
        query: &RaBitQQuery<'_>,
        eligible: Option<&[usize]>,
    ) -> Result<()> {
        let len = eligible.map_or(entries.len(), <[usize]>::len);
        let entry_at = |i| eligible.map_or(i, |ordinals| ordinals[i]);
        self.ids
            .try_reserve(len)
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        // A fixed leaf cutoff is conservative as subsequent admissions can only
        // improve the heap. Keep four-lane scoring and scalar tails unchanged.
        let cutoff = self.rough_cutoff();
        let mut source = Weak::new();
        let full = len / 4 * 4;
        for start in (0..full).step_by(4) {
            let ordinals: [_; 4] = std::array::from_fn(|lane| entry_at(start + lane));
            let codes = ordinals.map(|i| entries[i].code(components));
            let distances = score_batch(codes, query, cutoff)?;
            for (entry, distance) in ordinals.into_iter().zip(distances) {
                self.admit(entries, &location, &mut source, entry, distance)?;
            }
        }
        for i in full..len {
            let entry = entry_at(i);
            let [distance] = score_batch([entries[entry].code(components)], query, cutoff)?;
            self.admit(entries, &location, &mut source, entry, distance)?;
        }
        Ok(())
    }

    /// Every eligible identity is retained for corruption detection, including
    /// screened or evicted entries. Projection ownership belongs only to the heap.
    fn admit(
        &mut self,
        entries: &Arc<[CachedLeafEntry]>,
        location: &RecordLocation,
        source: &mut Weak<CandidateLeaf>,
        entry: usize,
        distance: Option<ApproximateDistance>,
    ) -> Result<()> {
        let id = entries[entry].record_id();
        self.id_bytes = self
            .id_bytes
            .checked_add(id.len())
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        self.ids.push(id.clone());
        let Some(distance) = distance else {
            self.truncated = true;
            return Ok(());
        };
        if self.candidates.len() == self.capacity {
            self.truncated = true;
            if self.capacity == 0
                || compare_finite(distance.rough(), self.candidates[0].distance().rough())
                    .then_with(|| id.cmp(self.candidates[0].record_id()))
                    != Ordering::Less
            {
                return Ok(());
            }
        }
        // Only an admitted candidate acquires a projection reference. Weak
        // ownership lets eviction release a leaf even during its own scan.
        let source = match source.upgrade() {
            Some(source) => source,
            None => {
                let bytes = if self.account_memory {
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
                *source = Arc::downgrade(&shared);
                shared
            }
        };
        let candidate = LeafCandidate {
            record_id: id.clone(),
            distance,
            projection: LeafProjection::Shared { source, entry },
        };
        if self.candidates.len() < self.capacity {
            self.candidates
                .try_reserve(1)
                .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
            self.candidates.push(candidate);
            let mut child = self.candidates.len() - 1;
            while child > 0 {
                let parent = (child - 1) / 2;
                if compare(&self.candidates[child], &self.candidates[parent]) != Ordering::Greater {
                    break;
                }
                self.candidates.swap(child, parent);
                child = parent;
            }
        } else {
            let old = std::mem::replace(&mut self.candidates[0], candidate);
            match &old.projection {
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
                if compare(&self.candidates[parent], &self.candidates[child]) != Ordering::Less {
                    break;
                }
                self.candidates.swap(parent, child);
                parent = child;
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

/// Refines all lanes that may enter the heap using the canonical f64 score.
fn score_batch<const N: usize>(
    codes: [DecodedRaBitQ7<&[i8]>; N],
    query: &RaBitQQuery<'_>,
    cutoff: Option<f64>,
) -> Result<[Option<ApproximateDistance>; N]> {
    let rejected = if let Some(cutoff) = cutoff {
        DecodedRaBitQ7::rough_lower_bounds(codes.each_ref(), query)
            .map_err(|_| Error::new(ErrorKind::Corruption))?
            .map(|lower| lower > cutoff)
    } else {
        [false; N]
    };
    if rejected.iter().all(|&rejected| !rejected) {
        return DecodedRaBitQ7::approximate_distances(codes.each_ref(), query)
            .map(|distances| distances.map(Some))
            .map_err(|_| Error::new(ErrorKind::Corruption));
    }
    let mut distances = [None; N];
    for lane in 0..N {
        if !rejected[lane] {
            let [distance] = DecodedRaBitQ7::approximate_distances([&codes[lane]], query)
                .map_err(|_| Error::new(ErrorKind::Corruption))?;
            distances[lane] = Some(distance);
        }
    }
    Ok(distances)
}

#[cfg(test)]
mod tests {
    use super::super::rabitq::{RaBitQ7, test_approximate_distance};
    use super::*;
    use crate::api::{Metric, PartitionKey, Value};
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

    fn push_scored(
        pool: &mut GlobalCandidatePool,
        entries: &Arc<[CachedLeafEntry]>,
        location: RecordLocation,
        scored: Vec<(usize, ApproximateDistance)>,
    ) -> Result<()> {
        pool.ids
            .try_reserve(scored.len())
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let mut source = Weak::new();
        for (entry, distance) in scored {
            pool.admit(entries, &location, &mut source, entry, Some(distance))?;
        }
        Ok(())
    }

    #[test]
    fn bounded_heap_matches_global_sort_and_shares_original_fields() {
        for budget in [0, 1, 7, 125, 180, 1000] {
            for reverse in [false, true] {
                let mut pool = GlobalCandidatePool::new(budget, true);
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
                            (i, distance)
                        })
                        .collect();
                    push_scored(&mut pool, &entries, location, scored).unwrap();
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
        let mut pool = GlobalCandidatePool::new(1, true);
        let (entries, location) = leaf(0);
        let scored = vec![
            (0, test_approximate_distance(0.0)),
            (1, test_approximate_distance(-0.0)),
        ];
        push_scored(&mut pool, &entries, location, scored).unwrap();
        let (selected, truncated) = pool.select(1).unwrap();
        assert!(truncated);
        assert_eq!(selected[0].record_id(), entries[0].record_id());
    }

    #[test]
    fn final_interval_filter_avoids_unnecessary_exact_reads() {
        let mut pool = GlobalCandidatePool::new(64, true);
        let (entries, location) = leaf(0);
        let scored = (0..20)
            .map(|i| (i, test_approximate_distance(i as f64)))
            .collect();
        push_scored(&mut pool, &entries, location, scored).unwrap();
        let (selected, truncated) = pool.select(3).unwrap();
        assert_eq!(selected.len(), 3);
        assert!(!truncated);
        assert_eq!(selected[2].distance().rough(), 2.0);
    }

    #[test]
    fn screened_entries_remain_accounted_and_detect_duplicate_membership() {
        let query = RaBitQQuery::new(&[0.0; 4], Metric::L2).unwrap();
        let mut components = Vec::new();
        let entries: Arc<[CachedLeafEntry]> = [0.0, 10.0, 10.0]
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                CachedLeafEntry::from_leaf(
                    LeafEntry::new(
                        Bytes::from_static(if i == 0 { b"best" } else { b"duplicate" }),
                        Vec::new(),
                        RaBitQ7::quantize(&[value; 4]).unwrap(),
                    ),
                    4,
                    &mut components,
                )
            })
            .collect::<Vec<_>>()
            .into();
        let location = RecordLocation::new(
            TreeKey::encode(&[], &[]).unwrap(),
            PartitionKey::new(1).unwrap(),
        );
        let mut pool = GlobalCandidatePool::new(1, true);
        assert_eq!(pool.rough_cutoff(), None);
        pool.push_leaf(&entries, &components, location.clone(), &query, Some(&[0]))
            .unwrap();
        let cutoff = pool.rough_cutoff().unwrap();
        assert!(
            DecodedRaBitQ7::rough_lower_bounds([&entries[1].code(&components)], &query).unwrap()[0]
                > cutoff
        );
        let before = pool.heap_bytes().unwrap();
        pool.push_leaf(&entries, &components, location, &query, Some(&[1, 2]))
            .unwrap();
        assert!(pool.truncated);
        assert_eq!(pool.ids.len(), 3);
        assert!(pool.heap_bytes().unwrap() > before);
        assert_eq!(
            pool.validate_membership().unwrap_err().kind(),
            ErrorKind::Corruption
        );
    }

    #[test]
    fn eviction_releases_arrays_but_remembers_duplicate_ids() {
        let mut pool = GlobalCandidatePool::new(1, true);
        let (entries, location) = leaf(0);
        let weak = Arc::downgrade(&entries);
        push_scored(
            &mut pool,
            &entries,
            location.clone(),
            vec![(0, test_approximate_distance(9.0))],
        )
        .unwrap();
        drop(entries);
        let (better, better_location) = leaf(1);
        push_scored(
            &mut pool,
            &better,
            better_location,
            vec![(0, test_approximate_distance(1.0))],
        )
        .unwrap();
        assert!(weak.upgrade().is_none());
        let (duplicate, _) = leaf(0);
        push_scored(
            &mut pool,
            &duplicate,
            location,
            vec![(0, test_approximate_distance(10.0))],
        )
        .unwrap();
        let error = pool.validate_membership().err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Corruption);
    }

    #[test]
    fn duplicate_rejected_ids_are_checked_even_with_no_heap_capacity() {
        for budget in [0, 1, 8] {
            let (entries, location) = leaf(0);
            let mut pool = GlobalCandidatePool::new(budget, true);
            let scored = [0, 1, 0]
                .into_iter()
                .map(|i| (i, test_approximate_distance(i as f64)))
                .collect();
            push_scored(&mut pool, &entries, location, scored).unwrap();
            assert_eq!(
                pool.validate_membership().err().unwrap().kind(),
                ErrorKind::Corruption
            );
        }
    }

    #[test]
    fn screened_leaf_scores_preserve_the_entire_global_pool() {
        for dimension in [1, 3, 7, 64, 768] {
            let query_components: Vec<f32> = (0..dimension)
                .map(|i| ((i * 31 % 97) as f32 - 48.0) / 48.0)
                .collect();
            let leaves: Vec<_> = (0..8_u32)
                .map(|leaf| {
                    let mut codes = Vec::new();
                    let entries: Arc<[CachedLeafEntry]> = (0..(9 + leaf * 11))
                        .map(|ordinal| {
                            let vector: Vec<f32> = (0..dimension)
                                .map(|i| {
                                    ((i * 47 + ordinal as usize * 19 + leaf as usize * 23) % 127)
                                        as f32
                                        / 63.0
                                        - 1.0
                                })
                                .collect();
                            CachedLeafEntry::from_leaf(
                                LeafEntry::new(
                                    Bytes::from(
                                        ((u64::from(leaf) << 32) | u64::from(ordinal))
                                            .to_be_bytes()
                                            .to_vec(),
                                    ),
                                    Vec::new(),
                                    RaBitQ7::quantize(&vector).unwrap(),
                                ),
                                dimension,
                                &mut codes,
                            )
                        })
                        .collect::<Vec<_>>()
                        .into();
                    (entries, codes)
                })
                .collect();
            for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
                let query = RaBitQQuery::new(&query_components, metric).unwrap();
                for budget in [0, 1, 7, 125] {
                    for filtered in [false, true] {
                        let mut reference = GlobalCandidatePool::new(budget, true);
                        let mut screened = GlobalCandidatePool::new(budget, true);
                        for (leaf, (entries, codes)) in leaves.iter().enumerate() {
                            let indices: Vec<_> = (0..entries.len())
                                .filter(|i| !filtered || i % 3 == 0)
                                .collect();
                            let location = RecordLocation::new(
                                TreeKey::encode(&[], &[]).unwrap(),
                                PartitionKey::new(leaf as u64 + 1).unwrap(),
                            );
                            let scored = indices
                                .iter()
                                .map(|&entry| {
                                    let code = entries[entry].code(codes);
                                    let [distance] =
                                        DecodedRaBitQ7::approximate_distances([&code], &query)
                                            .unwrap();
                                    (entry, distance)
                                })
                                .collect();
                            push_scored(&mut reference, entries, location.clone(), scored).unwrap();
                            screened
                                .push_leaf(
                                    entries,
                                    codes,
                                    location,
                                    &query,
                                    if filtered { Some(&indices) } else { None },
                                )
                                .unwrap();
                            let ordered = |pool: &GlobalCandidatePool| {
                                let mut values: Vec<_> = pool
                                    .candidates()
                                    .iter()
                                    .map(|candidate| {
                                        let distance = candidate.distance();
                                        (
                                            candidate.record_id().clone(),
                                            distance.rough().to_bits(),
                                            distance.lower().to_bits(),
                                            distance.upper().to_bits(),
                                        )
                                    })
                                    .collect();
                                values.sort_unstable();
                                values
                            };
                            assert_eq!(
                                ordered(&reference),
                                ordered(&screened),
                                "dim={dimension}, metric={metric:?}, budget={budget}, filtered={filtered}, leaf={leaf}"
                            );
                            assert_eq!(reference.truncated, screened.truncated);
                        }
                        reference.validate_membership().unwrap();
                        screened.validate_membership().unwrap();
                    }
                }
            }
        }
    }
}
