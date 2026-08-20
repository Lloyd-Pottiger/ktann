//! Deterministic binary K-means split training (ADR 0015).
//!
//! Split training turns one consistent snapshot of a split source into the two
//! target centroids the split state machine (#10) persists on the exposed
//! targets. Training output is a routing model, not persistent authority
//! (ADR 0014): concurrent source writes need not restart training, and every
//! committed split phase stays searchable regardless of centroid freshness.
//!
//! # Contract
//!
//! - **One consistent snapshot.** The loader reads every source entry — Leaf
//!   Entries with their Vector Records for a level-1 source, Child Entry
//!   centroids above level 1 — from one read transaction, then closes it.
//!   Training itself is a pure in-memory function of the loaded entries.
//! - **Deterministic protocol.** The seeding source centroid is the
//!   arithmetic mean of the loaded snapshot — never a persisted centroid,
//!   which the initial leaf root does not have and which concurrent writes
//!   may have moved past — so training stays a function of the snapshot
//!   alone. Farthest-pair seeding and every Lloyd round break distance ties
//!   by canonical Child or Record ID order, each round assigns exactly
//!   `floor(n/2)` entries to the left cluster, centroids accumulate in f64
//!   and convert to finite f32, assignment stability stops training early,
//!   and training stops after ten rounds (ADR 0015). The result depends only
//!   on the snapshot, never on scan paging or load batching: entries are
//!   reordered canonically before seeding.
//! - **Deliberately unbounded input.** Training reads the complete source
//!   with no sampling, memory, or CPU bound (ADR 0014); only each scan page
//!   and record-load batch is bounded, so resource use stays proportional to
//!   the source's actual size.
//! - **Fail closed.** A missing source Header, a Leaf Entry whose Vector
//!   Record is absent or no longer preprocesses cleanly, and a malformed or
//!   undersized training set are Corruption. Training changes no persistent
//!   state, so the searchable source stays intact for later diagnosis or
//!   retry. State validation — that the source is actually Splitting — is the
//!   caller's contract, not training's.

use std::fmt;

use bytes::Bytes;

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::maintenance::routing::kernel_for;
use crate::search::numeric::{VectorKernel, compare_finite};
use crate::storage::backend::{ReadOps, ScanLimits};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{IndexManifest, PartitionCentroid, PersistentValue};
use crate::storage::{LogicalRange, ReadLogicalTxn};

/// The maximum number of Lloyd rounds; a fixed format-v1 protocol choice
/// (ADR 0015).
const MAX_TRAINING_ROUNDS: usize = 10;

/// The bounds on one source-entry scan page.
///
/// Paging only shapes I/O: training deliberately loads the complete source
/// (ADR 0014), and canonical reordering before seeding makes the result
/// independent of page boundaries.
const ENTRY_SCAN_LIMITS: ScanLimits = ScanLimits {
    item_limit: 1_024,
    byte_limit: 1_048_576,
};

/// The number of Vector Records loaded by one point-read batch.
///
/// One record is one logical key, so one batch issues at most 128 encoded
/// keys — comfortably below backend batch ceilings, matching the search
/// rerank load shape.
const RECORD_LOAD_BATCH: usize = 128;

/// The two target centroids trained from one consistent source snapshot.
///
/// The left centroid leads the balanced cluster of exactly `floor(n/2)`
/// entries; the right centroid leads the remainder. The split state machine
/// persists them on the exposed targets in creation order.
#[derive(Clone, PartialEq)]
pub struct SplitCentroids {
    left: PartitionCentroid,
    right: PartitionCentroid,
}

impl SplitCentroids {
    /// Returns the left target centroid.
    #[must_use]
    pub const fn left(&self) -> &PartitionCentroid {
        &self.left
    }

    /// Returns the right target centroid.
    #[must_use]
    pub const fn right(&self) -> &PartitionCentroid {
        &self.right
    }
}

impl fmt::Debug for SplitCentroids {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SplitCentroids([REDACTED])")
    }
}

/// Trains the two split target centroids for `source` from one consistent
/// snapshot.
///
/// The source Header's level selects the training input: a level-1 (leaf)
/// source trains from its Vector Records, each preprocessed with the metric's
/// routing normalization and the persisted rotation; a higher-level source
/// trains from the full-f32 centroids in its Child Entries and reads no
/// Vector Records. The transaction is only read from; training holds no KV
/// locks and changes nothing, so any failure leaves the searchable source
/// intact.
pub async fn train_split_centroids<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    tree_key: &TreeKey,
    source: PartitionKey,
) -> Result<SplitCentroids> {
    let manifest = txn.bound_manifest().ok_or_else(Error::invalid_argument)?;
    let header = match txn
        .get(LogicalKey::Header {
            index: manifest.logical_index_id(),
            tree_key: tree_key.clone(),
            partition: source,
        })
        .await?
    {
        Some(PersistentValue::PartitionHeader(header)) => header,
        // A Header key decodes only as a Partition Header, so a wrong kind is
        // unreachable; a trained partition without a Header is Corruption
        // either way.
        _ => return Err(Error::new(ErrorKind::Corruption)),
    };
    let kernel = kernel_for(manifest)?;
    let trained = if header.level() == 1 {
        let entries = load_leaf_source(txn, manifest, tree_key, source, &kernel).await?;
        train(&kernel, entries)?
    } else {
        let entries = load_internal_source(txn, manifest, tree_key, source).await?;
        train(&kernel, entries)?
    };
    Ok(SplitCentroids {
        left: PartitionCentroid::new(trained.left),
        right: PartitionCentroid::new(trained.right),
    })
}

/// Loads one leaf source's training entries: each Leaf Entry's Vector Record,
/// preprocessed into routing space.
///
/// An absent Vector Record, or one that no longer passes the ingest-time
/// preprocessing contract, is Corruption (ADR 0015).
async fn load_leaf_source<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
    kernel: &VectorKernel,
) -> Result<Vec<(Bytes, Box<[f32]>)>> {
    let index = manifest.logical_index_id();
    let range = LogicalRange::leaf_entries(manifest, tree_key, source)?;
    let mut record_ids = Vec::new();
    let mut cursor = None;
    loop {
        let page = txn.scan(&range, cursor.as_ref(), ENTRY_SCAN_LIMITS).await?;
        for item in page.items() {
            let PersistentValue::LeafEntry(entry) = item.value() else {
                // The codec decodes a Leaf Entry key only as a Leaf Entry, so
                // another kind is unreachable but must stay fail-closed.
                return Err(Error::new(ErrorKind::Corruption));
            };
            record_ids.push(entry.record_id().clone());
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }

    let mut entries = Vec::with_capacity(record_ids.len());
    for batch in record_ids.chunks(RECORD_LOAD_BATCH) {
        let keys = batch
            .iter()
            .map(|id| LogicalKey::Record {
                index,
                id: id.clone(),
            })
            .collect();
        let values = txn.batch_get(keys).await?;
        for (id, value) in batch.iter().zip(values) {
            let Some(PersistentValue::VectorRecord(record)) = value else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            let routing = kernel
                .preprocess(record.vector())
                .map_err(|_| Error::new(ErrorKind::Corruption))?;
            entries.push((id.clone(), routing));
        }
    }
    Ok(entries)
}

/// Loads one internal source's training entries: each Child Entry's immutable
/// full-f32 centroid, which already lives in routing space.
async fn load_internal_source<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    source: PartitionKey,
) -> Result<Vec<(PartitionKey, Box<[f32]>)>> {
    let range = LogicalRange::child_entries(manifest, tree_key, source)?;
    let mut entries = Vec::new();
    let mut cursor = None;
    loop {
        let page = txn.scan(&range, cursor.as_ref(), ENTRY_SCAN_LIMITS).await?;
        for item in page.items() {
            let PersistentValue::ChildEntry(entry) = item.value() else {
                // The codec decodes a Child Entry key only as a Child Entry,
                // so another kind is unreachable but must stay fail-closed.
                return Err(Error::new(ErrorKind::Corruption));
            };
            entries.push((entry.child(), Box::from(entry.centroid())));
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    Ok(entries)
}

/// The in-memory outcome of [`train`]: the two target centroids and the number
/// of Lloyd rounds run.
#[derive(Clone, Debug, PartialEq)]
struct TrainedSplit {
    left: Box<[f32]>,
    right: Box<[f32]>,
    /// The rounds actually run; consumed by the in-module training tests.
    #[allow(dead_code)]
    rounds: usize,
}

/// Runs the deterministic balanced K-means protocol over one loaded source.
///
/// The entries are reordered by canonical ID before seeding, so the result is
/// a pure function of the entry set. Fewer than two entries cannot seed two
/// non-empty balanced clusters; a legal split source holds more than the
/// configured maximum (at least two) entries, so an undersized or malformed
/// training set is Corruption.
fn train<I: Ord>(kernel: &VectorKernel, mut entries: Vec<(I, Box<[f32]>)>) -> Result<TrainedSplit> {
    if entries.len() < 2 {
        return Err(Error::new(ErrorKind::Corruption));
    }
    for (_, vector) in &entries {
        if vector.len() != kernel.dimension() || vector.iter().any(|c| !c.is_finite()) {
            return Err(Error::new(ErrorKind::Corruption));
        }
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));

    // Every input is persistent training data validated above, so the
    // kernel's caller-versus-persistent error distinction collapses to
    // Corruption here.
    let distance = |vector: &[f32], centroid: &[f32]| -> Result<f64> {
        kernel
            .routing_distance(vector, centroid)
            .map_err(|_| Error::new(ErrorKind::Corruption))
    };

    let source = mean(
        kernel.dimension(),
        entries.len(),
        entries.iter().map(|entry| &*entry.1),
    )?;
    let left_seed = farthest(&entries, &source, &distance)?;
    let right_seed = farthest(&entries, &entries[left_seed].1, &distance)?;

    let mut left = entries[left_seed].1.clone();
    let mut right = entries[right_seed].1.clone();
    let half = entries.len() / 2;
    let mut previous: Option<Vec<bool>> = None;
    let mut rounds = 0_usize;
    loop {
        let assignment = assign(&entries, &left, &right, half, &distance)?;
        let stable = previous.as_ref() == Some(&assignment);
        left = cluster_mean(kernel.dimension(), &entries, &assignment, true)?;
        right = cluster_mean(kernel.dimension(), &entries, &assignment, false)?;
        previous = Some(assignment);
        rounds += 1;
        if stable || rounds == MAX_TRAINING_ROUNDS {
            return Ok(TrainedSplit {
                left,
                right,
                rounds,
            });
        }
    }
}

/// Orders entries by their distance difference to the two centroids and
/// assigns exactly `half` of them to the left cluster.
///
/// Entries are in canonical ID order, so the position tie-break on equal
/// differences is the Child or Record ID tie-break. The returned mask marks
/// left membership per entry position.
fn assign<I, D: Fn(&[f32], &[f32]) -> Result<f64>>(
    entries: &[(I, Box<[f32]>)],
    left: &[f32],
    right: &[f32],
    half: usize,
    distance: &D,
) -> Result<Vec<bool>> {
    let mut differences = Vec::with_capacity(entries.len());
    for (_, vector) in entries {
        differences.push(distance(vector, left)? - distance(vector, right)?);
    }
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| compare_finite(differences[a], differences[b]).then_with(|| a.cmp(&b)));
    let mut assignment = vec![false; entries.len()];
    for &member in &order[..half] {
        assignment[member] = true;
    }
    Ok(assignment)
}

/// Returns the position of the entry farthest from `from`.
///
/// Entries are in canonical ID order and a strictly greater distance is
/// required to win, so a distance tie keeps the smaller Child or Record ID.
fn farthest<I, D: Fn(&[f32], &[f32]) -> Result<f64>>(
    entries: &[(I, Box<[f32]>)],
    from: &[f32],
    distance: &D,
) -> Result<usize> {
    let mut farthest = 0_usize;
    let mut best = distance(&entries[0].1, from)?;
    for (index, (_, vector)) in entries.iter().enumerate().skip(1) {
        let candidate = distance(vector, from)?;
        if candidate > best {
            best = candidate;
            farthest = index;
        }
    }
    Ok(farthest)
}

/// Computes one cluster's centroid from an assignment mask.
///
/// A balanced assignment never produces an empty cluster: `half >= 1` and
/// `entries.len() - half >= 1` whenever training admits the entry set.
fn cluster_mean<I>(
    dimension: usize,
    entries: &[(I, Box<[f32]>)],
    assignment: &[bool],
    left: bool,
) -> Result<Box<[f32]>> {
    let count = assignment.iter().filter(|&&member| member == left).count();
    mean(
        dimension,
        count,
        entries
            .iter()
            .zip(assignment)
            .filter(move |(_, member)| **member == left)
            .map(|(entry, _)| &*entry.1),
    )
}

/// Computes the component-wise mean of `count` members with f64 accumulation,
/// converting to finite f32.
///
/// The mean of finite components lies within their range, so the conversion
/// cannot overflow; a non-finite component — including one produced by an
/// impossible empty cluster — is Corruption.
fn mean<'a>(
    dimension: usize,
    count: usize,
    members: impl Iterator<Item = &'a [f32]>,
) -> Result<Box<[f32]>> {
    let mut sums = vec![0.0_f64; dimension];
    for vector in members {
        for (sum, &component) in sums.iter_mut().zip(vector.iter()) {
            *sum += f64::from(component);
        }
    }
    let mut centroid = Vec::with_capacity(dimension);
    for sum in sums {
        let component = (sum / count as f64) as f32;
        if !component.is_finite() {
            return Err(Error::new(ErrorKind::Corruption));
        }
        centroid.push(component);
    }
    Ok(centroid.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Metric;

    const SEED: [u8; 32] = [7; 32];

    fn kernel(dimension: usize, metric: Metric) -> VectorKernel {
        VectorKernel::new(dimension, metric, SEED).expect("valid kernel")
    }

    /// Entries with canonical IDs `a`, `b`, ... in the given vector order.
    fn entries(values: &[&[f32]]) -> Vec<(Bytes, Box<[f32]>)> {
        values
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                (
                    Bytes::copy_from_slice(&[b'a' + index as u8]),
                    vector.to_vec().into_boxed_slice(),
                )
            })
            .collect()
    }

    fn assert_kind<T>(result: Result<T>, expected: ErrorKind) {
        assert_eq!(result.err().map(|error| error.kind()), Some(expected));
    }

    #[test]
    fn seeding_and_early_stop_on_a_one_dimensional_source() {
        // Source mean 3.25 seeds the farthest entry 10.0, whose farthest
        // partner is 0.0. The first balanced assignment {10, 2}|{0, 1} is
        // already stable, so training stops after the confirming round.
        let trained = train(
            &kernel(1, Metric::L2),
            entries(&[&[0.0], &[1.0], &[2.0], &[10.0]]),
        )
        .expect("trained");
        assert_eq!(&*trained.left, &[6.0]);
        assert_eq!(&*trained.right, &[0.5]);
        assert_eq!(trained.rounds, 2);
    }

    #[test]
    fn the_result_is_a_pure_function_of_the_entry_set() {
        let ordered = entries(&[&[0.0], &[1.0], &[2.0], &[10.0], &[4.5]]);
        let mut shuffled = ordered.clone();
        shuffled.reverse();
        let kernel = kernel(1, Metric::L2);
        let first = train(&kernel, ordered).expect("trained");
        let second = train(&kernel, shuffled).expect("trained");
        assert_eq!(first, second);
    }

    #[test]
    fn assignment_ties_break_by_canonical_id_order() {
        // Identical centroids make every distance difference zero, so the
        // balanced left cluster is exactly the first floor(n/2) canonical IDs.
        let entries = entries(&[&[0.0], &[1.0], &[2.0], &[10.0], &[4.5]]);
        let distance = |_: &[f32], _: &[f32]| -> Result<f64> { Ok(0.0) };
        let assignment = assign(&entries, &[0.0], &[0.0], 2, &distance).expect("assigned");
        assert_eq!(assignment, vec![true, true, false, false, false]);
    }

    #[test]
    fn seed_distance_ties_choose_the_smaller_id() {
        let kernel = kernel(1, Metric::L2);
        let distance = |vector: &[f32], centroid: &[f32]| kernel.routing_distance(vector, centroid);

        // Entries 10.0 ("a") and 0.0 ("b") tie at distance 25 from 5.0; the
        // smaller ID wins even though it holds the larger value.
        let tied = entries(&[&[10.0], &[0.0], &[5.0]]);
        assert_eq!(farthest(&tied, &[5.0], &distance).expect("farthest"), 0);

        // Swapping the values between the same IDs moves the winner with the
        // ID, proving the tie-break is the ID rather than the value.
        let swapped = entries(&[&[0.0], &[10.0], &[5.0]]);
        assert_eq!(farthest(&swapped, &[5.0], &distance).expect("farthest"), 0);
    }

    #[test]
    fn identical_entries_still_produce_finite_balanced_centroids() {
        // All distances and differences tie, so both seeds and the left
        // cluster resolve purely by canonical ID order.
        let trained =
            train(&kernel(1, Metric::L2), entries(&[&[5.0], &[5.0], &[5.0]])).expect("trained");
        assert_eq!(&*trained.left, &[5.0]);
        assert_eq!(&*trained.right, &[5.0]);
        assert_eq!(trained.rounds, 2);
    }

    #[test]
    fn cosine_training_ranks_by_negated_dot_product() {
        // In routing space cosine distance is the negated dot product, so the
        // antipodal entry pairs against the seed and the remaining entries
        // average into the right centroid.
        let trained = train(
            &kernel(2, Metric::Cosine),
            entries(&[&[1.0, 0.0], &[0.0, 1.0], &[-1.0, 0.0]]),
        )
        .expect("trained");
        assert_eq!(&*trained.left, &[1.0, 0.0]);
        assert_eq!(&*trained.right, &[-0.5, 0.5]);
        assert_eq!(trained.rounds, 2);
    }

    #[test]
    fn fewer_than_two_entries_are_corruption() {
        let kernel = kernel(1, Metric::L2);
        assert_kind(
            train(&kernel, Vec::<(Bytes, Box<[f32]>)>::new()),
            ErrorKind::Corruption,
        );
        assert_kind(train(&kernel, entries(&[&[1.0]])), ErrorKind::Corruption);
    }

    #[test]
    fn malformed_entry_vectors_are_corruption() {
        let kernel = kernel(2, Metric::L2);
        assert_kind(
            train(&kernel, entries(&[&[1.0], &[2.0]])),
            ErrorKind::Corruption,
        );
        assert_kind(
            train(&kernel, entries(&[&[1.0, f32::NAN], &[2.0, 3.0]])),
            ErrorKind::Corruption,
        );
    }

    #[test]
    fn extreme_magnitudes_convert_to_finite_f32() {
        let trained = train(
            &kernel(1, Metric::L2),
            entries(&[&[f32::MAX], &[-f32::MAX], &[0.0]]),
        )
        .expect("trained");
        assert!(trained.left.iter().all(|component| component.is_finite()));
        assert!(trained.right.iter().all(|component| component.is_finite()));
        assert_eq!(&*trained.left, &[f32::MAX]);
        assert_eq!(&*trained.right, &[-f32::MAX / 2.0]);
    }

    #[test]
    fn training_never_exceeds_the_ten_round_cap() {
        // Balanced binary training stabilizes within a few rounds in practice;
        // the ten-round cap is the deterministic termination backstop. Sweep
        // pseudo-random sources to guard the bound and the stable output.
        let mut state = 0x8f3a_2b19_u32;
        let mut rand = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) % 4_001) as f32 / 1_000.0 - 2.0
        };
        let kernel = kernel(3, Metric::L2);
        for n in 2..64_usize {
            let entries: Vec<(Bytes, Box<[f32]>)> = (0..n)
                .map(|index| {
                    (
                        Bytes::from(index.to_be_bytes().to_vec()),
                        vec![rand(), rand(), rand()].into_boxed_slice(),
                    )
                })
                .collect();
            let trained = train(&kernel, entries.clone()).expect("trained");
            assert!(trained.rounds <= MAX_TRAINING_ROUNDS);
            assert_eq!(train(&kernel, entries).expect("trained"), trained);
        }
    }
}
