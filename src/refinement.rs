//! Shared in-memory refinement of fixed partition topology.
//!
//! Callers admit retained inputs and scratch before constructing a planner.
//! Preparation normalizes vectors once; each tree refreshes all centroids even
//! with zero rounds. Persistence and artifact emission remain caller owned.

mod hnsw;

use std::collections::BTreeMap;

use crate::api::{IndexConfig, PartitionKey, Result};
use crate::search::numeric::VectorKernel;
use crate::storage::values::PartitionCentroid;

/// Per-leaf candidate bound shared by exact selection, HNSW and move scoring.
const NEIGHBOR_COUNT: usize = 32;

/// Sorted routing distances and leaf ordinals; unused slots sort last.
type LeafNeighbors = [(f64, usize); NEIGHBOR_COUNT];

/// Native candidate scratch is shared across sequential trees and rounds.
pub(crate) fn candidate_scratch_bytes(leaves: usize, rounds: usize) -> Result<usize> {
    if rounds == 0 {
        return Ok(0);
    }
    hnsw::scratch_bytes(leaves)
}

/// Existing partitions keep their identities and parent edges during refinement.
pub(crate) struct Part {
    pub(crate) key: PartitionKey,
    pub(crate) parent: Option<PartitionKey>,
    pub(crate) level: u32,
    pub(crate) centroid: PartitionCentroid,
    pub(crate) members: Vec<usize>,
    pub(crate) children: Vec<PartitionKey>,
}

/// Normalized inputs shared by every tree in one admitted refinement operation.
pub(crate) struct Planner {
    kernel: VectorKernel,
    vectors: Vec<Box<[f32]>>,
    minimum: usize,
    maximum: usize,
}

impl Planner {
    pub(crate) fn new(
        config: &IndexConfig,
        seed: [u8; 32],
        mut vectors: Vec<Box<[f32]>>,
        check: &impl Fn() -> Result<()>,
    ) -> Result<Self> {
        let kernel = VectorKernel::new(config.dimension(), config.metric(), seed)?;
        for vector in &mut vectors {
            check()?;
            *vector = kernel.preprocess(vector)?;
        }
        Ok(Self {
            kernel,
            vectors,
            minimum: config.min_leaf_partition_entries() as usize,
            maximum: config.max_leaf_partition_entries() as usize,
        })
    }

    /// Refines fixed topology in child-before-parent order, preserving identities.
    /// Returns key positions for callers retaining a different input order.
    /// The move observer lets persistence retain history without charging artifacts
    /// for a log they do not consume. Callers discard private state on failure.
    pub(crate) fn refine_tree(
        &self,
        parts: &mut [Part],
        rounds: usize,
        check: &impl Fn() -> Result<()>,
        on_move: impl FnMut(usize, usize, PartitionKey, PartitionKey),
    ) -> Result<BTreeMap<PartitionKey, usize>> {
        parts.sort_by_key(|part| (part.level, part.key));
        let leaves = parts.partition_point(|part| part.level == 1);
        for leaf in &mut parts[..leaves] {
            check()?;
            // Empty roots retain their existing center.
            if !leaf.members.is_empty() {
                leaf.centroid = mean(
                    &self.kernel,
                    leaf.members
                        .iter()
                        .map(|&position| self.vectors[position].as_ref()),
                    leaf.level,
                )?;
            }
        }
        refine(
            check,
            &self.kernel,
            &self.vectors,
            &mut parts[..leaves],
            self.minimum,
            self.maximum,
            rounds,
            on_move,
        )?;
        refresh_internal_centers(&self.kernel, parts, leaves, check)
    }
}

/// Refreshes the hierarchy in child-before-parent order using unweighted means.
fn refresh_internal_centers(
    kernel: &VectorKernel,
    parts: &mut [Part],
    leaves: usize,
    check: &impl Fn() -> Result<()>,
) -> Result<BTreeMap<PartitionKey, usize>> {
    let positions: BTreeMap<_, _> = parts
        .iter()
        .enumerate()
        .map(|(i, part)| (part.key, i))
        .collect();
    for i in leaves..parts.len() {
        check()?;
        parts[i].centroid = mean(
            kernel,
            parts[i]
                .children
                .iter()
                .map(|key| parts[positions[key]].centroid.components()),
            parts[i].level,
        )?;
    }
    Ok(positions)
}

/// Trains a routing model with canonical f64 accumulation; only leaves get dispersion.
fn mean<'a>(
    kernel: &VectorKernel,
    members: impl Iterator<Item = &'a [f32]>,
    level: u32,
) -> Result<PartitionCentroid> {
    let mut sum = vec![0.0_f64; kernel.dimension()];
    let mut count = 0_usize;
    for vector in members {
        for (sum, component) in sum.iter_mut().zip(vector) {
            *sum += f64::from(*component);
        }
        count += 1;
    }
    let center: Vec<f32> = sum
        .into_iter()
        .map(|sum| (sum / count.max(1) as f64) as f32)
        .collect();
    let dispersion = if level == 1 {
        kernel.leaf_dispersion(&center)
    } else {
        0.0
    };
    PartitionCentroid::with_dispersion(kernel.normalize_centroid(&center)?, dispersion)
}

/// Retains up to 32 candidate leaves in exact (distance, leaf ordinal) order.
/// Large trees use a centroid HNSW; small trees score each symmetric pair once,
/// updating both ends.
/// Only the bounded neighbor lists survive; no quadratic distance matrix is kept.
fn leaf_neighbors(
    checkpoint: &impl Fn() -> Result<()>,
    kernel: &VectorKernel,
    leaves: &[Part],
) -> Result<Vec<LeafNeighbors>> {
    if hnsw::enabled(leaves.len()) {
        return hnsw::neighbors(checkpoint, kernel, leaves);
    }
    // One allocation avoids one independently allocated buffer per leaf.
    let mut neighbors = vec![[(f64::INFINITY, usize::MAX); NEIGHBOR_COUNT]; leaves.len()];
    // Tile both axes so centroids and neighbor lists stay hot while scoring
    // the upper triangle. Each unordered pair still uses the same kernel once.
    const TILE: usize = 32;
    for source_start in (0..leaves.len()).step_by(TILE) {
        let source_end = (source_start + TILE).min(leaves.len());
        for target_start in (source_start..leaves.len()).step_by(TILE) {
            let target_end = (target_start + TILE).min(leaves.len());
            checkpoint()?;
            for source in source_start..source_end {
                let start = (source + 1).max(target_start);
                let (batches, tail) = leaves[start..target_end].as_chunks::<4>();
                for (batch, candidates) in batches.iter().enumerate() {
                    let scores = kernel.routing_distances_validated(
                        candidates
                            .each_ref()
                            .map(|candidate| candidate.centroid.components()),
                        leaves[source].centroid.components(),
                    );
                    for (lane, score) in scores.into_iter().enumerate() {
                        let target = start + batch * 4 + lane;
                        retain_neighbor(&mut neighbors[source], (score, target));
                        retain_neighbor(&mut neighbors[target], (score, source));
                    }
                }
                for (offset, candidate) in tail.iter().enumerate() {
                    let target = start + batches.len() * 4 + offset;
                    let score = kernel.routing_distances_validated(
                        [leaves[source].centroid.components()],
                        candidate.centroid.components(),
                    )[0];
                    retain_neighbor(&mut neighbors[source], (score, target));
                    retain_neighbor(&mut neighbors[target], (score, source));
                }
            }
        }
    }
    Ok(neighbors)
}

/// Inserts only improving candidates without growing beyond the neighbor bound.
fn retain_neighbor(neighbors: &mut LeafNeighbors, candidate: (f64, usize)) {
    let compare = |previous: &(f64, usize)| {
        previous
            .0
            .total_cmp(&candidate.0)
            .then(previous.1.cmp(&candidate.1))
    };
    let last = neighbors.len() - 1;
    if compare(&neighbors[last]).is_le() {
        return;
    }
    let position = neighbors.partition_point(|previous| compare(previous).is_lt());
    neighbors.copy_within(position..last, position + 1);
    neighbors[position] = candidate;
}

/// Greedy positive-gain moves use fixed centroids during one round, then
/// recompute spherical means. Exact counts constrain every accepted move.
/// The only root leaf is exempt from the ordinary nonroot minimum.
/// Only online planning retains accepted moves for later persistence; artifact
/// preparation needs only the final membership and centroids. Callers discard
/// their private planning state if any round fails.
#[expect(
    clippy::too_many_arguments,
    reason = "numerical inputs, occupancy bounds, and move observation are independent"
)]
fn refine(
    checkpoint: &impl Fn() -> Result<()>,
    kernel: &VectorKernel,
    vectors: &[Box<[f32]>],
    leaves: &mut [Part],
    minimum: usize,
    maximum: usize,
    rounds: usize,
    mut on_move: impl FnMut(usize, usize, PartitionKey, PartitionKey),
) -> Result<()> {
    // The planner preprocesses vectors and computes leaf centers with `mean`.
    // Each later round also uses `mean`, which validates and normalizes centers.
    // These finite, fixed-dimension inputs need no per-distance validation.
    if leaves.len() < 2 {
        return Ok(());
    }
    for round in 0..rounds {
        checkpoint()?;
        let neighbors = leaf_neighbors(checkpoint, kernel, leaves)?;
        let mut moves = Vec::new();
        for (source, leaf) in leaves.iter().enumerate() {
            checkpoint()?;
            let distances = &neighbors[source][..NEIGHBOR_COUNT.min(leaves.len() - 1)];
            for &position in &leaf.members {
                let vector = &vectors[position];
                let old =
                    kernel.routing_distances_validated([vector], leaf.centroid.components())[0];
                let mut best = (old, source);
                let (batches, tail) = distances.as_chunks::<4>();
                for targets in batches {
                    let distances = kernel.routing_distances_validated(
                        targets.map(|(_, target)| leaves[target].centroid.components()),
                        vector,
                    );
                    for (&(_, target), distance) in targets.iter().zip(distances) {
                        if distance < best.0 {
                            best = (distance, target);
                        }
                    }
                }
                for &(_, target) in tail {
                    let distance = kernel.routing_distances_validated(
                        [vector],
                        leaves[target].centroid.components(),
                    )[0];
                    if distance < best.0 {
                        best = (distance, target);
                    }
                }
                if best.1 != source {
                    moves.push((old - best.0, position, source, best.1));
                }
            }
        }
        drop(neighbors);
        moves.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
        let mut counts: Vec<_> = leaves.iter().map(|leaf| leaf.members.len()).collect();
        let mut targets = BTreeMap::new();
        for (_, position, source, target) in moves {
            if counts[source] > minimum && counts[target] < maximum {
                on_move(round, position, leaves[source].key, leaves[target].key);
                counts[source] -= 1;
                counts[target] += 1;
                targets.insert(position, target);
            }
        }
        crate::observe::metrics::refinement_round(targets.len());
        if targets.is_empty() {
            break;
        }
        let mut members = vec![Vec::new(); leaves.len()];
        for (source, leaf) in leaves.iter().enumerate() {
            for &position in &leaf.members {
                members[targets.get(&position).copied().unwrap_or(source)].push(position);
            }
        }
        for (leaf, mut members) in leaves.iter_mut().zip(members) {
            members.sort_unstable();
            leaf.centroid = mean(
                kernel,
                members.iter().map(|&position| vectors[position].as_ref()),
                leaf.level,
            )?;
            leaf.members = members;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rounds_refreshes_nested_trees_from_shared_normalized_inputs() {
        use crate::api::Metric;
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let config = IndexConfig::new(2, metric).unwrap();
            let vectors = vec![
                vec![2.0, 1.0].into_boxed_slice(),
                vec![1.0, 4.0].into_boxed_slice(),
            ];
            let kernel = VectorKernel::new(2, metric, [7; 32]).unwrap();
            let expected_leaves: Vec<_> = vectors
                .iter()
                .map(|vector| {
                    kernel
                        .normalize_centroid(&kernel.preprocess(vector).unwrap())
                        .unwrap()
                })
                .collect();
            let expected_root = mean(&kernel, expected_leaves.iter().map(Box::as_ref), 3).unwrap();
            let planner = Planner::new(&config, [7; 32], vectors, &|| Ok(())).unwrap();
            // Two trees reuse normalized vectors without transforming them a second time.
            for reverse in [false, true] {
                let mut parts: Vec<_> = [
                    (1, 3, vec![2, 3], vec![]),
                    (2, 2, vec![4], vec![]),
                    (3, 2, vec![5], vec![]),
                    (4, 1, vec![], vec![0]),
                    (5, 1, vec![], vec![1]),
                ]
                .into_iter()
                .map(|(key, level, children, members)| Part {
                    key: PartitionKey::new(key).unwrap(),
                    parent: None,
                    level,
                    centroid: PartitionCentroid::new(vec![999.0, 999.0]),
                    members,
                    children: children
                        .into_iter()
                        .map(|key| PartitionKey::new(key).unwrap())
                        .collect(),
                })
                .collect();
                if reverse {
                    parts.reverse();
                }
                let positions = planner
                    .refine_tree(&mut parts, 0, &|| Ok(()), |_, _, _, _| {
                        panic!("zero rounds must not emit moves");
                    })
                    .unwrap();
                for (leaf, expected) in [(4, &expected_leaves[0]), (5, &expected_leaves[1])] {
                    let part = &parts[positions[&PartitionKey::new(leaf).unwrap()]];
                    assert_eq!(part.centroid.components(), expected.as_ref());
                    assert_eq!(part.members, [usize::try_from(leaf - 4).unwrap()]);
                }
                assert_eq!(
                    parts[positions[&PartitionKey::new(1).unwrap()]].centroid,
                    expected_root
                );
            }
        }
    }

    #[test]
    fn symmetric_neighbors_match_exhaustive_order_for_all_metrics() {
        use crate::api::Metric;
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            for dimension in [1, 7, 8, 9, 128] {
                let kernel = VectorKernel::new(dimension, metric, [0; 32]).unwrap();
                for count in [0_usize, 1, 2, 31, 32, 33, 34, 65] {
                    let leaves: Vec<_> = (0..count)
                        .map(|i| Part {
                            key: PartitionKey::new(i as u64 + 2).unwrap(),
                            parent: None,
                            level: 1,
                            // Repeated and zero centers exercise exact distance ties.
                            centroid: PartitionCentroid::new(
                                kernel
                                    .normalize_centroid(
                                        &(0..dimension)
                                            .map(|d| {
                                                if i % 9 == 0 {
                                                    0.0
                                                } else {
                                                    ((i % 11 * 17 + d * 31) % 23) as f32 - 11.0
                                                }
                                            })
                                            .collect::<Vec<_>>(),
                                    )
                                    .unwrap(),
                            ),
                            members: vec![],
                            children: vec![],
                        })
                        .collect();
                    let actual = leaf_neighbors(&|| Ok(()), &kernel, &leaves).unwrap();
                    for (source, leaf) in leaves.iter().enumerate() {
                        let mut expected: Vec<_> = leaves
                            .iter()
                            .enumerate()
                            .filter(|(target, _)| *target != source)
                            .map(|(target, candidate)| {
                                (
                                    kernel
                                        .routing_distance(
                                            leaf.centroid.components(),
                                            candidate.centroid.components(),
                                        )
                                        .unwrap(),
                                    target,
                                )
                            })
                            .collect();
                        expected.sort_by(|left, right| {
                            left.0.total_cmp(&right.0).then(left.1.cmp(&right.1))
                        });
                        expected.truncate(NEIGHBOR_COUNT);
                        let bits = |neighbors: &[(f64, usize)]| {
                            neighbors
                                .iter()
                                .map(|&(distance, target)| (distance.to_bits(), target))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(
                            bits(&actual[source][..NEIGHBOR_COUNT.min(count.saturating_sub(1))]),
                            bits(&expected),
                            "metric={metric:?} dimension={dimension} count={count} source={source}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn neighbor_selection_propagates_cancellation_between_tiles() {
        use crate::api::{Error, ErrorKind, Metric};
        use std::cell::Cell;

        let kernel = VectorKernel::new(1, Metric::L2, [0; 32]).unwrap();
        let leaves: Vec<_> = (0..65)
            .map(|i| Part {
                key: PartitionKey::new(i + 1).unwrap(),
                parent: None,
                level: 1,
                centroid: PartitionCentroid::new(vec![i as f32]),
                members: Vec::new(),
                children: Vec::new(),
            })
            .collect();
        let checks = Cell::new(0);
        let error = leaf_neighbors(
            &|| {
                checks.set(checks.get() + 1);
                if checks.get() == 2 {
                    Err(Error::new(ErrorKind::Cancelled))
                } else {
                    Ok(())
                }
            },
            &kernel,
            &leaves,
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert_eq!(checks.get(), 2);
    }

    #[test]
    fn refinement_improves_assignment_and_respects_occupancy() {
        let kernel = VectorKernel::new(1, crate::api::Metric::L2, [0; 32]).unwrap();
        let vectors: Vec<Box<[f32]>> = [0.0, 1.0, 9.0, 10.0]
            .into_iter()
            .map(|x| vec![x].into())
            .collect();
        let mut leaves = vec![
            Part {
                parent: None,
                children: vec![],
                key: PartitionKey::new(2).unwrap(),
                level: 1,
                members: vec![0, 2],
                centroid: PartitionCentroid::new(vec![4.5]),
            },
            Part {
                parent: None,
                children: vec![],
                key: PartitionKey::new(3).unwrap(),
                level: 1,
                members: vec![1, 3],
                centroid: PartitionCentroid::new(vec![5.5]),
            },
        ];
        let mut moves = Vec::new();
        refine(
            &|| Ok(()),
            &kernel,
            &vectors,
            &mut leaves,
            2,
            2,
            2,
            |round, position, source, target| moves.push((round, position, source, target)),
        )
        .unwrap();
        assert_eq!(leaves[0].members, [0, 2]); // A full target cannot accept a move.
        assert!(moves.is_empty());
        refine(
            &|| Ok(()),
            &kernel,
            &vectors,
            &mut leaves,
            1,
            3,
            2,
            |round, position, source, target| moves.push((round, position, source, target)),
        )
        .unwrap();
        assert_eq!(leaves[0].members, [0, 1]);
        assert_eq!(leaves[1].members, [2, 3]);
        assert_eq!(leaves[0].centroid.components(), [0.5]);
        assert_eq!(leaves[1].centroid.components(), [9.5]);
        assert_eq!(
            moves,
            [
                (0, 1, leaves[1].key, leaves[0].key),
                (0, 2, leaves[0].key, leaves[1].key),
            ]
        );
    }

    #[test]
    fn refinement_batches_preserve_target_mapping_and_tail_neighbors() {
        let kernel = VectorKernel::new(1, crate::api::Metric::L2, [0; 32]).unwrap();
        let vectors: Vec<Box<[f32]>> = (0..9)
            .flat_map(|cluster| {
                [-0.1, 0.0, 0.1].map(|offset| vec![cluster as f32 * 10.0 + offset].into())
            })
            .collect();
        // Exchange one member between adjacent clusters. The ninth leaf is
        // unchanged and exercises the centroid and candidate batch tails.
        let mut leaves: Vec<_> = (0..9)
            .map(|cluster| Part {
                parent: None,
                children: vec![],
                key: PartitionKey::new(cluster as u64 + 2).unwrap(),
                level: 1,
                members: (cluster * 3..cluster * 3 + 3).collect(),
                centroid: PartitionCentroid::new(vec![cluster as f32 * 10.0]),
            })
            .collect();
        for pair in leaves[..8].as_chunks_mut::<2>().0 {
            let (left, right) = pair.split_at_mut(1);
            std::mem::swap(&mut left[0].members[2], &mut right[0].members[2]);
        }
        refine(
            &|| Ok(()),
            &kernel,
            &vectors,
            &mut leaves,
            1,
            5,
            2,
            |_, _, _, _| {},
        )
        .unwrap();
        for (cluster, leaf) in leaves.iter().enumerate() {
            assert_eq!(
                leaf.members,
                (cluster * 3..cluster * 3 + 3).collect::<Vec<_>>(),
                "cluster={cluster}"
            );
        }
    }
}
