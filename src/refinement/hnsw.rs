//! Ephemeral centroid graph for bounded, approximate refinement candidates.

use super::{LeafNeighbors, NEIGHBOR_COUNT, Part, retain_neighbor};
use crate::api::{Error, ErrorKind, Result};
use crate::search::{local_hnsw::native_error, numeric::VectorKernel};
use std::cell::Cell;

const M: usize = 8;
const EF_CONSTRUCTION: usize = 64;
const EF_SEARCH: usize = 64;

/// Small trees use exact pairs; larger trees use budget-admitted centroid graphs.
pub(super) fn enabled(leaves: usize) -> bool {
    leaves >= 1024
}

/// Additional peak reservation beyond the caller's retained inputs/neighbor lists.
/// Build scratch is included conservatively even while query scratch is live.
pub(super) fn scratch_bytes(leaves: usize) -> Result<usize> {
    if !enabled(leaves) {
        return Ok(0);
    }
    if leaves > ktann_usearch::MAX_NODES {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let build = ktann_usearch::build_bound(leaves, M).map_err(native_error)?;
    let query = ktann_usearch::workspace_bound(leaves, M).map_err(native_error)?;
    build
        .checked_add(query)
        .and_then(|bytes| {
            bytes.checked_add(std::mem::size_of::<[ktann_usearch::Hit; NEIGHBOR_COUNT + 1]>())
        })
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))
}

/// Rebuilds from current centers; native errors never select a different algorithm.
pub(super) fn neighbors(
    checkpoint: &impl Fn() -> Result<()>,
    kernel: &VectorKernel,
    leaves: &[Part],
) -> Result<Vec<LeafNeighbors>> {
    let n = leaves.len();
    checkpoint()?;
    let failure = Cell::new(None);
    let proceed = || match checkpoint() {
        Ok(()) => true,
        Err(error) => {
            failure.set(Some(error));
            false
        }
    };
    let result = (|| {
        let distance = |a: u32, b: u32| {
            kernel.routing_distances_validated(
                [leaves[a as usize].centroid.components()],
                leaves[b as usize].centroid.components(),
            )[0]
        };
        let build_limit = ktann_usearch::build_bound(n, M)?;
        let graph = ktann_usearch::Graph::build(
            n,
            M,
            EF_CONSTRUCTION,
            7,
            build_limit,
            &distance,
            &proceed,
        )?;
        let mut scratch =
            ktann_usearch::Workspace::new(n, M, ktann_usearch::workspace_bound(n, M)?)?;
        let mut hits = [ktann_usearch::Hit::default(); NEIGHBOR_COUNT + 1];
        let mut neighbors = vec![[(f64::INFINITY, usize::MAX); NEIGHBOR_COUNT]; n];
        for (source, retained) in neighbors.iter_mut().enumerate() {
            let query_distance = |_: u32, target: u32| distance(source as u32, target);
            graph.search(
                EF_SEARCH,
                &query_distance,
                &proceed,
                &mut scratch,
                &mut hits,
            )?;
            for hit in &hits {
                if hit.node as usize != source {
                    // Search already uses the exact routing kernel; no proxy scores survive.
                    retain_neighbor(retained, (hit.distance, hit.node as usize));
                }
            }
        }
        Ok(neighbors)
    })()
    .map_err(native_error);
    match failure.into_inner() {
        Some(error) => Err(error),
        None => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Metric, PartitionKey};
    use crate::storage::values::PartitionCentroid;

    fn leaves(kernel: &VectorKernel) -> Vec<Part> {
        (0..1024)
            .map(|i| Part {
                key: PartitionKey::new(i + 1).unwrap(),
                parent: None,
                level: 1,
                centroid: PartitionCentroid::new(
                    kernel
                        .normalize_centroid(&[1.0, (i % 31) as f32, (i % 47) as f32])
                        .unwrap(),
                ),
                members: vec![i as usize],
                children: vec![],
            })
            .collect()
    }

    #[test]
    fn candidates_are_deterministic_distinct_and_metric_correct() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let kernel = VectorKernel::new(3, metric, [7; 32]).unwrap();
            let leaves = leaves(&kernel);
            let first = super::super::leaf_neighbors(&|| Ok(()), &kernel, &leaves).unwrap();
            assert_eq!(first, neighbors(&|| Ok(()), &kernel, &leaves).unwrap());
            assert_eq!(first.len(), leaves.len());
            for (source, hits) in first.iter().enumerate() {
                let mut seen = std::collections::BTreeSet::new();
                for &(distance, target) in hits {
                    assert_ne!(source, target);
                    assert!(seen.insert(target));
                    assert_eq!(
                        distance,
                        kernel.routing_distances_validated(
                            [leaves[source].centroid.components()],
                            leaves[target].centroid.components(),
                        )[0]
                    );
                }
                assert!(hits.windows(2).all(|pair| pair[0] <= pair[1]));
            }
        }
    }

    #[test]
    fn native_cancellation_preserves_the_callers_error() {
        let kernel = VectorKernel::new(3, Metric::L2, [7; 32]).unwrap();
        let leaves = leaves(&kernel);
        let calls = Cell::new(0);
        let check = || {
            calls.set(calls.get() + 1);
            if calls.get() >= 20 {
                Err(Error::new(ErrorKind::DeadlineExceeded))
            } else {
                Ok(())
            }
        };
        assert_eq!(
            neighbors(&check, &kernel, &leaves).unwrap_err().kind(),
            ErrorKind::DeadlineExceeded
        );
        assert!(calls.get() >= 20);
    }

    #[test]
    fn approximate_refinement_preserves_membership_and_occupancy() {
        let kernel = VectorKernel::new(3, Metric::L2, [7; 32]).unwrap();
        let vectors: Vec<Box<[f32]>> = (0..3072)
            .map(|i| vec![(i / 3) as f32 * 10.0, (i % 3) as f32, 1.0].into())
            .collect();
        let mut leaves = leaves(&kernel);
        for (i, leaf) in leaves.iter_mut().enumerate() {
            leaf.members = vec![i * 3, i * 3 + 1, (i ^ 1) * 3 + 2];
            leaf.centroid = super::super::mean(
                &kernel,
                leaf.members.iter().map(|&i| vectors[i].as_ref()),
                leaf.level,
            )
            .unwrap();
        }
        let mut moves = 0;
        super::super::refine(
            &|| Ok(()),
            &kernel,
            &vectors,
            &mut leaves,
            2,
            4,
            2,
            |_, _, _, _| moves += 1,
        )
        .unwrap();
        assert!(moves > 0);
        assert!(
            leaves
                .iter()
                .all(|leaf| (2..=4).contains(&leaf.members.len()))
        );
        let mut members: Vec<_> = leaves
            .iter()
            .flat_map(|leaf| leaf.members.iter().copied())
            .collect();
        members.sort_unstable();
        assert_eq!(members, (0..vectors.len()).collect::<Vec<_>>());
        for leaf in &leaves {
            assert_eq!(
                leaf.centroid,
                super::super::mean(
                    &kernel,
                    leaf.members.iter().map(|&i| vectors[i].as_ref()),
                    leaf.level
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn reservation_follows_candidate_selection_and_rounds() {
        for n in [0, 1023] {
            assert_eq!(super::super::candidate_scratch_bytes(n, 1).unwrap(), 0);
        }
        for n in [1024, 65_537] {
            assert!(super::super::candidate_scratch_bytes(n, 1).unwrap() > 0);
            assert_eq!(super::super::candidate_scratch_bytes(n, 0).unwrap(), 0);
        }
        if let Some(unsupported) = ktann_usearch::MAX_NODES.checked_add(1) {
            assert_eq!(
                super::super::candidate_scratch_bytes(unsupported, 1)
                    .unwrap_err()
                    .kind(),
                ErrorKind::LimitExceeded
            );
        }
    }
}
