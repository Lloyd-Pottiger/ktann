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
                [&leaves[a as usize].center],
                &leaves[b as usize].center,
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

    fn leaves(kernel: &VectorKernel) -> Vec<Part> {
        (0..1024)
            .map(|i| Part {
                key: PartitionKey::new(i + 1).unwrap(),
                parent: None,
                level: 1,
                center: kernel
                    .normalize_centroid(&[1.0, (i % 31) as f32, (i % 47) as f32])
                    .unwrap(),
                members: vec![i as usize],
                children: vec![],
            })
            .collect()
    }

    /// Checks the same candidate contract for metric and large-graph cases.
    fn assert_candidates(kernel: &VectorKernel, leaves: &[Part], candidates: &[LeafNeighbors]) {
        assert_eq!(candidates.len(), leaves.len());
        for (source, hits) in candidates.iter().enumerate() {
            let mut seen = std::collections::BTreeSet::new();
            for &(distance, target) in hits {
                assert_ne!(source, target);
                assert!(seen.insert(target));
                assert_eq!(
                    distance,
                    kernel.routing_distances_validated(
                        [&leaves[source].center],
                        &leaves[target].center,
                    )[0]
                );
            }
            assert!(hits.windows(2).all(|pair| pair[0] <= pair[1]));
        }
    }

    #[test]
    fn candidates_are_deterministic_distinct_and_metric_correct() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let kernel = VectorKernel::new(3, metric, [7; 32]).unwrap();
            let leaves = leaves(&kernel);
            let first = super::super::leaf_neighbors(&|| Ok(()), &kernel, &leaves).unwrap();
            assert_eq!(first, neighbors(&|| Ok(()), &kernel, &leaves).unwrap());
            assert_candidates(&kernel, &leaves, &first);
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
                Err(Error::new(ErrorKind::Cancelled))
            } else {
                Ok(())
            }
        };
        assert_eq!(
            neighbors(&check, &kernel, &leaves).unwrap_err().kind(),
            ErrorKind::Cancelled
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
            leaf.center =
                super::super::mean(&kernel, leaf.members.iter().map(|&i| vectors[i].as_ref()))
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
                leaf.center,
                super::super::mean(&kernel, leaf.members.iter().map(|&i| vectors[i].as_ref()),)
                    .unwrap()
            );
        }
    }

    #[test]
    fn oversized_candidate_domain_fails_instead_of_selecting_exact() {
        if let Some(unsupported) = ktann_usearch::MAX_NODES.checked_add(1) {
            assert!(enabled(unsupported));
            assert_eq!(
                scratch_bytes(unsupported).unwrap_err().kind(),
                ErrorKind::LimitExceeded
            );
        }
        assert!(enabled(65_537));
    }

    #[test]
    #[ignore = "release-mode large candidate graph validation"]
    fn candidate_graph_above_partition_capacity() {
        let kernel = VectorKernel::new(3, Metric::L2, [7; 32]).unwrap();
        let leaves: Vec<_> = (0..65_537)
            .map(|i| Part {
                key: PartitionKey::new(i + 1).unwrap(),
                parent: None,
                level: 1,
                center: vec![(i % 257) as f32, (i / 257) as f32, 1.0].into(),
                members: vec![],
                children: vec![],
            })
            .collect();
        let started = std::time::Instant::now();
        let candidates = super::super::leaf_neighbors(&|| Ok(()), &kernel, &leaves).unwrap();
        let elapsed = started.elapsed().as_secs_f64();
        assert_candidates(&kernel, &leaves, &candidates);
        eprintln!(
            "large_refinement leaves={} seconds={elapsed:.3} graph_reservation_bytes={}",
            leaves.len(),
            scratch_bytes(leaves.len()).unwrap()
        );
    }

    #[test]
    fn graph_reservation_covers_native_work_and_respects_selection_bounds() {
        for n in [0, 1023] {
            assert_eq!(scratch_bytes(n).unwrap(), 0);
        }
        for n in [1024, 65_536, 65_537, 131_072] {
            assert!(
                scratch_bytes(n).unwrap()
                    >= ktann_usearch::build_bound(n, M).unwrap()
                        + ktann_usearch::workspace_bound(n, M).unwrap()
            );
            assert_eq!(super::super::candidate_scratch_bytes(n, 0).unwrap(), 0);
        }
    }
}
