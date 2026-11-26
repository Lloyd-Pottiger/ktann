//! Ephemeral centroid graph for bounded, approximate refinement candidates.

use super::{LeafCenter, LeafNeighbors, NEIGHBOR_COUNT, retain_neighbor};
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
    leaves: &[impl LeafCenter],
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
                [leaves[a as usize].center()],
                leaves[b as usize].center(),
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

/// Builds the centroid graph and executes batched candidate searches in parallel across threads.
pub(super) fn neighbors_parallel(
    checkpoint: &(dyn Fn() -> Result<()> + Sync),
    kernel: &VectorKernel,
    leaves: &[impl LeafCenter + Sync],
    parallelism: usize,
    memory_limit_bytes: usize,
) -> Result<Vec<LeafNeighbors>> {
    let n = leaves.len();
    if n > ktann_usearch::MAX_NODES {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    checkpoint()?;
    let build_limit = ktann_usearch::build_bound(n, M).map_err(native_error)?;
    let ws_bound = ktann_usearch::workspace_bound(n, M).map_err(native_error)?;
    let neighbors_bytes = n
        .checked_mul(std::mem::size_of::<LeafNeighbors>())
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    let fixed_bytes = build_limit
        .checked_add(neighbors_bytes)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    if fixed_bytes > memory_limit_bytes {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let remaining_for_ws = memory_limit_bytes - fixed_bytes;
    let max_affordable_workers = remaining_for_ws.checked_div(ws_bound).unwrap_or(8);
    if max_affordable_workers == 0 {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let workers = parallelism
        .clamp(1, 8)
        .min(n.max(1))
        .min(max_affordable_workers);

    let failure = std::sync::Mutex::new(None);
    let record_err = |error: Error| {
        let mut guard = failure.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(error);
        }
    };
    let proceed = || match checkpoint() {
        Ok(()) => true,
        Err(error) => {
            record_err(error);
            false
        }
    };
    let distance = |a: u32, b: u32| {
        kernel
            .routing_distances_validated([leaves[a as usize].center()], leaves[b as usize].center())
            [0]
    };
    let graph =
        ktann_usearch::Graph::build(n, M, EF_CONSTRUCTION, 7, build_limit, &distance, &proceed)
            .map_err(|e| {
                if let Some(err) = failure.lock().unwrap_or_else(|p| p.into_inner()).take() {
                    err
                } else {
                    native_error(e)
                }
            })?;

    let mut neighbors = vec![[(f64::INFINITY, usize::MAX); NEIGHBOR_COUNT]; n];
    let chunk_size = n.div_ceil(workers).max(1);
    let graph_ref = &graph;
    let proceed_ref = &proceed;

    let scope_res = std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for (chunk_idx, slice) in neighbors.chunks_mut(chunk_size).enumerate() {
            let base = chunk_idx * chunk_size;
            handles.push(scope.spawn(move || -> Result<()> {
                let mut scratch =
                    ktann_usearch::Workspace::new(n, M, ws_bound).map_err(native_error)?;
                let mut hits = [ktann_usearch::Hit::default(); NEIGHBOR_COUNT + 1];
                for (offset, retained) in slice.iter_mut().enumerate() {
                    let source = base + offset;
                    let q_vec = leaves[source].center();
                    let query_distance = |_: u32, target: u32| {
                        kernel
                            .routing_distances_validated([q_vec], leaves[target as usize].center())
                            [0]
                    };
                    let query_batch = |_: u32, nodes: &[u32], out: &mut [f64]| match nodes {
                        [a, b, c, d] => {
                            let scores = kernel.routing_distances_validated(
                                [
                                    leaves[*a as usize].center(),
                                    leaves[*b as usize].center(),
                                    leaves[*c as usize].center(),
                                    leaves[*d as usize].center(),
                                ],
                                q_vec,
                            );
                            out.copy_from_slice(&scores);
                        }
                        [a, b, c] => {
                            let scores = kernel.routing_distances_validated(
                                [
                                    leaves[*a as usize].center(),
                                    leaves[*b as usize].center(),
                                    leaves[*c as usize].center(),
                                ],
                                q_vec,
                            );
                            out.copy_from_slice(&scores);
                        }
                        [a, b] => {
                            let scores = kernel.routing_distances_validated(
                                [leaves[*a as usize].center(), leaves[*b as usize].center()],
                                q_vec,
                            );
                            out.copy_from_slice(&scores);
                        }
                        _ => {
                            for (dst, &node) in out.iter_mut().zip(nodes) {
                                *dst = kernel.routing_distances_validated(
                                    [q_vec],
                                    leaves[node as usize].center(),
                                )[0];
                            }
                        }
                    };
                    graph_ref
                        .search_batched(
                            EF_SEARCH,
                            &query_distance,
                            &query_batch,
                            proceed_ref,
                            &mut scratch,
                            &mut hits,
                        )
                        .map_err(native_error)?;
                    for hit in &hits {
                        if hit.node as usize != source {
                            retain_neighbor(retained, (hit.distance, hit.node as usize));
                        }
                    }
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().map_err(|_| Error::new(ErrorKind::Other))??;
        }
        Ok(())
    });

    if let Some(err) = failure.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(err);
    }
    scope_res?;
    Ok(neighbors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Metric, PartitionKey};
    use crate::refinement::Part;
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

        // Also cover cancellation during the multi-threaded batched search phase after Graph::build.
        let search_calls = std::sync::atomic::AtomicUsize::new(0);
        let search_check = || {
            let cur = search_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if cur >= 1500 {
                Err(Error::new(ErrorKind::DeadlineExceeded))
            } else {
                Ok(())
            }
        };
        assert_eq!(
            neighbors_parallel(&search_check, &kernel, &leaves, 4, 64 << 20)
                .unwrap_err()
                .kind(),
            ErrorKind::DeadlineExceeded
        );
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
