//! Budgeted refinement of a completed construction before the Forest Artifact seals.

use std::{collections::BTreeMap, path::Path};

use super::files::corrupt;
use super::forest::ForestOptions;
use crate::api::{Error, ErrorKind, IndexConfig, Result};
use crate::construction::{
    ConstructionRecord, ConstructionReport, PartitionPlan, Run, construct_replayable,
    construct_tree,
};
use crate::refinement::{
    LeafCenter, accept_moves, add_to_sum, best_move, candidate_scratch_bytes, finish_mean,
    leaf_neighbors,
};
use crate::search::numeric::VectorKernel;

/// Emit plans directly when refinement is omitted. Otherwise admit each retained
/// plan and reserve round workspace before spending the remainder on routing vectors.
pub(super) fn build_tree(
    directory: &Path,
    config: &IndexConfig,
    seed: [u8; 32],
    options: ForestOptions,
    records: impl Iterator<Item = Result<ConstructionRecord>>,
    mut emit: impl FnMut(PartitionPlan) -> Result<()>,
    check: &impl Fn() -> Result<()>,
) -> Result<ConstructionReport> {
    let Some(rounds) = options.refinement_rounds else {
        return construct_tree(
            directory,
            config,
            seed,
            options.construction(),
            records,
            emit,
        );
    };
    let limit = || Error::new(ErrorKind::LimitExceeded);
    // Include the replay reader, per-partition centroids and candidates, and per-entry IDs and moves.
    let mut remaining = options
        .refinement_memory_bytes
        .checked_sub(272 * 1024)
        .ok_or_else(limit)?;
    let mut plans = Vec::new();
    let (report, run) = construct_replayable(
        directory,
        config,
        seed,
        options.construction(),
        records,
        |plan| {
            remaining = remaining
                .checked_sub(2048 + config.dimension() * 24)
                .ok_or_else(limit)?;
            for entry in &plan.entries {
                remaining = remaining
                    .checked_sub(176 + 2 * entry.len())
                    .ok_or_else(limit)?;
            }
            plans.push(plan);
            Ok(())
        },
    )?;
    let leaves = plans.partition_point(|p| p.level == 1);
    let cache_budget = remaining
        .checked_sub(candidate_scratch_bytes(leaves, usize::from(rounds))?)
        .ok_or_else(limit)?;
    refine_plans(config, seed, &run, &mut plans, rounds, cache_budget, check)?;
    run.remove()?;
    for plan in plans {
        emit(plan)?;
    }
    Ok(report)
}

impl LeafCenter for PartitionPlan {
    fn center(&self) -> &[f32] {
        self.centroid.components()
    }
}

/// Construction emits leaves first in ascending key order, then their ancestors.
/// Positions in the sorted record stream break gain ties and fix the f64 addition order.
fn refine_plans(
    config: &IndexConfig,
    seed: [u8; 32],
    run: &Run,
    plans: &mut [PartitionPlan],
    rounds: u8,
    cache_budget: usize,
    check: &impl Fn() -> Result<()>,
) -> Result<()> {
    let kernel = VectorKernel::new(config.dimension(), config.metric(), seed)?;
    let n = plans.partition_point(|p| p.level == 1);
    refine_leaves(
        config,
        &kernel,
        run,
        &mut plans[..n],
        rounds,
        cache_budget,
        check,
    )?;
    if n < plans.len() {
        refresh_parents(&kernel, plans, check)?;
    }
    Ok(())
}

/// Counts enforce occupancy during arbitration. The record-to-leaf assignment is authoritative.
fn refine_leaves(
    config: &IndexConfig,
    kernel: &VectorKernel,
    run: &Run,
    leaves: &mut [PartitionPlan],
    rounds: u8,
    cache_budget: usize,
    check: &impl Fn() -> Result<()>,
) -> Result<()> {
    let mut counts: Vec<_> = leaves.iter().map(|leaf| leaf.entries.len()).collect();
    let mut rows: Vec<_> = leaves
        .iter_mut()
        .enumerate()
        .flat_map(|(i, leaf)| leaf.entries.drain(..).map(move |id| (id, i)))
        .collect();
    rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let dimension = kernel.dimension();
    let (ids, mut owners): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    let mut vectors = run.vectors(&ids, dimension, cache_budget, check)?;
    let mut sums = vec![0.0; leaves.len() * dimension];
    for round in 0..=rounds {
        sums.fill(0.0);
        vectors.scan(|i, vector| {
            check()?;
            let start = owners[i] * dimension;
            add_to_sum(&mut sums[start..start + dimension], vector);
            Ok(())
        })?;
        for ((leaf, &count), sum) in leaves
            .iter_mut()
            .zip(&counts)
            .zip(sums.chunks_exact(dimension))
        {
            if count != 0 {
                leaf.centroid = finish_mean(kernel, sum, count, 1)?;
            }
        }
        if round == rounds || leaves.len() < 2 {
            break;
        }
        let neighbors = leaf_neighbors(check, kernel, leaves)?;
        let mut moves = Vec::new();
        vectors.scan(|i, vector| {
            check()?;
            let source = owners[i];
            if let Some((gain, target)) =
                best_move(kernel, leaves, source, vector, &neighbors[source])
            {
                moves.push((gain, i, source, target));
            }
            Ok(())
        })?;
        drop(neighbors);
        let accepted = accept_moves(
            moves,
            &mut counts,
            config.min_leaf_partition_entries() as usize,
            config.max_leaf_partition_entries() as usize,
            |i, _, target| owners[i] = target,
        );
        if accepted == 0 {
            break;
        }
    }
    for (id, leaf) in ids.into_iter().zip(owners) {
        leaves[leaf].entries.push(id);
    }
    Ok(())
}

/// Internal plan entries are child keys in storage order. Construction emits every child before its parent.
fn refresh_parents(
    kernel: &VectorKernel,
    plans: &mut [PartitionPlan],
    check: &impl Fn() -> Result<()>,
) -> Result<()> {
    let mut centers = BTreeMap::<[u8; 8], &[f32]>::new();
    let mut sum = vec![0.0; kernel.dimension()];
    for plan in plans {
        check()?;
        if plan.level > 1 {
            sum.fill(0.0);
            for child in &plan.entries {
                add_to_sum(&mut sum, centers.get(child.as_ref()).ok_or_else(corrupt)?);
            }
            plan.centroid = finish_mean(kernel, &sum, plan.entries.len(), plan.level)?;
        }
        centers.insert(plan.key.get().to_be_bytes(), plan.centroid.components());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Metric, PartitionKey};
    use crate::construction::{ConstructionOptions, ConstructionRecord, construct_replayable};
    use crate::storage::values::PartitionCentroid;
    use bytes::Bytes;

    #[test]
    fn cache_budget_preserves_centroids_and_cross_leaf_moves() {
        let id = |i: u8| Bytes::from(vec![i; usize::from(i) + 1]);
        let key = |k| PartitionKey::new(k).unwrap();
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let config = IndexConfig::new(2, metric)
                .unwrap()
                .with_leaf_partition_entries(2, 8)
                .unwrap()
                .with_internal_partition_entries(2, 8)
                .unwrap();
            let dir = tempfile::tempdir().unwrap();
            let kernel = VectorKernel::new(2, metric, [0; 32]).unwrap();
            let records = (0u8..8).map(|i| {
                Ok(ConstructionRecord {
                    id: id(i),
                    vector: vec![if i < 4 { -1.0 } else { 1.0 }, 0.0].into_boxed_slice(),
                })
            });
            let (_, run) = construct_replayable(
                &dir.path().join("tree"),
                &config,
                [0; 32],
                ConstructionOptions {
                    sample_items: 4,
                    memory_bytes: 4 << 20,
                    scratch_bytes: 4 << 20,
                },
                records,
                |_| Ok(()),
            )
            .unwrap();
            for rounds in [0, 2] {
                let mut expected = None;
                for cache in [usize::MAX, 3 * (8 + 32), 0] {
                    // Each leaf deliberately starts with one member from the opposite cluster.
                    let mut plans = vec![
                        PartitionPlan {
                            key: key(2),
                            level: 1,
                            centroid: PartitionCentroid::new([999.0, 999.0]),
                            entries: [0, 1, 2, 4].map(id).into(),
                        },
                        PartitionPlan {
                            key: key(3),
                            level: 1,
                            centroid: PartitionCentroid::new([999.0, 999.0]),
                            entries: [3, 5, 6, 7].map(id).into(),
                        },
                        PartitionPlan {
                            key: key(1),
                            level: 2,
                            centroid: PartitionCentroid::new([999.0, 999.0]),
                            entries: [2u64, 3]
                                .map(|k| Bytes::copy_from_slice(&k.to_be_bytes()))
                                .into(),
                        },
                    ];
                    refine_plans(
                        &config,
                        [0; 32],
                        &run,
                        &mut plans,
                        rounds,
                        cache,
                        &|| Ok(()),
                    )
                    .unwrap();
                    assert_eq!(
                        plans[0].entries,
                        if rounds == 0 {
                            [0, 1, 2, 4]
                        } else {
                            [0, 1, 2, 3]
                        }
                        .map(id)
                    );
                    if rounds == 0 {
                        let negative = kernel.preprocess(&[-1.0, 0.0]).unwrap();
                        let positive = kernel.preprocess(&[1.0, 0.0]).unwrap();
                        let average: Vec<_> = negative
                            .iter()
                            .zip(positive.iter())
                            .map(|(&a, &b)| ((3.0 * f64::from(a) + f64::from(b)) / 4.0) as f32)
                            .collect();
                        assert_eq!(
                            plans[0].centroid.components(),
                            kernel.normalize_centroid(&average).unwrap().as_ref()
                        );
                    }
                    let bytes: Vec<_> = plans
                        .iter()
                        .map(|plan| {
                            let mut v = Vec::new();
                            crate::bulk::plan::encode(plan, &mut v);
                            v
                        })
                        .collect();
                    if let Some(prior) = &expected {
                        assert_eq!(&bytes, prior);
                    } else {
                        expected = Some(bytes);
                    }
                }
            }
            run.remove().unwrap();
        }
    }
}
