//! Converts construction plans to the shared refinement model and back.

use std::collections::BTreeMap;

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::refinement::{Part, Planner};
use crate::storage::values::PartitionCentroid;

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// Pure artifact preparation reuses the ordinary numerical assignment algorithm.
/// The caller admits retained input and scratch, and discards its private plans on failure.
pub(crate) fn refine_plans(
    config: &crate::api::IndexConfig,
    seed: [u8; 32],
    records: Vec<crate::construction::ConstructionRecord>,
    plans: &mut [crate::construction::PartitionPlan],
    rounds: u8,
    check: &impl Fn() -> Result<()>,
) -> Result<()> {
    let mut ids = Vec::with_capacity(records.len());
    let mut vectors = Vec::with_capacity(records.len());
    for record in records {
        check()?;
        ids.push(record.id);
        vectors.push(record.vector);
    }
    let planner = Planner::new(config, seed, vectors, check)?;
    let positions: BTreeMap<_, _> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i))
        .collect();
    let mut parts = Vec::with_capacity(plans.len());
    for plan in plans.iter_mut() {
        check()?;
        let members = if plan.level == 1 {
            plan.entries
                .iter()
                .map(|id| positions.get(id).copied().ok_or_else(corrupt))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let children = if plan.level > 1 {
            plan.entries
                .iter()
                .map(|id| {
                    PartitionKey::new(u64::from_be_bytes(
                        id.as_ref().try_into().map_err(|_| corrupt())?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        parts.push(Part {
            key: plan.key,
            parent: None,
            level: plan.level,
            centroid: std::mem::replace(&mut plan.centroid, PartitionCentroid::new([])),
            members,
            children,
        });
    }
    let positions = planner.refine_tree(&mut parts, rounds as usize, check, |_, _, _, _| {})?;
    for plan in plans {
        check()?;
        let part = &mut parts[positions[&plan.key]];
        plan.centroid = std::mem::replace(&mut part.centroid, PartitionCentroid::new([]));
        if plan.level == 1 {
            plan.entries = part.members.iter().map(|&i| ids[i].clone()).collect();
            plan.entries.sort();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::numeric::VectorKernel;
    use bytes::Bytes;

    #[test]
    fn artifact_zero_rounds_refreshes_centroids_without_changing_membership() {
        use crate::api::{IndexConfig, Metric};
        use crate::construction::{ConstructionRecord, PartitionPlan};
        let config = IndexConfig::new(1, Metric::L2).unwrap();
        let mut plans = vec![PartitionPlan {
            key: PartitionKey::new(1).unwrap(),
            level: 1,
            centroid: PartitionCentroid::new(vec![999.0]),
            entries: vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")],
        }];
        let before = plans[0].entries.clone();
        let records = vec![
            ConstructionRecord {
                id: before[0].clone(),
                vector: vec![2.0].into_boxed_slice(),
            },
            ConstructionRecord {
                id: before[1].clone(),
                vector: vec![4.0].into_boxed_slice(),
            },
        ];
        refine_plans(&config, [0; 32], records, &mut plans, 0, &|| Ok(())).unwrap();
        assert_eq!(plans[0].entries, before);
        let kernel = VectorKernel::new(1, Metric::L2, [0; 32]).unwrap();
        let expected = kernel.preprocess(&[3.0]).unwrap();
        assert_eq!(plans[0].centroid.components(), expected.as_ref());
    }
}
