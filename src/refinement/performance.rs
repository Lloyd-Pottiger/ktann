//! Reproducible CPU diagnostics; setup and output verification are untimed.

use super::*;
use crate::api::Metric;
use std::time::Instant;

#[test]
#[ignore = "release-mode CPU diagnostic"]
fn refinement_performance() {
    for (count, dimension) in [(256, 128), (4096, 128), (4096, 768)] {
        let kernel = VectorKernel::new(dimension, Metric::L2, [7; 32]).unwrap();
        let mut state = 7_u64;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1_u32 << 24) as f32
        };
        let leaves: Vec<_> = (0..count)
            .map(|i| Part {
                key: PartitionKey::new(i as u64 + 1).unwrap(),
                parent: None,
                level: 1,
                center: (0..dimension).map(|_| random()).collect(),
                members: Vec::new(),
                children: Vec::new(),
            })
            .collect();
        let mut expected = None;
        for repetition in 0..4 {
            let start = Instant::now();
            let neighbors = leaf_neighbors(&|| Ok(()), &kernel, &leaves).unwrap();
            let elapsed = start.elapsed();
            let digest = neighbors.iter().flatten().fold(0_u64, |digest, &(d, i)| {
                digest.rotate_left(7) ^ d.to_bits() ^ i as u64
            });
            assert_eq!(*expected.get_or_insert(digest), digest);
            eprintln!(
                "neighbors leaves={count} dimension={dimension} repetition={repetition} seconds={:.6} digest={digest:016x}",
                elapsed.as_secs_f64()
            );
        }
    }
}

#[test]
#[ignore = "release-mode CPU diagnostic"]
fn refinement_sift_performance() {
    use crate::construction::{ConstructionOptions, ConstructionRecord, construct_tree};
    use bytes::Bytes;

    let input = std::fs::read("tests/datadriven/data/siftsmall_base.fvecs").unwrap();
    let dimension = u32::from_le_bytes(input[..4].try_into().unwrap()) as usize;
    let vectors: Vec<Box<[f32]>> = input
        .chunks_exact(4 + dimension * 4)
        .map(|row| {
            row[4..]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect()
        })
        .collect();
    let config = IndexConfig::new(dimension, Metric::L2).unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let mut plans = Vec::new();
    construct_tree(
        &scratch.path().join("tree"),
        &config,
        [7; 32],
        ConstructionOptions {
            sample_items: 1024,
            memory_bytes: 256 * 1024 * 1024,
            scratch_bytes: 1024 * 1024 * 1024,
        },
        vectors.iter().enumerate().map(|(i, v)| {
            Ok(ConstructionRecord {
                id: Bytes::copy_from_slice(&(i as u64).to_be_bytes()),
                vector: v.clone(),
            })
        }),
        |plan| {
            plans.push(plan);
            Ok(())
        },
    )
    .unwrap();
    let planner = Planner::new(&config, [7; 32], vectors, &|| Ok(())).unwrap();
    let mut expected = None;
    for repetition in 0..4 {
        let mut parts: Vec<_> = plans
            .iter()
            .map(|p| Part {
                key: p.key,
                parent: None,
                level: p.level,
                center: p.centroid.clone(),
                members: if p.level == 1 {
                    p.entries
                        .iter()
                        .map(|v| u64::from_be_bytes(v.as_ref().try_into().unwrap()) as usize)
                        .collect()
                } else {
                    Vec::new()
                },
                children: if p.level > 1 {
                    p.entries
                        .iter()
                        .map(|v| {
                            PartitionKey::new(u64::from_be_bytes(v.as_ref().try_into().unwrap()))
                                .unwrap()
                        })
                        .collect()
                } else {
                    Vec::new()
                },
            })
            .collect();
        let start = Instant::now();
        let mut moved = 0;
        planner
            .refine_tree(&mut parts, 3, &|| Ok(()), |_, _, _, _| moved += 1)
            .unwrap();
        let elapsed = start.elapsed();
        let digest = parts.iter().fold(0_u64, |mut digest, p| {
            for &i in &p.members {
                digest = digest.rotate_left(7) ^ i as u64;
            }
            for &v in &p.center {
                digest = digest.rotate_left(7) ^ u64::from(v.to_bits());
            }
            digest
        });
        assert_eq!(*expected.get_or_insert((digest, moved)), (digest, moved));
        eprintln!(
            "siftsmall rounds=3 repetition={repetition} seconds={:.6} moves={moved} digest={digest:016x}",
            elapsed.as_secs_f64()
        );
    }
}

#[test]
#[ignore = "release-mode CPU diagnostic"]
fn refinement_large_performance() {
    for dimension in [128, 768] {
        let count = 4096;
        let occupancy = 32;
        let mut state = 17_u64;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1_u32 << 24) as f32
        };
        let centers: Vec<Vec<_>> = (0..count)
            .map(|_| (0..dimension).map(|_| random()).collect())
            .collect();
        let vectors = (0..count)
            .flat_map(|leaf| {
                let centers = &centers;
                (0..occupancy).map(move |member| {
                    // Misplace one eighth of the inputs to exercise accepted moves.
                    let center = &centers[(leaf + usize::from(member % 8 == 0)) % count];
                    center
                        .iter()
                        .enumerate()
                        .map(|(d, &v)| v + ((d + member * 17) % 11) as f32 * 0.001)
                        .collect::<Box<[f32]>>()
                })
            })
            .collect();
        let config = IndexConfig::new(dimension, Metric::L2).unwrap();
        let planner = Planner::new(&config, [7; 32], vectors, &|| Ok(())).unwrap();
        let mut expected = None;
        for repetition in 0..4 {
            let mut parts: Vec<_> = (0..count)
                .map(|leaf| Part {
                    key: PartitionKey::new(leaf as u64 + 1).unwrap(),
                    parent: None,
                    level: 1,
                    center: vec![0.0; dimension].into_boxed_slice(),
                    members: (leaf * occupancy..(leaf + 1) * occupancy).collect(),
                    children: Vec::new(),
                })
                .collect();
            let start = Instant::now();
            let mut moved = 0;
            planner
                .refine_tree(&mut parts, 3, &|| Ok(()), |_, _, _, _| moved += 1)
                .unwrap();
            let elapsed = start.elapsed();
            let digest = parts.iter().fold(0_u64, |mut digest, p| {
                for &i in &p.members {
                    digest = digest.rotate_left(7) ^ i as u64;
                }
                for &v in &p.center {
                    digest = digest.rotate_left(7) ^ u64::from(v.to_bits());
                }
                digest
            });
            assert_eq!(*expected.get_or_insert((digest, moved)), (digest, moved));
            eprintln!(
                "large leaves={count} dimension={dimension} rounds=3 repetition={repetition} seconds={:.6} moves={moved} digest={digest:016x}",
                elapsed.as_secs_f64()
            );
        }
    }
}
