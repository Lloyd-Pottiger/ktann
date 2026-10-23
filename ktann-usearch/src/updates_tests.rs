use super::*;

const LIMIT: usize = 16 * 1024 * 1024;

fn source() -> Graph {
    Graph::build(
        32,
        4,
        24,
        7,
        LIMIT,
        &|a, b| f64::from(a.abs_diff(b)),
        &|| true,
    )
    .unwrap()
}

#[test]
fn private_updates_remap_insertions_and_roundtrip_without_changing_the_source() {
    let original = source();
    let before = original.encode(LIMIT, &|| true).unwrap();
    // Insertions before or after the old children exercise changed and preserved
    // ordinals. Both cases also replace one old centroid.
    for mapping in [(2..34).chain(0..2).collect::<Vec<u32>>(), (0..34).collect()] {
        let copy = Graph::decode(&before, 32, 4, LIMIT, &|| true).unwrap();
        let coordinate = |slot: u32| {
            if slot == 4 {
                0.25
            } else {
                f64::from(mapping[slot as usize])
            }
        };
        let distance = |a, b| (coordinate(a) - coordinate(b)).abs();
        let updated = copy
            .evolve(&mapping, &[4], 24, 19, LIMIT, &distance, &|| true)
            .unwrap();
        let encoded = updated.encode(LIMIT, &|| true).unwrap();
        let restored = Graph::decode(&encoded, 34, 4, LIMIT, &|| true).unwrap();
        assert_eq!(restored.encode(LIMIT, &|| true).unwrap(), encoded);
        assert_eq!(original.encode(LIMIT, &|| true).unwrap(), before);
        let mut workspace = Workspace::new(34, 4, LIMIT).unwrap();
        let mut hits = vec![Hit::default(); 4];
        let final_coordinate = |node| {
            if node == mapping[4] {
                0.25
            } else {
                f64::from(node)
            }
        };
        restored
            .search(
                34,
                &|_, b| final_coordinate(b).abs(),
                &|| true,
                &mut workspace,
                &mut hits,
            )
            .unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.node).collect::<Vec<_>>(),
            [0, mapping[4], 1, 2]
        );
    }
}

#[test]
fn updates_reject_invalid_permutations_and_changed_slots() {
    for (mapping, changes) in [
        (vec![0; 32], vec![]),
        ((0..31).collect(), vec![]),
        ((0..32).collect(), vec![32]),
        ((0..32).collect(), vec![1, 1]),
    ] {
        assert!(matches!(
            source().evolve(&mapping, &changes, 24, 7, LIMIT, &|_, _| 0.0, &|| true),
            Err(Error::InvalidArgument)
        ));
    }
    assert!(matches!(
        source().evolve(
            &(0..32).collect::<Vec<_>>(),
            &[1],
            24,
            7,
            0,
            &|_, _| 0.0,
            &|| true
        ),
        Err(Error::LimitExceeded)
    ));
}

#[test]
fn every_update_cancellation_boundary_discards_private_state() {
    use std::cell::Cell;
    let bytes = source().encode(LIMIT, &|| true).unwrap();
    let mapping: Vec<u32> = (0..34).collect();
    let calls = Cell::new(0);
    let graph = Graph::decode(&bytes, 32, 4, LIMIT, &|| true).unwrap();
    graph
        .evolve(&mapping, &[0, 31], 24, 7, LIMIT, &|_, _| 0.0, &|| {
            calls.set(calls.get() + 1);
            true
        })
        .unwrap();
    for stop in 0..calls.get() {
        let graph = Graph::decode(&bytes, 32, 4, LIMIT, &|| true).unwrap();
        let current = Cell::new(0);
        let result = graph.evolve(&mapping, &[0, 31], 24, 7, LIMIT, &|_, _| 0.0, &|| {
            let next = current.get();
            current.set(next + 1);
            next != stop
        });
        assert!(
            matches!(result, Err(Error::Cancelled)),
            "checkpoint {stop}: {result:?}"
        );
    }
}

#[test]
fn retention_removes_entry_and_batch_nodes_and_roundtrips() {
    let original = source();
    let bytes = original.encode(LIMIT, &|| true).unwrap();
    for survivors in [
        vec![31],
        (0..32).filter(|&i| Some(i) != original.entry()).collect(),
        (0..32).step_by(3).collect(),
    ] {
        let distance =
            |a: u32, b: u32| f64::from(survivors[a as usize].abs_diff(survivors[b as usize]));
        let graph = Graph::decode(&bytes, 32, 4, LIMIT, &|| true)
            .unwrap()
            .retain(&survivors, 24, LIMIT, &distance, &|| true)
            .unwrap();
        let mut reached = vec![false; survivors.len()];
        let mut pending = vec![graph.entry().unwrap()];
        let mut edges = [0; 8];
        while let Some(node) = pending.pop() {
            if std::mem::replace(&mut reached[node as usize], true) {
                continue;
            }
            let count = graph.neighbors(node, 0, &mut edges).unwrap();
            pending.extend_from_slice(&edges[..count]);
        }
        assert!(reached.iter().all(|seen| *seen));
        let highest = (0..survivors.len() as u32)
            .map(|node| graph.level(node).unwrap())
            .max()
            .unwrap();
        assert_eq!(graph.level(graph.entry().unwrap()), Some(highest));
        let encoded = graph.encode(LIMIT, &|| true).unwrap();
        let restored = Graph::decode(&encoded, survivors.len(), 4, LIMIT, &|| true).unwrap();
        assert_eq!(encoded, restored.encode(LIMIT, &|| true).unwrap());
        let mut workspace = Workspace::new(survivors.len(), 4, LIMIT).unwrap();
        let mut hits = vec![Hit::default(); survivors.len()];
        restored
            .search(
                survivors.len(),
                &|_, a| f64::from(survivors[a as usize]),
                &|| true,
                &mut workspace,
                &mut hits,
            )
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.node).collect::<Vec<_>>(),
            (0..survivors.len() as u32).collect::<Vec<_>>()
        );
    }
    assert_eq!(bytes, original.encode(LIMIT, &|| true).unwrap());
}

#[test]
fn every_retention_cancellation_boundary_discards_private_state() {
    use std::cell::Cell;
    let bytes = source().encode(LIMIT, &|| true).unwrap();
    let survivors: Vec<_> = (0..32).step_by(2).collect();
    let calls = Cell::new(0);
    Graph::decode(&bytes, 32, 4, LIMIT, &|| true)
        .unwrap()
        .retain(
            &survivors,
            24,
            LIMIT,
            &|a, b| f64::from(a.abs_diff(b)),
            &|| {
                calls.set(calls.get() + 1);
                true
            },
        )
        .unwrap();
    for stop in 0..calls.get() {
        let current = Cell::new(0);
        let result = Graph::decode(&bytes, 32, 4, LIMIT, &|| true)
            .unwrap()
            .retain(
                &survivors,
                24,
                LIMIT,
                &|a, b| f64::from(a.abs_diff(b)),
                &|| {
                    let n = current.get();
                    current.set(n + 1);
                    n != stop
                },
            );
        assert!(
            matches!(result, Err(Error::Cancelled)),
            "checkpoint {stop}: {result:?}"
        );
    }
    for survivors in [vec![], vec![32], vec![1, 1], vec![2, 1]] {
        assert!(matches!(
            source().retain(&survivors, 24, LIMIT, &|_, _| 0.0, &|| true),
            Err(Error::InvalidArgument)
        ));
    }
    assert!(matches!(
        source().retain(&[1], 24, 0, &|_, _| 0.0, &|| true),
        Err(Error::LimitExceeded)
    ));
}

/// Fixed-budget routing quality and distance work for a wide parent graph.
#[test]
#[ignore = "release-mode deletion cost and recall experiment"]
fn wide_parent_retention_cost() {
    use std::{cell::Cell, time::Instant};
    let n = 4096;
    let mut seed = 19_u64;
    let vectors: Vec<Vec<f64>> = (0..n + 128)
        .map(|i| {
            (0..128)
                .map(|d| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 32) as u32 as f64 / u32::MAX as f64)
                        + ((i % 64 + d % 8) % 16) as f64 * 0.2
                })
                .collect()
        })
        .collect();
    // Optional real routing-vector sample; consume only the measured prefix.
    let vectors = if let Ok(path) = std::env::var("KTANN_GRAPH_BENCH_FVECS") {
        use std::io::Read;
        let mut input = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        (0..n + 128)
            .map(|_| {
                let mut dimension = [0; 4];
                input.read_exact(&mut dimension).unwrap();
                assert_eq!(u32::from_le_bytes(dimension), 128);
                (0..128)
                    .map(|_| {
                        let mut bytes = [0; 4];
                        input.read_exact(&mut bytes).unwrap();
                        f64::from(f32::from_le_bytes(bytes))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    } else {
        vectors
    };
    let squared = |a: usize, b: usize| {
        vectors[a]
            .iter()
            .zip(&vectors[b])
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
    };
    let original = Graph::build(
        n,
        16,
        64,
        19,
        LIMIT,
        &|a, b| squared(a as usize, b as usize),
        &|| true,
    )
    .unwrap();
    let bytes = original.encode(LIMIT, &|| true).unwrap();
    for removed in [1, 128] {
        for repeat in 0..3 {
            let survivors: Vec<u32> = (0..n as u32)
                .filter(|i| !(*i >= 100 && *i < 100 + removed))
                .collect();
            let calls = Cell::new(0_u64);
            let distance = |a: u32, b: u32| {
                calls.set(calls.get() + 1);
                squared(
                    survivors[a as usize] as usize,
                    survivors[b as usize] as usize,
                )
            };
            for method in ["rebuild", "retain"] {
                calls.set(0);
                let graph = Graph::decode(&bytes, n, 16, LIMIT, &|| true).unwrap();
                let started = Instant::now();
                let updated = if method == "retain" {
                    graph
                        .retain(&survivors, 64, LIMIT, &distance, &|| true)
                        .unwrap()
                } else {
                    drop(graph);
                    Graph::build(survivors.len(), 16, 64, 20, LIMIT, &distance, &|| true).unwrap()
                };
                let elapsed = started.elapsed().as_secs_f64();
                let work = calls.get();
                let mut ws = Workspace::new(survivors.len(), 16, LIMIT).unwrap();
                let mut hits = vec![Hit::default(); 10];
                let mut correct = 0;
                let mut search_seconds = 0.0;
                let query_distances = Cell::new(0_u64);
                for query in n..n + 128 {
                    let mut oracle: Vec<_> = survivors
                        .iter()
                        .enumerate()
                        .map(|(slot, &old)| (squared(query, old as usize), slot as u32))
                        .collect();
                    oracle.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    let search_start = Instant::now();
                    updated
                        .search(
                            64,
                            &|_, b| {
                                query_distances.set(query_distances.get() + 1);
                                squared(query, survivors[b as usize] as usize)
                            },
                            &|| true,
                            &mut ws,
                            &mut hits,
                        )
                        .unwrap();
                    search_seconds += search_start.elapsed().as_secs_f64();
                    correct += hits
                        .iter()
                        .filter(|h| oracle[..10].iter().any(|(_, slot)| *slot == h.node))
                        .count();
                }
                eprintln!(
                    "delete={removed} repeat={repeat} method={method} seconds={elapsed:.6} distances={work} recall={:.6} query_seconds={search_seconds:.6} query_distances={} resident={} peak={}",
                    correct as f64 / 1280.0,
                    query_distances.get(),
                    updated.resident_bytes(),
                    updated.peak_bytes()
                );
            }
        }
    }
}

#[test]
fn retention_allocation_failures_leave_source_usable() {
    let original = source();
    let bytes = original.encode(LIMIT, &|| true).unwrap();
    let survivors: Vec<_> = (0..32).step_by(2).collect();
    let metric = |a: u32, b: u32| f64::from(a.abs_diff(b));
    let updated = Graph::decode(&bytes, 32, 4, LIMIT, &|| true)
        .unwrap()
        .retain(&survivors, 24, LIMIT, &metric, &|| true)
        .unwrap();
    let mut failures = 0;
    for limit in (original.resident_bytes()..updated.peak_bytes()).step_by(64) {
        let candidate = Graph::decode(&bytes, 32, 4, LIMIT, &|| true)
            .unwrap()
            .retain(&survivors, 24, limit, &metric, &|| true);
        match candidate {
            Err(Error::LimitExceeded) => failures += 1,
            Ok(_) => {}
            other => panic!("{other:?}"),
        }
    }
    assert!(failures > 0);
    assert_eq!(bytes, original.encode(LIMIT, &|| true).unwrap());
}
