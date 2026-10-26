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
fn private_insertions_remap_and_roundtrip_without_changing_the_source() {
    let original = source();
    let before = original.encode(LIMIT, &|| true).unwrap();
    // Insertions before or after the old children exercise shifted and preserved
    // ordinals while keeping every existing distance unchanged.
    for mapping in [(2..34).chain(0..2).collect::<Vec<u32>>(), (0..34).collect()] {
        let copy = Graph::decode(&before, 32, 4, LIMIT, &|| true).unwrap();
        let distance =
            |a: u32, b: u32| f64::from(mapping[a as usize].abs_diff(mapping[b as usize]));
        let updated = copy
            .insert(&mapping, 24, 19, LIMIT, &distance, &|| true)
            .unwrap();
        let encoded = updated.encode(LIMIT, &|| true).unwrap();
        let restored = Graph::decode(&encoded, 34, 4, LIMIT, &|| true).unwrap();
        assert_eq!(restored.encode(LIMIT, &|| true).unwrap(), encoded);
        assert_eq!(original.encode(LIMIT, &|| true).unwrap(), before);
        let mut workspace = Workspace::new(34, 4, LIMIT).unwrap();
        let mut hits = vec![Hit::default(); 4];
        restored
            .search(
                34,
                &|_, b| f64::from(b),
                &|| true,
                &mut workspace,
                &mut hits,
            )
            .unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.node).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }
}

#[test]
fn insertions_reject_invalid_permutations() {
    for mapping in [vec![0; 32], (0..31).collect(), (1..33).collect()] {
        assert!(matches!(
            source().insert(&mapping, 24, 7, LIMIT, &|_, _| 0.0, &|| true),
            Err(Error::InvalidArgument)
        ));
    }
    assert!(matches!(
        source().insert(&(0..32).collect::<Vec<_>>(), 24, 7, 0, &|_, _| 0.0, &|| {
            true
        }),
        Err(Error::LimitExceeded)
    ));
}

#[test]
fn every_insertion_cancellation_boundary_discards_private_state() {
    use std::cell::Cell;
    let bytes = source().encode(LIMIT, &|| true).unwrap();
    let mapping: Vec<u32> = (0..34).collect();
    let calls = Cell::new(0);
    let graph = Graph::decode(&bytes, 32, 4, LIMIT, &|| true).unwrap();
    graph
        .insert(&mapping, 24, 7, LIMIT, &|_, _| 0.0, &|| {
            calls.set(calls.get() + 1);
            true
        })
        .unwrap();
    for stop in 0..calls.get() {
        let graph = Graph::decode(&bytes, 32, 4, LIMIT, &|| true).unwrap();
        let current = Cell::new(0);
        let result = graph.insert(&mapping, 24, 7, LIMIT, &|_, _| 0.0, &|| {
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
