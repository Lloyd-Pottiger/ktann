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
