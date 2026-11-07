use super::*;
use crate::api::{DataType, Metric, Value};

/// Holds borrowed test projections beside native topology for score and encoded
/// round-trip comparisons. Production only exposes encoded builds
/// and complete snapshot-bound bodies.
#[derive(Debug)]
pub(crate) struct TestGraph<'a> {
    centroids: Centroids<'a>,
    graph: ktann_usearch::Graph,
    identity: [u8; 32],
}

impl<'a> TestGraph<'a> {
    pub(crate) fn build(
        centroids: Centroids<'a>,
        parameters: Parameters,
        tree: &TreeKey,
        parent: PartitionKey,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let (graph, identity) =
            build_graph(centroids, parameters, tree, parent, working_bytes, check)?;
        Ok(Self {
            centroids,
            graph,
            identity,
        })
    }

    pub(crate) fn search(
        &self,
        query: &[f32],
        beam: usize,
        ef: usize,
        working_bytes: usize,
        scratch: &mut Workspace,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<(Vec<Candidate>, ktann_usearch::Stats)> {
        search_graph(
            self.centroids,
            &self.graph,
            query,
            beam,
            ef,
            working_bytes,
            scratch,
            check,
        )
    }
}

use std::cell::Cell;

const MEMORY: usize = 128 * 1024 * 1024;

/// Measures production routing on fixed graphs; construction and result digests
/// stay outside query timing. Fresh scratch matches the production read path.
#[test]
#[ignore = "release-only routing performance diagnostic"]
fn local_routing_performance() {
    use std::hint::black_box;
    use std::time::Instant;

    for (n, dimension) in [(512, 128), (4096, 128), (4096, 768), (16384, 128)] {
        let mut seed = 19_u64;
        let mut vector = || {
            (0..dimension)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 40) as f32 / (1_u32 << 24) as f32
                })
                .collect::<Vec<_>>()
        };
        let rows = children((0..n).map(|_| vector()).collect());
        let queries: Vec<_> = (0..64).map(|_| vector()).collect();
        let kernel = VectorKernel::new(dimension, Metric::L2, [0; 32]).unwrap();
        let parameters = params();
        let started = Instant::now();
        let graph = build(&kernel, &rows, parameters);
        let build_us = started.elapsed().as_micros();
        let run = |query: &[f32]| {
            let mut scratch = Workspace::new(n, parameters.m, MEMORY).unwrap();
            graph
                .search(query, 16, 64, MEMORY, &mut scratch, &|| Ok(()))
                .unwrap()
        };
        let mut digest = Sha256::new();
        for query in &queries {
            let (hits, stats) = run(query);
            digest.update((stats.distances as u64).to_le_bytes());
            digest.update((stats.visits as u64).to_le_bytes());
            for hit in hits {
                digest.update(hit.node.to_le_bytes());
                digest.update(hit.distance.to_bits().to_le_bytes());
            }
        }
        let scratch = Workspace::new(n, parameters.m, MEMORY).unwrap();
        println!(
            "n={n} d={dimension} build_us={build_us} workspace_bytes={} digest={:02x?}",
            scratch.allocated_bytes(),
            digest.finalize()
        );
        drop(scratch);
        for round in 0..5 {
            let started = Instant::now();
            for query in &queries {
                black_box(run(black_box(query)));
            }
            println!(
                "n={n} d={dimension} round={round} ns_per_query={}",
                started.elapsed().as_nanos() / queries.len() as u128
            );
        }
    }
}

fn key(value: u64) -> PartitionKey {
    PartitionKey::new(value).unwrap()
}

fn tree() -> TreeKey {
    TreeKey::encode(&[], &[]).unwrap()
}

fn params() -> Parameters {
    Parameters {
        m: 12,
        ef_construction: 128,
        seed: [7; 32],
    }
}

fn children(rows: Vec<Vec<f32>>) -> Vec<ChildEntry> {
    rows.into_iter()
        .enumerate()
        .map(|(i, row)| ChildEntry::new(key(i as u64 * 3 + 10), row))
        .collect()
}

fn build<'a>(
    kernel: &'a VectorKernel,
    children: &'a [ChildEntry],
    parameters: Parameters,
) -> TestGraph<'a> {
    let centroids = Centroids::new(
        kernel,
        children,
        crate::api::MAX_PARTITION_ENTRIES as usize,
        &|| Ok(()),
    )
    .unwrap();
    TestGraph::build(centroids, parameters, &tree(), key(1), MEMORY, &|| Ok(())).unwrap()
}

// Independent scalar oracle: no graph distance callback or production selector.
fn oracle(metric: Metric, rows: &[ChildEntry], query: &[f32], beam: usize) -> Vec<Candidate> {
    let mut hits: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(node, child)| {
            let distance = match metric {
                Metric::L2 => query
                    .iter()
                    .zip(child.centroid())
                    .map(|(&a, &b)| {
                        let delta = a as f64 - b as f64;
                        delta * delta
                    })
                    .sum(),
                Metric::Cosine | Metric::InnerProduct => -query
                    .iter()
                    .zip(child.centroid())
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>(),
            };
            Candidate {
                node: node as u32,
                distance,
            }
        })
        .collect();
    hits.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap()
            .then(a.node.cmp(&b.node))
    });
    hits.truncate(beam);
    hits
}

// Fixed integer generator keeps fixture vectors independent from graph RNG.
fn fixture(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

/// Validate built topology through the same encoding boundary used by serving.
fn validated_encoding(graph: &TestGraph<'_>, m: usize) -> Vec<u8> {
    let bytes = graph.graph.encode(MEMORY, &|| true).unwrap();
    ktann_usearch::Graph::decode(&bytes, graph.centroids.children.len(), m, MEMORY, &|| true)
        .unwrap();
    bytes
}

#[test]
fn rejects_invalid_inputs_and_admits_checked_resource_bounds() {
    let kernel = VectorKernel::new(2, Metric::L2, [0; 32]).unwrap();
    for rows in [
        children(vec![vec![0.0]]),
        children(vec![vec![f32::NAN, 0.0]]),
        children(vec![vec![f32::INFINITY, 0.0]]),
        vec![
            ChildEntry::new(key(3), [0.0, 0.0]),
            ChildEntry::new(key(3), [1.0, 1.0]),
        ],
        vec![
            ChildEntry::new(key(5), [0.0, 0.0]),
            ChildEntry::new(key(3), [1.0, 1.0]),
        ],
    ] {
        assert_eq!(
            Centroids::new(&kernel, &rows, 16, &|| Ok(()))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
    }
    let rows = children(vec![vec![0.0; 2]; 2]);
    for capacity in [0, 1, crate::api::MAX_PARTITION_ENTRIES as usize + 1] {
        assert!(Centroids::new(&kernel, &rows, capacity, &|| Ok(())).is_err());
    }
    let view = Centroids::new(&kernel, &rows, 2, &|| Ok(())).unwrap();
    for (m, ef) in [(0, 10), (1, 10), (10, 9), (usize::MAX, usize::MAX)] {
        let invalid = Parameters {
            m,
            ef_construction: ef,
            ..params()
        };
        assert_eq!(
            TestGraph::build(view, invalid, &tree(), key(1), MEMORY, &|| Ok(()))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
    }
    let bound = view.bytes + ktann_usearch::build_bound(2, params().m).unwrap();
    assert_eq!(
        TestGraph::build(view, params(), &tree(), key(1), bound - 1, &|| panic!(
            "must reject before native work"
        ))
        .unwrap_err()
        .kind(),
        ErrorKind::LimitExceeded
    );
    let graph = TestGraph::build(view, params(), &tree(), key(1), bound, &|| Ok(())).unwrap();
    assert!(graph.graph.peak_bytes() <= bound - view.bytes);
    let mut scratch = Workspace::new(2, params().m, MEMORY).unwrap();
    for (query, beam, ef) in [
        (&[0.0][..], 1, 8),
        (&[f32::NAN, 0.0][..], 1, 8),
        (&[0.0, 0.0][..], 0, 8),
        (&[0.0, 0.0][..], 1, 0),
    ] {
        assert_eq!(
            graph
                .search(query, beam, ef, MEMORY, &mut scratch, &|| Ok(()))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
    }
    // Graph queries retain one result buffer; native scratch is already included.
    let query = [0.0; 2];
    let query_budget = view.bytes
        + size_of_val(&query)
        + graph.graph.resident_bytes()
        + scratch.allocated_bytes()
        + size_of::<Candidate>()
        + 1024;
    assert_eq!(
        graph
            .search(&query, 1, 8, query_budget, &mut scratch, &|| Ok(()))
            .unwrap()
            .0,
        oracle(Metric::L2, &rows, &query, 1)
    );
    assert_eq!(
        graph
            .search(&query, 1, 8, query_budget - 1, &mut scratch, &|| Ok(()))
            .unwrap_err()
            .kind(),
        ErrorKind::LimitExceeded
    );
    assert_eq!(
        view.exact(&[0.0; 2], 1, 1, &|| Ok(())).unwrap_err().kind(),
        ErrorKind::LimitExceeded
    );
    assert!(ktann_usearch::build_bound(usize::MAX, 2).is_err());
}

#[test]
fn empty_singleton_small_and_exhaustive_queries_match_independent_oracle() {
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        let kernel = VectorKernel::new(2, metric, [0; 32]).unwrap();
        for n in 0..=8 {
            let rows = children(fixture(n, 2, 3));
            let graph = build(&kernel, &rows, params());
            validated_encoding(&graph, params().m);
            let mut scratch = Workspace::new(n, params().m, MEMORY).unwrap();
            let query = [0.2, -0.7];
            let expected = oracle(metric, &rows, &query, n);
            let (hits, stats) = graph
                .search(
                    &query,
                    usize::MAX,
                    usize::MAX,
                    MEMORY,
                    &mut scratch,
                    &|| Ok(()),
                )
                .unwrap();
            assert_eq!(hits, expected);
            assert_eq!(
                stats.distances, n,
                "exhaustive scan must score each child exactly once"
            );
            assert_eq!(
                graph
                    .centroids
                    .exact(&query, usize::MAX, MEMORY, &|| Ok(()))
                    .unwrap(),
                expected
            );
            for hit in hits {
                assert_eq!(graph.centroids.child(hit), rows[hit.node as usize].child());
            }
        }
    }
}

#[test]
fn numeric_range_zero_centroids_and_distance_ties_keep_kernel_semantics() {
    let rows = children(vec![
        vec![1e20, 1e20],
        vec![2e20, 2e20],
        vec![0.0, -0.0],
        vec![1e-8, 0.0],
    ]);
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        let kernel = VectorKernel::new(2, metric, [11; 32]).unwrap();
        let graph = build(&kernel, &rows, params());
        let mut scratch = Workspace::new(rows.len(), params().m, MEMORY).unwrap();
        for query in [[1e20, -1e20], [1.0, 0.0], [-0.0, 0.0]] {
            let (hits, _) = graph
                .search(&query, rows.len(), 1, MEMORY, &mut scratch, &|| Ok(()))
                .unwrap();
            assert_eq!(hits, oracle(metric, &rows, &query, rows.len()));
            for hit in hits {
                assert!(hit.distance.is_finite());
                assert_eq!(
                    hit.distance.to_bits(),
                    kernel
                        .routing_distance(&query, rows[hit.node as usize].centroid())
                        .unwrap()
                        .to_bits()
                );
            }
        }
    }
}

#[test]
fn duplicate_centroids_fill_full_parent_beam_with_bounded_connected_graph() {
    let rows = children(vec![vec![0.0; 2]; 256]);
    let kernel = VectorKernel::new(2, Metric::L2, [0; 32]).unwrap();
    let parameters = Parameters {
        m: 4,
        ef_construction: 32,
        ..params()
    };
    let graph = build(&kernel, &rows, parameters);
    validated_encoding(&graph, parameters.m);
    let mut scratch = Workspace::new(rows.len(), parameters.m, MEMORY).unwrap();
    let allocation = scratch.allocated_bytes();
    let allocations = scratch.allocation_count();
    for beam in [1, 8, 64, 255, 256] {
        let (hits, _) = graph
            .search(&[0.0; 2], beam, 32, MEMORY, &mut scratch, &|| Ok(()))
            .unwrap();
        assert_eq!(hits.len(), beam);
        assert!(hits.windows(2).all(|pair| pair[0].node < pair[1].node));
        assert_eq!(scratch.allocated_bytes(), allocation);
        assert_eq!(scratch.allocation_count(), allocations);
        let again = graph
            .search(&[0.0; 2], beam, 32, MEMORY, &mut scratch, &|| Ok(()))
            .unwrap()
            .0;
        assert_eq!(hits, again);
    }
}

#[test]
fn seeded_builds_and_partition_identity_are_reproducible() {
    let rows = children(fixture(512, 4, 937));
    let kernel = VectorKernel::new(4, Metric::InnerProduct, [0; 32]).unwrap();
    let first = build(&kernel, &rows, params());
    let encoded = validated_encoding(&first, params().m);
    let fingerprint = format!(
        "local-hnsw-encoding:{:02x?}",
        Sha256::digest(&encoded).as_slice()
    );
    if std::env::var_os("KTANN_HNSW_ENCODING_CHILD").is_some() {
        println!("{fingerprint}");
        return;
    }
    for _ in 0..2 {
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "search::local_hnsw::tests::seeded_builds_and_partition_identity_are_reproducible",
                "--nocapture",
            ])
            .env("KTANN_HNSW_ENCODING_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "{}",
            String::from_utf8_lossy(&child.stderr)
        );
        assert!(
            String::from_utf8_lossy(&child.stdout)
                .lines()
                .any(|line| line == fingerprint)
        );
    }
    let second = build(&kernel, &rows, params());
    assert_eq!(encoded, validated_encoding(&second, params().m));
    let third = build(
        &kernel,
        &rows,
        Parameters {
            seed: [8; 32],
            ..params()
        },
    );
    assert_ne!(encoded, validated_encoding(&third, params().m));
    let other_tree = TreeKey::encode(&[DataType::I64], &[Value::I64(3)]).unwrap();
    assert_ne!(
        partition_seed([7; 32], &tree(), key(1)),
        partition_seed([7; 32], &other_tree, key(1))
    );
    assert_ne!(
        partition_seed([7; 32], &tree(), key(1)),
        partition_seed([7; 32], &tree(), key(2))
    );
}

#[test]
fn independent_queries_have_good_coverage_for_all_metrics_and_seeds() {
    // The independent scalar oracle can round differently. Coverage compares
    // identities; native scores below must still match the routing kernel bits.
    let nodes = |hits: &[Candidate]| hits.iter().map(|hit| hit.node).collect::<Vec<_>>();
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        let kernel = VectorKernel::new(12, metric, [3; 32]).unwrap();
        let rows = children(
            fixture(512, 12, 413)
                .into_iter()
                .map(|row| {
                    let routing = kernel.preprocess(&row).unwrap();
                    kernel.normalize_centroid(&routing).unwrap().into_vec()
                })
                .collect(),
        );
        let queries = fixture(16, 12, 887);
        for seed in [[7; 32], [19; 32]] {
            let graph = build(&kernel, &rows, Parameters { seed, ..params() });
            validated_encoding(&graph, params().m);
            let mut scratch = Workspace::new(rows.len(), params().m, MEMORY).unwrap();
            let bytes = scratch.allocated_bytes();
            let allocations = scratch.allocation_count();
            let query = kernel.preprocess(&queries[0]).unwrap();
            assert_eq!(
                nodes(
                    &graph
                        .search(&query, 7, usize::MAX, MEMORY, &mut scratch, &|| Ok(()))
                        .unwrap()
                        .0
                ),
                nodes(&oracle(metric, &rows, &query, 7)),
                "clipped full exploration must recover the exact local top-k"
            );
            let mut correct = 0;
            let mut distances = 0;
            for query in &queries {
                let query = kernel.preprocess(query).unwrap();
                let truth = oracle(metric, &rows, &query, 16);
                assert_eq!(
                    nodes(
                        &graph
                            .centroids
                            .exact(&query, 16, MEMORY, &|| Ok(()))
                            .unwrap()
                    ),
                    nodes(&truth)
                );
                let (hits, stats) = graph
                    .search(&query, 16, 96, MEMORY, &mut scratch, &|| Ok(()))
                    .unwrap();
                assert_eq!(hits.len(), 16);
                assert!(
                    hits.windows(2)
                        .all(|pair| Ranked(pair[0]) < Ranked(pair[1]))
                );
                for hit in &hits {
                    assert_eq!(
                        hit.distance.to_bits(),
                        kernel
                            .routing_distance(&query, rows[hit.node as usize].centroid())
                            .unwrap()
                            .to_bits()
                    );
                    correct += usize::from(truth.iter().any(|expected| hit.node == expected.node));
                }
                distances += stats.distances;
                assert!(stats.visits <= rows.len() * (ktann_usearch::MAX_LEVEL as usize + 1));
                assert_eq!(scratch.allocation_count(), allocations);
                assert_eq!(
                    scratch.allocated_bytes(),
                    bytes,
                    "queries must not grow native allocations"
                );
            }
            let coverage = correct as f64 / (queries.len() * 16) as f64;
            eprintln!(
                "{metric:?} seed={} coverage={coverage:.4} mean_distances={:.1} build_peak={} graph={} scratch={bytes}",
                seed[0],
                distances as f64 / queries.len() as f64,
                graph.graph.peak_bytes(),
                graph.graph.resident_bytes()
            );
            assert!(
                coverage >= 0.90,
                "fixture quality regression: {metric:?}: {coverage}"
            );
        }
    }
}

#[test]
fn cancellation_unwinds_build_and_query_and_scratch_can_be_reused() {
    let rows = children(fixture(256, 8, 781));
    let kernel = VectorKernel::new(8, Metric::L2, [0; 32]).unwrap();
    let view = Centroids::new(&kernel, &rows, 256, &|| Ok(())).unwrap();
    let calls = Cell::new(0);
    let stop = || {
        calls.set(calls.get() + 1);
        if calls.get() > 30 {
            Err(Error::new(ErrorKind::DeadlineExceeded))
        } else {
            Ok(())
        }
    };
    assert_eq!(
        TestGraph::build(view, params(), &tree(), key(1), MEMORY, &stop)
            .unwrap_err()
            .kind(),
        ErrorKind::DeadlineExceeded
    );
    let graph = build(&kernel, &rows, params());
    let mut scratch = Workspace::new(256, params().m, MEMORY).unwrap();
    let query = [0.3; 8];
    let calls = Cell::new(0);
    let stop = || {
        calls.set(calls.get() + 1);
        if calls.get() > 3 {
            Err(Error::new(ErrorKind::Cancelled))
        } else {
            Ok(())
        }
    };
    for (beam, ef) in [(16, 256), (256, 1)] {
        calls.set(0);
        assert_eq!(
            graph
                .search(&query, beam, ef, MEMORY, &mut scratch, &stop)
                .unwrap_err()
                .kind(),
            ErrorKind::Cancelled
        );
        assert_eq!(
            graph
                .search(&query, beam, ef, MEMORY, &mut scratch, &|| Ok(()))
                .unwrap()
                .0,
            oracle(Metric::L2, &rows, &query, beam)
        );
    }
    // Scoring 256 inputs needs five checks; the eighth interrupts output extraction.
    calls.set(0);
    let stop_output = || {
        calls.set(calls.get() + 1);
        if calls.get() > 7 {
            Err(Error::new(ErrorKind::Cancelled))
        } else {
            Ok(())
        }
    };
    assert_eq!(
        view.exact(&query, 256, MEMORY, &stop_output)
            .unwrap_err()
            .kind(),
        ErrorKind::Cancelled
    );
}

#[test]
fn immutable_graph_supports_parallel_queries_with_separate_scratch() {
    let rows = children(fixture(128, 4, 947));
    let kernel = VectorKernel::new(4, Metric::L2, [0; 32]).unwrap();
    let graph = build(&kernel, &rows, params());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let mut scratch = Workspace::new(128, params().m, MEMORY).unwrap();
                    graph
                        .search(&[0.1; 4], 32, 64, MEMORY, &mut scratch, &|| Ok(()))
                        .unwrap()
                        .0
                })
            })
            .collect();
        let outputs: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]));
    });
}

#[path = "persistence_tests.rs"]
mod persistence_tests;
