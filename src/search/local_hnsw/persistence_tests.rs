use super::*;
use crate::api::LogicalIndexId;
use crate::search::local_hnsw::persistence::{GraphRecords, GraphSource};
use crate::storage::keys::LogicalKey;
use crate::storage::values::{
    GRAPH_CHUNK_BYTES, GraphChunk, MAX_GRAPH_VALUE_BYTES, PartitionHeader, PartitionState,
    PersistentValue, ValueCodec,
};
use bytes::Bytes;

impl<'a> TestGraph<'a> {
    fn persist(
        &self,
        index: LogicalIndexId,
        header: PartitionHeader,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<GraphRecords> {
        super::super::persistence::persist_graph(
            self.centroids,
            &self.graph,
            self.identity,
            index,
            header,
            working_bytes,
            check,
        )
    }

    fn restore(
        centroids: Centroids<'a>,
        source: GraphSource<'_>,
        descriptor: &crate::storage::values::GraphDescriptor,
        chunks: impl IntoIterator<Item = (u32, GraphChunk)>,
        working_bytes: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let (graph, identity) = super::super::persistence::restore_graph(
            centroids,
            source,
            descriptor,
            chunks,
            working_bytes,
            check,
        )?;
        Ok(Self {
            centroids,
            graph,
            identity,
        })
    }
}

fn index() -> LogicalIndexId {
    LogicalIndexId::new(7).unwrap()
}

fn header(n: usize) -> PartitionHeader {
    PartitionHeader::new(2, n as u32, 42, PartitionState::Ready).unwrap()
}

fn parameters() -> Parameters {
    Parameters {
        m: 8,
        ef_construction: 40,
        seed: [7; 32],
    }
}

fn source(tree: &TreeKey, n: usize) -> GraphSource<'_> {
    GraphSource {
        index: index(),
        tree,
        parent: key(1),
        header: header(n),
        parameters: parameters(),
    }
}

fn records(graph: &TestGraph<'_>) -> GraphRecords {
    graph
        .persist(
            index(),
            header(graph.centroids.children.len()),
            MEMORY,
            &|| Ok(()),
        )
        .unwrap()
}

#[test]
fn topology_states_preserve_graph_binding_and_batched_body_epochs() {
    let kernel = VectorKernel::new(3, Metric::L2, [0; 32]).unwrap();
    let rows = children(fixture(17, 3, 33));
    let graph = build(&kernel, &rows, parameters());
    let encoded = records(&graph);
    let tree = tree();
    for state in [
        PartitionState::Ready,
        PartitionState::Splitting,
        PartitionState::ReceivingSplit,
        PartitionState::DrainingSplit,
        PartitionState::Merging,
    ] {
        let source = GraphSource {
            header: PartitionHeader::new(2, rows.len() as u32, 42, state).unwrap(),
            ..source(&tree, rows.len())
        };
        // State changes neither replace the graph nor change its body identity.
        TestGraph::restore(
            graph.centroids,
            source,
            &encoded.descriptor,
            encoded.chunks(),
            MEMORY,
            &|| Ok(()),
        )
        .unwrap();
        let next = &rows[2..];
        let centroids = Centroids::new(&kernel, next, 4096, &|| Ok(())).unwrap();
        for epoch in [41, 42, 44] {
            let header = PartitionHeader::new(2, next.len() as u32, epoch, state).unwrap();
            let updated = super::super::persistence::update_records(
                graph.centroids,
                centroids,
                source,
                &encoded.descriptor,
                encoded.chunks(),
                header,
                MEMORY,
                &|| Ok(()),
            );
            if epoch <= 42 {
                assert_eq!(updated.unwrap_err().kind(), ErrorKind::InvalidArgument);
            } else {
                let updated = updated.unwrap();
                TestGraph::restore(
                    centroids,
                    GraphSource { header, ..source },
                    &updated.descriptor,
                    updated.chunks(),
                    MEMORY,
                    &|| Ok(()),
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn ordinary_updates_preserve_metric_scores_canonical_children_and_source_integrity() {
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        let kernel = VectorKernel::new(8, metric, [0; 32]).unwrap();
        for duplicate in [false, true] {
            let rows = children(if duplicate {
                vec![vec![0.0; 8]; 65]
            } else {
                fixture(65, 8, 91)
            });
            let graph = build(&kernel, &rows, parameters());
            let encoded = records(&graph);
            for operation in 0..3 {
                let mut next = rows.clone();
                match operation {
                    0 => next.insert(1, ChildEntry::new(key(11), vec![0.0; 8])),
                    1 => next[16] = ChildEntry::new(next[16].child(), vec![0.5; 8]),
                    _ => {
                        next.remove(graph.graph.entry().unwrap() as usize);
                    }
                }
                let centroids = Centroids::new(&kernel, &next, 4096, &|| Ok(())).unwrap();
                let next_header =
                    PartitionHeader::new(2, next.len() as u32, 43, PartitionState::Ready).unwrap();
                let updated = super::super::persistence::update_records(
                    graph.centroids,
                    centroids,
                    source(&tree(), rows.len()),
                    &encoded.descriptor,
                    encoded.chunks(),
                    next_header,
                    MEMORY,
                    &|| Ok(()),
                )
                .unwrap();
                let restored = TestGraph::restore(
                    centroids,
                    GraphSource {
                        header: next_header,
                        ..source(&tree(), next.len())
                    },
                    &updated.descriptor,
                    updated.chunks(),
                    MEMORY,
                    &|| Ok(()),
                )
                .unwrap();
                let mut scratch = Workspace::new(next.len(), parameters().m, MEMORY).unwrap();
                for query in fixture(3, 8, 301) {
                    let expected = centroids.exact(&query, 8, MEMORY, &|| Ok(())).unwrap();
                    let (actual, _) = restored
                        .search(&query, 8, next.len(), MEMORY, &mut scratch, &|| Ok(()))
                        .unwrap();
                    assert_eq!(
                        actual, expected,
                        "{metric:?} operation {operation}, duplicate={duplicate}"
                    );
                }
                assert_eq!(
                    super::super::persistence::update_records(
                        graph.centroids,
                        centroids,
                        source(&tree(), rows.len()),
                        &encoded.descriptor,
                        encoded.chunks(),
                        next_header,
                        0,
                        &|| Ok(())
                    )
                    .unwrap_err()
                    .kind(),
                    ErrorKind::LimitExceeded
                );
                let mut corrupted = encoded.descriptor.clone();
                corrupted.sha256[0] ^= 1;
                assert_eq!(
                    super::super::persistence::update_records(
                        graph.centroids,
                        centroids,
                        source(&tree(), rows.len()),
                        &corrupted,
                        encoded.chunks(),
                        next_header,
                        MEMORY,
                        &|| Ok(())
                    )
                    .unwrap_err()
                    .kind(),
                    ErrorKind::Corruption
                );
            }
        }
    }
}

#[test]
fn snapshot_roundtrip_preserves_topology_scores_and_queries() {
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        for n in [1, 2, 257] {
            let kernel = VectorKernel::new(4, metric, [0; 32]).unwrap();
            let rows = children(fixture(n, 4, 33));
            let graph = build(&kernel, &rows, parameters());
            let encoded = records(&graph);
            let restored = TestGraph::restore(
                graph.centroids,
                source(&tree(), n),
                &encoded.descriptor,
                encoded.chunks(),
                MEMORY,
                &|| Ok(()),
            )
            .unwrap();
            assert_eq!(
                restored.graph.encode(MEMORY, &|| true).unwrap(),
                graph.graph.encode(MEMORY, &|| true).unwrap()
            );
            let rebuilt = build(&kernel, &rows, parameters());
            assert_eq!(
                records(&rebuilt).chunks().collect::<Vec<_>>(),
                encoded.chunks().collect::<Vec<_>>()
            );
            let mut scratch = Workspace::new(n, 8, MEMORY).unwrap();
            for query in fixture(5, 4, 17) {
                for beam in [1, 7, n] {
                    let before = graph
                        .search(&query, beam, 40, MEMORY, &mut scratch, &|| Ok(()))
                        .unwrap()
                        .0;
                    let after = restored
                        .search(&query, beam, 40, MEMORY, &mut scratch, &|| Ok(()))
                        .unwrap()
                        .0;
                    assert_eq!(before, after);
                }
            }
        }
    }
}

#[test]
fn source_binding_covers_identity_header_configuration_and_centroids() {
    let kernel = VectorKernel::new(3, Metric::L2, [0; 32]).unwrap();
    let rows = children(fixture(17, 3, 33));
    let graph = build(&kernel, &rows, parameters());
    let encoded = records(&graph);
    let tree = tree();
    let expected = source(&tree, rows.len());
    let other_tree = TreeKey::encode(&[DataType::I64], &[Value::I64(1)]).unwrap();
    let mut cases = vec![
        GraphSource {
            tree: &other_tree,
            ..expected
        },
        GraphSource {
            index: LogicalIndexId::new(8).unwrap(),
            ..expected
        },
        GraphSource {
            parent: key(2),
            ..expected
        },
    ];
    for h in [
        PartitionHeader::new(3, 17, 42, PartitionState::Ready).unwrap(),
        PartitionHeader::new(2, 17, 43, PartitionState::Ready).unwrap(),
        header(16),
        PartitionHeader::new(1, 17, 42, PartitionState::Ready).unwrap(),
    ] {
        cases.push(GraphSource {
            header: h,
            ..expected
        });
    }
    for p in [
        Parameters {
            m: 9,
            ..parameters()
        },
        Parameters {
            ef_construction: 41,
            ..parameters()
        },
        Parameters {
            seed: [8; 32],
            ..parameters()
        },
    ] {
        cases.push(GraphSource {
            parameters: p,
            ..expected
        });
    }
    for source in cases {
        assert_eq!(
            TestGraph::restore(
                graph.centroids,
                source,
                &encoded.descriptor,
                encoded.chunks(),
                MEMORY,
                &|| Ok(())
            )
            .unwrap_err()
            .kind(),
            ErrorKind::Corruption
        );
    }
    for replacement in [
        ChildEntry::new(key(9), rows[0].centroid().to_vec()),
        ChildEntry::new(rows[0].child(), vec![2.0; 3]),
    ] {
        let mut changed = rows.clone();
        changed[0] = replacement;
        let view = Centroids::new(&kernel, &changed, 32, &|| Ok(())).unwrap();
        assert_eq!(
            TestGraph::restore(
                view,
                expected,
                &encoded.descriptor,
                encoded.chunks(),
                MEMORY,
                &|| Ok(())
            )
            .unwrap_err()
            .kind(),
            ErrorKind::Corruption
        );
    }
    let other = VectorKernel::new(3, Metric::InnerProduct, [0; 32]).unwrap();
    let view = Centroids::new(&other, &rows, 32, &|| Ok(())).unwrap();
    assert_eq!(
        TestGraph::restore(
            view,
            expected,
            &encoded.descriptor,
            encoded.chunks(),
            MEMORY,
            &|| Ok(())
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Corruption
    );
}

#[test]
fn numbered_chunks_are_bounded_complete_and_integrity_checked() {
    let kernel = VectorKernel::new(4, Metric::L2, [0; 32]).unwrap();
    let rows = children(fixture(2048, 4, 33));
    let graph = build(&kernel, &rows, parameters());
    let encoded = records(&graph);
    let chunks: Vec<_> = encoded.chunks().collect();
    assert!(chunks.len() > 1);
    // Value codecs and logical-key identity validation are on the real storage path.
    let codec = ValueCodec::bootstrap();
    for (ordinal, chunk) in &chunks {
        let value = PersistentValue::GraphChunk(chunk.clone());
        let bytes = codec.encode(&value).unwrap();
        assert!(bytes.len() <= MAX_GRAPH_VALUE_BYTES);
        let key = LogicalKey::GraphChunk {
            index: index(),
            tree_key: tree(),
            partition: key(1),
            chunk: *ordinal,
        };
        assert_eq!(codec.decode(&key, Bytes::from(bytes)).unwrap(), value);
    }
    assert_eq!(chunks[0].1.bytes().len(), GRAPH_CHUNK_BYTES);
    let mut bad = Vec::new();
    bad.push(chunks[..chunks.len() - 1].to_vec());
    let mut extra = chunks.clone();
    extra.push(chunks[0].clone());
    bad.push(extra);
    let mut order = chunks.clone();
    order.swap(0, 1);
    bad.push(order);
    let mut duplicate = chunks.clone();
    duplicate[1].0 = 0;
    bad.push(duplicate);
    let mut short = chunks.clone();
    short[0].1 = GraphChunk::new(short[0].1.bytes().slice(1..)).unwrap();
    bad.push(short);
    let mut corrupt = chunks.clone();
    let mut bytes = corrupt[0].1.bytes().to_vec();
    bytes[0] ^= 1;
    corrupt[0].1 = GraphChunk::new(bytes.into()).unwrap();
    bad.push(corrupt);
    for chunks in bad {
        assert_eq!(
            TestGraph::restore(
                graph.centroids,
                source(&tree(), rows.len()),
                &encoded.descriptor,
                chunks,
                MEMORY,
                &|| Ok(())
            )
            .unwrap_err()
            .kind(),
            ErrorKind::Corruption
        );
    }
    let restored = TestGraph::restore(
        graph.centroids,
        source(&tree(), rows.len()),
        &encoded.descriptor,
        chunks,
        MEMORY,
        &|| Ok(()),
    )
    .unwrap();
    assert_eq!(records(&restored).descriptor, encoded.descriptor);
    assert_eq!(
        graph
            .persist(index(), header(rows.len()), 0, &|| Ok(()))
            .unwrap_err()
            .kind(),
        ErrorKind::LimitExceeded
    );
    assert_eq!(
        TestGraph::restore(
            graph.centroids,
            source(&tree(), rows.len()),
            &encoded.descriptor,
            encoded.chunks(),
            0,
            &|| Ok(())
        )
        .unwrap_err()
        .kind(),
        ErrorKind::LimitExceeded
    );
    assert_eq!(
        TestGraph::restore(
            graph.centroids,
            source(&tree(), rows.len()),
            &encoded.descriptor,
            encoded.chunks(),
            MEMORY,
            &|| Err(Error::new(ErrorKind::Cancelled))
        )
        .unwrap_err()
        .kind(),
        ErrorKind::Cancelled
    );
}

#[tokio::test]
async fn typed_snapshot_scan_restores_a_usable_body() {
    use crate::api::IndexConfig;
    use crate::storage::backend::ScanLimits;
    use crate::storage::keys;
    use crate::storage::test_support::MockReadTxn;
    use crate::storage::values::{IndexLifecycle, IndexManifest};
    use crate::storage::{LogicalRange, ReadLogicalTxn};
    let rows = children(fixture(2048, 4, 33));
    let kernel = VectorKernel::new(4, Metric::L2, [0; 32]).unwrap();
    let graph = build(&kernel, &rows, parameters());
    let encoded = records(&graph);
    let manifest = IndexManifest::new(
        IndexLifecycle::Active,
        index(),
        IndexConfig::new(4, Metric::L2)
            .unwrap()
            .with_internal_partition_entries(128, 4096)
            .unwrap(),
        [0; 32],
        vec![],
    )
    .unwrap();
    let codec = ValueCodec::for_index(&manifest);
    let mut data = Vec::new();
    let descriptor_key = LogicalKey::GraphDescriptor {
        index: index(),
        tree_key: tree(),
        partition: key(1),
    };
    let header_key = LogicalKey::Header {
        index: index(),
        tree_key: tree(),
        partition: key(1),
    };
    let mut push = |key: LogicalKey, value: PersistentValue| {
        data.push((
            keys::encode_key(&key).unwrap(),
            codec.encode_for_key(&key, &value).unwrap(),
        ));
    };
    push(
        header_key.clone(),
        PersistentValue::PartitionHeader(header(rows.len())),
    );
    push(
        descriptor_key.clone(),
        PersistentValue::GraphDescriptor(encoded.descriptor.clone()),
    );
    for (ordinal, chunk) in encoded.chunks() {
        push(
            LogicalKey::GraphChunk {
                index: index(),
                tree_key: tree(),
                partition: key(1),
                chunk: ordinal,
            },
            PersistentValue::GraphChunk(chunk),
        );
    }
    for child in &rows {
        push(
            LogicalKey::ChildEntry {
                index: index(),
                tree_key: tree(),
                partition: key(1),
                child: child.child(),
            },
            PersistentValue::ChildEntry(child.clone()),
        );
    }
    let mut txn = ReadLogicalTxn::for_index(MockReadTxn::new(data), &manifest);
    let PersistentValue::GraphDescriptor(descriptor) =
        txn.get(descriptor_key).await.unwrap().unwrap()
    else {
        panic!()
    };
    let PersistentValue::PartitionHeader(header) = txn.get(header_key).await.unwrap().unwrap()
    else {
        panic!()
    };
    let mut children = Vec::new();
    let mut chunks = Vec::new();
    for range in [
        LogicalRange::child_entries(&manifest, &tree(), key(1)).unwrap(),
        LogicalRange::graph_chunks(&manifest, &tree(), key(1)).unwrap(),
    ] {
        let mut cursor = None;
        loop {
            let page = txn
                .scan(
                    &range,
                    cursor.as_ref(),
                    ScanLimits {
                        item_limit: 17,
                        byte_limit: 128 * 1024,
                    },
                )
                .await
                .unwrap();
            for item in page.items() {
                match (item.key(), item.value()) {
                    (_, PersistentValue::ChildEntry(child)) => children.push(child.clone()),
                    (LogicalKey::GraphChunk { chunk, .. }, PersistentValue::GraphChunk(value)) => {
                        chunks.push((*chunk, value.clone()))
                    }
                    _ => panic!("unexpected graph body family"),
                }
            }
            cursor = page.into_next_cursor();
            if cursor.is_none() {
                break;
            }
        }
    }
    let view = Centroids::new(&kernel, &children, 4096, &|| Ok(())).unwrap();
    let restored = TestGraph::restore(
        view,
        GraphSource {
            header,
            ..source(&tree(), rows.len())
        },
        &descriptor,
        chunks,
        MEMORY,
        &|| Ok(()),
    )
    .unwrap();
    let mut scratch = Workspace::new(rows.len(), 8, MEMORY).unwrap();
    assert_eq!(
        graph
            .search(&[0.; 4], 19, 40, MEMORY, &mut scratch, &|| Ok(()))
            .unwrap()
            .0,
        restored
            .search(&[0.; 4], 19, 40, MEMORY, &mut scratch, &|| Ok(()))
            .unwrap()
            .0
    );
}

#[test]
fn snapshot_binding_has_golden_bytes_and_canonical_zero() {
    let kernel = VectorKernel::new(3, Metric::L2, [0; 32]).unwrap();
    // Golden digest was independently calculated from the documented field order.
    let expected = [
        242, 140, 220, 18, 212, 63, 167, 14, 154, 97, 249, 42, 213, 18, 143, 222, 50, 143, 213, 19,
        88, 141, 95, 9, 98, 203, 188, 147, 118, 113, 24, 55,
    ];
    for zero in [0.0, -0.0] {
        let rows = children(vec![vec![1.0, zero, -2.0]]);
        assert_eq!(
            records(&build(&kernel, &rows, parameters()))
                .descriptor
                .binding,
            expected
        );
    }
}

#[test]
fn source_identity_has_golden_bytes_across_coordinate_blocks() {
    // Independently encoded SHA-256 digests cover a partial and multiple full blocks.
    let cases = [
        (
            257,
            [
                224, 4, 218, 42, 13, 226, 225, 94, 16, 65, 28, 45, 236, 192, 122, 146, 45, 53, 53,
                192, 253, 49, 7, 180, 112, 99, 155, 51, 251, 241, 27, 33,
            ],
        ),
        (
            768,
            [
                17, 2, 3, 97, 185, 132, 136, 23, 240, 165, 134, 47, 16, 109, 27, 200, 131, 168, 6,
                51, 160, 221, 220, 97, 6, 231, 60, 5, 100, 172, 209, 16,
            ],
        ),
    ];
    for (dimension, expected) in cases {
        let kernel = VectorKernel::new(dimension, Metric::L2, [0; 32]).unwrap();
        for zero in [0.0, -0.0] {
            let rows = children(
                (0..2)
                    .map(|row| {
                        (0..dimension)
                            .map(|i| {
                                let value = ((row * dimension + i) % 17) as f32 - 8.0;
                                if value == 0.0 { zero } else { value }
                            })
                            .collect()
                    })
                    .collect(),
            );
            let view = Centroids::new(&kernel, &rows, rows.len(), &|| Ok(())).unwrap();
            assert_eq!(
                super::super::persistence::identity(
                    view,
                    parameters(),
                    &tree(),
                    key(1),
                    &|| Ok(())
                )
                .unwrap(),
                expected
            );
        }
    }
}
