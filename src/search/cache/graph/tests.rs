use std::collections::BTreeMap;

use super::*;
use crate::api::{FixedGraphConfig, IndexConfig, LogicalIndexId};
use crate::search::local_hnsw::{Centroids, LocalHnsw};
use crate::storage::keys;
use crate::storage::test_support::{MockReadTxn, pk};
use crate::storage::values::expect_header;
use crate::storage::values::{IndexLifecycle, ValueCodec};

const W: usize = 16 * 1024 * 1024;

impl GraphRead {
    /// Unit tests execute pure restoration inline; the runtime test covers CPU dispatch.
    fn restore(
        self,
        cache: &PartitionCache,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<RoutingGraph> {
        check()?;
        match self {
            Self::Cached(graph) => Ok(graph),
            Self::Load(load) => load.restore(cache, check),
        }
    }
}

/// Exercises the real runtime CPU boundary with typed snapshot reads.
pub(crate) async fn exercise_runtime<B: crate::storage::backend::Backend>(
    context: &crate::runtime::OperationContext<B>,
) -> Result<()> {
    let fixture = Fixture::new(Metric::L2, 1, 1);
    let mut expected = None;
    for _ in 0..2 {
        let graph = context
            .read(
                &mut fixture.txn(),
                &fixture.manifest,
                BodyRequest {
                    tree_key: &fixture.tree,
                    partition: pk(1),
                    header: &fixture.header,
                },
                1024,
            )
            .await?;
        let hits = context
            .candidates(
                graph,
                vec![0.3, 0.1, 0.2],
                7,
                crate::api::RoutingMode::PartitionLocalHnsw { ef_search: 12 },
            )
            .await?;
        if let Some(expected) = &expected {
            assert_eq!(&hits, expected);
        }
        expected = Some(hits);
    }
    Ok(())
}

/// The same fixture supports changed epochs/projections and independent index identities.
struct Fixture {
    manifest: IndexManifest,
    tree: TreeKey,
    header: PartitionHeader,
    children: Vec<ChildEntry>,
    data: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl Fixture {
    fn new(metric: Metric, epoch: u64, index: u64) -> Self {
        Self::with_size(metric, epoch, index, 32, 2)
    }

    fn with_size(metric: Metric, epoch: u64, index: u64, count: u32, m: u32) -> Self {
        let config = IndexConfig::new(3, metric)
            .unwrap()
            .with_internal_partition_entries(1, count.max(2))
            .unwrap()
            .with_fixed_graph(FixedGraphConfig::new(m, 4 * m, [9; 32]).unwrap());
        let manifest = IndexManifest::new(
            IndexLifecycle::Active,
            LogicalIndexId::new(index).unwrap(),
            config,
            [7; 32],
            vec![],
        )
        .unwrap();
        let tree = TreeKey::encode(&[], &[]).unwrap();
        let header = PartitionHeader::new(2, count, epoch, PartitionState::Ready).unwrap();
        let children: Vec<_> = (0..count)
            .map(|i| {
                ChildEntry::new(
                    pk(u64::from(i) + 2),
                    vec![
                        i as f32 / count as f32,
                        epoch as f32 * 0.1,
                        (i % 5) as f32 / 5.0,
                    ],
                )
            })
            .collect();
        let kernel = VectorKernel::new(3, metric, [7; 32]).unwrap();
        let core = LocalHnsw::build(
            Centroids::new(&kernel, &children, count as usize, &|| Ok(())).unwrap(),
            manifest.config().fixed_graph().unwrap().parameters(),
            &tree,
            pk(1),
            W,
            &|| Ok(()),
        )
        .unwrap();
        let records = core
            .persist(manifest.logical_index_id(), header, W, &|| Ok(()))
            .unwrap();
        let mut fixture = Self {
            manifest,
            tree,
            header,
            children,
            data: BTreeMap::new(),
        };
        fixture.put(
            keys::header_key(fixture.index(), &fixture.tree, pk(1)),
            PersistentValue::PartitionHeader(header),
        );
        for entry in fixture.children.clone() {
            fixture.put(
                keys::child_entry_key(fixture.index(), &fixture.tree, pk(1), entry.child()),
                PersistentValue::ChildEntry(entry),
            );
        }
        fixture.put(
            keys::graph_descriptor_key(fixture.index(), &fixture.tree, pk(1)),
            PersistentValue::GraphDescriptor(records.descriptor.clone()),
        );
        for (ordinal, chunk) in records.chunks() {
            fixture.put(
                keys::graph_chunk_key(fixture.index(), &fixture.tree, pk(1), ordinal),
                PersistentValue::GraphChunk(chunk),
            );
        }
        fixture
    }

    fn index(&self) -> LogicalIndexId {
        self.manifest.logical_index_id()
    }

    fn put(&mut self, key: Vec<u8>, value: PersistentValue) {
        self.data.insert(
            key,
            ValueCodec::for_index(&self.manifest)
                .encode(&value)
                .unwrap(),
        );
    }

    fn txn(&self) -> ReadLogicalTxn<'_, MockReadTxn> {
        ReadLogicalTxn::for_index(MockReadTxn::new(self.data.clone()), &self.manifest)
    }

    async fn read(
        &self,
        cache: &PartitionCache,
        limit: usize,
        retained: usize,
    ) -> Result<GraphRead> {
        let mut txn = self.txn();
        let header = expect_header(
            txn.get(LogicalKey::Header {
                index: self.index(),
                tree_key: self.tree.clone(),
                partition: pk(1),
            })
            .await?,
        )?
        .ok_or_else(corrupt)?;
        read_graph(
            &mut txn,
            cache,
            &self.manifest,
            BodyRequest {
                tree_key: &self.tree,
                partition: pk(1),
                header: &header,
            },
            limit,
            retained,
            &|| Ok(()),
        )
        .await
    }

    async fn load(&self, cache: &PartitionCache) -> RoutingGraph {
        self.read(cache, W, 0)
            .await
            .unwrap()
            .restore(cache, &|| Ok(()))
            .unwrap()
    }
}

/// Forces continuations on both families and can expire the snapshot mid-graph.
struct PagedRead {
    raw: MockReadTxn,
    fail_at_scan: Option<usize>,
}

impl ReadOps for PagedRead {
    async fn get(&mut self, key: bytes::Bytes) -> Result<Option<bytes::Bytes>> {
        self.raw.get(key).await
    }

    async fn batch_get(&mut self, keys: Vec<bytes::Bytes>) -> Result<Vec<Option<bytes::Bytes>>> {
        self.raw.batch_get(keys).await
    }

    async fn scan(
        &mut self,
        range: &keys::KeyRange,
        mut limits: ScanLimits,
    ) -> Result<crate::storage::backend::ScanPage> {
        if self.fail_at_scan == Some(self.raw.scans) {
            return Err(Error::new(ErrorKind::RetryableAbort));
        }
        limits.item_limit = 1;
        self.raw.scan(range, limits).await
    }

    async fn batch_scan(
        &mut self,
        ranges: &[keys::KeyRange],
        limits: ScanLimits,
    ) -> Result<Vec<crate::storage::backend::ScanPage>> {
        self.raw.batch_scan(ranges, limits).await
    }
}

#[tokio::test]
async fn pagination_and_snapshot_expiry_never_publish_partial_graphs() {
    let fixture = Fixture::with_size(Metric::L2, 1, 1, 1024, 8);
    for fail_at_scan in [Some(1025), None] {
        let cache = PartitionCache::new(W as u64);
        let raw = PagedRead {
            raw: MockReadTxn::new(fixture.data.clone()),
            fail_at_scan,
        };
        let mut txn = ReadLogicalTxn::for_index(raw, &fixture.manifest);
        let result = read_graph(
            &mut txn,
            &cache,
            &fixture.manifest,
            BodyRequest {
                tree_key: &fixture.tree,
                partition: pk(1),
                header: &fixture.header,
            },
            W,
            0,
            &|| Ok(()),
        )
        .await;
        if fail_at_scan.is_some() {
            assert_eq!(result.err().unwrap().kind(), ErrorKind::RetryableAbort);
            assert_eq!(cache.len(), 0);
        } else {
            let graph = result.unwrap().restore(&cache, &|| Ok(())).unwrap();
            assert!(
                txn.into_raw().raw.scans > 1025,
                "multiple graph pages were read"
            );
            assert_eq!(
                graph
                    .search(&[0.3, 0.1, 0.2], 7, 12, &|| Ok(()))
                    .unwrap()
                    .0
                    .len(),
                7
            );
        }
    }
}

#[tokio::test]
async fn concurrent_cold_fills_and_shared_warm_queries_survive_eviction() {
    let fixture = Fixture::new(Metric::L2, 1, 1);
    let measure = PartitionCache::new(W as u64);
    let bytes = fixture.load(&measure).await.body.bytes;
    let cache = Arc::new(PartitionCache::new(bytes));
    let mut cold = Vec::new();
    for _ in 0..4 {
        cold.push(fixture.read(&cache, W, 0).await.unwrap());
    }
    let ready = Arc::new(std::sync::Barrier::new(4));
    let hits = std::thread::scope(|scope| {
        let tasks: Vec<_> = cold
            .into_iter()
            .map(|input| {
                let cache = Arc::clone(&cache);
                let ready = Arc::clone(&ready);
                scope.spawn(move || {
                    ready.wait();
                    input
                        .restore(&cache, &|| Ok(()))
                        .unwrap()
                        .search(&[0.3, 0.1, 0.2], 7, 12, &|| Ok(()))
                        .unwrap()
                        .0
                })
            })
            .collect();
        tasks
            .into_iter()
            .map(|task| task.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(hits.iter().all(|result| *result == hits[0]));
    assert_eq!(cache.total_bytes(), bytes);
    let mut warm = Vec::new();
    for _ in 0..4 {
        warm.push(fixture.load(&cache).await);
    }
    assert!(
        warm.iter()
            .all(|graph| Arc::ptr_eq(&graph.body, &warm[0].body))
    );
    for index in 2..=6 {
        Fixture::new(Metric::L2, 1, index).load(&cache).await;
    }
    assert!(matches!(
        fixture.read(&cache, W, 0).await.unwrap(),
        GraphRead::Load(_)
    ));
    std::thread::scope(|scope| {
        for graph in warm {
            let expected = &hits[0];
            scope.spawn(move || {
                assert_eq!(
                    &graph.search(&[0.3, 0.1, 0.2], 7, 12, &|| Ok(())).unwrap().0,
                    expected
                )
            });
        }
    });
    assert_eq!(
        fixture
            .load(&cache)
            .await
            .search(&[0.3, 0.1, 0.2], 7, 12, &|| Ok(()))
            .unwrap()
            .0,
        hits[0]
    );
}

#[tokio::test]
async fn restored_queries_match_native_search_for_all_metrics() {
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        let fixture = Fixture::new(metric, 1, 1);
        let cache = PartitionCache::new(W as u64);
        let loaded = fixture.load(&cache).await;
        let kernel = VectorKernel::new(3, metric, [7; 32]).unwrap();
        let parameters = fixture
            .manifest
            .config()
            .fixed_graph()
            .unwrap()
            .parameters();
        let core = LocalHnsw::build(
            Centroids::new(&kernel, &fixture.children, 32, &|| Ok(())).unwrap(),
            parameters,
            &fixture.tree,
            pk(1),
            W,
            &|| Ok(()),
        )
        .unwrap();
        let mut scratch = Workspace::new(32, 2, W).unwrap();
        let query = [0.3, 0.1, 0.2];
        let (expected, stats) = core
            .search(&query, 7, 12, W, &mut scratch, &|| Ok(()))
            .unwrap();
        let (actual, restored_stats) = loaded.search(&query, 7, 12, &|| Ok(())).unwrap();
        assert_eq!(
            actual,
            expected
                .iter()
                .map(|hit| (fixture.children[hit.node as usize].child(), hit.distance))
                .collect::<Vec<_>>()
        );
        assert_eq!(stats.distances, restored_stats.distances);
        let mut txn = fixture.txn();
        let warm = read_graph(
            &mut txn,
            &cache,
            &fixture.manifest,
            BodyRequest {
                tree_key: &fixture.tree,
                partition: pk(1),
                header: &fixture.header,
            },
            W,
            0,
            &|| Ok(()),
        )
        .await
        .unwrap();
        assert!(matches!(warm, GraphRead::Cached(_)));
        let raw = txn.into_raw();
        assert_eq!(
            raw.gets, 1,
            "warm graph reads validate the descriptor using the caller's snapshot Header"
        );
        assert_eq!(raw.scans, 0);
        let warm = warm.restore(&cache, &|| Ok(())).unwrap();
        let bytes = warm.body.bytes;
        assert_eq!(cache.total_bytes(), bytes);
        assert_eq!(warm.search(&query, 7, 12, &|| Ok(())).unwrap().0, actual);
    }
}

#[tokio::test]
async fn admission_is_identical_on_cold_warm_disabled_and_evicted_paths() {
    let fixture = Fixture::new(Metric::L2, 1, 1);
    let cache = PartitionCache::new(W as u64);
    let mut low = 1;
    let mut high = W;
    while low < high {
        let middle = (low + high) / 2;
        match fixture.read(&cache, middle, 0).await {
            Ok(_) => high = middle,
            Err(error) => {
                assert_eq!(error.kind(), ErrorKind::LimitExceeded);
                low = middle + 1;
            }
        }
    }
    for capacity in [0, 1, W as u64] {
        let cache = PartitionCache::new(capacity);
        let held = fixture
            .read(&cache, low, 0)
            .await
            .unwrap()
            .restore(&cache, &|| Ok(()))
            .unwrap();
        assert_eq!(cache.len(), usize::from(capacity == W as u64));
        assert_eq!(
            fixture.read(&cache, low - 1, 0).await.err().unwrap().kind(),
            ErrorKind::LimitExceeded
        );
        assert_eq!(
            fixture.read(&cache, low, 1).await.err().unwrap().kind(),
            ErrorKind::LimitExceeded
        );
        fixture.read(&cache, low + 100, 100).await.unwrap();
        assert!(
            !held
                .search(&[0.3, 0.1, 0.2], 32, 32, &|| Ok(()))
                .unwrap()
                .0
                .is_empty()
        );
    }
    let held = fixture.load(&cache).await;
    let bytes = held.body.bytes;
    let exact_capacity = PartitionCache::new(bytes);
    fixture.load(&exact_capacity).await;
    assert_eq!(exact_capacity.total_bytes(), bytes);
    let too_small = PartitionCache::new(bytes - 1);
    fixture.load(&too_small).await;
    assert_eq!(too_small.len(), 0);
}

#[tokio::test]
async fn historical_reads_and_racing_fills_preserve_newer_epoch_and_held_graphs() {
    let old = Fixture::new(Metric::L2, 1, 1);
    let new = Fixture::new(Metric::L2, 2, 1);
    let cache = PartitionCache::new(W as u64);
    let old_input = old.read(&cache, W, 0).await.unwrap();
    let held_new = new.load(&cache).await;
    let held_old = old_input.restore(&cache, &|| Ok(())).unwrap();
    assert_eq!(held_old.body.epoch, 1);
    assert_eq!(held_new.body.epoch, 2);
    assert!(matches!(
        new.read(&cache, W, 0).await.unwrap(),
        GraphRead::Cached(_)
    ));
    assert!(matches!(
        old.read(&cache, W, 0).await.unwrap(),
        GraphRead::Load(_)
    ));
    let other_index = Fixture::new(Metric::L2, 1, 2);
    assert!(matches!(
        other_index.read(&cache, W, 0).await.unwrap(),
        GraphRead::Load(_)
    ));
    // Queries retain a valid native owner after the cache and its other handles die.
    drop(cache);
    std::thread::scope(|scope| {
        for held in [held_old, held_new] {
            scope.spawn(move || {
                assert_eq!(
                    held.search(&[0.3, 0.1, 0.2], 7, 12, &|| Ok(()))
                        .unwrap()
                        .0
                        .len(),
                    7
                )
            });
        }
    });
}

#[tokio::test]
async fn corrupt_or_cancelled_inputs_never_publish_partial_bodies() {
    for case in 0..7 {
        let mut fixture = Fixture::new(Metric::L2, 1, 1);
        let descriptor_key = keys::graph_descriptor_key(fixture.index(), &fixture.tree, pk(1));
        let chunk_key = keys::graph_chunk_key(fixture.index(), &fixture.tree, pk(1), 0);
        match case {
            0 => {
                fixture.data.remove(&descriptor_key);
            }
            1 => {
                fixture.data.remove(&chunk_key);
            }
            2 => {
                let bytes = fixture.data.get_mut(&chunk_key).unwrap();
                *bytes.last_mut().unwrap() ^= 1;
            }
            3 => fixture.put(
                keys::child_entry_key(fixture.index(), &fixture.tree, pk(1), pk(2)),
                PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![7.0, 0.0, 0.0])),
            ),
            4 => fixture.put(
                keys::header_key(fixture.index(), &fixture.tree, pk(1)),
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(1, 32, 1, PartitionState::Ready).unwrap(),
                ),
            ),
            5 => fixture.put(
                keys::graph_chunk_key(fixture.index(), &fixture.tree, pk(1), 1),
                PersistentValue::GraphChunk(GraphChunk(bytes::Bytes::from_static(b"extra"))),
            ),
            _ => {}
        }
        let cache = PartitionCache::new(W as u64);
        let result = match fixture.read(&cache, W, 0).await {
            Ok(input) => input
                .restore(&cache, &|| {
                    if case == 6 {
                        Err(Error::new(ErrorKind::Cancelled))
                    } else {
                        Ok(())
                    }
                })
                .map(|_| ()),
            Err(error) => Err(error),
        };
        assert_eq!(
            result.err().unwrap().kind(),
            if case == 6 {
                ErrorKind::Cancelled
            } else {
                ErrorKind::Corruption
            }
        );
        assert_eq!(cache.len(), 0, "case {case}");
    }
}

#[tokio::test]
async fn warm_validation_rejects_changed_binding() {
    let mut fixture = Fixture::new(Metric::L2, 1, 1);
    let cache = PartitionCache::new(W as u64);
    fixture.load(&cache).await;
    let descriptor_key = keys::graph_descriptor_key(fixture.index(), &fixture.tree, pk(1));
    let codec = ValueCodec::for_index(&fixture.manifest);
    let PersistentValue::GraphDescriptor(mut descriptor) = codec
        .decode(
            &LogicalKey::GraphDescriptor {
                index: fixture.index(),
                tree_key: fixture.tree.clone(),
                partition: pk(1),
            },
            bytes::Bytes::copy_from_slice(&fixture.data[&descriptor_key]),
        )
        .unwrap()
    else {
        panic!()
    };
    descriptor.binding[0] ^= 1;
    fixture.put(descriptor_key, PersistentValue::GraphDescriptor(descriptor));
    assert_eq!(
        fixture.read(&cache, W, 0).await.err().unwrap().kind(),
        ErrorKind::Corruption
    );
}
