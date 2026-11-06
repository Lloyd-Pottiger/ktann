/// Runs real graph validation and routing inline to isolate traversal scheduling.
struct GraphExecution {
    cache: PartitionCache,
    working_bytes: usize,
    modes: std::sync::Mutex<Vec<RoutingMode>>,
}

impl RoutingExecution for GraphExecution {
    fn working_byte_limit(&self) -> usize {
        self.working_bytes
    }
    fn checkpoint(&self) -> Result<()> {
        Ok(())
    }

    async fn read<T: ReadOps>(
        &self,
        txn: &mut ReadLogicalTxn<'_, T>,
        manifest: &IndexManifest,
        request: super::BodyRequest<'_>,
        retained: usize,
    ) -> Result<super::super::cache::graph::RoutingGraph> {
        use super::super::cache::graph::{GraphRead, read_graph};
        match read_graph(
            txn,
            &self.cache,
            manifest,
            request,
            self.working_bytes,
            retained,
            &|| Ok(()),
        )
        .await?
        {
            GraphRead::Cached(graph) => Ok(graph),
            GraphRead::Load(load) => load.restore(&self.cache, &|| Ok(())),
        }
    }

    async fn candidates(
        &self,
        graph: super::super::cache::graph::RoutingGraph,
        query: Vec<f32>,
        beam: usize,
        mode: RoutingMode,
        tree: u32,
        mut references: PartitionReferences,
    ) -> Result<(Vec<(PartitionKey, f64)>, PartitionReferences)> {
        graph.validate_references(tree, &mut references, &|| Ok(()))?;
        self.modes.lock().unwrap().push(mode);
        let candidates = match mode {
            RoutingMode::Exact => graph.exact(&query, beam, &|| Ok(())),
            RoutingMode::PartitionLocalHnsw { ef_search } => {
                Ok(graph.search(&query, beam, ef_search, &|| Ok(()))?.0)
            }
        }?;
        Ok((candidates, references))
    }
}

/// A shallow tree's leaf sorts before a taller tree's non-root graph body.
struct MixedDepthGraph {
    manifest: IndexManifest,
    kernel: VectorKernel,
    trees: [EnumeratedTree; 2],
    items: Vec<(Vec<u8>, Vec<u8>)>,
}

const GRAPH_ROUTING_MODES: [RoutingMode; 2] = [
    RoutingMode::Exact,
    RoutingMode::PartitionLocalHnsw { ef_search: 8 },
];

impl MixedDepthGraph {
    fn new() -> Self {
        use crate::api::LocalGraphConfig;
        use crate::search::local_hnsw::{
            Centroids,
            persistence::{GraphSource, build_records},
        };

        let base = manifest();
        let manifest = IndexManifest::new(
            IndexLifecycle::Active,
            id(1),
            base.config()
                .clone()
                .with_local_graph(LocalGraphConfig::new(2, 8, SEED).unwrap()),
            SEED,
            base.bloom_parameters().to_vec(),
        )
        .unwrap();
        let kernel = VectorKernel::new(DIMENSION, Metric::L2, SEED).unwrap();
        let mut fixture = Fixture::new(&manifest);
        let a = fixture.tree(0);
        let b = fixture.tree(1);
        // After both roots, the shallow leaf sorts ahead of B's internal body.
        for (tree, parent, level, child, distance) in
            [(&a, 1, 2, 2, 0.0), (&b, 1, 3, 2, 10.0), (&b, 2, 2, 3, 1.0)]
        {
            let header = PartitionHeader::new(level, 1, 0, PartitionState::Ready).unwrap();
            fixture.header(tree, parent, level, 1, PartitionState::Ready);
            let children = vec![ChildEntry::new(pk(child), vec![distance, 0.0])];
            fixture.child(tree, parent, child, [distance, 0.0]);
            let records = build_records(
                Centroids::new(&kernel, &children, 1, &|| Ok(())).unwrap(),
                GraphSource {
                    index: id(1),
                    tree,
                    parent: pk(parent),
                    header,
                    parameters: manifest.config().local_graph().unwrap().parameters(),
                },
                16 << 20,
                &|| Ok(()),
            )
            .unwrap()
            .records;
            fixture.push(
                keys::graph_descriptor_key(id(1), tree, pk(parent)),
                PersistentValue::GraphDescriptor(records.descriptor.clone()),
            );
            for (ordinal, chunk) in records.chunks() {
                fixture.push(
                    keys::graph_chunk_key(id(1), tree, pk(parent), ordinal),
                    PersistentValue::GraphChunk(chunk),
                );
            }
        }
        fixture.header(&a, 2, 1, 1, PartitionState::Ready);
        fixture.entry(&a, 0, 2, "a", 0, [0.0, 0.0]);
        fixture.header(&b, 3, 1, 1, PartitionState::Ready);
        fixture.entry(&b, 1, 3, "b", 0, [1.0, 0.0]);
        let items = fixture.items;
        Self {
            manifest,
            kernel,
            trees: [tree_ref(&a), tree_ref(&b)],
            items,
        }
    }

    /// Gives each case its own cache, optionally primed by searching only the taller tree.
    async fn execution(&self, mode: RoutingMode, warm: bool) -> GraphExecution {
        let execution = GraphExecution {
            cache: PartitionCache::new(1 << 20),
            working_bytes: 16 << 20,
            modes: std::sync::Mutex::new(Vec::new()),
        };
        if warm {
            self.search(&execution, &self.trees[1..], mode, 10)
                .await
                .unwrap();
            execution.modes.lock().unwrap().clear();
        }
        execution
    }

    async fn search(
        &self,
        execution: &GraphExecution,
        trees: &[EnumeratedTree],
        mode: RoutingMode,
        partition_budget: u32,
    ) -> Result<TraversalOutcome> {
        let mut txn =
            ReadLogicalTxn::for_index(MockReadTxn::new(self.items.clone()), &self.manifest);
        super::traverse(
            &mut txn,
            &execution.cache,
            &self.kernel,
            TraversalRequest::new(&QUERY, trees, None, budgets(partition_budget, 10), 128)
                .unwrap()
                .with_routing_mode(mode),
            execution,
        )
        .await
    }
}

#[tokio::test]
async fn mixed_depth_graph_prefetch_preserves_routing_and_budgets() {
    let fixture = MixedDepthGraph::new();
    for mode in GRAPH_ROUTING_MODES {
        for warm in [false, true] {
            for (budget, expected_ids, graph_visits) in [
                (3, vec![b"a".as_slice()], 2),
                (4, vec![b"a".as_slice()], 3),
                (5, vec![b"a".as_slice(), b"b".as_slice()], 3),
            ] {
                let execution = fixture.execution(mode, warm).await;
                let outcome = fixture
                    .search(&execution, &fixture.trees, mode, budget)
                    .await
                    .unwrap();
                assert_eq!(outcome.visited_partitions(), budget);
                assert_eq!(outcome.partition_budget_exhausted, budget < 5);
                assert_eq!(outcome.visited_leaf_entries(), expected_ids.len() as u64);
                assert_eq!(candidate_ids(&outcome), expected_ids);
                assert_eq!(*execution.modes.lock().unwrap(), vec![mode; graph_visits]);
            }
        }
    }
}

#[tokio::test]
async fn mixed_depth_graph_prefetch_rejects_missing_descriptor() {
    for mode in GRAPH_ROUTING_MODES {
        for warm in [false, true] {
            let mut fixture = MixedDepthGraph::new();
            let execution = fixture.execution(mode, warm).await;
            // Remove authority after warming: cache hits must still read the descriptor.
            let descriptor = keys::graph_descriptor_key(id(1), fixture.trees[1].tree_key(), pk(2));
            fixture.items.retain(|(key, _)| *key != descriptor);
            let error = fixture
                .search(&execution, &fixture.trees, mode, 10)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Corruption);
        }
    }
}

#[tokio::test]
async fn mixed_depth_graph_prefetch_enforces_working_resource_admission() {
    let fixture = MixedDepthGraph::new();
    for mode in GRAPH_ROUTING_MODES {
        for warm in [false, true] {
            let mut execution = fixture.execution(mode, warm).await;
            execution.working_bytes = 1;
            // Start after the roots so their admission cannot mask a bypass at the mixed frontier.
            let mut state = super::Traversal::seed(&[], 10, true).unwrap();
            // Equal distances still order the leaf before the internal partition by tree.
            for (tree, level) in [(0, 1), (1, 2)] {
                state.frontier.inject(tree, pk(2), level).unwrap();
            }
            let request = TraversalRequest::new(&QUERY, &fixture.trees, None, budgets(10, 10), 128)
                .unwrap()
                .with_routing_mode(mode);
            let query = super::RaBitQQuery::new(&QUERY, Metric::L2).unwrap();
            let context = super::VisitContext {
                input_bytes: 0,
                execution: &execution,
                manifest: &fixture.manifest,
                cache: &execution.cache,
                kernel: &fixture.kernel,
                query: &query,
                request,
            };
            let mut txn = ReadLogicalTxn::for_index(
                MockReadTxn::new(fixture.items.clone()),
                &fixture.manifest,
            );
            assert_eq!(
                state.run(&mut txn, &context).await.unwrap_err().kind(),
                ErrorKind::LimitExceeded
            );
            assert_eq!(state.visited_leaf_entries, 1);
        }
    }
}

struct ExactOnly;

impl RoutingExecution for ExactOnly {
    fn working_byte_limit(&self) -> usize {
        256 * 1024 * 1024
    }
    fn checkpoint(&self) -> Result<()> {
        Ok(())
    }
    async fn read<T: ReadOps>(
        &self,
        _: &mut ReadLogicalTxn<'_, T>,
        _: &IndexManifest,
        _: super::BodyRequest<'_>,
        _: usize,
    ) -> Result<super::super::cache::graph::RoutingGraph> {
        unreachable!("mutable fixture")
    }
    async fn candidates(
        &self,
        _: super::super::cache::graph::RoutingGraph,
        _: Vec<f32>,
        _: usize,
        _: RoutingMode,
        _: u32,
        _: PartitionReferences,
    ) -> Result<(Vec<(PartitionKey, f64)>, PartitionReferences)> {
        unreachable!("mutable fixture")
    }
}

async fn traverse<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    cache: &PartitionCache,
    kernel: &VectorKernel,
    request: TraversalRequest<'_>,
) -> Result<TraversalOutcome> {
    super::traverse(txn, cache, kernel, request, &ExactOnly).await
}

use bytes::Bytes;

use crate::api::{
    CompareOp, DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, Metric, Predicate, Result,
    SearchBudgets, Value,
};
use crate::storage::ReadLogicalTxn;
use crate::storage::keys::{self, TreeKey};
use crate::storage::test_support::{MockReadTxn, id, pk};
use crate::storage::values::{
    BloomParameters, ChildEntry, IndexLifecycle, IndexManifest, LeafEntry, PartitionHeader,
    PartitionState, PartitionSynopsis, PartitionTransition, PersistentValue, TreeManifest,
    ValueCodec,
};

use super::super::cache::PartitionCache;
use super::super::cache::graph::PartitionReferences;
use super::super::numeric::VectorKernel;
use super::super::plan::EnumeratedTree;
use super::super::predicate::CompiledPredicate;
use super::super::rabitq::RaBitQ7;
use super::{
    DEFAULT_LEAF_BEAM, PartitionKey, ReadOps, RoutingExecution, RoutingMode, TraversalOutcome,
    TraversalRequest,
};

const DIMENSION: usize = 2;
const SEED: [u8; 32] = [9; 32];
/// The test query: at the L2 origin a stored vector's rough distance is
/// approximately its squared norm, so well-separated norms give a known
/// deterministic candidate order.
const QUERY: [f32; DIMENSION] = [0.0, 0.0];

fn manifest() -> IndexManifest {
    let fields = vec![
        FieldSchema::new("k", DataType::I64).expect("valid field"),
        FieldSchema::new("x", DataType::I64).expect("valid field"),
    ];
    let bloom = fields
        .iter()
        .map(|field| BloomParameters::derive(field.synopsis()).expect("valid synopsis"))
        .collect();
    let config = IndexConfig::new(DIMENSION, Metric::L2)
        .expect("valid config")
        .with_fields(fields)
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields");
    IndexManifest::new(IndexLifecycle::Active, id(1), config, SEED, bloom).expect("valid manifest")
}

fn tree_key(value: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(value)]).expect("canonical tree key")
}

/// The Tree Manifest every fixture tree uses: stable root pk1 with ample
/// reserved Partition Keys.
fn test_tree_manifest() -> TreeManifest {
    TreeManifest::new(pk(1), pk(1_000)).expect("valid tree manifest")
}

fn tree_ref(key: &TreeKey) -> EnumeratedTree {
    EnumeratedTree::new(key.clone(), test_tree_manifest())
}

fn budgets(partitions: u32, rerank: u32) -> SearchBudgets {
    SearchBudgets::new(1_024, partitions, rerank).expect("valid budgets")
}

fn encode(manifest: &IndexManifest, value: &PersistentValue) -> Vec<u8> {
    ValueCodec::for_index(manifest)
        .encode(value)
        .expect("encode value")
}

/// Accumulates encoded fixture state for one Logical Index.
struct Fixture<'a> {
    manifest: &'a IndexManifest,
    items: Vec<(Vec<u8>, Vec<u8>)>,
}

impl<'a> Fixture<'a> {
    fn new(manifest: &'a IndexManifest) -> Self {
        Self {
            manifest,
            items: Vec::new(),
        }
    }

    fn push(&mut self, key: Vec<u8>, value: PersistentValue) {
        self.items.push((key, encode(self.manifest, &value)));
    }

    /// Installs the Tree Manifest directory entry and returns the key.
    fn tree(&mut self, value: i64) -> TreeKey {
        let key = tree_key(value);
        self.push(
            keys::tree_manifest_key(self.manifest.logical_index_id(), &key),
            PersistentValue::TreeManifest(test_tree_manifest()),
        );
        key
    }

    fn header(
        &mut self,
        tree: &TreeKey,
        partition: u64,
        level: u32,
        count: u32,
        state: PartitionState,
    ) {
        self.push(
            keys::header_key(self.manifest.logical_index_id(), tree, pk(partition)),
            PersistentValue::PartitionHeader(
                PartitionHeader::new(level, count, 0, state).expect("valid header"),
            ),
        );
    }

    fn state(&mut self, tree: &TreeKey, partition: u64, transition: PartitionTransition) {
        self.push(
            keys::state_key(self.manifest.logical_index_id(), tree, pk(partition)),
            PersistentValue::PartitionState(transition),
        );
    }

    /// Installs the exact incremental Synopsis for rows `[tree_value, x]`.
    fn synopsis(&mut self, tree: &TreeKey, tree_value: i64, partition: u64, xs: &[i64]) {
        let mut synopsis = PartitionSynopsis::empty(self.manifest);
        for &x in xs {
            synopsis
                .expand(self.manifest, &[Value::I64(tree_value), Value::I64(x)])
                .expect("expand synopsis");
        }
        self.push(
            keys::synopsis_key(self.manifest.logical_index_id(), tree, pk(partition)),
            PersistentValue::PartitionSynopsis(synopsis),
        );
    }

    fn entry(
        &mut self,
        tree: &TreeKey,
        tree_value: i64,
        partition: u64,
        record_id: &str,
        x: i64,
        vector: [f32; DIMENSION],
    ) {
        let record_id = Bytes::copy_from_slice(record_id.as_bytes());
        let code = RaBitQ7::quantize(&vector).expect("quantize");
        let entry = LeafEntry::new(
            record_id.clone(),
            vec![Value::I64(tree_value), Value::I64(x)],
            code,
        );
        self.push(
            keys::leaf_entry_key(
                self.manifest.logical_index_id(),
                tree,
                pk(partition),
                &record_id,
            )
            .expect("leaf entry key"),
            PersistentValue::LeafEntry(entry),
        );
    }

    fn child(&mut self, tree: &TreeKey, parent: u64, child: u64, centroid: [f32; DIMENSION]) {
        self.push(
            keys::child_entry_key(
                self.manifest.logical_index_id(),
                tree,
                pk(parent),
                pk(child),
            ),
            PersistentValue::ChildEntry(ChildEntry::new(pk(child), centroid)),
        );
    }

    fn raw_entry(&mut self, tree: &TreeKey, partition: u64, record_id: &str, value: Vec<u8>) {
        let record_id = Bytes::copy_from_slice(record_id.as_bytes());
        self.items.push((
            keys::leaf_entry_key(
                self.manifest.logical_index_id(),
                tree,
                pk(partition),
                &record_id,
            )
            .expect("leaf entry key"),
            value,
        ));
    }

    fn raw_child(&mut self, tree: &TreeKey, parent: u64, child: u64, value: Vec<u8>) {
        self.items.push((
            keys::child_entry_key(
                self.manifest.logical_index_id(),
                tree,
                pk(parent),
                pk(child),
            ),
            value,
        ));
    }
}

async fn run(
    items: Vec<(Vec<u8>, Vec<u8>)>,
    manifest: &IndexManifest,
    trees: &[EnumeratedTree],
    predicate: Option<Predicate>,
    budgets: SearchBudgets,
    beam: u32,
) -> Result<TraversalOutcome> {
    let mut txn = ReadLogicalTxn::for_index(MockReadTxn::new(items), manifest);
    let kernel = VectorKernel::new(DIMENSION, Metric::L2, SEED).expect("valid kernel");
    let compiled = predicate
        .map(|predicate| CompiledPredicate::compile(predicate, manifest.config().fields()))
        .transpose()
        .expect("compile predicate");
    let cache = PartitionCache::new(1 << 20);
    traverse(
        &mut txn,
        &cache,
        &kernel,
        TraversalRequest::new(&QUERY, trees, compiled.as_ref(), budgets, beam)
            .expect("valid request"),
    )
    .await
}

fn candidate_ids(outcome: &TraversalOutcome) -> Vec<&[u8]> {
    let mut candidates: Vec<_> = outcome.candidates().iter().collect();
    candidates.sort_unstable_by(|a, b| {
        a.distance()
            .rough()
            .total_cmp(&b.distance().rough())
            .then_with(|| a.record_id().cmp(b.record_id()))
    });
    candidates
        .into_iter()
        .map(|candidate| candidate.record_id().as_ref())
        .collect()
}

fn assert_corruption(result: Result<TraversalOutcome>) {
    assert_eq!(
        result.err().map(|error| error.kind()),
        Some(ErrorKind::Corruption)
    );
}

/// Runs the default corruption traversal: no predicate, generous
/// budgets, and the default beam.
async fn assert_corrupt(
    items: Vec<(Vec<u8>, Vec<u8>)>,
    manifest: &IndexManifest,
    trees: &[EnumeratedTree],
) {
    assert_corruption(
        run(
            items,
            manifest,
            trees,
            None,
            budgets(16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await,
    );
}

#[test]
fn request_rejects_zero_beam() {
    assert_eq!(DEFAULT_LEAF_BEAM, 128);
    let trees = vec![];
    let budget = budgets(8, 8);
    assert!(TraversalRequest::new(&QUERY, &trees, None, budget, 0).is_err());
}

#[tokio::test]
async fn traversal_reads_each_header_once_across_cache_modes() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
    fixture.child(&tree, 1, 2, [1.0, 0.0]);
    fixture.child(&tree, 1, 3, [2.0, 0.0]);
    for (partition, record_id, vector) in [(2, "a", [1.0, 0.0]), (3, "b", [2.0, 0.0])] {
        fixture.header(&tree, partition, 1, 1, PartitionState::Ready);
        fixture.entry(&tree, 1, partition, record_id, 1, vector);
    }
    let trees = [tree_ref(&tree)];
    let kernel = VectorKernel::new(DIMENSION, Metric::L2, SEED).expect("valid kernel");
    for capacity in [0, 1, 1 << 20] {
        let cache = PartitionCache::new(capacity);
        for pass in 0..2 {
            let mut txn =
                ReadLogicalTxn::for_index(MockReadTxn::new(fixture.items.clone()), &manifest);
            let outcome = traverse(
                &mut txn,
                &cache,
                &kernel,
                TraversalRequest::new(&QUERY, &trees, None, budgets(8, 8), DEFAULT_LEAF_BEAM)
                    .expect("valid request"),
            )
            .await
            .expect("traverse");
            let raw = txn.into_raw();
            assert_eq!(
                raw.gets + raw.batch_sizes.iter().sum::<usize>(),
                3,
                "one Header read per visited partition: capacity={capacity}, pass={pass}"
            );
            let expected_scans = if capacity > 1 && pass == 1 { 0 } else { 3 };
            assert_eq!(
                raw.scans, expected_scans,
                "body scans: capacity={capacity}, pass={pass}"
            );
            assert_eq!(
                candidate_ids(&outcome),
                vec![b"a".as_slice(), b"b".as_slice()]
            );
            assert_eq!(outcome.visited_partitions(), 3);
            assert_eq!(outcome.visited_leaf_entries(), 2);
            assert!(!outcome.partition_budget_exhausted());
        }
    }
}

#[tokio::test]
async fn metadata_batches_are_bounded_and_read_each_key_once() {
    let manifest = manifest();
    let children = super::PARTITION_METADATA_BATCH as u32 + 6;
    let predicate = CompiledPredicate::compile(
        Predicate::Compare {
            field: FieldId(1),
            op: CompareOp::Eq,
            value: Value::I64(1),
        },
        manifest.config().fields(),
    )
    .expect("predicate");
    for (level, predicate) in [(2, None), (1, None), (1, Some(&predicate))] {
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, level + 1, children, PartitionState::Ready);
        for partition in 2..u64::from(children) + 2 {
            fixture.child(&tree, 1, partition, [partition as f32, 0.0]);
            fixture.header(&tree, partition, level, 0, PartitionState::Ready);
            if level == 1 {
                fixture.synopsis(&tree, 1, partition, &[]);
            }
        }
        let trees = [tree_ref(&tree)];
        let kernel = VectorKernel::new(DIMENSION, Metric::L2, SEED).expect("valid kernel");
        for capacity in [0, 1 << 20] {
            let cache = PartitionCache::new(capacity);
            for _ in 0..2 {
                let mut txn =
                    ReadLogicalTxn::for_index(MockReadTxn::new(fixture.items.clone()), &manifest);
                let outcome = traverse(
                    &mut txn,
                    &cache,
                    &kernel,
                    TraversalRequest::new(
                        &QUERY,
                        &trees,
                        predicate,
                        budgets(children + 2, 8),
                        4096,
                    )
                    .expect("valid request"),
                )
                .await
                .expect("traverse empty beam");
                assert_eq!(outcome.visited_partitions(), children + 1);
                assert!(!outcome.partition_budget_exhausted());
                let raw = txn.into_raw();
                assert!(
                    raw.batch_scan_sizes
                        .iter()
                        .all(|&size| { size <= super::BODY_PREFETCH_BATCH })
                );
                assert_eq!(raw.gets, 1, "only the root needs an individual Header read");
                let keys_per_partition = if predicate.is_some() { 2 } else { 1 };
                assert_eq!(
                    raw.batch_sizes.iter().sum::<usize>(),
                    children as usize * keys_per_partition
                );
                assert!(
                    raw.batch_sizes
                        .iter()
                        .all(|&size| size <= super::PARTITION_METADATA_BATCH * keys_per_partition)
                );
                assert!(
                    raw.batch_sizes.len() == 2,
                    "partition metadata requests are coalesced"
                );
            }
        }
    }
}

#[tokio::test]
async fn metadata_batch_does_not_read_beyond_partition_budget() {
    let manifest = manifest();
    for level in [1, 2] {
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, level + 1, 3, PartitionState::Ready);
        for partition in 2..5 {
            fixture.child(&tree, 1, partition, [partition as f32, 0.0]);
        }
        fixture.header(&tree, 2, level, 0, PartitionState::Ready);
        fixture.header(&tree, 3, level, 0, PartitionState::Ready);
        // The last selected child is corrupt (missing Header), but it is not
        // encountered until the query can fund all four partition visits.
        for limit in 1..4 {
            let outcome = run(
                fixture.items.clone(),
                &manifest,
                &[tree_ref(&tree)],
                None,
                budgets(limit, 8),
                DEFAULT_LEAF_BEAM,
            )
            .await
            .expect("unfunded missing Header is not read");
            assert_eq!(outcome.visited_partitions(), limit);
            assert!(outcome.partition_budget_exhausted());
        }
        assert_corruption(
            run(
                fixture.items,
                &manifest,
                &[tree_ref(&tree)],
                None,
                budgets(4, 8),
                DEFAULT_LEAF_BEAM,
            )
            .await,
        );
    }
}

#[tokio::test]
async fn filtered_metadata_batch_validates_only_funded_synopses() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
    for partition in 2..4 {
        fixture.child(&tree, 1, partition, [partition as f32, 0.0]);
        fixture.header(&tree, partition, 1, 0, PartitionState::Ready);
    }
    fixture.synopsis(&tree, 1, 2, &[]);
    // The second leaf is corrupt, but must not be read until funded.
    for limit in [2, 3] {
        let result = run(
            fixture.items.clone(),
            &manifest,
            &[tree_ref(&tree)],
            Some(Predicate::Compare {
                field: FieldId(1),
                op: CompareOp::Eq,
                value: Value::I64(1),
            }),
            budgets(limit, 8),
            DEFAULT_LEAF_BEAM,
        )
        .await;
        if limit == 2 {
            let outcome = result.expect("unfunded Synopsis is not read");
            assert_eq!(outcome.visited_partitions(), 2);
            assert!(outcome.partition_budget_exhausted());
        } else {
            assert_corruption(result);
        }
    }
}

#[tokio::test]
async fn empty_tree_visits_only_the_root() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 1, 0, PartitionState::Ready);
    fixture.synopsis(&tree, 1, 1, &[]);

    let outcome = run(
        fixture.items,
        &manifest,
        &[tree_ref(&tree)],
        None,
        budgets(8, 8),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(outcome.visited_partitions(), 1);
    assert_eq!(outcome.visited_leaf_entries(), 0);
    assert!(!outcome.partition_budget_exhausted());
    assert!(outcome.into_candidates().candidates().is_empty());
}

#[tokio::test]
async fn trees_advance_fairly_under_one_shared_budget() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let first = fixture.tree(1);
    fixture.header(&first, 1, 1, 2, PartitionState::Ready);
    fixture.synopsis(&first, 1, 1, &[0, 0]);
    fixture.entry(&first, 1, 1, "a4", 0, [2.0, 0.0]);
    fixture.entry(&first, 1, 1, "a16", 0, [4.0, 0.0]);
    let second = fixture.tree(2);
    fixture.header(&second, 1, 1, 2, PartitionState::Ready);
    fixture.synopsis(&second, 2, 1, &[0, 0]);
    fixture.entry(&second, 2, 1, "b1", 0, [1.0, 0.0]);
    fixture.entry(&second, 2, 1, "b9", 0, [3.0, 0.0]);
    let trees = vec![tree_ref(&first), tree_ref(&second)];

    // Both roots are seeded up front and every leaf is visited under a
    // generous budget; global selection owns their eventual ranking.
    let outcome = run(
        fixture.items.clone(),
        &manifest,
        &trees,
        None,
        budgets(16, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    let mut ids = candidate_ids(&outcome);
    ids.sort_unstable();
    assert_eq!(ids, vec![b"a16".as_slice(), b"a4", b"b1", b"b9"]);
    assert_eq!(outcome.visited_partitions(), 2);
    assert!(!outcome.partition_budget_exhausted());

    // Roots tie at distance zero, so enumeration order decides which tree
    // advances when the Partition budget funds only one visit.
    let outcome = run(
        fixture.items,
        &manifest,
        &trees,
        None,
        budgets(1, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(
        candidate_ids(&outcome),
        vec![b"a4".as_slice(), b"a16".as_slice()]
    );
    assert_eq!(outcome.visited_partitions(), 1);
    assert!(outcome.partition_budget_exhausted());
}

#[tokio::test]
async fn internal_distance_ties_break_to_the_smaller_partition_key() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    // A depth-2 tree whose two leaf children share one centroid, so the
    // expansion tie is decided by Partition Key alone.
    fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
    fixture.header(&tree, 2, 1, 1, PartitionState::Ready);
    fixture.header(&tree, 3, 1, 1, PartitionState::Ready);
    fixture.synopsis(&tree, 1, 2, &[0]);
    fixture.synopsis(&tree, 1, 3, &[0]);
    fixture.child(&tree, 1, 2, [1.0, 0.0]);
    fixture.child(&tree, 1, 3, [1.0, 0.0]);
    fixture.entry(&tree, 1, 2, "p2", 0, [1.0, 0.0]);
    fixture.entry(&tree, 1, 3, "p3", 0, [2.0, 0.0]);
    let trees = vec![tree_ref(&tree)];

    // The exact natural partition count completes without exhaustion.
    let outcome = run(
        fixture.items.clone(),
        &manifest,
        &trees,
        None,
        budgets(3, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(
        candidate_ids(&outcome),
        vec![b"p2".as_slice(), b"p3".as_slice()]
    );
    assert_eq!(outcome.visited_partitions(), 3);
    assert!(!outcome.partition_budget_exhausted());

    // One visit fewer admits the tied children in Partition Key order.
    let outcome = run(
        fixture.items,
        &manifest,
        &trees,
        None,
        budgets(2, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(candidate_ids(&outcome), vec![b"p2".as_slice()]);
    assert_eq!(outcome.visited_partitions(), 2);
    assert!(outcome.partition_budget_exhausted());
}

#[tokio::test]
async fn empty_leaf_does_not_increase_entry_usage() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
    fixture.header(&tree, 2, 1, 0, PartitionState::Ready);
    fixture.header(&tree, 3, 1, 1, PartitionState::Ready);
    fixture.synopsis(&tree, 1, 2, &[]);
    fixture.synopsis(&tree, 1, 3, &[0]);
    fixture.child(&tree, 1, 2, [0.0, 0.0]);
    fixture.child(&tree, 1, 3, [2.0, 0.0]);
    fixture.entry(&tree, 1, 3, "x", 0, [1.0, 0.0]);

    let outcome = run(
        fixture.items,
        &manifest,
        &[tree_ref(&tree)],
        None,
        budgets(16, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(candidate_ids(&outcome), vec![b"x".as_slice()]);
    assert_eq!(outcome.visited_partitions(), 3);
    assert_eq!(outcome.visited_leaf_entries(), 1);
}

#[tokio::test]
async fn beam_prunes_leaf_visits_without_budget_exhaustion() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    // A depth-3 tree: root pk1; internals pk2, pk3; leaves pk4..pk7.
    fixture.header(&tree, 1, 3, 2, PartitionState::Ready);
    fixture.header(&tree, 2, 2, 2, PartitionState::Ready);
    fixture.header(&tree, 3, 2, 2, PartitionState::Ready);
    for leaf in [4_u64, 5, 6, 7] {
        fixture.header(&tree, leaf, 1, 1, PartitionState::Ready);
        fixture.synopsis(&tree, 1, leaf, &[0]);
    }
    fixture.child(&tree, 1, 2, [0.0, 0.0]);
    fixture.child(&tree, 1, 3, [10.0, 0.0]);
    fixture.child(&tree, 2, 4, [0.0, 0.0]);
    fixture.child(&tree, 2, 5, [1.0, 0.0]);
    fixture.child(&tree, 3, 6, [10.0, 0.0]);
    fixture.child(&tree, 3, 7, [11.0, 0.0]);
    fixture.entry(&tree, 1, 4, "e4", 0, [1.0, 0.0]);
    fixture.entry(&tree, 1, 5, "e5", 0, [2.0, 0.0]);
    fixture.entry(&tree, 1, 6, "e6", 0, [3.0, 0.0]);
    fixture.entry(&tree, 1, 7, "e7", 0, [4.0, 0.0]);
    let trees = vec![tree_ref(&tree)];

    // A wide beam admits every body.
    let outcome = run(
        fixture.items.clone(),
        &manifest,
        &trees,
        None,
        budgets(16, 64),
        4,
    )
    .await
    .expect("traverse");
    assert_eq!(
        candidate_ids(&outcome),
        vec![
            b"e4".as_slice(),
            b"e5".as_slice(),
            b"e6".as_slice(),
            b"e7".as_slice()
        ]
    );
    assert_eq!(outcome.visited_partitions(), 7);
    assert!(!outcome.partition_budget_exhausted());

    // Beam 2 admits one level-2 internal and two leaves: pk3's subtree is
    // pruned by the beam, not by any budget.
    let outcome = run(
        fixture.items.clone(),
        &manifest,
        &trees,
        None,
        budgets(16, 64),
        2,
    )
    .await
    .expect("traverse");
    assert_eq!(
        candidate_ids(&outcome),
        vec![b"e4".as_slice(), b"e5".as_slice()]
    );
    assert_eq!(outcome.visited_partitions(), 4);
    assert!(!outcome.partition_budget_exhausted());

    // Beam 1 is a greedy single path to the nearest leaf.
    let outcome = run(fixture.items, &manifest, &trees, None, budgets(16, 64), 1)
        .await
        .expect("traverse");
    assert_eq!(candidate_ids(&outcome), vec![b"e4".as_slice()]);
    assert_eq!(outcome.visited_partitions(), 3);
    assert!(!outcome.partition_budget_exhausted());
}

#[tokio::test]
async fn beam_selects_children_globally_across_same_level_parents() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 3, 2, PartitionState::Ready);
    fixture.header(&tree, 2, 2, 4, PartitionState::Ready);
    fixture.header(&tree, 3, 2, 4, PartitionState::Ready);
    fixture.child(&tree, 1, 2, [0.0, 0.0]);
    fixture.child(&tree, 1, 3, [1.0, 0.0]);

    for (leaf, distance, record_id) in [
        (4, 2.0, "a2"),
        (5, 3.0, "a3"),
        (6, 4.0, "a4"),
        (7, 5.0, "a5"),
    ] {
        fixture.header(&tree, leaf, 1, 1, PartitionState::Ready);
        fixture.child(&tree, 2, leaf, [distance, 0.0]);
        fixture.entry(&tree, 1, leaf, record_id, 0, [distance, 0.0]);
    }
    for (leaf, distance, record_id) in [
        (8, 0.1, "b01"),
        (9, 0.2, "b02"),
        (10, 6.0, "b6"),
        (11, 7.0, "b7"),
    ] {
        fixture.header(&tree, leaf, 1, 1, PartitionState::Ready);
        fixture.child(&tree, 3, leaf, [distance, 0.0]);
        fixture.entry(&tree, 1, leaf, record_id, 0, [distance, 0.0]);
    }

    let outcome = run(
        fixture.items,
        &manifest,
        &[tree_ref(&tree)],
        None,
        budgets(16, 64),
        4,
    )
    .await
    .expect("traverse");
    assert_eq!(
        candidate_ids(&outcome),
        vec![
            b"b01".as_slice(),
            b"b02".as_slice(),
            b"a2".as_slice(),
            b"a3".as_slice(),
        ]
    );
    assert_eq!(outcome.visited_partitions(), 7);
    assert!(!outcome.partition_budget_exhausted());
}

#[tokio::test]
async fn synopsis_pruning_skips_provably_nonmatching_leaves() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
    fixture.header(&tree, 2, 1, 2, PartitionState::Ready);
    fixture.header(&tree, 3, 1, 2, PartitionState::Ready);
    fixture.child(&tree, 1, 2, [0.0, 0.0]);
    fixture.child(&tree, 1, 3, [10.0, 0.0]);
    fixture.synopsis(&tree, 1, 2, &[1, 2]);
    fixture.synopsis(&tree, 1, 3, &[100, 101]);
    fixture.entry(&tree, 1, 2, "p", 1, [1.0, 0.0]);
    fixture.entry(&tree, 1, 2, "q", 2, [2.0, 0.0]);
    fixture.entry(&tree, 1, 3, "r", 100, [3.0, 0.0]);
    fixture.entry(&tree, 1, 3, "s", 101, [4.0, 0.0]);
    let trees = vec![tree_ref(&tree)];

    // pk2's synopsis proves no entry can equal 100: its entries are never
    // read nor charged. pk3 is MayMatch and filters exactly.
    let predicate = Predicate::Compare {
        field: FieldId(1),
        op: CompareOp::Eq,
        value: Value::I64(100),
    };
    let outcome = run(
        fixture.items.clone(),
        &manifest,
        &trees,
        Some(predicate),
        budgets(16, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(candidate_ids(&outcome), vec![b"r".as_slice()]);
    assert_eq!(outcome.visited_partitions(), 3);
    assert_eq!(outcome.visited_leaf_entries(), 2);
    assert!(!outcome.partition_budget_exhausted());

    // A predicate that always holds over pk2's synopsis is AllMatch:
    // evaluation is skipped but every entry is still charged and admitted.
    let always = Predicate::And(vec![
        Predicate::Compare {
            field: FieldId(1),
            op: CompareOp::GreaterOrEqual,
            value: Value::I64(1),
        },
        Predicate::Compare {
            field: FieldId(1),
            op: CompareOp::LessOrEqual,
            value: Value::I64(2),
        },
    ]);
    let outcome = run(
        fixture.items,
        &manifest,
        &trees,
        Some(always),
        budgets(16, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(
        candidate_ids(&outcome),
        vec![b"p".as_slice(), b"q".as_slice()]
    );
    assert_eq!(outcome.visited_leaf_entries(), 2);
}

#[tokio::test]
async fn candidate_cap_preserves_record_id_ties_and_reports_truncation() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 1, 3, PartitionState::Ready);
    fixture.synopsis(&tree, 1, 1, &[0, 0, 0]);
    // The candidate pool owns ranking; this checks traversal supplies the codes,
    // Record IDs and cap, then forwards the selected pool and truncation signal.
    fixture.entry(&tree, 1, 1, "b", 0, [1.0, 0.0]);
    fixture.entry(&tree, 1, 1, "a", 0, [1.0, 0.0]);
    fixture.entry(&tree, 1, 1, "c", 0, [3.0, 0.0]);

    // A single-candidate global rerank budget reports budget exhaustion,
    // without reporting an interval-overlap cutoff.
    let outcome = run(
        fixture.items,
        &manifest,
        &[tree_ref(&tree)],
        None,
        budgets(16, 1),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(candidate_ids(&outcome), vec![b"a".as_slice()]);
    assert_eq!(outcome.visited_leaf_entries(), 3);
    assert!(outcome.into_candidates().select(1).unwrap().1);
}

#[tokio::test]
async fn non_root_split_family_searches_source_and_targets() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    // Transient fanout three: the draining source and both receiving
    // targets are reachable through the parent's current Child Entries.
    fixture.header(&tree, 1, 2, 3, PartitionState::Ready);
    fixture.child(&tree, 1, 2, [2.0, 0.0]);
    fixture.child(&tree, 1, 3, [0.0, 0.0]);
    fixture.child(&tree, 1, 4, [4.0, 0.0]);
    fixture.header(&tree, 2, 1, 1, PartitionState::DrainingSplit);
    fixture.state(
        &tree,
        2,
        PartitionTransition::DrainingSplit {
            left: pk(3),
            right: pk(4),
            started_at_unix_millis: 0,
        },
    );
    fixture.header(&tree, 3, 1, 1, PartitionState::ReceivingSplit);
    fixture.state(
        &tree,
        3,
        PartitionTransition::ReceivingSplit {
            source: pk(2),
            started_at_unix_millis: 0,
        },
    );
    fixture.header(&tree, 4, 1, 1, PartitionState::ReceivingSplit);
    fixture.state(
        &tree,
        4,
        PartitionTransition::ReceivingSplit {
            source: pk(2),
            started_at_unix_millis: 0,
        },
    );
    for leaf in [2_u64, 3, 4] {
        fixture.synopsis(&tree, 1, leaf, &[0]);
    }
    fixture.entry(&tree, 1, 2, "b", 0, [2.0, 0.0]);
    fixture.entry(&tree, 1, 3, "a", 0, [1.0, 0.0]);
    fixture.entry(&tree, 1, 4, "c", 0, [3.0, 0.0]);

    let outcome = run(
        fixture.items,
        &manifest,
        &[tree_ref(&tree)],
        None,
        budgets(16, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(
        candidate_ids(&outcome),
        vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]
    );
    assert_eq!(outcome.visited_partitions(), 4);
    assert_eq!(outcome.visited_leaf_entries(), 3);
    assert!(!outcome.partition_budget_exhausted());
}

#[tokio::test]
async fn splitting_root_searches_only_the_root_body() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 1, 1, PartitionState::Splitting);
    fixture.state(
        &tree,
        1,
        PartitionTransition::Splitting {
            left: pk(2),
            right: pk(3),
            started_at_unix_millis: 0,
        },
    );
    fixture.synopsis(&tree, 1, 1, &[0]);
    fixture.entry(&tree, 1, 1, "root", 0, [1.0, 0.0]);
    // Unexposed targets exist physically but stay unreachable to search.
    fixture.header(&tree, 2, 1, 1, PartitionState::ReceivingSplit);
    fixture.synopsis(&tree, 1, 2, &[0]);
    fixture.entry(&tree, 1, 2, "hidden", 0, [0.5, 0.0]);

    let outcome = run(
        fixture.items,
        &manifest,
        &[tree_ref(&tree)],
        None,
        budgets(16, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(candidate_ids(&outcome), vec![b"root".as_slice()]);
    assert_eq!(outcome.visited_partitions(), 1);
    assert_eq!(outcome.visited_leaf_entries(), 1);
}

#[tokio::test]
async fn draining_root_searches_root_and_both_targets() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 1, 1, PartitionState::DrainingSplit);
    fixture.state(
        &tree,
        1,
        PartitionTransition::DrainingSplit {
            left: pk(2),
            right: pk(3),
            started_at_unix_millis: 0,
        },
    );
    for (target, record_id, vector) in [(2_u64, "a", [1.0, 0.0]), (3_u64, "c", [3.0, 0.0])] {
        fixture.header(&tree, target, 1, 1, PartitionState::ReceivingSplit);
        fixture.state(
            &tree,
            target,
            PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 0,
            },
        );
        fixture.synopsis(&tree, 1, target, &[0]);
        fixture.entry(&tree, 1, target, record_id, 0, vector);
    }
    fixture.synopsis(&tree, 1, 1, &[0]);
    fixture.entry(&tree, 1, 1, "b", 0, [2.0, 0.0]);

    let outcome = run(
        fixture.items,
        &manifest,
        &[tree_ref(&tree)],
        None,
        budgets(16, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    let mut ids = candidate_ids(&outcome);
    ids.sort_unstable();
    assert_eq!(ids, vec![b"a".as_slice(), b"b", b"c"]);
    assert_eq!(outcome.visited_partitions(), 3);
    assert_eq!(outcome.visited_leaf_entries(), 3);
    assert!(!outcome.partition_budget_exhausted());
}

#[tokio::test]
async fn merging_source_is_searched_like_an_ordinary_partition() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
    fixture.child(&tree, 1, 2, [0.0, 0.0]);
    fixture.child(&tree, 1, 3, [2.0, 0.0]);
    fixture.header(&tree, 2, 1, 1, PartitionState::Merging);
    fixture.state(
        &tree,
        2,
        PartitionTransition::Merging {
            started_at_unix_millis: 0,
        },
    );
    fixture.synopsis(&tree, 1, 2, &[0]);
    fixture.entry(&tree, 1, 2, "a", 0, [1.0, 0.0]);
    fixture.header(&tree, 3, 1, 1, PartitionState::Ready);
    fixture.synopsis(&tree, 1, 3, &[0]);
    fixture.entry(&tree, 1, 3, "b", 0, [2.0, 0.0]);

    let outcome = run(
        fixture.items,
        &manifest,
        &[tree_ref(&tree)],
        None,
        budgets(16, 64),
        DEFAULT_LEAF_BEAM,
    )
    .await
    .expect("traverse");
    assert_eq!(
        candidate_ids(&outcome),
        vec![b"a".as_slice(), b"b".as_slice()]
    );
    assert_eq!(outcome.visited_partitions(), 3);
    assert_eq!(outcome.visited_leaf_entries(), 2);
}

#[tokio::test]
async fn missing_or_inconsistent_partition_headers_are_corruption() {
    for (case, root_level, child_level) in [
        ("missing root Header", None, None),
        ("missing child Header", Some(2), None),
        ("child level mismatch", Some(2), Some(2)),
    ] {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        if let Some(level) = root_level {
            fixture.header(&tree, 1, level, 1, PartitionState::Ready);
            fixture.child(&tree, 1, 2, [0.0, 0.0]);
        }
        if let Some(level) = child_level {
            fixture.header(&tree, 2, level, 0, PartitionState::Ready);
        }
        let outcome = run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            None,
            budgets(16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await;
        assert_eq!(
            outcome.err().map(|error| error.kind()),
            Some(ErrorKind::Corruption),
            "{case}"
        );
    }
}

#[tokio::test]
async fn second_incoming_child_entry_is_corruption() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 3, 2, PartitionState::Ready);
    fixture.header(&tree, 2, 2, 1, PartitionState::Ready);
    fixture.header(&tree, 3, 2, 1, PartitionState::Ready);
    fixture.header(&tree, 4, 1, 0, PartitionState::Ready);
    fixture.child(&tree, 1, 2, [0.0, 0.0]);
    fixture.child(&tree, 1, 3, [1.0, 0.0]);
    fixture.child(&tree, 2, 4, [0.0, 0.0]);
    // A second incoming Child Entry for one non-root partition.
    fixture.child(&tree, 3, 4, [0.0, 0.0]);

    assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
}

#[tokio::test]
async fn root_target_or_merge_states_are_corruption() {
    for state in [PartitionState::ReceivingSplit, PartitionState::Merging] {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 0, state);

        assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
    }
}

#[tokio::test]
async fn draining_root_requires_its_matching_state() {
    // The State value is missing entirely.
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 1, 0, PartitionState::DrainingSplit);
    assert_corrupt(fixture.items.clone(), &manifest, &[tree_ref(&tree)]).await;

    // Adding a State value with a different discriminator must also fail.
    fixture.state(
        &tree,
        1,
        PartitionTransition::Ready {
            started_at_unix_millis: 0,
        },
    );
    assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
}

#[tokio::test]
async fn missing_synopsis_with_predicate_is_corruption() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 2, 1, PartitionState::Ready);
    fixture.child(&tree, 1, 2, [0.0, 0.0]);
    fixture.header(&tree, 2, 1, 1, PartitionState::Ready);
    fixture.entry(&tree, 1, 2, "x", 0, [1.0, 0.0]);

    let predicate = Predicate::Compare {
        field: FieldId(1),
        op: CompareOp::Eq,
        value: Value::I64(0),
    };
    assert_corruption(
        run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            Some(predicate),
            budgets(16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await,
    );
}

#[tokio::test]
async fn malformed_entry_values_are_corruption() {
    // A truncated canonical Leaf Entry payload breaks the codec frame.
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 1, 1, PartitionState::Ready);
    fixture.synopsis(&tree, 1, 1, &[0, 1]);
    let mut value = encode(
        &manifest,
        &PersistentValue::LeafEntry(LeafEntry::new(
            Bytes::from_static(b"x"),
            vec![Value::I64(1), Value::I64(0)],
            RaBitQ7::quantize(&[1.0, 0.0]).expect("quantize"),
        )),
    );
    value.pop();
    fixture.raw_entry(&tree, 1, "x", value);
    assert_corrupt(fixture.items.clone(), &manifest, &[tree_ref(&tree)]).await;

    // A conservative MayMatch synopsis requires loading the body. Even
    // though the entry's field would fail the exact predicate, its damaged
    // code must still be rejected before filtering can discard it.
    assert_corruption(
        run(
            fixture.items,
            &manifest,
            &[tree_ref(&tree)],
            Some(Predicate::Compare {
                field: FieldId(1),
                op: CompareOp::Eq,
                value: Value::I64(1),
            }),
            budgets(16, 64),
            DEFAULT_LEAF_BEAM,
        )
        .await,
    );

    // A truncated canonical Child Entry payload likewise fails closed.
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 2, 1, PartitionState::Ready);
    let mut value = encode(
        &manifest,
        &PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.0, 0.0])),
    );
    value.pop();
    fixture.raw_child(&tree, 1, 2, value);
    assert_corrupt(fixture.items, &manifest, &[tree_ref(&tree)]).await;
}

#[tokio::test]
async fn duplicate_record_ids_across_leaves_are_corruption() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let first = fixture.tree(1);
    fixture.header(&first, 1, 1, 1, PartitionState::Ready);
    fixture.synopsis(&first, 1, 1, &[0]);
    fixture.entry(&first, 1, 1, "dup", 0, [1.0, 0.0]);
    let second = fixture.tree(2);
    fixture.header(&second, 1, 1, 1, PartitionState::Ready);
    fixture.synopsis(&second, 2, 1, &[0]);
    fixture.entry(&second, 2, 1, "dup", 0, [2.0, 0.0]);

    assert_corrupt(
        fixture.items,
        &manifest,
        &[tree_ref(&first), tree_ref(&second)],
    )
    .await;
}

/// Exercises the production policy directly, without benchmark features.
async fn filtered_search(
    fixture: &Fixture<'_>,
    trees: &[EnumeratedTree],
    visits: u32,
    beam: u32,
    working_bytes: usize,
    mode: RoutingMode,
) -> Result<TraversalOutcome> {
    let mut txn =
        ReadLogicalTxn::for_index(MockReadTxn::new(fixture.items.clone()), fixture.manifest);
    let kernel = VectorKernel::new(DIMENSION, Metric::L2, SEED).unwrap();
    let predicate = CompiledPredicate::compile(
        Predicate::Compare {
            field: FieldId(1),
            op: CompareOp::Eq,
            value: Value::I64(100),
        },
        fixture.manifest.config().fields(),
    )
    .unwrap();
    let execution = GraphExecution {
        cache: PartitionCache::new(1 << 20),
        working_bytes,
        modes: std::sync::Mutex::new(Vec::new()),
    };
    super::traverse(
        &mut txn,
        &execution.cache,
        &kernel,
        TraversalRequest::new(&QUERY, trees, Some(&predicate), budgets(visits, 64), beam)
            .unwrap()
            .with_routing_mode(mode),
        &execution,
    )
    .await
}

#[tokio::test]
async fn filtered_tree_allowances_are_independent_and_share_the_visit_budget() {
    for matching in [false, true] {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let a = fixture.tree(1);
        let b = fixture.tree(2);
        fixture.header(&a, 1, 2, 2, PartitionState::Ready);
        for leaf in [2, 3] {
            fixture.child(&a, 1, leaf, [leaf as f32, 0.0]);
            fixture.header(&a, leaf, 1, 1, PartitionState::Ready);
            fixture.synopsis(&a, 1, leaf, &[100]);
            fixture.entry(&a, 1, leaf, &format!("a{leaf}"), 100, [leaf as f32, 0.0]);
        }
        fixture.header(&b, 1, 3, 2, PartitionState::Ready);
        for (parent, leaf) in [(2, 4), (3, 5)] {
            let value = if matching && leaf == 5 { 100 } else { 0 };
            fixture.child(&b, 1, parent, [parent as f32, 0.0]);
            fixture.header(&b, parent, 2, 1, PartitionState::Ready);
            fixture.child(&b, parent, leaf, [leaf as f32, 0.0]);
            fixture.header(&b, leaf, 1, 1, PartitionState::Ready);
            // MayMatch does not prove an entry actually passes the predicate.
            fixture.synopsis(&b, 2, leaf, &[0, 200]);
            fixture.entry(&b, 2, leaf, &format!("b{leaf}"), value, [leaf as f32, 0.0]);
        }
        let trees = [tree_ref(&a), tree_ref(&b)];
        let result = filtered_search(&fixture, &trees, 7, 1, 1 << 20, RoutingMode::Exact)
            .await
            .unwrap();
        assert_eq!(result.visited_partitions(), 7);
        assert_eq!(result.visited_leaf_entries(), 3);
        assert!(!result.partition_budget_exhausted());
        assert_eq!(
            candidate_ids(&result),
            if matching {
                vec![b"a2".as_slice(), b"b5".as_slice()]
            } else {
                vec![b"a2".as_slice()]
            }
        );
        let limited = filtered_search(&fixture, &trees, 5, 1, 1 << 20, RoutingMode::Exact)
            .await
            .unwrap();
        assert_eq!(limited.visited_partitions(), 5);
        assert!(limited.partition_budget_exhausted());
    }
}

#[tokio::test]
async fn filtered_pending_growth_obeys_routing_working_bytes() {
    let manifest = manifest();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    fixture.header(&tree, 1, 2, 200, PartitionState::Ready);
    for child in 2..202 {
        fixture.child(&tree, 1, child, [child as f32, 0.0]);
    }
    let result = filtered_search(
        &fixture,
        &[tree_ref(&tree)],
        10,
        1,
        16 << 10,
        RoutingMode::Exact,
    )
    .await;
    assert_eq!(result.unwrap_err().kind(), ErrorKind::LimitExceeded);
}

#[tokio::test]
async fn graph_exact_filtering_retains_children_for_replacement() {
    use crate::api::LocalGraphConfig;
    use crate::search::local_hnsw::{
        Centroids,
        persistence::{GraphSource, build_records},
    };
    let base = manifest();
    let manifest = IndexManifest::new(
        IndexLifecycle::Active,
        id(1),
        base.config()
            .clone()
            .with_local_graph(LocalGraphConfig::new(2, 8, SEED).unwrap()),
        SEED,
        base.bloom_parameters().to_vec(),
    )
    .unwrap();
    let kernel = VectorKernel::new(DIMENSION, Metric::L2, SEED).unwrap();
    let mut fixture = Fixture::new(&manifest);
    let tree = fixture.tree(1);
    let header = PartitionHeader::new(2, 2, 0, PartitionState::Ready).unwrap();
    fixture.header(&tree, 1, 2, 2, PartitionState::Ready);
    let children = vec![
        ChildEntry::new(pk(2), vec![1.0, 0.0]),
        ChildEntry::new(pk(3), vec![10.0, 0.0]),
    ];
    for (leaf, value, distance) in [(2, 0, 1.0), (3, 100, 10.0)] {
        fixture.child(&tree, 1, leaf, [distance, 0.0]);
        fixture.header(&tree, leaf, 1, 1, PartitionState::Ready);
        fixture.synopsis(&tree, 1, leaf, &[value]);
        fixture.entry(&tree, 1, leaf, &format!("r{leaf}"), value, [distance, 0.0]);
    }
    let records = build_records(
        Centroids::new(&kernel, &children, 2, &|| Ok(())).unwrap(),
        GraphSource {
            index: id(1),
            tree: &tree,
            parent: pk(1),
            header,
            parameters: manifest.config().local_graph().unwrap().parameters(),
        },
        16 << 20,
        &|| Ok(()),
    )
    .unwrap()
    .records;
    fixture.push(
        keys::graph_descriptor_key(id(1), &tree, pk(1)),
        PersistentValue::GraphDescriptor(records.descriptor.clone()),
    );
    for (ordinal, chunk) in records.chunks() {
        fixture.push(
            keys::graph_chunk_key(id(1), &tree, pk(1), ordinal),
            PersistentValue::GraphChunk(chunk),
        );
    }
    let result = filtered_search(
        &fixture,
        &[tree_ref(&tree)],
        3,
        1,
        16 << 20,
        RoutingMode::Exact,
    )
    .await
    .unwrap();
    assert_eq!(candidate_ids(&result), vec![b"r3".as_slice()]);
    assert_eq!(result.visited_partitions(), 3);
    assert!(!result.partition_budget_exhausted());
    // Explicit approximate routing retains its existing bounded candidate semantics.
    let graph = filtered_search(
        &fixture,
        &[tree_ref(&tree)],
        3,
        1,
        16 << 20,
        RoutingMode::PartitionLocalHnsw { ef_search: 8 },
    )
    .await
    .unwrap();
    assert!(graph.candidates().is_empty());
    assert_eq!(graph.visited_partitions(), 2);
}

mod filtered_tests {
    use super::*;

    async fn search(
        fixture: &Fixture<'_>,
        tree: &TreeKey,
        visits: u32,
        beam: u32,
    ) -> Result<TraversalOutcome> {
        filtered_search(
            fixture,
            &[tree_ref(tree)],
            visits,
            beam,
            1 << 20,
            RoutingMode::Exact,
        )
        .await
    }

    /// The matching leaf is below an internal branch discarded by ordinary beam=1.
    fn two_branches(fixture: &mut Fixture<'_>, matching: bool) -> TreeKey {
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 3, 2, PartitionState::Ready);
        for (parent, leaf, distance, value) in
            [(2, 4, 0.0, 0), (3, 5, 10.0, if matching { 100 } else { 0 })]
        {
            fixture.child(&tree, 1, parent, [distance, 0.0]);
            fixture.header(&tree, parent, 2, 1, PartitionState::Ready);
            fixture.child(&tree, parent, leaf, [distance, 0.0]);
            fixture.header(&tree, leaf, 1, 1, PartitionState::Ready);
            fixture.synopsis(&tree, 1, leaf, &[value]);
            fixture.entry(&tree, 1, leaf, &format!("r{leaf}"), value, [distance, 0.0]);
        }
        tree
    }

    #[tokio::test]
    async fn may_match_without_hits_does_not_consume_leaf_allowance() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 3, PartitionState::Ready);
        for leaf in 2..=4 {
            fixture.child(&tree, 1, leaf, [leaf as f32, 0.0]);
            fixture.header(&tree, leaf, 1, 1, PartitionState::Ready);
            // Historical extrema permit 100, but only the final leaf has a match.
            fixture.synopsis(&tree, 1, leaf, &[0, 200]);
            fixture.entry(
                &tree,
                1,
                leaf,
                &format!("r{leaf}"),
                if leaf == 4 { 100 } else { 0 },
                [leaf as f32, 0.0],
            );
        }
        let result = search(&fixture, &tree, 10, 1).await.unwrap();
        assert_eq!(candidate_ids(&result), vec![b"r4".as_slice()]);
        assert_eq!(result.visited_leaf_entries(), 3);
        assert!(!result.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn effective_leaves_resume_parents_and_obey_total_limits() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = two_branches(&mut fixture, true);
        let result = search(&fixture, &tree, 10, 1).await.unwrap();
        assert_eq!(candidate_ids(&result), vec![b"r5".as_slice()]);
        assert_eq!(result.visited_partitions(), 5);
        assert!(!result.partition_budget_exhausted());
        let limited = search(&fixture, &tree, 3, 1).await.unwrap();
        assert!(limited.candidates().is_empty());
        assert_eq!(limited.visited_partitions(), 3);
        assert!(limited.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn all_rejected_leaves_finish_naturally_at_the_visit_limit() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = two_branches(&mut fixture, false);
        let result = search(&fixture, &tree, 5, 1).await.unwrap();
        assert_eq!(result.visited_partitions(), 5);
        assert!(result.candidates().is_empty());
        assert!(!result.partition_budget_exhausted());
    }

    #[tokio::test]
    async fn effective_leaves_refill_only_the_remaining_allowance() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 2, 4, PartitionState::Ready);
        for leaf in 2..=5 {
            let value = if leaf == 2 { 0 } else { 100 };
            fixture.child(&tree, 1, leaf, [leaf as f32, 0.0]);
            fixture.header(&tree, leaf, 1, 1, PartitionState::Ready);
            fixture.synopsis(&tree, 1, leaf, &[0, 200]);
            fixture.entry(
                &tree,
                1,
                leaf,
                &format!("r{leaf}"),
                value,
                [leaf as f32, 0.0],
            );
        }
        for visits in [4, 10] {
            let result = search(&fixture, &tree, visits, 2).await.unwrap();
            assert_eq!(
                candidate_ids(&result),
                vec![b"r3".as_slice(), b"r4".as_slice()]
            );
            assert_eq!(result.visited_partitions(), 4);
            assert_eq!(result.visited_leaf_entries(), 3);
            assert!(!result.partition_budget_exhausted());
        }
    }

    #[tokio::test]
    async fn effective_leaf_target_does_not_skip_root_split_targets() {
        let manifest = manifest();
        let mut fixture = Fixture::new(&manifest);
        let tree = fixture.tree(1);
        fixture.header(&tree, 1, 1, 0, PartitionState::DrainingSplit);
        fixture.state(
            &tree,
            1,
            PartitionTransition::DrainingSplit {
                left: pk(2),
                right: pk(3),
                started_at_unix_millis: 0,
            },
        );
        fixture.synopsis(&tree, 1, 1, &[]);
        for leaf in [2, 3] {
            fixture.header(&tree, leaf, 1, 1, PartitionState::ReceivingSplit);
            fixture.state(
                &tree,
                leaf,
                PartitionTransition::ReceivingSplit {
                    source: pk(1),
                    started_at_unix_millis: 0,
                },
            );
            fixture.synopsis(&tree, 1, leaf, &[100]);
            fixture.entry(&tree, 1, leaf, &format!("r{leaf}"), 100, [leaf as f32, 0.0]);
        }
        let result = search(&fixture, &tree, 10, 1).await.unwrap();
        assert_eq!(result.candidates().len(), 2);
        assert_eq!(result.visited_partitions(), 3);
        assert!(!result.partition_budget_exhausted());
        let limited = search(&fixture, &tree, 2, 1).await.unwrap();
        assert_eq!(limited.candidates().len(), 1);
        assert!(limited.partition_budget_exhausted());
    }
}
