//! Public bounded search operation contract tests (issue #30).
//!
//! Every test drives the public `Index::search` API against the deterministic
//! in-memory backend. Hits are compared with a brute-force exact-distance
//! oracle; budget usage, exhaustion flags, and overlap truncation are asserted
//! per dimension. Intermediate topology states are hand-installed through the
//! public storage API because the split state machines (#10) do not exist yet.

use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use ktann::api::{
    CompareOp, DataType, ErrorKind, FieldId, FieldSchema, Index, IndexConfig, LogicalIndexId,
    Metric, OperationOptions, Predicate, Record, SearchBudgets, SearchOptions, SearchOutcome,
    SearchRequest, Value,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, HardLimits, ScanLimits, WriteTxn,
};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{
    IndexLifecycle, IndexManifest, PartitionHeader, PartitionState, PartitionSynopsis,
    PartitionTransition, PersistentValue, RecordLocation,
};
use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};
use tokio_util::sync::CancellationToken;

use support::{DeterministicBackend, DeterministicConfig};

#[allow(dead_code)]
mod support;

#[derive(Clone)]
struct SharedBackend {
    inner: Arc<DeterministicBackend>,
}

impl SharedBackend {
    fn new(inner: DeterministicBackend) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl Backend for SharedBackend {
    type ReadTxn<'backend> = support::DeterministicReadTxn<'backend>;

    type WriteTxn<'backend> = support::DeterministicWriteTxn<'backend>;

    fn hard_limits(&self) -> HardLimits {
        self.inner.hard_limits()
    }

    fn admission_budget(&self) -> AdmissionBudget {
        self.inner.admission_budget()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn shutdown(&self) {
        self.inner.shutdown().await;
    }

    async fn begin_read(&self) -> ktann::api::Result<Self::ReadTxn<'_>> {
        self.inner.begin_read().await
    }

    async fn begin_write(&self) -> ktann::api::Result<Self::WriteTxn<'_>> {
        self.inner.begin_write().await
    }
}

fn shared_backend(config: DeterministicConfig) -> SharedBackend {
    SharedBackend::new(DeterministicBackend::new(config))
}

/// A one-dimensional L2 index: rotation is the identity at dimension 1, so the
/// brute-force oracle needs no numeric setup. `bucket` is the Tree Key field;
/// `score` is a nullable non-key field for predicate coverage.
fn config() -> IndexConfig {
    IndexConfig::new(1, Metric::L2)
        .expect("valid config")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
            FieldSchema::new("score", DataType::I64)
                .expect("valid field")
                .nullable(),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
}

fn make_runtime(backend: SharedBackend) -> Runtime<SharedBackend> {
    // Search fixtures pin exact intermediate topology states; background
    // maintenance workers would advance them concurrently.
    Runtime::new(backend, support::manual_maintenance_config()).expect("runtime is valid")
}

fn pk(value: u64) -> ktann::api::PartitionKey {
    ktann::api::PartitionKey::new(value).expect("test Partition Key is nonzero")
}

fn tree_key(bucket: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(bucket)]).expect("valid tree key")
}

/// One fixture record: Record ID, stored component, Tree Key field, and the
/// nullable non-key field.
type Row = (Vec<u8>, f32, i64, Option<i64>);

fn record(row: &Row) -> Record {
    let (id, x, bucket, score) = row;
    Record::new(
        Bytes::copy_from_slice(id),
        Arc::from([*x]),
        vec![
            Value::I64(*bucket),
            score.map(Value::I64).unwrap_or(Value::Null),
        ],
    )
    .expect("valid record")
}

async fn insert_all(index: &Index<SharedBackend>, rows: &[Row]) {
    for row in rows {
        index.insert(record(row)).await.expect("insert record");
    }
}

/// Builds a default backend/runtime/index triple; insert `rows` afterwards.
async fn setup() -> (SharedBackend, Runtime<SharedBackend>, Index<SharedBackend>) {
    let backend = shared_backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create index");
    (backend, runtime, index)
}

/// The exact oracle: squared L2 accumulated in f64, ordered by `(distance,
/// Record ID)`, truncated to `k`, with the public Euclidean distance exposed.
fn brute_force(
    rows: &[Row],
    query: f32,
    k: usize,
    keep: impl Fn(&Row) -> bool,
) -> Vec<(Bytes, f64)> {
    let mut scored: Vec<(Bytes, f64)> = rows
        .iter()
        .filter(|row| keep(row))
        .map(|(id, x, _, _)| {
            let difference = f64::from(query) - f64::from(*x);
            (Bytes::copy_from_slice(id), difference * difference)
        })
        .collect();
    scored.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(k);
    scored
        .into_iter()
        .map(|(id, squared)| (id, squared.sqrt()))
        .collect()
}

fn hit_parts(outcome: &SearchOutcome) -> Vec<(Bytes, f64)> {
    outcome
        .hits
        .iter()
        .map(|hit| (hit.id().clone(), hit.distance()))
        .collect()
}

fn assert_no_budget_exhaustion(outcome: &SearchOutcome) {
    assert!(
        !outcome.exhausted.scanned_tree_keys
            && !outcome.exhausted.visited_partitions
            && !outcome.exhausted.visited_leaf_entries
            && !outcome.exhausted.exact_rerank_candidates,
        "no budget dimension is exhausted: {:?}",
        outcome.exhausted
    );
}

fn search_request(k: usize) -> SearchRequest {
    SearchRequest::new(Arc::from([0.0_f32]), k).expect("valid request")
}

async fn read_manifest(backend: &SharedBackend, index: LogicalIndexId) -> IndexManifest {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    match txn
        .get(LogicalKey::Manifest(index))
        .await
        .expect("read manifest")
    {
        Some(PersistentValue::IndexManifest(manifest)) => manifest,
        _ => panic!("committed manifest must exist"),
    }
}

/// Four trees with two records each, ordered so the global top four are the
/// nearest record of every tree in bucket order.
fn fanout_rows() -> Vec<Row> {
    vec![
        (b"a0".to_vec(), 0.5, 0, Some(5)),
        (b"a1".to_vec(), 5.0, 0, Some(10)),
        (b"b0".to_vec(), 1.5, 1, Some(5)),
        (b"b1".to_vec(), 6.0, 1, Some(20)),
        (b"c0".to_vec(), 2.5, 2, Some(5)),
        (b"c1".to_vec(), 7.0, 2, None),
        (b"d0".to_vec(), 3.5, 3, Some(5)),
        (b"d1".to_vec(), 8.0, 3, None),
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_empty_index_returns_an_empty_outcome() {
    let (_backend, runtime, index) = setup().await;

    let outcome = index.search(search_request(4)).await.expect("search");
    assert!(outcome.hits.is_empty());
    assert_eq!(outcome.usage.scanned_tree_keys, 0);
    assert_eq!(outcome.usage.visited_partitions, 0);
    assert_eq!(outcome.usage.visited_leaf_entries, 0);
    assert_eq!(outcome.usage.exact_rerank_candidates, 0);
    assert_no_budget_exhaustion(&outcome);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_matches_brute_force_and_reports_exact_usage() {
    let (_backend, runtime, index) = setup().await;
    let rows: Vec<Row> = (0..20_u8)
        .map(|i| {
            (
                format!("r{i:02}").into_bytes(),
                f32::from(i) + 0.5,
                1,
                Some(i64::from(i)),
            )
        })
        .collect();
    insert_all(&index, &rows).await;

    let outcome = index.search(search_request(5)).await.expect("search");
    assert_eq!(hit_parts(&outcome), brute_force(&rows, 0.0, 5, |_| true));
    assert_eq!(outcome.usage.scanned_tree_keys, 1);
    assert_eq!(outcome.usage.visited_partitions, 1);
    assert_eq!(outcome.usage.visited_leaf_entries, 20);
    assert!(outcome.usage.exact_rerank_candidates >= 5);
    assert_no_budget_exhaustion(&outcome);

    // A warm-cache rerun over an unchanged snapshot is bit-identical.
    let rerun = index.search(search_request(5)).await.expect("search");
    assert_eq!(hit_parts(&rerun), hit_parts(&outcome));
    assert_eq!(rerun.usage, outcome.usage);
    assert_eq!(rerun.exhausted, outcome.exhausted);
    assert_eq!(
        rerun.rabitq_overlap_truncated,
        outcome.rabitq_overlap_truncated
    );
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_tree_fanout_merges_one_deterministic_order() {
    let (_backend, runtime, index) = setup().await;
    let rows = fanout_rows();
    insert_all(&index, &rows).await;

    let outcome = index.search(search_request(4)).await.expect("search");
    assert_eq!(hit_parts(&outcome), brute_force(&rows, 0.0, 4, |_| true));
    assert_eq!(outcome.usage.scanned_tree_keys, 4);
    assert_eq!(outcome.usage.visited_partitions, 4);
    assert_eq!(outcome.usage.visited_leaf_entries, 8);
    assert_no_budget_exhaustion(&outcome);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tree_key_predicate_scans_only_the_planned_range() {
    let (_backend, runtime, index) = setup().await;
    let rows = fanout_rows();
    insert_all(&index, &rows).await;

    let request = search_request(4).with_predicate(Predicate::Compare {
        field: FieldId(0),
        op: CompareOp::Eq,
        value: Value::I64(2),
    });
    let outcome = index.search(request).await.expect("search");
    assert_eq!(
        hit_parts(&outcome),
        brute_force(&rows, 0.0, 4, |row| row.2 == 2)
    );
    // Only the equality-narrowed tree leaves the directory: one decoded key,
    // one visited partition, and the tree's two entries.
    assert_eq!(outcome.usage.scanned_tree_keys, 1);
    assert_eq!(outcome.usage.visited_partitions, 1);
    assert_eq!(outcome.usage.visited_leaf_entries, 2);
    assert_no_budget_exhaustion(&outcome);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_key_predicates_filter_exactly_and_null_never_matches() {
    let (_backend, runtime, index) = setup().await;
    let rows = fanout_rows();
    insert_all(&index, &rows).await;

    let request = search_request(4).with_predicate(Predicate::Compare {
        field: FieldId(1),
        op: CompareOp::Eq,
        value: Value::I64(5),
    });
    let outcome = index.search(request).await.expect("search");
    assert_eq!(
        hit_parts(&outcome),
        brute_force(&rows, 0.0, 4, |row| row.3 == Some(5))
    );
    assert_no_budget_exhaustion(&outcome);

    // SQL UNKNOWN never qualifies: only IsNull returns the NULL-score rows.
    let request = search_request(4).with_predicate(Predicate::IsNull(FieldId(1)));
    let outcome = index.search(request).await.expect("search");
    assert_eq!(
        hit_parts(&outcome),
        brute_force(&rows, 0.0, 4, |row| row.3.is_none())
    );
    assert_no_budget_exhaustion(&outcome);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_predicate_matching_no_tree_key_yields_an_empty_outcome() {
    let (_backend, runtime, index) = setup().await;
    let rows = fanout_rows();
    insert_all(&index, &rows).await;

    // The empty IN is FALSE, so planning proves no Tree Key can match.
    let request = search_request(4).with_predicate(Predicate::In {
        field: FieldId(0),
        values: Vec::new(),
    });
    let outcome = index.search(request).await.expect("search");
    assert!(outcome.hits.is_empty());
    assert_eq!(outcome.usage.scanned_tree_keys, 0);
    assert_eq!(outcome.usage.visited_partitions, 0);
    assert_no_budget_exhaustion(&outcome);

    // A point predicate with no stored tree scans its range without a hit.
    let request = search_request(4).with_predicate(Predicate::Compare {
        field: FieldId(0),
        op: CompareOp::Eq,
        value: Value::I64(99),
    });
    let outcome = index.search(request).await.expect("search");
    assert!(outcome.hits.is_empty());
    assert_eq!(outcome.usage.scanned_tree_keys, 0);
    assert_no_budget_exhaustion(&outcome);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_budget_dimension_reports_exactly_its_own_exhaustion() {
    let rows = fanout_rows();

    // Scanned Tree Keys: only the first two trees in canonical order are
    // materialized, so hits come from buckets 0 and 1 alone.
    let (_backend, runtime, index) = setup().await;
    insert_all(&index, &rows).await;
    let options = SearchOptions::default()
        .with_scanned_tree_keys(2)
        .expect("valid override");
    let outcome = index
        .search(search_request(4).with_options(options))
        .await
        .expect("search");
    assert_eq!(outcome.usage.scanned_tree_keys, 2);
    assert!(outcome.exhausted.scanned_tree_keys);
    assert!(!outcome.exhausted.visited_partitions);
    assert!(!outcome.exhausted.visited_leaf_entries);
    assert!(!outcome.exhausted.exact_rerank_candidates);
    assert_eq!(
        hit_parts(&outcome),
        brute_force(&rows, 0.0, 4, |row| row.2 <= 1)
    );
    runtime.shutdown().await.expect("shutdown");

    // Visited partitions: roots tie at distance zero, so enumeration order
    // decides which two trees advance.
    let (_backend, runtime, index) = setup().await;
    insert_all(&index, &rows).await;
    let options = SearchOptions::default()
        .with_visited_partitions(2)
        .expect("valid override");
    let outcome = index
        .search(search_request(4).with_options(options))
        .await
        .expect("search");
    assert_eq!(outcome.usage.visited_partitions, 2);
    assert!(!outcome.exhausted.scanned_tree_keys);
    assert!(outcome.exhausted.visited_partitions);
    assert!(!outcome.exhausted.visited_leaf_entries);
    assert!(!outcome.exhausted.exact_rerank_candidates);
    assert_eq!(
        hit_parts(&outcome),
        brute_force(&rows, 0.0, 4, |row| row.2 <= 1)
    );
    runtime.shutdown().await.expect("shutdown");

    // Visited Leaf Entries: the budget funds the first three entries in
    // canonical Record ID order and stops mid-leaf.
    let (_backend, runtime, index) = setup().await;
    let rows: Vec<Row> = vec![
        (b"a".to_vec(), 1.0, 1, None),
        (b"b".to_vec(), 2.0, 1, None),
        (b"c".to_vec(), 3.0, 1, None),
        (b"d".to_vec(), 4.0, 1, None),
        (b"e".to_vec(), 5.0, 1, None),
    ];
    insert_all(&index, &rows).await;
    let options = SearchOptions::default()
        .with_visited_leaf_entries(3)
        .expect("valid override");
    let outcome = index
        .search(search_request(2).with_options(options))
        .await
        .expect("search");
    assert_eq!(outcome.usage.visited_leaf_entries, 3);
    assert!(outcome.exhausted.visited_leaf_entries);
    assert!(!outcome.exhausted.scanned_tree_keys);
    assert!(!outcome.exhausted.visited_partitions);
    assert!(!outcome.exhausted.exact_rerank_candidates);
    assert_eq!(
        hit_parts(&outcome),
        brute_force(&rows[..3], 0.0, 2, |_| true)
    );
    runtime.shutdown().await.expect("shutdown");

    // Exact rerank candidates: two trees of ten identical vectors each. Every
    // interval coincides, so the per-leaf caps and the merged selection both
    // truncate to the Runtime ceiling and report it.
    let backend = shared_backend(DeterministicConfig::default());
    let search_budgets =
        SearchBudgets::new(4_096, 1_024, 65_536, 3).expect("valid exact-rerank ceiling");
    let runtime_config = support::manual_maintenance_config()
        .with_default_search_budgets(search_budgets)
        .expect("valid runtime search budgets");
    let runtime = Runtime::new(backend, runtime_config).expect("runtime is valid");
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create index");
    let mut rows: Vec<Row> = Vec::new();
    for i in 0..10_u8 {
        rows.push((format!("a{i}").into_bytes(), 1.0, 1, None));
        rows.push((format!("b{i}").into_bytes(), 1.0, 2, None));
    }
    insert_all(&index, &rows).await;
    let outcome = index.search(search_request(2)).await.expect("search");
    assert_eq!(outcome.usage.exact_rerank_candidates, 3);
    assert!(outcome.exhausted.exact_rerank_candidates);
    assert!(outcome.rabitq_overlap_truncated);
    assert!(!outcome.exhausted.scanned_tree_keys);
    assert!(!outcome.exhausted.visited_partitions);
    assert!(!outcome.exhausted.visited_leaf_entries);
    assert_eq!(outcome.hits.len(), 2);
    assert!(outcome.hits.iter().all(|hit| hit.distance() == 1.0));

    // The depleted outcome is deterministic across identical resubmissions.
    let rerun = index.search(search_request(2)).await.expect("search");
    assert_eq!(hit_parts(&rerun), hit_parts(&outcome));
    assert_eq!(rerun.usage, outcome.usage);
    assert_eq!(rerun.exhausted, outcome.exhausted);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_draining_root_split_stays_searchable_with_exact_membership() {
    let (backend, runtime, index) = setup().await;
    let rows: Vec<Row> = vec![
        (b"a".to_vec(), 1.0, 7, Some(1)),
        (b"b".to_vec(), 2.0, 7, Some(2)),
        (b"c".to_vec(), 3.0, 7, Some(3)),
    ];
    insert_all(&index, &rows).await;

    let iid = index.logical_index_id();
    let manifest = read_manifest(&backend, iid).await;
    let key = tree_key(7);

    // Read the committed root Leaf Entries back: the RaBitQ7 encoding is
    // internal, so the crafted state moves the decoded envelopes rather than
    // re-encoding vectors.
    let mut entries = {
        let raw = backend.begin_read().await.expect("begin read");
        let mut txn = ReadLogicalTxn::for_index(raw, &manifest).expect("bind index");
        let range = LogicalRange::leaf_entries(&manifest, &key, pk(1)).expect("leaf range");
        let page = txn
            .scan(
                &range,
                None,
                ScanLimits {
                    item_limit: 16,
                    byte_limit: 1 << 20,
                },
            )
            .await
            .expect("scan root leaf");
        assert!(page.next_cursor().is_none(), "one page covers the root");
        let mut entries = std::collections::BTreeMap::new();
        for item in page.into_items() {
            let LogicalKey::LeafEntry { id, .. } = item.key() else {
                panic!("a Leaf Entry range holds only Leaf Entries");
            };
            let id = id.clone();
            let PersistentValue::LeafEntry(entry) = item.into_value() else {
                panic!("a Leaf Entry range holds only Leaf Entries");
            };
            entries.insert(id, entry);
        }
        entries
    };

    // Hand-install the committed DrainingSplit shape: the root keeps "b" and
    // drains "a" and "c" into the two published ReceivingSplit targets, with
    // every Header, State, Synopsis, and Record Location moved together.
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind index");
    let header = |partition, count, state| {
        (
            LogicalKey::Header {
                index: iid,
                tree_key: key.clone(),
                partition,
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, count, 7, state).expect("header"),
            ),
        )
    };
    let state = |partition, transition: PartitionTransition| {
        (
            LogicalKey::State {
                index: iid,
                tree_key: key.clone(),
                partition,
            },
            PersistentValue::PartitionState(transition),
        )
    };
    let synopsis = |partition, row: &Row| {
        let mut synopsis = PartitionSynopsis::empty(&manifest);
        synopsis
            .expand(
                &manifest,
                &[
                    Value::I64(row.2),
                    row.3.map(Value::I64).unwrap_or(Value::Null),
                ],
            )
            .expect("expand synopsis");
        (
            LogicalKey::Synopsis {
                index: iid,
                tree_key: key.clone(),
                partition,
            },
            PersistentValue::PartitionSynopsis(synopsis),
        )
    };
    let writes = [
        header(pk(1), 1, PartitionState::DrainingSplit),
        state(
            pk(1),
            PartitionTransition::DrainingSplit {
                left: pk(2),
                right: pk(3),
                started_at_unix_millis: 0,
            },
        ),
        header(pk(2), 1, PartitionState::ReceivingSplit),
        state(
            pk(2),
            PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 0,
            },
        ),
        synopsis(pk(2), &rows[0]),
        header(pk(3), 1, PartitionState::ReceivingSplit),
        state(
            pk(3),
            PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 0,
            },
        ),
        synopsis(pk(3), &rows[2]),
    ];
    for (key, value) in writes {
        txn.put(key, value).await.expect("write topology");
    }
    for (row, target) in [(&rows[0], pk(2)), (&rows[2], pk(3))] {
        let id = Bytes::copy_from_slice(&row.0);
        let entry = entries.remove(&id).expect("committed entry exists");
        txn.put(
            LogicalKey::LeafEntry {
                index: iid,
                tree_key: key.clone(),
                partition: target,
                id: id.clone(),
            },
            PersistentValue::LeafEntry(entry),
        )
        .await
        .expect("move leaf entry");
        txn.delete(LogicalKey::LeafEntry {
            index: iid,
            tree_key: key.clone(),
            partition: pk(1),
            id: id.clone(),
        })
        .await
        .expect("remove source entry");
        txn.put(
            LogicalKey::Location {
                index: iid,
                id: id.clone(),
            },
            PersistentValue::RecordLocation(RecordLocation::new(key.clone(), target)),
        )
        .await
        .expect("move record location");
    }
    txn.commit().await.expect("commit crafted split");

    // The committed intermediate state searches the root and both targets
    // with exact membership: every record is found exactly once.
    let outcome = index.search(search_request(10)).await.expect("search");
    assert_eq!(hit_parts(&outcome), brute_force(&rows, 0.0, 10, |_| true));
    assert_eq!(outcome.usage.scanned_tree_keys, 1);
    assert_eq!(outcome.usage.visited_partitions, 3);
    assert_eq!(outcome.usage.visited_leaf_entries, 3);
    assert_no_budget_exhaustion(&outcome);

    // Synopsis pruning applies to the crafted targets: only the root body
    // holds a score of 2, so the targets are provably skipped uncharged.
    let request = search_request(10).with_predicate(Predicate::Compare {
        field: FieldId(1),
        op: CompareOp::Eq,
        value: Value::I64(2),
    });
    let outcome = index.search(request).await.expect("search");
    assert_eq!(
        hit_parts(&outcome),
        brute_force(&rows, 0.0, 10, |row| row.3 == Some(2))
    );
    assert_eq!(outcome.usage.visited_partitions, 3);
    assert_eq!(outcome.usage.visited_leaf_entries, 1);
    assert_no_budget_exhaustion(&outcome);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_and_deadline_fail_the_whole_operation() {
    let (_backend, runtime, index) = setup().await;
    insert_all(&index, &fanout_rows()).await;

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let options = OperationOptions::default().with_cancellation(cancellation);
    let error = index
        .search_with_control(search_request(4), options)
        .await
        .expect_err("cancelled search");
    assert_eq!(error.kind(), ErrorKind::Cancelled);

    let options = OperationOptions::default().with_deadline(Instant::now());
    let error = index
        .search_with_control(search_request(4), options)
        .await
        .expect_err("expired deadline");
    assert_eq!(error.kind(), ErrorKind::DeadlineExceeded);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inconsistent_persistent_state_fails_closed() {
    // A Header count that disagrees with the decoded body is Corruption.
    let (backend, runtime, index) = setup().await;
    let rows: Vec<Row> = vec![(b"a".to_vec(), 1.0, 1, None), (b"b".to_vec(), 2.0, 1, None)];
    insert_all(&index, &rows).await;
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let iid = index.logical_index_id();
    let key = tree_key(1);
    {
        let raw = backend.begin_write().await.expect("begin write");
        let mut txn = WriteLogicalTxn::for_index(
            raw,
            &manifest,
            backend.hard_limits(),
            backend.admission_budget(),
        )
        .expect("bind index");
        txn.put(
            LogicalKey::Header {
                index: iid,
                tree_key: key.clone(),
                partition: pk(1),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 3, 9, PartitionState::Ready).expect("header"),
            ),
        )
        .await
        .expect("write mismatched header");
        txn.commit().await.expect("commit mismatch");
    }
    let error = index
        .search(search_request(4))
        .await
        .expect_err("count mismatch is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    runtime.shutdown().await.expect("shutdown");

    // A Record Location that disagrees with the candidate's leaf is
    // Corruption, never a silent deduplication or skip.
    let (backend, runtime, index) = setup().await;
    insert_all(&index, &rows).await;
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let iid = index.logical_index_id();
    {
        let raw = backend.begin_write().await.expect("begin write");
        let mut txn = WriteLogicalTxn::for_index(
            raw,
            &manifest,
            backend.hard_limits(),
            backend.admission_budget(),
        )
        .expect("bind index");
        txn.put(
            LogicalKey::Location {
                index: iid,
                id: Bytes::from_static(b"a"),
            },
            PersistentValue::RecordLocation(RecordLocation::new(tree_key(1), pk(99))),
        )
        .await
        .expect("write divergent location");
        txn.commit().await.expect("commit divergence");
    }
    let error = index
        .search(search_request(4))
        .await
        .expect_err("location divergence is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    runtime.shutdown().await.expect("shutdown");

    // Malformed Leaf Entry bytes fail closed at decode time.
    let (backend, runtime, index) = setup().await;
    insert_all(&index, &rows).await;
    let iid = index.logical_index_id();
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.put(
        Bytes::from(
            keys::leaf_entry_key(iid, &tree_key(1), pk(1), &Bytes::from_static(b"a"))
                .expect("leaf entry key"),
        ),
        Bytes::from_static(b"not-a-canonical-leaf-entry"),
    )
    .await
    .expect("write malformed entry");
    txn.commit().await.expect("commit malformed entry");
    let error = index
        .search(search_request(4))
        .await
        .expect_err("malformed entry is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    runtime.shutdown().await.expect("shutdown");
}

/// One Record ID reachable from two partitions at once — the leftover state
/// of a failed update in the reference corpus (issue #100, item A4). Where
/// CockroachDB dedupes and reranks the true distance, KTANN fails closed: a
/// duplicate Record ID in one snapshot is Corruption, never silently
/// deduplicated (design search.md section 6).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_duplicate_record_id_across_partitions_fails_closed() {
    let (backend, runtime, index) = setup().await;
    let rows: Vec<Row> = vec![
        (b"a".to_vec(), 1.0, 7, Some(1)),
        (b"b".to_vec(), 2.0, 7, Some(2)),
        (b"c".to_vec(), 3.0, 7, Some(3)),
    ];
    insert_all(&index, &rows).await;

    let iid = index.logical_index_id();
    let manifest = read_manifest(&backend, iid).await;
    let key = tree_key(7);

    // Read the committed root Leaf Entries back: the RaBitQ7 encoding is
    // internal, so the crafted state copies the decoded envelopes rather
    // than re-encoding vectors.
    let mut entries = {
        let raw = backend.begin_read().await.expect("begin read");
        let mut txn = ReadLogicalTxn::for_index(raw, &manifest).expect("bind index");
        let range = LogicalRange::leaf_entries(&manifest, &key, pk(1)).expect("leaf range");
        let page = txn
            .scan(
                &range,
                None,
                ScanLimits {
                    item_limit: 16,
                    byte_limit: 1 << 20,
                },
            )
            .await
            .expect("scan root leaf");
        assert!(page.next_cursor().is_none(), "one page covers the root");
        let mut entries = std::collections::BTreeMap::new();
        for item in page.into_items() {
            let LogicalKey::LeafEntry { id, .. } = item.key() else {
                panic!("a Leaf Entry range holds only Leaf Entries");
            };
            let id = id.clone();
            let PersistentValue::LeafEntry(entry) = item.into_value() else {
                panic!("a Leaf Entry range holds only Leaf Entries");
            };
            entries.insert(id, entry);
        }
        entries
    };

    // Hand-install a committed DrainingSplit shape where the failed-update
    // leftover is visible: "a" stays in the root body AND a stray copy lives
    // in the pk=2 target, while "c" drains cleanly to pk=3. Every Header,
    // State, and Synopsis is consistent; only the duplicated Record ID is
    // wrong.
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind index");
    let header = |partition, count, state| {
        (
            LogicalKey::Header {
                index: iid,
                tree_key: key.clone(),
                partition,
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, count, 7, state).expect("header"),
            ),
        )
    };
    let state = |partition, transition: PartitionTransition| {
        (
            LogicalKey::State {
                index: iid,
                tree_key: key.clone(),
                partition,
            },
            PersistentValue::PartitionState(transition),
        )
    };
    let synopsis = |partition, row: &Row| {
        let mut synopsis = PartitionSynopsis::empty(&manifest);
        synopsis
            .expand(
                &manifest,
                &[
                    Value::I64(row.2),
                    row.3.map(Value::I64).unwrap_or(Value::Null),
                ],
            )
            .expect("expand synopsis");
        (
            LogicalKey::Synopsis {
                index: iid,
                tree_key: key.clone(),
                partition,
            },
            PersistentValue::PartitionSynopsis(synopsis),
        )
    };
    let writes = [
        header(pk(1), 2, PartitionState::DrainingSplit),
        state(
            pk(1),
            PartitionTransition::DrainingSplit {
                left: pk(2),
                right: pk(3),
                started_at_unix_millis: 0,
            },
        ),
        header(pk(2), 1, PartitionState::ReceivingSplit),
        state(
            pk(2),
            PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 0,
            },
        ),
        synopsis(pk(2), &rows[0]),
        header(pk(3), 1, PartitionState::ReceivingSplit),
        state(
            pk(3),
            PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 0,
            },
        ),
        synopsis(pk(3), &rows[2]),
    ];
    for (key, value) in writes {
        txn.put(key, value).await.expect("write topology");
    }
    // The stray copy of "a" in pk=2: no delete at the root, no Location
    // update — the Record Location still names the root, as a torn
    // relocation would leave it.
    let id_a = Bytes::from_static(b"a");
    let entry_a = entries.remove(&id_a).expect("committed entry exists");
    txn.put(
        LogicalKey::LeafEntry {
            index: iid,
            tree_key: key.clone(),
            partition: pk(2),
            id: id_a.clone(),
        },
        PersistentValue::LeafEntry(entry_a),
    )
    .await
    .expect("copy leaf entry");
    // "c" drains cleanly: moved out of the root with its Location.
    let id_c = Bytes::from_static(b"c");
    let entry_c = entries.remove(&id_c).expect("committed entry exists");
    txn.put(
        LogicalKey::LeafEntry {
            index: iid,
            tree_key: key.clone(),
            partition: pk(3),
            id: id_c.clone(),
        },
        PersistentValue::LeafEntry(entry_c),
    )
    .await
    .expect("move leaf entry");
    txn.delete(LogicalKey::LeafEntry {
        index: iid,
        tree_key: key.clone(),
        partition: pk(1),
        id: id_c.clone(),
    })
    .await
    .expect("remove source entry");
    txn.put(
        LogicalKey::Location {
            index: iid,
            id: id_c.clone(),
        },
        PersistentValue::RecordLocation(RecordLocation::new(key.clone(), pk(3))),
    )
    .await
    .expect("move record location");
    txn.commit().await.expect("commit crafted state");

    let error = index
        .search(search_request(10))
        .await
        .expect_err("a duplicate Record ID across partitions is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_limits_paginate_scans_and_bound_batches() {
    // One item per backend scan page: directory enumeration and body loads
    // paginate through cursors without changing the outcome.
    let backend = shared_backend(DeterministicConfig {
        max_scan_page_items: 1,
        ..DeterministicConfig::default()
    });
    let runtime = make_runtime(backend);
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create index");
    let rows = fanout_rows();
    insert_all(&index, &rows).await;
    let outcome = index.search(search_request(4)).await.expect("search");
    assert_eq!(hit_parts(&outcome), brute_force(&rows, 0.0, 4, |_| true));
    assert_eq!(outcome.usage.scanned_tree_keys, 4);
    assert_no_budget_exhaustion(&outcome);
    runtime.shutdown().await.expect("shutdown");

    // A backend batch ceiling that admits every mutation batch (at most 9
    // keys for the tree-creating first insert) but is below the rerank
    // batch's 16 keys (8 candidates, 2 keys each) surfaces LimitExceeded
    // rather than a partial result.
    let backend = shared_backend(DeterministicConfig {
        max_batch_size: 12,
        ..DeterministicConfig::default()
    });
    let runtime = make_runtime(backend);
    let index = runtime
        .create_index("index", config())
        .await
        .expect("create index");
    let rows: Vec<Row> = (0..8_u8)
        .map(|i| (format!("r{i}").into_bytes(), 1.0, 1, None))
        .collect();
    insert_all(&index, &rows).await;
    let error = index
        .search(search_request(8))
        .await
        .expect_err("rerank batch exceeds the backend ceiling");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dropping_or_dropped_index_fails_closed_for_search() {
    let backend = shared_backend(DeterministicConfig::default());
    let runtime = make_runtime(backend.clone());
    let dropping = runtime
        .create_index("dropping", config())
        .await
        .expect("create index");

    // Flip the persisted Manifest to Dropping underneath the open handle.
    let manifest = read_manifest(&backend, dropping.logical_index_id()).await;
    {
        let raw = backend.begin_write().await.expect("begin write");
        let limits = backend.hard_limits();
        let budget = backend.admission_budget();
        let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);
        txn.put(
            LogicalKey::Manifest(manifest.logical_index_id()),
            PersistentValue::IndexManifest(
                manifest
                    .with_lifecycle(IndexLifecycle::Dropping)
                    .expect("dropping manifest"),
            ),
        )
        .await
        .expect("put dropping manifest");
        txn.commit().await.expect("commit dropping manifest");
    }
    let error = dropping
        .search(search_request(4))
        .await
        .expect_err("dropping index");
    assert_eq!(error.kind(), ErrorKind::IndexDropping);

    // A completed drop removes the Manifest; the stale handle fails closed.
    let dropped = runtime
        .create_index("dropped", config())
        .await
        .expect("create second index");
    runtime.drop_index("dropped").await.expect("drop");
    let error = dropped
        .search(search_request(4))
        .await
        .expect_err("dropped index");
    assert_eq!(error.kind(), ErrorKind::IndexNotFound);
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_requests_fail_before_admission() {
    let (_backend, runtime, index) = setup().await;
    insert_all(&index, &fanout_rows()).await;

    // The query dimension must match the Manifest.
    let request = SearchRequest::new(Arc::from([0.0_f32, 1.0]), 4).expect("valid request");
    let error = index.search(request).await.expect_err("wrong dimension");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);

    // A zero-k request is rejected at construction.
    assert!(SearchRequest::new(Arc::from([0.0_f32]), 0).is_err());
    runtime.shutdown().await.expect("shutdown");
}
