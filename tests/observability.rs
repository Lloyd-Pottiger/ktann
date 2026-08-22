//! Privacy-safe metrics and tracing audits (issue #36).
//!
//! These tests drive the public API on the deterministic backend with
//! canary-shaped sensitive data through success, failure, corruption, retry,
//! and maintenance paths, then scan the complete captured telemetry: no
//! metric, trace, or error rendering may contain a raw Index Name, Tree Key,
//! Record ID, field value, vector, or payload, and every label and trace
//! field must stay within the documented bounded allowlist (design
//! `runtime-operations.md` section 5). The remaining assertions prove that
//! operation, budget, cache, maintenance, commit, and verification outcomes
//! stay operationally distinguishable.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use ktann::api::{
    CompareOp, DataType, ErrorKind, FieldId, FieldSchema, Index, IndexConfig, Metric, Mutation,
    OperationOptions, Predicate, Record, RuntimeConfig, SearchOptions, SearchRequest,
    SynopsisConfig, Value, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys;
use tokio_util::sync::CancellationToken;

use support::observe::{audit_lock, capture};
use support::{CommitFault, DeterministicBackend, DeterministicConfig, SharedBackend, audit};

#[allow(dead_code)]
mod support;

/// Canary-shaped sensitive values. Every byte pattern a redaction bug could
/// leak is covered by one of these: Index Names, Tree Keys (the `bucket`
/// field is the Tree Key), Record IDs, field values, payloads, and a
/// distinctive vector component. The prefixes match every numbered record,
/// field value, and payload below.
const CANARIES: &[&str] = &[
    INDEX_NAME,
    DIRTY_INDEX_NAME,
    MISSING_INDEX_NAME,
    CANCEL_INDEX_NAME,
    RECORD_ID_PREFIX,
    FIELD_VALUE_PREFIX,
    PAYLOAD_PREFIX,
    VECTOR_CANARY,
];

const INDEX_NAME: &str = "canary-index-71c3f0a9";
const DIRTY_INDEX_NAME: &str = "canary-dirty-index-55b2d8e1";
const MISSING_INDEX_NAME: &str = "canary-missing-index-2e6a77b0";
const CANCEL_INDEX_NAME: &str = "canary-cancel-index-0f8d31b2";
const RECORD_ID_PREFIX: &str = "canary-record-id-";
const FIELD_VALUE_PREFIX: &str = "canary-field-value-";
const PAYLOAD_PREFIX: &str = "canary-payload-";
/// Exactly representable in f32, so a vector leak renders this exact prefix.
const VECTOR_CANARY: &str = "98304.5";

/// A 2-dimensional L2 index: `bucket` is the Tree Key field, `tag` a nullable
/// Bloom-synopsized String field, with tiny partitions to force splits.
fn index_config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
            FieldSchema::new("tag", DataType::String)
                .expect("valid field")
                .nullable()
                .with_synopsis(SynopsisConfig::MinMaxBloom {
                    expected_distinct: NonZeroU32::new(8).expect("nonzero"),
                    false_positive_rate: 0.01,
                })
                .expect("valid synopsis"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(2, 4)
        .expect("valid partition entries")
}

fn record(n: u32, bucket: i64) -> Record {
    Record::new(
        Bytes::from(format!("{RECORD_ID_PREFIX}{n}")),
        Arc::from([f32::from(n as u16), 98304.5]),
        vec![
            Value::I64(bucket),
            Value::String(format!("{FIELD_VALUE_PREFIX}{n}")),
        ],
    )
    .expect("valid record")
    .with_payload(Bytes::from(format!("{PAYLOAD_PREFIX}{n}")))
    .expect("valid payload")
}

fn search_request(k: usize) -> SearchRequest {
    SearchRequest::new(Arc::from([1.0_f32, 0.5]), k)
        .expect("valid request")
        .with_predicate(Predicate::Compare {
            field: FieldId(1),
            op: CompareOp::Eq,
            value: Value::String(format!("{FIELD_VALUE_PREFIX}3")),
        })
}

/// Polls until every partition of the index is `Ready`: all offered Fixups
/// have executed to completion.
async fn wait_for_maintenance(backend: &SharedBackend, index: &Index<SharedBackend>) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let partitions = audit::list_partitions(backend, index.logical_index_id())
            .await
            .expect("list partitions");
        if partitions
            .iter()
            .all(|(_, _, header)| header.state() == ktann::storage::values::PartitionState::Ready)
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for maintenance to settle"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Writes garbage bytes at one raw key, bypassing every invariant.
async fn raw_put(backend: &SharedBackend, key: Vec<u8>) {
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.put(Bytes::from(key), Bytes::from_static(b"garbage"))
        .await
        .expect("raw put");
    txn.commit().await.expect("commit");
}

/// Whether one captured metric series exists with exactly these labels.
fn has_series(series: &[(String, Vec<(String, String)>)], name: &str, labels: &[(&str, &str)]) {
    let mut expected: Vec<(String, String)> = labels
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect();
    expected.sort();
    assert!(
        series
            .iter()
            .any(|(series_name, series_labels)| series_name == name && *series_labels == expected),
        "missing series {name} with labels {labels:?}; captured: {series:?}"
    );
}

/// The full audit battery: success, failure, corruption, retry, cancellation,
/// import, verification, and maintenance paths, all under canary data.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn redaction_audit_covers_all_paths() {
    let _serialize = audit_lock().await;
    let capture = capture();
    capture.clear();

    // Phase A: success paths with live maintenance workers.
    let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()));
    let runtime = Runtime::new(
        backend.clone(),
        RuntimeConfig::default()
            .with_maintenance(1, 16)
            .and_then(|config| config.with_import_limits(1, 2))
            .expect("valid runtime config"),
    )
    .expect("runtime");
    let index = runtime
        .create_index(INDEX_NAME, index_config())
        .await
        .expect("create index");
    runtime.open_index(INDEX_NAME).await.expect("open index");

    for n in 0..3 {
        index.insert(record(n, 7)).await.expect("insert");
    }
    index.upsert(record(0, 7)).await.expect("upsert");
    let more = (3..10).map(|n| Mutation::Insert(record(n, 7))).collect();
    index.batch_mutate(more).await.expect("batch mutate");
    index
        .delete(Bytes::from(format!("{RECORD_ID_PREFIX}9")))
        .await
        .expect("delete");

    // Failure paths: duplicate record and a missing Index Name.
    let duplicate = index.insert(record(1, 7)).await.expect_err("duplicate");
    assert_eq!(duplicate.kind(), ErrorKind::RecordAlreadyExists);
    let missing = runtime
        .open_index(MISSING_INDEX_NAME)
        .await
        .expect_err("missing index");
    assert_eq!(missing.kind(), ErrorKind::IndexNotFound);

    // Reads: hits and misses with canary identifiers.
    index
        .get(
            Bytes::from(format!("{RECORD_ID_PREFIX}1")),
            Default::default(),
        )
        .await
        .expect("get");
    index
        .get(Bytes::from("canary-record-id-absent"), Default::default())
        .await
        .expect("get miss");
    index
        .batch_get(
            vec![
                Bytes::from(format!("{RECORD_ID_PREFIX}2")),
                Bytes::from("canary-record-id-absent"),
            ],
            Default::default(),
        )
        .await
        .expect("batch get");

    // Import Session backpressure: one in-flight slot, two batches.
    let mut session = index
        .import_session(Default::default())
        .expect("import session");
    session
        .submit(vec![Mutation::Insert(record(10, 7))])
        .await
        .expect("submit first batch");
    session
        .submit(vec![Mutation::Insert(record(11, 7))])
        .await
        .expect("submit second batch");
    let results = session.finish().await;
    assert_eq!(results.len(), 2);
    assert!(results[0].result.is_ok() && results[1].result.is_ok());

    // One search may rediscover threshold-crossing leaves; let every offered
    // Fixup finish so partition epochs are stable for the cached search.
    index.search(search_request(3)).await.expect("search");
    wait_for_maintenance(&backend, &index).await;

    // Searches: a predicate over the canary field exercises Bloom synopsis
    // pruning; the repeated search over the settled tree exercises cache
    // hits; the tiny Leaf Entry budget exhausts that dimension against any
    // non-empty leaf (a pruned partition is ineligible, so the Leaf Entry
    // dimension is the deterministic one).
    index
        .search(search_request(3))
        .await
        .expect("settled search");
    index
        .search(search_request(3))
        .await
        .expect("cached search");
    let constrained = SearchRequest::new(Arc::from([1.0_f32, 0.5]), 3)
        .expect("valid request")
        .with_options(
            SearchOptions::default()
                .with_visited_leaf_entries(1)
                .expect("valid override"),
        );
    index.search(constrained).await.expect("constrained search");

    // Verification: a complete, issue-free report.
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(
        report.complete,
        "healthy index verifies complete: {report:?}"
    );

    runtime.shutdown().await.expect("shutdown");

    // Phase B: retry, commit-unknown, corruption, and cancellation paths with
    // maintenance disabled so fault injection is deterministic.
    let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()));
    let runtime =
        Runtime::new(backend.clone(), support::manual_maintenance_config()).expect("runtime");
    let dirty = runtime
        .create_index(DIRTY_INDEX_NAME, index_config())
        .await
        .expect("create dirty index");
    for n in 12..14 {
        dirty.insert(record(n, 3)).await.expect("insert");
    }

    backend
        .inner()
        .push_fault(CommitFault::Abort)
        .expect("fault");
    dirty
        .insert(record(14, 3))
        .await
        .expect("insert after retry");

    backend
        .inner()
        .push_fault(CommitFault::UnknownApplied)
        .expect("fault");
    let unknown = dirty
        .insert(record(15, 3))
        .await
        .expect_err("unknown outcome");
    assert_eq!(unknown.kind(), ErrorKind::CommitOutcomeUnknown);

    // A malformed sibling key produces redacted verification issues.
    let mut malformed = keys::manifest_key(dirty.logical_index_id());
    malformed.push(0x00);
    raw_put(&backend, malformed).await;
    let dirty_report = dirty
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert_eq!(dirty_report.issues.len(), 1);

    // Corrupting the Manifest itself fails foreground operations closed.
    let manifest_key = keys::manifest_key(dirty.logical_index_id());
    raw_put(&backend, manifest_key).await;
    let corruption = dirty
        .search(search_request(2))
        .await
        .expect_err("corrupt search");
    assert_eq!(corruption.kind(), ErrorKind::Corruption);
    let corruption = dirty
        .get(
            Bytes::from(format!("{RECORD_ID_PREFIX}12")),
            Default::default(),
        )
        .await
        .expect_err("corrupt get");
    assert_eq!(corruption.kind(), ErrorKind::Corruption);
    let corruption = dirty
        .verify(VerifyOptions::default())
        .await
        .expect_err("corrupt verify");
    assert_eq!(corruption.kind(), ErrorKind::Corruption);

    // Caller-side cancellation and deadline exercise the aborted-task
    // observation path.
    let cancellable = runtime
        .create_index(CANCEL_INDEX_NAME, index_config())
        .await
        .expect("create cancellable index");
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = cancellable
        .search_with_control(
            search_request(2),
            OperationOptions::default().with_cancellation(token),
        )
        .await
        .expect_err("cancelled search");
    assert_eq!(cancelled.kind(), ErrorKind::Cancelled);
    let expired = cancellable
        .search_with_control(
            search_request(2),
            OperationOptions::default().with_deadline(Instant::now()),
        )
        .await
        .expect_err("expired search");
    assert_eq!(expired.kind(), ErrorKind::DeadlineExceeded);
    runtime.shutdown().await.expect("shutdown");

    // The complete capture is free of raw sensitive data.
    capture.assert_no_canaries(CANARIES);
    capture.assert_labels_bounded();
    capture.assert_trace_fields_bounded();

    // Operation, budget, cache, maintenance, commit, and verification
    // outcomes stay operationally distinguishable.
    let series = capture.metric_labels();
    for operation in [
        "create_index",
        "open_index",
        "insert",
        "upsert",
        "delete",
        "batch_mutate",
        "get",
        "batch_get",
        "search",
        "verify",
    ] {
        has_series(
            &series,
            "ktann.operation.total",
            &[("operation", operation), ("outcome", "ok")],
        );
    }
    for (operation, outcome) in [
        ("insert", "record_already_exists"),
        ("insert", "commit_outcome_unknown"),
        ("open_index", "index_not_found"),
        ("search", "corruption"),
        ("get", "corruption"),
        ("verify", "corruption"),
        ("search", "cancelled"),
        ("search", "deadline_exceeded"),
    ] {
        has_series(
            &series,
            "ktann.operation.total",
            &[("operation", operation), ("outcome", outcome)],
        );
    }
    has_series(
        &series,
        "ktann.operation.duration",
        &[("operation", "search"), ("outcome", "ok")],
    );
    has_series(&series, "ktann.write.retries", &[("operation", "insert")]);
    for dimension in [
        "scanned_tree_keys",
        "visited_partitions",
        "visited_leaf_entries",
        "exact_rerank_candidates",
    ] {
        has_series(
            &series,
            "ktann.search.budget.usage",
            &[("dimension", dimension)],
        );
    }
    has_series(
        &series,
        "ktann.search.budget.exhausted",
        &[("dimension", "visited_leaf_entries")],
    );
    has_series(
        &series,
        "ktann.cache.lookup",
        &[("level", "leaf"), ("result", "hit")],
    );
    has_series(
        &series,
        "ktann.cache.lookup",
        &[("level", "leaf"), ("result", "miss")],
    );
    has_series(
        &series,
        "ktann.cache.install",
        &[("level", "leaf"), ("result", "installed")],
    );
    has_series(&series, "ktann.fixup.admission", &[("outcome", "enqueued")]);
    has_series(&series, "ktann.fixup.execution", &[("outcome", "settled")]);
    has_series(&series, "ktann.fixup.state_age", &[("kind", "split")]);
    has_series(&series, "ktann.import.wait", &[("gate", "in_flight_slot")]);
    has_series(&series, "ktann.import.wait", &[("gate", "backlog")]);
    has_series(&series, "ktann.verify.reports", &[("outcome", "complete")]);
    has_series(
        &series,
        "ktann.verify.issues",
        &[("kind", "invalid_encoding")],
    );
    has_series(&series, "ktann.bloom.fill_ratio", &[]);

    // Trace capture proves the debug paths ran through the same policy:
    // operation and fixup spans carry only allowlisted identifiers, and
    // failure events carry only the stable error kind.
    let spans = capture.spans();
    let events = capture.events();
    assert!(
        spans.iter().any(|fields| {
            fields
                .iter()
                .any(|(field, value)| field == "operation" && value == "search")
                && fields.iter().any(|(field, _)| field == "logical_index_id")
        }),
        "missing search operation span"
    );
    assert!(
        spans.iter().any(|fields| {
            fields.iter().any(|(field, value)| {
                field == "tree_key_hash"
                    && value.len() == 64
                    && value.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        }),
        "missing fixup span with a stable Tree Key hash"
    );
    for error_kind in ["commit_outcome_unknown", "corruption"] {
        assert!(
            events.iter().any(|fields| {
                fields
                    .iter()
                    .any(|(field, value)| field == "error_kind" && value == error_kind)
            }),
            "missing failure event with error_kind {error_kind}"
        );
    }
    assert!(
        events.iter().any(|fields| {
            fields
                .iter()
                .any(|(field, value)| field == "operation" && value == "insert")
                && fields.iter().any(|(field, _)| field == "attempt")
        }),
        "missing retry event"
    );
}
