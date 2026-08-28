//! Metric recording contract tests (issue #100, item C3).
//!
//! `observability.rs` audits that captured telemetry never leaks caller data;
//! this suite asserts the other side: the documented `ktann.*` series (design
//! `runtime-operations.md` section 5, inventory in `src/observe.rs`) actually
//! fire with the expected labels and counts as the public API drives work —
//! foreground operations succeeding and failing, budget exhaustion, the
//! demand-driven Fixup queue, import gates, and verification.
//!
//! Metric names are not public API, so the expected strings are duplicated
//! here deliberately: the test is an independent check of the implementation
//! against the documented inventory. Volatile internals (retry counts, cache
//! hit ratios, timings) are asserted by presence, never by exact value.

use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{
    DataType, FieldId, FieldSchema, ImportOptions, IndexConfig, Metric, Mutation, Record,
    RuntimeConfig, SearchOptions, SearchRequest, UpsertResult, Value, VerifyOptions,
};
use ktann::runtime::Runtime;

use support::observe::CounterSeries;
use support::{DeterministicBackend, DeterministicConfig, SharedBackend, audit, observe};

#[allow(dead_code)]
mod support;

/// Returns one cumulative counter series value, or zero when absent.
fn counter(snapshot: &[CounterSeries], name: &str, labels: &[(&str, &str)]) -> u64 {
    snapshot
        .iter()
        .filter(|(series, series_labels, _)| {
            series == name
                && labels.iter().all(|(key, value)| {
                    series_labels
                        .iter()
                        .any(|label| label == &(key.to_string(), value.to_string()))
                })
        })
        .map(|(_, _, value)| *value)
        .sum()
}

/// Returns one counter series increase between cumulative snapshots.
fn delta(
    before: &[CounterSeries],
    after: &[CounterSeries],
    name: &str,
    labels: &[(&str, &str)],
) -> u64 {
    counter(after, name, labels) - counter(before, name, labels)
}

/// One-dimensional L2 vectors sharded by one i64 Tree Key field, with tiny
/// partitions so a dozen records exercises demand-driven splits.
fn index_config() -> IndexConfig {
    IndexConfig::new(1, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(2, 8)
        .expect("valid partition entries")
}

fn record(id: &str, x: f32) -> Record {
    record_in_bucket(id, x, 7)
}

fn record_in_bucket(id: &str, x: f32, bucket: i64) -> Record {
    Record::new(
        Bytes::copy_from_slice(id.as_bytes()),
        Arc::from([x]),
        vec![Value::I64(bucket)],
    )
    .expect("valid record")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_batches_offer_only_coalesced_actionable_partitions() {
    let _serial = observe::audit_lock().await;
    let capture = observe::capture();

    let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()));
    let config = RuntimeConfig::default()
        .with_maintenance(1, 16)
        .and_then(|config| config.with_import_limits(1, 1))
        .expect("valid runtime config");
    let runtime = Runtime::new(backend, config).expect("runtime");
    let index = runtime
        .create_index("actionable-fixups", index_config())
        .await
        .expect("create index");

    let before_healthy = capture.metric_counters();
    let healthy = (0..8_u8)
        .map(|id| Mutation::Insert(record_in_bucket(&format!("healthy-{id}"), f32::from(id), 8)))
        .collect();
    index.batch_mutate(healthy).await.expect("healthy batch");
    let after_healthy = capture.metric_counters();
    for outcome in ["enqueued", "duplicate", "saturated"] {
        assert_eq!(
            delta(
                &before_healthy,
                &after_healthy,
                "ktann.fixup.admission",
                &[("outcome", outcome)],
            ),
            0,
            "a healthy final Header must not be offered",
        );
    }

    let before_actionable = capture.metric_counters();
    let oversized = (0..9_u8)
        .map(|id| {
            Mutation::Insert(record_in_bucket(
                &format!("oversized-{id}"),
                f32::from(id),
                9,
            ))
        })
        .collect();
    index
        .batch_mutate(oversized)
        .await
        .expect("oversized batch");
    let after_actionable = capture.metric_counters();
    assert_eq!(
        delta(
            &before_actionable,
            &after_actionable,
            "ktann.fixup.admission",
            &[("outcome", "enqueued")],
        ),
        1,
        "one committed batch offers one partition once",
    );
    for outcome in ["duplicate", "saturated"] {
        assert_eq!(
            delta(
                &before_actionable,
                &after_actionable,
                "ktann.fixup.admission",
                &[("outcome", outcome)],
            ),
            0,
            "batch-local coalescing happens before queue admission",
        );
    }

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn operations_record_the_documented_series() {
    let _serial = observe::audit_lock().await;
    let capture = observe::capture();
    let before = capture.metric_counters();

    let backend = SharedBackend::new(DeterministicBackend::new(DeterministicConfig::default()));
    let config = RuntimeConfig::default()
        .with_maintenance(2, 16)
        .and_then(|config| config.with_import_limits(1, 1))
        .expect("valid runtime config");
    let runtime = Runtime::new(backend.clone(), config).expect("runtime");
    let index = runtime
        .create_index("metrics", index_config())
        .await
        .expect("create index");

    // Foreground mutations: twelve ok, one duplicate-ID failure, one
    // upsert replace, one delete.
    for id in 0..12_u8 {
        index
            .insert(record(&format!("r{id:03}"), f32::from(id)))
            .await
            .expect("insert");
    }
    let duplicate = index.insert(record("r000", 99.0)).await;
    assert_eq!(
        duplicate.expect_err("duplicate insert").kind(),
        ktann::api::ErrorKind::RecordAlreadyExists
    );
    let upserted = index.upsert(record("r001", 1.5)).await.expect("upsert");
    assert!(matches!(upserted, UpsertResult::Replaced));
    assert!(
        index
            .delete(Bytes::from_static(b"r002"))
            .await
            .expect("delete")
    );

    // Reads.
    index
        .get(Bytes::from_static(b"r001"), Default::default())
        .await
        .expect("get")
        .expect("record exists");
    let batch = index
        .batch_get(
            vec![Bytes::from_static(b"r001"), Bytes::from_static(b"r999")],
            Default::default(),
        )
        .await
        .expect("batch get");
    assert_eq!(batch.len(), 2);

    // An import session: two ordinary mutation batches.
    let mut session = index
        .import_session(ImportOptions::default())
        .expect("import session");
    for wave in 0..2_u8 {
        let mutations = (0..3_u8)
            .map(|n| {
                let id = 100 + wave * 3 + n;
                Mutation::Insert(record(&format!("r{id:03}"), f32::from(id)))
            })
            .collect();
        session.submit(mutations).await.expect("submit");
    }
    let results = session.finish().await;
    assert!(
        results
            .iter()
            .all(|result| result.result.as_ref().expect("batch ok").len() == 3)
    );

    // Demand-driven maintenance settles the 17 records into a split tree.
    audit::settle(&index, &backend, 17).await;

    // Searches: one with sufficient budgets, one with a leaf-entry budget
    // of one that must report exhaustion.
    index
        .search(SearchRequest::new(Arc::from([3.1_f32]), 1).expect("request"))
        .await
        .expect("search");
    let tight = index
        .search(
            SearchRequest::new(Arc::from([3.1_f32]), 1)
                .expect("request")
                .with_options(
                    SearchOptions::default()
                        .with_visited_leaf_entries(1)
                        .expect("valid budget"),
                ),
        )
        .await
        .expect("tight search");
    assert!(tight.exhausted.visited_leaf_entries);

    // Verification of the settled index completes.
    let report = index
        .verify(VerifyOptions::default())
        .await
        .expect("verify");
    assert!(report.complete);

    runtime.shutdown().await.expect("shutdown");

    // --- Counter assertions (before/after diff of the cumulative capture) ---
    let after = capture.metric_counters();
    let bumped = |name: &str, labels: &[(&str, &str)]| delta(&before, &after, name, labels);

    // Every foreground operation is counted exactly once by outcome.
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[("operation", "create_index"), ("outcome", "ok")]
        ),
        1
    );
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[("operation", "insert"), ("outcome", "ok")]
        ),
        12
    );
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[
                ("operation", "insert"),
                ("outcome", "record_already_exists")
            ]
        ),
        1
    );
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[("operation", "upsert"), ("outcome", "ok")]
        ),
        1
    );
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[("operation", "delete"), ("outcome", "ok")]
        ),
        1
    );
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[("operation", "get"), ("outcome", "ok")]
        ),
        1
    );
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[("operation", "batch_get"), ("outcome", "ok")]
        ),
        1
    );
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[("operation", "batch_mutate"), ("outcome", "ok")]
        ),
        2
    );
    assert_eq!(
        bumped(
            "ktann.operation.total",
            &[("operation", "verify"), ("outcome", "ok")]
        ),
        1
    );
    // The two explicit searches plus the settle polls; the exact poll count
    // is timing-dependent.
    assert!(
        bumped(
            "ktann.operation.total",
            &[("operation", "search"), ("outcome", "ok")]
        ) >= 2
    );

    // Exactly one budget exhaustion across all dimensions: the tight
    // leaf-entry search. The settle polls run with default budgets.
    for dimension in [
        "scanned_tree_keys",
        "visited_partitions",
        "exact_rerank_candidates",
    ] {
        assert_eq!(
            bumped("ktann.search.budget.exhausted", &[("dimension", dimension)]),
            0
        );
    }
    assert_eq!(
        bumped(
            "ktann.search.budget.exhausted",
            &[("dimension", "visited_leaf_entries")]
        ),
        1
    );

    // The demand-driven queue admitted and finished split work.
    assert!(bumped("ktann.fixup.admission", &[("outcome", "enqueued")]) >= 1);
    assert!(bumped("ktann.fixup.execution", &[("outcome", "settled")]) >= 1);
    assert!(
        bumped(
            "ktann.write.attempts",
            &[("operation", "batch_mutate"), ("outcome", "committed")]
        ) >= 2
    );
    assert_eq!(
        bumped(
            "ktann.write.attempts",
            &[("operation", "insert"), ("outcome", "failed")]
        ),
        1
    );
    assert!(
        bumped(
            "ktann.write.mutations",
            &[("operation", "batch_mutate"), ("outcome", "committed")]
        ) > 0
    );
    assert!(
        bumped(
            "ktann.write.mutation_bytes",
            &[("operation", "batch_mutate"), ("outcome", "committed")]
        ) > 0
    );
    assert!(
        bumped(
            "ktann.fixup.steps",
            &[("kind", "split"), ("result", "began")]
        ) >= 1
    );
    assert!(
        bumped(
            "ktann.fixup.steps",
            &[("kind", "split"), ("result", "drained")]
        ) >= 1
    );
    assert!(
        bumped(
            "ktann.fixup.steps",
            &[("kind", "split"), ("result", "completed")]
        ) >= 1
    );

    // The completed verification report is counted.
    assert_eq!(
        bumped("ktann.verify.reports", &[("outcome", "complete")]),
        1
    );

    // Searches consulted the Partition Cache.
    let cache_lookups: u64 = ["hit", "miss", "stale_miss"]
        .iter()
        .map(|result| {
            bumped(
                "ktann.cache.lookup",
                &[("level", "leaf"), ("result", result)],
            )
        })
        .sum();
    assert!(cache_lookups >= 1, "searches record cache lookups");

    // --- Presence assertions for histograms and gauges (values are ---
    // --- timing-dependent and never golden) ---
    let series: Vec<(String, Vec<(String, String)>)> = capture.metric_labels();
    let seen = |name: &str, labels: &[(&str, &str)]| {
        series.iter().any(|(series_name, series_labels)| {
            series_name == name
                && labels.iter().all(|(key, value)| {
                    series_labels
                        .iter()
                        .any(|l| l == &(key.to_string(), value.to_string()))
                })
        })
    };
    assert!(seen(
        "ktann.operation.duration",
        &[("operation", "insert"), ("outcome", "ok")]
    ));
    for dimension in [
        "scanned_tree_keys",
        "visited_partitions",
        "visited_leaf_entries",
        "exact_rerank_candidates",
    ] {
        assert!(seen(
            "ktann.search.budget.usage",
            &[("dimension", dimension)]
        ));
    }
    for stage in ["approximate_selection", "exact_reranking"] {
        assert!(seen("ktann.search.stage.duration", &[("stage", stage)]));
    }
    assert!(seen("ktann.import.wait", &[("gate", "in_flight_slot")]));
    assert!(seen("ktann.import.wait", &[("gate", "backlog")]));
    assert!(seen(
        "ktann.write.commit.duration",
        &[("operation", "batch_mutate"), ("outcome", "committed")]
    ));
    assert!(seen("ktann.fixup.drain.entries", &[("kind", "split")]));
    assert!(seen("ktann.fixup.state_age", &[("kind", "split")]));
    assert!(series.iter().any(|(name, _)| name == "ktann.fixup.backlog"));
}
