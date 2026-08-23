//! API-level recall parity scenario for the production adapter crates
//! (issue #100): the public Runtime/Index API over a real backend must meet
//! the same recall contract the deterministic-backend corpus pins.
//!
//! Each adapter test includes this file by path (`#[path = ...]`) and drives
//! it with its own backend handle, mirroring how `backend_contract.rs` is
//! shared. The scenario loads the siftsmall 1k prefix twice: an index whose
//! partitions can never split asserts exact recall (a flat scan is exact,
//! with and without an exact filter), and an index with tiny partitions
//! settles through demand-driven maintenance while a recall floor is polled.
//! Split timing under real workers is deliberately not controlled, so the
//! settled tree asserts a floor, never an exact golden.

// Included by path into both adapter crates, each using a different subset.
#![allow(dead_code)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use ktann::api::{
    CompareOp, DataType, ErrorKind, FieldId, FieldSchema, Index, IndexConfig, Metric, Mutation,
    MutationOutcome, Predicate, Record, RuntimeConfig, SearchRequest, Value, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;

/// The recall floor for the settled tree at the default beam width.
pub const SETTLED_RECALL_FLOOR: f64 = 0.9;

fn config(dimension: usize, minimum: u32, maximum: u32) -> IndexConfig {
    IndexConfig::new(dimension, Metric::L2)
        .expect("valid config")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("field"),
            FieldSchema::new("label", DataType::I64).expect("field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
        .with_partition_entries(minimum, maximum)
        .expect("valid partition entries")
}

/// Record IDs follow the corpus convention: the dataset ordinal.
fn record_id(ordinal: usize) -> Bytes {
    Bytes::from(format!("r{ordinal:06}"))
}

/// Loads the base set in bounded ordinary batches; `bucket` pins one tree and
/// `label` carries the ordinal so filters can select a prefix.
async fn load<B: Backend>(index: &Index<B>, base: &[Arc<[f32]>]) {
    for (batch, chunk) in base.chunks(100).enumerate() {
        let mutations: Vec<Mutation> = chunk
            .iter()
            .enumerate()
            .map(|(n, vector)| {
                let ordinal = batch * 100 + n;
                Mutation::Insert(
                    Record::new(
                        record_id(ordinal),
                        vector.clone(),
                        vec![Value::I64(1), Value::I64(ordinal as i64)],
                    )
                    .expect("record"),
                )
            })
            .collect();
        // Background fixups race the load on the tiny-partition index; a
        // contention-exhausted batch commits nothing, so retrying the same
        // batch is safe.
        let mut retries = 0_u32;
        let outcomes = loop {
            match index.batch_mutate(mutations.clone()).await {
                Ok(outcomes) => break outcomes,
                Err(error) if error.kind() == ErrorKind::ContentionExhausted => {
                    retries += 1;
                    assert!(retries <= 20, "load keeps exhausting contention retries");
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                Err(error) => panic!("batch failed: {error:?}"),
            }
        };
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome, MutationOutcome::Inserted)),
            "every batch insert succeeds"
        );
    }
}

/// Brute-force top-k Record IDs by exact L2 over the base set, ordered by
/// (distance, Record ID) — the same contract the corpus oracle mirrors.
fn truth_ids(base: &[Arc<[f32]>], query: &[f32], k: usize, label_le: Option<i64>) -> Vec<Bytes> {
    let mut scored: Vec<(f64, Bytes)> = base
        .iter()
        .enumerate()
        .filter(|(ordinal, _)| label_le.is_none_or(|limit| *ordinal as i64 <= limit))
        .map(|(ordinal, vector)| {
            let distance = vector
                .iter()
                .zip(query.iter())
                .map(|(a, b)| {
                    let delta = f64::from(*a) - f64::from(*b);
                    delta * delta
                })
                .sum();
            (distance, record_id(ordinal))
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("finite distances")
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

/// Mean recall@10 of the queries against the brute-force truth; an optional
/// exact filter restricts both the engine predicate and the truth to a label
/// prefix.
pub async fn recall<B: Backend>(
    index: &Index<B>,
    base: &[Arc<[f32]>],
    queries: &[Arc<[f32]>],
    label_le: Option<i64>,
) -> f64 {
    let k = 10;
    let mut total = 0.0;
    for query in queries {
        let mut request = SearchRequest::new(query.clone(), k).expect("request");
        if let Some(limit) = label_le {
            request = request.with_predicate(Predicate::Compare {
                field: FieldId(1),
                op: CompareOp::LessOrEqual,
                value: Value::I64(limit),
            });
        }
        let outcome = index.search(request).await.expect("search");
        let truth = truth_ids(base, query, k, label_le);
        let matched = truth
            .iter()
            .filter(|id| outcome.hits.iter().any(|hit| hit.id() == *id))
            .count();
        total += matched as f64 / k as f64;
    }
    total / queries.len() as f64
}

/// Runs the recall parity scenario against one backend. `base` is the
/// siftsmall 1k prefix and `queries` held-out siftsmall descriptors; both are
/// loaded by the caller so this file stays free of fixture-path concerns.
pub async fn run<B: Backend>(backend: B, base: Vec<Arc<[f32]>>, queries: Vec<Arc<[f32]>>) {
    assert!(!base.is_empty() && !queries.is_empty());
    let dimension = base[0].len();
    let runtime = Runtime::new(
        backend,
        RuntimeConfig::default()
            .with_maintenance(2, 16)
            .and_then(|config| config.with_attempts(32, 32))
            .and_then(|config| config.with_import_limits(1, 1))
            .expect("valid runtime config"),
    )
    .expect("runtime");

    // Partitions sized above the dataset: no split can trigger, search is an
    // exact flat scan, and recall must be 100% — with and without a filter.
    let flat = runtime
        .create_index("flat", config(dimension, 512, 2048))
        .await
        .expect("create flat index");
    load(&flat, &base).await;
    let flat_recall = recall(&flat, &base, &queries, None).await;
    assert_eq!(flat_recall, 1.0, "flat recall must be exact");
    let filtered_recall = recall(&flat, &base, &queries, Some(499)).await;
    assert_eq!(filtered_recall, 1.0, "filtered flat recall must be exact");

    // Tiny partitions: the load triggers demand-driven splits. Verify's
    // object counts prove internal partitions exist (Child Entries outnumber
    // the plain record entries), and the recall floor must then hold across
    // consecutive polls.
    let split = runtime
        .create_index("split", config(dimension, 8, 32))
        .await
        .expect("create split index");
    load(&split, &base).await;
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut stable = 0_u32;
    loop {
        let report = split
            .verify(VerifyOptions::default())
            .await
            .expect("verify");
        let structured = report.objects.entries > report.objects.vector_records;
        let value = recall(&split, &base, &queries, None).await;
        stable = if structured && value >= SETTLED_RECALL_FLOOR {
            stable + 1
        } else {
            0
        };
        if stable >= 3 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "settled-tree recall never reached the floor (structured={structured}, last {value})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let report = split
        .verify(VerifyOptions::default())
        .await
        .expect("final verify");
    assert!(
        report.complete && report.issues.is_empty(),
        "issues: {:?}",
        report.issues
    );

    runtime.shutdown().await.expect("shutdown");
}
