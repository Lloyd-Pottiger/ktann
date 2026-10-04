//! Opt-in full-corpus validation of audited publication on FoundationDB.
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use foundationdb::Database;
use ktann::api::{
    BulkBuildOptions, GetOptions, IndexConfig, Metric, OperationOptions, Record, RuntimeConfig,
    SearchOptions, SearchRequest,
};
use ktann::runtime::Runtime;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

use crate::resource::ResourceSnapshot;
#[path = "../../ktann-foundationdb/tests/support/mod.rs"]
mod support;

/// The builder itself exhaustively audits the immutable construction before
/// publishing. Queries and sampled gets then exercise the published index; this
/// does not claim that ordinary single-snapshot Index::verify scales to 1M FDB.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "requires local FoundationDB and cached Cohere1M; run with --release"]
async fn foundationdb_bulk_cohere_1m() {
    let mut data = crate::dataset::load_large("cohere-1m").expect("cached Cohere1M");
    assert_eq!(data.base.len(), 1_000_000);
    assert!(data.queries.len() >= 1000);
    let truth = data
        .ground_truth
        .take()
        .expect("official complete-corpus ground truth");
    assert!(truth.len() >= 1000);
    let _network = support::boot_foundationdb();
    let namespace = format!(
        "ktann-bulk-1m-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    eprintln!("large bulk validation namespace={namespace}");
    let cluster = std::env::var("FDB_CLUSTER_FILE").ok();
    let backend = || {
        FoundationDbBackend::new(
            Database::new(cluster.as_deref()).unwrap(),
            BackendNamespace::new(&namespace).unwrap(),
        )
    };
    let cleanup = backend();
    let runtime = Runtime::new(
        backend(),
        RuntimeConfig::default()
            .with_maintenance(0, 1)
            .unwrap()
            .with_import_limits(1, 1)
            .unwrap(),
    )
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(1800);
    let result: Result<(), String> = async {
        let before = ResourceSnapshot::capture()?;
        let started = Instant::now();
        // Consume source vectors rather than retaining an extra harness copy.
        let records = data.ids.iter().cloned().zip(std::mem::take(&mut data.base)).map(|(id, vector)| {
            Record::new(id, vector.to_vec(), vec![]).map_err(|e| super::error_at("record", e))
        }).collect::<Result<Vec<_>, _>>()?;
        let config = IndexConfig::new(768, Metric::Cosine).map_err(|e| super::error_at("config", e))?;
        let index = runtime.build_index("bulk", config, records, BulkBuildOptions::new(4 << 30).unwrap(), OperationOptions::default().with_deadline(deadline)).await.map_err(|e| super::error_at("build", e))?;
        let after = ResourceSnapshot::capture()?;
        eprintln!("audited publication completed: wall_seconds={:.3}, client_cpu_seconds={:.3}, peak_rss_bytes={}", started.elapsed().as_secs_f64(), after.cpu_seconds_since(before), after.peak_rss_bytes());
        let reopened = runtime.open_index("bulk").await.map_err(|e| super::error_at("reopen", e))?;
        if reopened.logical_index_id() != index.logical_index_id() { return Err("reopened a different logical index".to_owned()); }
        for position in [0, 500_000, 999_999] {
            if reopened.get_with_control(data.ids[position].clone(), GetOptions::default(), OperationOptions::default().with_deadline(deadline)).await.map_err(|e| super::error_at("get", e))?.is_none() {
                return Err(format!("missing sample record at {position}"));
            }
        }
        let options = SearchOptions::default().with_leaf_beam_size(132).unwrap();
        let mut matched = 0;
        for (query, expected) in data.queries.iter().zip(&truth).take(1000) {
            let expected: HashSet<_> = expected.iter().take(100).collect();
            if expected.len() != 100 { return Err("invalid ground-truth neighbor count".to_owned()); }
            let request = SearchRequest::new(Arc::clone(query), 100).unwrap().with_options(options);
            let outcome = reopened.search_with_control(request, OperationOptions::default().with_deadline(deadline)).await.map_err(|e| super::error_at("search", e))?;
            if outcome.hits.len() != 100 { return Err("short search result".to_owned()); }
            let returned: HashSet<_> = outcome.hits.iter().map(|hit| hit.id()).collect();
            if returned.len() != 100 { return Err("duplicate search hits".to_owned()); }
            matched += returned.intersection(&expected).count();
        }
        let recall = matched as f64 / 100_000.0;
        eprintln!("published FDB search: queries=1000, k=100, beam=132, recall={recall:.5}");
        if recall < 0.9 { return Err(format!("recall below target: {recall}")); }
        runtime.drop_index("bulk").await.map_err(|e| super::error_at("drop", e))?;
        Ok(())
    }.await;
    // Clear only this test's unique namespace even when the operation fails.
    runtime.shutdown().await.unwrap();
    support::clear_test_keys(&cleanup).await;
    result.expect("full-corpus audited publication and queries");
}
