//! Real FoundationDB coverage for the unpublished bulk-construction lifecycle.
use bytes::Bytes;
use foundationdb::Database;
use ktann::api::{
    BulkBuildOptions, ErrorKind, GetOptions, IndexConfig, Metric, OperationOptions, Record,
    RuntimeConfig, SearchRequest, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::WriteLogicalTxn;
use ktann::storage::backend::{Backend, ScanLimits};
use ktann::storage::keys::{self, LogicalKey};
use ktann::storage::values::{IndexLifecycle, PersistentValue};
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};
mod support;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_bulk_publication_mutation_and_interrupted_cleanup() {
    let _network = support::boot_foundationdb();
    let cluster = std::env::var("FDB_CLUSTER_FILE").ok();
    let backend = || {
        FoundationDbBackend::new(
            Database::new(cluster.as_deref()).unwrap(),
            BackendNamespace::new("ktann-capacity-refinement-native-contract").unwrap(),
        )
    };
    let cleanup = backend();
    support::clear_test_keys(&cleanup).await;
    let runtime = Runtime::new(
        backend(),
        RuntimeConfig::default()
            .with_maintenance(0, 1)
            .unwrap()
            .with_import_limits(1, 1)
            .unwrap(),
    )
    .unwrap();
    let config = IndexConfig::new(128, Metric::Cosine)
        .unwrap()
        .with_partition_entries(8, 32)
        .unwrap();
    let records: Vec<_> = (0_usize..1025)
        .map(|position| {
            let vector: Vec<_> = (0..128)
                .map(|axis| ((position * 17 + axis * 11) % 101) as f32 - 50.0)
                .collect();
            Record::new(Bytes::from(position.to_be_bytes().to_vec()), vector, vec![]).unwrap()
        })
        .collect();
    let index = runtime
        .build_index(
            "bulk",
            config.clone(),
            records,
            BulkBuildOptions::new(1 << 20).unwrap(),
            OperationOptions::default(),
        )
        .await
        .unwrap();
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty(), "{report:?}");
    assert!(
        report
            .topology
            .max_entries_by_level
            .values()
            .all(|count| *count <= 32)
    );
    assert_eq!(report.objects.vector_records, 1025);
    let result = index
        .search(SearchRequest::new(vec![1.0; 128], 10).unwrap())
        .await
        .unwrap();
    assert!(!result.hits.is_empty());
    let id = Bytes::from(0_usize.to_be_bytes().to_vec());
    index
        .upsert(Record::new(id.clone(), vec![1.0; 128], vec![]).unwrap())
        .await
        .unwrap();
    assert_eq!(
        index
            .get(id.clone(), GetOptions::default())
            .await
            .unwrap()
            .unwrap()
            .vector(),
        &[1.0; 128]
    );
    index.delete(id).await.unwrap();
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    let index_id = index.logical_index_id();
    runtime.shutdown().await.unwrap();

    // Install the exact durable state an interrupted pre-publication builder
    // leaves, with complete staged membership. No public handle is alive.
    let mut txn = WriteLogicalTxn::bootstrap(
        cleanup.begin_write().await.unwrap(),
        cleanup.hard_limits(),
        cleanup.admission_budget(),
    );
    let Some(PersistentValue::IndexManifest(manifest)) = txn
        .get_for_update(LogicalKey::Manifest(index_id))
        .await
        .unwrap()
    else {
        panic!("published manifest");
    };
    txn.put(
        LogicalKey::Manifest(index_id),
        PersistentValue::IndexManifest(
            manifest.with_lifecycle(IndexLifecycle::Building { owner: [7; 16] }),
        ),
    )
    .await
    .unwrap();
    txn.commit().await.unwrap();
    let restarted = Runtime::new(
        backend(),
        RuntimeConfig::default()
            .with_maintenance(0, 1)
            .unwrap()
            .with_import_limits(1, 1)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        restarted.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexBuilding
    );
    assert_eq!(
        restarted
            .create_index("bulk", config)
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::IndexBuilding
    );
    restarted.drop_index("bulk").await.unwrap();
    assert_eq!(
        restarted.open_index("bulk").await.unwrap_err().kind(),
        ErrorKind::IndexNotFound
    );
    restarted.shutdown().await.unwrap();
    let mut raw = cleanup.begin_read().await.unwrap();
    use ktann::storage::backend::ReadOps;
    assert!(
        raw.scan(
            &keys::index_range(index_id),
            ScanLimits {
                item_limit: 1,
                byte_limit: 4096
            }
        )
        .await
        .unwrap()
        .items()
        .is_empty()
    );
    drop(raw);
    support::clear_test_keys(&cleanup).await;
}
