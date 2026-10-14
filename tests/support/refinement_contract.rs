//! Public offline-refinement contract shared by Memory and FoundationDB.

use ktann::api::{
    GetOptions, Index, IndexConfig, Metric, PayloadProjection, RefineOptions, RuntimeConfig,
    SearchOptions, SearchRequest, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::{
    ReadLogicalTxn,
    backend::Backend,
    keys::LogicalKey,
    values::{IndexManifest, PersistentValue},
};

use super::fixture;

pub async fn setup<B: Backend>(
    backend: &B,
    runtime_backend: B,
) -> (Runtime<B>, Index<B>, IndexManifest) {
    let runtime = Runtime::new(
        runtime_backend,
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let index = runtime
        .create_index(
            "offline-refinement",
            IndexConfig::new(1, Metric::L2)
                .unwrap()
                .with_partition_entries(2, 8)
                .unwrap(),
        )
        .await
        .unwrap();
    let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await.unwrap());
    let Some(PersistentValue::IndexManifest(manifest)) = txn
        .get(LogicalKey::Manifest(index.logical_index_id()))
        .await
        .unwrap()
    else {
        panic!("manifest");
    };
    fixture::seed(&index, backend, &manifest).await.unwrap();
    (runtime, index, manifest)
}

pub async fn verify<B: Backend>(index: &Index<B>) {
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty(), "{report:?}");
}

pub async fn run<B: Backend>(backend: B, runtime_backend: B) {
    let (runtime, index, manifest) = setup(&backend, runtime_backend).await;
    verify(&index).await;
    // Warm both cached leaf bodies and the parent body before changing either.
    let query = SearchRequest::new(vec![4.0_f32], 2)
        .unwrap()
        .with_options(SearchOptions::default().with_leaf_beam_size(1).unwrap());
    assert_eq!(index.search(query.clone()).await.unwrap().hits.len(), 2);
    let mut before = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
    let root = LogicalKey::Header {
        index: index.logical_index_id(),
        tree_key: fixture::tree_key(),
        partition: fixture::partition(1),
    };
    let Some(PersistentValue::PartitionHeader(old_header)) =
        before.get(root.clone()).await.unwrap()
    else {
        panic!("header");
    };
    index
        .refine(RefineOptions::new(1 << 20).unwrap())
        .await
        .unwrap();
    verify(&index).await;
    assert_eq!(index.logical_index_id(), manifest.logical_index_id());
    for (position, value) in [-4.0_f32, -3.0, -2.0, 4.0, 2.0, 3.0, 4.0, -4.0]
        .into_iter()
        .enumerate()
    {
        let id = bytes::Bytes::from(vec![b'r', position as u8]);
        let record = index
            .get(id.clone(), GetOptions::default().with_payload())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(record.vector(), [value]);
        assert_eq!(
            record.payload(),
            &PayloadProjection::Present(bytes::Bytes::from(vec![position as u8]))
        );
        let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
        let Some(PersistentValue::RecordLocation(location)) = txn
            .get(LogicalKey::Location {
                index: index.logical_index_id(),
                id,
            })
            .await
            .unwrap()
        else {
            panic!("location");
        };
        assert_eq!(
            location.leaf(),
            fixture::partition(if value < 0.0 { 2 } else { 3 })
        );
    }
    let mut after = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
    let Some(PersistentValue::PartitionHeader(new_header)) = after.get(root.clone()).await.unwrap()
    else {
        panic!("header");
    };
    assert_eq!(old_header.entry_count(), new_header.entry_count());
    assert!(new_header.cache_epoch() > old_header.cache_epoch());
    // Old snapshots retain old routing while new snapshots and warmed caches agree.
    assert!(
        matches!(before.get(root).await.unwrap(), Some(PersistentValue::PartitionHeader(header)) if header == old_header)
    );
    for leaf in [2, 3] {
        let Some(PersistentValue::PartitionCentroid(center)) = after
            .get(LogicalKey::Centroid {
                index: index.logical_index_id(),
                tree_key: fixture::tree_key(),
                partition: fixture::partition(leaf),
            })
            .await
            .unwrap()
        else {
            panic!("centroid");
        };
        let Some(PersistentValue::ChildEntry(child)) = after
            .get(LogicalKey::ChildEntry {
                index: index.logical_index_id(),
                tree_key: fixture::tree_key(),
                partition: fixture::partition(1),
                child: fixture::partition(leaf),
            })
            .await
            .unwrap()
        else {
            panic!("edge");
        };
        assert_eq!(center.components(), child.centroid());
        assert_eq!(center.components(), [if leaf == 2 { -3.25 } else { 3.25 }]);
    }
    let hits = index.search(query).await.unwrap().hits;
    assert_eq!(hits.len(), 2);
    assert!(hits.iter().all(|hit| hit.distance() == 0.0));
    drop(before);
    drop(after);
    runtime.drop_index("offline-refinement").await.unwrap();
    runtime.shutdown().await.unwrap();
}
