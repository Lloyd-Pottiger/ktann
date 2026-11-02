//! Offline graph-index integrity, independent of candidate traversal and cache.
use super::*;
use ktann::api::{PartitionKey, VerifyIssueKind};
use ktann::storage::backend::{ReadOps, ScanLimits};
use ktann::storage::values::{GraphChunk, PartitionCentroid};

async fn replace(memory: &MemoryBackend, key: Bytes, value: Option<Bytes>) {
    let mut txn = memory.begin_write().await.unwrap();
    match value {
        Some(value) => txn.put(key, value).await.unwrap(),
        None => txn.delete(key).await.unwrap(),
    }
    txn.commit().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_graph_partition_is_audited_even_after_warming_search() {
    let memory = MemoryBackend::new();
    let runtime = build(memory.clone(), Metric::L2, false, 73).await;
    let index = runtime.open_index("routing").await.unwrap();
    index.search(request(HNSW, 128, false)).await.unwrap();
    let mut txn = ReadLogicalTxn::bootstrap(memory.begin_read().await.unwrap());
    let Some(PersistentValue::IndexManifest(manifest)) = txn
        .get(LogicalKey::Manifest(index.logical_index_id()))
        .await
        .unwrap()
    else {
        panic!()
    };
    let codec = ValueCodec::for_index(&manifest);
    let tree = keys::TreeKey::encode(&[], &[]).unwrap();
    let mut raw = memory.begin_read().await.unwrap();
    let page = raw
        .scan(
            &keys::index_range(index.logical_index_id()),
            ScanLimits {
                item_limit: 10_000,
                byte_limit: 16 << 20,
            },
        )
        .await
        .unwrap();
    assert!(page.next_start().is_none());
    let mut descriptors = Vec::new();
    let mut chunks = Vec::new();
    let mut centroids = Vec::new();
    let mut children = Vec::new();
    let mut leaf = None;
    for item in page.items() {
        let key = keys::decode_key(&[], item.key()).unwrap();
        match key {
            LogicalKey::GraphDescriptor { partition, .. } => {
                descriptors.push((partition, item.clone()))
            }
            LogicalKey::GraphChunk { partition, .. } => chunks.push((partition, item.clone())),
            LogicalKey::ChildEntry { .. } => children.push(item.clone()),
            LogicalKey::Centroid { partition, .. } => centroids.push((partition, item.clone())),
            LogicalKey::Header { partition, .. } => {
                if matches!(codec.decode(&key, item.value().clone()).unwrap(),
                    PersistentValue::PartitionHeader(h) if h.level() == 1)
                {
                    leaf = Some(partition);
                }
            }
            _ => {}
        }
    }
    drop(raw);
    assert!(descriptors.len() > 2, "exercise more than the root graph");
    // Delete every graph in turn, including parents a narrow search can omit.
    // A warm routing cache must not hide any missing persistent descriptor.
    for (partition, item) in &descriptors {
        replace(&memory, item.key().clone(), None).await;
        let report = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(report.complete);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == VerifyIssueKind::Reachability
                    && issue.partition_key == *partition),
            "{report:?}"
        );
        replace(&memory, item.key().clone(), Some(item.value().clone())).await;
    }
    // Missing and extra chunks, wrong-parent envelopes, and corrupt topology.
    let (_, chunk) = &chunks[0];
    let (_, descriptor) = &descriptors[0];
    let mut corrupt = chunk.value().to_vec();
    *corrupt.last_mut().unwrap() ^= 1;
    let extra = Bytes::from(keys::graph_chunk_key(
        index.logical_index_id(),
        &tree,
        descriptors[0].0,
        99,
    ));
    let leaf_graph = Bytes::from(keys::graph_descriptor_key(
        index.logical_index_id(),
        &tree,
        leaf.unwrap(),
    ));
    let orphan_graph = Bytes::from(keys::graph_descriptor_key(
        index.logical_index_id(),
        &tree,
        PartitionKey::new(1_000_000).unwrap(),
    ));
    let cases = [
        (chunk.key().clone(), None),
        (chunk.key().clone(), Some(Bytes::from(corrupt))),
        (
            descriptor.key().clone(),
            Some(descriptors[1].1.value().clone()),
        ),
        (
            extra,
            Some(Bytes::from(
                codec
                    .encode(&PersistentValue::GraphChunk(
                        GraphChunk::new(Bytes::from_static(b"x")).unwrap(),
                    ))
                    .unwrap(),
            )),
        ),
        (leaf_graph, Some(descriptor.value().clone())),
        (orphan_graph, Some(descriptor.value().clone())),
    ];
    for (key, value) in cases {
        let original = memory
            .begin_read()
            .await
            .unwrap()
            .get(key.clone())
            .await
            .unwrap();
        replace(&memory, key.clone(), value).await;
        let report = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(report.complete && !report.issues.is_empty(), "{report:?}");
        replace(&memory, key, original).await;
    }
    let child = &children[0];
    let key = keys::decode_key(&[], child.key()).unwrap();
    let PersistentValue::ChildEntry(entry) = codec.decode(&key, child.value().clone()).unwrap()
    else {
        panic!()
    };
    let changed = codec
        .encode(&PersistentValue::ChildEntry(
            ktann::storage::values::ChildEntry::new(entry.child(), vec![900.0; 3]),
        ))
        .unwrap();
    replace(&memory, child.key().clone(), Some(Bytes::from(changed))).await;
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        report.complete
            && report
                .issues
                .iter()
                .any(|i| i.kind == VerifyIssueKind::InvalidEncoding)
    );
    replace(&memory, child.key().clone(), Some(child.value().clone())).await;
    // The graph still matches its Child Entries, but those projections no
    // longer match a child's authoritative centroid. Native decoding alone
    // cannot detect this cross-partition inconsistency.
    let (_, centroid) = centroids.iter().find(|(p, _)| p.get() != 1).unwrap();
    let changed = codec
        .encode(&PersistentValue::PartitionCentroid(PartitionCentroid::new(
            vec![900.0; 3],
        )))
        .unwrap();
    replace(&memory, centroid.key().clone(), Some(Bytes::from(changed))).await;
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        report.complete
            && report
                .issues
                .iter()
                .any(|i| i.kind == VerifyIssueKind::Membership)
    );
    replace(
        &memory,
        centroid.key().clone(),
        Some(centroid.value().clone()),
    )
    .await;
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty(), "{report:?}");
    // A partial graph or tree must never be finalized into invented issues.
    for bytes in [1, 16 * 1024, 128 * 1024] {
        let bounded = index
            .verify(
                VerifyOptions::default()
                    .with_memory_limit_bytes(bytes)
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(bounded.issues.is_empty(), "{bounded:?}");
        if bytes == 1 {
            assert!(!bounded.complete);
        }
    }
    let bounded = index
        .verify(
            VerifyOptions::default()
                .with_object_limit(report.objects.total - 1)
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(!bounded.complete && bounded.issues.is_empty());
    for (_, item) in descriptors.iter().take(2) {
        replace(&memory, item.key().clone(), None).await;
    }
    let bounded = index
        .verify(VerifyOptions::default().with_issue_limit(1).unwrap())
        .await
        .unwrap();
    assert!(!bounded.complete && bounded.issues.len() == 1);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_verify_is_snapshot_bound_and_read_only() {
    let memory = MemoryBackend::new();
    let builder = build(memory.clone(), Metric::L2, false, 73).await;
    let delete = std::sync::Arc::new(std::sync::Mutex::new(None));
    let runtime = Runtime::new(
        SnapshotCut {
            memory: memory.clone(),
            delete: delete.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let index = runtime.open_index("routing").await.unwrap();
    let tree = keys::TreeKey::encode(&[], &[]).unwrap();
    let key = Bytes::from(keys::graph_chunk_key(
        index.logical_index_id(),
        &tree,
        PartitionKey::new(1).unwrap(),
        0,
    ));
    *delete.lock().unwrap() = Some(key.clone());
    let old = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(old.complete && old.issues.is_empty(), "{old:?}");
    let current = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(current.complete && !current.issues.is_empty());
    assert_eq!(
        memory.begin_read().await.unwrap().get(key).await.unwrap(),
        None
    );
    runtime.shutdown().await.unwrap();
    builder.shutdown().await.unwrap();
}
