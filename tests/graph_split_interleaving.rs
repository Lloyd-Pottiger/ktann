//! Nested graph splits must preserve the receiving parent's drain headroom.

use bytes::Bytes;
use ktann::api::{
    IndexConfig, LocalGraphConfig, Metric, PartitionKey, Record, RuntimeConfig, VerifyOptions,
};
use ktann::maintenance::{graph_topology, training};
use ktann::runtime::{GraphMaintenance, RetryPolicy, Runtime};
use ktann::storage::keys::TreeKey;
use ktann::storage::topology::{self, SplitExposure};
use ktann::storage::values::{IndexManifest, PartitionState, PartitionTransition};
use support::builders::{pk, read_txn, write_txn};
use support::topology_probe::{header_of, scan_child_entries, state_of};
use support::{MemoryBackend, read_manifest};

#[allow(dead_code)]
mod support;

async fn advance(
    backend: &MemoryBackend,
    manifest: &IndexManifest,
    tree: &TreeKey,
    partition: PartitionKey,
    executor: &GraphMaintenance,
) {
    graph_topology::advance(
        backend,
        manifest,
        tree,
        partition,
        1000,
        &RetryPolicy::for_fixup(&RuntimeConfig::default()),
        executor,
    )
    .await
    .expect("advance graph topology");
}

async fn complete_split(
    backend: &MemoryBackend,
    manifest: &IndexManifest,
    tree: &TreeKey,
    partition: PartitionKey,
    root_level: u32,
    executor: &GraphMaintenance,
) {
    for _ in 0..32 {
        advance(backend, manifest, tree, partition, executor).await;
        let header = header_of(backend, manifest, tree, partition).await;
        if header.is_none()
            || (partition == pk(1)
                && header
                    .is_some_and(|h| h.state() == PartitionState::Ready && h.level() == root_level))
        {
            return;
        }
    }
    panic!("split did not complete");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn child_graph_split_waits_for_receiving_parent_to_finish() {
    let _audit = support::observe::audit_lock().await;
    let backend = MemoryBackend::default();
    let runtime_config = RuntimeConfig::default().with_maintenance(0, 1).unwrap();
    let runtime = Runtime::new(backend.clone(), runtime_config.clone()).unwrap();
    let index = runtime
        .create_index(
            "nested-graph-split",
            IndexConfig::new(1, Metric::L2)
                .unwrap()
                .with_leaf_partition_entries(1, 4)
                .unwrap()
                .with_internal_partition_entries(1, 4)
                .unwrap()
                .with_local_graph(LocalGraphConfig::new(2, 4, [13; 32]).unwrap()),
        )
        .await
        .unwrap();
    for id in 0_u64..48 {
        index
            .upsert(
                Record::new(
                    Bytes::copy_from_slice(&id.to_be_bytes()),
                    vec![id as f32],
                    vec![],
                )
                .unwrap(),
            )
            .await
            .unwrap();
    }
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let tree = TreeKey::encode(&[], &[]).unwrap();
    let executor =
        GraphMaintenance::new(&runtime_config, tokio_util::sync::CancellationToken::new());
    complete_split(&backend, &manifest, &tree, pk(1), 2, &executor).await;

    // Three leaf splits grow the level-2 root to five entries, while all
    // resulting leaves remain large enough to split again.
    for _ in 0..3 {
        let mut leaves = Vec::new();
        for entry in scan_child_entries(&backend, &manifest, &tree, pk(1)).await {
            let header = header_of(&backend, &manifest, &tree, entry.child())
                .await
                .unwrap();
            leaves.push((header.entry_count(), entry.child()));
        }
        let (_, leaf) = leaves.into_iter().max().unwrap();
        complete_split(&backend, &manifest, &tree, leaf, 0, &executor).await;
    }
    assert_eq!(
        header_of(&backend, &manifest, &tree, pk(1))
            .await
            .unwrap()
            .entry_count(),
        5
    );

    advance(&backend, &manifest, &tree, pk(1), &executor).await;
    advance(&backend, &manifest, &tree, pk(1), &executor).await;
    advance(&backend, &manifest, &tree, pk(1), &executor).await;
    let PartitionTransition::DrainingSplit { left, right, .. } =
        state_of(&backend, &manifest, &tree, pk(1)).await.unwrap()
    else {
        panic!("internal root must be draining");
    };
    let mut nested = None;
    for parent in [left, right] {
        let header = header_of(&backend, &manifest, &tree, parent).await.unwrap();
        assert_eq!(header.state(), PartitionState::ReceivingSplit);
        if header.entry_count() > 0 && header.entry_count() <= 4 {
            for entry in scan_child_entries(&backend, &manifest, &tree, parent).await {
                if header_of(&backend, &manifest, &tree, entry.child())
                    .await
                    .unwrap()
                    .entry_count()
                    > 4
                {
                    nested = Some((parent, entry.child(), header));
                    break;
                }
            }
        }
    }
    let (parent, child, parent_before) =
        nested.expect("receiving parent with room and an overfull child");
    advance(&backend, &manifest, &tree, child, &executor).await;
    let child_before = header_of(&backend, &manifest, &tree, child).await.unwrap();
    let PartitionTransition::Splitting { left, right, .. } =
        state_of(&backend, &manifest, &tree, child).await.unwrap()
    else {
        panic!("child must have reserved split targets");
    };
    let centroids =
        training::train_split_centroids(read_txn(&backend, &manifest).await, &tree, child)
            .await
            .unwrap();
    let mut txn = write_txn(&backend, &manifest).await;
    assert_eq!(
        topology::expose_split_targets(
            &mut txn,
            &tree,
            child,
            [(left, centroids.left()), (right, centroids.right())],
            1000,
        )
        .await
        .unwrap(),
        SplitExposure::ParentNotAccepting,
        "low-level exposure must preserve the receiving parent's headroom",
    );
    txn.rollback().await;

    advance(&backend, &manifest, &tree, child, &executor).await;
    assert_eq!(
        header_of(&backend, &manifest, &tree, child).await,
        Some(child_before)
    );
    assert_eq!(
        header_of(&backend, &manifest, &tree, parent).await,
        Some(parent_before)
    );
    for target in [left, right] {
        assert!(
            header_of(&backend, &manifest, &tree, target)
                .await
                .is_none()
        );
    }

    // Finishing the ancestor makes the receiver Ready, so the same pending
    // child split can now expose, drain, and complete without restarting.
    complete_split(&backend, &manifest, &tree, pk(1), 3, &executor).await;
    assert_eq!(
        header_of(&backend, &manifest, &tree, parent)
            .await
            .unwrap()
            .state(),
        PartitionState::Ready
    );
    complete_split(&backend, &manifest, &tree, child, 0, &executor).await;
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        report.complete && report.issues.is_empty(),
        "{:?}",
        report.issues
    );
    assert_eq!(report.objects.vector_records, 48);
    assert_eq!(report.objects.record_locations, 48);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sibling_splits_share_parent_preparation_without_stale_publication() {
    let _audit = support::observe::audit_lock().await;
    let capture = support::observe::capture();
    let backend = MemoryBackend::default();
    let config = RuntimeConfig::default().with_maintenance(2, 32).unwrap();
    let runtime = Runtime::new(backend.clone(), support::manual_maintenance_config()).unwrap();
    let index = runtime
        .create_index(
            "sibling-graph-splits",
            IndexConfig::new(1, Metric::L2)
                .unwrap()
                .with_leaf_partition_entries(1, 4)
                .unwrap()
                .with_internal_partition_entries(1, 16)
                .unwrap()
                .with_local_graph(LocalGraphConfig::new(2, 4, [17; 32]).unwrap()),
        )
        .await
        .unwrap();
    for id in 0_u64..32 {
        index
            .upsert(
                Record::new(
                    Bytes::copy_from_slice(&id.to_be_bytes()),
                    vec![id as f32],
                    vec![],
                )
                .unwrap(),
            )
            .await
            .unwrap();
    }
    let manifest = read_manifest(&backend, index.logical_index_id()).await;
    let tree = TreeKey::encode(&[], &[]).unwrap();
    let executor = GraphMaintenance::new(&config, tokio_util::sync::CancellationToken::new());
    complete_split(&backend, &manifest, &tree, pk(1), 2, &executor).await;
    let children = scan_child_entries(&backend, &manifest, &tree, pk(1)).await;
    assert_eq!(children.len(), 2);
    let (left, right) = (children[0].child(), children[1].child());
    for child in [left, right] {
        advance(&backend, &manifest, &tree, child, &executor).await;
    }
    let retries_before: u64 = capture
        .metric_counters()
        .into_iter()
        .filter(|(name, _, _)| name == "ktann.write.retries")
        .map(|(_, _, value)| value)
        .sum();
    // join polls the second exposure while the first awaits native preparation.
    // Both change the same parent; the waiting plan must read it after ownership.
    tokio::join!(
        complete_split(&backend, &manifest, &tree, left, 0, &executor),
        complete_split(&backend, &manifest, &tree, right, 0, &executor),
    );
    assert!(header_of(&backend, &manifest, &tree, left).await.is_none());
    assert!(header_of(&backend, &manifest, &tree, right).await.is_none());
    let retries: u64 = capture
        .metric_counters()
        .into_iter()
        .filter(|(name, _, _)| name == "ktann.write.retries")
        .map(|(_, _, value)| value)
        .sum();
    assert_eq!(
        retries - retries_before,
        0,
        "sibling preparation must not publish a stale parent"
    );
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        report.complete && report.issues.is_empty(),
        "{:?}",
        report.issues
    );
    assert_eq!(report.objects.vector_records, 32);
    assert_eq!(report.objects.record_locations, 32);
    runtime.shutdown().await.unwrap();
}
