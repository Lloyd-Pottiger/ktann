//! Offline refinement preserves ordinary index integrity across partial failure.

use ktann::api::{ErrorKind, OperationOptions, RefineOptions};
use ktann::storage::{
    ReadLogicalTxn, WriteLogicalTxn,
    backend::Backend,
    keys::LogicalKey,
    values::{PartitionHeader, PartitionState, PersistentValue},
};
use ktann_memory::{
    MemoryBackend,
    test_support::{CommitFault, TestConfig},
};
use tokio_util::sync::CancellationToken;

#[path = "support/refinement_contract.rs"]
mod contract;
#[path = "support/refinement.rs"]
mod fixture;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refines_membership_and_centroids_preserving_partition_topology() {
    for graph in [false, true] {
        let backend = MemoryBackend::new();
        contract::run(backend.clone(), backend, graph).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_commits_leave_a_valid_index_without_replay() {
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        for committed_prefix in 0..3 {
            let backend = MemoryBackend::with_test_config(TestConfig::default());
            let (runtime, index, manifest) =
                contract::setup(&backend, backend.clone(), false).await;
            let mut faults = vec![CommitFault::Normal; committed_prefix];
            faults.push(fault);
            backend.set_fault_plan(faults).unwrap();
            assert_eq!(
                index
                    .refine(RefineOptions::new(1 << 20).unwrap())
                    .await
                    .unwrap_err()
                    .kind(),
                ErrorKind::CommitOutcomeUnknown
            );
            contract::verify(&index).await;
            // An ambiguous centroid commit still changes both projections or neither.
            let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
            for leaf in [2, 3] {
                let Some(PersistentValue::PartitionCentroid(center)) = txn
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
                let Some(PersistentValue::ChildEntry(edge)) = txn
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
                assert_eq!(center.components(), edge.centroid());
            }
            drop(txn);
            runtime.drop_index(index.name().as_str()).await.unwrap();
            runtime.shutdown().await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn input_limits_and_cancellation_do_not_modify_the_index() {
    let backend = MemoryBackend::with_test_config(TestConfig::default());
    let (runtime, index, _) = contract::setup(&backend, backend.clone(), false).await;
    let history = backend.history().len();
    assert_eq!(
        index
            .refine(RefineOptions::new(1).unwrap())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::LimitExceeded
    );
    let token = CancellationToken::new();
    token.cancel();
    assert_eq!(
        index
            .refine(
                RefineOptions::new(1 << 20)
                    .unwrap()
                    .with_operation_options(OperationOptions::default().with_cancellation(token))
            )
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(history, backend.history().len());
    // An explicit token must remain usable across planning and persistence.
    let token = CancellationToken::new();
    backend.set_fault_plan(vec![CommitFault::Abort]).unwrap();
    index
        .refine(
            RefineOptions::new(1 << 20).unwrap().with_operation_options(
                OperationOptions::default().with_cancellation(token.clone()),
            ),
        )
        .await
        .unwrap();
    assert!(!token.is_cancelled());
    contract::verify(&index).await;
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejects_unsettled_topology_before_writing() {
    let backend = MemoryBackend::with_test_config(TestConfig::default());
    let (runtime, index, manifest) = contract::setup(&backend, backend.clone(), false).await;
    let key = LogicalKey::Header {
        index: index.logical_index_id(),
        tree_key: fixture::tree_key(),
        partition: fixture::partition(2),
    };
    let mut txn = WriteLogicalTxn::for_index(
        backend.begin_write().await.unwrap(),
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    let Some(PersistentValue::PartitionHeader(header)) = txn.get(key.clone()).await.unwrap() else {
        panic!("header");
    };
    txn.put(
        key.clone(),
        PersistentValue::PartitionHeader(
            PartitionHeader::new(
                header.level(),
                header.entry_count(),
                header.cache_epoch(),
                PartitionState::Splitting,
            )
            .unwrap(),
        ),
    )
    .await
    .unwrap();
    txn.commit().await.unwrap();
    let history = backend.history().len();
    assert_eq!(
        index
            .refine(RefineOptions::new(1 << 20).unwrap())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(history, backend.history().len());
    let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
    assert!(
        matches!(txn.get(key).await.unwrap(), Some(PersistentValue::PartitionHeader(header)) if header.state() == PartitionState::Splitting)
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zero_rounds_refreshes_stale_centroids_without_moving_records() {
    let backend = MemoryBackend::new();
    let (runtime, index, manifest) = contract::setup(&backend, backend.clone(), false).await;
    let centroid_key = LogicalKey::Centroid {
        index: index.logical_index_id(),
        tree_key: fixture::tree_key(),
        partition: fixture::partition(2),
    };
    let edge_key = LogicalKey::ChildEntry {
        index: index.logical_index_id(),
        tree_key: fixture::tree_key(),
        partition: fixture::partition(1),
        child: fixture::partition(2),
    };
    index
        .refine(
            RefineOptions::new(1 << 20)
                .unwrap()
                .with_refinement_rounds(0)
                .unwrap(),
        )
        .await
        .unwrap();
    let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
    let Some(PersistentValue::PartitionCentroid(center)) = txn.get(centroid_key).await.unwrap()
    else {
        panic!("centroid");
    };
    let Some(PersistentValue::ChildEntry(edge)) = txn.get(edge_key).await.unwrap() else {
        panic!("edge");
    };
    assert_eq!(center.components(), [-1.25]);
    assert_eq!(center.components(), edge.centroid());
    for position in 0..8 {
        let Some(PersistentValue::RecordLocation(location)) = txn
            .get(LogicalKey::Location {
                index: index.logical_index_id(),
                id: bytes::Bytes::from(vec![b'r', position]),
            })
            .await
            .unwrap()
        else {
            panic!("location");
        };
        assert_eq!(
            location.leaf(),
            fixture::partition(if position < 4 { 2 } else { 3 })
        );
    }
    contract::verify(&index).await;
    runtime.shutdown().await.unwrap();
}

/// A graph update and its new projections must have the same commit outcome.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_centroid_commits_preserve_snapshots_and_do_not_replay_unknowns() {
    use ktann::api::{RoutingMode, SearchOptions, SearchRequest};
    use ktann::storage::graph_transitions::GraphTransitionSource;
    for fault in [
        CommitFault::Normal,
        CommitFault::Abort,
        CommitFault::UnknownApplied,
        CommitFault::UnknownNotApplied,
    ] {
        let backend = MemoryBackend::new();
        let (runtime, index, manifest) = contract::setup(&backend, backend.clone(), true).await;
        let query = SearchRequest::new(vec![4.0], 2).unwrap().with_options(
            SearchOptions::default()
                .with_routing_mode(RoutingMode::PartitionLocalHnsw { ef_search: 64 })
                .unwrap(),
        );
        index.search(query.clone()).await.unwrap();
        let mut snapshot =
            ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
        let before = GraphTransitionSource::read(
            &mut snapshot,
            &fixture::tree_key(),
            fixture::partition(1),
            64 << 20,
            &|| Ok(()),
        )
        .await
        .unwrap();
        let old_header = before.header().unwrap();
        let history = backend.history().len();
        backend.set_fault_plan(vec![fault]).unwrap();
        let result = index
            .refine(
                RefineOptions::new(1 << 20)
                    .unwrap()
                    .with_refinement_rounds(0)
                    .unwrap(),
            )
            .await;
        if matches!(
            fault,
            CommitFault::UnknownApplied | CommitFault::UnknownNotApplied
        ) {
            assert_eq!(result.unwrap_err().kind(), ErrorKind::CommitOutcomeUnknown);
            assert_eq!(
                backend.history().len() - history,
                1,
                "unknown outcomes are not replayed"
            );
        } else {
            result.unwrap();
        }
        contract::verify(&index).await;
        index.search(query).await.unwrap();
        // The pre-refinement snapshot retains a valid old graph after publication.
        let old = GraphTransitionSource::read(
            &mut snapshot,
            &fixture::tree_key(),
            fixture::partition(1),
            64 << 20,
            &|| Ok(()),
        )
        .await
        .unwrap();
        assert_eq!(old.header(), Some(old_header));
        let children = old.children().to_vec();
        old.prepare(Some(old_header), children, 64 << 20, &|| Ok(()))
            .unwrap();
        let mut current = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
        let graph = GraphTransitionSource::read(
            &mut current,
            &fixture::tree_key(),
            fixture::partition(1),
            64 << 20,
            &|| Ok(()),
        )
        .await
        .unwrap();
        let applied = fault != CommitFault::UnknownNotApplied;
        assert_eq!(
            graph.header().unwrap().cache_epoch(),
            old_header.cache_epoch() + if applied { 2 } else { 0 }
        );
        assert_eq!(
            graph.children()[0].centroid(),
            [if applied { -1.25 } else { -9.0 }]
        );
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_refinement_splits_oversized_transactions_and_rejects_damaged_graphs() {
    use ktann_memory::test_support::WriteFaultPoint;
    let backend = MemoryBackend::new();
    let (runtime, index, manifest) = contract::setup(&backend, backend.clone(), true).await;
    let history = backend.history().len();
    backend
        .set_write_fault_plan(
            WriteFaultPoint::BeginWrite,
            vec![ErrorKind::TransactionTooLarge],
        )
        .unwrap();
    let options = RefineOptions::new(1 << 20)
        .unwrap()
        .with_refinement_rounds(0)
        .unwrap();
    index.refine(options.clone()).await.unwrap();
    assert_eq!(
        backend.history().len() - history,
        2,
        "each admitted sibling batch commits one matching graph"
    );
    contract::verify(&index).await;
    let mut txn = WriteLogicalTxn::for_index(
        backend.begin_write().await.unwrap(),
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    txn.delete(LogicalKey::GraphChunk {
        index: index.logical_index_id(),
        tree_key: fixture::tree_key(),
        partition: fixture::partition(1),
        chunk: 0,
    })
    .await
    .unwrap();
    txn.commit().await.unwrap();
    let history = backend.history().len();
    assert_eq!(
        index.refine(options).await.unwrap_err().kind(),
        ErrorKind::Corruption
    );
    assert_eq!(
        backend.history().len(),
        history,
        "unchanged centroids must not hide or repair graph corruption"
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_refinement_respects_the_runtime_preparation_budget() {
    use ktann::api::RuntimeConfig;
    use ktann::runtime::Runtime;
    let backend = MemoryBackend::new();
    let (runtime, _, _) = contract::setup(&backend, backend.clone(), true).await;
    runtime.shutdown().await.unwrap();
    let bounded = Runtime::new(
        backend.clone(),
        RuntimeConfig::default()
            .with_maintenance(0, 1)
            .unwrap()
            .with_routing_working_bytes(1)
            .unwrap(),
    )
    .unwrap();
    let index = bounded.open_index("offline-refinement").await.unwrap();
    let history = backend.history().len();
    assert_eq!(
        index
            .refine(
                RefineOptions::new(1 << 20)
                    .unwrap()
                    .with_refinement_rounds(0)
                    .unwrap()
            )
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::LimitExceeded
    );
    assert_eq!(backend.history().len(), history);
    bounded.shutdown().await.unwrap();
}
