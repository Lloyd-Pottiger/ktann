//! Public mutable graph writes and real split/merge histories shared by adapters.
use bytes::Bytes;
use ktann::api::{
    ErrorKind, Index, IndexConfig, LocalGraphConfig, Metric, Record, RoutingMode, RuntimeConfig,
    SearchOptions, SearchRequest, VerifyOptions,
};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::ReadLogicalTxn;
use ktann::storage::backend::Backend;
use ktann::storage::keys::LogicalKey;
use ktann::storage::values::{PartitionState, PersistentValue};

#[allow(dead_code)]
#[path = "audit.rs"]
mod audit;
#[allow(dead_code)]
#[path = "oracle.rs"]
mod oracle;

fn record(id: u64, round: u64) -> Record {
    Record::new(
        Bytes::copy_from_slice(&id.to_be_bytes()),
        vec![id as f32 + round as f32 / 8.0, (id % 7) as f32],
        vec![],
    )
    .unwrap()
}

#[derive(Default, Debug)]
struct Coverage {
    internal_splits: usize,
    leaf_splits: usize,
    merges: usize,
    parent_changes: usize,
    max_level: u32,
}

async fn check<B: Backend>(index: &Index<B>) {
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        report.complete && report.issues.is_empty(),
        "{:?}",
        report.issues
    );
    // Repeat through the same handle to exercise cached graph/header epochs.
    for _ in 0..2 {
        let result = index
            .search(
                SearchRequest::new(vec![31.25, 3.0], 8)
                    .unwrap()
                    .with_options(
                        SearchOptions::default()
                            .with_routing_mode(RoutingMode::PartitionLocalHnsw { ef_search: 64 })
                            .unwrap(),
                    ),
            )
            .await
            .unwrap();
        assert!(
            result
                .hits
                .windows(2)
                .all(|p| p[0].distance() <= p[1].distance())
        );
    }
}

/// Advances the actual topology machines, restarting discovery after each
/// committed step. Faults are armed at a known maintenance commit boundary.
async fn settle<B: Backend>(
    backend: &B,
    index: &Index<B>,
    fault: &dyn Fn(&'static str, usize),
    clock: &mut u64,
    executor: &ktann::runtime::GraphMaintenance,
    coverage: &mut Coverage,
) -> usize {
    let mut read = ReadLogicalTxn::bootstrap(backend.begin_read().await.unwrap());
    let Some(PersistentValue::IndexManifest(manifest)) = read
        .get(LogicalKey::Manifest(index.logical_index_id()))
        .await
        .unwrap()
    else {
        panic!("missing manifest")
    };
    drop(read);
    let retry = RetryPolicy::for_fixup(&RuntimeConfig::default());
    let mut committed = 0;
    let mut stalled = std::collections::BTreeSet::new();
    for attempt in 0..4096 {
        let partitions = audit::list_partitions(backend, index.logical_index_id())
            .await
            .unwrap();
        let candidate = partitions
            .iter()
            .filter(|(tree, partition, _)| !stalled.contains(&(tree.clone(), *partition)))
            .filter(|(_, partition, header)| {
                matches!(
                    header.state(),
                    PartitionState::Splitting
                        | PartitionState::DrainingSplit
                        | PartitionState::Merging
                ) || (header.state() == PartitionState::Ready
                    && (header.entry_count()
                        > index.config().max_partition_entries(header.level())
                        || (partition.get() != 1
                            && header.entry_count()
                                < index.config().min_partition_entries(header.level()))))
            })
            .min_by_key(|(_, _, header)| {
                if matches!(
                    header.state(),
                    PartitionState::Splitting
                        | PartitionState::DrainingSplit
                        | PartitionState::Merging
                ) {
                    0
                } else if header.entry_count()
                    > index.config().max_partition_entries(header.level())
                {
                    1
                } else {
                    2
                }
            });
        let Some((tree, partition, _header)) = candidate else {
            return committed;
        };
        *clock += 100;
        fault("topology", attempt);
        let result = ktann::maintenance::graph_topology::advance(
            backend, &manifest, tree, *partition, *clock, &retry, executor,
        )
        .await;
        fault("topology-finished", attempt);
        let retry_outcome = result.is_err();
        match result {
            Ok(()) => {}
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                ) => {}
            Err(error) => panic!("topology attempt {attempt}: {error:?}"),
        }
        check(index).await;
        if retry_outcome {
            continue;
        }
        let after = audit::list_partitions(backend, index.logical_index_id())
            .await
            .unwrap();
        coverage.max_level = coverage
            .max_level
            .max(after.iter().map(|(_, _, h)| h.level()).max().unwrap_or(0));
        for (old_tree, old_pk, old_header) in &partitions {
            let next = after
                .iter()
                .find(|(tree, pk, _)| tree == old_tree && pk == old_pk);
            if old_header.state() == PartitionState::DrainingSplit
                && next.is_none_or(|(_, _, h)| h.state() == PartitionState::Ready)
            {
                if old_header.level() == 1 {
                    coverage.leaf_splits += 1;
                } else {
                    coverage.internal_splits += 1;
                }
            }
            if old_header.state() == PartitionState::Merging && next.is_none() {
                coverage.merges += 1;
            }
            if old_header.level() > 1
                && next.is_some_and(|(_, _, h)| h.entry_count() != old_header.entry_count())
            {
                coverage.parent_changes += 1;
            }
        }
        if after == partitions {
            // An exposed child can wait for its overfull parent to split;
            // try every independent candidate before declaring a fixed point.
            stalled.insert((tree.clone(), *partition));
        } else {
            committed += 1;
            stalled.clear();
        }
    }
    panic!("topology failed to converge")
}

/// Three grow/update/shrink cycles, recovery through a new Runtime, and a
/// concurrent reader observing every committed intermediate topology state.
pub async fn exercise<B: Backend>(backend: impl Fn() -> B, fault: &dyn Fn(&'static str, usize)) {
    let raw = backend();
    let runtime_config = RuntimeConfig::default().with_maintenance(0, 1).unwrap();
    let mut runtime = Runtime::new(backend(), runtime_config.clone()).unwrap();
    let config = IndexConfig::new(2, Metric::L2)
        .unwrap()
        .with_leaf_partition_entries(3, 6)
        .unwrap()
        .with_internal_partition_entries(3, 6)
        .unwrap()
        .with_mutable_graph(LocalGraphConfig::new(4, 16, [19; 32]).unwrap());
    let mut index = runtime
        .create_index("mutable-graph-history", config)
        .await
        .unwrap();
    let executor = ktann::runtime::GraphMaintenance::new(
        &runtime_config,
        tokio_util::sync::CancellationToken::new(),
    );
    let mut clock = 1000;
    let mut transitions = 0;
    let mut coverage = Coverage::default();
    for round in 0..3 {
        for id in 0..96 {
            if let Err(error) = index.upsert(record(id, round)).await {
                eprintln!(
                    "{}",
                    audit::render_tree(&raw, index.logical_index_id(), true, None)
                        .await
                        .unwrap()
                );
                panic!("round {round}, upsert {id}: {error:?}");
            }
        }
        transitions += settle(&raw, &index, fault, &mut clock, &executor, &mut coverage).await;
        check(&index).await;
        // A fresh handle forces body reload while the previous handle has cached
        // earlier parent epochs. Both must agree after updates and maintenance.
        let old = index.clone();
        runtime.shutdown().await.unwrap();
        runtime = Runtime::new(backend(), runtime_config.clone()).unwrap();
        index = runtime.open_index("mutable-graph-history").await.unwrap();
        check(&index).await;
        drop(old);
        let reader_index = index.clone();
        let reader = tokio::spawn(async move {
            for _ in 0..24 {
                let result = reader_index
                    .search(
                        SearchRequest::new(vec![31.0, 3.0], 8)
                            .unwrap()
                            .with_options(
                                SearchOptions::default()
                                    .with_routing_mode(RoutingMode::PartitionLocalHnsw {
                                        ef_search: 64,
                                    })
                                    .unwrap(),
                            ),
                    )
                    .await;
                assert!(result.is_ok(), "concurrent graph search: {result:?}");
                tokio::task::yield_now().await;
            }
        });
        for id in 12_u64..96 {
            index
                .delete(Bytes::copy_from_slice(&id.to_be_bytes()))
                .await
                .unwrap();
        }
        transitions += settle(&raw, &index, fault, &mut clock, &executor, &mut coverage).await;
        reader.await.unwrap();
        check(&index).await;
        let report = index.verify(VerifyOptions::default()).await.unwrap();
        assert_eq!(report.objects.vector_records, 12);
    }
    assert!(
        transitions > 0,
        "history must execute actual topology maintenance"
    );
    eprintln!("mutable graph topology coverage: {coverage:?}");
    assert!(
        coverage.max_level >= 3
            && coverage.internal_splits > 0
            && coverage.leaf_splits > 0
            && coverage.merges > 0
            && coverage.parent_changes > 0,
        "{coverage:?}"
    );
    runtime.drop_index("mutable-graph-history").await.unwrap();
    runtime.shutdown().await.unwrap();
}
