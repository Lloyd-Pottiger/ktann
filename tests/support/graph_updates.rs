//! Shared ordinary graph publication corpus; backend fault controls are injected
//! by the existing Memory harness, while production adapters run the same history.
use ktann::api::{DataType, ErrorKind, Index, LogicalIndexId, PartitionKey, Value, VerifyOptions};
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::graph_updates::{GraphUpdateSource, PreparedGraphUpdate};
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionSynopsis, PartitionTransition, PersistentValue,
};
use ktann::storage::{ReadLogicalTxn, WriteLogicalTxn};

const MEMORY: usize = 64 * 1024 * 1024;

fn report(phase: &str, started: std::time::Instant) {
    let micros = started.elapsed().as_micros();
    if std::env::var_os("KTANN_GRAPH_UPDATE_TIMING").is_some() {
        eprintln!("graph_update_timing,{phase},{micros}");
    }
}

fn prepare(source: GraphUpdateSource, next: Vec<ChildEntry>, phase: &str) -> PreparedGraphUpdate {
    let started = std::time::Instant::now();
    let update = source.prepare(next, MEMORY, &|| Ok(())).unwrap().unwrap();
    report(phase, started);
    update
}

async fn query<B: Backend>(index: &Index<B>) -> Vec<(bytes::Bytes, f64)> {
    let result = index
        .search(
            ktann::api::SearchRequest::new(vec![0.0, 1.0], 8)
                .unwrap()
                .with_options(
                    ktann::api::SearchOptions::default()
                        .with_routing_mode(ktann::api::RoutingMode::PartitionLocalHnsw {
                            ef_search: 64,
                        })
                        .unwrap(),
                ),
        )
        .await
        .unwrap();
    result
        .hits
        .iter()
        .map(|hit| (hit.id().clone(), hit.distance()))
        .collect()
}

async fn writer<'a, B: Backend>(
    backend: &'a B,
    manifest: &'a IndexManifest,
) -> WriteLogicalTxn<'a, B::WriteTxn<'a>> {
    WriteLogicalTxn::for_index(
        backend.begin_write().await.unwrap(),
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
}

async fn commit<T: WriteTxn>(
    txn: WriteLogicalTxn<'_, T>,
    fault: &dyn Fn(&'static str, usize),
    phase: &'static str,
    attempt: usize,
    started: std::time::Instant,
) -> bool {
    fault(phase, attempt);
    let committed = match txn.commit().await {
        Ok(()) => true,
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
            ) =>
        {
            false
        }
        Err(error) => panic!("{phase}: {error:?}"),
    };
    report(phase, started);
    committed
}

async fn source<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree: &TreeKey,
) -> GraphUpdateSource {
    let started = std::time::Instant::now();
    let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), manifest);
    let source = GraphUpdateSource::read(
        &mut txn,
        tree,
        PartitionKey::new(1).unwrap(),
        MEMORY,
        &|| Ok(()),
    )
    .await
    .unwrap();
    report("source-read", started);
    source
}

pub async fn exercise<B: Backend>(
    backend: &B,
    index: &Index<B>,
    id: LogicalIndexId,
    fault: &dyn Fn(&'static str, usize),
) {
    let mut bootstrap = ReadLogicalTxn::bootstrap(backend.begin_read().await.unwrap());
    let Some(PersistentValue::IndexManifest(manifest)) =
        bootstrap.get(LogicalKey::Manifest(id)).await.unwrap()
    else {
        panic!("manifest")
    };
    drop(bootstrap);
    let tree = TreeKey::encode(&[DataType::I64], &[Value::I64(0)]).unwrap();
    let root = PartitionKey::new(1).unwrap();
    let initial = source(backend, &manifest, &tree).await;
    let original = initial.children().to_vec();
    let mut next = original.clone();
    let mut centroid = next[0].centroid().to_vec();
    centroid[0] += 0.01;
    next[0] = ChildEntry::new(next[0].child(), centroid);
    let mut old_snapshot =
        ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
    let descriptor_key = LogicalKey::GraphDescriptor {
        index: id,
        tree_key: tree.clone(),
        partition: root,
    };
    let old_descriptor = old_snapshot.get(descriptor_key.clone()).await.unwrap();
    let Some(PersistentValue::GraphDescriptor(ref descriptor)) = old_descriptor else {
        panic!("descriptor")
    };
    assert!(
        descriptor.chunks() > 1,
        "fixture must exercise multi-chunk atomic replacement"
    );
    if std::env::var_os("KTANN_GRAPH_UPDATE_TIMING").is_some() {
        eprintln!(
            "graph_update_fixture,nodes={},dimensions={},chunks={},centroid_bytes={},working_limit={},write_limit={}",
            original.len(),
            manifest.config().dimension(),
            descriptor.chunks(),
            original
                .iter()
                .map(|child| child.centroid().len() * 4)
                .sum::<usize>(),
            MEMORY,
            9_000_000
        );
    }
    let expected_hits = query(index).await;
    // A shorter replacement removes stale tail chunks inside the atomic write.
    // Roll back this deliberately uncomposed topology change before publication.
    let smaller = source(backend, &manifest, &tree)
        .await
        .prepare(original[..original.len() / 2].to_vec(), MEMORY, &|| Ok(()))
        .unwrap()
        .unwrap();
    let mut shortened = writer(backend, &manifest).await;
    smaller.publish(&mut shortened).await.unwrap();
    assert!(
        shortened
            .get(LogicalKey::GraphChunk {
                index: id,
                tree_key: tree.clone(),
                partition: root,
                chunk: descriptor.chunks() - 1,
            })
            .await
            .unwrap()
            .is_none()
    );
    shortened.rollback().await;

    let mut stale_next = next.clone();
    let mut stale_centroid = stale_next[0].centroid().to_vec();
    stale_centroid[0] += 0.02;
    stale_next[0] = ChildEntry::new(stale_next[0].child(), stale_centroid);
    let stale = source(backend, &manifest, &tree)
        .await
        .prepare(stale_next, MEMORY, &|| Ok(()))
        .unwrap()
        .unwrap();
    let update = prepare(initial, next, "centroid-prepare");
    // Rolling back a complete replacement must leave the old body searchable.
    let mut aborted = writer(backend, &manifest).await;
    update.publish(&mut aborted).await.unwrap();
    aborted.rollback().await;
    let mut concurrent = writer(backend, &manifest).await;
    update.publish(&mut concurrent).await.unwrap();
    let verified = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        verified.complete && verified.issues.is_empty(),
        "{:?}",
        verified.issues
    );
    for attempt in 0..32 {
        let started = std::time::Instant::now();
        let mut txn = writer(backend, &manifest).await;
        update.publish(&mut txn).await.unwrap();
        if commit(txn, fault, "publish", attempt, started).await {
            break;
        }
        assert!(attempt < 31);
    }
    let mut txn = writer(backend, &manifest).await;
    assert!(update.published(&mut txn).await.unwrap());
    assert_eq!(
        stale.publish(&mut txn).await.unwrap_err().kind(),
        ErrorKind::RetryableAbort
    );
    txn.rollback().await;
    assert_eq!(query(index).await, expected_hits);
    assert_eq!(
        concurrent.commit().await.unwrap_err().kind(),
        ErrorKind::RetryableAbort
    );
    // The old snapshot still reads overwritten bytes, not a decoded cached copy.
    let historical = GraphUpdateSource::read(&mut old_snapshot, &tree, root, MEMORY, &|| Ok(()))
        .await
        .unwrap();
    assert_eq!(historical.children(), original);
    assert_eq!(
        old_snapshot.get(descriptor_key).await.unwrap(),
        old_descriptor
    );
    drop(old_snapshot);

    // Reserve a child identity without creating an orphan partition.
    let mut txn = writer(backend, &manifest).await;
    let child = ktann::storage::tree_manifest::reserve_partition_keys(&mut txn, &tree, 1)
        .await
        .unwrap()
        .next();
    txn.commit().await.unwrap();
    let current = source(backend, &manifest, &tree).await;
    let mut next = current.children().to_vec();
    let centroid = next[0].centroid().to_vec();
    next.push(ChildEntry::new(child, centroid.clone()));
    next.sort_by_key(ChildEntry::child);
    let update = prepare(current, next, "insert-prepare");
    for attempt in 0..32 {
        let started = std::time::Instant::now();
        let mut txn = writer(backend, &manifest).await;
        if !update.published(&mut txn).await.unwrap() {
            txn.put(
                LogicalKey::Header {
                    index: id,
                    tree_key: tree.clone(),
                    partition: child,
                },
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(1, 0, 0, PartitionState::Ready).unwrap(),
                ),
            )
            .await
            .unwrap();
            txn.put(
                LogicalKey::State {
                    index: id,
                    tree_key: tree.clone(),
                    partition: child,
                },
                PersistentValue::PartitionState(PartitionTransition::Ready {
                    started_at_unix_millis: 0,
                }),
            )
            .await
            .unwrap();
            txn.put(
                LogicalKey::Centroid {
                    index: id,
                    tree_key: tree.clone(),
                    partition: child,
                },
                PersistentValue::PartitionCentroid(PartitionCentroid::new(centroid.clone())),
            )
            .await
            .unwrap();
            txn.put(
                LogicalKey::Synopsis {
                    index: id,
                    tree_key: tree.clone(),
                    partition: child,
                },
                PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(&manifest)),
            )
            .await
            .unwrap();
            assert!(update.publish(&mut txn).await.unwrap());
        }
        if commit(txn, fault, "insert", attempt, started).await {
            break;
        }
        assert!(attempt < 31);
    }
    assert_eq!(query(index).await, expected_hits);

    // Preparation has no durable state. Dropping an unused preparation leaves
    // nothing to cancel or reclaim; the same result can be published later.
    let current = source(backend, &manifest, &tree).await;
    let update = prepare(current, original.clone(), "delete-prepare");
    for attempt in 0..32 {
        let started = std::time::Instant::now();
        let mut txn = writer(backend, &manifest).await;
        if !update.published(&mut txn).await.unwrap() {
            assert!(update.publish(&mut txn).await.unwrap());
            for key in [
                LogicalKey::Header {
                    index: id,
                    tree_key: tree.clone(),
                    partition: child,
                },
                LogicalKey::State {
                    index: id,
                    tree_key: tree.clone(),
                    partition: child,
                },
                LogicalKey::Centroid {
                    index: id,
                    tree_key: tree.clone(),
                    partition: child,
                },
                LogicalKey::Synopsis {
                    index: id,
                    tree_key: tree.clone(),
                    partition: child,
                },
            ] {
                txn.delete(key).await.unwrap();
            }
        }
        if commit(txn, fault, "delete", attempt, started).await {
            break;
        }
        assert!(attempt < 31);
    }
    let current = source(backend, &manifest, &tree).await;
    assert_eq!(current.children(), original);
    assert!(
        current
            .prepare(original, MEMORY, &|| Ok(()))
            .unwrap()
            .is_none()
    );
    let verified = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        verified.complete && verified.issues.is_empty(),
        "{:?}",
        verified.issues
    );
}
