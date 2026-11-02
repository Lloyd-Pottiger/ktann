//! Shared atomic graph transition corpus; backend fault controls are injected
//! by the existing Memory harness, while production adapters run the same history.
use ktann::api::{
    DataType, Error, ErrorKind, Index, LogicalIndexId, PartitionKey, Result, Value, VerifyOptions,
};
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::graph_transitions::{
    GraphTransitionSource, PreparedGraphTransition, protect_sources, publish_batch,
};
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionSynopsis, PartitionTransition, PersistentValue,
};
use ktann::storage::{ReadLogicalTxn, WriteLogicalTxn};

const MEMORY: usize = 64 * 1024 * 1024;

/// Test-owned topology edits; publication itself uses the production transition.
struct Update {
    graph: PreparedGraphTransition,
    index: LogicalIndexId,
    tree: TreeKey,
    header: PartitionHeader,
    changes: Vec<(PartitionKey, Option<ChildEntry>)>,
    expected_descriptor: Option<PersistentValue>,
}

impl Update {
    fn header_key(&self) -> LogicalKey {
        LogicalKey::Header {
            index: self.index,
            tree_key: self.tree.clone(),
            partition: PartitionKey::new(1).unwrap(),
        }
    }

    fn descriptor_key(&self) -> LogicalKey {
        LogicalKey::GraphDescriptor {
            index: self.index,
            tree_key: self.tree.clone(),
            partition: PartitionKey::new(1).unwrap(),
        }
    }

    /// Fault retries recognize the exact Header and descriptor observed in the
    /// prepared transaction, rather than treating any later epoch as success.
    async fn published<T: WriteTxn>(&self, txn: &mut WriteLogicalTxn<'_, T>) -> Result<bool> {
        if protect_sources(txn, std::slice::from_ref(&self.graph)).await? {
            return Ok(false);
        }
        if let Some(descriptor) = &self.expected_descriptor {
            let actual = txn
                .batch_get_for_update(vec![self.header_key(), self.descriptor_key()])
                .await?;
            if actual
                == vec![
                    Some(PersistentValue::PartitionHeader(self.header)),
                    Some(descriptor.clone()),
                ]
            {
                return Ok(true);
            }
        }
        Err(Error::new(ErrorKind::RetryableAbort))
    }

    async fn publish<T: WriteTxn>(&mut self, txn: &mut WriteLogicalTxn<'_, T>) -> Result<bool> {
        if self.published(txn).await? {
            return Ok(false);
        }
        for (child, entry) in &self.changes {
            let key = LogicalKey::ChildEntry {
                index: self.index,
                tree_key: self.tree.clone(),
                partition: PartitionKey::new(1).unwrap(),
                child: *child,
            };
            if let Some(entry) = entry {
                txn.put(key, PersistentValue::ChildEntry(entry.clone()))
                    .await?;
                txn.put(
                    LogicalKey::Centroid {
                        index: self.index,
                        tree_key: self.tree.clone(),
                        partition: *child,
                    },
                    PersistentValue::PartitionCentroid(PartitionCentroid::new(entry.centroid())),
                )
                .await?;
            } else {
                txn.delete(key).await?;
            }
        }
        txn.put(
            self.header_key(),
            PersistentValue::PartitionHeader(self.header),
        )
        .await?;
        publish_batch(txn, std::slice::from_ref(&self.graph)).await?;
        self.expected_descriptor = txn.get(self.descriptor_key()).await?;
        assert!(self.expected_descriptor.is_some());
        Ok(true)
    }
}

fn prepare(
    source: GraphTransitionSource,
    next: Vec<ChildEntry>,
    manifest: &IndexManifest,
    tree: &TreeKey,
) -> Update {
    let old = source.children();
    let mut changes = Vec::new();
    for entry in &next {
        if old
            .binary_search_by_key(&entry.child(), ChildEntry::child)
            .map(|i| &old[i])
            .ok()
            != Some(entry)
        {
            changes.push((entry.child(), Some(entry.clone())));
        }
    }
    for entry in old {
        if next
            .binary_search_by_key(&entry.child(), ChildEntry::child)
            .is_err()
        {
            changes.push((entry.child(), None));
        }
    }
    let header = source.header().unwrap();
    let header = PartitionHeader::new(
        header.level(),
        next.len() as u32,
        header.cache_epoch() + 1,
        header.state(),
    )
    .unwrap();
    let graph = source
        .prepare(Some(header), next, MEMORY, &|| Ok(()))
        .unwrap();
    Update {
        graph,
        index: manifest.logical_index_id(),
        tree: tree.clone(),
        header,
        changes,
        expected_descriptor: None,
    }
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
) -> bool {
    fault(phase, attempt);
    match txn.commit().await {
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
    }
}

async fn source<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree: &TreeKey,
) -> GraphTransitionSource {
    let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), manifest);
    GraphTransitionSource::read(
        &mut txn,
        tree,
        PartitionKey::new(1).unwrap(),
        MEMORY,
        &|| Ok(()),
    )
    .await
    .unwrap()
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

    let expected_hits = query(index).await;
    // A shorter replacement removes stale tail chunks inside the atomic write.
    // Roll back this deliberately uncomposed topology change before publication.
    let mut smaller = prepare(
        source(backend, &manifest, &tree).await,
        original[..original.len() / 2].to_vec(),
        &manifest,
        &tree,
    );
    let mut shortened = writer(backend, &manifest).await;
    assert!(
        protect_sources(&mut shortened, std::slice::from_ref(&smaller.graph))
            .await
            .unwrap()
    );
    // Source fencing is not proof that the topology owner installed its target.
    assert_eq!(
        publish_batch(&mut shortened, std::slice::from_ref(&smaller.graph))
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::RetryableAbort
    );
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

    // Header-only topology transitions still replace the snapshot-bound descriptor.
    let mut stale = prepare(
        source(backend, &manifest, &tree).await,
        original.clone(),
        &manifest,
        &tree,
    );
    let mut update = prepare(initial, original.clone(), &manifest, &tree);
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
        let mut txn = writer(backend, &manifest).await;
        update.publish(&mut txn).await.unwrap();
        if commit(txn, fault, "publish", attempt).await {
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
    let historical =
        GraphTransitionSource::read(&mut old_snapshot, &tree, root, MEMORY, &|| Ok(()))
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
    let mut update = prepare(current, next, &manifest, &tree);
    for attempt in 0..32 {
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
        if commit(txn, fault, "insert", attempt).await {
            break;
        }
        assert!(attempt < 31);
    }
    assert_eq!(query(index).await, expected_hits);

    // Preparation has no durable state. Dropping an unused preparation leaves
    // nothing to cancel or reclaim; the same result can be published later.
    let current = source(backend, &manifest, &tree).await;
    let mut update = prepare(current, original.clone(), &manifest, &tree);
    for attempt in 0..32 {
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
        if commit(txn, fault, "delete", attempt).await {
            break;
        }
        assert!(attempt < 31);
    }
    let current = source(backend, &manifest, &tree).await;
    assert_eq!(current.children(), original);
    let verified = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(
        verified.complete && verified.issues.is_empty(),
        "{:?}",
        verified.issues
    );
}
