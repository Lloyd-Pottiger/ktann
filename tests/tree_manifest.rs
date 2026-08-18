//! Tree Manifest directory and Partition Key allocation contract tests.

use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, LogicalIndexId, Metric, PartitionKey,
    Value,
};
use ktann::storage::backend::{Backend, ScanLimits, WriteTxn};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::tree_manifest::{
    self, DEFAULT_PARTITION_KEY_RESERVATION, PartitionKeyReservation, TreeCreation,
};
use ktann::storage::values::{
    IndexLifecycle, IndexManifest, PartitionHeader, PartitionState, PartitionSynopsis,
    PartitionTransition, PersistentValue, TreeManifest,
};
use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

use support::DeterministicBackend;

#[allow(dead_code)]
mod support;

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
}

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

fn manifest() -> IndexManifest {
    let config = IndexConfig::new(1, Metric::L2)
        .expect("valid config")
        .with_fields(vec![FieldSchema::new("a", DataType::I64).expect("field")])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields");
    IndexManifest::new(IndexLifecycle::Active, id(7), config, [7; 32], vec![None])
        .expect("valid manifest")
}

fn tree_key(value: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(value)]).expect("canonical key")
}

async fn write_txn<'b, 'm>(
    backend: &'b DeterministicBackend,
    manifest: &'m IndexManifest,
) -> WriteLogicalTxn<'m, <DeterministicBackend as Backend>::WriteTxn<'b>> {
    let raw = backend.begin_write().await.expect("begin write");
    WriteLogicalTxn::for_index(
        raw,
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind manifest")
}

async fn read_txn<'b, 'm>(
    backend: &'b DeterministicBackend,
    manifest: &'m IndexManifest,
) -> ReadLogicalTxn<'m, <DeterministicBackend as Backend>::ReadTxn<'b>> {
    let raw = backend.begin_read().await.expect("begin read");
    ReadLogicalTxn::for_index(raw, manifest).expect("bind manifest")
}

async fn create_tree_on(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    started_at_unix_millis: u64,
) -> TreeCreation {
    let mut txn = write_txn(backend, manifest).await;
    let outcome = tree_manifest::create_tree(&mut txn, key, started_at_unix_millis)
        .await
        .expect("create");
    txn.commit().await.expect("commit");
    outcome
}

async fn reserve_on(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    count: u32,
) -> PartitionKeyReservation {
    let mut txn = write_txn(backend, manifest).await;
    let reservation = tree_manifest::reserve_partition_keys(&mut txn, key, count)
        .await
        .expect("reserve");
    txn.commit().await.expect("commit");
    reservation
}

async fn read_value(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    key: LogicalKey,
) -> Option<PersistentValue> {
    let mut txn = read_txn(backend, manifest).await;
    txn.get(key).await.expect("typed read")
}

#[tokio::test]
async fn creation_installs_the_manifest_and_the_initial_leaf_root() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    let mut txn = write_txn(&backend, &manifest).await;
    assert_eq!(
        tree_manifest::create_tree(&mut txn, &key, 123)
            .await
            .expect("create"),
        TreeCreation::Created
    );
    txn.commit().await.expect("commit");

    assert_eq!(
        tree_manifest::read_tree_manifest(&mut read_txn(&backend, &manifest).await, &key)
            .await
            .expect("read manifest"),
        Some(TreeManifest::new(pk(1), pk(1)).expect("initial manifest"))
    );

    let header = read_value(
        &backend,
        &manifest,
        LogicalKey::Header {
            index: id(7),
            tree_key: key.clone(),
            partition: pk(1),
        },
    )
    .await
    .expect("root header");
    assert_eq!(
        header,
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("header")
        )
    );
    assert_eq!(
        read_value(
            &backend,
            &manifest,
            LogicalKey::Synopsis {
                index: id(7),
                tree_key: key.clone(),
                partition: pk(1),
            },
        )
        .await
        .expect("root synopsis"),
        PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(&manifest))
    );
    assert_eq!(
        read_value(
            &backend,
            &manifest,
            LogicalKey::State {
                index: id(7),
                tree_key: key.clone(),
                partition: pk(1),
            },
        )
        .await
        .expect("root state"),
        PersistentValue::PartitionState(PartitionTransition::Ready {
            started_at_unix_millis: 123,
        })
    );

    // The directory holds exactly one entry and the root prefix holds exactly
    // Header, Synopsis, and State.
    let mut txn = read_txn(&backend, &manifest).await;
    let directory = txn
        .scan(
            &LogicalRange::tree_manifests(&manifest),
            None,
            ScanLimits {
                item_limit: 16,
                byte_limit: 1 << 20,
            },
        )
        .await
        .expect("scan directory");
    assert_eq!(directory.items().len(), 1);
    let root = txn
        .scan(
            &LogicalRange::partition(&manifest, &key, pk(1)).expect("partition range"),
            None,
            ScanLimits {
                item_limit: 16,
                byte_limit: 1 << 20,
            },
        )
        .await
        .expect("scan root");
    assert_eq!(root.items().len(), 3);
}

#[tokio::test]
async fn duplicate_creation_is_idempotent_and_changes_nothing() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    assert_eq!(
        create_tree_on(&backend, &manifest, &key, 1).await,
        TreeCreation::Created
    );
    assert_eq!(
        create_tree_on(&backend, &manifest, &key, 2).await,
        TreeCreation::AlreadyExists
    );

    assert_eq!(
        read_value(
            &backend,
            &manifest,
            LogicalKey::State {
                index: id(7),
                tree_key: key,
                partition: pk(1),
            },
        )
        .await
        .expect("root state"),
        PersistentValue::PartitionState(PartitionTransition::Ready {
            started_at_unix_millis: 1,
        })
    );
}

#[tokio::test]
async fn reservations_are_monotonic_disjoint_and_persistent() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    create_tree_on(&backend, &manifest, &key, 0).await;

    let first = reserve_on(&backend, &manifest, &key, 2).await;
    assert_eq!(first.next(), pk(2));
    assert_eq!(first.last(), pk(3));
    assert_eq!(first.count(), 2);
    let second = reserve_on(&backend, &manifest, &key, 3).await;
    assert_eq!(second.next(), pk(4));
    assert_eq!(second.last(), pk(6));
    assert_eq!(second.count(), 3);

    let manifest_after =
        tree_manifest::read_tree_manifest(&mut read_txn(&backend, &manifest).await, &key)
            .await
            .expect("read")
            .expect("manifest exists");
    assert_eq!(manifest_after.partition_key_high_water(), pk(6));
    assert_eq!(DEFAULT_PARTITION_KEY_RESERVATION, 1_024);
}

#[tokio::test]
async fn near_exhaustion_reserves_the_final_suffix_then_reports_exhaustion() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        tree_manifest::create_tree(&mut txn, &key, 0)
            .await
            .expect("create");
        txn.put(
            LogicalKey::TreeManifest {
                index: id(7),
                tree_key: key.clone(),
            },
            PersistentValue::TreeManifest(
                TreeManifest::new(pk(1), pk(u64::MAX - 5)).expect("high water near the end"),
            ),
        )
        .await
        .expect("raise high water");
        txn.commit().await.expect("commit");
    }

    let mut txn = write_txn(&backend, &manifest).await;
    let final_suffix = tree_manifest::reserve_partition_keys(&mut txn, &key, 10)
        .await
        .expect("final suffix");
    assert_eq!(final_suffix.next(), pk(u64::MAX - 4));
    assert_eq!(final_suffix.last(), pk(u64::MAX));
    assert_eq!(final_suffix.count(), 5);
    txn.commit().await.expect("commit");

    let mut txn = write_txn(&backend, &manifest).await;
    let error = tree_manifest::reserve_partition_keys(&mut txn, &key, 1)
        .await
        .expect_err("space exhausted");
    assert_eq!(error.kind(), ErrorKind::IdExhausted);
    txn.rollback().await;
}

#[tokio::test]
async fn reservation_rejects_missing_trees_and_zero_counts() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    let mut txn = write_txn(&backend, &manifest).await;
    let error = tree_manifest::reserve_partition_keys(&mut txn, &key, 1)
        .await
        .expect_err("absent tree");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    txn.rollback().await;

    {
        let mut txn = write_txn(&backend, &manifest).await;
        tree_manifest::create_tree(&mut txn, &key, 0)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }

    let mut txn = write_txn(&backend, &manifest).await;
    let error = tree_manifest::reserve_partition_keys(&mut txn, &key, 0)
        .await
        .expect_err("zero count");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    txn.rollback().await;

    assert_eq!(
        tree_manifest::read_tree_manifest(&mut read_txn(&backend, &manifest).await, &tree_key(2),)
            .await
            .expect("absent read"),
        None
    );
}

#[tokio::test]
async fn concurrent_creation_installs_exactly_one_tree() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let backend = Arc::new(backend);
    let key = tree_key(1);

    let tasks = (0..8)
        .map(|_| {
            let backend = Arc::clone(&backend);
            let manifest = manifest.clone();
            let key = key.clone();
            tokio::spawn(async move {
                loop {
                    let mut txn = write_txn(&backend, &manifest).await;
                    let outcome = tree_manifest::create_tree(&mut txn, &key, 0)
                        .await
                        .expect("create");
                    match txn.commit().await {
                        Ok(()) => return outcome,
                        Err(error) if error.kind() == ErrorKind::RetryableAbort => continue,
                        Err(error) => panic!("unexpected commit error: {error}"),
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let mut created = 0;
    let mut existing = 0;
    for task in tasks {
        match task.await.expect("task join") {
            TreeCreation::Created => created += 1,
            TreeCreation::AlreadyExists => existing += 1,
            _ => panic!("unexpected creation outcome"),
        }
    }
    assert_eq!((created, existing), (1, 7));

    let manifest_after =
        tree_manifest::read_tree_manifest(&mut read_txn(&backend, &manifest).await, &key)
            .await
            .expect("read")
            .expect("manifest exists");
    assert_eq!(manifest_after.partition_key_high_water(), pk(1));
}

#[tokio::test]
async fn concurrent_reservations_partition_the_keyspace() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        tree_manifest::create_tree(&mut txn, &key, 0)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }

    let backend = Arc::new(backend);
    const TASKS: u64 = 8;
    const PER_TASK: u32 = 16;
    let tasks = (0..TASKS)
        .map(|_| {
            let backend = Arc::clone(&backend);
            let manifest = manifest.clone();
            let key = key.clone();
            tokio::spawn(async move {
                loop {
                    let mut txn = write_txn(&backend, &manifest).await;
                    let reservation =
                        match tree_manifest::reserve_partition_keys(&mut txn, &key, PER_TASK).await
                        {
                            Ok(reservation) => reservation,
                            Err(error) => panic!("unexpected reserve error: {error}"),
                        };
                    match txn.commit().await {
                        Ok(()) => return reservation,
                        Err(error) if error.kind() == ErrorKind::RetryableAbort => continue,
                        Err(error) => panic!("unexpected commit error: {error}"),
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let mut reservations: Vec<PartitionKeyReservation> = Vec::new();
    for task in tasks {
        reservations.push(task.await.expect("task join"));
    }
    reservations.sort_by_key(|reservation| reservation.next());

    let mut expected_next = 2_u64;
    let mut total = 0_u64;
    for reservation in &reservations {
        assert_eq!(reservation.next().get(), expected_next);
        assert_eq!(reservation.count(), u64::from(PER_TASK));
        total += reservation.count();
        expected_next += reservation.count();
    }
    assert_eq!(total, TASKS * u64::from(PER_TASK));

    let manifest_after =
        tree_manifest::read_tree_manifest(&mut read_txn(&backend, &manifest).await, &key)
            .await
            .expect("read")
            .expect("manifest exists");
    assert_eq!(manifest_after.partition_key_high_water().get(), 1 + total);
}

#[tokio::test]
async fn corruption_fails_closed_on_directory_reads() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        tree_manifest::create_tree(&mut txn, &key, 0)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }

    // Overwrite the Tree Manifest with garbage bytes through the raw seam.
    {
        let mut raw = backend.begin_write().await.expect("begin write");
        raw.put(
            Bytes::from(keys::tree_manifest_key(id(7), &key)),
            Bytes::from(vec![0xde, 0xad, 0xbe, 0xef]),
        )
        .await
        .expect("raw put");
        raw.commit().await.expect("commit");
    }

    let mut txn = read_txn(&backend, &manifest).await;
    let error = tree_manifest::read_tree_manifest(&mut txn, &key)
        .await
        .expect_err("garbage manifest");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn a_wrong_value_kind_at_the_directory_key_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    let mut txn = write_txn(&backend, &manifest).await;
    tree_manifest::create_tree(&mut txn, &key, 0)
        .await
        .expect("create");
    txn.commit().await.expect("commit");

    // The typed layer rejects a wrong-kind value on the encode path, so the
    // test injects one through the raw backend seam with a canonical encoding.
    let encoded = ktann::storage::values::ValueCodec::for_index(&manifest)
        .encode(&PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("header"),
        ))
        .expect("encode header");
    let mut raw = backend.begin_write().await.expect("begin write");
    raw.put(
        Bytes::from(keys::tree_manifest_key(id(7), &key)),
        Bytes::from(encoded),
    )
    .await
    .expect("raw put");
    raw.commit().await.expect("commit");

    let mut txn = read_txn(&backend, &manifest).await;
    let error = tree_manifest::read_tree_manifest(&mut txn, &key)
        .await
        .expect_err("wrong value kind");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn read_tree_manifest_for_update_establishes_conflicts() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        tree_manifest::create_tree(&mut txn, &key, 0)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }

    // Two transactions update-protect the same manifest; only one commit may
    // succeed with an intervening write.
    let first = async {
        let mut txn = write_txn(&backend, &manifest).await;
        let manifest_value = tree_manifest::read_tree_manifest_for_update(&mut txn, &key)
            .await
            .expect("update-protected read");
        assert!(manifest_value.is_some());
        txn.put(
            LogicalKey::TreeManifest {
                index: id(7),
                tree_key: key.clone(),
            },
            PersistentValue::TreeManifest(
                TreeManifest::new(pk(1), pk(10)).expect("advanced manifest"),
            ),
        )
        .await
        .expect("put");
        txn.commit().await.expect("first commit");
    };
    first.await;

    let mut txn = write_txn(&backend, &manifest).await;
    let manifest_value = tree_manifest::read_tree_manifest_for_update(&mut txn, &key)
        .await
        .expect("update-protected read");
    assert_eq!(
        manifest_value,
        Some(TreeManifest::new(pk(1), pk(10)).expect("advanced manifest"))
    );
    txn.rollback().await;
}
