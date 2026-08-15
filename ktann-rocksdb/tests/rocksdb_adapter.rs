//! Focused contract checks against a temporary local RocksDB database.

use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, InsertOutcome, Mutation, ReadOps, ScanLimits, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use rocksdb::{MemtableFactory, OptimisticTransactionDB, Options, SliceTransform};

#[path = "../../tests/support/backend_contract.rs"]
mod shared_backend_contract;

fn key(value: &'static [u8]) -> Bytes {
    Bytes::from_static(value)
}

fn range(start: &[u8], end: &[u8]) -> KeyRange {
    KeyRange::new(start.to_vec(), end.to_vec())
}

fn open_database(path: &Path) -> OptimisticTransactionDB {
    let mut options = Options::default();
    options.create_if_missing(true);
    OptimisticTransactionDB::open(&options, path).expect("open RocksDB")
}

#[tokio::test(flavor = "current_thread")]
async fn rocksdb_adapter_preserves_the_backend_contract() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let database_path = directory.path().join("database");
    let primary_namespace = BackendNamespace::new("ktann-issue-12-integration").expect("namespace");
    let isolated_namespace =
        BackendNamespace::new("ktann-issue-12-integration-isolated").expect("namespace");

    {
        let database = Arc::new(open_database(&database_path));
        let primary = RocksDbBackend::new(Arc::clone(&database), primary_namespace.clone());
        let isolated = RocksDbBackend::new(Arc::clone(&database), isolated_namespace.clone());

        assert_eq!(primary.admission_budget().max_mutations, 10_000);
        assert_eq!(primary.admission_budget().max_mutation_bytes, 1 << 20);
        assert_eq!(primary.hard_limits().max_value_bytes, u32::MAX as usize);
        assert!(!primary.capabilities().transactional_clear_range);
        shared_backend_contract::run(&primary).await;

        let mut write = primary.begin_write().await.expect("begin write");
        write
            .put(key(b"shared"), key(b"primary"))
            .await
            .expect("put");
        assert_eq!(
            write.get(key(b"shared")).await.expect("read your write"),
            Some(key(b"primary")),
        );
        write.commit().await.expect("commit primary");

        let mut isolated_write = isolated.begin_write().await.expect("begin isolated write");
        isolated_write
            .put(key(b"shared"), key(b"isolated"))
            .await
            .expect("isolated put");
        isolated_write.commit().await.expect("commit isolated");
        let mut isolated_read = isolated.begin_read().await.expect("begin isolated read");
        assert_eq!(
            isolated_read
                .get(key(b"shared"))
                .await
                .expect("isolated get"),
            Some(key(b"isolated")),
        );

        let mut old_snapshot = primary.begin_read().await.expect("begin old snapshot");
        let mut update = primary.begin_write().await.expect("begin update");
        update
            .put(key(b"shared"), key(b"updated"))
            .await
            .expect("update");
        update.commit().await.expect("commit update");
        assert_eq!(
            old_snapshot
                .get(key(b"shared"))
                .await
                .expect("snapshot get"),
            Some(key(b"primary")),
        );

        let mut insert = primary.begin_write().await.expect("begin insert");
        assert_eq!(
            insert
                .insert(key(b"unique"), key(b"first"))
                .await
                .expect("insert"),
            InsertOutcome::Inserted,
        );
        assert_eq!(
            insert
                .insert(key(b"unique"), key(b"second"))
                .await
                .expect("duplicate insert"),
            InsertOutcome::AlreadyExists,
        );
        insert.commit().await.expect("commit insert");

        let mut first_insert = primary
            .begin_write()
            .await
            .expect("begin first unique conflict");
        let mut second_insert = primary
            .begin_write()
            .await
            .expect("begin second unique conflict");
        assert_eq!(
            first_insert
                .insert(key(b"unique-conflict"), key(b"winner"))
                .await
                .expect("first unique insert"),
            InsertOutcome::Inserted,
        );
        assert_eq!(
            second_insert
                .insert(key(b"unique-conflict"), key(b"loser"))
                .await
                .expect("second unique insert"),
            InsertOutcome::Inserted,
        );
        first_insert
            .commit()
            .await
            .expect("commit first unique insert");
        assert_eq!(
            second_insert
                .commit()
                .await
                .expect_err("second unique insert conflicts")
                .kind(),
            ErrorKind::RetryableAbort,
        );
        let mut after_unique_conflict = primary
            .begin_read()
            .await
            .expect("read unique conflict winner");
        assert_eq!(
            after_unique_conflict
                .get(key(b"unique-conflict"))
                .await
                .expect("unique conflict value"),
            Some(key(b"winner")),
        );

        let mut rolled_back = primary.begin_write().await.expect("begin rollback");
        rolled_back
            .put(key(b"rolled-back"), key(b"value"))
            .await
            .expect("put");
        rolled_back.rollback().await;
        let mut after_rollback = primary.begin_read().await.expect("begin read");
        assert_eq!(
            after_rollback.get(key(b"rolled-back")).await.expect("get"),
            None,
        );

        let mut batch = primary.begin_write().await.expect("begin batch");
        batch
            .batch_mutate(vec![
                Mutation::Put {
                    key: key(b"batch/item"),
                    value: key(b"first"),
                },
                Mutation::Put {
                    key: key(b"batch/item"),
                    value: key(b"second"),
                },
                Mutation::Put {
                    key: key(b"batch/deleted"),
                    value: key(b"value"),
                },
                Mutation::Delete {
                    key: key(b"batch/deleted"),
                },
            ])
            .await
            .expect("batch mutate");
        assert_eq!(
            batch
                .batch_get(vec![
                    key(b"batch/item"),
                    key(b"batch/deleted"),
                    key(b"batch/item"),
                ])
                .await
                .expect("batch get"),
            vec![Some(key(b"second")), None, Some(key(b"second"))],
        );
        assert_eq!(
            batch
                .batch_get_for_update(vec![key(b"shared"), key(b"shared")])
                .await
                .expect("protected batch get"),
            vec![Some(key(b"updated")), Some(key(b"updated"))],
        );
        let batch_scan = batch
            .scan(
                &range(b"batch/", b"batch0"),
                ScanLimits {
                    item_limit: 10,
                    byte_limit: 1_024,
                },
            )
            .await
            .expect("write transaction scan");
        assert_eq!(batch_scan.items().len(), 1);
        assert_eq!(batch_scan.items()[0].value().as_ref(), b"second");
        batch.commit().await.expect("commit batch");

        let mut seed = primary.begin_write().await.expect("begin scan seed");
        seed.put(key(b"scan/a"), key(b"1")).await.expect("seed a");
        seed.put(key(b"scan/b"), key(b"2")).await.expect("seed b");
        seed.put(key(b"scan/c"), key(b"3")).await.expect("seed c");
        let large_value = Bytes::from(vec![7; 50_000]);
        seed.put(key(b"large/a"), large_value.clone())
            .await
            .expect("seed large a");
        seed.put(key(b"large/b"), large_value)
            .await
            .expect("seed large b");
        seed.commit().await.expect("commit scan seed");

        let mut scan = primary.begin_read().await.expect("begin scan");
        let first_page = scan
            .scan(
                &range(b"scan/", b"scan0"),
                ScanLimits {
                    item_limit: 2,
                    byte_limit: 1_024,
                },
            )
            .await
            .expect("scan first page");
        assert_eq!(
            first_page
                .items()
                .iter()
                .map(|item| item.key().as_ref())
                .collect::<Vec<_>>(),
            vec![&b"scan/a"[..], &b"scan/b"[..]],
        );
        let next = first_page.next_start().expect("scan cursor").clone();
        let byte_bounded = scan
            .scan(
                &range(b"scan/", b"scan0"),
                ScanLimits {
                    item_limit: 10,
                    byte_limit: b"scan/a".len() + 1,
                },
            )
            .await
            .expect("byte-bounded scan");
        assert_eq!(byte_bounded.items().len(), 1);
        assert_eq!(
            byte_bounded.next_start().expect("byte cursor").as_ref(),
            b"scan/b",
        );
        let second_page = scan
            .scan(
                &KeyRange::new(next.to_vec(), b"scan0".to_vec()),
                ScanLimits {
                    item_limit: 2,
                    byte_limit: 1_024,
                },
            )
            .await
            .expect("scan second page");
        assert_eq!(second_page.items()[0].key().as_ref(), b"scan/c");
        assert!(second_page.next_start().is_none());
        let capped_page = scan
            .scan(
                &range(b"large/", b"large0"),
                ScanLimits {
                    item_limit: 10,
                    byte_limit: usize::MAX,
                },
            )
            .await
            .expect("adapter-capped scan");
        assert_eq!(capped_page.items().len(), 1);
        assert_eq!(
            capped_page.next_start().expect("adapter cursor").as_ref(),
            b"large/b",
        );
        assert_eq!(
            scan.scan(
                &range(b"scan/", b"scan0"),
                ScanLimits {
                    item_limit: 0,
                    byte_limit: 1,
                },
            )
            .await
            .expect_err("zero item limit")
            .kind(),
            ErrorKind::InvalidArgument,
        );

        let mut first = primary.begin_write().await.expect("begin first conflict");
        let mut second = primary.begin_write().await.expect("begin second conflict");
        first
            .get_for_update(key(b"shared"))
            .await
            .expect("first protected read");
        second
            .get_for_update(key(b"shared"))
            .await
            .expect("second protected read");
        first
            .put(key(b"shared"), key(b"winner"))
            .await
            .expect("first put");
        second
            .put(key(b"shared"), key(b"loser"))
            .await
            .expect("second put");
        first.commit().await.expect("first commit");
        assert_eq!(
            second
                .commit()
                .await
                .expect_err("second commit conflicts")
                .kind(),
            ErrorKind::RetryableAbort,
        );

        let mut unsupported = primary.begin_write().await.expect("begin range clear");
        assert_eq!(
            unsupported
                .clear_range(&range(b"scan/", b"scan0"))
                .await
                .expect_err("range clear is unsupported")
                .kind(),
            ErrorKind::Unsupported,
        );
        unsupported.rollback().await;

        let mut count_limited = primary.begin_write().await.expect("begin count limit");
        let too_many = (0..=primary.admission_budget().max_mutations)
            .map(|_| Mutation::Delete {
                key: key(b"limit/count"),
            })
            .collect();
        assert_eq!(
            count_limited
                .batch_mutate(too_many)
                .await
                .expect_err("mutation count limit")
                .kind(),
            ErrorKind::LimitExceeded,
        );
        count_limited.rollback().await;

        let mut bytes_limited = primary.begin_write().await.expect("begin byte limit");
        assert_eq!(
            bytes_limited
                .put(
                    key(b"limit/bytes"),
                    Bytes::from(vec![0; primary.admission_budget().max_mutation_bytes]),
                )
                .await
                .expect_err("mutation byte limit")
                .kind(),
            ErrorKind::LimitExceeded,
        );
        bytes_limited.rollback().await;
    }

    let reopened = RocksDbBackend::new(open_database(&database_path), primary_namespace);
    let mut durable = reopened.begin_read().await.expect("begin durable read");
    assert_eq!(
        durable.get(key(b"shared")).await.expect("durable get"),
        Some(key(b"winner")),
    );
    drop(durable);
    drop(reopened);

    let isolated = RocksDbBackend::new(open_database(&database_path), isolated_namespace);
    let mut isolated_durable = isolated
        .begin_read()
        .await
        .expect("begin isolated durable read");
    assert_eq!(
        isolated_durable
            .get(key(b"shared"))
            .await
            .expect("isolated durable get"),
        Some(key(b"isolated")),
    );
}

#[tokio::test(flavor = "current_thread")]
async fn scan_uses_total_order_with_a_hash_memtable() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let namespace = BackendNamespace::new([]).expect("namespace");
    let physical_prefix_bytes = b"\0ktann-rocksdb\x01".len() + 1;
    let mut options = Options::default();
    options.create_if_missing(true);
    options.set_prefix_extractor(SliceTransform::create_fixed_prefix(
        physical_prefix_bytes + 1,
    ));
    options.set_allow_concurrent_memtable_write(false);
    options.set_memtable_factory(MemtableFactory::HashSkipList {
        bucket_count: 1_000,
        height: 4,
        branching_factor: 4,
    });
    let database = OptimisticTransactionDB::open(&options, directory.path()).expect("open RocksDB");
    let backend = RocksDbBackend::new(database, namespace);

    let mut seed = backend.begin_write().await.expect("begin seed");
    seed.put(key(b"a/item"), key(b"a")).await.expect("put a");
    seed.put(key(b"b/item"), key(b"b")).await.expect("put b");
    seed.commit().await.expect("commit seed");

    let mut read = backend.begin_read().await.expect("begin read");
    let page = read
        .scan(
            &range(b"", b"\xff"),
            ScanLimits {
                item_limit: 10,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("scan across prefixes");
    assert_eq!(
        page.items()
            .iter()
            .map(|item| item.key().as_ref())
            .collect::<Vec<_>>(),
        vec![&b"a/item"[..], &b"b/item"[..]],
    );
    assert!(page.next_start().is_none());
}
