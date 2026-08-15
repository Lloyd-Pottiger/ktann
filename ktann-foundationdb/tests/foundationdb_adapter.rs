//! Focused contract checks against a local FoundationDB 7.3 cluster.

use bytes::Bytes;
use foundationdb::Database;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, InsertOutcome, Mutation, ReadOps, ScanLimits, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

#[path = "../../tests/support/backend_contract.rs"]
mod shared_backend_contract;

fn key(value: &'static [u8]) -> Bytes {
    Bytes::from_static(value)
}

fn range(start: &[u8], end: &[u8]) -> KeyRange {
    KeyRange::new(start.to_vec(), end.to_vec())
}

#[expect(
    unsafe_code,
    reason = "the FoundationDB binding requires one process-global network boot"
)]
fn boot_foundationdb() -> foundationdb::api::NetworkAutoStop {
    // SAFETY: this integration-test binary contains one test, so it starts the
    // process-global FoundationDB network exactly once. The returned guard is
    // kept alive until every database, backend, and transaction has dropped.
    unsafe { foundationdb::boot() }
}

async fn clear_test_keys(backend: &FoundationDbBackend) {
    let mut transaction = backend.begin_write().await.expect("begin cleanup");
    transaction
        .clear_range(&range(b"", b"\xff"))
        .await
        .expect("clear test range");
    transaction.commit().await.expect("commit cleanup");
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_adapter_preserves_the_backend_contract() {
    let _network = boot_foundationdb();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let database = Database::new(cluster_file.as_deref()).expect("open FoundationDB");
    let primary = FoundationDbBackend::new(
        database,
        BackendNamespace::new("ktann-issue-6-integration").expect("namespace"),
    );
    let isolated = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("open isolated database handle"),
        BackendNamespace::new("ktann-issue-6-integration-isolated").expect("namespace"),
    );
    clear_test_keys(&primary).await;
    clear_test_keys(&isolated).await;
    assert_eq!(primary.admission_budget().max_mutations, 10_000);
    assert_eq!(primary.admission_budget().max_mutation_bytes, 1 << 20);
    assert_eq!(primary.hard_limits().max_value_bytes, 100_000);
    assert!(primary.capabilities().transactional_clear_range);
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
        b"scan/b"
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

    let mut clear = primary.begin_write().await.expect("begin range clear");
    clear
        .put(key(b"scan/before-clear"), key(b"removed"))
        .await
        .expect("put before clear");
    clear
        .clear_range(&range(b"scan/", b"scan0"))
        .await
        .expect("range clear");
    clear
        .put(key(b"scan/after-clear"), key(b"retained"))
        .await
        .expect("put after clear");
    clear.commit().await.expect("commit range clear");
    let reopened = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("reopen database"),
        BackendNamespace::new("ktann-issue-6-integration").expect("namespace"),
    );
    let mut durable = reopened.begin_read().await.expect("begin durable read");
    assert_eq!(
        durable.get(key(b"shared")).await.expect("durable get"),
        Some(key(b"winner")),
    );
    let after_clear = durable
        .scan(
            &range(b"scan/", b"scan0"),
            ScanLimits {
                item_limit: 10,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("scan cleared range");
    assert_eq!(after_clear.items().len(), 1);
    assert_eq!(after_clear.items()[0].key().as_ref(), b"scan/after-clear",);
    drop(durable);
    clear_test_keys(&reopened).await;
    clear_test_keys(&isolated).await;
}
