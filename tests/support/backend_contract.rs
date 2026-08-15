//! Backend-neutral behavioral contract shared by every adapter test.

use bytes::Bytes;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, InsertOutcome, Mutation, ReadOps, ScanLimits, WriteTxn};
use ktann::storage::keys::KeyRange;

fn key(value: &'static [u8]) -> Bytes {
    Bytes::from_static(value)
}

fn range(start: &[u8], end: &[u8]) -> KeyRange {
    KeyRange::new(start.to_vec(), end.to_vec())
}

/// Runs the public transaction contract unchanged against one empty backend.
pub async fn run<B: Backend>(backend: &B) {
    let hard_limits = backend.hard_limits();
    let admission_budget = backend.admission_budget();
    assert!(hard_limits.max_key_bytes > 0);
    assert!(hard_limits.max_value_bytes > 0);
    assert!(admission_budget.max_mutations > 0);
    assert!(admission_budget.max_mutation_bytes > 0);

    let mut seed = backend.begin_write().await.expect("begin snapshot seed");
    seed.put(key(b"shared-contract/snapshot"), key(b"old"))
        .await
        .expect("seed snapshot value");
    seed.commit().await.expect("commit snapshot seed");

    let mut old_snapshot = backend.begin_read().await.expect("begin old snapshot");
    let mut update = backend.begin_write().await.expect("begin snapshot update");
    update
        .put(key(b"shared-contract/snapshot"), key(b"new"))
        .await
        .expect("update snapshot value");
    update
        .put(key(b"shared-contract/later"), key(b"visible"))
        .await
        .expect("put later value");
    assert_eq!(
        update
            .get(key(b"shared-contract/snapshot"))
            .await
            .expect("read own write"),
        Some(key(b"new")),
    );
    update.commit().await.expect("commit snapshot update");
    assert_eq!(
        old_snapshot
            .get(key(b"shared-contract/snapshot"))
            .await
            .expect("read old snapshot"),
        Some(key(b"old")),
    );
    assert_eq!(
        old_snapshot
            .get(key(b"shared-contract/later"))
            .await
            .expect("read absent later key"),
        None,
    );
    drop(old_snapshot);

    let mut batch = backend.begin_write().await.expect("begin batch");
    batch
        .batch_mutate(vec![
            Mutation::Put {
                key: key(b"shared-contract/batch/a"),
                value: key(b"first"),
            },
            Mutation::Put {
                key: key(b"shared-contract/batch/a"),
                value: key(b"second"),
            },
            Mutation::Put {
                key: key(b"shared-contract/batch/b"),
                value: key(b"deleted"),
            },
            Mutation::Delete {
                key: key(b"shared-contract/batch/b"),
            },
        ])
        .await
        .expect("batch mutate");
    assert_eq!(
        batch
            .batch_get(vec![
                key(b"shared-contract/batch/a"),
                key(b"shared-contract/batch/b"),
                key(b"shared-contract/batch/a"),
            ])
            .await
            .expect("batch read own writes"),
        vec![Some(key(b"second")), None, Some(key(b"second"))],
    );
    let write_page = batch
        .scan(
            &range(b"shared-contract/batch/", b"shared-contract/batch0"),
            ScanLimits {
                item_limit: 10,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("write scan");
    assert_eq!(write_page.items().len(), 1);
    assert_eq!(write_page.items()[0].value().as_ref(), b"second");
    batch.commit().await.expect("commit batch");

    let mut scan_seed = backend.begin_write().await.expect("begin scan seed");
    for (entry_key, value) in [
        (b"shared-contract/scan/a".as_slice(), b"1".as_slice()),
        (b"shared-contract/scan/b".as_slice(), b"2".as_slice()),
        (b"shared-contract/scan/c".as_slice(), b"12345".as_slice()),
    ] {
        scan_seed
            .put(
                Bytes::copy_from_slice(entry_key),
                Bytes::copy_from_slice(value),
            )
            .await
            .expect("put scan seed");
    }
    scan_seed.commit().await.expect("commit scan seed");

    let mut scan = backend.begin_read().await.expect("begin scan");
    let first_page = scan
        .scan(
            &range(b"shared-contract/scan/", b"shared-contract/scan0"),
            ScanLimits {
                item_limit: 2,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("scan first page");
    assert_eq!(first_page.items().len(), 2);
    let next = first_page.next_start().expect("scan cursor").clone();
    assert!(next.as_ref() > first_page.items()[1].key().as_ref());
    let second_page = scan
        .scan(
            &KeyRange::new(next.to_vec(), b"shared-contract/scan0".to_vec()),
            ScanLimits {
                item_limit: 2,
                byte_limit: 1,
            },
        )
        .await
        .expect("scan second page");
    assert_eq!(second_page.items().len(), 1);
    assert_eq!(second_page.items()[0].value().as_ref(), b"12345");
    assert!(second_page.next_start().is_none());
    assert_eq!(
        scan.scan(
            &range(b"z", b"a"),
            ScanLimits {
                item_limit: 0,
                byte_limit: 1,
            },
        )
        .await
        .expect_err("zero scan limit")
        .kind(),
        ErrorKind::InvalidArgument,
    );
    drop(scan);

    let mut first_insert = backend.begin_write().await.expect("begin first insert");
    let mut second_insert = backend.begin_write().await.expect("begin second insert");
    assert_eq!(
        first_insert
            .insert(key(b"shared-contract/unique"), key(b"winner"))
            .await
            .expect("first insert"),
        InsertOutcome::Inserted,
    );
    assert_eq!(
        second_insert
            .insert(key(b"shared-contract/unique"), key(b"loser"))
            .await
            .expect("second insert"),
        InsertOutcome::Inserted,
    );
    first_insert.commit().await.expect("commit first insert");
    assert_eq!(
        second_insert
            .commit()
            .await
            .expect_err("second insert conflicts")
            .kind(),
        ErrorKind::RetryableAbort,
    );

    let mut aba_reader = backend.begin_write().await.expect("begin ABA reader");
    aba_reader
        .get_for_update(key(b"shared-contract/aba"))
        .await
        .expect("protected ABA read");
    let mut aba_write = backend.begin_write().await.expect("begin ABA write");
    aba_write
        .put(key(b"shared-contract/aba"), key(b"temporary"))
        .await
        .expect("ABA put");
    aba_write.commit().await.expect("commit ABA put");
    let mut aba_restore = backend.begin_write().await.expect("begin ABA restore");
    aba_restore
        .delete(key(b"shared-contract/aba"))
        .await
        .expect("ABA delete");
    aba_restore.commit().await.expect("commit ABA restore");
    aba_reader
        .put(key(b"shared-contract/aba"), key(b"stale"))
        .await
        .expect("stage stale ABA write");
    assert_eq!(
        aba_reader
            .commit()
            .await
            .expect_err("ABA commit conflicts")
            .kind(),
        ErrorKind::RetryableAbort,
    );

    let mut rollback = backend.begin_write().await.expect("begin rollback");
    rollback
        .put(key(b"shared-contract/rollback"), key(b"hidden"))
        .await
        .expect("put rollback value");
    rollback.rollback().await;
    let mut dropped = backend
        .begin_write()
        .await
        .expect("begin dropped transaction");
    dropped
        .put(key(b"shared-contract/dropped"), key(b"hidden"))
        .await
        .expect("put dropped value");
    drop(dropped);
    let mut after_abandon = backend.begin_read().await.expect("read abandoned writes");
    assert_eq!(
        after_abandon
            .batch_get(vec![
                key(b"shared-contract/rollback"),
                key(b"shared-contract/dropped"),
            ])
            .await
            .expect("batch get abandoned writes"),
        vec![None, None],
    );
    drop(after_abandon);

    let clear_range = range(b"shared-contract/clear/", b"shared-contract/clear0");
    if backend.capabilities().transactional_clear_range {
        let mut clear = backend.begin_write().await.expect("begin range clear");
        clear
            .clear_range(&clear_range)
            .await
            .expect("stage range clear");
        let mut concurrent = backend.begin_write().await.expect("begin concurrent put");
        concurrent
            .put(key(b"shared-contract/clear/concurrent"), key(b"removed"))
            .await
            .expect("put concurrent value");
        concurrent.commit().await.expect("commit concurrent put");
        clear.commit().await.expect("commit range clear");

        let mut after_clear = backend.begin_read().await.expect("read cleared range");
        assert_eq!(
            after_clear
                .get(key(b"shared-contract/clear/concurrent"))
                .await
                .expect("get cleared value"),
            None,
        );
    } else {
        let mut clear = backend
            .begin_write()
            .await
            .expect("begin unsupported clear");
        assert_eq!(
            clear
                .clear_range(&clear_range)
                .await
                .expect_err("range clear unsupported")
                .kind(),
            ErrorKind::Unsupported,
        );
    }
}
