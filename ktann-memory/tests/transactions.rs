//! Adapter-specific concurrency, isolation, and admission guarantees.

use bytes::Bytes;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, Mutation, ReadOps, WriteTxn};
use ktann_memory::MemoryBackend;

fn key() -> Bytes {
    Bytes::from_static(b"key")
}

#[tokio::test]
async fn clones_share_data_but_new_instances_are_isolated() {
    let backend = MemoryBackend::new();
    let clone = backend.clone();
    let independent = MemoryBackend::new();
    let mut write = backend.begin_write().await.unwrap();
    write
        .put(key(), Bytes::from_static(b"value"))
        .await
        .unwrap();
    write.commit().await.unwrap();
    let mut snapshot = clone.begin_read().await.unwrap();
    drop(backend);
    drop(clone);
    assert_eq!(
        snapshot.get(key()).await.unwrap(),
        Some(Bytes::from_static(b"value"))
    );
    assert!(
        independent
            .begin_read()
            .await
            .unwrap()
            .get(key())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn disjoint_writers_preserve_each_others_commits() {
    let backend = MemoryBackend::new();
    let mut first = backend.begin_write().await.unwrap();
    let mut second = backend.begin_write().await.unwrap();
    first
        .put(key(), Bytes::from_static(b"first"))
        .await
        .unwrap();
    second
        .put(Bytes::from_static(b"other"), Bytes::from_static(b"second"))
        .await
        .unwrap();
    first.commit().await.unwrap();
    assert!(second.get(key()).await.unwrap().is_none());
    second.commit().await.unwrap();
    let mut read = backend.begin_read().await.unwrap();
    assert_eq!(
        read.get(key()).await.unwrap(),
        Some(Bytes::from_static(b"first"))
    );
    assert_eq!(
        read.get(Bytes::from_static(b"other")).await.unwrap(),
        Some(Bytes::from_static(b"second"))
    );
}

#[tokio::test]
async fn absent_key_aba_before_protected_read_still_conflicts() {
    let backend = MemoryBackend::new();
    let mut old = backend.begin_write().await.unwrap();
    let mut insert = backend.begin_write().await.unwrap();
    insert
        .put(key(), Bytes::from_static(b"temporary"))
        .await
        .unwrap();
    insert.commit().await.unwrap();
    let mut delete = backend.begin_write().await.unwrap();
    delete.delete(key()).await.unwrap();
    delete.commit().await.unwrap();
    assert!(old.get_for_update(key()).await.unwrap().is_none());
    old.put(Bytes::from_static(b"must-not-commit"), Bytes::new())
        .await
        .unwrap();
    assert_eq!(
        old.commit().await.unwrap_err().kind(),
        ErrorKind::RetryableAbort
    );
    assert!(
        backend
            .begin_read()
            .await
            .unwrap()
            .get(Bytes::from_static(b"must-not-commit"))
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_protected_increments_do_not_lose_updates() {
    let backend = MemoryBackend::new();
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let backend = backend.clone();
        tasks.push(tokio::spawn(async move {
            for _ in 0..20 {
                loop {
                    let mut write = backend.begin_write().await.unwrap();
                    let value = write.get_for_update(key()).await.unwrap();
                    let count =
                        value.map_or(0, |v| u64::from_le_bytes(v.as_ref().try_into().unwrap()));
                    tokio::task::yield_now().await;
                    write
                        .put(key(), Bytes::copy_from_slice(&(count + 1).to_le_bytes()))
                        .await
                        .unwrap();
                    match write.commit().await {
                        Ok(()) => break,
                        Err(error) => assert_eq!(error.kind(), ErrorKind::RetryableAbort),
                    }
                }
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    let value = backend
        .begin_read()
        .await
        .unwrap()
        .get(key())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(u64::from_le_bytes(value.as_ref().try_into().unwrap()), 160);
}

#[tokio::test]
async fn admission_rejection_stages_no_partial_batch() {
    let backend = MemoryBackend::new();
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let mut write = backend.begin_write().await.unwrap();
    for mutations in [
        vec![Mutation::Put {
            key: key(),
            value: Bytes::from(vec![0; limits.max_value_bytes + 1]),
        }],
        vec![Mutation::Delete {
            key: Bytes::from(vec![0; limits.max_key_bytes + 1]),
        }],
        vec![
            Mutation::Put {
                key: key(),
                value: Bytes::new()
            };
            budget.max_mutations + 1
        ],
        vec![
            Mutation::Put {
                key: key(),
                value: Bytes::from(vec![0; limits.max_value_bytes])
            };
            budget.max_mutation_bytes / limits.max_value_bytes + 1
        ],
    ] {
        let mut batch = vec![Mutation::Put {
            key: Bytes::from_static(b"prefix"),
            value: Bytes::new(),
        }];
        batch.extend(mutations);
        assert_eq!(
            write.batch_mutate(batch).await.unwrap_err().kind(),
            ErrorKind::LimitExceeded
        );
        assert!(
            write
                .get(Bytes::from_static(b"prefix"))
                .await
                .unwrap()
                .is_none()
        );
    }
    // Rejections leave the transaction usable and consume no admission budget.
    write
        .put(
            Bytes::from(vec![0; limits.max_key_bytes]),
            Bytes::from(vec![0; limits.max_value_bytes]),
        )
        .await
        .unwrap();
    write.commit().await.unwrap();
}

#[tokio::test]
async fn repeated_mutations_consume_the_transaction_budget() {
    let backend = MemoryBackend::new();
    let mut write = backend.begin_write().await.unwrap();
    write
        .batch_mutate(vec![
            Mutation::Delete { key: key() };
            backend.admission_budget().max_mutations
        ])
        .await
        .unwrap();
    assert_eq!(
        write.delete(key()).await.unwrap_err().kind(),
        ErrorKind::LimitExceeded
    );
    write.rollback().await;

    let mut write = backend.begin_write().await.unwrap();
    let value = Bytes::from(vec![0; backend.hard_limits().max_value_bytes]);
    let count = backend.admission_budget().max_mutation_bytes / (key().len() + value.len());
    for _ in 0..count {
        write.put(key(), value.clone()).await.unwrap();
    }
    assert_eq!(
        write.put(key(), value).await.unwrap_err().kind(),
        ErrorKind::LimitExceeded
    );
    write.rollback().await;
}
