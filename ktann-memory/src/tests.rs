use super::*;

#[tokio::test]
async fn old_writers_expire_without_expiring_read_snapshots_and_history_is_reclaimed() {
    let backend = MemoryBackend::new();
    let key = Bytes::from_static(b"retained");
    let mut initial = backend.begin_write().await.unwrap();
    initial
        .put(key.clone(), Bytes::from_static(b"old"))
        .await
        .unwrap();
    initial.commit().await.unwrap();
    let mut snapshot = backend.begin_read().await.unwrap();
    let mut old = backend.begin_write().await.unwrap();
    old.get_for_update(key.clone()).await.unwrap();

    // Keep a writer open while actual commits exceed the conflict byte window.
    for _ in 0..10 {
        let mut write = backend.begin_write().await.unwrap();
        for n in 0..100_u16 {
            let mut deleted_key = vec![0; 9_000];
            deleted_key[..2].copy_from_slice(&n.to_be_bytes());
            write.delete(Bytes::from(deleted_key)).await.unwrap();
        }
        write.commit().await.unwrap();
    }
    {
        let state = backend.lock().unwrap();
        assert!(state.history_bytes <= MAX_HISTORY_BYTES);
        assert!(state.history_keys <= MAX_HISTORY_KEYS);
    }
    assert_eq!(
        old.commit().await.unwrap_err().kind(),
        ErrorKind::RetryableAbort
    );
    assert_eq!(
        snapshot.get(key).await.unwrap(),
        Some(Bytes::from_static(b"old"))
    );
    let state = backend.lock().unwrap();
    assert!(state.history.is_empty());
    assert!(state.modified.is_empty());
    assert!(state.writers.is_empty());
}

#[tokio::test]
async fn dropped_and_rolled_back_writers_release_conflict_history() {
    let backend = MemoryBackend::new();
    let first = backend.begin_write().await.unwrap();
    let second = backend.begin_write().await.unwrap();
    let mut committed = backend.begin_write().await.unwrap();
    committed
        .delete(Bytes::from_static(b"absent"))
        .await
        .unwrap();
    committed.commit().await.unwrap();
    drop(first);
    assert!(!backend.lock().unwrap().modified.is_empty());
    second.rollback().await;
    assert!(backend.lock().unwrap().modified.is_empty());
}
