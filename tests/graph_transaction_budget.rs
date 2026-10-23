//! Combined topology admission charges native conflicts and writes once issued.
use ktann::api::{IndexName, LogicalIndexId};
use ktann::storage::WriteLogicalTxn;
use ktann::storage::backend::Backend;
use ktann::storage::keys::{self, LogicalKey};
use ktann::storage::values::{IndexNameEntry, PersistentValue};
use ktann_memory::MemoryBackend;

#[tokio::test]
async fn affected_data_includes_conflicts_prefixes_and_actual_value_bytes() {
    let backend = MemoryBackend::default();
    let mut budget = backend.admission_budget();
    budget.mutation_key_overhead_bytes = 17;
    let mut txn = WriteLogicalTxn::bootstrap(
        backend.begin_write().await.unwrap(),
        backend.hard_limits(),
        budget,
    );
    let name = IndexName::new("graph-budget").unwrap();
    let key_bytes = keys::name_directory_key(&name).len();
    let key = LogicalKey::IndexNameDirectory(name);
    let conflict = 2 * (key_bytes + 17) + 64;
    let value =
        PersistentValue::IndexNameEntry(IndexNameEntry::new(LogicalIndexId::new(1).unwrap()));
    assert!(txn.get_for_update(key.clone()).await.unwrap().is_none());
    assert_eq!(txn.affected_bytes(), conflict);
    txn.batch_get_for_update(vec![key.clone(), key.clone()])
        .await
        .unwrap();
    assert_eq!(
        txn.affected_bytes(),
        conflict,
        "cached conflicts add no native work"
    );
    txn.put(key.clone(), value.clone()).await.unwrap();
    let value_bytes = txn.size().bytes() - key_bytes;
    let write = 3 * (key_bytes + 17) + value_bytes + 64;
    assert_eq!(txn.affected_bytes(), conflict + write);
    txn.get_for_update(key.clone()).await.unwrap();
    assert_eq!(txn.affected_bytes(), conflict + write);
    txn.delete(key.clone()).await.unwrap();
    assert_eq!(
        txn.affected_bytes(),
        conflict + write + 3 * (key_bytes + 17) + 64
    );
    txn.rollback().await;

    let mut txn = WriteLogicalTxn::bootstrap(
        backend.begin_write().await.unwrap(),
        backend.hard_limits(),
        budget,
    );
    txn.insert(key.clone(), value.clone()).await.unwrap();
    assert_eq!(txn.affected_bytes(), conflict + write);
    txn.insert(key, value).await.unwrap();
    assert_eq!(
        txn.affected_bytes(),
        conflict + write,
        "duplicate insert is cached"
    );
    txn.rollback().await;
}
