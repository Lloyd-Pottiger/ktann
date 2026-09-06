//! Shared builders for index/tree fixtures and typed transactions in tests.

use ktann::api::{
    DataType, FieldId, FieldSchema, IndexConfig, IndexName, LogicalIndexId, Metric, PartitionKey,
    Value,
};
use ktann::storage::backend::Backend;
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    IndexIdAllocator, IndexLifecycle, IndexManifest, IndexNameEntry, PersistentValue,
};
use ktann::storage::{ReadLogicalTxn, WriteLogicalTxn, tree_manifest};

use super::{DeterministicBackend, SharedBackend};

/// A nonzero Logical Index ID for fixtures.
pub fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
}

/// A nonzero Partition Key for fixtures.
pub fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

/// A one-dimensional L2 index with one I64 field that is also the Tree Key.
///
/// Rotation is the identity at dimension 1, so routing distances are plain
/// squared differences and trained centroids are plain arithmetic means;
/// fixtures need no numeric setup.
pub fn manifest() -> IndexManifest {
    let config = IndexConfig::new(1, Metric::L2)
        .expect("valid config")
        .with_fields(vec![FieldSchema::new("a", DataType::I64).expect("field")])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields");
    IndexManifest::new(IndexLifecycle::Active, id(7), config, [7; 32], vec![None])
        .expect("valid manifest")
}

/// The canonical Tree Key of one I64 field value.
pub fn tree_key(value: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(value)]).expect("canonical key")
}

/// Begins a typed write transaction bound to `manifest`.
pub async fn write_txn<'b, 'm>(
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

/// Begins a typed read transaction bound to `manifest`.
pub async fn read_txn<'b, 'm>(
    backend: &'b DeterministicBackend,
    manifest: &'m IndexManifest,
) -> ReadLogicalTxn<'m, <DeterministicBackend as Backend>::ReadTxn<'b>> {
    let raw = backend.begin_read().await.expect("begin read");
    ReadLogicalTxn::for_index(raw, manifest).expect("bind manifest")
}

/// Installs the tree's Tree Manifest and initial leaf root so fixtures can
/// grow the root shape from a committed empty root.
pub async fn create_committed_tree(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
) {
    let mut txn = write_txn(backend, manifest).await;
    tree_manifest::create_tree(&mut txn, key, 0)
        .await
        .expect("create tree");
    txn.commit().await.expect("commit tree");
}

/// Seeds the Index ID allocator, Index Name directory, and Manifest rows of
/// one named index holding `config`, committing each in its own transaction.
pub async fn seed_named_index(
    backend: &SharedBackend,
    name: &IndexName,
    logical_index_id: LogicalIndexId,
    lifecycle: IndexLifecycle,
    config: IndexConfig,
) -> IndexManifest {
    let manifest = IndexManifest::new(lifecycle, logical_index_id, config, [3; 32], vec![None])
        .expect("valid manifest");
    for (key, value) in [
        (
            LogicalKey::IndexIdAllocator,
            PersistentValue::IndexIdAllocator(IndexIdAllocator::new(logical_index_id.get())),
        ),
        (
            LogicalKey::IndexNameDirectory(name.clone()),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(logical_index_id)),
        ),
        (
            LogicalKey::Manifest(logical_index_id),
            PersistentValue::IndexManifest(manifest.clone()),
        ),
    ] {
        let raw = backend.begin_write().await.expect("begin write");
        let limits = backend.hard_limits();
        let budget = backend.admission_budget();
        let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);
        txn.put(key, value).await.expect("put lifecycle value");
        txn.commit().await.expect("commit lifecycle value");
    }
    manifest
}
