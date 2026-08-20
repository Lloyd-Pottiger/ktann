//! Deterministic binary K-means split training contract tests (#83).

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, LogicalIndexId, Metric, PartitionKey,
    Value,
};
use ktann::maintenance::training::{SplitCentroids, train_split_centroids};
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexLifecycle, IndexManifest, LeafEntry, PartitionHeader, PartitionState,
    PersistentValue, VectorRecord,
};
use ktann::storage::{ReadLogicalTxn, WriteLogicalTxn, tree_manifest};

use support::DeterministicBackend;

#[allow(dead_code)]
mod support;

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
}

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

fn config(metric: Metric) -> IndexConfig {
    IndexConfig::new(1, metric)
        .expect("valid config")
        .with_fields(vec![FieldSchema::new("a", DataType::I64).expect("field")])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
}

/// A one-dimensional L2 index: rotation is the identity at dimension 1, so
/// routing-space vectors equal the original vectors and expected centroids are
/// plain arithmetic means.
fn manifest() -> IndexManifest {
    IndexManifest::new(
        IndexLifecycle::Active,
        id(7),
        config(Metric::L2),
        [7; 32],
        vec![None],
    )
    .expect("valid manifest")
}

/// A one-dimensional Cosine index: preprocessing normalizes each record to
/// `[-1.0]` or `[1.0]` and the dimension-1 rotation is the identity, so the
/// trained centroids reveal whether normalization ran.
fn cosine_manifest() -> IndexManifest {
    IndexManifest::new(
        IndexLifecycle::Active,
        id(7),
        config(Metric::Cosine),
        [7; 32],
        vec![None],
    )
    .expect("valid manifest")
}

fn tree_key(value: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(value)]).expect("canonical key")
}

fn rid(value: u8) -> Bytes {
    Bytes::copy_from_slice(&[b'r', value])
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

async fn train(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    source: PartitionKey,
) -> ktann::api::Result<SplitCentroids> {
    train_split_centroids(&mut read_txn(backend, manifest).await, key, source).await
}

/// A dimension-1 Vector Record holding `vector`.
fn record(record_id: &Bytes, vector: f32) -> (LogicalKey, PersistentValue) {
    let key = LogicalKey::Record {
        index: id(7),
        id: record_id.clone(),
    };
    let value = PersistentValue::VectorRecord(VectorRecord::new(
        record_id.clone(),
        vec![vector],
        vec![Value::I64(0)],
    ));
    (key, value)
}

/// A dimension-1 Leaf Entry in `partition`; the all-zero code is the canonical
/// encoding of the zero vector, matching the established membership fixture.
/// Training reads only the Record ID projection, not the code.
fn leaf_entry(
    tree_key: &TreeKey,
    partition: PartitionKey,
    record_id: &Bytes,
) -> (LogicalKey, PersistentValue) {
    let key = LogicalKey::LeafEntry {
        index: id(7),
        tree_key: tree_key.clone(),
        partition,
        id: record_id.clone(),
    };
    let value = PersistentValue::LeafEntry(LeafEntry::new(
        record_id.clone(),
        vec![Value::I64(0)],
        Bytes::from_static(&[0; 14]),
    ));
    (key, value)
}

/// Commits typed values in chunks that stay inside the test backend's
/// admission budget.
async fn write_values(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    values: Vec<(LogicalKey, PersistentValue)>,
) {
    for chunk in values.chunks(800) {
        let mut txn = write_txn(backend, manifest).await;
        for (key, value) in chunk {
            txn.put(key.clone(), value.clone()).await.expect("put");
        }
        txn.commit().await.expect("commit");
    }
}

/// Installs the tree's Tree Manifest and initial leaf root (Partition Key 1).
async fn create_committed_tree(
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

/// Seeds a leaf source on Partition Key 1: one Vector Record and one Leaf
/// Entry per `(Record ID, vector)` pair.
async fn seed_leaf_source(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
    records: &[(Bytes, f32)],
) {
    create_committed_tree(backend, manifest, key).await;
    let mut values = Vec::new();
    for (record_id, vector) in records {
        values.push(record(record_id, *vector));
        values.push(leaf_entry(key, pk(1), record_id));
    }
    write_values(backend, manifest, values).await;
}

#[tokio::test]
async fn leaf_training_trains_from_the_original_vector_records() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_leaf_source(
        &backend,
        &manifest,
        &key,
        &[(rid(0), 0.0), (rid(1), 1.0), (rid(2), 2.0), (rid(3), 10.0)],
    )
    .await;

    // Source mean 3.25 seeds 10.0; its farthest partner is 0.0. The balanced
    // assignment {10, 2}|{0, 1} is immediately stable with means 6.0 and 0.5.
    let trained = train(&backend, &manifest, &key, pk(1))
        .await
        .expect("trained");
    assert_eq!(trained.left().components(), &[6.0]);
    assert_eq!(trained.right().components(), &[0.5]);

    // Re-reading the same committed state reproduces the identical pair.
    let retrained = train(&backend, &manifest, &key, pk(1))
        .await
        .expect("retrained");
    assert_eq!(trained, retrained);
}

#[tokio::test]
async fn cosine_leaf_training_applies_metric_preprocessing() {
    let backend = DeterministicBackend::default();
    let manifest = cosine_manifest();
    let key = tree_key(1);
    seed_leaf_source(
        &backend,
        &manifest,
        &key,
        &[(rid(0), 2.0), (rid(1), 4.0), (rid(2), -3.0)],
    )
    .await;

    // Normalization maps 2.0 and 4.0 to 1.0 and -3.0 to -1.0; unnormalized
    // training would instead produce -3.0 and 3.0.
    let trained = train(&backend, &manifest, &key, pk(1))
        .await
        .expect("trained");
    assert_eq!(trained.left().components(), &[-1.0]);
    assert_eq!(trained.right().components(), &[1.0]);
}

#[tokio::test]
async fn internal_training_reads_child_centroids_without_vector_records() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    let mut values = vec![(
        LogicalKey::Header {
            index: id(7),
            tree_key: key.clone(),
            partition: pk(1),
        },
        PersistentValue::PartitionHeader(
            PartitionHeader::new(2, 3, 0, PartitionState::Splitting).expect("header"),
        ),
    )];
    // No Vector Records exist at all: internal training must read only the
    // Child Entry centroids.
    for (child, centroid) in [(pk(2), 0.0), (pk(3), 2.0), (pk(4), 10.0)] {
        values.push((
            LogicalKey::ChildEntry {
                index: id(7),
                tree_key: key.clone(),
                partition: pk(1),
                child,
            },
            PersistentValue::ChildEntry(ChildEntry::new(child, vec![centroid])),
        ));
    }
    write_values(&backend, &manifest, values).await;

    // Source mean 4.0 seeds 10.0, whose farthest partner is 0.0; the balanced
    // assignment {10}|{0, 2} is immediately stable with means 10.0 and 1.0.
    let trained = train(&backend, &manifest, &key, pk(1))
        .await
        .expect("trained");
    assert_eq!(trained.left().components(), &[10.0]);
    assert_eq!(trained.right().components(), &[1.0]);
}

#[tokio::test]
async fn an_absent_source_vector_record_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    let mut values = Vec::new();
    for (record_id, vector) in [(rid(0), 0.0), (rid(1), 1.0), (rid(2), 2.0)] {
        if record_id != rid(1) {
            values.push(record(&record_id, vector));
        }
        values.push(leaf_entry(&key, pk(1), &record_id));
    }
    write_values(&backend, &manifest, values).await;

    let error = train(&backend, &manifest, &key, pk(1))
        .await
        .expect_err("absent Vector Record");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    // Training holds no locks and changes nothing: adding the missing record
    // makes the same source trainable without any repair.
    write_values(&backend, &manifest, vec![record(&rid(1), 1.0)]).await;
    let trained = train(&backend, &manifest, &key, pk(1))
        .await
        .expect("trained after the record appears");
    assert!(trained.left().components().iter().all(|c| c.is_finite()));
}

#[tokio::test]
async fn a_missing_source_header_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    let error = train(&backend, &manifest, &key, pk(1))
        .await
        .expect_err("missing source Header");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn an_empty_leaf_source_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;

    let error = train(&backend, &manifest, &key, pk(1))
        .await
        .expect_err("empty leaf source");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn malformed_leaf_entry_bytes_are_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_leaf_source(
        &backend,
        &manifest,
        &key,
        &[(rid(0), 0.0), (rid(1), 1.0), (rid(2), 2.0)],
    )
    .await;

    // Overwrite one Leaf Entry with garbage bytes through the raw seam.
    let mut raw = backend.begin_write().await.expect("begin write");
    raw.put(
        Bytes::from(keys::leaf_entry_key(id(7), &key, pk(1), &rid(1)).expect("key")),
        Bytes::from(vec![0xde, 0xad, 0xbe, 0xef]),
    )
    .await
    .expect("raw put");
    raw.commit().await.expect("commit");

    let error = train(&backend, &manifest, &key, pk(1))
        .await
        .expect_err("garbage Leaf Entry");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn training_loads_the_complete_source_across_pages_and_batches() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    // 1,100 entries exceed one 1,024-item scan page and one 128-record load
    // batch. IDs sort canonically in vector order; exact halves keep every
    // expected mean exactly representable.
    let records: Vec<(Bytes, f32)> = (0..1_100_u32)
        .map(|index| {
            (
                Bytes::from(format!("r{index:04}").into_bytes()),
                index as f32,
            )
        })
        .collect();
    seed_leaf_source(&backend, &manifest, &key, &records).await;

    // The source mean 549.5 ties between entries 0 and 1099; the smaller ID
    // seeds left, so the stable balanced split is the lower half against the
    // upper half.
    let trained = train(&backend, &manifest, &key, pk(1))
        .await
        .expect("trained");
    assert_eq!(trained.left().components(), &[274.5]);
    assert_eq!(trained.right().components(), &[824.5]);
}

#[tokio::test]
async fn an_unbound_transaction_is_rejected() {
    let backend = DeterministicBackend::default();
    let key = tree_key(1);

    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    let error = train_split_centroids(&mut txn, &key, pk(1))
        .await
        .expect_err("unbound transaction");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}
