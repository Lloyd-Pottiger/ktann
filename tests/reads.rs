//! Point and batch Vector Record read contract tests.

use std::time::Instant;

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, GetOptions, IndexConfig, IndexName, LogicalIndexId,
    Metric, OperationOptions, PartitionKey, PayloadProjection, RuntimeConfig, Value,
};
use ktann::storage::WriteLogicalTxn;
use ktann::storage::backend::{Backend, HardLimits, WriteTxn};
use ktann::storage::keys::{self, TreeKey};
use ktann::storage::values::{
    IndexIdAllocator, IndexLifecycle, IndexManifest, IndexNameEntry, OpaquePayload,
    PersistentValue, RecordLocation, ValueCodec, VectorRecord,
};
use tokio_util::sync::CancellationToken;

use support::{DeterministicBackend, DeterministicConfig, SharedBackend};

#[allow(dead_code)]
mod support;

fn backend(config: DeterministicConfig) -> SharedBackend {
    SharedBackend::new(DeterministicBackend::new(config))
}

fn batch_config(max_batch_size: usize) -> DeterministicConfig {
    DeterministicConfig {
        max_batch_size,
        ..DeterministicConfig::default()
    }
}

fn key_limit_config(max_key_bytes: usize) -> DeterministicConfig {
    DeterministicConfig {
        hard_limits: HardLimits {
            max_key_bytes,
            ..DeterministicConfig::default().hard_limits
        },
        ..DeterministicConfig::default()
    }
}

fn config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields")
}

fn make_runtime(backend: SharedBackend) -> ktann::runtime::Runtime<SharedBackend> {
    ktann::runtime::Runtime::new(backend, RuntimeConfig::default()).expect("runtime is valid")
}

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("valid id")
}

fn name(value: &str) -> IndexName {
    IndexName::new(value).expect("valid name")
}

fn tree_key(bucket: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(bucket)]).expect("valid tree key")
}

async fn seed_named_index(
    backend: &SharedBackend,
    index_name: &IndexName,
    logical_index_id: LogicalIndexId,
    lifecycle: IndexLifecycle,
) -> IndexManifest {
    let manifest = IndexManifest::new(lifecycle, logical_index_id, config(), [3; 32], vec![None])
        .expect("valid manifest");
    for (key, value) in [
        (
            keys::LogicalKey::IndexIdAllocator,
            PersistentValue::IndexIdAllocator(IndexIdAllocator::new(logical_index_id.get())),
        ),
        (
            keys::LogicalKey::IndexNameDirectory(index_name.clone()),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(logical_index_id)),
        ),
        (
            keys::LogicalKey::Manifest(logical_index_id),
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

async fn seed_record(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    id: &[u8],
    vector: Vec<f32>,
    bucket: i64,
    payload: Option<&[u8]>,
) {
    let raw = backend.begin_write().await.expect("begin write");
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let mut txn = WriteLogicalTxn::for_index(raw, manifest, limits, budget).expect("bind index");
    let id = Bytes::copy_from_slice(id);
    txn.put(
        keys::LogicalKey::Record {
            index: manifest.logical_index_id(),
            id: id.clone(),
        },
        PersistentValue::VectorRecord(VectorRecord::new(
            id.clone(),
            vector,
            vec![Value::I64(bucket)],
        )),
    )
    .await
    .expect("put record");
    txn.put(
        keys::LogicalKey::Location {
            index: manifest.logical_index_id(),
            id: id.clone(),
        },
        PersistentValue::RecordLocation(RecordLocation::new(
            tree_key(bucket),
            PartitionKey::new(1).expect("valid partition key"),
        )),
    )
    .await
    .expect("put location");
    if let Some(payload) = payload {
        txn.put(
            keys::LogicalKey::Payload {
                index: manifest.logical_index_id(),
                id: id.clone(),
            },
            PersistentValue::OpaquePayload(
                OpaquePayload::new(Bytes::copy_from_slice(payload)).expect("valid payload"),
            ),
        )
        .await
        .expect("put payload");
    }
    txn.commit().await.expect("commit record");
}

async fn put_raw(backend: &SharedBackend, key: Vec<u8>, value: Vec<u8>) {
    let mut raw = backend.begin_write().await.expect("begin write");
    raw.put(Bytes::from(key), Bytes::from(value))
        .await
        .expect("put raw value");
    raw.commit().await.expect("commit raw value");
}

async fn put_manifest(backend: &SharedBackend, manifest: IndexManifest) {
    let raw = backend.begin_write().await.expect("begin write");
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);
    txn.put(
        keys::LogicalKey::Manifest(manifest.logical_index_id()),
        PersistentValue::IndexManifest(manifest),
    )
    .await
    .expect("put manifest");
    txn.commit().await.expect("commit manifest");
}

/// A Runtime opened on a seeded Active index with seeded records.
struct Fixture {
    runtime: ktann::runtime::Runtime<SharedBackend>,
    index: ktann::api::Index<SharedBackend>,
}

async fn fixture(shared: SharedBackend) -> Fixture {
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    seed_record(
        &shared,
        &manifest,
        b"alpha",
        vec![1.0_f32, 2.0],
        1,
        Some(b"a-payload"),
    )
    .await;
    seed_record(&shared, &manifest, b"beta", vec![3.0_f32, 4.0], 2, None).await;
    seed_record(
        &shared,
        &manifest,
        b"gamma",
        vec![5.0_f32, 6.0],
        3,
        Some(b""),
    )
    .await;
    let runtime = make_runtime(shared);
    let index = runtime.open_index("docs").await.expect("open index");
    Fixture { runtime, index }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_returns_canonical_record_and_closed_payload_projection() {
    let fixture = fixture(backend(DeterministicConfig::default())).await;

    let record = fixture
        .index
        .get(Bytes::from_static(b"alpha"), GetOptions::default())
        .await
        .expect("get succeeds")
        .expect("record exists");
    assert_eq!(record.id(), &Bytes::from_static(b"alpha"));
    assert_eq!(record.vector(), &[1.0_f32, 2.0]);
    assert_eq!(record.fields(), &[Value::I64(1)]);
    assert_eq!(record.payload(), &PayloadProjection::NotLoaded);

    let with_payload = fixture
        .index
        .get(
            Bytes::from_static(b"alpha"),
            GetOptions::default().with_payload(),
        )
        .await
        .expect("get succeeds")
        .expect("record exists");
    assert_eq!(
        with_payload.payload(),
        &PayloadProjection::Present(Bytes::from_static(b"a-payload"))
    );

    // A requested but absent payload is `Absent`; an existing empty payload is
    // still `Present`.
    let absent_payload = fixture
        .index
        .get(
            Bytes::from_static(b"beta"),
            GetOptions::default().with_payload(),
        )
        .await
        .expect("get succeeds")
        .expect("record exists");
    assert_eq!(absent_payload.payload(), &PayloadProjection::Absent);

    let empty_payload = fixture
        .index
        .get(
            Bytes::from_static(b"gamma"),
            GetOptions::default().with_payload(),
        )
        .await
        .expect("get succeeds")
        .expect("record exists");
    assert_eq!(
        empty_payload.payload(),
        &PayloadProjection::Present(Bytes::new())
    );

    // Absent Record IDs return `Ok(None)`.
    assert!(
        fixture
            .index
            .get(
                Bytes::from_static(b"missing"),
                GetOptions::default().with_payload()
            )
            .await
            .expect("get succeeds")
            .is_none()
    );

    fixture.runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_get_preserves_order_duplicates_and_absence() {
    let fixture = fixture(backend(DeterministicConfig::default())).await;

    let ids = vec![
        Bytes::from_static(b"missing"),
        Bytes::from_static(b"alpha"),
        Bytes::from_static(b"alpha"),
        Bytes::from_static(b"beta"),
    ];
    let records = fixture
        .index
        .batch_get(ids, GetOptions::default().with_payload())
        .await
        .expect("batch_get succeeds");
    assert_eq!(records.len(), 4);
    assert!(records[0].is_none());
    let alpha = records[1].as_ref().expect("alpha exists");
    assert_eq!(alpha.id(), &Bytes::from_static(b"alpha"));
    assert_eq!(
        alpha.payload(),
        &PayloadProjection::Present(Bytes::from_static(b"a-payload"))
    );
    let duplicate = records[2].as_ref().expect("duplicate alpha exists");
    assert_eq!(duplicate.id(), alpha.id());
    assert_eq!(duplicate.vector(), alpha.vector());
    assert_eq!(duplicate.fields(), alpha.fields());
    assert_eq!(duplicate.payload(), alpha.payload());
    let beta = records[3].as_ref().expect("beta exists");
    assert_eq!(beta.id(), &Bytes::from_static(b"beta"));
    assert_eq!(beta.payload(), &PayloadProjection::Absent);

    // An empty batch succeeds with an empty result without backend access.
    assert!(
        fixture
            .index
            .batch_get(Vec::new(), GetOptions::default())
            .await
            .expect("empty batch succeeds")
            .is_empty()
    );

    fixture.runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_get_decodes_mixed_tree_keys() {
    let fixture = fixture(backend(DeterministicConfig::default())).await;

    let ids = vec![
        Bytes::from_static(b"gamma"),
        Bytes::from_static(b"alpha"),
        Bytes::from_static(b"beta"),
    ];
    let records = fixture
        .index
        .batch_get(ids, GetOptions::default())
        .await
        .expect("batch_get succeeds");
    let buckets = records
        .iter()
        .map(|record| record.as_ref().expect("record exists").fields()[0].clone())
        .collect::<Vec<_>>();
    assert_eq!(buckets, vec![Value::I64(3), Value::I64(1), Value::I64(2)]);

    fixture.runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reads_use_one_consistent_backend_snapshot() {
    let shared = backend(DeterministicConfig::default());
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    seed_record(&shared, &manifest, b"alpha", vec![1.0_f32, 2.0], 1, None).await;
    let runtime = make_runtime(shared.clone());
    let index = runtime.open_index("docs").await.expect("open index");

    // An uncommitted Record Group is invisible to the read snapshot.
    let raw = shared.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        shared.hard_limits(),
        shared.admission_budget(),
    )
    .expect("bind index");
    let id = Bytes::from_static(b"pending");
    txn.put(
        keys::LogicalKey::Record {
            index: manifest.logical_index_id(),
            id: id.clone(),
        },
        PersistentValue::VectorRecord(VectorRecord::new(
            id.clone(),
            vec![7.0_f32, 8.0],
            vec![Value::I64(7)],
        )),
    )
    .await
    .expect("put pending record");
    txn.put(
        keys::LogicalKey::Location {
            index: manifest.logical_index_id(),
            id: id.clone(),
        },
        PersistentValue::RecordLocation(RecordLocation::new(
            tree_key(7),
            PartitionKey::new(1).expect("valid partition key"),
        )),
    )
    .await
    .expect("put pending location");

    assert!(
        index
            .get(id.clone(), GetOptions::default())
            .await
            .expect("get succeeds")
            .is_none(),
        "an uncommitted Record Group is not visible to a fresh snapshot"
    );

    txn.commit().await.expect("commit pending record");
    let record = index
        .get(id, GetOptions::default())
        .await
        .expect("get succeeds")
        .expect("committed record is visible");
    assert_eq!(record.vector(), &[7.0_f32, 8.0]);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_manifest_fails_reads_closed() {
    let shared = backend(DeterministicConfig::default());
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    seed_record(&shared, &manifest, b"alpha", vec![1.0_f32, 2.0], 1, None).await;
    let runtime = make_runtime(shared.clone());
    let index = runtime.open_index("docs").await.expect("open index");

    put_manifest(
        &shared,
        manifest
            .with_lifecycle(IndexLifecycle::Dropping)
            .expect("dropping"),
    )
    .await;

    let get = index
        .get(Bytes::from_static(b"alpha"), GetOptions::default())
        .await
        .expect_err("dropping index rejects get");
    assert_eq!(get.kind(), ErrorKind::IndexDropping);
    let batch = index
        .batch_get(vec![Bytes::from_static(b"alpha")], GetOptions::default())
        .await
        .expect_err("dropping index rejects batch_get");
    assert_eq!(batch.kind(), ErrorKind::IndexDropping);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_index_fails_reads_closed() {
    let shared = backend(DeterministicConfig::default());
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    seed_record(&shared, &manifest, b"alpha", vec![1.0_f32, 2.0], 1, None).await;
    let runtime = make_runtime(shared.clone());
    let index = runtime.open_index("docs").await.expect("open index");

    runtime.drop_index("docs").await.expect("drop index");

    let get = index
        .get(Bytes::from_static(b"alpha"), GetOptions::default())
        .await
        .expect_err("dropped index rejects get");
    assert_eq!(get.kind(), ErrorKind::IndexNotFound);
    let batch = index
        .batch_get(vec![Bytes::from_static(b"alpha")], GetOptions::default())
        .await
        .expect_err("dropped index rejects batch_get");
    assert_eq!(batch.kind(), ErrorKind::IndexNotFound);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unsupported_manifest_format_fails_reads_closed() {
    let shared = backend(DeterministicConfig::default());
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    seed_record(&shared, &manifest, b"alpha", vec![1.0_f32, 2.0], 1, None).await;
    let runtime = make_runtime(shared.clone());
    let index = runtime.open_index("docs").await.expect("open index");

    // Rewrite the Manifest bytes with an unsupported whole-format version.
    let mut bytes = ValueCodec::bootstrap()
        .encode(&PersistentValue::IndexManifest(manifest.clone()))
        .expect("encode manifest");
    bytes[2..4].copy_from_slice(&2_u16.to_be_bytes());
    put_raw(
        &shared,
        keys::manifest_key(manifest.logical_index_id()),
        bytes,
    )
    .await;

    let error = index
        .get(Bytes::from_static(b"alpha"), GetOptions::default())
        .await
        .expect_err("unsupported manifest rejects get");
    assert_eq!(error.kind(), ErrorKind::UnsupportedFormat);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manifest_identity_mismatch_is_corruption() {
    let shared = backend(DeterministicConfig::default());
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    seed_record(&shared, &manifest, b"alpha", vec![1.0_f32, 2.0], 1, None).await;
    let runtime = make_runtime(shared.clone());
    let index = runtime.open_index("docs").await.expect("open index");

    // The same Logical Index ID persists with a different immutable config.
    let different = IndexManifest::new(
        IndexLifecycle::Active,
        manifest.logical_index_id(),
        IndexConfig::new(3, Metric::L2)
            .expect("valid dimension")
            .with_fields(vec![
                FieldSchema::new("bucket", DataType::I64).expect("valid field"),
            ])
            .expect("valid fields")
            .with_tree_key_fields(vec![FieldId(0)])
            .expect("valid tree key fields"),
        [4; 32],
        vec![None],
    )
    .expect("valid manifest");
    put_manifest(&shared, different).await;

    let error = index
        .get(Bytes::from_static(b"alpha"), GetOptions::default())
        .await
        .expect_err("identity mismatch is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn partial_record_groups_are_corruption() {
    let shared = backend(DeterministicConfig::default());
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    let runtime = make_runtime(shared.clone());
    let index = runtime.open_index("docs").await.expect("open index");

    // Record without Location.
    seed_record(
        &shared,
        &manifest,
        b"record-only",
        vec![1.0_f32, 2.0],
        1,
        None,
    )
    .await;
    delete_raw(
        &shared,
        keys::location_key(
            manifest.logical_index_id(),
            &Bytes::from_static(b"record-only"),
        )
        .expect("location key"),
    )
    .await;
    let error = index
        .get(Bytes::from_static(b"record-only"), GetOptions::default())
        .await
        .expect_err("record without location is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    // Location without Record.
    let raw = shared.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        shared.hard_limits(),
        shared.admission_budget(),
    )
    .expect("bind index");
    txn.put(
        keys::LogicalKey::Location {
            index: manifest.logical_index_id(),
            id: Bytes::from_static(b"location-only"),
        },
        PersistentValue::RecordLocation(RecordLocation::new(
            tree_key(1),
            PartitionKey::new(1).expect("valid partition key"),
        )),
    )
    .await
    .expect("put location");
    txn.commit().await.expect("commit location");
    let error = index
        .get(Bytes::from_static(b"location-only"), GetOptions::default())
        .await
        .expect_err("location without record is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    // Payload without Record, detected when the payload is requested.
    let raw = shared.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        shared.hard_limits(),
        shared.admission_budget(),
    )
    .expect("bind index");
    txn.put(
        keys::LogicalKey::Payload {
            index: manifest.logical_index_id(),
            id: Bytes::from_static(b"payload-only"),
        },
        PersistentValue::OpaquePayload(
            OpaquePayload::new(Bytes::from_static(b"p")).expect("payload"),
        ),
    )
    .await
    .expect("put payload");
    txn.commit().await.expect("commit payload");
    let error = index
        .get(
            Bytes::from_static(b"payload-only"),
            GetOptions::default().with_payload(),
        )
        .await
        .expect_err("payload without record is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_key_value_identity_mismatch_is_corruption() {
    let shared = backend(DeterministicConfig::default());
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    let runtime = make_runtime(shared.clone());
    let index = runtime.open_index("docs").await.expect("open index");

    // The value encodes one Record ID but the key addresses another.
    let value = ValueCodec::for_index(&manifest)
        .encode(&PersistentValue::VectorRecord(VectorRecord::new(
            Bytes::from_static(b"other"),
            vec![1.0_f32, 2.0],
            vec![Value::I64(1)],
        )))
        .expect("encode record");
    put_raw(
        &shared,
        keys::record_key(manifest.logical_index_id(), &Bytes::from_static(b"shown"))
            .expect("record key"),
        value,
    )
    .await;

    let error = index
        .get(Bytes::from_static(b"shown"), GetOptions::default())
        .await
        .expect_err("key/value identity mismatch is corruption");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_ids_fail_validation_with_position() {
    let fixture = fixture(backend(DeterministicConfig::default())).await;

    let error = fixture
        .index
        .get(Bytes::new(), GetOptions::default())
        .await
        .expect_err("empty id is invalid");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(error.position(), None);

    let error = fixture
        .index
        .get(Bytes::from(vec![0_u8; 257]), GetOptions::default())
        .await
        .expect_err("oversized id is invalid");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);

    let error = fixture
        .index
        .batch_get(
            vec![Bytes::from_static(b"alpha"), Bytes::new()],
            GetOptions::default(),
        )
        .await
        .expect_err("invalid batch id fails the whole batch");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(error.position(), Some(1));

    fixture.runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_reads_enforce_backend_batch_and_key_limits() {
    // The backend caps one batch_get at six keys: two IDs with payloads fit
    // exactly, three IDs with payloads exceed the cap.
    let shared = backend(batch_config(6));
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    for (record_id, bucket) in [("a", 1), ("b", 2), ("c", 3)] {
        seed_record(
            &shared,
            &manifest,
            record_id.as_bytes(),
            vec![1.0_f32, 2.0],
            bucket,
            Some(b"payload"),
        )
        .await;
    }
    let runtime = make_runtime(shared);
    let index = runtime.open_index("docs").await.expect("open index");

    let boundary = index
        .batch_get(
            vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")],
            GetOptions::default().with_payload(),
        )
        .await
        .expect("exactly-at-limit batch succeeds");
    assert_eq!(boundary.len(), 2);
    assert!(boundary.iter().all(Option::is_some));

    let error = index
        .batch_get(
            vec![
                Bytes::from_static(b"a"),
                Bytes::from_static(b"b"),
                Bytes::from_static(b"c"),
            ],
            GetOptions::default().with_payload(),
        )
        .await
        .expect_err("over-limit batch fails");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);

    runtime.shutdown().await.expect("shutdown");

    // Encoded keys that exceed the backend key ceiling fail with
    // LimitExceeded before any value is read.
    let shared = backend(key_limit_config(16));
    let manifest = seed_named_index(&shared, &name("docs"), id(1), IndexLifecycle::Active).await;
    seed_record(&shared, &manifest, b"abc", vec![1.0_f32, 2.0], 1, None).await;
    let runtime = make_runtime(shared);
    let index = runtime.open_index("docs").await.expect("open index");
    let error = index
        .get(Bytes::from_static(b"abcd"), GetOptions::default())
        .await
        .expect_err("oversized encoded key fails");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    assert!(
        index
            .get(Bytes::from_static(b"abc"), GetOptions::default())
            .await
            .expect("within-limit key reads")
            .is_some()
    );
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_and_deadline_fail_reads_before_work() {
    let fixture = fixture(backend(DeterministicConfig::default())).await;

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let options = OperationOptions::default().with_cancellation(cancellation);
    let error = fixture
        .index
        .get_with_control(Bytes::from_static(b"alpha"), GetOptions::default(), options)
        .await
        .expect_err("cancelled get fails");
    assert_eq!(error.kind(), ErrorKind::Cancelled);

    let options = OperationOptions::default().with_deadline(Instant::now());
    let error = fixture
        .index
        .batch_get_with_control(
            vec![Bytes::from_static(b"alpha")],
            GetOptions::default(),
            options,
        )
        .await
        .expect_err("expired deadline fails");
    assert_eq!(error.kind(), ErrorKind::DeadlineExceeded);

    fixture.runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reads_after_shutdown_fail_closed() {
    let fixture = fixture(backend(DeterministicConfig::default())).await;
    fixture.runtime.shutdown().await.expect("shutdown");

    let error = fixture
        .index
        .get(Bytes::from_static(b"alpha"), GetOptions::default())
        .await
        .expect_err("closed runtime rejects get");
    assert_eq!(error.kind(), ErrorKind::RuntimeClosed);
    let error = fixture
        .index
        .batch_get(vec![Bytes::from_static(b"alpha")], GetOptions::default())
        .await
        .expect_err("closed runtime rejects batch_get");
    assert_eq!(error.kind(), ErrorKind::RuntimeClosed);
}

async fn delete_raw(backend: &SharedBackend, key: Vec<u8>) {
    let mut raw = backend.begin_write().await.expect("begin write");
    raw.delete(Bytes::from(key))
        .await
        .expect("delete raw value");
    raw.commit().await.expect("commit raw delete");
}
