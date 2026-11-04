//! Logical Index lifecycle contract tests.

use std::num::NonZeroU32;

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, IndexName, LogicalIndexId, Metric,
    RuntimeConfig, SynopsisConfig, Value,
};
use ktann::storage::backend::{AdmissionBudget, Backend, Capabilities, WriteTxn};
use ktann::storage::keys;
use ktann::storage::values::{
    IndexIdAllocator, IndexLifecycle, IndexManifest, PersistentValue, ValueCodec, VectorRecord,
};
use ktann::storage::{ReadLogicalTxn, WriteLogicalTxn};

use support::builders::seed_named_index;
use support::{CommitFault, CommitOutcome, Durability, MemoryBackend, TestConfig};

#[allow(dead_code)]
mod support;

fn backend(config: TestConfig) -> MemoryBackend {
    MemoryBackend::with_test_config(config)
}

fn clear_config() -> TestConfig {
    TestConfig {
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        durability: Durability::Durable,
        ..TestConfig::default()
    }
}

fn no_clear_config() -> TestConfig {
    TestConfig {
        capabilities: Capabilities {
            transactional_clear_range: false,
        },
        durability: Durability::Durable,
        ..TestConfig::default()
    }
}

fn paged_config(page_mutations: usize) -> TestConfig {
    TestConfig {
        admission_budget: AdmissionBudget {
            max_mutations: page_mutations,
            max_mutation_bytes: 1 << 20,
            mutation_key_overhead_bytes: 0,
        },
        ..no_clear_config()
    }
}

fn config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2)
        .expect("valid dimension")
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
        ])
        .expect("valid fields")
}

fn config_with_dimension(dimension: usize) -> IndexConfig {
    IndexConfig::new(dimension, Metric::L2).expect("valid config")
}

fn make_runtime(backend: MemoryBackend) -> ktann::runtime::Runtime<MemoryBackend> {
    ktann::runtime::Runtime::new(
        backend,
        RuntimeConfig::default()
            .with_attempts(2, 4)
            .expect("valid attempts"),
    )
    .expect("test runs on a multi-thread runtime")
}

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("nonzero test id")
}

fn name(value: &str) -> IndexName {
    IndexName::new(value).expect("valid test name")
}

async fn read_manifest(backend: &MemoryBackend, name: &IndexName) -> Option<IndexManifest> {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    let entry = txn
        .get(keys::LogicalKey::IndexNameDirectory(name.clone()))
        .await
        .expect("read name");
    let Some(PersistentValue::IndexNameEntry(entry)) = entry else {
        return None;
    };
    match txn
        .get(keys::LogicalKey::Manifest(entry.logical_index_id()))
        .await
        .expect("read manifest")
    {
        Some(PersistentValue::IndexManifest(manifest)) => Some(manifest),
        _ => None,
    }
}

async fn read_allocator(backend: &MemoryBackend) -> u64 {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    match txn
        .get(keys::LogicalKey::IndexIdAllocator)
        .await
        .expect("read allocator")
    {
        Some(PersistentValue::IndexIdAllocator(allocator)) => allocator.high_water(),
        None => 0,
        Some(_) => panic!("wrong allocator value family"),
    }
}

async fn seed_allocator(backend: &MemoryBackend, high_water: u64) {
    let raw = backend.begin_write().await.expect("begin write");
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);
    txn.put(
        keys::LogicalKey::IndexIdAllocator,
        PersistentValue::IndexIdAllocator(IndexIdAllocator::new(high_water)),
    )
    .await
    .expect("put allocator");
    txn.commit().await.expect("commit allocator");
}

async fn seed_index_owned_keys(backend: &MemoryBackend, manifest: &IndexManifest, count: usize) {
    let codec = ValueCodec::for_index(manifest);
    for item in 0..count {
        let record_id = Bytes::copy_from_slice(format!("record-{item:03}").as_bytes());
        let raw_key = Bytes::copy_from_slice(
            &keys::record_key(manifest.logical_index_id(), &record_id).expect("record key"),
        );
        let raw_value = Bytes::copy_from_slice(
            &codec
                .encode(&PersistentValue::VectorRecord(VectorRecord::new(
                    record_id,
                    vec![1.0_f32, 2.0_f32],
                    vec![Value::I64(item as i64)],
                )))
                .expect("encode record"),
        );
        let mut raw = backend.begin_write().await.expect("begin write");
        raw.put(raw_key, raw_value).await.expect("put record");
        raw.commit().await.expect("commit record");
    }
}

async fn assert_drop_complete(backend: &MemoryBackend, name: &IndexName) {
    assert_eq!(read_manifest(backend, name).await, None);
    assert_eq!(backend.db_key_count(), 1, "only the allocator remains");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_open_and_drop_basic_contract() {
    let shared = backend(no_clear_config());
    let runtime = make_runtime(shared.clone());

    let index = runtime
        .create_index("docs", config())
        .await
        .expect("create index");
    assert_eq!(index.logical_index_id(), id(1));
    assert_eq!(index.config(), &config());
    assert_eq!(index.name().as_str(), "docs");

    let opened = runtime.open_index("docs").await.expect("open index");
    assert_eq!(opened.logical_index_id(), index.logical_index_id());
    assert_eq!(opened.config(), index.config());

    assert_eq!(
        runtime
            .create_index("docs", config_with_dimension(3))
            .await
            .expect_err("conflicting create")
            .kind(),
        ErrorKind::IndexAlreadyExists
    );

    for changed in [
        config().with_leaf_partition_entries(1, 512).unwrap(),
        config().with_leaf_partition_entries(64, 256).unwrap(),
        config().with_internal_partition_entries(1, 512).unwrap(),
        config().with_internal_partition_entries(64, 256).unwrap(),
    ] {
        assert_eq!(
            runtime
                .create_index("docs", changed)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::IndexAlreadyExists
        );
    }

    runtime.drop_index("docs").await.expect("drop index");
    assert_drop_complete(&shared, &name("docs")).await;
    assert_eq!(
        runtime
            .open_index("docs")
            .await
            .expect_err("open missing")
            .kind(),
        ErrorKind::IndexNotFound
    );
    runtime.drop_index("docs").await.expect("idempotent drop");

    let recreated = runtime
        .create_index("docs", config())
        .await
        .expect("recreate index");
    assert_eq!(recreated.logical_index_id(), id(2));

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_names_and_configs_fail_before_storage_work() {
    let shared = backend(no_clear_config());
    let runtime = make_runtime(shared.clone());

    assert_eq!(
        runtime
            .create_index("", config())
            .await
            .expect_err("empty name")
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        runtime
            .open_index("missing")
            .await
            .expect_err("missing index")
            .kind(),
        ErrorKind::IndexNotFound
    );
    assert_eq!(
        IndexConfig::new(0, Metric::L2)
            .expect_err("invalid config")
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(shared.db_key_count(), 0);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_retries_definite_aborts_without_reallocating() {
    let shared = backend(no_clear_config());
    shared
        .set_fault_plan(vec![CommitFault::Abort, CommitFault::Normal])
        .expect("set plan");
    let runtime = make_runtime(shared.clone());

    let index = runtime
        .create_index("docs", config())
        .await
        .expect("create retries");
    assert_eq!(index.logical_index_id(), id(1));
    assert_eq!(read_allocator(&shared).await, 1);
    let outcomes = shared
        .history()
        .into_iter()
        .map(|entry| entry.outcome)
        .collect::<Vec<_>>();
    assert_eq!(
        outcomes,
        vec![CommitOutcome::Aborted, CommitOutcome::Committed]
    );

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_applied_create_recovers_without_allocating_a_new_id() {
    let shared = backend(no_clear_config());
    shared
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("set plan");
    let runtime = make_runtime(shared.clone());

    let recovered = runtime
        .create_index("docs", config())
        .await
        .expect("recover applied create");
    assert_eq!(recovered.logical_index_id(), id(1));
    assert_eq!(read_allocator(&shared).await, 1);

    let identical = runtime
        .create_index("docs", config())
        .await
        .expect("identical retry");
    assert_eq!(identical.logical_index_id(), id(1));
    assert_eq!(read_allocator(&shared).await, 1);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_not_applied_create_never_reallocates_a_missing_name() {
    let shared = backend(no_clear_config());
    shared
        .set_fault_plan(vec![CommitFault::UnknownNotApplied])
        .expect("set plan");
    let runtime = make_runtime(shared.clone());

    assert_eq!(
        runtime
            .create_index("docs", config())
            .await
            .expect_err("unknown create")
            .kind(),
        ErrorKind::CommitOutcomeUnknown
    );
    assert_eq!(read_allocator(&shared).await, 0);

    let retried = runtime
        .create_index("docs", config())
        .await
        .expect("retry after unknown create");
    assert_eq!(retried.logical_index_id(), id(1));
    assert_eq!(read_allocator(&shared).await, 1);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_applied_create_reports_a_later_conflicting_config() {
    let shared = backend(no_clear_config());
    shared
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("set plan");
    let runtime = make_runtime(shared.clone());
    runtime
        .create_index("docs", config())
        .await
        .expect("applied create recovers");

    assert_eq!(
        runtime
            .create_index("docs", config_with_dimension(3))
            .await
            .expect_err("conflicting create")
            .kind(),
        ErrorKind::IndexAlreadyExists
    );
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_manifest_fails_open_and_create_closed_until_drop_completes() {
    let shared = backend(no_clear_config());
    let seeded = seed_named_index(
        &shared,
        &name("docs"),
        id(7),
        IndexLifecycle::Dropping,
        config(),
    )
    .await;
    let runtime = make_runtime(shared.clone());

    assert_eq!(
        runtime
            .open_index("docs")
            .await
            .expect_err("open dropping")
            .kind(),
        ErrorKind::IndexDropping
    );
    assert_eq!(
        runtime
            .create_index("docs", config())
            .await
            .expect_err("create dropping")
            .kind(),
        ErrorKind::IndexDropping
    );

    runtime.drop_index("docs").await.expect("resume dropping");
    assert_drop_complete(&shared, &name("docs")).await;
    assert_eq!(
        read_allocator(&shared).await,
        seeded.logical_index_id().get()
    );

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clear_range_drop_is_atomic_and_recovers_unknown_outcomes() {
    let shared = backend(clear_config());
    let seeded = seed_named_index(
        &shared,
        &name("docs"),
        id(5),
        IndexLifecycle::Dropping,
        config(),
    )
    .await;
    seed_index_owned_keys(&shared, &seeded, 20).await;
    shared
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("set plan");
    let runtime = make_runtime(shared.clone());

    runtime.drop_index("docs").await.expect("drop");
    assert_drop_complete(&shared, &name("docs")).await;
    assert_eq!(
        shared
            .history()
            .iter()
            .filter(|entry| entry.outcome == CommitOutcome::UnknownApplied)
            .count(),
        1
    );

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn point_delete_drop_is_bounded_resumable_and_restarts_after_unknown() {
    let shared = backend(paged_config(3));
    let seeded = seed_named_index(
        &shared,
        &name("docs"),
        id(9),
        IndexLifecycle::Dropping,
        config(),
    )
    .await;
    seed_index_owned_keys(&shared, &seeded, 11).await;
    let seeded_entries = shared.history().len();
    shared
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("set plan");
    let runtime = make_runtime(shared.clone());

    runtime.drop_index("docs").await.expect("paged drop");
    assert_drop_complete(&shared, &name("docs")).await;

    let history = shared.history();
    assert!(
        history[seeded_entries..]
            .iter()
            .any(|entry| entry.outcome == CommitOutcome::UnknownApplied)
    );
    assert!(
        history[seeded_entries..]
            .iter()
            .all(|entry| entry.mutations <= 3)
    );

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_recovers_when_the_initial_dropping_mark_was_not_applied() {
    let shared = backend(no_clear_config());
    seed_named_index(
        &shared,
        &name("docs"),
        id(4),
        IndexLifecycle::Active,
        config(),
    )
    .await;
    shared
        .set_fault_plan(vec![CommitFault::UnknownNotApplied])
        .expect("set plan");
    let runtime = make_runtime(shared.clone());

    runtime.drop_index("docs").await.expect("drop retries mark");
    assert_drop_complete(&shared, &name("docs")).await;
    assert_eq!(
        shared
            .history()
            .iter()
            .filter(|entry| entry.outcome == CommitOutcome::UnknownNotApplied)
            .count(),
        1
    );

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn logical_index_id_exhaustion_is_checked_and_preserves_allocator() {
    let shared = backend(no_clear_config());
    seed_allocator(&shared, u64::MAX).await;
    let runtime = make_runtime(shared.clone());

    assert_eq!(
        runtime
            .create_index("docs", config())
            .await
            .expect_err("exhausted allocator")
            .kind(),
        ErrorKind::IdExhausted
    );
    assert_eq!(read_allocator(&shared).await, u64::MAX);

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn durable_restart_reopens_created_and_dropped_lifecycle_states() {
    let shared = backend(clear_config());
    let runtime = make_runtime(shared.clone());
    runtime
        .create_index("docs", config())
        .await
        .expect("create");
    runtime.shutdown().await.expect("shutdown");

    let reopened_backend = shared.reopen();
    let runtime = make_runtime(reopened_backend.clone());
    let opened = runtime.open_index("docs").await.expect("reopen");
    assert_eq!(opened.logical_index_id(), id(1));

    runtime
        .drop_index("docs")
        .await
        .expect("drop after restart");
    assert_drop_complete(&reopened_backend, &name("docs")).await;

    let after_drop = reopened_backend.reopen();
    let runtime = make_runtime(after_drop);
    assert_eq!(
        runtime
            .open_index("docs")
            .await
            .expect_err("dropped state persisted")
            .kind(),
        ErrorKind::IndexNotFound
    );
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exhausted_unknown_budget_still_recovers_a_provable_completion() {
    let shared = backend(clear_config());
    let seeded = seed_named_index(
        &shared,
        &name("docs"),
        id(6),
        IndexLifecycle::Dropping,
        config(),
    )
    .await;
    seed_index_owned_keys(&shared, &seeded, 5).await;
    shared
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("set plan");
    let runtime = ktann::runtime::Runtime::new(
        shared.clone(),
        RuntimeConfig::default()
            .with_attempts(2, 1)
            .expect("one foreground attempt"),
    )
    .expect("runtime");

    runtime
        .drop_index("docs")
        .await
        .expect("completion is provable");
    assert_drop_complete(&shared, &name("docs")).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exhausted_unknown_budget_returns_unknown_when_drop_is_incomplete() {
    let shared = backend(no_clear_config());
    seed_named_index(
        &shared,
        &name("docs"),
        id(4),
        IndexLifecycle::Active,
        config(),
    )
    .await;
    shared
        .set_fault_plan(vec![CommitFault::UnknownNotApplied])
        .expect("set plan");
    let runtime = ktann::runtime::Runtime::new(
        shared.clone(),
        RuntimeConfig::default()
            .with_attempts(2, 1)
            .expect("one foreground attempt"),
    )
    .expect("runtime");

    assert_eq!(
        runtime
            .drop_index("docs")
            .await
            .expect_err("drop remains incomplete")
            .kind(),
        ErrorKind::CommitOutcomeUnknown
    );
    assert_eq!(
        read_manifest(&shared, &name("docs"))
            .await
            .expect("manifest remains")
            .lifecycle(),
        IndexLifecycle::Active
    );

    runtime
        .drop_index("docs")
        .await
        .expect("retry completes drop");
    assert_drop_complete(&shared, &name("docs")).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_open_and_drop_accept_tree_keys_and_bloom_synopses() {
    let shared = backend(no_clear_config());
    let runtime = make_runtime(shared.clone());
    let expected_distinct = NonZeroU32::new(100).expect("nonzero distinct count");
    let config = IndexConfig::new(4, Metric::Cosine)
        .expect("valid config")
        .with_fields(vec![
            FieldSchema::new("tenant", DataType::I64)
                .expect("valid field")
                .with_synopsis(SynopsisConfig::MinMaxBloom {
                    expected_distinct,
                    false_positive_rate: 0.01,
                })
                .expect("valid synopsis"),
            FieldSchema::new("tag", DataType::String).expect("valid field"),
        ])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree keys");

    let created = runtime
        .create_index("docs", config.clone())
        .await
        .expect("create bloom config");
    assert_eq!(created.config(), &config);
    let opened = runtime.open_index("docs").await.expect("open bloom config");
    assert_eq!(opened.config(), &config);
    runtime.drop_index("docs").await.expect("drop bloom config");
    assert_drop_complete(&shared, &name("docs")).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn point_delete_drop_resumes_from_prefix_after_restart() {
    let shared = backend(paged_config(3));
    let seeded = seed_named_index(
        &shared,
        &name("docs"),
        id(11),
        IndexLifecycle::Dropping,
        config(),
    )
    .await;
    seed_index_owned_keys(&shared, &seeded, 9).await;
    shared
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("set plan");
    let runtime = ktann::runtime::Runtime::new(
        shared.clone(),
        RuntimeConfig::default()
            .with_attempts(2, 1)
            .expect("one foreground attempt"),
    )
    .expect("runtime");

    assert_eq!(
        runtime
            .drop_index("docs")
            .await
            .expect_err("first attempt stays incomplete")
            .kind(),
        ErrorKind::CommitOutcomeUnknown
    );
    assert_eq!(
        read_manifest(&shared, &name("docs"))
            .await
            .expect("dropping manifest remains")
            .lifecycle(),
        IndexLifecycle::Dropping
    );
    runtime.shutdown().await.expect("shutdown");

    let reopened = shared.reopen();
    let runtime = make_runtime(reopened.clone());
    runtime
        .drop_index("docs")
        .await
        .expect("resume after restart");
    assert_drop_complete(&reopened, &name("docs")).await;
    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_recovers_when_the_initial_dropping_mark_was_applied() {
    let shared = backend(no_clear_config());
    seed_named_index(
        &shared,
        &name("docs"),
        id(8),
        IndexLifecycle::Active,
        config(),
    )
    .await;
    shared
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .expect("set plan");
    let runtime = make_runtime(shared.clone());

    runtime.drop_index("docs").await.expect("drop resumes");
    assert_drop_complete(&shared, &name("docs")).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn minimal_config_without_fields_lifecycle_works() {
    let shared = backend(clear_config());
    let runtime = make_runtime(shared.clone());
    let minimal = IndexConfig::new(1, Metric::InnerProduct).expect("minimal config");

    let created = runtime
        .create_index("minimal", minimal.clone())
        .await
        .expect("create minimal");
    assert_eq!(created.config(), &minimal);
    runtime.drop_index("minimal").await.expect("drop minimal");
    assert_drop_complete(&shared, &name("minimal")).await;

    runtime.shutdown().await.expect("shutdown");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_removes_only_the_named_index_owned_range() {
    let shared = backend(clear_config());
    let runtime = make_runtime(shared.clone());
    let first = runtime
        .create_index("first", config())
        .await
        .expect("create first");
    let second = runtime
        .create_index("second", config())
        .await
        .expect("create second");
    assert_eq!(first.logical_index_id(), id(1));
    assert_eq!(second.logical_index_id(), id(2));

    runtime.drop_index("first").await.expect("drop first");
    assert_eq!(
        runtime
            .open_index("first")
            .await
            .expect_err("first is gone")
            .kind(),
        ErrorKind::IndexNotFound
    );
    let second = runtime.open_index("second").await.expect("second survives");
    assert_eq!(second.logical_index_id(), id(2));

    runtime.shutdown().await.expect("shutdown");
}

#[test]
fn index_handle_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ktann::api::Index<MemoryBackend>>();
}

fn bulk_options() -> ktann::construction::ConstructionOptions {
    ktann::construction::ConstructionOptions {
        sample_items: 8,
        // Cover default wide internal partitions in the constructor reservation.
        memory_bytes: 64 * 1024 * 1024,
        scratch_bytes: 16 * 1024 * 1024,
    }
}

struct BulkSource(std::path::PathBuf);
impl BulkSource {
    fn new() -> (Self, ktann::bulk::InputSnapshot) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ktann-reservation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let input =
            ktann::bulk::InputSnapshot::create(&path, config(), 1024 * 1024, std::iter::empty())
                .unwrap();
        (Self(path), input)
    }
}
impl Drop for BulkSource {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bulk_reservation_reopens_blocks_serving_and_never_retargets_abort() {
    use ktann::api::BulkBuildStatus;
    for backend_config in [clear_config(), paged_config(4)] {
        let shared = backend(backend_config);
        let runtime = make_runtime(shared.clone());
        let (source, input) = BulkSource::new();
        let options = bulk_options();
        let job = runtime
            .start_bulk_build("bulk", &input, options)
            .await
            .unwrap();
        assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Preparing);
        assert_eq!(
            runtime.open_index("bulk").await.unwrap_err().kind(),
            ErrorKind::IndexBuilding
        );
        assert_eq!(
            runtime
                .create_index("bulk", config())
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::IndexBuilding
        );
        let repeated = runtime
            .start_bulk_build("bulk", &input, options)
            .await
            .unwrap();
        assert_eq!(job.logical_index_id(), repeated.logical_index_id());
        let mut other = options;
        other.sample_items += 1;
        assert_eq!(
            runtime
                .start_bulk_build("bulk", &input, other)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::IndexAlreadyExists
        );
        runtime.shutdown().await.unwrap();
        let reopened_runtime = make_runtime(shared.clone());
        let reopened = reopened_runtime.open_bulk_build("bulk").await.unwrap();
        assert_eq!(job.logical_index_id(), reopened.logical_index_id());
        assert_eq!(reopened.descriptor().input(), input.manifest());
        reopened.abort().await.unwrap();
        assert_drop_complete(&shared, &name("bulk")).await;
        assert_eq!(reopened.status().await.unwrap(), BulkBuildStatus::Aborted);
        assert!(source.0.join("data.bin").exists());
        let replacement = reopened_runtime
            .create_index("bulk", config())
            .await
            .unwrap();
        assert_ne!(replacement.logical_index_id(), reopened.logical_index_id());
        reopened.abort().await.unwrap();
        assert_eq!(
            reopened_runtime
                .open_index("bulk")
                .await
                .unwrap()
                .logical_index_id(),
            replacement.logical_index_id()
        );
        assert_eq!(read_allocator(&shared).await, 2);
        reopened_runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bulk_reservation_commit_recovery_and_validation_are_atomic() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownApplied,
        CommitFault::UnknownNotApplied,
    ] {
        let shared = backend(no_clear_config());
        let runtime = make_runtime(shared.clone());
        let (_source, input) = BulkSource::new();
        let options = bulk_options();
        let mut invalid = options;
        invalid.memory_bytes = 1;
        assert_eq!(
            runtime
                .start_bulk_build("bulk", &input, invalid)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(read_allocator(&shared).await, 0);
        shared.set_fault_plan(vec![fault]).unwrap();
        let started = runtime.start_bulk_build("bulk", &input, options).await;
        if matches!(fault, CommitFault::UnknownNotApplied) {
            assert_eq!(started.unwrap_err().kind(), ErrorKind::CommitOutcomeUnknown);
            assert_eq!(read_allocator(&shared).await, 0);
            assert_eq!(
                runtime.open_bulk_build("bulk").await.unwrap_err().kind(),
                ErrorKind::IndexNotFound
            );
        } else {
            let job = started.unwrap();
            assert_eq!(job.logical_index_id(), id(1));
            assert_eq!(
                runtime.open_bulk_build("bulk").await.unwrap().descriptor(),
                job.descriptor()
            );
            shared
                .set_fault_plan(vec![CommitFault::UnknownApplied])
                .unwrap();
            job.abort().await.unwrap();
        }
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bulk_controls_cancel_before_lifecycle_changes() {
    let shared = backend(no_clear_config());
    let runtime = make_runtime(shared.clone());
    let (_source, input) = BulkSource::new();
    let cancelled = tokio_util::sync::CancellationToken::new();
    cancelled.cancel();
    let control = ktann::api::OperationOptions::default().with_cancellation(cancelled);
    assert_eq!(
        runtime
            .start_bulk_build_with_control("bulk", &input, bulk_options(), control.clone())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(read_allocator(&shared).await, 0);
    let job = runtime
        .start_bulk_build("bulk", &input, bulk_options())
        .await
        .unwrap();
    assert_eq!(
        job.abort_with_control(control).await.unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(
        job.status().await.unwrap(),
        ktann::api::BulkBuildStatus::Preparing
    );
    job.abort().await.unwrap();
    assert_drop_complete(&shared, &name("bulk")).await;
    runtime.shutdown().await.unwrap();
}

/// Pauses one drop before its next snapshot, without delaying other runtimes.
#[derive(Clone)]
struct DropGate {
    memory: MemoryBackend,
    writes: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pause_write: usize,
    pause_read: bool,
    reached: std::sync::Arc<tokio::sync::Notify>,
    release: std::sync::Arc<tokio::sync::Notify>,
}

impl Backend for DropGate {
    type ReadTxn<'a> = support::MemoryReadTxn;
    type WriteTxn<'a> = support::MemoryWriteTxn<'a>;

    fn hard_limits(&self) -> ktann::storage::backend::HardLimits {
        self.memory.hard_limits()
    }

    fn admission_budget(&self) -> AdmissionBudget {
        self.memory.admission_budget()
    }

    fn capabilities(&self) -> Capabilities {
        self.memory.capabilities()
    }

    async fn begin_read(&self) -> ktann::api::Result<Self::ReadTxn<'_>> {
        if self.pause_read {
            self.reached.notify_one();
            self.release.notified().await;
        }
        self.memory.begin_read().await
    }

    async fn begin_write(&self) -> ktann::api::Result<Self::WriteTxn<'_>> {
        if self
            .writes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1
            == self.pause_write
        {
            self.reached.notify_one();
            self.release.notified().await;
        }
        self.memory.begin_write().await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_flight_drop_preserves_same_name_replacement() {
    // Exercise ordinary continuation, retry after an abort, and unknown final
    // deletion with both another loop attempt and exhausted-budget recovery.
    for cfg in [clear_config(), paged_config(32)] {
        for scenario in 0..4 {
            let shared = backend(cfg);
            let other = make_runtime(shared.clone());
            let original = other.create_index("docs", config()).await.expect("create");
            if scenario >= 2 {
                let raw = shared.begin_write().await.expect("begin");
                let mut txn = WriteLogicalTxn::bootstrap(
                    raw,
                    shared.hard_limits(),
                    shared.admission_budget(),
                );
                let manifest = read_manifest(&shared, &name("docs"))
                    .await
                    .expect("manifest");
                txn.put(
                    keys::LogicalKey::Manifest(original.logical_index_id()),
                    PersistentValue::IndexManifest(
                        manifest.with_lifecycle(IndexLifecycle::Dropping),
                    ),
                )
                .await
                .expect("mark dropping");
                txn.commit().await.expect("commit");
            }
            shared
                .set_fault_plan(vec![match scenario {
                    1 => CommitFault::Abort,
                    2 | 3 => CommitFault::UnknownApplied,
                    _ => CommitFault::Normal,
                }])
                .expect("fault plan");
            let gate = DropGate {
                memory: shared.clone(),
                writes: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                pause_write: if scenario == 3 { usize::MAX } else { 2 },
                pause_read: scenario == 3,
                reached: std::sync::Arc::new(tokio::sync::Notify::new()),
                release: std::sync::Arc::new(tokio::sync::Notify::new()),
            };
            let runtime = ktann::runtime::Runtime::new(
                gate.clone(),
                RuntimeConfig::default()
                    .with_attempts(2, if scenario == 3 { 1 } else { 4 })
                    .expect("attempts"),
            )
            .expect("runtime");
            let drop = tokio::spawn(async move {
                let result = runtime.drop_index("docs").await;
                runtime.shutdown().await.expect("shutdown");
                result
            });
            tokio::time::timeout(std::time::Duration::from_secs(10), gate.reached.notified())
                .await
                .expect("drop paused");
            if scenario >= 2 {
                assert!(read_manifest(&shared, &name("docs")).await.is_none());
                assert_eq!(
                    shared.history().last().expect("final deletion").outcome,
                    CommitOutcome::UnknownApplied
                );
            }
            other.drop_index("docs").await.expect("concurrent drop");
            let replacement = other
                .create_index("docs", config())
                .await
                .expect("recreate");
            assert_ne!(original.logical_index_id(), replacement.logical_index_id());
            let record = ktann::api::Record::new(
                Bytes::from_static(b"replacement"),
                vec![1.0, 2.0],
                vec![Value::I64(1)],
            )
            .expect("record");
            replacement.insert(record.clone()).await.expect("insert");
            gate.release.notify_one();
            drop.await.expect("join").expect("original drop complete");
            let opened = other
                .open_index("docs")
                .await
                .expect("replacement survives");
            assert_eq!(opened.logical_index_id(), replacement.logical_index_id());
            let stored = opened
                .get(record.id().clone(), ktann::api::GetOptions::default())
                .await
                .expect("get")
                .expect("record survives");
            assert_eq!(stored.vector(), record.vector());
            assert_eq!(stored.fields(), record.fields());
            other.shutdown().await.expect("shutdown");
        }
    }
}
