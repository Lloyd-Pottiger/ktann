//! Focused contract checks against a temporary local RocksDB database.
//!
//! Every test here owns one `tempfile` directory that is removed when the test
//! finishes, so databases are isolated from each other and from any caller-owned
//! database, and cleanup never depends on a shared path. Each adapter binds a
//! distinct Backend Namespace, and every test writes a bounded number of small
//! keys and values.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, InsertOutcome, Mutation, ReadOps, ScanLimits, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_rocksdb::{BackendNamespace, RocksDbBackend, RocksDbConfig};
use rocksdb::{MemtableFactory, OptimisticTransactionDB, Options, SliceTransform};

#[path = "../../tests/support/backend_contract.rs"]
mod shared_backend_contract;
mod support;

use shared_backend_contract::{BackendHarness, Fault, FaultInjection, RestartMode};
use support::{key, open_database, range};

/// Adapts a [`RocksDbBackend`] to the shared harness seam.
///
/// RocksDB cannot stage a controlled commit outcome, so fault injection is
/// declared unavailable. Backend restart is likewise declared unsupported
/// rather than silently skipped: the only honest restart is dropping the one
/// live database handle and reopening the path, but RocksDB locks the path for
/// the handle's whole lifetime and the unchanged shared suite keeps using this
/// harness after `restart()`. The two-phase `rocksdb_durability` binary proves
/// committed data survives a fresh process, and this file's reopen check proves
/// visibility through a new adapter after an orderly close.
struct RocksDbHarness {
    backend: RocksDbBackend,
}

impl BackendHarness for RocksDbHarness {
    type Backend = RocksDbBackend;

    fn backend(&self) -> &RocksDbBackend {
        &self.backend
    }

    fn fault_injection(&self) -> FaultInjection {
        FaultInjection::Unavailable
    }

    fn inject_fault(&self, _fault: Fault) {
        unreachable!("RocksDB cannot stage controlled commit faults");
    }

    fn restart_mode(&self) -> RestartMode {
        RestartMode::Unsupported
    }

    fn restart(&self) -> Self {
        unreachable!("RocksDB durability requires the external two-phase durability test");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn rocksdb_adapter_preserves_the_backend_contract() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let database_path = directory.path().join("database");
    let primary_namespace = BackendNamespace::new("ktann-issue-34-contract").expect("namespace");
    let isolated_namespace =
        BackendNamespace::new("ktann-issue-34-contract-isolated").expect("namespace");

    {
        let database = Arc::new(open_database(&database_path));
        let config = RocksDbConfig::default()
            .with_blocking_resource_limit(64)
            .expect("blocking limit");
        let primary =
            RocksDbBackend::with_config(Arc::clone(&database), primary_namespace.clone(), config);
        let isolated = RocksDbBackend::new(Arc::clone(&database), isolated_namespace.clone());

        // Adapter-declared facts are asserted here; the shared suite checks the
        // rest of the common contract.
        assert_eq!(primary.config().blocking_resource_limit(), 64);
        assert_eq!(primary.admission_budget().max_mutations, 10_000);
        assert_eq!(primary.admission_budget().max_mutation_bytes, 1 << 20);
        assert_eq!(primary.hard_limits().max_value_bytes, u32::MAX as usize);
        assert!(!primary.capabilities().transactional_clear_range);

        let harness = RocksDbHarness { backend: primary };
        shared_backend_contract::run_suite(&harness).await;
        let primary = harness.backend;

        // Adapter-specific: two namespaces over one database are isolated.
        let mut write = primary.begin_write().await.expect("begin write");
        write
            .put(key(b"shared"), key(b"primary"))
            .await
            .expect("put");
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

        // Adapter-specific: a scan page is capped at the adapter's 80 KiB byte
        // ceiling even when the caller requests an unbounded byte limit.
        let mut seed = primary.begin_write().await.expect("begin scan seed");
        let large_value = Bytes::from(vec![7; 50_000]);
        seed.put(key(b"large/a"), large_value.clone())
            .await
            .expect("seed large a");
        seed.put(key(b"large/b"), large_value)
            .await
            .expect("seed large b");
        seed.commit().await.expect("commit scan seed");
        let mut scan = primary.begin_read().await.expect("begin scan");
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
            b"large/a\x00",
        );

        // Adapter-specific: exceeding the declared admission budget is rejected.
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
        drop(scan);
        drop(isolated_read);
        primary.shutdown().await;
        isolated.shutdown().await;
    }

    // Adapter-specific: committed data survives a real reopen at the same path.
    let reopened = RocksDbBackend::new(open_database(&database_path), primary_namespace);
    let mut durable = reopened.begin_read().await.expect("begin durable read");
    assert_eq!(
        durable.get(key(b"shared")).await.expect("durable get"),
        Some(key(b"primary")),
    );
    drop(durable);
    reopened.shutdown().await;

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
    drop(isolated_durable);
    isolated.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_unique_insert_conflicts_instead_of_overwriting() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let namespace = BackendNamespace::new("ktann-issue-34-insert-conflict").expect("namespace");
    let backend = RocksDbBackend::new(open_database(directory.path()), namespace);

    let mut first = backend.begin_write().await.expect("begin first");
    let mut second = backend.begin_write().await.expect("begin second");
    assert_eq!(
        first
            .insert(key(b"unique"), key(b"1"))
            .await
            .expect("first insert"),
        InsertOutcome::Inserted,
    );
    assert_eq!(
        second
            .insert(key(b"unique"), key(b"2"))
            .await
            .expect("second insert"),
        InsertOutcome::Inserted,
    );

    first.commit().await.expect("first commit wins");
    // The loser observes RocksDB's real optimistic-conflict error, classified
    // as a retryable abort rather than a silent overwrite.
    let error = second.commit().await.expect_err("second insert conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    let mut read = backend.begin_read().await.expect("begin read");
    assert_eq!(
        read.get(key(b"unique")).await.expect("get"),
        Some(key(b"1"))
    );
    drop(read);
    backend.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn open_snapshot_paginates_consistently_across_concurrent_commits() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let namespace = BackendNamespace::new("ktann-issue-34-snapshot-pages").expect("namespace");
    let backend = RocksDbBackend::new(open_database(directory.path()), namespace);

    {
        let mut seed = backend.begin_write().await.expect("begin seed");
        for (suffix, value) in ["a", "b", "c", "d", "e", "f"].iter().zip(1_u8..=6) {
            seed.put(
                Bytes::from(format!("snap/{suffix}").into_bytes()),
                Bytes::from(vec![value]),
            )
            .await
            .expect("seed put");
        }
        seed.commit().await.expect("commit seed");
    }

    // The snapshot opens before the concurrent commit, so every page below
    // must observe the pre-commit state.
    let mut reader = backend.begin_read().await.expect("begin snapshot");
    {
        let mut concurrent = backend.begin_write().await.expect("begin concurrent");
        concurrent
            .delete(key(b"snap/d"))
            .await
            .expect("concurrent delete");
        concurrent
            .put(key(b"snap/z"), key(b"7"))
            .await
            .expect("concurrent put");
        concurrent.commit().await.expect("commit concurrent");
    }

    let limits = ScanLimits {
        item_limit: 2,
        byte_limit: 1_024,
    };
    let range = range(b"snap/", b"snap0");
    let mut pages = 0_usize;
    let mut start = range.start().to_vec();
    let mut items = Vec::new();
    loop {
        let page = reader
            .scan(&KeyRange::new(start.clone(), range.end().to_vec()), limits)
            .await
            .expect("snapshot page");
        for item in page.items() {
            items.push((item.key().to_vec(), item.value().to_vec()));
        }
        pages += 1;
        match page.next_start() {
            Some(next) => start = next.to_vec(),
            None => break,
        }
    }

    let expected: Vec<(Vec<u8>, Vec<u8>)> = ["a", "b", "c", "d", "e", "f"]
        .iter()
        .zip(1_u8..=6)
        .map(|(suffix, value)| (format!("snap/{suffix}").into_bytes(), vec![value]))
        .collect();
    assert_eq!(
        pages, 3,
        "six keys paginate as three pages of two without gaps or duplicates",
    );
    assert_eq!(
        items, expected,
        "the open snapshot keeps the deleted key and hides the inserted key",
    );
    drop(reader);
    backend.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn blocking_resource_limit_waits_for_a_live_transaction_slot() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let namespace = BackendNamespace::new("ktann-issue-34-blocking-limit").expect("namespace");
    let config = RocksDbConfig::default()
        .with_blocking_resource_limit(1)
        .expect("blocking limit");
    let backend = RocksDbBackend::with_config(open_database(directory.path()), namespace, config);

    {
        let mut seed = backend.begin_write().await.expect("begin seed");
        seed.put(key(b"seed"), key(b"1")).await.expect("seed put");
        seed.commit().await.expect("commit seed");
    }

    let mut holder = backend.begin_read().await.expect("first slot");
    // The one native actor slot is held, so the next open must wait
    // asynchronously instead of failing or overtaking the live transaction.
    assert!(
        tokio::time::timeout(Duration::from_millis(100), backend.begin_read())
            .await
            .is_err(),
        "second open must wait while the only slot is held",
    );
    assert_eq!(
        holder.get(key(b"seed")).await.expect("holder get"),
        Some(key(b"1")),
        "the live transaction keeps making progress while another open waits",
    );

    drop(holder);
    let successor = tokio::time::timeout(Duration::from_secs(1), backend.begin_read())
        .await
        .expect("native cleanup releases the slot")
        .expect("successor admits after release");
    drop(successor);
    backend.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
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
    drop(read);
    backend.shutdown().await;
}
