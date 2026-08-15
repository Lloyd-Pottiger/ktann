//! Focused contract checks against a temporary local RocksDB database.

use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, Mutation, ReadOps, ScanLimits, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use rocksdb::{MemtableFactory, OptimisticTransactionDB, Options, SliceTransform};

#[path = "../../tests/support/backend_contract.rs"]
mod shared_backend_contract;

use shared_backend_contract::{BackendHarness, Fault, FaultInjection, RestartMode};

fn key(value: &'static [u8]) -> Bytes {
    Bytes::from_static(value)
}

fn range(start: &[u8], end: &[u8]) -> KeyRange {
    KeyRange::new(start.to_vec(), end.to_vec())
}

fn open_database(path: &Path) -> OptimisticTransactionDB {
    let mut options = Options::default();
    options.create_if_missing(true);
    OptimisticTransactionDB::open(&options, path).expect("open RocksDB")
}

/// Adapts a [`RocksDbBackend`] to the shared harness seam.
///
/// RocksDB cannot stage controlled commit faults, and its durability is
/// exercised by reopening the database at the same path rather than by an
/// in-process restart.
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
        unreachable!("RocksDB durability is exercised by the adapter reopen test");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn rocksdb_adapter_preserves_the_backend_contract() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let database_path = directory.path().join("database");
    let primary_namespace = BackendNamespace::new("ktann-issue-18-contract").expect("namespace");
    let isolated_namespace =
        BackendNamespace::new("ktann-issue-18-contract-isolated").expect("namespace");

    {
        let database = Arc::new(open_database(&database_path));
        let primary = RocksDbBackend::new(Arc::clone(&database), primary_namespace.clone());
        let isolated = RocksDbBackend::new(Arc::clone(&database), isolated_namespace.clone());

        // Adapter-declared facts are asserted here; the shared suite checks the
        // rest of the common contract.
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
            b"large/b",
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
    }

    // Adapter-specific: committed data survives a real reopen at the same path.
    let reopened = RocksDbBackend::new(open_database(&database_path), primary_namespace);
    let mut durable = reopened.begin_read().await.expect("begin durable read");
    assert_eq!(
        durable.get(key(b"shared")).await.expect("durable get"),
        Some(key(b"primary")),
    );
    drop(durable);
    drop(reopened);

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
}

#[tokio::test(flavor = "current_thread")]
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
}
