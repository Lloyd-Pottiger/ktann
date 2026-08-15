//! Focused contract checks against a local FoundationDB 7.3 cluster.

use bytes::Bytes;
use foundationdb::Database;
use foundationdb::options::DatabaseOption;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, ReadOps, ScanLimits, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

#[path = "../../tests/support/backend_contract.rs"]
mod shared_backend_contract;

use shared_backend_contract::{BackendHarness, Fault, FaultInjection, RestartMode};

fn key(value: &'static [u8]) -> Bytes {
    Bytes::from_static(value)
}

fn range(start: &[u8], end: &[u8]) -> KeyRange {
    KeyRange::new(start.to_vec(), end.to_vec())
}

#[expect(
    unsafe_code,
    reason = "the FoundationDB binding requires one process-global network boot"
)]
fn boot_foundationdb() -> foundationdb::api::NetworkAutoStop {
    // SAFETY: this integration-test binary contains one test, so it starts the
    // process-global FoundationDB network exactly once. The returned guard is
    // kept alive until every database, backend, and transaction has dropped.
    unsafe { foundationdb::boot() }
}

async fn clear_test_keys(backend: &FoundationDbBackend) {
    let mut transaction = backend.begin_write().await.expect("begin cleanup");
    transaction
        .clear_range(&range(b"", b"\xff"))
        .await
        .expect("clear test range");
    transaction.commit().await.expect("commit cleanup");
}

/// Adapts a [`FoundationDbBackend`] to the shared harness seam.
///
/// FoundationDB reports unknown commit outcomes naturally but cannot stage a
/// specific outcome or restart the external cluster from the shared suite.
struct FoundationDbHarness {
    backend: FoundationDbBackend,
}

impl BackendHarness for FoundationDbHarness {
    type Backend = FoundationDbBackend;

    fn backend(&self) -> &FoundationDbBackend {
        &self.backend
    }

    fn fault_injection(&self) -> FaultInjection {
        FaultInjection::Unavailable
    }

    fn inject_fault(&self, _fault: Fault) {
        unreachable!("FoundationDB cannot stage controlled commit faults");
    }

    fn restart_mode(&self) -> RestartMode {
        RestartMode::Unsupported
    }

    fn restart(&self) -> Self {
        unreachable!("FoundationDB process restart requires the external durability test");
    }
}

/// Proves that FoundationDB's native transaction-size rejection is preserved.
async fn check_native_transaction_limit(cluster_file: Option<&str>) {
    let database = Database::new(cluster_file).expect("open size-limited database");
    database
        .set_option(DatabaseOption::TransactionSizeLimit(128))
        .expect("set transaction size limit");
    let backend = FoundationDbBackend::new(
        database,
        BackendNamespace::new("ktann-issue-20-native-limit").expect("namespace"),
    );

    // This write is below the adapter's admission budget but above the lower
    // native limit configured on this database handle.
    let mut transaction = backend.begin_write().await.expect("begin limited write");
    transaction
        .put(key(b"native-limit"), Bytes::from(vec![0_u8; 256]))
        .await
        .expect("stage limited write");
    let error = transaction
        .commit()
        .await
        .expect_err("native size rejection");
    assert_eq!(error.kind(), ErrorKind::TransactionTooLarge);
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_adapter_preserves_the_backend_contract() {
    let _network = boot_foundationdb();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let database = Database::new(cluster_file.as_deref()).expect("open FoundationDB");
    let primary = FoundationDbBackend::new(
        database,
        BackendNamespace::new("ktann-issue-20-contract").expect("namespace"),
    );
    let isolated = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("open isolated database handle"),
        BackendNamespace::new("ktann-issue-20-contract-isolated").expect("namespace"),
    );
    clear_test_keys(&primary).await;
    clear_test_keys(&isolated).await;

    // Adapter-declared facts are asserted here; the shared suite checks the
    // rest of the common contract.
    assert_eq!(primary.admission_budget().max_mutations, 10_000);
    assert_eq!(primary.admission_budget().max_mutation_bytes, 1 << 20);
    assert_eq!(primary.hard_limits().max_value_bytes, 100_000);
    assert!(primary.capabilities().transactional_clear_range);

    let harness = FoundationDbHarness { backend: primary };
    shared_backend_contract::run_suite(&harness).await;
    let primary = harness.backend;

    check_native_transaction_limit(cluster_file.as_deref()).await;

    // Adapter-specific: two namespaces over one cluster are isolated.
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

    // Adapter-specific: a transactional range clear removes a bounded logical
    // range while a later point put in the same transaction is retained, and
    // both survive a reopen against the same cluster.
    let mut clear = primary.begin_write().await.expect("begin range clear");
    clear
        .put(key(b"scan/before-clear"), key(b"removed"))
        .await
        .expect("put before clear");
    clear
        .clear_range(&range(b"scan/", b"scan0"))
        .await
        .expect("range clear");
    clear
        .put(key(b"scan/after-clear"), key(b"retained"))
        .await
        .expect("put after clear");
    clear.commit().await.expect("commit range clear");

    let reopened = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("reopen database"),
        BackendNamespace::new("ktann-issue-20-contract").expect("namespace"),
    );
    let mut durable = reopened.begin_read().await.expect("begin durable read");
    assert_eq!(
        durable.get(key(b"shared")).await.expect("durable get"),
        Some(key(b"primary")),
    );
    let after_clear = durable
        .scan(
            &range(b"scan/", b"scan0"),
            ScanLimits {
                item_limit: 10,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("scan cleared range");
    assert_eq!(after_clear.items().len(), 1);
    assert_eq!(after_clear.items()[0].key().as_ref(), b"scan/after-clear");
    drop(durable);
    clear_test_keys(&reopened).await;
    clear_test_keys(&isolated).await;
}
