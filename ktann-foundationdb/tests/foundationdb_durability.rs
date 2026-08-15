//! Two-phase durability check spanning a real FoundationDB server restart.

use bytes::Bytes;
use foundationdb::Database;
use ktann::storage::backend::{Backend, ReadOps, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

const PHASE_ENV: &str = "KTANN_FDB_DURABILITY_PHASE";

#[expect(
    unsafe_code,
    reason = "the FoundationDB binding requires one process-global network boot"
)]
fn boot_foundationdb() -> foundationdb::api::NetworkAutoStop {
    // SAFETY: this integration-test binary contains one test, so it starts the
    // process-global FoundationDB network exactly once. The returned guard is
    // retained until every database, backend, and transaction has dropped.
    unsafe { foundationdb::boot() }
}

async fn clear_test_keys(backend: &FoundationDbBackend) {
    let mut transaction = backend.begin_write().await.expect("begin cleanup");
    transaction
        .clear_range(&KeyRange::new(Vec::new(), vec![0xff]))
        .await
        .expect("clear durability range");
    transaction.commit().await.expect("commit cleanup");
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires write and verify phases separated by a FoundationDB server restart"]
async fn foundationdb_data_survives_server_restart() {
    let _network = boot_foundationdb();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let backend = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("open FoundationDB"),
        BackendNamespace::new("ktann-issue-20-durability").expect("namespace"),
    );
    let key = Bytes::from_static(b"committed-before-server-restart");

    match std::env::var(PHASE_ENV).as_deref() {
        Ok("write") => {
            clear_test_keys(&backend).await;
            let mut transaction = backend.begin_write().await.expect("begin durable write");
            transaction
                .put(key, Bytes::from_static(b"durable"))
                .await
                .expect("stage durable write");
            transaction.commit().await.expect("commit durable write");
        }
        Ok("verify") => {
            let mut transaction = backend.begin_read().await.expect("begin durable read");
            assert_eq!(
                transaction.get(key).await.expect("read durable value"),
                Some(Bytes::from_static(b"durable")),
            );
            drop(transaction);
            clear_test_keys(&backend).await;
        }
        _ => panic!("{PHASE_ENV} must be `write` or `verify`"),
    }
}
