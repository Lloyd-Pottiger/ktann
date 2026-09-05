//! Two-phase durability check spanning a real FoundationDB server restart.

use bytes::Bytes;
use foundationdb::Database;
use ktann::storage::backend::{Backend, ReadOps, WriteTxn};
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

mod support;

use support::{boot_foundationdb, clear_test_keys};

const PHASE_ENV: &str = "KTANN_FDB_DURABILITY_PHASE";

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
