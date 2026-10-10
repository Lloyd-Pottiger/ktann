//! Complete Bulk Build durability and automatic publication against FoundationDB.
use foundationdb::Database;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};
mod support;
use support::{boot_foundationdb, clear_test_keys};

#[path = "../../tests/support/bulk_load_adapter.rs"]
mod bulk_load_adapter;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_fenced_bulk_load_survives_reopen() {
    let _network = boot_foundationdb();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let backend = || {
        FoundationDbBackend::new(
            Database::new(cluster_file.as_deref()).unwrap(),
            BackendNamespace::new("ktann-bulk-load-verify").unwrap(),
        )
    };
    clear_test_keys(&backend()).await;
    let directory =
        std::env::temp_dir().join(format!("ktann-fdb-bulk-load-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    bulk_load_adapter::exercise(backend, &directory).await;
    clear_test_keys(&backend()).await;
    std::fs::remove_dir_all(directory).unwrap();
}
