//! RocksDB-specific fault checks against a temporary local database.
//!
//! RocksDB cannot stage controlled commit outcomes the way the deterministic
//! backend's fault plan can, and its commit path has no injectable fault switch
//! like FoundationDB's Client Buggify. This file instead corrupts a flushed
//! SST on disk and proves that real native failures reach the adapter's stable
//! error categories on both the point-read and the scan path. The shared suite
//! covers the real optimistic-conflict (`Busy`) classification against this
//! database, the commit-error mapping table is unit-tested, and the
//! deterministic backend remains the exhaustive evidence for both applied and
//! unapplied unknown commit outcomes.
//!
//! The test owns one `tempfile` directory that is removed when the test
//! finishes, binds a dedicated Backend Namespace, and writes one bounded value,
//! so the corrupted files never escape the test.

use std::fs::{self, OpenOptions};
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, ReadOps, ScanLimits, WriteTxn};
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};

mod support;

use support::{key, open_database, range};

/// Truncates every SST in `directory` to 20 bytes, destroying each footer so
/// any read of the table must fail a structure or checksum check.
fn truncate_ssts(directory: &Path) {
    let mut truncated = 0_usize;
    for entry in fs::read_dir(directory).expect("list database files") {
        let entry = entry.expect("directory entry");
        if entry.file_name().to_string_lossy().ends_with(".sst") {
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(entry.path())
                .expect("open SST for truncation")
                .set_len(20)
                .expect("truncate SST");
            truncated += 1;
        }
    }
    assert!(truncated >= 1, "flush must produce at least one SST");
}

#[tokio::test(flavor = "multi_thread")]
async fn corrupted_sstable_surfaces_corruption_on_point_read_and_scan() {
    let directory = tempfile::tempdir().expect("temporary database directory");
    let database = Arc::new(open_database(directory.path()));
    let namespace = BackendNamespace::new("ktann-issue-34-corruption").expect("namespace");
    let backend = RocksDbBackend::new(Arc::clone(&database), namespace);

    let mut txn = backend.begin_write().await.expect("begin seed");
    txn.put(key(b"corrupt/a"), Bytes::from(vec![0x42; 4_096]))
        .await
        .expect("put seed");
    txn.commit().await.expect("commit seed");

    // Move the value from the memtable into an on-disk SST before corrupting
    // it, so the read below cannot be served from memory.
    database.flush().expect("flush memtable");
    truncate_ssts(directory.path());

    let mut read = backend.begin_read().await.expect("begin read");
    let point = read.get(key(b"corrupt/a")).await;
    assert_eq!(
        point.expect_err("corrupted point read").kind(),
        ErrorKind::Corruption,
    );

    let scan = read
        .scan(
            &range(b"corrupt/", b"corrupt0"),
            ScanLimits {
                item_limit: 10,
                byte_limit: 1_024,
            },
        )
        .await;
    assert_eq!(
        scan.expect_err("corrupted scan").kind(),
        ErrorKind::Corruption,
    );
    drop(read);
    backend.shutdown().await;
}
