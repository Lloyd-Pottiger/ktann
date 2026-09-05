//! Two-phase WAL durability check spanning a real process boundary.
//!
//! Restarting an adapter over a reopened database in one process is evidence
//! of visibility, but only a fresh process proves that an acknowledged commit
//! (WAL enabled, `sync=true`) reached disk and is recovered on open. Run the
//! write phase, then run the verify phase from a new process against the same
//! database directory:
//!
//! ```sh
//! KTANN_ROCKSDB_DURABILITY_PHASE=write \
//!   KTANN_ROCKSDB_DURABILITY_PATH=/tmp/ktann-rocksdb-durability \
//!   cargo test -p ktann-rocksdb --test rocksdb_durability -- --ignored
//!
//! KTANN_ROCKSDB_DURABILITY_PHASE=verify \
//!   KTANN_ROCKSDB_DURABILITY_PATH=/tmp/ktann-rocksdb-durability \
//!   cargo test -p ktann-rocksdb --test rocksdb_durability -- --ignored
//! ```
//!
//! The write phase clears any previous database at the path before writing two
//! bounded keys; the verify phase destroys the database directory when it
//! finishes, so repeated runs stay isolated and cleaned up.

use std::fs;
use std::path::PathBuf;

use bytes::Bytes;
use ktann::storage::backend::{Backend, ReadOps, WriteTxn};
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use rocksdb::Options;

mod support;

use support::open_database;

const PHASE_ENV: &str = "KTANN_ROCKSDB_DURABILITY_PHASE";
const PATH_ENV: &str = "KTANN_ROCKSDB_DURABILITY_PATH";

/// The committed key/value probes the write phase stores and the verify phase
/// reads back.
const PROBE_ENTRIES: [(&[u8], &[u8]); 2] = [
    (b"committed-a", b"durable-a"),
    (b"committed-b", b"durable-b"),
];

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires write and verify phases in separate processes sharing one database path"]
async fn rocksdb_data_survives_process_restart() {
    let path = PathBuf::from(
        std::env::var(PATH_ENV).expect("KTANN_ROCKSDB_DURABILITY_PATH must name a directory"),
    );
    let namespace = BackendNamespace::new("ktann-issue-34-durability").expect("namespace");

    match std::env::var(PHASE_ENV).as_deref() {
        Ok("write") => {
            // Start from an empty directory so the probe cannot pass on data a
            // previous run left behind.
            if path.exists() {
                fs::remove_dir_all(&path).expect("clear previous durability database");
            }
            let backend = RocksDbBackend::new(open_database(&path), namespace);
            let mut transaction = backend.begin_write().await.expect("begin durable write");
            for (key, value) in PROBE_ENTRIES {
                transaction
                    .put(Bytes::from_static(key), Bytes::from_static(value))
                    .await
                    .expect("stage durable write");
            }
            transaction.commit().await.expect("commit durable write");
            // Crash instead of closing: the memtable is never flushed, so the
            // verify phase's open must recover the acknowledged commit from
            // the synced WAL alone.
            std::process::exit(0);
        }
        Ok("verify") => {
            let backend = RocksDbBackend::new(open_database(&path), namespace);
            let mut transaction = backend.begin_read().await.expect("begin durable read");
            for (key, value) in PROBE_ENTRIES {
                assert_eq!(
                    transaction
                        .get(Bytes::from_static(key))
                        .await
                        .expect("read durable value"),
                    Some(Bytes::from_static(value)),
                );
            }
            drop(transaction);
            // Consumes the adapter and releases the database handle (and its
            // lock) before the files are destroyed.
            backend.shutdown().await;
            // Remove the shared database so the next run starts from the same
            // clean state; destroy handles the RocksDB files and the directory
            // removal covers any leftover marker files.
            rocksdb::DB::destroy(&Options::default(), &path).expect("destroy durability database");
            if path.exists() {
                fs::remove_dir_all(&path).expect("remove durability directory");
            }
        }
        _ => panic!("{PHASE_ENV} must be `write` or `verify`"),
    }
}
