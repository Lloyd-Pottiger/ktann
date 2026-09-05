//! Helpers shared by the RocksDB adapter test binaries.
//!
//! Each integration-test file is its own crate and includes this module with
//! `mod support;`, mirroring how the adapter crates share the core crate's
//! `tests/support/` files by path.

// Each test binary uses a subset of these helpers.
#![allow(dead_code)]

use std::path::Path;

use bytes::Bytes;
use ktann::storage::keys::KeyRange;
use rocksdb::{OptimisticTransactionDB, Options};

/// Builds a static key or value fixture.
pub fn key(value: &'static [u8]) -> Bytes {
    Bytes::from_static(value)
}

/// Builds the `[start, end)` range fixture.
pub fn range(start: &[u8], end: &[u8]) -> KeyRange {
    KeyRange::new(start.to_vec(), end.to_vec())
}

/// Opens one database at `path`, creating it when missing.
pub fn open_database(path: &Path) -> OptimisticTransactionDB {
    let mut options = Options::default();
    options.create_if_missing(true);
    OptimisticTransactionDB::open(&options, path).expect("open RocksDB")
}
