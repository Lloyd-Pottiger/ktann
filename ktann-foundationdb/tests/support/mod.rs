//! Helpers shared by the FoundationDB adapter test binaries.
//!
//! Each integration-test file is its own crate and includes this module with
//! `mod support;`, mirroring how the adapter crates share the core crate's
//! `tests/support/` files by path.

// Each test binary uses a subset of these helpers.
#![allow(dead_code)]

use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_foundationdb::FoundationDbBackend;

/// Starts the process-global FoundationDB network for one test binary.
#[expect(
    unsafe_code,
    reason = "the FoundationDB binding requires one process-global network boot"
)]
pub fn boot_foundationdb() -> foundationdb::api::NetworkAutoStop {
    // SAFETY: each integration-test binary contains one test, so it starts the
    // process-global FoundationDB network exactly once. The returned guard is
    // kept alive until every database, backend, and transaction has dropped.
    unsafe { foundationdb::boot() }
}

/// Removes every key the test wrote within its Backend Namespace.
pub async fn clear_test_keys(backend: &FoundationDbBackend) {
    let mut transaction = backend.begin_write().await.expect("begin cleanup");
    transaction
        .clear_range(&KeyRange::new(b"".to_vec(), b"\xff".to_vec()))
        .await
        .expect("clear test range");
    transaction.commit().await.expect("commit cleanup");
}
