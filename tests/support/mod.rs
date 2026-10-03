//! Shared KTANN test fixtures. Transactions use the production memory adapter.

use ktann::api::{LogicalIndexId, RuntimeConfig};
use ktann::storage::keys::LogicalKey;
use ktann::storage::values::{IndexManifest, PersistentValue};
use ktann::storage::{ReadLogicalTxn, backend::Backend};

#[allow(
    unused_imports,
    reason = "each integration suite uses a different subset"
)]
pub use ktann_memory::test_support::{
    CommitFault, CommitOutcome, Durability, HistoryEntry, TestConfig,
};
#[allow(
    unused_imports,
    reason = "each integration suite uses a different subset"
)]
pub use ktann_memory::{MemoryBackend, MemoryReadTxn, MemoryWriteTxn};

pub mod audit;
pub mod builders;
pub mod datadriven;
pub mod dataset;
pub mod fixtures;
pub mod load_index;
pub mod observe;
pub mod oracle;
pub mod topology_probe;

/// A Runtime configuration without background maintenance workers.
///
/// Tests that drive the split/merge state machines one bounded transition at
/// a time — or assert exact intermediate topology under fixtures — use this
/// configuration so demand-driven Fixup scheduling cannot race their manual
/// drives. Scheduling itself is covered by `maintenance_scheduling.rs`.
pub fn manual_maintenance_config() -> RuntimeConfig {
    RuntimeConfig::default()
        .with_maintenance(0, 1)
        .and_then(|config| config.with_import_limits(1, 1))
        .expect("valid manual-maintenance config")
}

#[allow(unused_imports, reason = "only replay suites draw seeded randomness")]
pub use dataset::Rng;

/// Reads the manifest through a fresh committed snapshot.
pub async fn read_manifest(backend: &MemoryBackend, index: LogicalIndexId) -> IndexManifest {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    match txn
        .get(LogicalKey::Manifest(index))
        .await
        .expect("read manifest")
    {
        Some(PersistentValue::IndexManifest(manifest)) => manifest,
        _ => panic!("manifest must exist"),
    }
}
