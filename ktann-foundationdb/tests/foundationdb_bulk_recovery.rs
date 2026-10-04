//! Real client-process crashes at durable bulk-build boundaries.
use bytes::Bytes;
use foundationdb::Database;
use ktann::api::{
    BulkBuildOptions, ErrorKind, IndexConfig, LogicalIndexId, Metric, OperationOptions, Record,
    RuntimeConfig, VerifyOptions,
};
use ktann::runtime::Runtime;
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation,
    ReadOps, ReadTxn, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::{self, KeyRange, LogicalKey};
use ktann::storage::values::{IndexLifecycle, PersistentValue, ValueCodec};
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
mod support;

const RECORDS: usize = 1025;

/// Pauses only after a real FDB commit, so killing the child does not run Rust cleanup.
struct CrashBoundary {
    phase: String,
    marker: PathBuf,
}

struct CrashBackend {
    inner: FoundationDbBackend,
    boundary: Arc<CrashBoundary>,
}

impl Backend for CrashBackend {
    type ReadTxn<'a> = <FoundationDbBackend as Backend>::ReadTxn<'a>;
    type WriteTxn<'a> = CrashWrite<<FoundationDbBackend as Backend>::WriteTxn<'a>>;
    fn hard_limits(&self) -> HardLimits {
        self.inner.hard_limits()
    }
    fn admission_budget(&self) -> AdmissionBudget {
        self.inner.admission_budget()
    }
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
    async fn shutdown(&self) {
        self.inner.shutdown().await;
    }
    async fn begin_read(&self) -> ktann::api::Result<Self::ReadTxn<'_>> {
        self.inner.begin_read().await
    }
    async fn begin_write(&self) -> ktann::api::Result<Self::WriteTxn<'_>> {
        Ok(CrashWrite {
            inner: self.inner.begin_write().await?,
            boundary: Arc::clone(&self.boundary),
            pause: None,
        })
    }
}

struct CrashWrite<T> {
    inner: T,
    boundary: Arc<CrashBoundary>,
    pause: Option<LogicalIndexId>,
}

impl<T> CrashWrite<T> {
    /// Select boundaries by durable object semantics, not transaction ordinal.
    fn observe_put(&mut self, key: &Bytes, value: &Bytes) {
        let logical = keys::decode_key(&[], key).unwrap();
        match logical {
            LogicalKey::Record { index, .. } if self.boundary.phase == "partial" => {
                self.pause = Some(index)
            }
            LogicalKey::Manifest(index) if self.boundary.phase == "published" => {
                let PersistentValue::IndexManifest(manifest) = ValueCodec::bootstrap()
                    .decode(&logical, value.clone())
                    .unwrap()
                else {
                    panic!("manifest value")
                };
                if manifest.lifecycle() == IndexLifecycle::Active {
                    self.pause = Some(index);
                }
            }
            _ => {}
        }
    }
}

impl<T: ReadOps> ReadOps for CrashWrite<T> {
    async fn get(&mut self, key: Bytes) -> ktann::api::Result<Option<Bytes>> {
        self.inner.get(key).await
    }
    async fn batch_get(&mut self, keys: Vec<Bytes>) -> ktann::api::Result<Vec<Option<Bytes>>> {
        self.inner.batch_get(keys).await
    }
    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> ktann::api::Result<ScanPage> {
        self.inner.scan(range, limits).await
    }
    async fn batch_scan(
        &mut self,
        ranges: &[KeyRange],
        limits: ScanLimits,
    ) -> ktann::api::Result<Vec<ScanPage>> {
        self.inner.batch_scan(ranges, limits).await
    }
}
impl<T: ReadTxn> ReadTxn for CrashWrite<T> {}
impl<T: WriteTxn> WriteTxn for CrashWrite<T> {
    async fn get_for_update(&mut self, key: Bytes) -> ktann::api::Result<Option<Bytes>> {
        self.inner.get_for_update(key).await
    }
    async fn batch_get_for_update(
        &mut self,
        keys: Vec<Bytes>,
    ) -> ktann::api::Result<Vec<Option<Bytes>>> {
        self.inner.batch_get_for_update(keys).await
    }
    async fn put(&mut self, key: Bytes, value: Bytes) -> ktann::api::Result<()> {
        self.observe_put(&key, &value);
        self.inner.put(key, value).await
    }
    async fn insert(&mut self, key: Bytes, value: Bytes) -> ktann::api::Result<InsertOutcome> {
        self.observe_put(&key, &value);
        self.inner.insert(key, value).await
    }
    async fn delete(&mut self, key: Bytes) -> ktann::api::Result<()> {
        self.inner.delete(key).await
    }
    async fn batch_mutate(&mut self, mutations: Vec<Mutation>) -> ktann::api::Result<()> {
        for mutation in &mutations {
            if let Mutation::Put { key, value } = mutation {
                self.observe_put(key, value);
            }
        }
        self.inner.batch_mutate(mutations).await
    }
    async fn clear_range(&mut self, range: &KeyRange) -> ktann::api::Result<()> {
        self.inner.clear_range(range).await
    }
    async fn commit_with(self, start: CommitStart) -> ktann::api::Result<()> {
        self.inner.commit_with(start).await?;
        if let Some(index) = self.pause {
            // Rename publishes a complete marker to the monitoring parent.
            let pending = self.boundary.marker.with_extension("pending");
            std::fs::write(&pending, index.get().to_string()).unwrap();
            std::fs::rename(pending, &self.boundary.marker).unwrap();
            std::future::pending::<()>().await;
        }
        Ok(())
    }
    async fn rollback(self) {
        self.inner.rollback().await;
    }
}

/// Never leave a paused child behind if an assertion or timeout fails.
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn backend(namespace: &str) -> FoundationDbBackend {
    let cluster = std::env::var("FDB_CLUSTER_FILE").ok();
    FoundationDbBackend::new(
        Database::new(cluster.as_deref()).unwrap(),
        BackendNamespace::new(namespace).unwrap(),
    )
}
fn runtime_config() -> RuntimeConfig {
    RuntimeConfig::default()
        .with_maintenance(0, 1)
        .unwrap()
        .with_import_limits(1, 1)
        .unwrap()
}
fn config() -> IndexConfig {
    IndexConfig::new(16, Metric::Cosine)
        .unwrap()
        .with_partition_entries(8, 32)
        .unwrap()
}
fn records() -> Vec<Record> {
    (0..RECORDS)
        .map(|i| {
            Record::new(
                Bytes::from(i.to_be_bytes().to_vec()),
                (0..16)
                    .map(|j| ((i * 17 + j * 11) % 101) as f32 + 1.0)
                    .collect::<Vec<_>>(),
                vec![],
            )
            .unwrap()
        })
        .collect()
}
fn options() -> BulkBuildOptions {
    BulkBuildOptions::new(1 << 20).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_bulk_client_crash_recovery() {
    let _network = support::boot_foundationdb();
    if let Ok(phase) = std::env::var("KTANN_BULK_CRASH_PHASE") {
        let namespace = std::env::var("KTANN_BULK_CRASH_NAMESPACE").unwrap();
        let marker = std::env::var_os("KTANN_BULK_CRASH_MARKER").unwrap().into();
        let runtime = Runtime::new(
            CrashBackend {
                inner: backend(&namespace),
                boundary: Arc::new(CrashBoundary { phase, marker }),
            },
            runtime_config(),
        )
        .unwrap();
        runtime
            .build_index(
                "bulk",
                config(),
                records(),
                options(),
                OperationOptions::default(),
            )
            .await
            .unwrap();
        panic!("selected commit boundary was not reached");
    }

    for phase in ["partial", "published"] {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let namespace = format!("ktann-bulk-crash-{}-{stamp}-{phase}", std::process::id());
        eprintln!("bulk recovery phase={phase}, namespace={namespace}");
        let marker = std::env::temp_dir().join(format!("{namespace}.marker"));
        let mut child = ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "foundationdb_bulk_client_crash_recovery",
                    "--nocapture",
                ])
                .env("KTANN_BULK_CRASH_PHASE", phase)
                .env("KTANN_BULK_CRASH_NAMESPACE", &namespace)
                .env("KTANN_BULK_CRASH_MARKER", &marker)
                .spawn()
                .unwrap(),
        );
        tokio::time::timeout(Duration::from_secs(60), async {
            while !marker.exists() {
                assert!(
                    child.0.try_wait().unwrap().is_none(),
                    "child exited before durable boundary"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("child did not reach durable boundary");
        let index_id =
            LogicalIndexId::new(std::fs::read_to_string(&marker).unwrap().parse().unwrap())
                .unwrap();
        child.0.kill().unwrap();
        assert!(!child.0.wait().unwrap().success());
        std::fs::remove_file(&marker).unwrap();

        let runtime = Runtime::new(backend(&namespace), runtime_config()).unwrap();
        let cleanup = backend(&namespace);
        if phase == "partial" {
            assert_eq!(
                runtime.open_index("bulk").await.unwrap_err().kind(),
                ErrorKind::IndexBuilding
            );
            assert_eq!(
                runtime
                    .create_index("bulk", config())
                    .await
                    .unwrap_err()
                    .kind(),
                ErrorKind::IndexBuilding
            );
            let mut raw = cleanup.begin_read().await.unwrap();
            let page = raw
                .scan(
                    &keys::index_range(index_id),
                    ScanLimits {
                        item_limit: 100_000,
                        byte_limit: 8 << 20,
                    },
                )
                .await
                .unwrap();
            assert!(page.next_start().is_none());
            let stored = page
                .items()
                .iter()
                .filter(|item| {
                    matches!(
                        keys::decode_key(&[], item.key()).unwrap(),
                        LogicalKey::Record { .. }
                    )
                })
                .count();
            assert!(
                stored > 0 && stored < RECORDS,
                "fixture must contain a partial committed build"
            );
            drop(raw);
            runtime.drop_index("bulk").await.unwrap();
            let mut raw = cleanup.begin_read().await.unwrap();
            assert!(
                raw.scan(
                    &keys::index_range(index_id),
                    ScanLimits {
                        item_limit: 1,
                        byte_limit: 4096
                    }
                )
                .await
                .unwrap()
                .items()
                .is_empty()
            );
            drop(raw);
            let replacement = runtime
                .build_index(
                    "bulk",
                    config(),
                    records(),
                    options(),
                    OperationOptions::default(),
                )
                .await
                .unwrap();
            assert_ne!(replacement.logical_index_id(), index_id);
        }
        let index = runtime.open_index("bulk").await.unwrap();
        if phase == "published" {
            assert_eq!(index.logical_index_id(), index_id);
        }
        let report = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(report.complete && report.issues.is_empty());
        assert_eq!(report.objects.vector_records, RECORDS as u64);
        runtime.drop_index("bulk").await.unwrap();
        runtime.shutdown().await.unwrap();
        support::clear_test_keys(&cleanup).await;
    }
}
