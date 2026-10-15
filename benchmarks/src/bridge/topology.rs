//! Readiness snapshots and bounded maintenance for the bridge's single Tree Key.
//!
//! Header counts establish maintenance readiness, not full record integrity.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ktann::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result};
use ktann::maintenance::{merge, split};
use ktann::runtime::RetryPolicy;
use ktann::storage::backend::{Backend, ReadOps};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{IndexManifest, PartitionState, PersistentValue, ValueCodec};
use serde_json::{Value, json};

pub(super) const MAX_HEADER_SLOTS: u64 = 262_144;
pub(super) const MAX_ADVANCE_STEPS: usize = 32;

/// One consistent snapshot and at most 32 explicitly identified maintenance steps.
pub(super) struct Snapshot {
    pub(super) facts: Value,
    pub(super) ready: bool,
    /// Header fingerprint used only to avoid redundant rediscovery, never to prove readiness.
    pub(super) progress: u64,
    manifest: IndexManifest,
    tree_key: TreeKey,
    pending: Vec<Work>,
}

/// One source identified from its committed Header; ReceivingSplit is not a source.
enum Work {
    Split(PartitionKey),
    Merge(PartitionKey),
}

impl Snapshot {
    /// Advances cold sources directly, without relying on approximate query routing.
    /// Background workers may race these steps; the owning state machines revalidate
    /// durable authority and bound each transaction and retry sequence. Returns
    /// whether a step advanced work, so active draining need not wait for a poll.
    pub(super) async fn advance<B: Backend>(
        &self,
        backend: &B,
        retry: &RetryPolicy,
    ) -> Result<bool> {
        let mut progressed = false;
        for work in &self.pending {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .and_then(|time| u64::try_from(time.as_millis()).ok())
                .unwrap_or(0);
            let result = match *work {
                Work::Split(partition) => split::advance(
                    backend,
                    &self.manifest,
                    &self.tree_key,
                    partition,
                    now,
                    retry,
                )
                .await
                .map(|step| match step {
                    split::Advance::Idle => false,
                    split::Advance::Drained { moved, .. } => moved != 0,
                    _ => true,
                }),
                Work::Merge(partition) => merge::advance(
                    backend,
                    &self.manifest,
                    &self.tree_key,
                    partition,
                    now,
                    retry,
                )
                .await
                .map(|step| match step {
                    merge::Advance::Idle | merge::Advance::Stalled => false,
                    merge::Advance::Drained { moved, .. } => moved != 0,
                    _ => true,
                }),
            };
            match result {
                // A fresh readiness round re-reads authority after contention or an
                // uncertain commit. The Optimize deadline bounds repeated rounds.
                Err(error)
                    if matches!(
                        error.kind(),
                        ErrorKind::RetryableAbort
                            | ErrorKind::ContentionExhausted
                            | ErrorKind::CommitOutcomeUnknown
                    ) => {}
                result => progressed |= result?,
            }
        }
        Ok(progressed)
    }
}

/// A readiness round reads at most 262,144 allocated Header slots in 256-key
/// batches and, when requested, selects at most 32 maintenance sources. Missing slots are legal
/// allocator gaps or deleted partitions. All live Headers use one read snapshot.
pub(super) async fn snapshot<B: Backend>(
    backend: &B,
    index: LogicalIndexId,
    records: u64,
    collect_work: bool,
) -> Result<Snapshot> {
    let corrupt = || Error::new(ErrorKind::Corruption);
    let mut txn = backend.begin_read().await?;
    let bytes = txn
        .get(keys::manifest_key(index).into())
        .await?
        .ok_or_else(corrupt)?;
    let PersistentValue::IndexManifest(manifest) =
        ValueCodec::bootstrap().decode(&LogicalKey::Manifest(index), bytes)?
    else {
        return Err(corrupt());
    };
    let codec = ValueCodec::for_index(&manifest);
    let tree_key = TreeKey::encode(&[], &[])?;
    let tree_bytes = txn
        .get(keys::tree_manifest_key(index, &tree_key).into())
        .await?
        .ok_or_else(corrupt)?;
    let PersistentValue::TreeManifest(tree) = codec.decode(
        &LogicalKey::TreeManifest {
            index,
            tree_key: tree_key.clone(),
        },
        tree_bytes,
    )?
    else {
        return Err(corrupt());
    };
    let high_water = tree.partition_key_high_water().get();
    if high_water > MAX_HEADER_SLOTS {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let mut partitions_by_level = BTreeMap::<u32, u64>::new();
    let mut max_entries_by_level = BTreeMap::<u32, u32>::new();
    let mut progress = xxhash_rust::xxh3::Xxh3::new();
    let mut leaf_entries = 0;
    let mut partitions = 0;
    let mut transitional = 0;
    let mut actionable = 0;
    let mut root_present = false;
    let mut pending = Vec::new();
    for first in (1..=high_water).step_by(256) {
        let partitions_in_batch: Vec<_> = (first..=(first + 255).min(high_water))
            .map(PartitionKey::new)
            .collect::<Result<_>>()?;
        let header_keys = partitions_in_batch
            .iter()
            .map(|p| Bytes::from(keys::header_key(index, &tree_key, *p)))
            .collect();
        for (partition, bytes) in partitions_in_batch
            .into_iter()
            .zip(txn.batch_get(header_keys).await?)
        {
            let Some(bytes) = bytes else {
                continue;
            };
            progress.update(&partition.get().to_be_bytes());
            progress.update(&bytes);
            let key = LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition,
            };
            let PersistentValue::PartitionHeader(header) = codec.decode(&key, bytes)? else {
                return Err(corrupt());
            };
            root_present |= partition == tree.root();
            partitions += 1;
            *partitions_by_level.entry(header.level()).or_default() += 1;
            let max = max_entries_by_level.entry(header.level()).or_default();
            *max = (*max).max(header.entry_count());
            if header.level() == 1 {
                leaf_entries += u64::from(header.entry_count());
            }
            let in_transition = header.state() != PartitionState::Ready;
            transitional += u64::from(in_transition);
            let needs_work = in_transition
                || header.entry_count() > manifest.config().max_partition_entries()
                || (partition != tree.root()
                    && header.entry_count() < manifest.config().min_partition_entries());
            actionable += u64::from(needs_work && header.state() != PartitionState::ReceivingSplit);
            if collect_work && needs_work && pending.len() < MAX_ADVANCE_STEPS {
                match header.state() {
                    PartitionState::ReceivingSplit => {}
                    PartitionState::Merging => pending.push(Work::Merge(partition)),
                    PartitionState::Ready
                        if header.entry_count() < manifest.config().min_partition_entries() =>
                    {
                        pending.push(Work::Merge(partition))
                    }
                    _ => pending.push(Work::Split(partition)),
                }
            }
        }
    }
    if !root_present {
        return Err(corrupt());
    }
    Ok(Snapshot {
        progress: progress.digest(),
        ready: leaf_entries == records && actionable == 0 && transitional == 0,
        facts: json!({"kind":"single-tree header snapshot (not full integrity verification)","records_from_leaf_headers":leaf_entries,"partitions":partitions,"max_level":partitions_by_level.keys().next_back(),"partitions_by_level":partitions_by_level,"max_entries_by_level":max_entries_by_level,"actionable":actionable,"transitional":transitional,"allocated_header_slots":high_water}),
        manifest,
        tree_key,
        pending,
    })
}

#[cfg(all(test, feature = "rocksdb"))]
mod tests {
    use super::*;
    use ktann::api::{IndexConfig, Metric, Mutation, Record, RuntimeConfig, VerifyOptions};
    use ktann::runtime::Runtime;
    use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
    use std::sync::Arc;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn header_readiness_matches_full_audit_and_rejects_pending_split() {
        let directory = tempfile::tempdir().unwrap();
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        let database =
            Arc::new(rocksdb::OptimisticTransactionDB::open(&options, directory.path()).unwrap());
        let backend = RocksDbBackend::new(
            database,
            BackendNamespace::new("bridge-topology-test").unwrap(),
        );
        let (backend, _) = crate::backend::MeasuredBackend::new(backend);
        let runtime = Runtime::new(
            backend.clone(),
            RuntimeConfig::default().with_maintenance(0, 1024).unwrap(),
        )
        .unwrap();
        let index = runtime
            .create_index(
                "test",
                IndexConfig::new(2, Metric::L2)
                    .unwrap()
                    // Readiness follows the Manifest, not the bridge's fixed thresholds.
                    .with_partition_entries(16, 64)
                    .unwrap(),
            )
            .await
            .unwrap();
        let mutations = (0_i64..64)
            .map(|id| {
                Mutation::Insert(
                    Record::new(
                        Bytes::copy_from_slice(&id.to_be_bytes()),
                        vec![id as f32, 1.],
                        vec![],
                    )
                    .unwrap(),
                )
            })
            .collect();
        index.batch_mutate(mutations).await.unwrap();
        let metadata = snapshot(&backend, index.logical_index_id(), 64, false)
            .await
            .unwrap();
        let full = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(full.complete && full.issues.is_empty());
        assert!(metadata.ready);
        assert_eq!(
            metadata.facts["records_from_leaf_headers"],
            full.objects.vector_records
        );
        assert_eq!(metadata.facts["partitions"], full.topology.partitions);
        assert!(
            !snapshot(&backend, index.logical_index_id(), 65, true)
                .await
                .unwrap()
                .ready
        );
        index
            .insert(
                Record::new(
                    Bytes::copy_from_slice(&64_i64.to_be_bytes()),
                    vec![64., 1.],
                    vec![],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        let pending = snapshot(&backend, index.logical_index_id(), 65, true)
            .await
            .unwrap();
        assert!(!pending.ready);
        assert!(
            matches!(pending.pending.as_slice(), [Work::Split(partition)] if partition.get() == 1)
        );
        assert_eq!(pending.facts["actionable"], 1);
        runtime.shutdown().await.unwrap();
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_identical_vectors_converge() {
        let directory = tempfile::tempdir().unwrap();
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        let database =
            Arc::new(rocksdb::OptimisticTransactionDB::open(&options, directory.path()).unwrap());
        let backend =
            RocksDbBackend::new(database, BackendNamespace::new("cold-maintenance").unwrap());
        let (backend, _) = crate::backend::MeasuredBackend::new(backend);
        let loader = Runtime::new(
            backend.clone(),
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let index = loader
            .create_index(
                "cold",
                IndexConfig::new(2, Metric::L2)
                    .unwrap()
                    .with_partition_entries(16, 64)
                    .unwrap(),
            )
            .await
            .unwrap();
        for first in (0_i64..257).step_by(32) {
            let records = (first..(first + 32).min(257))
                .map(|id| {
                    Mutation::Insert(
                        Record::new(
                            Bytes::copy_from_slice(&id.to_be_bytes()),
                            vec![0., 0.],
                            vec![],
                        )
                        .unwrap(),
                    )
                })
                .collect();
            index.batch_mutate(records).await.unwrap();
        }
        // One real split leaves a Ready sibling and an oversized cold sibling.
        // All-zero L2 vectors keep their centroids exactly equal under rotation.
        let id = index.logical_index_id();
        let manifest = snapshot(&backend, id, 257, false).await.unwrap().manifest;
        let tree = TreeKey::encode(&[], &[]).unwrap();
        let retry = RetryPolicy::for_fixup(&RuntimeConfig::default());
        let mut source = PartitionKey::new(1).unwrap();
        for _ in 0..2 {
            for step in 0..100 {
                if matches!(
                    split::advance(&backend, &manifest, &tree, source, 0, &retry)
                        .await
                        .unwrap(),
                    split::Advance::Completed { .. }
                ) {
                    break;
                }
                assert!(step < 99, "fixture split must finish");
            }
            let state = snapshot(&backend, id, 257, true).await.unwrap();
            let [Work::Split(partition)] = state.pending.as_slice() else {
                panic!(
                    "fixture must retain one cold oversized source: {}",
                    state.facts
                )
            };
            source = *partition;
        }
        let runtime = Runtime::new(
            backend.clone(),
            RuntimeConfig::default().with_maintenance(1, 1).unwrap(),
        )
        .unwrap();
        let index = runtime.open_index("cold").await.unwrap();
        let mut last = Value::Null;
        for _ in 0..100 {
            let state = snapshot(&backend, index.logical_index_id(), 257, true)
                .await
                .unwrap();
            if state.ready {
                let report = index.verify(VerifyOptions::default()).await.unwrap();
                assert!(report.complete && report.issues.is_empty());
                assert_eq!(report.objects.vector_records, 257);
                runtime.shutdown().await.unwrap();
                return;
            }
            state.advance(&backend, &retry).await.unwrap();
            last = state.facts;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("cold maintenance did not converge: {last}");
    }
}
