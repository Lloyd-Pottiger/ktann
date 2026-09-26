//! Read-only benchmark readiness snapshot. This is not a full record-integrity audit.
//!
//! The bridge creates exactly one Tree Key and performs no deletes. Header counts
//! suffice to observe split/merge readiness without re-reading and re-encoding a
//! million Vector Records on every poll. Production codecs own the persistent format.

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result};
use ktann::storage::backend::{Backend, ReadOps};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{PartitionState, PersistentValue, ValueCodec};
use serde_json::{Value, json};

/// One consistent snapshot and bounded centroid probes for rediscovery.
pub(super) struct Snapshot {
    pub(super) facts: Value,
    pub(super) ready: bool,
    /// Header fingerprint used only to avoid redundant rediscovery, never to prove readiness.
    pub(super) progress: u64,
    pub(super) probes: Vec<Arc<[f32]>>,
    /// The stable root has no stored centroid; any imported vector can touch it.
    pub(super) needs_root_probe: bool,
}

/// A readiness round reads at most 262,144 allocated Header slots in 256-key
/// batches and retains at most 32 maintenance probes. Missing slots are legal
/// allocator gaps or deleted partitions. All live Headers use one read snapshot.
pub(super) async fn snapshot<B: Backend>(
    backend: &B,
    index: LogicalIndexId,
    records: u64,
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
    if high_water > 262_144 {
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
    let mut probe_keys = Vec::new();
    let mut needs_root_probe = false;
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
            if needs_work && partition == tree.root() {
                needs_root_probe = true;
            }
            if needs_work && partition != tree.root() && probe_keys.len() < 32 {
                probe_keys.push(partition);
            }
        }
    }
    if !root_present {
        return Err(corrupt());
    }
    let centroid_keys = probe_keys
        .iter()
        .map(|p| Bytes::from(keys::centroid_key(index, &tree_key, *p)))
        .collect();
    let mut probes = Vec::new();
    for (partition, bytes) in probe_keys
        .into_iter()
        .zip(txn.batch_get(centroid_keys).await?)
    {
        let key = LogicalKey::Centroid {
            index,
            tree_key: tree_key.clone(),
            partition,
        };
        let PersistentValue::PartitionCentroid(centroid) =
            codec.decode(&key, bytes.ok_or_else(corrupt)?)?
        else {
            return Err(corrupt());
        };
        probes.push(Arc::from(centroid.components()));
    }
    Ok(Snapshot {
        progress: progress.digest(),
        ready: leaf_entries == records && actionable == 0 && transitional == 0,
        facts: json!({"kind":"single-tree header snapshot (not full integrity verification)","records_from_leaf_headers":leaf_entries,"partitions":partitions,"max_level":partitions_by_level.keys().next_back(),"partitions_by_level":partitions_by_level,"max_entries_by_level":max_entries_by_level,"actionable":actionable,"transitional":transitional,"allocated_header_slots":high_water}),
        probes,
        needs_root_probe,
    })
}

#[cfg(all(test, feature = "rocksdb"))]
mod tests {
    use super::*;
    use ktann::api::{IndexConfig, Metric, Mutation, Record, RuntimeConfig, VerifyOptions};
    use ktann::runtime::Runtime;
    use ktann_rocksdb::{BackendNamespace, RocksDbBackend};

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
        let metadata = snapshot(&backend, index.logical_index_id(), 64)
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
            !snapshot(&backend, index.logical_index_id(), 65)
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
        let pending = snapshot(&backend, index.logical_index_id(), 65)
            .await
            .unwrap();
        assert!(!pending.ready);
        assert!(pending.needs_root_probe);
        assert!(pending.probes.is_empty());
        assert_eq!(pending.facts["actionable"], 1);
        runtime.shutdown().await.unwrap();
    }
}
