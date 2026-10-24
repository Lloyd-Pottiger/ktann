//! Bounded Header readiness snapshots for the bridge's single Tree Key.
//!
//! Header counts establish readiness, not full record integrity.

use std::collections::BTreeMap;

use bytes::Bytes;
use ktann::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result};
use ktann::storage::backend::{Backend, ReadOps};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{PartitionState, PersistentValue, ValueCodec};
use serde_json::{Value, json};

pub(super) const MAX_HEADER_SLOTS: u64 = 262_144;

/// One consistent view of committed Header counts and topology readiness.
pub(super) struct Snapshot {
    pub(super) facts: Value,
    pub(super) ready: bool,
    #[allow(dead_code, reason = "consumed by the isolated write benchmark")]
    pub(super) pending: Vec<PartitionKey>,
}

/// A readiness round reads at most 262,144 allocated Header slots in 256-key
/// batches. Missing slots are legal
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
    if high_water > MAX_HEADER_SLOTS {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let mut partitions_by_level = BTreeMap::<u32, u64>::new();
    let mut max_entries_by_level = BTreeMap::<u32, u32>::new();
    let mut leaf_entries = 0;
    let mut partitions = 0;
    let mut transitional = 0;
    let mut actionable = 0;
    let mut pending = Vec::new();
    let mut root_present = false;
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
                || header.entry_count() > manifest.config().max_partition_entries(header.level())
                || (partition != tree.root()
                    && header.entry_count()
                        < manifest.config().min_partition_entries(header.level()));
            if needs_work && header.state() != PartitionState::ReceivingSplit {
                actionable += 1;
                pending.push(partition);
            }
        }
    }
    if !root_present {
        return Err(corrupt());
    }
    Ok(Snapshot {
        pending,
        ready: leaf_entries == records && actionable == 0 && transitional == 0,
        facts: json!({"kind":"single-tree header snapshot (not full integrity verification)","records_from_leaf_headers":leaf_entries,"partitions":partitions,"max_level":partitions_by_level.keys().next_back(),"partitions_by_level":partitions_by_level,"max_entries_by_level":max_entries_by_level,"actionable":actionable,"transitional":transitional,"allocated_header_slots":high_water}),
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
                    .with_leaf_partition_entries(16, 64)
                    .and_then(|config| config.with_internal_partition_entries(16, 64))
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
        assert_eq!(pending.facts["actionable"], 1);
        runtime.shutdown().await.unwrap();
    }
}
