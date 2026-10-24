//! Snapshot-proof rejection and conflict-read coverage without a native backend.
use super::*;
use crate::api::{IndexConfig, Metric};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::{
    CommitStart, HardLimits, InsertOutcome, Mutation, ReadOps, ScanPage,
};
use crate::storage::test_support::{MockReadTxn, pk};
use crate::storage::values::{IndexLifecycle, ValueCodec};
use std::sync::{Arc, Mutex};

struct ProofTxn {
    read: MockReadTxn,
    protected: Arc<Mutex<Vec<Bytes>>>,
}

impl ReadOps for ProofTxn {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.read.get(key).await
    }
    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.read.batch_get(keys).await
    }
    async fn scan(&mut self, range: &keys::KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        self.read.scan(range, limits).await
    }
    async fn batch_scan(
        &mut self,
        ranges: &[keys::KeyRange],
        limits: ScanLimits,
    ) -> Result<Vec<ScanPage>> {
        self.read.batch_scan(ranges, limits).await
    }
}

impl WriteTxn for ProofTxn {
    async fn get_for_update(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.protected.lock().unwrap().push(key.clone());
        self.read.get(key).await
    }
    async fn batch_get_for_update(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.protected.lock().unwrap().extend(keys.iter().cloned());
        self.read.batch_get(keys).await
    }
    async fn put(&mut self, _: Bytes, _: Bytes) -> Result<()> {
        unreachable!("proof only reads")
    }
    async fn insert(&mut self, _: Bytes, _: Bytes) -> Result<InsertOutcome> {
        unreachable!("proof only reads")
    }
    async fn delete(&mut self, _: Bytes) -> Result<()> {
        unreachable!("proof only reads")
    }
    async fn batch_mutate(&mut self, _: Vec<Mutation>) -> Result<()> {
        unreachable!("proof only reads")
    }
    async fn clear_range(&mut self, _: &keys::KeyRange) -> Result<()> {
        unreachable!("proof only reads")
    }
    async fn commit_with(self, _: CommitStart) -> Result<()> {
        unreachable!("proof only reads")
    }
    async fn rollback(self) {}
}

fn fixture() -> (IndexManifest, TreeKey, Vec<(LogicalKey, PersistentValue)>) {
    let manifest = IndexManifest::new(
        IndexLifecycle::Active,
        LogicalIndexId::new(1).unwrap(),
        IndexConfig::new(2, Metric::L2).unwrap(),
        [0; 32],
        vec![],
    )
    .unwrap();
    let tree = TreeKey::encode(&[], &[]).unwrap();
    let index = manifest.logical_index_id();
    let mut rows = vec![(
        LogicalKey::Manifest(index),
        PersistentValue::IndexManifest(manifest.clone()),
    )];
    for (key, level, count) in [(1, 3, 2), (2, 2, 1), (3, 2, 1)] {
        rows.push((
            LogicalKey::Header {
                index,
                tree_key: tree.clone(),
                partition: pk(key),
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(level, count, 7, PartitionState::Ready).unwrap(),
            ),
        ));
    }
    for (parent, child) in [(1, 2), (1, 3), (2, 4), (3, 5)] {
        rows.push((
            LogicalKey::ChildEntry {
                index,
                tree_key: tree.clone(),
                partition: pk(parent),
                child: pk(child),
            },
            PersistentValue::ChildEntry(ChildEntry::new(pk(child), vec![1.0, 0.0])),
        ));
    }
    (manifest, tree, rows)
}

fn raw(manifest: &IndexManifest, rows: &[(LogicalKey, PersistentValue)]) -> MockReadTxn {
    let codec = ValueCodec::for_index(manifest);
    MockReadTxn::new(rows.iter().map(|(key, value)| {
        (
            keys::encode_key(key).unwrap(),
            codec.encode_for_key(key, value).unwrap(),
        )
    }))
}

#[tokio::test]
async fn proof_protects_every_parent_and_rejects_changed_or_missing_parents() {
    let (manifest, tree, rows) = fixture();
    let mut read = ReadLogicalTxn::for_index(raw(&manifest, &rows), &manifest);
    assert_eq!(
        find_incoming_edge(&mut read, &manifest, &tree, pk(4), 1)
            .await
            .unwrap()
            .unwrap()
            .0,
        pk(2)
    );
    let proof = IncomingEdgeProof::read(&mut read, &manifest, &tree, 1, 4096)
        .await
        .unwrap();
    assert!(
        IncomingEdgeProof::read(&mut read, &manifest, &tree, 1, 128)
            .await
            .is_err()
    );
    for scenario in 0..4 {
        let mut changed = rows.clone();
        match scenario {
            1 => {
                // A different parent changes: the old unique-edge proof is stale.
                let (_, value) = changed.iter_mut().find(|(key, _)| matches!(key, LogicalKey::Header { partition, .. } if *partition == pk(3))).unwrap();
                *value = PersistentValue::PartitionHeader(PartitionHeader::new(2, 1, 8, PartitionState::Ready).unwrap());
            }
            2 => changed.retain(|(key, _)| !matches!(key, LogicalKey::ChildEntry { partition, child, .. } if *partition == pk(1) && *child == pk(3))),
            3 => {
                let (_, value) = changed.iter_mut().find(|(key, _)| matches!(key, LogicalKey::Header { partition, .. } if *partition == pk(1))).unwrap();
                *value = PersistentValue::PartitionHeader(PartitionHeader::new(1, 0, 8, PartitionState::Ready).unwrap());
            }
            _ => {}
        }
        let protected = Arc::new(Mutex::new(Vec::new()));
        let mut write = WriteLogicalTxn::for_index(
            ProofTxn {
                read: raw(&manifest, &changed),
                protected: protected.clone(),
            },
            &manifest,
            HardLimits {
                max_key_bytes: 10000,
                max_value_bytes: 1 << 20,
            },
            AdmissionBudget {
                max_mutations: 1000,
                max_mutation_bytes: 1 << 20,
                mutation_key_overhead_bytes: 0,
            },
        );
        assert_eq!(
            proof.validate(&mut write, &tree).await.unwrap(),
            scenario == 0
        );
        if scenario == 0 {
            for parent in [2, 3] {
                assert!(
                    protected.lock().unwrap().contains(&Bytes::from(
                        keys::encode_key(&LogicalKey::Header {
                            index: manifest.logical_index_id(),
                            tree_key: tree.clone(),
                            partition: pk(parent)
                        })
                        .unwrap()
                    ))
                );
            }
        }
    }
}
