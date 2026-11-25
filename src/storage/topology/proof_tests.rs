//! Paged discovery, snapshot-proof rejection, and conflict-read coverage.
use super::*;
use crate::api::{IndexConfig, LogicalIndexId, Metric};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::{
    AdmissionBudget, CommitStart, HardLimits, InsertOutcome, Mutation, ReadOps, ScanPage,
};
use crate::storage::keys;
use crate::storage::test_support::{MockReadTxn, pk};
use crate::storage::values::{IndexLifecycle, PersistentValue, ValueCodec};
use bytes::Bytes;
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
async fn proof_isolates_distinct_parents_and_rejects_stale_or_corrupt_edges() {
    let (manifest, tree, rows) = fixture();
    let index = manifest.logical_index_id();
    let mut read = ReadLogicalTxn::for_index(raw(&manifest, &rows), &manifest);
    let incoming = find_incoming_edge(&mut read, &manifest, &tree, pk(4), 1)
        .await
        .unwrap();
    assert_eq!(incoming.as_ref().unwrap().0, pk(2));
    let proof = IncomingEdgeProof::read(&mut read, &manifest, &tree, pk(4), 1)
        .await
        .unwrap();
    assert_eq!(proof.parent(), pk(2));
    // Each call descends from level 3 to level 2 by scanning only the root's
    // child range (`1`), then resolves `pk(4)` at level 2 via point reads
    // without scanning the two level-2 bodies.
    assert_eq!(read.into_raw().batch_scan_sizes, vec![1, 1]);

    // Missing or duplicate incoming edges in the discovery snapshot fail closed.
    let mut missing_edge = rows.clone();
    missing_edge.retain(|(key, _)| {
        !matches!(
            key,
            LogicalKey::ChildEntry { partition, child, .. }
                if *partition == pk(2) && *child == pk(4)
        )
    });
    let mut read = ReadLogicalTxn::for_index(raw(&manifest, &missing_edge), &manifest);
    assert_eq!(
        IncomingEdgeProof::read(&mut read, &manifest, &tree, pk(4), 1)
            .await
            .unwrap_err()
            .kind(),
        crate::api::ErrorKind::Corruption,
    );

    let mut duplicate_edge = rows.clone();
    duplicate_edge.push((
        LogicalKey::ChildEntry {
            index,
            tree_key: tree.clone(),
            partition: pk(3),
            child: pk(4),
        },
        PersistentValue::ChildEntry(ChildEntry::new(pk(4), vec![1.0, 0.0])),
    ));
    let mut read = ReadLogicalTxn::for_index(raw(&manifest, &duplicate_edge), &manifest);
    assert_eq!(
        IncomingEdgeProof::read(&mut read, &manifest, &tree, pk(4), 1)
            .await
            .unwrap_err()
            .kind(),
        crate::api::ErrorKind::Corruption,
    );

    let make_write = |changed: Vec<(LogicalKey, PersistentValue)>| {
        let protected = Arc::new(Mutex::new(Vec::new()));
        let write = WriteLogicalTxn::for_index(
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
        (write, protected)
    };

    // Unchanged state and unrelated sibling-parent Header updates both validate
    // without update-protecting sibling parent `pk(3)`.
    for bump_sibling_parent in [false, true] {
        let mut changed = rows.clone();
        if bump_sibling_parent {
            let (_, value) = changed
                .iter_mut()
                .find(|(key, _)| {
                    matches!(key, LogicalKey::Header { partition, .. } if *partition == pk(3))
                })
                .unwrap();
            *value = PersistentValue::PartitionHeader(
                PartitionHeader::new(2, 1, 8, PartitionState::Ready).unwrap(),
            );
        }
        let (mut write, protected) = make_write(changed);
        assert!(proof.validate(&mut write, &tree).await.unwrap());
        let locked = protected.lock().unwrap().clone();
        assert!(
            locked.contains(&Bytes::from(
                keys::encode_key(&LogicalKey::Header {
                    index,
                    tree_key: tree.clone(),
                    partition: pk(2),
                })
                .unwrap()
            ))
        );
        assert!(
            locked.contains(&Bytes::from(
                keys::encode_key(&LogicalKey::ChildEntry {
                    index,
                    tree_key: tree.clone(),
                    partition: pk(2),
                    child: pk(4),
                })
                .unwrap()
            ))
        );
        assert!(
            !locked.contains(&Bytes::from(
                keys::encode_key(&LogicalKey::Header {
                    index,
                    tree_key: tree.clone(),
                    partition: pk(3),
                })
                .unwrap()
            ))
        );
    }

    // Relocating `pk(4)` from `pk(2)` to `pk(3)` invalidates the proof cleanly (`Ok(false)`).
    let mut moved = missing_edge;
    moved.push((
        LogicalKey::ChildEntry {
            index,
            tree_key: tree.clone(),
            partition: pk(3),
            child: pk(4),
        },
        PersistentValue::ChildEntry(ChildEntry::new(pk(4), vec![1.0, 0.0])),
    ));
    let (mut write, _) = make_write(moved);
    assert!(!proof.validate(&mut write, &tree).await.unwrap());

    // A duplicate edge or orphaned owning parent in the write snapshot fails closed.
    let (mut write, _) = make_write(duplicate_edge);
    assert_eq!(
        proof.validate(&mut write, &tree).await.unwrap_err().kind(),
        crate::api::ErrorKind::Corruption,
    );

    let mut orphaned_parent = rows.clone();
    orphaned_parent.retain(|(key, _)| {
        !matches!(
            key,
            LogicalKey::ChildEntry { partition, child, .. }
                if *partition == pk(1) && *child == pk(2)
        )
    });
    let (mut write, _) = make_write(orphaned_parent);
    assert_eq!(
        proof.validate(&mut write, &tree).await.unwrap_err().kind(),
        crate::api::ErrorKind::Corruption,
    );
}

#[tokio::test]
async fn discovery_visits_every_edge_once_across_uneven_pages_and_waves() {
    use std::collections::BTreeMap;

    let (manifest, tree_key, _) = fixture();
    let mut items = Vec::new();
    let mut expected = BTreeMap::new();
    // Empty, short, exact-page and multiple-page bodies span two waves.
    let sizes = [0, 1, 128, 129, 257, 2, 0, 130, 3, 258];
    let bodies: Vec<_> = (1..=sizes.len() as u64).map(pk).collect();
    for (&body, size) in bodies.iter().zip(sizes) {
        for child in 100..100 + size {
            let child = pk(child);
            let centroid = vec![child.get() as f32; manifest.config().dimension()];
            items.push((
                keys::child_entry_key(manifest.logical_index_id(), &tree_key, body, child),
                ValueCodec::for_index(&manifest)
                    .encode(&PersistentValue::ChildEntry(ChildEntry::new(
                        child, centroid,
                    )))
                    .expect("encode child"),
            ));
            expected.insert((body, child), 0);
        }
    }
    let mut read = ReadLogicalTxn::for_index(MockReadTxn::new(items), &manifest);
    super::scan_child_edges(
        &mut read,
        &manifest,
        &tree_key,
        &bodies,
        &mut |body, entry| {
            *expected
                .get_mut(&(body, entry.child()))
                .expect("known edge") += 1;
            Ok(())
        },
    )
    .await
    .expect("discovery");
    assert!(expected.values().all(|&visits| visits == 1));
    let raw = read.into_raw();
    assert!(
        raw.batch_scan_sizes
            .iter()
            .all(|&size| size <= super::DISCOVERY_BODY_BATCH)
    );
    assert!(raw.batch_scan_sizes.iter().any(|&size| size > 1));
}
