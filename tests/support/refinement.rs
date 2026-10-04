//! Ready-tree fixture with two deliberately misplaced outliers.

use bytes::Bytes;
use ktann::api::{Index, PartitionKey, Record, Result};
use ktann::storage::WriteLogicalTxn;
use ktann::storage::backend::Backend;
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionSynopsis, PartitionTransition, PersistentValue, RecordLocation, TreeManifest,
};

pub fn partition(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("nonzero partition")
}

pub fn tree_key() -> TreeKey {
    TreeKey::encode(&[], &[]).expect("empty tree key")
}

/// Seeds real encoded records through the index, then exposes two Ready leaves.
///
/// Requires a fresh one-dimensional L2 index with no fields and maintenance
/// disabled. Leaf 2 contains [-4,-3,-2,4], leaf 3 contains [2,3,4,-4].
/// Dimension-one rotation is identity, so the projections are their means.
/// The existing index manifest supplies the production rotation seed and codec.
pub async fn seed<B: Backend>(
    index: &Index<B>,
    backend: &B,
    manifest: &IndexManifest,
) -> Result<Vec<Record>> {
    let records: Vec<_> = [-4.0_f32, -3.0, -2.0, 4.0, 2.0, 3.0, 4.0, -4.0]
        .into_iter()
        .enumerate()
        .map(|(position, value)| {
            Record::new(Bytes::from(vec![b'r', position as u8]), vec![value], vec![])
                .and_then(|record| record.with_payload(Bytes::from(vec![position as u8])))
        })
        .collect::<Result<_>>()?;
    for record in &records {
        index.upsert(record.clone()).await?;
    }
    let logical_index = manifest.logical_index_id();
    let key = tree_key();
    let root = partition(1);
    let raw = backend.begin_write().await?;
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    let root_header_key = LogicalKey::Header {
        index: logical_index,
        tree_key: key.clone(),
        partition: root,
    };
    let Some(PersistentValue::PartitionHeader(root_header)) =
        txn.get_for_update(root_header_key.clone()).await?
    else {
        panic!("upserts created the root");
    };
    txn.put(
        LogicalKey::TreeManifest {
            index: logical_index,
            tree_key: key.clone(),
        },
        PersistentValue::TreeManifest(TreeManifest::new(root, partition(3))?),
    )
    .await?;
    txn.put(
        root_header_key,
        PersistentValue::PartitionHeader(PartitionHeader::new(
            2,
            2,
            root_header.cache_epoch() + 1,
            PartitionState::Ready,
        )?),
    )
    .await?;
    txn.delete(LogicalKey::Synopsis {
        index: logical_index,
        tree_key: key.clone(),
        partition: root,
    })
    .await?;
    for (leaf, centroid) in [(partition(2), -1.25_f32), (partition(3), 1.25_f32)] {
        txn.put(
            LogicalKey::State {
                index: logical_index,
                tree_key: key.clone(),
                partition: leaf,
            },
            PersistentValue::PartitionState(PartitionTransition::Ready {
                started_at_unix_millis: 0,
            }),
        )
        .await?;
        txn.put(
            LogicalKey::Header {
                index: logical_index,
                tree_key: key.clone(),
                partition: leaf,
            },
            PersistentValue::PartitionHeader(PartitionHeader::new(1, 4, 0, PartitionState::Ready)?),
        )
        .await?;
        txn.put(
            LogicalKey::Centroid {
                index: logical_index,
                tree_key: key.clone(),
                partition: leaf,
            },
            PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![centroid])),
        )
        .await?;
        txn.put(
            LogicalKey::ChildEntry {
                index: logical_index,
                tree_key: key.clone(),
                partition: root,
                child: leaf,
            },
            PersistentValue::ChildEntry(ChildEntry::new(leaf, vec![centroid])),
        )
        .await?;
        txn.put(
            LogicalKey::Synopsis {
                index: logical_index,
                tree_key: key.clone(),
                partition: leaf,
            },
            PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(manifest)),
        )
        .await?;
    }
    for (position, record) in records.iter().enumerate() {
        let old_key = LogicalKey::LeafEntry {
            index: logical_index,
            tree_key: key.clone(),
            partition: root,
            id: record.id().clone(),
        };
        let Some(PersistentValue::LeafEntry(entry)) = txn.get_for_update(old_key.clone()).await?
        else {
            panic!("upsert created root entry");
        };
        let leaf = partition(if position < 4 { 2 } else { 3 });
        txn.put(
            LogicalKey::LeafEntry {
                index: logical_index,
                tree_key: key.clone(),
                partition: leaf,
                id: record.id().clone(),
            },
            PersistentValue::LeafEntry(entry),
        )
        .await?;
        txn.delete(old_key).await?;
        txn.put(
            LogicalKey::Location {
                index: logical_index,
                id: record.id().clone(),
            },
            PersistentValue::RecordLocation(RecordLocation::new(key.clone(), leaf)),
        )
        .await?;
    }
    txn.commit().await?;
    Ok(records)
}
