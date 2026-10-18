//! Graph key/value envelopes through each production adapter's typed storage path.

use bytes::Bytes;
use ktann::api::{IndexConfig, LogicalIndexId, Metric, PartitionKey};
use ktann::storage::backend::{Backend, ScanLimits};
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    GRAPH_CHUNK_BYTES, GraphChunk, IndexLifecycle, IndexManifest, PersistentValue, ValueCodec,
};
use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

/// A complete two-chunk body proves the maximum KV size and pagination contract.
/// Topology restoration and source binding are tested by the owning core module.
pub async fn exercise<B: Backend>(backend: &B) {
    let index = LogicalIndexId::new(0x19504).unwrap();
    let partition = PartitionKey::new(1).unwrap();
    let tree = TreeKey::encode(&[], &[]).unwrap();
    let manifest = IndexManifest::new(
        IndexLifecycle::Active,
        index,
        IndexConfig::new(1, Metric::L2)
            .unwrap()
            .with_internal_partition_entries(4096, 16384)
            .unwrap(),
        [0; 32],
        vec![],
    )
    .unwrap();
    let key = LogicalKey::GraphDescriptor {
        index,
        tree_key: tree.clone(),
        partition,
    };
    let codec = ValueCodec::for_index(&manifest);
    // Canonical descriptor fixture: a sufficiently large node bound permits the body.
    let mut bytes = vec![0x12, if cfg!(target_endian = "little") { 1 } else { 2 }];
    bytes.extend_from_slice(&9000_u32.to_be_bytes());
    bytes.extend_from_slice(&1_u64.to_be_bytes());
    bytes.extend_from_slice(&((GRAPH_CHUNK_BYTES + 1) as u64).to_be_bytes());
    bytes.extend_from_slice(&[0; 64]);
    let descriptor = codec.decode(&key, bytes.into()).unwrap();
    let mut write = WriteLogicalTxn::for_index(
        backend.begin_write().await.unwrap(),
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    write.put(key.clone(), descriptor.clone()).await.unwrap();
    let mut expected = Vec::new();
    for (chunk, length) in [(0, GRAPH_CHUNK_BYTES), (1, 1)] {
        let key = LogicalKey::GraphChunk {
            index,
            tree_key: tree.clone(),
            partition,
            chunk,
        };
        let value = PersistentValue::GraphChunk(
            GraphChunk::new(Bytes::from(vec![chunk as u8; length])).unwrap(),
        );
        write.put(key.clone(), value.clone()).await.unwrap();
        expected.push((key, value));
    }
    write.commit().await.unwrap();
    let mut read = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
    assert_eq!(read.get(key).await.unwrap(), Some(descriptor));
    let range = LogicalRange::graph_chunks(&manifest, &tree, partition).unwrap();
    let mut cursor = None;
    let mut found = Vec::new();
    loop {
        let page = read
            .scan(
                &range,
                cursor.as_ref(),
                ScanLimits {
                    item_limit: 1,
                    byte_limit: 128 * 1024,
                },
            )
            .await
            .unwrap();
        for item in page.items() {
            found.push((item.key().clone(), item.value().clone()));
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(found, expected);
}
