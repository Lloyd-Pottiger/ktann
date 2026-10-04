//! Ready-to-Ready refinement preserves the atomic membership contract.

use bytes::Bytes;
use ktann::api::{ErrorKind, IndexConfig, Metric, RuntimeConfig};
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;
use ktann::storage::keys::LogicalKey;
use ktann::storage::topology::{self, Movement};
use ktann::storage::values::{IndexManifest, PartitionState, PersistentValue};
use ktann::storage::{ReadLogicalTxn, WriteLogicalTxn};
use ktann_memory::MemoryBackend;

#[path = "support/refinement.rs"]
mod fixture;

async fn setup() -> (MemoryBackend, IndexManifest) {
    let backend = MemoryBackend::new();
    let runtime = Runtime::new(
        backend.clone(),
        RuntimeConfig::default()
            .with_maintenance(0, 1)
            .and_then(|config| config.with_import_limits(1, 1))
            .expect("manual maintenance"),
    )
    .expect("runtime");
    let index = runtime
        .create_index(
            "refinement-storage",
            IndexConfig::new(1, Metric::L2)
                .expect("config")
                .with_partition_entries(2, 8)
                .expect("capacity"),
        )
        .await
        .expect("index");
    let mut txn = ReadLogicalTxn::bootstrap(backend.begin_read().await.expect("read"));
    let manifest = match txn
        .get(LogicalKey::Manifest(index.logical_index_id()))
        .await
        .expect("manifest")
    {
        Some(PersistentValue::IndexManifest(manifest)) => manifest,
        _ => panic!("index manifest"),
    };
    fixture::seed(&index, &backend, &manifest)
        .await
        .expect("fixture");
    (backend, manifest)
}

async fn read(
    backend: &MemoryBackend,
    manifest: &IndexManifest,
    key: LogicalKey,
) -> Option<PersistentValue> {
    let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.expect("read"), manifest);
    txn.get(key).await.expect("value")
}

fn header_key(manifest: &IndexManifest, partition: u64) -> LogicalKey {
    LogicalKey::Header {
        index: manifest.logical_index_id(),
        tree_key: fixture::tree_key(),
        partition: fixture::partition(partition),
    }
}
fn entry_key(manifest: &IndexManifest, partition: u64, id: Bytes) -> LogicalKey {
    LogicalKey::LeafEntry {
        index: manifest.logical_index_id(),
        tree_key: fixture::tree_key(),
        partition: fixture::partition(partition),
        id,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ready_refinement_moves_membership_and_epochs_without_rewriting_records() {
    let (backend, manifest) = setup().await;
    let id = Bytes::from_static(&[b'r', 3]);
    let index = manifest.logical_index_id();
    let source_key = entry_key(&manifest, 2, id.clone());
    let target_key = entry_key(&manifest, 3, id.clone());
    let entry = read(&backend, &manifest, source_key.clone()).await;
    let record_key = LogicalKey::Record {
        index,
        id: id.clone(),
    };
    let payload_key = LogicalKey::Payload {
        index,
        id: id.clone(),
    };
    let record = read(&backend, &manifest, record_key.clone()).await;
    let payload = read(&backend, &manifest, payload_key.clone()).await;
    let parent = read(&backend, &manifest, header_key(&manifest, 1)).await;
    let synopsis_key = LogicalKey::Synopsis {
        index,
        tree_key: fixture::tree_key(),
        partition: fixture::partition(3),
    };
    let synopsis = read(&backend, &manifest, synopsis_key.clone()).await;
    let mut txn = WriteLogicalTxn::for_index(
        backend.begin_write().await.expect("write"),
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    let candidates = topology::read_leaf_drain_candidates(
        &mut txn,
        &fixture::tree_key(),
        fixture::partition(2),
        std::slice::from_ref(&id),
    )
    .await
    .expect("candidates");
    let moves = vec![(
        candidates.into_iter().next().expect("slot").expect("entry"),
        fixture::partition(3),
    )];
    assert_eq!(
        topology::relocate_leaf_entries(
            &mut txn,
            &fixture::tree_key(),
            fixture::partition(2),
            moves,
            Movement::Refine
        )
        .await
        .expect("move"),
        1
    );
    // Readers cannot observe the relocation before its one commit.
    assert_eq!(read(&backend, &manifest, source_key.clone()).await, entry);
    assert_eq!(read(&backend, &manifest, target_key.clone()).await, None);
    txn.commit().await.expect("commit");
    assert_eq!(read(&backend, &manifest, source_key).await, None);
    assert_eq!(read(&backend, &manifest, target_key).await, entry);
    let Some(PersistentValue::RecordLocation(location)) =
        read(&backend, &manifest, LogicalKey::Location { index, id }).await
    else {
        panic!("record location");
    };
    assert_eq!(location.leaf(), fixture::partition(3));
    assert_eq!(location.tree_key(), &fixture::tree_key());
    for (partition, count) in [(2, 3), (3, 5)] {
        let Some(PersistentValue::PartitionHeader(header)) =
            read(&backend, &manifest, header_key(&manifest, partition)).await
        else {
            panic!("partition header");
        };
        assert_eq!(header.entry_count(), count);
        assert_eq!(header.cache_epoch(), 1);
        assert_eq!(header.state(), PartitionState::Ready);
    }
    assert_eq!(read(&backend, &manifest, record_key).await, record);
    assert_eq!(read(&backend, &manifest, payload_key).await, payload);
    assert_eq!(
        read(&backend, &manifest, header_key(&manifest, 1)).await,
        parent
    );
    assert_eq!(read(&backend, &manifest, synopsis_key).await, synopsis);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aborted_refinement_leaves_ready_membership_unchanged() {
    let (backend, manifest) = setup().await;
    let id = Bytes::from_static(&[b'r', 3]);
    let source_key = entry_key(&manifest, 2, id.clone());
    let entry = read(&backend, &manifest, source_key.clone()).await;
    let header = read(&backend, &manifest, header_key(&manifest, 2)).await;
    let mut txn = WriteLogicalTxn::for_index(
        backend.begin_write().await.expect("write"),
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    let candidates = topology::read_leaf_drain_candidates(
        &mut txn,
        &fixture::tree_key(),
        fixture::partition(2),
        std::slice::from_ref(&id),
    )
    .await
    .expect("candidates");
    topology::relocate_leaf_entries(
        &mut txn,
        &fixture::tree_key(),
        fixture::partition(2),
        vec![(
            candidates.into_iter().next().expect("slot").expect("entry"),
            fixture::partition(3),
        )],
        Movement::Refine,
    )
    .await
    .expect("move");
    drop(txn);
    assert_eq!(read(&backend, &manifest, source_key).await, entry);
    assert_eq!(
        read(&backend, &manifest, entry_key(&manifest, 3, id)).await,
        None
    );
    assert_eq!(
        read(&backend, &manifest, header_key(&manifest, 2)).await,
        header
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn refinement_rejects_a_move_back_into_its_source() {
    let (backend, manifest) = setup().await;
    let id = Bytes::from_static(&[b'r', 3]);
    let mut txn = WriteLogicalTxn::for_index(
        backend.begin_write().await.expect("write"),
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    let candidates = topology::read_leaf_drain_candidates(
        &mut txn,
        &fixture::tree_key(),
        fixture::partition(2),
        &[id],
    )
    .await
    .expect("candidates");
    let error = topology::relocate_leaf_entries(
        &mut txn,
        &fixture::tree_key(),
        fixture::partition(2),
        vec![(
            candidates.into_iter().next().expect("slot").expect("entry"),
            fixture::partition(2),
        )],
        Movement::Refine,
    )
    .await
    .expect_err("same leaf");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}
