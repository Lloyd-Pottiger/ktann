//! Admission and native commit for the bounded two-graph publication envelope.
//! Opaque payloads exercise storage sizing; graph semantics have separate histories.
use bytes::Bytes;
use foundationdb::Database;
use ktann::storage::backend::{Backend, Mutation, ReadOps, WriteTxn};
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};
mod support;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn largest_atomic_graph_shapes_fit_adapter_and_native_limits() {
    let _network = support::boot_foundationdb();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let backend = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).unwrap(),
        BackendNamespace::new("ktann-atomic-graph-size").unwrap(),
    );
    support::clear_test_keys(&backend).await;
    let key_bytes = backend.hard_limits().max_key_bytes;
    let key = |ordinal: usize| {
        let mut key = vec![b'g'; key_bytes];
        key[..8].copy_from_slice(&(ordinal as u64).to_be_bytes());
        Bytes::from(key)
    };
    for capacity in [8192, 16384] {
        for tail_count in [0, 3] {
            let mut seed = backend.begin_write().await.unwrap();
            for ordinal in 100..100 + tail_count {
                seed.put(key(ordinal), Bytes::from_static(b"old tail"))
                    .await
                    .unwrap();
            }
            seed.commit().await.unwrap();
            let mut mutations = Vec::new();
            let mut ordinal = 0;
            for nodes in [127, capacity + 2] {
                // M=16, all nodes at the supported maximum level four.
                let mut remaining = 40 + nodes * 416;
                while remaining > 0 {
                    let payload = remaining.min(99_995);
                    mutations.push(Mutation::Put {
                        key: key(ordinal),
                        value: Bytes::from(vec![0; payload + 5]),
                    });
                    ordinal += 1;
                    remaining -= payload;
                }
            }
            // Tail deletes share the 900 KB authority envelope. Three maximum
            // keys cost <91 KB; eight authority values cost <795 KB.
            let authority_values = if tail_count == 0 { 9 } else { 8 };
            for ordinal in 0..authority_values {
                mutations.push(Mutation::Put {
                    key: Bytes::from(format!("authority/{ordinal}")),
                    value: Bytes::from(vec![0; 99_000]),
                });
            }
            for ordinal in 100..100 + tail_count {
                mutations.push(Mutation::Delete { key: key(ordinal) });
            }
            let started = std::time::Instant::now();
            let mut txn = backend.begin_write().await.unwrap();
            txn.batch_mutate(mutations).await.unwrap();
            txn.commit().await.unwrap();
            let mut snapshot = backend.begin_read().await.unwrap();
            assert!(snapshot.get(key(0)).await.unwrap().is_some());
            for ordinal in 100..100 + tail_count {
                assert!(snapshot.get(key(ordinal)).await.unwrap().is_none());
            }
            println!(
                "capacity={capacity} tail_deletes={tail_count} elapsed={:?}",
                started.elapsed()
            );
            support::clear_test_keys(&backend).await;
        }
    }
}
