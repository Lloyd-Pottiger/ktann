//! Sharded Bulk Build exercised through the public verifier, online codecs, and sub-suites.

use std::fs;

use bytes::Bytes;
use ktann::api::{
    BulkBuildOptions, BulkBuildStatus, BulkWorkerOptions, DataType, ErrorKind, FieldId,
    FieldSchema, GetOptions, IndexConfig, Metric, Mutation, PayloadProjection, Record,
    RuntimeConfig, SynopsisConfig, Value, VerifyOptions,
};
use ktann::bulk::InputSnapshot;
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann::storage::ReadLogicalTxn;
use ktann::storage::backend::{Backend, ReadOps, ScanLimits, WriteTxn};
use ktann::storage::keys::{self, LogicalKey};
use ktann::storage::values::{IndexLifecycle, IndexManifest, PersistentValue, ValueCodec};
use ktann_memory::MemoryBackend;

fn config(metric: Metric, keyed: bool) -> IndexConfig {
    let config = IndexConfig::new(3, metric)
        .unwrap()
        .with_leaf_partition_entries(2, 4)
        .and_then(|config| config.with_internal_partition_entries(2, 4))
        .unwrap()
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64)
                .unwrap()
                .with_synopsis(SynopsisConfig::MinMaxBloom {
                    expected_distinct: std::num::NonZeroU32::new(8).unwrap(),
                    false_positive_rate: 0.01,
                })
                .unwrap(),
            FieldSchema::new("name", DataType::String)
                .unwrap()
                .with_synopsis(SynopsisConfig::MinMaxBloom {
                    expected_distinct: std::num::NonZeroU32::new(8).unwrap(),
                    false_positive_rate: 0.01,
                })
                .unwrap(),
        ])
        .unwrap();
    if keyed {
        config
            .with_tree_key_fields(vec![FieldId(1), FieldId(0)])
            .unwrap()
    } else {
        config
    }
}

fn record(id: u64) -> Record {
    let record = Record::new(
        Bytes::copy_from_slice(&id.to_be_bytes()),
        vec![
            id as f32 + 1.0,
            (id % 7) as f32 + 1.0,
            (id % 11) as f32 + 1.0,
        ],
        vec![
            Value::I64((id % 3) as i64),
            Value::string(if id.is_multiple_of(2) { "a\0b" } else { "a" }).unwrap(),
        ],
    )
    .unwrap();
    match id % 3 {
        0 => record,
        1 => record.with_payload(Bytes::new()).unwrap(),
        _ => record.with_payload(Bytes::from_static(b"payload")).unwrap(),
    }
}

fn construction_options() -> ConstructionOptions {
    ConstructionOptions {
        sample_items: 8,
        memory_bytes: 2 * 1024 * 1024,
        scratch_bytes: 16 * 1024 * 1024,
    }
}

fn runtime(backend: MemoryBackend) -> Runtime<MemoryBackend> {
    Runtime::new(
        backend,
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap()
}

async fn put_manifest(backend: &MemoryBackend, manifest: &IndexManifest) {
    let mut txn = backend.begin_write().await.unwrap();
    txn.put(
        Bytes::from(keys::manifest_key(manifest.logical_index_id())),
        Bytes::from(
            ValueCodec::bootstrap()
                .encode(&PersistentValue::IndexManifest(manifest.clone()))
                .unwrap(),
        ),
    )
    .await
    .unwrap();
    txn.commit().await.unwrap();
}

async fn encoded_leaf(backend: &MemoryBackend, manifest: &IndexManifest, id: u64) -> Vec<u8> {
    let mut txn = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), manifest);
    let record_id = record(id).id().clone();
    let Some(PersistentValue::RecordLocation(location)) = txn
        .get(LogicalKey::Location {
            index: manifest.logical_index_id(),
            id: record_id.clone(),
        })
        .await
        .unwrap()
    else {
        panic!("location")
    };
    let Some(PersistentValue::LeafEntry(entry)) = txn
        .get(LogicalKey::LeafEntry {
            index: manifest.logical_index_id(),
            tree_key: location.tree_key().clone(),
            partition: location.leaf(),
            id: record_id,
        })
        .await
        .unwrap()
    else {
        panic!("entry")
    };
    ValueCodec::for_index(manifest)
        .encode(&PersistentValue::LeafEntry(entry))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bulk_emits_record_groups_in_canonical_escaped_id_order() {
    let directory = tempfile::tempdir().unwrap();
    let backend = MemoryBackend::new();
    let rt = runtime(backend.clone());
    let cfg = IndexConfig::new(3, Metric::L2)
        .unwrap()
        .with_leaf_partition_entries(2, 4)
        .and_then(|c| c.with_internal_partition_entries(2, 4))
        .unwrap();
    let ids: &[&[u8]] = &[b"a\xff", b"\0\0", b"a", b"\xff", b"a\0", b"\0"];
    let input = InputSnapshot::create(
        &directory.path().join("input"),
        cfg,
        1 << 20,
        ids.iter().map(|id| {
            Record::new(Bytes::copy_from_slice(id), vec![1.0, 2.0, 3.0], vec![])?
                .with_payload(Bytes::from_static(b"payload"))
        }),
    )
    .unwrap();
    let job = rt
        .start_bulk_build("escaped", &input, construction_options())
        .await
        .unwrap();
    let work = directory.path().join("work");
    fs::create_dir(&work).unwrap();
    let index = job.complete(BulkWorkerOptions::new(work)).await.unwrap();
    let report = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty());

    let id_num = index.logical_index_id();
    let mut start = keys::index_range(id_num).start().to_vec();
    start.push(0x01);
    let mut end = keys::index_range(id_num).start().to_vec();
    end.push(0x02);
    let mut rtxn = backend.begin_read().await.unwrap();
    let page = rtxn
        .scan(
            &keys::KeyRange::new(start, end),
            ScanLimits {
                item_limit: 100,
                byte_limit: 100_000,
            },
        )
        .await
        .unwrap();
    let actual: Vec<Bytes> = page.items().iter().map(|it| it.key().clone()).collect();
    let mut expected = Vec::new();
    for id in ids {
        let id = Bytes::copy_from_slice(id);
        for key in [
            keys::record_key(id_num, &id),
            keys::location_key(id_num, &id),
            keys::payload_key(id_num, &id),
        ] {
            expected.push(Bytes::from(key.unwrap()));
        }
    }
    expected.sort();
    assert_eq!(actual, expected);
    rt.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_memory_and_disk_modes_match_each_other_and_online_leaf_values() {
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        for keyed in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let backend = MemoryBackend::new();
            let rt = runtime(backend.clone());
            let cfg = config(metric, keyed);
            let input = InputSnapshot::create(
                &directory.path().join("input"),
                cfg.clone(),
                1024 * 1024,
                (0..73).rev().map(|id| Ok(record(id))),
            )
            .unwrap();

            // Build 1: zero-disk in-memory mode.
            let job_mem = rt
                .start_bulk_build("bulk-mem", &input, construction_options())
                .await
                .unwrap();
            let work_mem = directory.path().join("work-mem");
            fs::create_dir(&work_mem).unwrap();
            let (idx_mem, rep_mem) = job_mem
                .complete_with_report(BulkWorkerOptions::new(work_mem), Default::default())
                .await
                .unwrap();
            assert!(rep_mem.in_memory_mode);
            assert_eq!(rep_mem.scratch_written_bytes, 0);

            // Build 2: constrained disk BuildUnit mode (`memory_bytes` forces disk spill).
            let job_disk = rt
                .start_bulk_build("bulk-disk", &input, construction_options())
                .await
                .unwrap();
            let work_disk = directory.path().join("work-disk");
            fs::create_dir(&work_disk).unwrap();
            let mut disk_opts = BulkWorkerOptions::new(work_disk);
            disk_opts.memory_bytes = 8 * 1024; // < needed resident footprint, >= 16 rows
            let (idx_disk, rep_disk) = job_disk
                .complete_with_report(disk_opts, Default::default())
                .await
                .unwrap();
            assert!(!rep_disk.in_memory_mode);
            assert!(rep_disk.scratch_written_bytes > 0);

            // Build 3: disk BuildUnit mode with multi-round refinement (heavy + light shards).
            let job_disk_ref = rt
                .start_bulk_build(
                    "bulk-disk-ref",
                    &input,
                    BulkBuildOptions {
                        construction: construction_options(),
                        refinement_rounds: Some(2),
                    },
                )
                .await
                .unwrap();
            let work_disk_ref = directory.path().join("work-disk-ref");
            fs::create_dir(&work_disk_ref).unwrap();
            let mut disk_ref_opts = BulkWorkerOptions::new(work_disk_ref);
            disk_ref_opts.distributed_workers = 2;
            disk_ref_opts.memory_bytes = 256 * 1024;
            let (idx_disk_ref, rep_disk_ref) = job_disk_ref
                .complete_with_report(disk_ref_opts, Default::default())
                .await
                .unwrap();
            assert!(!rep_disk_ref.in_memory_mode);
            assert!(rep_disk_ref.scratch_written_bytes > 0);
            assert_eq!(
                job_disk_ref.status().await.unwrap(),
                BulkBuildStatus::Published
            );

            for idx in [&idx_mem, &idx_disk, &idx_disk_ref] {
                let verified = idx.verify(VerifyOptions::default()).await.unwrap();
                assert!(
                    verified.complete && verified.issues.is_empty(),
                    "{verified:?}"
                );
                assert_eq!(verified.objects.vector_records, 73);
                for id in 0..73 {
                    let expected = record(id);
                    let actual = idx
                        .get(expected.id().clone(), GetOptions::default().with_payload())
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(actual.vector(), expected.vector());
                    assert_eq!(actual.fields(), expected.fields());
                    let payload = expected.payload().map_or(PayloadProjection::Absent, |p| {
                        PayloadProjection::Present(p.clone())
                    });
                    assert_eq!(actual.payload(), &payload);
                }
                // Post-publication cleanup reclaims BuildWorkspace and transient BuildUnits
                // while keeping compact BuildDescriptor and BuildProgress for idempotent status/reports.
                let lid = idx.logical_index_id();
                let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await.unwrap());
                assert!(
                    rtxn.get(LogicalKey::BuildWorkspace(lid))
                        .await
                        .unwrap()
                        .is_none()
                );
                match rtxn.get(LogicalKey::BuildProgress(lid)).await.unwrap() {
                    Some(PersistentValue::BuildProgress(p)) => {
                        assert!(p.total_entries() > 0);
                        assert_eq!(p.validation_slices(), 1);
                    }
                    other => panic!("expected compact BuildProgress, got {other:?}"),
                }
                assert!(
                    rtxn.get(LogicalKey::BuildDescriptor(lid))
                        .await
                        .unwrap()
                        .is_some()
                );
                drop(rtxn);
                let mut raw_rtxn = backend.begin_read().await.unwrap();
                let unit_start = keys::build_unit_key(lid, 0, 0);
                let unit_end = keys::build_unit_key(lid, u8::MAX, u32::MAX);
                let leftover_units = raw_rtxn
                    .scan(
                        &keys::KeyRange::new(unit_start, unit_end),
                        ScanLimits {
                            item_limit: 10,
                            byte_limit: 64 * 1024,
                        },
                    )
                    .await
                    .unwrap();
                assert!(leftover_units.items().is_empty());
            }

            // Idempotent `complete_with_report` after publication + cleanup still reports exact counts.
            let idempotent_work = directory.path().join("work-idempotent");
            fs::create_dir(&idempotent_work).unwrap();
            let (_, rep_again) = job_mem
                .complete_with_report(BulkWorkerOptions::new(idempotent_work), Default::default())
                .await
                .unwrap();
            assert_eq!(rep_again.records, 73);
            assert!(rep_again.trees > 0);
            assert!(rep_again.partitions > 0);
            assert!(rep_again.entries > 0);

            // Compare LeafEntry encoding bit-for-bit against online inserts sharing rotation_seed.
            let mut rtxn = ReadLogicalTxn::bootstrap(backend.begin_read().await.unwrap());
            let Some(PersistentValue::IndexManifest(manifest)) = rtxn
                .get(LogicalKey::Manifest(idx_mem.logical_index_id()))
                .await
                .unwrap()
            else {
                panic!("manifest")
            };
            drop(rtxn);
            let online = rt.create_index("online", cfg.clone()).await.unwrap();
            let online_manifest = IndexManifest::new(
                IndexLifecycle::Active,
                online.logical_index_id(),
                cfg,
                *manifest.rotation_seed(),
                manifest.bloom_parameters().to_vec(),
            )
            .unwrap();
            put_manifest(&backend, &online_manifest).await;
            let online = rt.open_index("online").await.unwrap();
            online
                .batch_mutate((0..73).map(|id| Mutation::Insert(record(id))).collect())
                .await
                .unwrap();
            for id in 0..73 {
                assert_eq!(
                    encoded_leaf(&backend, &manifest, id).await,
                    encoded_leaf(&backend, &online_manifest, id).await
                );
            }
            rt.shutdown().await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cross_segment_duplicate_ids_fail_with_record_already_exists() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(MemoryBackend::new());
    let cfg = IndexConfig::new(2, Metric::L2)
        .unwrap()
        .with_leaf_partition_entries(2, 4)
        .and_then(|c| c.with_internal_partition_entries(2, 4))
        .unwrap();
    // 17,000 items span two segments (`seg-00000000.bin` and `seg-00000001.bin`);
    // duplicate ID `42` in segment 1 must be caught by bucketed global ID verification.
    let input = InputSnapshot::create(
        &dir.path().join("source"),
        cfg,
        64 << 20,
        (0..17_000_u64).map(|i| {
            let id = if i == 16_900 { 42 } else { i };
            Record::new(
                Bytes::copy_from_slice(&id.to_be_bytes()),
                vec![i as f32 + 1.0, 1.0],
                vec![],
            )
        }),
    )
    .unwrap();
    let job = rt
        .start_bulk_build("dup", &input, construction_options())
        .await
        .unwrap();
    let work = dir.path().join("work");
    fs::create_dir(&work).unwrap();
    assert_eq!(
        job.complete(BulkWorkerOptions::new(work))
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::RecordAlreadyExists
    );
    rt.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resident_mode_does_not_pin_large_payload_frames_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(MemoryBackend::new());
    let cfg = IndexConfig::new(4, Metric::L2)
        .unwrap()
        .with_leaf_partition_entries(4, 16)
        .and_then(|c| c.with_internal_partition_entries(2, 8))
        .unwrap();
    // 200 records * 16 KiB payload = 3.2 MiB of raw segment frames, while resident `memory_bytes`
    // is 8 MiB; verify resident fast path runs cleanly and preserves exact payloads.
    let big_payload = Bytes::from(vec![0xAB_u8; 16 * 1024]);
    let input = InputSnapshot::create(
        &dir.path().join("source"),
        cfg,
        64 << 20,
        (0..200_u64).map(|i| {
            Record::new(
                Bytes::copy_from_slice(&i.to_be_bytes()),
                vec![i as f32, 1.0, 2.0, 3.0],
                vec![],
            )
            .and_then(|r| r.with_payload(big_payload.clone()))
        }),
    )
    .unwrap();
    let job = rt
        .start_bulk_build(
            "payload-unpinned",
            &input,
            ConstructionOptions {
                sample_items: 8,
                memory_bytes: 16 * 1024 * 1024,
                scratch_bytes: 64 * 1024 * 1024,
            },
        )
        .await
        .unwrap();
    let work = dir.path().join("work");
    fs::create_dir(&work).unwrap();
    let mut opts = BulkWorkerOptions::new(work);
    opts.memory_bytes = 8 * 1024 * 1024;
    let (idx, rep) = job
        .complete_with_report(opts, Default::default())
        .await
        .unwrap();
    assert!(rep.in_memory_mode);
    assert_eq!(rep.records, 200);
    let sample = idx
        .get(
            Bytes::copy_from_slice(&137_u64.to_be_bytes()),
            GetOptions::default().with_payload(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(sample.payload(), &PayloadProjection::Present(big_payload));
    rt.shutdown().await.unwrap();
}

#[path = "bulk_serving/load.rs"]
mod load;

#[path = "bulk_serving/graphs.rs"]
mod graphs;

#[path = "bulk_serving/routing.rs"]
mod routing;

#[path = "bulk_serving/filtered.rs"]
mod filtered;
