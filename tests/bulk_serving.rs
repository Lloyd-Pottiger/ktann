//! Serving artifacts exercised through the ordinary verifier and online codecs.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, GetOptions, IndexConfig, Metric, Mutation,
    PayloadProjection, Record, RuntimeConfig, SynopsisConfig, Value, VerifyOptions,
};
use ktann::bulk::{ForestArtifact, ForestOptions, InputSnapshot, ServingArtifact, ServingOptions};
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann::storage::ReadLogicalTxn;
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys::{self, LogicalKey};
use ktann::storage::values::{IndexLifecycle, IndexManifest, PersistentValue, ValueCodec};
use ktann_memory::MemoryBackend;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ktann-serving-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn config(metric: Metric, keyed: bool) -> IndexConfig {
    let config = IndexConfig::new(3, metric)
        .unwrap()
        .with_partition_entries(2, 4)
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
fn options() -> ForestOptions {
    ForestOptions {
        tree: ConstructionOptions {
            min_partition_entries: 2,
            max_partition_entries: 4,
            sample_items: 8,
            memory_bytes: 2 * 1024 * 1024,
            scratch_bytes: 16 * 1024 * 1024,
        },
        sort_memory_bytes: 1024 * 1024,
        sort_scratch_bytes: 32 * 1024 * 1024,
    }
}
fn serving_options(backend: &MemoryBackend) -> ServingOptions {
    ServingOptions {
        memory_bytes: 8 * 1024 * 1024,
        scratch_bytes: 64 * 1024 * 1024,
        hard_limits: backend.hard_limits(),
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
async fn serving_streams_record_groups_in_escaped_id_order() {
    let directory = Directory::new();
    let backend = MemoryBackend::new();
    let runtime = runtime(backend.clone());
    let config = IndexConfig::new(3, Metric::L2)
        .unwrap()
        .with_partition_entries(2, 4)
        .unwrap();
    let ids: &[&[u8]] = &[b"a\xff", b"\0\0", b"a", b"\xff", b"a\0", b"\0"];
    let input = InputSnapshot::create(
        &directory.0.join("input"),
        config,
        1 << 20,
        ids.iter().map(|id| {
            Record::new(Bytes::copy_from_slice(id), vec![1.0, 2.0, 3.0], vec![])?
                .with_payload(Bytes::from_static(b"payload"))
        }),
    )
    .unwrap();
    let job = runtime
        .start_bulk_build("bulk", &input, options().tree)
        .await
        .unwrap();
    let manifest = job.index_manifest();
    let (forest, _) = ForestArtifact::build(
        &directory.0.join("forest"),
        &input,
        *manifest.rotation_seed(),
        options(),
        1 << 20,
    )
    .unwrap();
    let (artifact, _) = ServingArtifact::build(
        &directory.0.join("serving"),
        &input,
        &forest,
        manifest,
        serving_options(&backend),
        1 << 20,
    )
    .unwrap();
    let actual: Vec<_> = artifact
        .reader()
        .unwrap()
        .map(|entry| entry.unwrap().key)
        .collect();
    let mut expected = Vec::new();
    for id in ids {
        let id = Bytes::copy_from_slice(id);
        let index = manifest.logical_index_id();
        for key in [
            keys::record_key(index, &id),
            keys::location_key(index, &id),
            keys::payload_key(index, &id),
        ] {
            expected.push(Bytes::from(key.unwrap()));
        }
    }
    expected.sort();
    assert_eq!(&actual[..expected.len()], expected);
    assert!(actual.windows(2).all(|pair| pair[0] < pair[1]));
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn encoded_artifact_verifies_and_matches_online_leaf_values_for_every_metric() {
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        for keyed in [false, true] {
            let directory = Directory::new();
            let backend = MemoryBackend::new();
            let runtime = runtime(backend.clone());
            let config = config(metric, keyed);
            let input = InputSnapshot::create(
                &directory.0.join("input"),
                config.clone(),
                1024 * 1024,
                (0..73).rev().map(|id| Ok(record(id))),
            )
            .unwrap();
            let job = runtime
                .start_bulk_build("bulk", &input, options().tree)
                .await
                .unwrap();
            let manifest = job.index_manifest().clone();
            let (forest, _) = ForestArtifact::build(
                &directory.0.join("forest"),
                &input,
                *manifest.rotation_seed(),
                options(),
                1024 * 1024,
            )
            .unwrap();
            let (artifact, report) = ServingArtifact::build(
                &directory.0.join("serving"),
                &input,
                &forest,
                &manifest,
                serving_options(&backend),
                4 * 1024 * 1024,
            )
            .unwrap();
            assert_eq!(report.records, 73);
            assert!(report.partitions > 1);
            assert!(!directory.0.join("serving/scratch").exists());
            let reopened = ServingArtifact::open(
                &directory.0.join("serving"),
                artifact.manifest().clone(),
                &input,
                &forest,
                &manifest,
                serving_options(&backend),
            )
            .unwrap();
            job.load_serving(
                &reopened,
                ktann::api::BulkLoadOptions {
                    max_mutations: 17,
                    max_bytes: 4096,
                },
            )
            .await
            .unwrap();
            assert_eq!(
                job.status().await.unwrap(),
                ktann::api::BulkBuildStatus::Loaded {
                    entries: report.keys
                }
            );
            // Test fixture only: publication still requires sealed backend proofs.
            assert_eq!(
                runtime.open_index("bulk").await.unwrap_err().kind(),
                ErrorKind::IndexBuilding
            );
            put_manifest(&backend, &manifest.with_lifecycle(IndexLifecycle::Active)).await;
            let index = runtime.open_index("bulk").await.unwrap();
            let verified = index.verify(VerifyOptions::default()).await.unwrap();
            assert!(
                verified.complete && verified.issues.is_empty(),
                "{verified:?}"
            );
            assert_eq!(verified.objects.vector_records, 73);
            for id in 0..73 {
                let expected = record(id);
                let actual = index
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
            let online = runtime
                .create_index("online", config.clone())
                .await
                .unwrap();
            let online_manifest = IndexManifest::new(
                IndexLifecycle::Active,
                online.logical_index_id(),
                config.clone(),
                *manifest.rotation_seed(),
                manifest.bloom_parameters().to_vec(),
            )
            .unwrap();
            put_manifest(&backend, &online_manifest).await;
            let online = runtime.open_index("online").await.unwrap();
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
            let mut corrupt = backend.begin_write().await.unwrap();
            corrupt
                .put(
                    Bytes::from(keys::build_descriptor_key(manifest.logical_index_id())),
                    Bytes::from_static(b"invalid descriptor"),
                )
                .await
                .unwrap();
            corrupt.commit().await.unwrap();
            assert!(
                index
                    .verify(VerifyOptions::default())
                    .await
                    .unwrap()
                    .issues
                    .iter()
                    .any(|issue| issue.kind == ktann::api::VerifyIssueKind::InvalidEncoding)
            );
            runtime.shutdown().await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serving_empty_input_and_resource_failures_never_publish_or_seal_partial_output() {
    let directory = Directory::new();
    let backend = MemoryBackend::new();
    let runtime = runtime(backend.clone());
    for count in [0, 10] {
        let input = InputSnapshot::create(
            &directory.0.join(format!("input-{count}")),
            config(Metric::L2, true),
            1024 * 1024,
            (0..count).map(|id| Ok(record(id))),
        )
        .unwrap();
        let job = runtime
            .start_bulk_build(&format!("bulk-{count}"), &input, options().tree)
            .await
            .unwrap();
        let manifest = job.index_manifest();
        let (forest, _) = ForestArtifact::build(
            &directory.0.join(format!("forest-{count}")),
            &input,
            *manifest.rotation_seed(),
            options(),
            1024 * 1024,
        )
        .unwrap();
        let path = directory.0.join(format!("serving-{count}"));
        let (artifact, report) = ServingArtifact::build(
            &path,
            &input,
            &forest,
            manifest,
            serving_options(&backend),
            1024 * 1024,
        )
        .unwrap();
        assert_eq!(report.records, count);
        artifact.verify().unwrap();
        if count == 0 {
            assert_eq!(report.keys, 0);
            assert_eq!(report.partitions, 0);
        } else {
            for (label, memory, scratch, value, output) in [
                ("memory", 1, 1 << 20, 1 << 20, 1 << 20),
                ("scratch", 8 << 20, 1, 1 << 20, 1 << 20),
                ("value", 8 << 20, 64 << 20, 8, 1 << 20),
                ("output", 8 << 20, 64 << 20, 1 << 20, 130),
            ] {
                let failed = directory.0.join(label);
                let mut limits = serving_options(&backend);
                limits.memory_bytes = memory;
                limits.scratch_bytes = scratch;
                limits.hard_limits.max_value_bytes = value;
                assert!(
                    ServingArtifact::build(&failed, &input, &forest, manifest, limits, output)
                        .is_err()
                );
                assert!(!failed.join("manifest.bin").exists());
            }
        }
        assert_eq!(
            runtime
                .open_index(&format!("bulk-{count}"))
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::IndexBuilding
        );
    }
    runtime.shutdown().await.unwrap();
}

fn rewrite_forest(
    path: &std::path::Path,
    artifact: &ktann::bulk::ArtifactManifest,
    edit: impl FnOnce(&mut Vec<Vec<u8>>),
) -> ktann::bulk::ArtifactManifest {
    use sha2::{Digest, Sha256};
    let bytes = fs::read(path.join("data.bin")).unwrap();
    let mut frames = Vec::new();
    let mut cursor = 41;
    while cursor < bytes.len() {
        let size = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        frames.push(bytes[cursor..cursor + size].to_vec());
        cursor += size + 32;
    }
    edit(&mut frames);
    let mut data = bytes[..41].to_vec();
    for frame in &frames {
        data.extend_from_slice(&(frame.len() as u32).to_be_bytes());
        data.extend_from_slice(frame);
        data.extend_from_slice(&Sha256::digest(frame));
    }
    let mut manifest = artifact.encode();
    manifest[41..49].copy_from_slice(&(frames.len() as u64).to_be_bytes());
    manifest[49..57].copy_from_slice(&(data.len() as u64).to_be_bytes());
    manifest[57..].copy_from_slice(&Sha256::digest(&data));
    fs::write(path.join("data.bin"), data).unwrap();
    fs::write(path.join("manifest.bin"), manifest).unwrap();
    ktann::bulk::ArtifactManifest::decode(&manifest).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serving_exact_joins_reject_validly_framed_but_wrong_membership_and_edges() {
    for defect in ["duplicate", "wrong_tree", "root_parent", "level"] {
        let directory = Directory::new();
        let backend = MemoryBackend::new();
        let runtime = runtime(backend.clone());
        let keyed = matches!(defect, "duplicate" | "wrong_tree");
        let count = if keyed { 2 } else { 30 };
        let input = InputSnapshot::create(
            &directory.0.join("input"),
            config(Metric::L2, keyed),
            1024 * 1024,
            (0..count).map(|id| Ok(record(id))),
        )
        .unwrap();
        let job = runtime
            .start_bulk_build("bulk", &input, options().tree)
            .await
            .unwrap();
        let index = job.index_manifest();
        let path = directory.0.join("forest");
        let (forest, _) = ForestArtifact::build(
            &path,
            &input,
            *index.rotation_seed(),
            options(),
            1024 * 1024,
        )
        .unwrap();
        let expected = rewrite_forest(&path, forest.manifest(), |frames| {
            if keyed {
                let offsets = frames
                    .iter()
                    .map(|frame| {
                        4 + u32::from_be_bytes(frame[..4].try_into().unwrap()) as usize
                            + 16
                            + 12
                            + 2
                    })
                    .collect::<Vec<_>>();
                let a = frames[0][offsets[0]..offsets[0] + 8].to_vec();
                let b = frames[1][offsets[1]..offsets[1] + 8].to_vec();
                frames[0][offsets[0]..offsets[0] + 8].copy_from_slice(&b);
                if defect == "wrong_tree" {
                    frames[1][offsets[1]..offsets[1] + 8].copy_from_slice(&a);
                }
            } else {
                let root = frames.last_mut().unwrap();
                if defect == "level" {
                    let level = u32::from_be_bytes(root[12..16].try_into().unwrap());
                    root[12..16].copy_from_slice(&(level + 1).to_be_bytes());
                } else {
                    root[34..42].copy_from_slice(&1_u64.to_be_bytes());
                }
            }
        });
        let forged =
            ForestArtifact::open(&path, expected, &input, *index.rotation_seed(), options())
                .unwrap();
        forged.verify().unwrap(); // counts/checksums alone do not prove a join
        let output = directory.0.join("serving");
        assert_eq!(
            ServingArtifact::build(
                &output,
                &input,
                &forged,
                index,
                serving_options(&backend),
                1024 * 1024
            )
            .err()
            .unwrap()
            .kind(),
            ErrorKind::Corruption,
            "{defect}"
        );
        assert!(!output.join("manifest.bin").exists());
        runtime.shutdown().await.unwrap();
    }
}

#[path = "bulk_serving/load.rs"]
mod load;
