//! Durable source and topology artifacts: restart, integrity, and failure bounds.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use ktann::api::{DataType, Error, ErrorKind, FieldSchema, IndexConfig, Metric, Record, Value};
use ktann::bulk::ConstructionOptions;
use ktann::bulk::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest, InputSnapshot, InputSnapshotWriter};
use ktann::test_support::ForestArtifact;

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ktann-bulk-artifacts-{}-{}",
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

fn config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2)
        .unwrap()
        .with_partition_entries(2, 4)
        .unwrap()
}

fn options() -> ConstructionOptions {
    ConstructionOptions {
        min_partition_entries: 2,
        max_partition_entries: 4,
        sample_items: 8,
        memory_bytes: 2 * 1024 * 1024,
        scratch_bytes: 8 * 1024 * 1024,
    }
}

fn record(id: u64) -> Record {
    Record::new(
        Bytes::copy_from_slice(&id.to_be_bytes()),
        vec![id as f32 + 1.0, (id % 7) as f32],
        vec![],
    )
    .unwrap()
}

#[test]
fn full_records_survive_restart_with_payload_presence_and_canonical_fields() {
    let directory = Directory::new();
    let path = directory.0.join("input");
    let config = config()
        .with_fields(vec![
            FieldSchema::new("enabled", DataType::Bool).unwrap(),
            FieldSchema::new("count", DataType::I64).unwrap(),
            FieldSchema::new("weight", DataType::F64).unwrap(),
            FieldSchema::new("label", DataType::String)
                .unwrap()
                .nullable(),
        ])
        .unwrap();
    let records: Vec<_> = (0..3)
        .map(|id| {
            let record = Record::new(
                Bytes::copy_from_slice(&[id, 0, 255]),
                vec![1.0, -0.0],
                vec![
                    Value::Bool(true),
                    Value::I64(-9),
                    Value::F64(-0.0),
                    Value::Null,
                ],
            )
            .unwrap();
            match id {
                0 => record,
                1 => record.with_payload(Bytes::new()).unwrap(),
                _ => record
                    .with_payload(Bytes::from_static(b"opaque\0payload"))
                    .unwrap(),
            }
        })
        .collect();
    let snapshot = InputSnapshot::create(
        &path,
        config.clone(),
        1_000_000,
        records.iter().cloned().map(Ok),
    )
    .unwrap();
    let descriptor = snapshot.manifest().encode();
    drop(snapshot);
    let reopened = InputSnapshot::open(
        &path,
        config.clone(),
        ArtifactManifest::decode(&descriptor).unwrap(),
    )
    .unwrap();
    assert_eq!(reopened.manifest().items(), 3);
    let loaded = reopened
        .reader()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    for (original, loaded) in records.iter().zip(loaded) {
        assert_eq!(original.id(), loaded.id());
        assert_eq!(original.fields(), loaded.fields());
        assert_eq!(original.payload(), loaded.payload());
        assert_eq!(loaded.vector()[1].to_bits(), 0);
    }
    let incompatible = config.with_fields(vec![]).unwrap();
    assert_eq!(
        InputSnapshot::open(&path, incompatible, reopened.manifest().clone())
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidArgument
    );
}

#[test]
fn empty_snapshot_and_tree_are_complete_verifiable_artifacts() {
    let directory = Directory::new();
    let input = InputSnapshot::create(&directory.0.join("input"), config(), 130, []).unwrap();
    input.verify().unwrap();
    assert_eq!(input.manifest().items(), 0);
    let (artifact, report) = ForestArtifact::build(
        &directory.0.join("plan"),
        &input,
        [7; 32],
        forest_options(),
        130,
    )
    .unwrap();
    assert_eq!(report.records, 0);
    artifact
        .reader()
        .unwrap()
        .try_for_each(|row| row.map(|_| ()))
        .unwrap();
    assert_eq!(artifact.manifest().items(), 0);
}

#[test]
fn input_failures_and_quota_exhaustion_never_seal_or_overwrite() {
    let directory = Directory::new();
    let failed = directory.0.join("failed");
    let result = InputSnapshot::create(
        &failed,
        config(),
        1_000_000,
        [Ok(record(0)), Err(Error::new(ErrorKind::Cancelled))],
    );
    assert_eq!(result.err().unwrap().kind(), ErrorKind::Cancelled);
    assert!(!failed.join("manifest.bin").exists());
    let partial = fs::read(failed.join("data.partial")).unwrap();
    assert!(InputSnapshot::create(&failed, config(), 1_000_000, []).is_err());
    assert_eq!(fs::read(failed.join("data.partial")).unwrap(), partial);
    let quota = directory.0.join("quota");
    assert_eq!(
        InputSnapshot::create(&quota, config(), 130, [Ok(record(0))])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::LimitExceeded
    );
    assert!(!quota.join("manifest.bin").exists());
    let invalid = Record::new(Bytes::from_static(b"bad"), vec![1.0], vec![]).unwrap();
    assert_eq!(
        InputSnapshot::create(
            &directory.0.join("invalid"),
            config(),
            1_000_000,
            [Ok(invalid)]
        )
        .err()
        .unwrap()
        .kind(),
        ErrorKind::InvalidArgument
    );
}

#[test]
fn duplicate_ids_and_output_quota_fail_before_plan_sealing() {
    let directory = Directory::new();
    let input = InputSnapshot::create(
        &directory.0.join("input"),
        config(),
        1_000_000,
        [Ok(record(0)), Ok(record(0))],
    )
    .unwrap();
    let path = directory.0.join("duplicate");
    assert_eq!(
        ForestArtifact::build(&path, &input, [7; 32], forest_options(), 1_000_000)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::RecordAlreadyExists
    );
    assert!(!path.join("manifest.bin").exists());
    let unique = InputSnapshot::create(
        &directory.0.join("unique"),
        config(),
        1_000_000,
        [Ok(record(0))],
    )
    .unwrap();
    let path = directory.0.join("quota");
    assert_eq!(
        ForestArtifact::build(&path, &unique, [7; 32], forest_options(), 130)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::LimitExceeded
    );
    assert!(!path.join("manifest.bin").exists());
}

#[test]
fn changed_truncated_extended_and_forged_length_files_are_rejected() {
    for mutation in 0..5 {
        let directory = Directory::new();
        let path = directory.0.join("input");
        let snapshot = InputSnapshot::create(&path, config(), 1_000_000, [Ok(record(0))]).unwrap();
        let mut bytes = fs::read(path.join("data.bin")).unwrap();
        match mutation {
            0 => bytes[50] ^= 1,
            1 => {
                bytes.pop();
            }
            2 => bytes.push(0),
            3 => bytes[41..45].copy_from_slice(&u32::MAX.to_be_bytes()),
            _ => bytes[9] ^= 1,
        }
        fs::write(path.join("data.bin"), bytes).unwrap();
        assert_eq!(snapshot.verify().unwrap_err().kind(), ErrorKind::Corruption);
    }
}

#[test]
fn swapping_valid_frames_fails_whole_file_identity_before_plan_sealing() {
    let directory = Directory::new();
    let path = directory.0.join("input");
    let snapshot =
        InputSnapshot::create(&path, config(), 1_000_000, [Ok(record(0)), Ok(record(1))]).unwrap();
    let mut bytes = fs::read(path.join("data.bin")).unwrap();
    let frame_size = (bytes.len() - 41) / 2;
    bytes[41..].rotate_left(frame_size);
    fs::write(path.join("data.bin"), bytes).unwrap();
    let mut reader = snapshot.reader().unwrap();
    assert_eq!(reader.next().unwrap().unwrap().id(), record(1).id());
    assert!(reader.next().unwrap().is_ok());
    assert_eq!(
        reader.next().unwrap().unwrap_err().kind(),
        ErrorKind::Corruption
    );
    assert!(reader.next().is_none());
    let output = directory.0.join("plan");
    assert_eq!(
        ForestArtifact::build(&output, &snapshot, [7; 32], forest_options(), 1_000_000)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Corruption
    );
    assert!(!output.join("manifest.bin").exists());
}

#[test]
fn manifest_identity_and_version_are_checked_independently_of_data() {
    let directory = Directory::new();
    let first = directory.0.join("first");
    let second = directory.0.join("second");
    let a = InputSnapshot::create(&first, config(), 1_000_000, [Ok(record(0))]).unwrap();
    let b = InputSnapshot::create(&second, config(), 1_000_000, [Ok(record(1))]).unwrap();
    fs::copy(second.join("manifest.bin"), first.join("manifest.bin")).unwrap();
    assert_eq!(a.verify().unwrap_err().kind(), ErrorKind::Corruption);
    let mut encoded = b.manifest().encode();
    assert_eq!(encoded.len(), ARTIFACT_MANIFEST_BYTES);
    encoded[7] += 1;
    assert_eq!(
        ArtifactManifest::decode(&encoded).unwrap_err().kind(),
        ErrorKind::UnsupportedFormat
    );
    assert_eq!(
        ArtifactManifest::decode(&encoded[..encoded.len() - 1])
            .unwrap_err()
            .kind(),
        ErrorKind::Corruption
    );
}

fn forest_options() -> ktann::bulk::ForestOptions {
    ktann::bulk::ForestOptions {
        tree: options(),
        sort_memory_bytes: 556 * 1024,
        sort_scratch_bytes: 16 * 1024 * 1024,
    }
}

fn forest_config(metric: Metric) -> IndexConfig {
    use ktann::api::FieldId;
    IndexConfig::new(2, metric)
        .unwrap()
        .with_partition_entries(2, 4)
        .unwrap()
        .with_fields(vec![
            FieldSchema::new("bucket", DataType::I64).unwrap(),
            FieldSchema::new("name", DataType::String).unwrap(),
        ])
        .unwrap()
        .with_tree_key_fields(vec![FieldId(1), FieldId(0)])
        .unwrap()
}

fn forest_record(id: u64) -> Record {
    Record::new(
        Bytes::copy_from_slice(&id.to_be_bytes()),
        vec![id as f32 + 1.0, (id % 7) as f32],
        vec![
            Value::I64((id % 7) as i64 - 3),
            Value::string(if id.is_multiple_of(2) { "a\0b" } else { "a" }).unwrap(),
        ],
    )
    .unwrap()
    .with_payload(Bytes::from_static(b"original payload"))
    .unwrap()
}

type ForestShape = std::collections::BTreeMap<(Vec<u8>, u64), (u32, Vec<u32>, Vec<Bytes>)>;
fn forest_shape(artifact: &ktann::test_support::ForestArtifact) -> ForestShape {
    artifact
        .reader()
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            let plan = row.partition;
            (
                (row.tree_key.as_bytes().to_vec(), plan.key.get()),
                (
                    plan.level,
                    plan.centroid.iter().map(|x| x.to_bits()).collect(),
                    plan.entries,
                ),
            )
        })
        .collect()
}

#[test]
fn forest_globally_groups_exact_membership_and_matches_independent_tree_builds() {
    use ktann::storage::keys::TreeKey;
    use ktann::test_support::ForestArtifact;
    use ktann::test_support::construction::{ConstructionRecord, construct_tree};
    use std::collections::{BTreeMap, BTreeSet};
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        let directory = Directory::new();
        let source = InputSnapshot::create(
            &directory.0.join("input"),
            forest_config(metric),
            4 * 1024 * 1024,
            (0..317).rev().map(|id| Ok(forest_record(id))),
        )
        .unwrap();
        let (artifact, report) = ForestArtifact::build(
            &directory.0.join("forest"),
            &source,
            [7; 32],
            forest_options(),
            4 * 1024 * 1024,
        )
        .unwrap();
        artifact
            .reader()
            .unwrap()
            .try_for_each(|row| row.map(|_| ()))
            .unwrap();
        assert_eq!(report.records, 317);
        assert_eq!(report.trees, 14);
        assert!(report.peak_sort_scratch_bytes <= forest_options().sort_scratch_bytes);
        assert!(!directory.0.join("forest/sort").exists());
        assert!(!directory.0.join("forest/tree").exists());
        let expected_manifest = ArtifactManifest::decode(&artifact.manifest().encode()).unwrap();
        drop(artifact);
        let reopened = ForestArtifact::open(
            &directory.0.join("forest"),
            expected_manifest,
            &source,
            [7; 32],
            forest_options(),
        )
        .unwrap();
        let shape = forest_shape(&reopened);
        assert_eq!(shape.len() as u64, report.partitions);
        let mut expected: BTreeMap<Vec<u8>, Vec<ConstructionRecord>> = BTreeMap::new();
        for id in 0..317 {
            let record = forest_record(id);
            let tree = TreeKey::encode(
                &[DataType::String, DataType::I64],
                &[record.fields()[1].clone(), record.fields()[0].clone()],
            )
            .unwrap();
            expected
                .entry(tree.as_bytes().to_vec())
                .or_default()
                .push(ConstructionRecord {
                    id: record.id().clone(),
                    vector: record.vector().into(),
                });
        }
        let mut exact = ForestShape::new();
        for (ordinal, (tree, records)) in expected.into_iter().enumerate() {
            construct_tree(
                &directory.0.join(format!("independent-{ordinal}")),
                2,
                metric,
                [7; 32],
                options(),
                records.into_iter().map(Ok),
                |plan| {
                    exact.insert(
                        (tree.clone(), plan.key.get()),
                        (
                            plan.level,
                            plan.centroid.iter().map(|x| x.to_bits()).collect(),
                            plan.entries,
                        ),
                    );
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(shape, exact);
        let mut all_ids = BTreeSet::new();
        for (_, (level, _, entries)) in shape {
            if level == 1 {
                for id in entries {
                    assert!(all_ids.insert(id));
                }
            }
        }
        assert_eq!(all_ids.len(), 317);
    }
}

#[test]
fn forest_sort_spills_and_source_order_do_not_change_topology() {
    use ktann::test_support::ForestArtifact;
    let directory = Directory::new();
    let a = InputSnapshot::create(
        &directory.0.join("a"),
        forest_config(Metric::L2),
        4 * 1024 * 1024,
        (0..173).map(|id| Ok(forest_record(id))),
    )
    .unwrap();
    let b = InputSnapshot::create(
        &directory.0.join("b"),
        forest_config(Metric::L2),
        4 * 1024 * 1024,
        (0..173).rev().map(|id| Ok(forest_record(id))),
    )
    .unwrap();
    let (small, _) = ForestArtifact::build(
        &directory.0.join("small"),
        &a,
        [9; 32],
        forest_options(),
        4 * 1024 * 1024,
    )
    .unwrap();
    let mut roomy = forest_options();
    roomy.sort_memory_bytes = 4 * 1024 * 1024;
    let (large, _) = ForestArtifact::build(
        &directory.0.join("large"),
        &b,
        [9; 32],
        roomy,
        4 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(forest_shape(&small), forest_shape(&large));
    let (repeat, _) = ForestArtifact::build(
        &directory.0.join("repeat"),
        &a,
        [9; 32],
        forest_options(),
        4 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(small.manifest(), repeat.manifest());
    assert_eq!(
        ForestArtifact::open(
            &directory.0.join("small"),
            small.manifest().clone(),
            &a,
            [8; 32],
            forest_options()
        )
        .err()
        .unwrap()
        .kind(),
        ErrorKind::InvalidArgument
    );
    let regrouped = InputSnapshot::open(
        &directory.0.join("a"),
        forest_config(Metric::L2)
            .with_tree_key_fields(vec![ktann::api::FieldId(0)])
            .unwrap(),
        a.manifest().clone(),
    )
    .unwrap();
    assert!(
        ForestArtifact::open(
            &directory.0.join("small"),
            small.manifest().clone(),
            &regrouped,
            [9; 32],
            forest_options()
        )
        .is_err()
    );
}

#[test]
fn forest_rejects_cross_tree_duplicates_before_output_and_enforces_quotas() {
    use ktann::test_support::ForestArtifact;
    let directory = Directory::new();
    let duplicate = Record::new(
        forest_record(0).id().clone(),
        vec![4.0, 5.0],
        forest_record(1).fields().to_vec(),
    )
    .unwrap();
    let input = InputSnapshot::create(
        &directory.0.join("input"),
        forest_config(Metric::L2),
        4 * 1024 * 1024,
        (0..100)
            .map(|id| Ok(forest_record(id)))
            .chain([Ok(duplicate)]),
    )
    .unwrap();
    let path = directory.0.join("duplicate");
    assert_eq!(
        ForestArtifact::build(&path, &input, [0; 32], forest_options(), 4 * 1024 * 1024)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::RecordAlreadyExists
    );
    assert!(!path.join("manifest.bin").exists());
    assert!(!path.join("tree").exists());
    let unique = InputSnapshot::create(
        &directory.0.join("unique"),
        forest_config(Metric::L2),
        4 * 1024 * 1024,
        (0..100).map(|id| Ok(forest_record(id))),
    )
    .unwrap();
    let mut tiny = forest_options();
    tiny.sort_scratch_bytes = 100;
    assert_eq!(
        ForestArtifact::build(
            &directory.0.join("quota"),
            &unique,
            [0; 32],
            tiny,
            4 * 1024 * 1024
        )
        .err()
        .unwrap()
        .kind(),
        ErrorKind::LimitExceeded
    );
    tiny = forest_options();
    tiny.sort_memory_bytes = 1;
    let invalid = directory.0.join("invalid");
    assert_eq!(
        ForestArtifact::build(&invalid, &unique, [0; 32], tiny, 4 * 1024 * 1024)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert!(!invalid.exists());
    let full = directory.0.join("full");
    assert_eq!(
        ForestArtifact::build(&full, &unique, [0; 32], forest_options(), 130)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::LimitExceeded
    );
    assert!(!full.join("manifest.bin").exists());
}

#[test]
fn forest_empty_input_has_no_tree_and_changed_source_cannot_seal() {
    use ktann::test_support::ForestArtifact;
    let directory = Directory::new();
    for (index, config) in [config(), forest_config(Metric::L2)]
        .into_iter()
        .enumerate()
    {
        let input = InputSnapshot::create(
            &directory.0.join(format!("input-{index}")),
            config,
            1024,
            std::iter::empty(),
        )
        .unwrap();
        let (artifact, report) = ForestArtifact::build(
            &directory.0.join(format!("plan-{index}")),
            &input,
            [0; 32],
            forest_options(),
            130,
        )
        .unwrap();
        artifact
            .reader()
            .unwrap()
            .try_for_each(|row| row.map(|_| ()))
            .unwrap();
        assert_eq!(report.records, 0);
        assert_eq!(report.trees, 0);
        assert_eq!(report.partitions, 0);
        assert!(artifact.reader().unwrap().next().is_none());
    }
    let path = directory.0.join("damaged-input");
    let input = InputSnapshot::create(
        &path,
        forest_config(Metric::L2),
        1024 * 1024,
        (0..4).map(|id| Ok(forest_record(id))),
    )
    .unwrap();
    let file = path.join("data.bin");
    let mut bytes = fs::read(&file).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(file, bytes).unwrap();
    let output = directory.0.join("damaged-output");
    assert_eq!(
        ForestArtifact::build(&output, &input, [0; 32], forest_options(), 1024 * 1024)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Corruption
    );
    assert!(!output.join("manifest.bin").exists());
    assert!(!output.join("tree").exists());
}

#[test]
fn forest_reader_rejects_invalid_tree_envelopes_and_premature_roots() {
    use ktann::test_support::ForestArtifact;
    use sha2::{Digest, Sha256};
    for root_early in [false, true] {
        let directory = Directory::new();
        let input = InputSnapshot::create(
            &directory.0.join("input"),
            forest_config(Metric::L2),
            1024 * 1024,
            (0..100).map(|id| Ok(forest_record(id))),
        )
        .unwrap();
        let path = directory.0.join("forest");
        let (artifact, _) =
            ForestArtifact::build(&path, &input, [0; 32], forest_options(), 1024 * 1024).unwrap();
        let mut data = fs::read(path.join("data.bin")).unwrap();
        let length = u32::from_be_bytes(data[41..45].try_into().unwrap()) as usize;
        let body = &mut data[45..45 + length];
        if root_early {
            let tree_len = u32::from_be_bytes(body[..4].try_into().unwrap()) as usize;
            body[4 + tree_len..12 + tree_len].copy_from_slice(&1_u64.to_be_bytes());
        } else {
            body[..4].copy_from_slice(&u32::MAX.to_be_bytes());
        }
        let frame_hash = Sha256::digest(body);
        data[45 + length..77 + length].copy_from_slice(&frame_hash);
        let mut manifest = artifact.manifest().encode();
        manifest[57..].copy_from_slice(&Sha256::digest(&data));
        fs::write(path.join("data.bin"), &data).unwrap();
        fs::write(path.join("manifest.bin"), manifest).unwrap();
        let reopened = ForestArtifact::open(
            &path,
            ArtifactManifest::decode(&manifest).unwrap(),
            &input,
            [0; 32],
            forest_options(),
        )
        .unwrap();
        assert_eq!(
            reopened
                .reader()
                .unwrap()
                .try_for_each(|row| row.map(|_| ()))
                .unwrap_err()
                .kind(),
            ErrorKind::Corruption
        );
        let mut reader = reopened.reader().unwrap();
        while let Some(row) = reader.next() {
            if row.is_err() {
                assert!(reader.next().is_none());
                break;
            }
        }
    }
}

#[test]
fn streaming_snapshot_matches_one_pass_and_never_seals_failed_input() {
    let directory = Directory::new();
    let rows: Vec<_> = (0..31).map(record).collect();
    let one = InputSnapshot::create(
        &directory.0.join("one"),
        config(),
        1_000_000,
        rows.iter().cloned().map(Ok),
    )
    .unwrap();
    let path = directory.0.join("stream");
    let mut writer = InputSnapshotWriter::new(&path, config(), 1_000_000).unwrap();
    for batch in rows.chunks(7) {
        writer = writer.append(batch.iter().cloned().map(Ok)).unwrap();
        assert!(!path.join("manifest.bin").exists());
    }
    let streamed = writer.seal().unwrap();
    streamed.verify().unwrap();
    assert_eq!(one.manifest(), streamed.manifest());
    assert_eq!(
        fs::read(directory.0.join("one/data.bin")).unwrap(),
        fs::read(path.join("data.bin")).unwrap()
    );

    for (name, quota, rows) in [
        (
            "invalid",
            1_000_000,
            vec![Ok(record(0)), Err(Error::new(ErrorKind::Other))],
        ),
        (
            "quota",
            ARTIFACT_MANIFEST_BYTES as u64 + 41,
            vec![Ok(record(0))],
        ),
    ] {
        let path = directory.0.join(name);
        assert!(
            InputSnapshotWriter::new(&path, config(), quota)
                .unwrap()
                .append(rows)
                .is_err()
        );
        assert!(!path.join("manifest.bin").exists());
    }
    let path = directory.0.join("cancelled");
    drop(
        InputSnapshotWriter::new(&path, config(), 1_000_000)
            .unwrap()
            .append([Ok(record(0))])
            .unwrap(),
    );
    assert!(!path.join("manifest.bin").exists());
}

#[test]
fn receipt_sorting_preserves_source_and_forest_bytes_with_spills() {
    use ktann::bulk::PreparedInputWriter;
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        for count in [0, 73, 3000] {
            let dir = Directory::new();
            let config = forest_config(metric);
            let mut options = forest_options();
            options.sort_memory_bytes *= 2;
            let records = || (0..count).rev().map(|id| Ok(forest_record(id)));
            let original = InputSnapshot::create(
                &dir.0.join("original"),
                config.clone(),
                8_000_000,
                records(),
            )
            .unwrap();
            let mut writer = PreparedInputWriter::new(
                &dir.0.join("source"),
                &dir.0.join("sorting"),
                config,
                8_000_000,
                options,
            )
            .unwrap();
            let mut input = records();
            loop {
                let batch: Vec<_> = input.by_ref().take(17).collect();
                if batch.is_empty() {
                    break;
                }
                writer = writer.append(batch).unwrap();
            }
            let prepared = writer.seal().unwrap();
            assert_eq!(original.manifest(), prepared.source().manifest());
            let (before, _) = ForestArtifact::build(
                &dir.0.join("before"),
                &original,
                [7; 32],
                options,
                8_000_000,
            )
            .unwrap();
            let (after, _) =
                ForestArtifact::build_prepared(&dir.0.join("after"), prepared, [7; 32], 8_000_000)
                    .unwrap();
            assert_eq!(before.manifest(), after.manifest());
            assert_eq!(forest_shape(&before), forest_shape(&after));
            assert!(!dir.0.join("sorting").exists());
        }
    }
}

#[test]
fn receipt_sorting_rejects_cross_batch_duplicates_and_invalid_records() {
    use ktann::bulk::PreparedInputWriter;
    let dir = Directory::new();
    let mut options = forest_options();
    options.sort_memory_bytes *= 2;
    let writer = PreparedInputWriter::new(
        &dir.0.join("source"),
        &dir.0.join("sorting"),
        forest_config(Metric::L2),
        8_000_000,
        options,
    )
    .unwrap();
    let writer = writer
        .append((0..3000).map(|id| Ok(forest_record(id))))
        .unwrap();
    let prepared = writer
        .append([Ok(forest_record(0))])
        .unwrap()
        .seal()
        .unwrap();
    assert_eq!(
        ForestArtifact::build_prepared(&dir.0.join("forest"), prepared, [7; 32], 8_000_000)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::RecordAlreadyExists
    );
    assert!(!dir.0.join("forest/manifest.bin").exists());
    let writer = PreparedInputWriter::new(
        &dir.0.join("invalid"),
        &dir.0.join("invalid-sort"),
        forest_config(Metric::L2),
        8_000_000,
        options,
    )
    .unwrap();
    assert_eq!(
        writer.append([Ok(record(1))]).err().unwrap().kind(),
        ErrorKind::InvalidArgument
    );
    assert!(!dir.0.join("invalid/manifest.bin").exists());
}

#[test]
fn receipt_sorting_rejects_local_duplicates_before_sealing() {
    let dir = Directory::new();
    let mut options = forest_options();
    options.sort_memory_bytes *= 2;
    let writer = ktann::bulk::PreparedInputWriter::new(
        &dir.0.join("source"),
        &dir.0.join("sort"),
        forest_config(Metric::L2),
        8_000_000,
        options,
    )
    .unwrap();
    let error = writer
        .append((0..10000).map(|_| Ok(forest_record(1))))
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::RecordAlreadyExists);
    assert!(!dir.0.join("source/manifest.bin").exists());
}

#[test]
fn receipt_sorting_validates_configuration_before_creating_files() {
    let dir = Directory::new();
    let invalid = config()
        .with_tree_key_fields(vec![ktann::api::FieldId(99)])
        .unwrap();
    let mut options = forest_options();
    options.sort_memory_bytes *= 2;
    let error = ktann::bulk::PreparedInputWriter::new(
        &dir.0.join("source"),
        &dir.0.join("sort"),
        invalid,
        8_000_000,
        options,
    )
    .err()
    .unwrap();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 0);
}

#[test]
fn variable_id_forest_is_identical_across_construction_budgets_and_preparation() {
    use ktann::bulk::PreparedInputWriter;
    for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
        for tied in [true, false] {
            let dir = Directory::new();
            let config = IndexConfig::new(7, metric)
                .unwrap()
                .with_partition_entries(2, 4)
                .unwrap();
            let records: Vec<_> = (0_u64..257)
                .rev()
                .map(|id| {
                    let encoded = id.to_be_bytes();
                    let first = encoded.iter().position(|byte| *byte != 0).unwrap_or(7);
                    let mut key = encoded[first..].to_vec();
                    if id % 2 == 0 {
                        key.resize(255, 0);
                    }
                    Record::new(
                        Bytes::from(key),
                        (0..7)
                            .map(|axis| {
                                if axis == 0 {
                                    1.0
                                } else if tied {
                                    0.0
                                } else {
                                    (id % 3) as f32
                                }
                            })
                            .collect::<Vec<_>>(),
                        vec![],
                    )
                    .unwrap()
                })
                .collect();
            let input = InputSnapshot::create(
                &dir.0.join("input"),
                config.clone(),
                8_000_000,
                records.iter().cloned().map(Ok),
            )
            .unwrap();
            let mut small = forest_options();
            small.tree.memory_bytes = 160 * 1024;
            small.sort_memory_bytes *= 2;
            let mut large = small;
            large.tree.memory_bytes = 4 * 1024 * 1024;
            let (resident, _) =
                ForestArtifact::build(&dir.0.join("resident"), &input, [7; 32], large, 8_000_000)
                    .unwrap();
            let expected = forest_shape(&resident);
            let (streamed, _) =
                ForestArtifact::build(&dir.0.join("streamed"), &input, [7; 32], small, 8_000_000)
                    .unwrap();
            assert_eq!(forest_shape(&streamed), expected);
            let prepared = PreparedInputWriter::new(
                &dir.0.join("prepared-input"),
                &dir.0.join("sorting"),
                config,
                8_000_000,
                small,
            )
            .unwrap()
            .append(records.into_iter().map(Ok))
            .unwrap()
            .seal()
            .unwrap();
            let (prepared, _) = ForestArtifact::build_prepared(
                &dir.0.join("prepared-forest"),
                prepared,
                [7; 32],
                8_000_000,
            )
            .unwrap();
            assert_eq!(forest_shape(&prepared), expected);
        }
    }
}
