//! Segmented input snapshot tests: rollover, restart, integrity, and quota bounds.

use std::fs;

use bytes::Bytes;
use ktann::api::{DataType, Error, ErrorKind, FieldSchema, IndexConfig, Metric, Record, Value};
use ktann::bulk::{INPUT_MANIFEST_BYTES, InputManifest, InputSnapshot, InputSnapshotWriter};

fn config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2)
        .unwrap()
        .with_leaf_partition_entries(2, 4)
        .and_then(|config| config.with_internal_partition_entries(2, 4))
        .unwrap()
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
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("input");
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
        InputManifest::decode(&descriptor).unwrap(),
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
fn multi_segment_rollover_preserves_exact_order_and_relocatability() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("segmented");
    let count = 20_000_u64;
    let writer = InputSnapshotWriter::new(&path, config(), 64 << 20).unwrap();
    let writer = writer.append((0..count).map(|id| Ok(record(id)))).unwrap();
    assert!(path.join("seg-00000000.bin").exists());
    assert!(path.join("data.partial").exists());
    assert!(!path.join("manifest.bin").exists());
    let snapshot = writer.seal().unwrap();
    assert!(!path.join("data.partial").exists());
    assert!(path.join("seg-00000001.bin").exists());
    assert!(path.join("segments.bin").exists());
    assert!(path.join("manifest.bin").exists());

    let relocated = directory.path().join("relocated");
    fs::rename(&path, &relocated).unwrap();
    let reopened = InputSnapshot::open(&relocated, config(), snapshot.manifest().clone()).unwrap();
    assert_eq!(reopened.manifest().items(), count);
    let mut seen = 0_u64;
    for item in reopened.reader().unwrap() {
        let rec = item.unwrap();
        assert_eq!(rec.id(), record(seen).id());
        seen += 1;
    }
    assert_eq!(seen, count);
}

#[test]
fn input_failures_and_quota_exhaustion_never_seal_or_overwrite() {
    let directory = tempfile::tempdir().unwrap();
    let failed = directory.path().join("failed");
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

    let quota = directory.path().join("quota");
    assert_eq!(
        InputSnapshot::create(&quota, config(), 150, [Ok(record(0))])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::LimitExceeded
    );
    assert!(!quota.join("manifest.bin").exists());

    let invalid = Record::new(Bytes::from_static(b"bad"), vec![1.0], vec![]).unwrap();
    assert_eq!(
        InputSnapshot::create(
            &directory.path().join("invalid"),
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
fn changed_truncated_extended_and_forged_length_segment_files_are_rejected() {
    for mutation in 0..5 {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input");
        let snapshot = InputSnapshot::create(&path, config(), 1_000_000, [Ok(record(0))]).unwrap();
        let seg_path = path.join("seg-00000000.bin");
        let mut bytes = fs::read(&seg_path).unwrap();
        match mutation {
            0 => bytes[50] ^= 1,
            1 => {
                bytes.pop();
            }
            2 => bytes.push(0),
            3 => bytes[42..46].copy_from_slice(&u32::MAX.to_be_bytes()),
            _ => bytes[10] ^= 1,
        }
        fs::write(&seg_path, bytes).unwrap();
        assert_eq!(snapshot.verify().unwrap_err().kind(), ErrorKind::Corruption);
    }
}

#[test]
fn swapping_valid_frames_fails_segment_digest_verification() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("input");
    let snapshot =
        InputSnapshot::create(&path, config(), 1_000_000, [Ok(record(0)), Ok(record(1))]).unwrap();
    let seg_path = path.join("seg-00000000.bin");
    let mut bytes = fs::read(&seg_path).unwrap();
    let frame_size = (bytes.len() - 42) / 2;
    bytes[42..].rotate_left(frame_size);
    fs::write(&seg_path, bytes).unwrap();
    let mut reader = snapshot.reader().unwrap();
    assert_eq!(reader.next().unwrap().unwrap().id(), record(1).id());
    assert!(reader.next().unwrap().is_ok());
    assert_eq!(
        reader.next().unwrap().unwrap_err().kind(),
        ErrorKind::Corruption
    );
    assert!(reader.next().is_none());
    assert_eq!(snapshot.verify().unwrap_err().kind(), ErrorKind::Corruption);
}

#[test]
fn manifest_identity_and_version_are_checked_independently_of_data() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    let a = InputSnapshot::create(&first, config(), 1_000_000, [Ok(record(0))]).unwrap();
    let b = InputSnapshot::create(&second, config(), 1_000_000, [Ok(record(1))]).unwrap();
    fs::copy(second.join("manifest.bin"), first.join("manifest.bin")).unwrap();
    assert_eq!(a.verify().unwrap_err().kind(), ErrorKind::Corruption);

    let mut encoded = b.manifest().encode();
    assert_eq!(encoded.len(), INPUT_MANIFEST_BYTES);
    for kind in [1, 4, u8::MAX] {
        encoded[9] = kind;
        assert_eq!(
            InputManifest::decode(&encoded).unwrap_err().kind(),
            ErrorKind::Corruption
        );
    }
    encoded = b.manifest().encode();
    let mut seg = fs::read(second.join("segments.bin")).unwrap();
    let version = ktann::storage::FORMAT_VERSION.to_be_bytes();
    assert_eq!(&encoded[7..9], &version);
    assert_eq!(&seg[7..9], &version);
    for unsupported in [0_u16, 2, 256, u16::MAX] {
        encoded[7..9].copy_from_slice(&unsupported.to_be_bytes());
        assert_eq!(
            InputManifest::decode(&encoded).unwrap_err().kind(),
            ErrorKind::UnsupportedFormat
        );
        seg[7..9].copy_from_slice(&unsupported.to_be_bytes());
        fs::write(second.join("segments.bin"), &seg).unwrap();
        assert_eq!(
            InputSnapshot::open(&second, config(), b.manifest().clone())
                .err()
                .unwrap()
                .kind(),
            ErrorKind::UnsupportedFormat
        );
    }
    assert_eq!(
        InputManifest::decode(&encoded[..encoded.len() - 1])
            .unwrap_err()
            .kind(),
        ErrorKind::Corruption
    );
}
