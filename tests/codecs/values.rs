//! Canonical persistent-value codec contract tests.

use std::num::NonZeroU32;

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, IndexName, LogicalIndexId, Metric,
    PartitionKey, SynopsisConfig, Value,
};
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    BloomParameters, ChildEntry, IndexIdAllocator, IndexLifecycle, IndexManifest, IndexNameEntry,
    LeafEntry, OpaquePayload, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionSynopsis, PartitionTransition, PersistentValue, RecordLocation, TreeManifest,
    ValueCodec, VectorRecord,
};

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
}

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

fn minimal_manifest() -> IndexManifest {
    IndexManifest::new(
        IndexLifecycle::Active,
        id(1),
        IndexConfig::new(1, Metric::L2).expect("valid config"),
        [0; 32],
        vec![],
    )
    .expect("valid manifest")
}

fn rich_manifest() -> IndexManifest {
    let tenant = FieldSchema::new("tenant", DataType::String).expect("valid field");
    let score = FieldSchema::new("score", DataType::F64)
        .expect("valid field")
        .nullable()
        .with_synopsis(SynopsisConfig::MinMaxBloom {
            expected_distinct: NonZeroU32::new(10).expect("nonzero"),
            false_positive_rate: 0.5,
        })
        .expect("valid synopsis");
    let config = IndexConfig::new(2, Metric::Cosine)
        .expect("valid config")
        .with_fields(vec![tenant, score])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid Tree Key fields")
        .with_partition_entries(2, 4)
        .expect("valid partition limits");
    let mut seed = [0_u8; 32];
    for (value, byte) in seed.iter_mut().zip(0_u8..) {
        *value = byte;
    }
    let score_bloom =
        BloomParameters::derive(config.fields()[1].synopsis()).expect("derive Bloom parameters");
    IndexManifest::new(
        IndexLifecycle::Dropping,
        id(7),
        config,
        seed,
        vec![None, score_bloom],
    )
    .expect("valid manifest")
}

fn index_codec(manifest: &IndexManifest) -> ValueCodec<'_> {
    ValueCodec::for_index(manifest)
}

fn rich_synopsis(manifest: &IndexManifest) -> PartitionSynopsis {
    let mut synopsis = PartitionSynopsis::empty(manifest);
    for fields in [
        vec![Value::string("a").expect("valid String"), Value::Null],
        vec![
            Value::string("z").expect("valid String"),
            Value::f64(1.0).expect("finite"),
        ],
        vec![
            Value::string("m").expect("valid String"),
            Value::f64(2.0).expect("finite"),
        ],
    ] {
        synopsis.expand(manifest, &fields).expect("expand synopsis");
    }
    synopsis
}

fn decode(
    codec: ValueCodec<'_>,
    key: &LogicalKey,
    bytes: &[u8],
) -> ktann::api::Result<PersistentValue> {
    codec.decode(key, Bytes::copy_from_slice(bytes))
}

fn key_for_value(index: LogicalIndexId, value: &PersistentValue) -> LogicalKey {
    let tree_key = TreeKey::encode(&[], &[]).expect("empty Tree Key");
    match value {
        PersistentValue::IndexIdAllocator(_) => LogicalKey::IndexIdAllocator,
        PersistentValue::IndexNameEntry(_) => {
            LogicalKey::IndexNameDirectory(IndexName::new("index").expect("valid Index Name"))
        }
        PersistentValue::IndexManifest(value) => LogicalKey::Manifest(value.logical_index_id()),
        PersistentValue::TreeManifest(_) => LogicalKey::TreeManifest { index, tree_key },
        PersistentValue::VectorRecord(value) => LogicalKey::Record {
            index,
            id: value.record_id().clone(),
        },
        PersistentValue::OpaquePayload(_) => LogicalKey::Payload {
            index,
            id: Bytes::from_static(b"r"),
        },
        PersistentValue::RecordLocation(_) => LogicalKey::Location {
            index,
            id: Bytes::from_static(b"r"),
        },
        PersistentValue::PartitionHeader(_) => LogicalKey::Header {
            index,
            tree_key,
            partition: pk(1),
        },
        PersistentValue::PartitionCentroid(_) => LogicalKey::Centroid {
            index,
            tree_key,
            partition: pk(1),
        },
        PersistentValue::ChildEntry(value) => LogicalKey::ChildEntry {
            index,
            tree_key,
            partition: pk(1),
            child: value.child(),
        },
        PersistentValue::LeafEntry(value) => LogicalKey::LeafEntry {
            index,
            tree_key,
            partition: pk(1),
            id: value.record_id().clone(),
        },
        PersistentValue::PartitionSynopsis(_) => LogicalKey::Synopsis {
            index,
            tree_key,
            partition: pk(1),
        },
        PersistentValue::PartitionState(_) => LogicalKey::State {
            index,
            tree_key,
            partition: pk(1),
        },
        _ => panic!("test helper does not support this persistent value"),
    }
}

fn decode_value(
    codec: ValueCodec<'_>,
    index: LogicalIndexId,
    value: &PersistentValue,
    bytes: &[u8],
) -> ktann::api::Result<PersistentValue> {
    decode(codec, &key_for_value(index, value), bytes)
}

fn round_trip(codec: ValueCodec<'_>, index: LogicalIndexId, value: &PersistentValue) {
    let bytes = codec.encode(value).expect("encode");
    let key = key_for_value(index, value);
    assert_eq!(decode(codec, &key, &bytes).expect("decode"), *value);
}

fn assert_corrupt(result: ktann::api::Result<PersistentValue>) {
    assert_eq!(
        result.expect_err("must fail closed").kind(),
        ErrorKind::Corruption
    );
}

#[test]
fn leaf_entry_codec_rejects_noncanonical_nested_rabitq7() {
    let manifest = minimal_manifest();
    let codec = index_codec(&manifest);
    let mut negative_zero = [0_u8; 14];
    negative_zero[12] = 1;
    let malformed = PersistentValue::LeafEntry(LeafEntry::new(
        Bytes::from_static(b"r"),
        Vec::<Value>::new(),
        Bytes::copy_from_slice(&negative_zero),
    ));
    assert_eq!(
        codec
            .encode(&malformed)
            .expect_err("encode must reject")
            .kind(),
        ErrorKind::InvalidArgument
    );

    let valid = PersistentValue::LeafEntry(LeafEntry::new(
        Bytes::from_static(b"r"),
        Vec::<Value>::new(),
        Bytes::from_static(&[0; 14]),
    ));
    let mut bytes = codec.encode(&valid).expect("encode valid Leaf Entry");
    let payload_start = bytes.len() - 14;
    bytes[payload_start + 12] = 1;
    assert_corrupt(decode_value(codec, id(1), &valid, &bytes));
}

#[test]
fn namespace_and_manifest_golden_bytes() {
    let codec = ValueCodec::bootstrap();
    assert_eq!(
        codec
            .encode(&PersistentValue::IndexIdAllocator(IndexIdAllocator::new(0)))
            .expect("encode"),
        b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
    );
    assert_eq!(
        codec
            .encode(&PersistentValue::IndexNameEntry(IndexNameEntry::new(id(1))))
            .expect("encode"),
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01"
    );

    let mut expected = vec![0x02, 0x01, 0x00, 0x01, 0x01, 0x00];
    expected.extend_from_slice(&1_u64.to_be_bytes());
    expected.extend_from_slice(&1_u32.to_be_bytes());
    expected.push(0x00);
    expected.extend_from_slice(&0_u16.to_be_bytes());
    expected.extend_from_slice(&0_u16.to_be_bytes());
    expected.extend_from_slice(&16_u32.to_be_bytes());
    expected.extend_from_slice(&128_u32.to_be_bytes());
    expected.extend_from_slice(&[0; 32]);
    assert_eq!(
        codec
            .encode(&PersistentValue::IndexManifest(minimal_manifest()))
            .expect("encode"),
        expected
    );
}

#[test]
fn index_value_family_golden_bytes() {
    let manifest = minimal_manifest();
    let codec = index_codec(&manifest);
    let empty_tree_key = TreeKey::encode(&[], &[]).expect("empty Tree Key");

    assert_eq!(
        codec
            .encode(&PersistentValue::TreeManifest(
                TreeManifest::new(pk(1), pk(1_024)).expect("valid Tree Manifest")
            ))
            .expect("encode"),
        b"\x03\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x04\x00"
    );
    assert_eq!(
        codec
            .encode(&PersistentValue::VectorRecord(VectorRecord::new(
                Bytes::from_static(b"r"),
                vec![1.5_f32],
                Vec::<Value>::new()
            )))
            .expect("encode"),
        b"\x04\x01\x00\x01r\x00\x00\x00\x01\x3f\xc0\x00\x00\x00\x00"
    );
    assert_eq!(
        codec
            .encode(&PersistentValue::OpaquePayload(
                OpaquePayload::new(Bytes::from_static(b"abc")).expect("valid payload")
            ))
            .expect("encode"),
        b"\x05\x01\x00\x00\x00\x03abc"
    );
    assert_eq!(
        codec
            .encode(&PersistentValue::RecordLocation(RecordLocation::new(
                empty_tree_key,
                pk(2)
            )))
            .expect("encode"),
        b"\x06\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02"
    );
    assert_eq!(
        codec
            .encode(&PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 3, 4, PartitionState::Ready,).expect("valid Header")
            ))
            .expect("encode"),
        b"\x07\x01\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00"
    );
    assert_eq!(
        codec
            .encode(&PersistentValue::PartitionCentroid(PartitionCentroid::new(
                vec![2.0_f32]
            )))
            .expect("encode"),
        b"\x08\x01\x00\x00\x00\x01\x40\x00\x00\x00"
    );
    assert_eq!(
        codec
            .encode(&PersistentValue::ChildEntry(ChildEntry::new(
                pk(2),
                vec![2.0_f32]
            )))
            .expect("encode"),
        b"\x09\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x40\x00\x00\x00"
    );

    let mut leaf = vec![0x0a, 0x01, 0x00, 0x01, b'r', 0x00, 0x00];
    leaf.extend_from_slice(&14_u32.to_be_bytes());
    leaf.extend_from_slice(&[0; 14]);
    assert_eq!(
        codec
            .encode(&PersistentValue::LeafEntry(LeafEntry::new(
                Bytes::from_static(b"r"),
                Vec::<Value>::new(),
                Bytes::from_static(&[0; 14]),
            )))
            .expect("encode"),
        leaf
    );
    assert_eq!(
        codec
            .encode(&PersistentValue::PartitionSynopsis(
                PartitionSynopsis::empty(&manifest,)
            ))
            .expect("encode"),
        b"\x0b\x01\x00\x00"
    );

    let mut state = vec![0x0c, 0x01, 0x01];
    state.extend_from_slice(&5_u64.to_be_bytes());
    state.extend_from_slice(&2_u64.to_be_bytes());
    state.extend_from_slice(&3_u64.to_be_bytes());
    assert_eq!(
        codec
            .encode(&PersistentValue::PartitionState(
                PartitionTransition::Splitting {
                    left: pk(2),
                    right: pk(3),
                    started_at_unix_millis: 5,
                }
            ))
            .expect("encode"),
        state
    );
}

#[test]
fn incrementally_constructed_synopsis_has_golden_bytes_and_round_trips() {
    let manifest = rich_manifest();
    let value = PersistentValue::PartitionSynopsis(rich_synopsis(&manifest));
    let bytes = index_codec(&manifest).encode(&value).expect("encode");
    assert_eq!(
        bytes,
        b"\x0b\x01\x00\x02\x02\x04\x00\x00\x00\x01a\x04\x00\x00\x00\x01z\x03\x03\x3f\xf0\x00\x00\x00\x00\x00\x00\x03\x40\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x80\x01\x00"
    );
    assert_eq!(
        decode_value(index_codec(&manifest), id(7), &value, &bytes).expect("decode"),
        value
    );
}

#[test]
fn derived_bloom_shape_meets_the_configured_false_positive_bound() {
    let manifest = rich_manifest();
    let parameters = manifest.bloom_parameters()[1].expect("score Bloom");
    assert_eq!((parameters.bit_count(), parameters.hash_count()), (20, 1));

    let occupied_bit_bound = 10.0 / f64::from(parameters.bit_count());
    assert!(occupied_bit_bound <= 0.5);

    let strict = SynopsisConfig::MinMaxBloom {
        expected_distinct: NonZeroU32::new(1).expect("nonzero"),
        false_positive_rate: 0.01,
    };
    let strict = BloomParameters::derive(&strict)
        .expect("derive strict Bloom")
        .expect("Bloom parameters");
    assert_eq!((strict.bit_count(), strict.hash_count()), (100, 1));
}

#[test]
fn every_value_family_round_trips() {
    let bootstrap = ValueCodec::bootstrap();
    let manifest = rich_manifest();
    round_trip(
        bootstrap,
        id(1),
        &PersistentValue::IndexIdAllocator(IndexIdAllocator::new(u64::MAX)),
    );
    round_trip(
        bootstrap,
        id(1),
        &PersistentValue::IndexNameEntry(IndexNameEntry::new(id(u64::MAX))),
    );
    round_trip(
        bootstrap,
        id(1),
        &PersistentValue::IndexManifest(manifest.clone()),
    );

    let codec = index_codec(&manifest);
    let tree_key = TreeKey::encode(
        &[DataType::String],
        &[Value::string("tenant-a").expect("valid String")],
    )
    .expect("valid Tree Key");
    let synopsis = rich_synopsis(&manifest);
    let values = [
        PersistentValue::TreeManifest(
            TreeManifest::new(pk(1), pk(1_024)).expect("valid Tree Manifest"),
        ),
        PersistentValue::VectorRecord(VectorRecord::new(
            Bytes::from_static(b"r"),
            vec![1.0_f32, -2.0],
            vec![
                Value::string("tenant-a").expect("valid String"),
                Value::f64(1.5).expect("finite"),
            ],
        )),
        PersistentValue::OpaquePayload(
            OpaquePayload::new(Bytes::new()).expect("valid empty payload"),
        ),
        PersistentValue::RecordLocation(RecordLocation::new(tree_key, pk(9))),
        PersistentValue::PartitionHeader(
            PartitionHeader::new(2, u32::MAX, u64::MAX, PartitionState::DrainingSplit)
                .expect("valid Header"),
        ),
        PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![0.5_f32, -0.5])),
        PersistentValue::ChildEntry(ChildEntry::new(pk(10), vec![0.25_f32, -0.25])),
        PersistentValue::LeafEntry(LeafEntry::new(
            Bytes::from_static(b"record"),
            vec![
                Value::string("tenant-a").expect("valid String"),
                Value::Null,
            ],
            Bytes::from_static(&[0; 15]),
        )),
        PersistentValue::PartitionSynopsis(synopsis),
        PersistentValue::PartitionState(PartitionTransition::ReceivingSplit {
            source: pk(2),
            started_at_unix_millis: 123,
        }),
    ];
    for value in &values {
        round_trip(codec, id(7), value);
    }
}

fn assert_truncated_and_trailing(
    codec: ValueCodec<'_>,
    index: LogicalIndexId,
    value: &PersistentValue,
) {
    let bytes = codec.encode(value).expect("encode corpus value");
    for cut in 0..bytes.len() {
        assert_corrupt(decode_value(codec, index, value, &bytes[..cut]));
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert_corrupt(decode_value(codec, index, value, &trailing));
}

#[test]
fn every_value_family_rejects_truncation_and_trailing_bytes() {
    let bootstrap = ValueCodec::bootstrap();
    let manifest = minimal_manifest();
    for value in [
        PersistentValue::IndexIdAllocator(IndexIdAllocator::new(1)),
        PersistentValue::IndexNameEntry(IndexNameEntry::new(id(1))),
        PersistentValue::IndexManifest(manifest.clone()),
    ] {
        assert_truncated_and_trailing(bootstrap, id(1), &value);
    }

    let codec = index_codec(&manifest);
    let values = [
        PersistentValue::TreeManifest(
            TreeManifest::new(pk(1), pk(2)).expect("valid Tree Manifest"),
        ),
        PersistentValue::VectorRecord(VectorRecord::new(
            Bytes::from_static(b"r"),
            vec![1.0_f32],
            Vec::<Value>::new(),
        )),
        PersistentValue::OpaquePayload(
            OpaquePayload::new(Bytes::from_static(b"x")).expect("valid payload"),
        ),
        PersistentValue::RecordLocation(RecordLocation::new(
            TreeKey::encode(&[], &[]).expect("empty Tree Key"),
            pk(1),
        )),
        PersistentValue::PartitionHeader(
            PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("valid Header"),
        ),
        PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![0.0_f32])),
        PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.0_f32])),
        PersistentValue::LeafEntry(LeafEntry::new(
            Bytes::from_static(b"r"),
            Vec::<Value>::new(),
            Bytes::from_static(&[0; 14]),
        )),
        PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(&manifest)),
        PersistentValue::PartitionState(PartitionTransition::Merging {
            started_at_unix_millis: 0,
        }),
    ];
    for value in &values {
        assert_truncated_and_trailing(codec, id(1), value);
    }
}

#[test]
fn version_and_type_fail_closed() {
    let bootstrap = ValueCodec::bootstrap();
    let manifest = minimal_manifest();
    let manifest_value = PersistentValue::IndexManifest(manifest.clone());
    let manifest_key = key_for_value(id(1), &manifest_value);
    let mut bytes = bootstrap.encode(&manifest_value).expect("encode Manifest");

    bytes[1] = 2;
    assert_eq!(
        decode(bootstrap, &manifest_key, &bytes)
            .expect_err("unknown Manifest codec")
            .kind(),
        ErrorKind::UnsupportedFormat
    );

    let name_value = PersistentValue::IndexNameEntry(IndexNameEntry::new(id(1)));
    let name_key = key_for_value(id(1), &name_value);
    let mut bytes = bootstrap.encode(&name_value).expect("encode mapping");
    bytes[1] = 2;
    assert_eq!(
        decode(bootstrap, &name_key, &bytes)
            .expect_err("unknown bootstrap codec")
            .kind(),
        ErrorKind::UnsupportedFormat
    );

    bytes = bootstrap.encode(&manifest_value).expect("encode Manifest");
    bytes[2..4].copy_from_slice(&2_u16.to_be_bytes());
    assert_eq!(
        decode(bootstrap, &manifest_key, &bytes)
            .expect_err("unknown whole format")
            .kind(),
        ErrorKind::UnsupportedFormat
    );

    bytes = bootstrap.encode(&manifest_value).expect("encode Manifest");
    bytes[4] = 2;
    assert_eq!(
        decode(bootstrap, &manifest_key, &bytes)
            .expect_err("unknown declared codec")
            .kind(),
        ErrorKind::UnsupportedFormat
    );

    let codec = index_codec(&manifest);
    let header = PersistentValue::PartitionHeader(
        PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("valid Header"),
    );
    let mut bytes = codec.encode(&header).expect("encode Header");
    bytes[1] = 2;
    assert_corrupt(decode_value(codec, id(1), &header, &bytes));

    let bytes = codec.encode(&header).expect("encode Header");
    let state = PersistentValue::PartitionState(PartitionTransition::Merging {
        started_at_unix_millis: 0,
    });
    assert_corrupt(decode_value(codec, id(1), &state, &bytes));
}

#[test]
fn repeated_identities_must_match_the_logical_key() {
    let manifest = minimal_manifest();
    let bootstrap = ValueCodec::bootstrap();
    let manifest_value = PersistentValue::IndexManifest(manifest.clone());
    let bytes = bootstrap.encode(&manifest_value).expect("encode Manifest");
    assert_corrupt(decode(bootstrap, &LogicalKey::Manifest(id(2)), &bytes));

    let codec = index_codec(&manifest);
    let tree_key = TreeKey::encode(&[], &[]).expect("empty Tree Key");

    let record = PersistentValue::VectorRecord(VectorRecord::new(
        Bytes::from_static(b"record"),
        vec![0.0_f32],
        Vec::<Value>::new(),
    ));
    let bytes = codec.encode(&record).expect("encode Vector Record");
    assert_corrupt(decode(
        codec,
        &LogicalKey::Record {
            index: id(1),
            id: Bytes::from_static(b"other"),
        },
        &bytes,
    ));

    let leaf = PersistentValue::LeafEntry(LeafEntry::new(
        Bytes::from_static(b"record"),
        Vec::<Value>::new(),
        Bytes::from_static(&[0; 14]),
    ));
    let bytes = codec.encode(&leaf).expect("encode Leaf Entry");
    assert_corrupt(decode(
        codec,
        &LogicalKey::LeafEntry {
            index: id(1),
            tree_key: tree_key.clone(),
            partition: pk(1),
            id: Bytes::from_static(b"other"),
        },
        &bytes,
    ));

    let child = PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.0_f32]));
    let bytes = codec.encode(&child).expect("encode Child Entry");
    assert_corrupt(decode(
        codec,
        &LogicalKey::ChildEntry {
            index: id(1),
            tree_key,
            partition: pk(1),
            child: pk(3),
        },
        &bytes,
    ));
}

#[test]
fn index_codec_rejects_keys_owned_by_another_logical_index() {
    let manifest = minimal_manifest();
    let codec = index_codec(&manifest);
    let value = PersistentValue::TreeManifest(
        TreeManifest::new(pk(1), pk(2)).expect("valid Tree Manifest"),
    );
    let bytes = codec.encode(&value).expect("encode Tree Manifest");
    assert_corrupt(decode(
        codec,
        &LogicalKey::TreeManifest {
            index: id(2),
            tree_key: TreeKey::encode(&[], &[]).expect("empty Tree Key"),
        },
        &bytes,
    ));
}

#[test]
fn manifest_rejects_malformed_identity_configuration_and_discriminants() {
    let codec = ValueCodec::bootstrap();
    let value = PersistentValue::IndexManifest(minimal_manifest());

    let mut bytes = codec.encode(&value).expect("encode");
    bytes[5] = 0xff;
    assert_corrupt(decode_value(codec, id(1), &value, &bytes));

    let mut bytes = codec.encode(&value).expect("encode");
    bytes[6..14].fill(0);
    assert_corrupt(decode_value(codec, id(1), &value, &bytes));

    let mut bytes = codec.encode(&value).expect("encode");
    bytes[14..18].fill(0);
    assert_corrupt(decode_value(codec, id(1), &value, &bytes));

    let mut bytes = codec.encode(&value).expect("encode");
    bytes[18] = 0xff;
    assert_corrupt(decode_value(codec, id(1), &value, &bytes));

    let rich = PersistentValue::IndexManifest(rich_manifest());
    let mut bytes = codec.encode(&rich).expect("encode rich Manifest");
    let parameters = [0, 0, 0, 20, 1];
    let offset = bytes
        .windows(parameters.len())
        .position(|window| window == parameters)
        .expect("derived Bloom parameters");
    bytes[offset + 3] -= 1;
    assert_corrupt(decode_value(codec, id(7), &rich, &bytes));
}

#[test]
fn dimensional_count_identity_and_size_invariants_fail_closed() {
    let manifest = minimal_manifest();
    let codec = index_codec(&manifest);

    let name = PersistentValue::IndexNameEntry(IndexNameEntry::new(id(1)));
    let mut bytes = ValueCodec::bootstrap().encode(&name).expect("encode");
    bytes[2..10].fill(0);
    assert_corrupt(decode_value(ValueCodec::bootstrap(), id(1), &name, &bytes));

    let tree = PersistentValue::TreeManifest(
        TreeManifest::new(pk(1), pk(2)).expect("valid Tree Manifest"),
    );
    let mut bytes = codec.encode(&tree).expect("encode");
    bytes[2..10].copy_from_slice(&2_u64.to_be_bytes());
    assert_corrupt(decode_value(codec, id(1), &tree, &bytes));

    let vector = PersistentValue::VectorRecord(VectorRecord::new(
        Bytes::from_static(b"r"),
        vec![0.0_f32],
        Vec::<Value>::new(),
    ));
    let mut bytes = codec.encode(&vector).expect("encode");
    bytes[5..9].copy_from_slice(&2_u32.to_be_bytes());
    assert_corrupt(decode_value(codec, id(1), &vector, &bytes));
    let mut bytes = codec.encode(&vector).expect("encode");
    bytes[9..13].copy_from_slice(&(-0.0_f32).to_bits().to_be_bytes());
    assert_corrupt(decode_value(codec, id(1), &vector, &bytes));

    let payload =
        PersistentValue::OpaquePayload(OpaquePayload::new(Bytes::new()).expect("valid payload"));
    let mut bytes = codec.encode(&payload).expect("encode");
    bytes[2..6].copy_from_slice(&((64 * 1_024 + 1) as u32).to_be_bytes());
    assert_corrupt(decode_value(codec, id(1), &payload, &bytes));

    let location = PersistentValue::RecordLocation(RecordLocation::new(
        TreeKey::encode(&[], &[]).expect("empty Tree Key"),
        pk(1),
    ));
    let mut bytes = codec.encode(&location).expect("encode");
    let length = bytes.len();
    bytes[length - 8..].fill(0);
    assert_corrupt(decode_value(codec, id(1), &location, &bytes));

    let header = PersistentValue::PartitionHeader(
        PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("valid Header"),
    );
    let mut bytes = codec.encode(&header).expect("encode");
    bytes[2..6].fill(0);
    assert_corrupt(decode_value(codec, id(1), &header, &bytes));

    let centroid = PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![0.0_f32]));
    let mut bytes = codec.encode(&centroid).expect("encode");
    bytes[6..10].copy_from_slice(&f32::INFINITY.to_bits().to_be_bytes());
    assert_corrupt(decode_value(codec, id(1), &centroid, &bytes));

    let child = PersistentValue::ChildEntry(ChildEntry::new(pk(2), vec![0.0_f32]));
    let mut bytes = codec.encode(&child).expect("encode");
    bytes[2..10].fill(0);
    assert_corrupt(decode_value(codec, id(1), &child, &bytes));

    let leaf = PersistentValue::LeafEntry(LeafEntry::new(
        Bytes::from_static(b"r"),
        Vec::<Value>::new(),
        Bytes::from_static(&[0; 14]),
    ));
    let mut bytes = codec.encode(&leaf).expect("encode");
    bytes[2..4].fill(0);
    assert_corrupt(decode_value(codec, id(1), &leaf, &bytes));
    let mut bytes = codec.encode(&leaf).expect("encode");
    bytes[7..11].copy_from_slice(&13_u32.to_be_bytes());
    bytes.pop();
    assert_corrupt(decode_value(codec, id(1), &leaf, &bytes));

    let state = PersistentValue::PartitionState(PartitionTransition::Splitting {
        left: pk(2),
        right: pk(3),
        started_at_unix_millis: 0,
    });
    let mut bytes = codec.encode(&state).expect("encode");
    let left = bytes[11..19].to_vec();
    bytes[19..27].copy_from_slice(&left);
    assert_corrupt(decode_value(codec, id(1), &state, &bytes));
}

#[test]
fn schema_and_synopsis_invariants_fail_closed() {
    let manifest = rich_manifest();
    let codec = index_codec(&manifest);

    let record = PersistentValue::VectorRecord(VectorRecord::new(
        Bytes::from_static(b"r"),
        vec![1.0_f32, 2.0],
        vec![
            Value::string("tenant-a").expect("valid String"),
            Value::f64(1.0).expect("finite"),
        ],
    ));
    let mut bytes = codec.encode(&record).expect("encode");
    // Record ID framing, dimension, and two f32 components occupy bytes 2..17.
    bytes[17..19].copy_from_slice(&1_u16.to_be_bytes());
    assert_corrupt(decode_value(codec, id(7), &record, &bytes));

    let synopsis = PersistentValue::PartitionSynopsis(rich_synopsis(&manifest));
    let mut bytes = codec.encode(&synopsis).expect("encode");
    bytes[4] = 0x80;
    assert_corrupt(decode_value(codec, id(7), &synopsis, &bytes));

    let mut bytes = codec.encode(&synopsis).expect("encode");
    bytes[4] |= 0b01;
    assert_corrupt(decode_value(codec, id(7), &synopsis, &bytes));

    let mut bytes = codec.encode(&synopsis).expect("encode");
    let bloom_bytes = usize::try_from(
        manifest.bloom_parameters()[1]
            .expect("score Bloom")
            .bit_count(),
    )
    .expect("u32 fits usize")
    .div_ceil(8);
    let bloom_start = bytes.len() - bloom_bytes;
    bytes[bloom_start..].fill(0);
    assert_corrupt(decode_value(codec, id(7), &synopsis, &bytes));

    let mut bytes = codec.encode(&synopsis).expect("encode");
    let bloom_start = bytes.len() - bloom_bytes;
    let unrelated_bit = (0..manifest.bloom_parameters()[1]
        .expect("score Bloom")
        .bit_count())
        .map(|bit| usize::try_from(bit).expect("u32 fits usize"))
        .find(|bit| bytes[bloom_start + bit / 8] & (1 << (bit % 8)) == 0)
        .expect("Bloom has an unset bit");
    bytes[bloom_start..].fill(0);
    bytes[bloom_start + unrelated_bit / 8] = 1 << (unrelated_bit % 8);
    assert_corrupt(decode_value(codec, id(7), &synopsis, &bytes));
}

#[test]
fn every_transition_variant_round_trips() {
    let manifest = minimal_manifest();
    let codec = index_codec(&manifest);
    let transitions = [
        PartitionTransition::Ready {
            started_at_unix_millis: 42,
        },
        PartitionTransition::Splitting {
            left: pk(2),
            right: pk(3),
            started_at_unix_millis: 42,
        },
        PartitionTransition::ReceivingSplit {
            source: pk(4),
            started_at_unix_millis: 42,
        },
        PartitionTransition::DrainingSplit {
            left: pk(5),
            right: pk(6),
            started_at_unix_millis: 42,
        },
        PartitionTransition::Merging {
            started_at_unix_millis: 42,
        },
    ];
    for transition in transitions {
        round_trip(codec, id(1), &PersistentValue::PartitionState(transition));
    }
}

#[test]
fn reopened_manifest_reproduces_identical_index_value_bytes() {
    let manifest = rich_manifest();
    let bootstrap = ValueCodec::bootstrap();
    let manifest_bytes = bootstrap
        .encode(&PersistentValue::IndexManifest(manifest.clone()))
        .expect("encode Manifest");
    let manifest_key = LogicalKey::Manifest(manifest.logical_index_id());
    let PersistentValue::IndexManifest(reopened) =
        decode(bootstrap, &manifest_key, &manifest_bytes).expect("decode Manifest")
    else {
        panic!("expected Index Manifest")
    };
    let value = PersistentValue::LeafEntry(LeafEntry::new(
        Bytes::from_static(b"record"),
        vec![
            Value::string("tenant-a").expect("valid String"),
            Value::f64(2.5).expect("finite"),
        ],
        Bytes::from_static(&[0; 15]),
    ));
    assert_eq!(
        ValueCodec::for_index(&manifest)
            .encode(&value)
            .expect("encode in first process"),
        ValueCodec::for_index(&reopened)
            .encode(&value)
            .expect("encode after reopen")
    );
}
