//! Canonical logical-key codec contract tests.
//!
//! Golden vectors pin the version-1 byte layout for every key family and edge
//! value; property tests cover ordering, round trips, and fail-closed decoding.

use std::cmp::Ordering;

use bytes::Bytes;
use ktann::api::{DataType, ErrorKind, IndexName, LogicalIndexId, PartitionKey, Value};
use ktann::storage::keys::{
    LogicalKey, TreeKey, centroid_key, child_entry_key, decode_key, header_key,
    index_id_allocator_key, index_range, leaf_entry_key, location_key, manifest_key,
    name_directory_key, partition_range, payload_key, record_key, state_key, synopsis_key,
    tree_manifest_key, tree_manifest_prefix_range, tree_manifest_range,
};

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test id is nonzero")
}

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test partition key is nonzero")
}

fn tk(types: &[DataType], values: &[Value]) -> Vec<u8> {
    TreeKey::encode(types, values)
        .expect("test tree key is canonical")
        .as_bytes()
        .to_vec()
}

fn is_corrupt(types: &[DataType], bytes: &[u8]) -> bool {
    matches!(decode_key(types, bytes), Err(error) if error.kind() == ErrorKind::Corruption)
}

// ---------------------------------------------------------------------------
// Golden vectors: namespace-level and index-level families.
// ---------------------------------------------------------------------------

#[test]
fn allocator_and_name_directory_golden_bytes() {
    assert_eq!(index_id_allocator_key(), b"\x01\x00\x00");
    let name = IndexName::new("a").expect("valid name");
    assert_eq!(name_directory_key(&name), b"\x01\x00\x01a");
    let nested = IndexName::new("x\x00y").expect("embedded NUL is valid UTF-8");
    assert_eq!(name_directory_key(&nested), b"\x01\x00\x01x\x00y");
}

#[test]
fn index_level_family_golden_bytes() {
    assert_eq!(
        manifest_key(id(1)),
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00"
    );
    assert_eq!(
        record_key(id(1), &Bytes::from_static(b"r")).expect("valid id"),
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x01r\x00\x00"
    );
    assert_eq!(
        location_key(id(1), &Bytes::from_static(b"r")).expect("valid id"),
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x01r\x00\x01"
    );
    assert_eq!(
        payload_key(id(1), &Bytes::from_static(b"r")).expect("valid id"),
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x01r\x00\x02"
    );

    let empty = TreeKey::encode(&[], &[]).expect("empty tree key");
    assert_eq!(
        tree_manifest_key(id(1), &empty),
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x03"
    );
}

#[test]
fn record_group_golden_bytes_escape_embedded_nul() {
    let record_id = Bytes::from_static(b"a\0b");
    assert_eq!(
        record_key(id(1), &record_id).expect("valid id"),
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x01a\x00\xffb\x00\x00"
    );
    assert_eq!(
        decode_key(&[], &record_key(id(1), &record_id).expect("valid id")).expect("decode"),
        LogicalKey::Record {
            index: id(1),
            id: record_id,
        }
    );
}

#[test]
fn record_values_are_adjacent_and_ordered_by_record_id() {
    let index = id(1);
    let record_ids = [
        Bytes::from_static(b"a"),
        Bytes::from_static(b"a\0"),
        Bytes::from_static(b"b"),
    ];
    let keys: Vec<Vec<u8>> = record_ids
        .iter()
        .flat_map(|record_id| {
            [
                record_key(index, record_id).expect("valid id"),
                location_key(index, record_id).expect("valid id"),
                payload_key(index, record_id).expect("valid id"),
            ]
        })
        .collect();

    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(sorted, keys);

    for (record_id, group) in record_ids.iter().zip(keys.chunks_exact(3)) {
        assert!(matches!(
            decode_key(&[], &group[0]).expect("decode Record"),
            LogicalKey::Record { id, .. } if id == *record_id
        ));
        assert!(matches!(
            decode_key(&[], &group[1]).expect("decode Location"),
            LogicalKey::Location { id, .. } if id == *record_id
        ));
        assert!(matches!(
            decode_key(&[], &group[2]).expect("decode Payload"),
            LogicalKey::Payload { id, .. } if id == *record_id
        ));
    }
}

#[test]
fn partition_family_golden_bytes() {
    // Tree Key = (String "a") = 0x61 0x00.
    let types = [DataType::String];
    let tree_key = TreeKey::encode(&types, &[Value::string("a").expect("short string")])
        .expect("canonical tree key");
    let prefix = b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x04\x61\x00";
    let partition = b"\x00\x00\x00\x00\x00\x00\x00\x01";

    let mut expected = prefix.to_vec();
    expected.extend_from_slice(partition);
    expected.push(0x00);
    assert_eq!(header_key(id(1), &tree_key, pk(1)), expected);

    let mut expected = prefix.to_vec();
    expected.extend_from_slice(partition);
    expected.push(0x01);
    assert_eq!(synopsis_key(id(1), &tree_key, pk(1)), expected);

    let mut expected = prefix.to_vec();
    expected.extend_from_slice(partition);
    expected.push(0x02);
    assert_eq!(state_key(id(1), &tree_key, pk(1)), expected);

    let mut expected = prefix.to_vec();
    expected.extend_from_slice(partition);
    expected.push(0x03);
    assert_eq!(centroid_key(id(1), &tree_key, pk(1)), expected);

    let mut expected = prefix.to_vec();
    expected.extend_from_slice(partition);
    expected.extend_from_slice(&[0x04, b'x']);
    assert_eq!(
        leaf_entry_key(id(1), &tree_key, pk(1), &Bytes::from_static(b"x")).expect("valid id"),
        expected
    );

    let mut expected = prefix.to_vec();
    expected.extend_from_slice(partition);
    expected.extend_from_slice(&[0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02]);
    assert_eq!(child_entry_key(id(1), &tree_key, pk(1), pk(2)), expected);
}

// ---------------------------------------------------------------------------
// Golden vectors: Tree Key scalar encodings and edge values.
// ---------------------------------------------------------------------------

#[test]
fn tree_key_bool_golden_bytes() {
    assert_eq!(tk(&[DataType::Bool], &[Value::Bool(false)]), b"\x00");
    assert_eq!(tk(&[DataType::Bool], &[Value::Bool(true)]), b"\x01");
}

#[test]
fn tree_key_i64_golden_bytes() {
    assert_eq!(
        tk(&[DataType::I64], &[Value::I64(i64::MIN)]),
        b"\x00\x00\x00\x00\x00\x00\x00\x00"
    );
    assert_eq!(
        tk(&[DataType::I64], &[Value::I64(-1)]),
        b"\x7f\xff\xff\xff\xff\xff\xff\xff"
    );
    assert_eq!(
        tk(&[DataType::I64], &[Value::I64(0)]),
        b"\x80\x00\x00\x00\x00\x00\x00\x00"
    );
    assert_eq!(
        tk(&[DataType::I64], &[Value::I64(1)]),
        b"\x80\x00\x00\x00\x00\x00\x00\x01"
    );
    assert_eq!(
        tk(&[DataType::I64], &[Value::I64(i64::MAX)]),
        b"\xff\xff\xff\xff\xff\xff\xff\xff"
    );
}

#[test]
fn tree_key_f64_golden_bytes() {
    assert_eq!(
        tk(&[DataType::F64], &[Value::F64(f64::MIN)]),
        b"\x00\x10\x00\x00\x00\x00\x00\x00"
    );
    assert_eq!(
        tk(&[DataType::F64], &[Value::F64(-1.0)]),
        b"\x40\x0f\xff\xff\xff\xff\xff\xff"
    );
    assert_eq!(
        tk(&[DataType::F64], &[Value::F64(0.0)]),
        b"\x80\x00\x00\x00\x00\x00\x00\x00"
    );
    assert_eq!(
        tk(&[DataType::F64], &[Value::F64(1.0)]),
        b"\xbf\xf0\x00\x00\x00\x00\x00\x00"
    );
    assert_eq!(
        tk(&[DataType::F64], &[Value::F64(f64::MAX)]),
        b"\xff\xef\xff\xff\xff\xff\xff\xff"
    );
}

#[test]
fn tree_key_string_golden_bytes() {
    assert_eq!(
        tk(&[DataType::String], &[Value::string("").expect("empty")]),
        b"\x00"
    );
    assert_eq!(
        tk(&[DataType::String], &[Value::string("a").expect("short")]),
        b"\x61\x00"
    );
    assert_eq!(
        tk(
            &[DataType::String],
            &[Value::string("a\0b").expect("embedded NUL")]
        ),
        b"\x61\x00\xff\x62\x00"
    );
    assert_eq!(
        tk(
            &[DataType::String],
            &[Value::string("\u{4f60}\u{597d}").expect("multibyte")]
        ),
        "\u{4f60}\u{597d}"
            .as_bytes()
            .iter()
            .copied()
            .chain([0x00])
            .collect::<Vec<u8>>()
    );
}

#[test]
fn tree_key_tuple_golden_bytes() {
    let types = [DataType::String, DataType::I64];
    let values = [Value::string("a").expect("short"), Value::I64(1)];
    assert_eq!(
        tk(&types, &values),
        b"\x61\x00\x80\x00\x00\x00\x00\x00\x00\x01"
    );
}

// ---------------------------------------------------------------------------
// Ordering properties.
// ---------------------------------------------------------------------------

fn cmp_value(ty: DataType, a: &Value, b: &Value) -> Ordering {
    match ty {
        DataType::Bool => match (a, b) {
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            _ => panic!("bool type mismatch"),
        },
        DataType::I64 => match (a, b) {
            (Value::I64(a), Value::I64(b)) => a.cmp(b),
            _ => panic!("i64 type mismatch"),
        },
        DataType::F64 => match (a, b) {
            (Value::F64(a), Value::F64(b)) => a.total_cmp(b),
            _ => panic!("f64 type mismatch"),
        },
        DataType::String => match (a, b) {
            (Value::String(a), Value::String(b)) => a.as_bytes().cmp(b.as_bytes()),
            _ => panic!("string type mismatch"),
        },
        _ => panic!("unknown data type"),
    }
}

fn typed_cmp(types: &[DataType], a: &[Value], b: &[Value]) -> Ordering {
    for (ty, (a, b)) in types.iter().zip(a.iter().zip(b)) {
        let order = cmp_value(*ty, a, b);
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

#[test]
fn tree_key_byte_order_matches_typed_order() {
    let cases: &[(&[DataType], Vec<Vec<Value>>)] = &[
        (
            &[DataType::Bool],
            vec![vec![Value::Bool(false)], vec![Value::Bool(true)]],
        ),
        (
            &[DataType::I64],
            vec![
                vec![Value::I64(i64::MIN)],
                vec![Value::I64(-1)],
                vec![Value::I64(0)],
                vec![Value::I64(1)],
                vec![Value::I64(i64::MAX)],
            ],
        ),
        (
            &[DataType::F64],
            vec![
                vec![Value::F64(f64::MIN)],
                vec![Value::F64(-1.5)],
                vec![Value::F64(-0.5)],
                vec![Value::F64(0.0)],
                vec![Value::F64(0.5)],
                vec![Value::F64(f64::MAX)],
            ],
        ),
        (
            &[DataType::String],
            vec![
                vec![Value::string("").expect("empty")],
                vec![Value::string("a").expect("short")],
                vec![Value::string("a\0").expect("nul")],
                vec![Value::string("a\0b").expect("nul")],
                vec![Value::string("ab").expect("short")],
                vec![Value::string("b").expect("short")],
            ],
        ),
        (
            &[DataType::String, DataType::I64],
            vec![
                vec![Value::string("a").expect("short"), Value::I64(-1)],
                vec![Value::string("a").expect("short"), Value::I64(0)],
                vec![Value::string("a").expect("short"), Value::I64(1)],
                vec![Value::string("b").expect("short"), Value::I64(i64::MIN)],
                vec![Value::string("b").expect("short"), Value::I64(0)],
            ],
        ),
    ];

    for (types, tuples) in cases {
        let mut by_typed = tuples.clone();
        by_typed.sort_by(|a, b| typed_cmp(types, a, b));
        let mut by_bytes = tuples.clone();
        by_bytes.sort_by_key(|a| tk(types, a));
        assert_eq!(by_typed, by_bytes, "types {types:?}");
    }
}

// ---------------------------------------------------------------------------
// Round trips.
// ---------------------------------------------------------------------------

fn encode_key(key: &LogicalKey) -> Vec<u8> {
    match key {
        LogicalKey::IndexIdAllocator => index_id_allocator_key(),
        LogicalKey::IndexNameDirectory(name) => name_directory_key(name),
        LogicalKey::Manifest(index) => manifest_key(*index),
        LogicalKey::Record { index, id } => record_key(*index, id).expect("valid id"),
        LogicalKey::Location { index, id } => location_key(*index, id).expect("valid id"),
        LogicalKey::Payload { index, id } => payload_key(*index, id).expect("valid id"),
        LogicalKey::TreeManifest { index, tree_key } => tree_manifest_key(*index, tree_key),
        LogicalKey::Header {
            index,
            tree_key,
            partition,
        } => header_key(*index, tree_key, *partition),
        LogicalKey::Synopsis {
            index,
            tree_key,
            partition,
        } => synopsis_key(*index, tree_key, *partition),
        LogicalKey::State {
            index,
            tree_key,
            partition,
        } => state_key(*index, tree_key, *partition),
        LogicalKey::Centroid {
            index,
            tree_key,
            partition,
        } => centroid_key(*index, tree_key, *partition),
        LogicalKey::LeafEntry {
            index,
            tree_key,
            partition,
            id,
        } => leaf_entry_key(*index, tree_key, *partition, id).expect("valid id"),
        LogicalKey::ChildEntry {
            index,
            tree_key,
            partition,
            child,
        } => child_entry_key(*index, tree_key, *partition, *child),
        _ => panic!("unhandled logical key variant"),
    }
}

#[test]
fn every_key_family_round_trips() {
    let types = [DataType::String, DataType::I64];
    let tree_key = TreeKey::encode(
        &types,
        &[Value::string("a\0b").expect("nul"), Value::I64(-3)],
    )
    .expect("canonical tree key");
    let index = id(7);
    let partition = pk(9);

    let keys = [
        LogicalKey::IndexIdAllocator,
        LogicalKey::IndexNameDirectory(IndexName::new("name").expect("valid")),
        LogicalKey::Manifest(index),
        LogicalKey::Record {
            index,
            id: Bytes::from_static(b"rec"),
        },
        LogicalKey::Location {
            index,
            id: Bytes::from_static(b"rec"),
        },
        LogicalKey::Payload {
            index,
            id: Bytes::from_static(b"rec"),
        },
        LogicalKey::TreeManifest {
            index,
            tree_key: tree_key.clone(),
        },
        LogicalKey::Header {
            index,
            tree_key: tree_key.clone(),
            partition,
        },
        LogicalKey::Synopsis {
            index,
            tree_key: tree_key.clone(),
            partition,
        },
        LogicalKey::State {
            index,
            tree_key: tree_key.clone(),
            partition,
        },
        LogicalKey::Centroid {
            index,
            tree_key: tree_key.clone(),
            partition,
        },
        LogicalKey::LeafEntry {
            index,
            tree_key: tree_key.clone(),
            partition,
            id: Bytes::from_static(b"leaf"),
        },
        LogicalKey::ChildEntry {
            index,
            tree_key: tree_key.clone(),
            partition,
            child: pk(10),
        },
    ];

    for key in &keys {
        let encoded = encode_key(key);
        let decoded = decode_key(&types, &encoded).expect("key decodes");
        assert_eq!(&decoded, key, "round trip of {key:?}");
    }
}

#[test]
fn tree_key_values_round_trip() {
    let types = [
        DataType::Bool,
        DataType::I64,
        DataType::F64,
        DataType::String,
    ];
    let values = [
        Value::Bool(true),
        Value::I64(-42),
        Value::F64(2.5),
        Value::string("h\u{e9}llo\0world").expect("utf-8 with NUL"),
    ];
    let tree_key = TreeKey::encode(&types, &values).expect("canonical");
    assert_eq!(tree_key.values(&types).expect("decodes"), values);
}

// ---------------------------------------------------------------------------
// Prefix and range construction.
// ---------------------------------------------------------------------------

#[test]
fn index_and_partition_ranges_are_contiguous() {
    let one = id(1);
    let two = id(2);
    assert_eq!(index_range(one).end(), index_range(two).start());

    let types = [DataType::I64];
    let tree_key = TreeKey::encode(&types, &[Value::I64(0)]).expect("canonical");
    let range = partition_range(id(1), &tree_key, pk(1));
    assert!(range.start() < range.end());
    let header = header_key(id(1), &tree_key, pk(1));
    let leaf =
        leaf_entry_key(id(1), &tree_key, pk(1), &Bytes::from_static(b"z")).expect("valid id");
    assert!(header.as_slice() >= range.start() && header.as_slice() < range.end());
    assert!(leaf.as_slice() >= range.start() && leaf.as_slice() < range.end());
}

#[test]
fn tree_manifest_prefix_range_contains_matching_keys() {
    let types = [DataType::String];
    let index = id(1);

    let a = TreeKey::encode(&types, &[Value::string("a").expect("short")]).expect("canonical");
    let b = TreeKey::encode(&types, &[Value::string("b").expect("short")]).expect("canonical");

    let range = tree_manifest_prefix_range(index, &types, &[Value::string("a").expect("short")])
        .expect("valid prefix");
    let ka = tree_manifest_key(index, &a);
    let kb = tree_manifest_key(index, &b);
    assert!(ka.as_slice() >= range.start() && ka.as_slice() < range.end());
    assert!(!(kb.as_slice() >= range.start() && kb.as_slice() < range.end()));
}

#[test]
fn tree_manifest_empty_prefix_matches_full_directory() {
    let index = id(1);
    let types: [DataType; 0] = [];
    let full = tree_manifest_range(index);
    let by_prefix = tree_manifest_prefix_range(index, &types, &[]).expect("empty prefix");
    assert_eq!(by_prefix.start(), full.start());
    assert_eq!(by_prefix.end(), full.end());
}

// ---------------------------------------------------------------------------
// Fail-closed decoding.
// ---------------------------------------------------------------------------

#[test]
fn decode_rejects_every_truncation_of_every_valid_key() {
    let types = [DataType::String];
    let tree_key =
        TreeKey::encode(&types, &[Value::string("a").expect("short")]).expect("canonical");
    let index = id(1);
    let partition = pk(1);

    let valid = [
        index_id_allocator_key(),
        name_directory_key(&IndexName::new("n").expect("valid")),
        manifest_key(index),
        record_key(index, &Bytes::from_static(b"r")).expect("valid id"),
        location_key(index, &Bytes::from_static(b"r")).expect("valid id"),
        payload_key(index, &Bytes::from_static(b"r")).expect("valid id"),
        tree_manifest_key(index, &tree_key),
        header_key(index, &tree_key, partition),
        synopsis_key(index, &tree_key, partition),
        state_key(index, &tree_key, partition),
        centroid_key(index, &tree_key, partition),
        leaf_entry_key(index, &tree_key, partition, &Bytes::from_static(b"x")).expect("valid id"),
        child_entry_key(index, &tree_key, partition, pk(2)),
    ];

    for key in &valid {
        for cut in 0..key.len() {
            assert!(
                is_corrupt(&types, &key[..cut]),
                "truncation of {key:?} at {cut} must fail"
            );
        }
    }
}

#[test]
fn decode_rejects_trailing_bytes_on_fixed_terminal_keys() {
    let types = [DataType::String];
    let tree_key =
        TreeKey::encode(&types, &[Value::string("a").expect("short")]).expect("canonical");
    let index = id(1);
    let partition = pk(1);

    let fixed = [
        index_id_allocator_key(),
        manifest_key(index),
        tree_manifest_key(index, &tree_key),
        header_key(index, &tree_key, partition),
        synopsis_key(index, &tree_key, partition),
        state_key(index, &tree_key, partition),
        centroid_key(index, &tree_key, partition),
        child_entry_key(index, &tree_key, partition, pk(2)),
    ];
    for key in &fixed {
        let mut trailing = key.clone();
        trailing.push(0x00);
        assert!(
            is_corrupt(&types, &trailing),
            "trailing byte on {key:?} must fail"
        );
    }
}

#[test]
fn decode_rejects_unknown_discriminators() {
    let types: [DataType; 0] = [];
    // Unknown version.
    assert!(is_corrupt(&types, b"\x02\x00\x00"));
    // Unknown scope.
    assert!(is_corrupt(&types, b"\x01\xff\x00"));
    // Unknown namespace kind.
    assert!(is_corrupt(&types, b"\x01\x00\xff"));
    // Unknown index kind (id 1, kind 0x0a).
    assert!(is_corrupt(
        &types,
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x0a"
    ));
    // Unknown partition subkind.
    assert!(is_corrupt(
        &types,
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x04\x00\x00\x00\x00\x00\x00\x00\x01\xff"
    ));
}

#[test]
fn decode_rejects_zero_identities() {
    let types: [DataType; 0] = [];
    // Zero Logical Index ID in a manifest key.
    assert!(is_corrupt(
        &types,
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ));
    // Zero Partition Key in a header key (empty tree key, id 1).
    assert!(is_corrupt(
        &types,
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ));
}

#[test]
fn decode_rejects_noncanonical_tree_key_scalars() {
    let types_f64 = [DataType::F64];
    // -0.0 encodes to 0x7f ff ff ff ff ff ff ff ff and is noncanonical.
    let neg_zero =
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x03\x7f\xff\xff\xff\xff\xff\xff\xff".to_vec();
    assert!(is_corrupt(&types_f64, &neg_zero));

    // Non-finite F64 (infinity sign bit 0 -> encoded 0xff f0 00 ...).
    let infinity =
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x03\xff\xf0\x00\x00\x00\x00\x00\x00".to_vec();
    assert!(is_corrupt(&types_f64, &infinity));

    // Noncanonical Bool (0x02).
    let types_bool = [DataType::Bool];
    assert!(is_corrupt(
        &types_bool,
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x03\x02"
    ));

    // Unterminated String field.
    let types_string = [DataType::String];
    assert!(is_corrupt(
        &types_string,
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x03\x61"
    ));

    // Invalid UTF-8 String field (0xff followed by terminator).
    assert!(is_corrupt(
        &types_string,
        b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x03\xff\x00"
    ));
}

#[test]
fn decode_rejects_invalid_utf8_name() {
    let types: [DataType; 0] = [];
    assert!(is_corrupt(&types, b"\x01\x00\x01\xff"));
}

#[test]
fn decode_rejects_overlong_record_id_and_name() {
    let types: [DataType; 0] = [];

    // 257-byte record id after a Record kind.
    let mut key = vec![0x01, 0x01];
    key.extend_from_slice(&1_u64.to_be_bytes());
    key.push(0x01);
    key.extend(std::iter::repeat_n(b'x', 257));
    assert!(is_corrupt(&types, &key));

    // 256-byte Index Name after the directory kind.
    let mut key = vec![0x01, 0x00, 0x01];
    key.extend(std::iter::repeat_n(b'y', 256));
    assert!(is_corrupt(&types, &key));
}

#[test]
fn encode_rejects_invalid_input() {
    let types_string = [DataType::String];
    let types_f64 = [DataType::F64];

    // Type mismatch, NULL, and length mismatch.
    assert_eq!(
        TreeKey::encode(&types_string, &[Value::I64(0)])
            .expect_err("type mismatch")
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        TreeKey::encode(&types_string, &[Value::Null])
            .expect_err("null field")
            .kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        TreeKey::encode(&types_string, &[])
            .expect_err("length mismatch")
            .kind(),
        ErrorKind::InvalidArgument
    );

    // Non-finite and noncanonical F64.
    for bad in [f64::NAN, f64::INFINITY, -0.0] {
        assert_eq!(
            TreeKey::encode(&types_f64, &[Value::F64(bad)])
                .expect_err("noncanonical f64")
                .kind(),
            ErrorKind::InvalidArgument
        );
    }

    // Overlong string.
    let overlong = Value::String("x".repeat(1_025));
    assert_eq!(
        TreeKey::encode(&types_string, &[overlong])
            .expect_err("overlong string")
            .kind(),
        ErrorKind::InvalidArgument
    );

    // Empty and overlong record id.
    assert_eq!(
        record_key(id(1), &Bytes::new())
            .expect_err("empty id")
            .kind(),
        ErrorKind::InvalidArgument
    );
    let overlong_id = Bytes::from(vec![0u8; 257]);
    assert_eq!(
        record_key(id(1), &overlong_id)
            .expect_err("overlong id")
            .kind(),
        ErrorKind::InvalidArgument
    );

    // Prefix naming more fields than the schema has.
    assert_eq!(
        tree_manifest_prefix_range(id(1), &types_f64, &[Value::F64(0.0), Value::F64(0.0)])
            .expect_err("overlong prefix")
            .kind(),
        ErrorKind::InvalidArgument
    );
}
