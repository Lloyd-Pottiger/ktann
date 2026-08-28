//! Typed logical-storage operation contract tests.

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, IndexName, LogicalIndexId, Metric,
    PartitionKey, Value,
};
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, HardLimits, InsertOutcome, ScanLimits, WriteTxn,
};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{
    IndexIdAllocator, IndexLifecycle, IndexManifest, IndexNameEntry, LeafEntry, PartitionCentroid,
    PartitionHeader, PartitionState, PartitionSynopsis, PartitionTransition, PersistentValue,
    TreeManifest, ValueCodec, ValueKind, VectorRecord,
};
use ktann::storage::{
    LogicalRange, MutationBuilder, ReadLogicalTxn, TransactionSize, WriteLogicalTxn,
};

use super::support::{DeterministicBackend, DeterministicConfig, DeterministicWriteTxn};

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
}

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

fn manifest() -> IndexManifest {
    IndexManifest::new(
        IndexLifecycle::Active,
        id(7),
        IndexConfig::new(1, Metric::L2).expect("valid config"),
        [7; 32],
        vec![],
    )
    .expect("valid manifest")
}

fn tree_key() -> TreeKey {
    TreeKey::encode(&[], &[]).expect("empty Tree Key is canonical")
}

fn record_key(record_id: &'static [u8]) -> LogicalKey {
    LogicalKey::Record {
        index: id(7),
        id: Bytes::from_static(record_id),
    }
}

fn record(record_id: &'static [u8]) -> PersistentValue {
    PersistentValue::VectorRecord(VectorRecord::new(
        Bytes::from_static(record_id),
        vec![1.0_f32],
        Vec::<Value>::new(),
    ))
}

#[tokio::test]
async fn absent_point_and_batch_reads_preserve_shape() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, &manifest).expect("bind manifest");

    assert_eq!(txn.get(record_key(b"missing")).await.expect("get"), None);

    let keys = vec![record_key(b"a"), record_key(b"b"), record_key(b"a")];
    assert_eq!(
        txn.batch_get(keys).await.expect("batch get"),
        vec![None, None, None]
    );
    assert!(
        txn.batch_get(Vec::new())
            .await
            .expect("empty batch")
            .is_empty()
    );
}

#[tokio::test]
async fn namespace_manifest_and_name_directory_operations_are_typed() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let name = IndexName::new("documents").expect("valid name");
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);

    let inserted = txn
        .insert(
            LogicalKey::IndexNameDirectory(name.clone()),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id(7))),
        )
        .await
        .expect("typed insert");
    assert_eq!(inserted, InsertOutcome::Inserted);
    let size_after_insert = txn.size();
    let duplicate = txn
        .insert(
            LogicalKey::IndexNameDirectory(name.clone()),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id(7))),
        )
        .await
        .expect("duplicate insert");
    assert_eq!(duplicate, InsertOutcome::AlreadyExists);
    assert_eq!(txn.size(), size_after_insert);

    let mut mutations = txn.mutations();
    mutations
        .put(
            LogicalKey::Manifest(id(7)),
            PersistentValue::IndexManifest(manifest.clone()),
        )
        .expect("queue manifest");
    mutations
        .put(
            LogicalKey::IndexIdAllocator,
            PersistentValue::IndexIdAllocator(IndexIdAllocator::new(7)),
        )
        .expect("queue allocator");
    txn.apply(mutations).await.expect("apply mutations");
    txn.commit().await.expect("commit");

    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    assert_eq!(
        txn.get(LogicalKey::IndexIdAllocator)
            .await
            .expect("allocator"),
        Some(PersistentValue::IndexIdAllocator(IndexIdAllocator::new(7)))
    );
    assert_eq!(
        txn.get(LogicalKey::IndexNameDirectory(name))
            .await
            .expect("name entry"),
        Some(PersistentValue::IndexNameEntry(IndexNameEntry::new(id(7))))
    );
    assert_eq!(
        txn.get(LogicalKey::Manifest(id(7)))
            .await
            .expect("manifest"),
        Some(PersistentValue::IndexManifest(manifest))
    );
}

#[tokio::test]
async fn duplicate_insert_does_not_require_remaining_mutation_budget() {
    let backend = DeterministicBackend::new(DeterministicConfig {
        admission_budget: AdmissionBudget {
            max_mutations: 1,
            max_mutation_bytes: 1_024,
            mutation_key_overhead_bytes: 0,
        },
        ..DeterministicConfig::default()
    });
    let name = IndexName::new("documents").expect("valid name");
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn =
        WriteLogicalTxn::bootstrap(raw, backend.hard_limits(), backend.admission_budget());

    let first = txn
        .insert(
            LogicalKey::IndexNameDirectory(name.clone()),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id(7))),
        )
        .await
        .expect("first insert");
    assert_eq!(first, InsertOutcome::Inserted);
    let full_size = txn.size();
    let duplicate = txn
        .insert(
            LogicalKey::IndexNameDirectory(name),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id(7))),
        )
        .await
        .expect("duplicate insert at full budget");
    assert_eq!(duplicate, InsertOutcome::AlreadyExists);
    assert_eq!(txn.size(), full_size);
    txn.commit().await.expect("commit insert");
}

#[tokio::test]
async fn partition_scans_decode_mixed_families_and_page_without_read_ahead() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let tree_key = tree_key();
    let partition = pk(1);
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn =
        WriteLogicalTxn::for_index(raw, &manifest, limits, budget).expect("bind manifest");
    let mut mutations = txn.mutations();

    for (record_id, value) in [(b"b".as_slice(), record(b"b")), (b"a", record(b"a"))] {
        mutations
            .put(record_key(record_id), value)
            .expect("queue record");
        mutations
            .put(
                LogicalKey::LeafEntry {
                    index: id(7),
                    tree_key: tree_key.clone(),
                    partition,
                    id: Bytes::copy_from_slice(record_id),
                },
                PersistentValue::LeafEntry(LeafEntry::new(
                    Bytes::copy_from_slice(record_id),
                    Vec::<Value>::new(),
                    Bytes::from_static(&[0; 14]),
                )),
            )
            .expect("queue Leaf Entry");
    }
    mutations
        .put(
            LogicalKey::Centroid {
                index: id(7),
                tree_key: tree_key.clone(),
                partition,
            },
            PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![0.0_f32])),
        )
        .expect("queue centroid");
    mutations
        .put(
            LogicalKey::State {
                index: id(7),
                tree_key: tree_key.clone(),
                partition,
            },
            PersistentValue::PartitionState(PartitionTransition::Ready {
                started_at_unix_millis: 1,
            }),
        )
        .expect("queue state");
    mutations
        .put(
            LogicalKey::Synopsis {
                index: id(7),
                tree_key: tree_key.clone(),
                partition,
            },
            PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(&manifest)),
        )
        .expect("queue synopsis");
    mutations
        .put(
            LogicalKey::Header {
                index: id(7),
                tree_key: tree_key.clone(),
                partition,
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 2, 9, PartitionState::Ready).expect("valid header"),
            ),
        )
        .expect("queue header");
    mutations
        .put(
            LogicalKey::TreeManifest {
                index: id(7),
                tree_key: tree_key.clone(),
            },
            PersistentValue::TreeManifest(
                TreeManifest::new(pk(1), pk(1)).expect("valid Tree Manifest"),
            ),
        )
        .expect("queue Tree Manifest");

    txn.apply(mutations).await.expect("apply mutations");
    txn.commit().await.expect("commit");

    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, &manifest).expect("bind manifest");
    let range =
        LogicalRange::partition(&manifest, &tree_key, partition).expect("valid partition range");
    let mut cursor = None;
    let mut kinds = Vec::new();
    loop {
        let page = txn
            .scan(
                &range,
                cursor.as_ref(),
                ScanLimits {
                    item_limit: 2,
                    byte_limit: usize::MAX,
                },
            )
            .await
            .expect("typed partition scan");
        kinds.extend(page.items().iter().map(|item| item.value().kind()));
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(
        kinds,
        vec![
            ValueKind::PartitionHeader,
            ValueKind::PartitionSynopsis,
            ValueKind::PartitionState,
            ValueKind::PartitionCentroid,
            ValueKind::LeafEntry,
            ValueKind::LeafEntry,
        ]
    );

    let leaf_range = LogicalRange::leaf_entries(&manifest, &tree_key, partition)
        .expect("valid Leaf Entry range");
    let page = txn
        .scan(
            &leaf_range,
            None,
            ScanLimits {
                item_limit: 10,
                byte_limit: usize::MAX,
            },
        )
        .await
        .expect("typed Leaf Entry scan");
    assert_eq!(page.items().len(), 2);
    assert!(
        page.items()
            .iter()
            .all(|item| item.value().kind() == ValueKind::LeafEntry)
    );
    assert!(page.next_cursor().is_none());
}

#[tokio::test]
async fn reads_and_scans_fail_closed_on_key_value_identity_mismatch() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let codec = ValueCodec::for_index(&manifest);
    let mismatched = codec
        .encode(&record(b"actual"))
        .expect("encode mismatched record");
    let raw_key =
        keys::record_key(id(7), &Bytes::from_static(b"requested")).expect("encode record key");
    let mut raw = backend.begin_write().await.expect("begin write");
    raw.put(Bytes::from(raw_key), Bytes::from(mismatched))
        .await
        .expect("seed corruption");
    raw.commit().await.expect("commit corruption");

    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, &manifest).expect("bind manifest");
    let error = txn
        .get(record_key(b"requested"))
        .await
        .expect_err("identity mismatch must fail closed");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    let range = LogicalRange::index(&manifest);
    let error = txn
        .scan(
            &range,
            None,
            ScanLimits {
                item_limit: 10,
                byte_limit: usize::MAX,
            },
        )
        .await
        .expect_err("scan identity mismatch must fail closed");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[test]
fn mutation_builder_enforces_count_and_byte_limits_without_partial_change() {
    let manifest = manifest();
    let hard_limits = HardLimits {
        max_key_bytes: 1_024,
        max_value_bytes: 4_096,
    };
    let count_budget = AdmissionBudget {
        max_mutations: 2,
        max_mutation_bytes: usize::MAX,
        mutation_key_overhead_bytes: 0,
    };
    let mut builder =
        MutationBuilder::for_index(&manifest, hard_limits, count_budget).expect("valid builder");
    builder.delete(record_key(b"a")).expect("first delete");
    builder.delete(record_key(b"b")).expect("second delete");
    let full_size = builder.size();
    let error = builder
        .delete(record_key(b"c"))
        .expect_err("third mutation exceeds count");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    assert_eq!(builder.size(), full_size);

    let generous_budget = AdmissionBudget {
        max_mutations: 1,
        max_mutation_bytes: usize::MAX,
        mutation_key_overhead_bytes: 0,
    };
    let mut sizing =
        MutationBuilder::for_index(&manifest, hard_limits, generous_budget).expect("valid builder");
    sizing
        .put(record_key(b"sized"), record(b"sized"))
        .expect("sizing mutation");
    let exact_bytes = sizing.size().bytes();

    let byte_budget = AdmissionBudget {
        max_mutations: 1,
        max_mutation_bytes: exact_bytes - 1,
        mutation_key_overhead_bytes: 0,
    };
    let mut limited =
        MutationBuilder::for_index(&manifest, hard_limits, byte_budget).expect("valid builder");
    let error = limited
        .put(record_key(b"sized"), record(b"sized"))
        .expect_err("mutation exceeds byte budget");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    assert_eq!(limited.size(), TransactionSize::default());
}

#[test]
fn mutation_builder_rejects_wrong_value_family_and_hard_limits() {
    let manifest = manifest();
    let hard_limits = HardLimits {
        max_key_bytes: 1,
        max_value_bytes: 4_096,
    };
    let budget = AdmissionBudget {
        max_mutations: 10,
        max_mutation_bytes: usize::MAX,
        mutation_key_overhead_bytes: 0,
    };
    let mut builder =
        MutationBuilder::for_index(&manifest, hard_limits, budget).expect("valid builder");
    let error = builder
        .delete(record_key(b"a"))
        .expect_err("logical key exceeds hard limit");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    assert_eq!(builder.size(), TransactionSize::default());

    let mut builder = MutationBuilder::for_index(
        &manifest,
        HardLimits {
            max_key_bytes: 1_024,
            max_value_bytes: 4_096,
        },
        budget,
    )
    .expect("valid builder");
    let error = builder
        .put(
            record_key(b"a"),
            PersistentValue::PartitionSynopsis(PartitionSynopsis::empty(&manifest)),
        )
        .expect_err("wrong value family");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(builder.size(), TransactionSize::default());
}

#[test]
fn tree_local_inputs_must_match_the_bound_tree_key_schema() {
    let config = IndexConfig::new(1, Metric::L2)
        .expect("valid base config")
        .with_fields(vec![
            FieldSchema::new("tenant", DataType::I64).expect("valid field"),
        ])
        .expect("valid schema")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid Tree Key fields");
    let manifest = IndexManifest::new(IndexLifecycle::Active, id(7), config, [7; 32], vec![None])
        .expect("valid manifest");
    let wrong_tree_key = TreeKey::encode(
        &[DataType::String],
        &[Value::string("wrong schema").expect("valid string")],
    )
    .expect("valid Tree Key under another schema");

    let error = LogicalRange::partition(&manifest, &wrong_tree_key, pk(1))
        .expect_err("range must reject a foreign Tree Key schema");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);

    let mut builder = MutationBuilder::for_index(
        &manifest,
        HardLimits {
            max_key_bytes: 1_024,
            max_value_bytes: 4_096,
        },
        AdmissionBudget {
            max_mutations: 1,
            max_mutation_bytes: 1_024,
            mutation_key_overhead_bytes: 0,
        },
    )
    .expect("valid builder");
    let error = builder
        .delete(LogicalKey::Header {
            index: id(7),
            tree_key: wrong_tree_key,
            partition: pk(1),
        })
        .expect_err("typed key must reject a foreign Tree Key schema");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(builder.size(), TransactionSize::default());
}

#[tokio::test]
async fn range_clear_is_included_in_transaction_admission() {
    let config = DeterministicConfig {
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        admission_budget: AdmissionBudget {
            max_mutations: 1,
            max_mutation_bytes: 1_024,
            mutation_key_overhead_bytes: 0,
        },
        ..DeterministicConfig::default()
    };
    let backend = DeterministicBackend::new(config);
    let manifest = manifest();
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind manifest");
    let range = LogicalRange::index(&manifest);
    let raw_range = keys::index_range(id(7));
    let expected_bytes = raw_range.start().len() + raw_range.end().len();

    txn.clear_range(&range).await.expect("clear range");
    assert_eq!(txn.size().mutations(), 1);
    assert_eq!(txn.size().bytes(), expected_bytes);
    let mut mutations = txn.mutations();
    let error = mutations
        .delete(record_key(b"over-budget"))
        .expect_err("range clear consumes the mutation budget");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    txn.commit().await.expect("commit clear");

    let history = backend.history();
    let committed = history.last().expect("one commit history entry");
    assert_eq!(committed.mutations, 1);
    assert_eq!(committed.mutation_bytes, expected_bytes);
}

#[tokio::test]
async fn applying_a_builder_charges_its_exact_final_size() {
    let config = DeterministicConfig::default();
    let backend = DeterministicBackend::new(config);
    let manifest = manifest();
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind manifest");
    let mut mutations = txn.mutations();
    mutations
        .put(record_key(b"z"), record(b"z"))
        .expect("queue z");
    mutations
        .put(record_key(b"a"), record(b"a"))
        .expect("queue a");
    mutations
        .delete(record_key(b"z"))
        .expect("replace z put with delete");
    let expected = mutations.size();

    txn.apply(mutations).await.expect("apply");
    assert_eq!(txn.size(), expected);
    txn.commit().await.expect("commit");

    let history = backend.history();
    let committed = history.last().expect("one commit history entry");
    assert_eq!(committed.mutations, expected.mutations());
    assert_eq!(committed.mutation_bytes, expected.bytes());
    assert_eq!(committed.distinct_keys, expected.mutations());
}

#[test]
fn manifest_write_with_different_immutable_config_is_rejected_when_bound() {
    let manifest = manifest();
    let different = IndexManifest::new(
        IndexLifecycle::Active,
        id(7),
        IndexConfig::new(2, Metric::L2).expect("valid config"),
        [7; 32],
        vec![],
    )
    .expect("valid manifest");
    let hard_limits = HardLimits {
        max_key_bytes: 1_024,
        max_value_bytes: 4_096,
    };
    let budget = AdmissionBudget {
        max_mutations: 1,
        max_mutation_bytes: 1_024,
        mutation_key_overhead_bytes: 0,
    };

    let mut builder =
        MutationBuilder::for_index(&manifest, hard_limits, budget).expect("bind manifest");
    let error = builder
        .put(
            LogicalKey::Manifest(id(7)),
            PersistentValue::IndexManifest(different),
        )
        .expect_err("different immutable config must be rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(builder.size(), TransactionSize::default());
}

#[tokio::test]
async fn duplicate_insert_fails_closed_on_corrupt_existing_value() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    // Seed a VectorRecord whose embedded record_id ("actual") disagrees with the
    // logical key's record_id ("requested").
    let codec = ValueCodec::for_index(&manifest);
    let mismatched = codec
        .encode(&record(b"actual"))
        .expect("encode mismatched record");
    let raw_key =
        keys::record_key(id(7), &Bytes::from_static(b"requested")).expect("encode record key");
    {
        let mut raw = backend.begin_write().await.expect("begin write");
        raw.put(Bytes::from(raw_key), Bytes::from(mismatched))
            .await
            .expect("seed corruption");
        raw.commit().await.expect("commit corruption");
    }

    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind manifest");
    // A duplicate insert at the corrupted key must fail closed rather than
    // report AlreadyExists on undecodable bytes.
    let error = txn
        .insert(record_key(b"requested"), record(b"requested"))
        .await
        .expect_err("corrupt existing value must fail closed");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;
}

#[tokio::test]
async fn duplicate_insert_establishes_an_update_protected_conflict() {
    let config = DeterministicConfig {
        admission_budget: AdmissionBudget {
            max_mutations: 1,
            max_mutation_bytes: 1_024,
            mutation_key_overhead_bytes: 0,
        },
        ..DeterministicConfig::default()
    };
    let backend = DeterministicBackend::new(config);
    let name = IndexName::new("documents").expect("valid name");
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();

    // Seed a committed name-directory entry.
    {
        let raw = backend.begin_write().await.expect("begin seed");
        let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);
        txn.insert(
            LogicalKey::IndexNameDirectory(name.clone()),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id(7))),
        )
        .await
        .expect("seed insert");
        txn.commit().await.expect("commit seed");
    }

    // Fill this transaction's mutation budget, so a duplicate insert falls into
    // the budget-exhausted path that must still establish a conflict.
    let raw = backend.begin_write().await.expect("begin writer");
    let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);
    txn.put(
        LogicalKey::IndexIdAllocator,
        PersistentValue::IndexIdAllocator(IndexIdAllocator::new(1)),
    )
    .await
    .expect("fill budget");
    let outcome = txn
        .insert(
            LogicalKey::IndexNameDirectory(name.clone()),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id(8))),
        )
        .await
        .expect("duplicate insert");
    assert_eq!(outcome, InsertOutcome::AlreadyExists);

    // A concurrent overwrite of the same key must conflict with the duplicate
    // insert's update-protected read.
    let raw = backend.begin_write().await.expect("begin concurrent");
    let mut concurrent = WriteLogicalTxn::bootstrap(raw, limits, budget);
    concurrent
        .put(
            LogicalKey::IndexNameDirectory(name),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id(9))),
        )
        .await
        .expect("concurrent put");
    concurrent.commit().await.expect("concurrent commit");

    let error = txn
        .commit()
        .await
        .expect_err("duplicate insert read conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);
}

#[tokio::test]
async fn range_and_cursor_bind_the_tree_key_schema() {
    let backend = DeterministicBackend::default();
    let string_manifest = IndexManifest::new(
        IndexLifecycle::Active,
        id(7),
        IndexConfig::new(1, Metric::L2)
            .expect("valid base")
            .with_fields(vec![
                FieldSchema::new("tenant", DataType::String).expect("valid field"),
            ])
            .expect("valid schema")
            .with_tree_key_fields(vec![FieldId(0)])
            .expect("valid Tree Key fields"),
        [7; 32],
        vec![None],
    )
    .expect("valid manifest");
    let int_manifest = IndexManifest::new(
        IndexLifecycle::Active,
        id(7),
        IndexConfig::new(1, Metric::L2)
            .expect("valid base")
            .with_fields(vec![
                FieldSchema::new("tenant", DataType::I64).expect("valid field"),
            ])
            .expect("valid schema")
            .with_tree_key_fields(vec![FieldId(0)])
            .expect("valid Tree Key fields"),
        [7; 32],
        vec![None],
    )
    .expect("valid manifest");

    let range = LogicalRange::tree_manifests(&string_manifest);
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, &int_manifest).expect("bind int manifest");
    let error = txn
        .scan(
            &range,
            None,
            ScanLimits {
                item_limit: 10,
                byte_limit: usize::MAX,
            },
        )
        .await
        .expect_err("range built under a different schema must be rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}

#[tokio::test]
async fn reading_a_manifest_with_different_immutable_config_fails_closed() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let different = IndexManifest::new(
        IndexLifecycle::Active,
        id(7),
        IndexConfig::new(2, Metric::L2).expect("valid config"),
        [7; 32],
        vec![],
    )
    .expect("valid manifest");

    // Seed a Manifest whose immutable config disagrees with the bound manifest.
    let codec = ValueCodec::bootstrap();
    let bytes = codec
        .encode(&PersistentValue::IndexManifest(different))
        .expect("encode manifest");
    {
        let mut raw = backend.begin_write().await.expect("begin write");
        raw.put(Bytes::from(keys::manifest_key(id(7))), Bytes::from(bytes))
            .await
            .expect("seed manifest");
        raw.commit().await.expect("commit manifest");
    }

    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, &manifest).expect("bind manifest");
    let error = txn
        .get(LogicalKey::Manifest(id(7)))
        .await
        .expect_err("different immutable config must fail closed");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn insert_at_exhausted_budget_for_absent_key_returns_limit_exceeded() {
    let config = DeterministicConfig {
        admission_budget: AdmissionBudget {
            max_mutations: 1,
            max_mutation_bytes: 1_024,
            mutation_key_overhead_bytes: 0,
        },
        ..DeterministicConfig::default()
    };
    let backend = DeterministicBackend::new(config);
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::bootstrap(raw, limits, budget);
    txn.put(
        LogicalKey::IndexIdAllocator,
        PersistentValue::IndexIdAllocator(IndexIdAllocator::new(1)),
    )
    .await
    .expect("fill budget");

    // The budget is exhausted and the key is absent, so this is a new mutation
    // that cannot be admitted — it must surface LimitExceeded, not AlreadyExists.
    let error = txn
        .insert(
            LogicalKey::IndexNameDirectory(IndexName::new("absent").expect("valid name")),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id(7))),
        )
        .await
        .expect_err("absent key at exhausted budget");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    txn.rollback().await;
}

/// Seeds one committed Record value and resets the backend call counters.
async fn seed_record(backend: &DeterministicBackend, record_id: &'static [u8]) {
    let manifest = manifest();
    let limits = backend.hard_limits();
    let budget = backend.admission_budget();
    let raw = backend.begin_write().await.expect("begin seed");
    let mut txn = WriteLogicalTxn::for_index(raw, &manifest, limits, budget).expect("bind index");
    txn.put(record_key(record_id), record(record_id))
        .await
        .expect("seed record");
    txn.commit().await.expect("commit seed");
    backend.reset_operation_counts();
}

async fn index_write_txn<'b, 'm>(
    backend: &'b DeterministicBackend,
    manifest: &'m IndexManifest,
) -> WriteLogicalTxn<'m, DeterministicWriteTxn<'b>> {
    let raw = backend.begin_write().await.expect("begin write");
    WriteLogicalTxn::for_index(
        raw,
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind index")
}

#[tokio::test]
async fn repeat_reads_reuse_the_transaction_cache() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    seed_record(&backend, b"cached").await;
    let mut txn = index_write_txn(&backend, &manifest).await;

    // A plain read followed by another plain read hits the backend once.
    let first = txn.get(record_key(b"cached")).await.expect("first get");
    let second = txn.get(record_key(b"cached")).await.expect("second get");
    assert_eq!(first, second);
    assert_eq!(backend.operation_counts().get, 1);

    // An update-protected read of a key cached by a plain read must go to the
    // backend anyway: the conflict is established only by a native
    // update-protected read. A later update-protected read reuses the cache.
    txn.get_for_update(record_key(b"cached"))
        .await
        .expect("update read");
    txn.get_for_update(record_key(b"cached"))
        .await
        .expect("cached update read");
    assert_eq!(backend.operation_counts().get_for_update, 1);

    // A batch read merges cached keys with one native call for the rest, in
    // input order and with duplicates preserved.
    let values = txn
        .batch_get(vec![
            record_key(b"cached"),
            record_key(b"other"),
            record_key(b"cached"),
        ])
        .await
        .expect("batch get");
    assert_eq!(values, vec![first.clone(), None, first]);
    assert_eq!(backend.operation_counts().batch_get, 1);
    // The update-protected batch reuses the conflict-established entry and
    // fetches only the remaining key.
    txn.batch_get_for_update(vec![record_key(b"cached"), record_key(b"other")])
        .await
        .expect("batch update read");
    assert_eq!(backend.operation_counts().batch_get_for_update, 1);
    txn.rollback().await;
}

#[tokio::test]
async fn cached_update_protected_reads_still_conflict() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    seed_record(&backend, b"cached").await;

    let mut txn = index_write_txn(&backend, &manifest).await;
    // The second read is served from the cache, so the transaction establishes
    // exactly one native conflict on the key.
    txn.get_for_update(record_key(b"cached"))
        .await
        .expect("update read");
    txn.get_for_update(record_key(b"cached"))
        .await
        .expect("cached update read");
    assert_eq!(backend.operation_counts().get_for_update, 1);
    txn.put(record_key(b"other"), record(b"other"))
        .await
        .expect("write something");

    let mut concurrent = index_write_txn(&backend, &manifest).await;
    concurrent
        .put(record_key(b"cached"), record(b"cached"))
        .await
        .expect("concurrent put");
    concurrent.commit().await.expect("concurrent commit");

    let error = txn.commit().await.expect_err("cached read conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);
}

#[tokio::test]
async fn writes_refresh_cached_reads() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    seed_record(&backend, b"cached").await;
    let mut txn = index_write_txn(&backend, &manifest).await;

    assert!(txn.get(record_key(b"cached")).await.expect("get").is_some());
    // A write refreshes the cache entry, so the next read observes this
    // transaction's own write without another backend read.
    txn.put(record_key(b"cached"), record(b"cached"))
        .await
        .expect("overwrite");
    let after_write = txn.get(record_key(b"cached")).await.expect("re-read");
    assert_eq!(after_write, Some(record(b"cached")));
    assert_eq!(backend.operation_counts().get, 1);

    txn.delete(record_key(b"cached")).await.expect("delete");
    assert_eq!(txn.get(record_key(b"cached")).await.expect("re-read"), None);
    assert_eq!(backend.operation_counts().get, 1);
    txn.rollback().await;
}
