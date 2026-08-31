//! Backend-neutral public API contract tests.

use std::error::Error as _;
use std::io;
use std::num::NonZeroU32;
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{
    DataType, Error, ErrorKind, FieldId, FieldSchema, ImportOptions, IndexConfig, IndexName,
    Metric, Mutation, PayloadProjection, Predicate, Record, RuntimeConfig, SearchBudgets,
    SearchHit, SearchOptions, SearchRequest, SynopsisConfig, Value, VerifyOptions,
    validate_mutations,
};

fn assert_invalid<T>(result: ktann::api::Result<T>) {
    match result {
        Ok(_) => panic!("expected InvalidArgument"),
        Err(error) => assert_eq!(error.kind(), ErrorKind::InvalidArgument),
    }
}

#[test]
fn core_crate_is_linkable() {
    use ktann as _;
}

#[test]
fn immutable_configuration_enforces_schema_contract() -> ktann::api::Result<()> {
    assert_invalid(IndexConfig::new(0, Metric::L2));
    assert_invalid(IndexConfig::new(16_385, Metric::L2));

    let duplicate_fields = vec![
        FieldSchema::new("tag", DataType::String)?,
        FieldSchema::new("tag", DataType::String)?,
    ];
    assert_invalid(IndexConfig::new(4, Metric::L2)?.with_fields(duplicate_fields));

    let nullable_tree_key = FieldSchema::new("tenant", DataType::I64)?.nullable();
    let config = IndexConfig::new(4, Metric::L2)?
        .with_fields(vec![nullable_tree_key])?
        .with_tree_key_fields(vec![FieldId(0)])?;
    assert_invalid(config.validate());

    let field = FieldSchema::new("tenant", DataType::I64)?;
    let config = IndexConfig::new(4, Metric::L2)?.with_fields(vec![field])?;
    assert_invalid(
        config
            .clone()
            .with_tree_key_fields(vec![FieldId(0), FieldId(0)]),
    );
    assert_invalid(config.with_partition_entries(65_536, 65_536));

    let tree_keys_first = IndexConfig::new(4, Metric::L2)?
        .with_tree_key_fields(vec![FieldId(0)])?
        .with_fields(vec![FieldSchema::new("tenant", DataType::I64)?])?;
    tree_keys_first.validate()?;
    Ok(())
}

#[test]
fn bloom_configuration_rejects_non_finite_and_unsafe_values() -> ktann::api::Result<()> {
    let expected_distinct = NonZeroU32::new(10).expect("test constant is nonzero");
    for false_positive_rate in [f64::NAN, f64::INFINITY, 0.0, 1.0] {
        assert_invalid(FieldSchema::new("tag", DataType::String)?.with_synopsis(
            SynopsisConfig::MinMaxBloom {
                expected_distinct,
                false_positive_rate,
            },
        ));
    }
    let enormous_bloom =
        FieldSchema::new("tag", DataType::String)?.with_synopsis(SynopsisConfig::MinMaxBloom {
            expected_distinct: NonZeroU32::new(u32::MAX).expect("maximum u32 is nonzero"),
            false_positive_rate: 0.000_001,
        })?;
    assert_invalid(IndexConfig::new(4, Metric::L2)?.with_fields(vec![enormous_bloom]));
    Ok(())
}

#[test]
fn records_validate_finite_vectors_and_typed_fields() -> ktann::api::Result<()> {
    assert_invalid(Record::new(
        Bytes::from_static(b"id"),
        Arc::from([f32::NAN]),
        Vec::<Value>::new(),
    ));

    let schema = vec![FieldSchema::new("score", DataType::F64)?];
    let mut record = Record::new(
        Bytes::from_static(b"id"),
        Arc::from([1.0_f32, 2.0]),
        vec![Value::F64(-0.0)],
    )?;
    record.validate(2, &schema)?;
    let Value::F64(score) = record.fields()[0] else {
        panic!("validated field has the declared type");
    };
    assert_eq!(score.to_bits(), 0.0_f64.to_bits());

    let mut wrong_type = Record::new(
        Bytes::from_static(b"other"),
        Arc::from([1.0_f32, 2.0]),
        vec![Value::Bool(true)],
    )?;
    assert_invalid(wrong_type.validate(2, &schema));
    Ok(())
}

#[test]
fn mutation_batches_reject_duplicate_ids_with_position() -> ktann::api::Result<()> {
    let first = Record::new(
        Bytes::from_static(b"same"),
        Arc::from([1.0_f32]),
        Vec::<Value>::new(),
    )?;
    let mut mutations = vec![
        Mutation::Insert(first),
        Mutation::Delete(Bytes::from_static(b"same")),
    ];
    let error = validate_mutations(&mut mutations, 1, &[]).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(error.position(), Some(1));
    Ok(())
}

#[test]
fn predicates_enforce_shape_and_types() -> ktann::api::Result<()> {
    let fields = vec![
        FieldSchema::new("published", DataType::Bool)?,
        FieldSchema::new("score", DataType::F64)?,
        FieldSchema::new("name", DataType::String)?,
    ];
    let mut compare_null = Predicate::Compare {
        field: FieldId(0),
        op: ktann::api::CompareOp::Eq,
        value: Value::Null,
    };
    assert_invalid(compare_null.validate(&fields));

    let mut wrong_type = Predicate::In {
        field: FieldId(0),
        values: vec![Value::I64(1)],
    };
    assert_invalid(wrong_type.validate(&fields));

    let mut null_in_list = Predicate::In {
        field: FieldId(0),
        values: vec![Value::Bool(true), Value::Null],
    };
    assert_invalid(null_in_list.validate(&fields));

    let mut unknown_field = Predicate::IsNull(FieldId(3));
    assert_invalid(unknown_field.validate(&fields));

    let mut non_finite = Predicate::Compare {
        field: FieldId(1),
        op: ktann::api::CompareOp::Eq,
        value: Value::F64(f64::INFINITY),
    };
    assert_invalid(non_finite.validate(&fields));

    let mut oversized_string = Predicate::Compare {
        field: FieldId(2),
        op: ktann::api::CompareOp::Eq,
        value: Value::String("x".repeat(1_025)),
    };
    assert_invalid(oversized_string.validate(&fields));

    let mut maximum_in = Predicate::In {
        field: FieldId(0),
        values: vec![Value::Bool(false); 1_024],
    };
    maximum_in.validate(&fields)?;
    let mut oversized_in = Predicate::In {
        field: FieldId(0),
        values: vec![Value::Bool(false); 1_025],
    };
    assert_invalid(oversized_in.validate(&fields));

    let mut maximum_nodes = Predicate::And(vec![Predicate::IsNull(FieldId(0)); 1_023]);
    maximum_nodes.validate(&fields)?;
    let mut oversized_nodes = Predicate::And(vec![Predicate::IsNull(FieldId(0)); 1_024]);
    assert_invalid(oversized_nodes.validate(&fields));

    let mut maximum_depth = Predicate::And(vec![]);
    for _ in 0..63 {
        maximum_depth = Predicate::Not(Box::new(maximum_depth));
    }
    maximum_depth.validate(&fields)?;
    let mut too_deep = Predicate::And(vec![]);
    for _ in 0..64 {
        too_deep = Predicate::Not(Box::new(too_deep));
    }
    assert_invalid(too_deep.validate(&fields));

    let mut negative_zero = Predicate::Compare {
        field: FieldId(1),
        op: ktann::api::CompareOp::Eq,
        value: Value::F64(-0.0),
    };
    negative_zero.validate(&fields)?;
    let Predicate::Compare {
        value: Value::F64(canonical_zero),
        ..
    } = negative_zero
    else {
        panic!("comparison shape changed during validation");
    };
    assert_eq!(canonical_zero.to_bits(), 0.0_f64.to_bits());
    Ok(())
}

#[test]
fn search_rejects_invalid_k_dimension_and_budgets() -> ktann::api::Result<()> {
    assert_invalid(SearchRequest::new(Arc::from([1.0_f32]), 0));
    assert_invalid(SearchRequest::new(Arc::from([f32::INFINITY]), 1));
    assert_invalid(SearchOptions::default().with_visited_partitions(16_385));
    assert_invalid(SearchOptions::default().with_leaf_beam_size(0));
    assert_invalid(SearchOptions::default().with_leaf_beam_size(16_385));
    assert_eq!(
        SearchOptions::default()
            .with_leaf_beam_size(8)?
            .leaf_beam_size(),
        Some(8)
    );
    assert_eq!(SearchOptions::default().leaf_beam_size(), None);

    let tight_runtime_budgets = SearchBudgets::new(4_096, 1_024, 65_536, 9)?;
    let mut request = SearchRequest::new(Arc::from([1.0_f32, 2.0]), 10)?;
    assert_invalid(request.validate(2, &[], tight_runtime_budgets));

    for (k, expected) in [(1, 64), (10, 64), (43, 65), (65_536, 65_536)] {
        let mut request = SearchRequest::new(Arc::from([1.0_f32, 2.0]), k)?;
        let budgets = request.validate(2, &[], SearchBudgets::default())?;
        assert_eq!(budgets.exact_rerank_candidates(), expected);
    }
    let runtime_cap = SearchBudgets::new(4_096, 1_024, 65_536, 12)?;
    let mut request = SearchRequest::new(Arc::from([1.0_f32, 2.0]), 10)?;
    assert_eq!(
        request
            .validate(2, &[], runtime_cap)?
            .exact_rerank_candidates(),
        12
    );

    let mut wrong_dimension = SearchRequest::new(Arc::from([1.0_f32]), 1)?;
    assert_invalid(wrong_dimension.validate(2, &[], SearchBudgets::default()));
    assert_invalid(SearchHit::new(Bytes::from_static(b"id"), f64::NAN));
    Ok(())
}

#[test]
fn runtime_and_verify_limits_fail_closed() {
    assert_eq!(RuntimeConfig::default().import_backlog_watermark(), 2);
    assert_invalid(RuntimeConfig::default().with_foreground_operation_limit(0));
    assert_invalid(RuntimeConfig::default().with_foreground_operation_limit(65_537));
    assert_eq!(
        RuntimeConfig::default()
            .with_foreground_operation_limit(7)
            .expect("positive foreground limit")
            .foreground_operation_limit(),
        7
    );
    assert_invalid(RuntimeConfig::default().with_maintenance(2, 1));
    assert_invalid(RuntimeConfig::default().with_maintenance(0, 0));
    // Zero workers disables background maintenance scheduling; the queue
    // capacity bound still applies and must cover the import watermark.
    assert_eq!(
        RuntimeConfig::default()
            .with_maintenance(0, 1_024)
            .expect("zero workers disables background maintenance")
            .maintenance_workers(),
        0
    );
    let unsafe_import_limits = RuntimeConfig::default()
        .with_import_limits(1, 1_025)
        .expect("builder defers cross-setting validation");
    assert_invalid(unsafe_import_limits.validate());
    assert_invalid(ImportOptions::default().with_max_in_flight_batches(0));
    assert_eq!(
        ImportOptions::default()
            .with_max_in_flight_batches(2)
            .expect("positive in-flight override")
            .max_in_flight_batches(),
        Some(2)
    );
    assert_eq!(ImportOptions::default().max_in_flight_batches(), None);
    assert_invalid(RuntimeConfig::default().with_stalled_timeout(Default::default()));
    assert_invalid(VerifyOptions::default().with_issue_limit(10_001));
    assert_invalid(VerifyOptions::default().with_memory_limit_bytes(1_073_741_825));
}

#[test]
fn errors_preserve_sources_without_rendering_them() {
    let secret = "sensitive-record-id";
    let error = Error::with_source(
        ErrorKind::Backend,
        io::Error::other(format!("backend failed for {secret}")),
    );
    assert!(error.source().is_some());
    assert!(!error.to_string().contains(secret));
    assert!(!format!("{error:?}").contains(secret));
}

#[test]
fn sensitive_public_values_have_redacted_debug() -> ktann::api::Result<()> {
    let index_name = IndexName::new("secret-index")?;
    assert!(!format!("{index_name:?}").contains(index_name.as_str()));

    let record = Record::new(
        Bytes::from_static(b"secret-id"),
        Arc::from([42.0_f32]),
        vec![Value::String("secret-field".into())],
    )?
    .with_payload(Bytes::from_static(b"secret-payload"))?;
    let debug = format!("{record:?}");
    for secret in ["secret-id", "42", "secret-field", "secret-payload"] {
        assert!(!debug.contains(secret));
    }
    assert!(!format!("{:?}", Value::String("secret-field".into())).contains("secret-field"));
    assert!(
        !format!(
            "{:?}",
            Predicate::In {
                field: FieldId(0),
                values: vec![Value::String("secret-field".into())],
            }
        )
        .contains("secret-field")
    );
    assert!(
        !format!(
            "{:?}",
            PayloadProjection::Present(Bytes::from_static(b"secret-payload"))
        )
        .contains("secret-payload")
    );
    Ok(())
}

#[test]
fn owned_public_values_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Record>();
    assert_send_sync::<SearchRequest>();
    assert_send_sync::<RuntimeConfig>();
    assert_send_sync::<Error>();
}
