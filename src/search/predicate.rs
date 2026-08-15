//! Schema-compiled exact Filter Predicate evaluation.

use std::cmp::Ordering;

use crate::api::{
    CompareOp, DataType, Error, ErrorKind, FieldSchema, MAX_STRING_BYTES, Predicate, Result, Value,
    typed_order,
};
use crate::storage::values::{BloomParameters, FieldSynopsis, IndexManifest, PartitionSynopsis};

/// A conservative pruning decision for one leaf partition.
#[expect(
    clippy::enum_variant_names,
    reason = "NoMatch, MayMatch, and AllMatch are the documented domain terms"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SynopsisClassification {
    /// No current entry can make the predicate TRUE.
    NoMatch,
    /// Some current entry may make the predicate TRUE.
    MayMatch,
    /// Every current entry makes the predicate TRUE.
    AllMatch,
}

/// A schema-bound Filter Predicate ready for exact Leaf Entry evaluation.
///
/// Compilation validates and owns the caller's bounded expression, so exact
/// evaluation never depends on a backend codec or backend-specific value type.
pub(crate) struct CompiledPredicate {
    expression: CompiledExpression,
    field_count: usize,
    referenced_fields: Box<[CompiledField]>,
}

impl CompiledPredicate {
    /// Validates and compiles one owned public predicate against `fields`.
    pub(crate) fn compile(mut predicate: Predicate, fields: &[FieldSchema]) -> Result<Self> {
        predicate.validate(fields)?;
        let mut referenced = vec![false; fields.len()];
        mark_referenced_fields(&predicate, &mut referenced)?;
        let referenced_fields = fields
            .iter()
            .enumerate()
            .filter(|(index, _)| referenced[*index])
            .map(|(index, field)| CompiledField {
                index,
                data_type: field.data_type(),
                nullable: field.is_nullable(),
            })
            .collect();
        Ok(Self {
            expression: CompiledExpression::compile(predicate)?,
            field_count: fields.len(),
            referenced_fields,
        })
    }

    /// Returns whether exact stored fields satisfy the Filter Predicate.
    ///
    /// SQL `FALSE` and `UNKNOWN` both reject a record. A wrong projection length
    /// or referenced value that disagrees with the compiled schema is
    /// persistent corruption; the canonical decoder owns validation of fields
    /// the expression never observes.
    pub(crate) fn matches(&self, fields: &[Value]) -> Result<bool> {
        Ok(self.evaluate_truth(fields)? == TruthValue::True)
    }

    /// Classifies a leaf using its exact count and conservative synopsis.
    pub(crate) fn classify(
        &self,
        manifest: &IndexManifest,
        synopsis: &PartitionSynopsis,
        entry_count: u32,
    ) -> Result<SynopsisClassification> {
        if manifest.config().fields().len() != self.field_count
            || !synopsis.has_shape_for(manifest)
            || self.referenced_fields.iter().any(|compiled| {
                manifest
                    .config()
                    .fields()
                    .get(compiled.index)
                    .is_none_or(|field| {
                        field.data_type() != compiled.data_type
                            || field.is_nullable() != compiled.nullable
                    })
            })
        {
            return Err(corrupt());
        }
        if entry_count == 0 {
            return Ok(SynopsisClassification::NoMatch);
        }
        if synopsis
            .fields()
            .iter()
            .any(|field| !field.has_null() && field.minimum().is_none())
        {
            return Err(corrupt());
        }

        let possible = self.expression.possible_truths(manifest, synopsis)?;
        if !possible.contains(TruthValue::True) {
            Ok(SynopsisClassification::NoMatch)
        } else if possible == TruthSet::TRUE {
            Ok(SynopsisClassification::AllMatch)
        } else {
            Ok(SynopsisClassification::MayMatch)
        }
    }

    fn evaluate_truth(&self, fields: &[Value]) -> Result<TruthValue> {
        self.validate_stored_fields(fields)?;
        self.expression.evaluate(fields)
    }

    fn validate_stored_fields(&self, fields: &[Value]) -> Result<()> {
        if fields.len() != self.field_count {
            return Err(corrupt());
        }
        for field in &self.referenced_fields {
            let value = fields.get(field.index).ok_or_else(corrupt)?;
            let valid = match value {
                Value::Null => field.nullable,
                Value::Bool(_) => field.data_type == DataType::Bool,
                Value::I64(_) => field.data_type == DataType::I64,
                Value::F64(value) => {
                    field.data_type == DataType::F64
                        && value.is_finite()
                        && (*value != 0.0 || value.to_bits() == 0.0_f64.to_bits())
                }
                Value::String(value) => {
                    field.data_type == DataType::String && value.len() <= MAX_STRING_BYTES
                }
            };
            if !valid {
                return Err(corrupt());
            }
        }
        Ok(())
    }
}

/// The schema facts needed to validate one decoded stored value.
#[derive(Clone, Copy)]
struct CompiledField {
    index: usize,
    data_type: DataType,
    nullable: bool,
}

/// The bounded public AST after schema validation and value canonicalization.
enum CompiledExpression {
    And(Box<[Self]>),
    Or(Box<[Self]>),
    Not(Box<Self>),
    Compare {
        field: usize,
        op: CompareOp,
        value: Value,
    },
    In {
        field: usize,
        values: CompiledIn,
    },
    IsNull(usize),
    IsNotNull(usize),
}

impl CompiledExpression {
    /// Converts an already validated public AST without changing its shape.
    fn compile(predicate: Predicate) -> Result<Self> {
        Ok(match predicate {
            Predicate::And(children) => Self::And(
                children
                    .into_iter()
                    .map(Self::compile)
                    .collect::<Result<Vec<_>>>()?
                    .into_boxed_slice(),
            ),
            Predicate::Or(children) => Self::Or(
                children
                    .into_iter()
                    .map(Self::compile)
                    .collect::<Result<Vec<_>>>()?
                    .into_boxed_slice(),
            ),
            Predicate::Not(child) => Self::Not(Box::new(Self::compile(*child)?)),
            Predicate::Compare { field, op, value } => Self::Compare {
                field: usize::from(field.0),
                op,
                value,
            },
            Predicate::In { field, values } => Self::In {
                field: usize::from(field.0),
                values: CompiledIn::compile(values)?,
            },
            Predicate::IsNull(field) => Self::IsNull(usize::from(field.0)),
            Predicate::IsNotNull(field) => Self::IsNotNull(usize::from(field.0)),
        })
    }

    /// Evaluates one node with the SQL three-valued truth tables.
    fn evaluate(&self, fields: &[Value]) -> Result<TruthValue> {
        match self {
            Self::And(children) => {
                let mut result = TruthValue::True;
                for child in children {
                    match child.evaluate(fields)? {
                        TruthValue::False => return Ok(TruthValue::False),
                        TruthValue::Unknown => result = TruthValue::Unknown,
                        TruthValue::True => {}
                    }
                }
                Ok(result)
            }
            Self::Or(children) => {
                let mut result = TruthValue::False;
                for child in children {
                    match child.evaluate(fields)? {
                        TruthValue::True => return Ok(TruthValue::True),
                        TruthValue::Unknown => result = TruthValue::Unknown,
                        TruthValue::False => {}
                    }
                }
                Ok(result)
            }
            Self::Not(child) => Ok(child.evaluate(fields)?.not()),
            Self::Compare { field, op, value } => {
                let stored = fields.get(*field).ok_or_else(corrupt)?;
                if matches!(stored, Value::Null) {
                    return Ok(TruthValue::Unknown);
                }
                Ok(TruthValue::from_bool(compare(stored, value, *op)?))
            }
            Self::In { field, values } => values.evaluate(fields.get(*field).ok_or_else(corrupt)?),
            Self::IsNull(field) => fields
                .get(*field)
                .map(|value| TruthValue::from_bool(matches!(value, Value::Null)))
                .ok_or_else(corrupt),
            Self::IsNotNull(field) => fields
                .get(*field)
                .map(|value| TruthValue::from_bool(!matches!(value, Value::Null)))
                .ok_or_else(corrupt),
        }
    }

    /// Evaluates this expression over every truth value allowed by a synopsis.
    fn possible_truths(
        &self,
        manifest: &IndexManifest,
        synopsis: &PartitionSynopsis,
    ) -> Result<TruthSet> {
        match self {
            Self::And(children) => children.iter().try_fold(TruthSet::TRUE, |left, child| {
                Ok(left.and(child.possible_truths(manifest, synopsis)?))
            }),
            Self::Or(children) => children.iter().try_fold(TruthSet::FALSE, |left, child| {
                Ok(left.or(child.possible_truths(manifest, synopsis)?))
            }),
            Self::Not(child) => Ok(child.possible_truths(manifest, synopsis)?.not()),
            Self::Compare { field, op, value } => {
                let (field_synopsis, data_type, parameters) =
                    synopsis_field(manifest, synopsis, *field)?;
                compare_truths(field_synopsis, *op, value, data_type, parameters)
            }
            Self::In { field, values } => {
                let (field_synopsis, data_type, parameters) =
                    synopsis_field(manifest, synopsis, *field)?;
                in_truths(field_synopsis, values, data_type, parameters)
            }
            Self::IsNull(field) => {
                let (field, _, _) = synopsis_field(manifest, synopsis, *field)?;
                presence_truths(field.has_null(), field.minimum().is_some())
            }
            Self::IsNotNull(field) => {
                let (field, _, _) = synopsis_field(manifest, synopsis, *field)?;
                presence_truths(field.minimum().is_some(), field.has_null())
            }
        }
    }
}

/// A typed, sorted, deduplicated membership set.
///
/// Sorting once during request compilation bounds each exact `IN` evaluation
/// to logarithmic comparisons without hashing backend-specific encodings.
struct CompiledIn(Box<[Value]>);

impl CompiledIn {
    fn compile(mut values: Vec<Value>) -> Result<Self> {
        let Some(first) = values.first() else {
            return Ok(Self(values.into_boxed_slice()));
        };
        if matches!(first, Value::Null)
            || values
                .iter()
                .skip(1)
                .any(|value| typed_order(first, value).is_none())
        {
            return Err(Error::invalid_argument());
        }
        values.sort_unstable_by(|left, right| {
            typed_order(left, right).expect("validated IN values share one scalar domain")
        });
        values.dedup_by(|left, right| typed_order(left, right) == Some(Ordering::Equal));
        Ok(Self(values.into_boxed_slice()))
    }

    fn evaluate(&self, stored: &Value) -> Result<TruthValue> {
        if self.0.is_empty() {
            return Ok(TruthValue::False);
        }
        if matches!(stored, Value::Null) {
            return Ok(TruthValue::Unknown);
        }
        if typed_order(&self.0[0], stored).is_none() {
            return Err(corrupt());
        }
        let contains = self
            .0
            .binary_search_by(|value| {
                typed_order(value, stored).expect("validated IN values share the stored domain")
            })
            .is_ok();
        Ok(TruthValue::from_bool(contains))
    }
}

/// Marks the schema fields whose persistent values exact evaluation observes.
fn mark_referenced_fields(predicate: &Predicate, referenced: &mut [bool]) -> Result<()> {
    match predicate {
        Predicate::And(children) | Predicate::Or(children) => {
            for child in children {
                mark_referenced_fields(child, referenced)?;
            }
            Ok(())
        }
        Predicate::Not(child) => mark_referenced_fields(child, referenced),
        Predicate::Compare { field, .. }
        | Predicate::In { field, .. }
        | Predicate::IsNull(field)
        | Predicate::IsNotNull(field) => referenced
            .get_mut(usize::from(field.0))
            .map(|is_referenced| *is_referenced = true)
            .ok_or_else(Error::invalid_argument),
    }
}

fn synopsis_field<'a>(
    manifest: &IndexManifest,
    synopsis: &'a PartitionSynopsis,
    index: usize,
) -> Result<(&'a FieldSynopsis, DataType, Option<BloomParameters>)> {
    let field = manifest.config().fields().get(index).ok_or_else(corrupt)?;
    let field_synopsis = synopsis.fields().get(index).ok_or_else(corrupt)?;
    if !field_synopsis.has_null() && field_synopsis.minimum().is_none() {
        return Err(corrupt());
    }
    Ok((
        field_synopsis,
        field.data_type(),
        manifest.bloom_parameters()[index],
    ))
}

fn compare_truths(
    synopsis: &FieldSynopsis,
    op: CompareOp,
    value: &Value,
    data_type: DataType,
    parameters: Option<BloomParameters>,
) -> Result<TruthSet> {
    let mut truths = TruthSet::EMPTY;
    if synopsis.has_null() {
        truths.insert(TruthValue::Unknown);
    }
    let (Some(minimum), Some(maximum)) = (synopsis.minimum(), synopsis.maximum()) else {
        return Ok(truths);
    };
    let minimum_order = typed_order(minimum, value).ok_or_else(corrupt)?;
    let maximum_order = typed_order(maximum, value).ok_or_else(corrupt)?;

    let (true_possible, false_possible) = match op {
        CompareOp::Eq | CompareOp::NotEq => {
            let equality_possible = minimum_order != Ordering::Greater
                && maximum_order != Ordering::Less
                && synopsis.bloom_might_contain(value, data_type, parameters);
            let equality_certain =
                minimum_order == Ordering::Equal && maximum_order == Ordering::Equal;
            if op == CompareOp::Eq {
                (equality_possible, !equality_certain)
            } else {
                (!equality_certain, equality_possible)
            }
        }
        CompareOp::Lt => (
            minimum_order == Ordering::Less,
            maximum_order != Ordering::Less,
        ),
        CompareOp::LessOrEqual => (
            minimum_order != Ordering::Greater,
            maximum_order == Ordering::Greater,
        ),
        CompareOp::Gt => (
            maximum_order == Ordering::Greater,
            minimum_order != Ordering::Greater,
        ),
        CompareOp::GreaterOrEqual => (
            maximum_order != Ordering::Less,
            minimum_order == Ordering::Less,
        ),
    };
    if true_possible {
        truths.insert(TruthValue::True);
    }
    if false_possible {
        truths.insert(TruthValue::False);
    }
    Ok(truths)
}

fn in_truths(
    synopsis: &FieldSynopsis,
    values: &CompiledIn,
    data_type: DataType,
    parameters: Option<BloomParameters>,
) -> Result<TruthSet> {
    if values.0.is_empty() {
        return Ok(TruthSet::FALSE);
    }

    let mut truths = TruthSet::EMPTY;
    if synopsis.has_null() {
        truths.insert(TruthValue::Unknown);
    }
    let (Some(minimum), Some(maximum)) = (synopsis.minimum(), synopsis.maximum()) else {
        return Ok(truths);
    };
    let mut true_possible = false;
    for value in &values.0 {
        let minimum_order = typed_order(minimum, value).ok_or_else(corrupt)?;
        let maximum_order = typed_order(maximum, value).ok_or_else(corrupt)?;
        if minimum_order != Ordering::Greater
            && maximum_order != Ordering::Less
            && synopsis.bloom_might_contain(value, data_type, parameters)
        {
            true_possible = true;
            break;
        }
    }
    if true_possible {
        truths.insert(TruthValue::True);
    }

    let only_value_is_listed = typed_order(minimum, maximum) == Some(Ordering::Equal)
        && values
            .0
            .binary_search_by(|value| {
                typed_order(value, minimum)
                    .expect("compiled IN values match the synopsis scalar domain")
            })
            .is_ok();
    if !only_value_is_listed {
        truths.insert(TruthValue::False);
    }
    Ok(truths)
}

fn presence_truths(true_possible: bool, false_possible: bool) -> Result<TruthSet> {
    let mut truths = TruthSet::EMPTY;
    if true_possible {
        truths.insert(TruthValue::True);
    }
    if false_possible {
        truths.insert(TruthValue::False);
    }
    if truths == TruthSet::EMPTY {
        return Err(corrupt());
    }
    Ok(truths)
}

/// A bit set of SQL truth values possible for entries in one leaf.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TruthSet(u8);

impl TruthSet {
    const EMPTY: Self = Self(0);
    const FALSE: Self = Self(1 << TruthValue::False as u8);
    const TRUE: Self = Self(1 << TruthValue::True as u8);

    fn insert(&mut self, truth: TruthValue) {
        self.0 |= 1 << truth as u8;
    }

    const fn contains(self, truth: TruthValue) -> bool {
        self.0 & (1 << truth as u8) != 0
    }

    fn not(self) -> Self {
        let mut result = Self::EMPTY;
        for truth in [TruthValue::False, TruthValue::True, TruthValue::Unknown] {
            if self.contains(truth) {
                result.insert(truth.not());
            }
        }
        result
    }

    fn and(self, right: Self) -> Self {
        self.map_binary(right, TruthValue::and)
    }

    fn or(self, right: Self) -> Self {
        self.map_binary(right, TruthValue::or)
    }

    fn map_binary(
        self,
        right_set: Self,
        operation: fn(TruthValue, TruthValue) -> TruthValue,
    ) -> Self {
        const VALUES: [TruthValue; 3] = [TruthValue::False, TruthValue::True, TruthValue::Unknown];
        let mut result = Self::EMPTY;
        for left in VALUES.into_iter().filter(|truth| self.contains(*truth)) {
            for right in VALUES
                .into_iter()
                .filter(|truth| right_set.contains(*truth))
            {
                result.insert(operation(left, right));
            }
        }
        result
    }
}

/// One exact SQL truth value used internally during expression evaluation.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TruthValue {
    False,
    True,
    Unknown,
}

impl TruthValue {
    const fn from_bool(value: bool) -> Self {
        if value { Self::True } else { Self::False }
    }

    const fn not(self) -> Self {
        match self {
            Self::False => Self::True,
            Self::True => Self::False,
            Self::Unknown => Self::Unknown,
        }
    }

    const fn and(self, right: Self) -> Self {
        match (self, right) {
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::True, Self::True) => Self::True,
        }
    }

    const fn or(self, right: Self) -> Self {
        match (self, right) {
            (Self::True, _) | (_, Self::True) => Self::True,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::False, Self::False) => Self::False,
        }
    }
}

/// Applies one typed comparison after NULL handling.
fn compare(left: &Value, right: &Value, op: CompareOp) -> Result<bool> {
    let ordering = stored_order(left, right)?;
    Ok(match op {
        CompareOp::Eq => ordering == Ordering::Equal,
        CompareOp::NotEq => ordering != Ordering::Equal,
        CompareOp::Lt => ordering == Ordering::Less,
        CompareOp::LessOrEqual => ordering != Ordering::Greater,
        CompareOp::Gt => ordering == Ordering::Greater,
        CompareOp::GreaterOrEqual => ordering != Ordering::Less,
    })
}

/// Orders two same-domain non-NULL values using the persistent scalar contract.
fn stored_order(left: &Value, right: &Value) -> Result<Ordering> {
    typed_order(left, right).ok_or_else(corrupt)
}

const fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use proptest::prelude::*;

    use crate::api::{
        CompareOp, DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, LogicalIndexId, Metric,
        Predicate, SynopsisConfig, Value,
    };
    use crate::storage::values::{
        BloomParameters, IndexLifecycle, IndexManifest, PartitionSynopsis,
    };

    use super::{CompiledPredicate, SynopsisClassification, TruthValue};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum OracleTruth {
        False,
        True,
        Unknown,
    }

    #[test]
    fn empty_expressions_and_nulls_follow_sql_where_semantics() {
        let schema = vec![FieldSchema::new("value", DataType::I64).unwrap().nullable()];
        let cases = [
            (Predicate::And(vec![]), Value::Null, TruthValue::True),
            (Predicate::Or(vec![]), Value::Null, TruthValue::False),
            (
                Predicate::In {
                    field: FieldId(0),
                    values: vec![],
                },
                Value::Null,
                TruthValue::False,
            ),
            (
                Predicate::Compare {
                    field: FieldId(0),
                    op: CompareOp::Eq,
                    value: Value::I64(0),
                },
                Value::Null,
                TruthValue::Unknown,
            ),
            (
                Predicate::Not(Box::new(Predicate::Compare {
                    field: FieldId(0),
                    op: CompareOp::Eq,
                    value: Value::I64(0),
                })),
                Value::Null,
                TruthValue::Unknown,
            ),
        ];

        for (predicate, value, expected) in cases {
            let compiled = CompiledPredicate::compile(predicate, &schema).unwrap();
            assert_eq!(compiled.evaluate_truth(&[value]).unwrap(), expected);
        }
    }

    #[test]
    fn malformed_stored_fields_fail_closed() {
        let bool_schema = vec![FieldSchema::new("flag", DataType::Bool).unwrap()];
        let compiled =
            CompiledPredicate::compile(Predicate::IsNotNull(FieldId(0)), &bool_schema).unwrap();
        for fields in [vec![], vec![Value::Null], vec![Value::I64(0)]] {
            assert_eq!(
                compiled.matches(&fields).unwrap_err().kind(),
                ErrorKind::Corruption
            );
        }

        let f64_schema = vec![FieldSchema::new("score", DataType::F64).unwrap()];
        let compiled =
            CompiledPredicate::compile(Predicate::IsNotNull(FieldId(0)), &f64_schema).unwrap();
        for value in [Value::F64(f64::NAN), Value::F64(-0.0)] {
            assert_eq!(
                compiled.matches(&[value]).unwrap_err().kind(),
                ErrorKind::Corruption
            );
        }

        let string_schema = vec![FieldSchema::new("name", DataType::String).unwrap()];
        let compiled =
            CompiledPredicate::compile(Predicate::IsNotNull(FieldId(0)), &string_schema).unwrap();
        assert_eq!(
            compiled
                .matches(&[Value::String("x".repeat(1_025))])
                .unwrap_err()
                .kind(),
            ErrorKind::Corruption
        );
    }

    proptest! {
        #[test]
        fn compiled_evaluation_matches_three_valued_oracle(
            predicate in predicate_strategy(),
            fields in stored_fields_strategy(),
        ) {
            let schema = test_schema();
            let expected = oracle_evaluate(&predicate, &fields);
            let compiled = CompiledPredicate::compile(predicate, &schema).unwrap();

            prop_assert_eq!(compiled.evaluate_truth(&fields).unwrap(), to_compiled_truth(expected));
            prop_assert_eq!(compiled.matches(&fields).unwrap(), expected == OracleTruth::True);
        }

        #[test]
        fn historical_synopsis_classification_is_sound_after_deletes_and_moves(
            predicate in predicate_strategy(),
            entries in prop::collection::vec(stored_fields_strategy(), 0..12),
            first_cut in 0_usize..12,
            second_cut in 0_usize..12,
        ) {
            let schema = test_bloom_schema();
            let manifest = test_manifest(schema.clone());
            let compiled = CompiledPredicate::compile(predicate, &schema).unwrap();
            let first = first_cut.min(second_cut).min(entries.len());
            let second = first_cut.max(second_cut).min(entries.len());
            let target_initial = &entries[..first];
            let moved = &entries[first..second];
            let retained = &entries[second..];

            let mut source = PartitionSynopsis::empty(&manifest);
            for fields in &entries[first..] {
                source.expand(&manifest, fields).unwrap();
            }
            assert_classification_is_sound(
                compiled.classify(&manifest, &source, retained.len() as u32).unwrap(),
                &compiled,
                retained,
            );

            let mut target = PartitionSynopsis::empty(&manifest);
            for fields in target_initial {
                target.expand(&manifest, fields).unwrap();
            }
            for fields in moved {
                target.expand(&manifest, fields).unwrap();
            }
            let mut rebuilt_target = PartitionSynopsis::empty(&manifest);
            for fields in entries[..second].iter().rev() {
                rebuilt_target.expand(&manifest, fields).unwrap();
            }
            prop_assert_eq!(&target, &rebuilt_target);
            assert_classification_is_sound(
                compiled.classify(&manifest, &target, second as u32).unwrap(),
                &compiled,
                &entries[..second],
            );
        }
    }

    #[test]
    fn bloom_misses_prune_but_saturation_stays_conservative() {
        let expected_distinct = NonZeroU32::new(2).unwrap();
        let field = FieldSchema::new("value", DataType::I64)
            .unwrap()
            .with_synopsis(SynopsisConfig::MinMaxBloom {
                expected_distinct,
                false_positive_rate: 0.01,
            })
            .unwrap();
        let schema = vec![field];
        let roomy_manifest = test_manifest(schema.clone());
        let mut synopsis = PartitionSynopsis::empty(&roomy_manifest);
        synopsis.expand(&roomy_manifest, &[Value::I64(0)]).unwrap();
        synopsis
            .expand(&roomy_manifest, &[Value::I64(1_000)])
            .unwrap();
        let absent = (1..1_000)
            .map(Value::I64)
            .find(|value| {
                !synopsis.fields()[0].bloom_might_contain(
                    value,
                    DataType::I64,
                    roomy_manifest.bloom_parameters()[0],
                )
            })
            .expect("large Bloom has a definite miss in the stored range");
        let predicate = Predicate::Compare {
            field: FieldId(0),
            op: CompareOp::Eq,
            value: absent.clone(),
        };
        assert_eq!(
            CompiledPredicate::compile(predicate.clone(), &schema)
                .unwrap()
                .classify(&roomy_manifest, &synopsis, 2)
                .unwrap(),
            SynopsisClassification::NoMatch
        );

        let saturated_schema = vec![
            FieldSchema::new("value", DataType::I64)
                .unwrap()
                .with_synopsis(SynopsisConfig::MinMaxBloom {
                    expected_distinct: NonZeroU32::new(1).unwrap(),
                    false_positive_rate: 0.9,
                })
                .unwrap(),
        ];
        let saturated_manifest = test_manifest(saturated_schema);
        let mut saturated = PartitionSynopsis::empty(&saturated_manifest);
        saturated
            .expand(&saturated_manifest, &[Value::I64(0)])
            .unwrap();
        saturated
            .expand(&saturated_manifest, &[Value::I64(1_000)])
            .unwrap();
        let mut entry_count = 2;
        for candidate in 1..1_000 {
            if saturated.fields()[0]
                .bloom()
                .is_some_and(|bytes| bytes[0] & 0b11 == 0b11)
            {
                break;
            }
            if Value::I64(candidate) != absent {
                saturated
                    .expand(&saturated_manifest, &[Value::I64(candidate)])
                    .unwrap();
                entry_count += 1;
            }
        }
        assert_eq!(
            saturated.fields()[0].bloom().map(bytes::Bytes::as_ref),
            Some(&[0b11][..])
        );
        assert_eq!(
            CompiledPredicate::compile(predicate, &schema)
                .unwrap()
                .classify(&saturated_manifest, &saturated, entry_count)
                .unwrap(),
            SynopsisClassification::MayMatch
        );
    }

    #[test]
    fn null_truth_sets_and_not_produce_safe_classifications() {
        let schema = vec![FieldSchema::new("value", DataType::I64).unwrap().nullable()];
        let manifest = test_manifest(schema.clone());
        let mut synopsis = PartitionSynopsis::empty(&manifest);
        synopsis.expand(&manifest, &[Value::Null]).unwrap();
        synopsis.expand(&manifest, &[Value::I64(5)]).unwrap();

        let equals_five = Predicate::Compare {
            field: FieldId(0),
            op: CompareOp::Eq,
            value: Value::I64(5),
        };
        assert_eq!(
            CompiledPredicate::compile(equals_five.clone(), &schema)
                .unwrap()
                .classify(&manifest, &synopsis, 2)
                .unwrap(),
            SynopsisClassification::MayMatch
        );
        assert_eq!(
            CompiledPredicate::compile(Predicate::Not(Box::new(equals_five.clone())), &schema)
                .unwrap()
                .classify(&manifest, &synopsis, 2)
                .unwrap(),
            SynopsisClassification::NoMatch
        );
        assert_eq!(
            CompiledPredicate::compile(
                Predicate::Or(vec![Predicate::IsNull(FieldId(0)), equals_five]),
                &schema,
            )
            .unwrap()
            .classify(&manifest, &synopsis, 2)
            .unwrap(),
            SynopsisClassification::MayMatch
        );

        let mut only_five = PartitionSynopsis::empty(&manifest);
        only_five.expand(&manifest, &[Value::I64(5)]).unwrap();
        assert_eq!(
            CompiledPredicate::compile(
                Predicate::Compare {
                    field: FieldId(0),
                    op: CompareOp::Eq,
                    value: Value::I64(5),
                },
                &schema,
            )
            .unwrap()
            .classify(&manifest, &only_five, 1)
            .unwrap(),
            SynopsisClassification::AllMatch
        );
        assert_eq!(
            CompiledPredicate::compile(Predicate::IsNotNull(FieldId(0)), &schema)
                .unwrap()
                .classify(&manifest, &synopsis, 0)
                .unwrap(),
            SynopsisClassification::NoMatch
        );
    }

    fn test_schema() -> Vec<FieldSchema> {
        vec![
            FieldSchema::new("bool", DataType::Bool).unwrap().nullable(),
            FieldSchema::new("i64", DataType::I64).unwrap().nullable(),
            FieldSchema::new("f64", DataType::F64).unwrap().nullable(),
            FieldSchema::new("string", DataType::String)
                .unwrap()
                .nullable(),
        ]
    }

    fn test_bloom_schema() -> Vec<FieldSchema> {
        test_schema()
            .into_iter()
            .map(|field| {
                field
                    .with_synopsis(SynopsisConfig::MinMaxBloom {
                        expected_distinct: NonZeroU32::new(16).unwrap(),
                        false_positive_rate: 0.01,
                    })
                    .unwrap()
            })
            .collect()
    }

    fn test_manifest(fields: Vec<FieldSchema>) -> IndexManifest {
        let bloom_parameters = fields
            .iter()
            .map(|field| BloomParameters::derive(field.synopsis()).unwrap())
            .collect();
        let config = IndexConfig::new(1, Metric::L2)
            .unwrap()
            .with_fields(fields)
            .unwrap();
        IndexManifest::new(
            IndexLifecycle::Active,
            LogicalIndexId::new(1).unwrap(),
            config,
            [0; 32],
            bloom_parameters,
        )
        .unwrap()
    }

    fn assert_classification_is_sound(
        classification: SynopsisClassification,
        compiled: &CompiledPredicate,
        entries: &[Vec<Value>],
    ) {
        let truths = entries
            .iter()
            .map(|fields| compiled.evaluate_truth(fields).unwrap())
            .collect::<Vec<_>>();
        match classification {
            SynopsisClassification::NoMatch => {
                assert!(!truths.contains(&TruthValue::True));
            }
            SynopsisClassification::AllMatch => {
                assert!(!truths.is_empty());
                assert!(truths.iter().all(|truth| *truth == TruthValue::True));
            }
            SynopsisClassification::MayMatch => {}
        }
    }

    fn stored_fields_strategy() -> impl Strategy<Value = Vec<Value>> {
        (
            nullable(bool_value()),
            nullable(i64_value()),
            nullable(f64_value()),
            nullable(string_value()),
        )
            .prop_map(|(bool_value, i64_value, f64_value, string_value)| {
                vec![bool_value, i64_value, f64_value, string_value]
            })
    }

    fn predicate_strategy() -> impl Strategy<Value = Predicate> {
        let comparisons = prop_oneof![
            (compare_op(), bool_value()).prop_map(|(op, value)| Predicate::Compare {
                field: FieldId(0),
                op,
                value,
            }),
            (compare_op(), i64_value()).prop_map(|(op, value)| Predicate::Compare {
                field: FieldId(1),
                op,
                value,
            }),
            (compare_op(), f64_value()).prop_map(|(op, value)| Predicate::Compare {
                field: FieldId(2),
                op,
                value,
            }),
            (compare_op(), string_value()).prop_map(|(op, value)| Predicate::Compare {
                field: FieldId(3),
                op,
                value,
            }),
        ];
        let membership = prop_oneof![
            prop::collection::vec(bool_value(), 0..=4).prop_map(|values| Predicate::In {
                field: FieldId(0),
                values,
            }),
            prop::collection::vec(i64_value(), 0..=4).prop_map(|values| Predicate::In {
                field: FieldId(1),
                values,
            }),
            prop::collection::vec(f64_value(), 0..=4).prop_map(|values| Predicate::In {
                field: FieldId(2),
                values,
            }),
            prop::collection::vec(string_value(), 0..=4).prop_map(|values| Predicate::In {
                field: FieldId(3),
                values,
            }),
        ];
        let null_checks = (0_u16..4, any::<bool>()).prop_map(|(field, is_null)| {
            if is_null {
                Predicate::IsNull(FieldId(field))
            } else {
                Predicate::IsNotNull(FieldId(field))
            }
        });

        prop_oneof![comparisons, membership, null_checks].prop_recursive(4, 64, 4, |inner| {
            prop_oneof![
                inner
                    .clone()
                    .prop_map(|child| Predicate::Not(Box::new(child))),
                prop::collection::vec(inner.clone(), 0..=4).prop_map(Predicate::And),
                prop::collection::vec(inner, 0..=4).prop_map(Predicate::Or),
            ]
        })
    }

    fn compare_op() -> impl Strategy<Value = CompareOp> {
        prop::sample::select(vec![
            CompareOp::Eq,
            CompareOp::NotEq,
            CompareOp::Lt,
            CompareOp::LessOrEqual,
            CompareOp::Gt,
            CompareOp::GreaterOrEqual,
        ])
    }

    fn nullable(value: impl Strategy<Value = Value>) -> impl Strategy<Value = Value> {
        prop_oneof![Just(Value::Null), value]
    }

    fn bool_value() -> impl Strategy<Value = Value> {
        any::<bool>().prop_map(Value::Bool)
    }

    fn i64_value() -> impl Strategy<Value = Value> {
        prop::sample::select(vec![i64::MIN, -1, 0, 1, i64::MAX]).prop_map(Value::I64)
    }

    fn f64_value() -> impl Strategy<Value = Value> {
        prop::sample::select(vec![
            f64::MIN,
            -1.0,
            -f64::MIN_POSITIVE,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            f64::MAX,
        ])
        .prop_map(Value::F64)
    }

    fn string_value() -> impl Strategy<Value = Value> {
        prop::sample::select(vec![
            String::new(),
            "a".to_owned(),
            "aa".to_owned(),
            "z".to_owned(),
            "é".to_owned(),
            "x".repeat(1_024),
        ])
        .prop_map(Value::String)
    }

    fn oracle_evaluate(predicate: &Predicate, fields: &[Value]) -> OracleTruth {
        match predicate {
            Predicate::And(children) => children.iter().fold(OracleTruth::True, |left, child| {
                oracle_and(left, oracle_evaluate(child, fields))
            }),
            Predicate::Or(children) => children.iter().fold(OracleTruth::False, |left, child| {
                oracle_or(left, oracle_evaluate(child, fields))
            }),
            Predicate::Not(child) => oracle_not(oracle_evaluate(child, fields)),
            Predicate::Compare { field, op, value } => {
                oracle_compare(&fields[usize::from(field.0)], value, *op)
            }
            Predicate::In { values, .. } if values.is_empty() => OracleTruth::False,
            Predicate::In { field, values } => {
                let stored = &fields[usize::from(field.0)];
                if matches!(stored, Value::Null) {
                    OracleTruth::Unknown
                } else if values
                    .iter()
                    .any(|value| oracle_compare(stored, value, CompareOp::Eq) == OracleTruth::True)
                {
                    OracleTruth::True
                } else {
                    OracleTruth::False
                }
            }
            Predicate::IsNull(field) => {
                OracleTruth::from_bool(matches!(fields[usize::from(field.0)], Value::Null))
            }
            Predicate::IsNotNull(field) => {
                OracleTruth::from_bool(!matches!(fields[usize::from(field.0)], Value::Null))
            }
        }
    }

    fn oracle_compare(left: &Value, right: &Value, op: CompareOp) -> OracleTruth {
        if matches!(left, Value::Null) {
            return OracleTruth::Unknown;
        }
        let ordering = match (left, right) {
            (Value::Bool(left), Value::Bool(right)) => left.cmp(right),
            (Value::I64(left), Value::I64(right)) => left.cmp(right),
            (Value::F64(left), Value::F64(right)) => left.partial_cmp(right).unwrap(),
            (Value::String(left), Value::String(right)) => left.as_bytes().cmp(right.as_bytes()),
            _ => panic!("oracle received values from different domains"),
        };
        OracleTruth::from_bool(match op {
            CompareOp::Eq => ordering == std::cmp::Ordering::Equal,
            CompareOp::NotEq => ordering != std::cmp::Ordering::Equal,
            CompareOp::Lt => ordering == std::cmp::Ordering::Less,
            CompareOp::LessOrEqual => ordering != std::cmp::Ordering::Greater,
            CompareOp::Gt => ordering == std::cmp::Ordering::Greater,
            CompareOp::GreaterOrEqual => ordering != std::cmp::Ordering::Less,
        })
    }

    const fn oracle_and(left: OracleTruth, right: OracleTruth) -> OracleTruth {
        match (left, right) {
            (OracleTruth::False, _) | (_, OracleTruth::False) => OracleTruth::False,
            (OracleTruth::True, OracleTruth::True) => OracleTruth::True,
            _ => OracleTruth::Unknown,
        }
    }

    const fn oracle_or(left: OracleTruth, right: OracleTruth) -> OracleTruth {
        match (left, right) {
            (OracleTruth::True, _) | (_, OracleTruth::True) => OracleTruth::True,
            (OracleTruth::False, OracleTruth::False) => OracleTruth::False,
            _ => OracleTruth::Unknown,
        }
    }

    const fn oracle_not(value: OracleTruth) -> OracleTruth {
        match value {
            OracleTruth::False => OracleTruth::True,
            OracleTruth::True => OracleTruth::False,
            OracleTruth::Unknown => OracleTruth::Unknown,
        }
    }

    const fn to_compiled_truth(value: OracleTruth) -> TruthValue {
        match value {
            OracleTruth::False => TruthValue::False,
            OracleTruth::True => TruthValue::True,
            OracleTruth::Unknown => TruthValue::Unknown,
        }
    }

    impl OracleTruth {
        const fn from_bool(value: bool) -> Self {
            if value { Self::True } else { Self::False }
        }
    }
}
