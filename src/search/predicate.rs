//! Schema-compiled exact Filter Predicate evaluation.

use std::cmp::Ordering;

use crate::api::{
    CompareOp, DataType, Error, ErrorKind, FieldSchema, MAX_STRING_BYTES, Predicate, Result, Value,
};

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
}

/// A typed, sorted, deduplicated membership set.
///
/// Sorting once during request compilation bounds each exact `IN` evaluation
/// to logarithmic comparisons without hashing backend-specific encodings.
enum CompiledIn {
    Empty,
    Bool(Box<[bool]>),
    I64(Box<[i64]>),
    F64(Box<[f64]>),
    String(Box<[String]>),
}

impl CompiledIn {
    fn compile(values: Vec<Value>) -> Result<Self> {
        let Some(first) = values.first() else {
            return Ok(Self::Empty);
        };
        match first {
            Value::Bool(_) => {
                let mut values = collect_values(values, |value| match value {
                    Value::Bool(value) => Some(value),
                    _ => None,
                })?;
                values.sort_unstable();
                values.dedup();
                Ok(Self::Bool(values.into_boxed_slice()))
            }
            Value::I64(_) => {
                let mut values = collect_values(values, |value| match value {
                    Value::I64(value) => Some(value),
                    _ => None,
                })?;
                values.sort_unstable();
                values.dedup();
                Ok(Self::I64(values.into_boxed_slice()))
            }
            Value::F64(_) => {
                let mut values = collect_values(values, |value| match value {
                    Value::F64(value) => Some(value),
                    _ => None,
                })?;
                values.sort_by(f64::total_cmp);
                values.dedup_by(|left, right| left.total_cmp(right) == Ordering::Equal);
                Ok(Self::F64(values.into_boxed_slice()))
            }
            Value::String(_) => {
                let mut values = collect_values(values, |value| match value {
                    Value::String(value) => Some(value),
                    _ => None,
                })?;
                values.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
                values.dedup_by(|left, right| left.as_bytes() == right.as_bytes());
                Ok(Self::String(values.into_boxed_slice()))
            }
            Value::Null => Err(Error::invalid_argument()),
        }
    }

    fn evaluate(&self, stored: &Value) -> Result<TruthValue> {
        if matches!(self, Self::Empty) {
            return Ok(TruthValue::False);
        }
        if matches!(stored, Value::Null) {
            return Ok(TruthValue::Unknown);
        }
        let contains = match (self, stored) {
            (Self::Bool(values), Value::Bool(stored)) => values.binary_search(stored).is_ok(),
            (Self::I64(values), Value::I64(stored)) => values.binary_search(stored).is_ok(),
            (Self::F64(values), Value::F64(stored)) => values
                .binary_search_by(|value| value.total_cmp(stored))
                .is_ok(),
            (Self::String(values), Value::String(stored)) => values
                .binary_search_by(|value| value.as_bytes().cmp(stored.as_bytes()))
                .is_ok(),
            _ => return Err(corrupt()),
        };
        Ok(TruthValue::from_bool(contains))
    }
}

/// Moves one validated `IN` list into its concrete scalar representation.
fn collect_values<T>(values: Vec<Value>, convert: impl Fn(Value) -> Option<T>) -> Result<Vec<T>> {
    values
        .into_iter()
        .map(|value| convert(value).ok_or_else(Error::invalid_argument))
        .collect()
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

/// One exact SQL truth value used internally during expression evaluation.
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
}

/// Applies one typed comparison after NULL handling.
fn compare(left: &Value, right: &Value, op: CompareOp) -> Result<bool> {
    let ordering = typed_order(left, right)?;
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
fn typed_order(left: &Value, right: &Value) -> Result<Ordering> {
    match (left, right) {
        (Value::Bool(left), Value::Bool(right)) => Ok(left.cmp(right)),
        (Value::I64(left), Value::I64(right)) => Ok(left.cmp(right)),
        (Value::F64(left), Value::F64(right)) => left.partial_cmp(right).ok_or_else(corrupt),
        (Value::String(left), Value::String(right)) => Ok(left.as_bytes().cmp(right.as_bytes())),
        _ => Err(corrupt()),
    }
}

const fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::api::{CompareOp, DataType, ErrorKind, FieldId, FieldSchema, Predicate, Value};

    use super::{CompiledExpression, CompiledIn, CompiledPredicate, TruthValue};

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

    #[test]
    fn in_compilation_sorts_and_deduplicates_typed_values() {
        let schema = vec![FieldSchema::new("value", DataType::I64).unwrap()];
        let compiled = CompiledPredicate::compile(
            Predicate::In {
                field: FieldId(0),
                values: vec![Value::I64(2), Value::I64(-1), Value::I64(2), Value::I64(0)],
            },
            &schema,
        )
        .unwrap();

        let CompiledExpression::In {
            values: CompiledIn::I64(values),
            ..
        } = &compiled.expression
        else {
            panic!("expected compiled i64 membership set");
        };
        assert_eq!(values.as_ref(), &[-1, 0, 2]);
        assert!(compiled.matches(&[Value::I64(2)]).unwrap());
        assert!(!compiled.matches(&[Value::I64(1)]).unwrap());
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
