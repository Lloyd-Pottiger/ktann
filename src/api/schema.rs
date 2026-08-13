//! Vector Record schema and Filter Predicate values.

use std::fmt;
use std::num::NonZeroU32;

use super::{Error, FieldId, Result};

pub(crate) const MAX_FIELDS: usize = 16;
pub(crate) const MAX_STRING_BYTES: usize = 1_024;
const MAX_FIELD_NAME_BYTES: usize = 255;
const MAX_PREDICATE_NODES: usize = 1_024;
const MAX_PREDICATE_DEPTH: usize = 64;
const MAX_IN_VALUES: usize = 1_024;
pub(crate) const MAX_ENCODED_SYNOPSIS_BYTES: usize = 64 * 1_024;

/// The exact distance metric used by one Logical Index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Cosine distance.
    Cosine,
    /// Negative inner product, where a smaller value ranks first.
    InnerProduct,
}

/// A typed field's value domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DataType {
    /// Boolean values.
    Bool,
    /// Signed 64-bit integer values.
    I64,
    /// Finite 64-bit floating-point values.
    F64,
    /// UTF-8 strings of at most 1 KiB.
    String,
}

/// The conservative Partition Synopsis maintained for a field.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SynopsisConfig {
    /// Exact NULL-presence flags and non-NULL minimum/maximum values.
    MinMax,
    /// `MinMax` plus a fixed-size Bloom summary for equality and `IN` pruning.
    MinMaxBloom {
        /// Expected number of distinct non-NULL values in one partition.
        expected_distinct: NonZeroU32,
        /// Desired Bloom false-positive rate in the open interval `(0, 1)`.
        false_positive_rate: f64,
    },
}

impl SynopsisConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::MinMax => Ok(()),
            Self::MinMaxBloom {
                false_positive_rate,
                ..
            } if false_positive_rate.is_finite()
                && *false_positive_rate > 0.0
                && *false_positive_rate < 1.0 =>
            {
                Ok(())
            }
            Self::MinMaxBloom { .. } => Err(Error::invalid_argument()),
        }
    }

    pub(crate) fn bloom_bytes(&self) -> Result<usize> {
        let Self::MinMaxBloom {
            expected_distinct,
            false_positive_rate,
        } = self
        else {
            return Ok(0);
        };
        self.validate()?;
        let bits = -(f64::from(expected_distinct.get()) * false_positive_rate.ln())
            / std::f64::consts::LN_2.powi(2);
        if !bits.is_finite() || bits > (usize::MAX as f64) {
            return Err(Error::invalid_argument());
        }
        let bits = bits.ceil() as usize;
        bits.checked_add(7)
            .map(|rounded| rounded / 8)
            .ok_or_else(Error::invalid_argument)
    }
}

/// A typed Vector Record field value.
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub enum Value {
    /// SQL `NULL`.
    Null,
    /// A Boolean value.
    Bool(bool),
    /// A signed 64-bit integer value.
    I64(i64),
    /// A finite 64-bit floating-point value.
    F64(f64),
    /// An unnormalized UTF-8 string.
    String(String),
}

impl fmt::Debug for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Null => "Null",
            Self::Bool(_) => "Bool([REDACTED])",
            Self::I64(_) => "I64([REDACTED])",
            Self::F64(_) => "F64([REDACTED])",
            Self::String(_) => "String([REDACTED])",
        })
    }
}

impl Value {
    /// Constructs a finite floating-point value and canonicalizes negative zero.
    pub fn f64(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::invalid_argument());
        }
        Ok(Self::F64(if value == 0.0 { 0.0 } else { value }))
    }

    /// Constructs a string value within the v1 size limit.
    pub fn string(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.len() > MAX_STRING_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(Self::String(value))
    }

    pub(crate) fn validate_for(&mut self, data_type: DataType, nullable: bool) -> Result<()> {
        match self {
            Self::Null if nullable => Ok(()),
            Self::Null => Err(Error::invalid_argument()),
            Self::Bool(_) if data_type == DataType::Bool => Ok(()),
            Self::I64(_) if data_type == DataType::I64 => Ok(()),
            Self::F64(value) if data_type == DataType::F64 && value.is_finite() => {
                if *value == 0.0 {
                    *value = 0.0;
                }
                Ok(())
            }
            Self::String(value)
                if data_type == DataType::String && value.len() <= MAX_STRING_BYTES =>
            {
                Ok(())
            }
            _ => Err(Error::invalid_argument()),
        }
    }

    const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

/// The schema of one positional Vector Record field.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FieldSchema {
    name: String,
    data_type: DataType,
    nullable: bool,
    synopsis: SynopsisConfig,
}

impl FieldSchema {
    /// Creates a non-null field with a `MinMax` Partition Synopsis.
    pub fn new(name: impl Into<String>, data_type: DataType) -> Result<Self> {
        let name = name.into();
        if name.is_empty() || name.len() > MAX_FIELD_NAME_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            name,
            data_type,
            nullable: false,
            synopsis: SynopsisConfig::MinMax,
        })
    }

    /// Allows this field to contain `NULL`.
    #[must_use]
    pub const fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }

    /// Configures the Partition Synopsis for this field.
    pub fn with_synopsis(mut self, synopsis: SynopsisConfig) -> Result<Self> {
        synopsis.validate()?;
        self.synopsis = synopsis;
        Ok(self)
    }

    /// Returns the original, unnormalized field name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the field's value domain.
    #[must_use]
    pub const fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns whether the field accepts `NULL`.
    #[must_use]
    pub const fn is_nullable(&self) -> bool {
        self.nullable
    }

    /// Returns the field's Partition Synopsis configuration.
    #[must_use]
    pub const fn synopsis(&self) -> &SynopsisConfig {
        &self.synopsis
    }
}

/// A typed comparison operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompareOp {
    /// Equal.
    Eq,
    /// Not equal.
    NotEq,
    /// Less than.
    Lt,
    /// Less than or equal.
    LessOrEqual,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    GreaterOrEqual,
}

/// An exact typed Filter Predicate.
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub enum Predicate {
    /// Logical conjunction. An empty conjunction is true.
    And(Vec<Predicate>),
    /// Logical disjunction. An empty disjunction is false.
    Or(Vec<Predicate>),
    /// Logical negation using SQL three-valued logic.
    Not(Box<Predicate>),
    /// Compares one field with a non-NULL typed value.
    Compare {
        /// Field position.
        field: FieldId,
        /// Comparison operator.
        op: CompareOp,
        /// Non-NULL comparison value.
        value: Value,
    },
    /// Tests membership in a list of non-NULL typed values.
    In {
        /// Field position.
        field: FieldId,
        /// Non-NULL values. An empty list is false.
        values: Vec<Value>,
    },
    /// Tests whether a field is `NULL`.
    IsNull(FieldId),
    /// Tests whether a field is not `NULL`.
    IsNotNull(FieldId),
}

impl fmt::Debug for Predicate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Predicate([REDACTED])")
    }
}

impl Predicate {
    /// Validates structure, field references, and typed values against a schema.
    pub fn validate(&mut self, fields: &[FieldSchema]) -> Result<()> {
        let mut nodes = 0_usize;
        self.validate_at(fields, 1, &mut nodes)
    }

    fn validate_at(
        &mut self,
        fields: &[FieldSchema],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<()> {
        *nodes = nodes.checked_add(1).ok_or_else(Error::invalid_argument)?;
        if depth > MAX_PREDICATE_DEPTH || *nodes > MAX_PREDICATE_NODES {
            return Err(Error::invalid_argument());
        }
        match self {
            Self::And(children) | Self::Or(children) => {
                for child in children {
                    child.validate_at(fields, depth + 1, nodes)?;
                }
                Ok(())
            }
            Self::Not(child) => child.validate_at(fields, depth + 1, nodes),
            Self::Compare { field, value, .. } => {
                if value.is_null() {
                    return Err(Error::invalid_argument());
                }
                validate_predicate_value(*field, value, fields)
            }
            Self::In { field, values } => {
                if values.len() > MAX_IN_VALUES || values.iter().any(Value::is_null) {
                    return Err(Error::invalid_argument());
                }
                for value in values {
                    validate_predicate_value(*field, value, fields)?;
                }
                Ok(())
            }
            Self::IsNull(field) | Self::IsNotNull(field) => {
                field_schema(*field, fields).map(|_| ())
            }
        }
    }
}

fn field_schema(field: FieldId, fields: &[FieldSchema]) -> Result<&FieldSchema> {
    fields
        .get(usize::from(field.0))
        .ok_or_else(Error::invalid_argument)
}

fn validate_predicate_value(
    field: FieldId,
    value: &mut Value,
    fields: &[FieldSchema],
) -> Result<()> {
    value.validate_for(field_schema(field, fields)?.data_type, false)
}
