//! Backend-neutral identifiers.

use std::fmt;
use std::num::NonZeroU64;

use super::{Error, Result};

/// A validated caller-chosen Index Name.
///
/// Names contain `1..=255` UTF-8 bytes and retain their original bytes without
/// normalization. `Debug` is redacted; use [`IndexName::as_str`] when the raw
/// value is intentionally required.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct IndexName(String);

impl IndexName {
    /// Validates and owns an Index Name.
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.is_empty() || name.len() > 255 {
            return Err(Error::invalid_argument());
        }
        Ok(Self(name))
    }

    /// Returns the original, unnormalized Index Name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for IndexName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("IndexName([REDACTED])")
    }
}

/// The never-reused identity of a Logical Index within a Backend Namespace.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LogicalIndexId(NonZeroU64);

impl LogicalIndexId {
    /// Creates a nonzero Logical Index ID.
    pub fn new(value: u64) -> Result<Self> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(Error::invalid_argument)
    }

    /// Returns the integer identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// The stable identity of a partition within one Tree Key.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PartitionKey(NonZeroU64);

impl PartitionKey {
    /// Creates a nonzero Partition Key.
    pub fn new(value: u64) -> Result<Self> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(Error::invalid_argument)
    }

    /// Returns the integer identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// The zero-based position of a field in a Vector Record schema.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FieldId(pub u16);

/// A process-local identity assigned to an accepted Import Session batch.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BatchToken(NonZeroU64);

impl BatchToken {
    /// Creates a nonzero Batch Token.
    pub fn new(value: u64) -> Result<Self> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(Error::invalid_argument)
    }

    /// Returns the process-local integer identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}
