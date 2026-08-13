//! Vector Records and point-read projections.

use std::fmt;
use std::sync::Arc;

use bytes::Bytes;

use super::schema::FieldSchema;
use super::{Error, Result, Value};

const MAX_RECORD_ID_BYTES: usize = 256;
const MAX_PAYLOAD_BYTES: usize = 64 * 1_024;

/// An engine-owned Vector Record accepted by insert and upsert.
#[derive(Clone)]
pub struct Record {
    id: Bytes,
    vector: Arc<[f32]>,
    fields: Box<[Value]>,
    payload: Option<Bytes>,
}

impl Record {
    /// Creates a Vector Record and validates limits independent of a schema.
    pub fn new(
        id: Bytes,
        vector: impl Into<Arc<[f32]>>,
        fields: impl Into<Box<[Value]>>,
    ) -> Result<Self> {
        let record = Self {
            id,
            vector: vector.into(),
            fields: fields.into(),
            payload: None,
        };
        record.validate_shape()?;
        Ok(record)
    }

    /// Stores an Opaque Payload; `Some(empty)` remains distinct from absence.
    pub fn with_payload(mut self, payload: Bytes) -> Result<Self> {
        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(Error::invalid_argument());
        }
        self.payload = Some(payload);
        Ok(self)
    }

    /// Validates this record against one Logical Index configuration.
    pub fn validate(&mut self, dimension: usize, schema: &[FieldSchema]) -> Result<()> {
        self.validate_shape()?;
        if self.vector.len() != dimension || self.fields.len() != schema.len() {
            return Err(Error::invalid_argument());
        }
        for (value, field) in self.fields.iter_mut().zip(schema) {
            value.validate_for(field.data_type(), field.is_nullable())?;
        }
        Ok(())
    }

    /// Returns the opaque Record ID.
    #[must_use]
    pub const fn id(&self) -> &Bytes {
        &self.id
    }

    /// Returns the original finite vector.
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// Returns the positional typed fields.
    #[must_use]
    pub fn fields(&self) -> &[Value] {
        &self.fields
    }

    /// Returns the optional Opaque Payload.
    #[must_use]
    pub const fn payload(&self) -> Option<&Bytes> {
        self.payload.as_ref()
    }

    fn validate_shape(&self) -> Result<()> {
        if self.id.is_empty()
            || self.id.len() > MAX_RECORD_ID_BYTES
            || self.vector.iter().any(|component| !component.is_finite())
            || self
                .payload
                .as_ref()
                .is_some_and(|payload| payload.len() > MAX_PAYLOAD_BYTES)
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}

impl fmt::Debug for Record {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Record")
            .field("id", &"[REDACTED]")
            .field("vector", &"[REDACTED]")
            .field("fields", &"[REDACTED]")
            .field("payload", &self.payload.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

/// The closed Opaque Payload projection returned by point reads.
#[derive(Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum PayloadProjection {
    /// The caller did not request the payload.
    NotLoaded,
    /// The Vector Record has no payload.
    Absent,
    /// The payload exists, including when its bytes are empty.
    Present(Bytes),
}

impl fmt::Debug for PayloadProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotLoaded => "NotLoaded",
            Self::Absent => "Absent",
            Self::Present(_) => "Present([REDACTED])",
        })
    }
}

/// A Vector Record returned by a point read.
#[derive(Clone)]
pub struct StoredRecord {
    id: Bytes,
    vector: Arc<[f32]>,
    fields: Box<[Value]>,
    payload: PayloadProjection,
}

impl StoredRecord {
    /// Constructs a storage-returned Vector Record projection.
    #[must_use]
    pub fn new(
        id: Bytes,
        vector: Arc<[f32]>,
        fields: Box<[Value]>,
        payload: PayloadProjection,
    ) -> Self {
        Self {
            id,
            vector,
            fields,
            payload,
        }
    }

    /// Returns the opaque Record ID.
    #[must_use]
    pub const fn id(&self) -> &Bytes {
        &self.id
    }

    /// Returns the original vector.
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// Returns the positional typed fields.
    #[must_use]
    pub fn fields(&self) -> &[Value] {
        &self.fields
    }

    /// Returns the closed Opaque Payload projection.
    #[must_use]
    pub const fn payload(&self) -> &PayloadProjection {
        &self.payload
    }
}

impl fmt::Debug for StoredRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoredRecord")
            .field("id", &"[REDACTED]")
            .field("vector", &"[REDACTED]")
            .field("fields", &"[REDACTED]")
            .field("payload", &"[REDACTED]")
            .finish()
    }
}
