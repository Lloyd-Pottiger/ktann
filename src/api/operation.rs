//! Foreground Mutation, point-read, operation-control, and import values.

use std::collections::HashSet;
use std::fmt;
use std::time::Instant;

use bytes::Bytes;
use tokio_util::sync::CancellationToken;

use super::{BatchToken, Error, Record, Result};

/// One atomic Foreground Mutation item.
#[derive(Clone)]
#[non_exhaustive]
pub enum Mutation {
    /// Inserts a Vector Record only when its Record ID is absent.
    Insert(Record),
    /// Creates or fully replaces a Vector Record.
    Upsert(Record),
    /// Idempotently deletes a Vector Record by Record ID.
    Delete(Bytes),
}

impl Mutation {
    pub(crate) fn id(&self) -> &Bytes {
        match self {
            Self::Insert(record) | Self::Upsert(record) => record.id(),
            Self::Delete(id) => id,
        }
    }
}

impl fmt::Debug for Mutation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Insert(_) => "Insert([REDACTED])",
            Self::Upsert(_) => "Upsert([REDACTED])",
            Self::Delete(_) => "Delete([REDACTED])",
        })
    }
}

/// The result of one upsert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum UpsertResult {
    /// No Vector Record previously had the Record ID.
    Created,
    /// The previous Vector Record was fully replaced.
    Replaced,
}

/// The result of one item in an atomic mutation batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MutationOutcome {
    /// Insert created the Vector Record.
    Inserted,
    /// Upsert completed.
    Upserted {
        /// Whether the upsert replaced an existing Vector Record.
        replaced: bool,
    },
    /// Delete completed.
    Deleted {
        /// Whether the Vector Record existed.
        existed: bool,
    },
}

/// Validates a complete atomic mutation batch before storage access.
///
/// An empty batch is valid. A duplicate or invalid Record ID, invalid Vector
/// Record, or schema mismatch returns `InvalidArgument` with the first invalid
/// input's zero-based position.
pub fn validate_mutations(
    mutations: &mut [Mutation],
    dimension: usize,
    schema: &[super::FieldSchema],
) -> Result<()> {
    let mut ids = HashSet::with_capacity(mutations.len());
    for (position, mutation) in mutations.iter_mut().enumerate() {
        let id = mutation.id();
        if id.is_empty() || id.len() > 256 || !ids.insert(id.clone()) {
            return Err(Error::invalid_argument().at_position(position));
        }
        match mutation {
            Mutation::Insert(record) | Mutation::Upsert(record) => record
                .validate(dimension, schema)
                .map_err(|error| error.at_position(position))?,
            Mutation::Delete(_) => {}
        }
    }
    Ok(())
}

/// Controls an operation independently from its logical request.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct OperationOptions {
    deadline: Option<Instant>,
    cancellation: Option<CancellationToken>,
}

impl OperationOptions {
    /// Sets an optional monotonic deadline.
    #[must_use]
    pub const fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Sets an explicit cloneable cancellation token.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Returns the monotonic deadline.
    #[must_use]
    pub const fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Returns the cancellation token.
    #[must_use]
    pub const fn cancellation(&self) -> Option<&CancellationToken> {
        self.cancellation.as_ref()
    }
}

/// Options for a point read.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct GetOptions {
    include_payload: bool,
}

impl GetOptions {
    /// Requests the Opaque Payload projection.
    #[must_use]
    pub const fn with_payload(mut self) -> Self {
        self.include_payload = true;
        self
    }

    /// Returns whether the Opaque Payload should be loaded.
    #[must_use]
    pub const fn includes_payload(self) -> bool {
        self.include_payload
    }
}

/// Bounded Import Session admission options.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct ImportOptions {
    in_flight_batches: Option<usize>,
}

impl ImportOptions {
    /// Overrides the Runtime's positive in-flight batch limit.
    pub fn with_in_flight_batches(mut self, batches: usize) -> Result<Self> {
        if batches == 0 {
            return Err(Error::invalid_argument());
        }
        self.in_flight_batches = Some(batches);
        Ok(self)
    }

    /// Returns the optional in-flight batch override.
    #[must_use]
    pub const fn in_flight_batches(self) -> Option<usize> {
        self.in_flight_batches
    }
}

/// One Import Session batch result in submission order.
#[derive(Debug)]
#[non_exhaustive]
pub struct ImportBatchResult {
    /// The unique process-local Batch Token returned by submission.
    pub token: BatchToken,
    /// The ordinary atomic batch result.
    pub result: Result<Vec<MutationOutcome>>,
}
