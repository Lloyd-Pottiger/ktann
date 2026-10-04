//! Structured public errors.

use std::error::Error as StdError;
use std::fmt;

/// A convenient result alias for KTANN operations and validation.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable caller-visible error categories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Caller input or caller-derived arithmetic is invalid.
    InvalidArgument,
    /// The requested Index Name already has a conflicting Logical Index.
    IndexAlreadyExists,
    /// The requested Logical Index does not exist.
    IndexNotFound,
    /// The requested Logical Index is being dropped.
    IndexDropping,
    /// Insert found an existing Vector Record with the same Record ID.
    RecordAlreadyExists,
    /// The persistent format is known but unsupported by this build.
    UnsupportedFormat,
    /// The backend does not support the requested operation.
    Unsupported,
    /// The operation cannot fit the backend's declared transaction limits.
    TransactionTooLarge,
    /// The operation exceeds a hard limit or bounded admission capacity.
    LimitExceeded,
    /// Whole-operation retry attempts were exhausted.
    ContentionExhausted,
    /// The backend aborted the transaction and the whole operation may be retried.
    RetryableAbort,
    /// The backend cannot determine whether commit succeeded.
    CommitOutcomeUnknown,
    /// A persistent non-reusable identifier space is exhausted.
    IdExhausted,
    /// The operation deadline elapsed before commit began.
    DeadlineExceeded,
    /// The operation was explicitly cancelled before commit began.
    Cancelled,
    /// The Runtime no longer accepts work.
    RuntimeClosed,
    /// A backend operation failed.
    Backend,
    /// An unclassified backend failure without a more specific category.
    Other,
    /// Persistent encoding or a committed invariant is invalid.
    Corruption,
}

impl ErrorKind {
    const fn message(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid argument",
            Self::IndexAlreadyExists => "index already exists",
            Self::IndexNotFound => "index not found",
            Self::IndexDropping => "index is being dropped",
            Self::RecordAlreadyExists => "record already exists",
            Self::UnsupportedFormat => "unsupported format",
            Self::Unsupported => "backend does not support operation",
            Self::TransactionTooLarge => "transaction too large",
            Self::LimitExceeded => "limit exceeded",
            Self::ContentionExhausted => "contention retries exhausted",
            Self::RetryableAbort => "transaction aborted, retry",
            Self::CommitOutcomeUnknown => "commit outcome unknown",
            Self::IdExhausted => "identifier space exhausted",
            Self::DeadlineExceeded => "deadline exceeded",
            Self::Cancelled => "operation cancelled",
            Self::RuntimeClosed => "runtime closed",
            Self::Backend => "backend error",
            Self::Other => "unclassified backend error",
            Self::Corruption => "corruption detected",
        }
    }
}

/// A structured KTANN error with an optional diagnostic source.
///
/// `Display` and `Debug` deliberately expose only the stable kind and an
/// optional input position. The diagnostic source remains available through
/// [`std::error::Error::source`] without being rendered implicitly, preventing
/// accidental disclosure of Index Names, Record IDs, field values, vectors,
/// payloads, and raw Tree Keys.
pub struct Error {
    kind: ErrorKind,
    position: Option<usize>,
    source: Option<Box<dyn StdError + Send + Sync + 'static>>,
}

impl Error {
    /// Creates an error with no diagnostic source.
    #[must_use]
    pub const fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            position: None,
            source: None,
        }
    }

    /// Creates an error while preserving a diagnostic source.
    #[must_use]
    pub fn with_source<E>(kind: ErrorKind, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self {
            kind,
            position: None,
            source: Some(Box::new(source)),
        }
    }

    /// Returns the stable error category.
    #[must_use]
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Returns the offending batch input position when one is known.
    #[must_use]
    pub const fn position(&self) -> Option<usize> {
        self.position
    }

    pub(crate) const fn invalid_argument() -> Self {
        Self::new(ErrorKind::InvalidArgument)
    }

    pub(crate) const fn at_position(mut self, position: usize) -> Self {
        self.position = Some(position);
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.kind.message())?;
        if let Some(position) = self.position {
            write!(formatter, " at input position {position}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Error")
            .field("kind", &self.kind)
            .field("position", &self.position)
            .field("has_source", &self.source.is_some())
            .finish()
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}
