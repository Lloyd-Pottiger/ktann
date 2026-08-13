//! Bounded read-only verification request and report values.

use std::fmt;

use bytes::Bytes;

use super::{Error, LogicalIndexId, OperationOptions, PartitionKey, Result};

const DEFAULT_ISSUES: usize = 100;
const MAX_ISSUES: usize = 10_000;
const DEFAULT_OBJECTS: u64 = 1_000_000;
const MAX_OBJECTS: u64 = 100_000_000;
const DEFAULT_MEMORY_BYTES: u64 = 64 * 1_024 * 1_024;
const MAX_MEMORY_BYTES: u64 = 1_024 * 1_024 * 1_024;

/// Resource and operation-control limits for one read-only verification.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VerifyOptions {
    issue_limit: usize,
    object_limit: u64,
    memory_limit_bytes: u64,
    operation: OperationOptions,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            issue_limit: DEFAULT_ISSUES,
            object_limit: DEFAULT_OBJECTS,
            memory_limit_bytes: DEFAULT_MEMORY_BYTES,
            operation: OperationOptions::default(),
        }
    }
}

impl VerifyOptions {
    /// Sets the independent positive issue limit.
    pub fn with_issue_limit(mut self, limit: usize) -> Result<Self> {
        self.issue_limit = limit;
        self.validate()?;
        Ok(self)
    }

    /// Sets the independent positive decoded logical-object limit.
    pub fn with_object_limit(mut self, limit: u64) -> Result<Self> {
        self.object_limit = limit;
        self.validate()?;
        Ok(self)
    }

    /// Sets the independent positive resident-memory limit.
    pub fn with_memory_limit_bytes(mut self, limit: u64) -> Result<Self> {
        self.memory_limit_bytes = limit;
        self.validate()?;
        Ok(self)
    }

    /// Sets deadline and cancellation control for the verification snapshot.
    #[must_use]
    pub fn with_operation_options(mut self, operation: OperationOptions) -> Self {
        self.operation = operation;
        self
    }

    /// Validates all hard safety limits.
    pub fn validate(&self) -> Result<()> {
        if self.issue_limit == 0
            || self.issue_limit > MAX_ISSUES
            || self.object_limit == 0
            || self.object_limit > MAX_OBJECTS
            || self.memory_limit_bytes == 0
            || self.memory_limit_bytes > MAX_MEMORY_BYTES
            || usize::try_from(self.memory_limit_bytes).is_err()
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }

    /// Returns the maximum number of reported issues.
    #[must_use]
    pub const fn issue_limit(&self) -> usize {
        self.issue_limit
    }

    /// Returns the maximum decoded logical-object count.
    #[must_use]
    pub const fn object_limit(&self) -> u64 {
        self.object_limit
    }

    /// Returns the maximum resident-memory byte count.
    #[must_use]
    pub const fn memory_limit_bytes(&self) -> u64 {
        self.memory_limit_bytes
    }

    /// Returns deadline and cancellation control.
    #[must_use]
    pub const fn operation_options(&self) -> &OperationOptions {
        &self.operation
    }
}

/// A stable coarse verification issue category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VerifyIssueKind {
    /// Persistent bytes are malformed, noncanonical, or unsupported here.
    InvalidEncoding,
    /// A reachable object is missing or an unreachable object exists.
    Reachability,
    /// A child or Leaf Entry has invalid or duplicate membership.
    Membership,
    /// A stored exact count disagrees with its logical entries.
    CountMismatch,
    /// Vector Record, Record Location, and Leaf Entry projections disagree.
    RecordProjectionMismatch,
    /// A Partition Synopsis could produce unsound pruning.
    SynopsisNotConservative,
}

/// Safe verification identifiers for one issue.
#[derive(Clone)]
#[non_exhaustive]
pub struct VerifyIssue {
    /// Stable coarse issue category.
    pub kind: VerifyIssueKind,
    /// Logical Index ID containing the issue.
    pub logical_index_id: LogicalIndexId,
    /// Stable hash of the Tree Key, never the raw Tree Key.
    pub tree_key_hash: [u8; 32],
    /// Partition Key containing or referencing the issue.
    pub partition_key: PartitionKey,
    /// Optional opaque Record ID for programmatic correlation.
    pub record_id: Option<Bytes>,
}

impl fmt::Debug for VerifyIssue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifyIssue")
            .field("kind", &self.kind)
            .field("logical_index_id", &self.logical_index_id)
            .field("tree_key_hash", &self.tree_key_hash)
            .field("partition_key", &self.partition_key)
            .field("record_id", &self.record_id.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

/// Decoded logical-object counts observed by verification.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct VerifyObjectCounts {
    /// Total decoded logical objects charged to the object limit.
    pub total: u64,
    /// Vector Records decoded.
    pub vector_records: u64,
    /// Record Locations decoded.
    pub record_locations: u64,
    /// Partition bodies decoded.
    pub partitions: u64,
    /// Internal and Leaf Entries decoded.
    pub entries: u64,
}

/// The bounded result of one read-only verification snapshot.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct VerifyReport {
    /// Whether the full Logical Index was verified before any limit was reached.
    pub complete: bool,
    /// Ordered bounded issues.
    pub issues: Vec<VerifyIssue>,
    /// Decoded logical-object counts.
    pub objects: VerifyObjectCounts,
}
