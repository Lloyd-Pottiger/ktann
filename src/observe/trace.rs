//! Span and event construction under the redaction policy of
//! [`crate::observe`].
//!
//! Spans and events carry only Logical Index IDs, Partition Keys, stable
//! Tree Key hashes, bounded label strings from [`super::labels`], counts, and
//! error kinds. Error sources are never recorded: an adapter-native source
//! may embed backend-internal strings, so events record only the stable
//! [`ErrorKind`].

use std::fmt;

use tracing::Span;

use crate::api::{Error, ErrorKind, LogicalIndexId, PartitionKey};
use crate::storage::keys::{TreeKey, tree_key_hash};

use super::labels::{FixupKind, Operation, error_kind};

/// The lowercase hex rendering of one stable Tree Key hash.
struct HexHash([u8; 32]);

impl fmt::Display for HexHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// One foreground operation span: the operation label and, when the caller
/// knows it, the Logical Index ID.
pub(crate) fn operation_span(operation: Operation, index: Option<LogicalIndexId>) -> Span {
    let span = tracing::debug_span!(
        "ktann.operation",
        operation = operation.as_str(),
        logical_index_id = tracing::field::Empty,
    );
    if let Some(index) = index {
        span.record("logical_index_id", index.get());
    }
    span
}

/// One Fixup execution span: the rediscovery identity plus the stable Tree
/// Key hash, never the raw Tree Key.
pub(crate) fn fixup_span(
    index: LogicalIndexId,
    partition: PartitionKey,
    tree_key: &TreeKey,
) -> Span {
    tracing::debug_span!(
        "ktann.fixup",
        logical_index_id = index.get(),
        partition_key = partition.get(),
        tree_key_hash = %HexHash(tree_key_hash(tree_key)),
        kind = tracing::field::Empty,
    )
}

/// Emits the levelled completion event for one failed operation.
///
/// Corruption is an operator-visible integrity failure and logs at ERROR; a
/// commit of unknown outcome requires the documented recovery protocol and
/// logs at WARN; every other failure is an ordinary operation result at
/// DEBUG. Only the stable error kind is recorded.
pub(crate) fn operation_failed(operation: Operation, error: &Error) {
    let kind = error_kind(error.kind());
    match error.kind() {
        ErrorKind::Corruption => {
            tracing::error!(
                operation = operation.as_str(),
                error_kind = kind,
                "ktann.operation failed"
            );
        }
        ErrorKind::CommitOutcomeUnknown => {
            tracing::warn!(
                operation = operation.as_str(),
                error_kind = kind,
                "ktann.operation failed"
            );
        }
        _ => {
            tracing::debug!(
                operation = operation.as_str(),
                error_kind = kind,
                "ktann.operation failed"
            );
        }
    }
}

/// Emits one whole-attempt retry event before the backoff wait.
pub(crate) fn write_retrying(operation: Operation, attempt: u32) {
    tracing::debug!(operation = operation.as_str(), attempt, "ktann.write retry");
}

/// Records the state machine a Fixup execution resolved to.
pub(crate) fn fixup_kind(span: &Span, kind: FixupKind) {
    span.record("kind", kind.as_str());
}
