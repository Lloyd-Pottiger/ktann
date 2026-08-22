//! Bounded label values: the documented allowlist of design
//! `runtime-operations.md` section 5 as closed enums, under the redaction
//! policy of [`crate::observe`].
//!
//! Every metric label value and bounded trace field is a `&'static str`
//! produced here; no constructor in this module accepts caller data.

use crate::api::{ErrorKind, VerifyIssueKind};
use crate::search::cache::PartitionKind;

/// The metric label keys used across the `ktann.*` namespace.
pub(crate) mod key {
    /// One observed operation.
    pub(crate) const OPERATION: &str = "operation";
    /// The outcome of one operation, report, or commit.
    pub(crate) const OUTCOME: &str = "outcome";
    /// One Search Budget dimension.
    pub(crate) const DIMENSION: &str = "dimension";
    /// The Partition Cache level.
    pub(crate) const LEVEL: &str = "level";
    /// The bounded result of one cache access.
    pub(crate) const RESULT: &str = "result";
    /// The Fixup state machine or verification issue kind.
    pub(crate) const KIND: &str = "kind";
    /// One Import Session admission gate.
    pub(crate) const GATE: &str = "gate";
}

/// One observed foreground operation or maintenance write step (key
/// `operation`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Operation {
    /// `Runtime::create_index`.
    CreateIndex,
    /// `Runtime::open_index`.
    OpenIndex,
    /// `Runtime::drop_index`.
    DropIndex,
    /// `Index::insert`.
    Insert,
    /// `Index::upsert`.
    Upsert,
    /// `Index::delete`.
    Delete,
    /// `Index::batch_mutate`, including one Import Session batch.
    BatchMutate,
    /// `Index::get`.
    Get,
    /// `Index::batch_get`.
    BatchGet,
    /// `Index::search`.
    Search,
    /// `Index::verify`.
    Verify,
    /// One split Fixup write step; labels `ktann.write.retries` only.
    SplitFixup,
    /// One merge Fixup write step; labels `ktann.write.retries` only.
    MergeFixup,
}

impl Operation {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CreateIndex => "create_index",
            Self::OpenIndex => "open_index",
            Self::DropIndex => "drop_index",
            Self::Insert => "insert",
            Self::Upsert => "upsert",
            Self::Delete => "delete",
            Self::BatchMutate => "batch_mutate",
            Self::Get => "get",
            Self::BatchGet => "batch_get",
            Self::Search => "search",
            Self::Verify => "verify",
            Self::SplitFixup => "split_fixup",
            Self::MergeFixup => "merge_fixup",
        }
    }
}

/// The outcome of one operation (key `outcome`): success or the stable
/// [`ErrorKind`] category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OperationOutcome {
    /// The operation succeeded.
    Ok,
    /// The operation failed with the stable error category.
    Error(ErrorKind),
}

impl OperationOutcome {
    /// Classifies one operation result.
    pub(crate) fn from_result<T>(result: &crate::api::Result<T>) -> Self {
        match result {
            Ok(_) => Self::Ok,
            Err(error) => Self::Error(error.kind()),
        }
    }

    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error(kind) => error_kind(kind),
        }
    }
}

/// The bounded label value of one stable error category.
pub(crate) const fn error_kind(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::InvalidArgument => "invalid_argument",
        ErrorKind::IndexAlreadyExists => "index_already_exists",
        ErrorKind::IndexNotFound => "index_not_found",
        ErrorKind::IndexDropping => "index_dropping",
        ErrorKind::RecordAlreadyExists => "record_already_exists",
        ErrorKind::UnsupportedFormat => "unsupported_format",
        ErrorKind::Unsupported => "unsupported",
        ErrorKind::TransactionTooLarge => "transaction_too_large",
        ErrorKind::LimitExceeded => "limit_exceeded",
        ErrorKind::ContentionExhausted => "contention_exhausted",
        ErrorKind::RetryableAbort => "retryable_abort",
        ErrorKind::CommitOutcomeUnknown => "commit_outcome_unknown",
        ErrorKind::IdExhausted => "id_exhausted",
        ErrorKind::DeadlineExceeded => "deadline_exceeded",
        ErrorKind::Cancelled => "cancelled",
        ErrorKind::RuntimeClosed => "runtime_closed",
        ErrorKind::Backend => "backend",
        ErrorKind::Other => "other",
        ErrorKind::Corruption => "corruption",
    }
}

/// One Search Budget dimension (key `dimension`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BudgetDimension {
    /// Tree Keys decoded and checked from the directory.
    ScannedTreeKeys,
    /// Distinct partition bodies logically visited.
    VisitedPartitions,
    /// Leaf Entries read and considered under the exact Filter Predicate.
    VisitedLeafEntries,
    /// Vector Records read and exactly reranked.
    ExactRerankCandidates,
}

impl BudgetDimension {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ScannedTreeKeys => "scanned_tree_keys",
            Self::VisitedPartitions => "visited_partitions",
            Self::VisitedLeafEntries => "visited_leaf_entries",
            Self::ExactRerankCandidates => "exact_rerank_candidates",
        }
    }
}

/// The bounded label value of one Partition Cache level (key `level`).
pub(crate) const fn cache_level(kind: PartitionKind) -> &'static str {
    match kind {
        PartitionKind::Leaf => "leaf",
        PartitionKind::Internal => "internal",
    }
}

/// The result of one Partition Cache lookup (key `result`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CacheLookupResult {
    /// The cached body matched the snapshot Header epoch.
    Hit,
    /// A cached older epoch was evicted and reported as a miss.
    StaleMiss,
    /// Nothing usable was cached, or a cached newer epoch missed without
    /// eviction.
    Miss,
}

impl CacheLookupResult {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::StaleMiss => "stale_miss",
            Self::Miss => "miss",
        }
    }
}

/// The result of one Partition Cache install (key `result`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CacheInstallResult {
    /// The body was published.
    Installed,
    /// The body exceeded the cache capacity and was skipped.
    SkippedOversized,
    /// A racing fill already published a newer epoch; the stale body was
    /// skipped.
    SkippedStale,
}

impl CacheInstallResult {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Installed => "installed",
            Self::SkippedOversized => "skipped_oversized",
            Self::SkippedStale => "skipped_stale",
        }
    }
}

/// One Fixup state machine (key `kind`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FixupKind {
    /// The split state machine.
    Split,
    /// The merge state machine.
    Merge,
}

impl FixupKind {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Merge => "merge",
        }
    }
}

/// The outcome of offering one Fixup key (key `outcome`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FixupAdmission {
    /// The key took a queue slot.
    Enqueued,
    /// The key was already pending or running; the offer coalesced.
    Duplicate,
    /// The queue was full; the offer was dropped.
    Saturated,
}

impl FixupAdmission {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Enqueued => "enqueued",
            Self::Duplicate => "duplicate",
            Self::Saturated => "saturated",
        }
    }
}

/// The outcome of one Fixup execution (key `outcome`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FixupExecution {
    /// The partition reached a state with no work for either state machine.
    Settled,
    /// The partition is `Merging` with no legal target.
    Stalled,
    /// Execution stopped early: error, step exhaustion, cancellation, or
    /// backend release.
    Retired,
}

impl FixupExecution {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Settled => "settled",
            Self::Stalled => "stalled",
            Self::Retired => "retired",
        }
    }
}

/// One Import Session admission gate (key `gate`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImportGate {
    /// The bounded in-flight batch slot.
    InFlightSlot,
    /// The maintenance backlog watermark.
    Backlog,
}

impl ImportGate {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::InFlightSlot => "in_flight_slot",
            Self::Backlog => "backlog",
        }
    }
}

/// The completeness outcome of one verification report (key `outcome`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VerifyCompletion {
    /// The full Logical Index was verified.
    Complete,
    /// A reached limit stopped the audit early.
    Incomplete,
}

impl VerifyCompletion {
    /// The bounded label value.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Incomplete => "incomplete",
        }
    }
}

/// The bounded label value of one verification issue kind (key `kind`).
pub(crate) const fn verify_issue(kind: VerifyIssueKind) -> &'static str {
    match kind {
        VerifyIssueKind::InvalidEncoding => "invalid_encoding",
        VerifyIssueKind::Reachability => "reachability",
        VerifyIssueKind::Membership => "membership",
        VerifyIssueKind::CountMismatch => "count_mismatch",
        VerifyIssueKind::RecordProjectionMismatch => "record_projection_mismatch",
        VerifyIssueKind::SynopsisNotConservative => "synopsis_not_conservative",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every label value is non-empty lowercase snake_case: bounded,
    /// exporter-safe, and free of caller data by shape.
    fn assert_bounded(values: &[&'static str]) {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        for window in sorted.windows(2) {
            assert_ne!(window[0], window[1], "duplicate label value");
        }
        for value in values {
            assert!(!value.is_empty());
            assert!(
                value
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
                "label value {value:?} is not snake_case"
            );
        }
    }

    #[test]
    fn label_values_are_bounded_snake_case() {
        assert_bounded(&[
            Operation::CreateIndex.as_str(),
            Operation::OpenIndex.as_str(),
            Operation::DropIndex.as_str(),
            Operation::Insert.as_str(),
            Operation::Upsert.as_str(),
            Operation::Delete.as_str(),
            Operation::BatchMutate.as_str(),
            Operation::Get.as_str(),
            Operation::BatchGet.as_str(),
            Operation::Search.as_str(),
            Operation::Verify.as_str(),
            Operation::SplitFixup.as_str(),
            Operation::MergeFixup.as_str(),
        ]);
        assert_bounded(&[
            OperationOutcome::Ok.as_str(),
            error_kind(ErrorKind::InvalidArgument),
            error_kind(ErrorKind::IndexAlreadyExists),
            error_kind(ErrorKind::IndexNotFound),
            error_kind(ErrorKind::IndexDropping),
            error_kind(ErrorKind::RecordAlreadyExists),
            error_kind(ErrorKind::UnsupportedFormat),
            error_kind(ErrorKind::Unsupported),
            error_kind(ErrorKind::TransactionTooLarge),
            error_kind(ErrorKind::LimitExceeded),
            error_kind(ErrorKind::ContentionExhausted),
            error_kind(ErrorKind::RetryableAbort),
            error_kind(ErrorKind::CommitOutcomeUnknown),
            error_kind(ErrorKind::IdExhausted),
            error_kind(ErrorKind::DeadlineExceeded),
            error_kind(ErrorKind::Cancelled),
            error_kind(ErrorKind::RuntimeClosed),
            error_kind(ErrorKind::Backend),
            error_kind(ErrorKind::Other),
            error_kind(ErrorKind::Corruption),
        ]);
        assert_bounded(&[
            BudgetDimension::ScannedTreeKeys.as_str(),
            BudgetDimension::VisitedPartitions.as_str(),
            BudgetDimension::VisitedLeafEntries.as_str(),
            BudgetDimension::ExactRerankCandidates.as_str(),
            cache_level(PartitionKind::Leaf),
            cache_level(PartitionKind::Internal),
            CacheLookupResult::Hit.as_str(),
            CacheLookupResult::StaleMiss.as_str(),
            CacheLookupResult::Miss.as_str(),
            CacheInstallResult::Installed.as_str(),
            CacheInstallResult::SkippedOversized.as_str(),
            CacheInstallResult::SkippedStale.as_str(),
            FixupKind::Split.as_str(),
            FixupKind::Merge.as_str(),
            FixupAdmission::Enqueued.as_str(),
            FixupAdmission::Duplicate.as_str(),
            FixupAdmission::Saturated.as_str(),
            FixupExecution::Settled.as_str(),
            FixupExecution::Stalled.as_str(),
            FixupExecution::Retired.as_str(),
            ImportGate::InFlightSlot.as_str(),
            ImportGate::Backlog.as_str(),
            VerifyCompletion::Complete.as_str(),
            VerifyCompletion::Incomplete.as_str(),
            verify_issue(VerifyIssueKind::InvalidEncoding),
            verify_issue(VerifyIssueKind::Reachability),
            verify_issue(VerifyIssueKind::Membership),
            verify_issue(VerifyIssueKind::CountMismatch),
            verify_issue(VerifyIssueKind::RecordProjectionMismatch),
            verify_issue(VerifyIssueKind::SynopsisNotConservative),
        ]);
    }

    #[test]
    fn outcome_classification_reads_the_error_kind() {
        let ok: crate::api::Result<()> = Ok(());
        assert_eq!(OperationOutcome::from_result(&ok), OperationOutcome::Ok);
        let err: crate::api::Result<()> = Err(crate::api::Error::new(ErrorKind::Corruption));
        assert_eq!(OperationOutcome::from_result(&err).as_str(), "corruption");
    }
}
