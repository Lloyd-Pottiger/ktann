//! Emission helpers for the `ktann.*` metric namespace.
//!
//! Each helper emits one documented series with labels taken only from the
//! bounded enums in [`super::labels`]. All helpers are no-ops until a metrics
//! recorder is installed.

use std::time::Duration;

use crate::api::{SearchBudgetExhaustion, SearchBudgetUsage, VerifyIssueKind, VerifyReport};
use crate::search::cache::PartitionKind;

use super::labels::{
    BudgetDimension, CacheInstallResult, CacheLookupResult, FixupAdmission, FixupExecution,
    FixupKind, ImportGate, Operation, OperationOutcome, VerifyCompletion, cache_level, key,
    verify_issue,
};

/// The metric names of the `ktann.*` namespace; the inventory table in
/// [`super`](crate::observe) documents kind and labels of each series.
pub(crate) mod names {
    pub(crate) const OPERATION_TOTAL: &str = "ktann.operation.total";
    pub(crate) const OPERATION_DURATION: &str = "ktann.operation.duration";
    pub(crate) const WRITE_RETRIES: &str = "ktann.write.retries";
    pub(crate) const SEARCH_BUDGET_USAGE: &str = "ktann.search.budget.usage";
    pub(crate) const SEARCH_BUDGET_EXHAUSTED: &str = "ktann.search.budget.exhausted";
    pub(crate) const CACHE_LOOKUP: &str = "ktann.cache.lookup";
    pub(crate) const CACHE_INSTALL: &str = "ktann.cache.install";
    pub(crate) const CACHE_BYTES: &str = "ktann.cache.bytes";
    pub(crate) const FIXUP_ADMISSION: &str = "ktann.fixup.admission";
    pub(crate) const FIXUP_BACKLOG: &str = "ktann.fixup.backlog";
    pub(crate) const FIXUP_EXECUTION: &str = "ktann.fixup.execution";
    pub(crate) const FIXUP_STATE_AGE: &str = "ktann.fixup.state_age";
    pub(crate) const BLOOM_FILL_RATIO: &str = "ktann.bloom.fill_ratio";
    pub(crate) const IMPORT_WAIT: &str = "ktann.import.wait";
    pub(crate) const VERIFY_REPORTS: &str = "ktann.verify.reports";
    pub(crate) const VERIFY_ISSUES: &str = "ktann.verify.issues";
}

/// Records one finished foreground operation's outcome and latency.
pub(crate) fn operation_finished(
    operation: Operation,
    outcome: OperationOutcome,
    duration: Duration,
) {
    metrics::counter!(
        names::OPERATION_TOTAL,
        key::OPERATION => operation.as_str(),
        key::OUTCOME => outcome.as_str(),
    )
    .increment(1);
    metrics::histogram!(
        names::OPERATION_DURATION,
        key::OPERATION => operation.as_str(),
        key::OUTCOME => outcome.as_str(),
    )
    .record(duration.as_secs_f64());
}

/// Counts one whole-attempt write retry.
pub(crate) fn write_retried(operation: Operation) {
    metrics::counter!(names::WRITE_RETRIES, key::OPERATION => operation.as_str()).increment(1);
}

/// Records the logical budget usage and exhaustion of one search.
pub(crate) fn search_budget(usage: &SearchBudgetUsage, exhausted: &SearchBudgetExhaustion) {
    let dimensions = [
        (
            BudgetDimension::ScannedTreeKeys,
            usage.scanned_tree_keys,
            exhausted.scanned_tree_keys,
        ),
        (
            BudgetDimension::VisitedPartitions,
            usage.visited_partitions,
            exhausted.visited_partitions,
        ),
        (
            BudgetDimension::VisitedLeafEntries,
            usage.visited_leaf_entries,
            exhausted.visited_leaf_entries,
        ),
        (
            BudgetDimension::ExactRerankCandidates,
            usage.exact_rerank_candidates,
            exhausted.exact_rerank_candidates,
        ),
    ];
    for (dimension, used, prevented) in dimensions {
        metrics::histogram!(names::SEARCH_BUDGET_USAGE, key::DIMENSION => dimension.as_str())
            .record(f64::from(used));
        if prevented {
            metrics::counter!(
                names::SEARCH_BUDGET_EXHAUSTED,
                key::DIMENSION => dimension.as_str(),
            )
            .increment(1);
        }
    }
}

/// Counts one Partition Cache lookup by level and result.
pub(crate) fn cache_lookup(level: PartitionKind, result: CacheLookupResult) {
    metrics::counter!(
        names::CACHE_LOOKUP,
        key::LEVEL => cache_level(level),
        key::RESULT => result.as_str(),
    )
    .increment(1);
}

/// Counts one Partition Cache install by level and result.
pub(crate) fn cache_install(level: PartitionKind, result: CacheInstallResult) {
    metrics::counter!(
        names::CACHE_INSTALL,
        key::LEVEL => cache_level(level),
        key::RESULT => result.as_str(),
    )
    .increment(1);
}

/// Publishes the currently accounted bytes of all cached bodies.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn cache_bytes(bytes: u64) {
    metrics::gauge!(names::CACHE_BYTES).set(bytes as f64);
}

/// Counts Fixup queue offers by admission outcome.
pub(crate) fn fixup_admission(result: FixupAdmission, count: u64) {
    metrics::counter!(names::FIXUP_ADMISSION, key::OUTCOME => result.as_str()).increment(count);
}

/// Publishes the current Fixup backlog: pending plus running.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn fixup_backlog(backlog: usize) {
    metrics::gauge!(names::FIXUP_BACKLOG).set(backlog as f64);
}

/// Counts one finished Fixup execution's outcome.
pub(crate) fn fixup_execution(result: FixupExecution) {
    metrics::counter!(names::FIXUP_EXECUTION, key::OUTCOME => result.as_str()).increment(1);
}

/// Records the wall-clock age of the durable partition state one Fixup step
/// advanced; a future persisted timestamp saturates to zero (design
/// `runtime-operations.md` §3).
pub(crate) fn fixup_state_age(kind: FixupKind, now_unix_millis: u64, started_at_unix_millis: u64) {
    let age = Duration::from_millis(now_unix_millis.saturating_sub(started_at_unix_millis));
    metrics::histogram!(names::FIXUP_STATE_AGE, key::KIND => kind.as_str())
        .record(age.as_secs_f64());
}

/// Records the set-bit ratio of one Bloom filter after a membership
/// expansion.
pub(crate) fn bloom_fill_ratio(ratio: f64) {
    metrics::histogram!(names::BLOOM_FILL_RATIO).record(ratio);
}

/// Records one Import Session admission wait behind one gate.
pub(crate) fn import_wait(gate: ImportGate, duration: Duration) {
    metrics::histogram!(names::IMPORT_WAIT, key::GATE => gate.as_str())
        .record(duration.as_secs_f64());
}

/// Records one verification report's completeness and per-kind issue counts.
pub(crate) fn verify_report(report: &VerifyReport) {
    let completion = if report.complete {
        VerifyCompletion::Complete
    } else {
        VerifyCompletion::Incomplete
    };
    metrics::counter!(names::VERIFY_REPORTS, key::OUTCOME => completion.as_str()).increment(1);
    for kind in [
        VerifyIssueKind::InvalidEncoding,
        VerifyIssueKind::Reachability,
        VerifyIssueKind::Membership,
        VerifyIssueKind::CountMismatch,
        VerifyIssueKind::RecordProjectionMismatch,
        VerifyIssueKind::SynopsisNotConservative,
    ] {
        let count = report
            .issues
            .iter()
            .filter(|issue| issue.kind == kind)
            .count() as u64;
        if count > 0 {
            metrics::counter!(names::VERIFY_ISSUES, key::KIND => verify_issue(kind))
                .increment(count);
        }
    }
}

#[cfg(test)]
mod tests {
    use metrics_util::debugging::{DebuggingRecorder, Snapshotter};

    use super::*;
    use crate::api::{VerifyIssue, VerifyIssueKind, VerifyObjectCounts};

    fn recorder() -> (DebuggingRecorder, Snapshotter) {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        (recorder, snapshotter)
    }

    #[test]
    fn emissions_use_only_allowlisted_labels() {
        let (recorder, snapshotter) = recorder();
        metrics::with_local_recorder(&recorder, || {
            operation_finished(
                Operation::Search,
                OperationOutcome::Ok,
                Duration::from_millis(3),
            );
            write_retried(Operation::SplitFixup);
            search_budget(
                &SearchBudgetUsage {
                    scanned_tree_keys: 4,
                    visited_partitions: 2,
                    visited_leaf_entries: 9,
                    exact_rerank_candidates: 1,
                },
                &SearchBudgetExhaustion {
                    scanned_tree_keys: true,
                    ..Default::default()
                },
            );
            cache_lookup(PartitionKind::Leaf, CacheLookupResult::Hit);
            cache_install(PartitionKind::Internal, CacheInstallResult::SkippedStale);
            cache_bytes(512);
            fixup_admission(FixupAdmission::Duplicate, 2);
            fixup_backlog(3);
            fixup_execution(FixupExecution::Settled);
            fixup_state_age(FixupKind::Split, 7_000, 1_000);
            bloom_fill_ratio(0.25);
            import_wait(ImportGate::Backlog, Duration::from_millis(2));
            verify_report(&VerifyReport {
                complete: false,
                issues: vec![VerifyIssue {
                    kind: VerifyIssueKind::Membership,
                    logical_index_id: crate::api::LogicalIndexId::new(7).expect("nonzero id"),
                    tree_key_hash: [7_u8; 32],
                    partition_key: crate::api::PartitionKey::new(1).expect("nonzero key"),
                    record_id: None,
                }],
                objects: VerifyObjectCounts::default(),
            });
        });

        let snapshot = snapshotter.snapshot();
        let mut seen_names = Vec::new();
        for (key, _unit, _description, _value) in snapshot.into_vec() {
            let name = key.key().name().to_owned();
            seen_names.push(name.clone());
            assert!(
                name.starts_with("ktann."),
                "name {name} leaves the namespace"
            );
            for label in key.key().labels() {
                let value = label.value();
                assert!(
                    value.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
                    "label value {value:?} is not a bounded enum"
                );
            }
        }
        // Every documented family was exercised above.
        for expected in [
            names::OPERATION_TOTAL,
            names::OPERATION_DURATION,
            names::WRITE_RETRIES,
            names::SEARCH_BUDGET_USAGE,
            names::SEARCH_BUDGET_EXHAUSTED,
            names::CACHE_LOOKUP,
            names::CACHE_INSTALL,
            names::CACHE_BYTES,
            names::FIXUP_ADMISSION,
            names::FIXUP_BACKLOG,
            names::FIXUP_EXECUTION,
            names::FIXUP_STATE_AGE,
            names::BLOOM_FILL_RATIO,
            names::IMPORT_WAIT,
            names::VERIFY_REPORTS,
            names::VERIFY_ISSUES,
        ] {
            assert!(
                seen_names.iter().any(|name| name == expected),
                "missing series {expected}"
            );
        }
    }
}
