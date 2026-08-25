//! Capture harness for the privacy-safe metrics and tracing audits (issue #36).
//!
//! One global `metrics` recorder and one global `tracing` subscriber capture
//! every emission the process makes. Audit tests serialize through
//! [`audit_lock`], drive operations with canary-shaped sensitive data, and
//! then scan the whole capture: no captured metric name, label, span, or event
//! may contain a canary, and every label and trace field must stay within the
//! documented bounded allowlist (design `runtime-operations.md` section 5).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use tracing::field::Visit;
use tracing::{Event, Id, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};

/// One captured span or event: its rendered `(field, value)` pairs.
type CapturedFields = Vec<(String, String)>;

/// One captured counter series: `(name, sorted (label, value) pairs,
/// cumulative value)`.
pub(crate) type CounterSeries = (String, Vec<(String, String)>, u64);

/// The installed capture stack: metric snapshots plus captured span and event
/// fields.
pub(crate) struct Capture {
    snapshotter: Snapshotter,
    spans: Arc<Mutex<HashMap<Id, CapturedFields>>>,
    events: Arc<Mutex<Vec<CapturedFields>>>,
}

/// Formats tracing fields into structured pairs without ever interpreting
/// them.
#[derive(Default)]
struct FieldVisitor {
    fields: CapturedFields,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.fields
            .push((field.name().to_owned(), format!("{value:?}")));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.fields
            .push((field.name().to_owned(), value.to_owned()));
    }
}

struct CaptureLayer {
    spans: Arc<Mutex<HashMap<Id, CapturedFields>>>,
    events: Arc<Mutex<Vec<CapturedFields>>>,
}

impl CaptureLayer {
    fn lock<V>(mutex: &Mutex<V>) -> MutexGuard<'_, V> {
        match mutex.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_new_span(
        &self,
        attributes: &tracing::span::Attributes<'_>,
        id: &Id,
        _context: Context<'_, S>,
    ) {
        let mut visitor = FieldVisitor::default();
        attributes.record(&mut visitor);
        Self::lock(&self.spans).insert(id.clone(), visitor.fields);
    }

    fn on_record(&self, id: &Id, values: &tracing::span::Record<'_>, _context: Context<'_, S>) {
        // Fields declared Empty at creation arrive here when recorded later.
        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        if let Some(fields) = Self::lock(&self.spans).get_mut(id) {
            fields.extend(visitor.fields);
        }
    }

    fn on_event(&self, event: &Event<'_>, _context: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        Self::lock(&self.events).push(visitor.fields);
    }
}

/// Installs the global capture stack once and returns it.
pub(crate) fn capture() -> &'static Capture {
    static CAPTURE: OnceLock<Capture> = OnceLock::new();
    CAPTURE.get_or_init(|| {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        recorder.install().expect("global recorder installs once");
        let spans = Arc::new(Mutex::new(HashMap::new()));
        let events = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry()
            .with(CaptureLayer {
                spans: Arc::clone(&spans),
                events: Arc::clone(&events),
            })
            .with(tracing_subscriber::filter::LevelFilter::TRACE);
        tracing::subscriber::set_global_default(subscriber)
            .expect("global subscriber installs once");
        Capture {
            snapshotter,
            spans,
            events,
        }
    })
}

/// Serializes audit tests: the capture stack is process-global, so only one
/// audit may clear and inspect it at a time.
pub(crate) async fn audit_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

impl Capture {
    /// Clears the captured spans and events. Metric state is cumulative;
    /// callers diff [`Capture::metric_labels`] snapshots instead.
    pub(crate) fn clear(&self) {
        CaptureLayer::lock(&self.spans).clear();
        CaptureLayer::lock(&self.events).clear();
    }

    /// Every metric series as `(name, sorted (label, value) pairs)`.
    pub(crate) fn metric_labels(&self) -> Vec<(String, Vec<(String, String)>)> {
        self.snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _unit, _description, _value)| {
                let mut labels: Vec<(String, String)> = key
                    .key()
                    .labels()
                    .map(|label| (label.key().to_owned(), label.value().to_owned()))
                    .collect();
                labels.sort();
                (key.key().name().to_owned(), labels)
            })
            .collect()
    }

    /// Every counter series as `(name, sorted (label, value) pairs,
    /// cumulative value)`. Gauges and histograms are excluded; callers diff
    /// two snapshots because metric state is cumulative.
    pub(crate) fn metric_counters(&self) -> Vec<CounterSeries> {
        self.snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter_map(|(key, _unit, _description, value)| {
                let DebugValue::Counter(value) = value else {
                    return None;
                };
                let mut labels: Vec<(String, String)> = key
                    .key()
                    .labels()
                    .map(|label| (label.key().to_owned(), label.value().to_owned()))
                    .collect();
                labels.sort();
                Some((key.key().name().to_owned(), labels, value))
            })
            .collect()
    }

    /// Every captured span's fields.
    pub(crate) fn spans(&self) -> Vec<CapturedFields> {
        CaptureLayer::lock(&self.spans).values().cloned().collect()
    }

    /// Every captured event's fields.
    pub(crate) fn events(&self) -> Vec<CapturedFields> {
        CaptureLayer::lock(&self.events).clone()
    }

    /// Asserts no canary string appears anywhere in the capture: metric
    /// names, label keys and values, and every span and event field.
    #[track_caller]
    pub(crate) fn assert_no_canaries(&self, canaries: &[&str]) {
        let mut rendered: Vec<String> = Vec::new();
        for (name, labels) in self.metric_labels() {
            rendered.push(name);
            for (key, value) in labels {
                rendered.push(key);
                rendered.push(value);
            }
        }
        for fields in self.spans().into_iter().chain(self.events()) {
            for (field, value) in fields {
                rendered.push(field);
                rendered.push(value);
            }
        }
        for canary in canaries {
            for haystack in &rendered {
                assert!(
                    !haystack.contains(canary),
                    "canary {canary:?} leaked into captured telemetry: {haystack:?}"
                );
            }
        }
    }

    /// Asserts every captured metric label key and value stays within the
    /// documented bounded allowlist.
    #[track_caller]
    pub(crate) fn assert_labels_bounded(&self) {
        for (name, labels) in self.metric_labels() {
            assert!(
                name.starts_with("ktann."),
                "metric {name} leaves the ktann.* namespace"
            );
            for (key, value) in labels {
                let allowed = ALLOWED_LABEL_VALUES
                    .iter()
                    .find(|(allowed_key, _)| *allowed_key == key);
                let Some((_, values)) = allowed else {
                    panic!("metric {name} uses undocumented label key {key:?}");
                };
                assert!(
                    values.contains(&value.as_str()),
                    "metric {name} label {key:?} has unbounded value {value:?}"
                );
            }
        }
    }

    /// Asserts every captured span and event field name stays within the
    /// documented trace allowlist.
    #[track_caller]
    pub(crate) fn assert_trace_fields_bounded(&self) {
        let allowed: HashSet<&str> = ALLOWED_TRACE_FIELDS.iter().copied().collect();
        for fields in self.spans().into_iter().chain(self.events()) {
            for (field, _value) in fields {
                assert!(
                    allowed.contains(field.as_str()),
                    "trace field {field:?} is outside the documented allowlist"
                );
            }
        }
    }
}

/// The documented bounded metric label allowlist (design
/// `runtime-operations.md` section 5). Duplicating it here makes the audit an
/// independent check of the implementation against the contract.
const ALLOWED_LABEL_VALUES: &[(&str, &[&str])] = &[
    ("backend", &["rocksdb", "foundationdb"]),
    (
        "operation",
        &[
            "create_index",
            "open_index",
            "drop_index",
            "insert",
            "upsert",
            "delete",
            "batch_mutate",
            "get",
            "batch_get",
            "search",
            "verify",
            "split_fixup",
            "merge_fixup",
        ],
    ),
    (
        "outcome",
        &[
            "ok",
            // Stable ErrorKind categories.
            "invalid_argument",
            "index_already_exists",
            "index_not_found",
            "index_dropping",
            "record_already_exists",
            "unsupported_format",
            "unsupported",
            "transaction_too_large",
            "limit_exceeded",
            "contention_exhausted",
            "retryable_abort",
            "commit_outcome_unknown",
            "id_exhausted",
            "deadline_exceeded",
            "cancelled",
            "runtime_closed",
            "backend",
            "other",
            "corruption",
            // Native commit outcomes.
            "committed",
            "retryable",
            "unknown",
            "failed",
            // Fixup admission and execution outcomes.
            "enqueued",
            "duplicate",
            "saturated",
            "settled",
            "stalled",
            "retired",
            // Verification completeness.
            "complete",
            "incomplete",
        ],
    ),
    (
        "dimension",
        &[
            "scanned_tree_keys",
            "visited_partitions",
            "visited_leaf_entries",
            "exact_rerank_candidates",
        ],
    ),
    ("stage", &["approximate_selection", "exact_reranking"]),
    ("level", &["leaf", "internal"]),
    (
        "result",
        &[
            "hit",
            "stale_miss",
            "miss",
            "installed",
            "skipped_oversized",
            "skipped_stale",
        ],
    ),
    (
        "kind",
        &[
            "split",
            "merge",
            "invalid_encoding",
            "reachability",
            "membership",
            "count_mismatch",
            "record_projection_mismatch",
            "synopsis_not_conservative",
        ],
    ),
    ("gate", &["in_flight_slot", "backlog"]),
];

/// The documented bounded trace field allowlist: Logical Index ID, Partition
/// Key, stable Tree Key hash, bounded label strings, counts, and error kinds.
const ALLOWED_TRACE_FIELDS: &[&str] = &[
    "message",
    "operation",
    "logical_index_id",
    "partition_key",
    "tree_key_hash",
    "kind",
    "attempt",
    "error_kind",
];
