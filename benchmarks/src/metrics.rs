//! In-process capture of KTANN's bounded metrics for benchmark reports.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use metrics::atomics::AtomicU64;
use metrics::{Counter, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit};
use metrics_util::registry::{AtomicStorage, Registry};

use crate::report::{
    AdmissionSummary, CacheSummary, Distribution, OperationClass, WriteAttribution,
};

/// One process-global recorder used by a single benchmark worker.
#[derive(Debug)]
pub struct MetricCapture {
    /// Shared metric storage; the installed recorder and capture handle own it together.
    registry: Arc<MetricRegistry>,
    /// Direct handle used by allocation-free maintenance polling.
    fixup_backlog: Arc<AtomicU64>,
}

impl MetricCapture {
    /// Installs a metrics recorder for this worker process.
    ///
    /// # Errors
    ///
    /// Returns an error when another process-global recorder is already set.
    pub fn install() -> Result<Self, String> {
        let registry = Arc::new(MetricRegistry::atomic());
        let fixup_backlog =
            registry.get_or_create_gauge(&Key::from_static_name("ktann.fixup.backlog"), Arc::clone);
        metrics::set_global_recorder(CaptureRecorder {
            registry: Arc::clone(&registry),
        })
        .map_err(|error| format!("install metrics recorder: {error}"))?;
        Ok(Self {
            registry,
            fixup_backlog,
        })
    }

    /// Takes one interval snapshot while preserving process-state gauges.
    ///
    /// Counters and histograms are consumed so successive reports describe
    /// disjoint intervals. Gauges are loaded without mutation because they
    /// represent current state; in particular, Fixup backlog polling must
    /// distinguish an unchanged nonzero backlog from an explicit transition
    /// to zero.
    #[must_use]
    pub fn snapshot(&self) -> CapturedMetrics {
        let mut captured = CapturedMetrics::default();
        self.registry.visit_counters(|key, counter| {
            let value = counter.swap(0, Ordering::SeqCst);
            if value > 0 {
                captured
                    .counters
                    .insert(SeriesKey::from_metric_key(key), value);
            }
        });
        self.registry.visit_gauges(|key, gauge| {
            captured.gauges.insert(
                SeriesKey::from_metric_key(key),
                f64::from_bits(gauge.load(Ordering::SeqCst)),
            );
        });
        self.registry.visit_histograms(|key, histogram| {
            let mut values = Vec::new();
            histogram.clear_with(|block| values.extend_from_slice(block));
            if !values.is_empty() {
                captured
                    .histograms
                    .insert(SeriesKey::from_metric_key(key), values);
            }
        });
        captured
    }

    /// Reads the current pending-plus-running Structure Maintenance count.
    ///
    /// Maintenance polling uses this narrow path so it does not allocate a
    /// complete interval snapshot or consume counters and histograms while CPU
    /// and Backend IO remain under observation.
    #[must_use]
    pub fn fixup_backlog(&self) -> usize {
        finite_usize(f64::from_bits(self.fixup_backlog.load(Ordering::SeqCst))).unwrap_or_default()
    }
}

/// Atomic storage shared by the process-global recorder and its snapshot handle.
type MetricRegistry = Registry<Key, AtomicStorage>;

/// Recorder whose snapshot semantics match the benchmark accounting model.
///
/// KTANN writes through the process-global `metrics` facade. This recorder
/// keeps the facade unchanged while exposing interval counters/histograms and
/// non-destructive gauges to [`MetricCapture`]. Units and descriptions are not
/// retained because report construction matches the repository's fixed,
/// bounded metric names and labels directly.
#[derive(Debug)]
struct CaptureRecorder {
    /// Registry holding every metric handle returned to KTANN.
    registry: Arc<MetricRegistry>,
}

impl Recorder for CaptureRecorder {
    fn describe_counter(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}

    fn describe_gauge(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}

    fn describe_histogram(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}

    fn register_counter(&self, key: &Key, _metadata: &Metadata<'_>) -> Counter {
        self.registry
            .get_or_create_counter(key, |counter| Counter::from_arc(Arc::clone(counter)))
    }

    fn register_gauge(&self, key: &Key, _metadata: &Metadata<'_>) -> Gauge {
        self.registry
            .get_or_create_gauge(key, |gauge| Gauge::from_arc(Arc::clone(gauge)))
    }

    fn register_histogram(&self, key: &Key, _metadata: &Metadata<'_>) -> Histogram {
        self.registry
            .get_or_create_histogram(key, |histogram| Histogram::from_arc(Arc::clone(histogram)))
    }
}

/// One interval's counters/histograms plus current process-state gauges.
///
/// Series retain their complete bounded label set. Exact matching prevents a
/// newly introduced label dimension from being silently folded into an older
/// baseline and makes schema drift visible in report review.
#[derive(Debug, Default)]
pub struct CapturedMetrics {
    /// Monotonic event totals keyed by complete label identity.
    counters: BTreeMap<SeriesKey, u64>,
    /// Last observed process-state values keyed by complete label identity.
    gauges: BTreeMap<SeriesKey, f64>,
    /// Raw finite observations retained until report aggregation.
    histograms: BTreeMap<SeriesKey, Vec<f64>>,
}

impl CapturedMetrics {
    /// Returns one counter with the exact label set, or zero.
    #[must_use]
    pub fn counter(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        self.counters
            .get(&SeriesKey::new(name, labels))
            .copied()
            .unwrap_or_default()
    }

    /// Returns Runtime foreground-admission rejections for one public class.
    #[must_use]
    pub fn foreground_admission_rejections(&self, class: OperationClass) -> u64 {
        let operation = match class {
            OperationClass::Search => "search",
            OperationClass::Write => "upsert",
        };
        self.counter(
            "ktann.foreground.admission",
            &[("operation", operation), ("outcome", "rejected")],
        )
    }

    /// Returns one histogram with the exact label set.
    #[must_use]
    pub fn histogram(&self, name: &str, labels: &[(&str, &str)]) -> Vec<f64> {
        self.histograms
            .get(&SeriesKey::new(name, labels))
            .cloned()
            .unwrap_or_default()
    }

    /// Returns all observations from a histogram grouped by one label.
    #[must_use]
    pub fn histograms_by_label(&self, name: &str, label: &str) -> BTreeMap<String, Vec<f64>> {
        self.histograms
            .iter()
            .filter(|(key, _)| key.name == name)
            .filter_map(|(key, values)| {
                key.label(label)
                    .map(|value| (value.to_owned(), values.clone()))
            })
            .collect()
    }

    /// Returns all counters grouped by their rendered label values.
    #[must_use]
    pub fn counters_rendered(&self, name: &str) -> BTreeMap<String, u64> {
        self.counters
            .iter()
            .filter(|(key, _)| key.name == name)
            .map(|(key, value)| (key.render_labels(), *value))
            .collect()
    }

    /// Returns all histograms grouped by their complete rendered label sets.
    #[must_use]
    pub fn distributions_rendered(&self, name: &str) -> BTreeMap<String, Distribution> {
        self.histograms
            .iter()
            .filter(|(key, _)| key.name == name)
            .map(|(key, values)| {
                (
                    key.render_labels(),
                    Distribution::from_samples(values.clone()),
                )
            })
            .collect()
    }

    /// Converts operation-attributed write series into the report schema.
    #[must_use]
    pub fn write_attribution(&self) -> WriteAttribution {
        WriteAttribution {
            attempts: self.counters_rendered("ktann.write.attempts"),
            retries: self.counters_rendered("ktann.write.retries"),
            mutation_operations: self.counters_rendered("ktann.write.mutations"),
            mutation_bytes: self.counters_rendered("ktann.write.mutation_bytes"),
            commit_wait_ms: self
                .distributions_rendered("ktann.write.commit.duration")
                .into_iter()
                .map(|(labels, distribution)| (labels, distribution.seconds_to_milliseconds()))
                .collect(),
        }
    }

    /// Converts captured cache series into the report schema.
    #[must_use]
    pub fn cache_summary(&self) -> CacheSummary {
        let accounted_bytes = self
            .gauges
            .iter()
            .find(|(key, _)| key.name == "ktann.cache.bytes")
            .and_then(|(_, value)| finite_u64(*value));
        CacheSummary {
            lookups: self.counters_rendered("ktann.cache.lookup"),
            installs: self.counters_rendered("ktann.cache.install"),
            accounted_bytes,
        }
    }

    /// Converts captured admission series into the report schema.
    #[must_use]
    pub fn admission_summary(&self) -> AdmissionSummary {
        let milliseconds = |name: &str| {
            let values = self
                .histograms
                .iter()
                .filter(|(key, _)| key.name == name)
                .flat_map(|(_, values)| values.iter().copied())
                .collect();
            Distribution::from_samples(values).seconds_to_milliseconds()
        };
        let import_wait_ms = self
            .histograms_by_label("ktann.import.wait", "gate")
            .into_iter()
            .map(|(gate, values)| {
                (
                    gate,
                    Distribution::from_samples(values).seconds_to_milliseconds(),
                )
            })
            .collect();
        AdmissionSummary {
            blocking_wait_ms: milliseconds("ktann.backend.blocking.wait"),
            blocking_held_ms: milliseconds("ktann.backend.blocking.held"),
            import_wait_ms,
        }
    }

    /// Returns the total whole-operation write retry count.
    #[must_use]
    pub fn write_retries(&self) -> u64 {
        self.counters
            .iter()
            .filter(|(key, _)| key.name == "ktann.write.retries")
            .map(|(_, value)| *value)
            .sum()
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// A metric family plus its canonical sorted label set.
struct SeriesKey {
    name: String,
    labels: Vec<(String, String)>,
}

impl SeriesKey {
    /// Copies a facade key into the stable owned identity used by reports.
    fn from_metric_key(key: &Key) -> Self {
        let mut labels = key
            .labels()
            .map(|label| (label.key().to_owned(), label.value().to_owned()))
            .collect::<Vec<_>>();
        labels.sort();
        Self {
            name: key.name().to_owned(),
            labels,
        }
    }

    /// Builds the same canonical label ordering used for captured series.
    fn new(name: &str, labels: &[(&str, &str)]) -> Self {
        let mut labels: Vec<_> = labels
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect();
        labels.sort();
        Self {
            name: name.to_owned(),
            labels,
        }
    }

    /// Returns one bounded label value when the series carries it.
    fn label(&self, name: &str) -> Option<&str> {
        self.labels
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// Produces stable `key=value` text for report map keys.
    fn render_labels(&self) -> String {
        self.labels
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Converts a nonnegative integral gauge without permitting wraparound.
fn finite_u64(value: f64) -> Option<u64> {
    if value.is_finite() && value >= 0.0 && value <= u64::MAX as f64 {
        Some(value as u64)
    } else {
        None
    }
}

/// Converts a nonnegative integral gauge to a process-local resource count.
fn finite_usize(value: f64) -> Option<usize> {
    if value.is_finite() && value >= 0.0 && value <= usize::MAX as f64 {
        Some(value as usize)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{CapturedMetrics, MetricCapture, SeriesKey};

    #[test]
    fn write_series_preserve_complete_attribution_labels() {
        let mut captured = CapturedMetrics::default();
        let retryable = [
            ("operation", "batch_mutate"),
            ("outcome", "retryable_abort"),
        ];
        captured
            .counters
            .insert(SeriesKey::new("ktann.write.attempts", &retryable), 3);
        captured.counters.insert(
            SeriesKey::new("ktann.write.retries", &[("operation", "batch_mutate")]),
            2,
        );
        captured
            .counters
            .insert(SeriesKey::new("ktann.write.mutations", &retryable), 12);
        captured.counters.insert(
            SeriesKey::new("ktann.write.mutation_bytes", &retryable),
            480,
        );
        captured.histograms.insert(
            SeriesKey::new(
                "ktann.write.commit.duration",
                &[("operation", "split_fixup"), ("outcome", "committed")],
            ),
            vec![0.001, 0.003],
        );

        let writes = captured.write_attribution();
        assert_eq!(
            writes
                .attempts
                .get("operation=batch_mutate,outcome=retryable_abort"),
            Some(&3)
        );
        assert_eq!(writes.retries.get("operation=batch_mutate"), Some(&2));
        assert_eq!(
            writes
                .mutation_operations
                .get("operation=batch_mutate,outcome=retryable_abort"),
            Some(&12)
        );
        assert_eq!(
            writes
                .mutation_bytes
                .get("operation=batch_mutate,outcome=retryable_abort"),
            Some(&480)
        );
        let commit_wait = writes
            .commit_wait_ms
            .get("operation=split_fixup,outcome=committed")
            .expect("commit wait series");
        assert_eq!(commit_wait.count, 2);
        assert_eq!(commit_wait.mean, 2.0);
        assert_eq!(commit_wait.p95, 3.0);
    }

    #[test]
    fn gauges_remain_sticky_until_an_explicit_update() {
        let capture =
            MetricCapture::install().expect("recorder installs once in this test process");
        metrics::gauge!("ktann.fixup.backlog").set(3.0);
        assert_eq!(capture.fixup_backlog(), 3);

        // Repeated interval snapshots do not mutate current process state.
        let _ = capture.snapshot();
        assert_eq!(capture.fixup_backlog(), 3);

        metrics::gauge!("ktann.fixup.backlog").set(0.0);
        assert_eq!(capture.fixup_backlog(), 0);
    }
}
