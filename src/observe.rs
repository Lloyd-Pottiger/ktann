//! Metrics, tracing, and privacy (design `runtime-operations.md` section 5).
//!
//! KTANN emits through the `metrics` and `tracing` facades; without an
//! installed recorder or subscriber every emission is a no-op. This module is
//! the only place metric series and span fields are constructed, so the
//! documented redaction policy holds by construction:
//!
//! - Metric label values come only from the bounded enums in [`labels`]. Raw
//!   Index Names, Tree Keys, Record IDs, field values, vectors, and payloads
//!   can never become a label, because no constructor here accepts caller
//!   data.
//! - Trace spans and events carry only Logical Index IDs, Partition Keys,
//!   stable Tree Key hashes, bounded label strings, counts, and error kinds.
//!   Error sources are never recorded: adapter-native errors may embed
//!   backend-internal strings and stay reachable only through
//!   `std::error::Error::source`.
//!
//! All series live in one `ktann.*` namespace. Metric names and span nesting
//! are not public API. Durations are recorded in seconds, ratios in `0.0..=1.0`,
//! and sizes in bytes.
//!
//! # Metric inventory
//!
//! | Name | Kind | Labels |
//! | --- | --- | --- |
//! | `ktann.operation.total` | counter | operation, outcome |
//! | `ktann.operation.duration` | histogram | operation, outcome |
//! | `ktann.write.retries` | counter | operation |
//! | `ktann.search.budget.usage` | histogram | dimension |
//! | `ktann.search.budget.exhausted` | counter | dimension |
//! | `ktann.search.stage.duration` | histogram | stage |
//! | `ktann.cache.lookup` | counter | level, result |
//! | `ktann.cache.install` | counter | level, result |
//! | `ktann.cache.bytes` | gauge | — |
//! | `ktann.fixup.admission` | counter | outcome |
//! | `ktann.fixup.backlog` | gauge | — |
//! | `ktann.fixup.execution` | counter | outcome |
//! | `ktann.fixup.state_age` | histogram | kind |
//! | `ktann.bloom.fill_ratio` | histogram | — |
//! | `ktann.import.wait` | histogram | gate |
//! | `ktann.verify.reports` | counter | outcome |
//! | `ktann.verify.issues` | counter | kind |
//!
//! Backend adapters additionally emit `ktann.backend.commit`
//! {backend, outcome} and, for RocksDB, `ktann.backend.blocking.wait` and
//! `ktann.backend.blocking.held` {backend}; those names are adapter-local
//! because adapters are separate crates.

pub(crate) mod labels;
pub(crate) mod metrics;
pub(crate) mod trace;
