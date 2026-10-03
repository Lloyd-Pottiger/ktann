//! Metrics, tracing, and telemetry privacy.
//!
//! Emissions use the `metrics` and `tracing` facades. Labels come only from
//! bounded enums. Traces contain Logical Index IDs, Partition Keys, stable
//! Tree Key hashes, bounded labels, counts, and error kinds. Raw caller data
//! and adapter error sources are never recorded.
//!
//! Series use the `ktann.*` namespace. Durations are seconds, sizes are bytes,
//! and ratios are in `0.0..=1.0`. Names and span nesting are internal details.
//!
//! # Metric inventory
//!
//! | Name | Kind | Labels |
//! | --- | --- | --- |
//! | `ktann.operation.total` | counter | operation, outcome |
//! | `ktann.operation.duration` | histogram | operation, outcome |
//! | `ktann.foreground.admission` | counter | operation, outcome |
//! | `ktann.write.retries` | counter | operation |
//! | `ktann.mutation.stage.duration` | histogram | stage |
//! | `ktann.write.attempts` | counter | operation, outcome |
//! | `ktann.write.mutations` | counter | operation, outcome |
//! | `ktann.write.mutation_bytes` | counter | operation, outcome |
//! | `ktann.write.commit.duration` | histogram | operation, outcome |
//! | `ktann.search.budget.usage` | histogram | dimension |
//! | `ktann.search.budget.exhausted` | counter | dimension |
//! | `ktann.search.stage.duration` | histogram | stage |
//! | `ktann.cache.lookup` | counter | level, result |
//! | `ktann.cache.install` | counter | level, result |
//! | `ktann.cache.bytes` | gauge | — |
//! | `ktann.fixup.admission` | counter | outcome |
//! | `ktann.fixup.backlog` | gauge | — |
//! | `ktann.fixup.execution` | counter | outcome |
//! | `ktann.fixup.steps` | counter | kind, result |
//! | `ktann.fixup.drain.entries` | histogram | kind |
//! | `ktann.fixup.state_age` | histogram | kind |
//! | `ktann.bloom.fill_ratio` | histogram | — |
//! | `ktann.bulk.refinement.rounds` | counter | — |
//! | `ktann.bulk.refinement.moves` | counter | — |
//! | `ktann.import.wait` | histogram | gate |
//! | `ktann.import.concurrency.limit` | histogram | direction |
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
