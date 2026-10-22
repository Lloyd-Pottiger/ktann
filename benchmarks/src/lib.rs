//! Reproducible whole-system performance baselines for KTANN.

#![deny(unsafe_code)]

mod backend;
pub mod cli;
mod compare;
mod dataset;
mod metrics;
mod report;
mod resource;
mod routing;
mod runner;

pub mod bridge;

/// Repeated mutable graph update and rebuild quality experiments.
#[cfg(feature = "rocksdb")]
pub mod update_history;

#[path = "../../tests/support/oracle.rs"]
#[expect(
    dead_code,
    reason = "shared oracle also exposes integration-only filter helpers"
)]
mod oracle;
