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

// Shared measurement and dataset helpers used by standalone benchmark binaries.
pub use backend::MeasuredBackend;
pub use dataset::{limit_with_query_offset, load, load_large};
pub use metrics::MetricCapture;
pub use resource::ResourceSnapshot;

#[path = "../../tests/support/oracle.rs"]
#[expect(
    dead_code,
    reason = "shared oracle also exposes integration-only filter helpers"
)]
mod oracle;
