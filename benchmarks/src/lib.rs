//! Reproducible whole-system performance baselines for KTANN.

#![deny(unsafe_code)]

mod backend;
pub mod cli;
mod compare;
mod dataset;
mod metrics;
mod report;
mod resource;
mod runner;
