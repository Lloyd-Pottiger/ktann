//! Backend transaction contract, logical codecs, and typed storage operations.
//!
//! This module owns raw Logical Keys and persistent values. Algorithms access
//! index storage through typed operations. The [`keys`] submodule defines the
//! Logical Key namespace and canonical Tree Key codec; [`values`] encodes and
//! decodes persistent values; [`backend`] defines the backend-neutral
//! transactional KV contract; and [`topology`] implements atomic partition
//! transitions and entry moves (ADR 0014).

pub mod backend;
pub mod keys;
pub mod membership;
mod operations;
#[cfg(test)]
pub(crate) mod test_support;
pub mod topology;
mod tree_key;
pub mod tree_manifest;
pub mod values;

pub(crate) use operations::LogicalReader;
pub use operations::{
    LogicalRange, LogicalScanCursor, LogicalScanItem, LogicalScanPage, MutationBuilder,
    ReadLogicalTxn, RecordGroupRead, TransactionSize, WriteLogicalTxn,
};
