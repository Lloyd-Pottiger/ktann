//! Backend transaction contract, logical codecs, and typed storage operations.
//!
//! This module is the sole owner of raw logical keys and persistent values.
//! Algorithm modules hand the storage layer typed operations and never build
//! keys themselves. The [`keys`] submodule implements the version-1 Logical Key
//! namespace and exposes the canonical Tree Key codec; the [`values`] submodule
//! implements the version-1 persistent value codecs; the [`backend`]
//! submodule defines the backend-neutral transactional KV contract; and the
//! [`topology`] submodule implements the typed atomic split-state transitions
//! and structural entry moves (ADR 0014).

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
