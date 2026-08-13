//! Caller-visible values, validation, configuration, and errors.
//!
//! The types in this module are independent of any storage backend. Values
//! which depend on a Logical Index's immutable configuration expose explicit
//! validation methods so callers can reject an operation before opening a
//! storage transaction.

mod config;
mod error;
mod identifiers;
mod operation;
mod record;
mod schema;
mod search;
mod verify;

pub use config::{IndexConfig, RuntimeConfig};
pub use error::{Error, ErrorKind, Result};
pub use identifiers::{BatchToken, FieldId, IndexName, LogicalIndexId, PartitionKey};
pub use operation::{
    GetOptions, ImportBatchResult, ImportOptions, Mutation, MutationOutcome, OperationOptions,
    UpsertResult, validate_mutations,
};
pub use record::{PayloadProjection, Record, StoredRecord};
pub use schema::{CompareOp, DataType, FieldSchema, Metric, Predicate, SynopsisConfig, Value};
pub use search::{
    SearchBudgetExhaustion, SearchBudgetUsage, SearchBudgets, SearchHit, SearchOptions,
    SearchOutcome, SearchRequest,
};
pub use verify::{VerifyIssue, VerifyIssueKind, VerifyObjectCounts, VerifyOptions, VerifyReport};
