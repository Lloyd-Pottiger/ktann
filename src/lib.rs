//! K-means Tree Approximate Nearest Neighbor.
//!
//! KTANN is an embeddable vector-index library backed by transactional
//! key-value storage. The crate owns the backend-neutral API, algorithms,
//! logical storage, and persistent codecs.
//!
//! Public request values validate independently from storage and Search:
//!
//! ```
//! use std::sync::Arc;
//!
//! use bytes::Bytes;
//! use ktann::api::{
//!     DataType, FieldSchema, IndexConfig, Metric, Record, SearchRequest, Value,
//! };
//!
//! let config = IndexConfig::new(3, Metric::Cosine)?
//!     .with_fields(vec![FieldSchema::new("published", DataType::Bool)?])?;
//! let mut record = Record::new(
//!     Bytes::from_static(b"article-1"),
//!     Arc::from([1.0_f32, 0.0, 0.0]),
//!     vec![Value::Bool(true)],
//! )?;
//! record.validate(config.dimension(), config.fields())?;
//!
//! let mut search = SearchRequest::new(Arc::from([1.0_f32, 0.0, 0.0]), 10)?;
//! search.validate(
//!     config.dimension(),
//!     config.fields(),
//!     Default::default(),
//! )?;
//! # Ok::<(), ktann::api::Error>(())
//! ```

pub mod api;
pub mod maintenance;
mod observe;
pub mod runtime;
pub mod search;
pub mod storage;
