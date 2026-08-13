//! K-means Tree Approximate Nearest Neighbor.
//!
//! KTANN is an embeddable vector-index library backed by transactional
//! key-value storage. The crate owns the backend-neutral API, algorithms,
//! logical storage, and persistent codecs.

pub mod api;
pub mod maintenance;
pub mod runtime;
pub mod search;
pub mod storage;
