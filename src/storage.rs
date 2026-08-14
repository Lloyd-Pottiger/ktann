//! Backend transaction contract, logical codecs, and typed storage operations.
//!
//! This module is the sole owner of raw logical keys and persistent values.
//! Algorithm modules hand the storage layer typed operations and never build
//! keys themselves. The [`keys`] submodule implements the version-1 Logical Key
//! namespace and exposes the canonical Tree Key codec; the [`values`] submodule
//! implements the version-1 persistent value codecs; and the [`backend`]
//! submodule defines the backend-neutral transactional KV contract. Typed atomic
//! operations are added by later stages.

pub mod backend;
pub mod keys;
mod tree_key;
pub mod values;
