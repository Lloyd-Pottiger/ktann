//! Backend transaction contract, logical codecs, and typed storage operations.
//!
//! This module is the sole owner of raw logical keys and persistent values.
//! Algorithm modules hand the storage layer typed operations and never build
//! keys themselves. The [`keys`] submodule implements the version-1 logical key
//! codecs; the backend transaction contract, persistent value codecs, and typed
//! atomic operations are added by later stages.

pub mod keys;
