//! Canonical persistent-format codec contract tests.
//!
//! Golden vectors pin the version-1 byte layout for every key and value
//! family; property tests cover ordering, round trips, and fail-closed
//! decoding. The two modules are independent pure-codec suites sharing one
//! test target.

#[path = "codecs/keys.rs"]
mod keys;
#[path = "codecs/values.rs"]
mod values;
