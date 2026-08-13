//! FoundationDB storage adapter for KTANN.
//!
//! This crate owns FoundationDB transaction mechanics, physical key prefixes,
//! backend limits, error classification, and capabilities. Logical index
//! algorithms and persistent values remain in [`ktann`].
