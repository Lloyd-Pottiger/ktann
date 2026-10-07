//! Finite source snapshots preserving complete records before index allocation.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::api::{DataType, Error, IndexConfig, Record, Result};
use crate::storage::values::source;

use super::files::{ArtifactManifest, Reader, Writer};

/// An immutable finite input snapshot containing complete original records.
///
/// Record order is preserved. Duplicate IDs are retained here and rejected by
/// the construction/preparation stage, where a global external sort is available.
/// This handle owns no cleanup: the caller owns the directory even on failure.
#[derive(Clone)]
pub struct InputSnapshot {
    directory: PathBuf,
    config: IndexConfig,
    manifest: ArtifactManifest,
}

impl InputSnapshot {
    /// Streams records into a new directory and durably seals its manifest.
    ///
    /// `maximum_bytes` bounds the complete snapshot including file framing and
    /// the manifest. Only one record is buffered. Invalid input or quota/IO
    /// failure leaves an unsealed directory for caller-owned reclamation.
    pub fn create(
        directory: &Path,
        config: IndexConfig,
        maximum_bytes: u64,
        records: impl IntoIterator<Item = Result<Record>>,
    ) -> Result<Self> {
        config.validate()?;
        let mut writer = Writer::new(directory, 0, schema_binding(&config), maximum_bytes)?;
        for record in records {
            writer.append(&source::encode(&config, record?)?)?;
        }
        Ok(Self {
            directory: directory.to_path_buf(),
            config,
            manifest: writer.seal()?,
        })
    }

    /// Reopens a snapshot against the identity saved by its owner.
    ///
    /// The schema must match; metric, synopsis policy, Tree Key selection, and
    /// partition sizes are consumer configuration, not source record shape.
    /// Opening checks framing and file identity metadata. Consume a reader to
    /// EOF, or call [`Self::verify`], to verify all contents and the final hash.
    pub fn open(directory: &Path, config: IndexConfig, expected: ArtifactManifest) -> Result<Self> {
        config.validate()?;
        if !expected.matches(0, schema_binding(&config)) {
            return Err(Error::invalid_argument());
        }
        let snapshot = Self {
            directory: directory.to_path_buf(),
            config,
            manifest: expected,
        };
        snapshot.reader()?;
        Ok(snapshot)
    }

    /// The descriptor to persist before assigning work that consumes this file.
    #[must_use]
    pub const fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Opens a new bounded, checksumming record stream.
    ///
    /// A dropped reader has verified only its consumed frames, not the complete
    /// snapshot. Successful exhaustion verifies count, exact length, and hash.
    pub fn reader(&self) -> Result<InputReader> {
        Ok(InputReader {
            reader: Reader::open(&self.directory, &self.manifest, 1024 * 1024)?,
            config: self.config.clone(),
            failed: false,
        })
    }

    /// Checks every frame, canonical record body, and the complete file hash.
    pub fn verify(&self) -> Result<()> {
        for record in self.reader()? {
            record?;
        }
        Ok(())
    }

    pub(crate) const fn config(&self) -> &IndexConfig {
        &self.config
    }

    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }
}

/// A fused iterator of full records from one immutable input snapshot.
pub struct InputReader {
    reader: Reader,
    config: IndexConfig,
    failed: bool,
}

impl Iterator for InputReader {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        let result = self
            .reader
            .next()?
            .and_then(|bytes| source::decode(&self.config, bytes));
        self.failed = result.is_err();
        Some(result)
    }
}

impl std::iter::FusedIterator for InputReader {}

fn schema_binding(config: &IndexConfig) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"KTANN bulk input schema v1");
    hash.update((config.dimension() as u32).to_be_bytes());
    hash.update((config.fields().len() as u16).to_be_bytes());
    for field in config.fields() {
        hash.update((field.name().len() as u16).to_be_bytes());
        hash.update(field.name().as_bytes());
        hash.update([match field.data_type() {
            DataType::Bool => 0,
            DataType::I64 => 1,
            DataType::F64 => 2,
            DataType::String => 3,
        }]);
        hash.update([u8::from(field.is_nullable())]);
    }
    hash.finalize().into()
}
