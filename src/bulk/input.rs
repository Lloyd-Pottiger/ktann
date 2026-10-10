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
        InputSnapshotWriter::new(directory, config, maximum_bytes)?
            .append(records)?
            .seal()
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

/// Incrementally writes a finite input without an intermediate staging file.
///
/// The caller owns the directory, including unsealed files after cancellation
/// or failure. Only `seal` returns an immutable snapshot suitable for a build.
/// Each append consumes the writer so an error cannot later seal a partial batch.
pub struct InputSnapshotWriter {
    directory: PathBuf,
    config: IndexConfig,
    writer: Writer,
}

impl InputSnapshotWriter {
    /// Creates an exclusive snapshot directory and reserves its framing quota.
    pub fn new(directory: &Path, config: IndexConfig, maximum_bytes: u64) -> Result<Self> {
        config.validate()?;
        let writer = Writer::new(directory, 0, schema_binding(&config), maximum_bytes)?;
        Ok(Self {
            directory: directory.to_path_buf(),
            config,
            writer,
        })
    }

    /// Appends a bounded-memory stream of full records in arrival order.
    ///
    /// Only one encoded record is buffered. Failure consumes this writer and
    /// leaves an unsealed directory; it does not acknowledge a partial batch.
    pub fn append(mut self, records: impl IntoIterator<Item = Result<Record>>) -> Result<Self> {
        for record in records {
            self.writer
                .append(&source::encode(&self.config, &mut record?)?)?;
        }
        Ok(self)
    }

    /// Flushes and durably seals the snapshot without rereading its records.
    pub fn seal(self) -> Result<InputSnapshot> {
        Ok(InputSnapshot {
            directory: self.directory,
            config: self.config,
            manifest: self.writer.seal()?,
        })
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

/// Receipt-time source capture and bounded Tree Key/ID sorting.
/// Appending is synchronous: callers must apply backpressure and run it on a
/// blocking executor. Neither partition training nor publication starts here.
/// Both directories remain caller owned, including after failure or cancellation.
pub struct PreparedInputWriter {
    source: InputSnapshotWriter,
    preparation: super::forest::ForestPreparation,
}

/// One-use receipt-time work bound to an immutable, original-order source.
/// This is not a durable checkpoint. If lost, reopen the source and run the
/// ordinary worker; it recomputes the same forest from the source snapshot.
pub struct PreparedInput {
    pub(super) source: InputSnapshot,
    pub(super) preparation: super::forest::ForestPreparation,
}

impl PreparedInput {
    /// The durable source identity used to reserve or resume the build.
    pub fn source(&self) -> &InputSnapshot {
        &self.source
    }

    pub(crate) fn matches(
        &self,
        config: &IndexConfig,
        manifest: &ArtifactManifest,
        options: super::ForestOptions,
    ) -> bool {
        self.source.manifest == *manifest
            && self.source.config == *config
            && self.preparation.options == options
    }
}

impl PreparedInputWriter {
    /// Creates source capture and sorting within the declared shared sort budget.
    /// Row buffers split the memory remaining after one shared IO reservation.
    /// Scratch shares one ceiling; the two sorters perform IO serially.
    pub fn new(
        source_directory: &Path,
        preparation_directory: &Path,
        config: IndexConfig,
        maximum_bytes: u64,
        options: super::ForestOptions,
    ) -> Result<Self> {
        config.validate()?;
        let preparation =
            super::forest::ForestPreparation::new(preparation_directory, &config, options)?;
        Ok(Self {
            source: InputSnapshotWriter::new(source_directory, config, maximum_bytes)?,
            preparation,
        })
    }

    /// Captures and sorts a batch before acknowledging it. Consuming ownership
    /// prevents an error from subsequently sealing a partially accepted batch.
    pub fn append(mut self, records: impl IntoIterator<Item = Result<Record>>) -> Result<Self> {
        for record in records {
            let mut record = record?;
            let encoded = source::encode(&self.source.config, &mut record)?;
            self.source.writer.append(&encoded)?;
            self.preparation.append(&self.source.config, &record)?;
        }
        Ok(self)
    }

    /// Seals the original source. Global duplicate checking and final merges
    /// complete when the prepared work is consumed by forest construction.
    pub fn seal(self) -> Result<PreparedInput> {
        Ok(PreparedInput {
            source: self.source.seal()?,
            preparation: self.preparation,
        })
    }
}
