//! Segmented finite source snapshots preserving complete records before index allocation.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use sha2::{Digest, Sha256};

use crate::api::{DataType, Error, IndexConfig, Record, Result};
use crate::storage::values::{MAX_VALUE_BYTES, source};

use super::files::{
    InputManifest, SegmentFileReader, SegmentMeta, SegmentReader, SegmentedWriter, open_segments,
};

/// An immutable segmented input snapshot containing complete original records.
///
/// Record arrival order is preserved across segments. Duplicate Record IDs are
/// retained here and rejected globally during Bulk Build ingestion. This handle
/// owns no cleanup: the caller owns the directory even after failure or abort.
#[derive(Clone)]
pub struct InputSnapshot {
    directory: PathBuf,
    config: IndexConfig,
    manifest: InputManifest,
    segments: Arc<[SegmentMeta]>,
    verified_segments: Arc<[AtomicBool]>,
}

impl InputSnapshot {
    /// Caller-owned snapshot directory used by durable build recovery.
    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }

    /// Streams records into a new segmented directory and durably seals its manifest.
    ///
    /// `maximum_bytes` bounds the complete snapshot including segment headers,
    /// per-frame checksums, segment index, and manifest. Only one record is
    /// buffered in memory.
    ///
    /// # Errors
    ///
    /// Returns validation, quota, or I/O errors without sealing the directory.
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

    /// Reopens a snapshot against the trusted identity persisted by its owner.
    ///
    /// The schema must match; metric, synopsis policy, Tree Key selection, and
    /// partition capacities are consumer configuration rather than source shape.
    ///
    /// # Errors
    ///
    /// Returns [`crate::api::ErrorKind::InvalidArgument`] on schema mismatch and
    /// [`crate::api::ErrorKind::Corruption`] or I/O errors on invalid files.
    pub fn open(directory: &Path, config: IndexConfig, expected: InputManifest) -> Result<Self> {
        config.validate()?;
        if !expected.matches(schema_binding(&config)) {
            return Err(Error::invalid_argument());
        }
        let segments = open_segments(directory, &expected)?;
        let verified_segments = (0..segments.len())
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<_>>()
            .into();
        Ok(Self {
            directory: directory.to_path_buf(),
            config,
            manifest: expected,
            segments: Arc::from(segments),
            verified_segments,
        })
    }

    /// Trusted identity to persist before assigning work that consumes this snapshot.
    #[must_use]
    pub const fn manifest(&self) -> &InputManifest {
        &self.manifest
    }

    /// Verified metadata for each immutable segment in arrival order.
    pub(crate) fn segments(&self) -> &[SegmentMeta] {
        &self.segments
    }

    /// Creates a bounded `O(1)`-FD reader for monotonic `(seg_idx, frame_offset)` accesses.
    #[must_use]
    pub(crate) fn frame_reader(&self) -> SegmentFileReader<'_> {
        SegmentFileReader::new(
            &self.directory,
            schema_binding(&self.config),
            &self.segments,
            &self.verified_segments,
        )
    }

    /// Opens one segment's raw encoded frame reader for parallel ingestion.
    pub(crate) fn encoded_segment(&self, index: usize) -> Result<SegmentReader> {
        let meta = *self
            .segments
            .get(index)
            .ok_or_else(Error::invalid_argument)?;
        SegmentReader::open(
            &self.directory,
            index,
            schema_binding(&self.config),
            meta,
            MAX_VALUE_BYTES,
            Some((Arc::clone(&self.verified_segments), index)),
        )
    }

    /// Opens one segment's raw encoded frame reader with a compact buffer for strided sampling.
    pub(crate) fn encoded_segment_strided(&self, index: usize) -> Result<SegmentReader> {
        let meta = *self
            .segments
            .get(index)
            .ok_or_else(Error::invalid_argument)?;
        SegmentReader::open_strided(
            &self.directory,
            index,
            schema_binding(&self.config),
            meta,
            MAX_VALUE_BYTES,
        )
    }

    /// Opens a sequential checksumming record stream across all segments.
    ///
    /// Exhausting the reader verifies every segment's frame and whole-segment
    /// SHA-256 digests.
    ///
    /// # Errors
    ///
    /// Returns I/O or corruption errors if the snapshot directory, manifest,
    /// segment index, or first segment is missing or invalid.
    pub fn reader(&self) -> Result<InputReader> {
        open_segments(&self.directory, &self.manifest)?;
        let mut reader = InputReader {
            snapshot: self.clone(),
            next_segment: 0,
            current: None,
            failed: false,
        };
        reader.advance_segment()?;
        Ok(reader)
    }

    /// Verifies every segment, frame, canonical record body, and segment hash.
    ///
    /// # Errors
    ///
    /// Returns [`crate::api::ErrorKind::Corruption`] or I/O errors on invalid data.
    pub fn verify(&self) -> Result<()> {
        for record in self.reader()? {
            record?;
        }
        Ok(())
    }

    /// Configuration bound into the source identity.
    #[must_use]
    pub const fn config(&self) -> &IndexConfig {
        &self.config
    }
}

/// Incrementally writes a finite segmented input without an extra staging copy.
///
/// Each `append` consumes the writer so an error cannot seal a partial batch.
pub struct InputSnapshotWriter {
    config: IndexConfig,
    writer: SegmentedWriter,
}

impl InputSnapshotWriter {
    /// Creates an exclusive snapshot directory and reserves its framing quota.
    ///
    /// # Errors
    ///
    /// Returns validation, quota, or directory-creation errors.
    pub fn new(directory: &Path, config: IndexConfig, maximum_bytes: u64) -> Result<Self> {
        config.validate()?;
        let writer = SegmentedWriter::new(directory, schema_binding(&config), maximum_bytes)?;
        Ok(Self { config, writer })
    }

    /// Appends a bounded-memory stream of full records in arrival order.
    ///
    /// # Errors
    ///
    /// Returns record validation, quota, or I/O errors and consumes the writer.
    pub fn append(mut self, records: impl IntoIterator<Item = Result<Record>>) -> Result<Self> {
        for record in records {
            self.writer
                .append(&source::encode(&self.config, &mut record?)?)?;
        }
        Ok(self)
    }

    /// Flushes and durably seals all segments, `segments.bin`, and `manifest.bin`.
    ///
    /// # Errors
    ///
    /// Returns I/O or quota errors if sealing fails.
    pub fn seal(self) -> Result<InputSnapshot> {
        let (directory, manifest, segments) = self.writer.seal()?;
        let verified_segments = (0..segments.len())
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<_>>()
            .into();
        Ok(InputSnapshot {
            directory,
            config: self.config,
            manifest,
            segments: Arc::from(segments),
            verified_segments,
        })
    }
}

/// A fused iterator of full records across all segments of an [`InputSnapshot`].
pub struct InputReader {
    snapshot: InputSnapshot,
    next_segment: usize,
    current: Option<SegmentReader>,
    failed: bool,
}

impl InputReader {
    fn advance_segment(&mut self) -> Result<bool> {
        if self.next_segment >= self.snapshot.segments.len() {
            self.current = None;
            return Ok(false);
        }
        let seg = self.snapshot.encoded_segment(self.next_segment)?;
        self.next_segment += 1;
        self.current = Some(seg);
        Ok(true)
    }
}

impl Iterator for InputReader {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            let reader = self.current.as_mut()?;
            match reader.next_frame() {
                Some(Ok(bytes)) => {
                    let decoded = source::decode(&self.snapshot.config, bytes);
                    if decoded.is_err() {
                        self.failed = true;
                    }
                    return Some(decoded);
                }
                Some(Err(err)) => {
                    self.failed = true;
                    return Some(Err(err));
                }
                None => match self.advance_segment() {
                    Ok(true) => continue,
                    Ok(false) => return None,
                    Err(err) => {
                        self.failed = true;
                        return Some(Err(err));
                    }
                },
            }
        }
    }
}

impl std::iter::FusedIterator for InputReader {}

pub(super) fn schema_binding(config: &IndexConfig) -> [u8; 32] {
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
