//! Immutable files with bounded frames and a separately persisted identity.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::api::{Error, ErrorKind, Result};

const MAGIC: &[u8; 8] = b"KTANNBF\x01";
const HEADER_BYTES: u64 = 41;
const FRAME_OVERHEAD: u64 = 36;

/// Size of the canonical version-1 artifact manifest.
pub const ARTIFACT_MANIFEST_BYTES: usize = 89;

/// Identity of a completed input snapshot or construction artifact.
///
/// Paths are deliberately excluded: workers may relocate the identical bytes.
/// Persist this value in the owning job/task, rather than trusting a manifest
/// read from a worker directory after a restart.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactManifest {
    kind: u8,
    binding: [u8; 32],
    items: u64,
    bytes: u64,
    digest: [u8; 32],
}

impl ArtifactManifest {
    /// Number of records or partitions in the artifact.
    #[must_use]
    pub const fn items(&self) -> u64 {
        self.items
    }

    /// Exact data-file length, including framing and checksums.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// SHA-256 of the complete data file.
    #[must_use]
    pub const fn sha256(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Canonical bytes suitable for persisting in an owning task descriptor.
    #[must_use]
    pub fn encode(&self) -> [u8; ARTIFACT_MANIFEST_BYTES] {
        let mut bytes = [0; ARTIFACT_MANIFEST_BYTES];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8] = self.kind;
        bytes[9..41].copy_from_slice(&self.binding);
        bytes[41..49].copy_from_slice(&self.items.to_be_bytes());
        bytes[49..57].copy_from_slice(&self.bytes.to_be_bytes());
        bytes[57..].copy_from_slice(&self.digest);
        bytes
    }

    /// Decodes a bounded manifest; it does not verify the referenced file.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != ARTIFACT_MANIFEST_BYTES {
            return Err(corrupt());
        }
        if bytes[..7] != MAGIC[..7] {
            return Err(corrupt());
        }
        if bytes[7] != MAGIC[7] {
            return Err(Error::new(ErrorKind::UnsupportedFormat));
        }
        if !matches!(bytes[8], 0..=3) {
            return Err(corrupt());
        }
        let manifest = Self {
            kind: bytes[8],
            binding: bytes[9..41].try_into().expect("fixed binding"),
            items: u64::from_be_bytes(bytes[41..49].try_into().expect("fixed count")),
            bytes: u64::from_be_bytes(bytes[49..57].try_into().expect("fixed length")),
            digest: bytes[57..].try_into().expect("fixed digest"),
        };
        if manifest.bytes < HEADER_BYTES
            || manifest.items > (manifest.bytes - HEADER_BYTES) / (FRAME_OVERHEAD + 1)
        {
            return Err(corrupt());
        }
        Ok(manifest)
    }

    /// Digest of the fixed data header, before any framed entries.
    pub(crate) fn initial_sha256(&self) -> [u8; 32] {
        Sha256::digest(&self.encode()[..HEADER_BYTES as usize]).into()
    }

    pub(crate) const fn is_input(&self) -> bool {
        self.kind == 0
    }

    pub(super) fn matches(&self, kind: u8, binding: [u8; 32]) -> bool {
        self.kind == kind && self.binding == binding
    }
}

pub(super) fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

pub(super) fn io_error(error: std::io::Error) -> Error {
    Error::new(if error.kind() == std::io::ErrorKind::UnexpectedEof {
        ErrorKind::Corruption
    } else {
        ErrorKind::Backend
    })
}

/// Owns a newly created directory. Failure leaves only unsealed, caller-owned
/// files; no recovery or cleanup ever overwrites another attempt's directory.
pub(super) struct Writer {
    directory: PathBuf,
    file: BufWriter<File>,
    hash: Sha256,
    manifest: ArtifactManifest,
    quota: u64,
}

impl Writer {
    pub(super) fn new(directory: &Path, kind: u8, binding: [u8; 32], quota: u64) -> Result<Self> {
        if quota < HEADER_BYTES + ARTIFACT_MANIFEST_BYTES as u64 {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        fs::create_dir(directory).map_err(io_error)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("data.partial"))
            .map_err(io_error)?;
        let mut writer = Self {
            directory: directory.to_path_buf(),
            file: BufWriter::new(file),
            hash: Sha256::new(),
            manifest: ArtifactManifest {
                kind,
                binding,
                items: 0,
                bytes: 0,
                digest: [0; 32],
            },
            quota,
        };
        writer.write(MAGIC)?;
        writer.write(&[kind])?;
        writer.write(&binding)?;
        Ok(writer)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.file.write_all(bytes).map_err(io_error)?;
        self.hash.update(bytes);
        self.manifest.bytes += bytes.len() as u64;
        Ok(())
    }

    pub(super) fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let length = u32::try_from(bytes.len()).map_err(|_| Error::invalid_argument())?;
        let size = self
            .manifest
            .bytes
            .checked_add(FRAME_OVERHEAD + u64::from(length) + ARTIFACT_MANIFEST_BYTES as u64)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        if bytes.is_empty() || size > self.quota {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        self.write(&length.to_be_bytes())?;
        self.write(bytes)?;
        self.write(&Sha256::digest(bytes))?;
        self.manifest.items += 1;
        Ok(())
    }

    pub(super) fn seal(mut self) -> Result<ArtifactManifest> {
        self.file.flush().map_err(io_error)?;
        self.file.get_ref().sync_all().map_err(io_error)?;
        self.manifest.digest = self.hash.finalize().into();
        drop(self.file);
        fs::rename(
            self.directory.join("data.partial"),
            self.directory.join("data.bin"),
        )
        .map_err(io_error)?;
        // Persist the data name before making the completion marker visible.
        File::open(&self.directory)
            .and_then(|file| file.sync_all())
            .map_err(io_error)?;
        let mut manifest = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join("manifest.partial"))
            .map_err(io_error)?;
        manifest
            .write_all(&self.manifest.encode())
            .map_err(io_error)?;
        manifest.sync_all().map_err(io_error)?;
        drop(manifest);
        fs::rename(
            self.directory.join("manifest.partial"),
            self.directory.join("manifest.bin"),
        )
        .map_err(io_error)?;
        File::open(&self.directory)
            .and_then(|file| file.sync_all())
            .map_err(io_error)?;
        File::open(
            self.directory
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )
        .and_then(|file| file.sync_all())
        .map_err(io_error)?;
        Ok(self.manifest)
    }
}

/// A streaming verification pass; at most one caller-bounded frame is resident.
pub(super) struct Reader {
    file: BufReader<File>,
    hash: Sha256,
    manifest: ArtifactManifest,
    remaining: u64,
    consumed: u64,
    maximum_frame: usize,
    done: bool,
}

impl Reader {
    pub(super) fn open(
        directory: &Path,
        expected: &ArtifactManifest,
        maximum_frame: usize,
    ) -> Result<Self> {
        let mut manifest = File::open(directory.join("manifest.bin")).map_err(io_error)?;
        if manifest.metadata().map_err(io_error)?.len() != ARTIFACT_MANIFEST_BYTES as u64 {
            return Err(corrupt());
        }
        let mut bytes = [0; ARTIFACT_MANIFEST_BYTES];
        manifest.read_exact(&mut bytes).map_err(io_error)?;
        if ArtifactManifest::decode(&bytes)? != *expected {
            return Err(corrupt());
        }
        let file = File::open(directory.join("data.bin")).map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() != expected.bytes {
            return Err(corrupt());
        }
        let mut reader = Self {
            file: BufReader::new(file),
            hash: Sha256::new(),
            manifest: expected.clone(),
            remaining: expected.items,
            consumed: 0,
            maximum_frame,
            done: false,
        };
        let mut header = [0; HEADER_BYTES as usize];
        reader.read(&mut header)?;
        if &header[..8] != MAGIC || header[8] != expected.kind || header[9..] != expected.binding {
            return Err(corrupt());
        }
        Ok(reader)
    }

    fn read(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.file.read_exact(bytes).map_err(io_error)?;
        self.hash.update(&*bytes);
        self.consumed += bytes.len() as u64;
        Ok(())
    }

    pub(super) fn prefix_sha256(&self) -> [u8; 32] {
        self.hash.clone().finalize().into()
    }

    fn frame(&mut self) -> Result<Option<Bytes>> {
        if self.remaining == 0 {
            let mut tail = [0];
            if self.consumed != self.manifest.bytes
                || self.file.read(&mut tail).map_err(io_error)? != 0
                || <[u8; 32]>::from(self.hash.clone().finalize()) != self.manifest.digest
            {
                return Err(corrupt());
            }
            return Ok(None);
        }
        let mut length = [0; 4];
        self.read(&mut length)?;
        let length = u32::from_be_bytes(length) as usize;
        if length == 0
            || length > self.maximum_frame
            || length as u64 + 32 > self.manifest.bytes.saturating_sub(self.consumed)
        {
            return Err(corrupt());
        }
        let mut bytes = vec![0; length];
        self.read(&mut bytes)?;
        let mut digest = [0; 32];
        self.read(&mut digest)?;
        if <[u8; 32]>::from(Sha256::digest(&bytes)) != digest {
            return Err(corrupt());
        }
        self.remaining -= 1;
        Ok(Some(Bytes::from(bytes)))
    }

    pub(super) fn next(&mut self) -> Option<Result<Bytes>> {
        if self.done {
            return None;
        }
        match self.frame() {
            Ok(Some(bytes)) => Some(Ok(bytes)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(error) => {
                self.done = true;
                Some(Err(error))
            }
        }
    }
}
