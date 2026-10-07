//! One immutable serving artifact and its atomically advanced load checkpoint.

use std::fmt;

use crate::api::{Error, Result};
use crate::bulk::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest};

use super::corrupt;
use super::wire::{Decoder, Encoder};

/// Durable fencing and progress for one index's serving-data load.
///
/// The artifact cannot be replaced after registration. A new invocation takes
/// over by increasing the epoch; data and progress commit together. Completion
/// means every artifact entry was loaded, not that publication is authorized.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildLoad {
    pub(crate) artifact: ArtifactManifest,
    pub(crate) epoch: u64,
    pub(crate) entries: u64,
    pub(crate) prefix_sha256: [u8; 32],
    pub(crate) complete: bool,
    pub(crate) sealed: bool,
}

impl BuildLoad {
    /// The immutable, sealed input identity.
    #[must_use]
    pub fn artifact(&self) -> &ArtifactManifest {
        &self.artifact
    }
    /// Monotonic takeover epoch, starting at one.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }
    /// Number of sorted entries committed with this checkpoint.
    #[must_use]
    pub const fn entries(&self) -> u64 {
        self.entries
    }
    /// SHA-256 of the data-file prefix corresponding to committed entries.
    #[must_use]
    pub const fn prefix_sha256(&self) -> &[u8; 32] {
        &self.prefix_sha256
    }
    /// Whether the complete stream was loaded and its checksum verified.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    fn validate(&self) -> Result<()> {
        if !self.artifact.is_serving()
            || self.epoch == 0
            || (self.sealed && !self.complete)
            || self.entries > self.artifact.items()
            || (self.entries == 0 && self.prefix_sha256 != self.artifact.initial_sha256())
            || (self.complete
                && (self.entries != self.artifact.items()
                    || self.prefix_sha256 != *self.artifact.sha256()))
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}

impl fmt::Debug for BuildLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuildLoad")
            .field("epoch", &self.epoch)
            .field("entries", &self.entries)
            .field("complete", &self.complete)
            .finish_non_exhaustive()
    }
}

pub(super) fn encode(encoder: &mut Encoder, load: &BuildLoad) -> Result<()> {
    load.validate()?;
    encoder.sized_bytes(&load.artifact.encode(), ARTIFACT_MANIFEST_BYTES)?;
    encoder.u64(load.epoch);
    encoder.u64(load.entries);
    encoder.bytes(&load.prefix_sha256);
    encoder.u8(u8::from(load.complete));
    encoder.u8(u8::from(load.sealed));
    Ok(())
}

pub(super) fn decode(decoder: &mut Decoder) -> Result<BuildLoad> {
    let artifact = ArtifactManifest::decode(&decoder.sized_bytes(ARTIFACT_MANIFEST_BYTES)?)?;
    let epoch = decoder.u64()?;
    let entries = decoder.u64()?;
    let prefix_sha256 = decoder.array()?;
    let complete = match decoder.u8()? {
        0 => false,
        1 => true,
        _ => return Err(corrupt()),
    };
    let sealed = match decoder.u8()? {
        0 => false,
        1 => true,
        _ => return Err(corrupt()),
    };
    let load = BuildLoad {
        artifact,
        epoch,
        entries,
        prefix_sha256,
        complete,
        sealed,
    };
    load.validate().map_err(|_| corrupt())?;
    Ok(load)
}
