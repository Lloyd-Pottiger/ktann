//! One fenced progress record for loading and exact backend validation.
use super::{
    corrupt,
    wire::{Decoder, Encoder},
};
use crate::api::{Error, Result};
use crate::bulk::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest};
use bytes::Bytes;

/// The durable stage and only the checkpoint data meaningful in that stage.
#[derive(Clone, Eq, PartialEq)]
pub enum BuildPhase {
    /// Serving writes are allowed under the current load epoch.
    Loading {
        /// Number of entries committed atomically with this checkpoint.
        entries: u64,
        /// Digest of the committed artifact prefix.
        prefix_sha256: [u8; 32],
    },
    /// The entire artifact was loaded and verified through EOF.
    Loaded,
    /// Serving writes are frozen; the backend is compared with the artifact.
    Validating {
        /// Next backend key to scan; empty at the start of validation.
        cursor: Bytes,
        /// Number of serving entries compared so far.
        entries: u64,
        /// Digest of the compared artifact prefix.
        prefix_sha256: [u8; 32],
    },
    /// Both the backend scan and artifact EOF verification completed.
    Validated,
}

impl std::fmt::Debug for BuildPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loading { entries, .. } => f
                .debug_struct("Loading")
                .field("entries", entries)
                .finish_non_exhaustive(),
            Self::Loaded => f.write_str("Loaded"),
            Self::Validating { entries, .. } => f
                .debug_struct("Validating")
                .field("entries", entries)
                .finish_non_exhaustive(),
            Self::Validated => f.write_str("Validated"),
        }
    }
}

/// Immutable artifact identity and atomic progress through loading/validation.
/// Publication remains governed exclusively by the Index Manifest lifecycle.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildProgress {
    pub(crate) artifact: ArtifactManifest,
    pub(crate) epoch: u64,
    pub(crate) phase: BuildPhase,
}
impl BuildProgress {
    /// The immutable serving artifact bound at the first load claim.
    pub fn artifact(&self) -> &ArtifactManifest {
        &self.artifact
    }
    /// Monotonic load takeover epoch, starting at one.
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }
    /// Current durable phase.
    pub const fn phase(&self) -> &BuildPhase {
        &self.phase
    }
    /// Entries written or compared in the current phase.
    pub fn entries(&self) -> u64 {
        match &self.phase {
            BuildPhase::Loading { entries, .. } | BuildPhase::Validating { entries, .. } => {
                *entries
            }
            BuildPhase::Loaded | BuildPhase::Validated => self.artifact.items(),
        }
    }
    /// Digest of the written/compared artifact prefix in the current phase.
    pub fn prefix_sha256(&self) -> &[u8; 32] {
        match &self.phase {
            BuildPhase::Loading { prefix_sha256, .. }
            | BuildPhase::Validating { prefix_sha256, .. } => prefix_sha256,
            BuildPhase::Loaded | BuildPhase::Validated => self.artifact.sha256(),
        }
    }
    fn validate(&self) -> Result<()> {
        if !self.artifact.is_serving()
            || self.epoch == 0
            || self.entries() > self.artifact.items()
            || (self.entries() == 0 && self.prefix_sha256() != &self.artifact.initial_sha256())
            || matches!(&self.phase, BuildPhase::Validating { cursor, .. } if cursor.len() > 16 * 1024)
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}
impl std::fmt::Debug for BuildProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Do not expose a validation cursor, which can contain caller IDs.
        let phase = match self.phase {
            BuildPhase::Loading { .. } => "loading",
            BuildPhase::Loaded => "loaded",
            BuildPhase::Validating { .. } => "validating",
            BuildPhase::Validated => "validated",
        };
        f.debug_struct("BuildProgress")
            .field("epoch", &self.epoch)
            .field("phase", &phase)
            .field("entries", &self.entries())
            .finish_non_exhaustive()
    }
}
pub(super) fn encode(e: &mut Encoder, p: &BuildProgress) -> Result<()> {
    p.validate()?;
    e.sized_bytes(&p.artifact.encode(), ARTIFACT_MANIFEST_BYTES)?;
    e.u64(p.epoch);
    match &p.phase {
        BuildPhase::Loading {
            entries,
            prefix_sha256,
        } => {
            e.u8(0);
            e.u64(*entries);
            e.bytes(prefix_sha256);
        }
        BuildPhase::Loaded => e.u8(1),
        BuildPhase::Validating {
            cursor,
            entries,
            prefix_sha256,
        } => {
            e.u8(2);
            e.sized_bytes(cursor, 16 * 1024)?;
            e.u64(*entries);
            e.bytes(prefix_sha256);
        }
        BuildPhase::Validated => e.u8(3),
    }
    Ok(())
}
pub(super) fn decode(d: &mut Decoder) -> Result<BuildProgress> {
    let artifact = ArtifactManifest::decode(&d.sized_bytes(ARTIFACT_MANIFEST_BYTES)?)?;
    let epoch = d.u64()?;
    let phase = match d.u8()? {
        0 => BuildPhase::Loading {
            entries: d.u64()?,
            prefix_sha256: d.array()?,
        },
        1 => BuildPhase::Loaded,
        2 => BuildPhase::Validating {
            cursor: d.sized_bytes(16 * 1024)?,
            entries: d.u64()?,
            prefix_sha256: d.array()?,
        },
        3 => BuildPhase::Validated,
        _ => return Err(corrupt()),
    };
    let progress = BuildProgress {
        artifact,
        epoch,
        phase,
    };
    progress.validate().map_err(|_| corrupt())?;
    Ok(progress)
}
