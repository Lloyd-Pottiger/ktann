//! Namespace-owned preparation state and durable file-reclamation inventory.
use super::{
    corrupt,
    wire::{Decoder, Encoder},
};
use crate::api::{BulkLoadOptions, BulkWorkerOptions, Error, ErrorKind, Result};
use crate::bulk::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest};
use crate::storage::backend::HardLimits;
use std::{fmt, path::PathBuf};

/// An accepted immutable artifact in an attempt-owned directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedArtifact {
    pub epoch: u64,
    pub manifest: ArtifactManifest,
}
/// Namespace-scoped ownership survives abort and ordinary index drop.
/// The random token identifies exactly one child of the configured workspace.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildWorkspace {
    pub(crate) options: BulkWorkerOptions,
    pub(crate) hard_limits: HardLimits,
    pub(crate) token: [u8; 32],
    pub(crate) epoch: u64,
    pub(crate) forest: Option<PreparedArtifact>,
    pub(crate) serving: Option<PreparedArtifact>,
    pub(crate) failure: Option<ErrorKind>,
}
impl BuildWorkspace {
    pub(crate) fn directory(&self) -> PathBuf {
        self.options.workspace.join(
            self.token
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
        )
    }
    pub(crate) fn attempt(&self, epoch: u64) -> PathBuf {
        self.directory().join(format!("attempt-{epoch}"))
    }
    pub(crate) fn validate(&self) -> Result<()> {
        self.options.validate()?;
        if self.epoch == 0
            || self.token == [0; 32]
            || self.hard_limits.max_key_bytes == 0
            || self.hard_limits.max_value_bytes == 0
            || self
                .forest
                .as_ref()
                .is_some_and(|a| a.epoch == 0 || a.epoch > self.epoch || !a.manifest.is_forest())
            || self.serving.as_ref().is_some_and(|a| {
                a.epoch == 0
                    || a.epoch > self.epoch
                    || !a.manifest.is_serving()
                    || self.forest.is_none()
            })
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}
impl fmt::Debug for BuildWorkspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuildWorkspace")
            .field("epoch", &self.epoch)
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}
pub(super) fn encode(e: &mut Encoder, w: &BuildWorkspace) -> Result<()> {
    w.validate()?;
    encode_options(e, &w.options)?;
    e.u64(w.hard_limits.max_key_bytes as u64);
    e.u64(w.hard_limits.max_value_bytes as u64);
    e.bytes(&w.token);
    e.u64(w.epoch);
    for artifact in [&w.forest, &w.serving] {
        e.u8(u8::from(artifact.is_some()));
        if let Some(a) = artifact {
            e.u64(a.epoch);
            e.sized_bytes(&a.manifest.encode(), ARTIFACT_MANIFEST_BYTES)?;
        }
    }
    e.u8(match w.failure {
        None => 0,
        Some(ErrorKind::Corruption) => 1,
        Some(ErrorKind::InvalidArgument) => 2,
        Some(ErrorKind::LimitExceeded) => 3,
        Some(ErrorKind::Other) => 4,
        Some(ErrorKind::UnsupportedFormat) => 5,
        Some(ErrorKind::IdExhausted) => 6,
        Some(ErrorKind::RecordAlreadyExists) => 7,
        Some(ErrorKind::TransactionTooLarge) => 8,
        _ => return Err(Error::invalid_argument()),
    });
    Ok(())
}
pub(super) fn decode(d: &mut Decoder) -> Result<BuildWorkspace> {
    let options = decode_options(d)?;
    let hard_limits = HardLimits {
        max_key_bytes: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
        max_value_bytes: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
    };
    let token = d.array()?;
    let epoch = d.u64()?;
    fn artifact(d: &mut Decoder) -> Result<Option<PreparedArtifact>> {
        match d.u8()? {
            0 => Ok(None),
            1 => Ok(Some(PreparedArtifact {
                epoch: d.u64()?,
                manifest: ArtifactManifest::decode(&d.sized_bytes(ARTIFACT_MANIFEST_BYTES)?)?,
            })),
            _ => Err(corrupt()),
        }
    }
    let forest = artifact(d)?;
    let serving = artifact(d)?;
    let failure = match d.u8()? {
        0 => None,
        1 => Some(ErrorKind::Corruption),
        2 => Some(ErrorKind::InvalidArgument),
        3 => Some(ErrorKind::LimitExceeded),
        4 => Some(ErrorKind::Other),
        5 => Some(ErrorKind::UnsupportedFormat),
        6 => Some(ErrorKind::IdExhausted),
        7 => Some(ErrorKind::RecordAlreadyExists),
        8 => Some(ErrorKind::TransactionTooLarge),
        _ => return Err(corrupt()),
    };
    let value = BuildWorkspace {
        options,
        hard_limits,
        token,
        epoch,
        forest,
        serving,
        failure,
    };
    value.validate().map_err(|_| corrupt())?;
    Ok(value)
}

pub(super) fn encode_options(e: &mut Encoder, o: &BulkWorkerOptions) -> Result<()> {
    o.validate()?;
    e.sized_bytes(
        o.workspace
            .to_str()
            .ok_or_else(Error::invalid_argument)?
            .as_bytes(),
        3900,
    )?;
    for n in [
        o.sort_memory_bytes as u64,
        o.sort_scratch_bytes,
        o.serving_memory_bytes as u64,
        o.serving_scratch_bytes,
        o.max_artifact_bytes,
        o.load.max_mutations as u64,
        o.load.max_bytes as u64,
    ] {
        e.u64(n);
    }
    Ok(())
}

pub(super) fn decode_options(d: &mut Decoder) -> Result<BulkWorkerOptions> {
    let path = d.sized_bytes(3900)?;
    let workspace = PathBuf::from(std::str::from_utf8(&path).map_err(|_| corrupt())?);
    let options = BulkWorkerOptions {
        workspace,
        sort_memory_bytes: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
        sort_scratch_bytes: d.u64()?,
        serving_memory_bytes: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
        serving_scratch_bytes: d.u64()?,
        max_artifact_bytes: d.u64()?,
        load: BulkLoadOptions {
            max_mutations: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
            max_bytes: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
        },
    };
    options.validate().map_err(|_| corrupt())?;
    Ok(options)
}
