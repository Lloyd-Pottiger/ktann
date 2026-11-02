//! Immutable Bulk Build request identity, separate from the serving Manifest.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::api::{Error, Result};
use crate::bulk::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest};
use crate::construction::ConstructionOptions;

use super::corrupt;
use super::wire::{Decoder, Encoder};

const MAX_SOURCE_PATH_BYTES: usize = 4096;

/// The immutable request that owns a newly reserved Logical Index.
///
/// Config, rotation, and Bloom parameters belong to the Index Manifest. This
/// descriptor lives at a separate key so ordinary serving reads do not load
/// source locators or construction options.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildDescriptor {
    source: PathBuf,
    input: ArtifactManifest,
    options: ConstructionOptions,
    refinement_rounds: Option<u8>,
}

impl BuildDescriptor {
    /// Creates a bounded request descriptor with an absolute UTF-8 source path.
    pub fn new(
        source: PathBuf,
        input: ArtifactManifest,
        options: ConstructionOptions,
    ) -> Result<Self> {
        if !input.is_input()
            || !source.is_absolute()
            || source
                .to_str()
                .is_none_or(|path| path.len() > MAX_SOURCE_PATH_BYTES || path.contains('\0'))
        {
            return Err(Error::invalid_argument());
        }
        options.validate()?;
        Ok(Self {
            source,
            input,
            options,
            refinement_rounds: None,
        })
    }

    /// Binds optional refinement to this reservation; None and Some(0) differ.
    pub fn with_refinement_rounds(mut self, rounds: Option<u8>) -> Result<Self> {
        if rounds.is_some_and(|rounds| usize::from(rounds) > crate::api::MAX_REFINEMENT_ROUNDS) {
            return Err(Error::invalid_argument());
        }
        self.refinement_rounds = rounds;
        Ok(self)
    }

    /// Requested bounded refinement rounds, before final forest acceptance.
    pub const fn refinement_rounds(&self) -> Option<u8> {
        self.refinement_rounds
    }

    /// Absolute source snapshot directory; its contents must match `input`.
    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Expected immutable source identity.
    #[must_use]
    pub const fn input(&self) -> &ArtifactManifest {
        &self.input
    }

    /// Deterministic construction parameters and resource ceilings.
    #[must_use]
    pub const fn options(&self) -> ConstructionOptions {
        self.options
    }
}

impl fmt::Debug for BuildDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BuildDescriptor([REDACTED])")
    }
}

pub(super) fn encode(encoder: &mut Encoder, descriptor: &BuildDescriptor) -> Result<()> {
    encoder.sized_bytes(
        descriptor
            .source
            .to_str()
            .expect("validated UTF-8")
            .as_bytes(),
        MAX_SOURCE_PATH_BYTES,
    )?;
    encoder.sized_bytes(&descriptor.input.encode(), ARTIFACT_MANIFEST_BYTES)?;
    let options = descriptor.options;
    encoder.u64(options.sample_items as u64);
    encoder.u64(options.memory_bytes as u64);
    encoder.u64(options.scratch_bytes);
    encoder.u8(descriptor.refinement_rounds.map_or(0, |rounds| rounds + 1));
    Ok(())
}

pub(super) fn decode(decoder: &mut Decoder) -> Result<BuildDescriptor> {
    let path = decoder.sized_bytes(MAX_SOURCE_PATH_BYTES)?;
    let source = PathBuf::from(std::str::from_utf8(&path).map_err(|_| corrupt())?);
    let input = ArtifactManifest::decode(&decoder.sized_bytes(ARTIFACT_MANIFEST_BYTES)?)?;
    let options = ConstructionOptions {
        sample_items: usize::try_from(decoder.u64()?).map_err(|_| corrupt())?,
        memory_bytes: usize::try_from(decoder.u64()?).map_err(|_| corrupt())?,
        scratch_bytes: decoder.u64()?,
    };
    let refinement_rounds = match decoder.u8()? {
        0 => None,
        tag @ 1..=6 => Some(tag - 1),
        _ => return Err(corrupt()),
    };
    BuildDescriptor::new(source, input, options)
        .and_then(|d| d.with_refinement_rounds(refinement_rounds))
        .map_err(|_| corrupt())
}
