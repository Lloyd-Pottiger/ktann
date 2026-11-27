//! Immutable Bulk Build request identity, separate from the serving Manifest.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::api::{BulkBuildOptions, Error, Result};
use crate::bulk::{INPUT_MANIFEST_BYTES, InputManifest};

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
    input: InputManifest,
    options: BulkBuildOptions,
}

impl BuildDescriptor {
    /// Creates a bounded request descriptor with an absolute UTF-8 source path.
    pub fn new(source: PathBuf, input: InputManifest, options: BulkBuildOptions) -> Result<Self> {
        if !source.is_absolute()
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
        })
    }

    /// Requested bounded refinement rounds, before serving emission.
    #[must_use]
    pub const fn refinement_rounds(&self) -> Option<u8> {
        self.options.refinement_rounds
    }

    /// Absolute source snapshot directory; its contents must match `input`.
    #[must_use]
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Expected immutable source identity.
    #[must_use]
    pub const fn input(&self) -> &InputManifest {
        &self.input
    }

    /// Deterministic construction and refinement semantics.
    #[must_use]
    pub const fn options(&self) -> BulkBuildOptions {
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
    encoder.sized_bytes(&descriptor.input.encode(), INPUT_MANIFEST_BYTES)?;
    encoder.u64(descriptor.options.sample_items as u64);
    encoder.u8(descriptor
        .options
        .refinement_rounds
        .map_or(0, |rounds| rounds + 1));
    Ok(())
}

pub(super) fn decode(decoder: &mut Decoder) -> Result<BuildDescriptor> {
    let path = decoder.sized_bytes(MAX_SOURCE_PATH_BYTES)?;
    let source = PathBuf::from(std::str::from_utf8(&path).map_err(|_| corrupt())?);
    let input = InputManifest::decode(&decoder.sized_bytes(INPUT_MANIFEST_BYTES)?)?;
    let sample_items = usize::try_from(decoder.u64()?).map_err(|_| corrupt())?;
    let refinement_rounds = match decoder.u8()? {
        0 => None,
        tag @ 1..=6 => Some(tag - 1),
        _ => return Err(corrupt()),
    };
    BuildDescriptor::new(
        source,
        input,
        BulkBuildOptions {
            sample_items,
            refinement_rounds,
        },
    )
    .map_err(|_| corrupt())
}
