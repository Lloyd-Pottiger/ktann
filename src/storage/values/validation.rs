//! Bounded proof cursor over immutable sealed backend data.
use super::{
    corrupt,
    wire::{Decoder, Encoder},
};
use crate::api::{Error, Result};
use crate::bulk::{ARTIFACT_MANIFEST_BYTES, ArtifactManifest};
use bytes::Bytes;
/// Persistent exact-comparison proof for a sealed serving artifact.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildValidation {
    pub(crate) artifact: ArtifactManifest,
    pub(crate) cursor: Bytes,
    pub(crate) entries: u64,
    pub(crate) prefix_sha256: [u8; 32],
    pub(crate) complete: bool,
}
impl BuildValidation {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.artifact.is_serving()
            || self.cursor.len() > 16 * 1024
            || self.entries > self.artifact.items()
            || (self.entries == 0 && self.prefix_sha256 != self.artifact.initial_sha256())
            || (self.complete
                && (!self.cursor.is_empty()
                    || self.entries != self.artifact.items()
                    || self.prefix_sha256 != *self.artifact.sha256()))
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}
pub(super) fn encode(e: &mut Encoder, v: &BuildValidation) -> Result<()> {
    v.validate()?;
    e.sized_bytes(&v.artifact.encode(), ARTIFACT_MANIFEST_BYTES)?;
    e.sized_bytes(&v.cursor, 16 * 1024)?;
    e.u64(v.entries);
    e.bytes(&v.prefix_sha256);
    e.u8(u8::from(v.complete));
    Ok(())
}
pub(super) fn decode(d: &mut Decoder) -> Result<BuildValidation> {
    let artifact = ArtifactManifest::decode(&d.sized_bytes(ARTIFACT_MANIFEST_BYTES)?)?;
    let cursor = d.sized_bytes(16 * 1024)?;
    let entries = d.u64()?;
    let prefix_sha256 = d.array()?;
    let complete = match d.u8()? {
        0 => false,
        1 => true,
        _ => return Err(corrupt()),
    };
    let v = BuildValidation {
        artifact,
        cursor,
        entries,
        prefix_sha256,
        complete,
    };
    v.validate().map_err(|_| corrupt())?;
    Ok(v)
}

impl std::fmt::Debug for BuildValidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildValidation")
            .field("entries", &self.entries)
            .field("complete", &self.complete)
            .finish_non_exhaustive()
    }
}
