//! Durable scheduler queue and revocable worker lease.
use super::{
    corrupt,
    wire::{Decoder, Encoder},
    workspace,
};
use crate::api::{BulkWorkerOptions, Error, Result};

/// Namespace queue entry; its random owner token fences all scheduled writes.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildSchedule {
    pub(crate) name: crate::api::IndexName,
    pub(crate) options: BulkWorkerOptions,
    pub(crate) owner: [u8; 32],
    pub(crate) expires_ms: u64,
}
impl std::fmt::Debug for BuildSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildSchedule")
            .field("claimed", &(self.owner != [0; 32]))
            .finish_non_exhaustive()
    }
}
pub(super) fn encode(e: &mut Encoder, value: &BuildSchedule) -> Result<()> {
    if value.owner != [0; 32] && value.expires_ms == 0 {
        return Err(Error::invalid_argument());
    }
    e.sized_bytes(value.name.as_str().as_bytes(), 255)?;
    workspace::encode_options(e, &value.options)?;
    e.bytes(&value.owner);
    e.u64(value.expires_ms);
    Ok(())
}
pub(super) fn decode(d: &mut Decoder) -> Result<BuildSchedule> {
    let name = d.sized_bytes(255)?;
    let name = crate::api::IndexName::new(std::str::from_utf8(&name).map_err(|_| corrupt())?)
        .map_err(|_| corrupt())?;
    let value = BuildSchedule {
        name,
        options: workspace::decode_options(d)?,
        owner: d.array()?,
        expires_ms: d.u64()?,
    };
    if value.owner != [0; 32] && value.expires_ms == 0 {
        return Err(corrupt());
    }
    Ok(value)
}
