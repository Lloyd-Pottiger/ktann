//! Durable namespace queue request for automatic Bulk Build execution.

use super::{
    corrupt,
    wire::{Decoder, Encoder},
    workspace,
};
use crate::api::{BulkWorkerOptions, Result};

/// Namespace queue entry; worker ownership leases live in `BuildScheduleSlot`.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildSchedule {
    pub(crate) name: crate::api::IndexName,
    pub(crate) options: BulkWorkerOptions,
}

impl std::fmt::Debug for BuildSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildSchedule").finish_non_exhaustive()
    }
}

pub(super) fn encode(e: &mut Encoder, value: &BuildSchedule) -> Result<()> {
    e.sized_bytes(value.name.as_str().as_bytes(), 255)?;
    workspace::encode_options(e, &value.options)
}

pub(super) fn decode(d: &mut Decoder) -> Result<BuildSchedule> {
    let name = d.sized_bytes(255)?;
    let name = crate::api::IndexName::new(std::str::from_utf8(&name).map_err(|_| corrupt())?)
        .map_err(|_| corrupt())?;
    Ok(BuildSchedule {
        name,
        options: workspace::decode_options(d)?,
    })
}
