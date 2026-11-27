//! Per-unit worker lease and homomorphic multiset digest record.

use std::fmt;

use crate::api::{Error, Result};
use crate::bulk::LtHash;

use super::corrupt;
use super::wire::{Decoder, Encoder};

/// Persistent lease and completion digest for one `BuildUnit`.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildUnit {
    pub(crate) epoch: u64,
    pub(crate) owner: [u8; 32],
    pub(crate) expires_ms: u64,
    pub(crate) complete: bool,
    pub(crate) records: u64,
    pub(crate) kv_count: u64,
    pub(crate) aux_count: u64,
    pub(crate) kv_lthash: LtHash,
    pub(crate) aux_lthash: LtHash,
    pub(crate) aux2_lthash: LtHash,
}

impl BuildUnit {
    pub(crate) const fn unclaimed(epoch: u64) -> Self {
        Self {
            epoch,
            owner: [0; 32],
            expires_ms: 0,
            complete: false,
            records: 0,
            kv_count: 0,
            aux_count: 0,
            kv_lthash: LtHash::ZERO,
            aux_lthash: LtHash::ZERO,
            aux2_lthash: LtHash::ZERO,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.epoch == 0 || (self.owner != [0; 32] && !self.complete && self.expires_ms == 0) {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}

impl fmt::Debug for BuildUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuildUnit")
            .field("epoch", &self.epoch)
            .field("claimed", &(self.owner != [0; 32]))
            .field("complete", &self.complete)
            .field("kv_count", &self.kv_count)
            .finish_non_exhaustive()
    }
}

pub(super) fn encode(e: &mut Encoder, u: &BuildUnit) -> Result<()> {
    u.validate()?;
    e.u64(u.epoch);
    e.bytes(&u.owner);
    e.u64(u.expires_ms);
    e.u8(u8::from(u.complete));
    e.u64(u.records);
    e.u64(u.kv_count);
    e.u64(u.aux_count);
    e.bytes(&u.kv_lthash.encode());
    e.bytes(&u.aux_lthash.encode());
    e.bytes(&u.aux2_lthash.encode());
    Ok(())
}

pub(super) fn decode(d: &mut Decoder) -> Result<BuildUnit> {
    let epoch = d.u64()?;
    let owner = d.array()?;
    let expires_ms = d.u64()?;
    let complete = match d.u8()? {
        0 => false,
        1 => true,
        _ => return Err(corrupt()),
    };
    let records = d.u64()?;
    let kv_count = d.u64()?;
    let aux_count = d.u64()?;
    let kv_lthash = LtHash::decode(&d.array()?);
    let aux_lthash = LtHash::decode(&d.array()?);
    let aux2_lthash = LtHash::decode(&d.array()?);
    let u = BuildUnit {
        epoch,
        owner,
        expires_ms,
        complete,
        records,
        kv_count,
        aux_count,
        kv_lthash,
        aux_lthash,
        aux2_lthash,
    };
    u.validate().map_err(|_| corrupt())?;
    Ok(u)
}
