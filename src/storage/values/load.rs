//! Persistent Bulk Build phase, inventory counts, and multiset-digest expectations.

use std::fmt;

use bytes::Bytes;

use crate::api::{Error, Result};
use crate::bulk::LtHash;

use super::corrupt;
use super::wire::{Decoder, Encoder};

pub(crate) const MAX_SPLIT_KEYS: usize = 255;
const MAX_SPLIT_KEY_BYTES: usize = 16 * 1024;

/// High-level persisted phase of one Bulk Build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuildPhase {
    /// Plan or intermediate shard units are being prepared.
    Preparing,
    /// Serving KV entries are being emitted into the hidden index.
    Loading,
    /// All serving KV entries and expected multiset digests are sealed.
    Loaded,
    /// Writer lanes are revoked and parallel slice validation is in progress.
    Validating,
}

impl BuildPhase {
    const fn to_u8(self) -> u8 {
        match self {
            Self::Preparing => 0,
            Self::Loading => 1,
            Self::Loaded => 2,
            Self::Validating => 3,
        }
    }

    fn from_u8(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::Preparing),
            1 => Ok(Self::Loading),
            2 => Ok(Self::Loaded),
            3 => Ok(Self::Validating),
            _ => Err(corrupt()),
        }
    }
}

/// Durable stage progress and sealed inventory digest of one Bulk Build.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildProgress {
    pub(crate) phase: BuildPhase,
    pub(crate) epoch: u64,
    pub(crate) total_records: u64,
    pub(crate) loaded_entries: u64,
    pub(crate) total_entries: u64,
    pub(crate) trees: u64,
    pub(crate) partitions: u64,
    pub(crate) expected_kv_lthash: LtHash,
    pub(crate) validation_split_keys: Vec<Bytes>,
}

impl BuildProgress {
    /// Current persisted phase.
    #[must_use]
    pub const fn phase(&self) -> BuildPhase {
        self.phase
    }

    /// Monotonic attempt epoch, starting at one.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Serving entries committed so far.
    #[must_use]
    pub const fn loaded_entries(&self) -> u64 {
        self.loaded_entries
    }

    /// Total serving entries expected once sealed.
    #[must_use]
    pub const fn total_entries(&self) -> u64 {
        self.total_entries
    }

    /// Disjoint keyspace slices (`validation_split_keys.len() + 1`) used during validation.
    #[must_use]
    pub const fn validation_slices(&self) -> u32 {
        (self.validation_split_keys.len() as u32) + 1
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.epoch == 0
            || self.loaded_entries > self.total_entries
            || (matches!(self.phase, BuildPhase::Loaded | BuildPhase::Validating)
                && self.loaded_entries != self.total_entries)
            || self.validation_split_keys.len() > MAX_SPLIT_KEYS
            || self
                .validation_split_keys
                .iter()
                .any(|k| k.is_empty() || k.len() > MAX_SPLIT_KEY_BYTES)
            || self.validation_split_keys.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}

impl fmt::Debug for BuildProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuildProgress")
            .field("phase", &self.phase)
            .field("epoch", &self.epoch)
            .field("loaded_entries", &self.loaded_entries)
            .field("total_entries", &self.total_entries)
            .finish_non_exhaustive()
    }
}

pub(super) fn encode(encoder: &mut Encoder, p: &BuildProgress) -> Result<()> {
    p.validate()?;
    encoder.u8(p.phase.to_u8());
    encoder.u64(p.epoch);
    encoder.u64(p.total_records);
    encoder.u64(p.loaded_entries);
    encoder.u64(p.total_entries);
    encoder.u64(p.trees);
    encoder.u64(p.partitions);
    encoder.bytes(&p.expected_kv_lthash.encode());
    encoder.u16(p.validation_split_keys.len() as u16);
    for key in &p.validation_split_keys {
        encoder.sized_bytes(key, MAX_SPLIT_KEY_BYTES)?;
    }
    Ok(())
}

pub(super) fn decode(decoder: &mut Decoder) -> Result<BuildProgress> {
    let phase = BuildPhase::from_u8(decoder.u8()?)?;
    let epoch = decoder.u64()?;
    let total_records = decoder.u64()?;
    let loaded_entries = decoder.u64()?;
    let total_entries = decoder.u64()?;
    let trees = decoder.u64()?;
    let partitions = decoder.u64()?;
    let lthash_bytes: [u8; 64] = decoder.array()?;
    let expected_kv_lthash = LtHash::decode(&lthash_bytes);
    let split_count = usize::from(decoder.u16()?);
    if split_count > MAX_SPLIT_KEYS {
        return Err(corrupt());
    }
    let mut validation_split_keys = Vec::with_capacity(split_count);
    for _ in 0..split_count {
        validation_split_keys.push(decoder.sized_bytes(MAX_SPLIT_KEY_BYTES)?);
    }
    let value = BuildProgress {
        phase,
        epoch,
        total_records,
        loaded_entries,
        total_entries,
        trees,
        partitions,
        expected_kv_lthash,
        validation_split_keys,
    };
    value.validate().map_err(|_| corrupt())?;
    Ok(value)
}
