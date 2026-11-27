//! Namespace-owned workspace ledger and terminal failure record.

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use crate::api::{BulkLoadOptions, BulkWorkerOptions, Error, ErrorKind, Result};
use crate::storage::backend::HardLimits;

use super::corrupt;
use super::wire::{Decoder, Encoder};

/// Namespace-scoped workspace ownership survives abort and ordinary index drop.
/// The random token identifies one owned child directory of the configured workspace.
#[derive(Clone, Eq, PartialEq)]
pub struct BuildWorkspace {
    pub(crate) options: BulkWorkerOptions,
    pub(crate) hard_limits: HardLimits,
    pub(crate) token: [u8; 32],
    /// Non-zero while writers for `epoch` are authorized; `[0; 32]` once frozen for validation.
    pub(crate) writer_token: [u8; 32],
    pub(crate) epoch: u64,
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

    pub(crate) fn validate(&self) -> Result<()> {
        self.options.validate()?;
        if self.epoch == 0
            || self.token == [0; 32]
            || self.hard_limits.max_key_bytes == 0
            || self.hard_limits.max_value_bytes == 0
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
    e.bytes(&w.writer_token);
    e.u64(w.epoch);
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
    let writer_token = d.array()?;
    let epoch = d.u64()?;
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
        writer_token,
        epoch,
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
    e.u64(o.parallelism as u64);
    e.u64(o.distributed_workers as u64);
    e.u64(o.unit_lease.as_secs());
    e.u32(o.unit_lease.subsec_nanos());
    for n in [
        o.memory_bytes as u64,
        o.graph_memory_bytes as u64,
        o.scratch_bytes,
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
    let parallelism = usize::try_from(d.u64()?).map_err(|_| corrupt())?;
    let distributed_workers = usize::try_from(d.u64()?).map_err(|_| corrupt())?;
    let lease_secs = d.u64()?;
    let lease_nanos = d.u32()?;
    if lease_nanos >= 1_000_000_000 {
        return Err(corrupt());
    }
    let options = BulkWorkerOptions {
        workspace,
        parallelism,
        distributed_workers,
        unit_lease: Duration::new(lease_secs, lease_nanos),
        memory_bytes: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
        graph_memory_bytes: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
        scratch_bytes: d.u64()?,
        load: BulkLoadOptions {
            max_mutations: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
            max_bytes: usize::try_from(d.u64()?).map_err(|_| corrupt())?,
        },
    };
    options.validate().map_err(|_| corrupt())?;
    Ok(options)
}
