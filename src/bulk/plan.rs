//! Canonical partition-plan bodies shared by forest readers and writers.
use super::files::corrupt;
use crate::api::{PartitionKey, Result};
use crate::construction::PartitionPlan;
use bytes::Bytes;

pub(super) fn encode(plan: &PartitionPlan) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&plan.key.get().to_be_bytes());
    bytes.extend_from_slice(&plan.level.to_be_bytes());
    bytes.extend_from_slice(&(plan.entries.len() as u32).to_be_bytes());
    for component in &plan.centroid {
        bytes.extend_from_slice(
            &if *component == 0.0 {
                0
            } else {
                component.to_bits()
            }
            .to_be_bytes(),
        );
    }
    for entry in &plan.entries {
        bytes.extend_from_slice(&(entry.len() as u16).to_be_bytes());
        bytes.extend_from_slice(entry);
    }
    bytes
}

pub(super) fn decode(
    mut bytes: Bytes,
    dimension: usize,
    maximum_entries: u32,
) -> Result<PartitionPlan> {
    use bytes::Buf;

    if bytes.len() < 16 + dimension * 4 {
        return Err(corrupt());
    }
    let key = PartitionKey::new(bytes.get_u64()).map_err(|_| corrupt())?;
    let level = bytes.get_u32();
    let count = bytes.get_u32();
    if level == 0 || count == 0 || count > maximum_entries {
        return Err(corrupt());
    }
    let mut centroid = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        let bits = bytes.get_u32();
        let value = f32::from_bits(bits);
        if !value.is_finite() || bits == (-0.0_f32).to_bits() {
            return Err(corrupt());
        }
        centroid.push(value);
    }
    // Check a minimum body size before allocating the entry inventory.
    if count as usize > bytes.len() / 3 {
        return Err(corrupt());
    }
    let mut entries = Vec::with_capacity(count as usize);
    for _ in 0..count {
        if bytes.len() < 2 {
            return Err(corrupt());
        }
        let length = bytes.get_u16() as usize;
        if length == 0 || length > 256 || length > bytes.len() || (level > 1 && length != 8) {
            return Err(corrupt());
        }
        let entry = bytes.split_to(length);
        if entries.last().is_some_and(|previous| previous >= &entry)
            || (level > 1 && entry.as_ref() == [0; 8])
        {
            return Err(corrupt());
        }
        entries.push(entry);
    }
    if !bytes.is_empty() {
        return Err(corrupt());
    }
    Ok(PartitionPlan {
        key,
        level,
        centroid: centroid.into_boxed_slice(),
        entries,
    })
}
