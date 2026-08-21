//! Persistent Tree Manifest and Partition authority values.

use std::fmt;

use crate::api::{Error, PartitionKey, Result};

use super::corrupt;
use super::data::{decode_vector, encode_vector};
use super::manifest::IndexManifest;
use super::wire::{Decoder, Encoder};

/// The directory and Partition Key allocator state for one Tree Key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TreeManifest {
    root: PartitionKey,
    partition_key_high_water: PartitionKey,
}

impl TreeManifest {
    /// Creates a Tree Manifest whose stable root is Partition Key 1.
    pub fn new(root: PartitionKey, partition_key_high_water: PartitionKey) -> Result<Self> {
        if root.get() != 1 || partition_key_high_water < root {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            root,
            partition_key_high_water,
        })
    }

    /// Returns the stable root Partition Key.
    #[must_use]
    pub const fn root(self) -> PartitionKey {
        self.root
    }

    /// Returns the greatest Partition Key reserved for the tree.
    #[must_use]
    pub const fn partition_key_high_water(self) -> PartitionKey {
        self.partition_key_high_water
    }
}

/// The state discriminator duplicated in a Partition Header for traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PartitionState {
    /// The partition accepts its ordinary operations.
    Ready,
    /// Split target identities have been reserved.
    Splitting,
    /// A published split target is receiving entries.
    ReceivingSplit,
    /// The source is draining into two published targets.
    DrainingSplit,
    /// The source is draining into reselected Ready targets.
    Merging,
}

impl PartitionState {
    /// Whether a leaf in this state accepts foreground writes and structural
    /// move-ins.
    ///
    /// A `DrainingSplit` or `Merging` source accepts move-outs only: its
    /// entries are leaving, and its exact zero count is the completion proof
    /// that no insert may race.
    pub(crate) const fn accepts_writes(self) -> bool {
        matches!(self, Self::Ready | Self::Splitting | Self::ReceivingSplit)
    }
}

/// Small mutable operational metadata for one partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartitionHeader {
    level: u32,
    entry_count: u32,
    cache_epoch: u64,
    state: PartitionState,
}

impl PartitionHeader {
    /// Creates a structurally valid Partition Header.
    pub fn new(
        level: u32,
        entry_count: u32,
        cache_epoch: u64,
        state: PartitionState,
    ) -> Result<Self> {
        if level == 0 {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            level,
            entry_count,
            cache_epoch,
            state,
        })
    }

    /// Returns the tree level; leaves are level one.
    #[must_use]
    pub const fn level(self) -> u32 {
        self.level
    }

    /// Returns the exact number of entries.
    #[must_use]
    pub const fn entry_count(self) -> u32 {
        self.entry_count
    }

    /// Returns the persistent cache-validation epoch.
    #[must_use]
    pub const fn cache_epoch(self) -> u64 {
        self.cache_epoch
    }

    /// Returns the traversal state discriminator.
    #[must_use]
    pub const fn state(self) -> PartitionState {
        self.state
    }
}

/// A full-f32 immutable routing centroid.
#[derive(Clone, PartialEq)]
pub struct PartitionCentroid(Box<[f32]>);

impl PartitionCentroid {
    /// Creates an immutable centroid.
    #[must_use]
    pub fn new(components: impl Into<Box<[f32]>>) -> Self {
        Self(components.into())
    }

    /// Returns the centroid components.
    #[must_use]
    pub fn components(&self) -> &[f32] {
        &self.0
    }
}

impl fmt::Debug for PartitionCentroid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PartitionCentroid([REDACTED])")
    }
}

/// The durable topology state and references for one partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PartitionTransition {
    /// The partition accepts ordinary operations.
    Ready {
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
    /// A source has reserved two target identities.
    Splitting {
        /// The left target.
        left: PartitionKey,
        /// The right target.
        right: PartitionKey,
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
    /// A published target is receiving entries from one source.
    ReceivingSplit {
        /// The source partition.
        source: PartitionKey,
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
    /// A source is draining into two published targets.
    DrainingSplit {
        /// The left target.
        left: PartitionKey,
        /// The right target.
        right: PartitionKey,
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
    /// A source is draining into targets reselected per batch.
    Merging {
        /// Milliseconds since the Unix epoch when this state began.
        started_at_unix_millis: u64,
    },
}

impl PartitionTransition {
    /// Returns the Header discriminator corresponding to this state.
    #[must_use]
    pub const fn state(self) -> PartitionState {
        match self {
            Self::Ready { .. } => PartitionState::Ready,
            Self::Splitting { .. } => PartitionState::Splitting,
            Self::ReceivingSplit { .. } => PartitionState::ReceivingSplit,
            Self::DrainingSplit { .. } => PartitionState::DrainingSplit,
            Self::Merging { .. } => PartitionState::Merging,
        }
    }

    /// Returns milliseconds since the Unix epoch when this state began.
    #[must_use]
    pub const fn started_at_unix_millis(self) -> u64 {
        match self {
            Self::Ready {
                started_at_unix_millis,
            }
            | Self::Splitting {
                started_at_unix_millis,
                ..
            }
            | Self::ReceivingSplit {
                started_at_unix_millis,
                ..
            }
            | Self::DrainingSplit {
                started_at_unix_millis,
                ..
            }
            | Self::Merging {
                started_at_unix_millis,
            } => started_at_unix_millis,
        }
    }
}

pub(super) fn encode_tree_manifest(encoder: &mut Encoder, manifest: TreeManifest) {
    encoder.u64(manifest.root.get());
    encoder.u64(manifest.partition_key_high_water.get());
}

pub(super) fn decode_tree_manifest(decoder: &mut Decoder) -> Result<TreeManifest> {
    let root = decoder.partition_key()?;
    let high_water = decoder.partition_key()?;
    TreeManifest::new(root, high_water).map_err(|_| corrupt())
}

pub(super) fn encode_partition_header(encoder: &mut Encoder, header: PartitionHeader) {
    encoder.u32(header.level);
    encoder.u32(header.entry_count);
    encoder.u64(header.cache_epoch);
    encoder.u8(encode_state_kind(header.state));
}

pub(super) fn decode_partition_header(decoder: &mut Decoder) -> Result<PartitionHeader> {
    let level = decoder.u32()?;
    let entry_count = decoder.u32()?;
    let cache_epoch = decoder.u64()?;
    let state = decode_state_kind(decoder.u8()?)?;
    PartitionHeader::new(level, entry_count, cache_epoch, state).map_err(|_| corrupt())
}

pub(super) fn encode_partition_centroid(
    encoder: &mut Encoder,
    manifest: &IndexManifest,
    centroid: &PartitionCentroid,
) -> Result<()> {
    encode_vector(
        encoder,
        manifest.config().dimension(),
        centroid.components(),
    )
}

pub(super) fn decode_partition_centroid(
    decoder: &mut Decoder,
    manifest: &IndexManifest,
) -> Result<PartitionCentroid> {
    Ok(PartitionCentroid::new(decode_vector(
        decoder,
        manifest.config().dimension(),
    )?))
}

pub(super) fn encode_partition_state(
    encoder: &mut Encoder,
    transition: PartitionTransition,
) -> Result<()> {
    validate_transition(transition)?;
    encoder.u8(encode_state_kind(transition.state()));
    encoder.u64(transition.started_at_unix_millis());
    match transition {
        PartitionTransition::Ready { .. } | PartitionTransition::Merging { .. } => {}
        PartitionTransition::Splitting { left, right, .. }
        | PartitionTransition::DrainingSplit { left, right, .. } => {
            encoder.u64(left.get());
            encoder.u64(right.get());
        }
        PartitionTransition::ReceivingSplit { source, .. } => encoder.u64(source.get()),
    }
    Ok(())
}

pub(super) fn decode_partition_state(decoder: &mut Decoder) -> Result<PartitionTransition> {
    let kind = decode_state_kind(decoder.u8()?)?;
    let started_at_unix_millis = decoder.u64()?;
    let transition = match kind {
        PartitionState::Ready => PartitionTransition::Ready {
            started_at_unix_millis,
        },
        PartitionState::Splitting => PartitionTransition::Splitting {
            left: decoder.partition_key()?,
            right: decoder.partition_key()?,
            started_at_unix_millis,
        },
        PartitionState::ReceivingSplit => PartitionTransition::ReceivingSplit {
            source: decoder.partition_key()?,
            started_at_unix_millis,
        },
        PartitionState::DrainingSplit => PartitionTransition::DrainingSplit {
            left: decoder.partition_key()?,
            right: decoder.partition_key()?,
            started_at_unix_millis,
        },
        PartitionState::Merging => PartitionTransition::Merging {
            started_at_unix_millis,
        },
    };
    validate_transition(transition).map_err(|_| corrupt())?;
    Ok(transition)
}

fn validate_transition(transition: PartitionTransition) -> Result<()> {
    match transition {
        PartitionTransition::Splitting { left, right, .. }
        | PartitionTransition::DrainingSplit { left, right, .. }
            if left == right =>
        {
            Err(Error::invalid_argument())
        }
        _ => Ok(()),
    }
}

const fn encode_state_kind(kind: PartitionState) -> u8 {
    match kind {
        PartitionState::Ready => 0,
        PartitionState::Splitting => 1,
        PartitionState::ReceivingSplit => 2,
        PartitionState::DrainingSplit => 3,
        PartitionState::Merging => 4,
    }
}

fn decode_state_kind(tag: u8) -> Result<PartitionState> {
    match tag {
        0 => Ok(PartitionState::Ready),
        1 => Ok(PartitionState::Splitting),
        2 => Ok(PartitionState::ReceivingSplit),
        3 => Ok(PartitionState::DrainingSplit),
        4 => Ok(PartitionState::Merging),
        _ => Err(corrupt()),
    }
}
