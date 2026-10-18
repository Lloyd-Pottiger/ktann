//! Partition-local graph envelopes. Topology and source validation are performed
//! when the complete, ordered graph body is assembled by local_hnsw.

use super::{
    corrupt,
    wire::{Decoder, Encoder},
};
use crate::api::{Error, Result};
use bytes::Bytes;

/// Maximum complete graph value, including its type tag and length prefix.
pub const MAX_GRAPH_VALUE_BYTES: usize = 64 * 1024;
/// Fixed payload width of every nonfinal chunk.
pub const GRAPH_CHUNK_BYTES: usize = MAX_GRAPH_VALUE_BYTES - 5;

/// Integrity envelope for one internal partition's immutable topology.
///
/// The Header epoch remains the body-version authority. Entry, levels, and
/// adjacency exist only in the topology payload; centroid projections remain
/// exclusively in Child Entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphDescriptor {
    pub(crate) nodes: u32,
    pub(crate) epoch: u64,
    pub(crate) bytes: u64,
    pub(crate) binding: [u8; 32],
    pub(crate) sha256: [u8; 32],
}

impl GraphDescriptor {
    /// Returns the format-defined number of contiguous chunks.
    pub fn chunks(&self) -> u32 {
        self.bytes.div_ceil(GRAPH_CHUNK_BYTES as u64) as u32
    }

    fn validate(&self) -> Result<()> {
        if self.nodes == 0
            || self.nodes as usize > ktann_usearch::MAX_NODES
            || self.bytes < ktann_usearch::NATIVE_HEADER_BYTES as u64
            || self.bytes
                > ktann_usearch::encoded_bound(self.nodes as usize, ktann_usearch::MAX_NODES / 2)
                    .map_err(|_| corrupt())? as u64
        {
            return Err(corrupt());
        }
        Ok(())
    }
}

/// A nonempty slice of topology bytes. Chunk ordinals belong to Logical Keys.
#[derive(Clone, PartialEq, Eq)]
pub struct GraphChunk(pub(crate) Bytes);

impl GraphChunk {
    /// Creates one bounded chunk; full-size and final-chunk rules are checked
    /// against the descriptor when assembling the body.
    pub fn new(bytes: Bytes) -> Result<Self> {
        if bytes.is_empty() || bytes.len() > GRAPH_CHUNK_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(Self(bytes))
    }

    /// Returns the topology payload without its value envelope.
    pub fn bytes(&self) -> &Bytes {
        &self.0
    }
}

pub(super) fn encode_descriptor(encoder: &mut Encoder, value: &GraphDescriptor) -> Result<()> {
    value.validate().map_err(|_| Error::invalid_argument())?;
    encoder.u8(ktann_usearch::NATIVE_ENCODING);
    encoder.u32(value.nodes);
    encoder.u64(value.epoch);
    encoder.u64(value.bytes);
    encoder.bytes(&value.binding);
    encoder.bytes(&value.sha256);
    Ok(())
}

pub(super) fn decode_descriptor(decoder: &mut Decoder) -> Result<GraphDescriptor> {
    if decoder.u8()? != ktann_usearch::NATIVE_ENCODING {
        return Err(corrupt());
    }
    let value = GraphDescriptor {
        nodes: decoder.u32()?,
        epoch: decoder.u64()?,
        bytes: decoder.u64()?,
        binding: decoder.array()?,
        sha256: decoder.array()?,
    };
    value.validate()?;
    Ok(value)
}

pub(super) fn encode_chunk(encoder: &mut Encoder, chunk: &GraphChunk) -> Result<()> {
    encoder.sized_bytes(&chunk.0, GRAPH_CHUNK_BYTES)
}

pub(super) fn decode_chunk(decoder: &mut Decoder) -> Result<GraphChunk> {
    GraphChunk::new(decoder.sized_bytes(GRAPH_CHUNK_BYTES)?).map_err(|_| corrupt())
}

impl std::fmt::Debug for GraphChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphChunk")
            .field("bytes", &self.0.len())
            .finish()
    }
}
