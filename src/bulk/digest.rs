//! Homomorphic multiset digests for order-independent Bulk Build verification.
//!
//! Each item is mapped via domain-separated SHA-512 into eight 64-bit lanes
//! (`512` bits total) and combined with wrapping lane-wise addition (`LtHash`
//! over `(Z / 2^64 Z)^8`). Addition is commutative and associative, allowing
//! disjoint shards and parallel validation slices to prove exact set equality
//! without materializing or sorting a global key/value artifact.

use sha2::{Digest, Sha512};

use crate::api::PartitionKey;

const DOMAIN_KV: u8 = 0x01;
const DOMAIN_ID: u8 = 0x02;
const DOMAIN_MEMBERSHIP: u8 = 0x03;
const DOMAIN_EDGE: u8 = 0x04;

/// Canonical serialized byte width of an [`LtHash`].
pub const LTHASH_BYTES: usize = 64;

/// A 512-bit commutative multiset digest over eight `u64` lanes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LtHash([u64; 8]);

impl LtHash {
    /// Identity digest of an empty multiset.
    pub const ZERO: Self = Self([0; 8]);

    /// Hashes one encoded logical key/value pair.
    #[must_use]
    pub fn from_kv(key: &[u8], value: &[u8]) -> Self {
        Self::hash(DOMAIN_KV, |h| {
            h.update((key.len() as u32).to_be_bytes());
            h.update(key);
            h.update((value.len() as u32).to_be_bytes());
            h.update(value);
        })
    }

    /// Hashes one canonical Record ID.
    #[must_use]
    pub fn from_id(id: &[u8]) -> Self {
        Self::hash(DOMAIN_ID, |h| {
            h.update((id.len() as u32).to_be_bytes());
            h.update(id);
        })
    }

    /// Hashes one `(Record ID, Tree Key, Leaf Partition Key, canonical fields)` membership binding.
    #[must_use]
    pub fn from_membership(
        tree_key: &[u8],
        partition: PartitionKey,
        id: &[u8],
        fields_bytes: &[u8],
    ) -> Self {
        Self::hash(DOMAIN_MEMBERSHIP, |h| {
            h.update((tree_key.len() as u32).to_be_bytes());
            h.update(tree_key);
            h.update(partition.get().to_be_bytes());
            h.update((id.len() as u32).to_be_bytes());
            h.update(id);
            h.update((fields_bytes.len() as u32).to_be_bytes());
            h.update(fields_bytes);
        })
    }

    /// Hashes one `(Tree Key, child Partition Key, encoded centroid)` parent-child binding.
    #[must_use]
    pub fn from_edge(tree_key: &[u8], child: PartitionKey, centroid_bytes: &[u8]) -> Self {
        Self::hash(DOMAIN_EDGE, |h| {
            h.update((tree_key.len() as u32).to_be_bytes());
            h.update(tree_key);
            h.update(child.get().to_be_bytes());
            h.update((centroid_bytes.len() as u32).to_be_bytes());
            h.update(centroid_bytes);
        })
    }

    /// Combines another multiset digest in-place.
    pub fn add_assign(&mut self, other: Self) {
        for (dst, src) in self.0.iter_mut().zip(other.0) {
            *dst = dst.wrapping_add(src);
        }
    }

    /// Returns the combined multiset digest of `self` and `other`.
    #[must_use]
    pub const fn wrapping_add(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 8 {
            self.0[i] = self.0[i].wrapping_add(other.0[i]);
            i += 1;
        }
        self
    }

    /// Encodes lanes in canonical big-endian order.
    #[must_use]
    pub fn encode(&self) -> [u8; LTHASH_BYTES] {
        let mut out = [0_u8; LTHASH_BYTES];
        for (i, lane) in self.0.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_be_bytes());
        }
        out
    }

    /// Decodes lanes from canonical big-endian bytes.
    #[must_use]
    pub fn decode(bytes: &[u8; LTHASH_BYTES]) -> Self {
        let mut lanes = [0_u64; 8];
        for (i, lane) in lanes.iter_mut().enumerate() {
            let chunk: [u8; 8] = bytes[i * 8..(i + 1) * 8].try_into().expect("fixed lane");
            *lane = u64::from_be_bytes(chunk);
        }
        Self(lanes)
    }

    fn hash(domain: u8, feed: impl FnOnce(&mut Sha512)) -> Self {
        let mut h = Sha512::new();
        h.update(b"KTANN-LTHASH-V1");
        h.update([domain]);
        feed(&mut h);
        let digest: [u8; 64] = h.finalize().into();
        let mut lanes = [0_u64; 8];
        for (i, lane) in lanes.iter_mut().enumerate() {
            *lane = u64::from_le_bytes(digest[i * 8..(i + 1) * 8].try_into().expect("lane"));
        }
        Self(lanes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiset_addition_is_order_independent_and_sensitive_to_edits() {
        let a = LtHash::from_kv(b"k1", b"v1");
        let b = LtHash::from_kv(b"k2", b"v2");
        let c = LtHash::from_kv(b"k3", b"v3");
        assert_eq!(
            a.wrapping_add(b).wrapping_add(c),
            c.wrapping_add(a).wrapping_add(b)
        );
        assert_ne!(
            a.wrapping_add(b),
            a.wrapping_add(LtHash::from_kv(b"k2", b"v2-mut"))
        );
        let encoded = a.wrapping_add(b).encode();
        assert_eq!(LtHash::decode(&encoded), a.wrapping_add(b));
    }
}
