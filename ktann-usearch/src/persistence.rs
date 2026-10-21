//! Strict validation of the pinned USearch generic native stream before loading.

use crate::{Error, Graph, MetricContext, build_bound, ffi, validate};

/// Native u64-key/u32-slot USearch 2.26.4 stream: 1 is little-, 2 big-endian.
/// The owning envelope must reject a different host's encoding before loading.
pub const NATIVE_ENCODING: u8 = if cfg!(target_endian = "little") { 1 } else { 2 };
/// Five native-endian u64 fields; checked against the C++ header at compile time.
pub const NATIVE_HEADER_BYTES: usize = 40;

fn checkpoint(check: &dyn Fn() -> bool) -> Result<(), Error> {
    if check() {
        Ok(())
    } else {
        Err(Error::Cancelled)
    }
}

/// Maximum native stream size, including full reserved adjacency capacity.
pub fn encoded_bound(n: usize, m: usize) -> Result<usize, Error> {
    validate(n, m)?;
    // Per node: level-table i16, key u64, embedded i16, then each layer's
    // u32 degree and reserved slots (2*M at base, M at every upper layer).
    let layers = crate::max_level(m) as usize;
    n.checked_mul(12 + 4 * (layers + 1) + 4 * (2 * m + layers * m))
        .and_then(|bytes| bytes.checked_add(NATIVE_HEADER_BYTES))
        .ok_or(Error::LimitExceeded)
}

/// Admission includes input bytes, native load allocations, and validation scratch.
pub fn decode_bound(n: usize, m: usize, bytes: usize) -> Result<usize, Error> {
    build_bound(n, m)?
        .checked_add(bytes)
        .and_then(|v| v.checked_add(n * (2 * size_of::<usize>() + 6) + 1024))
        .ok_or(Error::LimitExceeded)
}

/// Checked native-endian field reads. No untrusted integer becomes a pointer.
struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Cursor<'_> {
    fn field<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let start = self.position;
        self.skip(N)?;
        self.bytes[start..self.position]
            .try_into()
            .map_err(|_| Error::InvalidEncoding)
    }

    fn level(&mut self) -> Result<u16, Error> {
        Ok(u16::from_ne_bytes(self.field()?))
    }

    fn word(&mut self) -> Result<u32, Error> {
        Ok(u32::from_ne_bytes(self.field()?))
    }

    fn wide(&mut self) -> Result<u64, Error> {
        Ok(u64::from_ne_bytes(self.field()?))
    }

    fn skip(&mut self, bytes: usize) -> Result<(), Error> {
        self.position = self
            .position
            .checked_add(bytes)
            .ok_or(Error::InvalidEncoding)?;
        if self.position > self.bytes.len() {
            return Err(Error::InvalidEncoding);
        }
        Ok(())
    }
}

fn reserved<T>(count: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::LimitExceeded)?;
    Ok(values)
}

impl Graph {
    /// Saves through USearch's native serializer into one exactly sized buffer.
    /// Unused adjacency slots are preserved as opaque native state, not validated
    /// as edges. No Rust implementation of the native serializer is maintained.
    pub fn encode(&self, limit: usize, check: &dyn Fn() -> bool) -> Result<Vec<u8>, Error> {
        checkpoint(check)?;
        let length = self.inner.encoded_size();
        let required = self
            .resident_bytes()
            .checked_add(length)
            .and_then(|v| v.checked_add(1024))
            .ok_or(Error::LimitExceeded)?;
        if required > limit {
            return Err(Error::LimitExceeded);
        }
        let mut bytes = reserved(length)?;
        bytes.resize(length, 0);
        let context = MetricContext {
            distance: &|_, _| unreachable!(),
            check,
        };
        self.inner.save(&mut bytes, &context)?;
        Ok(bytes)
    }

    /// Validates the entire pinned native layout before invoking USearch's loader.
    /// Checks both copies of levels, ordinal keys, capacities, degrees, target
    /// layers, duplicate/self edges, maximal entry, reachability and exact length.
    /// No metric, rebuild, pruning or connectivity repair runs during loading.
    pub fn decode(
        bytes: &[u8],
        n: usize,
        m: usize,
        limit: usize,
        check: &dyn Fn() -> bool,
    ) -> Result<Self, Error> {
        checkpoint(check)?;
        if bytes.len() > encoded_bound(n, m)? {
            return Err(Error::InvalidEncoding);
        }
        let mut cursor = Cursor { bytes, position: 0 };
        if cursor.wide()? != n as u64
            || cursor.wide()? != m as u64
            || cursor.wide()? != (2 * m) as u64
        {
            return Err(Error::InvalidEncoding);
        }
        let max_level = cursor.wide()?;
        let entry = cursor.wide()?;
        if (n == 0 && (max_level != u64::MAX || entry != 0))
            || (n != 0 && (max_level > u64::from(crate::max_level(m)) || entry >= n as u64))
        {
            return Err(Error::InvalidEncoding);
        }
        if decode_bound(n, m, bytes.len())? > limit {
            return Err(Error::LimitExceeded);
        }
        let entry = entry as usize;
        let mut levels = reserved(n)?;
        let mut offsets = reserved(n)?;
        for _ in 0..n {
            checkpoint(check)?;
            let level = cursor.level()?;
            if u32::from(level) > crate::max_level(m) {
                return Err(Error::InvalidEncoding);
            }
            levels.push(level);
        }
        let mut seen = reserved(n)?;
        seen.resize(n, 0_u32);
        let mut stamp = 0_u32;
        // The level table is complete, so validate both layout and active edges
        // in one pass. An embedded level must equal its allocation's level.
        for (node, &level) in levels.iter().enumerate() {
            checkpoint(check)?;
            if cursor.wide()? != node as u64 || cursor.level()? != level {
                return Err(Error::InvalidEncoding);
            }
            offsets.push(cursor.position);
            for layer in 0..=level {
                stamp += 1; // At most MAX_NODES * (MAX_LEVEL+1), below u32::MAX.
                let capacity = if layer == 0 { 2 * m } else { m };
                let degree = cursor.word()? as usize;
                if degree > capacity.min(n.saturating_sub(1)) {
                    return Err(Error::InvalidEncoding);
                }
                for edge in 0..degree {
                    if edge % 64 == 0 {
                        checkpoint(check)?;
                    }
                    let target = cursor.word()? as usize;
                    if target >= n
                        || target == node
                        || levels[target] < layer
                        || seen[target] == stamp
                    {
                        return Err(Error::InvalidEncoding);
                    }
                    seen[target] = stamp;
                }
                cursor.skip(4 * (capacity - degree))?;
            }
        }
        if cursor.position != bytes.len()
            || (n != 0
                && (u64::from(levels[entry]) != max_level
                    || Some(&levels[entry]) != levels.iter().max()))
        {
            return Err(Error::InvalidEncoding);
        }
        seen.fill(0);
        let mut pending = reserved(n)?;
        if n != 0 {
            pending.push(entry);
            seen[entry] = 1;
        }
        let mut reached = 0;
        while let Some(node) = pending.pop() {
            checkpoint(check)?;
            reached += 1;
            cursor.position = offsets[node];
            for edge in 0..cursor.word()? {
                if edge % 64 == 0 {
                    checkpoint(check)?;
                }
                let target = cursor.word()? as usize;
                if seen[target] == 0 {
                    seen[target] = 1;
                    pending.push(target);
                }
            }
        }
        if reached != n {
            return Err(Error::InvalidEncoding);
        }
        drop((levels, offsets, seen, pending));
        let context = MetricContext {
            distance: &|_, _| unreachable!(),
            check,
        };
        let inner = ffi::restore(bytes, n as u32, m as u32, limit - bytes.len(), &context)?;
        Ok(Self { inner, n, m })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Hit, Workspace};
    use std::cell::Cell;
    const LIMIT: usize = 128 * 1024 * 1024;

    // Independent fixture for the pinned native layout. Inactive slots contain
    // arbitrary old state, as permitted by the upstream native serializer.
    fn topology() -> Vec<u8> {
        let mut bytes = Vec::new();
        for field in [3_u64, 2, 4, 1, 0] {
            bytes.extend_from_slice(&field.to_ne_bytes());
        }
        for level in [1_u16, 1, 0] {
            bytes.extend_from_slice(&level.to_ne_bytes());
        }
        for (node, layers) in [
            vec![vec![1], vec![1]],
            vec![vec![2], vec![0]],
            vec![vec![0]],
        ]
        .into_iter()
        .enumerate()
        {
            bytes.extend_from_slice(&(node as u64).to_ne_bytes());
            bytes.extend_from_slice(&((layers.len() - 1) as u16).to_ne_bytes());
            for (layer, neighbors) in layers.into_iter().enumerate() {
                bytes.extend_from_slice(&(neighbors.len() as u32).to_ne_bytes());
                let capacity = if layer == 0 { 4 } else { 2 };
                for slot in 0..capacity {
                    bytes.extend_from_slice(
                        &neighbors
                            .get(slot)
                            .copied()
                            .unwrap_or(u32::MAX)
                            .to_ne_bytes(),
                    );
                }
            }
        }
        bytes
    }

    #[test]
    fn native_golden_and_byte_preserving_roundtrip() {
        let mut empty = Vec::new();
        for field in [0_u64, 2, 4, u64::MAX, 0] {
            empty.extend_from_slice(&field.to_ne_bytes());
        }
        let mut singleton = Vec::new();
        for field in [1_u64, 2, 4, 0, 0] {
            singleton.extend_from_slice(&field.to_ne_bytes());
        }
        singleton.resize(72, 0);
        for (n, bytes) in [(0, empty), (1, singleton), (3, topology())] {
            let graph = Graph::decode(&bytes, n, 2, LIMIT, &|| true).unwrap();
            assert_eq!(graph.encode(LIMIT, &|| true).unwrap(), bytes);
            assert!(graph.peak_bytes() <= decode_bound(n, 2, bytes.len()).unwrap());
            let mut scratch = Workspace::new(n, 2, LIMIT).unwrap();
            for beam in 0..=n {
                let mut hits = vec![Hit::default(); beam];
                graph
                    .search(
                        3,
                        &|query, node| {
                            assert_eq!(query as usize, n);
                            assert!((node as usize) < n);
                            f64::from(node)
                        },
                        &|| true,
                        &mut scratch,
                        &mut hits,
                    )
                    .unwrap();
            }
        }
    }

    #[test]
    fn truncation_metadata_and_topology_corruption_fail_before_loading() {
        let bytes = topology();
        for length in 0..bytes.len() {
            assert_eq!(
                Graph::decode(&bytes[..length], 3, 2, LIMIT, &|| true).unwrap_err(),
                Error::InvalidEncoding
            );
        }
        let mut cases = Vec::new();
        for (offset, field) in [
            (0, 4_u64),
            (8, 3),
            (16, 5),
            (24, 2),
            (32, 3),
            (32, 2),
            (46, 1),
        ] {
            let mut bad = bytes.clone();
            bad[offset..offset + 8].copy_from_slice(&field.to_ne_bytes());
            cases.push(bad);
        }
        for (offset, field) in [(40, 17_u16), (40, u16::MAX), (54, 0)] {
            let mut bad = bytes.clone();
            bad[offset..offset + 2].copy_from_slice(&field.to_ne_bytes());
            cases.push(bad);
        }
        for (offset, field) in [(56, 3_u32), (60, 0), (60, 3), (80, 2), (102, 0)] {
            let mut bad = bytes.clone();
            bad[offset..offset + 4].copy_from_slice(&field.to_ne_bytes());
            cases.push(bad);
        }
        let mut duplicate = bytes.clone();
        duplicate[56..60].copy_from_slice(&2_u32.to_ne_bytes());
        duplicate[64..68].copy_from_slice(&1_u32.to_ne_bytes());
        cases.push(duplicate);
        let mut trailing = bytes;
        trailing.push(0);
        cases.push(trailing);
        for bytes in cases {
            assert_eq!(
                Graph::decode(&bytes, 3, 2, LIMIT, &|| true).unwrap_err(),
                Error::InvalidEncoding
            );
        }
    }

    #[test]
    fn arbitrary_byte_mutations_preserve_safe_ordinals_and_layers() {
        let bytes = topology();
        for offset in 0..bytes.len() {
            for value in [0, 1, 2, 16, 17, 255] {
                let mut changed = bytes.clone();
                changed[offset] = value;
                if let Ok(graph) = Graph::decode(&changed, 3, 2, LIMIT, &|| true) {
                    assert_eq!(graph.encode(LIMIT, &|| true).unwrap(), changed);
                    let mut scratch = Workspace::new(3, 2, LIMIT).unwrap();
                    graph
                        .search(
                            3,
                            &|a, b| {
                                assert_eq!(a, 3);
                                assert!(b < 3);
                                f64::from(b)
                            },
                            &|| true,
                            &mut scratch,
                            &mut [Hit::default(); 2],
                        )
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn native_allocation_failures_and_every_stream_cancellation_clean_up() {
        // A node larger than 64 KiB also exercises cancellation after a partial
        // node copy. Large M does not imply any valid edges for a singleton.
        let graph = Graph::build(1, 32768, 32768, 17, LIMIT, &|_, _| 0., &|| true).unwrap();
        let bytes = graph.encode(LIMIT, &|| true).unwrap();
        assert!(bytes.len() > 65536);
        let loaded = Graph::decode(&bytes, 1, 32768, LIMIT, &|| true).unwrap();
        let peak = loaded.peak_bytes();
        // Bypass operation admission only in this unit test to force native OOM
        // at its actual allocation boundaries, including partially loaded graphs.
        for (n, m, bytes) in [(1, 32768, bytes), (3, 2, topology())] {
            let context = MetricContext {
                distance: &|_, _| unreachable!(),
                check: &|| true,
            };
            let ceiling = if n == 1 {
                peak
            } else {
                Graph::decode(&bytes, n, m, LIMIT, &|| true)
                    .unwrap()
                    .peak_bytes()
            };
            assert!(ceiling + bytes.len() <= decode_bound(n, m, bytes.len()).unwrap());
            let stride = if n == 1 { 65536 } else { 64 };
            for limit in (0..ceiling).step_by(stride).chain([ceiling - 1]) {
                let result = ffi::restore(&bytes, n as u32, m as u32, limit, &context);
                assert_eq!(
                    result.map_err(Error::from).err(),
                    Some(Error::LimitExceeded)
                );
            }
            let calls = Cell::new(0);
            let count = || {
                calls.set(calls.get() + 1);
                true
            };
            let context = MetricContext {
                distance: &|_, _| unreachable!(),
                check: &count,
            };
            assert!(ffi::restore(&bytes, n as u32, m as u32, LIMIT, &context).is_ok());
            for stop in 0..calls.get() {
                let seen = Cell::new(0);
                let check = || {
                    let step = seen.get();
                    seen.set(step + 1);
                    step != stop
                };
                let context = MetricContext {
                    distance: &|_, _| unreachable!(),
                    check: &check,
                };
                assert_eq!(
                    ffi::restore(&bytes, n as u32, m as u32, LIMIT, &context)
                        .map_err(Error::from)
                        .err(),
                    Some(Error::Cancelled)
                );
            }
        }
        let calls = Cell::new(0);
        graph
            .encode(LIMIT, &|| {
                calls.set(calls.get() + 1);
                true
            })
            .unwrap();
        for stop in 0..calls.get() {
            let seen = Cell::new(0);
            assert_eq!(
                graph
                    .encode(LIMIT, &|| {
                        let step = seen.get();
                        seen.set(step + 1);
                        step != stop
                    })
                    .unwrap_err(),
                Error::Cancelled
            );
        }
        assert_eq!(graph.encode(0, &|| true).unwrap_err(), Error::LimitExceeded);
        assert_eq!(
            Graph::decode(&topology(), 3, 2, 0, &|| true).unwrap_err(),
            Error::LimitExceeded
        );
        assert_eq!(
            Graph::decode(&topology(), 3, 2, LIMIT, &|| false).unwrap_err(),
            Error::Cancelled
        );
    }
}
#[cfg(test)]
mod process_tests {
    use super::*;

    #[test]
    fn process_fixture() {
        if std::env::var_os("KTANN_GRAPH_CODEC_CHILD").is_none() {
            return;
        }
        let graph = Graph::build(
            64,
            4,
            32,
            17,
            1 << 24,
            &|a, b| f64::from(a.abs_diff(b)),
            &|| true,
        )
        .unwrap();
        let bytes = graph.encode(1 << 24, &|| true).unwrap();
        let restored = Graph::decode(&bytes, 64, 4, 1 << 24, &|| true).unwrap();
        assert_eq!(restored.encode(1 << 24, &|| true).unwrap(), bytes);
        println!("TOPOLOGY={bytes:02x?}");
    }

    #[test]
    fn fresh_processes_encode_the_same_graph() {
        let run = || {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "persistence::process_tests::process_fixture",
                    "--nocapture",
                ])
                .env("KTANN_GRAPH_CODEC_CHILD", "1")
                .output()
                .unwrap();
            assert!(output.status.success());
            String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .find(|line| line.starts_with("TOPOLOGY="))
                .unwrap()
                .to_owned()
        };
        assert_eq!(run(), run());
    }
}
