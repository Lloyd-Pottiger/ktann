//! Checked-in real-dataset fixtures: byte reading and format parsing.
//!
//! This module is deliberately self-contained (no `support::` dependencies):
//! adapter-crate tests include it by path (`#[path = "..."]`) to load the
//! same fixtures without dragging in the rest of the test-support tree. See
//! `tests/datadriven/data/README.md` for fixture provenance.

// Included by path into the adapter crates, which use a subset of the API.
#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;

/// Reads one fixture file by plain file name (no path separators) from `dir`.
///
/// Bad names and unreadable files are corpus-authoring errors and panic.
#[must_use]
pub fn read_bytes(dir: &Path, name: &str) -> Vec<u8> {
    assert!(
        !name.is_empty() && !name.contains(['/', '\\']) && !name.contains(".."),
        "bad fixture name `{name}`"
    );
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("read fixture {}: {error}", path.display()))
}

/// Loads the vectors of one fixture: `.fvecs` (per vector one little-endian
/// i32 dimension prefix followed by that many little-endian f32 components)
/// and `.idx3-ubyte` (MNIST IDX: a big-endian header — magic 0x803, count,
/// rows, columns — followed by `count` row-major images, converted to raw f32
/// intensities in `[0, 255]`) are supported.
#[must_use]
pub fn read_vectors(dir: &Path, name: &str) -> Vec<Arc<[f32]>> {
    let bytes = read_bytes(dir, name);
    if name.ends_with(".fvecs") {
        parse_fvecs(&bytes, name)
    } else if name.ends_with(".idx3-ubyte") {
        parse_idx3_ubyte(&bytes, name)
    } else {
        panic!("unknown fixture format `{name}`")
    }
}

/// Reads an `.ivecs` fixture (the `.fvecs` layout with i32 components); used
/// to check the brute-force oracle against published ground truth.
#[must_use]
pub fn read_ivecs(dir: &Path, name: &str) -> Vec<Vec<i32>> {
    let bytes = read_bytes(dir, name);
    assert!(bytes.len() >= 4, "bad ivecs fixture `{name}`");
    let width = i32::from_le_bytes(bytes[..4].try_into().expect("prefix")) as usize;
    let record = 4 + width * 4;
    assert!(
        width > 0 && bytes.len() % record == 0,
        "truncated ivecs fixture `{name}`"
    );
    (0..bytes.len() / record)
        .map(|index| {
            let payload = &bytes[index * record + 4..(index + 1) * record];
            payload
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("component")))
                .collect()
        })
        .collect()
}

/// Parses the `.fvecs` layout; panics on truncated or inconsistent bytes.
fn parse_fvecs(bytes: &[u8], name: &str) -> Vec<Arc<[f32]>> {
    assert!(bytes.len() >= 4, "bad fvecs fixture `{name}`");
    let dimension = i32::from_le_bytes(bytes[..4].try_into().expect("prefix")) as usize;
    let record = 4 + dimension * 4;
    assert!(
        dimension > 0 && bytes.len() % record == 0,
        "truncated fvecs fixture `{name}`"
    );
    (0..bytes.len() / record)
        .map(|index| {
            let payload = &bytes[index * record..(index + 1) * record];
            assert!(
                i32::from_le_bytes(payload[..4].try_into().expect("prefix")) as usize == dimension,
                "inconsistent fvecs dimension in fixture `{name}`"
            );
            payload[4..]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("component")))
                .collect()
        })
        .collect()
}

/// Parses the MNIST IDX ubyte layout; panics on truncated or inconsistent
/// bytes.
fn parse_idx3_ubyte(bytes: &[u8], name: &str) -> Vec<Arc<[f32]>> {
    let header = |index: usize| -> i32 {
        bytes
            .get(index * 4..index * 4 + 4)
            .and_then(|chunk| chunk.try_into().ok())
            .map(i32::from_be_bytes)
            .unwrap_or_else(|| panic!("bad idx3-ubyte fixture `{name}`"))
    };
    assert!(
        header(0) == 0x803,
        "bad idx3-ubyte magic in fixture `{name}`"
    );
    let count = header(1) as usize;
    let dimension = (header(2) * header(3)) as usize;
    assert!(
        bytes.len() == 16 + count * dimension,
        "truncated idx3-ubyte fixture `{name}`"
    );
    (0..count)
        .map(|index| {
            bytes[16 + index * dimension..16 + (index + 1) * dimension]
                .iter()
                .map(|byte| f32::from(*byte))
                .collect()
        })
        .collect()
}
