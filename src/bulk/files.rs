//! Segmented immutable input files and atomic workspace file helpers.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::api::{Error, ErrorKind, Result};
use crate::storage::FORMAT_VERSION;
use crate::storage::values::MAX_VALUE_BYTES;

const MAGIC: &[u8; 7] = b"KTANNBF";
const HEADER_BYTES: u64 = 42;
const SEGMENTS_HEADER_BYTES: u64 = 46;
const SEGMENT_ENTRY_BYTES: u64 = 48;
const FRAME_OVERHEAD: u64 = 36;
const IO_BUFFER_BYTES: usize = 256 * 1024;

const KIND_INPUT: u8 = 0;
const KIND_SEGMENTS_INDEX: u8 = 1;

/// Target maximum records per immutable input segment, enabling parallel ingestion.
const SEGMENT_MAX_ITEMS: u64 = 16_384;
/// Target maximum framed bytes per immutable input segment.
const SEGMENT_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Size of the canonical input snapshot manifest.
pub const INPUT_MANIFEST_BYTES: usize = 90;

/// Identity of a sealed segmented [`super::InputSnapshot`].
///
/// Paths are excluded so snapshots remain relocatable across directories and
/// hosts. Persist this value in the owning build descriptor rather than
/// trusting an unverified file in a worker directory after restart.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputManifest {
    binding: [u8; 32],
    items: u64,
    bytes: u64,
    digest: [u8; 32],
}

impl InputManifest {
    fn header(kind: u8, binding: &[u8; 32]) -> [u8; HEADER_BYTES as usize] {
        let mut bytes = [0; HEADER_BYTES as usize];
        bytes[..7].copy_from_slice(MAGIC);
        bytes[7..9].copy_from_slice(&FORMAT_VERSION.to_be_bytes());
        bytes[9] = kind;
        bytes[10..].copy_from_slice(binding);
        bytes
    }

    /// Number of records across all segments in the snapshot.
    #[must_use]
    pub const fn items(&self) -> u64 {
        self.items
    }

    /// Exact total length of `segments.bin` plus all segment data files.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// SHA-256 of `segments.bin`, which binds every segment's count, length, and SHA-256.
    #[must_use]
    pub const fn sha256(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Canonical 90-byte representation persisted in a [`crate::storage::values::BuildDescriptor`].
    #[must_use]
    pub fn encode(&self) -> [u8; INPUT_MANIFEST_BYTES] {
        let mut bytes = [0; INPUT_MANIFEST_BYTES];
        bytes[..HEADER_BYTES as usize].copy_from_slice(&Self::header(KIND_INPUT, &self.binding));
        bytes[42..50].copy_from_slice(&self.items.to_be_bytes());
        bytes[50..58].copy_from_slice(&self.bytes.to_be_bytes());
        bytes[58..].copy_from_slice(&self.digest);
        bytes
    }

    /// Decodes a bounded manifest; it does not verify referenced snapshot files.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedFormat`] on version mismatch and
    /// [`ErrorKind::Corruption`] on malformed or inconsistent bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != INPUT_MANIFEST_BYTES {
            return Err(corrupt());
        }
        let binding = parse_header(&bytes[..42], KIND_INPUT)?;
        let manifest = Self {
            binding,
            items: u64::from_be_bytes(bytes[42..50].try_into().expect("fixed count")),
            bytes: u64::from_be_bytes(bytes[50..58].try_into().expect("fixed length")),
            digest: bytes[58..].try_into().expect("fixed digest"),
        };
        if manifest.bytes < SEGMENTS_HEADER_BYTES
            || (manifest.items == 0 && manifest.bytes != SEGMENTS_HEADER_BYTES)
            || (manifest.items > 0
                && manifest.items > (manifest.bytes - SEGMENTS_HEADER_BYTES) / (FRAME_OVERHEAD + 1))
        {
            return Err(corrupt());
        }
        Ok(manifest)
    }

    pub(super) fn matches(&self, binding: [u8; 32]) -> bool {
        self.binding == binding
    }
}

fn parse_header(bytes: &[u8], expected_kind: u8) -> Result<[u8; 32]> {
    if bytes.len() < HEADER_BYTES as usize || &bytes[..7] != MAGIC {
        return Err(corrupt());
    }
    if u16::from_be_bytes(bytes[7..9].try_into().expect("fixed version")) != FORMAT_VERSION {
        return Err(Error::new(ErrorKind::UnsupportedFormat));
    }
    if bytes[9] != expected_kind {
        return Err(corrupt());
    }
    Ok(bytes[10..42].try_into().expect("fixed binding"))
}

/// Verified metadata for one immutable segment file `seg-{index:08x}.bin`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SegmentMeta {
    pub items: u64,
    pub bytes: u64,
    pub sha256: [u8; 32],
}

pub(crate) fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

pub(crate) fn io_error(error: std::io::Error) -> Error {
    Error::new(if error.kind() == std::io::ErrorKind::UnexpectedEof {
        ErrorKind::Corruption
    } else {
        ErrorKind::Backend
    })
}

pub(crate) fn segment_file_name(index: usize) -> String {
    format!("seg-{index:08x}.bin")
}

const SEGMENT_PARTIAL_NAME: &str = "data.partial";

fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|d| d.sync_all())
        .map_err(io_error)
}

fn write_sealed_file(dir: &Path, partial: &str, final_name: &str, bytes: &[u8]) -> Result<()> {
    let partial_path = dir.join(partial);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial_path)
        .map_err(io_error)?;
    file.write_all(bytes).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    drop(file);
    fs::rename(&partial_path, dir.join(final_name)).map_err(io_error)?;
    sync_dir(dir)
}

struct ActiveSegment {
    index: usize,
    file: BufWriter<File>,
    hash: Sha256,
    items: u64,
    bytes: u64,
}

/// Streams records into bounded segment files inside a newly created directory.
pub(super) struct SegmentedWriter {
    directory: PathBuf,
    binding: [u8; 32],
    segments: Vec<SegmentMeta>,
    active: Option<ActiveSegment>,
    total_items: u64,
    data_bytes: u64,
    quota: u64,
}

impl SegmentedWriter {
    pub(super) fn new(directory: &Path, binding: [u8; 32], quota: u64) -> Result<Self> {
        let min_bytes = SEGMENTS_HEADER_BYTES + INPUT_MANIFEST_BYTES as u64;
        if quota < min_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        fs::create_dir(directory).map_err(io_error)?;
        Ok(Self {
            directory: directory.to_path_buf(),
            binding,
            segments: Vec::new(),
            active: None,
            total_items: 0,
            data_bytes: 0,
            quota,
        })
    }

    pub(super) fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let length = u32::try_from(bytes.len()).map_err(|_| Error::invalid_argument())?;
        if bytes.is_empty() {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let rotate = self.active.as_ref().is_some_and(|seg| {
            seg.items >= SEGMENT_MAX_ITEMS
                || seg.bytes.saturating_add(FRAME_OVERHEAD + u64::from(length)) > SEGMENT_MAX_BYTES
        });
        if rotate {
            self.finish_active()?;
        }
        let new_seg_overhead = if self.active.is_none() {
            HEADER_BYTES + SEGMENT_ENTRY_BYTES
        } else {
            0
        };
        let committed_index_bytes =
            SEGMENTS_HEADER_BYTES + (self.segments.len() as u64) * SEGMENT_ENTRY_BYTES;
        let active_index_bytes = u64::from(self.active.is_some()) * SEGMENT_ENTRY_BYTES;
        let projected = committed_index_bytes
            .checked_add(active_index_bytes)
            .and_then(|n| n.checked_add(self.data_bytes))
            .and_then(|n| n.checked_add(new_seg_overhead))
            .and_then(|n| n.checked_add(FRAME_OVERHEAD + u64::from(length)))
            .and_then(|n| n.checked_add(INPUT_MANIFEST_BYTES as u64))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        if projected > self.quota {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        if self.active.is_none() {
            let index = self.segments.len();
            if u32::try_from(index).is_err() {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(self.directory.join(SEGMENT_PARTIAL_NAME))
                .map_err(io_error)?;
            let mut file = BufWriter::with_capacity(IO_BUFFER_BYTES, file);
            let header = InputManifest::header(KIND_INPUT, &self.binding);
            file.write_all(&header).map_err(io_error)?;
            let mut hash = Sha256::new();
            hash.update(header);
            self.data_bytes += HEADER_BYTES;
            self.active = Some(ActiveSegment {
                index,
                file,
                hash,
                items: 0,
                bytes: HEADER_BYTES,
            });
        }
        let seg = self.active.as_mut().expect("active segment");
        let len_bytes = length.to_be_bytes();
        let frame_digest = Sha256::digest(bytes);
        seg.file.write_all(&len_bytes).map_err(io_error)?;
        seg.file.write_all(bytes).map_err(io_error)?;
        seg.file.write_all(&frame_digest).map_err(io_error)?;
        seg.hash.update(len_bytes);
        seg.hash.update(bytes);
        seg.hash.update(frame_digest);
        let added = FRAME_OVERHEAD + u64::from(length);
        seg.bytes += added;
        seg.items += 1;
        self.data_bytes += added;
        self.total_items += 1;
        Ok(())
    }

    fn finish_active(&mut self) -> Result<()> {
        let Some(mut seg) = self.active.take() else {
            return Ok(());
        };
        seg.file.flush().map_err(io_error)?;
        let inner = seg
            .file
            .into_inner()
            .map_err(|e| io_error(e.into_error()))?;
        inner.sync_all().map_err(io_error)?;
        drop(inner);
        let sha256: [u8; 32] = seg.hash.finalize().into();
        fs::rename(
            self.directory.join(SEGMENT_PARTIAL_NAME),
            self.directory.join(segment_file_name(seg.index)),
        )
        .map_err(io_error)?;
        self.segments.push(SegmentMeta {
            items: seg.items,
            bytes: seg.bytes,
            sha256,
        });
        Ok(())
    }

    pub(super) fn seal(mut self) -> Result<(PathBuf, InputManifest, Vec<SegmentMeta>)> {
        self.finish_active()?;
        let count =
            u32::try_from(self.segments.len()).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        let mut index_bytes =
            Vec::with_capacity(SEGMENTS_HEADER_BYTES as usize + self.segments.len() * 48);
        index_bytes.extend_from_slice(&InputManifest::header(KIND_SEGMENTS_INDEX, &self.binding));
        index_bytes.extend_from_slice(&count.to_be_bytes());
        for seg in &self.segments {
            index_bytes.extend_from_slice(&seg.items.to_be_bytes());
            index_bytes.extend_from_slice(&seg.bytes.to_be_bytes());
            index_bytes.extend_from_slice(&seg.sha256);
        }
        let digest: [u8; 32] = Sha256::digest(&index_bytes).into();
        let total_bytes = (index_bytes.len() as u64)
            .checked_add(self.data_bytes)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;

        write_sealed_file(
            &self.directory,
            "segments.partial",
            "segments.bin",
            &index_bytes,
        )?;
        let manifest = InputManifest {
            binding: self.binding,
            items: self.total_items,
            bytes: total_bytes,
            digest,
        };
        write_sealed_file(
            &self.directory,
            "manifest.partial",
            "manifest.bin",
            &manifest.encode(),
        )?;
        sync_dir(
            self.directory
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?;
        Ok((self.directory, manifest, self.segments))
    }
}

/// Validates `manifest.bin`, `segments.bin`, and segment file sizes on snapshot reopen.
pub(super) fn open_segments(
    directory: &Path,
    expected: &InputManifest,
) -> Result<Vec<SegmentMeta>> {
    let mut manifest_file = File::open(directory.join("manifest.bin")).map_err(io_error)?;
    if manifest_file.metadata().map_err(io_error)?.len() != INPUT_MANIFEST_BYTES as u64 {
        return Err(corrupt());
    }
    let mut manifest_bytes = [0; INPUT_MANIFEST_BYTES];
    manifest_file
        .read_exact(&mut manifest_bytes)
        .map_err(io_error)?;
    if InputManifest::decode(&manifest_bytes)? != *expected {
        return Err(corrupt());
    }

    let mut seg_file = File::open(directory.join("segments.bin")).map_err(io_error)?;
    let seg_file_len = seg_file.metadata().map_err(io_error)?.len();
    if seg_file_len < SEGMENTS_HEADER_BYTES
        || !(seg_file_len - SEGMENTS_HEADER_BYTES).is_multiple_of(SEGMENT_ENTRY_BYTES)
    {
        return Err(corrupt());
    }
    let count_from_len = (seg_file_len - SEGMENTS_HEADER_BYTES) / SEGMENT_ENTRY_BYTES;
    if count_from_len > u64::from(u32::MAX) || seg_file_len > 256 * 1024 * 1024 {
        return Err(corrupt());
    }
    let mut index_bytes = vec![0; seg_file_len as usize];
    seg_file.read_exact(&mut index_bytes).map_err(io_error)?;
    let binding = parse_header(&index_bytes[..42], KIND_SEGMENTS_INDEX)?;
    if <[u8; 32]>::from(Sha256::digest(&index_bytes)) != expected.digest
        || binding != expected.binding
    {
        return Err(corrupt());
    }
    let count = u32::from_be_bytes(index_bytes[42..46].try_into().expect("count")) as usize;
    if count as u64 != count_from_len {
        return Err(corrupt());
    }
    let mut segments = Vec::with_capacity(count);
    let mut sum_items = 0_u64;
    let mut sum_bytes = seg_file_len;
    for i in 0..count {
        let base = 46 + i * 48;
        let items = u64::from_be_bytes(index_bytes[base..base + 8].try_into().expect("items"));
        let bytes = u64::from_be_bytes(index_bytes[base + 8..base + 16].try_into().expect("bytes"));
        let sha256: [u8; 32] = index_bytes[base + 16..base + 48]
            .try_into()
            .expect("sha256");
        if items == 0
            || bytes < HEADER_BYTES
            || items > (bytes - HEADER_BYTES) / (FRAME_OVERHEAD + 1)
        {
            return Err(corrupt());
        }
        let actual_len = fs::metadata(directory.join(segment_file_name(i)))
            .map_err(io_error)?
            .len();
        if actual_len != bytes {
            return Err(corrupt());
        }
        sum_items = sum_items.checked_add(items).ok_or_else(corrupt)?;
        sum_bytes = sum_bytes.checked_add(bytes).ok_or_else(corrupt)?;
        segments.push(SegmentMeta {
            items,
            bytes,
            sha256,
        });
    }
    if sum_items != expected.items || sum_bytes != expected.bytes {
        return Err(corrupt());
    }
    Ok(segments)
}

/// Streams and checksum-verifies one immutable input segment.
pub(crate) struct SegmentReader {
    file: BufReader<File>,
    hash: Sha256,
    meta: SegmentMeta,
    remaining: u64,
    consumed: u64,
    maximum_frame: usize,
    done: bool,
    verified_tracker: Option<(Arc<[AtomicBool]>, usize)>,
}

impl SegmentReader {
    pub(super) fn open(
        directory: &Path,
        index: usize,
        binding: [u8; 32],
        meta: SegmentMeta,
        maximum_frame: usize,
        verified_tracker: Option<(Arc<[AtomicBool]>, usize)>,
    ) -> Result<Self> {
        Self::open_with_capacity(
            directory,
            index,
            binding,
            meta,
            maximum_frame,
            IO_BUFFER_BYTES,
            verified_tracker,
        )
    }

    pub(super) fn open_strided(
        directory: &Path,
        index: usize,
        binding: [u8; 32],
        meta: SegmentMeta,
        maximum_frame: usize,
    ) -> Result<Self> {
        Self::open_with_capacity(directory, index, binding, meta, maximum_frame, 4096, None)
    }

    fn open_with_capacity(
        directory: &Path,
        index: usize,
        binding: [u8; 32],
        meta: SegmentMeta,
        maximum_frame: usize,
        buffer_capacity: usize,
        verified_tracker: Option<(Arc<[AtomicBool]>, usize)>,
    ) -> Result<Self> {
        let file = File::open(directory.join(segment_file_name(index))).map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() != meta.bytes {
            return Err(corrupt());
        }
        let mut reader = Self {
            file: BufReader::with_capacity(buffer_capacity, file),
            hash: Sha256::new(),
            meta,
            remaining: meta.items,
            consumed: 0,
            maximum_frame,
            done: false,
            verified_tracker,
        };
        let mut header = [0; HEADER_BYTES as usize];
        reader.read(&mut header)?;
        if parse_header(&header, KIND_INPUT)? != binding {
            return Err(corrupt());
        }
        Ok(reader)
    }

    fn read(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.file.read_exact(bytes).map_err(io_error)?;
        self.hash.update(&*bytes);
        self.consumed += bytes.len() as u64;
        Ok(())
    }

    fn frame(&mut self) -> Result<Option<(u64, u32, Bytes)>> {
        if self.remaining == 0 {
            let mut tail = [0];
            if self.consumed != self.meta.bytes
                || self.file.read(&mut tail).map_err(io_error)? != 0
                || <[u8; 32]>::from(self.hash.clone().finalize()) != self.meta.sha256
            {
                return Err(corrupt());
            }
            if let Some((flags, idx)) = &self.verified_tracker
                && let Some(flag) = flags.get(*idx)
            {
                flag.store(true, AtomicOrdering::Release);
            }
            return Ok(None);
        }
        let frame_offset = self.consumed;
        let mut length = [0; 4];
        self.read(&mut length)?;
        let body_len = u32::from_be_bytes(length);
        let length = body_len as usize;
        if length == 0
            || length > self.maximum_frame
            || length as u64 + 32 > self.meta.bytes.saturating_sub(self.consumed)
        {
            return Err(corrupt());
        }
        let mut bytes = vec![0; length];
        self.read(&mut bytes)?;
        let mut digest = [0; 32];
        self.read(&mut digest)?;
        if <[u8; 32]>::from(Sha256::digest(&bytes)) != digest {
            return Err(corrupt());
        }
        self.remaining -= 1;
        Ok(Some((frame_offset, body_len, Bytes::from(bytes))))
    }

    pub(crate) fn next_frame_with_span(&mut self) -> Option<Result<(u64, u32, Bytes)>> {
        if self.done {
            return None;
        }
        match self.frame() {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(error) => {
                self.done = true;
                Some(Err(error))
            }
        }
    }

    pub(crate) fn next_frame(&mut self) -> Option<Result<Bytes>> {
        self.next_frame_with_span()
            .map(|res| res.map(|(_, _, bytes)| bytes))
    }

    /// Reads every `stride`-th record across the full segment while skipping intermediate
    /// frame bodies via `seek_relative` so uniform segment sampling does `O(sample)` I/O.
    pub(crate) fn visit_strided_frames(
        mut self,
        stride: u64,
        mut on_frame: impl FnMut(Bytes) -> Result<()>,
    ) -> Result<()> {
        let stride = stride.max(1);
        let mut idx = 0_u64;
        while self.remaining > 0 {
            let mut len_b = [0_u8; 4];
            self.file.read_exact(&mut len_b).map_err(io_error)?;
            self.consumed += 4;
            let body_len = u32::from_be_bytes(len_b) as usize;
            let skip_bytes = (body_len as u64).checked_add(32).ok_or_else(corrupt)?;
            if body_len == 0
                || body_len > self.maximum_frame
                || skip_bytes > self.meta.bytes.saturating_sub(self.consumed)
            {
                return Err(corrupt());
            }
            if idx.is_multiple_of(stride) {
                let mut bytes = vec![0_u8; body_len];
                self.file.read_exact(&mut bytes).map_err(io_error)?;
                let mut digest = [0_u8; 32];
                self.file.read_exact(&mut digest).map_err(io_error)?;
                self.consumed += skip_bytes;
                if <[u8; 32]>::from(Sha256::digest(&bytes)) != digest {
                    return Err(corrupt());
                }
                on_frame(Bytes::from(bytes))?;
            } else {
                let delta = i64::try_from(skip_bytes).map_err(|_| corrupt())?;
                self.file.seek_relative(delta).map_err(io_error)?;
                self.consumed += skip_bytes;
            }
            self.remaining -= 1;
            idx += 1;
        }
        if self.consumed != self.meta.bytes {
            return Err(corrupt());
        }
        Ok(())
    }
}

/// Bounded single-file handle cache with coalesced forward read-ahead and segment authentication.
///
/// Keeps at most one open OS file descriptor regardless of segment count (`O(1)` FDs even at
/// 610,000+ segments), verifies the 42-byte segment header and whole-segment SHA-256 on first
/// access if not yet verified in this process, and coalesces forward reads through a `256 KiB`
/// window to avoid per-record `pread` syscalls.
pub(crate) struct SegmentFileReader<'a> {
    directory: &'a Path,
    binding: [u8; 32],
    segments: &'a [SegmentMeta],
    verified: &'a [AtomicBool],
    current: Option<(u32, File, u64)>,
    window: Vec<u8>,
    window_offset: u64,
    window_len: usize,
}

impl<'a> SegmentFileReader<'a> {
    #[must_use]
    pub(crate) fn new(
        directory: &'a Path,
        binding: [u8; 32],
        segments: &'a [SegmentMeta],
        verified: &'a [AtomicBool],
    ) -> Self {
        Self {
            directory,
            binding,
            segments,
            verified,
            current: None,
            window: vec![0_u8; IO_BUFFER_BYTES],
            window_offset: 0,
            window_len: 0,
        }
    }

    pub(crate) fn read_frame(&mut self, seg_idx: u32, offset: u64, body_len: u32) -> Result<Bytes> {
        let need_open = self
            .current
            .as_ref()
            .is_none_or(|(idx, _, _)| *idx != seg_idx);
        if need_open {
            let meta = self.segments.get(seg_idx as usize).ok_or_else(corrupt)?;
            let mut file = File::open(self.directory.join(segment_file_name(seg_idx as usize)))
                .map_err(io_error)?;
            if file.metadata().map_err(io_error)?.len() != meta.bytes || meta.bytes < HEADER_BYTES {
                return Err(corrupt());
            }
            let already_verified = self
                .verified
                .get(seg_idx as usize)
                .is_some_and(|f| f.load(AtomicOrdering::Acquire));
            if already_verified {
                let mut header = [0_u8; HEADER_BYTES as usize];
                file.read_exact(&mut header).map_err(io_error)?;
                if parse_header(&header, KIND_INPUT)? != self.binding {
                    return Err(corrupt());
                }
            } else {
                verify_whole_segment(&mut file, self.binding, *meta)?;
                if let Some(flag) = self.verified.get(seg_idx as usize) {
                    flag.store(true, AtomicOrdering::Release);
                }
            }
            self.current = Some((seg_idx, file, meta.bytes));
            self.window_offset = 0;
            self.window_len = 0;
        }
        let &(_, ref file, seg_bytes) = self.current.as_ref().expect("opened segment");
        let body_usize = body_len as usize;
        if offset < HEADER_BYTES || body_usize == 0 || body_usize > MAX_VALUE_BYTES {
            return Err(corrupt());
        }
        let total = 4_usize
            .checked_add(body_usize)
            .and_then(|n| n.checked_add(32))
            .ok_or_else(corrupt)?;
        let end_offset = offset.checked_add(total as u64).ok_or_else(corrupt)?;
        if end_offset > seg_bytes {
            return Err(corrupt());
        }
        if total <= self.window.len() {
            let win_end = self.window_offset + self.window_len as u64;
            if offset < self.window_offset || end_offset > win_end {
                let read_len = ((seg_bytes - offset) as usize).min(self.window.len());
                read_at_exact(file, &mut self.window[..read_len], offset)?;
                self.window_offset = offset;
                self.window_len = read_len;
            }
            let start = (offset - self.window_offset) as usize;
            let raw = &self.window[start..start + total];
            return verify_frame_slice(raw, body_len);
        }
        read_verified_frame_at(file, offset, body_len)
    }
}

fn verify_whole_segment(
    file: &mut File,
    expected_binding: [u8; 32],
    meta: SegmentMeta,
) -> Result<()> {
    use std::io::{Seek as _, SeekFrom};
    file.seek(SeekFrom::Start(0)).map_err(io_error)?;
    let mut header = [0_u8; HEADER_BYTES as usize];
    file.read_exact(&mut header).map_err(io_error)?;
    if parse_header(&header, KIND_INPUT)? != expected_binding {
        return Err(corrupt());
    }
    let mut hasher = Sha256::new();
    hasher.update(header);
    let mut buf = vec![0_u8; IO_BUFFER_BYTES];
    let mut remaining = meta.bytes - HEADER_BYTES;
    while remaining > 0 {
        let to_read = (remaining as usize).min(buf.len());
        file.read_exact(&mut buf[..to_read]).map_err(io_error)?;
        hasher.update(&buf[..to_read]);
        remaining -= to_read as u64;
    }
    if <[u8; 32]>::from(hasher.finalize()) != meta.sha256 {
        return Err(corrupt());
    }
    Ok(())
}

fn read_at_exact(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset).map_err(io_error)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Seek as _, SeekFrom};
        let mut cloned = file.try_clone().map_err(io_error)?;
        cloned.seek(SeekFrom::Start(offset)).map_err(io_error)?;
        cloned.read_exact(buf).map_err(io_error)
    }
}

fn verify_frame_slice(buf: &[u8], body_len: u32) -> Result<Bytes> {
    let body_usize = body_len as usize;
    if u32::from_be_bytes(buf[..4].try_into().expect("frame len")) != body_len {
        return Err(corrupt());
    }
    let body = &buf[4..4 + body_usize];
    if <[u8; 32]>::from(Sha256::digest(body)) != buf[4 + body_usize..] {
        return Err(corrupt());
    }
    Ok(Bytes::copy_from_slice(body))
}

/// Reads and SHA-256-verifies one record frame at `frame_offset` without mutating file cursors.
pub(crate) fn read_verified_frame_at(
    file: &File,
    frame_offset: u64,
    body_len: u32,
) -> Result<Bytes> {
    let body_usize = body_len as usize;
    if frame_offset < HEADER_BYTES || body_usize == 0 || body_usize > MAX_VALUE_BYTES {
        return Err(corrupt());
    }
    let total = 4_usize
        .checked_add(body_usize)
        .and_then(|n| n.checked_add(32))
        .ok_or_else(corrupt)?;
    let mut buf = vec![0_u8; total];
    read_at_exact(file, &mut buf, frame_offset)?;
    if u32::from_be_bytes(buf[..4].try_into().expect("frame len")) != body_len {
        return Err(corrupt());
    }
    let body = &buf[4..4 + body_usize];
    if <[u8; 32]>::from(Sha256::digest(body)) != buf[4 + body_usize..] {
        return Err(corrupt());
    }
    Ok(Bytes::from(buf).slice(4..4 + body_usize))
}

/// Writes a workspace file atomically via a sibling temporary file, syncing both the
/// file and its parent directory before returning the written byte length.
pub(crate) fn write_atomic(
    path: &Path,
    write_fn: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<u64> {
    let mut nonce = [0_u8; 16];
    getrandom::fill(&mut nonce).map_err(|e| Error::with_source(ErrorKind::Other, e))?;
    let nonce_hex: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(Error::invalid_argument)?;
    let temp_path = path.with_file_name(format!(".{file_name}.tmp-{nonce_hex}"));
    let result = (|| -> Result<u64> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(io_error)?;
        let mut writer = BufWriter::with_capacity(IO_BUFFER_BYTES, file);
        write_fn(&mut writer)?;
        writer.flush().map_err(io_error)?;
        let file = writer.into_inner().map_err(|e| io_error(e.into_error()))?;
        let bytes = file.metadata().map_err(io_error)?.len();
        file.sync_all().map_err(io_error)?;
        drop(file);
        fs::rename(&temp_path, path).map_err(io_error)?;
        sync_dir(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?;
        Ok(bytes)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}
