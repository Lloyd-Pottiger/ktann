//! Recomputable, byte-bounded binary merge sorting for input preparation.

use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::api::{Error, ErrorKind, Result};

use super::files::{corrupt, io_error};

// Amortize file syscalls while keeping two caller readers plus a three-buffer
// merge within validate_memory's existing 512 KiB IO/metadata reservation.
const BUFFER_BYTES: usize = 64 * 1024;

/// One sortable projection. Neither vectors nor field values appear in Debug.
#[derive(Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct Row {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

impl Row {
    fn encoded_bytes(&self) -> usize {
        8 + self.key.len() + self.value.len()
    }
}

pub(super) struct Run {
    path: PathBuf,
    bytes: u64,
    rows: u64,
}

/// All runs share a quota, including inputs retained during merge writes.
pub(super) struct Space {
    directory: PathBuf,
    quota: u64,
    maximum_row: usize,
    next: u64,
    live: u64,
    pub peak: u64,
    pub written: u64,
}

impl Space {
    pub fn new(directory: &Path, quota: u64, maximum_row: usize) -> Result<Self> {
        fs::create_dir(directory).map_err(io_error)?;
        Ok(Self {
            directory: directory.to_owned(),
            quota,
            maximum_row,
            next: 0,
            live: 0,
            peak: 0,
            written: 0,
        })
    }

    fn writer(&mut self) -> Result<(Run, BufWriter<File>)> {
        let path = self.directory.join(self.next.to_string());
        self.next = self.next.checked_add(1).ok_or_else(limit)?;
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(io_error)?;
        Ok((
            Run {
                path,
                bytes: 0,
                rows: 0,
            },
            BufWriter::with_capacity(BUFFER_BYTES, file),
        ))
    }

    fn append(&mut self, run: &mut Run, writer: &mut BufWriter<File>, row: &Row) -> Result<()> {
        let bytes = row.encoded_bytes();
        if bytes > self.maximum_row {
            return Err(limit());
        }
        let next = self.live.checked_add(bytes as u64).ok_or_else(limit)?;
        if next > self.quota {
            return Err(limit());
        }
        writer
            .write_all(&(row.key.len() as u32).to_be_bytes())
            .map_err(io_error)?;
        writer
            .write_all(&(row.value.len() as u32).to_be_bytes())
            .map_err(io_error)?;
        writer.write_all(&row.key).map_err(io_error)?;
        writer.write_all(&row.value).map_err(io_error)?;
        run.rows = run.rows.checked_add(1).ok_or_else(limit)?;
        run.bytes += bytes as u64;
        self.live = next;
        self.peak = self.peak.max(next);
        self.written = self.written.checked_add(bytes as u64).ok_or_else(limit)?;
        Ok(())
    }

    pub fn reader(&self, run: &Run) -> Result<RunReader> {
        let file = File::open(&run.path).map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() != run.bytes {
            return Err(corrupt());
        }
        Ok(RunReader {
            file: BufReader::with_capacity(BUFFER_BYTES, file),
            remaining: run.rows,
            bytes: run.bytes,
            maximum_row: self.maximum_row,
        })
    }

    pub fn remove(&mut self, run: Run) -> Result<()> {
        fs::remove_file(run.path).map_err(io_error)?;
        self.live -= run.bytes;
        Ok(())
    }

    fn merge(&mut self, left: Run, right: Run) -> Result<Run> {
        let mut a = self.reader(&left)?;
        let mut b = self.reader(&right)?;
        let (mut run, mut output) = self.writer()?;
        let mut x = a.next()?;
        let mut y = b.next()?;
        while x.is_some() || y.is_some() {
            let take_left = match (&x, &y) {
                (Some(x), Some(y)) => x <= y,
                (Some(_), None) => true,
                _ => false,
            };
            if take_left {
                self.append(&mut run, &mut output, x.as_ref().expect("selected row"))?;
                x = a.next()?;
            } else {
                self.append(&mut run, &mut output, y.as_ref().expect("selected row"))?;
                y = b.next()?;
            }
        }
        output.flush().map_err(io_error)?;
        drop((a, b, output));
        self.remove(left)?;
        self.remove(right)?;
        Ok(run)
    }
}

pub(super) struct RunReader {
    file: BufReader<File>,
    remaining: u64,
    bytes: u64,
    maximum_row: usize,
}

impl RunReader {
    pub fn next(&mut self) -> Result<Option<Row>> {
        if self.remaining == 0 {
            if self.bytes != 0 {
                return Err(corrupt());
            }
            return Ok(None);
        }
        let mut lengths = [0; 8];
        self.file.read_exact(&mut lengths).map_err(io_error)?;
        let key = u32::from_be_bytes(lengths[..4].try_into().expect("fixed length")) as usize;
        let value = u32::from_be_bytes(lengths[4..].try_into().expect("fixed length")) as usize;
        let total = key
            .checked_add(value)
            .and_then(|n| n.checked_add(8))
            .ok_or_else(corrupt)?;
        if total > self.maximum_row || total as u64 > self.bytes {
            return Err(corrupt());
        }
        let mut row = Row {
            key: vec![0; key],
            value: vec![0; value],
        };
        self.file.read_exact(&mut row.key).map_err(io_error)?;
        self.file.read_exact(&mut row.value).map_err(io_error)?;
        self.remaining -= 1;
        self.bytes -= total as u64;
        Ok(Some(row))
    }
}

/// Accounts allocated row payloads and row slots separately; sort_unstable
/// allocates no additional array. Binary carry bounds live run descriptors.
pub(super) struct Sorter<'a> {
    space: &'a mut Space,
    rows: Vec<Row>,
    budget: usize,
    payload_bytes: usize,
    runs: Vec<Option<Run>>,
}

pub(super) fn validate_memory(memory: usize, maximum_row: usize) -> Result<usize> {
    // While spilling, the caller can retain one source row and two merge
    // readers hold rows. Reserve a fourth row for transient decoding/copying,
    // IO buffers and fixed run/path metadata. The remaining budget owns rows.
    let reserved = maximum_row
        .checked_mul(4)
        .and_then(|n| n.checked_add(512 * 1024))
        .ok_or_else(limit)?;
    let budget = memory
        .checked_sub(reserved)
        .ok_or_else(Error::invalid_argument)?;
    if budget < maximum_row + std::mem::size_of::<Row>() {
        return Err(Error::invalid_argument());
    }
    Ok(budget)
}

impl<'a> Sorter<'a> {
    pub fn new(space: &'a mut Space, memory: usize) -> Result<Self> {
        let budget = validate_memory(memory, space.maximum_row)?;
        Ok(Self {
            space,
            rows: Vec::new(),
            budget,
            payload_bytes: 0,
            runs: Vec::new(),
        })
    }

    pub fn push(&mut self, row: Row) -> Result<()> {
        if row.encoded_bytes() > self.space.maximum_row {
            return Err(limit());
        }
        let payload = row
            .key
            .capacity()
            .checked_add(row.value.capacity())
            .ok_or_else(limit)?;
        let slot = std::mem::size_of::<Row>();
        let slots = self.rows.capacity().max(self.rows.len() + 1);
        let required = self
            .payload_bytes
            .checked_add(payload)
            .and_then(|n| n.checked_add(slots.checked_mul(slot)?))
            .ok_or_else(limit)?;
        if required > self.budget {
            self.spill()?;
        }
        if payload + slot > self.budget {
            return Err(limit());
        }
        if self.rows.len() == self.rows.capacity() {
            let available = (self.budget - self.payload_bytes - payload) / slot;
            let desired = self
                .rows
                .capacity()
                .max(16)
                .saturating_mul(2)
                .min(available);
            self.rows
                .try_reserve_exact(desired - self.rows.len())
                .map_err(|_| limit())?;
        }
        self.payload_bytes += payload;
        self.rows.push(row);
        Ok(())
    }

    fn spill(&mut self) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let mut rows = std::mem::take(&mut self.rows);
        self.payload_bytes = 0;
        rows.sort_unstable();
        let (mut run, mut output) = self.space.writer()?;
        for row in rows {
            self.space.append(&mut run, &mut output, &row)?;
        }
        output.flush().map_err(io_error)?;
        drop(output);
        let mut level = 0;
        loop {
            if level == self.runs.len() {
                self.runs.push(Some(run));
                break;
            }
            if let Some(other) = self.runs[level].take() {
                run = self.space.merge(other, run)?;
                level += 1;
            } else {
                self.runs[level] = Some(run);
                break;
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Run> {
        self.spill()?;
        let mut result = None;
        for run in self.runs.into_iter().flatten() {
            result = Some(match result {
                None => run,
                Some(other) => self.space.merge(other, run)?,
            });
        }
        match result {
            Some(run) => Ok(run),
            None => {
                let (run, mut file) = self.space.writer()?;
                file.flush().map_err(io_error)?;
                Ok(run)
            }
        }
    }
}

fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}
