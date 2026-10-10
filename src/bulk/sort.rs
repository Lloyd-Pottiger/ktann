//! Recomputable, byte-bounded multiway merge sorting for input preparation.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::api::{Error, ErrorKind, Result};

use super::files::{corrupt, io_error};

// Two caller IO buffers, eight 32 KiB merge readers and one 64 KiB writer
// use at most 448 KiB of the 512 KiB IO/metadata reservation.
const BUFFER_BYTES: usize = 64 * 1024;
const MERGE_BUFFER_BYTES: usize = 32 * 1024;
const MAX_FAN_IN: usize = 8;

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
    file_id: u64,
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

    pub fn writer(&mut self) -> Result<(Run, BufWriter<File>)> {
        let file_id = self.next;
        let path = self.directory.join(file_id.to_string());
        self.next = self.next.checked_add(1).ok_or_else(limit)?;
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(io_error)?;
        Ok((
            Run {
                file_id,
                bytes: 0,
                rows: 0,
            },
            BufWriter::with_capacity(BUFFER_BYTES, file),
        ))
    }

    pub fn append(&mut self, run: &mut Run, writer: &mut BufWriter<File>, row: &Row) -> Result<()> {
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
        self.reader_with_capacity(run, BUFFER_BYTES)
    }

    fn reader_with_capacity(&self, run: &Run, capacity: usize) -> Result<RunReader> {
        let file = File::open(self.directory.join(run.file_id.to_string())).map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() != run.bytes {
            return Err(corrupt());
        }
        Ok(RunReader {
            file: BufReader::with_capacity(capacity, file),
            remaining: run.rows,
            bytes: run.bytes,
            maximum_row: self.maximum_row,
        })
    }

    pub fn reclaim_directory(&self) -> Result<()> {
        fs::remove_dir(&self.directory).map_err(io_error)
    }

    pub fn remove(&mut self, run: Run) -> Result<()> {
        fs::remove_file(self.directory.join(run.file_id.to_string())).map_err(io_error)?;
        self.live -= run.bytes;
        Ok(())
    }

    fn merge(&mut self, inputs: Vec<Run>) -> Result<Run> {
        let capacity = if inputs.len() <= 4 {
            BUFFER_BYTES
        } else {
            MERGE_BUFFER_BYTES
        };
        let mut readers = inputs
            .iter()
            .map(|run| self.reader_with_capacity(run, capacity))
            .collect::<Result<Vec<_>>>()?;
        let mut heads = BinaryHeap::with_capacity(readers.len());
        for (index, reader) in readers.iter_mut().enumerate() {
            if let Some(row) = reader.next()? {
                heads.push(Reverse((row, index)));
            }
        }
        let (mut run, mut output) = self.writer()?;
        while let Some(Reverse((row, index))) = heads.pop() {
            self.append(&mut run, &mut output, &row)?;
            drop(row);
            if let Some(next) = readers[index].next()? {
                heads.push(Reverse((next, index)));
            }
        }
        output.flush().map_err(io_error)?;
        drop((readers, output));
        for input in inputs {
            self.remove(input)?;
        }
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
/// allocates no additional array. Base-fan-in carry bounds live run descriptors.
pub(super) struct Sorter {
    rows: Vec<Row>,
    budget: usize,
    payload_bytes: usize,
    runs: Vec<Vec<Run>>,
    fan_in: usize,
    reject_duplicate_keys: bool,
}

pub(super) fn validate_memory(memory: usize, maximum_row: usize) -> Result<usize> {
    memory_plan(memory, maximum_row, 1).map(|(_, budget)| budget)
}

// Larger rows or small budgets reduce fan-in, rather than rejecting a budget
// that can still support a two-way merge. All merge heads are explicitly charged.
fn memory_plan(memory: usize, maximum_row: usize, buffers: usize) -> Result<(usize, usize)> {
    let minimum = maximum_row
        .checked_add(std::mem::size_of::<Row>())
        .and_then(|row| row.checked_mul(buffers))
        .ok_or_else(limit)?;
    for fan_in in (2..=MAX_FAN_IN).rev() {
        let Some(reserved) = maximum_row
            .checked_mul(fan_in + 2)
            .and_then(|n| n.checked_add(512 * 1024))
        else {
            continue;
        };
        if let Some(budget) = memory
            .checked_sub(reserved)
            .filter(|budget| *budget >= minimum)
        {
            return Ok((fan_in, budget));
        }
    }
    Err(Error::invalid_argument())
}

impl Sorter {
    pub fn new(space: &Space, memory: usize) -> Result<Self> {
        let (fan_in, budget) = memory_plan(memory, space.maximum_row, 1)?;
        Ok(Self {
            rows: Vec::new(),
            budget,
            payload_bytes: 0,
            runs: Vec::new(),
            fan_in,
            reject_duplicate_keys: false,
        })
    }

    // Paired receipt sorters retain disjoint row buffers, but push/finish run
    // serially. Reserve merge heads and IO buffers once for the active sorter.
    // The first sorter checks local ID duplicates as each run is written.
    pub fn pair(space: &Space, memory: usize) -> Result<(Self, Self)> {
        let (fan_in, budget) = memory_plan(memory, space.maximum_row, 2)?;
        let first = budget / 2;
        let make = |budget, reject_duplicate_keys| Self {
            rows: Vec::new(),
            budget,
            payload_bytes: 0,
            runs: Vec::new(),
            fan_in,
            reject_duplicate_keys,
        };
        Ok((make(first, true), make(budget - first, false)))
    }

    pub fn push(&mut self, space: &mut Space, row: Row) -> Result<()> {
        if row.encoded_bytes() > space.maximum_row {
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
            self.spill(space)?;
        }
        if payload + slot > self.budget {
            return Err(limit());
        }
        if self.rows.len() == self.rows.capacity() {
            // Reserve room for the payloads of new slots too. Reserving only
            // Row headers can exhaust the budget with mostly empty capacity,
            // forcing a spill even when the actual uniform rows would fit.
            let available = self.rows.len()
                + (self.budget - self.payload_bytes - self.rows.len() * slot) / (slot + payload);
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

    fn spill(&mut self, space: &mut Space) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let mut rows = std::mem::take(&mut self.rows);
        self.payload_bytes = 0;
        rows.sort_unstable();
        if self.reject_duplicate_keys && rows.windows(2).any(|pair| pair[0].key == pair[1].key) {
            return Err(Error::new(ErrorKind::RecordAlreadyExists));
        }
        let (mut run, mut output) = space.writer()?;
        for row in rows {
            space.append(&mut run, &mut output, &row)?;
        }
        output.flush().map_err(io_error)?;
        drop(output);
        let mut level = 0;
        loop {
            if level == self.runs.len() {
                self.runs.push(Vec::with_capacity(self.fan_in));
            }
            self.runs[level].push(run);
            if self.runs[level].len() < self.fan_in {
                break;
            }
            run = space.merge(std::mem::take(&mut self.runs[level]))?;
            level += 1;
        }
        Ok(())
    }

    /// Checks global key uniqueness and reclaims spilled runs. A buffered-only
    /// input needs no output file because its sorted rows are consumed here.
    pub fn unique_keys(mut self, space: &mut Space) -> Result<bool> {
        if self.runs.is_empty() {
            self.rows.sort_unstable();
            return Ok(self.rows.windows(2).all(|rows| rows[0].key != rows[1].key));
        }
        let run = self.finish(space)?;
        let mut reader = space.reader(&run)?;
        let mut previous = None;
        let mut unique = true;
        while let Some(row) = reader.next()? {
            if previous.as_ref() == Some(&row.key) {
                unique = false;
                break;
            }
            previous = Some(row.key);
        }
        drop(reader);
        space.remove(run)?;
        Ok(unique)
    }

    pub fn finish(mut self, space: &mut Space) -> Result<Run> {
        self.spill(space)?;
        let mut runs: Vec<_> = self.runs.into_iter().flatten().collect();
        // Merge smaller runs first. The first partial fan-in leaves a count
        // congruent to one modulo (fan_in-1), avoiding repeated large merges.
        while runs.len() > 1 {
            runs.sort_unstable_by_key(|run| (run.bytes, run.file_id));
            let count = if runs.len() <= self.fan_in {
                runs.len()
            } else {
                2 + (runs.len() - 2) % (self.fan_in - 1)
            };
            let inputs = runs.drain(..count).collect();
            runs.push(space.merge(inputs)?);
        }
        match runs.pop() {
            Some(run) => Ok(run),
            None => {
                let (run, mut file) = space.writer()?;
                file.flush().map_err(io_error)?;
                Ok(run)
            }
        }
    }
}

fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Directory(PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn paired_sorters_do_not_spill_uniform_rows_that_fit_the_budget() {
        let directory = Directory(
            std::env::temp_dir().join(format!("ktann-paired-fit-{}", std::process::id())),
        );
        let mut space = Space::new(&directory.0, 1024 * 1024, 128).unwrap();
        let (mut ids, mut trees) = Sorter::pair(&space, 2 * 1024 * 1024).unwrap();
        for id in 0..10000_u64 {
            ids.push(
                &mut space,
                Row {
                    key: id.to_be_bytes().to_vec(),
                    value: Vec::new(),
                },
            )
            .unwrap();
            trees
                .push(
                    &mut space,
                    Row {
                        key: Vec::new(),
                        value: vec![0; 18],
                    },
                )
                .unwrap();
        }
        assert_eq!(space.written, 0);
        assert!(ids.unique_keys(&mut space).unwrap());
        let run = trees.finish(&mut space).unwrap();
        space.remove(run).unwrap();
        assert_eq!(space.live, 0);
    }

    #[test]
    fn paired_sorters_accept_the_minimum_shared_memory_budget() {
        let directory =
            Directory(std::env::temp_dir().join(format!("ktann-paired-{}", std::process::id())));
        let row_bound = 128;
        let memory = 512 * 1024 + 6 * row_bound + 2 * std::mem::size_of::<Row>();
        let space = Space::new(&directory.0, 1024 * 1024, row_bound).unwrap();
        assert!(Sorter::pair(&space, memory).is_ok());
        assert_eq!(
            Sorter::pair(&space, memory - 1).err().unwrap().kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn uniqueness_checks_buffered_and_spilled_keys_and_reclaims_runs() {
        for memory in [
            512 * 1024 + 5 * 128 + std::mem::size_of::<Row>(),
            1024 * 1024,
        ] {
            for duplicate in [false, true] {
                let directory = Directory(std::env::temp_dir().join(format!(
                    "ktann-unique-{}-{memory}-{duplicate}",
                    std::process::id()
                )));
                let mut space = Space::new(&directory.0, 1024 * 1024, 128).unwrap();
                let mut sorter = Sorter::new(&space, memory).unwrap();
                for id in (0..400_u16).rev() {
                    let key = if duplicate && id == 399 { 0 } else { id };
                    sorter
                        .push(
                            &mut space,
                            Row {
                                key: key.to_be_bytes().to_vec(),
                                value: id.to_be_bytes().to_vec(),
                            },
                        )
                        .unwrap();
                }
                assert_eq!(sorter.unique_keys(&mut space).unwrap(), !duplicate);
                assert_eq!(space.live, 0);
                assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
            }
        }
    }

    #[test]
    fn bounded_merges_preserve_full_lexical_order_and_reclaim_runs() {
        let row_bound = 128;
        let overhead = std::mem::size_of::<Row>();
        for (case, memory, count) in [
            (0, 512 * 1024 + 5 * row_bound + overhead, 400_u16),
            (
                1,
                512 * 1024 + 10 * row_bound + 10 * (row_bound + overhead),
                2000,
            ),
        ] {
            let directory = Directory(std::env::temp_dir().join(format!(
                    "ktann-merge-{}-{case}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                )));
            let mut space = Space::new(&directory.0, 1024 * 1024, row_bound).unwrap();
            let mut expected: Vec<_> = (0..count)
                .rev()
                .map(|id| {
                    (
                        (id % 17).to_be_bytes().to_vec(),
                        vec![(id % 251) as u8; (id % 80 + 1) as usize],
                    )
                })
                .collect();
            let mut sorter = Sorter::new(&space, memory).unwrap();
            for (key, value) in &expected {
                sorter
                    .push(
                        &mut space,
                        Row {
                            key: key.clone(),
                            value: value.clone(),
                        },
                    )
                    .unwrap();
            }
            let run = sorter.finish(&mut space).unwrap();
            let mut reader = space.reader(&run).unwrap();
            let mut actual = Vec::new();
            while let Some(row) = reader.next().unwrap() {
                actual.push((row.key, row.value));
            }
            expected.sort();
            assert_eq!(actual, expected);
            drop(reader);
            space.remove(run).unwrap();
            assert_eq!(space.live, 0);
            assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
        }
    }
}
