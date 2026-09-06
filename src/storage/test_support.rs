//! Test doubles and fixtures shared by core-crate unit tests.

use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound;

use bytes::Bytes;

use crate::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result};

use super::backend::{ReadOps, ScanItem, ScanLimits, ScanPage};
use super::keys::{self, KeyRange};

/// Returns a test Logical Index ID.
pub(crate) fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
}

/// Returns a test Partition Key.
pub(crate) fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

/// A snapshot read mock over committed key-value bytes.
///
/// Point reads serve queued scripted results before falling back to the map,
/// and batched gets enforce the configured backend batch ceiling. Scans count
/// themselves and page the map forward under the backend scan contract:
/// non-zero limits, one oversized first item carried alone, and a peek-ahead
/// continuation. The `with_failing_*` builders turn one operation family into
/// a tripwire for stages that must never use it.
pub(crate) struct MockReadTxn {
    pub(crate) data: BTreeMap<Vec<u8>, Vec<u8>>,
    /// Scripted point-read results; a non-empty queue overrides the map.
    pub(crate) scripted_gets: VecDeque<Option<Vec<u8>>>,
    /// The number of scans performed.
    pub(crate) scans: usize,
    /// The maximum number of keys one batched get accepts.
    pub(crate) max_batch_size: usize,
    scans_fail: bool,
    batch_gets_fail: bool,
}

impl MockReadTxn {
    /// Creates a read mock over `items`.
    pub(crate) fn new(items: impl IntoIterator<Item = (Vec<u8>, Vec<u8>)>) -> Self {
        Self {
            data: items.into_iter().collect(),
            scripted_gets: VecDeque::new(),
            scans: 0,
            max_batch_size: 10_000,
            scans_fail: false,
            batch_gets_fail: false,
        }
    }

    /// Fails every scan: the stage under test must only point-read.
    pub(crate) fn with_failing_scans(mut self) -> Self {
        self.scans_fail = true;
        self
    }

    /// Fails every batched get: the stage under test must not batch reads.
    pub(crate) fn with_failing_batch_gets(mut self) -> Self {
        self.batch_gets_fail = true;
        self
    }
}

impl ReadOps for MockReadTxn {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        if let Some(scripted) = self.scripted_gets.pop_front() {
            return Ok(scripted.map(Bytes::from));
        }
        Ok(self.data.get(key.as_ref()).cloned().map(Bytes::from))
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        if self.batch_gets_fail {
            return Err(Error::new(ErrorKind::Backend));
        }
        if keys.len() > self.max_batch_size {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        Ok(keys
            .iter()
            .map(|key| self.data.get(key.as_ref()).cloned().map(Bytes::from))
            .collect())
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        if self.scans_fail {
            return Err(Error::new(ErrorKind::Backend));
        }
        self.scans += 1;
        if limits.item_limit == 0 || limits.byte_limit == 0 {
            return Err(Error::invalid_argument());
        }
        let mut iter = self
            .data
            .range::<[u8], _>((Bound::Included(range.start()), Bound::Excluded(range.end())))
            .peekable();
        let mut items = Vec::new();
        let mut bytes = 0_usize;
        while let Some((key, value)) = iter.peek() {
            let size = key.len() + value.len();
            if items.is_empty() && size > limits.byte_limit {
                let (key, value) = iter.next().expect("peeked item exists");
                items.push(ScanItem::new(
                    Bytes::copy_from_slice(key),
                    Bytes::copy_from_slice(value),
                ));
                break;
            }
            if items.len() >= limits.item_limit || bytes + size > limits.byte_limit {
                break;
            }
            let (key, value) = iter.next().expect("peeked item exists");
            items.push(ScanItem::new(
                Bytes::copy_from_slice(key),
                Bytes::copy_from_slice(value),
            ));
            bytes += size;
        }
        if iter.peek().is_some() {
            ScanPage::continued(items, keys::MAX_TREE_KEY_BYTES + 64)
        } else {
            Ok(ScanPage::terminal(items))
        }
    }

    async fn batch_scan(
        &mut self,
        ranges: &[KeyRange],
        limits: ScanLimits,
    ) -> Result<Vec<ScanPage>> {
        let mut pages = Vec::with_capacity(ranges.len());
        for range in ranges {
            pages.push(self.scan(range, limits).await?);
        }
        Ok(pages)
    }
}
