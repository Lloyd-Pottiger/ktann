//! The snapshot-validated Partition Cache (design `search.md` section 8, ADR
//! 0010).
//!
//! The Runtime shares one byte-bounded, process-local cache of decoded
//! partition search bodies. Internal bodies hold Child Entries (Child Partition
//! Key and full-f32 centroid); leaf bodies hold Leaf Entries (Record ID,
//! absolute RaBitQ7 search data, and exact filter fields). Empty bodies are
//! cached too. State, Synopsis, Vector Record, and Payload data is never
//! cached.
//!
//! Validation follows the snapshot Header: a search reads the Header from its
//! own consistent snapshot, derives the body kind from the Header level, and
//! may reuse a cached body only when the cached epoch equals the snapshot
//! Header's cache epoch. A cached older epoch is a miss and is evicted; a
//! cached newer epoch means the search holds a historical snapshot, so it
//! misses without evicting the useful newer entry. On a miss the same search
//! transaction scans and decodes the complete body and rechecks the same
//! snapshot Header before publishing; it never fills from a separate latest
//! snapshot. Corruption is never cached: a body that fails to decode or fails
//! the Header recheck is a `Corruption` error and nothing is installed.
//!
//! Entries are immutable and never pinned. Concurrent misses may duplicate work
//! and race to publish equal or newer epochs; there is deliberately no
//! singleflight waiter or cancellation state. A body larger than the cache
//! capacity is served but never installed, keeping memory bounded. The eviction
//! policy is an internal benchmark-tunable detail, not a persistent or public
//! compatibility contract; cache warmth never changes logical search-budget
//! accounting.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "tree traversal (#9) and the public search operation (#30) consume the \
                  partition cache"
    )
)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result, Value};
use crate::storage::backend::{ReadOps, ScanLimits};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    ChildEntry, IndexManifest, LeafEntry, PartitionHeader, PersistentValue,
};
use crate::storage::{LogicalRange, LogicalScanCursor, ReadLogicalTxn};

/// One bounded page of a complete body scan.
///
/// A body is decoded page by page so one huge partition cannot force one
/// unbounded backend read; the decoded body itself is exactly what the cache
/// accounts and bounds.
const BODY_SCAN_LIMITS: ScanLimits = ScanLimits {
    item_limit: 256,
    byte_limit: 1_024 * 1_024,
};

/// The search-body kind of a partition, derived from its Header level.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum PartitionKind {
    /// A level-one partition holding Leaf Entries.
    Leaf,
    /// A partition above level one holding Child Entries.
    Internal,
}

impl PartitionKind {
    /// Derives the body kind from the snapshot Header level; leaves are level
    /// one.
    #[must_use]
    const fn from_level(level: u32) -> Self {
        if level == 1 {
            Self::Leaf
        } else {
            Self::Internal
        }
    }
}

/// The identity of one cached partition body.
///
/// The kind is part of the key, so a partition whose level changed can never
/// serve a body of the wrong kind.
#[derive(Clone, Eq, Hash, PartialEq)]
struct CacheKey {
    index: LogicalIndexId,
    tree_key: TreeKey,
    partition: PartitionKey,
    kind: PartitionKind,
}

/// The decoded search body of one partition.
#[derive(Clone, PartialEq)]
pub(crate) enum BodyEntries {
    /// The decoded Leaf Entries of a leaf partition.
    Leaf(Box<[LeafEntry]>),
    /// The decoded Child Entries of an internal partition.
    Internal(Box<[ChildEntry]>),
}

/// One immutable decoded partition body with its validation epoch.
///
/// Handles are shared behind `Arc`; entries are never mutated or pinned in
/// place.
pub(crate) struct CachedBody {
    epoch: u64,
    bytes: u64,
    entries: BodyEntries,
}

impl CachedBody {
    /// Returns the Header cache epoch this body was decoded from.
    #[must_use]
    pub(crate) const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns the accounted decoded size in bytes.
    #[must_use]
    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the decoded body entries.
    #[must_use]
    pub(crate) const fn entries(&self) -> &BodyEntries {
        &self.entries
    }
}

/// A byte-bounded process-shared cache of decoded partition search bodies.
pub(crate) struct PartitionCache {
    capacity_bytes: u64,
    inner: Mutex<CacheInner>,
}

struct CacheInner {
    slots: HashMap<CacheKey, CacheSlot>,
    /// The lazy LRU access log: one record per lookup hit or install, matched
    /// to its slot by tick. Stale records are skipped on eviction.
    access: VecDeque<AccessRecord>,
    tick: u64,
    total_bytes: u64,
}

struct CacheSlot {
    body: Arc<CachedBody>,
    tick: u64,
}

struct AccessRecord {
    key: CacheKey,
    tick: u64,
}

impl PartitionCache {
    /// Creates a cache holding at most `capacity_bytes` of decoded bodies.
    ///
    /// A zero capacity disables caching: every body is oversized and skipped.
    #[must_use]
    pub(crate) fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            inner: Mutex::new(CacheInner {
                slots: HashMap::new(),
                access: VecDeque::new(),
                tick: 0,
                total_bytes: 0,
            }),
        }
    }

    /// Returns the configured byte capacity.
    #[must_use]
    pub(crate) const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Looks up a body for the snapshot-visible Header epoch.
    ///
    /// An equal epoch is a hit. A cached older epoch is stale: it is evicted
    /// and reported as a miss. A cached newer epoch means the caller holds a
    /// historical snapshot: it misses without evicting the newer entry.
    #[must_use]
    pub(crate) fn lookup(&self, key_parts: &CacheKeyParts, epoch: u64) -> Option<Arc<CachedBody>> {
        let key = key_parts.key();
        let mut inner = self.lock();
        let cached_epoch = inner.slots.get(&key).map(|slot| slot.body.epoch);
        match cached_epoch {
            Some(cached) if cached == epoch => {
                inner.touch(&key);
                inner.slots.get(&key).map(|slot| Arc::clone(&slot.body))
            }
            Some(cached) if cached < epoch => {
                inner.remove(&key);
                None
            }
            Some(_) | None => None,
        }
    }

    /// Publishes a decoded body.
    ///
    /// A body larger than the capacity is skipped. An existing entry with a
    /// strictly newer epoch is never downgraded by a racing fill; an equal or
    /// newer epoch replaces it. Installation evicts until the cache is within
    /// its byte capacity.
    pub(crate) fn install(&self, key_parts: &CacheKeyParts, body: Arc<CachedBody>) {
        if body.bytes > self.capacity_bytes {
            return;
        }
        let key = key_parts.key();
        let mut inner = self.lock();
        if let Some(existing) = inner.slots.get(&key) {
            if existing.body.epoch > body.epoch {
                return;
            }
            inner.remove(&key);
        }
        let tick = inner.next_tick();
        inner.total_bytes += body.bytes;
        inner.slots.insert(key.clone(), CacheSlot { body, tick });
        inner.access.push_back(AccessRecord { key, tick });
        inner.evict_within_capacity(self.capacity_bytes);
        inner.compact_access_log();
    }

    /// Returns the number of cached bodies.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.lock().slots.len()
    }

    /// Returns the accounted bytes of all cached bodies.
    #[cfg(test)]
    fn total_bytes(&self) -> u64 {
        self.lock().total_bytes
    }

    fn lock(&self) -> MutexGuard<'_, CacheInner> {
        match self.inner.lock() {
            Ok(inner) => inner,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl CacheInner {
    fn next_tick(&mut self) -> u64 {
        if self.tick == u64::MAX {
            // Renumber from an empty access log; every live slot restarts at
            // tick zero so ticks stay unique within the new numbering.
            self.access.clear();
            for slot in self.slots.values_mut() {
                slot.tick = 0;
            }
            self.tick = 0;
        }
        self.tick += 1;
        self.tick
    }

    fn touch(&mut self, key: &CacheKey) {
        let tick = self.next_tick();
        if let Some(slot) = self.slots.get_mut(key) {
            slot.tick = tick;
        }
        self.access.push_back(AccessRecord {
            key: key.clone(),
            tick,
        });
        self.compact_access_log();
    }

    fn remove(&mut self, key: &CacheKey) {
        if let Some(slot) = self.slots.remove(key) {
            self.total_bytes -= slot.body.bytes;
        }
    }

    fn evict_within_capacity(&mut self, capacity_bytes: u64) {
        while self.total_bytes > capacity_bytes {
            let Some(record) = self.access.pop_front() else {
                // Every live slot has exactly one matching access record, so a
                // live over-capacity cache never reaches this guard.
                debug_assert!(self.slots.is_empty());
                self.slots.clear();
                self.total_bytes = 0;
                break;
            };
            if self
                .slots
                .get(&record.key)
                .is_some_and(|slot| slot.tick == record.tick)
            {
                self.remove(&record.key);
            }
        }
    }

    /// Rebuilds the access log when stale records dominate it.
    fn compact_access_log(&mut self) {
        if self.access.len() <= self.slots.len().saturating_mul(2).saturating_add(64) {
            return;
        }
        let mut records: Vec<AccessRecord> = self
            .slots
            .iter()
            .map(|(key, slot)| AccessRecord {
                key: key.clone(),
                tick: slot.tick,
            })
            .collect();
        records.sort_by_key(|record| record.tick);
        self.access = records.into_iter().collect();
    }
}

/// The caller-visible parts of a cache key: one partition of one tree of one
/// Logical Index, with the body kind derived from the snapshot Header.
#[derive(Clone)]
pub(crate) struct CacheKeyParts {
    index: LogicalIndexId,
    tree_key: TreeKey,
    partition: PartitionKey,
    kind: PartitionKind,
}

impl CacheKeyParts {
    /// Creates the key parts for one partition body.
    #[must_use]
    pub(crate) const fn new(
        index: LogicalIndexId,
        tree_key: TreeKey,
        partition: PartitionKey,
        kind: PartitionKind,
    ) -> Self {
        Self {
            index,
            tree_key,
            partition,
            kind,
        }
    }

    fn key(&self) -> CacheKey {
        CacheKey {
            index: self.index,
            tree_key: self.tree_key.clone(),
            partition: self.partition,
            kind: self.kind,
        }
    }
}

/// Loads one partition's decoded search body, validated against the snapshot
/// Header and cached under its cache epoch.
///
/// The Header is read first from `txn`'s one consistent snapshot. An equal
/// cached epoch is served from `cache` without touching the body. On a miss
/// the same transaction scans and decodes the complete body, the decoded entry
/// count must equal the Header's exact entry count, and the same snapshot
/// Header is rechecked before publishing; any disagreement is Corruption and
/// nothing is cached. The body is returned even when it is too large to cache.
pub(crate) async fn load_body<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    cache: &PartitionCache,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<Arc<CachedBody>> {
    let index = manifest.logical_index_id();
    let header = read_header(txn, index, tree_key, partition).await?;
    let kind = PartitionKind::from_level(header.level());
    let key_parts = CacheKeyParts::new(index, tree_key.clone(), partition, kind);
    if let Some(body) = cache.lookup(&key_parts, header.cache_epoch()) {
        return Ok(body);
    }

    let range = match kind {
        PartitionKind::Leaf => LogicalRange::leaf_entries(manifest, tree_key, partition),
        PartitionKind::Internal => LogicalRange::child_entries(manifest, tree_key, partition),
    }?;

    let mut leaf_entries: Vec<LeafEntry> = Vec::new();
    let mut child_entries: Vec<ChildEntry> = Vec::new();
    let mut bytes = size_of::<CachedBody>() as u64;
    let mut cursor: Option<LogicalScanCursor> = None;
    loop {
        let page = txn.scan(&range, cursor.as_ref(), BODY_SCAN_LIMITS).await?;
        for item in page.items() {
            match (kind, item.value()) {
                (PartitionKind::Leaf, PersistentValue::LeafEntry(entry)) => {
                    bytes = bytes.saturating_add(leaf_entry_bytes(entry));
                    leaf_entries.push(entry.clone());
                }
                (PartitionKind::Internal, PersistentValue::ChildEntry(entry)) => {
                    bytes = bytes.saturating_add(child_entry_bytes(entry));
                    child_entries.push(entry.clone());
                }
                _ => return Err(corruption()),
            }
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    let entry_count = leaf_entries.len() + child_entries.len();
    if entry_count != header.entry_count() as usize {
        return Err(corruption());
    }

    // Recheck the same snapshot Header before publishing: the body is only
    // cacheable under the epoch it was decoded from.
    let rechecked = read_header(txn, index, tree_key, partition).await?;
    if rechecked != header {
        return Err(corruption());
    }

    let entries = match kind {
        PartitionKind::Leaf => BodyEntries::Leaf(leaf_entries.into_boxed_slice()),
        PartitionKind::Internal => BodyEntries::Internal(child_entries.into_boxed_slice()),
    };
    let body = Arc::new(CachedBody {
        epoch: header.cache_epoch(),
        bytes,
        entries,
    });
    cache.install(&key_parts, Arc::clone(&body));
    Ok(body)
}

async fn read_header<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    index: LogicalIndexId,
    tree_key: &TreeKey,
    partition: PartitionKey,
) -> Result<PartitionHeader> {
    let key = LogicalKey::Header {
        index,
        tree_key: tree_key.clone(),
        partition,
    };
    match txn.get(key).await? {
        Some(PersistentValue::PartitionHeader(header)) => Ok(header),
        _ => Err(corruption()),
    }
}

/// The accounted decoded size of one Leaf Entry: the envelope, the Record ID
/// and RaBitQ7 byte strings, and every filter field including string payloads.
fn leaf_entry_bytes(entry: &LeafEntry) -> u64 {
    let mut bytes = size_of::<LeafEntry>() as u64;
    bytes = bytes.saturating_add(entry.record_id().len() as u64);
    bytes = bytes.saturating_add(entry.rabitq7().len() as u64);
    for field in entry.fields() {
        bytes = bytes.saturating_add(field_bytes(field));
    }
    bytes
}

fn field_bytes(field: &Value) -> u64 {
    let payload = match field {
        Value::String(value) => value.len(),
        _ => 0,
    };
    (size_of::<Value>() as u64).saturating_add(payload as u64)
}

/// The accounted decoded size of one Child Entry: the envelope, the Child
/// Partition Key, and the full-f32 centroid.
fn child_entry_bytes(entry: &ChildEntry) -> u64 {
    (size_of::<ChildEntry>() as u64).saturating_add(4 * entry.centroid().len() as u64)
}

fn corruption() -> Error {
    Error::new(ErrorKind::Corruption)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    use std::ops::Bound;
    use std::sync::Arc;

    use bytes::Bytes;

    use crate::api::{Error, ErrorKind, IndexConfig, LogicalIndexId, Metric, PartitionKey, Result};
    use crate::storage::ReadLogicalTxn;
    use crate::storage::backend::{ReadOps, ScanItem, ScanLimits, ScanPage};
    use crate::storage::keys::{self, KeyRange, TreeKey};
    use crate::storage::values::{
        ChildEntry, IndexLifecycle, IndexManifest, LeafEntry, PartitionHeader, PartitionState,
        PersistentValue, ValueCodec,
    };

    use super::super::rabitq::RaBitQ7;
    use super::{
        BodyEntries, CacheKeyParts, CachedBody, PartitionCache, PartitionKind, child_entry_bytes,
        leaf_entry_bytes, load_body,
    };

    const MAX_KEY_BYTES: usize = 1_024;

    fn index() -> LogicalIndexId {
        LogicalIndexId::new(7).expect("test Logical Index ID is nonzero")
    }

    fn pk(value: u64) -> PartitionKey {
        PartitionKey::new(value).expect("test Partition Key is nonzero")
    }

    fn manifest() -> IndexManifest {
        IndexManifest::new(
            IndexLifecycle::Active,
            index(),
            IndexConfig::new(1, Metric::L2).expect("valid config"),
            [7; 32],
            vec![],
        )
        .expect("valid manifest")
    }

    fn tree_key() -> TreeKey {
        TreeKey::encode(&[], &[]).expect("empty Tree Key is canonical")
    }

    fn parts(partition: u64, kind: PartitionKind) -> CacheKeyParts {
        CacheKeyParts::new(index(), tree_key(), pk(partition), kind)
    }

    fn leaf_entry(id: &[u8]) -> LeafEntry {
        LeafEntry::new(
            Bytes::copy_from_slice(id),
            Vec::new(),
            RaBitQ7::quantize(&[1.0]).expect("valid vector"),
        )
    }

    fn child_entry(child: u64) -> ChildEntry {
        ChildEntry::new(pk(child), vec![1.0])
    }

    fn cached_leaf_body(epoch: u64, ids: &[&[u8]]) -> Arc<CachedBody> {
        let entries: Vec<LeafEntry> = ids.iter().map(|id| leaf_entry(id)).collect();
        let mut bytes = size_of::<CachedBody>() as u64;
        for entry in &entries {
            bytes += leaf_entry_bytes(entry);
        }
        Arc::new(CachedBody {
            epoch,
            bytes,
            entries: BodyEntries::Leaf(entries.into_boxed_slice()),
        })
    }

    fn cached_internal_body(epoch: u64, children: &[u64]) -> Arc<CachedBody> {
        let entries: Vec<ChildEntry> = children.iter().map(|child| child_entry(*child)).collect();
        let mut bytes = size_of::<CachedBody>() as u64;
        for entry in &entries {
            bytes += child_entry_bytes(entry);
        }
        Arc::new(CachedBody {
            epoch,
            bytes,
            entries: BodyEntries::Internal(entries.into_boxed_slice()),
        })
    }

    fn leaf_ids(body: &CachedBody) -> Vec<Bytes> {
        match body.entries() {
            BodyEntries::Leaf(entries) => entries
                .iter()
                .map(|entry| entry.record_id().clone())
                .collect(),
            BodyEntries::Internal(_) => panic!("expected a leaf body"),
        }
    }

    #[test]
    fn an_equal_epoch_is_a_hit() {
        let cache = PartitionCache::new(1 << 20);
        let key = parts(1, PartitionKind::Leaf);
        cache.install(&key, cached_leaf_body(5, &[b"a"]));

        let hit = cache.lookup(&key, 5).expect("equal epoch hits");
        assert_eq!(hit.epoch(), 5);
        assert_eq!(leaf_ids(&hit), vec![Bytes::from_static(b"a")]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn a_cached_older_epoch_misses_and_is_evicted() {
        let cache = PartitionCache::new(1 << 20);
        let key = parts(1, PartitionKind::Leaf);
        cache.install(&key, cached_leaf_body(5, &[b"a"]));

        assert!(cache.lookup(&key, 6).is_none());
        // The stale entry was evicted, so its own epoch misses too.
        assert!(cache.lookup(&key, 5).is_none());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn a_cached_newer_epoch_survives_a_historical_lookup() {
        let cache = PartitionCache::new(1 << 20);
        let key = parts(1, PartitionKind::Leaf);
        cache.install(&key, cached_leaf_body(6, &[b"a"]));

        // A historical snapshot misses but must not evict the newer entry.
        assert!(cache.lookup(&key, 5).is_none());
        assert!(cache.lookup(&key, 6).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn install_never_downgrades_a_newer_epoch() {
        let cache = PartitionCache::new(1 << 20);
        let key = parts(1, PartitionKind::Leaf);
        cache.install(&key, cached_leaf_body(6, &[b"new"]));
        cache.install(&key, cached_leaf_body(5, &[b"old"]));
        assert_eq!(
            leaf_ids(&cache.lookup(&key, 6).expect("hit")),
            vec![Bytes::from_static(b"new")]
        );
        assert!(cache.lookup(&key, 5).is_none());

        // A racing fill with an equal epoch may replace the entry.
        cache.install(&key, cached_leaf_body(6, &[b"other"]));
        assert_eq!(
            leaf_ids(&cache.lookup(&key, 6).expect("hit")),
            vec![Bytes::from_static(b"other")]
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn kinds_do_not_share_entries() {
        let cache = PartitionCache::new(1 << 20);
        let leaf = parts(1, PartitionKind::Leaf);
        let internal = parts(1, PartitionKind::Internal);
        cache.install(&leaf, cached_leaf_body(3, &[b"a"]));
        cache.install(&internal, cached_internal_body(3, &[2, 3]));

        assert!(matches!(
            cache.lookup(&leaf, 3).expect("leaf hit").entries(),
            BodyEntries::Leaf(_)
        ));
        assert!(matches!(
            cache.lookup(&internal, 3).expect("internal hit").entries(),
            BodyEntries::Internal(_)
        ));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn an_oversized_body_is_skipped() {
        let body = cached_leaf_body(1, &[b"a"]);
        let cache = PartitionCache::new(body.bytes() - 1);
        let key = parts(1, PartitionKind::Leaf);
        cache.install(&key, body);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_bytes(), 0);
        assert!(cache.lookup(&key, 1).is_none());

        // A zero capacity disables caching entirely, even for empty bodies.
        let disabled = PartitionCache::new(0);
        disabled.install(&key, cached_leaf_body(1, &[]));
        assert_eq!(disabled.len(), 0);
    }

    #[test]
    fn total_bytes_stay_within_capacity_under_churn() {
        let cache = PartitionCache::new(4_096);
        let mut state = 0x2545_F491_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        let mut installed_any_eviction = false;
        for round in 0..200_u64 {
            let partition = next() % 64 + 1;
            let epoch = round + 1;
            let id = [(next() % 26) as u8 + b'a'];
            let key = parts(u64::from(partition), PartitionKind::Leaf);
            cache.install(&key, cached_leaf_body(epoch, &[&id]));
            assert!(cache.total_bytes() <= cache.capacity_bytes());
            // Every lookup that hits returns exactly the requested epoch.
            let query_epoch = u64::from(next()) % (epoch + 1);
            if let Some(hit) = cache.lookup(&key, query_epoch) {
                assert_eq!(hit.epoch(), query_epoch);
            }
            installed_any_eviction |= cache.len() < 64;
        }
        assert!(installed_any_eviction, "the churn must exercise eviction");
    }

    #[test]
    fn concurrent_fills_never_serve_a_stale_epoch() {
        let cache = Arc::new(PartitionCache::new(1 << 16));
        let mut handles = Vec::new();
        for thread in 0..8_u32 {
            let cache = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                let mut state = 0x9E37_79B9_u32.wrapping_mul(thread + 1) | 1;
                for _ in 0..5_000 {
                    // xorshift32: deterministic per thread, replayable.
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    let partition = u64::from(state % 4 + 1);
                    let epoch = u64::from(state % 8 + 1);
                    let key = parts(partition, PartitionKind::Leaf);
                    if state % 2 == 0 {
                        if let Some(hit) = cache.lookup(&key, epoch) {
                            assert_eq!(hit.epoch(), epoch);
                        }
                    } else {
                        cache.install(&key, cached_leaf_body(epoch, &[b"x"]));
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("cache thread did not panic");
        }
        // Every remaining entry still validates against its own epoch.
        for partition in 1..=4_u64 {
            for epoch in 1..=8_u64 {
                let key = parts(partition, PartitionKind::Leaf);
                if let Some(hit) = cache.lookup(&key, epoch) {
                    assert_eq!(hit.epoch(), epoch);
                }
            }
        }
    }

    /// A snapshot read mock over a raw key/value map, counting body scans.
    #[derive(Clone, Default)]
    struct MockReadTxn {
        data: BTreeMap<Vec<u8>, Vec<u8>>,
        scans: usize,
        /// Scripted point-read results; a non-empty queue overrides the map.
        scripted_gets: VecDeque<Option<Vec<u8>>>,
    }

    impl MockReadTxn {
        fn new(data: BTreeMap<Vec<u8>, Vec<u8>>) -> Self {
            Self {
                data,
                scans: 0,
                scripted_gets: VecDeque::new(),
            }
        }
    }

    impl ReadOps for MockReadTxn {
        async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
            if let Some(scripted) = self.scripted_gets.pop_front() {
                return Ok(scripted.map(Bytes::from));
            }
            Ok(self.data.get(key.as_ref()).cloned().map(Bytes::from))
        }

        async fn batch_get(&mut self, _keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
            Err(Error::new(ErrorKind::Backend))
        }

        async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
            self.scans += 1;
            let mut items = Vec::new();
            let mut bytes = 0_usize;
            let start = Bound::Included(range.start().to_vec());
            let end = Bound::Excluded(range.end().to_vec());
            for (key, value) in self.data.range((start, end)) {
                let item_bytes = key.len() + value.len();
                let full = items.len() >= limits.item_limit
                    || bytes.saturating_add(item_bytes) > limits.byte_limit;
                // A page always carries its first item, even when oversized.
                if full && !items.is_empty() {
                    return ScanPage::continued(items, MAX_KEY_BYTES);
                }
                bytes += item_bytes;
                items.push(ScanItem::new(
                    Bytes::copy_from_slice(key),
                    Bytes::copy_from_slice(value),
                ));
            }
            Ok(ScanPage::terminal(items))
        }
    }

    fn encode(manifest: &IndexManifest, value: &PersistentValue) -> Vec<u8> {
        ValueCodec::for_index(manifest)
            .encode(value)
            .expect("encode test value")
    }

    fn header_item(
        manifest: &IndexManifest,
        partition: u64,
        level: u32,
        entry_count: u32,
        epoch: u64,
    ) -> (Vec<u8>, Vec<u8>) {
        let key = keys::header_key(index(), &tree_key(), pk(partition));
        let header = PartitionHeader::new(level, entry_count, epoch, PartitionState::Ready)
            .expect("valid header");
        (
            key,
            encode(manifest, &PersistentValue::PartitionHeader(header)),
        )
    }

    fn leaf_item(manifest: &IndexManifest, partition: u64, id: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let key = keys::leaf_entry_key(
            index(),
            &tree_key(),
            pk(partition),
            &Bytes::copy_from_slice(id),
        )
        .expect("valid leaf entry key");
        (
            key,
            encode(manifest, &PersistentValue::LeafEntry(leaf_entry(id))),
        )
    }

    fn child_item(manifest: &IndexManifest, partition: u64, child: u64) -> (Vec<u8>, Vec<u8>) {
        let key = keys::child_entry_key(index(), &tree_key(), pk(partition), pk(child));
        (
            key,
            encode(manifest, &PersistentValue::ChildEntry(child_entry(child))),
        )
    }

    fn leaf_data(
        manifest: &IndexManifest,
        ids: &[&[u8]],
        epoch: u64,
    ) -> BTreeMap<Vec<u8>, Vec<u8>> {
        let mut data = BTreeMap::new();
        let (key, value) = header_item(manifest, 1, 1, ids.len() as u32, epoch);
        data.insert(key, value);
        for id in ids {
            let (key, value) = leaf_item(manifest, 1, id);
            data.insert(key, value);
        }
        data
    }

    async fn load(
        cache: &PartitionCache,
        manifest: &IndexManifest,
        mock: MockReadTxn,
        partition: u64,
    ) -> Result<(Arc<CachedBody>, MockReadTxn)> {
        let mut txn = ReadLogicalTxn::for_index(mock, manifest).expect("bind manifest");
        let body = load_body(&mut txn, cache, manifest, &tree_key(), pk(partition)).await?;
        Ok((body, txn.into_raw()))
    }

    #[tokio::test]
    async fn a_miss_scans_decodes_and_publishes_the_body() {
        let manifest = manifest();
        let cache = PartitionCache::new(1 << 20);
        let mock = MockReadTxn::new(leaf_data(&manifest, &[b"a", b"b"], 11));

        let (first, mock) = load(&cache, &manifest, mock, 1).await.expect("load");
        assert_eq!(mock.scans, 1);
        assert_eq!(first.epoch(), 11);
        assert_eq!(
            leaf_ids(&first),
            vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")]
        );

        let (second, mock) = load(&cache, &manifest, mock, 1).await.expect("load");
        assert_eq!(mock.scans, 1, "the equal-epoch hit performs no body scan");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn an_epoch_bump_forces_a_reload_and_evicts_the_stale_body() {
        let manifest = manifest();
        let cache = PartitionCache::new(1 << 20);
        let mock = MockReadTxn::new(leaf_data(&manifest, &[b"a"], 11));

        let (first, mut mock) = load(&cache, &manifest, mock, 1).await.expect("load");
        // A body mutation commits with exactly one epoch increment.
        mock.data = leaf_data(&manifest, &[b"b"], 12);
        let (second, mock) = load(&cache, &manifest, mock, 1).await.expect("load");
        assert_eq!(mock.scans, 2);
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(second.epoch(), 12);
        assert_eq!(leaf_ids(&second), vec![Bytes::from_static(b"b")]);

        // The stale epoch-11 entry was evicted by the first lookup at epoch 12.
        assert!(cache.lookup(&parts(1, PartitionKind::Leaf), 11).is_none());
        assert!(cache.lookup(&parts(1, PartitionKind::Leaf), 12).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[tokio::test]
    async fn internal_bodies_decode_child_entries() {
        let manifest = manifest();
        let cache = PartitionCache::new(1 << 20);
        let mut data = BTreeMap::new();
        let (key, value) = header_item(&manifest, 1, 2, 2, 7);
        data.insert(key, value);
        for child in [2, 3] {
            let (key, value) = child_item(&manifest, 1, child);
            data.insert(key, value);
        }

        let (body, _) = load(&cache, &manifest, MockReadTxn::new(data), 1)
            .await
            .expect("load");
        match body.entries() {
            BodyEntries::Internal(entries) => {
                let children: Vec<u64> = entries.iter().map(|entry| entry.child().get()).collect();
                assert_eq!(children, vec![2, 3]);
            }
            BodyEntries::Leaf(_) => panic!("expected an internal body"),
        }
    }

    #[tokio::test]
    async fn an_empty_body_is_cached() {
        let manifest = manifest();
        let cache = PartitionCache::new(1 << 20);
        let mock = MockReadTxn::new(leaf_data(&manifest, &[], 3));

        let (first, mock) = load(&cache, &manifest, mock, 1).await.expect("load");
        assert!(leaf_ids(&first).is_empty());
        let (second, mock) = load(&cache, &manifest, mock, 1).await.expect("load");
        assert_eq!(mock.scans, 1);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn a_missing_header_is_corruption() {
        let manifest = manifest();
        let cache = PartitionCache::new(1 << 20);
        let error = load(&cache, &manifest, MockReadTxn::default(), 1)
            .await
            .map(|_| ())
            .expect_err("a partition without a Header fails closed");
        assert_eq!(error.kind(), ErrorKind::Corruption);
        assert_eq!(cache.len(), 0);
    }

    #[tokio::test]
    async fn a_corrupt_body_fails_closed_and_is_never_cached() {
        let manifest = manifest();
        let cache = PartitionCache::new(1 << 20);
        let mut data = leaf_data(&manifest, &[b"a"], 11);
        let (entry_key, _) = leaf_item(&manifest, 1, b"a");
        data.insert(entry_key.clone(), vec![0xFF; 3]);

        let error = load(&cache, &manifest, MockReadTxn::new(data.clone()), 1)
            .await
            .map(|_| ())
            .expect_err("malformed entry bytes fail closed");
        assert_eq!(error.kind(), ErrorKind::Corruption);
        assert_eq!(cache.len(), 0, "corruption is never cached");

        // Once repaired, the same snapshot loads and caches normally.
        let mut repaired = data;
        let (_, value) = leaf_item(&manifest, 1, b"a");
        repaired.insert(entry_key, value);
        let (_, mock) = load(&cache, &manifest, MockReadTxn::new(repaired), 1)
            .await
            .expect("repaired load");
        assert_eq!(mock.scans, 1);
        assert_eq!(cache.len(), 1);
    }

    #[tokio::test]
    async fn an_entry_count_mismatch_is_corruption_and_not_cached() {
        let manifest = manifest();
        let cache = PartitionCache::new(1 << 20);
        let mut data = BTreeMap::new();
        // The Header claims two entries; the body holds only one.
        let (key, value) = header_item(&manifest, 1, 1, 2, 11);
        data.insert(key, value);
        let (key, value) = leaf_item(&manifest, 1, b"a");
        data.insert(key, value);

        let error = load(&cache, &manifest, MockReadTxn::new(data), 1)
            .await
            .map(|_| ())
            .expect_err("a divergent exact entry count fails closed");
        assert_eq!(error.kind(), ErrorKind::Corruption);
        assert_eq!(cache.len(), 0);
    }

    #[tokio::test]
    async fn a_header_recheck_mismatch_is_corruption_and_not_cached() {
        let manifest = manifest();
        let cache = PartitionCache::new(1 << 20);
        let mut mock = MockReadTxn::new(leaf_data(&manifest, &[b"a"], 11));
        // A misbehaving read path returns a changed Header on the recheck.
        let (_, epoch_11) = header_item(&manifest, 1, 1, 1, 11);
        let (_, epoch_12) = header_item(&manifest, 1, 1, 1, 12);
        mock.scripted_gets = [Some(epoch_11), Some(epoch_12)].into_iter().collect();

        let error = load(&cache, &manifest, mock, 1)
            .await
            .map(|_| ())
            .expect_err("a changed snapshot Header fails closed");
        assert_eq!(error.kind(), ErrorKind::Corruption);
        assert_eq!(cache.len(), 0, "a failed recheck publishes nothing");
    }

    #[tokio::test]
    async fn an_oversized_body_is_served_but_never_cached() {
        let manifest = manifest();
        let cache = PartitionCache::new(1);
        let mock = MockReadTxn::new(leaf_data(&manifest, &[b"a"], 11));

        let (body, mock) = load(&cache, &manifest, mock, 1).await.expect("load");
        assert_eq!(leaf_ids(&body), vec![Bytes::from_static(b"a")]);
        assert_eq!(cache.len(), 0);
        let (_, mock) = load(&cache, &manifest, mock, 1).await.expect("load");
        assert_eq!(mock.scans, 2, "an uncacheable body is re-scanned");
    }
}
