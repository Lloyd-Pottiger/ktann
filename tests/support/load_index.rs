//! The `load-index` corpus directive's fixture installer (issue #100, item C2).
//!
//! A `load-index` block annotates one tree's exact persistent topology in
//! `format-tree` shape — partition lines nested by two-space indentation, leaf
//! record lines in `insert` syntax — and installs it directly, including
//! in-flight split/merge intermediate states that are tedious to reach by
//! driving the state machines. Installed states are byte-equivalent to what
//! the state machines persist: the same authority keys (Header and State per
//! partition, non-root Centroid, leaf-only Synopsis, Child Entries including
//! the dual parent links of a non-root mid-drain target) and verbatim moved
//! Leaf Entry envelopes, so the `split-step`, `merge-step`, `search`,
//! `validate`, and `format-tree` directives all work on them afterwards.
//!
//! The install is insert-then-move: fixtures cannot encode Leaf Entries
//! themselves (the RaBitQ7 quantization path is `pub(crate)`), so every
//! fixture record first commits through the public mutation API into the
//! target tree's root leaf, the decoded envelopes are read back, and one
//! write transaction then rewrites the topology and moves the entries into
//! place. Header entry counts are always computed from what the installer
//! writes, never taken from the text.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use ktann::api::{Index, Mutation, PartitionKey, Record};
use ktann::storage::backend::Backend;
use ktann::storage::keys::{LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexManifest, LeafEntry, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionSynopsis, PartitionTransition, PersistentValue, RecordLocation, TreeManifest,
};
use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn, tree_manifest};

use super::{SharedBackend, read_manifest};

/// The state annotation of one fixture partition line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixtureState {
    /// The partition accepts its ordinary operations.
    Ready,
    /// Split target identities are reserved; the targets have no lines.
    Splitting {
        /// The left target.
        left: PartitionKey,
        /// The right target.
        right: PartitionKey,
    },
    /// A published target is receiving entries from one source.
    ReceivingSplit {
        /// The source partition.
        source: PartitionKey,
    },
    /// The source is draining into two published targets.
    DrainingSplit {
        /// The left target.
        left: PartitionKey,
        /// The right target.
        right: PartitionKey,
    },
    /// The source is draining into reselected Ready targets.
    Merging,
}

impl FixtureState {
    /// The Header discriminator corresponding to this state.
    fn kind(self) -> PartitionState {
        match self {
            Self::Ready => PartitionState::Ready,
            Self::Splitting { .. } => PartitionState::Splitting,
            Self::ReceivingSplit { .. } => PartitionState::ReceivingSplit,
            Self::DrainingSplit { .. } => PartitionState::DrainingSplit,
            Self::Merging => PartitionState::Merging,
        }
    }

    /// The persistent State value of this state at one start time.
    fn transition(self, started_at_unix_millis: u64) -> PartitionTransition {
        match self {
            Self::Ready => PartitionTransition::Ready {
                started_at_unix_millis,
            },
            Self::Splitting { left, right } => PartitionTransition::Splitting {
                left,
                right,
                started_at_unix_millis,
            },
            Self::ReceivingSplit { source } => PartitionTransition::ReceivingSplit {
                source,
                started_at_unix_millis,
            },
            Self::DrainingSplit { left, right } => PartitionTransition::DrainingSplit {
                left,
                right,
                started_at_unix_millis,
            },
            Self::Merging => PartitionTransition::Merging {
                started_at_unix_millis,
            },
        }
    }
}

/// One parsed partition line of a `load-index` fixture.
#[derive(Clone, Debug)]
pub struct FixturePartition {
    /// The partition key.
    pub key: PartitionKey,
    /// The annotated tree level; leaves are level 1.
    pub level: u32,
    /// The annotated state with its parameters.
    pub state: FixtureState,
    /// The centroid components: required on a non-root line, forbidden on the
    /// root (the persisted root never has a Centroid key).
    pub centroid: Option<Arc<[f32]>>,
    /// The enclosing line's partition key (`None` only for the root line): the
    /// parent in the indentation nesting, which for a `ReceivingSplit` line is
    /// its source.
    pub parent_line: Option<PartitionKey>,
    /// The Record IDs nested under this line, in text order (leaves only).
    pub records: Vec<Bytes>,
}

/// One parsed `load-index` fixture: one tree's exact annotated topology plus
/// the record bodies to install.
#[derive(Default)]
pub struct LoadFixture {
    /// The partition lines in text order; the first is the root.
    pub partitions: Vec<FixturePartition>,
    /// Every fixture record, in text order.
    pub records: Vec<Record>,
}

/// The derived install data for one validated fixture partition.
struct PlannedPartition {
    /// The partition holding this partition's incoming Child Entry: the
    /// nesting parent for an ordinary child, the source's parent for a
    /// non-root mid-drain target (dual-linked), and `None` for the root and
    /// for root-split targets.
    edge_parent: Option<PartitionKey>,
    /// The exact Header entry count: member entries for a leaf, Child Entries
    /// (including any target edges) for an internal partition.
    count: u32,
}

/// The derived install data for one validated fixture.
struct InstallPlan {
    /// One entry per fixture partition, in text order.
    partitions: Vec<PlannedPartition>,
    /// The greatest Partition Key mentioned anywhere (lines and
    /// left=/right=/source= references), so the persisted high-water mark
    /// keeps every named key reserved.
    high_water: u64,
    /// The deepest tree level in the fixture.
    max_level: u32,
}

impl LoadFixture {
    /// Validates the parsed fixture and derives its install plan.
    ///
    /// Every violation panics with the directive line: a malformed fixture is
    /// a corpus authoring error and must fail fast here, never as a later
    /// storage Corruption.
    fn validate(&self, line: usize) -> InstallPlan {
        let root = root_key();
        let Some(first) = self.partitions.first() else {
            panic!("load-index at line {line}: the fixture needs at least the root line");
        };
        assert!(
            first.key == root && first.parent_line.is_none(),
            "load-index at line {line}: the first line must be `pk=1`, the stable root"
        );
        assert!(
            first.centroid.is_none(),
            "load-index at line {line}: the persisted root never has a centroid"
        );
        assert!(
            !matches!(
                first.state,
                FixtureState::ReceivingSplit { .. } | FixtureState::Merging
            ),
            "load-index at line {line}: the root can never be a split target or merge source"
        );

        // Unique partition keys; per-line basics.
        let mut by_key: BTreeMap<PartitionKey, &FixturePartition> = BTreeMap::new();
        for partition in &self.partitions {
            assert!(
                by_key.insert(partition.key, partition).is_none(),
                "load-index at line {line}: duplicate partition key pk={}",
                partition.key.get()
            );
            assert!(
                partition.level >= 1,
                "load-index at line {line}: pk={} level must be at least 1",
                partition.key.get()
            );
            if partition.level > 1 {
                assert!(
                    partition.records.is_empty(),
                    "load-index at line {line}: record lines nest only under a level-1 leaf, \
                     but pk={} is level {}",
                    partition.key.get(),
                    partition.level
                );
            }
            if partition.key != root {
                assert!(
                    partition.parent_line.is_some(),
                    "load-index at line {line}: non-root partition pk={} needs a nesting parent \
                     line",
                    partition.key.get()
                );
                assert!(
                    partition.centroid.is_some(),
                    "load-index at line {line}: non-root partition pk={} needs `centroid=`",
                    partition.key.get()
                );
            }
        }

        // Unique Record IDs across the fixture.
        let mut record_ids: BTreeSet<&Bytes> = BTreeSet::new();
        for partition in &self.partitions {
            for id in &partition.records {
                assert!(
                    record_ids.insert(id),
                    "load-index at line {line}: duplicate record id {}",
                    super::datadriven::show_id(id)
                );
            }
        }

        // Split target references: a Splitting source's targets are reserved
        // and have no lines; a DrainingSplit source's targets must appear as
        // ReceivingSplit lines naming it.
        for partition in &self.partitions {
            let (left, right) = match partition.state {
                FixtureState::Splitting { left, right }
                | FixtureState::DrainingSplit { left, right } => (left, right),
                _ => continue,
            };
            assert!(
                left != right,
                "load-index at line {line}: pk={} split targets must differ",
                partition.key.get()
            );
            for target in [left, right] {
                if matches!(partition.state, FixtureState::Splitting { .. }) {
                    assert!(
                        !by_key.contains_key(&target),
                        "load-index at line {line}: Splitting pk={} target pk={} is reserved, \
                         not exposed: it must not appear as a line",
                        partition.key.get(),
                        target.get()
                    );
                    continue;
                }
                let target_line = by_key.get(&target).unwrap_or_else(|| {
                    panic!(
                        "load-index at line {line}: DrainingSplit pk={} target pk={} needs a \
                         partition line nested under the source",
                        partition.key.get(),
                        target.get()
                    )
                });
                assert!(
                    matches!(
                        target_line.state,
                        FixtureState::ReceivingSplit { source } if source == partition.key
                    ),
                    "load-index at line {line}: DrainingSplit pk={} target pk={} must be \
                     annotated `state=ReceivingSplit source={}`",
                    partition.key.get(),
                    target.get(),
                    partition.key.get()
                );
            }
        }

        // Levels and incoming edges: a real child sits exactly one level below
        // its parent; a ReceivingSplit target nests under its DrainingSplit
        // source at the source's own level.
        let mut edge_parents: Vec<Option<PartitionKey>> = Vec::with_capacity(self.partitions.len());
        for partition in &self.partitions {
            let edge_parent = if partition.key == root {
                None
            } else {
                match partition.state {
                    FixtureState::ReceivingSplit { source } => {
                        let source_line = by_key.get(&source).unwrap_or_else(|| {
                            panic!(
                                "load-index at line {line}: ReceivingSplit pk={} names unknown \
                                 source pk={}",
                                partition.key.get(),
                                source.get()
                            )
                        });
                        assert!(
                            matches!(
                                source_line.state,
                                FixtureState::DrainingSplit { left, right }
                                    if left == partition.key || right == partition.key
                            ),
                            "load-index at line {line}: ReceivingSplit pk={} source pk={} must \
                             be DrainingSplit and name it as a target",
                            partition.key.get(),
                            source.get()
                        );
                        assert!(
                            partition.parent_line == Some(source),
                            "load-index at line {line}: ReceivingSplit pk={} must nest under its \
                             source pk={}",
                            partition.key.get(),
                            source.get()
                        );
                        assert!(
                            partition.level == source_line.level,
                            "load-index at line {line}: ReceivingSplit pk={} level {} must equal \
                             its source's level {}",
                            partition.key.get(),
                            partition.level,
                            source_line.level
                        );
                        // A root-split target carries no parent edge; a
                        // non-root mid-drain target is dual-linked into the
                        // source's parent.
                        if source == root {
                            None
                        } else {
                            source_line.parent_line
                        }
                    }
                    _ => {
                        let parent = partition.parent_line.expect("validated non-root parent");
                        let parent_line = by_key.get(&parent).expect("parent line is known");
                        assert!(
                            Some(parent_line.level) == partition.level.checked_add(1),
                            "load-index at line {line}: pk={} level {} must be exactly one below \
                             its parent pk={} level {}",
                            partition.key.get(),
                            partition.level,
                            parent.get(),
                            parent_line.level
                        );
                        Some(parent)
                    }
                }
            };
            edge_parents.push(edge_parent);
        }

        // Exact counts: leaf member entries; internal Child Entries including
        // any dual-linked target edges.
        let counts: Vec<u32> = self
            .partitions
            .iter()
            .map(|partition| {
                if partition.level == 1 {
                    u32::try_from(partition.records.len()).expect("leaf entry count fits u32")
                } else {
                    let children = edge_parents
                        .iter()
                        .filter(|parent| **parent == Some(partition.key))
                        .count();
                    u32::try_from(children).expect("child entry count fits u32")
                }
            })
            .collect();

        let mut high_water = root.get();
        for partition in &self.partitions {
            high_water = high_water.max(partition.key.get());
            match partition.state {
                FixtureState::Splitting { left, right }
                | FixtureState::DrainingSplit { left, right } => {
                    high_water = high_water.max(left.get()).max(right.get());
                }
                FixtureState::ReceivingSplit { source } => {
                    high_water = high_water.max(source.get());
                }
                FixtureState::Ready | FixtureState::Merging => {}
            }
        }
        let max_level = self
            .partitions
            .iter()
            .map(|partition| partition.level)
            .max()
            .expect("a validated fixture has a root");

        InstallPlan {
            partitions: edge_parents
                .into_iter()
                .zip(counts)
                .map(|(edge_parent, count)| PlannedPartition { edge_parent, count })
                .collect(),
            high_water,
            max_level,
        }
    }
}

/// The counts describing one installed fixture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstallSummary {
    /// The installed record count.
    pub records: usize,
    /// The installed partition count.
    pub partitions: usize,
    /// The deepest installed tree level.
    pub max_level: u32,
}

/// Installs one fixture's exact persistent state into the target tree and
/// returns its summary counts.
///
/// `started_at_unix_millis` and `cache_epoch` come from the harness's
/// deterministic maintenance clock: the former fills every State timestamp
/// and the latter stamps every Header the install writes, so an earlier
/// `search` that warmed the runtime partition cache under the tree's
/// pre-install epochs can never serve a stale body.
pub async fn install(
    backend: &SharedBackend,
    index: &Index<SharedBackend>,
    fixture: &LoadFixture,
    tree_key: &TreeKey,
    started_at_unix_millis: u64,
    cache_epoch: u64,
    line: usize,
) -> InstallSummary {
    let plan = fixture.validate(line);
    let manifest = read_manifest(backend, index.logical_index_id()).await;
    let iid = manifest.logical_index_id();
    let root = root_key();

    // The target tree must be empty: not yet present, or present with an
    // empty level-1 Ready root (untouched since `new-index`). The persisted
    // high-water mark never moves backward, keeping every Partition Key the
    // tree ever reserved retired.
    let high_water = plan
        .high_water
        .max(existing_high_water(backend, &manifest, tree_key, line).await);

    // Phase 1: every fixture record commits through the public mutation API
    // and lands in the tree's root leaf (foreground inserts never refuse an
    // over-full leaf).
    if !fixture.records.is_empty() {
        index
            .batch_mutate(
                fixture
                    .records
                    .iter()
                    .cloned()
                    .map(Mutation::Insert)
                    .collect(),
            )
            .await
            .unwrap_or_else(|error| {
                panic!("load-index at line {line}: fixture insert failed: {error:?}")
            });
    }

    // Phase 2: read the decoded Leaf Entry envelopes back from the root; the
    // RaBitQ7 encoding is internal, so the install moves the committed
    // envelopes rather than re-encoding vectors.
    let mut entries = read_root_entries(backend, &manifest, tree_key).await;

    // Phase 3: one transaction writes the whole fixture state.
    let raw = backend.begin_write().await.expect("begin write");
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind index");
    txn.put(
        LogicalKey::TreeManifest {
            index: iid,
            tree_key: tree_key.clone(),
        },
        PersistentValue::TreeManifest(
            TreeManifest::new(
                root,
                PartitionKey::new(high_water).expect("the high-water mark is nonzero"),
            )
            .expect("valid tree manifest"),
        ),
    )
    .await
    .expect("write tree manifest");
    for (partition, planned) in fixture.partitions.iter().zip(&plan.partitions) {
        let key = partition.key;
        txn.put(
            LogicalKey::Header {
                index: iid,
                tree_key: tree_key.clone(),
                partition: key,
            },
            PersistentValue::PartitionHeader(
                PartitionHeader::new(
                    partition.level,
                    planned.count,
                    cache_epoch,
                    partition.state.kind(),
                )
                .expect("valid header"),
            ),
        )
        .await
        .expect("write header");
        txn.put(
            LogicalKey::State {
                index: iid,
                tree_key: tree_key.clone(),
                partition: key,
            },
            PersistentValue::PartitionState(partition.state.transition(started_at_unix_millis)),
        )
        .await
        .expect("write state");
        if let Some(centroid) = &partition.centroid {
            txn.put(
                LogicalKey::Centroid {
                    index: iid,
                    tree_key: tree_key.clone(),
                    partition: key,
                },
                PersistentValue::PartitionCentroid(PartitionCentroid::new(centroid.to_vec())),
            )
            .await
            .expect("write centroid");
        }
        if partition.level == 1 {
            // Leaf synopses cover exactly the member entries placed here.
            let mut synopsis = PartitionSynopsis::empty(&manifest);
            for id in &partition.records {
                let entry = entries.get(id).unwrap_or_else(|| {
                    panic!("load-index at line {line}: fixture record missing from the root leaf")
                });
                synopsis
                    .expand(&manifest, entry.fields())
                    .expect("expand synopsis");
            }
            txn.put(
                LogicalKey::Synopsis {
                    index: iid,
                    tree_key: tree_key.clone(),
                    partition: key,
                },
                PersistentValue::PartitionSynopsis(synopsis),
            )
            .await
            .expect("write synopsis");
        } else if key == root {
            // A root that becomes internal leaves its leaf-only Synopsis stale.
            txn.delete(LogicalKey::Synopsis {
                index: iid,
                tree_key: tree_key.clone(),
                partition: key,
            })
            .await
            .expect("delete stale root synopsis");
        }
        if let Some(parent) = planned.edge_parent {
            let centroid = partition
                .centroid
                .as_ref()
                .expect("validated non-root centroid");
            txn.put(
                LogicalKey::ChildEntry {
                    index: iid,
                    tree_key: tree_key.clone(),
                    partition: parent,
                    child: key,
                },
                PersistentValue::ChildEntry(ChildEntry::new(key, centroid.to_vec())),
            )
            .await
            .expect("write child entry");
        }
    }

    // Entry moves: every record placed outside the root moves its decoded
    // Leaf Entry to the target leaf and repoints its Record Location.
    for partition in &fixture.partitions {
        if partition.level != 1 || partition.key == root {
            continue;
        }
        for id in &partition.records {
            let entry = entries.remove(id).unwrap_or_else(|| {
                panic!("load-index at line {line}: fixture record missing from the root leaf")
            });
            txn.put(
                LogicalKey::LeafEntry {
                    index: iid,
                    tree_key: tree_key.clone(),
                    partition: partition.key,
                    id: id.clone(),
                },
                PersistentValue::LeafEntry(entry),
            )
            .await
            .expect("move leaf entry");
            txn.delete(LogicalKey::LeafEntry {
                index: iid,
                tree_key: tree_key.clone(),
                partition: root,
                id: id.clone(),
            })
            .await
            .expect("remove root entry");
            txn.put(
                LogicalKey::Location {
                    index: iid,
                    id: id.clone(),
                },
                PersistentValue::RecordLocation(RecordLocation::new(
                    tree_key.clone(),
                    partition.key,
                )),
            )
            .await
            .expect("repoint record location");
        }
    }
    txn.commit().await.expect("commit load-index install");

    InstallSummary {
        records: fixture.records.len(),
        partitions: fixture.partitions.len(),
        max_level: plan.max_level,
    }
}

/// The stable root Partition Key of every tree.
fn root_key() -> PartitionKey {
    PartitionKey::new(1).expect("Partition Key 1 is nonzero")
}

/// Reads the target tree's existing Partition Key high-water mark, proving on
/// the way that the tree is still empty: absent, or present with an empty
/// level-1 Ready root.
async fn existing_high_water(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    line: usize,
) -> u64 {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, manifest).expect("bind index");
    let Some(existing) = tree_manifest::read_tree_manifest(&mut txn, tree_key)
        .await
        .expect("read tree manifest")
    else {
        return root_key().get();
    };
    let iid = manifest.logical_index_id();
    let header = txn
        .get(LogicalKey::Header {
            index: iid,
            tree_key: tree_key.clone(),
            partition: root_key(),
        })
        .await
        .expect("read root header");
    let state = txn
        .get(LogicalKey::State {
            index: iid,
            tree_key: tree_key.clone(),
            partition: root_key(),
        })
        .await
        .expect("read root state");
    let empty = matches!(
        (&header, &state),
        (
            Some(PersistentValue::PartitionHeader(header)),
            Some(PersistentValue::PartitionState(PartitionTransition::Ready { .. }))
        ) if header.level() == 1 && header.entry_count() == 0 && header.state() == PartitionState::Ready
    );
    assert!(
        empty,
        "load-index at line {line}: the target tree must be empty (only an untouched \
         post-`new-index` tree may be loaded over)"
    );
    existing.partition_key_high_water().get()
}

/// Reads back the decoded Leaf Entry envelopes the insert phase committed to
/// the tree's root leaf, keyed by Record ID.
async fn read_root_entries(
    backend: &SharedBackend,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
) -> BTreeMap<Bytes, LeafEntry> {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::for_index(raw, manifest).expect("bind index");
    let range = LogicalRange::leaf_entries(manifest, tree_key, root_key()).expect("leaf range");
    let mut entries = BTreeMap::new();
    for item in super::audit::scan_all(&mut txn, &range)
        .await
        .expect("scan root leaf")
    {
        let LogicalKey::LeafEntry { id, .. } = item.key() else {
            panic!("a Leaf Entry range holds only Leaf Entries");
        };
        let id = id.clone();
        let PersistentValue::LeafEntry(entry) = item.into_value() else {
            panic!("a Leaf Entry range holds only Leaf Entries");
        };
        entries.insert(id, entry);
    }
    entries
}
