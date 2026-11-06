//! Snapshot metadata validation, partition visit accounting and maintenance discovery.
//!
//! Root transitions are applied before empty-body or Synopsis pruning so every
//! committed topology remains searchable, even when the source body is skipped.

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::maintenance::fixup;
use crate::search::cache::graph::RoutingExecution;
use crate::search::predicate::{CompiledPredicate, SynopsisClassification};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::ReadOps;
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    PartitionHeader, PartitionState, PartitionSynopsis, PartitionTransition, PersistentValue,
    expect_header, expect_synopsis,
};

use super::VisitContext;
use super::frontier::{Frontier, FrontierEntry};

/// A funded, metadata-validated visit whose body survived Synopsis pruning.
pub(super) struct PreparedVisit<'a> {
    pub(super) entry: FrontierEntry,
    pub(super) header: PartitionHeader,
    pub(super) predicate: Option<&'a CompiledPredicate>,
}

/// Owns the effects of admitted metadata reads, including visits pruned without a body.
#[derive(Default)]
pub(super) struct PartitionReader {
    visited_partitions: u32,
    maintenance: Vec<(TreeKey, PartitionKey)>,
}

impl PartitionReader {
    pub(super) fn visited_partitions(&self) -> u32 {
        self.visited_partitions
    }

    pub(super) fn into_maintenance(self) -> Vec<(TreeKey, PartitionKey)> {
        self.maintenance
    }

    /// Charges a visit and validates metadata before any body prefetch.
    pub(super) async fn prepare<'a, T: ReadOps>(
        &mut self,
        txn: &mut ReadLogicalTxn<'_, T>,
        context: &VisitContext<'a, impl RoutingExecution>,
        frontier: &mut Frontier,
        entry: FrontierEntry,
        prefetched_metadata: Option<(PartitionHeader, Option<PartitionSynopsis>)>,
    ) -> Result<Option<PreparedVisit<'a>>> {
        self.visited_partitions = self
            .visited_partitions
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let index = context.manifest.logical_index_id();
        let tree_key = context.request.trees[entry.tree as usize].tree_key();

        // Single-partition visits also pair the Synopsis with its Header when
        // the referencing edge proves a leaf and the request needs a predicate.
        let (header, synopsis) = if let Some(metadata) = prefetched_metadata {
            metadata
        } else {
            let header_key = LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition: entry.partition,
            };
            if context.request.predicate.is_some() && entry.expected_level == Some(1) {
                let mut values = txn
                    .batch_get(vec![
                        header_key,
                        LogicalKey::Synopsis {
                            index,
                            tree_key: tree_key.clone(),
                            partition: entry.partition,
                        },
                    ])
                    .await?;
                let synopsis = expect_synopsis(values.pop().flatten())?
                    .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
                let header = expect_header(values.pop().flatten())?
                    .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
                (header, Some(synopsis))
            } else {
                let header = expect_header(txn.get(header_key).await?)?
                    .ok_or_else(|| Error::new(ErrorKind::Corruption))?;
                (header, None)
            }
        };

        // Every Child Entry descends exactly one level, so a referenced
        // partition's Header must agree with the level its edge established;
        // the same check rejects cycles before they can repeat a partition.
        if let Some(expected) = entry.expected_level
            && header.level() != expected
        {
            return Err(Error::new(ErrorKind::Corruption));
        }
        let level = header.level();
        // A visit rediscovers only work the committed Header can advance.
        if fixup::is_actionable(context.manifest.config(), entry.partition, header) {
            self.maintenance.push((tree_key.clone(), entry.partition));
        }

        if entry.expected_level.is_none() {
            Self::apply_root_state(txn, frontier, index, tree_key, entry, &header).await?;
        }

        // Root split targets have already been injected above. The exact count
        // proves an empty internal body has no graph or routing candidates.
        if level > 1
            && header.entry_count() == 0
            && context.manifest.config().local_graph().is_some()
        {
            return Ok(None);
        }

        let predicate = if level > 1 {
            None
        } else {
            let effective_predicate = match context.request.predicate {
                Some(predicate) => {
                    let synopsis = match synopsis {
                        Some(synopsis) => synopsis,
                        None => {
                            let value = txn
                                .get(LogicalKey::Synopsis {
                                    index,
                                    tree_key: tree_key.clone(),
                                    partition: entry.partition,
                                })
                                .await?;
                            expect_synopsis(value)?
                                .ok_or_else(|| Error::new(ErrorKind::Corruption))?
                        }
                    };
                    match predicate.classify(context.manifest, &synopsis, header.entry_count())? {
                        // The synopsis proves no entry can satisfy the
                        // predicate: prune the leaf without charging entries.
                        SynopsisClassification::NoMatch => return Ok(None),
                        // Every entry provably matches: skip per-entry
                        // evaluation but still charge every entry read.
                        SynopsisClassification::AllMatch => None,
                        SynopsisClassification::MayMatch => Some(predicate),
                    }
                }
                None => None,
            };
            // The exact Header count authoritatively proves emptiness.
            if header.entry_count() == 0 {
                return Ok(None);
            }
            effective_predicate
        };
        Ok(Some(PreparedVisit {
            entry,
            header,
            predicate,
        }))
    }

    /// Applies the root-only intermediate topology rules.
    ///
    /// A Splitting root's targets are unexposed, so only the root body is
    /// searched. A DrainingSplit root's targets have no incoming Child Entry
    /// yet, so both persisted targets join the root's own level's frontier
    /// explicitly. A root can never be a split target or merge source.
    async fn apply_root_state<T: ReadOps>(
        txn: &mut ReadLogicalTxn<'_, T>,
        frontier: &mut Frontier,
        index: crate::api::LogicalIndexId,
        tree_key: &TreeKey,
        entry: FrontierEntry,
        header: &PartitionHeader,
    ) -> Result<()> {
        match header.state() {
            PartitionState::Ready | PartitionState::Splitting => Ok(()),
            PartitionState::DrainingSplit => {
                let transition = txn
                    .get(LogicalKey::State {
                        index,
                        tree_key: tree_key.clone(),
                        partition: entry.partition,
                    })
                    .await?;
                let (left, right) =
                    match transition {
                        Some(PersistentValue::PartitionState(
                            PartitionTransition::DrainingSplit { left, right, .. },
                        )) => (left, right),
                        _ => return Err(Error::new(ErrorKind::Corruption)),
                    };
                frontier.inject(entry.tree, left, header.level())?;
                frontier.inject(entry.tree, right, header.level())?;
                Ok(())
            }
            PartitionState::ReceivingSplit | PartitionState::Merging => {
                Err(Error::new(ErrorKind::Corruption))
            }
        }
    }
}
