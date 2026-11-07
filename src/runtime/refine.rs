//! Caller-exclusive refinement of an existing, settled index.
//!
//! Planning is in memory. Persistence reuses ordinary write attempts and atomic
//! membership moves; there is no construction lifecycle or whole-operation commit.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use bytes::Bytes;
#[cfg(test)]
use tokio_util::sync::CancellationToken;

use crate::api::{Error, ErrorKind, PartitionKey, RefineOptions, Result};
use crate::observe::labels::Operation;
use crate::refinement::{Part, Planner};
use crate::storage::backend::{Backend, ScanLimits};
use crate::storage::graph_transitions::{GraphTransitionSource, protect_sources, publish_batch};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    IndexManifest, PartitionCentroid, PartitionState, PartitionTransition, PersistentValue,
};
use crate::storage::{LogicalRange, LogicalScanCursor, LogicalScanPage, topology};

use super::compute::controlled_compute;
use super::{OperationContext, check_control, lifecycle::RetryPolicy, reads, writes};

const PAGE: ScanLimits = ScanLimits {
    item_limit: 128,
    byte_limit: 1 << 20,
};

/// A loaded tree and its planned refinement rounds; member indexes refer to the shared record arrays.
struct Tree {
    key: TreeKey,
    parts: Vec<Part>,
    rounds: Vec<Vec<Move>>,
}

/// The bounded refinement input, with record IDs and vectors shared across the loaded topology.
struct Loaded {
    trees: Vec<Tree>,
    ids: Vec<Bytes>,
    vectors: Vec<Box<[f32]>>,
}

/// One accepted move; each round contains at most one move per record.
struct Move {
    source: PartitionKey,
    target: PartitionKey,
    id: Bytes,
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

fn charge(used: &mut usize, bytes: usize, options: &RefineOptions) -> Result<()> {
    *used = used
        .checked_add(bytes)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    if *used > options.input_bytes {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    Ok(())
}

/// Exclusive access permits short read transactions without a renewable public snapshot.
async fn page<B: Backend>(
    context: &OperationContext<B>,
    manifest: &IndexManifest,
    range: &LogicalRange,
    cursor: Option<&LogicalScanCursor>,
) -> Result<LogicalScanPage> {
    context.checkpoint()?;
    let backend = context.backend();
    reads::open_validated_read(backend.as_ref(), manifest)
        .await?
        .scan(range, cursor, PAGE)
        .await
}

/// Loads and cross-checks records, locations, and tree membership under the refinement input budget.
async fn load<B: Backend>(
    context: &OperationContext<B>,
    manifest: &IndexManifest,
    options: &RefineOptions,
) -> Result<Loaded> {
    let index = manifest.logical_index_id();
    let mut used = 0;
    let mut trees = Vec::new();
    let mut cursor = None;
    loop {
        let (items, next) = page(
            context,
            manifest,
            &LogicalRange::tree_manifests(manifest),
            cursor.as_ref(),
        )
        .await?
        .into_parts();
        for item in items {
            let LogicalKey::TreeManifest { tree_key, .. } = item.key() else {
                return Err(corrupt());
            };
            let PersistentValue::TreeManifest(tree) = item.value() else {
                return Err(corrupt());
            };
            charge(
                &mut used,
                tree_key.as_bytes().len() + std::mem::size_of::<Tree>(),
                options,
            )?;
            trees.push((tree_key.clone(), tree.root()));
        }
        cursor = next;
        if cursor.is_none() {
            break;
        }
    }
    let backend = context.backend();
    let mut loaded = Loaded {
        trees: Vec::new(),
        ids: Vec::new(),
        vectors: Vec::new(),
    };
    for (key, root) in trees {
        let mut pending = VecDeque::from([(root, None, None)]);
        let mut seen = BTreeSet::new();
        let mut parts = Vec::new();
        while let Some((partition, parent, expected_level)) = pending.pop_front() {
            context.checkpoint()?;
            if !seen.insert(partition) {
                return Err(corrupt());
            }
            charge(
                &mut used,
                std::mem::size_of::<Part>() + manifest.config().dimension() * 4,
                options,
            )?;
            let mut txn = reads::open_validated_read(backend.as_ref(), manifest).await?;
            let mut values = txn
                .batch_get(vec![
                    LogicalKey::Header {
                        index,
                        tree_key: key.clone(),
                        partition,
                    },
                    LogicalKey::State {
                        index,
                        tree_key: key.clone(),
                        partition,
                    },
                ])
                .await?
                .into_iter();
            let Some(Some(PersistentValue::PartitionHeader(header))) = values.next() else {
                return Err(corrupt());
            };
            let Some(Some(PersistentValue::PartitionState(state))) = values.next() else {
                return Err(corrupt());
            };
            if header.state() != PartitionState::Ready
                || !matches!(state, PartitionTransition::Ready { .. })
            {
                return Err(Error::invalid_argument());
            }
            if expected_level.is_some_and(|level| level != header.level()) {
                return Err(corrupt());
            }
            drop(txn);
            if header.entry_count() > manifest.config().max_partition_entries(header.level())
                || (parent.is_some()
                    && header.entry_count()
                        < manifest.config().min_partition_entries(header.level()))
            {
                return Err(Error::invalid_argument());
            }
            let mut part = Part {
                key: partition,
                parent,
                level: header.level(),
                centroid: PartitionCentroid::new(vec![0.0; manifest.config().dimension()]),
                members: Vec::new(),
                children: Vec::new(),
            };
            let range = if header.level() == 1 {
                LogicalRange::leaf_entries(manifest, &key, partition)?
            } else {
                LogicalRange::child_entries(manifest, &key, partition)?
            };
            let mut cursor = None;
            loop {
                let mut txn = reads::open_validated_read(backend.as_ref(), manifest).await?;
                context.checkpoint()?;
                let (items, next) = txn.scan(&range, cursor.as_ref(), PAGE).await?.into_parts();
                if header.level() == 1 {
                    let mut entries = Vec::with_capacity(items.len());
                    for item in items {
                        let PersistentValue::LeafEntry(entry) = item.into_value() else {
                            return Err(corrupt());
                        };
                        entries.push(entry);
                    }
                    let ids = entries
                        .iter()
                        .map(|entry| entry.record_id().clone())
                        .collect();
                    let records = txn.read_record_groups(ids, false).await?;
                    for (entry, group) in entries.into_iter().zip(records) {
                        let (record, location, _) = group.ok_or_else(corrupt)?.into_parts();
                        if record.fields() != entry.fields()
                            || location.tree_key() != &key
                            || location.leaf() != partition
                        {
                            return Err(corrupt());
                        }
                        charge(
                            &mut used,
                            record.record_id().len()
                                + record.vector().len() * 4
                                + std::mem::size_of::<Bytes>()
                                + std::mem::size_of::<usize>(),
                            options,
                        )?;
                        part.members.push(loaded.ids.len());
                        loaded.ids.push(record.record_id().clone());
                        loaded.vectors.push(record.vector().into());
                    }
                } else {
                    for item in items {
                        let PersistentValue::ChildEntry(child) = item.into_value() else {
                            return Err(corrupt());
                        };
                        charge(&mut used, std::mem::size_of::<PartitionKey>(), options)?;
                        part.children.push(child.child());
                        pending.push_back((
                            child.child(),
                            Some(partition),
                            Some(header.level() - 1),
                        ));
                    }
                }
                cursor = next;
                if cursor.is_none() {
                    break;
                }
            }
            if part.members.len() + part.children.len() != header.entry_count() as usize {
                return Err(corrupt());
            }
            parts.push(part);
        }
        loaded.trees.push(Tree {
            key,
            parts,
            rounds: Vec::new(),
        });
    }
    // Trees and rounds execute sequentially, so reserve only the largest graph.
    let mut candidate_bytes = 0;
    for tree in &loaded.trees {
        candidate_bytes = candidate_bytes.max(crate::refinement::candidate_scratch_bytes(
            tree.parts.iter().filter(|part| part.level == 1).count(),
            options.rounds,
        )?);
    }
    charge(&mut used, candidate_bytes, options)?;
    Ok(loaded)
}

/// Produces moves against existing leaves and recomputes the fixed routing hierarchy.
fn plan(
    mut loaded: Loaded,
    manifest: &IndexManifest,
    options: &RefineOptions,
    checkpoint: &impl Fn() -> Result<()>,
) -> Result<Vec<Tree>> {
    let planner = Planner::new(
        manifest.config(),
        *manifest.rotation_seed(),
        loaded.vectors,
        checkpoint,
    )?;
    for tree in &mut loaded.trees {
        let history = &mut tree.rounds;
        planner.refine_tree(
            &mut tree.parts,
            options.rounds,
            checkpoint,
            |round, position, source, target| {
                if history.len() == round {
                    history.push(Vec::new());
                }
                history[round].push(Move {
                    source,
                    target,
                    id: loaded.ids[position].clone(),
                });
            },
        )?;
        // Only final routing centers and ordered moves are needed during writes.
        for part in &mut tree.parts {
            part.members = Vec::new();
            part.children = Vec::new();
        }
    }
    Ok(loaded.trees)
}

/// Refines an ordinary index using existing retry/commit boundaries.
pub(crate) async fn run<B: Backend>(
    context: &mut OperationContext<B>,
    manifest: &IndexManifest,
    options: RefineOptions,
    retry: RetryPolicy,
) -> Result<()> {
    context.checkpoint()?;
    let loaded = load(context, manifest, &options).await?;
    let planning_manifest = manifest.clone();
    let trees = controlled_compute(
        context.options.clone(),
        context.cpu_admission.clone().expect("refinement admission"),
        move |control| {
            plan(loaded, &planning_manifest, &options, &|| {
                check_control(control)
            })
        },
    )
    .await?;
    let backend = context.backend();
    for tree in trees {
        for round in &tree.rounds {
            let mut offset = 0;
            let mut batch = 128;
            while offset < round.len() {
                let end = (offset + batch).min(round.len());
                let prefix = &round[offset..end];
                let result = writes::run_write_attempts(
                    backend.as_ref(),
                    Some(context),
                    manifest,
                    &retry,
                    Operation::Refine,
                    |txn| {
                        let key = &tree.key;
                        Box::pin(async move {
                            // One round moves each record at most once, so all sources
                            // exist at the start of this transaction.
                            let candidates: Vec<_> = prefix
                                .iter()
                                .map(|step| (step.source, step.id.clone()))
                                .collect();
                            let entries =
                                topology::read_leaf_drain_candidates(txn, key, &candidates).await?;
                            let moves = entries
                                .into_iter()
                                .zip(prefix)
                                .map(|(entry, step)| Ok((entry.ok_or_else(corrupt)?, step.target)))
                                .collect::<Result<Vec<_>>>()?;
                            topology::relocate_leaf_entries(
                                txn,
                                key,
                                moves,
                                topology::Movement::Refine,
                            )
                            .await?;
                            Ok(())
                        })
                    },
                )
                .await;
                match result {
                    Ok(()) => offset = end,
                    Err(error)
                        if batch > 1
                            && matches!(
                                error.kind(),
                                ErrorKind::LimitExceeded | ErrorKind::TransactionTooLarge
                            ) =>
                    {
                        batch = batch.div_ceil(2);
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        if manifest.config().local_graph().is_some() {
            publish_graph_centroids(context, manifest, &tree, &retry).await?;
            continue;
        }
        for part in &tree.parts {
            let Some(parent) = part.parent else {
                continue;
            };
            writes::run_write_attempts(
                backend.as_ref(),
                Some(context),
                manifest,
                &retry,
                Operation::Refine,
                |txn| {
                    let key = &tree.key;
                    Box::pin(async move {
                        topology::update_centroid(txn, key, part.key, parent, part.centroid.clone())
                            .await
                    })
                },
            )
            .await?;
        }
    }
    Ok(())
}

/// Publishes bounded sibling batches, child levels before parent levels. Preparing
/// one replacement per batch avoids rebuilding a wide graph for every centroid.
/// Caller exclusivity keeps plans stable; source fences still protect each commit.
async fn publish_graph_centroids<B: Backend>(
    context: &mut OperationContext<B>,
    manifest: &IndexManifest,
    tree: &Tree,
    retry: &RetryPolicy,
) -> Result<()> {
    let mut parents: BTreeMap<(u32, PartitionKey), Vec<&Part>> = BTreeMap::new();
    for part in &tree.parts {
        if let Some(parent) = part.parent {
            parents.entry((part.level, parent)).or_default().push(part);
        }
    }
    let backend = context.backend();
    for ((_, parent), parts) in parents {
        let mut offset = 0;
        let mut batch = parts.len().min(128);
        while offset < parts.len() {
            context.checkpoint()?;
            let end = (offset + batch).min(parts.len());
            let prefix = &parts[offset..end];
            let mut read = reads::open_validated_read(backend.as_ref(), manifest).await?;
            let source = GraphTransitionSource::read(
                &mut read,
                &tree.key,
                parent,
                context.routing_working_bytes,
                &|| context.checkpoint(),
            )
            .await?;
            drop(read);
            // Admit both the replacement body and the temporary update copies
            // before allocating them. Native preparation retains the same ceiling.
            let required = source
                .retained_bytes()?
                .checked_add(source.projection_bytes(source.children().len())?)
                .and_then(|bytes| bytes.checked_add(source.projection_bytes(prefix.len()).ok()?))
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
            if required > context.routing_working_bytes {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
            let updates: Vec<_> = prefix
                .iter()
                .map(|part| (part.key, part.centroid.clone()))
                .collect();
            let working_bytes = context.routing_working_bytes;
            let graphs = controlled_compute(
                context.options.clone(),
                context.cpu_admission.clone().expect("refinement admission"),
                move |control| {
                    Ok(
                        [source.prepare_centroid_update(updates, working_bytes, &|| {
                            check_control(control)
                        })?],
                    )
                },
            )
            .await?;
            let result = writes::run_write_attempts(
                backend.as_ref(),
                Some(context),
                manifest,
                retry,
                Operation::Refine,
                |txn| {
                    let graphs = &graphs;
                    let key = &tree.key;
                    Box::pin(async move {
                        if !protect_sources(txn, graphs).await? {
                            // Replanning around a changed parent would violate the
                            // caller-exclusive input used for numerical refinement.
                            return Err(Error::new(ErrorKind::ContentionExhausted));
                        }
                        for part in prefix {
                            topology::update_centroid(
                                txn,
                                key,
                                part.key,
                                parent,
                                part.centroid.clone(),
                            )
                            .await?;
                        }
                        publish_batch(txn, graphs).await
                    })
                },
            )
            .await;
            match result {
                Ok(()) => {
                    for graph in graphs {
                        graph.cache_committed(&context.partition_cache);
                    }
                    offset = end;
                }
                Err(error)
                    if batch > 1
                        && matches!(
                            error.kind(),
                            ErrorKind::LimitExceeded | ErrorKind::TransactionTooLarge
                        ) =>
                {
                    batch = batch.div_ceil(2);
                }
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_planning_cancels_only_its_child_and_retains_admission() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        };
        struct Held(Arc<AtomicBool>);
        impl Drop for Held {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let released = Arc::new(AtomicBool::new(false));
        let held = Held(released.clone());
        let caller = CancellationToken::new();
        let options = crate::api::OperationOptions::default().with_cancellation(caller.clone());
        let (started_send, started) = tokio::sync::oneshot::channel();
        let (resume, wait) = mpsc::channel();
        let (finished_send, finished) = tokio::sync::oneshot::channel();
        let owner = tokio::spawn(async move {
            controlled_compute(options, held, move |control| {
                started_send.send(()).unwrap();
                wait.recv_timeout(std::time::Duration::from_secs(10))
                    .unwrap();
                finished_send
                    .send(check_control(control).unwrap_err().kind())
                    .unwrap();
                Ok(())
            })
            .await
        });
        started.await.unwrap();
        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        assert!(!released.load(Ordering::SeqCst));
        resume.send(()).unwrap();
        assert_eq!(finished.await.unwrap(), ErrorKind::Cancelled);
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while !released.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(!caller.is_cancelled());
    }
}
