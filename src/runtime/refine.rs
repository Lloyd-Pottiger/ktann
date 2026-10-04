//! Caller-exclusive refinement of an existing, settled index.
//!
//! Planning is in memory. Persistence reuses ordinary write attempts and atomic
//! membership moves; there is no construction lifecycle or whole-operation commit.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use bytes::Bytes;
use tokio_util::sync::CancellationToken;

use crate::api::{Error, ErrorKind, PartitionKey, RefineOptions, Result};
use crate::observe::labels::Operation;
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{Backend, ScanLimits};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{IndexManifest, PartitionState, PartitionTransition, PersistentValue};
use crate::storage::{LogicalRange, LogicalScanCursor, LogicalScanPage, topology};

use super::{OperationContext, check_control, lifecycle::RetryPolicy, reads, writes};

const PAGE: ScanLimits = ScanLimits {
    item_limit: 128,
    byte_limit: 1 << 20,
};

/// Existing partitions keep their identities and parent edges during refinement.
struct Part {
    key: PartitionKey,
    parent: Option<PartitionKey>,
    level: u32,
    center: Box<[f32]>,
    members: Vec<usize>,
    children: Vec<PartitionKey>,
}

struct Tree {
    key: TreeKey,
    parts: Vec<Part>,
    rounds: Vec<Vec<Move>>,
}

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
            if header.entry_count() > manifest.config().max_partition_entries()
                || (parent.is_some()
                    && header.entry_count() < manifest.config().min_partition_entries())
            {
                return Err(Error::invalid_argument());
            }
            let mut part = Part {
                key: partition,
                parent,
                level: header.level(),
                center: vec![0.0; manifest.config().dimension()].into_boxed_slice(),
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
        parts.sort_by_key(|part| (part.level, part.key));
        loaded.trees.push(Tree {
            key,
            parts,
            rounds: Vec::new(),
        });
    }
    Ok(loaded)
}

/// Produces moves against existing leaves and recomputes the fixed routing hierarchy.
fn plan(
    mut loaded: Loaded,
    manifest: &IndexManifest,
    options: &RefineOptions,
    checkpoint: &impl Fn() -> Result<()>,
) -> Result<Vec<Tree>> {
    let kernel = VectorKernel::new(
        manifest.config().dimension(),
        manifest.config().metric(),
        *manifest.rotation_seed(),
    )?;
    for vector in &mut loaded.vectors {
        checkpoint()?;
        *vector = kernel.preprocess(vector)?;
    }
    for tree in &mut loaded.trees {
        let leaf_count = tree.parts.partition_point(|part| part.level == 1);
        // Recompute even unchanged leaf memberships: imported centroids may be stale.
        for leaf in &mut tree.parts[..leaf_count] {
            checkpoint()?;
            if !leaf.members.is_empty() {
                leaf.center = mean(
                    &kernel,
                    leaf.members
                        .iter()
                        .map(|&position| loaded.vectors[position].as_ref()),
                )?;
            }
        }
        tree.rounds = refine(
            checkpoint,
            &kernel,
            &loaded.vectors,
            &mut tree.parts[..leaf_count],
            manifest.config().min_partition_entries() as usize,
            manifest.config().max_partition_entries() as usize,
            options,
        )?
        .into_iter()
        .map(|round| {
            round
                .into_iter()
                .map(|(position, source, target)| Move {
                    source,
                    target,
                    id: loaded.ids[position].clone(),
                })
                .collect()
        })
        .collect();
        let positions: BTreeMap<_, _> = tree
            .parts
            .iter()
            .enumerate()
            .map(|(i, part)| (part.key, i))
            .collect();
        for i in leaf_count..tree.parts.len() {
            checkpoint()?;
            let center = mean(
                &kernel,
                tree.parts[i]
                    .children
                    .iter()
                    .map(|key| tree.parts[positions[key]].center.as_ref()),
            )?;
            tree.parts[i].center = center;
        }
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
                        topology::update_centroid(txn, key, part.key, parent, part.center.clone())
                            .await
                    })
                },
            )
            .await?;
        }
    }
    Ok(())
}
/// Computes a metric-correct mean with f64 accumulation in canonical order.
fn mean<'a>(kernel: &VectorKernel, members: impl Iterator<Item = &'a [f32]>) -> Result<Box<[f32]>> {
    let mut sum = vec![0.0_f64; kernel.dimension()];
    let mut count = 0_usize;
    for vector in members {
        for (sum, component) in sum.iter_mut().zip(vector) {
            *sum += f64::from(*component);
        }
        count += 1;
    }
    let center: Vec<f32> = sum
        .into_iter()
        .map(|sum| (sum / count.max(1) as f64) as f32)
        .collect();
    kernel.normalize_centroid(&center)
}

/// Greedy positive-gain moves use fixed centroids during one round, then
/// recompute spherical means. Exact counts constrain every accepted move.
/// The only root leaf is exempt from the ordinary nonroot minimum.
fn refine(
    checkpoint: &impl Fn() -> Result<()>,
    kernel: &VectorKernel,
    vectors: &[Box<[f32]>],
    leaves: &mut [Part],
    minimum: usize,
    maximum: usize,
    options: &RefineOptions,
) -> Result<Vec<Vec<(usize, PartitionKey, PartitionKey)>>> {
    let mut rounds = Vec::new();
    if leaves.len() < 2 {
        return Ok(rounds);
    }
    for _ in 0..options.rounds {
        checkpoint()?;
        let mut accepted = Vec::new();
        let mut moves = Vec::new();
        for (source, leaf) in leaves.iter().enumerate() {
            checkpoint()?;
            let mut distances = Vec::with_capacity(leaves.len() - 1);
            // Reuse the search kernel's independent accumulators without
            // changing any distance bits or the deterministic target ordering.
            let (batches, tail) = leaves.as_chunks::<4>();
            for (batch, candidates) in batches.iter().enumerate() {
                let scores = kernel.routing_centroid_distances(
                    &leaf.center,
                    candidates
                        .each_ref()
                        .map(|candidate| candidate.center.as_ref()),
                )?;
                for (lane, distance) in scores.into_iter().enumerate() {
                    let target = batch * 4 + lane;
                    if target != source {
                        distances.push((distance, target));
                    }
                }
            }
            for (offset, candidate) in tail.iter().enumerate() {
                let target = batches.len() * 4 + offset;
                if target != source {
                    distances.push((
                        kernel.routing_distance(&leaf.center, &candidate.center)?,
                        target,
                    ));
                }
            }
            let compare = |left: &(f64, usize), right: &(f64, usize)| {
                left.0.total_cmp(&right.0).then(left.1.cmp(&right.1))
            };
            if distances.len() > 32 {
                distances.select_nth_unstable_by(32, compare);
                distances.truncate(32);
            }
            distances.sort_unstable_by(compare);
            for &position in &leaf.members {
                let vector = &vectors[position];
                let old = kernel.routing_distance(vector, &leaf.center)?;
                let mut best = (old, source);
                let (batches, tail) = distances.as_chunks::<4>();
                for targets in batches {
                    let distances = kernel.routing_centroid_distances(
                        vector,
                        targets.map(|(_, target)| leaves[target].center.as_ref()),
                    )?;
                    for (&(_, target), distance) in targets.iter().zip(distances) {
                        if distance < best.0 {
                            best = (distance, target);
                        }
                    }
                }
                for &(_, target) in tail {
                    let distance = kernel.routing_distance(vector, &leaves[target].center)?;
                    if distance < best.0 {
                        best = (distance, target);
                    }
                }
                if best.1 != source {
                    moves.push((old - best.0, position, source, best.1));
                }
            }
        }
        moves.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
        let mut counts: Vec<_> = leaves.iter().map(|leaf| leaf.members.len()).collect();
        let mut targets = BTreeMap::new();
        for (_, position, source, target) in moves {
            if counts[source] > minimum && counts[target] < maximum {
                accepted.push((position, leaves[source].key, leaves[target].key));
                counts[source] -= 1;
                counts[target] += 1;
                targets.insert(position, target);
            }
        }
        crate::observe::metrics::refinement_round(targets.len());
        if targets.is_empty() {
            break;
        }
        rounds.push(accepted);
        let mut members = vec![Vec::new(); leaves.len()];
        for (source, leaf) in leaves.iter().enumerate() {
            for &position in &leaf.members {
                members[targets.get(&position).copied().unwrap_or(source)].push(position);
            }
        }
        for (leaf, mut members) in leaves.iter_mut().zip(members) {
            members.sort_unstable();
            leaf.center = mean(
                kernel,
                members.iter().map(|&position| vectors[position].as_ref()),
            )?;
            leaf.members = members;
        }
    }
    Ok(rounds)
}

/// Keeps heavy CPU work off Tokio and cancels it when its owning future drops.
/// Existing foreground admission bounds the number of these owned tasks.
async fn controlled_compute<T: Send + 'static>(
    options: crate::api::OperationOptions,
    keep_alive: impl Send + 'static,
    work: impl FnOnce(&crate::api::OperationOptions) -> Result<T> + Send + 'static,
) -> Result<T> {
    let cancellation = options
        .cancellation()
        .map(CancellationToken::child_token)
        .unwrap_or_default();
    let _guard = cancellation.clone().drop_guard();
    let options = options.with_cancellation(cancellation);
    tokio::task::spawn_blocking(move || {
        // Retain admission even after the owning async future is cancelled.
        let _keep_alive = keep_alive;
        check_control(&options)?;
        work(&options)
    })
    .await
    .map_err(|error| Error::with_source(ErrorKind::Other, error))?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refinement_improves_assignment_and_respects_occupancy() {
        let kernel = VectorKernel::new(1, crate::api::Metric::L2, [0; 32]).unwrap();
        let vectors: Vec<Box<[f32]>> = [0.0, 1.0, 9.0, 10.0]
            .into_iter()
            .map(|x| vec![x].into())
            .collect();
        let mut leaves = vec![
            Part {
                parent: None,
                children: vec![],
                key: PartitionKey::new(2).unwrap(),
                level: 1,
                members: vec![0, 2],
                center: vec![4.5].into(),
            },
            Part {
                parent: None,
                children: vec![],
                key: PartitionKey::new(3).unwrap(),
                level: 1,
                members: vec![1, 3],
                center: vec![5.5].into(),
            },
        ];
        let options = RefineOptions::new(100).unwrap();
        refine(&|| Ok(()), &kernel, &vectors, &mut leaves, 2, 2, &options).unwrap();
        assert_eq!(leaves[0].members, [0, 2]); // A full target cannot accept a move.
        refine(&|| Ok(()), &kernel, &vectors, &mut leaves, 1, 3, &options).unwrap();
        assert_eq!(leaves[0].members, [0, 1]);
        assert_eq!(leaves[1].members, [2, 3]);
        assert_eq!(leaves[0].center.as_ref(), [0.5]);
        assert_eq!(leaves[1].center.as_ref(), [9.5]);
    }

    #[test]
    fn refinement_batches_preserve_target_mapping_and_tail_neighbors() {
        let kernel = VectorKernel::new(1, crate::api::Metric::L2, [0; 32]).unwrap();
        let vectors: Vec<Box<[f32]>> = (0..9)
            .flat_map(|cluster| {
                [-0.1, 0.0, 0.1].map(|offset| vec![cluster as f32 * 10.0 + offset].into())
            })
            .collect();
        // Exchange one member between adjacent clusters. The ninth leaf is
        // unchanged and exercises the centroid and candidate batch tails.
        let mut leaves: Vec<_> = (0..9)
            .map(|cluster| Part {
                parent: None,
                children: vec![],
                key: PartitionKey::new(cluster as u64 + 2).unwrap(),
                level: 1,
                members: (cluster * 3..cluster * 3 + 3).collect(),
                center: vec![cluster as f32 * 10.0].into(),
            })
            .collect();
        for pair in leaves[..8].as_chunks_mut::<2>().0 {
            let (left, right) = pair.split_at_mut(1);
            std::mem::swap(&mut left[0].members[2], &mut right[0].members[2]);
        }
        let options = RefineOptions::new(4096).unwrap();
        refine(&|| Ok(()), &kernel, &vectors, &mut leaves, 1, 5, &options).unwrap();
        for (cluster, leaf) in leaves.iter().enumerate() {
            assert_eq!(
                leaf.members,
                (cluster * 3..cluster * 3 + 3).collect::<Vec<_>>(),
                "cluster={cluster}"
            );
        }
    }

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
