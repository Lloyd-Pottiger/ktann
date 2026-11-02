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
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{Backend, ScanLimits};
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{IndexManifest, PartitionState, PartitionTransition, PersistentValue};
use crate::storage::{LogicalRange, LogicalScanCursor, LogicalScanPage, topology};

use super::compute::controlled_compute;
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
        let history = &mut tree.rounds;
        refine(
            checkpoint,
            &kernel,
            &loaded.vectors,
            &mut tree.parts[..leaf_count],
            manifest.config().min_leaf_partition_entries() as usize,
            manifest.config().max_leaf_partition_entries() as usize,
            options.rounds,
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
        refresh_internal_centers(&kernel, &mut tree.parts, leaf_count, checkpoint)?;
        // Only final routing centers and ordered moves are needed during writes.
        for part in &mut tree.parts {
            part.members = Vec::new();
            part.children = Vec::new();
        }
    }
    Ok(loaded.trees)
}

/// Refreshes the hierarchy in child-before-parent order using unweighted means.
fn refresh_internal_centers(
    kernel: &VectorKernel,
    parts: &mut [Part],
    leaves: usize,
    check: &impl Fn() -> Result<()>,
) -> Result<BTreeMap<PartitionKey, usize>> {
    let positions: BTreeMap<_, _> = parts
        .iter()
        .enumerate()
        .map(|(i, part)| (part.key, i))
        .collect();
    for i in leaves..parts.len() {
        check()?;
        parts[i].center = mean(
            kernel,
            parts[i]
                .children
                .iter()
                .map(|key| parts[positions[key]].center.as_ref()),
        )?;
    }
    Ok(positions)
}

/// Pure artifact preparation reuses the ordinary numerical assignment algorithm.
/// The caller admits all retained input and planning scratch before invoking it.
pub(crate) fn refine_plans(
    config: &crate::api::IndexConfig,
    seed: [u8; 32],
    records: Vec<crate::construction::ConstructionRecord>,
    plans: &mut [crate::construction::PartitionPlan],
    rounds: u8,
    check: &impl Fn() -> Result<()>,
) -> Result<()> {
    let kernel = VectorKernel::new(config.dimension(), config.metric(), seed)?;
    let mut ids = Vec::with_capacity(records.len());
    let mut vectors = Vec::with_capacity(records.len());
    for record in records {
        check()?;
        ids.push(record.id);
        vectors.push(kernel.preprocess(&record.vector)?);
    }
    let positions: BTreeMap<_, _> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i))
        .collect();
    let mut parts = Vec::with_capacity(plans.len());
    for plan in plans.iter_mut() {
        check()?;
        let members = if plan.level == 1 {
            plan.entries
                .iter()
                .map(|id| positions.get(id).copied().ok_or_else(corrupt))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let children = if plan.level > 1 {
            plan.entries
                .iter()
                .map(|id| {
                    PartitionKey::new(u64::from_be_bytes(
                        id.as_ref().try_into().map_err(|_| corrupt())?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let center = if plan.level == 1 {
            mean(&kernel, members.iter().map(|&i| vectors[i].as_ref()))?
        } else {
            plan.centroid.clone()
        };
        parts.push(Part {
            key: plan.key,
            parent: None,
            level: plan.level,
            center,
            members,
            children,
        });
    }
    parts.sort_by_key(|part| (part.level, part.key));
    let leaves = parts.partition_point(|part| part.level == 1);
    refine(
        check,
        &kernel,
        &vectors,
        &mut parts[..leaves],
        config.min_leaf_partition_entries() as usize,
        config.max_leaf_partition_entries() as usize,
        rounds as usize,
        |_, _, _, _| {},
    )?;
    let positions = refresh_internal_centers(&kernel, &mut parts, leaves, check)?;
    for plan in plans {
        check()?;
        let part = &mut parts[positions[&plan.key]];
        plan.centroid = std::mem::take(&mut part.center);
        if plan.level == 1 {
            plan.entries = part.members.iter().map(|&i| ids[i].clone()).collect();
            plan.entries.sort();
        }
    }
    Ok(())
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
/// Only online planning retains accepted moves for later persistence; artifact
/// preparation needs only the final membership and centroids. Callers discard
/// their private planning state if any round fails.
#[expect(
    clippy::too_many_arguments,
    reason = "numerical inputs, occupancy bounds, and move observation are independent"
)]
fn refine(
    checkpoint: &impl Fn() -> Result<()>,
    kernel: &VectorKernel,
    vectors: &[Box<[f32]>],
    leaves: &mut [Part],
    minimum: usize,
    maximum: usize,
    rounds: usize,
    mut on_move: impl FnMut(usize, usize, PartitionKey, PartitionKey),
) -> Result<()> {
    if leaves.len() < 2 {
        return Ok(());
    }
    for round in 0..rounds {
        checkpoint()?;
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
                on_move(round, position, leaves[source].key, leaves[target].key);
                counts[source] -= 1;
                counts[target] += 1;
                targets.insert(position, target);
            }
        }
        crate::observe::metrics::refinement_round(targets.len());
        if targets.is_empty() {
            break;
        }
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_zero_rounds_refreshes_centroids_without_changing_membership() {
        use crate::api::{IndexConfig, Metric};
        use crate::construction::{ConstructionRecord, PartitionPlan};
        let config = IndexConfig::new(1, Metric::L2).unwrap();
        let mut plans = vec![PartitionPlan {
            key: PartitionKey::new(1).unwrap(),
            level: 1,
            centroid: vec![999.0].into_boxed_slice(),
            entries: vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")],
        }];
        let before = plans[0].entries.clone();
        let records = vec![
            ConstructionRecord {
                id: before[0].clone(),
                vector: vec![2.0].into_boxed_slice(),
            },
            ConstructionRecord {
                id: before[1].clone(),
                vector: vec![4.0].into_boxed_slice(),
            },
        ];
        refine_plans(&config, [0; 32], records, &mut plans, 0, &|| Ok(())).unwrap();
        assert_eq!(plans[0].entries, before);
        let kernel = VectorKernel::new(1, Metric::L2, [0; 32]).unwrap();
        let expected = mean(
            &kernel,
            [
                kernel.preprocess(&[2.0]).unwrap(),
                kernel.preprocess(&[4.0]).unwrap(),
            ]
            .iter()
            .map(AsRef::as_ref),
        )
        .unwrap();
        assert_eq!(plans[0].centroid, expected);
    }

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
        let mut moves = Vec::new();
        refine(
            &|| Ok(()),
            &kernel,
            &vectors,
            &mut leaves,
            2,
            2,
            options.rounds,
            |round, position, source, target| moves.push((round, position, source, target)),
        )
        .unwrap();
        assert_eq!(leaves[0].members, [0, 2]); // A full target cannot accept a move.
        assert!(moves.is_empty());
        refine(
            &|| Ok(()),
            &kernel,
            &vectors,
            &mut leaves,
            1,
            3,
            options.rounds,
            |round, position, source, target| moves.push((round, position, source, target)),
        )
        .unwrap();
        assert_eq!(leaves[0].members, [0, 1]);
        assert_eq!(leaves[1].members, [2, 3]);
        assert_eq!(leaves[0].center.as_ref(), [0.5]);
        assert_eq!(leaves[1].center.as_ref(), [9.5]);
        assert_eq!(
            moves,
            [
                (0, 1, leaves[1].key, leaves[0].key),
                (0, 2, leaves[0].key, leaves[1].key),
            ]
        );
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
        refine(
            &|| Ok(()),
            &kernel,
            &vectors,
            &mut leaves,
            1,
            5,
            options.rounds,
            |_, _, _, _| {},
        )
        .unwrap();
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
