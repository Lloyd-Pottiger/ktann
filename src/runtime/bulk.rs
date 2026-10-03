//! Unpublished, fenced bulk construction and capacity-constrained refinement.

use std::collections::BTreeMap;
use tokio_util::sync::CancellationToken;

use crate::api::{
    BulkBuildOptions, Error, ErrorKind, IndexConfig, IndexName, LogicalIndexId, PartitionKey,
    Record, Result, Value, VerifyOptions,
};
use crate::maintenance::training::construction_groups;
use crate::search::numeric::VectorKernel;
use crate::search::rabitq::RaBitQ7;
use crate::storage::backend::Backend;
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{
    ChildEntry, IndexIdAllocator, IndexLifecycle, IndexManifest, IndexNameEntry, LeafEntry,
    OpaquePayload, PartitionCentroid, PartitionHeader, PartitionState, PartitionSynopsis,
    PartitionTransition, PersistentValue, RecordLocation, TreeManifest, VectorRecord,
};
use crate::storage::{ReadLogicalTxn, WriteLogicalTxn};

use super::{OperationContext, lifecycle, verify};

/// A construction partition; members are record positions for leaves and
/// indices into `parts` for internal partitions. Parents are created last.
struct Part {
    key: PartitionKey,
    level: u32,
    members: Vec<usize>,
    center: Box<[f32]>,
}

struct Tree {
    key: TreeKey,
    parts: Vec<Part>,
}

struct Plan {
    vectors: Vec<Box<[f32]>>,
    trees: Vec<Tree>,
    owners: Vec<(usize, usize)>,
}

/// Reject malformed or oversized source data before reserving an Index Name.
fn validate_input(
    config: &IndexConfig,
    records: &mut [Record],
    options: &BulkBuildOptions,
    checkpoint: &impl Fn() -> Result<()>,
) -> Result<BTreeMap<TreeKey, Vec<usize>>> {
    checkpoint()?;
    config.validate()?;
    records.sort_by(|left, right| left.id().cmp(right.id()));
    if records.windows(2).any(|pair| pair[0].id() == pair[1].id()) {
        return Err(Error::new(ErrorKind::RecordAlreadyExists));
    }
    let types: Vec<_> = config
        .tree_key_fields()
        .iter()
        .map(|field| config.fields()[usize::from(field.0)].data_type())
        .collect();
    let mut tree_members = BTreeMap::<TreeKey, Vec<usize>>::new();
    let mut bytes = 0_usize;
    for (position, record) in records.iter_mut().enumerate() {
        checkpoint()?;
        record.validate(config.dimension(), config.fields())?;
        let values: Vec<_> = config
            .tree_key_fields()
            .iter()
            .map(|field| record.fields()[usize::from(field.0)].clone())
            .collect();
        tree_members
            .entry(TreeKey::encode(&types, &values)?)
            .or_default()
            .push(position);
        if config.metric() == crate::api::Metric::Cosine
            && record.vector().iter().all(|value| *value == 0.0)
        {
            return Err(Error::invalid_argument());
        }
        let size = record
            .id()
            .len()
            .checked_add(record.vector().len() * 4)
            .and_then(|n| n.checked_add(record.payload().map_or(0, |payload| payload.len())))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        bytes = bytes
            .checked_add(size)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        for value in record.fields() {
            let size = match value {
                Value::Null => 1,
                Value::Bool(_) => 2,
                Value::I64(_) | Value::F64(_) => 9,
                Value::String(value) => value.len() + 5,
            };
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        }
        if bytes > options.input_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
    }
    Ok(tree_members)
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

impl Plan {
    fn new(
        checkpoint: &impl Fn() -> Result<()>,
        manifest: &IndexManifest,
        records: &[Record],
        tree_members: BTreeMap<TreeKey, Vec<usize>>,
        options: &BulkBuildOptions,
    ) -> Result<Self> {
        let config = manifest.config();
        let kernel = VectorKernel::new(
            config.dimension(),
            config.metric(),
            *manifest.rotation_seed(),
        )?;
        let mut vectors = Vec::with_capacity(records.len());
        for record in records {
            checkpoint()?;
            vectors.push(kernel.preprocess(record.vector())?);
        }
        let maximum = config.max_partition_entries() as usize;
        let minimum = config.min_partition_entries() as usize;
        // Validated configuration guarantees 2 * minimum <= maximum. Balanced
        // power-of-two subdivision produces non-root groups above half this
        // target, preserving minimum occupancy while leaving refinement space.
        let initial_leaf_target = (maximum / 2).max(2 * minimum);
        let mut trees = Vec::with_capacity(tree_members.len());
        let mut owners = vec![(0, 0); records.len()];
        for (key, members) in tree_members {
            checkpoint()?;
            let groups = construction_groups(
                &kernel,
                members
                    .iter()
                    .map(|&position| (position, vectors[position].clone()))
                    .collect(),
                initial_leaf_target,
                checkpoint,
            )?;
            let mut parts = Vec::with_capacity(groups.len());
            for (leaf, members) in groups.into_iter().enumerate() {
                let center = mean(
                    &kernel,
                    members.iter().map(|&position| vectors[position].as_ref()),
                )?;
                parts.push(Part {
                    key: PartitionKey::new(leaf as u64 + 2)?,
                    level: 1,
                    members,
                    center,
                });
            }
            refine(
                checkpoint, &kernel, &vectors, &mut parts, minimum, maximum, options,
            )?;
            let leaf_count = parts.len();
            for (leaf, part) in parts.iter().enumerate() {
                for &position in &part.members {
                    owners[position] = (trees.len(), leaf);
                }
            }
            let mut level: Vec<_> = (0..leaf_count).collect();
            let mut height = 1;
            while level.len() > 1 {
                checkpoint()?;
                height += 1;
                let groups = construction_groups(
                    &kernel,
                    level
                        .iter()
                        .map(|&part| (part, parts[part].center.clone()))
                        .collect(),
                    maximum,
                    checkpoint,
                )?;
                let mut next = Vec::with_capacity(groups.len());
                for members in groups {
                    let center = mean(
                        &kernel,
                        members.iter().map(|&part| parts[part].center.as_ref()),
                    )?;
                    next.push(parts.len());
                    parts.push(Part {
                        key: PartitionKey::new(parts.len() as u64 + 2)?,
                        level: height,
                        members,
                        center,
                    });
                }
                level = next;
            }
            // Every Tree Manifest fixes its root at partition 1; no identity
            // has been published, so changing this planned key is safe.
            parts.last_mut().expect("a nonempty tree has a root").key = PartitionKey::new(1)?;
            trees.push(Tree { key, parts });
        }
        Ok(Self {
            vectors,
            trees,
            owners,
        })
    }

    fn topology(
        &self,
        manifest: &IndexManifest,
        records: &[Record],
        checkpoint: &impl Fn() -> Result<()>,
    ) -> Result<Vec<(LogicalKey, PersistentValue)>> {
        let index = manifest.logical_index_id();
        let mut output = Vec::new();
        for tree in &self.trees {
            checkpoint()?;
            let high_water = tree
                .parts
                .iter()
                .map(|part| part.key)
                .max()
                .expect("nonempty tree");
            output.push((
                LogicalKey::TreeManifest {
                    index,
                    tree_key: tree.key.clone(),
                },
                PersistentValue::TreeManifest(TreeManifest::new(
                    PartitionKey::new(1)?,
                    high_water,
                )?),
            ));
            for part in &tree.parts {
                checkpoint()?;
                output.push((
                    LogicalKey::Header {
                        index,
                        tree_key: tree.key.clone(),
                        partition: part.key,
                    },
                    PersistentValue::PartitionHeader(PartitionHeader::new(
                        part.level,
                        u32::try_from(part.members.len())
                            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?,
                        0,
                        PartitionState::Ready,
                    )?),
                ));
                output.push((
                    LogicalKey::State {
                        index,
                        tree_key: tree.key.clone(),
                        partition: part.key,
                    },
                    PersistentValue::PartitionState(PartitionTransition::Ready {
                        started_at_unix_millis: lifecycle::now_unix_millis(),
                    }),
                ));
                output.push((
                    LogicalKey::Centroid {
                        index,
                        tree_key: tree.key.clone(),
                        partition: part.key,
                    },
                    PersistentValue::PartitionCentroid(PartitionCentroid::new(part.center.clone())),
                ));
                if part.level == 1 {
                    let mut synopsis = PartitionSynopsis::empty(manifest);
                    for &position in &part.members {
                        checkpoint()?;
                        synopsis.expand(manifest, records[position].fields())?;
                    }
                    output.push((
                        LogicalKey::Synopsis {
                            index,
                            tree_key: tree.key.clone(),
                            partition: part.key,
                        },
                        PersistentValue::PartitionSynopsis(synopsis),
                    ));
                } else {
                    for &child in &part.members {
                        checkpoint()?;
                        let child = &tree.parts[child];
                        output.push((
                            LogicalKey::ChildEntry {
                                index,
                                tree_key: tree.key.clone(),
                                partition: part.key,
                                child: child.key,
                            },
                            PersistentValue::ChildEntry(ChildEntry::new(
                                child.key,
                                child.center.clone(),
                            )),
                        ));
                    }
                }
            }
        }
        Ok(output)
    }

    fn record(
        &self,
        manifest: &IndexManifest,
        record: &Record,
        position: usize,
    ) -> Result<Vec<(LogicalKey, PersistentValue)>> {
        let index = manifest.logical_index_id();
        let id = record.id().clone();
        let (tree, leaf) = self.owners[position];
        let tree = &self.trees[tree];
        let partition = tree.parts[leaf].key;
        let mut output = vec![
            (
                LogicalKey::Record {
                    index,
                    id: id.clone(),
                },
                PersistentValue::VectorRecord(VectorRecord::new(
                    id.clone(),
                    Box::from(record.vector()),
                    Box::from(record.fields()),
                )),
            ),
            (
                LogicalKey::Location {
                    index,
                    id: id.clone(),
                },
                PersistentValue::RecordLocation(RecordLocation::new(tree.key.clone(), partition)),
            ),
            (
                LogicalKey::LeafEntry {
                    index,
                    tree_key: tree.key.clone(),
                    partition,
                    id: id.clone(),
                },
                PersistentValue::LeafEntry(LeafEntry::new(
                    id.clone(),
                    Box::from(record.fields()),
                    RaBitQ7::quantize(&self.vectors[position])?,
                )),
            ),
        ];
        if let Some(payload) = record.payload() {
            output.push((
                LogicalKey::Payload { index, id },
                PersistentValue::OpaquePayload(OpaquePayload::new(payload.clone())?),
            ));
        }
        Ok(output)
    }
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
    options: &BulkBuildOptions,
) -> Result<()> {
    if leaves.len() < 2 {
        return Ok(());
    }
    for _ in 0..options.rounds {
        checkpoint()?;
        let mut neighbors = Vec::with_capacity(leaves.len());
        for (source, leaf) in leaves.iter().enumerate() {
            checkpoint()?;
            let mut distances = Vec::with_capacity(leaves.len() - 1);
            for (target, candidate) in leaves.iter().enumerate() {
                if target != source {
                    distances.push((
                        kernel.routing_distance(&leaf.center, &candidate.center)?,
                        target,
                    ));
                }
            }
            distances.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
            distances.truncate(options.neighbors);
            neighbors.push(
                distances
                    .into_iter()
                    .map(|(_, target)| target)
                    .collect::<Vec<_>>(),
            );
        }
        let mut moves = Vec::new();
        for (source, leaf) in leaves.iter().enumerate() {
            checkpoint()?;
            for &position in &leaf.members {
                let vector = &vectors[position];
                let old = kernel.routing_distance(vector, &leaf.center)?;
                let mut best = (old, source);
                for &target in &neighbors[source] {
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
                counts[source] -= 1;
                counts[target] += 1;
                targets.insert(position, target);
            }
        }
        crate::observe::metrics::bulk_refinement_round(targets.len());
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

/// Reserves a durable construction name. An exact manifest and owner nonce fence
/// subsequent writes; an existing Building index is never borrowed by another builder.
async fn reserve<B: Backend>(
    context: &OperationContext<B>,
    name: &IndexName,
    config: &IndexConfig,
    retry: &lifecycle::RetryPolicy,
) -> Result<IndexManifest> {
    let backend = context.backend();
    let mut owner = [0; 16];
    getrandom::fill(&mut owner).map_err(|error| Error::with_source(ErrorKind::Other, error))?;
    let mut failures = 0;
    loop {
        context.checkpoint()?;
        let raw = backend.begin_write().await?;
        let mut txn =
            WriteLogicalTxn::bootstrap(raw, backend.hard_limits(), backend.admission_budget());
        if let Some(value) = txn
            .get_for_update(LogicalKey::IndexNameDirectory(name.clone()))
            .await?
        {
            let PersistentValue::IndexNameEntry(entry) = value else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            let value = txn
                .get(LogicalKey::Manifest(entry.logical_index_id()))
                .await?;
            let Some(PersistentValue::IndexManifest(manifest)) = value else {
                return Err(Error::new(ErrorKind::Corruption));
            };
            return Err(Error::new(match manifest.lifecycle() {
                IndexLifecycle::Building { .. } => ErrorKind::IndexBuilding,
                IndexLifecycle::Dropping => ErrorKind::IndexDropping,
                IndexLifecycle::Active => ErrorKind::IndexAlreadyExists,
            }));
        }
        let high_water = match txn.get_for_update(LogicalKey::IndexIdAllocator).await? {
            None => 0,
            Some(PersistentValue::IndexIdAllocator(allocator)) => allocator.high_water(),
            Some(_) => return Err(Error::new(ErrorKind::Corruption)),
        };
        let next = high_water
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::IdExhausted))?;
        let id = LogicalIndexId::new(next)?;
        let manifest = IndexManifest::new(
            IndexLifecycle::Building { owner },
            id,
            config.clone(),
            lifecycle::derive_rotation_seed(id),
            lifecycle::derive_bloom_parameters(config)?,
        )?;
        txn.put(
            LogicalKey::IndexIdAllocator,
            PersistentValue::IndexIdAllocator(IndexIdAllocator::new(next)),
        )
        .await?;
        txn.put(
            LogicalKey::IndexNameDirectory(name.clone()),
            PersistentValue::IndexNameEntry(IndexNameEntry::new(id)),
        )
        .await?;
        txn.put(
            LogicalKey::Manifest(id),
            PersistentValue::IndexManifest(manifest.clone()),
        )
        .await?;
        match txn.commit().await {
            Ok(()) => return Ok(manifest),
            Err(error) if error.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(crate::observe::labels::Operation::BuildIndex, &mut failures)
                    .await?
            }
            Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => {
                let raw = backend.begin_read().await?;
                let mut txn = ReadLogicalTxn::bootstrap(raw);
                let name_value = txn
                    .get(LogicalKey::IndexNameDirectory(name.clone()))
                    .await?;
                let manifest_value = txn.get(LogicalKey::Manifest(id)).await?;
                if matches!(name_value, Some(PersistentValue::IndexNameEntry(entry)) if entry.logical_index_id() == id)
                    && matches!(manifest_value, Some(PersistentValue::IndexManifest(ref current)) if current == &manifest)
                {
                    return Ok(manifest);
                }
                return Err(error);
            }
            Err(error) => return Err(error),
        }
    }
}

/// Every staging transaction conflicts with drop/publication of its manifest.
async fn fenced_txn<'a, B: Backend>(
    backend: &'a B,
    manifest: &'a IndexManifest,
) -> Result<WriteLogicalTxn<'a, B::WriteTxn<'a>>> {
    let raw = backend.begin_write().await?;
    let mut txn = WriteLogicalTxn::for_index(
        raw,
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    let current = txn
        .get_for_update(LogicalKey::Manifest(manifest.logical_index_id()))
        .await?;
    match current {
        Some(PersistentValue::IndexManifest(ref current)) if current == manifest => Ok(txn),
        Some(PersistentValue::IndexManifest(current))
            if current.lifecycle() == IndexLifecycle::Dropping =>
        {
            Err(Error::new(ErrorKind::IndexDropping))
        }
        None => Err(Error::new(ErrorKind::IndexNotFound)),
        _ => Err(Error::new(ErrorKind::Corruption)),
    }
}

/// Idempotent deterministic groups may be replayed after unknown commits.
/// A record group is indivisible so even Building records have exact membership.
/// Typed transaction accounting enforces adapter admission and hard limits;
/// an oversized batch is rolled back and reduced, never partially committed.
async fn stage<B: Backend>(
    context: &OperationContext<B>,
    manifest: &IndexManifest,
    groups: &[Vec<(LogicalKey, PersistentValue)>],
    retry: &lifecycle::RetryPolicy,
) -> Result<()> {
    let backend = context.backend();
    let mut offset = 0;
    let mut batch = 256_usize.min(groups.len().max(1));
    let mut failures = 0;
    while offset < groups.len() {
        context.checkpoint()?;
        let end = (offset + batch).min(groups.len());
        let mut txn = fenced_txn(backend.as_ref(), manifest).await?;
        let mut put_error = None;
        'groups: for group in &groups[offset..end] {
            for (key, value) in group {
                if let Err(error) = txn.put(key.clone(), value.clone()).await {
                    put_error = Some(error);
                    break 'groups;
                }
            }
        }
        if let Some(error) = put_error {
            txn.rollback().await;
            if batch > 1
                && matches!(
                    error.kind(),
                    ErrorKind::LimitExceeded | ErrorKind::TransactionTooLarge
                )
            {
                batch = batch.div_ceil(2);
                continue;
            }
            return Err(error);
        }
        match txn.commit().await {
            Ok(()) => {
                offset = end;
                failures = 0;
            }
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::RetryableAbort | ErrorKind::CommitOutcomeUnknown
                ) =>
            {
                retry
                    .wait_or_exhaust(crate::observe::labels::Operation::BuildIndex, &mut failures)
                    .await?
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Cancels a planning task when its owning foreground future is dropped.
struct CancelPlanning(CancellationToken);
impl Drop for CancelPlanning {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Shared cooperative control for validation and numerical planning.
struct PlanningControl {
    cancellation: CancellationToken,
    options: crate::api::OperationOptions,
}
impl PlanningControl {
    fn checkpoint(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            return Err(Error::new(ErrorKind::Cancelled));
        }
        super::check_control(&self.options)
    }
}

/// Keeps heavy CPU work off Tokio and cancels it when its owning future drops.
/// Existing foreground admission bounds the number of these owned tasks.
async fn controlled_compute<T: Send + 'static>(
    options: crate::api::OperationOptions,
    keep_alive: impl Send + 'static,
    work: impl FnOnce(&PlanningControl) -> Result<T> + Send + 'static,
) -> Result<T> {
    let cancellation = CancellationToken::new();
    let _guard = CancelPlanning(cancellation.clone());
    tokio::task::spawn_blocking(move || {
        // Retain admission even after the owning async future is cancelled.
        let _keep_alive = keep_alive;
        let control = PlanningControl {
            cancellation,
            options,
        };
        control.checkpoint()?;
        work(&control)
    })
    .await
    .map_err(|error| Error::with_source(ErrorKind::Other, error))?
}

/// Builds one index; publication is the only caller-visible irreversible commit.
pub(crate) async fn build<B: Backend>(
    context: &mut OperationContext<B>,
    name: IndexName,
    config: IndexConfig,
    records: Vec<Record>,
    options: BulkBuildOptions,
    retry: lifecycle::RetryPolicy,
) -> Result<IndexManifest> {
    let validation_config = config.clone();
    let validation_options = options.clone();
    let (records, tree_members) = controlled_compute(
        context.options.clone(),
        context.cpu_admission.clone().expect("bulk admission"),
        move |control| {
            let mut records = records;
            let tree_members = validate_input(
                &validation_config,
                &mut records,
                &validation_options,
                &|| control.checkpoint(),
            )?;
            Ok((records, tree_members))
        },
    )
    .await?;
    let manifest = reserve(context, &name, &config, &retry).await?;
    let planning_manifest = manifest.clone();
    let (plan, records, topology) = controlled_compute(
        context.options.clone(),
        context.cpu_admission.clone().expect("bulk admission"),
        move |control| {
            let plan = Plan::new(
                &|| control.checkpoint(),
                &planning_manifest,
                &records,
                tree_members,
                &options,
            )?;
            let topology = plan
                .topology(&planning_manifest, &records, &|| control.checkpoint())?
                .into_iter()
                .map(|row| {
                    control.checkpoint()?;
                    Ok(vec![row])
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((plan, records, topology))
        },
    )
    .await?;
    stage(context, &manifest, &topology, &retry).await?;
    drop(topology);
    // Record projections stay bounded; the complete dataset is never copied
    // into a second collection of encoded persistent values.
    for first in (0..records.len()).step_by(64) {
        let end = (first + 64).min(records.len());
        let mut values = Vec::with_capacity(end - first);
        for (position, record) in records.iter().enumerate().take(end).skip(first) {
            values.push(plan.record(&manifest, record, position)?);
        }
        stage(context, &manifest, &values, &retry).await?;
    }
    drop(plan);
    drop(records);
    let report = verify::verify_build(
        context,
        &manifest,
        VerifyOptions::default()
            .with_object_limit(100_000_000)?
            .with_memory_limit_bytes(1 << 30)?,
    )
    .await?;
    if !report.complete {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    if !report.issues.is_empty() {
        return Err(Error::new(ErrorKind::Corruption));
    }
    let backend = context.backend();
    let active = manifest.with_lifecycle(IndexLifecycle::Active);
    let mut failures = 0;
    loop {
        context.checkpoint()?;
        let mut txn = fenced_txn(backend.as_ref(), &manifest).await?;
        txn.put(
            LogicalKey::Manifest(manifest.logical_index_id()),
            PersistentValue::IndexManifest(active.clone()),
        )
        .await?;
        match context.commit(move |start| txn.commit_with(start)).await {
            Ok(()) => return Ok(active),
            Err(error) if error.kind() == ErrorKind::RetryableAbort => {
                retry
                    .wait_or_exhaust(crate::observe::labels::Operation::BuildIndex, &mut failures)
                    .await?
            }
            Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => {
                let raw = backend.begin_read().await?;
                let mut txn = ReadLogicalTxn::for_index(raw, &active);
                if matches!(txn.get(LogicalKey::Manifest(active.logical_index_id())).await?,
                    Some(PersistentValue::IndexManifest(ref current)) if current == &active)
                {
                    return Ok(active);
                }
                return Err(error);
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };

    #[test]
    fn initial_leaf_slack_preserves_partition_occupancy_boundaries() {
        for (minimum, maximum, count, expected_leaves) in [
            (2, 4, 4, 1),
            (2, 4, 5, 2),
            (2, 4, 9, 4),
            (16, 512, 512, 2),
            (16, 512, 513, 4),
            (127, 255, 255, 2),
        ] {
            let config = IndexConfig::new(1, crate::api::Metric::L2)
                .unwrap()
                .with_partition_entries(minimum, maximum)
                .unwrap();
            let manifest = IndexManifest::new(
                IndexLifecycle::Building { owner: [1; 16] },
                LogicalIndexId::new(1).unwrap(),
                config,
                [0; 32],
                vec![],
            )
            .unwrap();
            let records: Vec<_> = (0_usize..count)
                .map(|position| {
                    Record::new(
                        Bytes::from(position.to_be_bytes().to_vec()),
                        vec![position as f32],
                        vec![],
                    )
                    .unwrap()
                })
                .collect();
            let plan = Plan::new(
                &|| Ok(()),
                &manifest,
                &records,
                BTreeMap::from([(TreeKey::encode(&[], &[]).unwrap(), (0..count).collect())]),
                &BulkBuildOptions::new(1 << 20)
                    .unwrap()
                    .with_refinement_rounds(0)
                    .unwrap(),
            )
            .unwrap();
            let parts = &plan.trees[0].parts;
            assert_eq!(
                parts.iter().filter(|part| part.level == 1).count(),
                expected_leaves
            );
            for part in parts {
                assert!(part.members.len() <= maximum as usize);
                if part.key != PartitionKey::new(1).unwrap() {
                    assert!(part.members.len() >= minimum as usize);
                }
            }
        }
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
                key: PartitionKey::new(2).unwrap(),
                level: 1,
                members: vec![0, 2],
                center: vec![4.5].into(),
            },
            Part {
                key: PartitionKey::new(3).unwrap(),
                level: 1,
                members: vec![1, 3],
                center: vec![5.5].into(),
            },
        ];
        let options = BulkBuildOptions::new(100).unwrap();
        refine(&|| Ok(()), &kernel, &vectors, &mut leaves, 2, 2, &options).unwrap();
        assert_eq!(leaves[0].members, [0, 2]); // A full target cannot accept a move.
        refine(&|| Ok(()), &kernel, &vectors, &mut leaves, 1, 3, &options).unwrap();
        assert_eq!(leaves[0].members, [0, 1]);
        assert_eq!(leaves[1].members, [2, 3]);
        assert_eq!(leaves[0].center.as_ref(), [0.5]);
        assert_eq!(leaves[1].center.as_ref(), [9.5]);
    }

    /// Exercise real planning and serialization, not just caller error returns.
    /// Aborting the owning async task drops the guard; after the blocked
    /// checkpoint resumes, the blocking task must stop at that checkpoint.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_planning_owner_stops_blocking_work() {
        assert_dropped_owner_stops_work(false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_topology_owner_stops_blocking_work() {
        assert_dropped_owner_stops_work(true).await;
    }

    async fn assert_dropped_owner_stops_work(topology: bool) {
        let config = IndexConfig::new(8, crate::api::Metric::L2)
            .unwrap()
            .with_partition_entries(2, 4)
            .unwrap();
        let manifest = IndexManifest::new(
            IndexLifecycle::Building { owner: [1; 16] },
            LogicalIndexId::new(1).unwrap(),
            config,
            [0; 32],
            vec![],
        )
        .unwrap();
        let records: Vec<_> = (0_usize..1000)
            .map(|position| {
                Record::new(
                    Bytes::from(position.to_be_bytes().to_vec()),
                    vec![position as f32; 8],
                    vec![],
                )
                .unwrap()
            })
            .collect();
        let calls = Arc::new(AtomicUsize::new(0));
        let (started_send, started) = tokio::sync::oneshot::channel();
        let (resume, resume_receive) = mpsc::channel();
        let (finished_send, finished) = tokio::sync::oneshot::channel();
        let worker_calls = calls.clone();
        let released = Arc::new(AtomicUsize::new(0));
        struct HeldAdmission(Arc<AtomicUsize>);
        impl Drop for HeldAdmission {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let admission = HeldAdmission(released.clone());
        let owner = tokio::spawn(async move {
            controlled_compute(
                crate::api::OperationOptions::default(),
                admission,
                move |control| {
                    let started_send = std::sync::Mutex::new(Some(started_send));
                    let checkpoint = || {
                        let call = worker_calls.fetch_add(1, Ordering::SeqCst);
                        if call == 50 {
                            started_send
                                .lock()
                                .unwrap()
                                .take()
                                .unwrap()
                                .send(())
                                .unwrap();
                            resume_receive
                                .recv_timeout(std::time::Duration::from_secs(10))
                                .unwrap();
                        }
                        control.checkpoint()
                    };
                    let members = BTreeMap::from([(
                        TreeKey::encode(&[], &[]).unwrap(),
                        (0..records.len()).collect(),
                    )]);
                    let options = BulkBuildOptions::new(1 << 20).unwrap();
                    let result = if topology {
                        let plan =
                            Plan::new(&|| Ok(()), &manifest, &records, members, &options).unwrap();
                        plan.topology(&manifest, &records, &checkpoint).map(|_| ())
                    } else {
                        Plan::new(&checkpoint, &manifest, &records, members, &options).map(|_| ())
                    };
                    finished_send
                        .send(result.err().map(|error| error.kind()))
                        .unwrap();
                    Ok(())
                },
            )
            .await
            .unwrap();
        });
        started.await.unwrap();
        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        assert_eq!(
            released.load(Ordering::SeqCst),
            0,
            "CPU task still owns admission"
        );
        resume.send(()).unwrap();
        assert_eq!(finished.await.unwrap(), Some(ErrorKind::Cancelled));
        assert_eq!(calls.load(Ordering::SeqCst), 51);
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while released.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }
}
