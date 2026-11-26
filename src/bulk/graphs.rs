//! Direct partition-local HNSW graph construction from final in-memory Child Entries.

use std::time::Instant;

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::local_hnsw::{
    Centroids,
    persistence::{GraphSource, build_records},
};
use crate::search::numeric::VectorKernel;
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{ChildEntry, IndexManifest, PartitionHeader, PersistentValue};

/// Builds and encodes one internal partition's local HNSW graph when configured.
///
/// Operates directly on the final sorted [`ChildEntry`] slice in memory without
/// serializing and re-decoding intermediate partition KV bytes.
#[expect(
    clippy::too_many_arguments,
    reason = "internal graph construction takes manifest, kernel, partition identity, children, budget, and cancellation check"
)]
pub(crate) fn build_internal_graph(
    index: &IndexManifest,
    kernel: &VectorKernel,
    tree_key: &TreeKey,
    partition: PartitionKey,
    header: PartitionHeader,
    children: &[ChildEntry],
    budget: usize,
    check: &(dyn Fn() -> Result<()> + Sync),
) -> Result<Vec<(LogicalKey, PersistentValue)>> {
    let Some(local_graph) = index.config().local_graph() else {
        return Ok(Vec::new());
    };
    if header.level() <= 1 || children.is_empty() {
        return Ok(Vec::new());
    }
    if children.len() != header.entry_count() as usize {
        return Err(Error::new(ErrorKind::Corruption));
    }
    let centroid_bytes = children
        .len()
        .checked_mul(
            index
                .config()
                .dimension()
                .checked_mul(4)
                .and_then(|n| n.checked_add(std::mem::size_of::<ChildEntry>()))
                .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?,
        )
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    if centroid_bytes > budget {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let centroids = Centroids::new(
        kernel,
        children,
        index.config().max_internal_partition_entries() as usize,
        check,
    )?;
    let started = Instant::now();
    let records = build_records(
        centroids,
        GraphSource {
            index: index.logical_index_id(),
            tree: tree_key,
            parent: partition,
            header,
            parameters: local_graph.parameters(),
        },
        budget,
        check,
    )?
    .records;
    ::metrics::histogram!("ktann.routing.build_seconds").record(started.elapsed().as_secs_f64());
    ::metrics::counter!("ktann.routing.stored_graph_bytes").increment(records.descriptor.bytes);

    let index_id = index.logical_index_id();
    let mut out = Vec::with_capacity(1 + records.descriptor.chunks() as usize);
    out.push((
        LogicalKey::GraphDescriptor {
            index: index_id,
            tree_key: tree_key.clone(),
            partition,
        },
        PersistentValue::GraphDescriptor(records.descriptor.clone()),
    ));
    for (chunk, value) in records.chunks() {
        check()?;
        out.push((
            LogicalKey::GraphChunk {
                index: index_id,
                tree_key: tree_key.clone(),
                partition,
                chunk,
            },
            PersistentValue::GraphChunk(value),
        ));
    }
    Ok(out)
}
