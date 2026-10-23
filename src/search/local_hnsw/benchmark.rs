//! Benchmark-only global leaf-centroid routing over a verified, immutable index.
//! This is not a production cache or a persisted global graph.

use crate::api::{Error, ErrorKind, LogicalIndexId, PartitionKey, Result};
use crate::search::local_hnsw::{self, Centroids, Workspace};
use crate::search::numeric::VectorKernel;
use crate::storage::backend::{Backend, ReadOps};
use crate::storage::keys::{self, LogicalKey, TreeKey};
use crate::storage::values::{ChildEntry, PersistentValue, ValueCodec};

/// One globally resident comparison index. It is limited to a single Tree Key
/// and USearch's supported node count; exceeding either bound is an error.
pub struct GlobalLeafRouter {
    index: LogicalIndexId,
    admission: std::sync::Arc<tokio::sync::Semaphore>,
    kernel: VectorKernel,
    children: Vec<ChildEntry>,
    graph: Option<ktann_usearch::Graph>,
    m: usize,
    limit: usize,
    resident_bytes: usize,
    centroid_bytes: usize,
    overhead_bytes: usize,
}

impl GlobalLeafRouter {
    /// Loads final leaf centroids in one snapshot and optionally builds a graph.
    /// `limit` covers centroids, kernel, native build/search working storage and
    /// candidate buffers, independently of the runtime's partition cache.
    pub(crate) async fn load<B: Backend>(
        backend: &B,
        index: LogicalIndexId,
        limit: usize,
        hnsw: bool,
        concurrency: usize,
    ) -> Result<Self> {
        let corrupt = || Error::new(ErrorKind::Corruption);
        let mut txn = backend.begin_read().await?;
        let key = LogicalKey::Manifest(index);
        let raw = txn
            .get(keys::manifest_key(index).into())
            .await?
            .ok_or_else(corrupt)?;
        let PersistentValue::IndexManifest(manifest) = ValueCodec::bootstrap().decode(&key, raw)?
        else {
            return Err(corrupt());
        };
        let config = manifest.config();
        let parameters = config
            .local_graph()
            .ok_or_else(Error::invalid_argument)?
            .parameters();
        if !config.tree_key_fields().is_empty()
            || limit == 0
            || concurrency == 0
            || concurrency > tokio::sync::Semaphore::MAX_PERMITS
        {
            return Err(Error::invalid_argument());
        }
        let kernel = VectorKernel::new(
            config.dimension(),
            config.metric(),
            *manifest.rotation_seed(),
        )?;
        let codec = ValueCodec::for_index(&manifest);
        let tree_key = TreeKey::encode(&[], &[])?;
        let raw = txn
            .get(keys::tree_manifest_key(index, &tree_key).into())
            .await?
            .ok_or_else(corrupt)?;
        let PersistentValue::TreeManifest(tree) = codec.decode(
            &LogicalKey::TreeManifest {
                index,
                tree_key: tree_key.clone(),
            },
            raw,
        )?
        else {
            return Err(corrupt());
        };
        // Header enumeration is setup, never part of query timing. Fail explicitly
        // instead of scanning an arbitrary allocator history or shrinking data.
        if tree.partition_key_high_water().get() > 262_144 {
            return Err(limit_error());
        }
        let per_leaf = config
            .dimension()
            .checked_mul(4)
            .and_then(|n| n.checked_add(std::mem::size_of::<ChildEntry>() + 64))
            .ok_or_else(limit_error)?;
        let mut resident_bytes = kernel.heap_bytes();
        let capacity =
            (tree.partition_key_high_water().get() as usize).min(ktann_usearch::MAX_NODES);
        resident_bytes = resident_bytes
            .checked_add(capacity * std::mem::size_of::<ChildEntry>())
            .ok_or_else(limit_error)?;
        if resident_bytes > limit {
            return Err(limit_error());
        }
        let mut children = Vec::new();
        children
            .try_reserve_exact(capacity)
            .map_err(|_| limit_error())?;
        for id in 1..=tree.partition_key_high_water().get() {
            let partition = PartitionKey::new(id)?;
            let Some(raw) = txn
                .get(keys::header_key(index, &tree_key, partition).into())
                .await?
            else {
                continue;
            };
            if resident_bytes
                .checked_add(raw.len() * 2 + per_leaf)
                .is_none_or(|n| n > limit)
            {
                return Err(limit_error());
            }
            let PersistentValue::PartitionHeader(header) = codec.decode(
                &LogicalKey::Header {
                    index,
                    tree_key: tree_key.clone(),
                    partition,
                },
                raw,
            )?
            else {
                return Err(corrupt());
            };
            if header.level() == 1 {
                if children.len() == ktann_usearch::MAX_NODES {
                    return Err(limit_error());
                }
                resident_bytes = resident_bytes
                    .checked_add(per_leaf)
                    .ok_or_else(limit_error)?;
                let raw = txn
                    .get(keys::centroid_key(index, &tree_key, partition).into())
                    .await?
                    .ok_or_else(corrupt)?;
                let PersistentValue::PartitionCentroid(centroid) = codec.decode(
                    &LogicalKey::Centroid {
                        index,
                        tree_key: tree_key.clone(),
                        partition,
                    },
                    raw,
                )?
                else {
                    return Err(corrupt());
                };
                children.push(ChildEntry::new(partition, centroid.components().to_vec()));
            }
        }
        drop(txn);
        if children.is_empty() {
            return Err(Error::invalid_argument());
        }
        let centroids = Centroids::new(&kernel, &children, ktann_usearch::MAX_NODES, &|| Ok(()))?;
        let centroid_bytes = centroids.bytes;
        let overhead_bytes = resident_bytes.saturating_sub(centroid_bytes);
        let graph = if hnsw {
            let budget = limit.checked_sub(overhead_bytes).ok_or_else(limit_error)?;
            Some(
                local_hnsw::build_graph(
                    centroids,
                    parameters,
                    &tree_key,
                    PartitionKey::new(1)?,
                    budget,
                    &|| Ok(()),
                )?
                .0,
            )
        } else {
            None
        };
        resident_bytes = resident_bytes
            .checked_add(
                graph
                    .as_ref()
                    .map_or(0, ktann_usearch::Graph::resident_bytes),
            )
            .ok_or_else(limit_error)?;
        Ok(Self {
            index,
            admission: std::sync::Arc::new(tokio::sync::Semaphore::new(concurrency)),
            kernel,
            children,
            graph,
            m: parameters.m,
            limit,
            resident_bytes,
            centroid_bytes,
            overhead_bytes,
        })
    }

    /// Admits one complete global-routing plus production-search operation.
    /// The permit follows blocking work so cancellation cannot release it early.
    pub(crate) async fn permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit> {
        self.admission
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::new(ErrorKind::Cancelled))
    }

    /// Number of final leaf centroids in this global comparison.
    pub fn leaves(&self) -> usize {
        self.children.len()
    }

    /// Conservative retained centroid/kernel/native graph bytes; not process RSS.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Routes a query without changing native graph scores. Search scratch is
    /// private per operation; concurrent callers share only immutable state.
    pub(crate) fn route(
        &self,
        index: LogicalIndexId,
        query: &[f32],
        beam: usize,
        ef: usize,
    ) -> Result<Vec<(PartitionKey, f64)>> {
        self.route_mode(index, query, beam, ef, true)
    }

    /// Untimed exact centroid reference over the identical resident layout.
    pub fn exact(
        &self,
        index: LogicalIndexId,
        query: &[f32],
        beam: usize,
    ) -> Result<Vec<(PartitionKey, f64)>> {
        self.route_mode(index, query, beam, 1, false)
    }

    fn route_mode(
        &self,
        index: LogicalIndexId,
        query: &[f32],
        beam: usize,
        ef: usize,
        use_graph: bool,
    ) -> Result<Vec<(PartitionKey, f64)>> {
        if index != self.index {
            return Err(Error::invalid_argument());
        }
        let started = std::time::Instant::now();
        let query = self.kernel.preprocess(query)?;
        let centroids = Centroids {
            kernel: &self.kernel,
            children: &self.children,
            bytes: self.centroid_bytes,
        };
        let map_bytes = beam
            .min(self.children.len())
            .checked_mul(std::mem::size_of::<(PartitionKey, f64)>())
            .ok_or_else(limit_error)?;
        let budget = self
            .limit
            .checked_sub(self.overhead_bytes)
            .and_then(|n| n.checked_sub(query.len() * 8))
            .and_then(|n| n.checked_sub(map_bytes))
            .ok_or_else(limit_error)?;
        let hits = if let Some(graph) = &self.graph
            && use_graph
            && beam < self.children.len()
        {
            let available = self
                .limit
                .checked_sub(self.resident_bytes)
                .ok_or_else(limit_error)?;
            let mut scratch = Workspace::new(self.children.len(), self.m, available)
                .map_err(local_hnsw::native_error)?;
            ::metrics::histogram!("ktann.routing.effective_ef")
                .record(ef.max(beam).min(self.children.len()) as f64);
            let (hits, stats) = local_hnsw::search_graph(
                centroids,
                graph,
                &query,
                beam,
                ef,
                budget,
                &mut scratch,
                &|| Ok(()),
            )?;
            ::metrics::counter!("ktann.routing.distances").increment(stats.distances as u64);
            ::metrics::counter!("ktann.routing.graph_visits").increment(stats.visits as u64);
            hits
        } else {
            let available = budget
                .checked_sub(
                    self.graph
                        .as_ref()
                        .map_or(0, ktann_usearch::Graph::resident_bytes),
                )
                .ok_or_else(limit_error)?;
            let hits = centroids.exact(&query, beam, available, &|| Ok(()))?;
            ::metrics::counter!("ktann.routing.distances").increment(self.children.len() as u64);
            hits
        };
        ::metrics::histogram!("ktann.routing.compute_seconds")
            .record(started.elapsed().as_secs_f64());
        Ok(hits
            .into_iter()
            .map(|hit| (self.children[hit.node as usize].child(), hit.distance))
            .collect())
    }
}

fn limit_error() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

impl std::fmt::Debug for GlobalLeafRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalLeafRouter")
            .field("leaves", &self.leaves())
            .field("resident_bytes", &self.resident_bytes)
            .finish_non_exhaustive()
    }
}
