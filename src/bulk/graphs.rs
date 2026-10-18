//! One-at-a-time graph preparation from the exact canonical serving projections.

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::local_hnsw::{
    Centroids,
    persistence::{GraphSource, build_records},
};
use crate::search::numeric::VectorKernel;
use crate::storage::keys::{LogicalKey, TreeKey};
use crate::storage::values::{ChildEntry, IndexManifest, PartitionHeader, PersistentValue};

/// Retains only one internal body; leaf bodies never allocate graph state.
pub(super) struct GraphPreparation<'a> {
    index: &'a IndexManifest,
    kernel: VectorKernel,
    budget: usize,
    body: Option<(TreeKey, PartitionKey, PartitionHeader, Vec<ChildEntry>)>,
}

impl<'a> GraphPreparation<'a> {
    pub(super) fn new(index: &'a IndexManifest, budget: usize) -> Result<Self> {
        Ok(Self {
            index,
            kernel: VectorKernel::new(
                index.config().dimension(),
                index.config().metric(),
                *index.rotation_seed(),
            )?,
            budget,
            body: None,
        })
    }

    /// Observe canonical bytes after finishing the preceding body at its boundary.
    pub(super) fn observe(&mut self, key: &LogicalKey, value: PersistentValue) -> Result<()> {
        match (key, value) {
            (
                LogicalKey::Header {
                    tree_key,
                    partition,
                    ..
                },
                PersistentValue::PartitionHeader(header),
            ) if header.level() > 1 => {
                // Admit the centroid allocation before collecting any child. The core
                // subsequently admits the combined native construction/encoding peak.
                let bytes = (header.entry_count() as usize)
                    .checked_mul(
                        self.index
                            .config()
                            .dimension()
                            .checked_mul(4)
                            .and_then(|n| n.checked_add(std::mem::size_of::<ChildEntry>()))
                            .ok_or_else(limit)?,
                    )
                    .ok_or_else(limit)?;
                if bytes > self.budget {
                    return Err(limit());
                }
                self.body = Some((
                    tree_key.clone(),
                    *partition,
                    header,
                    Vec::with_capacity(header.entry_count() as usize),
                ));
            }
            (
                LogicalKey::ChildEntry {
                    tree_key,
                    partition,
                    ..
                },
                PersistentValue::ChildEntry(child),
            ) => {
                let (tree, parent, header, children) = self.body.as_mut().ok_or_else(corrupt)?;
                if tree != tree_key
                    || parent != partition
                    || children.len() >= header.entry_count() as usize
                {
                    return Err(corrupt());
                }
                children.push(child);
            }
            _ => {}
        }
        Ok(())
    }

    /// Build and serialize once, after every child projection is final.
    pub(super) fn finish(
        &mut self,
        check: &dyn Fn() -> Result<()>,
        mut emit: impl FnMut(LogicalKey, PersistentValue) -> Result<()>,
    ) -> Result<()> {
        let Some((tree_key, partition, header, children)) = self.body.take() else {
            return Ok(());
        };
        if children.len() != header.entry_count() as usize {
            return Err(corrupt());
        }
        let parameters = self
            .index
            .config()
            .fixed_graph()
            .expect("fixed preparation")
            .parameters();
        let centroids = Centroids::new(
            &self.kernel,
            &children,
            self.index.config().max_internal_partition_entries() as usize,
            check,
        )?;
        let records = build_records(
            centroids,
            GraphSource {
                index: self.index.logical_index_id(),
                tree: &tree_key,
                parent: partition,
                header,
                parameters,
            },
            self.budget,
            check,
        )?;
        let index = self.index.logical_index_id();
        emit(
            LogicalKey::GraphDescriptor {
                index,
                tree_key: tree_key.clone(),
                partition,
            },
            PersistentValue::GraphDescriptor(records.descriptor.clone()),
        )?;
        for (chunk, value) in records.chunks() {
            check()?;
            emit(
                LogicalKey::GraphChunk {
                    index,
                    tree_key: tree_key.clone(),
                    partition,
                    chunk,
                },
                PersistentValue::GraphChunk(value),
            )?;
        }
        Ok(())
    }
}

fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{FixedGraphConfig, IndexConfig, LogicalIndexId, Metric, Record};
    use crate::bulk::{
        ForestArtifact, ForestOptions, InputSnapshot, ServingArtifact, ServingOptions,
    };
    use crate::search::local_hnsw::{Workspace, persistence::GraphSource};
    use crate::storage::backend::HardLimits;
    use crate::storage::keys;
    use crate::storage::values::{GraphChunk, GraphDescriptor, IndexLifecycle, ValueCodec};
    use bytes::Bytes;
    use std::collections::BTreeMap;

    #[test]
    fn fixed_artifact_bytes_are_reproducible_in_independent_processes() {
        let run = || {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["bulk::graphs::tests::final_artifact_graphs_restore_against_exact_projections_for_all_metrics", "--exact", "--nocapture"])
                .output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .filter(|line| line.starts_with("GRAPH-ARTIFACT:"))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        let first = run();
        assert_eq!(first.len(), 9);
        assert_eq!(first, run());
    }

    #[test]
    fn final_artifact_graphs_restore_against_exact_projections_for_all_metrics() {
        let root = std::env::temp_dir().join(format!("ktann-final-graphs-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let budget = 16 << 20;
        for (metric_id, metric) in [Metric::L2, Metric::Cosine, Metric::InnerProduct]
            .into_iter()
            .enumerate()
        {
            for (round_id, rounds) in [None, Some(0), Some(2)].into_iter().enumerate() {
                let directory = root.join(format!("{metric_id}-{round_id}"));
                std::fs::create_dir(&directory).unwrap();
                let config = IndexConfig::new(2, metric)
                    .unwrap()
                    .with_leaf_partition_entries(2, 4)
                    .unwrap()
                    .with_internal_partition_entries(2, 4)
                    .unwrap()
                    .with_fixed_graph(FixedGraphConfig::new(2, 8, [4; 32]).unwrap());
                let index = IndexManifest::new(
                    IndexLifecycle::Building,
                    LogicalIndexId::new(1).unwrap(),
                    config.clone(),
                    [3; 32],
                    vec![],
                )
                .unwrap();
                let input = InputSnapshot::create(
                    &directory.join("input"),
                    config,
                    1 << 20,
                    (0u64..73).map(|i| {
                        Record::new(
                            Bytes::copy_from_slice(&i.to_be_bytes()),
                            vec![i as f32 + 1.0, (i % 7) as f32 + 1.0],
                            vec![],
                        )
                    }),
                )
                .unwrap();
                let options = ForestOptions {
                    sample_items: 8,
                    refinement_rounds: rounds,
                    refinement_memory_bytes: budget,
                    tree_memory_bytes: budget,
                    tree_scratch_bytes: 1 << 26,
                    sort_memory_bytes: budget,
                    sort_scratch_bytes: 1 << 26,
                };
                let (forest, _) = ForestArtifact::build(
                    &directory.join("forest"),
                    &input,
                    [3; 32],
                    options,
                    1 << 24,
                )
                .unwrap();
                let serving_options = ServingOptions {
                    memory_bytes: budget,
                    graph_memory_bytes: budget,
                    scratch_bytes: 1 << 26,
                    hard_limits: HardLimits {
                        max_key_bytes: 10_000,
                        max_value_bytes: 100_000,
                    },
                };
                let (artifact, _) = ServingArtifact::build(
                    &directory.join("serving"),
                    &input,
                    &forest,
                    &index,
                    serving_options,
                    1 << 24,
                )
                .unwrap();
                let (again, _) = ServingArtifact::build(
                    &directory.join("again"),
                    &input,
                    &forest,
                    &index,
                    serving_options,
                    1 << 24,
                )
                .unwrap();
                assert_eq!(artifact.manifest(), again.manifest());
                println!("GRAPH-ARTIFACT: {:?}", artifact.manifest().encode());
                #[derive(Default)]
                struct Body {
                    header: Option<PartitionHeader>,
                    children: Vec<ChildEntry>,
                    descriptor: Option<GraphDescriptor>,
                    chunks: Vec<(u32, GraphChunk)>,
                }
                let mut bodies = BTreeMap::<u64, Body>::new();
                let codec = ValueCodec::for_index(&index);
                for row in artifact.reader().unwrap() {
                    let row = row.unwrap();
                    let key = keys::decode_key(&[], &row.key).unwrap();
                    let value = codec.decode(&key, row.value).unwrap();
                    match (key, value) {
                        (
                            LogicalKey::Header { partition, .. },
                            PersistentValue::PartitionHeader(header),
                        ) => {
                            bodies.entry(partition.get()).or_default().header = Some(header);
                        }
                        (
                            LogicalKey::ChildEntry { partition, .. },
                            PersistentValue::ChildEntry(child),
                        ) => {
                            bodies
                                .entry(partition.get())
                                .or_default()
                                .children
                                .push(child);
                        }
                        (
                            LogicalKey::GraphDescriptor { partition, .. },
                            PersistentValue::GraphDescriptor(descriptor),
                        ) => {
                            bodies.entry(partition.get()).or_default().descriptor =
                                Some(descriptor);
                        }
                        (
                            LogicalKey::GraphChunk {
                                partition, chunk, ..
                            },
                            PersistentValue::GraphChunk(value),
                        ) => {
                            bodies
                                .entry(partition.get())
                                .or_default()
                                .chunks
                                .push((chunk, value));
                        }
                        _ => {}
                    }
                }
                let kernel = VectorKernel::new(2, metric, [3; 32]).unwrap();
                let tree = TreeKey::encode(&[], &[]).unwrap();
                let mut internal_levels = std::collections::BTreeSet::new();
                for (key, body) in bodies {
                    let header = body.header.unwrap();
                    if header.level() == 1 {
                        assert!(body.descriptor.is_none() && body.chunks.is_empty());
                        continue;
                    }
                    internal_levels.insert(header.level());
                    let centroids = Centroids::new(&kernel, &body.children, 4, &|| Ok(())).unwrap();
                    let graph = crate::search::local_hnsw::body::FixedInternalBody::restore(
                        body.children.clone().into_boxed_slice(),
                        VectorKernel::new(2, metric, [3; 32]).unwrap(),
                        GraphSource {
                            index: index.logical_index_id(),
                            tree: &tree,
                            parent: PartitionKey::new(key).unwrap(),
                            header,
                            parameters: index.config().fixed_graph().unwrap().parameters(),
                        },
                        body.descriptor.unwrap(),
                        body.chunks,
                        budget,
                        &|| Ok(()),
                    )
                    .unwrap();
                    let query = kernel.preprocess(&[5.0, 2.0]).unwrap();
                    let n = body.children.len();
                    let mut scratch = Workspace::new(n, 8, budget).unwrap();
                    let hits = graph
                        .search(&query, n, 8, budget, &mut scratch, &|| Ok(()))
                        .unwrap()
                        .0;
                    let exact = centroids.exact(&query, n, budget, &|| Ok(())).unwrap();
                    assert_eq!(hits, exact);
                }
                assert!(internal_levels.len() >= 2);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
