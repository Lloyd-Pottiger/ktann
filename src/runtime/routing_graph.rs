//! Foreground-owned graph loading and native query execution, ready for traversal.

use super::{OperationContext, check_control, compute::controlled_compute};
use crate::api::{PartitionKey, Result};
use crate::search::cache::graph::{GraphRead, RoutingGraph, read_graph};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::{Backend, ReadOps};
use crate::storage::keys::TreeKey;
use crate::storage::values::IndexManifest;

impl<B: Backend> OperationContext<B> {
    /// Uses the complete query's already validated transaction; it never opens a
    /// replacement snapshot. The caller reserves all other retained routing state
    /// and releases/consumes each parent before requesting another graph.
    pub(crate) async fn read_routing_graph<T: ReadOps>(
        &self,
        txn: &mut ReadLogicalTxn<'_, T>,
        manifest: &IndexManifest,
        tree: &TreeKey,
        parent: PartitionKey,
        retained_bytes: usize,
    ) -> Result<RoutingGraph> {
        let admission = self.cpu_admission.as_ref().expect("search CPU admission");
        let input = read_graph(
            txn,
            &self.partition_cache,
            manifest,
            tree,
            parent,
            admission.inner.config.routing_working_bytes(),
            retained_bytes,
            &|| self.checkpoint(),
        )
        .await?;
        match input {
            GraphRead::Cached(graph) => {
                self.checkpoint()?;
                Ok(graph)
            }
            GraphRead::Load(load) => {
                let cache = self.partition_cache();
                controlled_compute(self.options.clone(), admission.clone(), move |options| {
                    load.restore(&cache, &|| check_control(options))
                })
                .await
            }
        }
    }

    /// Consumes one parent handle. Native scratch is query-private, and admission
    /// remains held until the blocking task exits even if the async future drops.
    pub(crate) async fn search_routing_graph(
        &self,
        graph: RoutingGraph,
        query: Vec<f32>,
        beam: usize,
        ef: usize,
    ) -> Result<(Vec<(PartitionKey, f64)>, ktann_usearch::Stats)> {
        controlled_compute(
            self.options.clone(),
            self.cpu_admission.clone().expect("search CPU admission"),
            move |options| graph.search(&query, beam, ef, &|| check_control(options)),
        )
        .await
    }
}
