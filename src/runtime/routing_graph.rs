//! Foreground-owned graph loading and native query execution, ready for traversal.

use super::{OperationContext, check_control, compute::controlled_compute};
use crate::api::{PartitionKey, Result};
use crate::search::cache::BodyRequest;
use crate::search::cache::graph::{GraphRead, RoutingGraph, read_graph};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::{Backend, ReadOps};
use crate::storage::values::IndexManifest;

impl<B: Backend> OperationContext<B> {
    /// Uses the complete query's already validated transaction; it never opens a
    /// replacement snapshot. The caller reserves all other retained routing state
    /// and releases/consumes each parent before requesting another graph.
    pub(crate) async fn read_routing_graph<T: ReadOps>(
        &self,
        txn: &mut ReadLogicalTxn<'_, T>,
        manifest: &IndexManifest,
        request: BodyRequest<'_>,
        retained_bytes: usize,
    ) -> Result<RoutingGraph> {
        let admission = self.cpu_admission.as_ref().expect("search CPU admission");
        let input = read_graph(
            txn,
            &self.partition_cache,
            manifest,
            request,
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
}

impl<B: Backend> crate::search::cache::graph::RoutingExecution for OperationContext<B> {
    fn checkpoint(&self) -> Result<()> {
        self.checkpoint()
    }

    async fn read<T: ReadOps>(
        &self,
        txn: &mut ReadLogicalTxn<'_, T>,
        manifest: &IndexManifest,
        request: BodyRequest<'_>,
        retained_bytes: usize,
    ) -> Result<RoutingGraph> {
        self.read_routing_graph(txn, manifest, request, retained_bytes)
            .await
    }

    async fn candidates(
        &self,
        graph: RoutingGraph,
        query: Vec<f32>,
        beam: usize,
        mode: crate::api::RoutingMode,
    ) -> Result<Vec<(PartitionKey, f64)>> {
        controlled_compute(
            self.options.clone(),
            self.cpu_admission.clone().expect("search CPU admission"),
            move |options| match mode {
                crate::api::RoutingMode::PartitionLocalHnsw { ef_search } => Ok(graph
                    .search(&query, beam, ef_search, &|| check_control(options))?
                    .0),
                crate::api::RoutingMode::Exact => {
                    graph.exact(&query, beam, &|| check_control(options))
                }
            },
        )
        .await
    }
}
