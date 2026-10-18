//! Foreground-owned graph loading and native query execution, ready for traversal.

use super::{OperationContext, check_control, compute::controlled_compute};
use crate::api::{PartitionKey, Result};
use crate::search::cache::BodyRequest;
use crate::search::cache::graph::{GraphRead, RoutingGraph, read_graph};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::{Backend, ReadOps};
use crate::storage::values::IndexManifest;

impl<B: Backend> crate::search::cache::graph::RoutingExecution for OperationContext<B> {
    fn checkpoint(&self) -> Result<()> {
        self.checkpoint()
    }

    /// Loads through the query's snapshot and restores under its CPU admission.
    async fn read<T: ReadOps>(
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
            move |options| {
                let started = std::time::Instant::now();
                let result = match mode {
                    crate::api::RoutingMode::PartitionLocalHnsw { ef_search } => {
                        let (hits, stats) =
                            graph.search(&query, beam, ef_search, &|| check_control(options))?;
                        ::metrics::counter!("ktann.routing.distances")
                            .increment(stats.distances as u64);
                        ::metrics::counter!("ktann.routing.graph_visits")
                            .increment(stats.visits as u64);
                        hits
                    }
                    crate::api::RoutingMode::Exact => {
                        let count = graph.children().len();
                        let hits = graph.exact(&query, beam, &|| check_control(options))?;
                        ::metrics::counter!("ktann.routing.distances").increment(count as u64);
                        hits
                    }
                };
                ::metrics::histogram!("ktann.routing.compute_seconds")
                    .record(started.elapsed().as_secs_f64());
                Ok(result)
            },
        )
        .await
    }
}
