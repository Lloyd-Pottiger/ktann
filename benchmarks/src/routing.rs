//! Frozen-dataset experiment settings shared by CLI workers and reports.
use ktann::api::{LocalGraphConfig, RoutingMode};
use serde::{Deserialize, Serialize};

/// Explicit routing comparison; global variants are benchmark-only.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    /// Production exact tree routing.
    #[default]
    Exact,
    /// Production partition-local graph routing.
    PartitionLocalHnsw,
    /// Exact scan over all final leaf centroids.
    GlobalExact,
    /// One in-memory graph over all final leaf centroids.
    GlobalHnsw,
}

/// Fully resolved experiment inputs. Unknown fields are rejected, including typos.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Experiment {
    /// Routing policy; all policies use the same graph-index serving format.
    pub mode: Mode,
    /// Graph maximum upper-layer degree.
    pub m: u32,
    /// Graph construction expansion.
    pub ef_construction: u32,
    /// Little-endian u64 in the first eight bytes of the existing 32-byte graph seed.
    pub graph_seed: u64,
    /// Requested query expansion (effective expansion is also bounded by beam/n).
    pub ef_search: u32,
    /// Per-operation production routing working memory.
    pub routing_working_bytes: usize,
    /// Global baseline retained plus private operation memory ceiling.
    pub global_working_bytes: usize,
    /// Existing construction sampling control.
    pub sample_items: usize,
    /// Per-worker construction/serving/refinement memory ceiling.
    pub build_memory_bytes: usize,
    /// Per-worker scratch and artifact byte ceiling.
    pub build_scratch_bytes: u64,
}

impl Default for Experiment {
    fn default() -> Self {
        Self {
            mode: Mode::Exact,
            m: 16,
            ef_construction: 200,
            graph_seed: 0,
            ef_search: 200,
            routing_working_bytes: 256 << 20,
            global_working_bytes: 256 << 20,
            sample_items: 1024,
            build_memory_bytes: 256 << 20,
            build_scratch_bytes: 64 << 30,
        }
    }
}

impl Experiment {
    /// Validates semantic and resource inputs before a worker creates an index.
    pub fn validate(&self) -> Result<(), String> {
        self.graph().map_err(|e| e.to_string())?;
        if self.ef_search == 0
            || self.routing_working_bytes == 0
            || self.global_working_bytes == 0
            || self.sample_items == 0
            || self.build_memory_bytes == 0
            || self.build_scratch_bytes == 0
        {
            return Err("routing experiment controls must be positive".into());
        }
        Ok(())
    }

    /// Resolves the same construction controls for the native runner and bridge.
    pub fn construction(&self) -> ktann::construction::ConstructionOptions {
        ktann::construction::ConstructionOptions {
            sample_items: self.sample_items,
            memory_bytes: self.build_memory_bytes,
            scratch_bytes: self.build_scratch_bytes,
        }
    }

    /// Applies explicit memory/scratch ceilings at their existing Bulk Build owner.
    pub fn worker(&self, workspace: std::path::PathBuf) -> ktann::api::BulkWorkerOptions {
        let mut worker = ktann::api::BulkWorkerOptions::new(workspace);
        worker.sort_memory_bytes = self.build_memory_bytes;
        worker.serving_memory_bytes = self.build_memory_bytes;
        worker.graph_memory_bytes = self.build_memory_bytes;
        worker.refinement_memory_bytes = self.build_memory_bytes;
        worker.sort_scratch_bytes = self.build_scratch_bytes;
        worker.serving_scratch_bytes = self.build_scratch_bytes;
        worker.max_artifact_bytes = self.build_scratch_bytes;
        worker
    }

    /// Persistent graph parameters, identical across routing policies.
    pub fn graph(&self) -> ktann::api::Result<LocalGraphConfig> {
        let mut seed = [0; 32];
        seed[..8].copy_from_slice(&self.graph_seed.to_le_bytes());
        LocalGraphConfig::new(self.m, self.ef_construction, seed)
    }

    /// Production routing selection; global policies bypass internal traversal.
    pub fn production_mode(&self) -> RoutingMode {
        match self.mode {
            Mode::PartitionLocalHnsw => RoutingMode::PartitionLocalHnsw {
                ef_search: self.ef_search as usize,
            },
            _ => RoutingMode::Exact,
        }
    }

    /// Whether to prepare the bounded global comparison before warmup.
    pub fn global(&self) -> bool {
        matches!(self.mode, Mode::GlobalExact | Mode::GlobalHnsw)
    }
}
