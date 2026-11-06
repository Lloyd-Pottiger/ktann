//! Deterministic benchmark scenarios and their validated configurations.

use ktann::api::{
    DataType, FieldId, FieldSchema, IndexConfig, Metric, RuntimeConfig, SearchBudgets,
    SearchOptions,
};

use crate::report::{
    AdmissionTarget, BudgetConfiguration, SearchBudgetConfiguration, WorkloadDispatch,
};
use crate::runner::error_at;

/// Workers used after import diagnostics to converge persistent topology.
pub(crate) const CONVERGENCE_MAINTENANCE_WORKERS: usize = 2;

/// Queue capacity for demand-driven Structure Maintenance.
const FIXUP_QUEUE_CAPACITY: usize = 1_024;

/// One fixed scenario in a named scale profile.
#[derive(Clone, Debug)]
pub struct ScenarioSpec {
    /// Stable scenario name used as the comparison key.
    pub name: &'static str,
    /// Frozen-dataset experimental construction and routing controls.
    pub routing_experiment: Option<crate::routing::Experiment>,
    /// Public fixture or synthetic distribution.
    pub dataset: &'static str,
    /// Human-readable scale profile.
    pub profile: &'static str,
    /// Indexed vector count.
    pub base_vectors: usize,
    /// Held-out query count.
    pub query_vectors: usize,
    /// Query-window start used by bounded large diagnostic runs.
    pub query_offset: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Distance metric used by the Logical Index and supplied ground truth.
    pub metric: Metric,
    /// Replayable dataset and workload seed.
    pub seed: u64,
    /// Percentage of operations that are searches.
    pub search_percent: u8,
    /// Whether updates concentrate on a small conflict set.
    pub hot_updates: bool,
    /// Runtime Partition Cache capacity.
    pub partition_cache_bytes: u64,
    /// Runtime foreground concurrency and wait bound.
    pub foreground_limit: usize,
    /// RocksDB native actor bound; ignored by other Backends.
    pub blocking_resource_limit: Option<usize>,
    /// Retained RocksDB fixture directory for controlled search comparisons.
    pub persisted_index: Option<std::path::PathBuf>,
    /// Reopen a verified fixture instead of importing it again.
    pub reuse_index: bool,
    /// Concurrent clients in the timed workload.
    pub concurrency: usize,
    /// Large-profile client counts measured on the same imported index.
    pub query_concurrency_sweep: Vec<usize>,
    /// Dispatch policy for the bounded concurrent clients.
    pub dispatch: WorkloadDispatch,
    /// Operations outside the timed region that establish steady state.
    pub warmup_operations: usize,
    /// Operations in the timed region.
    pub measured_operations: usize,
    /// Search result count.
    pub k: usize,
    /// Per-request Search Budget and traversal overrides.
    pub search_options: SearchOptions,
    /// Per-level beam used while importing records into the tree.
    pub write_beam_size: u32,
    /// Offline refinement rounds after ordinary import; None skips refinement.
    pub refinement_rounds: Option<usize>,
    /// Ordered single-variable beam values; empty for ordinary scenarios.
    pub leaf_beam_sweep: Vec<u32>,
    /// Minimum non-root leaf occupancy.
    pub min_leaf_partition_entries: u32,
    /// Leaf split threshold.
    pub max_leaf_partition_entries: u32,
    /// Minimum non-root internal occupancy.
    pub min_internal_partition_entries: u32,
    /// Internal split threshold.
    pub max_internal_partition_entries: u32,
    /// Whether this scenario measures the import-to-search lifecycle.
    pub lifecycle: bool,
    /// Records per atomic batch load batch in a lifecycle scenario.
    pub import_batch_size: usize,
    /// Background Structure Maintenance workers.
    pub maintenance_workers: usize,
}

/// Returns the complete deterministic scenario matrix for one scale profile.
///
/// # Errors
///
/// Returns an error when `profile` is unknown.
pub fn scenarios(profile: &str) -> Result<Vec<ScenarioSpec>, String> {
    match profile {
        "smoke" => Ok(smoke_scenarios()),
        "full" => Ok(baseline_scenarios("full")),
        "diagnostic" => Ok(baseline_scenarios("diagnostic")),
        "large" => large_scenarios(),
        _ => Err(format!(
            "unknown profile `{profile}`; expected smoke, full, large, or diagnostic"
        )),
    }
}

/// Small deterministic matrix used to catch runner and adapter breakage in CI.
fn smoke_scenarios() -> Vec<ScenarioSpec> {
    let common = ScenarioSpec {
        name: "ann-warm-cache",
        dataset: "clustered",
        profile: "smoke",
        base_vectors: 256,
        query_vectors: 8,
        query_offset: 0,
        dimension: 16,
        metric: Metric::L2,
        seed: 0x38_0001,
        search_percent: 100,
        hot_updates: false,
        partition_cache_bytes: 4 << 20,
        foreground_limit: 8,
        blocking_resource_limit: None,
        persisted_index: None,
        reuse_index: false,
        concurrency: 2,
        dispatch: WorkloadDispatch::Continuous,
        warmup_operations: 16,
        measured_operations: 64,
        k: 10,
        search_options: SearchOptions::default(),
        write_beam_size: 8,
        refinement_rounds: None,
        routing_experiment: None,
        query_concurrency_sweep: Vec::new(),
        leaf_beam_sweep: Vec::new(),
        min_leaf_partition_entries: 8,
        max_leaf_partition_entries: 32,
        min_internal_partition_entries: 8,
        max_internal_partition_entries: 32,
        lifecycle: false,
        import_batch_size: 32,
        maintenance_workers: 2,
    };
    vec![
        common.clone(),
        ScenarioSpec {
            name: "quality-routing-smoke",
            routing_experiment: Some(crate::routing::Experiment::default()),
            maintenance_workers: 0,
            leaf_beam_sweep: vec![4, 16],
            ..common.clone()
        },
        ScenarioSpec {
            name: "ann-cache-disabled",
            partition_cache_bytes: 0,
            ..common.clone()
        },
        ScenarioSpec {
            name: "mixed-search50-update50-hot",
            search_percent: 50,
            hot_updates: true,
            concurrency: 4,
            measured_operations: 80,
            ..common.clone()
        },
        ScenarioSpec {
            name: "backend-admission-saturated",
            search_percent: 50,
            hot_updates: true,
            blocking_resource_limit: Some(1),
            concurrency: 16,
            foreground_limit: 4,
            dispatch: WorkloadDispatch::FixedWaves(admission_target()),
            measured_operations: 640,
            ..common.clone()
        },
        ScenarioSpec {
            name: "import-to-search-lifecycle",
            lifecycle: true,
            search_percent: 100,
            concurrency: 1,
            warmup_operations: 0,
            measured_operations: 8,
            hot_updates: false,
            blocking_resource_limit: None,
            persisted_index: None,
            reuse_index: false,
            dispatch: WorkloadDispatch::Continuous,
            ..common.clone()
        },
    ]
}

/// Scheduled baselines and opt-in diagnostics share the same measurement scale.
fn baseline_scenarios(profile: &'static str) -> Vec<ScenarioSpec> {
    let ann = |name, dataset, base_vectors, query_vectors, dimension, seed| ScenarioSpec {
        name,
        dataset,
        profile,
        base_vectors,
        query_vectors,
        query_offset: 0,
        dimension,
        metric: Metric::L2,
        seed,
        search_percent: 100,
        hot_updates: false,
        partition_cache_bytes: 64 << 20,
        foreground_limit: 32,
        blocking_resource_limit: None,
        persisted_index: None,
        reuse_index: false,
        concurrency: 4,
        dispatch: WorkloadDispatch::Continuous,
        warmup_operations: query_vectors,
        measured_operations: query_vectors.saturating_mul(10),
        k: 10,
        search_options: SearchOptions::default(),
        write_beam_size: 8,
        refinement_rounds: None,
        routing_experiment: None,
        query_concurrency_sweep: Vec::new(),
        leaf_beam_sweep: Vec::new(),
        min_leaf_partition_entries: 32,
        max_leaf_partition_entries: 128,
        min_internal_partition_entries: 32,
        max_internal_partition_entries: 128,
        lifecycle: false,
        import_batch_size: 50,
        maintenance_workers: 2,
    };
    let clustered = ann("ann-clustered", "clustered", 5_000, 100, 128, 0x38_1001);
    if profile == "diagnostic" {
        return vec![
            ann(
                "ann-fashion-mnist",
                "fashion-mnist",
                10_000,
                20,
                784,
                0x38_1003,
            ),
            ScenarioSpec {
                name: "backend-admission-saturated",
                search_percent: 50,
                hot_updates: true,
                blocking_resource_limit: Some(1),
                foreground_limit: 8,
                concurrency: 32,
                measured_operations: 2_000,
                dispatch: WorkloadDispatch::FixedWaves(admission_target()),
                ..clustered.clone()
            },
        ];
    }
    vec![
        clustered.clone(),
        ann("ann-skewed", "skewed", 5_000, 100, 128, 0x38_1004),
        ann("ann-duplicates", "duplicates", 5_000, 100, 128, 0x38_1005),
        ScenarioSpec {
            name: "ann-cache-disabled",
            partition_cache_bytes: 0,
            ..clustered.clone()
        },
        ScenarioSpec {
            name: "mixed-search95-update5",
            search_percent: 95,
            measured_operations: 2_000,
            concurrency: 8,
            ..clustered.clone()
        },
        ScenarioSpec {
            name: "mixed-search50-update50-hot",
            search_percent: 50,
            hot_updates: true,
            measured_operations: 2_000,
            concurrency: 8,
            ..clustered.clone()
        },
        ScenarioSpec {
            name: "import-to-search-lifecycle",
            dataset: "siftsmall",
            base_vectors: 10_000,
            query_vectors: 100,
            dimension: 128,
            seed: 0x38_1006,
            lifecycle: true,
            search_percent: 100,
            concurrency: 1,
            warmup_operations: 0,
            measured_operations: 100,
            hot_updates: false,
            blocking_resource_limit: None,
            persisted_index: None,
            reuse_index: false,
            dispatch: WorkloadDispatch::Continuous,
            ..clustered.clone()
        },
    ]
}

/// Fixed public million-vector quality curves reserved for optimized workers.
fn large_scenarios() -> Result<Vec<ScenarioSpec>, String> {
    let defaults = RuntimeConfig::default();
    let scenario = |name, dataset, dimension, metric| -> Result<ScenarioSpec, String> {
        let index = IndexConfig::new(dimension, metric)
            .map_err(|error| error_at("configure large index", error))?;
        Ok(ScenarioSpec {
            name,
            dataset,
            profile: "large",
            base_vectors: 1_000_000,
            query_vectors: 1_000,
            query_offset: 0,
            dimension,
            metric,
            seed: 0x38_2001,
            search_percent: 100,
            hot_updates: false,
            partition_cache_bytes: defaults.partition_cache_bytes(),
            foreground_limit: defaults.foreground_operation_limit(),
            blocking_resource_limit: None,
            persisted_index: None,
            reuse_index: false,
            concurrency: 16,
            dispatch: WorkloadDispatch::Continuous,
            warmup_operations: 1_000,
            // Repeat the fixed query corpus to reduce short-interval timing noise.
            measured_operations: 10_000,
            k: 100,
            search_options: SearchOptions::default(),
            write_beam_size: defaults.write_beam_size(),
            refinement_rounds: None,
            routing_experiment: None,
            query_concurrency_sweep: Vec::new(),
            leaf_beam_sweep: vec![1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 192, 256, 384],
            // Follow the library layout defaults; capacity overrides can independently
            // control leaf scan size and internal routing fanout.
            min_leaf_partition_entries: index.min_leaf_partition_entries(),
            max_leaf_partition_entries: index.max_leaf_partition_entries(),
            min_internal_partition_entries: index.min_internal_partition_entries(),
            max_internal_partition_entries: index.max_internal_partition_entries(),
            lifecycle: false,
            import_batch_size: 50,
            maintenance_workers: defaults.maintenance_workers(),
        })
    };
    Ok(vec![
        scenario("quality-cohere-1m", "cohere-1m", 768, Metric::Cosine)?,
        scenario("quality-sift-1m", "sift-1m", 128, Metric::L2)?,
    ])
}

/// Builds the complete process-local configuration used by benchmark workers.
pub(crate) fn runtime_config(
    spec: &ScenarioSpec,
    maintenance_workers: usize,
) -> Result<RuntimeConfig, String> {
    let defaults = RuntimeConfig::default();
    let fixup_queue_capacity = if spec.profile == "large" {
        defaults.fixup_queue_capacity()
    } else {
        FIXUP_QUEUE_CAPACITY
    };
    let defaults = if let Some(experiment) = &spec.routing_experiment {
        defaults
            .with_routing_working_bytes(experiment.routing_working_bytes)
            .map_err(|e| e.to_string())?
    } else {
        defaults
    };
    let config = defaults
        .with_foreground_operation_limit(spec.foreground_limit)
        .and_then(|config| config.with_maintenance(maintenance_workers, fixup_queue_capacity))
        .and_then(|config| config.with_partition_cache_bytes(spec.partition_cache_bytes))
        .and_then(|config| config.with_write_beam_size(spec.write_beam_size));
    let config = if spec.profile == "large" {
        config
    } else {
        config.and_then(|config| config.with_attempts(32, 32))
    };
    config
        .and_then(|config| config.validate().map(|()| config))
        .map_err(|error| error_at("configure runtime", error))
}

/// Resolves the exact limits used by scenario requests through the public API.
pub(crate) fn search_budget_configuration(
    defaults: SearchBudgets,
    options: SearchOptions,
    k: usize,
) -> Result<SearchBudgetConfiguration, String> {
    let effective = options
        .resolve(defaults, k)
        .map_err(|error| error_at("resolve search budgets", error))?;
    Ok(SearchBudgetConfiguration {
        scanned_tree_keys: BudgetConfiguration {
            runtime_default: defaults.scanned_tree_keys(),
            request_override: options.scanned_tree_keys(),
            effective_limit: effective.scanned_tree_keys(),
        },
        visited_partitions: BudgetConfiguration {
            runtime_default: defaults.visited_partitions(),
            request_override: options.visited_partitions(),
            effective_limit: effective.visited_partitions(),
        },
        exact_rerank_candidates: BudgetConfiguration {
            runtime_default: defaults.exact_rerank_candidates(),
            request_override: None,
            effective_limit: effective.exact_rerank_candidates(),
        },
    })
}

/// Returns the stable public name of a configured distance metric.
pub(crate) const fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::L2 => "l2",
        Metric::Cosine => "cosine",
        Metric::InnerProduct => "inner_product",
        _ => "unknown",
    }
}

/// Builds the public Index configuration shared by steady and lifecycle cases.
pub(crate) fn index_config(spec: &ScenarioSpec) -> Result<IndexConfig, String> {
    let config = IndexConfig::new(spec.dimension, spec.metric).and_then(|config| {
        config
            .with_leaf_partition_entries(
                spec.min_leaf_partition_entries,
                spec.max_leaf_partition_entries,
            )?
            .with_internal_partition_entries(
                spec.min_internal_partition_entries,
                spec.max_internal_partition_entries,
            )
    });
    let config = if spec.profile == "large" || spec.routing_experiment.is_some() {
        config
    } else {
        config
            .and_then(|config| {
                config.with_fields(vec![
                    FieldSchema::new("bucket", DataType::I64)?,
                    FieldSchema::new("generation", DataType::I64)?,
                ])
            })
            .and_then(|config| config.with_tree_key_fields(vec![FieldId(0)]))
    };
    let config = config.and_then(|config| {
        if let Some(experiment) = &spec.routing_experiment {
            Ok(config.with_local_graph(experiment.graph()?))
        } else {
            Ok(config)
        }
    });
    config.map_err(|error| error_at("configure index", error))
}

/// Defines a sample-rich mixed region while preserving visible overload.
const fn admission_target() -> AdmissionTarget {
    AdmissionTarget {
        minimum_accepted_per_class: 100,
        minimum_rejection_rate: 0.25,
        maximum_rejection_rate: 0.75,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scenario_has_a_jointly_valid_runtime_configuration() {
        for profile in ["smoke", "full", "large", "diagnostic"] {
            for scenario in scenarios(profile).expect("known profile") {
                for workers in [
                    scenario.maintenance_workers,
                    CONVERGENCE_MAINTENANCE_WORKERS,
                ] {
                    runtime_config(&scenario, workers).unwrap_or_else(|error| {
                        panic!("{} runtime configuration: {error}", scenario.name)
                    });
                }
            }
        }
    }

    #[test]
    fn report_exposes_the_k_derived_effective_rerank_limit() {
        let budgets =
            search_budget_configuration(SearchBudgets::default(), SearchOptions::default(), 10)
                .expect("default search budgets resolve");

        let rerank = budgets.exact_rerank_candidates;
        assert_eq!(rerank.runtime_default, 65_536);
        assert_eq!(rerank.request_override, None);
        assert_eq!(rerank.effective_limit, 64);
    }

    #[test]
    fn report_distinguishes_a_request_budget_override() {
        let options = SearchOptions::default()
            .with_visited_partitions(64)
            .expect("valid request override");
        let budgets = search_budget_configuration(SearchBudgets::default(), options, 10)
            .expect("overridden search budgets resolve");

        let partitions = budgets.visited_partitions;
        assert_eq!(partitions.runtime_default, 1_024);
        assert_eq!(partitions.request_override, Some(64));
        assert_eq!(partitions.effective_limit, 64);
    }

    #[test]
    fn large_profile_uses_library_defaults_for_the_top_100_beam_sweep() {
        for scenario in scenarios("large").expect("large profile") {
            assert_eq!(
                scenario.leaf_beam_sweep,
                [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 192, 256, 384]
            );
            assert_eq!(scenario.k, 100);
            assert_eq!(scenario.search_options, SearchOptions::default());
            assert_eq!(scenario.blocking_resource_limit, None);
            let defaults = ktann::api::RuntimeConfig::default();
            let actual = runtime_config(&scenario, scenario.maintenance_workers)
                .expect("large runtime config");
            assert_eq!(
                actual.partition_cache_bytes(),
                defaults.partition_cache_bytes()
            );
            assert_eq!(
                actual.foreground_operation_limit(),
                defaults.foreground_operation_limit()
            );
            assert_eq!(actual.maintenance_workers(), defaults.maintenance_workers());
            assert_eq!(
                actual.fixup_queue_capacity(),
                defaults.fixup_queue_capacity()
            );
            assert_eq!(actual.foreground_attempts(), defaults.foreground_attempts());
            assert_eq!(actual.fixup_attempts(), defaults.fixup_attempts());
            assert_eq!(actual.write_beam_size(), defaults.write_beam_size());
            let budgets = scenario
                .search_options
                .resolve(actual.default_search_budgets(), scenario.k)
                .expect("large search budgets resolve");
            assert_eq!(
                budgets,
                SearchOptions::default()
                    .resolve(defaults.default_search_budgets(), 100)
                    .expect("default search budgets resolve")
            );
            assert_eq!(budgets.exact_rerank_candidates(), 125);
            assert_eq!(
                super::index_config(&scenario).expect("large index config"),
                ktann::api::IndexConfig::new(scenario.dimension, scenario.metric)
                    .expect("default index config")
            );
        }
    }
}
