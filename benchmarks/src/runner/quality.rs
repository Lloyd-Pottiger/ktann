//! Quality sweeps over a constructed or verified persistent fixture.

use std::time::Instant;

use ktann::api::Index;
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;

use crate::backend::{BackendCounters, MeasuredBackend};
use crate::dataset::BenchmarkDataset;
use crate::metrics::MetricCapture;
use crate::report::{OperationClass, QualityPoint, QualitySweepMeasurements, Topology};
use crate::resource::ResourceSnapshot;
use crate::runner::construction::{
    prepare_index, settle_timeout, verified_topology, verify_measured_state,
};
use crate::runner::error_at;
use crate::runner::workload::{ExactTruth, exact_truth, measure_steady_workload, search_request};
use crate::scenario::{ScenarioSpec, index_config};
use crate::{dataset, oracle};

/// Measures an ordered beam curve over one loaded and converged Logical Index.
pub(super) async fn run_quality_sweep<B: Backend>(
    runtime: &Runtime<MeasuredBackend<B>>,
    backend_counters: &BackendCounters,
    metric_capture: &MetricCapture,
    spec: &ScenarioSpec,
    dataset: &mut BenchmarkDataset,
) -> Result<(Topology, QualitySweepMeasurements), String> {
    // Reuse requires the same dataset and construction inputs.
    let identity = format!(
        "{:?};base={};write-beam={};refinement={:?};sample-items={:?}",
        index_config(spec)?,
        dataset::checksum(&dataset.ids, &dataset.base, &[]),
        spec.write_beam_size,
        spec.refinement_rounds,
        spec.routing_experiment.as_ref().map(|e| e.sample_items)
    );
    let manifest_path = spec
        .persisted_index
        .as_ref()
        .map(|path| path.join("fixture.json"));
    let (index, topology, construction) = if spec.reuse_index {
        let path = manifest_path
            .as_ref()
            .expect("reuse has a fixture directory");
        let persisted: PersistedFixture = serde_json::from_slice(
            &std::fs::read(path).map_err(|error| format!("read fixture manifest: {error}"))?,
        )
        .map_err(|error| format!("decode fixture manifest: {error}"))?;
        if persisted.identity != identity {
            return Err("persisted fixture construction inputs differ".to_owned());
        }
        let index = runtime
            .open_index("benchmark")
            .await
            .map_err(|error| error_at("reopen fixture index", error))?;
        let topology = verified_topology(
            &index,
            spec,
            "verify reused fixture",
            Instant::now() + settle_timeout(spec),
        )
        .await?;
        if topology != persisted.topology {
            return Err("persisted fixture topology changed".to_owned());
        }
        (index, topology, None)
    } else {
        let (index, topology, construction) =
            prepare_index(runtime, backend_counters, metric_capture, spec, dataset).await?;
        if let Some(path) = &manifest_path {
            let fixture = PersistedFixture {
                identity: identity.clone(),
                topology: topology.clone(),
            };
            let bytes = serde_json::to_vec_pretty(&fixture)
                .map_err(|error| format!("encode fixture manifest: {error}"))?;
            std::fs::write(path, bytes)
                .map_err(|error| format!("write fixture manifest: {error}"))?;
        }
        (index, topology, Some(construction))
    };
    if spec.routing_experiment.is_none() && topology.max_level.is_none_or(|level| level < 3) {
        return Err(format!(
            "large quality topology has max level {:?} with {} partitions by level {:?}; expected at least three searchable levels",
            topology.max_level, topology.partitions, topology.partitions_by_level
        ));
    }
    let mut routing_setup = serde_json::json!({"fixture_identity": identity});
    if spec.routing_experiment.is_some() {
        let started = Instant::now();
        routing_setup["leaf_layout_sha256"] = serde_json::json!(
            index
                .leaf_layout_fingerprint()
                .await
                .map_err(|e| error_at("leaf layout fingerprint", e))?
        );
        routing_setup["fingerprint_seconds"] = serde_json::json!(started.elapsed().as_secs_f64());
    }
    // Compute exact truth before releasing the base vectors, including
    // bounded subsets whose full-corpus ground truth no longer applies.
    let truth = exact_truth(dataset, spec.metric, spec.k);
    // Release the imported million-vector corpus before measuring search.
    dataset.ids = Vec::new();
    dataset.base = Vec::new();
    let concurrencies = if spec.query_concurrency_sweep.is_empty() {
        std::slice::from_ref(&spec.concurrency)
    } else {
        &spec.query_concurrency_sweep
    };
    let mut points = Vec::with_capacity(spec.leaf_beam_sweep.len() * concurrencies.len());
    for concurrency in concurrencies {
        for beam in &spec.leaf_beam_sweep {
            let mut point = spec.clone();
            point.concurrency = *concurrency;
            point.search_options = point
                .search_options
                .with_leaf_beam_size(*beam)
                .map_err(|error| error_at("configure quality point", error))?;
            let mut measurements = measure_steady_workload(
                &index,
                backend_counters,
                metric_capture,
                &point,
                dataset,
                Some(&truth),
            )
            .await?;
            // getrusage exposes only the process-lifetime high-water mark, which
            // setup already established and cannot attribute to one beam point.
            measurements.peak_rss_bytes = None;
            points.push(QualityPoint {
                leaf_beam_size: *beam,
                concurrency: *concurrency,
                measurements,
            });
        }
        // Each client count retains the full quality/completion contract.
        let start = points.len() - spec.leaf_beam_sweep.len();
        validate_quality_frontier(
            &points[start..],
            spec.measured_operations,
            spec.routing_experiment.is_none(),
        )?;
    }
    if spec.routing_experiment.is_some() {
        routing_setup["measured_process_peak_rss_bytes"] =
            serde_json::json!(ResourceSnapshot::capture()?.peak_rss_bytes());
    }
    // All latency, IO, CPU and peak-RSS snapshots precede diagnostic replay.
    if spec.routing_experiment.is_some() {
        // The fixture is frozen for the whole sweep. Resolve truth membership
        // once after timing, rather than repeating identical backend reads per beam.
        let mut truth_locations = Vec::with_capacity(truth.len());
        for expected in &truth {
            truth_locations.push(
                index
                    .benchmark_truth_leaves(expected.iter().map(|(id, _)| id.clone()).collect())
                    .await
                    .map_err(|e| error_at("truth leaf membership", e))?,
            );
        }
        for point in &mut points {
            let mut diagnostic_spec = spec.clone();
            diagnostic_spec.search_options = diagnostic_spec
                .search_options
                .with_leaf_beam_size(point.leaf_beam_size)
                .map_err(|e| e.to_string())?;
            point.measurements.quality_diagnostics = Some(
                quality_diagnostics(&index, dataset, &diagnostic_spec, &truth, &truth_locations)
                    .await?,
            );
        }
    }
    verify_measured_state(&index, spec).await?;
    Ok((
        topology,
        QualitySweepMeasurements {
            routing_setup: spec.routing_experiment.as_ref().map(|_| routing_setup),
            construction,
            points,
        },
    ))
}

/// Completed, verified persisted fixture identity and topology.
#[derive(serde::Serialize, serde::Deserialize)]
struct PersistedFixture {
    identity: String,
    topology: Topology,
}

/// Checks completion, and optionally requires the standard fixture to expose a frontier.
fn validate_quality_frontier(
    points: &[QualityPoint],
    expected_searches: usize,
    require_variation: bool,
) -> Result<(), String> {
    let expected_searches = u64::try_from(expected_searches)
        .map_err(|_| "large quality search count exceeds u64".to_owned())?;
    let mut recalls = Vec::with_capacity(points.len());
    let mut leaf_work = Vec::with_capacity(points.len());
    for point in points {
        let search = point
            .measurements
            .operations
            .get(&OperationClass::Search)
            .ok_or_else(|| format!("leaf beam {} has no search results", point.leaf_beam_size))?;
        let recall =
            point.measurements.recall_at_k.as_ref().ok_or_else(|| {
                format!("leaf beam {} has no recall results", point.leaf_beam_size)
            })?;
        let leaf_work_usage = &point.measurements.visited_leaf_entries;
        if search.attempted != expected_searches
            || search.accepted != expected_searches
            || recall.queries != expected_searches
            || leaf_work_usage.count != expected_searches
        {
            return Err(format!(
                "leaf beam {} completed {}/{} searches with {} recall and {} Leaf Entry samples; expected {expected_searches}",
                point.leaf_beam_size,
                search.accepted,
                search.attempted,
                recall.queries,
                leaf_work_usage.count,
            ));
        }
        recalls.push(recall.mean);
        leaf_work.push(leaf_work_usage.mean);
    }
    let recall_moves = recalls
        .iter()
        .copied()
        .reduce(f64::min)
        .zip(recalls.iter().copied().reduce(f64::max))
        .is_some_and(|(minimum, maximum)| maximum > minimum);
    let work_moves = leaf_work.windows(2).any(|window| window[1] > window[0]);
    if require_variation && points.len() > 1 && (!recall_moves || !work_moves) {
        return Err(
            "large quality sweep did not produce a nontrivial quality/work frontier".to_owned(),
        );
    }
    Ok(())
}

/// Replays unique held-out queries after every timed resource counter is captured.
async fn quality_diagnostics<B: Backend>(
    index: &Index<MeasuredBackend<B>>,
    dataset: &BenchmarkDataset,
    spec: &ScenarioSpec,
    truth: &[ExactTruth],
    truth_locations: &[Vec<(ktann::api::PartitionKey, u64)>],
) -> Result<crate::report::QualityDiagnostics, String> {
    let mut queries = Vec::with_capacity(dataset.queries.len());
    for (query, expected) in truth.iter().enumerate() {
        let request = search_request(dataset, spec, query)?;
        let (outcome, diagnostic) = index
            .benchmark_search(request, expected.iter().map(|(id, _)| id.clone()).collect())
            .await
            .map_err(|e| error_at("quality replay", e))?;
        let truth_leaves = expected
            .iter()
            .zip(&truth_locations[query])
            .map(
                |((id, _), &(leaf, entries))| crate::report::TruthLeafQuality {
                    leaf: leaf.get(),
                    entries,
                    scanned: diagnostic.leaves.contains(&leaf),
                    returned: outcome.hits.iter().any(|hit| hit.id() == id),
                },
            )
            .collect();
        let denominator = expected.len() as f64;
        queries.push(crate::report::QueryQuality {
            query,
            true_neighbor_leaf_coverage: diagnostic.neighbors_in_leaves as f64 / denominator,
            global_candidate_coverage: diagnostic.neighbors_in_global_pool as f64 / denominator,
            rerank_candidate_coverage: diagnostic.neighbors_reranked as f64 / denominator,
            final_recall: oracle::recall_ids(outcome.hits.iter().map(|hit| hit.id()), expected),
            truth_leaves,
        });
    }
    Ok(crate::report::QualityDiagnostics { queries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{Distribution, RecallSummary, SteadyStateMeasurements};
    use crate::runner::tests::operation_summary;

    fn quality_point(beam: u32, searches: u64, recall: f64, leaf_work: f64) -> QualityPoint {
        let mut measurements = SteadyStateMeasurements {
            recall_at_k: Some(RecallSummary {
                queries: searches,
                mean: recall,
                min: recall,
            }),
            ..Default::default()
        };
        measurements.operations.insert(
            OperationClass::Search,
            operation_summary(searches, searches, 0),
        );
        measurements.visited_leaf_entries = Distribution {
            count: searches,
            mean: leaf_work,
            ..Default::default()
        };
        QualityPoint {
            leaf_beam_size: beam,
            concurrency: 16,
            measurements,
        }
    }

    #[test]
    fn quality_frontier_requires_complete_results_for_every_beam() {
        let mut points = vec![
            quality_point(1, 2, 0.5, 10.0),
            quality_point(2, 2, 1.0, 20.0),
        ];
        assert!(validate_quality_frontier(&points, 2, true).is_ok());

        points[0].measurements.recall_at_k = None;
        assert!(validate_quality_frontier(&points, 2, true).is_err());
    }

    #[test]
    fn single_beam_diagnostic_still_requires_every_search_to_complete() {
        let mut points = vec![quality_point(32, 2, 0.9, 10.0)];
        assert!(validate_quality_frontier(&points, 2, true).is_ok());
        points[0]
            .measurements
            .recall_at_k
            .as_mut()
            .expect("recall")
            .queries = 1;
        assert!(validate_quality_frontier(&points, 2, true).is_err());
    }

    #[test]
    fn experimental_frontier_allows_plateaus_but_requires_completion() {
        let mut points = vec![
            quality_point(4, 2, 1.0, 20.0),
            quality_point(16, 2, 1.0, 20.0),
        ];
        assert!(validate_quality_frontier(&points, 2, false).is_ok());
        assert!(validate_quality_frontier(&points, 2, true).is_err());
        points[0].measurements.recall_at_k.as_mut().unwrap().queries = 1;
        assert!(validate_quality_frontier(&points, 2, false).is_err());
    }
}
