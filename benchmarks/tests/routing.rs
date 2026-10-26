//! Real-adapter routing comparisons use one persisted immutable fixture.
#![cfg(feature = "rocksdb")]
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

fn run(root: &Path, mode: &str, reuse: bool) -> Output {
    let experiment = json!({"mode": mode, "m":8,"ef_construction":40,"graph_seed":7,"ef_search":16,"build_memory_bytes":33554432});
    Command::new(env!("CARGO_BIN_EXE_ktann-bench"))
        .args([
            "run",
            "--backend",
            "rocksdb",
            "--profile",
            "smoke",
            "--scenario",
            "quality-routing-smoke",
            "--leaf-beam-size",
            "4",
            "--partition-cache-bytes",
            "0",
            "--warmup-operations",
            "0",
            "--measured-operations",
            "32",
            "--refinement-rounds",
            "0",
            "--routing-experiment",
        ])
        .arg(experiment.to_string())
        .arg("--rocksdb-path")
        .arg(root.join("fixture"))
        .args(["--reuse-index", if reuse { "true" } else { "false" }])
        .arg("--output")
        .arg(root.join(format!("{mode}.json")))
        .output()
        .unwrap()
}

#[test]
fn all_routes_share_layout_and_keep_diagnostics_out_of_measurement() {
    let root = tempfile::tempdir().unwrap();
    let mut fingerprint = None;
    let mut rerank = None;
    for (ordinal, mode) in ["exact", "partition-local-hnsw"].iter().enumerate() {
        let output = run(root.path(), mode, ordinal > 0);
        assert!(
            output.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let suite: Value = serde_json::from_slice(
            &std::fs::read(root.path().join(format!("{mode}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(suite["schema_version"], 10);
        let report = &suite["reports"][0];
        assert_eq!(report["configuration"]["measured_operations"], 32);
        assert_eq!(report["configuration"]["routing_experiment"]["mode"], *mode);
        assert_eq!(
            report["configuration"]["routing_experiment"]["graph_seed"],
            7
        );
        let budgets = &report["configuration"]["search_budgets"]["exact_rerank_candidates"];
        assert_eq!(rerank.get_or_insert_with(|| budgets.clone()), budgets);
        let sweep = &report["measurements"]["values"];
        let identity = &sweep["routing_setup"]["leaf_layout_sha256"];
        assert_eq!(
            fingerprint.get_or_insert_with(|| identity.clone()),
            identity
        );
        assert_eq!(sweep["construction"].is_null(), ordinal > 0);
        if ordinal == 0 {
            let settings = &sweep["construction"]["bulk"];
            for field in [
                "construction_memory_bytes",
                "sort_memory_bytes",
                "serving_memory_bytes",
                "graph_memory_bytes",
                "refinement_memory_bytes",
            ] {
                assert_eq!(settings[field], 33554432, "{field}: {settings}");
            }
        }
        let point = &sweep["points"][0]["measurements"];
        assert_eq!(point["per_query_recall"].as_array().unwrap().len(), 32);
        assert!(point["routing"]["distances"].as_u64().unwrap() > 0);
        assert!(point["routing"]["graph_loads"].as_u64().unwrap() > 0);
        let diagnostics = &point["quality_diagnostics"];
        assert_eq!(diagnostics["queries"].as_array().unwrap().len(), 8);
        for quality in diagnostics["queries"].as_array().unwrap() {
            let truth_leaves = quality["truth_leaves"].as_array().unwrap();
            assert!(!truth_leaves.is_empty());
            let scanned = truth_leaves
                .iter()
                .filter(|leaf| leaf["scanned"] == true)
                .count();
            let returned = truth_leaves
                .iter()
                .filter(|leaf| leaf["returned"] == true)
                .count();
            assert_eq!(
                scanned as f64 / truth_leaves.len() as f64,
                quality["true_neighbor_leaf_coverage"].as_f64().unwrap()
            );
            assert_eq!(
                returned as f64 / truth_leaves.len() as f64,
                quality["final_recall"].as_f64().unwrap()
            );
            for leaf in truth_leaves {
                assert!(leaf["entries"].as_u64().unwrap() > 0);
            }
            let values: Vec<f64> = [
                "true_neighbor_leaf_coverage",
                "local_candidate_coverage",
                "rerank_candidate_coverage",
                "final_recall",
            ]
            .iter()
            .map(|name| quality[name].as_f64().unwrap())
            .collect();
            assert!(
                values.windows(2).all(|pair| pair[0] >= pair[1]),
                "{quality}"
            );
        }
    }
}

#[test]
fn rejects_invalid_experiment_before_backend_setup() {
    for config in [
        r#"{"mode":"typo"}"#,
        r#"{"ef_search":0}"#,
        r#"{"m":0}"#,
        r#"{"unknown":1}"#,
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_ktann-bench"))
            .args([
                "run",
                "--backend",
                "rocksdb",
                "--scenario",
                "quality-routing-smoke",
                "--routing-experiment",
                config,
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{config}");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("dataset load started"));
    }
}
