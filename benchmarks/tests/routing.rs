//! Real-adapter routing comparisons use one persisted immutable fixture.
#![cfg(feature = "rocksdb")]
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

fn run(root: &Path, mode: &str, reuse: bool, budget: Option<usize>) -> Output {
    let mut experiment = json!({"mode": mode, "m":8,"ef_construction":40,"graph_seed":7,"ef_search":16,"build_memory_bytes":33554432});
    if let Some(bytes) = budget {
        experiment["global_working_bytes"] = json!(bytes);
    }
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
    for (ordinal, mode) in [
        "exact",
        "partition-local-hnsw",
        "global-exact",
        "global-hnsw",
    ]
    .iter()
    .enumerate()
    {
        let output = run(root.path(), mode, ordinal > 0, None);
        assert!(
            output.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let suite: Value = serde_json::from_slice(
            &std::fs::read(root.path().join(format!("{mode}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(suite["schema_version"], 9);
        let report = &suite["reports"][0];
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
        assert_eq!(point["per_query_recall"].as_array().unwrap().len(), 64);
        assert!(point["routing"]["distances"].as_u64().unwrap() > 0);
        if ordinal < 2 {
            assert!(point["routing"]["graph_loads"].as_u64().unwrap() > 0);
        }
        let diagnostics = &point["quality_diagnostics"];
        assert!(diagnostics["global_reference_skipped"].is_null());
        assert_eq!(diagnostics["queries"].as_array().unwrap().len(), 8);
        for quality in diagnostics["queries"].as_array().unwrap() {
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
            if *mode == "global-exact" {
                assert_eq!(quality["global_centroid_coverage"], 1.0);
            }
        }
    }
    let global_report: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("global-exact.json")).unwrap())
            .unwrap();
    let reference_resident_bytes = global_report["reports"][0]["measurements"]["values"]["routing_setup"]["global"]["resident_bytes"].as_u64().unwrap() as usize;
    let output = run(
        root.path(),
        "exact",
        true,
        Some(reference_resident_bytes + 256),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("exact.json")).unwrap()).unwrap();
    let skipped = report["reports"][0]["measurements"]["values"]["points"][0]["measurements"]["quality_diagnostics"]["global_reference_skipped"].as_str().unwrap();
    assert!(skipped.contains("reference query"), "{skipped}");

    // A diagnostic-only global reference may skip; the production route must run.
    let output = run(root.path(), "exact", true, Some(1));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("exact.json")).unwrap()).unwrap();
    assert!(report["reports"][0]["measurements"]["values"]["points"][0]["measurements"]["quality_diagnostics"]["global_reference_skipped"].is_string());

    // The selected global mode must fail, not silently fall back to tree routing.
    let output = run(root.path(), "global-hnsw", true, Some(1));
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("global baseline refused"));
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
