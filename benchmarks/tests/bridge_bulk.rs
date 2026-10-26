//! Real bridge coverage: finite staging must publish before search is available.
#![cfg(feature = "rocksdb")]
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Bridge {
    process: Child,
    directory: tempfile::TempDir,
    stream: UnixStream,
}
impl Bridge {
    fn start() -> Self {
        Self::start_with(None)
    }

    fn start_with(experiment: Option<Value>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("bridge.sock");
        let scratch = directory.path().join("scratch");
        std::fs::create_dir(&scratch).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_ktann-vdbbench-bridge"));
        if let Some(experiment) = experiment {
            command
                .arg("--routing-experiment")
                .arg(experiment.to_string());
        }
        let mut process = command
            .arg("--backend")
            .arg("rocksdb")
            .arg("--socket")
            .arg(&socket)
            .arg("--database")
            .arg(directory.path().join("database"))
            .arg("--report")
            .arg(directory.path().join("report.json"))
            .env("TMPDIR", &scratch)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        let stream = loop {
            if let Ok(stream) = UnixStream::connect(&socket) {
                break stream;
            }
            assert!(
                process.try_wait().unwrap().is_none(),
                "bridge exited before startup"
            );
            if Instant::now() >= deadline {
                process.kill().unwrap();
                panic!("bridge startup timeout");
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(60)))
            .unwrap();
        Self {
            process,
            directory,
            stream,
        }
    }
    fn request(&mut self, mut request: Value) -> Value {
        request["version"] = json!(1);
        self.exchange(&serde_json::to_vec(&request).unwrap())
    }
    fn insert(&mut self, ids: &[i64], vectors: &[Vec<f32>]) -> Value {
        assert_eq!(ids.len(), vectors.len());
        let dimension = vectors[0].len();
        let mut data = b"KTI\x01".to_vec();
        data.extend_from_slice(&(ids.len() as u32).to_be_bytes());
        data.extend_from_slice(&(dimension as u32).to_be_bytes());
        for id in ids {
            data.extend_from_slice(&id.to_le_bytes());
        }
        for vector in vectors {
            assert_eq!(vector.len(), dimension);
            for value in vector {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        self.exchange(&data)
    }
    fn exchange(&mut self, data: &[u8]) -> Value {
        self.stream
            .write_all(&(data.len() as u32).to_be_bytes())
            .unwrap();
        self.stream.write_all(data).unwrap();
        let mut size = [0; 4];
        self.stream.read_exact(&mut size).unwrap();
        let mut bytes = vec![0; u32::from_be_bytes(size) as usize];
        self.stream.read_exact(&mut bytes).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }
    fn stop(&mut self) {
        assert_eq!(self.request(json!({"op":"shutdown"}))["ok"], true);
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = self.process.try_wait().unwrap() {
                assert!(status.success());
                assert_eq!(
                    std::fs::read_dir(self.directory.path().join("scratch"))
                        .unwrap()
                        .count(),
                    0,
                    "bridge must remove its temporary workspace on shutdown"
                );
                break;
            }
            assert!(Instant::now() < deadline, "bridge shutdown timeout");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
impl Drop for Bridge {
    fn drop(&mut self) {
        if self.process.try_wait().ok().flatten().is_none() {
            let _ = self.process.kill();
            let _ = self.process.wait();
        }
    }
}
#[test]
fn staged_bulk_build_publishes_and_reports_real_build_phases() {
    let mut bridge = Bridge::start();
    assert_eq!(
        bridge.request(json!({"op":"reset","dimension":4,"metric":"L2","refinement_rounds":2,"refinement_memory_bytes":33554432,"dataset":"bulk-test"}))["ok"],
        true
    );
    for start in [0, 50, 100, 150] {
        let ids: Vec<_> = (start..start + 50).collect();
        let vectors: Vec<_> = ids.iter().map(|id| vec![*id as f32, 1., 2., 3.]).collect();
        let response = bridge.insert(&ids, &vectors);
        assert_eq!(response["ok"], true, "{response}");
    }
    assert_eq!(
        bridge.request(json!({"op":"search","vector":[7.,1.,2.,3.],"k":1}))["ok"],
        false
    );
    assert_eq!(
        bridge.request(json!({"op":"optimize","records":200}))["ok"],
        true
    );
    assert_eq!(
        bridge.request(json!({"op":"optimize","records":200}))["ok"],
        true
    );
    let response = bridge.request(json!({"op":"search","vector":[7.,1.,2.,3.],"k":1}));
    assert_eq!(response["ok"], true);
    assert_eq!(response["result"]["ids"], json!([7]));
    assert_eq!(bridge.insert(&[201], &[vec![1., 2., 3., 4.]])["ok"], false);
    bridge.stop();
    let report: Value = serde_json::from_slice(
        &std::fs::read(bridge.directory.path().join("report.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["build_mode"], "bulk");
    assert_eq!(report["ready"], true);
    assert_eq!(report["bulk_build"]["refinement_rounds"], 2);
    assert_eq!(report["bulk_build"]["refinement_memory_bytes"], 33554432);
    for phase in [
        "snapshot_seconds",
        "prepare_load_seconds",
        "validate_publish_cleanup_seconds",
    ] {
        assert!(report["bulk_build"][phase].as_f64().unwrap() > 0.);
    }
}
#[test]
fn duplicate_staged_ids_fail_before_publication() {
    let mut bridge = Bridge::start();
    assert_eq!(
        bridge.request(json!({"op":"reset","dimension":4,"metric":"L2","refinement_rounds":2,"refinement_memory_bytes":33554432,"dataset":"duplicate"}))["ok"],
        true
    );
    let response = bridge.insert(&[1, 1], &[vec![1., 2., 3., 4.], vec![2., 3., 4., 5.]]);
    assert_eq!(response["ok"], true, "{response}");
    assert_eq!(
        bridge.request(json!({"op":"optimize","records":2}))["ok"],
        false
    );
    assert_eq!(
        bridge.request(json!({"op":"search","vector":[1.,2.,3.,4.],"k":1}))["ok"],
        false
    );
    bridge.stop();
}

#[test]
fn refinement_limit_failure_stays_terminal_and_keeps_search_closed() {
    let mut bridge = Bridge::start();
    assert_eq!(
        bridge.request(
            json!({"op":"reset","dimension":4,"metric":"L2","dataset":"limit",
            "refinement_rounds":2,"refinement_memory_bytes":1})
        )["ok"],
        true
    );
    assert_eq!(bridge.insert(&[1], &[vec![1., 2., 3., 4.]])["ok"], true);
    for _ in 0..2 {
        let response = bridge.request(json!({"op":"optimize","records":1}));
        assert_eq!(response["ok"], false);
        assert!(
            response["error"]["message"]
                .as_str()
                .unwrap()
                .contains("LimitExceeded")
        );
        assert_eq!(
            bridge.request(json!({"op":"search","vector":[1.,2.,3.,4.],"k":1}))["ok"],
            false
        );
    }
    bridge.stop();
    let report: Value = serde_json::from_slice(
        &std::fs::read(bridge.directory.path().join("report.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["ready"], false);
    assert!(report["bulk_build"].is_object());
    assert_eq!(report["bulk_build"]["refinement_memory_bytes"], 1);
    assert!(report["bulk_build"]["forest_seconds"].is_null());
    assert!(report["phases"]["optimize_seconds"].as_f64().unwrap() > 0.);
}

#[test]
fn fixed_bridge_routes_after_artifact_refinement() {
    for mode in ["partition-local-hnsw", "global-hnsw"] {
        let mut bridge = Bridge::start_with(Some(
            json!({"mode":mode, "m":8, "ef_construction":40, "graph_seed":9, "ef_search":16, "build_memory_bytes":33554432}),
        ));
        let reset = bridge.request(json!({"op":"reset","dimension":4,"metric":"L2","refinement_rounds":1,
            "refinement_memory_bytes":8388608,"dataset":"fixed", "min_leaf_partition_entries":4,
            "max_leaf_partition_entries":16,"min_internal_partition_entries":4,"max_internal_partition_entries":16,"leaf_beam":4}));
        assert_eq!(reset["ok"], true, "{reset}");
        for start in [0, 50, 100, 150] {
            let ids: Vec<_> = (start..start + 50).collect();
            let vectors: Vec<_> = ids.iter().map(|id| vec![*id as f32, 1., 2., 3.]).collect();
            assert_eq!(bridge.insert(&ids, &vectors)["ok"], true);
        }
        let optimize = bridge.request(json!({"op":"optimize","records":200}));
        assert_eq!(optimize["ok"], true, "{optimize}");
        let result = bridge.request(json!({"op":"search","vector":[7.,1.,2.,3.],"k":1}));
        assert_eq!(result["result"]["ids"], json!([7]), "{result}");
        bridge.stop();
        let report: Value = serde_json::from_slice(
            &std::fs::read(bridge.directory.path().join("report.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report["configuration"]["routing_experiment"]["mode"], mode);
        assert_eq!(report["bulk_build"]["refinement_memory_bytes"], 8388608);
        for field in [
            "construction_memory_bytes",
            "sort_memory_bytes",
            "serving_memory_bytes",
            "graph_memory_bytes",
        ] {
            assert_eq!(report["bulk_build"][field], 33554432, "{field}: {report}");
        }
    }
}

#[test]
fn global_router_failure_keeps_the_published_index_closed_to_search() {
    let mut bridge = Bridge::start_with(Some(json!({
        "mode":"global-hnsw", "global_working_bytes":1,
    })));
    assert_eq!(
        bridge.request(json!({
            "op":"reset", "dimension":4, "metric":"L2", "dataset":"router-limit",
            "refinement_rounds":0, "refinement_memory_bytes":33554432,
        }))["ok"],
        true
    );
    assert_eq!(bridge.insert(&[1], &[vec![1., 2., 3., 4.]])["ok"], true);
    for _ in 0..2 {
        let result = bridge.request(json!({"op":"optimize", "records":1}));
        assert_eq!(result["ok"], false, "{result}");
        assert!(
            result["error"]["message"]
                .as_str()
                .unwrap()
                .contains("LimitExceeded")
        );
        assert_eq!(
            bridge.request(json!({"op":"health"}))["result"]["ready"],
            false
        );
        assert_eq!(
            bridge.request(json!({"op":"search", "vector":[1.,2.,3.,4.], "k":1}))["ok"],
            false
        );
    }
    bridge.stop();
    let report: Value = serde_json::from_slice(
        &std::fs::read(bridge.directory.path().join("report.json")).unwrap(),
    )
    .unwrap();
    assert!(
        report["bulk_build"]["validate_publish_cleanup_seconds"]
            .as_f64()
            .unwrap()
            > 0.
    );
    assert_eq!(report["ready"], false);
}

#[test]
fn invalid_refinement_controls_do_not_consume_reset() {
    let mut bridge = Bridge::start();
    for (rounds, memory) in [(6, 33554432), (0, 0)] {
        assert_eq!(
            bridge.request(json!({
                "op":"reset", "dimension":4, "metric":"L2", "dataset":"invalid-controls",
                "refinement_rounds":rounds, "refinement_memory_bytes":memory,
            }))["ok"],
            false
        );
    }
    assert_eq!(
        bridge.request(json!({
            "op":"reset", "dimension":4, "metric":"L2", "dataset":"valid-controls",
            "refinement_rounds":0, "refinement_memory_bytes":33554432,
        }))["ok"],
        true
    );
    bridge.stop();
}
