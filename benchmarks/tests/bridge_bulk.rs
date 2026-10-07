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
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("bridge.sock");
        let mut process = Command::new(env!("CARGO_BIN_EXE_ktann-vdbbench-bridge"))
            .arg("--backend")
            .arg("rocksdb")
            .arg("--socket")
            .arg(&socket)
            .arg("--database")
            .arg(directory.path().join("database"))
            .arg("--report")
            .arg(directory.path().join("report.json"))
            .arg("--bulk-workspace")
            .arg(directory.path().join("bulk"))
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
        let data = serde_json::to_vec(&request).unwrap();
        self.stream
            .write_all(&(data.len() as u32).to_be_bytes())
            .unwrap();
        self.stream.write_all(&data).unwrap();
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
        bridge.request(json!({"op":"reset","dimension":4,"metric":"L2","dataset":"bulk-test"}))["ok"],
        true
    );
    for start in [0, 50, 100, 150] {
        let ids: Vec<_> = (start..start + 50).collect();
        let vectors: Vec<_> = ids.iter().map(|id| vec![*id as f32, 1., 2., 3.]).collect();
        assert_eq!(
            bridge.request(json!({"op":"insert","ids":ids,"vectors":vectors}))["ok"],
            true
        );
    }
    assert_eq!(
        bridge.request(json!({"op":"search","vector":[7.,1.,2.,3.],"k":1}))["ok"],
        false
    );
    assert!(!bridge.directory.path().join("bulk/source").exists());
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
    assert_eq!(
        bridge.request(json!({"op":"insert","ids":[201],"vectors":[[1.,2.,3.,4.]]}))["ok"],
        false
    );
    bridge.stop();
    let report: Value = serde_json::from_slice(
        &std::fs::read(bridge.directory.path().join("report.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(report["build_mode"], "bulk");
    assert_eq!(report["ready"], true);
    assert!(report["phases"]["committed_import_seconds"].is_null());
    for phase in [
        "snapshot_seconds",
        "prepare_load_seconds",
        "validate_publish_cleanup_seconds",
    ] {
        assert!(report["bulk_build"][phase].as_f64().unwrap() > 0.);
    }
    assert!(!bridge.directory.path().join("bulk/input.bin").exists());
}
#[test]
fn duplicate_staged_ids_fail_before_publication() {
    let mut bridge = Bridge::start();
    assert_eq!(
        bridge.request(json!({"op":"reset","dimension":4,"metric":"L2","dataset":"duplicate"}))["ok"],
        true
    );
    assert_eq!(
        bridge.request(json!({"op":"insert","ids":[1,1],"vectors":[[1.,2.,3.,4.],[2.,3.,4.,5.]]}))
            ["ok"],
        true
    );
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
