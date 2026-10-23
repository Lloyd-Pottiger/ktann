//! Shared adapter graph-update history with deterministic commit failures.
#[path = "support/bulk_load_adapter.rs"]
mod bulk_load_adapter;

use bulk_load_adapter::graph_history;

use ktann_memory::{MemoryBackend, test_support::CommitFault};

fn backend() -> MemoryBackend {
    let mut config = ktann_memory::test_support::TestConfig::default();
    config.hard_limits.max_key_bytes = 10_000;
    config.hard_limits.max_value_bytes = 100_000;
    MemoryBackend::with_test_config(config)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_graph_updates_and_commit_recovery() {
    let root = std::env::temp_dir().join(format!(
        "ktann-graph-updates-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let backend = backend();
    bulk_load_adapter::exercise(|| backend.clone(), &root).await;
    for (name, fault) in [
        ("abort", CommitFault::Abort),
        ("unknown-applied", CommitFault::UnknownApplied),
        ("unknown-not-applied", CommitFault::UnknownNotApplied),
    ] {
        let backend = self::backend();
        bulk_load_adapter::exercise_updates(
            &|| backend.clone(),
            &root.join(name),
            &|_, attempt| {
                if attempt == 0 {
                    backend.push_fault(fault).unwrap();
                }
            },
        )
        .await;
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "release-mode 2D history phase cost probe; set KTANN_GRAPH_UPDATE_TIMING=1"]
async fn graph_update_cost_probe() {
    let root = std::env::temp_dir().join(format!("ktann-graph-update-cost-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    for repeat in 0..3 {
        let backend = backend();
        bulk_load_adapter::exercise_updates(
            &|| backend.clone(),
            &root.join(repeat.to_string()),
            &|_, _| {},
        )
        .await;
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graph_topology_commit_faults() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownApplied,
        CommitFault::UnknownNotApplied,
    ] {
        let backend = self::backend();
        graph_history::exercise(|| backend.clone(), &|phase, attempt| {
            if phase == "topology-finished" {
                backend.set_fault_plan(Vec::new()).unwrap();
            } else if attempt == 0 {
                backend.push_fault(fault).unwrap();
            }
        })
        .await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_graph_history() {
    let backend = self::backend();
    graph_history::exercise(|| backend.clone(), &|_, _| {}).await;
}
