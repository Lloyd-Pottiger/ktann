//! Repeated mutable graph quality experiment.
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ktann_benchmarks::update_history::run().await
}
