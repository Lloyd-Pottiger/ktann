//! Private VectorDBBench bridge executable.

fn main() -> std::process::ExitCode {
    match ktann_benchmarks::bridge::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ktann-vdbbench-bridge: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
