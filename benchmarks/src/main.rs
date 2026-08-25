//! Command-line entry point for KTANN performance baselines.

use std::process::ExitCode;

/// Maps the library command result to a conventional process exit status.
fn main() -> ExitCode {
    match ktann_benchmarks::cli::run(std::env::args_os()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ktann-bench: {error}");
            ExitCode::FAILURE
        }
    }
}
