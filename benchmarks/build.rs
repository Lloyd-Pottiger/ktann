//! Captures build identity that cannot be reconstructed from the running host.

use std::process::Command;

/// Exposes compiler and code-generation identity to the runtime report.
///
/// These values must be captured here because the finished executable cannot
/// reliably reconstruct the Cargo profile or flags that produced it.
fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let rustc_identity = Command::new(&rustc)
        .arg("--version")
        .arg("--verbose")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().replace('\n', " | "))
        .unwrap_or_else(|| "unavailable".to_owned());
    println!("cargo:rustc-env=KTANN_BENCH_BUILD_RUSTC={rustc_identity}");

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unavailable".to_owned());
    println!("cargo:rustc-env=KTANN_BENCH_BUILD_PROFILE={profile}");

    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .map(|flags| flags.replace('\u{1f}', " "))
        .or_else(|_| std::env::var("RUSTFLAGS"))
        .unwrap_or_default();
    println!("cargo:rustc-env=KTANN_BENCH_RUSTFLAGS={rustflags}");
}
