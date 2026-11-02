//! Builds the pinned graph-only USearch bridge without fast-math.

fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("native/graph.cpp")
        .include(".")
        .include("vendor")
        .std("c++17")
        .flag_if_supported("-ffp-contract=off")
        .flag_if_supported("/fp:strict")
        .compile("ktann_usearch");
    for path in [
        "src/lib.rs",
        "native/graph.h",
        "native/graph.cpp",
        "vendor/usearch/index.hpp",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
}
