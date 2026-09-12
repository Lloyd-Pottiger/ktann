# Balanced training validation

Balanced split training selects the median membership boundary instead of fully
sorting distance differences. The comparator retains canonical ID tie-breaking.
When membership is unchanged, training reuses the previous centroids while still
counting the confirming round. Numeric accumulation order and persistent results
are unchanged.

The independent full-sort/recompute oracle compares centroid bits, round counts,
and errors across all three metrics, dimensions 1/3/16, empty and singleton
inputs, odd/even sizes, ties, duplicate vectors, and extreme finite values.
Existing protocol tests and storage uniqueness/conflict regression tests remain.

## Release microbenchmark

Run on an otherwise idle host:

```sh
cargo test --release -p ktann --lib training_release_comparison -- --ignored --nocapture
```

The comparison uses production `Bytes` IDs and measures 20 training calls per
sample, including identical input cloning on both sides. Six alternating
reference/optimized pairs cover each shape. The reference retains full sorting
and unconditional centroid recomputation. The benchmark is ignored in normal
test runs; it asserts successful training without imposing a timing SLA.

Results below are median milliseconds per sample on Apple M1 Pro, Rust 1.85,
release mode. These measure the training kernel, not whole-system import or
search latency. No FoundationDB RPC reduction or backend-wide speedup is claimed.

| Dimensions | Entries | Reference ms | Optimized ms | Change |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 128 | 0.550 | 0.476 | -13.4% |
| 16 | 1024 | 5.388 | 4.438 | -17.6% |
| 128 | 128 | 4.836 | 4.713 | -2.6% |
| 128 | 1024 | 29.489 | 28.568 | -3.1% |
| 768 | 128 | 22.649 | 21.612 | -4.6% |
| 768 | 1024 | 182.336 | 172.612 | -5.3% |

Measured on 2026-09-12. The raw log is retained locally at
`.benchmark-data/results/issue-162/training-only-final.log` in the primary checkout.
