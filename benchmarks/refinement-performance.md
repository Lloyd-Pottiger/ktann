# Refinement locality comparison

Historical exact-selection measurements, before the
[hybrid HNSW candidate policy](refinement-hnsw-experiment.md). Running these
diagnostics on current code uses the hybrid policy for supported large trees.

Measured on 2026-11-07 with Rust 1.99.1, default release flags, on an x86-64
Intel Xeon host exposing 8 CPUs. The diagnostic runs serially on an otherwise
idle host. No additional threads or dependencies are introduced by the change.

## Comparison

Baseline: `86d948daec5740b633fe9b497b98bd2ac6989cfa` numerical implementation,
with the new `src/refinement/performance.rs` diagnostic compiled in.
Candidate: the same implementation with upper-triangle pair evaluation tiled
in 32-by-32 leaf blocks. The full kernel still evaluates every unordered pair
once; fixed-size sorted neighbor lists, distance reduction, tie-breaking,
capacity checks and move ordering are unchanged. The tiles reuse centroids and
neighbor lists without copying them or retaining a distance matrix.

Run the command in [Refinement CPU diagnostic](README.md#refinement-cpu-diagnostic)
on both revisions, preserving each binary before building the next. Baseline,
candidate, and baseline again were run in separate processes. Each case has
four repetitions, with the first discarded as warmup. The table uses the median
of the final three repetitions from the candidate and repeated baseline runs.
Input generation and normalization are excluded. Synthetic leaf planning has
4096 leaves, 131,072 vectors, and up to three refinement rounds. SIFT small uses
10,000 real 128-dimensional vectors, default capacities, construction sample
size 1024, seed `[7; 32]`, and three requested rounds.

| CPU workload | Baseline | Tiled | Time reduction |
| --- | ---: | ---: | ---: |
| Neighbor selection: 4096 leaves, 128 dimensions | 0.4760 s | 0.3901 s | 18.0% |
| Neighbor selection: 4096 leaves, 768 dimensions | 2.3630 s | 1.9493 s | 17.5% |
| Synthetic leaf planning: 128 dimensions | 2.0972 s | 1.7519 s | 16.5% |
| Synthetic leaf planning: 768 dimensions | 10.5359 s | 9.2819 s | 11.9% |
| SIFT small tree planning | 47.03 ms | 47.70 ms | No demonstrated gain |

Candidate synthetic-planning ranges were 1.740–1.763 s and 9.234–9.393 s;
repeated baseline ranges were 2.084–2.115 s and 10.506–10.674 s. The first
baseline process measured medians of 2.185 s and 10.407 s respectively, also
slower than the candidate. The 256-leaf neighbor case takes only about 2–3 ms;
its candidate range overlaps the baseline, so it does not demonstrate a gain.

All baseline/candidate outputs agree on neighbor digests, final membership and
centroid digests, and accepted move counts. Synthetic cases accept 18,667 moves
at 128 dimensions and 16,087 at 768 dimensions; SIFT small accepts 2,716.
The unit oracle additionally compares exact ordered neighbors for L2, cosine,
and inner product, including duplicates, ties, and tile/batch tails.

## Resource and scope limits

Neighbor storage remains 512 bytes per leaf on this 64-bit host. No extra heap
allocation, worker, IO, or persistent format change is needed. Cancellation is
checked between tiles. This improves locality, not asymptotic work: neighbor
selection is still quadratic in the number of leaves, and high-dimensional
vector scoring remains substantial.

These results demonstrate CPU-stage improvements on the larger measured cases,
not an end-to-end Bulk Build or query throughput improvement. Synthetic cases
exercise leaf planning without internal projections; SIFT includes the actual
constructed tree. Artifact IO, local graph construction, backend publication,
peak process RSS, and million-vector end-to-end runs were not measured here.
A fixed-size max-heap alternative for neighbor retention showed no stable
additional improvement and was discarded.
