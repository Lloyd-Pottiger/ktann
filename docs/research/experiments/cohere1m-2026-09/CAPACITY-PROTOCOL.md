# Cache capacity screen, predeclared before capacity runs

Same fixed1M tree and ground truth as the baseline. Change only the runtime
Partition Cache budget from512 MiB (B) to1 GiB (C), a bounded extra512 MiB.
Use one newly frozen executable for BOTH budgets. No core search changes or
decoded-representation candidate. Rust scenario configuration reads the explicit
budget and samples worker RSS after timing, while the warmed cache remains live.
This is post-point resident memory, not peak search memory or total macOS footprint.

Run B/C/C/B sequentially; beams64 and384, one thousand warmups and two thousand
measured queries per point, concurrency4/workers8. Both operating points must keep
identical recall, logical budgets and index hashes. Compare run-level QPS, latency
percentiles and CPU; inspect cache misses, backend IO and actual accounted bytes
for the mechanism. Budget1024 must be recorded in candidate configuration rather
than passed off as an implementation improvement at unchanged memory.

No cold-cache or FoundationDB benefit will be claimed. The user explicitly accepts
moderate memory growth and warm-cache-only gains. Keep the existing full six-point
baseline separate; the capacity screen is a paired follow-up at two fixed points.
Construction fields in reuse reports remain the original512 MiB fixture build.
