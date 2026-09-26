#!/usr/bin/env python3
"""Combine bounded client histograms without modifying canonical runner results."""
import argparse
import json
from pathlib import Path


def combine(directory):
    reports = [json.loads(path.read_text()) for path in sorted(directory.glob("client-*.json"))]
    fields = ("search_calls", "round_trip_seconds_sum", "python_encode_seconds", "python_decode_seconds",
              "ktann_seconds_sum", "bridge_ipc_and_queue_seconds")
    totals = {field: sum(report[field] for report in reports) for field in fields}
    buckets = [sum(report["latency_log2_nanoseconds_histogram"][i] for report in reports) for i in range(64)]
    cumulative = 0
    p50 = None
    if totals["search_calls"]:
        for i, count in enumerate(buckets):
            cumulative += count
            if cumulative >= (totals["search_calls"] + 1) // 2:
                p50 = 2 ** i / 1e9
                break
    starts = [r["first_insert_monotonic_ns"] for r in reports if r["first_insert_monotonic_ns"] is not None]
    ends = [r["last_search_monotonic_ns"] for r in reports if r["last_search_monotonic_ns"] is not None]
    return {"continuous_first_insert_through_final_search_seconds": (max(ends) - min(starts)) / 1e9 if starts and ends else None,
            "label": "KTANN plus benchmark bridge — companion (all client contexts)",
            "protocol_version": 1, "client_contexts": len(reports), **totals,
            "search_round_trip_p50_upper_bound_seconds": p50,
            "latency_log2_nanoseconds_histogram": buckets,
            "overhead_definition": "client round trip minus KTANN search, including JSON, IPC, scheduling and lock/queue wait; not pure socket latency"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("client_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(combine(args.client_directory), indent=2))
