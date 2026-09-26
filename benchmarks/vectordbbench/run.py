#!/usr/bin/env python3
"""One optimized bridge and one unmodified canonical VectorDBBench run."""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

from companion import combine
from datasets import digest, prepare
from install import REVISION


def run(args):
    if subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip() != REVISION:
        raise ValueError("unsupported VectorDBBench revision")
    for field in ("checkout", "bridge", "cache", "sift", "output"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    if args.case == "cohere-1m":
        provenance = prepare(args.case, args.cache, args.output)
    else:
        if args.sift is None:
            raise ValueError("--sift is required for SIFT1M")
        provenance = json.loads((args.sift / "provenance.json").read_text())
    (args.output / "dataset.json").write_text(json.dumps(provenance, indent=2))
    env = {**os.environ, "DATASET_LOCAL_DIR": str(args.cache), "RESULTS_LOCAL_DIR": str(args.output / "canonical"),
           "LOG_FILE": str(args.output / "runner.log"), "IR_DATASETS_HOME": str(args.output / "ir-datasets"),
           "MPLCONFIGDIR": str(args.output / "matplotlib")}
    os.environ.update({key: env[key] for key in ("LOG_FILE", "IR_DATASETS_HOME", "MPLCONFIGDIR")})
    from vectordb_bench.backend.clients.ktann.ktann import Connection
    with tempfile.TemporaryDirectory(prefix="kvdb-", dir="/tmp") as temporary:
        socket = str(Path(temporary) / "bridge.sock")
        database = str(args.output / "rocksdb") if args.backend == "rocksdb" else f"ktann-vdbbench-{Path(temporary).name}"
        command = [str(args.bridge), "--backend", args.backend, "--socket", socket, "--database", database,
                   "--report", str(args.output / "bridge.json")]
        cli = [str(Path(sys.executable).parent / "vectordbbench"), "ktann", "--socket-path", socket,
               "--dataset-identity", args.case, "--companion-dir", str(args.output / "clients"),
               "--db-label", f"{args.backend}-bridge-v1", "--task-label", "KTANN plus benchmark bridge",
               "--load-concurrency", "1", "--insert-batch-size", "50", "--k", "100",
               "--num-concurrency", args.concurrency,
               "--concurrency-duration", str(args.duration)]
        if args.leaf_beam is not None:
            cli += ["--leaf-beam", str(args.leaf_beam)]
        if args.case == "cohere-1m":
            cli += ["--case-type", "Performance768D1M"]
        else:
            for name, checksum in provenance["converted_sha256"].items():
                if digest(args.sift / name) != checksum:
                    raise ValueError(f"converted dataset checksum mismatch: {name}")
            cli += ["--case-type", "PerformanceCustomDataset", "--custom-case-name", "SIFT1M L2",
                    "--custom-dataset-name", "sift1m", "--custom-dataset-dir", str(args.sift),
                    "--custom-dataset-size", "1000000", "--custom-dataset-dim", "128",
                    "--custom-dataset-metric-type", "L2", "--custom-dataset-file-count", "1"]
        (args.output / "invocation.json").write_text(json.dumps({"label": "KTANN plus benchmark bridge", "upstream_revision": REVISION,
            "bridge_sha256": digest(args.bridge), "bridge_command": command, "canonical_command": cli, "host": platform.platform(), "cpu_count": os.cpu_count(),
            "python": sys.version, "ktann_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}, indent=2))
        with (args.output / "bridge.log").open("w") as log:
            bridge = subprocess.Popen(command, env=env, stderr=log)
            try:
                deadline = time.monotonic() + 30
                while True:
                    if bridge.poll() is not None:
                        raise RuntimeError("bridge exited during startup; see bridge.log")
                    try:
                        connection = Connection(socket)
                        try:
                            connection.request("health")
                        finally:
                            connection.close()
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("bridge startup")
                        time.sleep(.05)
                subprocess.run(cli, env=env, check=True)
            finally:
                if bridge.poll() is None:
                    bridge.send_signal(signal.SIGINT)
                    try:
                        bridge.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        bridge.kill()
                        bridge.wait()
                        raise TimeoutError("bridge cleanup did not complete")
                (args.output / "client-summary.json").write_text(json.dumps(combine(args.output / "clients"), indent=2))
            if bridge.returncode:
                raise RuntimeError("bridge failed; see bridge.log")
            canonical_files = sorted((args.output / "canonical").rglob("result_*.json"))
            cases = [case for path in canonical_files for case in json.loads(path.read_text())["results"]]
            if len(cases) != 1 or cases[0]["label"] != ":)":
                raise RuntimeError("canonical runner did not report one successful case")
            metrics = cases[0]["metrics"]
            expected = [int(value) for value in args.concurrency.split(",")]
            if metrics["conc_num_list"] != expected or len(metrics["conc_qps_list"]) != len(expected):
                raise RuntimeError("canonical runner did not retain every requested concurrency")
            if metrics["qps"] != max(metrics["conc_qps_list"]):
                raise RuntimeError("canonical headline QPS differs from per-concurrency maximum")
            # These p50 values are already canonical in the pinned upstream revision.
            # Copy them by name; never recompute or replace upstream metric fields.
            (args.output / "canonical-p50.json").write_text(json.dumps({
                "source_files": [str(p.relative_to(args.output)) for p in canonical_files],
                "serial_latency_p50": metrics["serial_latency_p50"],
                "conc_num_list": metrics["conc_num_list"],
                "conc_latency_p50_list": metrics["conc_latency_p50_list"]}, indent=2))
            report = json.loads((args.output / "bridge.json").read_text())
            if not report["ready"] or report["records"] != 1000000 or report["searches"] == 0:
                raise RuntimeError("canonical run did not finish load/optimize/search; inspect canonical results and runner.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sift", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["rocksdb", "foundationdb"], required=True)
    parser.add_argument("--case", choices=["cohere-1m", "sift-1m"], required=True)
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--leaf-beam", type=int)
    run(parser.parse_args())
