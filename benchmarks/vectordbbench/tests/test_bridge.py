"""Real bridge + pinned upstream classes; run with the installed checkout on PYTHONPATH."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.ktann.config import KTANNCaseConfig, KTANNConfig
from vectordb_bench.backend.clients.ktann.ktann import BridgeError, Connection, KTANN, MAX_FRAME


def worker(client, operation, queue):
    try:
        with client.init():
            if operation == "load":
                vectors = [[float(i), 1., 2., 3.] for i in range(200)]
                count, error = client.insert_embeddings(vectors, list(range(200)))
                if error:
                    raise error
                queue.put(count)
            elif operation == "optimize":
                client.optimize(200)
                queue.put("ready")
            else:
                queue.put([client.search_embedding([float(i), 1., 2., 3.], 1) for i in range(10)])
    except BaseException as error:
        queue.put((type(error).__name__, str(error)))


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="kvdb-", dir="/tmp")
        self.root = Path(self.directory.name)
        self.socket = str(self.root / "b.sock")
        self.process = None
        self.backend = os.environ.get("KTANN_TEST_BACKEND", "rocksdb")
        self.database = str(self.root / "db") if self.backend == "rocksdb" else "ktann-vdbbench-test-" + self.root.name
        self.addCleanup(self.cleanup)
        self.start()

    def start(self):
        self.log = (self.root / "stderr.log").open("a")
        self.process = subprocess.Popen([os.environ["KTANN_BRIDGE_BIN"], "--backend", self.backend,
                                        "--socket", self.socket, "--database", self.database,
                                        "--report", str(self.root / "bridge.json")], stderr=self.log)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.fail((self.root / "stderr.log").read_text())
            try:
                connection = Connection(self.socket)
                connection.socket.settimeout(2)
                try:
                    connection.request("health")
                finally:
                    connection.close()
                return
            except (OSError, BridgeError):
                time.sleep(.05)
        self.fail("bridge did not become healthy")

    def stop(self):
        connection = Connection(self.socket)
        connection.request("shutdown")
        connection.close()
        self.assertEqual(self.process.wait(timeout=30), 0)
        self.log.close()
        self.assertFalse(Path(self.socket).exists())
        self.process = None

    def cleanup(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.log.close()
        self.directory.cleanup()

    def client(self, metric=MetricType.L2):
        return KTANN(4, {"socket_path": self.socket, "dataset": "synthetic-test-200",
                         "companion_dir": str(self.root / "clients")}, KTANNCaseConfig(metric_type=metric), drop_old=True)

    def spawn(self, client, operation, count=1):
        context = mp.get_context("spawn")
        queue = context.Queue()
        processes = [context.Process(target=worker, args=(client, operation, queue)) for _ in range(count)]
        for process in processes:
            process.start()
        results = [queue.get(timeout=90) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
        queue.close()
        queue.join_thread()
        return results

    def test_spawn_handoff_concurrent_search_and_restart(self):
        self.assertIs(DB.KTANN.init_cls, KTANN)
        self.assertIs(DB.KTANN.config_cls, KTANNConfig)
        client = self.client()
        self.assertIsNone(pickle.loads(pickle.dumps(client))._connection)
        with client.init():
            self.assertIsNone(pickle.loads(pickle.dumps(client))._connection)
        self.assertEqual(self.spawn(client, "load"), [200])
        self.assertEqual(self.spawn(client, "optimize"), ["ready"])
        self.assertEqual(self.spawn(client, "search", 4), [[[i] for i in range(10)]] * 4)
        self.stop()
        report = json.loads((self.root / "bridge.json").read_text())
        self.assertTrue(report["ready"])
        self.assertEqual(report["records"], 200)
        self.assertEqual(report["searches"], 40)
        self.assertGreater(report["continuous_first_insert_through_final_search_seconds"], 0)
        self.assertEqual(sum(json.loads(p.read_text())["search_calls"] > 0 for p in (self.root / "clients").glob("client-*.json")), 4)
        self.start()  # proves Runtime shutdown released native RocksDB ownership
        self.client()
        self.stop()

    def test_boundaries_and_error_mapping(self):
        client = self.client(MetricType.COSINE)
        with client.init():
            with self.assertRaises(BridgeError):
                client.search_embedding([1., 0., 0., 0.])
            count, error = client.insert_embeddings([[1., 0., 0., 0.]], [-(1 << 63)])
            self.assertEqual((count, error), (1, None))
            count, error = client.insert_embeddings([[1., 0., 0., 0.]], [-(1 << 63)])
            self.assertEqual(count, 0)
            self.assertTrue(error.non_retryable)
            count, error = client.insert_embeddings([[1., 2.]], [3])
            self.assertEqual(count, 0)
            self.assertTrue(error.non_retryable)
            with self.assertRaises(BridgeError):
                client.optimize(2)
            client.optimize(1)
            self.assertEqual(client.search_embedding([1., 0., 0., 0.], 1), [-(1 << 63)])
        for length in (0, MAX_FRAME + 1):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
                raw.settimeout(5)
                raw.connect(self.socket)
                raw.sendall(struct.pack("!I", length))
                self.assertEqual(raw.recv(1), b"")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
            raw.connect(self.socket)
            data = b'{"version":999,"op":"health"}'
            raw.sendall(struct.pack("!I", len(data)) + data)
            length = struct.unpack("!I", raw.recv(4))[0]
            response = json.loads(raw.recv(length))
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["kind"], "protocol")

    def test_canonical_cli_retains_metrics_and_all_concurrency_results(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        data = self.root / "data"
        data.mkdir()
        vectors = np.array([[float(i), 1., 2., 3.] for i in range(200)], dtype=np.float32)
        queries = vectors[:10]
        truth = np.argsort(((queries[:, None, :] - vectors[None, :, :]) ** 2).sum(axis=2), axis=1, kind="stable")[:, :100]
        pq.write_table(pa.table({"id": list(range(200)), "emb": vectors.tolist()}), data / "train.parquet")
        pq.write_table(pa.table({"id": list(range(10)), "emb": queries.tolist()}), data / "test.parquet")
        pq.write_table(pa.table({"id": list(range(10)), "neighbors_id": truth.tolist()}), data / "neighbors.parquet")
        env = {**os.environ, "RESULTS_LOCAL_DIR": str(self.root / "canonical"),
               "LOG_FILE": str(self.root / "cli.log"), "IR_DATASETS_HOME": str(self.root / "ir"),
               "MPLCONFIGDIR": str(self.root / "mpl")}
        command = [str(Path(sys.executable).parent / "vectordbbench"), "ktann", "--socket-path", self.socket,
                   "--dataset-identity", "synthetic-test-200", "--companion-dir", str(self.root / "clients"),
                   "--case-type", "PerformanceCustomDataset", "--custom-case-name", "bridge-test",
                   "--custom-dataset-name", "bridge-test", "--custom-dataset-dir", str(data),
                   "--custom-dataset-size", "200", "--custom-dataset-dim", "4",
                   "--custom-dataset-metric-type", "L2", "--custom-dataset-file-count", "1",
                   "--load-concurrency", "1", "--k", "10", "--num-concurrency", "1,2", "--concurrency-duration", "1"]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        files = list((self.root / "canonical").rglob("result_*.json"))
        self.assertEqual(len(files), 1, result.stderr)
        case, = json.loads(files[0].read_text())["results"]
        self.assertEqual(case["label"], ":)", result.stderr)
        metrics = case["metrics"]
        self.assertAlmostEqual(metrics["load_duration"], metrics["insert_duration"] + metrics["optimize_duration"], delta=.0002)
        self.assertEqual(metrics["conc_num_list"], [1, 2])
        self.assertEqual(metrics["qps"], max(metrics["conc_qps_list"]))
        for field in ("conc_latency_p50_list", "conc_latency_p95_list", "conc_latency_p99_list"):
            self.assertEqual(len(metrics[field]), 2)
        self.assertGreaterEqual(metrics["serial_latency_p99"], metrics["serial_latency_p95"])
        self.assertGreaterEqual(metrics["serial_latency_p95"], metrics["serial_latency_p50"])
        self.assertGreater(metrics["recall"], .9)

    def test_partial_frame_shutdown_and_crash_restart(self):
        client = self.client()
        with client.init():
            self.assertEqual(client.insert_embeddings([[1., 2., 3., 4.]], [7]), (1, None))
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as incomplete:
            incomplete.settimeout(5)
            incomplete.connect(self.socket)
            incomplete.sendall(struct.pack("!I", 100))  # body deliberately never arrives
            self.stop()
            self.assertEqual(incomplete.recv(1), b"")
        self.start()
        self.client()
        self.process.kill()
        self.process.wait(timeout=10)
        self.log.close()
        self.process = None
        self.assertTrue(Path(self.socket).exists())
        # Explicit recovery removes only this test's socket after process death.
        Path(self.socket).unlink()
        self.start()
        self.client()
        self.stop()

    def test_live_socket_is_not_unlinked(self):
        second = subprocess.run([os.environ["KTANN_BRIDGE_BIN"], "--backend", self.backend,
                                 "--socket", self.socket, "--database", self.database,
                                 "--report", str(self.root / "other.json")], capture_output=True, timeout=30)
        self.assertNotEqual(second.returncode, 0)
        connection = Connection(self.socket)
        self.assertFalse(connection.request("health")["ready"])
        connection.close()
