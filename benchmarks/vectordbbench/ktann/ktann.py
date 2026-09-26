"""Thin pickleable adapter; all database ownership stays in the Rust process."""
import json
import math
import os
import operator
from pathlib import Path
import socket
import struct
import threading
import time
import uuid
from contextlib import contextmanager

from ..api import PartialInsertError, VectorDB
from ...filter import FilterOp
from ...payload import PayloadProfile

VERSION = 1
MAX_FRAME = 8 << 20
MAX_BATCH = 50


class BridgeError(RuntimeError):
    """A terminal wire/KTANN error. Unknown inserts must never be auto-replayed."""


class Connection:
    """One sequential connection, opened inside the current worker only."""
    def __init__(self, path):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(3605)
        try:
            self.socket.connect(path)
        except BaseException:
            self.socket.close()
            raise
        self.lock = threading.Lock()
        self.calls = 0
        self.elapsed = self.encode = self.decode = self.ktann = 0.0
        self.latency_buckets = [0] * 64
        self.first_insert_ns = self.last_search_ns = None

    def close(self):
        self.socket.close()

    def _read(self, size):
        result = bytearray()
        while len(result) < size:
            chunk = self.socket.recv(size - len(result))
            if not chunk:
                raise BridgeError("bridge disconnected; insert outcome may be unknown")
            result.extend(chunk)
        return result

    def request(self, op, **fields):
        if op == "insert" and self.first_insert_ns is None:
            self.first_insert_ns = time.monotonic_ns()
        started = time.perf_counter()
        data = json.dumps({"version": VERSION, "op": op, **fields}, separators=(",", ":"), allow_nan=False).encode()
        encoded = time.perf_counter()
        if len(data) > MAX_FRAME:
            raise ValueError("bridge frame exceeds 8 MiB")
        with self.lock:
            self.socket.sendall(struct.pack("!I", len(data)) + data)
            size, = struct.unpack("!I", self._read(4))
            if not 0 < size <= MAX_FRAME:
                raise BridgeError("invalid response length")
            data = self._read(size)
            decode_started = time.perf_counter()
            response = json.loads(data)
            finished = time.perf_counter()
            if response.get("version") != VERSION:
                raise BridgeError("bridge version mismatch")
            if not response.get("ok"):
                raise BridgeError(str(response.get("error")))
            result = response["result"]
            if op == "search":
                self.last_search_ns = time.monotonic_ns()
                self.calls += 1
                self.elapsed += finished - started
                self.encode += encoded - started
                self.decode += finished - decode_started
                self.ktann += result["ktann_seconds"]
                bucket = min(63, max(0, int((finished - started) * 1e9).bit_length()))
                self.latency_buckets[bucket] += 1
            return result

    def report(self):
        return {"label": "KTANN plus benchmark bridge — client companion", "protocol_version": VERSION,
                "first_insert_monotonic_ns": self.first_insert_ns, "last_search_monotonic_ns": self.last_search_ns,
                "pid": os.getpid(), "search_calls": self.calls, "round_trip_seconds_sum": self.elapsed,
                "python_encode_seconds": self.encode, "python_decode_seconds": self.decode,
                "ktann_seconds_sum": self.ktann,
                "bridge_ipc_and_queue_seconds": max(0, self.elapsed - self.ktann),
                "latency_log2_nanoseconds_histogram": self.latency_buckets}


class KTANN(VectorDB):
    supported_filter_types = [FilterOp.NonFilter]

    def __init__(self, dim, db_config, db_case_config, collection_name="vdbbench", drop_old=False,
                 with_scalar_labels=False, **kwargs):
        if with_scalar_labels or kwargs.get("multitenant_tenant_labels"):
            raise ValueError("KTANN bridge supports unfiltered single-tenant IDs-only ANN")
        if not drop_old:
            raise ValueError("KTANN benchmark requires a fresh bridge and full load (drop_old=True)")
        self.name = "KTANN plus benchmark bridge"
        self.dim = dim
        self.db_config = dict(db_config)
        self._connection = None
        self._owner_pid = None
        metric = str(db_case_config.metric_type)
        if metric not in {"L2", "COSINE"}:
            raise ValueError("only L2 and COSINE are supported")
        connection = Connection(self.db_config["socket_path"])
        try:
            connection.request("reset", dimension=dim, metric=metric,
                               dataset=self.db_config["dataset"], **db_case_config.search_param())
        finally:
            connection.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_owner_pid"] = None
        return state

    @contextmanager
    def init(self):
        if self._connection is not None:
            raise RuntimeError("nested init is unsupported")
        self._connection = Connection(self.db_config["socket_path"])
        self._owner_pid = os.getpid()
        try:
            yield
        finally:
            connection, self._connection = self._connection, None
            self._owner_pid = None
            connection.close()
            if connection.calls or connection.first_insert_ns is not None:
                directory = Path(self.db_config["companion_dir"])
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"client-{os.getpid()}-{uuid.uuid4().hex}.json"
                target.write_text(json.dumps(connection.report(), indent=2))

    def _request(self, op, **fields):
        if self._connection is None or self._owner_pid != os.getpid():
            raise RuntimeError("use init() inside each worker process")
        return self._connection.request(op, **fields)

    def prepare_filter(self, filters):
        if filters.type != FilterOp.NonFilter:
            raise ValueError("filters are unsupported")

    def insert_embeddings(self, embeddings, metadata, labels_data=None, **kwargs):
        inserted = 0
        try:
            if labels_data is not None or kwargs.get("tenant") is not None:
                raise ValueError("labels and tenants are unsupported")
            if len(embeddings) != len(metadata):
                raise ValueError("IDs/vectors length mismatch")
            # Split at the client boundary; each wire batch is a bounded atomic import.
            for offset in range(0, len(metadata), MAX_BATCH):
                ids = [operator.index(i) for i in metadata[offset:offset + MAX_BATCH]]
                if any(not -(1 << 63) <= i < (1 << 63) for i in ids):
                    raise ValueError("IDs must be signed 64-bit integers")
                vectors = [[float(v) for v in row] for row in embeddings[offset:offset + MAX_BATCH]]
                if any(len(row) != self.dim or any(not math.isfinite(v) for v in row) for row in vectors):
                    raise ValueError("wrong dimension or nonfinite vector")
                result = self._request("insert", ids=ids, vectors=vectors)
                if result["inserted"] != len(ids):
                    raise BridgeError("incomplete insert response")
                inserted += result["inserted"]
            return inserted, None
        except Exception as error:
            # The upstream loader recognizes non_retryable, preserving known prefixes.
            return inserted, PartialInsertError(str(error), inserted_count=inserted, cause=error)

    def optimize(self, data_size=None):
        if data_size is None:
            raise ValueError("optimize requires the expected dataset size")
        self._request("optimize", records=int(data_size))

    def search_embedding(self, query, k=100, payload_profile=PayloadProfile.IDS_ONLY, tenant=None):
        if payload_profile != PayloadProfile.IDS_ONLY or tenant is not None:
            raise ValueError("only IDs-only, single-tenant search is supported")
        return self._request("search", vector=[float(v) for v in query], k=int(k))["ids"]
