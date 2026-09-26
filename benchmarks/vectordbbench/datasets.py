#!/usr/bin/env python3
"""Verify pinned inputs and expose SIFT1M as a normal upstream custom dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256", part_bytes=None):
    with path.open("rb") as stream:
        if algorithm == "sha256":
            return hashlib.file_digest(stream, "sha256").hexdigest()
        if algorithm != "s3_etag_md5":
            raise ValueError(f"unsupported checksum {algorithm}")
        parts = []
        while data := stream.read(part_bytes or 8 * 1024 * 1024):
            parts.append(hashlib.md5(data).digest())
        return parts[0].hex() if len(parts) == 1 else hashlib.md5(b"".join(parts)).hexdigest() + f"-{len(parts)}"


def prepare(case, cache, output, download=False):
    manifest_path = ROOT / "datasets" / f"{case}.json"
    manifest = json.loads(manifest_path.read_text())
    for file in manifest["files"]:
        path = cache / file["path"]
        if not path.exists() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".download")
            urllib.request.urlretrieve(file["url"], temporary)
            temporary.replace(path)
        if path.stat().st_size != file["bytes"]:
            raise ValueError(f"size mismatch: {path}")
        checksum = file["checksum"]
        if digest(path, checksum["algorithm"], checksum.get("part_bytes")) != checksum["value"]:
            raise ValueError(f"checksum mismatch: {path}")
    if case == "cohere-1m":
        return manifest
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=True)
    for file in manifest["files"]:
        role = file["role"]
        width = 100 if role == "ground_truth" else 128
        count = 1000000 if role == "base" else 10000
        words = np.memmap(cache / file["path"], dtype="<i4", mode="r").reshape(count, width + 1)
        if not np.all(words[:, 0] == width):
            raise ValueError("noncanonical vecs dimension prefix")
        values = words[:, 1:] if role == "ground_truth" else words[:, 1:].view("<f4")
        name = {"base": "train", "queries": "test", "ground_truth": "neighbors"}[role]
        field = "neighbors_id" if role == "ground_truth" else "emb"
        value_type = pa.int32() if role == "ground_truth" else pa.float32()
        schema = pa.schema([("id", pa.int64()), (field, pa.list_(value_type))])
        with pq.ParquetWriter(output / f"{name}.parquet", schema, compression="zstd") as writer:
            for offset in range(0, count, 10000):
                chunk = values[offset:offset + 10000]
                arrays = pa.ListArray.from_arrays(pa.array(np.arange(len(chunk) + 1) * width, type=pa.int32()),
                                                 pa.array(chunk.reshape(-1), type=value_type))
                writer.write_table(pa.Table.from_arrays([pa.array(np.arange(offset, offset + len(chunk))), arrays], schema=schema))
    manifest = {**manifest, "conversion": "fvecs/ivecs to Parquet; all 1M rows and 10K queries, original zero-based IDs and top-100 truth unchanged",
                "converted_sha256": {p.name: digest(p) for p in sorted(output.glob("*.parquet"))}}
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["cohere-1m", "sift-1m"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.case, args.cache, args.output, args.download)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2))
