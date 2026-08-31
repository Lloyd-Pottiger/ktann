# Large ANN dataset cache

The optimized `large` profile consumes fixed public datasets without checking
their multi-gigabyte files into KTANN. The versioned manifests under `v1/`
pin the source revision or S3 object version, exact byte length, checksum,
metric, shape, query count, and supplied exact-neighbor ground truth.

By default the runner reads `/tmp/vectordb_bench/dataset`, matching
VectorDBBench. Set `KTANN_BENCH_DATASET_CACHE` to use a persistent cache. The
relative Cohere paths are
the same as VectorDBBench, so an existing prepared cache is reused directly.

## Cohere 1M cosine

KTANN pins the dataset contract from
`Lloyd-Pottiger/VectorDBBench@08fabb2df7436bbaffb549f6d1144d66f14b9fb2`:
one million 768-dimensional base vectors, 1,000 held-out queries, and supplied
exact neighbors. VectorDBBench validates only remote/local file length; KTANN
also verifies the pinned S3 ETag checksum before decoding Parquet.

```sh
cache=${KTANN_BENCH_DATASET_CACHE:-/tmp/vectordb_bench/dataset}
mkdir -p "$cache/cohere/cohere_medium_1m"
curl --fail --location --continue-at - \
  --output "$cache/cohere/cohere_medium_1m/shuffle_train.parquet" \
  'https://assets.zilliz.com/benchmark/cohere_medium_1m/shuffle_train.parquet?versionId=9XIbR3Rr33QqVLWG97mWh73NjH01VZQ8'
curl --fail --location --continue-at - \
  --output "$cache/cohere/cohere_medium_1m/test.parquet" \
  'https://assets.zilliz.com/benchmark/cohere_medium_1m/test.parquet?versionId=a8Wf7zPE9YCKhiPw0nTF4jEhrGs7h.CC'
curl --fail --location --continue-at - \
  --output "$cache/cohere/cohere_medium_1m/neighbors.parquet" \
  'https://assets.zilliz.com/benchmark/cohere_medium_1m/neighbors.parquet?versionId=UCL_GDV5HL078p3o48ek7CZLMvCxHT1x'
```

## SIFT1M L2

VectorDBBench's `sift_small_500k` object set has train/test files but declares
`with_gt = false` and publishes no `neighbors.parquet`; it therefore cannot
satisfy issue #126's supplied-ground-truth requirement. The L2 curve uses the
original TexMex SIFT1M corpus mirrored at the fixed
`qbo-odp/sift1m@bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b` revision. The
mirror preserves the original `fvecs`/`ivecs` files and publishes their SHA-256
LFS object IDs.

```sh
cache=${KTANN_BENCH_DATASET_CACHE:-/tmp/vectordb_bench/dataset}
mkdir -p "$cache/sift1m"
root='https://huggingface.co/datasets/qbo-odp/sift1m/resolve/bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b'
curl --fail --location --continue-at - --output "$cache/sift1m/sift_base.fvecs" "$root/sift_base.fvecs"
curl --fail --location --continue-at - --output "$cache/sift1m/sift_query.fvecs" "$root/sift_query.fvecs"
curl --fail --location --continue-at - --output "$cache/sift1m/sift_groundtruth.ivecs" "$root/sift_groundtruth.ivecs"
```

The benchmark refuses missing, truncated, checksum-mismatched, malformed, or
shape-inconsistent files before creating a Backend. This validation is setup
work and is excluded from measured search intervals.
