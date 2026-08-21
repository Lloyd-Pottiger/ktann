# Real-dataset fixtures

Checked-in vector fixtures for the data-driven end-to-end corpus
(`tests/datadriven/realdata.kddt`) and the oracle ground-truth cross-check
(`tests/oracle_groundtruth.rs`). All files are loaded through
`tests/support/dataset.rs` (`file:` specs); Record IDs are dataset ordinals
(`r000000` ..).

| File | Vectors | Dims | Source |
| ---- | ------- | ---- | ------ |
| `siftsmall_base.fvecs` | 10,000 | 128 | INRIA TEXMEX `siftsmall` base set (`.fvecs`, little-endian f32) |
| `siftsmall_query.fvecs` | 100 | 128 | INRIA TEXMEX `siftsmall` held-out query set |
| `siftsmall_groundtruth.ivecs` | 100 x 100 | i32 | INRIA TEXMEX published top-100 neighbors per query |
| `fashion_mnist_base.idx3-ubyte` | 10,000 | 784 | Zalando Research fashion-mnist `t10k-images` (IDX ubyte) |
| `fashion_mnist_query.idx3-ubyte` | 20 | 784 | First 20 images of the fashion-mnist `train` split (held out from the base set) |

## Provenance and terms

- The `siftsmall` files are byte-identical to those in the INRIA TEXMEX
  `siftsmall.tar.gz` distribution (downloaded from the public mirror in
  `TileDB-Inc/TileDB-Vector-Search`, `external/test_data/files/siftsmall/`,
  where `input_vectors.fvecs` is the base set). The TEXMEX datasets are
  provided for evaluation/research use; see
  <http://corpus-texmex.irisa.fr/>.
- The fashion-mnist files derive from the official distribution at
  <https://github.com/zalandoresearch/fashion-mnist> (MIT License).
  `fashion_mnist_base.idx3-ubyte` is the gunzipped `t10k-images-idx3-ubyte`
  unchanged; `fashion_mnist_query.idx3-ubyte` truncates the gunzipped
  `train-images-idx3-ubyte` header to a count of 20 and keeps the first 20
  image payloads.

Fixtures are append-only: corpus expectations depend on their exact bytes, so
replacing one requires regenerating and reviewing the affected expectations
(`KTANN_REWRITE=1 cargo test --test e2e`).
