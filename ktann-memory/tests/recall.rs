//! Real-vector recall, filtering, maintenance, and lifecycle verification.

use ktann_memory::MemoryBackend;
use std::path::Path;

#[path = "../../tests/support/adapter_recall.rs"]
mod adapter_recall;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_recall_matches_the_corpus_contract() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/datadriven/data");
    let mut base = fixtures::read_vectors(&fixture_dir, "siftsmall_base.fvecs");
    base.truncate(1000);
    let mut queries = fixtures::read_vectors(&fixture_dir, "siftsmall_query.fvecs");
    queries.truncate(20);
    adapter_recall::run(MemoryBackend::new(), base, queries).await;
}
