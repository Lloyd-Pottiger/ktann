//! Create an in-memory index, insert a vector, and search it.

use bytes::Bytes;
use ktann::api::{IndexConfig, Metric, Record, RuntimeConfig, SearchRequest};
use ktann::runtime::Runtime;
use ktann_memory::MemoryBackend;

#[tokio::main]
async fn main() -> ktann::api::Result<()> {
    let runtime = Runtime::new(MemoryBackend::new(), RuntimeConfig::default())?;
    let index = runtime
        .create_index("vectors", IndexConfig::new(3, Metric::L2)?)
        .await?;

    index
        .insert(Record::new(
            Bytes::from_static(b"first"),
            vec![1.0, 0.0, 0.0],
            vec![],
        )?)
        .await?;

    let result = index
        .search(SearchRequest::new(vec![1.0, 0.0, 0.0], 1)?)
        .await?;
    for hit in &result.hits {
        println!("distance: {}", hit.distance());
    }

    runtime.shutdown().await?;
    Ok(())
}
