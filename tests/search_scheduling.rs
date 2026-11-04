//! Reproducible search scheduling diagnostic; use an otherwise idle host.
//! Measures worker fairness separately from throughput; it makes no SLA claim.
use bytes::Bytes;
use ktann::api::{
    IndexConfig, Metric, Mutation, OperationOptions, Record, RuntimeConfig, SearchRequest,
};
use ktann::runtime::Runtime;
use ktann_memory::MemoryBackend;
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
#[ignore = "release scheduling diagnostic; run on an otherwise idle host"]
async fn search_executor_latency_probe() -> ktann::api::Result<()> {
    tokio::spawn(probe()).await.unwrap()
}

async fn probe() -> ktann::api::Result<()> {
    for (count, dimension, k) in [(32_u32, 8_u32, 10), (4096, 768, 100)] {
        let config = RuntimeConfig::default()
            .with_maintenance(0, 64)?
            .with_partition_cache_bytes(128 * 1024 * 1024)?;
        let runtime = Runtime::new(MemoryBackend::new(), config)?;
        let index = runtime
            .create_index("probe", IndexConfig::new(dimension as usize, Metric::L2)?)
            .await?;
        for start in (0..count).step_by(32) {
            let records = (start..start + 32)
                .map(|id| {
                    let vector: Vec<f32> = (0..dimension)
                        .map(|d| {
                            ((id.wrapping_mul(1664525)
                                .wrapping_add(d.wrapping_mul(1013904223)))
                                % 65521) as f32
                                / 65521.0
                        })
                        .collect();
                    Ok(Mutation::Insert(Record::new(
                        Bytes::copy_from_slice(&id.to_be_bytes()),
                        vector,
                        vec![],
                    )?))
                })
                .collect::<ktann::api::Result<Vec<_>>>()?;
            index.batch_mutate(records).await?;
        }
        let mut times = Vec::new();
        let mut digest = Sha256::new();
        for iteration in 0..51 {
            let start = Instant::now();
            let heartbeat = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(1)).await;
                start.elapsed()
            });
            tokio::task::yield_now().await;
            let result = index
                .search(SearchRequest::new(vec![0.5; dimension as usize], k)?)
                .await?;
            let elapsed = start.elapsed();
            let beat = heartbeat.await.unwrap();
            for hit in &result.hits {
                digest.update(hit.id());
                digest.update(hit.distance().to_bits().to_le_bytes());
            }
            if iteration == 0 {
                println!("n={count} d={dimension} cold={elapsed:?} timer={beat:?}");
            } else {
                times.push((elapsed, beat));
            }
        }
        times.sort_by_key(|x| x.0);
        println!(
            "n={count} d={dimension} warm_p50={:?} warm_p95={:?} hits_sha256={:02x?}",
            times[25].0,
            times[47].0,
            digest.finalize()
        );
        times.sort_by_key(|x| x.1);
        println!(
            "n={count} timer_p50={:?} timer_p95={:?}",
            times[25].1, times[47].1
        );
        let throughput_started = Instant::now();
        for _ in 0..100 {
            index
                .search(SearchRequest::new(vec![0.5; dimension as usize], k)?)
                .await?;
        }
        println!(
            "n={count} sequential_warm_qps={:.1}",
            100.0 / throughput_started.elapsed().as_secs_f64()
        );
        let start = Instant::now();
        let result = index
            .search_with_control(
                SearchRequest::new(vec![0.5; dimension as usize], k)?,
                OperationOptions::default().with_deadline(start + Duration::from_millis(1)),
            )
            .await;
        println!(
            "n={count} deadline_elapsed={:?} result={:?}",
            start.elapsed(),
            result.map(|o| o.hits.len()).map_err(|e| e.kind())
        );
        runtime.shutdown().await?;
    }
    Ok(())
}
