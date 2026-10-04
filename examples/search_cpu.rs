//! Reproducible warmed-cache public-API search CPU benchmark.
//!
//! Run `cargo run --release --example search_cpu -- [dimension] [queries]
//! [concurrency_csv] [records] [cosine|l2] [k]` (defaults: 768 1000 1,4,16
//! 5000 cosine 100). A single leaf avoids maintenance and topology variation.
//! Each phase performs identical query work and checks every result against
//! its warmed checksum. Use `/usr/bin/time -l` for process resource accounting.

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use ktann::api::{
    IndexConfig, Metric, Mutation, MutationOutcome, Record, RuntimeConfig, SearchOutcome,
    SearchRequest,
};
use ktann::runtime::Runtime;
use ktann_memory::MemoryBackend;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use tokio::task::JoinSet;

/// Produces deterministic, finite, nonzero vectors without a random dependency.
fn vector(seed: usize, dimension: usize) -> Arc<[f32]> {
    let mut state = (seed as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..dimension)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            (value >> 40) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}

/// Hashes exact hit ordering, IDs, and distance bits for deterministic checking.
fn checksum(outcome: &SearchOutcome, k: usize) -> u64 {
    assert_eq!(outcome.hits.len(), k, "unexpected hit count");
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for hit in &outcome.hits {
        assert!(hit.distance().is_finite(), "non-finite search distance");
        for byte in hit
            .id()
            .iter()
            .copied()
            .chain(hit.distance().to_bits().to_le_bytes())
        {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// Refreshes only this process and returns cumulative CPU milliseconds.
fn cpu_millis(system: &mut System, pid: Pid) -> Result<u64, Box<dyn Error>> {
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_cpu(),
    );
    system
        .process(pid)
        .map(|process| process.accumulated_cpu_time())
        .ok_or_else(|| "benchmark process missing from CPU sample".into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let dimension = args.first().map_or(Ok(768), |s| s.parse::<usize>())?;
    let queries = args.get(1).map_or(Ok(1000), |s| s.parse::<usize>())?;
    let concurrency: Vec<usize> = args
        .get(2)
        .map_or("1,4,16", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let records = args.get(3).map_or(Ok(5000), |s| s.parse::<usize>())?;
    let metric = match args.get(4).map_or("cosine", String::as_str) {
        "cosine" => Metric::Cosine,
        "l2" => Metric::L2,
        _ => return Err("metric must be cosine or l2".into()),
    };
    let k = args.get(5).map_or(Ok(100), |s| s.parse::<usize>())?;
    if queries == 0 || concurrency.is_empty() || concurrency.contains(&0) || records < k || k == 0 {
        return Err(
            "queries, concurrency, and k must be positive; records must be at least k".into(),
        );
    }
    let maximum = u32::try_from(records)?
        .checked_add(1)
        .ok_or("record count overflow")?;
    let runtime = Runtime::new(
        MemoryBackend::new(),
        RuntimeConfig::default()
            .with_partition_cache_bytes(512 * 1024 * 1024)?
            .with_foreground_operation_limit(
                *concurrency.iter().max().expect("nonempty concurrency"),
            )?,
    )?;
    let index = runtime
        .create_index(
            "search-cpu",
            IndexConfig::new(dimension, metric)?.with_partition_entries(1, maximum)?,
        )
        .await?;
    for chunk in (0..records).collect::<Vec<_>>().chunks(100) {
        let mutations = chunk
            .iter()
            .map(|&id| {
                Record::new(
                    Bytes::copy_from_slice(&(id as u64).to_be_bytes()),
                    vector(id, dimension),
                    vec![],
                )
                .map(Mutation::Insert)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outcomes = index.batch_mutate(mutations).await?;
        assert!(
            outcomes
                .iter()
                .all(|o| matches!(o, MutationOutcome::Inserted))
        );
    }
    let pool: Arc<[Arc<[f32]>]> = (0..32).map(|q| vector(records + q, dimension)).collect();
    let mut expected = Vec::with_capacity(pool.len());
    for query in pool.iter() {
        expected.push(checksum(
            &index.search(SearchRequest::new(query.clone(), k)?).await?,
            k,
        ));
    }
    let expected: Arc<[u64]> = expected.into();
    println!(
        "dimension={dimension} records={records} metric={metric:?} k={k} queries={queries} query_pool={} cache_bytes=536870912",
        pool.len()
    );
    let pid = sysinfo::get_current_pid()?;
    let mut system = System::new();
    for concurrency in concurrency {
        let cpu_before = cpu_millis(&mut system, pid)?;
        let start = Instant::now();
        let mut tasks = JoinSet::new();
        for worker in 0..concurrency {
            let index = index.clone();
            let pool = pool.clone();
            let expected = expected.clone();
            tasks.spawn(async move {
                let mut latencies = Vec::new();
                let mut total = 0_u64;
                let mut reranked = 0_u64;
                for ordinal in (worker..queries).step_by(concurrency) {
                    let slot = ordinal % pool.len();
                    let start = Instant::now();
                    let outcome = index
                        .search(SearchRequest::new(pool[slot].clone(), k).expect("valid request"))
                        .await
                        .expect("search succeeds");
                    latencies.push(start.elapsed().as_secs_f64());
                    reranked += u64::from(outcome.usage.exact_rerank_candidates);
                    let actual = checksum(&outcome, k);
                    assert_eq!(
                        actual, expected[slot],
                        "search result changed for query {slot}"
                    );
                    total = total.wrapping_add(actual);
                }
                (latencies, total, reranked)
            });
        }
        let mut latencies = Vec::with_capacity(queries);
        let mut total = 0_u64;
        let mut reranked = 0_u64;
        while let Some(result) = tasks.join_next().await {
            let (times, hash, count) = result?;
            reranked += count;
            latencies.extend(times);
            total = total.wrapping_add(hash);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let cpu_after = cpu_millis(&mut system, pid)?;
        let cpu_seconds = cpu_after
            .checked_sub(cpu_before)
            .ok_or("process CPU counter decreased")? as f64
            / 1000.0;
        assert_eq!(latencies.len(), queries);
        latencies.sort_unstable_by(f64::total_cmp);
        println!(
            "concurrency={concurrency} elapsed_s={elapsed:.6} qps={:.3} cpu_s={cpu_seconds:.6} cpu_cores={:.3} cpu_us_per_query={:.3} p50_ms={:.3} p99_ms={:.3} exact_rerank_candidates={reranked} checksum={total:016x}",
            queries as f64 / elapsed,
            cpu_seconds / elapsed,
            cpu_seconds * 1_000_000.0 / queries as f64,
            latencies[(queries - 1) / 2] * 1000.0,
            latencies[(queries - 1) * 99 / 100] * 1000.0
        );
    }
    runtime.shutdown().await?;
    Ok(())
}
