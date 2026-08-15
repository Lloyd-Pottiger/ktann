//! FoundationDB-specific commit-outcome fault checks against a real cluster.

use bytes::Bytes;
use foundationdb::Database;
use foundationdb::api::{FdbApiBuilder, NetworkAutoStop};
use foundationdb::options::NetworkOption;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, ReadOps, WriteTxn};
use ktann::storage::keys::KeyRange;
use ktann_foundationdb::{BackendNamespace, FoundationDbBackend};

const MAX_BUGGIFIED_COMMITS: usize = 256;

#[expect(
    unsafe_code,
    reason = "the FoundationDB binding requires one process-global network boot"
)]
fn boot_foundationdb_for_faults() -> NetworkAutoStop {
    let network = FdbApiBuilder::default()
        .build()
        .expect("initialize FoundationDB API")
        .set_option(NetworkOption::ClientBuggifySectionActivatedProbability(100))
        .expect("activate client buggify sections")
        .set_option(NetworkOption::ClientBuggifySectionFiredProbability(25))
        .expect("bound client buggify firing probability");

    // SAFETY: this integration-test binary contains one test, so it starts the
    // process-global network exactly once and retains the guard until every
    // FoundationDB object has been dropped.
    unsafe { network.boot() }.expect("boot FoundationDB network")
}

#[expect(
    unsafe_code,
    reason = "FoundationDB explicitly permits Client Buggify runtime control"
)]
fn set_client_buggify(enabled: bool) {
    let option = if enabled {
        NetworkOption::ClientBuggifyEnable
    } else {
        NetworkOption::ClientBuggifyDisable
    };
    // SAFETY: Client Buggify is the FoundationDB-supported runtime fault-test
    // switch. This process owns the one initialized network and calls no other
    // process-global FoundationDB configuration concurrently.
    unsafe { option.apply() }.expect("set client buggify state");
}

async fn clear_test_keys(backend: &FoundationDbBackend) {
    let mut transaction = backend.begin_write().await.expect("begin cleanup");
    transaction
        .clear_range(&KeyRange::new(Vec::new(), vec![0xff]))
        .await
        .expect("clear fault-test range");
    transaction.commit().await.expect("commit cleanup");
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires a local FoundationDB 7.3 client and cluster"]
async fn foundationdb_maps_an_applied_unknown_commit_outcome() {
    let _network = boot_foundationdb_for_faults();
    let cluster_file = std::env::var("FDB_CLUSTER_FILE").ok();
    let backend = FoundationDbBackend::new(
        Database::new(cluster_file.as_deref()).expect("open FoundationDB"),
        BackendNamespace::new("ktann-issue-20-commit-faults").expect("namespace"),
    );
    let mut observed_applied_unknown = false;

    // Start from an empty namespace while fault injection is disabled. Client
    // Buggify short-circuits before registering fault sites when disabled, so
    // the following enable still activates every commit site.
    clear_test_keys(&backend).await;
    set_client_buggify(true);

    // Client Buggify injects the same commit errors as the production client.
    // Attempts are bounded; each key is unique so an unknown result can be
    // classified afterward without retrying or overwriting that mutation.
    for attempt in 0..MAX_BUGGIFIED_COMMITS {
        let key = Bytes::from(format!("unknown-{attempt:03}").into_bytes());
        let mut transaction = backend.begin_write().await.expect("begin faulted write");
        transaction
            .put(key.clone(), Bytes::from_static(b"committed"))
            .await
            .expect("stage faulted write");

        match transaction.commit().await {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::RetryableAbort => {}
            Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => {
                set_client_buggify(false);
                let mut read = backend.begin_read().await.expect("read unknown outcome");
                if read
                    .get(key.clone())
                    .await
                    .expect("get unknown outcome")
                    .is_some()
                {
                    observed_applied_unknown = true;
                    break;
                }
                set_client_buggify(true);
            }
            Err(error) => panic!("unexpected commit error category: {:?}", error.kind()),
        }
    }

    set_client_buggify(false);
    assert!(
        observed_applied_unknown,
        "no applied unknown commit observed within {MAX_BUGGIFIED_COMMITS} bounded attempts"
    );
    clear_test_keys(&backend).await;
}
