# KTANN FoundationDB adapter

This crate maps KTANN's backend-neutral transactional KV interface onto
FoundationDB 7.3. It compiles against API version 730 and requires the
FoundationDB 7.3 client library at build and runtime.

The embedding process must start the FoundationDB network exactly once, keep
its network guard alive longer than every `FoundationDbBackend`, and pass an
already-open `foundationdb::Database` to the adapter. Each adapter instance
adds a versioned physical prefix containing its caller-selected Backend
Namespace; logical codecs and index algorithms remain in `ktann`.

## Physical key format

Every FoundationDB key has this exact physical prefix before its opaque KTANN
logical key:

```text
00 6b 74 61 6e 6e 01 <namespace-length:u8> <namespace-bytes> <logical-key>
```

`00 6b 74 61 6e 6e` is the KTANN marker and `01` is the FoundationDB physical
format version. Backend Namespaces are limited to 255 bytes. Length delimiting
keeps adjacent namespace values disjoint without escaping the logical key, and
the adapter subtracts the complete prefix length from FoundationDB's 10,000
byte physical-key limit.

The adapter exposes FoundationDB's 100,000-byte value limit and applies the
accepted conservative defaults of 10,000 mutations, 1 MiB of physical mutation
bytes, and 80 KiB per scan page. Native affected-data accounting remains the
final authority and maps an oversized transaction to `TransactionTooLarge`.

## Local integration test

1. Install a FoundationDB 7.3 client and a local single-node server from the
   [FoundationDB releases](https://github.com/apple/foundationdb/releases).
   The client package must provide `libfdb_c` to the linker.
2. Start the server and verify it with `fdbcli --exec status`.
3. If the cluster file is not in FoundationDB's platform-default location, set
   `FDB_CLUSTER_FILE` to its absolute path.
4. Run the ignored adapter and fault tests:

   ```sh
   cargo test -p ktann-foundationdb \
     --test foundationdb_adapter --test foundationdb_faults -- --ignored
   ```

The tests use dedicated Backend Namespaces and clear their test keys after
execution. The adapter test runs every in-process case from the unchanged shared
backend suite, including snapshot consistency, read-your-writes, point conflict
ranges, item/byte-bounded ordered scan pagination, unique insertion, rollback,
and atomic range clear. It also lowers the native transaction-size limit on one
database handle to prove the adapter's `TransactionTooLarge` mapping without
generating a large cluster workload.

The separate fault-test process enables FoundationDB's Client Buggify facility.
It makes at most 256 small commits and requires one transaction that is applied
by the real cluster while the client reports `CommitOutcomeUnknown`; the test
then disables fault injection, verifies the committed key, and clears the
namespace. FoundationDB cannot deterministically stage each exact commit result,
so the shared suite declares controlled fault injection unavailable for this
adapter. The deterministic backend remains the exhaustive evidence for both
applied and unapplied unknown outcomes.

Durability uses a separate two-phase test because restarting a client handle is
not evidence that the server durably stored an acknowledged commit. Run the
write phase, restart the same FoundationDB server without deleting its data
directory, then run the verify phase:

```sh
KTANN_FDB_DURABILITY_PHASE=write \
  cargo test -p ktann-foundationdb --test foundationdb_durability -- --ignored

# Restart FoundationDB here, preserving its data directory.

KTANN_FDB_DURABILITY_PHASE=verify \
  cargo test -p ktann-foundationdb --test foundationdb_durability -- --ignored
```

CI performs this service restart explicitly. The in-process shared harness
therefore declares controlled fault injection and backend restart unsupported;
the fault and durability binaries provide the adapter-specific evidence instead
of silently weakening those shared cases.
