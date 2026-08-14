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
4. Run the ignored adapter test:

   ```sh
   cargo test -p ktann-foundationdb --test foundationdb_adapter -- --ignored
   ```

The test uses two dedicated Backend Namespaces, clears its test keys before and
after execution, and covers namespace isolation, snapshot consistency,
read-your-writes, point conflicts, item/byte-bounded ordered scan pagination,
unique insertion, rollback, atomic range clear, and visibility through a newly
opened database handle.
