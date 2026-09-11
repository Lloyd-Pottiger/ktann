# Allocate non-reusable Logical Index IDs and govern them with manifests

KTANN allocates Logical Index IDs one at a time from a persistent monotonic high-water mark and never reuses them; therefore a recreated index receives a new physical key range without an incarnation or per-ID retirement tombstone. A caller supplies a stable Index Name: create atomically allocates an ID and installs a versioned manifest, an identical retry returns the current ID, and conflicting configuration is rejected; after drop completes the name may identify a newly allocated ID. If create commit has an unknown outcome, KTANN opens a fresh snapshot and reads that name: an Active manifest with the identical configuration recovers and returns its Index, a different configuration returns IndexAlreadyExists, and absence returns CommitOutcomeUnknown without allocating another ID because absence cannot prove that the uncertain commit will never become visible.

Open accepts only Index Name, reconstructs all immutable configuration from the
current adapter's manifest, and validates that adapter's supported format. A
Dropping manifest returns
IndexDropping rather than a handle that cannot operate. Every operation
validates that the name manifest is Active and still names its handle's
ID—batched with its other initial reads where possible—while mutations update-
protect that read; thus an old handle rejects a new same-name index rather than
silently retargeting.

Drop is a synchronous idempotent state machine: it changes Active to Dropping, resumes an already-Dropping manifest, clears that Logical Index ID's data range, and deletes the manifest last; an absent manifest is already successfully dropped. FoundationDB clears the complete range and deletes the Manifest atomically, so after CommitOutcomeUnknown an absent Manifest proves completion and the same Dropping Manifest permits retry. A point-delete backend instead rereads the Manifest and restarts its remaining range from the prefix beginning after an unknown outcome. Create for the same name returns IndexDropping until cleanup removes the manifest, preserving the authoritative recovery entry for the old ID. Unsupported formats are rejected and immutable schema, metric, quantizer, or Tree-Key changes require a new Logical Index rather than online migration.

IndexConfig contains persistent caller choices: dimension, metric, ordered field schema, ordered Tree Key fields, per-field MinMax or MinMaxBloom synopsis policy, and partition entry thresholds. RuntimeConfig controls process caches, maintenance, and default search policy. SearchOptions provides request-level search settings. Adapter configuration owns backend controls such as RocksDB blocking concurrency.

The Persistent Format defines RaBitQ7, binary fanout, Lloyd iteration count, rotation algorithm, and safety limits. The Index Manifest stores the canonical caller configuration, persistent format version, Logical Index ID, rotation seed, and concrete Bloom bit/hash counts. Idempotent create compares the canonical caller configuration.
