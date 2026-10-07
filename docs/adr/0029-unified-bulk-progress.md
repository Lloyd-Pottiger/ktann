# ADR 0029: One durable Bulk Build progress record

Status: Accepted. Refines the descriptor/load/proof record layout in
[ADR 0027](0027-bulk-workspace-and-publication.md).

## Context

Separate load and validation records duplicate the artifact identity and encode
one phase across completion/sealed flags and proof-record presence. Each caller
must maintain those relationships even though the stages advance serially.

## Decision

Keep one index-owned Build Progress record containing the immutable Serving
Artifact, load epoch, and a phase: Loading, Loaded, Validating, or Validated.
Only Loading and Validating contain their respective checkpoints. Counts alone
never imply completion: artifact EOF is required for Loaded, and full backend
scan plus artifact EOF are required for Validated.

Sealing atomically changes Loaded to Validating on the same key that every load
chunk update-protects. Validation advances that key; activation requires its
Validated state and all existing manifest, workspace, and schedule fences.
Keep loading and validation execution/retry loops separate. Persisted progress
survives workspace cleanup; publication remains solely the Manifest lifecycle.

Remove the separate validation key/value family. No released consumer requires
compatibility. Workspace preparation authority, namespace cleanup ownership and
scheduler leases remain separate because they protect different lifetimes.

## Consequences

Cross-record phase and artifact consistency checks disappear. Proof transactions
read one progress record, and sealing writes one record. Terminal phases derive
counts/digests from the artifact rather than persisting duplicate values.
Codec goldens and recovery tests change together; old in-development build state
is not readable by this format.
