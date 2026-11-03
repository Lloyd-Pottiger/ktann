# Share one persistent format version across index storage and bulk artifacts

Status: Accepted; amended 2026-11-03 to unify bulk and index format ownership.

The Persistent Format defines the encoding of Logical Keys, stored values,
adapter physical keys, and algorithms that determine persisted bytes, including
RaBitQ7 payloads and bulk source, forest, and serving artifacts. Its version is
stored as a big-endian u16 in the Index Manifest and in bulk manifests and file
headers. The supported version is 1 (`storage::FORMAT_VERSION`). Bulk artifacts
reuse index value encodings, so both formats change under this shared version. An index is opened through the
adapter that owns its Backend Namespace.

Logical Keys begin with a namespace or index scope tag. Stored values begin
with a type tag followed by their payload. Adapter physical keys consist of a
backend marker, a length-delimited Backend Namespace, and the Logical Key.
These tags and lengths identify ownership and structure. Canonical encodings
preserve deterministic bytes, ordered scans, and namespace isolation.

Loading an Index Manifest validates the stored format version; an unsupported
version returns UnsupportedFormat. Decoders validate scope and type tags,
lengths, canonical scalars, identities, and structural invariants. Invalid
encodings and wrong key/value pairings return Corruption, including in
allocator and Index Name values.
