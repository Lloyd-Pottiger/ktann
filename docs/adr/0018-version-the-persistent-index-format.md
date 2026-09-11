# Store the persistent format version in the Index Manifest

The Persistent Format defines the encoding of Logical Keys, stored values,
adapter physical keys, and algorithms that determine persisted bytes, including
RaBitQ7 payloads. Its version is stored in the Index Manifest. The supported
version is 1 (`FORMAT_VERSION`). An index is opened through the adapter that
owns its Backend Namespace.

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
