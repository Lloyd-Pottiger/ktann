# Version the persistent index format as one whole

The Index Manifest stores one whole persistent-format marker (`FORMAT_VERSION`,
currently 2). Every typed value carries a one-byte type tag followed directly by
its payload. There is one implemented encoding, and all current consumers move
with the code; values and the Manifest carry no independent value-codec version.
The whole-format marker governs value layouts and persistent algorithms,
including nested RaBitQ7 payloads. A layout change updates that marker.

Logical keys and adapter physical-key encodings retain their separate version
markers. All versions are scoped to opening an index through the same adapter;
they do not define cross-backend interchange. KTANN performs no in-place
migration and never guesses compatibility. Existing development data must be
recreated when its persisted layout is unsupported.

An unsupported Manifest format returns UnsupportedFormat. Malformed values,
wrong key/value type pairings, unknown Partition State discriminants,
noncanonical scalars, and illegal value combinations return Corruption,
including in allocator and Index Name values. Decoders never treat unknown data
as Ready, Missing, or another permissive fallback.
