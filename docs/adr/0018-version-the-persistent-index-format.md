# Version the persistent index format as one whole

The Index Manifest stores one core value-format version, and every typed value
carries a type tag and codec version so wrong key/value pairings and unsupported
encodings fail closed. Each adapter separately versions its physical key
encoding. All versions are scoped to opening an index through the same adapter;
they do not define cross-backend interchange. KTANN v1 performs no in-place
migration and never guesses compatibility. An unsupported Manifest format or
its declared codec combination returns UnsupportedFormat. Once a Manifest
declares a supported format, an unknown value tag, codec variant, Partition
State discriminant, or illegal value combination is local Corruption rather
than speculative evidence of a newer format; decoders never treat unknown data
as Ready, Missing, or another permissive fallback.
