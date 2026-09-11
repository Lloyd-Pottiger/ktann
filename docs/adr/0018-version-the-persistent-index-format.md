# Version the persistent index format as one whole

The Index Manifest stores the single whole-format marker (`FORMAT_VERSION = 1`).
It governs logical keys, values, adapter physical encodings, and persistent
algorithms, including nested RaBitQ7 payloads. KTANN has no stable release and
all current consumers move with the code. During this unreleased phase, encoding
changes update the sole implementation directly while the marker remains 1.

Logical keys begin with a scope tag. Typed values begin with a type tag followed
directly by their payload. Adapter prefixes contain a backend marker and a
length-delimited Backend Namespace. These tags and lengths identify ownership
and structure; keys, values, and physical prefixes carry no independent format
versions. Namespace isolation and canonical encoding remain mandatory.

The whole-format marker is scoped to opening an index through the same adapter;
it does not define cross-backend interchange. An unsupported Manifest marker
returns UnsupportedFormat. Malformed values, wrong key/value type pairings,
unknown discriminants, noncanonical scalars, and illegal value combinations
return Corruption, including in allocator and Index Name values.

There is one implemented layout and no migration, dual decoding, or compatibility
path. Development data must be recreated after incompatible encoding changes;
marker 1 does not promise compatibility between unreleased builds. Decoders
never treat unknown data as Ready, Missing, or another permissive fallback.
