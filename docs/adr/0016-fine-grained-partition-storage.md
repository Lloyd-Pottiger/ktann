# Store partition authority and entries as independent KV values

A partition stores authority State, immutable Centroid, exact-count/cache-epoch Header, strong Synopsis, and each Child or Leaf Entry as independently addressable KV values. Compared with rewriting one monolithic partition value, this layout reduces write amplification and enables bounded structural moves, but same-leaf foreground mutations still serialize through the Header and Synopsis keys to preserve exact counts and strong pruning summaries.

Vector Record, Record Location, and optional Opaque Payload are also separate values. Location contains the canonical Tree Key and leaf Partition Key, allowing structural moves to update only the small authority mapping rather than rewrite the original vector, fields, or payload; point reads recover all requested pieces with one batch-get. Every index-scoped key, including its Index Manifest, occupies one contiguous Logical Index ID range. The name directory and namespace ID allocator are outside that range. Range-clear drop removes the index range and name mapping atomically; paged drop preserves the Dropping Manifest until the remaining data is removed.

Within that range, physical locality follows access ownership rather than value type. Record, Location, and Payload keys are adjacent under each Record ID. The Tree Directory is a separate Tree-Key-ordered range so broad enumeration reads exactly one item per tree. All other tree data is grouped first by canonical Tree Key and then Partition Key; State, Centroid, Header, Synopsis, and every Child or Leaf Entry share that partition prefix. This makes one partition a contiguous clearable range and keeps tree-local traversal and maintenance adjacent, at the cost of repeating the Tree Key already accepted by the no-Tree-ID design. Logical Index ID and Partition Key are nonzero fixed-width big-endian u64 values; zero is invalid. Tree Key and Record ID components use one self-terminating memcomparable encoding pass.

Persistent value validation checks type tags, lengths, and structure. Backend
storage checksums detect physical corruption. Once a split or merge source's
exact update-protected Header count reaches zero, one final transaction switches
topology and removes the source. It clears the complete partition prefix where
supported; otherwise it point-deletes the fixed State, Header, Synopsis, and
Centroid keys. Completion does not rescan entries or retain a terminal state for
paged cleanup; see [split completion](0014-expose-then-drain-splits.md).

Full index drop instead performs bounded deletion while update-protecting the
same Dropping Index Manifest and Logical Index ID. Paged deletion preserves that
Manifest until the remaining data is removed. Unknown commit outcomes restart
from the prefix beginning; already-deleted keys need no persistent delete cursor.
