# Store an engine-owned Vector Record with the index

KTANN owns and versions the Vector Record containing the original f32 vector and typed filter fields, while an optional opaque payload is stored separately and fetched only explicitly. The Vector Record, payload, Record Location, and affected index entries commit in the same backend transaction; this rejects caller-defined source-value encoding in exchange for correctness, exact reranking, and atomic source/index mutation without callbacks into application storage.

Opaque Payload accepts `0..=64 KiB` raw bytes. `Some(empty)` is an existing empty payload and remains distinct from None; because upsert is a full replacement, None deletes any previous Payload value.

Dimension is limited to `1..=16,384` as an algorithm and resource-safety bound:
the maximum original f32 vector is 64 KiB, and RaBitQ7 encoding, rotation,
rerank, and complete-source split training otherwise scale into an extreme per-
record and `entries * dimension` cost. Create additionally checks only the
selected Backend's value admission limit and may reject a smaller range. KTANN
deliberately rejects dimensions above the algorithm bound rather than add vector
chunking or unbounded split/search work; acceptance by one backend has no
meaning for another.
