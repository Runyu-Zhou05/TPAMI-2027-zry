# Closure review: model/data identity and committed export

Verdict: **PASS after remediation**.

- Base snapshot validation requires the pinned revision, expected two-shard index, all downloaded files, and matching Hugging Face ETags; queue provenance records their SHA-256 values.
- Complete Rotation data requires the exact canonical generated Parquet hashes in addition to raw hashes, exact Arrow schema, generator identity, and row semantics.
- Checkpoint manifests are schema v2 content manifests; resume and export recompute their hashes.
- Export resolves only the committed tracker step, audits exact parameter counts, and records current weight plus config/processor/tokenizer hashes.
- Post-training attempts reject cross-run relabeling and revalidate export weights, support files, the 24-row/six-subtask smoke, required artifacts, Table II projection, and log on resume.

Synthetic negative fixtures rejected a weightless base snapshot, modified base/export/checkpoint shards, noncanonical complete data, cross-run post checkpoint, and incomplete/tampered post-smoke evidence.

