# Closure review: immutable evaluation and paper tables

Verdict: **PASS after remediation**.

- Full/detail attempt fingerprints bind actual model bytes and loading metadata, the complete VLMEvalKit Python tree, pre-run package freeze, exact data bytes, dataset classes, and generation config.
- Resume requires an exact preflight and unchanged config, then revalidates prediction/score hashes and coverage.
- The 60/40 split is deterministic and checksum-backed. MMBench keeps complete four-row circular groups; SEED is nine-category stratified. Existing splits and detail runs recompute source MD5, full row content, indices, partition digests, and split ID.
- Table aggregation requires authenticated immutable manifests, paired roles/run IDs/models/protocols, score containment/hashes, prediction binding, and either no detail inputs or a compatible pair. Final packaging requires the paired 40% detail pair.
- RealWorldQA and VStar follow the owner-resolved checksum-pinned MCQ protocols.

CPU tests rejected mutated exported weights, a one-row self-attested split, unpaired detail manifests, changed attempt config/provenance, and incompatible table inputs.

