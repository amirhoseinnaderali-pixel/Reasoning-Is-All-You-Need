# Benchmark provenance

The repository currently contains `ioi_multi_view.json` as a zero-byte file. Therefore the benchmark population, task IDs, visible-test hashes, hidden-test hashes, and benchmark materialization hash cannot truthfully be frozen from the current repository state.

EXP-001 therefore fails closed until a benchmark owner materializes the intended task population and records:

- benchmark source/provenance;
- exact task IDs;
- visible-test hashes;
- hidden-test hashes;
- complete materialization SHA-256;
- matching hidden-test secret artifact.

The machine-readable contract lives in `benchmark/manifest.json`. Do not replace missing benchmark evidence with synthetic or historical results.
