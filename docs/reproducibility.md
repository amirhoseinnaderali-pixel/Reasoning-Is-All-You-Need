# Reproducibility

Validation mode performs software-only tests.

Smoke mode uses the real model adapter and real Docker sandbox against `benchmark/smoke_task.json`. Smoke artifacts are always marked `VALIDATION_ONLY` and never count as EXP-001 evidence. Smoke preflight still requires real model credentials and Docker.

Real mode requires frozen benchmark materialization, credentials, Docker, a digest-pinned execution image, frozen model/runtime settings, externally isolated hidden tests, and zero provider retries.

The exact protocol lives in `configs/exp001.yaml`. New runs receive new run IDs and never overwrite existing result artifacts. Every record carries task-level visible/hidden test hashes, candidate-set identity, configuration/model/benchmark hashes, seed, token usage, debug steps, wall-clock and environment metadata.
