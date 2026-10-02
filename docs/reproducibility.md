# Reproducibility

Validation mode performs software-only checks.

Smoke mode uses the real model adapter and real Docker sandbox against `benchmark/smoke_task.json`. Smoke artifacts are always marked `VALIDATION_ONLY` and never count as EXP-001 evidence.

Real mode requires frozen benchmark materialization, credentials, Docker, a digest-pinned execution image, frozen model/runtime settings, and externally isolated hidden tests.

The exact protocol lives in `configs/exp001.yaml`. New runs receive new run IDs and never overwrite existing result artifacts.
