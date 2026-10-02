# Benchmark provenance

## Legitimate source recovered

The repository history and public project documentation identify Humanity's Last Code Exam's IOI benchmark as the intended upstream source. The public immutable source used for the materialization is:

- repository: `Humanity-s-Last-Code-Exam/HLCE`
- path: `HLCE/IOI_scripts/examples/ioi_contest_problems_chatgpt-4o-latest.jsonl`
- Git blob SHA: `01bba2e3569bbec3dfa22990b8c0f0394af79d39`
- source SHA-256: `824fbb624efbd595499f51c78e24ef47a9b4da2612208c1f15f68843daa59db0`
- rows: 93

The project also explicitly references the `HumanLastCodeExam/ioi` dataset.

## Frozen public materialization

`benchmark/materialized.json` is a deterministic 93-task public benchmark materialization. Each task contains a stable `data_uuid`-derived task ID, the upstream refined problem statement, public sample tests when present, source provenance, a task hash, and a visible-test hash.

Public materialization SHA-256:

`85b88a94b7abb62b2caa86895d2d8edf271bd7716ecb22868881ef6940c04925`

Manifest identity SHA-256:

`4e9cac2f5f22516d132c198bc0f66f612c52b4cf7c569719dfd3547ee7d8f453`

## Hidden-test boundary

The public HLCE source does **not** provide the hidden/full-judge test artifact required for an offline, reproducible EXP-001 run. The published HLCE IOI evaluation instructions use the IOI Codeforces contest for definitive scoring rather than distributing a local hidden-test bundle.

Therefore no hidden tests have been fabricated, copied from visible samples, or silently substituted.

The manifest intentionally remains:

`PUBLIC_MATERIALIZATION_FROZEN_PENDING_HIDDEN_ARTIFACT`

Real mode remains fail-closed until an external artifact supplies one hidden-test set and verified hidden-test hash per frozen task. The artifact must remain unavailable during generation, planning, candidate selection, optimization, refinement, and debugging.
