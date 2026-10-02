# CTTR-VPS

## Collective Test-Time Reasoning for Verified Program Synthesis

### Research question
How does additional inference-time computation change the probability of producing an objectively correct executable solution?

### Frozen research instrument

EXP-001 explicitly separates:

- **C0** Single-pass generation
- **C1** Multi-sample generation + objective visible-test selection
- **C2** Self-refinement without execution feedback
- **C3** Execution-based refinement using visible tests
- **C4** CTTR-VPS: planning → candidate generation → deterministic optimization/deduplication → objective visible-test selection → execution-based debugging → final candidate

Exact model/revision, retry policy, generation parameters, seed, candidate budgets, token budgets, wall-clock budget, Docker image, runtime limits, and reproducibility fields are frozen in `configs/exp001.yaml`.

### Scientific status

- **IMPLEMENTED:** frozen protocol, objective evaluation architecture, controlled budgets, immutable result schema, secure Docker sandbox, real execution path, validation/smoke/real modes, audit and analysis infrastructure.
- **VALIDATED:** validation and CI gates are implemented; successful CI completion is not yet claimed here.
- **SCIENTIFICALLY AUDITED:** fail-closed audit implemented; the current audit is expected to remain **FAIL** until the benchmark is frozen.
- **PUBLIC BENCHMARK MATERIALIZATION:** **FROZEN** — 93 tasks from the recovered HLCE IOI source are materialized and hashed.
- **READY FOR REAL EXECUTION:** **BLOCKED** until the external hidden-test artifact is supplied and hashed for every task.
- **REAL SMOKE PASSED:** not claimed.
- **EXP-001 EXECUTED:** not claimed.

### Run modes

Validation performs software-only invariant tests:

`python scripts/run_experiment.py --mode validation`

Smoke uses the real model adapter and real Docker sandbox against the validation-only smoke fixture:

`export GOOGLE_API_KEY='...'`
`python scripts/run_experiment.py --mode smoke --method single_pass`

Real mode requires frozen benchmark materialization, credentials, Docker, and isolated hidden tests:

`python scripts/run_experiment.py --mode real`

No mock fallback exists in real mode.

### Results

No empirical benchmark accuracy, solved-rate, significance result, or performance claim is reported from historical runs. Smoke artifacts are explicitly `VALIDATION_ONLY` and are not EXP-001 evidence.

See `docs/scientific_protocol.md`, `docs/benchmark_provenance.md`, `docs/result_schema_v2.md`, `docs/reproducibility.md`, and `docs/audit.md`.
