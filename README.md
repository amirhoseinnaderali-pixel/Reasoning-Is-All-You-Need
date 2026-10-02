# CTTR-VPS

## Collective Test-Time Reasoning for Verified Program Synthesis

### Portfolio status

**REGISTERED — HISTORICAL RESEARCH CASE STUDY**

CTTR-VPS is the research identity of this repository. The implementation studies inference-time computation for C++ program synthesis on IOI-style algorithmic tasks.

### Research question
How does additional inference-time computation change the probability of producing an objectively correct executable solution?

### Core pipeline
problem preprocessing -> multi-model planning -> multi-candidate code generation -> optimization -> objective candidate selection -> iterative execution-based debugging.

### Controlled evaluation
The experiment harness provides single-pass generation, multi-sample generation, self-refinement, execution-based refinement, and CTTR-VPS. The primary metric is objective test-set correctness. Candidate count, model calls, debugging steps, latency, and cost proxies are also recorded.

### Results
**No new controlled benchmark result is claimed.** Historical evidence is preserved and analyzed in [`docs/research_report.md`](docs/research_report.md). The repository contains four historical IOI task artifact sets with 14 retained planning records and 54 labeled generated-code variants, but it does not preserve final hidden/full-judge telemetry needed to compute a defensible historical success rate.

### Run
Set credentials outside Git:

    export GOOGLE_API_KEYS='key1,key2,...'
    export OLLAMA_API_KEYS='key1,key2,...'

Install:

    pip install -r requirements.txt

Run:

    python scripts/run_experiment.py --config configs/baseline_single_pass.yaml

Aggregate:

    python scripts/evaluate.py --results results

### Limitations
Visible samples are not hidden/full-judge evaluation. Model/provider versions and API availability can change. Additional test-time computation has an explicit latency/cost trade-off.

See [`docs/research_report.md`](docs/research_report.md), docs/architecture.md, docs/experiments.md, docs/result_schema.md, and docs/research_positioning.md.

### History
The repository was originally named Reasoning-Is-All-You-Need. Original modules and research artifacts are retained.