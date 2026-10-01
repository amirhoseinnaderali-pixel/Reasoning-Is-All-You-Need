# CTTR-VPS Architecture

Research question: under fixed problem sets and comparable compute budgets, how does structured multi-model planning, iterative candidate generation/refinement, and execution-based selection affect correctness and inference cost?

H1: structured test-time computation combining diverse planning, candidate generation, refinement, and execution verification improves problem-level correctness versus single-pass generation under matched compute. This is a hypothesis, not a finding.

Pipeline: problem -> multi-view preprocessing -> multi-model planning -> explicit plan selection -> candidate generation -> refinement -> optimization -> compile/execute -> debugging -> structured result.

Important implementation facts: the legacy code does not implement calibrated probabilistic consensus; planning artifacts are saved and the old orchestrator later selects the first stored plan. Candidate generation does not establish correctness. Some legacy repair paths inspect only a subset of tests. Visible samples are not hidden/full-judge evaluation.

| Stage | Main input/output | Failure modes | Cost signals |
|---|---|---|---|
| Preprocessing | Statement -> structured views | Missing fields, malformed model JSON | Calls, latency |
| Planning | Problem view -> plans | API failure, disagreement, bad algorithm | Calls/model, latency |
| Generation | Plan/problem -> C++ candidates | Empty/invalid output | Candidate count, calls |
| Refinement | Candidate/context -> revised candidates | Regression, sample overfitting | Rounds, calls |
| Optimization | Candidate pool -> optimized candidates | Semantic regression | Calls, test outcomes |
| Verification | C++ + tests -> outcomes | Compile/runtime error, timeout, wrong output | Runtime, memory |
| Selection | Evaluations -> selected candidate | Incomplete evidence | Selection rule |

Package layout: cttr_vps/config.py for environment/YAML; legacy.py for adapters; methods.py for methods; evaluation.py for compile/execute; result_schema.py for records; results.py for aggregation; scripts/ for experiment control.

Security: credentials are environment based. Credentials previously committed to Git should be rotated; removing them from the current tree does not erase Git history.