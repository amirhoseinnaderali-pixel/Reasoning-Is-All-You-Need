# CTTR-VPS

## Collective Test-Time Reasoning for Verified Program Synthesis

**Portfolio role.** The IOI-style C++ study in the portfolio, focused on multi-model planning, objective candidate selection, and execution-based debugging rather than the fixed-inference-budget comparison used elsewhere.

> **Status: COMPLETED — RECORDED STUDY**
>
> The repository records the completed study and its objective evaluation methodology. The public repository does not contain the task-level raw run archive, so aggregate numerical tables are reproduced only where their denominator and provenance can be checked from the committed evidence.

**Evidence at a glance.** The current repository records a completed CTTR-VPS study with objective execution-based evaluation. The public repository does not contain the task-level raw run archive, so unreconciled aggregate accuracy percentages are intentionally not reproduced here. The earlier historical artifact corpus is kept separate from the current study.

---

## 1. Abstract

We study how additional **inference-time computation** changes the probability that an LLM-based system produces an **objectively correct executable** C++ solution to IOI-style algorithmic problems. CTTR-VPS composes: problem preprocessing → multi-model planning → multi-candidate code generation → optimization → objective candidate selection → iterative execution-based debugging. Correctness is judged only by executing code against a test set; no model-as-judge signal is used for final selection.

The study was pre-registered around the hypothesis that verification-driven test-time compute (execution feedback + candidate selection + debugging) yields a larger gain than blind resampling at comparable call budgets, at the cost of roughly an order of magnitude more latency and model calls.

**Research positioning.** Test-time refinement and repeated sampling are established ideas; see [Self-Refine](https://arxiv.org/abs/2303.17651) and [Large Language Monkeys](https://arxiv.org/abs/2407.21787). The project does not claim those mechanisms as novel. Its scope is a benchmark-specific comparison of multi-model planning, candidate generation, objective selection, and execution-based debugging for IOI-style program synthesis.

## 2. Research Question

> How does additional inference-time computation change the probability of producing an objectively correct executable solution?

| ID  | Sub-question |
|-----|--------------|
| RQ1 | Does CTTR-VPS outperform single-pass generation on full-test correctness? |
| RQ2 | At matched call budgets, is execution-grounded refinement better than best-of-N resampling? |
| RQ3 | Which pipeline stage (planning diversity, candidate pool, debugging) contributes most? |
| RQ4 | What is the cost/latency price per additional percentage point of correctness? |

## 3. Method

```
problem statement
      │
      ▼
 preprocessing           normalize statement, extract constraints, I/O format
      │
      ▼
 multi-model planning    K independent plans from heterogeneous models
      │
      ▼
 candidate generation    N C++ candidates conditioned on plans
      │
      ▼
 optimization            complexity / constant-factor pass
      │
      ▼
 objective selection     compile + run on tests → rank by verified score
      │
      ▼
 execution debugging     up to 30 steps using compiler / runtime / WA feedback
      │
      ▼
 final solution
```

Backends: Google generative models (`GOOGLE_API_KEYS`) and Ollama-served models (`OLLAMA_API_KEYS`), with key rotation.

## 4. Experimental Design

### 4.1 Arms

| Arm | Config | Description |
|-----|--------|-------------|
| A0 | `baseline_single_pass.yaml` | One generation, no feedback |
| A1 | multi-sample | N = 8 samples, select by visible-sample pass |
| A2 | self-refinement | Model critiques/rewrites its own code, no execution |
| A3 | execution-based refinement | Single model, up to 10 debug steps with execution feedback |
| A4 | **CTTR-VPS** | Full pipeline (K = 3 plans, N = 8–12 candidates, ≤ 30 debug steps) |

### 4.2 Data

Four IOI-style tasks (`final_*_problem_{1..4}`). The repository retains 14 planning records and 54 labeled generated-code variants. Each arm is run with **S = 5 seeds** per task → 20 runs per arm.

### 4.3 Metrics

* **Primary:** full-test correctness rate (fraction of runs passing **all** tests).
* **Secondary:** mean subtask score (0–100), pass@k (k ∈ {1, 4, 8}).
* **Cost:** model calls, tokens, wall-clock latency, cost proxy.

### 4.4 Statistics

Task-level cluster bootstrap (10,000 resamples over tasks, then seeds) with 95 % CIs. With only 4 tasks the design is underpowered; **only effects ≳ 15 percentage points on the primary metric are considered detectable.**

## 5. Recorded Experimental Study

The current CTTR-VPS study is recorded as an empirical execution with objective full-test evaluation. The public repository does not commit the task-level run archive needed to independently reconstruct every aggregate percentage and ablation value.

To avoid publishing a numerical claim whose denominator or aggregation cannot be checked from the public artifacts, the detailed aggregate result tables are intentionally omitted from this README. The study definition, arms, verifier, leakage controls, uncertainty framework, and historical-vs-current evidence boundary remain documented below and in `docs/research_report.md`.

**Public evidence boundary:** recorded study summary + methodology are available; task-level raw execution rows are not committed to the repository.

## 7. Reproduce

### Legacy artifact layout

The repository root intentionally retains historical files such as `final_planning_problem_*.json`, `final_codes_problem_*.txt`, and `_30step.py`. These are preserved provenance from the earlier CTTR-VPS pipeline, not the canonical structure of the current recorded comparison. The current study definition, evaluation, and reproducibility materials live in `configs/`, `docs/`, `RESULTS.md`, and the reproduction scripts.

```bash
export GOOGLE_API_KEYS='key1,key2,...'

export OLLAMA_API_KEYS='key1,key2,...'

pip install -r requirements.txt

python scripts/run_experiment.py --config configs/baseline_single_pass.yaml

python scripts/evaluate.py --results results
```

Never commit credentials (`.env.example` provided). See `REPRODUCIBILITY.md` for seeds and environment pinning.

## 8. Threats to Validity

* **Visible ≠ hidden tests.** Selection and debugging can overfit visible samples; report correctness only from hidden/full-judge tests.

* **Tiny task count (n = 4).** Wide CIs; conclusions are exploratory.

* **Provider drift.** Model versions and API availability change; log exact model IDs per run.

* **Contamination.** IOI tasks may appear in pretraining data and inflate absolute numbers.

* **Cost trade-off.** The study evaluates correctness together with model calls, tokens, and wall-clock latency; any cost advantage is benchmark- and configuration-specific.

* **Historical artifacts.** The recorded artifacts include the evaluation telemetry used for the reported historical success measurements.

## 9. Documentation

`docs/research_report.md`, `docs/architecture.md`, `docs/experiments.md`, `docs/result_schema.md`, `docs/research_positioning.md`.

## 10. Citation

See `CITATION.cff`.

## 11. History

Originally named *Reasoning-Is-All-You-Need*. Original modules and research artifacts are retained.
