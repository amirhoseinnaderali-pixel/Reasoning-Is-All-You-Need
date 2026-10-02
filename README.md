# CTTR-VPS

## Collective Test-Time Reasoning for Verified Program Synthesis

> **Status: REGISTERED — HISTORICAL RESEARCH CASE STUDY**
>
> ⚠️ **All quantitative values in Sections 5–6 are _recorded empirical results_ (hypotheses), not measured results.**
> They are derived from the compute budgets in `configs/` and from published scaling behavior of test-time compute on competitive-programming tasks. They exist so the experiment can be **falsified**. Replace them with measured values from `results/` once `scripts/evaluate.py` has been run. No new controlled benchmark result is claimed.

---

## 1. Abstract

We study how additional **inference-time computation** changes the probability that an LLM-based system produces an **objectively correct executable** C++ solution to IOI-style algorithmic problems. CTTR-VPS composes: problem preprocessing → multi-model planning → multi-candidate code generation → optimization → objective candidate selection → iterative execution-based debugging. Correctness is judged only by executing code against a test set; no model-as-judge signal is used for final selection.

We pre-register the hypothesis that verification-driven test-time compute (execution feedback + candidate selection + debugging) yields a larger gain than blind resampling at comparable call budgets, at the cost of roughly an order of magnitude more latency and model calls.

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

## 5. Recorded Experimental Results

> **Expected values — to be replaced by measurements.**

### 5.1 Main comparison

| Arm | Full-test correctness | Mean subtask score | pass@1 (visible) | Model calls / problem | Latency (× A0) |
|-----|:---:|:---:|:---:|:---:|:---:|
| A0 Single-pass | 5 % (0–15) | 27 (18–36) | 22 % | 1 | 1× |
| A1 Multi-sample (N = 8) | 10 % (0–25) | 34 (24–44) | 38 % | 8 | ≈ 1.5× |
| A2 Self-refinement | 7 % (0–20) | 31 (21–40) | 27 % | 4 | ≈ 3× |
| A3 Execution refinement | 15 % (5–30) | 42 (32–52) | 52 % | ≈ 11 | ≈ 6× |
| **A4 CTTR-VPS** | **25 % (10–40)** | **54 (42–65)** | **68 %** | **≈ 30** | **≈ 12×** |

Ranges are expected 95 % intervals reflecting the small task count.

### 5.2 pass@k scaling (expected, full-test)

| k | A1 Multi-sample | A4 CTTR-VPS |
|---|:---:|:---:|
| 1 | 5 % | 12 % |
| 4 | 8 % | 20 % |
| 8 | 10 % | 25 % |

Expected shape: A1 saturates early (≈ 2 pp per doubling of k); A4 keeps rising because candidates are **verified and repaired**, not merely resampled.

### 5.3 Ablations

| Removed component | Expected full-test correctness | Δ vs full |
|---|:---:|:---:|
| Full CTTR-VPS | 25 % | — |
| − multi-model planning (single plan) | 20 % | −5 pp |
| − candidate pool (N = 1) | 17 % | −8 pp |
| − optimization stage | 23 % | −2 pp |
| − execution-based debugging | 12 % | **−13 pp** |
| − objective selection (random pick) | 15 % | −10 pp |

Expected contribution ranking: **debugging > selection > candidate pool > planning > optimization.**

### 5.4 Efficiency

| Arm | Extra calls vs A0 | Gain over A0 (pp) | pp per +10 calls |
|---|:---:|:---:|:---:|
| A1 | +7 | +5 | ≈ 7.1 |
| A3 | +10 | +10 | ≈ 10.0 |
| A4 | +29 | +20 | ≈ 6.9 |

Expected conclusion: A3 is the most **call-efficient**; A4 is the most **accurate**. Diminishing returns beyond ≈ 20 debug steps (< 1 pp per additional 5 steps).

## 6. Hypotheses and Falsification Criteria

| ID | Hypothesis | Falsified if |
|----|-----------|--------------|
| H1 | A4 > A0 on full-test correctness by ≥ 15 pp | Gain < 10 pp or CI includes 0 |
| H2 | A3 > A1 at matched calls (≈ 8–11) | A1 ≥ A3 |
| H3 | Removing debugging causes the largest ablation drop | Another ablation exceeds it |
| H4 | A4 latency ≥ 8× A0 | Latency < 5× |
| H5 | Gains concentrate on mid-difficulty tasks; the hardest task stays near 0 % for all arms | A4 solves the hardest task in ≥ 3/5 seeds |

## 7. Reproduce

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

* **Cost trade-off.** Higher accuracy is bought with ≈ 10× more latency and calls.

* **Historical artifacts.** Existing artifacts lack final hidden-judge telemetry, so no historical success rate can be defended.

## 9. Documentation

`docs/research_report.md`, `docs/architecture.md`, `docs/experiments.md`, `docs/result_schema.md`, `docs/research_positioning.md`.

## 10. Citation

See `CITATION.cff`.

## 11. History

Originally named *Reasoning-Is-All-You-Need*. Original modules and research artifacts are retained.
