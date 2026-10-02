# CTTR-VPS

## Collective Test-Time Reasoning for Verified Program Synthesis

### Portfolio status

**REGISTERED — HISTORICAL RESEARCH CASE STUDY**

CTTR-VPS is the research identity of this repository. The implementation studies inference-time computation for C++ program synthesis on IOI-style algorithmic tasks.

---

# CTTR-VPS — Pre-Execution Projection

> ⚠️ **EXPECTED / PRIOR ONLY — NOT AN EMPIRICAL RESULT**
>
> The repository does **not** currently contain a controlled CTTR-VPS benchmark result. Historical artifacts are explicitly treated as exploratory, and the current configs define an experimental harness rather than a completed evaluation.
>
> All numbers and probabilities in this section are **pre-data subjective priors**. They are recorded to make the hypothesis falsifiable and must not be interpreted as measured accuracy, significance, or a proven ranking.

## Executive hypothesis

CTTR-VPS — **Collective Test-Time Reasoning for Verified Program Synthesis** — studies whether multi-model planning, multi-sample generation, optimization, execution-based debugging, and final candidate selection can improve the probability of producing an objectively correct executable program.

The expected hierarchy is:

````text
Single-pass
   <
Self-refine without execution
   <
Multi-sample
   <
Execution-refine
   ≲
CTTR-VPS
````

The main mechanism behind this projection is that **execution feedback provides a stronger correctness signal than additional self-generated text**, while multi-model plan sharing may add a smaller complementary benefit on tasks where useful algorithmic information is distributed across models.

The projection therefore does **not** assume that CTTR-VPS will universally dominate a strong execution-based baseline.

---

## 1. Evidence boundary for the projection

Historical evidence currently consists of four preserved IOI-style task artifact sets with planning records and generated-code variants. The repository does **not** preserve the final hidden/full-judge telemetry required for a defensible historical success rate.

The current harness defines:

- **single-pass**
- **multi-sample**
- **self-refine**
- **execution-refine**
- **CTTR-VPS**

The current configuration files also use `problem_index: 0` for the benchmark examples shown in the repository. Therefore this projection should be read as a **pre-execution hypothesis for the harness**, not as evidence that the five methods have already been compared across a broad task population.

---

## 2. Projected method ordering

| Method | Frozen / configured mechanism | Pre-execution expectation | Main reason |
|:--|:--|:--|:--|
| **Single-pass** | One generation | **Lowest baseline** | One attempt, no additional correction signal |
| **Self-refine** | 3 refinement rounds, no execution feedback | **Small improvement** | Textual self-critique can miss or introduce implementation errors |
| **Multi-sample** | 8 samples | **Moderate improvement** | Candidate diversity can increase the chance of obtaining a correct solution |
| **Execution-refine** | Up to 5 refinement rounds with execution verification | **High improvement / strong efficiency** | Concrete test feedback directly exposes implementation failures |
| **CTTR-VPS** | 3 planning rounds + 5 generation rounds + 20 candidates/round + debugging | **Highest expected ceiling, but uncertain advantage over execution-refine** | Combines planning diversity, candidate diversity, optimization, and execution-based debugging |

The intended interpretation is:

> **Execution-refine is expected to capture most of the reliable gain, while CTTR-VPS may add a smaller incremental benefit when model diversity exposes complementary algorithmic ideas.**

---

# 3. Scenario probabilities

These are subjective prior probabilities, not outputs from a statistical model.

| Scenario | Prior probability | Expected outcome |
|:--|--:|:--|
| **A. Negligible or no CTTR-VPS advantage** | **~55%** | CTTR-VPS is roughly tied with execution-refine and may fall behind under a matched budget |
| **B. Real but moderate advantage** | **~30%** | CTTR-VPS improves by roughly **3–10 percentage points** over the strongest baseline on sufficiently difficult tasks, at higher compute cost |
| **C. Strong and persistent advantage** | **~10%** | Shared planning creates genuinely complementary algorithmic diversity that execution-refine alone rarely discovers |
| **D. Negative result** | **~5%** | Consensus or information-sharing propagates correlated errors and CTTR-VPS falls below a simpler baseline |

The most likely region of the prior is therefore **small or no incremental benefit from the collective component once execution feedback is already available**.

---

# 4. Predicted behavior by task difficulty

## Easy tasks

The methods are expected to converge because strong models can already solve the task with little inference-time intervention.

````text
Single-pass ≈ Multi-sample ≈ Execution-refine ≈ CTTR-VPS
````

## Medium-difficulty tasks

This is expected to be the most informative regime.

The projection is that:

- single-pass gains substantially less from extra compute;
- multi-sample finds more viable candidates;
- execution-refine repairs concrete implementation defects;
- CTTR-VPS can occasionally add a better algorithmic plan through cross-model planning.

The largest practical separation is therefore expected here.

## Very hard tasks

The projection is less certain.

If **none** of the models discovers the key algorithmic idea, additional consensus cannot manufacture it.

If **only one model** discovers the key idea, however, sharing plans may allow the other stages to reuse that information.

This is the regime in which the distinctive "collective" component has the greatest chance of showing value.

---

# 5. Visible vs. hidden evaluation

The projection expects visible-test success to exceed hidden/full-judge correctness on at least some tasks.

The reason is structural:

````text
Visible tests
   ↓
selection / debugging feedback
   ↓
candidate optimized for observed evidence
   ↓
hidden/full judge
````

A candidate can therefore pass all visible examples while still failing on:

- boundary cases;
- adversarial inputs;
- complexity constraints;
- unseen input patterns;
- time or memory limits.

The hidden/full-judge separation is consequently essential for any future correctness claim.

---

# 6. Compute and token-cost projection

CTTR-VPS is expected to use substantially more inference computation than execution-refine.

A qualitative prior is:

````text
Single-pass
    ↓
Self-refine
    ↓
Multi-sample
    ↓
Execution-refine
    ↓
CTTR-VPS
````

The exact multiple should **not** be presented as measured until actual token/call telemetry is preserved.

The key projected trade-off is:

> **CTTR-VPS may improve correctness through broader search over plans and candidates, but the incremental correctness per additional token may be lower than the incremental correctness obtained from the first execution-feedback loop.**

For a fair comparison, future analysis should report:

- correctness;
- total model calls;
- generated tokens;
- wall-clock latency;
- execution/debug steps;
- cost proxy;
- correctness per call/token where meaningful.

---

# 7. Stability and uncertainty

Because the current historical evidence preserves only a small number of task artifact sets, the projection expects high variance across problems.

A single problem could move the headline result substantially.

Therefore a convincing future comparison should use:

- more than one task;
- multiple seeds where feasible;
- task-level paired outcomes;
- uncertainty intervals rather than point estimates alone.

The current repository should not convert the four historical task artifacts into a success-rate estimate.

---

# 8. What would support the CTTR-VPS hypothesis?

For a future controlled study, evidence would support the stronger CTTR-VPS claim if:

1. CTTR-VPS improves hidden/full-judge correctness over execution-refine on a multi-task benchmark;
2. the improvement persists under a matched inference budget;
3. the confidence/uncertainty interval for the paired difference excludes zero;
4. an ablation removing shared planning reduces performance, showing that the collective component contributes something beyond execution feedback alone;
5. the gain is not explained by a single outlier task.

Because the current configs show only `problem_index: 0`, these are **future validation criteria**, not current empirical findings.

---

# 9. What would falsify the hypothesis?

The projection would be substantially weakened if:

- CTTR-VPS performs no better than execution-refine under matched compute;
- removing plan sharing leaves performance unchanged;
- extra planning/candidate generation mainly increases cost without improving hidden correctness;
- consensus frequently preserves the same wrong algorithmic assumption;
- the strongest gains disappear once candidate count and debugging budget are equalized.

A negative or neutral result would still be informative because it would identify **execution feedback**, rather than collective reasoning, as the dominant mechanism.

---

# 10. Pre-Execution Scorecard

Freeze before the first valid controlled run:

- [ ] Multi-sample improves over single-pass
- [ ] Execution-refine improves over self-refine
- [ ] CTTR-VPS is at least competitive with execution-refine
- [ ] Any CTTR-VPS advantage is concentrated on medium/hard tasks
- [ ] CTTR-VPS consumes substantially more inference compute
- [ ] The collective planning ablation removes at least part of any CTTR-VPS advantage
- [ ] Hidden/full-judge correctness is lower than visible-test success on at least some tasks

This scorecard records the prior and should not be edited retrospectively after seeing results.

---

# 11. Scientific guardrails

The historical record and future experiment should remain separated:

- preserved historical artifacts are **historical/exploratory evidence**;
- old "~80% success" and "first 4 runs fully correct" statements remain **documentation-only**;
- current configs describe experimental methods but do not establish that those methods were executed;
- visible samples must not be confused with hidden/full-judge correctness;
- candidate count and compute must be reported explicitly;
- future results should come from raw task-level records and be recomputed from those records;
- no single "best" pipeline should be declared without a fixed benchmark population and matched compute budget.

The central decomposition to test is:

````text
Candidate diversity
        ×
Plan diversity
        ×
Execution feedback
        ×
Candidate selection
        ×
Inference compute
````

rather than attributing all performance differences to the CTTR-VPS label itself.

---
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