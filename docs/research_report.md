# CTTR-VPS Historical Research Case Study

## Research Question

How does additional inference-time computation change the probability of producing an objectively correct executable solution?

The historical system pursued this question through multi-model planning, multi-stage code generation, optimization, and execution-based debugging for IOI-style programming tasks.

## Hypothesis

The historical project proposed that additional test-time reasoning and verification could improve executable-program quality, while increasing inference cost and latency.

This report does **not** claim that the hypothesis was established.

## Evidence Inventory

The historical repository snapshot preserves four end-to-end task artifact sets:

1. `final_planning_problem_1.json` + `final_codes_problem_1.txt`
2. `final_planning_problem_2.json` + `final_codes_problem_2.txt`
3. `final_planning_problem_3.json` + `final_code_problem_3.txt`
4. `final_planning_problem_4.json` + `final_codes_problem4.txt`

These files were introduced together in the original repository upload commit:

`6501b64e99a017dfbc728d4ca909ba4c93949e87` — **2025-11-16**

The benchmark source used by the historical pipeline is the repository's `ioi_multi_view.json`. The current blob is:

`3c8d546f94d3757ff5beec7f57c8c219bc299202`

The four preserved cases correspond to task indices 0–3 in that artifact:

| Case | Task | Task ID | Planning records | Preserved code variants |
|---|---|---|---:|---:|
| H1 | E. Memory | `ps_34ed2ceeb12ceab464ae8af48cca5bbf8f3ae887` | 4 | 16 |
| H2 | C. Quality of Living | `ps_87a3a133e68621915d1276d8980fac0ace7d09d2` | 4 | 14 |
| H3 | E. Friend | `ps_0003bd5cea4f3cb318943bdc2815aa339784a8b3` | 2 | 10 |
| H4 | A. Arranging Shoes | `ps_ff9715751d51f006e17870ce3d9742747df2c674` | 4 | 14 |

**Derived artifact totals:** 14 preserved planning records and 54 labeled code variants across the four cases.

These counts are derived directly from committed JSON/code artifacts. They are not estimates of model calls.

## Historical Method

The original `main.py`, `planning.py`, `cpp_pipe.py`, `optim.py`, and `_30step.py` define the historical pipeline.

### Planning

The historical planning module declares a 34-model planning pool spanning Gemini, DeepSeek, Qwen, GPT-OSS, GLM, Kimi, and MiniMax families.

The preserved planning JSON files contain only the successful/retained planning records that were written to those artifacts. They should therefore not be interpreted as a complete call log.

### Code generation

The historical `main.py` configured five sequential code-generation phases, each requesting 20 attempts and passing the previous phase's candidates forward.

The repository does not preserve the corresponding per-phase output directories in Git. Therefore **100 requested generation calls per case must not be reported as 100 observed successful calls**.

### Optimization

The historical optimizer ran four sequential optimization stages over a strong-model pool. The source code explicitly selected the **first successful optimization result** at each stage. This is an implementation detail of the historical system, not a validated scientific selection rule.

### Debugging / execution

The historical debugger used a sequential model pool and execution feedback. The source code preserved in the original snapshot monitored only the first three supplied tests inside one execution-feedback path. Later hardening explicitly identified this as a methodological defect.

## Results

### What is directly supported

The repository directly preserves:

- four named IOI task cases;
- planning artifacts for all four cases;
- 54 labeled generated-code variants across those cases;
- executable C++ solution artifacts;
- the historical pipeline source that generated/planned/optimized/debugged them.

### What is **not** directly supported

The repository does **not** preserve a raw judge/evaluator record establishing:

- final hidden/full-judge acceptance for each case;
- per-task score;
- success rate;
- exact successful API call count;
- exact total API call count;
- end-to-end runtime for each case;
- token usage or monetary cost;
- a compute-matched baseline comparison.

The repository's `.gitignore` explicitly excludes `results/`, `experiments/runs/`, and `*.log`, and the later research audit confirms that historical headline results were based on only a very small number of completed runs.

Accordingly, the historical headline statement in the old README that the “first 4 runs” were fully correct is classified here as **DOCUMENTATION ONLY**, not as a measured result.

Likewise, the old “~80% success” claim is **DOCUMENTATION ONLY** and is not reproduced as a result.

## Evidence Levels

| Evidence level | Historical evidence in repository |
|---|---|
| **RAW EXECUTION EVIDENCE** | No preserved final judge log, score file, runtime log, or execution record was found. |
| **DERIVED FROM RAW RESULTS** | 4 task artifact sets, 14 retained planning records, 54 labeled code variants, task mapping, and source-level pipeline configuration. |
| **HISTORICAL/EXPLORATORY** | The four-task end-to-end artifact corpus showing planning and code-generation outputs. |
| **DOCUMENTATION ONLY** | “first 4 runs fully correct” and “~80% success” statements from the old README. |

## Analysis

The strongest defensible observation is about **engineering execution**, not measured correctness.

The repository shows that the historical system was actually taken beyond a toy prompt:

- it created structured multi-view IOI task representations;
- it generated multiple algorithm plans from a diverse model pool;
- it generated and retained many C++ candidates;
- it ran multi-stage optimization;
- it integrated execution-based debugging.

For the four preserved cases, the planning artifacts converge on recognizable algorithmic structures such as deterministic memory matching, binary-search-plus-prefix-sums for Quality of Living, tree DP for Friend, and Fenwick-tree/inversion-count reasoning for Arranging Shoes.

However, the historical repository does not retain enough judge telemetry to measure whether the additional compute improved objective correctness. In particular, the missing judge logs prevent a defensible success-rate estimate and prevent a compute-normalized comparison against C0-style baselines.

## Conclusion

**Historical conclusion:** the repository contains concrete evidence that CTTR-VPS was implemented and exercised on at least four IOI task cases, producing multi-model planning artifacts and a substantial set of generated C++ candidates.

The historical evidence does **not** establish a general correctness rate, does not establish superiority over simpler methods, and does not support the old ~80% claim as a measured result.

A portfolio description should therefore emphasize the **research system and preserved execution artifacts**, not an unsupported accuracy number.

## Limitations

1. The historical task sample is only four preserved cases.
2. Final hidden/full-judge outcomes are not preserved in the repository.
3. Exact model-call counts are missing from historical logs.
4. End-to-end runtime and token/cost telemetry are missing.
5. The historical pipeline used changing model pools and legacy selection behavior rather than a frozen compute-normalized comparison.
6. One historical execution-feedback path inspected only the first three supplied tests.
7. Historical artifacts come from a legacy pipeline whose source was subsequently hardened; the current hardened EXP-001 is a separate protocol and has **not** been executed.
8. The preserved artifact timestamp is the original repository upload date, 2025-11-16; it should not be presented as the exact execution date of every API/model call.

## Portfolio Interpretation

The strongest honest portfolio framing is:

> **Built and exercised a multi-model test-time reasoning pipeline for IOI-style C++ program synthesis, preserving multi-model planning and 54 generated-code artifacts across four task cases; subsequently audited the system's selection, evaluation, budgeting, and reproducibility weaknesses.**

This is supported by repository artifacts without claiming an unsupported benchmark success rate.
