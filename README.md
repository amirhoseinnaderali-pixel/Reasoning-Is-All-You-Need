# CTTR-VPS
## Collective Test-Time Reasoning for Verified Program Synthesis

CTTR-VPS investigates whether **multi-model planning, candidate selection, and execution-based debugging** can improve the correctness of generated C++ solutions to IOI-style programming problems.

**Status:** The repository records a completed study. The public repository does not include the full task-level raw run archive, so aggregate accuracy claims cannot currently be independently reconstructed.

## How it works

```text
Problem preprocessing
        ↓
Multi-model planning
        ↓
Generate multiple C++ candidates
        ↓
Compile, run visible tests, and select
        ↓
Debug using compiler/runtime/wrong-answer feedback
        ↓
Final solution
```

The full pipeline uses additional inference-time calls rather than fine-tuning the models. Candidate correctness is evaluated by executing code, not by asking an LLM to judge its own answer.

## Experimental comparison

| Arm | Method | What it tests |
|---|---|---|
| **A0** | Single-pass generation | One model call, no feedback |
| **A1** | Multi-sampling | Generate 8 candidates and select using visible tests |
| **A2** | Self-refinement | Critique and rewrite without execution feedback |
| **A3** | Execution-based refinement | Iterative debugging using execution feedback |
| **A4** | **CTTR-VPS** | Multi-model planning, candidate generation, objective selection, and debugging |

The goal is to compare correctness against the added inference cost, not to assume that the most complex pipeline is automatically best.

## Benchmark and metrics

- **Tasks:** 4 IOI-style C++ problem cases.
- **Seeds:** 5 per task and arm in the registered design.
- **Primary metric:** full-test correctness — passing all tests.
- **Secondary metrics:** subtask score, pass@k, model calls, tokens, wall-clock latency, and cost proxy.
- **Evaluation:** executable code is tested in a sandbox. Visible tests may guide selection/debugging; hidden or full-judge tests are reserved for final evaluation.

With only four distinct tasks, statistical power is limited and conclusions should be treated as exploratory.

## Results and evidence

The study is recorded as completed, but the full task-level run archive is not committed publicly. For that reason, this README does not claim a numerical winner or reproduce aggregate accuracy percentages that cannot be checked from the public artifacts. The older historical files are retained as provenance and should not be confused with the current controlled comparison.

## Run

Install dependencies and configure the credentials for the model backend you use:

```bash
pip install -r requirements.txt
export GOOGLE_API_KEYS='your-key'
export OLLAMA_API_KEYS='your-key'
```

Run a configured method and summarize saved results:

```bash
python scripts/run_experiment.py --config configs/cttr_vps.yaml
python scripts/evaluate.py --results results
```

Before running, check that the dataset path in the selected config exists and that the config matches the backend and budget you intend to use. Keep API keys outside the repository; never commit credentials.

## Key files

- `configs/` — experiment and baseline configurations
- `cttr_vps/` — pipeline methods, evaluation, and result schema
- `scripts/run_experiment.py` — run a configured method
- `scripts/evaluate.py` — summarize saved result records
- `docs/architecture.md` and `docs/research_report.md` — implementation details and study context
- `REPRODUCIBILITY.md` — notes for recording reproducible runs
