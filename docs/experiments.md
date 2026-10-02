# Experimental Protocol

## Research question

How does additional inference-time computation change the probability of producing an objectively correct executable solution?

## Frozen EXP-001 conditions

| Condition | Controlled inference-time budget | Candidate selection |
|---|---:|---|
| C0 | 1 model call | final candidate |
| C1 | 8 model calls | objective visible-test selection |
| C2 | 4 model calls | final self-refined candidate |
| C3 | 4 model calls | final execution-refined candidate |
| C4 | 13 model calls | planning + 8 candidates + objective selection + up to 3 debug calls |

The exact seed, model revision, token parameters, execution image, runtime limits, and budget enforcement are in `configs/exp001.yaml`.

## Evaluation order

Generation and refinement receive visible tests only. The final candidate is selected before hidden tests are loaded. Hidden evaluation is a separate execution step and is never used to generate, select, optimize, or debug a candidate.

## Reporting

Primary:
- task-level hidden-test solved rate

Secondary:
- compilation rate
- visible-test pass rate
- hidden failure-mode distribution
- model calls and token usage
- refinement/debug steps
- wall-clock time
- compute/accuracy trade-off
- paired per-task/per-seed comparison against C0
- uncertainty intervals

Do not rank or declare a winning method before real execution. Historical runs are not controlled benchmark evidence.
