# Research Audit — Reasoning-Is-All-You-Need

## Research question

> How does additional inference-time computation change the probability of producing an objectively correct executable solution?

The architecture combines candidate generation, iterative refinement, optimization, and execution-based debugging.

## What the historical project demonstrated

- The repository contains an end-to-end C++ generation pipeline for IOI-style tasks.
- The historical README reports four early fully correct runs.
- The pipeline records intermediate code candidates and uses execution feedback during debugging.

These observations do not establish an 80% benchmark success rate.

## Methodological issues found

1. Hard-coded Google and Ollama credentials were present in source files.
2. The optimizer selected the first successful model response rather than an objectively verified best candidate.
3. The main pipeline then used candidate index 0 from the optimizer output, preserving the first-result bias.
4. One execution-feedback path evaluated only the first three tests.
5. The pipeline has large fixed sleeps that add wall-clock delay without being part of the algorithmic compute budget.
6. Historical headline results are based on a very small number of completed runs.
7. There is no compute-normalized ablation of the full pipeline.

## Research redesign

Candidate selection now uses objective execution tests before the 30-step debugger.

Future evaluation should compare controlled compute budgets:
- 1 candidate / 1 execution path
- 5 candidates
- 20 candidates
- 20 candidates + iterative debugging

For each budget report:
- task-level solved rate
- test pass rate
- number of model calls
- execution/debug steps
- wall-clock time
- API cost proxy when available

## Hypothesis

> Increasing test-time computation should improve the probability of solving an executable programming task, with diminishing returns and increasing inference cost.

This is falsifiable.

## Important distinction

LLM self-reported quality or consensus is not the primary metric. The primary metric is whether the generated program passes an objective verifier.