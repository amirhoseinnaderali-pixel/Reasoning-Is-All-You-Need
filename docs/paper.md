# CTTR-VPS: Collective Test-Time Reasoning for Verified Program Synthesis

## Abstract
This project studies an inference-time pipeline for generating C++ programs for algorithmic problems. It combines structured problem representations, multi-model planning, candidate generation, refinement, optimization, compilation, testing, and model-assisted debugging. [EXPERIMENT REQUIRED] Controlled comparisons, full benchmark evaluation, ablations, and cost analysis have not yet established empirical benefit.

## Introduction
LLM-generated programs may contain algorithmic, syntax, runtime, or output errors. The research question is whether additional test-time computation and execution feedback improve reliability without modifying model weights.

## Problem Definition
Given an algorithmic problem, constraints, and tests, generate a correct C++ program. Generation, compilation, visible-test success, and hidden full-judge acceptance are distinct outcomes.

## Related Work
[LITERATURE REVIEW REQUIRED] Verify original publications covering pass@k, sampled program synthesis, self-refinement, multi-agent systems, execution-guided generation, and test-time compute scaling.

## Methodology
The legacy system performs multi-view preprocessing, multi-model planning, repeated candidate generation, optimization, compilation, testing, and model-assisted debugging. The new harness adds baseline interfaces and structured result records.

## Experimental Design
Use a fixed, versioned benchmark subset. Match or report model/provider versions, sampling parameters, call budgets, refinement rounds, compiler, timeout, and memory. Repeat stochastic methods and retain raw artifacts.

## Baselines
Single-pass, multi-sample, self-refinement, execution-based refinement, and CTTR-VPS.

## Metrics
Primary: problem-level solved rate under a stated evaluation regime. Secondary: compile rate, visible-test correctness, pass@k, model calls, latency, tokens, and cost.

## Results
[EXPERIMENT REQUIRED] No controlled results are reported.

## Ablations
[EXPERIMENT REQUIRED] Remove or reduce planning diversity, multi-view preprocessing, refinement, execution feedback, optimization, and final verification; vary model diversity and refinement rounds.

## Error Analysis
Categorize incorrect algorithms, preprocessing errors, compile/runtime errors, timeout, memory failure, wrong output, provider failure, and optimization regression.

## Limitations
Visible tests are incomplete evidence; provider/model versions change; test-time compute increases latency/cost; current planning is not calibrated consensus; token/cost telemetry is incomplete.

## Future Work
Add dataset hashing and stable IDs, full-judge integration, provider-level token/cost telemetry, stronger sandbox validation, matched-budget baselines, ablations, and uncertainty analysis.

## Conclusion
CTTR-VPS is a framework for empirically studying multi-stage inference-time program synthesis. Its effectiveness remains an empirical question.