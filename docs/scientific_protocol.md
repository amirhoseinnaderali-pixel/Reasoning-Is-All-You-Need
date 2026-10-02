# CTTR-VPS Scientific Protocol

## Research question

How does additional inference-time computation change the probability of producing an objectively correct executable solution?

## Frozen conditions

EXP-001 uses C0 Single-pass, C1 Multi-sample, C2 Self-refinement, C3 Execution-based refinement, and C4 CTTR-VPS. The exact condition budgets, seed, model revision, generation parameters, Docker execution limits, and result schema are frozen in `configs/exp001.yaml`.

Visible tests are used only for generation-time execution feedback and objective candidate selection. Hidden tests are loaded only after the final candidate is selected.

## Condition semantics

C0 generates one candidate.

C1 generates eight independent candidates and objectively selects among them using visible tests.

C2 generates one candidate and performs up to three model-only self-refinement rounds without execution feedback.

C3 generates one candidate and performs up to three refinement rounds using visible execution feedback.

C4 performs planning, candidate generation, deterministic deduplication, visible-test objective selection, and execution-based debugging.

## Compute accounting

Every model call is counted. Requested output-token limits and provider-reported token usage are recorded when available. Debug/refinement steps are explicit. The runner fails closed when call or token budgets are exceeded.

## Objective correctness

Compilation, runtime failure, timeout, memory-limit failure, sandbox failure, and wrong output are separate failure modes. A model/API failure cannot become a source candidate. Passing visible tests is never reported as hidden/full-judge correctness.

## Real execution

Real mode requires frozen benchmark materialization, credentials, Docker, a digest-pinned execution image, frozen model/runtime settings, and an external hidden-test artifact. No mock fallback is permitted.

## Current status

The software hardening and fail-closed execution infrastructure are implemented. The repository cannot enter real EXP-001 execution until the external benchmark materialization and hidden-test artifact are supplied and hashed. No empirical result is inferred from historical runs.
