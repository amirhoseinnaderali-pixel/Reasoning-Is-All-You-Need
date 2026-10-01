# Reasoning-Is-All-You-Need

An end-to-end test-time reasoning system for generating and debugging C++ solutions to IOI-style programming tasks.

## Research question

> How does additional inference-time computation change the probability of producing an objectively correct executable solution?

## Pipeline

`planning → multi-candidate code generation → optimization → objective candidate selection → iterative execution-based debugging`

The system is designed as a compute-allocation study: additional candidate generation and debugging are treated as inference-time resources.

## Research hardening

This branch makes the following changes:
- removes hard-coded API credentials from active source files;
- removes first-success/first-candidate selection bias from the optimizer path;
- evaluates optimizer candidates with objective execution tests before final debugging;
- records the selected candidate and every candidate's test performance;
- makes the long stage delay configurable instead of fixed;
- documents the distinction between exploratory historical runs and controlled results.

## Current evidence

The repository contains historical successful runs, including four early fully correct examples reported in the original project. These are useful demonstrations but are not enough to establish a general benchmark success rate.

## Controlled research direction

Compare fixed test-time compute budgets such as 1, 5, and 20 generated candidates, with and without iterative debugging.

Primary metric: objective verifier pass rate.

Secondary metrics: model calls, debugging steps, latency, and cost proxy.

## Run requirements

Set credentials outside the repository:

```bash
export GOOGLE_API_KEYS='key1,key2,...'
export OLLAMA_API_KEYS='key1,key2,...'
export PIPELINE_STAGE_DELAY_SECONDS=0
```

Then install the dependencies listed by the original project and run the desired pipeline task.

## Status

Implemented: objective candidate selection, credential cleanup, configurable delay, research audit and reporting protocol.

Not executed here: the expensive model/API benchmark.