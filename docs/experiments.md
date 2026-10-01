# Experimental Protocol

## Baselines
| Method | Multi-model planning | Multiple candidates | Refinement | Execution feedback | Optimization |
|---|---:|---:|---:|---:|---:|
| Single-pass | No | No | No | Final evaluation | No |
| Multi-sample | No | Yes | No | Candidate evaluation | No |
| Self-refinement | No | One evolving candidate | Yes | No during rounds | No |
| Execution-refine | No | One evolving candidate | Yes | Yes | No |
| CTTR-VPS | Yes | Yes | Yes | Yes | Yes |

## Ablations
| Variant | Multi-model | Multi-view | Refinement | Execution feedback | Optimization | Verification |
|---|---:|---:|---:|---:|---:|---:|
| Single pass | No | Fixed | No | No | No | Yes |
| No planning diversity | No | Yes | Yes | Yes | Yes | Yes |
| No multi-view | Yes | No | Yes | Yes | Yes | Yes |
| No refinement | Yes | Yes | No | Yes | Yes | Yes |
| No execution feedback | Yes | Yes | Yes | No | Yes | Yes |
| No optimization | Yes | Yes | Yes | Yes | No | Yes |
| Reduced model pool | Reduced | Yes | Yes | Yes | Yes | Yes |
| Reduced refinement | Yes | Yes | Reduced | Yes | Yes | Yes |
| Full CTTR-VPS | Yes | Yes | Yes | Yes | Yes | Yes |

## Controlled protocol
1. Pin dataset version/hash and exact problem IDs.
2. Distinguish visible samples from official/full-judge evaluation.
3. Hold compiler, standard, timeout, memory, and output normalization constant.
4. Record model/provider versions, sampling parameters, candidate counts, refinement rounds, and stopping criteria.
5. Report model calls, wall time, and token/API cost when available.
6. Repeat stochastic methods and aggregate at problem level.
7. Preserve configs, raw outputs, logs, and result.json.
8. Use uncertainty intervals and paired tests before inferential claims.

## Metrics
Primary: problem-level solved rate under a stated evaluation regime.
Secondary: compile rate, supplied-test pass rate, execution success, pass@k, calls/problem, refinement rounds, wall time, tokens, and cost.
Failures: preprocessing error, incorrect algorithm, compile error, runtime error, timeout, memory failure, wrong output, provider failure, no candidate, optimization regression.

## Results
Not yet evaluated. Existing artifacts are historical evidence, not a controlled benchmark comparison.