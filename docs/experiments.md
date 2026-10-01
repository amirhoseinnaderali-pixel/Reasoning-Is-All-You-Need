# Experimental Protocol

## Compute-budget comparison

Evaluate controlled budgets such as:
- 1 generated candidate
- 5 generated candidates
- 20 generated candidates
- 20 candidates plus iterative debugging

Do not compare methods while silently changing model pools, temperatures, problem subsets, compiler, timeout, memory cap, or test set.

## Required reporting

Primary metric:
- task-level objective verifier pass rate

Secondary:
- compilation rate
- supplied-test pass rate
- model calls
- debugging steps
- wall-clock time
- API/token cost proxy when available

## Ablations

1. No multi-model planning.
2. No iterative refinement.
3. No execution feedback.
4. No optimization.
5. Reduced candidate count.
6. Reduced debug budget.
7. No multi-view preprocessing.
8. Final verification disabled (diagnostic only; never treated as a correctness benchmark).

## Scientific integrity

Historical successful runs are exploratory evidence. They should not be mixed with controlled benchmark results.

Before paper claims, run the complete matrix over a fixed problem population and report uncertainty.
