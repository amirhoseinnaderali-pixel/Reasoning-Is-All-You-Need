# Experimental Protocol and Recorded Study

## Compute-budget comparison

The completed study evaluates controlled budgets across the documented arms. For reproducibility and future reruns, preserve the following compute-budget structure:
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

The completed study uses a fixed four-task population, five seeds per arm, objective execution-based evaluation, and reported uncertainty. Future replications should preserve the same controls or explicitly document any change.
