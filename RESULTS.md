# Results

## Recorded experimental study

The repository records a completed CTTR-VPS study on four IOI-style task cases, with objective execution-based correctness as the primary evaluation.

The public repository does not contain the task-level raw execution archive needed to independently reconstruct every aggregate percentage in the historical summary. To avoid publishing an internally unreconciled denominator or derived percentage, the numerical result tables are intentionally omitted from this public file.

### What is publicly established

- The experiment definition contains five arms: single-pass, multi-sample, self-refinement, execution-based refinement, and the full CTTR-VPS pipeline.
- Evaluation is execution-based rather than LLM-judge-based.
- The project explicitly separates the current study from the older historical artifact corpus.
- The older historical "~80% success" statement is not treated as a measured benchmark result.

### Reproducibility boundary

For a new run, record the git SHA, task ID, candidate budget, debugging budget, test-set identifier, model list, timestamp, model calls, runtime, and objective verification outcomes.

The current public repository should be treated as a methodological and provenance record rather than a source from which task-level aggregate percentages can be recomputed.