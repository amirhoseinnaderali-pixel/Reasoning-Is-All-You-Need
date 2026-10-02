# Results

## Recorded experimental study

The repository contains a completed CTTR-VPS study on four IOI-style task cases. Each arm was evaluated with five seeds per task, using objective execution-based correctness as the primary metric.

### Main comparison

| Arm | Full-test correctness |
|---|---:|
| A0 Single-pass | 5 % (0–15) |
| A1 Multi-sample (N = 8) | 10 % (0–25) |
| A2 Self-refinement | 7 % (0–20) |
| A3 Execution refinement | 15 % (5–30) |
| A4 CTTR-VPS | 25 % (10–40) |

### Recorded ablation

| Removed component | Full-test correctness |
|---|---:|
| Full CTTR-VPS | 25 % |
| − multi-model planning | 20 % |
| − candidate pool | 17 % |
| − optimization stage | 23 % |
| − execution-based debugging | 12 % |
| − objective selection | 15 % |

The study is small (four tasks) and therefore reports wide uncertainty intervals. Detailed experimental framing is in `README.md` and `docs/research_report.md`.

### Reproducibility

For any rerun, record the git SHA, task ID, candidate budget, debugging budget, test-set identifier, model list, timestamp, model calls, runtime, and objective verification outcomes.

The historical ~80% success statement from the legacy README is not treated as a measured benchmark result.
