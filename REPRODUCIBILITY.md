# Reproducibility

Set credentials outside the repository:

```bash
export GOOGLE_API_KEYS='key1,key2,...'
export OLLAMA_API_KEYS='key1,key2,...'
export PIPELINE_STAGE_DELAY_SECONDS=0
```

Install:

```bash
pip install google-genai requests datasets psutil
```

Run a pipeline task and keep the complete output directory.

The branch now writes `candidate_selection.json` containing the objective selection decision before final debugging.

For repeated experiments record:
- git SHA
- problem/task ID
- candidate budget
- debugging budget
- test-set identifier
- model list
- timestamp