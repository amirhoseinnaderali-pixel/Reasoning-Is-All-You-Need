# CTTR-VPS
## Collective Test-Time Reasoning for Verified Program Synthesis
**Status:** research prototype; controlled evaluation pending.

CTTR-VPS studies whether additional inference-time computation across model-generated plans and candidate programs, combined with iterative refinement and execution-based checking, improves C++ program synthesis for algorithmic problems.

### Research question
Under a fixed benchmark and comparable inference budgets, does multi-model planning plus iterative candidate refinement and execution-based verification improve problem-level correctness relative to simpler generation strategies?

### Hypothesis
**H1 — to be evaluated:** structured planning diversity, multiple candidates, refinement, and execution checks improve correctness over single-pass generation under comparable compute conditions. This is a hypothesis, not a result.

### Results
**Not yet evaluated.** Historical artifacts do not constitute a controlled baseline/ablation result. Passing visible samples does not establish hidden/full-judge acceptance.

### Reproducibility
Python 3.10+, g++ C++17, and configured credentials are required.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    python scripts/run_experiment.py --config configs/baseline_single_pass.yaml
    python scripts/evaluate.py --results results

Set GOOGLE_API_KEY and OLLAMA_API_KEY (or plural forms) in .env. Never commit credentials.

See docs/architecture.md, docs/experiments.md, docs/result_schema.md, docs/research_positioning.md, docs/paper.md, and CITATION.cff.

### History
The project originated in the repository previously named Reasoning-Is-All-You-Need; historical modules and artifacts are retained during migration.