# Research Portfolio Audit

## Scope

This audit covers the public repositories visible on the author's GitHub profile at the time of review. There are 36 public repository entries; several are duplicate/alias repos (for example `agentCoder`/`agent_coder`, `Phi-to-Qwen-Knowledge-Distillation`/`Phi-4-to-Qwen-Knowledge-Distillation`, and two similarly named medical-DPO repos), so they are treated as one research thread where appropriate.

## Evidence standard

Each project is classified at four levels:

- **Implemented** - the architecture/code exists.
- **Executed** - the code/notebook has recorded execution artifacts.
- **Measured** - a numerical metric or reproducible benchmark is present.
- **Demonstrated** - the experiment actually supports the stated research claim, with an appropriate comparison/baseline.

A README claim is not treated as a measured result unless it is supported by artifacts in the repository.

## Main research tree

```
FOUNDATION
  ML / RL / NLP basics
      |
      +--> Parameter-efficient adaptation
      |       |
      |       +--> T5: Full FT vs Soft Prompt vs Adapter vs AdapterHub vs LoRA
      |       +--> Llama / Qwen LoRA + QLoRA
      |       +--> DPO
      |       +--> Distributed training
      |       +--> Knowledge distillation
      |       +--> Curriculum learning
      |
      +--> Reasoning at inference time
              |
              +--> N-Queens prompt-time iterative refinement
              +--> Planner / Logic / Judge / Replanner
              +--> Graph-of-Thought representations
              +--> Multi-agent generation
              +--> Code execution feedback
              +--> Multi-round planning / consensus
              +--> Test-time compute for algorithmic code generation
```

## Project-by-project research interpretation

### 1. Reasoning-Is-All-You-Need
**Question:** Can additional inference-time computation across multiple models, planning rounds, optimization, execution, and debugging improve complex code generation without updating model weights?

**Implemented:** Multi-stage planning, 20+ candidate generation per phase, multi-stage optimization, executable C++ sandbox, sequential debugging with many models.

**Executed evidence:** Real planning JSONs, generated C++ files, and end-to-end artifacts for IOI-style problems are present. The README reports four fully correct initial benchmark runs.

**Measured:** Evidence exists for individual successful runs, but a benchmark-wide success rate is not established in the repo.

**Not demonstrated yet:** A controlled comparison against:
1. one-shot single-model generation,
2. single-model self-refinement,
3. independent best-of-N sampling,
4. multi-model consensus without iterative redistribution,
5. the complete proposed pipeline.

**Highest-value next experiment:** Run a fixed IOI evaluation set and report pass rate, compile rate, average inference cost, latency, and success as a function of test-time compute budget.

**Important implementation issue found:** `optim.py` gathers multiple optimized candidates but selects the first successful result rather than the candidate with the best measured test/time/memory score. This weakens the experimental claim.

**Security issue found:** API credentials were hard-coded in source. A separate branch, `security-and-research-audit`, moves them to environment variables. Previously exposed credentials should still be rotated/revoked because Git history may retain them.

### 2. LLM_reasoning_solve_NQueen_with_no_code
**Question:** Can a small frozen model behave more like a search procedure when the previous response is fed back as context?

**Evidence:** The repo gives a verifiable N-Queens trace and explicitly records a zero-conflict final state for the example.

**Status:** Executed demonstration; not yet a systematic experiment.

**Missing:** Success rate over N, number of refinement rounds, token/compute budget, and comparison against one-shot prompting.

### 3. graph_reasoning
**Question:** Can reasoning traces from multiple models be represented as a graph so that repeated ideas, dependencies, contradictions, and communities become explicit?

**Evidence:** Actual graph artifacts exist. One artifact contains 17 solutions, 212 thought nodes and 1,339 edges; thought types include hypotheses, verification, synthesis, obstacles, and conclusions.

**Status:** Strong systems/representation artifact.

**Missing:** Does graph-based aggregation improve task accuracy or reduce reasoning cost? A graph visualization alone does not establish that.

### 4. Reasoning-Agent---Multi-Stage-AI-Reasoning-System
**Question:** Can a planner/solver/judge/replanner loop improve answer quality?

**Implementation:** Planner -> Logic Agent -> Judge -> conditional Replanner.

**Status:** Prototype.

**Missing:** Human or task-grounded evaluation showing that replanning beats the original answer. The 0.3 threshold is hard-coded and the judge uses the same small model family.

### 5. multi-agent-react-sandbox
**Question:** Can multiple agents generate, execute, observe, and refine code in a sandbox?

**Evidence:** A 24-agent run artifact exists.

**Critical audit finding:** The stored results contain 24 solutions but four are Gemini quota errors, and all recorded outputs are empty. The sandbox implementation waits for the process to exit before sending stdin; programs that wait for input can therefore time out before receiving it. The stored benchmark artifact is not valid evidence of successful test-time reasoning.

**Next step:** Fix stdin handling first, then rerun the benchmark and compare one-shot vs ReAct refinement.

### 6. DPO-Training-Benchmark-Performance-Comparison-of-Distributed-Training-Methods-for-LLMs
**Question:** Which distributed training strategy gives the best throughput/resource trade-off for DPO on dual T4 GPUs?

**Measured result:** TinyLlama-1.1B, Anthropic HH-RLHF, dual T4 benchmark:
- Standard: 2.67 samples/s, 336.79 s, 2.48 GB
- FSDP: 1.74 samples/s, 517.47 s, 2.91 GB
- ZeRO-2: 3.71 samples/s, 242.60 s, 3.62 GB
- ZeRO-2 optimized: 4.31 samples/s, 208.76 s, 3.62 GB
- ZeRO-3: 0.82 samples/s, 1097.17 s, 4.20 GB

**Status:** Strong measured systems experiment.

**Boundary:** This result is specific to the tested model, workload, GPU setup, and configuration; it should not be generalized to all model sizes.

### 7. dpo-mistal_7b-med-deepspeed-zero2-wandb
**Question:** Can DPO training be run efficiently for a medical adaptation task?

**Status:** Feasibility/training pipeline, not a demonstrated medical improvement experiment.

**Missing:** Base vs DPO evaluation on a held-out medical benchmark with a task-specific metric.

### 8. qwen-math-reasoning
**Question:** Can a small Qwen model be adapted to step-by-step mathematical reasoning with LoRA under T4 constraints?

**Evidence:** 7,473 train / 1,319 test GSM8K examples were loaded; training ran on a Tesla T4; LoRA used about 0.87% trainable parameters; both fine-tuned and fresh base-model testing code exists.

**Status:** Executed training + qualitative evaluation.

**Missing:** Aggregate before/after GSM8K accuracy. The repository does not establish that fine-tuning improved the model.

### 9. Fine-tune-Qwen2.5-Coder-14B-on-HumanEval-MBPP-using-LoRA
**Question:** Can a 14B code model be adapted with LoRA on a T4 to improve coding performance?

**Evidence:** Base-model tests, mixed HumanEval/MBPP training pipeline, LoRA configuration, checkpoints and post-training sample generation are implemented.

**Status:** Executed adaptation demo.

**Missing:** Pass@1/Pass@k on HumanEval/MBPP before vs after, using code execution rather than only visual inspection.

### 10. DistillLlama-Curriculum
**Question:** Can a small/quantized coding model gain capability progressively through curriculum stages and teacher distillation?

**Design:** Foundation -> Algorithms -> Debugging -> Advanced reasoning, with optional teacher-generated data.

**Status:** Strong research hypothesis and engineering implementation.

**Missing:** Controlled stage-by-stage held-out evaluation. README's indicative accuracy figures are explicitly not a robust benchmark.

### 11. Phi-to-Qwen-Knowledge-Distillation
**Question:** Can reasoning/coding capability from a 14B Phi teacher be transferred into a 0.5B Qwen student using sparse soft labels and selective fine-tuning?

**Implementation:** Top-k teacher distributions, hard-label CE + KL distillation, last-four-layer training, DeepSpeed/FP16 memory optimization.

**Status:** Feasibility implementation. The notebook code supports training, but the repository does not establish student capability improvement.

**Missing:** Teacher vs student vs student-SFT-only on a held-out coding benchmark. Also inspect whether the tiny number of teacher samples and the near-zero distillation-loss behavior seen in earlier artifacts indicate a weak KD signal.

### 12. FineTune_T5_on_imdb
**Question:** How does parameter-efficient adaptation compare with full fine-tuning on accuracy and trainable parameter count?

**Measured evidence:** The notebooks contain explicit validation results. The strongest observed entries include roughly 90.0% for full fine-tuning, 90.6% for one Adapter setup, 90.4% for AdapterHub, and about 89.5% for LoRA; soft-prompt variants range lower.

**Status:** Strong quantitative foundational experiment.

**Contribution:** This is the clearest early origin of the later PEFT / efficient adaptation thread.

### 13. vulnerability-detection-ai
**Question:** Can GraphCodeBERT + LoRA provide resource-efficient multi-class vulnerability detection?

**Status:** Substantial engineering, monitoring, and training infrastructure.

**Missing:** Reproducible precision/recall/F1 tables and controlled base-vs-LoRA-vs-distillation comparisons. README metrics are not enough by themselves.

### 14. RadVision-7B / Lingshu7B_Finetuning
**Question:** Can a medical VLM be adapted for chest-X-ray report generation under constrained memory?

**Design:** vision adaptation -> multimodal projector alignment -> LLM adaptation with QLoRA/checkpointing.

**Status:** Architecture/feasibility prototype.

**Missing:** Reliable completed training run and held-out radiology evaluation metrics. The notebook currently has infrastructure/dataset-loading evidence but not a defensible performance result.

### 15. Multimodal_Rag
**Question:** Can text and image retrieval be combined in a local multimodal RAG pipeline?

**Status:** System implementation.

**Missing:** Retrieval recall, grounded answer accuracy, hallucination/faithfulness and text-only vs multimodal ablation.

### 16. AutoTune-Research-Assistan / auto-finetune-llm
**Question:** Can model, dataset and fine-tuning method selection be partially automated from Hugging Face, Kaggle and arXiv information?

**Status:** Tooling/prototype.

**Missing:** Recommendation accuracy/relevance evaluation against expert choices or historical successful fine-tuning setups.

### 17. Creating-a-Q-A-dataset-with-the-DeepSeek-671B-free-API
**Question:** Can a large model be used to automatically transform source text into structured Q&A training data?

**Status:** Data-generation infrastructure.

**Missing:** factual correctness, diversity, duplication and human-quality checks.

### 18. codechain / agentCoder / agent_coder
**Question:** Can multiple model calls improve code generation and user-facing coding assistance?

**Status:** Application engineering / product prototype.

**Missing:** controlled comparison with a single strong model. `agentCoder` and `agent_coder` appear to be duplicate/renamed copies.

### 19. ox / ox-vs-claude-vs-grok
**Question:** Can generated Python solutions be optimized and compared by correctness, runtime and memory?

**Evidence:** Actual per-problem result JSONs and aggregate README metrics exist; the reported aggregate is 14/14 for OX and Claude vs 12/14 for Grok across the five-problem suite.

**Status:** Quantitative benchmark artifact.

**Boundary:** Only five problems; timing differences are therefore an early signal, not a broad model ranking.

### 20. SmartRAG
**Question:** Can a RAG system dynamically decide whether retrieval is needed?

**Audit finding:** The implementation has likely routing bugs. FAISS `similarity_search_with_score` returns a distance-like score for common FAISS setups, yet the code treats larger values as stronger similarity; additionally, the conditional route maps the retrieval-needed case to the direct-LLM node. This undermines the current result.

**Next step:** Fix the routing semantics and add a retrieval-needed/not-needed test set.

### 21. agent1
Minimal single-node LangGraph + Ollama example. Useful as an architectural stepping stone, not research evidence.

### 22. DistillLlama-LoRA-Finetune
Short LoRA/SFT tutorial on a 4-bit DeepSeek-R1-Distill-Llama model.

**Status:** Training implementation, no research-grade evaluation.

### 23. 1-fine-tuning-structured-output-llama2
Structured-output fine-tuning / model packaging experiment.

**Status:** Infrastructure/demo; no defensible aggregate performance result found.

### 24. Audio2Entity
Whisper -> BERT NER pipeline.

**Status:** Application prototype; no NER benchmark reported.

### 25. Diabetes-Prediction-Project
Classical ML/XGBoost health prediction.

**Evidence:** README reports 75% test accuracy.

**Status:** Applied ML baseline/project, not part of the modern LLM research spine.

### 26. Sentiment-Analysis-of-COVID-19-Vaccine-Tweets
Word2Vec + CNN/LSTM/FNN sentiment pipeline.

**Status:** Older NLP foundation/application work. Reported accuracy/F1 exist, but this is not central to the later LLM direction.

### 27. RL
SAC, behavioral cloning, DQN, REINFORCE and PPO course implementations.

**Contribution:** Establishes RL foundations that later connect conceptually to preference optimization and reasoning, but these notebooks are not themselves LLM-RL research.

### 28. ML_course_practices
Broad classical ML/deep learning/recommendation/anomaly-detection coursework.

**Contribution:** Foundation rather than research evidence.

### 29. deepresearch-agent
Very small skeleton; not yet evidence of a functioning deep-research system.

## What the whole portfolio is actually becoming

The strongest common thread is **capability improvement under compute constraints**.

There are two mechanisms:

### A. Add compute at inference time without changing weights

```
small/frozen model
    ↓
iterative prompt refinement
    ↓
multiple reasoning paths
    ↓
structured reasoning graph
    ↓
multiple agents
    ↓
code execution / environment feedback
    ↓
multi-round consensus
```

Research question:

> When model weights are fixed, how should additional inference-time computation be allocated so that reasoning becomes more reliable?

### B. Change only a small part of the model efficiently

```
base model
   ↓
full FT vs PEFT comparison
   ↓
LoRA / QLoRA
   ↓
DPO
   ↓
distributed DPO
   ↓
knowledge distillation
   ↓
curriculum + distillation
```

Research question:

> Under limited memory and compute, how can a small model acquire new reasoning/task capabilities with the least parameter and training cost?

### C. The bridge between A and B

The most interesting unifying direction is:

> **Efficient reasoning under compute constraints: combining parameter-efficient adaptation with test-time computation to improve reasoning quality per unit of compute.**

This bridge is better supported by the portfolio than treating every application area (medical, RAG, vulnerability, sentiment, diabetes) as a separate specialization.

## Highest-value open experiments

1. **Test-time compute scaling curve**
   - baseline: one-shot model
   - + self-refinement
   - + best-of-N
   - + multi-model consensus
   - + execution feedback
   - measure accuracy vs number of model calls/tokens/time.

2. **Reasoning-agent ablation**
   - Planner only
   - Planner + Logic
   - Planner + Logic + Judge
   - full Replanner
   - use tasks with objective correctness, not subjective judge scores.

3. **Coding model before/after**
   - Qwen2.5-Coder base
   - LoRA
   - curriculum
   - curriculum + distillation
   - evaluate HumanEval+/MBPP/IOI-style held-out tasks with execution-based Pass@1.

4. **KD ablation**
   - student SFT only
   - SFT + full teacher targets
   - SFT + top-k KD
   - vary temperature and alpha/beta
   - compare quality and memory/training cost.

5. **Joint adaptation + test-time compute**
   - same student model
   - LoRA only
   - test-time reasoning only
   - LoRA + test-time reasoning
   - report quality / token budget / latency / VRAM.

## Final research identity supported by the current evidence

The portfolio is strongest when described as a progression from:

**efficient model adaptation -> reasoning improvement -> test-time computation -> executable feedback -> resource-aware reasoning systems.**

It is not yet a collection of completed papers. It is a coherent set of implementations and a few strong quantitative experiments pointing toward one research question.

The main missing ingredient is not another project. It is **controlled experiments that turn the strongest implementations into evidence**.
