# reasoning is all you neeeed ::)




# intuition 
Imagine a classroom where every student is trying to solve the same problem.
Some students are naturally strong and write great answers right away.
Others might not be the best at solving the whole problem, but they’re really good at reading everyone else’s reasoning and spotting the strongest idea.
Now picture all these students sharing their thought process out loud.
They start learning from each other, picking up better ideas, fixing mistakes, and slowly moving toward a solution they all agree is the best.
Even the weaker student—who can’t fully solve the problem alone—can still choose the best answer from the group.
That’s exactly what happens with the models here:
they propose ideas, read each other’s reasoning, filter, refine, and converge on the strongest solution.
And that collective “brainstorming” is what makes the system reach a much better final answer.

This project is an advanced system for **automatic C++ code generation** for competitive programming (IOI‑style) problems using multiple AI models.

At a high level:

- **Live demo**: [`https://huggingface.co/spaces/amirhoseinnaderali/coder`](https://huggingface.co/spaces/amirhoseinnaderali/coder)  
- **Benchmark**: evaluated on https://huggingface.co/datasets/HumanLastCodeExam/ioi 
- **Quality signal**: the **first 4 runs** on this benchmark are **fully correct** and show **high‑quality C++ code** generated end‑to‑end  
- **Test‑time computation**: uses a **multi‑round, multi‑model planning loop** where models share and refine each other’s solutions until they reach a consensus  
- **Claim**: given enough compute and API budget, this system can realistically reach **~80% success** on this IOI benchmark; current limitations come mainly from **API rate limits and cost**, not from the core algorithmic capabilities.

---

### Table of Contents

- **Introduction**
- **Key Features**
- **System Architecture**
- **Installation & Setup**
- **Project Structure**
- **How to Use**
- **File Descriptions**
- **Pipeline Process**
- **AI Models Used**
- **Hugging Face Demo & IOI Dataset**
- **Limitations & Notes**
- **Usage Examples**
- **Resources & References**
- **Author & License**

---

### Introduction

This system uses advanced AI techniques and multiple different models to:

1. **Algorithm Planning** – Generate an algorithmic plan for solving a problem using 30+ AI models
2. **Code Generation** – Convert the algorithmic plan into C++ code
3. **Optimization** – Optimize the code for speed and memory
4. **Debugging & Testing** – Debug and test the code using 30 different models

The goal is to go from an IOI problem statement to a high‑quality, optimized, and tested C++ solution with minimal human intervention.

---

### Key Features

#### 1. Multi‑Stage Planning
- **Phase 1**: Generate initial algorithmic plans with 30+ AI models in parallel  
- **Phase 2**: Refine and improve the plans using results from Phase 1  
- **Phase 3**: Produce a final, detailed algorithmic plan

#### 2. Five‑Phase Code Generation
- Each phase uses **20 parallel attempts** with different models
- Each phase uses the previous phase’s results as input
- All code variants are saved for analysis and fallback

#### 3. Multi‑Stage Optimization
- 4 consecutive optimization stages
- Different API keys/models can be used in each stage
- The best code is selected from all optimization results

#### 4. Debugging with 30 Models
- 30 different models are used for debugging sequentially
- The pipeline stops once all tests pass
- The final code is saved as the canonical solution

#### 5. Progressive Disclosure Architecture
- Information is revealed in stages according to need
- Multiple “views” for different stages (planning, algorithm, implementation)
- Prompts are optimized to match the current stage

---

### System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: IOI Problem                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 0: Preprocessing (preproccess.py)                    │
│  - Structure the problem                                    │
│  - Create multiple views                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1–3: Planning (planning.py)                          │
│  - Phase 1: Initial plans (30+ models in parallel)          │
│  - Phase 2: Plan refinement                                 │
│  - Phase 3: Final detailed plan                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4–8: Code Generation (cpp_pipe.py)                   │
│  - Phase 1: Initial code (20 models in parallel)            │
│  - Phase 2–5: Progressive improvements (20 models each)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 9: Optimization (optim.py)                           │
│  - 4 optimization stages                                    │
│  - Uses stronger models per stage                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 10: Debug & Test (_30step.py)                        │
│  - 30 models sequentially                                   │
│  - Stops on success                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT: Final C++ Code                   │
└─────────────────────────────────────────────────────────────┘
```

---

### Installation & Setup

#### Requirements

```bash
# Python 3.8 or higher
python --version

# Install required libraries
pip install google-genai requests datasets psutil
```

#### Optional: Install Ollama

If you want to use local Ollama models:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download required models (example)
ollama pull deepseek-v3.1:671b-cloud
```

#### API Keys Configuration

Set your API keys in the corresponding files:

- `main.py`: lists `api_key_google_list` and `api_key_ollama_list`
- `planning.py`: uses the key passed into its functions
- `cpp_pipe.py`: list `api_key_google_list`

---

### Project Structure

```text
nvidia/
├── main.py                 # Main orchestrator for the whole pipeline
├── planning.py             # Algorithm planning module
├── preproccess.py          # Preprocessing & problem structuring
├── ollama_client.py        # Ollama API client
├── cpp_pipe.py             # C++ code generation & improvement pipeline
├── optim.py                # Code optimization module
├── _30step.py              # 30‑step debugging module
├── ioi_multi_view.json     # Preprocessed IOI problems (multi‑view)
├── README.md               # Main README (this file, English)
└── pipeline_results/       # Pipeline outputs
    ├── phase1.txt          # Phase 1 codes
    ├── phase2.txt          # Phase 2 codes
    ├── phase3.txt          # Phase 3 codes
    ├── phase4.txt          # Phase 4 codes
    ├── phase5.txt          # Phase 5 codes
    ├── input_codes_*.txt   # Optimized code variants
    ├── planning_results*.json  # Planning results
    └── final_code.cpp      # Final solution
```

---

### How to Use

#### Simple Usage

```python
from main import daily_using

# Run the whole pipeline for a sample problem
daily_using()
```

#### Advanced Usage

```python
import asyncio
from main import plan_to_code
from planning import planning

# 1. Planning
planning_results = planning(
    problem="Your problem statement",
    api_key_google="your-google-key",
    api_key_ollama="your-ollama-key",
    filename="planning_results1.json",
    previous_planning_results=[],
    output_dir="output"
)

# 2. Code generation
plan = planning_results[0]["algorithm"]  # Use the first plan
tests = [{"input": "...", "expected_output": "..."}]

asyncio.run(plan_to_code(
    plan=plan,
    problem="Your problem statement",
    output_dir="output",
    tests=tests,
    speed=1,
    memory=512
))
```

#### Batch Processing

```python
from main import test_pipeline

# Process all problems in ioi_multi_view.json
test_pipeline()
```

---

### File Descriptions

#### `main.py`

- **Responsibility**: Orchestrates the entire pipeline
- **Key functions**:
  - `phaze1_planning()`: Planning – phase 1
  - `phaze2_planning_multiple()`: Planning – phase 2
  - `phaze3_planning_multiple()`: Planning – phase 3
  - `plan_()`: Run the full planning process
  - `plan_to_code()`: Convert a plan into C++ code
  - `optimizerrr()`: Run optimization
  - `test_pipeline()`: Batch processing over dataset
  - `daily_using()`: Simple one‑shot usage

#### `planning.py`

- **Responsibility**: Generate algorithmic plans using multiple AI models
- **Features**:
  - Uses 30+ different AI models
  - Parallel execution for speed
  - JSON output parsing and validation
  - Saves results to JSON files
- **Key functions**:
  - `planning()`: Main planning function
  - `call_models_parallel()`: Parallel model invocation
  - `call_google_model()`: Gemini call helper
  - `call_ollama_model()`: Ollama call helper
  - `parse_planning_output()`: Parse model outputs
  - `save_to_file()`: Save results

#### `preproccess.py`

- **Responsibility**: Preprocess IOI problems & build structured multi‑view representations
- **Features**:
  - Progressive Disclosure Architecture
  - Multi‑level views (`planning_view`, `algorithm_view`, `implementation_view`)
  - Batch processing with resume
  - Safe checkpointing
- **Key functions**:
  - `process_dataset_with_resume()`
  - `call_google_model()`
  - `create_multi_level_views()`
  - `process_ioi_dataset_problem()`
  - `safe_json_parse()`
  - `save_checkpoint()`

#### `ollama_client.py`

- **Responsibility**: Communication with the Ollama API
- **Features**:
  - Streaming response support
  - Timeout handling
  - Parsing JSON and text responses
- **Key functions**:
  - `chat_completion()`
  - `_collect_streaming_response()`

#### `cpp_pipe.py`

- **Responsibility**: Main pipeline for C++ code generation and improvement
- **Features**:
  - `CppSandbox`: Safe environment for compilation and execution
  - Code generation with 20 models in parallel
  - Iterative bug fixing and improvement
  - Performance optimization (time & memory)
  - Resource management (timeout, memory limit)
- **Key classes & functions**:
  - `CppSandbox`
  - `phase1_initial_code_generation()`
  - `phase2_iterative_fixing()`
  - `phase3_4_parallel_optimization()`
  - `call_model()`
  - `extract_code_from_response()`

#### `optim.py`

- **Responsibility**: Multi‑stage code optimization with strong models
- **Features**:
  - Parallel optimization with multiple strong models
  - Several sequential optimization stages
  - Save results of each stage
  - Select the best final code
- **Key functions**:
  - `optimizer()`
  - `optimize_best_code()`
  - `optimize_with_single_model()`
  - `load_codes_from_file()`

#### `_30step.py`

- **Responsibility**: Debugging and bug fixing with 30 different AI models
- **Features**:
  - Sequential debugging with 30 models
  - Stop early if all tests pass
  - Save final fixed code
- **Key functions**:
  - `_30step()`
  - `debug_with_all_models()`
  - `debug_and_fix_with_model()`
  - `load_test_cases_from_json()`

---

### Pipeline Process

#### Step 0: Preprocessing

1. Load the problem from a dataset or JSON file  
2. Structure the problem using an AI model  
3. Create multiple views:
   - `planning_view`: minimal information for algorithm planning
   - `algorithm_view`: core logic and constraints
   - `implementation_view`: full technical details

#### Steps 1–3: Planning

1. **Phase 1**: Send `planning_view` to 30+ AI models in parallel  
2. **Phase 2**: Refine the initial plans using phase 1 outputs  
3. **Phase 3**: Produce a final, detailed algorithmic plan  

**Output**: `planning_results1.json`, `planning_results2.json`, `planning_results3.json`

#### Steps 4–8: Code Generation

1. **Phase 1**: Generate initial C++ code from the plan (20 models in parallel)  
2. **Phase 2**: Improve the code using phase 1 results  
3. **Phase 3**: Further improvements  
4. **Phase 4**: Advanced refinements  
5. **Phase 5**: Final refinement  

**Output**: `phase1.txt` … `phase5.txt`

#### Step 9: Optimization

1. **Stage 1**: Optimize with the first API key / model set  
2. **Stage 2**: Optimize with a second configuration  
3. **Stage 3**: Further optimization  
4. **Stage 4**: Final optimization  

**Output**: `input_codes_1.txt` … `input_codes_4.txt`

#### Step 10: Debug & Test

1. Compile the C++ code  
2. Run tests (from the IOI dataset or custom tests)  
3. On failure, request a fix from a model  
4. Repeat with up to 30 different models  

**Output**: `final_code.cpp`

---

### AI Models Used

#### Planning Models (30+)

- **Gemini**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`, …
- **DeepSeek**: `deepseek-v3.1:671b-cloud`
- **Qwen**: `qwen3-coder:480b-cloud`
- **GPT-OSS**: `gpt-oss:120b-cloud`
- **GLM**: `glm-4.6:cloud`
- **Minimax**: `minimax-m2:cloud`
- **Kimi**: `kimi-k2:1t-cloud`

#### Code Generation Models (20+)

- Largely similar to the planning models  
- Focused more on code‑specialized variants

#### Debugging Models (30)

- Full list is in `_30step.py`  
- Includes Gemini, DeepSeek, Qwen, and others

---

### Hugging Face Demo & IOI Dataset

You can try an online demo of this system here:

- **Hugging Face Space**: [`https://huggingface.co/spaces/amirhoseinnaderali/coder`](https://huggingface.co/spaces/amirhoseinnaderali/coder)

In the current demo, the system is being **evaluated on IOI‑style tasks**.  
The **first 4 results shown in the Space come directly from the official IOI dataset** and are **fully correct solutions** generated by the pipeline. They are used as an initial benchmark to demonstrate that the models can correctly handle real IOI tasks and produce accepted C++ solutions.

One of the key problems used for evaluation is:

- **IOI 2010 Day 1 – Problem C: “Quality of Living”**  
  (Codeforces mirror: [`https://ioi.contest.codeforces.com/group/32KGsXgiKA/contest/103756/problem/C`](https://ioi.contest.codeforces.com/group/32KGsXgiKA/contest/103756/problem/C))

#### Dataset / Problem Description – “Quality of Living”

The **Quality of Living** problem models a city as an \\(R \\times C\\) grid of blocks:

- Each block \\((a, b)\\) has a unique **quality rank** from 1 to \\(R \cdot C\\)  
  - `1` = best quality  
  - `R*C` = worst quality  
- The city planning department wants to find a **rectangular sub‑grid** of size \\(H \times W\\) (north–south × west–east), where \\(H\\) and \\(W\\) are **odd** and do not exceed \\(R\\) and \\(C\\).
- For each candidate \\(H \times W\\) rectangle, we consider all \\(H \cdot W\\) quality ranks inside it and compute the **median**:
  - Because \\(H \cdot W\\) is odd, the median is the value \\(m\\) such that  
    the number of ranks **better than** \\(m\\) equals the number of ranks **worse than** \\(m\\).

The task is to implement a procedure:

```text
rectangle(R, C, H, W, Q)
```

where:

- `R, C` – grid dimensions  
- `H, W` – sub‑rectangle dimensions (both odd)  
- `Q[a][b]` – quality rank at row `a`, column `b`  

The function must return:

- The **best (numerically smallest) possible median quality rank** among all \\(H \times W\\) rectangles.

This problem is used as a **benchmark dataset** in this project.  
The pipeline:

1. Reads the **problem statement and constraints** (e.g., \\(R, C \le 3000\\))  
2. Designs an algorithm (typically using prefix sums and binary search over the rank, or similar techniques)  
3. Generates optimized C++ code that satisfies the time limit (5 seconds) and memory constraints (256 MB)  
4. Validates the solution on multiple official test cases.

The **first 4 evaluation results** in the Hugging Face Space correspond to real test instances of this dataset and show that the system:

- Understands the problem semantics  
- Produces correct algorithms  
- Generates **fully accepted C++ implementations** from end to end.

Based on the current architecture and the quality of the generated solutions, **I claim that this system can realistically reach around 80% success on this IOI benchmark**, under sufficient compute and API budget.  
However, **strict API rate limits and cost constraints** make running the full benchmark at scale time‑consuming in practice.  
Even so, the **4 completed runs** already illustrate the **high quality of the generated C++ code** and give an early signal of the system’s potential performance on the full benchmark.

#### Test‑Time Computation & Multi‑Model Consensus (New Idea)

This project also explores a **new idea in test‑time computation** based on **iterative, multi‑model planning**:

- In the *first* planning round, we send the problem to a pool of (for example) 20 models.  
- Typically, only a subset (e.g., **4 out of 20 models**) produce a fully correct or clearly superior plan in this first round.  
- Instead of stopping there, we **share the best answers/plans across all models** and run **several additional planning rounds**.  
- In each round, every model sees:
  - The original problem
  - Its own previous attempts
  - The best ideas and partial solutions from *all other* models
- Over 3 or more iterations, this process acts like a **test‑time “deliberation and aggregation” loop**, where models:
  - Correct each other’s mistakes  
  - Borrow strong sub‑ideas (invariants, data structures, complexity arguments)  
  - Gradually move toward a **shared consensus algorithm**.

In practice, this creates a **powerful form of test‑time computation**: we are not just sampling one model once, but running a *small multi‑agent system* at test time, where:

- Multiple models think in parallel  
- Their solutions are merged, compared, and redistributed  
- Later rounds are strictly better informed than earlier ones  

This consensus‑driven, multi‑round planning is what you see in this system’s behavior on the IOI dataset and in the **first 4 fully correct solutions** displayed in the Hugging Face Space  
([`https://huggingface.co/spaces/amirhoseinnaderali/coder`](https://huggingface.co/spaces/amirhoseinnaderali/coder)).

---

### Limitations & Notes

#### Technical Limitations

1. **API Rate Limits** – You may hit rate limits depending on provider  
2. **Timeouts** – Each API call has a timeout (typically 60–120 seconds)  
3. **Memory** – Default execution memory limit is 512 MB  
4. **Time** – Default execution time limit is 5 seconds per run

#### Important Notes

1. **API Keys** – Ensure your keys are valid and active  
2. **Ollama** – If using Ollama, the local server must be running  
3. **g++** – Required to compile the generated C++ code  
4. **Disk Space** – Pipeline outputs can become large over time  
5. **Network** – Internet access is required for remote API calls

#### Optimization Techniques

1. **Parallel Execution** – Heavy use of concurrency to speed up runs  
2. **Caching** – Intermediate results are saved to disk  
3. **Resume** – Processing can resume from checkpoints after failures  
4. **Error Handling** – Robust error handling to keep the pipeline running

---

### Usage Examples

#### Example 1: Simple Usage

```python
from main import daily_using

# Run the pipeline for a sample IOI problem
daily_using()
```

#### Example 2: Process a Specific Problem

```python
import json
import asyncio
from main import plan_to_code, plan_

# 1. Planning (for a specific index in the IOI dataset)
plan_(index=0, output_dir="my_output")

# 2. Load plans
with open("my_output/planning_results3.json", "r") as f:
    plans = json.load(f)

# 3. Code generation
plan = plans[0]
tests = [{"input": "...", "expected_output": "..."}]

asyncio.run(plan_to_code(
    plan=plan,
    problem="Your IOI problem",
    output_dir="my_output",
    tests=tests,
    speed=1,
    memory=512
))
```

---

### Resources & References

- **Google Gemini API**: [`https://ai.google.dev/`](https://ai.google.dev/)
- **Ollama Documentation**: [`https://ollama.com/docs`](https://ollama.com/docs)
- **IOI Problems (Official)**: [`https://ioinformatics.org/`](https://ioinformatics.org/)
- **IOI 2010 Day 1 – Quality of Living (Codeforces mirror)**:  
  [`https://ioi.contest.codeforces.com/group/32KGsXgiKA/contest/103756/problem/C`](https://ioi.contest.codeforces.com/group/32KGsXgiKA/contest/103756/problem/C)
- **Hugging Face Demo Space**:  
  [`https://huggingface.co/spaces/amirhoseinnaderali/coder`](https://huggingface.co/spaces/amirhoseinnaderali/coder)

---

### Author

- **Amirhosein Nadeerali**

---

### License

This project is intended for **educational and research use**.

---

### Acknowledgements

Special thanks to the developers of the AI models and libraries used in this project.


