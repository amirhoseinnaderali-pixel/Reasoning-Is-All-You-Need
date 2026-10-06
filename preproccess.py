import json
import re
import time
from string import Template
from ollama_client import chat_completion
from datasets import load_dataset
import requests
import os


def safe_json_parse(raw_output: str):
    """Robustly extract and fix JSON from model output."""
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    cleaned = cleaned.strip()
    cleaned = ''.join(ch for ch in cleaned if ch.isprintable())
    
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode failed:", e)
        cleaned = cleaned.replace(",}", "}")
        cleaned = cleaned.replace(",]", "]")
        try:
            return json.loads(cleaned)
        except Exception:
            return {"error": str(e), "raw_output": raw_output}


def save_checkpoint(data, output_file: str):
    """
    Persist results safely so progress is never lost.
    Writes atomically to avoid corruption on interruptions.
    """
    tmp_file = f"{output_file}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, output_file)


# ============================================================================
# BEST APPROACH: Progressive Disclosure Architecture
# ============================================================================
# مدل اول summary میخونه، بعد problem، بعد details
# داده کامل هست ولی به ترتیب اولویت چیده شده

FEW_SHOT_OUTPUT = """
{
  "core": {
    "title": "E. Memory",
    "contest": "IOI 2010 day 2",
    "one_line": "Match 50 cards (25 letter-pairs A-Y) in ≤100 faceup() calls"
  },
  
  "problem": {
    "given": [
      "50 face-down cards",
      "Each letter A-Y appears exactly twice",
      "Cards shuffled randomly"
    ],
    "goal": "Collect 25 candies by matching all letter pairs",
    "mechanism": "Call faceup(C) to reveal card C; after every 2 calls both flip back",
    "success_condition": "First time both copies of a letter are face-up simultaneously → 1 candy"
  },
  
  "constraints": {
    "critical": [
      "Cards flip back after every 2 faceup() calls",
      "Cannot call faceup() on currently face-up card",
      "Must collect all 25 candies before terminating"
    ],
    "subtasks": {
      "1": {"points": 50, "req": "Any valid strategy (e.g., 2450 calls acceptable)"},
      "2": {"points": 50, "req": "At most 100 faceup() calls total"}
    }
  },
  
  "implementation": {
    "function_to_write": "void play()",
    "available_api": {
      "faceup": {
        "signature": "char faceup(int C)",
        "params": "C: card number (1-50)",
        "returns": "Letter on card C (A-Y)",
        "side_effect": "After 2nd call, both cards flip back"
      }
    },
    "requirements": [
      "Cards are 1-indexed (1 to 50)",
      "Use std::vector<T> not raw arrays",
      "No main() function",
      "Match exact grader signatures"
    ]
  },
  
  "limits": {
    "time": "1 second",
    "memory": "256 megabytes"
  },
  
  "samples": [],
  
  "original_statement": "FULL original problem text preserved here for reference..."
}
"""


def call_google_model(problem_payload: str, api_key: str):
    """
    Uses progressive disclosure: most important info first, details later
    """
    prompt_template = Template("""You are an expert at structuring competitive programming problems for AI planning agents.

### Your Task:
Transform the raw problem JSON into a **progressive disclosure format** where:
1. Most critical info comes first (what to do, key constraint)
2. Problem mechanics come second (how it works)
3. Implementation details come last

### Output Structure (STRICT):

{
  "core": {
    "title": "Problem name",
    "contest": "e.g., IOI 2010 day 2",
    "one_line": "Complete task description in ≤15 words including main constraint"
  },
  
  "problem": {
    "given": ["List", "of", "initial", "conditions"],
    "goal": "Single sentence: what must be achieved",
    "mechanism": "How the system works (API, state changes)",
    "success_condition": "Exact condition for winning/earning points"
  },
  
  "constraints": {
    "critical": ["Rules that affect strategy"],
    "subtasks": {
      "1": {"points": X, "req": "requirement"},
      "2": {"points": Y, "req": "requirement"}
    }
  },
  
  "implementation": {
    "function_to_write": "Exact signature needed",
    "available_api": {
      "function_name": {
        "signature": "full signature",
        "params": "parameter descriptions",
        "returns": "return value description",
        "side_effect": "any side effects"
      }
    },
    "requirements": ["Implementation constraints like indexing, types, etc."]
  },
  
  "limits": {
    "time": "e.g., 1 second",
    "memory": "e.g., 256 megabytes"
  },
  
  "samples": [],
  
  "original_statement": "PRESERVE FULL ORIGINAL TEXT HERE - verbatim copy for reference"
}

### Critical Rules:
1. **core.one_line**: Must be ≤15 words, include the main constraint (e.g., "in ≤100 calls")
2. **problem.given**: Array of SHORT strings, each one fact
3. **problem.mechanism**: How the game/system works in 1-2 sentences
4. **constraints.critical**: Only rules that directly impact algorithm design
5. **implementation.available_api**: Document EACH function with full details
6. **original_statement**: COPY THE ENTIRE original problem text verbatim - DO NOT summarize this field
7. Output ONLY valid JSON, no markdown, no extra text

### Example Output:
$few_shot_output

---

Now process this problem:
$problem_payload
""")
    
    prompt = prompt_template.substitute(
        few_shot_output=FEW_SHOT_OUTPUT,
        problem_payload=problem_payload,
    )

    response = chat_completion(model="deepseek-v3.1:671b-cloud", prompt=prompt, api_key=api_key)

    if isinstance(response, str):
        raw_output = response
    elif hasattr(response, "text"):
        raw_output = response.text
    else:
        raw_output = str(response)
    
    return safe_json_parse(raw_output)


def create_multi_level_views(cleaned_problem):
    """
    Creates different 'views' of the same problem for different stages:
    - planning_view: Ultra-minimal for initial brainstorming
    - algorithm_view: Core logic without implementation details
    - implementation_view: Everything needed to code
    """
    return {
        # ===== VIEW 1: Planning (17 models brainstorm here) =====
        "planning_view": {
            "title": cleaned_problem["core"]["title"],
            "task": cleaned_problem["core"]["one_line"],
            "given": cleaned_problem["problem"]["given"],
            "goal": cleaned_problem["problem"]["goal"],
            "key_rules": cleaned_problem["constraints"]["critical"],
            "target": cleaned_problem["constraints"]["subtasks"]
        },
        
        # ===== VIEW 2: Algorithm Design (after consensus) =====
        "algorithm_view": {
            **cleaned_problem["core"],
            **cleaned_problem["problem"],
            "constraints": cleaned_problem["constraints"],
            "limits": cleaned_problem["limits"]
        },
        
        # ===== VIEW 3: Implementation (final coding) =====
        "implementation_view": cleaned_problem,  # Full structure
        
        # ===== VIEW 4: Raw Original (for reference) =====
        "original": cleaned_problem.get("original_statement", "")
    }


def process_ioi_dataset_problem(problem_row):
    """
    Extracts minimal but complete info from dataset row.
    Preserves original statement for reference.
    """
    structured_payload = {
        "title": str(problem_row.get("title", "")).strip(),
        "contest_date": str(problem_row.get("date", "")).strip(),
        "full_statement": str(problem_row.get("statement", "")).strip(),
        "time_limit": str(problem_row.get("time_limit", "")).strip(),
        "memory_limit": str(problem_row.get("memory_limit", "")).strip(),
        "samples": problem_row.get("samples", []),
        "tags": problem_row.get("tags", []),
    }

    return json.dumps(structured_payload, ensure_ascii=False, indent=2)


def process_dataset_with_resume(
    dataset_name: str,
    api_key: str,
    output_file: str = "cleaned_problems_best.json",
    batch_size: int = 5
):
    """
    Process dataset with:
    - Multi-level view generation
    - Automatic resume
    - Batch saving
    - Full data preservation
    """
    ds = load_dataset(dataset_name)
    problems = ds['train']
    total = len(problems)
    
    cleaned_problems = []
    start_index = 21
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                cleaned_problems = json.load(f)
            start_index = len(cleaned_problems)
            print(f"📂 Resumed from {start_index}/{total}")
        except Exception as e:
            print(f"⚠️ Starting fresh: {e}")
    
    for i in range(91, total):
        retries = 0
        max_retries = 5
        base_delay = 2.0
        
        while retries < max_retries:
            try:
                problem = problems[i]
                
                # Step 1: Clean with model
                cleaned = call_google_model(
                    process_ioi_dataset_problem(problem), 
                    api_key
                )
                
                # Step 2: Generate multi-level views
                multi_view = create_multi_level_views(cleaned)
                
                # Step 3: Add metadata
                result = {
                    **multi_view,
                    "_meta": {
                        "index": i,
                        "problem_url": str(problem.get("problem_url", "")).strip(),
                        "uuid": str(problem.get("data_uuid", "")).strip(),
                    }
                }
                
                cleaned_problems.append(result)
                save_checkpoint(cleaned_problems, output_file)
                print(f"✅ {i+1}/{total} - {cleaned['core']['title']}")
                
                if (i + 1) % batch_size == 0:
                    print(f"💾 Autosave checkpoint at {i+1}/{total}")
                
                break
                
            except requests.exceptions.HTTPError as e:
                if getattr(e.response, "status_code", None) == 429:
                    delay = base_delay * (2 ** retries)
                    retries += 1
                    print(f"⏳ Rate limit. Retry {retries}/{max_retries} in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    print(f"❌ HTTP error at {i+1}: {e}")
                    raise
                    
            except Exception as e:
                delay = base_delay * (2 ** retries)
                retries += 1
                print(f"⚠️ Error at {i+1}: {e}. Retry {retries}/{max_retries}")
                time.sleep(delay)
        
        if retries >= max_retries:
            print(f"❌ Failed at {i+1} after {max_retries} retries")
            break
    
    # Final save
    save_checkpoint(cleaned_problems, output_file)
    
    print(f"\n{'='*60}")
    print(f"✅ Complete! Processed {len(cleaned_problems)}/{total}")
    print(f"{'='*60}")
    
    return cleaned_problems


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    api_key = "AIzaSyDANYHSgMUfyzNNG94L2RaL8kAHh4dkvTM"
    
    cleaned = process_dataset_with_resume(
        dataset_name="HumanLastCodeExam/ioi",
        api_key=api_key,
        output_file="ioi_multi_view.json",
        batch_size=5
    )
    
    # Demo: Show how to use different views
    if cleaned:
        sample = cleaned[0]
        print("\n" + "="*60)
        print("📋 PLANNING VIEW (for 17-model brainstorm):")
        print("="*60)
        print(json.dumps(sample["planning_view"], indent=2, ensure_ascii=False))
        
        print("\n" + "="*60)
        print("🧠 ALGORITHM VIEW (after consensus):")
        print("="*60)
        print(json.dumps(sample["algorithm_view"], indent=2, ensure_ascii=False))
        
        print("\n" + "="*60)
        print("💻 How to access full implementation details:")
        print("="*60)
        print("sample['implementation_view']['implementation']")
        
        print("\n" + "="*60)
        print("📄 Original statement preserved at:")
        print("="*60)
        print("sample['original'][:200] + '...'")