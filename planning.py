from google import genai
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama_client import chat_completion



PLANNING_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-06-05",
    "gemini-pro-latest",
    "deepseek-v3.1:671b-cloud",      # 671B پارامتر - غول reasoning
    "kimi-k2:1t-cloud",
    "qwen3-coder:480b-cloud",        # 480B - مخصوص کد ولی planning هم خوبه
    "gpt-oss:120b-cloud", 
    "glm-4.6:cloud",                 # GLM-4 - balanced و سریع
    "minimax-m2:cloud",           
    
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.0-pro-exp",
    "gemini-2.0-pro-exp-02-05",
    "gemini-exp-1206",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-exp",
    
    # Tier C
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite-preview",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-2.5-computer-use-preview-11-11",
    

]


from google import genai
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama_client import chat_completion

# 10 مدل متنوع و قوی


EXPECTED_PLANNING_KEYS = {
    "algorithm",
    "approach",
    "time_complexity",
    "space_complexity",
}


def parse_planning_output(raw_text: str):
    """Parse model output to structured JSON"""
    if not raw_text:
        return None, "Empty response"

    candidate = raw_text.strip()
    
    # حذف markdown blocks
    candidate = candidate.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # سعی کن JSON رو از متن extract کنی
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None, "Response is not valid JSON"
        snippet = match.group(0)
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            return None, f"JSON decode error: {exc}"

    if not isinstance(parsed, dict):
        return None, "Response is not a JSON object"

    missing_keys = EXPECTED_PLANNING_KEYS - parsed.keys()
    if missing_keys:
        return None, f"Missing keys: {', '.join(sorted(missing_keys))}"

    normalized = {
        "algorithm": str(parsed["algorithm"]).strip(),
        "approach": str(parsed["approach"]).strip(),
        "time_complexity": str(parsed["time_complexity"]).strip(),
        "space_complexity": str(parsed["space_complexity"]).strip(),
    }
    return normalized, None


def call_google_model(model_name: str, prompt: str, api_key: str, timeout: int = 60):
    """Call Google Gemini model with timeout"""
    import threading
    
    client = genai.Client(api_key=api_key)
    result_container = {"response": None, "error": None, "done": False}
    
    def api_call():
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            result_container["response"] = response
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            result_container["done"] = True
    
    # Start API call in a thread
    thread = threading.Thread(target=api_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if not result_container["done"]:
        return {
            "model": model_name,
            "output": "",
            "success": False,
            "error": f"Google API call timeout after {timeout}s"
        }
    
    if result_container["error"]:
        return {
            "model": model_name,
            "output": "",
            "success": False,
            "error": result_container["error"]
        }
    
    try:
        parsed_output, error = parse_planning_output(result_container["response"].text)
        if parsed_output is None:
            return {
                "model": model_name,
                "output": result_container["response"].text,
                "success": False,
                "error": error
            }

        return {
            "model": model_name,
            "output": parsed_output,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "model": model_name,
            "output": "",
            "success": False,
            "error": str(e)
        }


def call_ollama_model(model: str, prompt: str, api_key: str):
    """Call Ollama model"""
    try:
        output = chat_completion(model=model, prompt=prompt, api_key=api_key)
        parsed_output, error = parse_planning_output(output)
        if parsed_output is None:
            return {
                "model": model,
                "output": output,
                "success": False,
                "error": error
            }

        return {
            "model": model,
            "output": parsed_output,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "model": model,
            "output": "",
            "success": False,
            "error": str(e)
        }


def save_to_file(results: list, filename: str,output_dir: str):
    """Save successful results to JSON file"""
    successful_results = []
    for result in results:
        if not result["success"]:
            print(f"❌ Error calling {result['model']}: {result['error']}")
            continue
        
        structured_entry = {
            "model": result["model"],
            **result["output"],
        }
        successful_results.append(structured_entry)

    if not successful_results:
        print("⚠️  No successful results to save.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir+"/"+filename, "w", encoding="utf-8") as f:
        json.dump(successful_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(successful_results)}/{len(results)} results to {filename}")


def call_models_parallel(prompt: str, api_key_google: str, api_key_ollama: str):
    """Call all models in parallel with timeout"""
    import time
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    
    results = [None] * len(PLANNING_MODELS)
    API_TIMEOUT = 60  # 60 seconds timeout per API call

    def _dispatch(model: str, index: int):
        # Add small delay to avoid overwhelming API
        time.sleep(index * 0.3)  # Stagger requests: 0s, 0.3s, 0.6s, 0.9s...
        try:
            if model.startswith("gemini-"):
                print(f"🔄 Calling Google model: {model}")
                return index, call_google_model(model, prompt, api_key_google)
            else:
                print(f"🔄 Calling Ollama model: {model}")
                return index, call_ollama_model(model=model, prompt=prompt, api_key=api_key_ollama)
        except Exception as e:
            print(f"❌ Error in thread for {model}: {str(e)}")
            return index, {
                "model": model,
                "output": "",
                "success": False,
                "error": f"Thread exception: {str(e)}"
            }

    # Reduce max workers to avoid rate limiting (was 8, now 4)
    max_workers = min(4, len(PLANNING_MODELS))
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_dispatch, model, idx): (idx, model) for idx, model in enumerate(PLANNING_MODELS)}
        
        for future in as_completed(futures, timeout=API_TIMEOUT * len(PLANNING_MODELS)):
            try:
                index, result = future.result(timeout=API_TIMEOUT)
                results[index] = result
            except FuturesTimeoutError:
                idx, model = futures[future]
                print(f"⏱️ Timeout for model {model} (index {idx})")
                results[idx] = {
                    "model": model,
                    "output": "",
                    "success": False,
                    "error": f"API call timeout after {API_TIMEOUT}s"
                }
            except Exception as e:
                idx, model = futures[future]
                print(f"❌ Exception for model {model}: {str(e)}")
                results[idx] = {
                    "model": model,
                    "output": "",
                    "success": False,
                    "error": f"Exception: {str(e)}"
                }

    return [result for result in results if result is not None]


def planning(problem: str, api_key_google: str, api_key_ollama: str, filename: str,previous_planning_results: list,output_dir: str):
 
    planning_template = """
You are an expert competitive programmer analyzing IOI-level problems.

PROBLEM:
{problem}

previous_planning_results:
{previous_planning_results}

YOUR TASK:
Design ONE complete algorithmic approach. Think deeply about:

1. PROBLEM TYPE: What category? (Graph, DP, Greedy, Math, etc.)
2. ALGORITHM CHOICE: Which specific algorithm/data structure?
3. KEY INSIGHTS: What's the core trick or observation?
4. EDGE CASES: What could break the solution?
5. COMPLEXITY: Time and space requirements

Be specific and avoid obvious brute-force approaches.

OUTPUT FORMAT (CRITICAL):
Return ONLY a valid JSON object with NO markdown, NO explanations:
{{
  "algorithm": "concise algorithm name",
  "approach": "2-4 sentence description of key steps and insights",
  "time_complexity": "Big-O notation",
  "space_complexity": "Big-O notation"
}}
"""
    
    planning_prompt = planning_template.format(problem=problem,previous_planning_results=previous_planning_results)
    #print(planning_prompt)
    print(f"\n{'='*60}")
    print("🚀 Starting Planning Phase")
    print(f"{'='*60}\n")
    
    results = call_models_parallel(planning_prompt, api_key_google, api_key_ollama)
    
    save_to_file(results, filename,output_dir)
    
    #Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"✅ Planning Complete: {successful}/{len(results)} models succeeded")
    print(f"{'='*60}\n")
    
    return results