

import os
import subprocess
import tempfile
import shutil
import time
import asyncio
import json
import re
import psutil
import signal
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

from ollama_client import chat_completion
from google import genai
api_key_google_list =[
        "AIzaSyBXC7krDh4mvI4VPKFUHpmkDrEcigOE00o",
        "AIzaSyAiSR_exmQehaC7Q0HPnuQUhr0S9jCtQFs",
        "AIzaSyAVhlHdikARNiTbyJLEBtExGBJPTCWucOg",
        "AIzaSyBt9wnZwb6gGm13gXIDLAs01JuF3PoSnBw",
    ]


def call_model(model: str, prompt: str,api_key_google: str,api_key_ollama: str, timeout: int = 120) -> dict:
    import time
    import threading
    global _current_google_api_key_index
    
    if model.startswith("gemini-"):
        # Use threading for timeout (works in all environments)
        result_container = {"response": None, "error": None, "done": False}
        
        def api_call():
            try:
                client = genai.Client(api_key=api_key_google)
                response = client.models.generate_content(
                    model=model, 
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
            return {"success": False, "output": "", "error": f"Google API call timeout after {timeout}s"}
        
        if result_container["error"]:
            return {"success": False, "output": "", "error": result_container["error"]}
        
        return {"success": True, "output": result_container["response"].text}
        
    else:
        # For non-gemini models, use ollama as before (already has timeout)
        try:
            output = chat_completion(model=model, prompt=prompt,api_key=api_key_ollama)
            return {"success": True, "output": output}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}




IMPLEMENTATION_MODELS=IMPLEMENTATION_MODELS = [
    
"gpt-oss:120b-cloud",
"gemini-2.5-flash" ,                # قدرتمند و متعادل
    "glm-4.6:cloud",
    "minimax-m2:cloud",
    "deepseek-v3.1:671b-cloud",
    "qwen3-coder:480b-cloud",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-pro",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    
    "gemini-2.5-flash-preview-05-20",
    "gemini-flash-latest",
    "gemini-2.0-flash-exp"
    ]
# ============================================================================
# Data Classes for Tracking
# ============================================================================

@dataclass
class ExecutionMetrics:
    """Metrics for a single execution"""
    success: bool
    output: str
    error: str
    time_ms: float
    memory_mb: float
    timeout: bool
    iteration: int
    model_used: str
    test_passed: bool
    timestamp: str


@dataclass
class PhaseResult:
    """Result from a pipeline phase"""
    phase_name: str
    start_step: int
    end_step: int
    code: list[str]
    metrics: List[ExecutionMetrics]
    duration_seconds: float


# ============================================================================
# C++ Sandbox Executor
# ============================================================================

class CppSandbox:
    """Secure C++ code executor with resource limits"""
    
    def __init__(self, timeout: int = 5, memory_limit_mb: int = 512):
        """
        Initialize sandbox with limits
        
        Args:
            timeout: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
        """
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.temp_dir = None
        self.source_file = None
        self.binary_file = None
        self.process = None
        
    def _setup_temp_dir(self):
        """Create temporary directory for compilation and execution"""
        self.temp_dir = tempfile.mkdtemp(prefix="cpp_sandbox_")
        self.source_file = os.path.join(self.temp_dir, "solution.cpp")
        self.binary_file = os.path.join(self.temp_dir, "solution")
        
    def compile(self, cpp_code: str) -> Tuple[bool, str]:
        """
        Compile C++ code
        
        Args:
            cpp_code: C++ source code as string
            
        Returns:
            (success: bool, error_message: str)
        """
        try:
            # Setup temporary directory
            self._setup_temp_dir()
            
            # Write code to file
            with open(self.source_file, 'w') as f:
                f.write(cpp_code)
            
            # Compile with g++
            compile_cmd = [
                'g++',
                '-std=c++17',
                '-O2',
                '-Wall',
                '-Wextra',
                self.source_file,
                '-o',
                self.binary_file
            ]
            
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                error_msg = result.stderr if result.stderr else "Compilation failed"
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"
    
    def execute(self, input_data: str = "") -> Dict:
        """
        Execute compiled binary safely with resource limits
        
        Args:
            input_data: Input to pass to the program via stdin
            
        Returns:
            Dictionary with execution results and metrics
        """
        if not self.binary_file or not os.path.exists(self.binary_file):
            return {
                'success': False,
                'output': '',
                'error': 'Binary not found. Run compile() first.',
                'time_ms': 0.0,
                'memory_mb': 0.0,
                'timeout': False
            }
        
        try:
            # Start timing
            start_time = time.perf_counter()
            
            # Prepare process with resource limits
            env = os.environ.copy()
            
            # Start process
            self.process = subprocess.Popen(
                [self.binary_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.temp_dir,
                env=env,
                preexec_fn=self._set_resource_limits
            )
            
            # Monitor memory in separate thread
            max_memory_mb = [0.0]
            memory_exceeded = [False]
            
            def monitor_memory():
                try:
                    proc = psutil.Process(self.process.pid)
                    while self.process.poll() is None:
                        try:
                            mem_info = proc.memory_info()
                            current_mb = mem_info.rss / (1024 * 1024)
                            max_memory_mb[0] = max(max_memory_mb[0], current_mb)
                            
                            # Kill if memory limit exceeded
                            if current_mb > self.memory_limit_mb:
                                memory_exceeded[0] = True
                                self.process.kill()
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            break
                        time.sleep(0.01)
                except:
                    pass
            
            import threading
            monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
            monitor_thread.start()
            
            # Execute with timeout
            try:
                stdout, stderr = self.process.communicate(
                    input=input_data.encode('utf-8'),
                    timeout=self.timeout
                )
                
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                
                output = stdout.decode('utf-8', errors='replace')
                error = stderr.decode('utf-8', errors='replace')
                
                success = self.process.returncode == 0 and not memory_exceeded[0]
                
                if memory_exceeded[0]:
                    error = f"Memory limit exceeded ({self.memory_limit_mb} MB)"
                
                return {
                    'success': success,
                    'output': output,
                    'error': error,
                    'time_ms': round(execution_time_ms, 2),
                    'memory_mb': round(max_memory_mb[0], 2),
                    'timeout': False
                }
                
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                
                return {
                    'success': False,
                    'output': '',
                    'error': f'Execution timeout ({self.timeout}s)',
                    'time_ms': round(execution_time_ms, 2),
                    'memory_mb': round(max_memory_mb[0], 2),
                    'timeout': True
                }
                
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Execution error: {str(e)}',
                'time_ms': 0.0,
                'memory_mb': 0.0,
                'timeout': False
            }
    
    def _set_resource_limits(self):
        """Set resource limits for the child process (Unix only)"""
        try:
            import resource
            
            # Set CPU time limit (seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
            
            # Set memory limit (bytes)
            memory_bytes = self.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # Set file size limit (prevent large output files)
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
            
        except (ImportError, ValueError):
            # resource module not available or limits not supported
            pass
    
    def cleanup(self):
        """Remove temporary files and kill any running processes"""
        # Kill process if still running
        if self.process and self.process.poll() is None:
            try:
                self.process.kill()
                self.process.wait()
            except:
                pass
        
        # Remove temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
        
        self.temp_dir = None
        self.source_file = None
        self.binary_file = None
        self.process = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ============================================================================
# API Configuration and Utilities
# ============================================================================

       

    
_GOOGLE_API_KEY = (
    
        "AIzaSyD4-OYglZP9aqgtvLiJ5zLdWWmWMYMWENQ", 
        "AIzaSyAcuYSRxIu7Ikq5oaiYSp85ZhD6ned5GWI",
        "AIzaSyB3geZNufz1qM92gAxJpIzK0M3FOyzoStk",
        "AIzaSyDANYHSgMUfyzNNG94L2RaL8kAHh4dkvTM",
)
_OLLAMA_API_KEY = (
    "66b7ca3198584136a86660733672b5ab.NO-wYz2AeqN7Bf0rRSrLkb0H",
        "3423a52360bf468588b6c80e6957ea1d.nQBGpbmZzCvwWysfb6ORjbGd",
        "77408cf3484946d8bb8cf37220ad2721.837tfoOLJmV3FEka43ozQlZF",
        "9c75046d041a4dca811fd2eaaf3e5696.RH4yyGnyj-qwU8BLCRSr4j7P"

)


def _build_model_configs() -> List[Dict[str, str]]:
    configs: List[Dict[str, str]] = []
    for model_name in IMPLEMENTATION_MODELS:
        if not isinstance(model_name, str) or not model_name.strip():
            continue
        provider = "google-genai" if model_name.startswith("gemini-") else "ollama"
        configs.append({
            "provider": provider,
            "model": model_name.strip(),
            "name": model_name.strip(),
        })
    return configs


class ModelConfig:
    """Configuration for different models"""

    _ALL_MODELS = _build_model_configs()
    STRONG_MODELS = list(_ALL_MODELS)
    WEAK_MODELS = list(_ALL_MODELS)
    OPTIMIZATION_MODELS = list(_ALL_MODELS)


def _truncate_message(message: Optional[str], limit: int = 140) -> str:
    """Safely truncate long log messages."""
    if not message:
        return ""
    trimmed = message.strip()
    if len(trimmed) <= limit:
        return trimmed
    return f"{trimmed[: limit - 3]}..."


def log_step(step: int, message: str) -> None:
    """Print a formatted pipeline step message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Step {step:03d}: {message}")


async def call_model_with_retry____________________________________________(
    provider: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    max_retries: int = 3,
    temperature: float = 0.7,
    api_key_google: str = ""
) -> str:
  
    del provider  # retained for backwards compatibility
    del temperature

    combined_prompt = prompt
    if system_prompt:
        combined_prompt = f"{system_prompt.strip()}\n\n{prompt}"

    loop = asyncio.get_running_loop()

    async def _invoke() -> dict:
        # Use first API key as placeholder (call_model handles rotation internally)
        ollama_key = _OLLAMA_API_KEY[0] if _OLLAMA_API_KEY else ""
        return await loop.run_in_executor(
            None,
            lambda: call_model(model, combined_prompt, api_key_google=api_key_google,api_key_ollama="") 
        )

    last_error: str | None = None
    for attempt in range(1):
        try:
            result = await _invoke()
            if isinstance(result, dict) and result.get("success"):
                return result.get("output", "")
            error_message = ""
            if isinstance(result, dict):
                error_message = result.get("error") or ""
            if not error_message:
                error_message = "Model call failed without an explicit error message."
            last_error = error_message
        except Exception as exc:  # noqa: BLE001 - propagate meaningful failure
            last_error = str(exc)
        if attempt < max_retries - 1:
            # Reduced delay: 0.5s, 1s instead of 1s, 2s, 4s
            await asyncio.sleep(0.5 * (attempt + 1))

    raise RuntimeError(last_error or "Model call failed after retries.")


def extract_code_from_response(response: str) -> str:
    """Extract C++ code from model response"""
    # Look for code blocks
    if "```cpp" in response:
        parts = response.split("```cpp")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    elif "```c++" in response:
        parts = response.split("```c++")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    elif "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code = parts[1]
            return code.strip()
    
    # If no code blocks, return the whole response
    return response.strip()


# ============================================================================
# Phase 1: Plan to Code (Steps 1-20)
# ============================================================================

async def plan_to_code(
    plan: str,
    problem: str,
    best_code: str,
    model_config: Dict,
    iteration: int,
    api_key_google: str = "",
    speed: int = 1,
    memory: int = 512
) -> Tuple[str, ExecutionMetrics]:
    """
    Generate initial C++ code from plan using strong models
    
    Args:
        plan: High-level plan/algorithm description
        problem: Problem description
        model_config: Model configuration dict
        iteration: Current iteration number (1-20)
        
    Returns:
        (generated_code, metrics)
    """
    start_time = time.time()
    
    system_prompt = """You are an expert C++ programmer. Generate clean, efficient C++ code.
Always include necessary headers and a main() function.
Focus on correctness and clarity."""
    
    user_prompt = f"""
    Current C++ Code:
{best_code}

Problem:
{problem}

Plan/Algorithm:
{plan}

Generate a complete C++ solution that implements this plan.
Include all necessary headers (#include <iostream>, etc.).
The code should compile with g++ -std=c++17 and solve the problem correctly.

Return ONLY the C++ code, wrapped in ```cpp code blocks."""
    
    try:
        response = await call_model_with_retry____________________________________________(
            provider=model_config["provider"],
            model=model_config["model"],
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            api_key_google=api_key_google
        )
        
        code = extract_code_from_response(response)
        
        metrics = ExecutionMetrics(
            success=True,
            output=code[:200],
            error="",
            time_ms=(time.time() - start_time) * 1000,
            memory_mb=0.0,
            timeout=False,
            iteration=iteration,
            model_used=model_config["name"],
            test_passed=False,
            timestamp=datetime.now().isoformat()
        )
        
        return code, metrics
        
    except Exception as e:
        metrics = ExecutionMetrics(
            success=False,
            output="",
            error=str(e),
            time_ms=(time.time() - start_time) * 1000,
            memory_mb=0.0,
            timeout=False,
            iteration=iteration,
            model_used=model_config["name"],
            test_passed=False,
            timestamp=datetime.now().isoformat()
        )
        return "", metrics


async def phase1_initial_code_generation(
    plan: str,
    problem: str,
    code: List[str],
    num_attempts: int = 20,
    api_key_google: str = "",
    speed: int = 1,
    memory: int = 512
) -> PhaseResult:
    """
    Phase 1: Generate initial code using strong models (Steps 1-20)
    
    Executes in parallel - all models run simultaneously
    """
    print(f"\n{'='*60}")
    print("PHASE 1: Initial Code Generation (Steps 1-20) - PARALLEL MODE")
    print(f"{'='*60}")
    
    phase_start = time.time()
    
    # Prepare all tasks for parallel execution
    tasks = []
    # Distribute all codes from list to models (round-robin)
    # If we have codes, distribute them; otherwise use empty string
    code_list = code if code and isinstance(code, list) else ([] if isinstance(code, str) else [])
    code_str = ""
    for code_item in code_list:
        code_str += code_item + "\n"+"next code is :"

    for i in range(num_attempts):
        model_config = ModelConfig.STRONG_MODELS[i % len(ModelConfig.STRONG_MODELS)]
        step_num = i + 1
        log_step(step_num, f"Scheduling parallel generation with model {model_config['name']}")
        
        
        task = plan_to_code(plan, problem, code_str, model_config, step_num, api_key_google=api_key_google,speed=speed,memory=memory)
        tasks.append(task)
    
    # Execute all tasks in parallel
    print(f"\n🚀 Launching {num_attempts} models in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    all_metrics = []
    best_code = []
    
    for i, result in enumerate(results):
        step_num = i + 1
        
        if isinstance(result, Exception):
            # Handle exception
            metrics = ExecutionMetrics(
                success=False,
                output="",
                error=str(result),
                time_ms=0.0,
                memory_mb=0.0,
                timeout=False,
                iteration=step_num,
                model_used=ModelConfig.STRONG_MODELS[i % len(ModelConfig.STRONG_MODELS)]["name"],
                test_passed=False,
                timestamp=datetime.now().isoformat()
            )
            all_metrics.append(metrics)
            log_step(step_num, f"Generation failed with exception: {_truncate_message(str(result))}")
        else:
            code, metrics = result
            all_metrics.append(metrics)
            
            if metrics.success and len(code) > 0:
                # Keep the longest/best code
                best_code.append(code)
                log_step(step_num, f"✓ Generated candidate code ({len(code)} characters)")
            elif not metrics.success:
                log_step(step_num, f"✗ Generation failed: {_truncate_message(metrics.error)}")
            else:
                log_step(step_num, "⚠ Model returned empty code")
    
    phase_duration = time.time() - phase_start
    
    # Select the best code (longest one) from the list and convert to string
    selected_code = ""
    if best_code:
        selected_code = max(best_code, key=len)
    
    print(f"\n✓ Generated {num_attempts} code variants in PARALLEL")
    if selected_code:
        print(f"✓ Best code length: {len(selected_code)} characters")
    else:
        print(f"⚠ No successful code generation")
    print(f"✓ Phase duration: {phase_duration:.2f}s")
    print(f"✓ Speedup: ~{num_attempts}x faster than sequential")
    
    return PhaseResult(
        phase_name="Initial Code Generation (Parallel)",
        start_step=1,
        end_step=20,
        code=best_code,
        
        metrics=all_metrics,
        duration_seconds=phase_duration
    )


# ============================================================================
# Phase 2: Execute and Fix (Steps 21-70)
# ============================================================================

async def execute_and_fix(
    code: str,
    problem: str,
    tests: List[Dict],
    model_config: Dict,
    iteration: int
) -> Tuple[str, ExecutionMetrics]:
    """
    Execute code, capture errors, and fix with weak models
    
    Args:
        code: Current C++ code
        problem: Problem description
        tests: List of test cases
        model_config: Model configuration
        iteration: Current iteration number
        
    Returns:
        (fixed_code, metrics)
    """
    start_time = time.time()
    
    # Try to compile and execute
    with CppSandbox(timeout=5, memory_limit_mb=512) as sandbox:
        # Compile
        compile_success, compile_error = sandbox.compile(code)
        
        if not compile_success:
            # Code doesn't compile - ask model to fix
            system_prompt = "You are a C++ expert. Fix compilation errors."
            
            user_prompt = f"""This C++ code has compilation errors:

```cpp
{code}
```

Compilation Error:
{compile_error}

Fix the compilation errors and return the corrected code.
Return ONLY the fixed C++ code in ```cpp blocks."""
            
            try:
                response = await call_model_with_retry____________________________________________(
                    provider=model_config["provider"],
                    model=model_config["model"],
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3
                )
                
                fixed_code = extract_code_from_response(response)
                
                metrics = ExecutionMetrics(
                    success=False,
                    output="",
                    error=compile_error,
                    time_ms=(time.time() - start_time) * 1000,
                    memory_mb=0.0,
                    timeout=False,
                    iteration=iteration,
                    model_used=model_config["name"],
                    test_passed=False,
                    timestamp=datetime.now().isoformat()
                )
                
                return fixed_code, metrics
                
            except Exception as e:
                metrics = ExecutionMetrics(
                    success=False,
                    output="",
                    error=f"Model error: {str(e)}",
                    time_ms=(time.time() - start_time) * 1000,
                    memory_mb=0.0,
                    timeout=False,
                    iteration=iteration,
                    model_used=model_config["name"],
                    test_passed=False,
                    timestamp=datetime.now().isoformat()
                )
                return code, metrics
        
        # Code compiles - run tests
        test_results = []
        all_passed = True
        
        for test in tests[:3]:  # Run first 3 tests
            test_input = test.get("input", "")
            expected_output = test.get("expected_output", "")
            
            exec_result = sandbox.execute(test_input)
            
            actual_output = exec_result["output"].strip()
            # Convert expected_output to string before calling strip()
            expected_str = str(expected_output).strip()
            passed = actual_output == expected_str
            
            test_results.append({
                "input": test_input,
                "expected": expected_output,
                "actual": actual_output,
                "passed": passed,
                "error": exec_result["error"]
            })
            
            if not passed:
                all_passed = False
        
        if all_passed:
            # All tests passed!
            metrics = ExecutionMetrics(
                success=True,
                output="All tests passed",
                error="",
                time_ms=(time.time() - start_time) * 1000,
                memory_mb=test_results[0].get("memory_mb", 0) if test_results else 0,
                timeout=False,
                iteration=iteration,
                model_used=model_config["name"],
                test_passed=True,
                timestamp=datetime.now().isoformat()
            )
            return code, metrics
        
        # Some tests failed - ask model to fix
        failed_tests = [t for t in test_results if not t["passed"]]
        
        system_prompt = "You are a C++ expert. Fix code to pass all test cases."
        
        user_prompt = f"""Problem:
{problem}

Current C++ Code:
```cpp
{code}
```

Failed Test Cases:
"""
        for i, test in enumerate(failed_tests[:2]):  # Show first 2 failures
            user_prompt += f"""
Test {i+1}:
Input: {test['input']}
Expected: {test['expected']}
Actual: {test['actual']}
Error: {test['error']}
"""
        
        user_prompt += """
Fix the code to pass all test cases correctly.
Return ONLY the fixed C++ code in ```cpp blocks."""
        
        try:
            response = await call_model_with_retry____________________________________________(
                provider=model_config["provider"],
                model=model_config["model"],
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            
            fixed_code = extract_code_from_response(response)
            
            metrics = ExecutionMetrics(
                success=True,
                output=f"{len(failed_tests)} tests failed",
                error="",
                time_ms=(time.time() - start_time) * 1000,
                memory_mb=0.0,
                timeout=False,
                iteration=iteration,
                model_used=model_config["name"],
                test_passed=False,
                timestamp=datetime.now().isoformat()
            )
            
            return fixed_code, metrics
            
        except Exception as e:
            metrics = ExecutionMetrics(
                success=False,
                output="",
                error=f"Model error: {str(e)}",
                time_ms=(time.time() - start_time) * 1000,
                memory_mb=0.0,
                timeout=False,
                iteration=iteration,
                model_used=model_config["name"],
                test_passed=False,
                timestamp=datetime.now().isoformat()
            )
            return code, metrics


async def phase2_iterative_fixing(
    initial_code: str,
    problem: str,
    tests: List[Dict],
    max_iterations: int = 50
) -> PhaseResult:
    """
    Phase 2: Iteratively execute and fix code (Steps 21-70)
    
    Executes sequentially so each model can improve upon the previous model's solution
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Iterative Fixing (Steps 21-70)")
    print(f"{'='*60}")
    
    phase_start = time.time()
    current_code = initial_code
    all_metrics = []
    
    for i in range(max_iterations):
        model_config = ModelConfig.WEAK_MODELS[i % len(ModelConfig.WEAK_MODELS)]
        step_num = 21 + i
        log_step(step_num, f"Iterative fixing with model {model_config['name']}")
        
        # Execute sequentially (await directly) so each model improves the previous result
        new_code, metrics = await execute_and_fix(
            current_code,
            problem,
            tests,
            model_config,
            step_num
        )
        
        all_metrics.append(metrics)
        
        if metrics.test_passed:
            log_step(step_num, "All monitored tests passed. Exiting iterative fixing early.")
            current_code = new_code
            break
        
        if not metrics.success and metrics.error:
            log_step(step_num, f"Execution/model error: {_truncate_message(metrics.error)}")
        
        if len(new_code) > 0:
            current_code = new_code
            log_step(step_num, f"Applied model suggestion ({len(current_code)} characters); continuing validation.")
        else:
            log_step(step_num, "No code changes produced; keeping previous version.")
        
        if metrics.output and not metrics.test_passed:
            log_step(step_num, f"Status: {_truncate_message(metrics.output)}")
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{max_iterations} iterations completed")
    
    phase_duration = time.time() - phase_start
    
    print(f"\n✓ Completed {len(all_metrics)} sequential fix iterations")
    print(f"✓ Phase duration: {phase_duration:.2f}s")
    
    return PhaseResult(
        phase_name="Iterative Fixing",
        start_step=21,
        end_step=70,
        code=current_code,
        metrics=all_metrics,
        duration_seconds=phase_duration
    )


# ============================================================================
# Phase 3: Speed Optimization (Steps 71-85)
# ============================================================================




# ============================================================================
# Phase 3&4 Combined: Parallel Optimization with Specific Constraints
# ============================================================================

async def optimize_with_constraints(
    candidate_codes: List[str],
    problem: str,
    tests: List[Dict],
    target_time_ms: float,
    target_memory_mb: float,
    step_num: int,
    model_config: Dict
) -> Tuple[List[str], ExecutionMetrics]:



    start_time = time.time()
    code_len = len(candidate_codes)
    newline = '\n\n'
    
    system_prompt = """You are a C++ optimization expert. 
    Analyze multiple code solutions, pick the best one, and improve it to meet performance constraints."""

    versions_text = newline.join([f"VERSION {i+1} :\n```cpp\n{c}\n```" for i, c in enumerate(candidate_codes)])
    user_prompt = f"""You have {code_len} optimized versions of this code:

    {versions_text}

    CONSTRAINTS:
    - Time must be ≤ {target_time_ms}ms
    - Memory must be ≤ {target_memory_mb}MB

    TASK:
    1. Pick the best version (or combine ideas from multiple versions)
    2. Make improvements to meet BOTH constraints
    3. Return only the final optimized code

        Output format:
    [final code]
    ```cpp
    
    ```
    """

 
    
    try:
        response = await call_model_with_retry____________________________________________(
            provider=model_config["provider"],
            model=model_config["model"],
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4
        )
        
        optimized_code = extract_code_from_response(response)





        
        metrics = ExecutionMetrics(
            success=True,
            output=f"Optimization applied (target: {target_time_ms:.2f}ms, {target_memory_mb:.2f}MB)",
            error="",
            time_ms=(time.time() - start_time) * 1000,
            memory_mb=0.0,
            timeout=False,
            iteration=step_num,
            model_used=model_config["name"],
            test_passed=False,
            timestamp=datetime.now().isoformat()
        )
        
        # Return as list to match function signature
        if optimized_code:
            return [optimized_code], metrics
        else:
            return candidate_codes if candidate_codes else [], metrics
        
    except Exception as e:
        metrics = ExecutionMetrics(
            success=False,
            output="",
            error=str(e),
            time_ms=(time.time() - start_time) * 1000,
            memory_mb=0.0,
            timeout=False,
            iteration=step_num,
            model_used=model_config["name"],
            test_passed=False,
            timestamp=datetime.now().isoformat()
        )
        return [], metrics


async def phase3_4_parallel_optimization(
    code: List[str],
    problem: str,
    tests: List[Dict],
    target_time_ms: float,
    target_memory_mb: float,
    num_models: int = 15
) -> PhaseResult:
    model_configs = ModelConfig.STRONG_MODELS
    print(f"\n{'='*60}")
    print("PHASE 3&4 COMBINED: Parallel Optimization (Steps 71-100)")
    print(f"{'='*60}")
    print(f"Target Constraints:")
    print(f"  - Execution Time: {target_time_ms:.2f}ms or less")
    print(f"  - Memory Usage: {target_memory_mb:.2f}MB or less")
    print(f"Running {num_models} models in PARALLEL...")
    print(f"{'='*60}")
    
    phase_start = time.time()
    
    # Create tasks for all models to run in parallel
    tasks = []
    for i in range(num_models):
        model_config = model_configs[i % len(model_configs)]
        step_num = 71 + i
        
        # Create async task for each model
        task = optimize_with_constraints(code, problem, tests, target_time_ms, target_memory_mb, step_num, model_config)
        tasks.append(task)
        log_step(step_num, f"Launching parallel optimization with {model_config['name']}")
    
    # Execute all optimizations in parallel
    print(f"\n{'='*60}")
    print(f"Executing {num_models} parallel optimizations...")
    print(f"{'='*60}")
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    all_metrics = []
    valid_results = []
    
    for i, result in enumerate(results):
        step_num = 71 + i
        
        if isinstance(result, Exception):
            log_step(step_num, f"Error: {_truncate_message(str(result))}")
            continue
            
        optimized_code_list, metrics = result
        all_metrics.append(metrics)
        
        if not metrics.success:
            log_step(step_num, f"Optimization failed: {_truncate_message(metrics.error)}")
            continue
        
        # Convert list to string for compilation
        if not optimized_code_list:
            log_step(step_num, "No optimized code returned")
            continue
        optimized_code = optimized_code_list[0] if isinstance(optimized_code_list, list) else optimized_code_list
        
        # Test the optimized code
        with CppSandbox(timeout=5, memory_limit_mb=512) as sandbox:
            compile_success, compile_error = sandbox.compile(optimized_code)
            
            if not compile_success:
                log_step(step_num, f"Compilation failed: {_truncate_message(compile_error)}")
                continue
                
            if not tests:
                log_step(step_num, "No tests provided; accepting code without benchmark.")
                valid_results.append((optimized_code, 0.0, 0.0, metrics, step_num))
                continue
                
            # Execute all tests and calculate average performance
            total_time = 0.0
            total_memory = 0.0
            test_count = 0
            all_passed = True
            
            for test in tests:
                exec_result = sandbox.execute(test.get("input", ""))
                if exec_result["success"]:
                    total_time += exec_result["time_ms"]
                    total_memory += exec_result["memory_mb"]
                    test_count += 1
                    
                    # Check if output matches (if expected output is provided)
                    expected = test.get("expected_output", "")
                    if expected:
                        actual = exec_result["output"].strip()
                        if actual != expected.strip():
                            all_passed = False
                else:
                    all_passed = False
                    break
            
            if test_count > 0 and all_passed:
                avg_time = total_time / test_count
                avg_memory = total_memory / test_count
                
                # Calculate constraint satisfaction score
                time_ratio = avg_time / target_time_ms if target_time_ms > 0 else 1.0
                memory_ratio = avg_memory / target_memory_mb if target_memory_mb > 0 else 1.0
                
                meets_time = avg_time <= target_time_ms
                meets_memory = avg_memory <= target_memory_mb
                
                status = "✓" if (meets_time and meets_memory) else "○"
                log_step(
                    step_num,
                    f"{status} Avg: {avg_time:.2f}ms ({time_ratio*100:.1f}%), {avg_memory:.2f}MB ({memory_ratio*100:.1f}%)"
                )
                
                valid_results.append((optimized_code, avg_time, avg_memory, metrics, step_num))
            else:
                log_step(step_num, "Tests failed or execution error")
    
    # Select best result
    if not valid_results:
        print("\n⚠ No valid optimizations produced. Returning original code.")
        best_code = code
        best_time = float('inf')
        best_memory = float('inf')
    else:
        # Score each result: prioritize meeting constraints, then minimize both time and memory
        def score_result(result):
            _, avg_time, avg_memory, _, _ = result
            
            # Perfect score if both constraints are met
            if avg_time <= target_time_ms and avg_memory <= target_memory_mb:
                # Lower is better when constraints are met
                return (0, avg_time + avg_memory)
            
            # Otherwise, penalize based on how much constraints are exceeded
            time_penalty = max(0, avg_time - target_time_ms)
            memory_penalty = max(0, avg_memory - target_memory_mb)
            return (1, time_penalty + memory_penalty)
        
        valid_results.sort(key=score_result)
        best_code, best_time, best_memory, _, best_step = valid_results[0]
        
        print(f"\n{'='*60}")
        print(f"BEST RESULT SELECTED: Step {best_step}")
        print(f"{'='*60}")
        print(f"✓ Execution Time: {best_time:.2f}ms (target: {target_time_ms:.2f}ms)")
        print(f"✓ Memory Usage: {best_memory:.2f}MB (target: {target_memory_mb:.2f}MB)")
        
        if best_time <= target_time_ms and best_memory <= target_memory_mb:
            print(f"✓ BOTH constraints SATISFIED! 🎉")
        elif best_time <= target_time_ms:
            print(f"○ Time constraint satisfied, memory needs improvement")
        elif best_memory <= target_memory_mb:
            print(f"○ Memory constraint satisfied, time needs improvement")
        else:
            print(f"○ Both constraints need improvement")
    
    phase_duration = time.time() - phase_start
    
    print(f"\n✓ Completed {num_models} parallel optimizations")
    print(f"✓ Valid results: {len(valid_results)}/{num_models}")
    print(f"✓ Phase duration: {phase_duration:.2f}s")
    
    # Ensure code is a list for PhaseResult
    code_list = best_code if isinstance(best_code, list) else [best_code] if best_code else []
    
    return PhaseResult(
        phase_name="Parallel Optimization (Speed & Memory)",
        start_step=71,
        end_step=100,
        code=code_list,
        metrics=all_metrics,
        duration_seconds=phase_duration
    )

from optim import optimizer
# ============================================================================
# Helper Functions for Constraints
# ============================================================================

def parse_time_limit(time_str: str) -> float:
    """
    Parse time limit string to milliseconds
    
    Examples:
        "1 second" -> 1000.0
        "5 seconds" -> 5000.0
        "0.5 seconds" -> 500.0
    """
    time_str = time_str.lower().strip()
    
    # Extract number
    import re
    match = re.search(r'[\d.]+', time_str)
    if not match:
        return 1000.0  # Default 1 second
    
    value = float(match.group())
    
    # Convert to milliseconds
    if 'second' in time_str:
        return value * 1000.0
    elif 'ms' in time_str or 'millisecond' in time_str:
        return value
    else:
        return value * 1000.0  # Assume seconds if no unit


def parse_memory_limit(memory_str: str) -> float:
    """
    Parse memory limit string to megabytes
    
    Examples:
        "256 megabytes" -> 256.0
        "1024 megabytes" -> 1024.0
        "1 GB" -> 1024.0
        "512 MB" -> 512.0
    """
    memory_str = memory_str.lower().strip()
    
    # Extract number
    import re
    match = re.search(r'[\d.]+', memory_str)
    if not match:
        return 256.0  # Default 256 MB
    
    value = float(match.group())
    
    # Convert to MB
    if 'gb' in memory_str or 'gigabyte' in memory_str:
        return value * 1024.0
    elif 'mb' in memory_str or 'megabyte' in memory_str:
        return value
    elif 'kb' in memory_str or 'kilobyte' in memory_str:
        return value / 1024.0
    else:
        return value  # Assume MB if no unit


def load_constraints_from_json(problem_index: Optional[int] = None, json_path: str = "ioi_multi_view.json") -> Tuple[float, float]:
    """
    Load time and memory constraints from ioi_multi_view.json
    
    Args:
        problem_index: Index of the problem in the JSON array (None to skip)
        json_path: Path to the JSON file
        
    Returns:
        (target_time_ms, target_memory_mb)
    """
    if problem_index is None:
        # Default constraints if no index provided
        return 1000.0, 256.0
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or problem_index >= len(data):
            print(f"⚠ Invalid problem index {problem_index}. Using default constraints.")
            return 1000.0, 256.0
        
        problem_data = data[problem_index]
        limits = problem_data.get('algorithm_view', {}).get('limits', {})
        
        time_str = limits.get('time', '1 second')
        memory_str = limits.get('memory', '256 megabytes')
        
        target_time_ms = parse_time_limit(time_str)
        target_memory_mb = parse_memory_limit(memory_str)
        
        print(f"\n{'='*60}")
        print(f"LOADED CONSTRAINTS FROM JSON (Index: {problem_index})")
        print(f"{'='*60}")
        print(f"  Time Limit: {time_str} → {target_time_ms:.2f}ms")
        print(f"  Memory Limit: {memory_str} → {target_memory_mb:.2f}MB")
        print(f"{'='*60}")
        
        return target_time_ms, target_memory_mb
        
    except Exception as e:
        print(f"⚠ Error loading constraints from JSON: {e}")
        print(f"⚠ Using default constraints: 1000ms, 256MB")
        return 1000.0, 256.0


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

async def run_pipeline(
    plan: str,
    problem: str,
    tests: List[Dict],
    output_dir: str = "pipeline_results",
    problem_index: Optional[int] = None
) -> Dict:
 
    print("\n" + "="*60)
    print("100-STEP MULTI-MODEL CODE GENERATION PIPELINE")
    print("="*60)
    
    pipeline_start = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: Initial Code Generation (Steps 1-20)
    code = []
    phase1_result = await phase1_initial_code_generation(plan, problem,code=code ,num_attempts=20,api_key_google=api_key_google_list[0])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase1.txt"), "w") as f:
        for i, code_item in enumerate(phase1_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")

    # Pass ALL codes from phase1 to phase2 (not just one)
    phase2_result = await phase1_initial_code_generation(plan, problem, phase1_result.code if phase1_result.code else [], num_attempts=20,api_key_google=api_key_google_list[1])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase2.txt"), "w") as f:
        for i, code_item in enumerate(phase2_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")

    # Pass ALL codes from phase2 to phase3 (not just one)
    phase3_result = await phase1_initial_code_generation(plan, problem, phase2_result.code if phase2_result.code else [], num_attempts=20,api_key_google=api_key_google_list[2])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase3.txt"), "w") as f:
        for i, code_item in enumerate(phase3_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")

    # Pass ALL codes from phase3 to phase4 (not just one)
    phase4_result = await phase1_initial_code_generation(plan, problem, phase3_result.code if phase3_result.code else [], num_attempts=20,api_key_google=api_key_google_list[3])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase4.txt"), "w") as f:
        for i, code_item in enumerate(phase4_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")
   
    phase5_result = await phase1_initial_code_generation(plan, problem, phase4_result.code if phase4_result.code else [], num_attempts=20,api_key_google=api_key_google_list[0])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase5.txt"), "w") as f:
        for i, code_item in enumerate(phase5_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")
    


    

    

    # Phase 3&4 Combined: Parallel Optimization (Steps 71-100)
    print("\n" + "="*60)
    print("RUNNING PARALLEL OPTIMIZATION WITH CONSTRAINTS")
    print("="*60)
    
    
    best_code = optimizer()
    # Ensure best_code is a list for the next phase
    from _30step import _30step
    all_results = await _30step(best_code, tests)


    


  
   

    


  
    
 
    
    # Save phase 2 results
    
    
    return 


# ============================================================================
# Example Usage
# ============================================================================

async def cpp_pipeline(
    problem: Union[str, Dict],
    plan: Union[str, Dict],
    tests: Union[List[Dict], Dict, None],
    output_dir: str = "pipeline_results2",
    problem_index: Optional[int] = None
):
    """
    Run the 100-step pipeline using the provided inputs.
    
    Args:
        problem: Problem description
        plan: Algorithm plan
        tests: Test cases
        output_dir: Directory to save results
        problem_index: Index in ioi_multi_view.json to load constraints from
    
    Prints a concise overview before delegating to the main `run_pipeline` orchestrator.
    """
    plan_text = plan if isinstance(plan, str) else json.dumps(plan, indent=2)
    if isinstance(problem, str):
        problem_text = problem
    else:
        problem_text = json.dumps(problem, indent=2)
    
    if isinstance(tests, list):
        tests_list = [t for t in tests if isinstance(t, dict)]
    elif isinstance(tests, dict):
        # Accept dictionaries that contain a direct list of tests or need value extraction
        if "tests" in tests and isinstance(tests["tests"], list):
            tests_list = [t for t in tests["tests"] if isinstance(t, dict)]
        else:
            tests_list = [t for t in tests.values() if isinstance(t, dict)]
    else:
        tests_list = []
    
    print("\n" + "="*60)
    print("PIPELINE INPUT OVERVIEW")
    print("="*60)
    print(f"Output directory: {output_dir}")
    if problem_index is not None:
        print(f"Problem index (ioi_multi_view.json): {problem_index}")
    print(f"Problem summary preview: {_truncate_message(problem_text, 180)}")
    print(f"Plan preview: {_truncate_message(plan_text, 180)}")
    print(f"Test cases detected: {len(tests_list)}")
    if not tests_list:
        print("Warning: No structured test cases detected. Speed/memory benchmarks will be skipped.")
    print("="*60)
    
    return await run_pipeline(
        plan=plan_text,
        problem=problem_text,
        tests=tests_list,
        output_dir=output_dir,
        problem_index=problem_index
    )


if __name__ == "__main__":
    # Set API keys as environment variables before running:
    # export GOOGLE_API_KEY="your-google-genai-key"
    # export OLLAMA_API_KEY="optional-if-your-endpoint-requires-it"



    problem = "You are given a list of n integers, and you need to find the k-th largest element in the list. The k-th largest element is the element that would appear in the k-th position if the list were sorted in descending order."
    plan = "To find the k-th largest element in the list, we can use the following steps: 1. Sort the list in descending order. 2. Return the element at the k-th position."
    tests = [{"input": [3, 1, 2, 4, 5], "expected_output": 4}, {"input": [1, 2, 3, 4, 5], "expected_output": 5}]
    output_dir = "pipeline_results"
    problem_index = 0
    asyncio.run(cpp_pipeline(problem, plan, tests, output_dir, problem_index))

