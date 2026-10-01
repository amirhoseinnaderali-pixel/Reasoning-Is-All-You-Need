IMPLEMENTATION_MODELS = [
    "gpt-oss:120b-cloud",
    "gemini-2.5-flash",
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

from typing import List, Dict, Tuple, Optional
import time
import asyncio
import json
import re
import os
from datetime import datetime
from cpp_pipe import (
    call_model_with_retry____________________________________________,
    extract_code_from_response,
    ExecutionMetrics,
    ModelConfig,
    PhaseResult,
    log_step,
    CppSandbox,
    _truncate_message
)

# API keys are supplied through the environment. Never commit credentials.
api_key_google_list = [
    key.strip() for key in os.getenv("GOOGLE_API_KEYS", "").split(",") if key.strip()
]
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


def _get_model_config(model_name: str) -> Dict:
    """
    ساخت model config از نام مدل
    
    Args:
        model_name: نام مدل
        
    Returns:
        دیکشنری config مدل
    """
    provider = "google-genai" if model_name.startswith("gemini-") else "ollama"
    return {
        "provider": provider,
        "model": model_name.strip(),
        "name": model_name.strip()
    }


async def debug_and_fix_with_model(
    code: str,
    tests: List[Dict],
    model_config: Dict,
    step_num: int,
    api_key_google: str = ""
) -> Tuple[str, ExecutionMetrics]:
    """
    اجرای کد، پیدا کردن مشکلات و رفع آن‌ها با یک مدل
    
    Args:
        code: کد C++ فعلی
        tests: لیست تست کیس‌ها
        model_config: تنظیمات مدل
        step_num: شماره گام
        api_key_google: کلید API گوگل
        
    Returns:
        (کد اصلاح شده, metrics)
    """
    start_time = time.time()
    
    # اجرا در sandbox
    with CppSandbox(timeout=5, memory_limit_mb=512) as sandbox:
        # کامپایل
        print(f"  🔨 Compiling code...")
        compile_success, compile_error = sandbox.compile(code)
        
        if not compile_success:
            print(f"  ❌ Compilation failed!")
            print(f"  Error: {compile_error[:200]}")
            print(f"  🤖 Asking model to fix compilation errors...")
            # خطای کامپایل - درخواست رفع از مدل
            system_prompt = "You are a C++ expert. Fix ONLY compilation errors. Make MINIMAL changes to the code."
            
            user_prompt = f"""This C++ code has compilation errors. Fix ONLY the errors, do NOT make major changes:

```cpp
{code}
```

Compilation Error:
{compile_error}

Fix ONLY the compilation errors. Keep the code structure the same.
Return ONLY the fixed C++ code in ```cpp blocks."""
            
            try:
                response = await call_model_with_retry____________________________________________(
                    provider=model_config["provider"],
                    model=model_config["model"],
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    api_key_google=api_key_google
                )
                
                fixed_code = extract_code_from_response(response)
                
                metrics = ExecutionMetrics(
                    success=False,
                    output="",
                    error=compile_error,
                    time_ms=(time.time() - start_time) * 1000,
                    memory_mb=0.0,
                    timeout=False,
                    iteration=step_num,
                    model_used=model_config["name"],
                    test_passed=False,
                    timestamp=datetime.now().isoformat()
                )
                
                return fixed_code if fixed_code else code, metrics
                
            except Exception as e:
                metrics = ExecutionMetrics(
                    success=False,
                    output="",
                    error=f"Model error: {str(e)}",
                    time_ms=(time.time() - start_time) * 1000,
                    memory_mb=0.0,
                    timeout=False,
                    iteration=step_num,
                    model_used=model_config["name"],
                    test_passed=False,
                    timestamp=datetime.now().isoformat()
                )
                return code, metrics
        
        # کد کامپایل شد - اجرای تست‌ها
        print(f"  ✅ Compilation successful!")
        test_results = []
        all_passed = True
        
        print(f"  🧪 Running {len(tests)} test case(s)...")
        for i, test in enumerate(tests, 1):
            test_input = test.get("input", "")
            expected_output = test.get("expected_output", "")
            
            exec_result = sandbox.execute(test_input)
            
            actual_output = exec_result["output"].strip()
            expected_str = str(expected_output).strip()
            passed = actual_output == expected_str
            
            test_results.append({
                "input": test_input,
                "expected": expected_output,
                "actual": actual_output,
                "passed": passed,
                "error": exec_result["error"]
            })
            
            # نمایش نتیجه هر تست
            status = "✅" if passed else "❌"
            print(f"    Test {i}: {status}", end="")
            if passed:
                print(f" Passed (Output: {actual_output[:50]})")
            else:
                print(f" Failed")
                print(f"      Expected: {expected_str[:100]}")
                print(f"      Actual:   {actual_output[:100]}")
                if exec_result["error"]:
                    print(f"      Error:    {exec_result['error'][:100]}")
            
            if not passed:
                all_passed = False
        
        if all_passed:
            # همه تست‌ها پاس شدند
            print(f"  ✅ All tests passed!")
            metrics = ExecutionMetrics(
                success=True,
                output="All tests passed",
                error="",
                time_ms=(time.time() - start_time) * 1000,
                memory_mb=test_results[0].get("memory_mb", 0) if test_results else 0,
                timeout=False,
                iteration=step_num,
                model_used=model_config["name"],
                test_passed=True,
                timestamp=datetime.now().isoformat()
            )
            return code, metrics
        
        # برخی تست‌ها ناموفق - درخواست رفع از مدل
        failed_tests = [t for t in test_results if not t["passed"]]
        
        print(f"  ❌ {len(failed_tests)} test(s) failed!")
        print(f"  🤖 Asking model to fix bugs...")
        
        system_prompt = "You are a C++ expert. Fix ONLY small bugs to make tests pass. Make MINIMAL changes."
        
        user_prompt = f"""This C++ code fails some test cases. Fix ONLY the bugs, do NOT rewrite the code:

```cpp
{code}
```

Failed Test Cases:
"""
        for i, test in enumerate(failed_tests[:3]):  # نمایش 3 تست ناموفق اول
            user_prompt += f"""
Test {i+1}:
Input: {test['input']}
Expected: {test['expected']}
Actual: {test['actual']}
Error: {test['error']}
"""
        
        user_prompt += """
Fix ONLY the bugs to make all tests pass. Keep the code structure the same.
Return ONLY the fixed C++ code in ```cpp blocks."""
        
        try:
            response = await call_model_with_retry____________________________________________(
                provider=model_config["provider"],
                model=model_config["model"],
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                api_key_google=api_key_google
            )
            
            fixed_code = extract_code_from_response(response)
            
            metrics = ExecutionMetrics(
                success=True,
                output=f"{len(failed_tests)} tests failed",
                error="",
                time_ms=(time.time() - start_time) * 1000,
                memory_mb=0.0,
                timeout=False,
                iteration=step_num,
                model_used=model_config["name"],
                test_passed=False,
                timestamp=datetime.now().isoformat()
            )
            
            return fixed_code if fixed_code else code, metrics
            
        except Exception as e:
            metrics = ExecutionMetrics(
                success=False,
                output="",
                error=f"Model error: {str(e)}",
                time_ms=(time.time() - start_time) * 1000,
                memory_mb=0.0,
                timeout=False,
                iteration=step_num,
                model_used=model_config["name"],
                test_passed=False,
                timestamp=datetime.now().isoformat()
            )
            return code, metrics


async def debug_with_all_models(
    code: str,
    tests: List[Dict],
    problem: str = "",
    api_key_google: Optional[str] = None
) -> Tuple[str, List[ExecutionMetrics]]:
    """
    دیباگ و رفع مشکلات کد با استفاده از همه مدل‌ها
    هر مدل یک گام اجرا می‌کند
    
    Args:
        code: کد C++ اولیه
        tests: لیست تست کیس‌ها
        problem: توضیح مسئله (اختیاری)
        api_key_google: کلید API گوگل (اختیاری)
        
    Returns:
        (کد نهایی اصلاح شده, لیست metrics)
    """
    print(f"\n{'='*60}")
    print(f"DEBUGGING WITH {len(IMPLEMENTATION_MODELS)} MODELS")
    print(f"{'='*60}")
    print(f"Starting with code length: {len(code)} characters")
    print(f"Test cases: {len(tests)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    current_code = code
    all_metrics = []
    
    # استفاده از API key پیش‌فرض اگر داده نشده
    if api_key_google is None:
        api_key_google = api_key_google_list[0] if api_key_google_list else ""
    
    # نمایش کد اولیه
    print("\n" + "="*60)
    print("INITIAL CODE:")
    print("="*60)
    print(current_code)
    print("="*60 + "\n")
    
    # اجرای یک گام با هر مدل
    for step, model_name in enumerate(IMPLEMENTATION_MODELS, start=1):
        print("\n" + "="*60)
        print(f"STEP {step}/{len(IMPLEMENTATION_MODELS)}")
        print("="*60)
        model_config = _get_model_config(model_name)
        log_step(step, f"Model: {model_config['name']}")
        log_step(step, f"Provider: {model_config['provider']}")
        
        # نمایش کد فعلی
        print(f"\n📝 CURRENT CODE (Step {step}):")
        print("-" * 60)
        print(current_code)
        print("-" * 60)
        
        # اجرای دیباگ و رفع مشکل
        print(f"\n🔄 Executing code and running tests...")
        new_code, metrics = await debug_and_fix_with_model(
            current_code,
            tests,
            model_config,
            step,
            api_key_google=api_key_google
        )
        
        all_metrics.append(metrics)
        
        # نمایش نتایج
        print(f"\n📊 RESULTS (Step {step}):")
        print("-" * 60)
        print(f"  Model: {model_config['name']}")
        print(f"  Success: {metrics.success}")
        print(f"  Test Passed: {metrics.test_passed}")
        print(f"  Time: {metrics.time_ms:.2f}ms")
        if metrics.error:
            print(f"  Error: {metrics.error[:200]}")
        if metrics.output:
            print(f"  Output: {metrics.output[:200]}")
        print("-" * 60)
        
        # بررسی نتیجه
        if metrics.test_passed:
            print(f"\n✅ SUCCESS! All tests passed at step {step}!")
            log_step(step, "✓ All tests passed! No further fixes needed.")
            current_code = new_code
            print(f"\n📝 FINAL CODE (Step {step}):")
            print("-" * 60)
            print(current_code)
            print("-" * 60)
            break
        
        if not metrics.success and metrics.error:
            print(f"\n⚠️  ERROR at step {step}:")
            print(f"  {_truncate_message(metrics.error)}")
            log_step(step, f"⚠ Error: {_truncate_message(metrics.error)}")
        
        # بررسی تغییرات کد
        if len(new_code) > 0 and new_code != current_code:
            previous_code = current_code
            current_code = new_code
            print(f"\n✏️  CODE UPDATED at step {step}:")
            print(f"  Previous length: {len(previous_code)} chars")
            print(f"  New length: {len(new_code)} chars")
            print(f"\n📝 NEW CODE (Step {step}):")
            print("-" * 60)
            print(new_code)
            print("-" * 60)
            log_step(step, f"✓ Code updated ({len(new_code)} characters)")
        else:
            print(f"\n○ No changes made at step {step}")
            log_step(step, "○ No changes made")
        
        if metrics.output and not metrics.test_passed:
            print(f"  Status: {_truncate_message(metrics.output)}")
            log_step(step, f"Status: {_truncate_message(metrics.output)}")
        
        print("="*60)
    
    duration = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("🎉 DEBUGGING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 SUMMARY:")
    print(f"  • Total steps executed: {len(all_metrics)}/{len(IMPLEMENTATION_MODELS)}")
    print(f"  • Total duration: {duration:.2f}s")
    print(f"  • Average time per step: {duration/len(all_metrics):.2f}s")
    print(f"  • Initial code length: {len(code)} characters")
    print(f"  • Final code length: {len(current_code)} characters")
    
    # شمارش تست‌های پاس شده
    passed_count = sum(1 for m in all_metrics if m.test_passed)
    if passed_count > 0:
        passed_steps = [i+1 for i, m in enumerate(all_metrics) if m.test_passed]
        print(f"  • ✅ Tests passed at step(s): {passed_steps}")
        print(f"  • 🎯 First success at step: {passed_steps[0]}")
    else:
        print(f"  • ⚠️  No steps achieved all tests passing")
    
    # نمایش آمار مدل‌ها
    successful_models = [m.model_used for m in all_metrics if m.test_passed]
    if successful_models:
        print(f"  • 🏆 Successful model(s): {successful_models[0]}")
    
    print(f"\n📝 FINAL CODE:")
    print("-" * 60)
    print(current_code)
    print("-" * 60)
    print(f"{'='*60}\n")
    
    return current_code, all_metrics


def extract_codes_from_file(file_path: str) -> List[Dict[str, str]]:

    codes = []
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return codes
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # پیدا کردن همه RESULT ها
    pattern = r'RESULT (\d+) \(from ([^)]+)\):\s*\n-+\s*\n(.*?)(?=\nRESULT \d+ \(from|$)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        result_num = match.group(1)
        model_name = match.group(2).strip()
        code = match.group(3).strip()
        
        # حذف کدهای خیلی کوتاه (احتمالاً ناقص)

    
    return codes


def load_test_cases_from_json(json_path: str, index: int = 2) -> List[Dict]:

    if not os.path.exists(json_path):
        print(f"⚠️  File not found: {json_path}")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if index >= len(data):
        print(f"⚠️  Index {index} out of range. Total problems: {len(data)}")
        return []
    
    problem = data[index]
    
    # پیدا کردن samples
    samples = []
    if 'implementation_view' in problem:
        samples = problem['implementation_view'].get('samples', [])
    elif 'algorithm_view' in problem:
        samples = problem['algorithm_view'].get('samples', [])
    
    # تبدیل به فرمت مورد نیاز
    test_cases = []
    for sample in samples:
        test_cases.append({
            'input': sample.get('input', ''),
            'expected_output': sample.get('output', '').strip()
        })
    
    return test_cases


# مثال استفاده و تست
async def _30step(code: str, test: List[Dict],output_dir: str):

        # کد پیش‌فرض
    sample_code = """#include <cstdio>
#    #include <cstdlib>
#    #include <algorithm>

/*---------------------------  Fast scanner  ---------------------------*/
#                   struct FastScanner {
    static const std::size_t SZ = 1 << 20;          // 1 MiB buffer
    char buf[SZ];
    std::size_t idx = 0, len = 0;

    inline char readChar() {
        if (idx >= len) {
            len = fread(buf, 1, SZ, stdin);
            idx = 0;
            if (len == 0) return 0;               // EOF
        }
        return buf[idx++];
    }

    template <typename T>
    bool nextInt(T &out) {                         // reads signed integer
        char c = readChar();
        if (!c) return false;

        // skip non‑digit / non‑sign characters
        while (c != '-' && (c < '0' || c > '9')) {
            c = readChar();
            if (!c) return false;
        }

        bool neg = false;
        if (c == '-') {
            neg = true;
            c = readChar();
        }

        T val = 0;
        while (c >= '0' && c <= '9') {
            val = val * 10 + (c - '0');
            c = readChar();
        }
        out = neg ? -val : val;
        return true;
    }
};

/*---------------------------  Fast output  ---------------------------*/
static inline void writeLL(long long x) {
    char out[32];
    int p = 0;
    if (x == 0) {
        putchar_unlocked('0');
        putchar_unlocked('\n');
        return;
    }
    bool neg = x < 0;
    if (neg) x = -x;
    while (x) {
        out[p++] = char('0' + (x % 10));
        x /= 10;
    }
    if (neg) putchar_unlocked('-');
    while (p--) putchar_unlocked(out[p]);
    putchar_unlocked('\n');
}

/*---------------------------  Main  ---------------------------*/
int main() {
    FastScanner in;
    long long n_ll, k_ll;
    if (!in.nextInt(n_ll) || !in.nextInt(k_ll)) return 0;

    if (n_ll <= 0 || k_ll <= 0 || k_ll > n_ll) return 0;

    std::size_t n = static_cast<std::size_t>(n_ll);
    std::size_t k = static_cast<std::size_t>(k_ll);

    // allocate contiguous array
    long long *a = static_cast<long long*>(malloc(n * sizeof(long long)));
    if (!a) return 0;                     // allocation failure

    for (std::size_t i = 0; i < n; ++i) {
        if (!in.nextInt(a[i])) {          // malformed input
            free(a);
            return 0;
        }
    }

    // k‑th largest → element at position k‑1 after nth_element with greater<>
    std::nth_element(a, a + (k - 1), a + n,
                     [](const long long &x, const long long &y) { return x > y; });

    writeLL(a[k - 1]);

    free(a);
    return 0;
}
"""
    
    
    # بارگذاری تست کیس‌ها
    #test_cases = [
    #    {
    #      "input": "6\n13 3 6 20 10 15\n0 0 0 1 1 2 2 1 0 0",
    #      "output": "35"
    #    }
    #  ]
    
 # اجرای دیباگ
    final_code, metrics = await debug_with_all_models(
                code=code,
                tests=test,
                problem="",
                api_key_google=api_key_google_list[1]
            )
            
            # بررسی نتیجه
    
    # نوشتن کد نهایی در فایل
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_code.cpp")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_code)
    
    print(f"\n✅ Final code written to: {output_file}")
    
    return final_code



