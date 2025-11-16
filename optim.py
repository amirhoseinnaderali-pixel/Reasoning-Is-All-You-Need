import asyncio
import os
import re
from datetime import datetime
from typing import List, Tuple
from cpp_pipe import (
    ModelConfig,
    call_model_with_retry____________________________________________,
    extract_code_from_response
)


def load_codes_from_file(file_path: str) -> List[str]:
    """
    کدها را از فایل می‌خواند و به صورت لیست برمی‌گرداند
    پشتیبانی از دو فرمت:
    1. "=== Code Variant \d+ ===" (از فایل‌های phase)
    2. "RESULT \d+ (from ...)" یا "RESULT \d+ (fallback)" (از فایل‌های optimize)
    
    Args:
        file_path: مسیر فایل
    
    Returns:
        لیست کدهای C++
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return []
    
    codes = []
    current_code = []
    in_code_block = False
    skip_next_separator = False
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines:
        stripped = line.strip()
        
        # Skip header lines
        if stripped.startswith("All optimization results") or stripped.startswith("Fallback code"):
            continue
        
        # Check for separator lines
        if stripped == "=" * 80:
            if skip_next_separator:
                skip_next_separator = False
                continue
            # Separator line - if we have code, save it
            if current_code and in_code_block:
                code_text = "".join(current_code).strip()
                if code_text:
                    codes.append(code_text)
                current_code = []
                in_code_block = False
            continue
        
        if stripped == "-" * 80:
            skip_next_separator = True
            in_code_block = True
            continue
        
        # Check for "=== Code Variant \d+ ===" format
        if re.match(r"^=== Code Variant \d+ ===$", stripped):
            if current_code:
                code_text = "".join(current_code).strip()
                if code_text:
                    codes.append(code_text)
                current_code = []
            in_code_block = True
            continue
        
        # Check for "RESULT \d+ (from ...)" or "RESULT \d+ (fallback)" format
        if re.match(r"^RESULT \d+ \(", stripped):
            if current_code:
                code_text = "".join(current_code).strip()
                if code_text:
                    codes.append(code_text)
                current_code = []
            in_code_block = True
            continue
        
        # Add code lines (only if we're in a code block or it's not a header line)
        if in_code_block:
            current_code.append(line)
        elif stripped and not (stripped.startswith("All") or stripped.startswith("Fallback")):
            # If not in a code block but line is not empty and not a header, start collecting
            in_code_block = True
            current_code.append(line)
    
    # آخرین کد را اضافه کن
    if current_code:
        code_text = "".join(current_code).strip()
        if code_text:
            codes.append(code_text)
    
    print(f"✓ Loaded {len(codes)} code variants from {file_path}")
    return codes


async def optimize_with_single_model(
    candidate_codes: List[str],
    model_config: dict,
    user_prompt: str,
    system_prompt: str,
    api_key_google: str = ""
) -> Tuple[str, str]:

    try:
        response = await call_model_with_retry____________________________________________(
            provider=model_config["provider"],
            model=model_config["model"],
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            api_key_google=api_key_google
        )
        
        optimized_code = extract_code_from_response(response)
        return optimized_code if optimized_code else "", model_config["name"]
    except Exception as e:
        print(f"  ✗ Error with {model_config['name']}: {str(e)[:100]}")
        return "", model_config["name"]


async def optimize_best_code(candidate_codes: List[str], output_dir: str = "pipeline_results", api_key_google: str = "", input_file_name: str = "") -> str:

    if not candidate_codes:
        return ""
    
    # استفاده از همه مدل‌های قوی
    model_configs = ModelConfig.STRONG_MODELS
    print(f"\n{'='*60}")
    print(f"Running optimization with {len(model_configs)} strong models in PARALLEL...")
    print(f"{'='*60}")
    
    # ساخت prompt برای انتخاب و بهینه‌سازی بهترین کد
    code_count = len(candidate_codes)
    versions_text = "\n\n".join([
        f"VERSION {i+1}:\n```cpp\n{code}\n```" 
        for i, code in enumerate(candidate_codes)
    ])
    
    system_prompt = """You are a C++ optimization expert. 
Analyze multiple code solutions, pick the best one, and improve it further."""

    user_prompt = f"""You have {code_count} C++ code versions:

{versions_text}

TASK:
1. Analyze all versions and pick the BEST one
2. Improve it further (optimize for speed and memory)
3. Return ONLY the final optimized code

Output format:
```cpp
[your optimized code here]
```
"""
    
    # ایجاد tasks برای همه مدل‌ها به صورت موازی
    tasks = []
    for model_config in model_configs:
        task = optimize_with_single_model(
            candidate_codes,
            model_config,
            user_prompt,
            system_prompt,
            api_key_google=api_key_google
        )
        tasks.append(task)
        print(f"  → Launched: {model_config['name']}")
    
    # اجرای همه بهینه‌سازی‌ها به صورت موازی
    print(f"\n{'='*60}")
    print("Waiting for all models to complete...")
    print(f"{'='*60}")
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # جمع‌آوری نتایج موفق
    successful_results = []
    for i, result in enumerate(results):
        model_name = model_configs[i]["name"]
        
        if isinstance(result, Exception):
            print(f"  ✗ {model_name}: Exception - {str(result)[:100]}")
            continue
        
        optimized_code, model_name_result = result
        if optimized_code:
            successful_results.append((optimized_code, model_name_result))
            print(f"  ✓ {model_name_result}: Success ({len(optimized_code)} chars)")
        else:
            print(f"  ○ {model_name_result}: Empty response")
    
    # انتخاب بهترین نتیجه (اولین نتیجه موفق)
    os.makedirs(output_dir, exist_ok=True)
    
    if not successful_results:
        print("\n⚠ No successful optimizations! Returning first input code.")
        fallback_code = candidate_codes[0] if candidate_codes else ""
        # Save fallback code to file so it can be loaded later
        if fallback_code and input_file_name:
            with open(input_file_name, "w", encoding="utf-8") as f:
                f.write(f"Fallback code (no successful optimizations):\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"RESULT 1 (fallback):\n")
                f.write("-" * 80 + "\n")
                f.write(fallback_code)
                f.write("\n\n")
            print(f"✓ Fallback code saved to: {input_file_name}")
        return fallback_code
    
    # استفاده از اولین نتیجه موفق
    best_code, best_model = successful_results[0]
    
    print(f"\n{'='*60}")
    print(f"✓ Selected best result from: {best_model}")
    print(f"✓ Total successful: {len(successful_results)}/{len(model_configs)}")
    print(f"{'='*60}")
    
    # ذخیره همه نتایج موفق
    if input_file_name:
        with open(input_file_name, "w", encoding="utf-8") as f:
            f.write(f"All optimization results ({len(successful_results)} successful):\n")
            f.write("=" * 80 + "\n\n")
            for i, (code, model_name) in enumerate(successful_results, 1):
                f.write(f"RESULT {i} (from {model_name}):\n")
                f.write("-" * 80 + "\n")
                f.write(code)
                f.write("\n\n")
        print(f"✓ All results saved to: {input_file_name}")
    

    
    return best_code


# مثال استفاده
async def optimizer(code_list: List[str],output_dir: str):
    # خواندن کدها از فایل phase5.txt



    api_key_google_list =[
        "AIzaSyBXC7krDh4mvI4VPKFUHpmkDrEcigOE00o",
        "AIzaSyAiSR_exmQehaC7Q0HPnuQUhr0S9jCtQFs",
        "AIzaSyAVhlHdikARNiTbyJLEBtExGBJPTCWucOg",
        "AIzaSyBt9wnZwb6gGm13gXIDLAs01JuF3PoSnBw",
    ]

    #phase5_file = "pipeline_results/phase5.txt"
    #code_list = load_codes_from_file(phase5_file)
    

    print("code_list_0 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    print(f"\nProcessing {len(code_list)} code variants...")
    input_file_name=output_dir+"/input_codes_1.txt"
    result = await optimize_best_code(code_list, api_key_google=api_key_google_list[0], input_file_name=input_file_name)
    code_list = load_codes_from_file(input_file_name)
    # Fallback: if file is empty, use the result string
    if not code_list and result:
        code_list = [result]
    print("code_list_1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    
    input_file_name=output_dir+"/input_codes_2.txt"
    result = await optimize_best_code(code_list, api_key_google=api_key_google_list[1], input_file_name=input_file_name)
    code_list = load_codes_from_file(input_file_name)
    # Fallback: if file is empty, use the result string
    if not code_list and result:
        code_list = [result]
    print("code_list_2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    input_file_name=output_dir+"/input_codes_3.txt"
    result = await optimize_best_code(code_list, api_key_google=api_key_google_list[2], input_file_name=input_file_name)
    code_list = load_codes_from_file(input_file_name)
    # Fallback: if file is empty, use the result string
    if not code_list and result:
        code_list = [result]
    print("code_list_3 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    

    input_file_name=output_dir+"/input_codes_4.txt"
    result = await optimize_best_code(code_list, api_key_google=api_key_google_list[3], input_file_name=input_file_name)
    code_list = load_codes_from_file(input_file_name)
    # Fallback: if file is empty, use the result string
    if not code_list and result:
        code_list = [result]
    print("code_list_4 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    # Final fallback: if still empty, return original code_list or result
    if not code_list:
        if result:
            code_list = [result]
        elif code_list is None or len(code_list) == 0:
            # Last resort: return empty list (will be handled in main.py)
            print("⚠ Warning: No codes available after optimization")
            return []
    
    return code_list