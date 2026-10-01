from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .config import google_keys, ollama_keys
from .evaluation import evaluate_candidate, select_best
from .legacy import available_models, debug_all, debug_code, generate_code, optimize


def problem_text(problem: dict[str, Any]) -> str:
    return json.dumps(problem.get("implementation_view", problem), ensure_ascii=False, indent=2)


def verify_config(config: dict[str, Any]) -> tuple[float, int]:
    value = config.get("verification", {}) or {}
    return float(value.get("timeout_s", 5)), int(value.get("memory_mb", 512))


async def run_single_pass(problem: dict, tests: list[dict], config: dict, output_dir: str) -> dict:
    models = config.get("models") or [config.get("model", "gemini-2.5-flash")]
    model = str(models[0])
    gkeys, okeys = google_keys(), ollama_keys()
    key = gkeys[0] if model.startswith("gemini-") and gkeys else (okeys[0] if okeys else "")
    code, metrics = await generate_code("", problem_text(problem), model, api_key_google=key, api_key_ollama=(ollama_keys()[0] if not model.startswith("gemini-") and ollama_keys() else ""))
    timeout_s, memory_mb = verify_config(config)
    evaluation = evaluate_candidate(code, tests, timeout_s, memory_mb) if code else {"compiled": False, "tests_passed": 0, "tests_failed": len(tests), "total_tests": len(tests), "all_passed": False, "error_type": "generation_failed"}
    return {"code": code, "candidates": [code] if code else [], "evaluations": [evaluation], "model_calls": 1, "events": [getattr(metrics, "__dict__", {})], "models": [model]}


async def run_multi_sample(problem: dict, tests: list[dict], config: dict, output_dir: str) -> dict:
    count = int(config.get("samples", 8))
    models = list(config.get("models") or available_models())
    models = [models[i % len(models)] for i in range(count)]
    gkeys, okeys = google_keys(), ollama_keys()

    tasks = []
    for name in models:
        key = gkeys[0] if str(name).startswith("gemini-") and gkeys else (okeys[0] if okeys else "")
        tasks.append(generate_code("", problem_text(problem), str(name), api_key_google=key, api_key_ollama=(okeys[0] if not str(name).startswith("gemini-") and okeys else "")))
    outputs = await asyncio.gather(*tasks, return_exceptions=True)

    candidates, events, used_models = [], [], []
    for name, output in zip(models, outputs):
        if isinstance(output, Exception):
            events.append({"model": name, "error": str(output)})
            continue
        code, metrics = output
        used_models.append(name)
        events.append({"model": name, "metrics": getattr(metrics, "__dict__", {})})
        if code:
            candidates.append(code)

    timeout_s, memory_mb = verify_config(config)
    evaluations = [evaluate_candidate(code, tests, timeout_s, memory_mb) for code in candidates]
    selected = select_best(evaluations)
    return {"code": candidates[selected] if selected is not None else "", "candidates": candidates, "evaluations": evaluations, "model_calls": count, "events": events, "models": used_models}


async def run_self_refine(problem: dict, tests: list[dict], config: dict, output_dir: str) -> dict:
    model = str(config.get("model", "gemini-2.5-flash"))
    key = google_keys()[0] if model.startswith("gemini-") and google_keys() else ""
    rounds = int(config.get("refinement_rounds", 3))
    code = ""
    events = []
    for i in range(rounds + 1):
        code, metrics = await generate_code("", problem_text(problem), model, iteration=i + 1, api_key_google=key)
        events.append({"round": i + 1, "metrics": getattr(metrics, "__dict__", {})})
    timeout_s, memory_mb = verify_config(config)
    evaluation = evaluate_candidate(code, tests, timeout_s, memory_mb)
    return {"code": code, "candidates": [code] if code else [], "evaluations": [evaluation], "model_calls": rounds + 1, "events": events, "models": [model]}


async def run_execution_refine(problem: dict, tests: list[dict], config: dict, output_dir: str) -> dict:
    model = str(config.get("model", "gemini-2.5-flash"))
    key = google_keys()[0] if model.startswith("gemini-") and google_keys() else ""
    rounds = int(config.get("refinement_rounds", 5))
    code, _ = await generate_code("", problem_text(problem), model, api_key_google=key)
    events = []
    calls = 1
    timeout_s, memory_mb = verify_config(config)
    for i in range(rounds):
        evaluation = evaluate_candidate(code, tests, timeout_s, memory_mb)
        events.append({"round": i + 1, "evaluation": evaluation})
        if evaluation["all_passed"]:
            break
        code, metrics = await debug_code(code, tests, problem_text(problem), model, api_key_google=key)
        calls += 1
        events.append({"round": i + 1, "metrics": getattr(metrics, "__dict__", {})})
    final = evaluate_candidate(code, tests, timeout_s, memory_mb)
    return {"code": code, "candidates": [code], "evaluations": [final], "model_calls": calls, "events": events, "models": [model]}


async def run_cttr_vps(problem: dict, tests: list[dict], config: dict, output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from planning import planning

    gkeys, okeys = google_keys(), ollama_keys()
    if not gkeys or not okeys:
        raise RuntimeError("CTTR-VPS requires GOOGLE_API_KEYS and OLLAMA_API_KEYS.")

    planning_rounds = int(config.get("planning_rounds", 3))
    planning_events = []
    previous = []
    final_plans = []
    planning_problem = json.dumps(problem.get("planning_view", problem), ensure_ascii=False, indent=2)

    for round_id in range(planning_rounds):
        result = planning(
            planning_problem,
            gkeys[round_id % len(gkeys)],
            okeys[round_id % len(okeys)],
            f"planning_round_{round_id + 1:02d}.json",
            previous_planning_results=previous,
            output_dir=str(out),
        )
        planning_events.append(result)
        previous = result
        final_plans = result

    if not final_plans:
        raise RuntimeError("No planning results were produced.")

    chosen = final_plans[0]
    plan = json.dumps({
        "algorithm": chosen.get("algorithm", ""),
        "approach": chosen.get("approach", ""),
        "time_complexity": chosen.get("time_complexity", ""),
        "space_complexity": chosen.get("space_complexity", ""),
    }, ensure_ascii=False, indent=2)

    generation_rounds = int(config.get("generation_rounds", 5))
    per_round = int(config.get("candidates_per_round", 20))
    current_candidates = []
    generation_events = []

    for round_id in range(generation_rounds):
        models = available_models()[:per_round]
        tasks = []
        for index, model in enumerate(models):
            key = gkeys[round_id % len(gkeys)] if model.startswith("gemini-") else ""
            tasks.append(generate_code(plan, json.dumps(problem.get("algorithm_view", problem), ensure_ascii=False, indent=2), model, iteration=index + 1, api_key_google=key, api_key_ollama=(okeys[round_id % len(okeys)] if not model.startswith("gemini-") and okeys else "")))
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        round_candidates = []
        events = []
        for model, output in zip(models, outputs):
            if isinstance(output, Exception):
                events.append({"model": model, "error": str(output)})
                continue
            code, metrics = output
            events.append({"model": model, "metrics": getattr(metrics, "__dict__", {})})
            if code:
                round_candidates.append(code)
        generation_events.append(events)
        (out / f"generation_round_{round_id + 1:02d}.json").write_text(json.dumps({"events": events, "candidate_count": len(round_candidates)}, indent=2), encoding="utf-8")
        if round_candidates:
            current_candidates = round_candidates

    optimized = await optimize(current_candidates, str(out)) if current_candidates else []
    candidates = optimized if optimized else current_candidates
    timeout_s, memory_mb = verify_config(config)
    evaluations = [evaluate_candidate(code, tests, timeout_s, memory_mb) for code in candidates]
    selected = select_best(evaluations)
    selected_code = candidates[selected] if selected is not None else ""

    debug_events = []
    if selected_code and config.get("debug_models", True):
        selected_code, debug_metrics = await debug_all(selected_code, tests, json.dumps(problem.get("algorithm_view", problem), ensure_ascii=False, indent=2), gkeys[0])
        debug_events = [getattr(item, "__dict__", {}) for item in debug_metrics]

    final_eval = evaluate_candidate(selected_code, tests, timeout_s, memory_mb) if selected_code else {"compiled": False, "tests_passed": 0, "tests_failed": len(tests), "total_tests": len(tests), "all_passed": False, "error_type": "generation_failed"}
    return {
        "code": selected_code,
        "candidates": candidates,
        "evaluations": [final_eval],
        "model_calls": sum(len(x) for x in planning_events) + generation_rounds * per_round + (4 * len(available_models()) if candidates else 0) + len(debug_events),
        "events": {"planning": planning_events, "generation": generation_events, "debugging": debug_events},
        "models": available_models(),
    }


METHODS = {
    "single_pass": run_single_pass,
    "multi_sample": run_multi_sample,
    "self_refine": run_self_refine,
    "execution_refine": run_execution_refine,
    "cttr_vps": run_cttr_vps,
}
