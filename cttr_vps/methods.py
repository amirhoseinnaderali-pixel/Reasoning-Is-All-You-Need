from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import get_google_api_keys, get_ollama_api_keys, require_google_api_key
from .evaluation import evaluate_candidate, metric_snapshot, select_best_verified
from .legacy import debug_all_models, debug_code, generate_code, optimize_codes, run_planning_round, strong_models


def _normalise_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def select_plan(planning_results: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select a representative plan using an explicit modal-label + detail rule."""
    successful = [item for item in planning_results if item.get("algorithm")]
    if not successful:
        raise RuntimeError("No successful planning result is available.")

    labels = Counter(_normalise_label(str(item["algorithm"])) for item in successful)
    modal_label, _ = labels.most_common(1)[0]
    candidates = [item for item in successful if _normalise_label(str(item["algorithm"])) == modal_label]
    chosen = max(candidates, key=lambda item: len(str(item.get("approach", ""))))
    return json.dumps(
        {
            "algorithm": chosen.get("algorithm", ""),
            "approach": chosen.get("approach", ""),
            "time_complexity": chosen.get("time_complexity", ""),
            "space_complexity": chosen.get("space_complexity", ""),
        },
        ensure_ascii=False,
        indent=2,
    ), chosen


def _model_names(names: list[str] | None, count: int) -> list[str]:
    if names:
        return [names[i % len(names)] for i in range(count)]
    pool = strong_models()
    return [pool[i % len(pool)]["name"] for i in range(count)]


async def generate_samples(
    plan: str,
    problem: str,
    *,
    model_names: list[str] | None = None,
    count: int = 1,
    best_code: str = "",
    api_key_google: str = "",
    api_key_ollama: str = "",
    speed: int = 1,
    memory: int = 512,
) -> tuple[list[str], list[dict[str, Any]]]:
    names = _model_names(model_names, count)
    tasks = [
        generate_code(
            plan,
            problem,
            best_code=best_code,
            model_name=name,
            api_key_google=api_key_google if name.startswith("gemini-") else "",
            api_key_ollama=api_key_ollama if not name.startswith("gemini-") else "",
            iteration=i + 1,
            speed=speed,
            memory=memory,
        )
        for i, name in enumerate(names)
    ]
    outputs = await asyncio.gather(*tasks, return_exceptions=True)
    candidates: list[str] = []
    events: list[dict[str, Any]] = []
    for i, output in enumerate(outputs):
        if isinstance(output, Exception):
            events.append({"model": names[i], "success": False, "error": str(output)})
            continue
        code, metrics = output
        event = {"model": names[i], "success": bool(metrics.success and code), "metrics": getattr(metrics, "__dict__", {})}
        events.append(event)
        if code:
            candidates.append(code)
    return candidates, events


async def run_single_pass(plan: str, problem: str, tests: list[dict[str, Any]], config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    google_key = require_google_api_key()
    ollama_keys = get_ollama_api_keys()
    model = config.get("model", "gemini-2.5-flash")
    code, event = await generate_code(
        plan,
        problem,
        model_name=model,
        api_key_google=google_key if model.startswith("gemini-") else "",
        api_key_ollama=ollama_keys[0] if ollama_keys else "",
    )
    evaluation = evaluate_candidate(code, tests, timeout_s=float(config.get("verification", {}).get("timeout_s", 5)), memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)), max_tests=config.get("verification", {}).get("max_tests")) if code else {"compiled": False, "tests_passed": 0, "tests_failed": len(tests), "solved": False, "latency_seconds": None, "memory_mb": None, "verification_attempts": 0, "error_type": "generation_failed"}
    return {"code": code, "candidates": [code] if code else [], "evaluations": [evaluation], "events": [getattr(event, "__dict__", {})], "model_calls": 1}


async def run_multi_sample(plan: str, problem: str, tests: list[dict[str, Any]], config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    google_key = require_google_api_key()
    ollama_keys = get_ollama_api_keys()
    count = int(config.get("samples", 8))
    candidates, events = await generate_samples(
        plan,
        problem,
        count=count,
        api_key_google=google_key,
        api_key_ollama=ollama_keys[0] if ollama_keys else "",
    )
    evaluations = [
        evaluate_candidate(c, tests, timeout_s=float(config.get("verification", {}).get("timeout_s", 5)), memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)), max_tests=config.get("verification", {}).get("max_tests"))
        for c in candidates
    ]
    selected = select_best_verified(evaluations)
    return {"code": candidates[selected] if selected is not None else "", "candidates": candidates, "evaluations": evaluations, "events": events, "model_calls": count}
    events: list[dict[str, Any]] = []
    for i, output in enumerate(outputs):
        if isinstance(output, Exception):
            events.append({"model": names[i], "success": False, "error": str(output)})
            continue
        code, metrics = output
        event = {"model": names[i], "success": bool(metrics.success and code), "metrics": getattr(metrics, "__dict__", {})}
        events.append(event)
        if code:
            candidates.append(code)
    return candidates, events


async def run_single_pass(plan: str, problem: str, tests: list[dict[str, Any]], config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    key = require_google_api_key()
    model = config.get("model", "gemini-2.5-flash")
    code, event = await generate_code(plan, problem, model_name=model, api_key_google=key)
    evaluation = evaluate_candidate(
        code,
        tests,
        timeout_s=float(config.get("verification", {}).get("timeout_s", 5)),
        memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)),
        max_tests=config.get("verification", {}).get("max_tests"),
    ) if code else {"compiled": False, "tests_passed": 0, "tests_failed": len(tests), "solved": False, "latency_seconds": None, "memory_mb": None, "verification_attempts": 0, "error_type": "generation_failed"}
    return {"code": code, "candidates": [code] if code else [], "evaluations": [evaluation], "events": [getattr(event, "__dict__", {})], "model_calls": 1}


async def run_multi_sample(plan: str, problem: str, tests: list[dict[str, Any]], config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    key = require_google_api_key()
    count = int(config.get("samples", 8))
    candidates, events = await generate_samples(plan, problem, count=count, api_key_google=key)
    evaluations = [
        evaluate_candidate(c, tests, timeout_s=float(config.get("verification", {}).get("timeout_s", 5)), memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)), max_tests=config.get("verification", {}).get("max_tests"))
        for c in candidates
    ]
    selected = select_best_verified(evaluations)
    return {"code": candidates[selected] if selected is not None else "", "candidates": candidates, "evaluations": evaluations, "events": events, "model_calls": count}


async def run_self_refinement(plan: str, problem: str, tests: list[dict[str, Any]], config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    key = require_google_api_key()
    model = config.get("model", "gemini-2.5-flash")
    rounds = int(config.get("refinement_rounds", 3))
    code = ""
    events: list[dict[str, Any]] = []
    for i in range(rounds + 1):
        new_code, metrics = await generate_code(plan, problem, best_code=code, model_name=model, api_key_google=key, iteration=i + 1)
        code = new_code or code
        events.append({"round": i + 1, "model": model, "metrics": getattr(metrics, "__dict__", {})})
    evaluation = evaluate_candidate(code, tests, timeout_s=float(config.get("verification", {}).get("timeout_s", 5)), memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)), max_tests=config.get("verification", {}).get("max_tests"))
    return {"code": code, "candidates": [code] if code else [], "evaluations": [evaluation], "events": events, "model_calls": rounds + 1}


async def run_execution_refinement(plan: str, problem: str, tests: list[dict[str, Any]], config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    key = require_google_api_key()
    model = config.get("model", "gemini-2.5-flash")
    rounds = int(config.get("refinement_rounds", 5))
    code, _ = await generate_code(plan, problem, model_name=model, api_key_google=key)
    events: list[dict[str, Any]] = []
    calls = 1
    for i in range(rounds):
        evaluation = evaluate_candidate(code, tests, timeout_s=float(config.get("verification", {}).get("timeout_s", 5)), memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)), max_tests=config.get("verification", {}).get("max_tests"))
        events.append({"round": i + 1, "evaluation": metric_snapshot(evaluation)})
        if evaluation["solved"]:
            break
        code, metrics = await debug_code(code, tests, problem, model, api_key_google=key)
        calls += 1
        events.append({"round": i + 1, "model": model, "metrics": getattr(metrics, "__dict__", {})})
    final_eval = evaluate_candidate(code, tests, timeout_s=float(config.get("verification", {}).get("timeout_s", 5)), memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)), max_tests=config.get("verification", {}).get("max_tests"))
    return {"code": code, "candidates": [code], "evaluations": [final_eval], "events": events, "model_calls": calls}


async def run_cttr_vps(plan_hint: str, problem: dict[str, Any], tests: list[dict[str, Any]], config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    google_keys = get_google_api_keys()
    ollama_keys = get_ollama_api_keys()
    if not google_keys:
        raise RuntimeError("CTTR-VPS requires GOOGLE_API_KEY or GOOGLE_API_KEYS.")
    if not ollama_keys:
        raise RuntimeError("CTTR-VPS planning requires OLLAMA_API_KEY or OLLAMA_API_KEYS.")

    problem_text = json.dumps(problem.get("planning_view", problem), ensure_ascii=False, indent=2)
    rounds = int(config.get("planning_rounds", 3))
    previous: list[Any] = []
    planning_events: list[Any] = []
    final_round: list[dict[str, Any]] = []
    for round_id in range(rounds):
        final_round = await run_planning_round(
            problem_text,
            google_key=google_keys[round_id % len(google_keys)],
            ollama_key=ollama_keys[round_id % len(ollama_keys)],
            filename=f"planning_round_{round_id + 1:02d}.json",
            previous_results=previous,
            output_dir=str(out),
        )
        planning_events.append(final_round)
        previous = final_round

    plan_text, selected_plan = select_plan(final_round)
    algorithm_problem = json.dumps(problem.get("algorithm_view", problem), ensure_ascii=False, indent=2)

    generation_rounds = int(config.get("generation_rounds", 5))
    candidates: list[str] = []
    generation_events: list[Any] = []
    context = ""
    for round_id in range(generation_rounds):
        count = int(config.get("candidates_per_round", 20))
        round_candidates, events = await generate_samples(
            plan_text,
            algorithm_problem,
            count=count,
            best_code=context,
            api_key_google=google_keys[round_id % len(google_keys)],
        )
        generation_events.append(events)
        round_file = out / f"generation_round_{round_id + 1:02d}.json"
        round_file.write_text(json.dumps({"candidates": round_candidates, "events": events}, ensure_ascii=False, indent=2), encoding="utf-8")
        if round_candidates:
            candidates = round_candidates
            context = "\n\nNEXT CANDIDATE:\n\n".join(round_candidates)

    optimized = await optimize_codes(candidates, str(out)) if candidates else []
    optimization_candidates = optimized if optimized else candidates
    evaluations = [
        evaluate_candidate(
            code,
            tests,
            timeout_s=float(config.get("verification", {}).get("timeout_s", 5)),
            memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)),
            max_tests=config.get("verification", {}).get("max_tests"),
        )
        for code in optimization_candidates
        if code
    ]
    selected = select_best_verified(evaluations)
    current_code = optimization_candidates[selected] if selected is not None else (optimization_candidates[0] if optimization_candidates else "")

    debug_models = config.get("debug_models", True)
    debug_events: list[Any] = []
    if current_code and debug_models:
        current_code, debug_metrics = await debug_all_models(current_code, tests, algorithm_problem, google_keys[0])
        debug_events = [getattr(item, "__dict__", {}) for item in debug_metrics]
        final_evaluation = evaluate_candidate(current_code, tests, timeout_s=float(config.get("verification", {}).get("timeout_s", 5)), memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)), max_tests=config.get("verification", {}).get("max_tests"))
    else:
        final_evaluation = evaluate_candidate(current_code, tests, timeout_s=float(config.get("verification", {}).get("timeout_s", 5)), memory_limit_mb=int(config.get("verification", {}).get("memory_mb", 512)), max_tests=config.get("verification", {}).get("max_tests")) if current_code else {"compiled": False, "tests_passed": 0, "tests_failed": len(tests), "solved": False, "latency_seconds": None, "memory_mb": None, "verification_attempts": 0, "error_type": "generation_failed"}

    return {
        "code": current_code,
        "candidates": optimization_candidates,
        "evaluations": [final_evaluation],
        "events": {
            "planning": planning_events,
            "selected_plan": selected_plan,
            "generation": generation_events,
            "optimization_candidates": len(optimization_candidates),
            "debugging": debug_events,
        },
        "model_calls": (
            sum(len(r) for r in planning_events)
            + generation_rounds * int(config.get("candidates_per_round", 20))
            + (4 * len(strong_models()) if optimization_candidates else 0)
            + len(debug_events)
        ),
    }


METHODS = {
    "single_pass": run_single_pass,
    "multi_sample": run_multi_sample,
    "self_refine": run_self_refinement,
    "execution_refine": run_execution_refinement,
    "cttr_vps": run_cttr_vps,
}
