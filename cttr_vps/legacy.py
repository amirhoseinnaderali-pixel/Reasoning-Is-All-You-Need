from __future__ import annotations

from typing import Any


def get_model_config(model_name: str) -> dict[str, str]:
    from cpp_pipe import ModelConfig

    for item in ModelConfig.STRONG_MODELS:
        if item["name"] == model_name:
            return dict(item)
    provider = "google-genai" if model_name.startswith("gemini-") else "ollama"
    return {"provider": provider, "model": model_name, "name": model_name}

async def generate_code(
    plan: str,
    problem: str,
    *,
    best_code: str = "",
    model_name: str,
    api_key_google: str = "",
    api_key_ollama: str = "",
    iteration: int = 1,
    speed: int = 1,
    memory: int = 512,
) -> tuple[str, Any]:
    from cpp_pipe import plan_to_code

    return await plan_to_code(
        plan,
        problem,
        best_code,
        get_model_config(model_name),
        iteration,
        api_key_google=api_key_google,
        speed=speed,
        memory=memory,
    )


async def debug_code(
    code: str,
    tests: list[dict[str, Any]],
    problem: str,
    model_name: str,
    api_key_google: str = "",
    api_key_ollama: str = "",
):
    from _30step import debug_and_fix_with_model

    return await debug_and_fix_with_model(
        code,
        tests,
        get_model_config(model_name),
        iteration=1,
        problem=problem,
        api_key_google=api_key_google,
    )


async def run_planning_round(
    problem_text: str,
    *,
    google_key: str,
    ollama_key: str,
    filename: str,
    previous_results: list[Any],
    output_dir: str,
):
    from planning import planning

    return planning(
        problem_text,
        google_key,
        ollama_key,
        filename,
        previous_planning_results=previous_results,
        output_dir=output_dir,
    )


async def optimize_codes(candidate_codes: list[str], output_dir: str) -> list[str]:
    from optim import optimizer

    return await optimizer(candidate_codes, output_dir)


async def debug_all_models(
    code: str,
    tests: list[dict[str, Any]],
    problem: str,
    google_key: str,
):
    from _30step import debug_with_all_models

    return await debug_with_all_models(code, tests, problem=problem, api_key_google=google_key)
        previous_planning_results=previous_results,
        output_dir=output_dir,
    )


async def optimize_codes(candidate_codes: list[str], output_dir: str) -> list[str]:
    from optim import optimizer

    return await optimizer(candidate_codes, output_dir)


async def debug_all_models(code: str, tests: list[dict[str, Any]], problem: str, google_key: str):
    from _30step import debug_with_all_models

    return await debug_with_all_models(code, tests, problem=problem, api_key_google=google_key)
