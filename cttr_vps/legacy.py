from __future__ import annotations

from typing import Any


def model_config(model_name: str) -> dict[str, str]:
    from cpp_pipe import ModelConfig
    for item in ModelConfig.STRONG_MODELS:
        if item["name"] == model_name:
            return dict(item)
    provider = "google-genai" if model_name.startswith("gemini-") else "ollama"
    return {"provider": provider, "model": model_name, "name": model_name}


def available_models() -> list[str]:
    from cpp_pipe import ModelConfig
    return [item["name"] for item in ModelConfig.STRONG_MODELS]


async def generate_code(plan: str, problem: str, model_name: str, iteration: int = 1, api_key_google: str = "", api_key_ollama: str = ""):
    from cpp_pipe import plan_to_code
    return await plan_to_code(
        plan,
        problem,
        "",
        model_config(model_name),
        iteration,
        api_key_google=api_key_google,
        api_key_ollama=api_key_ollama,
    )


async def debug_code(code: str, tests: list[dict[str, Any]], problem: str, model_name: str, api_key_google: str = ""):
    from _30step import debug_and_fix_with_model
    return await debug_and_fix_with_model(
        code,
        tests,
        model_config(model_name),
        step_num=1,
        api_key_google=api_key_google,
    )


async def debug_all(code: str, tests: list[dict[str, Any]], problem: str, api_key_google: str = ""):
    from _30step import debug_with_all_models
    return await debug_with_all_models(code, tests, problem=problem, api_key_google=api_key_google)


async def optimize(candidates: list[str], output_dir: str) -> list[str]:
    from optim import optimizer
    return await optimizer(candidates, output_dir)
