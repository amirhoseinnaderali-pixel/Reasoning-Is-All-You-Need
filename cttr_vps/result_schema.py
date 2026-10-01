from __future__ import annotations

SCHEMA_VERSION = "1.0"


def make_record(
    *,
    experiment_id: str,
    problem_id: str,
    method: str,
    candidate_count: int,
    model_calls: int,
    wall_time_seconds: float,
    evaluation: dict,
    config: dict,
    events=None,
    models=None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "problem_id": problem_id,
        "method": method,
        "models": models or [],
        "candidate_count": candidate_count,
        "model_calls": model_calls,
        "compiled": bool(evaluation.get("compiled", False)),
        "tests_passed": int(evaluation.get("tests_passed", 0)),
        "tests_failed": int(evaluation.get("tests_failed", 0)),
        "total_tests": int(evaluation.get("total_tests", 0)),
        "solved": bool(evaluation.get("all_passed", False)),
        "latency_ms": evaluation.get("latency_ms"),
        "memory_mb": evaluation.get("memory_mb"),
        "verification_attempts": int(evaluation.get("total_tests", 0)),
        "error_type": evaluation.get("error_type"),
        "wall_time_seconds": wall_time_seconds,
        "config": config,
        "events": events or [],
    }
