from __future__ import annotations

from dataclasses import asdict
from typing import Any


def classify_error(compiled: bool, execution_results: list[dict[str, Any]]) -> str | None:
    if not compiled:
        return "compilation_error"
    if not execution_results:
        return None
    for result in execution_results:
        if result.get("timeout"):
            return "timeout"
        if not result.get("success", False):
            return "runtime_error"
        if not result.get("passed", False):
            return "wrong_output"
    return None


def evaluate_candidate(
    code: str,
    tests: list[dict[str, Any]],
    *,
    timeout_s: float = 5.0,
    memory_limit_mb: int = 512,
    max_tests: int | None = None,
) -> dict[str, Any]:
    """Compile and execute one candidate; never marks untested code as solved."""
    from cpp_pipe import CppSandbox

    selected_tests = tests if max_tests is None else tests[:max_tests]
    with CppSandbox(timeout=timeout_s, memory_limit_mb=memory_limit_mb) as sandbox:
        compiled, compile_error = sandbox.compile(code)
        if not compiled:
            return {
                "compiled": False,
                "tests_passed": 0,
                "tests_failed": len(selected_tests),
                "solved": False,
                "latency_seconds": None,
                "memory_mb": None,
                "verification_attempts": len(selected_tests),
                "error_type": "compilation_error",
                "compile_error": compile_error,
                "tests": [],
            }

        results: list[dict[str, Any]] = []
        total_ms = 0.0
        total_memory = 0.0
        passed = 0

        for test in selected_tests:
            result = sandbox.execute(str(test.get("input", "")))
            expected = str(test.get("expected_output", test.get("output", ""))).strip()
            actual = str(result.get("output", "")).strip()
            ok = bool(result.get("success", False)) and actual == expected
            row = dict(result)
            row["passed"] = ok
            row["expected_output"] = expected
            row["actual_output"] = actual
            results.append(row)
            total_ms += float(result.get("time_ms", 0.0) or 0.0)
            total_memory += float(result.get("memory_mb", 0.0) or 0.0)
            if ok:
                passed += 1

        latency_seconds = total_ms / 1000.0 if results else 0.0
        avg_memory_mb = total_memory / len(results) if results else 0.0
        return {
            "compiled": True,
            "tests_passed": passed,
            "tests_failed": len(results) - passed,
            "solved": bool(results) and passed == len(results),
            "latency_seconds": latency_seconds,
            "memory_mb": avg_memory_mb,
            "verification_attempts": len(results),
            "error_type": classify_error(True, results),
            "tests": results,
        }


def select_best_verified(evaluations: list[dict[str, Any]]) -> int | None:
    """Return an index using explicit, reproducible selection criteria."""
    if not evaluations:
        return None

    ranked = sorted(
        range(len(evaluations)),
        key=lambda i: (
            not evaluations[i].get("solved", False),
            -int(evaluations[i].get("tests_passed", 0)),
            evaluations[i].get("latency_seconds") if evaluations[i].get("latency_seconds") is not None else float("inf"),
            evaluations[i].get("memory_mb") if evaluations[i].get("memory_mb") is not None else float("inf"),
        ),
    )
    return ranked[0]


def metric_snapshot(evaluation: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "compiled",
        "tests_passed",
        "tests_failed",
        "solved",
        "latency_seconds",
        "memory_mb",
        "verification_attempts",
        "error_type",
    )
    return {key: evaluation.get(key) for key in keys}
