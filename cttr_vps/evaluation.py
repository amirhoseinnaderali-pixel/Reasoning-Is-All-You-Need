from __future__ import annotations

from typing import Any


def evaluate_candidate(code: str, tests: list[dict[str, Any]], timeout_s: float = 5.0, memory_mb: int = 512) -> dict[str, Any]:
    from cpp_pipe import CppSandbox

    with CppSandbox(timeout=timeout_s, memory_limit_mb=memory_mb) as sandbox:
        compiled, compile_error = sandbox.compile(code)
        if not compiled:
            return {
                "compiled": False,
                "tests_passed": 0,
                "tests_failed": len(tests),
                "total_tests": len(tests),
                "all_passed": False,
                "latency_ms": None,
                "memory_mb": None,
                "error_type": "compile_error",
                "compile_error": compile_error,
                "tests": [],
            }

        rows = []
        passed = 0
        total_ms = 0.0
        total_memory = 0.0

        for index, test in enumerate(tests, 1):
            result = sandbox.execute(str(test.get("input", "")))
            actual = str(result.get("output", "")).strip()
            expected = str(test.get("expected_output", test.get("output", ""))).strip()
            ok = bool(result.get("success", False)) and actual == expected
            rows.append({
                "test_index": index,
                "passed": ok,
                "expected": expected,
                "actual": actual,
                "error": result.get("error", ""),
                "time_ms": result.get("time_ms", 0.0),
                "memory_mb": result.get("memory_mb", 0.0),
                "timeout": result.get("timeout", False),
            })
            passed += int(ok)
            total_ms += float(result.get("time_ms", 0.0) or 0.0)
            total_memory += float(result.get("memory_mb", 0.0) or 0.0)

        error_type = None
        if any(row["timeout"] for row in rows):
            error_type = "timeout"
        elif any(row["error"] and not row["passed"] for row in rows):
            error_type = "runtime_error"
        elif passed < len(rows):
            error_type = "wrong_output"

        return {
            "compiled": True,
            "tests_passed": passed,
            "tests_failed": len(rows) - passed,
            "total_tests": len(rows),
            "all_passed": bool(rows) and passed == len(rows),
            "latency_ms": total_ms,
            "memory_mb": total_memory / len(rows) if rows else 0.0,
            "error_type": error_type,
            "tests": rows,
        }


def select_best(evaluations: list[dict[str, Any]]) -> int | None:
    if not evaluations:
        return None
    return max(
        range(len(evaluations)),
        key=lambda i: (
            int(evaluations[i].get("all_passed", False)),
            evaluations[i].get("tests_passed", 0),
            int(evaluations[i].get("compiled", False)),
            -(evaluations[i].get("latency_ms") if evaluations[i].get("latency_ms") is not None else 10**18),
            -(evaluations[i].get("memory_mb") if evaluations[i].get("memory_mb") is not None else 10**18),
            -i,
        ),
    )
