from cttr_vps.evaluation import classify_error, select_best_verified


def test_error_classification_distinguishes_compile_and_wrong_output():
    assert classify_error(False, []) == "compilation_error"
    assert classify_error(True, [{"success": True, "passed": False}]) == "wrong_output"
    assert classify_error(True, [{"success": False, "timeout": True, "passed": False}]) == "timeout"


def test_selection_prefers_solved_candidate():
    rows = [
        {"solved": False, "tests_passed": 4, "latency_seconds": 0.1, "memory_mb": 1},
        {"solved": True, "tests_passed": 5, "latency_seconds": 0.2, "memory_mb": 2},
    ]
    assert select_best_verified(rows) == 1
