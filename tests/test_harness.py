from cttr_vps.evaluation import select_best
from cttr_vps.results import pass_at_k


def test_selection_prefers_all_passed():
    rows = [
        {"all_passed": False, "tests_passed": 3, "compiled": True, "latency_ms": 1, "memory_mb": 1},
        {"all_passed": True, "tests_passed": 3, "compiled": True, "latency_ms": 2, "memory_mb": 2},
    ]
    assert select_best(rows) == 1


def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
