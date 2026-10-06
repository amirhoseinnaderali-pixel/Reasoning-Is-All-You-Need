from __future__ import annotations

from cttr_vps.evaluation import select_best


def test_selection_does_not_pick_first_candidate_when_a_later_candidate_solves_all_tests():
    evaluations = [
        {
            "compiled": True,
            "tests_passed": 1,
            "total_tests": 3,
            "all_passed": False,
            "latency_ms": 1.0,
            "memory_mb": 1.0,
        },
        {
            "compiled": True,
            "tests_passed": 3,
            "total_tests": 3,
            "all_passed": True,
            "latency_ms": 3.0,
            "memory_mb": 2.0,
        },
    ]
    assert select_best(evaluations) == 1


def test_selection_returns_no_candidate_for_empty_input():
    assert select_best([]) is None
