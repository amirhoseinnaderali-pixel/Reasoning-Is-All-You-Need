from cttr_vps.methods import select_plan


def test_select_plan_uses_modal_normalized_algorithm_name():
    rows = [
        {"success": True, "algorithm": "Binary Search", "approach": "short", "time_complexity": "O(n)", "space_complexity": "O(n)"},
        {"success": True, "algorithm": "binary-search!", "approach": "a longer approach", "time_complexity": "O(n log n)", "space_complexity": "O(n)"},
        {"success": True, "algorithm": "Greedy", "approach": "other", "time_complexity": "O(n)", "space_complexity": "O(1)"},
    ]
    plan, chosen = select_plan(rows)
    assert chosen["algorithm"] == "Binary Search"
    assert "a longer approach" in plan
