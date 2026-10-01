from cttr_vps.result_schema import new_record
from cttr_vps.results import aggregate, pass_at_k


def test_new_record_is_explicit_about_missing_measurements():
    record = new_record("exp-1", "problem-1", "single_pass").to_dict()
    assert record["solved"] is False
    assert record["tokens"] is None
    assert record["latency_seconds"] is None
    assert record["schema_version"] == "1.0"


def test_pass_at_k_edge_cases():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(0, 0, 1) is None


def test_aggregate_keeps_method_counts():
    summary = aggregate([
        {"method": "single_pass", "solved": True, "model_calls": 1, "latency_seconds": 1.0},
        {"method": "single_pass", "solved": False, "model_calls": 1, "latency_seconds": None},
        {"method": "cttr_vps", "solved": False, "model_calls": 5, "latency_seconds": 2.0},
    ])
    rows = {row["method"]: row for row in summary["methods"]}
    assert rows["single_pass"]["problems"] == 2
    assert rows["single_pass"]["solved"] == 1
    assert rows["single_pass"]["success_rate"] == 0.5
