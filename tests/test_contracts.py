import os
from pathlib import Path
import pytest
from cttr_vps.hardening import hidden_path,load_protocol
def test_hidden_tests_must_be_outside_repo(monkeypatch):
    monkeypatch.setenv("CTTR_HIDDEN_TESTS_PATH",str(Path.cwd()/ "benchmark/hidden.json"))
    with pytest.raises(RuntimeError): hidden_path()
def test_condition_budgets_are_explicit():
    p=load_protocol()["conditions"]
    assert {"C0","C1","C2","C3","C4"}==set(p)
    assert all("max_model_calls" in x and "max_debug_steps" in x for x in p.values())
    assert p["C4"]["max_model_calls"]>p["C1"]["max_model_calls"]


def test_retry_policy_is_fail_closed():
    p=load_protocol()["model"]["retry_policy"]
    assert p["max_retries"]==0
    assert p["on_retryable_error"]=="fail_closed"
