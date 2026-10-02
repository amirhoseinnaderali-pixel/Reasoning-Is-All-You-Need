import json
from cttr_vps.hardening import load_manifest,verify_public_materialization

def test_public_materialization_matches_manifest():
    m=load_manifest()
    result=verify_public_materialization(m)
    assert result["task_count"]==93
    assert result["materialization_sha256"]==m["materialization_sha256"]

def test_public_materialization_contains_no_hidden_tests():
    m=load_manifest()
    rows=json.loads(open("benchmark/materialized.json",encoding="utf-8").read())["tasks"]
    assert all("hidden_tests" not in t or t["hidden_tests"] is None for t in rows)
    assert m["hidden_tests"]["status"]=="MISSING"
