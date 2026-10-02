import hashlib, json
from pathlib import Path
from cttr_vps.hardening import candidate_set_hash,load_protocol,preflight
from cttr_vps.result_schema_v2 import validate_record
def test_candidate_hash_is_stable():
    assert candidate_set_hash(["a","b"])==candidate_set_hash(["a","b"])
    assert candidate_set_hash(["a","b"])!=candidate_set_hash(["b","a"])
def test_protocol_is_exp001_and_digest_pinned():
    p=load_protocol(); assert p["experiment_id"]=="EXP-001"; assert "@" in p["execution"]["image"]
def test_real_preflight_fails_closed_without_frozen_benchmark():
    r=preflight("real"); assert r["status"]=="FAIL"; assert any("Benchmark" in x for x in r["failures"])
def test_schema_rejects_missing_fields():
    assert validate_record({})

