from cttr_vps.hardening import candidate_set_hash,load_protocol,preflight,load_manifest,verify_public_materialization

def test_candidate_hash_is_stable():
    assert candidate_set_hash(["a","b"])==candidate_set_hash(["a","b"])
    assert candidate_set_hash(["a","b"])!=candidate_set_hash(["b","a"])

def test_protocol_is_exp001_and_digest_pinned():
    p=load_protocol();assert p["experiment_id"]=="EXP-001";assert "@" in p["execution"]["image"]

def test_real_preflight_fails_closed_on_missing_hidden_artifact():
    r=preflight("real");assert r["status"]=="FAIL";assert any("not fully frozen" in x or "Hidden-test artifact" in x for x in r["failures"])

def test_public_materialization_is_verified():
    result=verify_public_materialization(load_manifest());assert result["task_count"]==93
