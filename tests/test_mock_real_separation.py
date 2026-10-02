import json
from pathlib import Path
from scripts.scientific_audit import result_checks
from cttr_vps.hardening import load_protocol
def test_non_real_artifact_cannot_claim_solved(tmp_path,monkeypatch):
    r=tmp_path/"results";(r/"x").mkdir(parents=True)
    row={"schema_version":"2.0","experiment_id":"EXP-001","run_id":"x","task_id":"t","seed":1,"method":"single_pass","candidate_set_hash":"0"*64,"config_hash":"x","benchmark_hash":"x","model_config_hash":"x","model_revision":"gemini-2.5-flash","candidate_count":1,"model_calls":1,"token_usage":[],"refinement_debug_steps":0,"visible_evaluation":{},"hidden_evaluation":{},"solved":True,"failure_type":None,"wall_clock_seconds":0,"budget_usage":{},"environment":{},"git_sha":"x","execution_mode":"smoke","status":"VALIDATION_ONLY"}
    (r/"x"/"result.json").write_text(json.dumps(row))
    import scripts.scientific_audit as audit
    monkeypatch.setattr(audit,"ROOT",tmp_path)
    assert any("non_real_marked_solved" in x for x in result_checks(load_protocol()))
