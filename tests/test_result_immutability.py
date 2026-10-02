from cttr_vps.result_schema_v2 import write_immutable

def test_result_cannot_overwrite(tmp_path):
    r={"schema_version":"2.0","experiment_id":"EXP-001","run_id":"x","task_id":"t","seed":1,
       "method":"single_pass","candidate_set_hash":"0"*64,"config_hash":"x","benchmark_hash":"x",
       "dependency_lock_hash":"0"*64,"model_config_hash":"x","model_revision":"gemini-2.5-flash",
       "candidate_count":1,"generated_candidate_count":1,"visible_test_hash":"1"*64,"hidden_test_hash":None,
       "model_calls":1,"token_usage":[],"refinement_debug_steps":0,"visible_evaluation":{},
       "hidden_evaluation":None,"solved":False,"failure_type":"compile_error","wall_clock_seconds":0,
       "budget_usage":{},"environment":{},"git_sha":"x","execution_mode":"validation","status":"VALIDATION_ONLY"}
    p=tmp_path/"result.json";write_immutable(p,r)
    try:write_immutable(p,r)
    except FileExistsError:return
    assert False
