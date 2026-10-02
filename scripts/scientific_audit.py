from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cttr_vps.hardening import preflight,load_protocol,load_manifest,config_hash
from cttr_vps.result_schema_v2 import validate_record

def result_checks(protocol):
    errors=[]
    root=ROOT/"results"
    if not root.exists(): return errors
    for p in root.rglob("result.json"):
        try:r=json.loads(p.read_text())
        except Exception:errors.append(f"invalid_json:{p}");continue
        errors.extend(f"{p}:{x}" for x in validate_record(r))
        cond=protocol["conditions"].get(r.get("method"),{})
        if r.get("execution_mode")!="real" and r.get("solved"):errors.append(f"non_real_marked_solved:{p}")
        if r.get("model_calls",0)>cond.get("max_model_calls",10**9):errors.append(f"call_budget:{p}")
        if r.get("refinement_debug_steps",0)>cond.get("max_debug_steps",10**9):errors.append(f"debug_budget:{p}")
        if r.get("status")=="COMPLETED" and r.get("execution_mode")!="real":errors.append(f"status_mode_mismatch:{p}")
    return errors

def main():
    p=preflight("real");m=load_manifest();c=load_protocol();errs=result_checks(c)
    checks={"benchmark_frozen":m.get("status")=="FROZEN","benchmark_materialized":bool(m.get("materialization_sha256")),"model_frozen":bool(c.get("model",{}).get("model_revision")),"runtime_frozen":"@" in c.get("execution",{}).get("image",""),"hidden_isolated":m.get("hidden_tests",{}).get("available_during_generation") is False,"immutable_results":c.get("reproducibility",{}).get("immutable_results") is True,"real_preflight":p["status"]=="PASS","result_artifacts_valid":not errs}
    out={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"preflight":p,"config_hash":config_hash(c),"result_errors":errs}
    print(json.dumps(out,indent=2));(ROOT/"audit").mkdir(exist_ok=True);(ROOT/"audit/scientific_audit.json").write_text(json.dumps(out,indent=2,sort_keys=True))
    raise SystemExit(0 if out["status"]=="PASS" else 1)
if __name__=="__main__":main()
