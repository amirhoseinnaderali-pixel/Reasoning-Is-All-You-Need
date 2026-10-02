from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any
SCHEMA_VERSION="2.0"
REQUIRED={"schema_version","experiment_id","run_id","task_id","seed","method","candidate_set_hash","config_hash","benchmark_hash","model_config_hash","model_revision","candidate_count","model_calls","token_usage","refinement_debug_steps","visible_evaluation","hidden_evaluation","solved","failure_type","wall_clock_seconds","budget_usage","environment","git_sha","execution_mode","status"}
def validate_record(r:dict[str,Any])->list[str]:
    e=sorted(REQUIRED-set(r))
    if r.get("schema_version")!=SCHEMA_VERSION:e.append("schema_version")
    if r.get("execution_mode") not in {"validation","smoke","real"}:e.append("execution_mode")
    if r.get("status") not in {"VALIDATION_ONLY","COMPLETED","BLOCKED","FAILED","BUDGET_VIOLATION"}:e.append("status")
    if not isinstance(r.get("candidate_set_hash"),str) or len(r.get("candidate_set_hash",""))!=64:e.append("candidate_set_hash")
    return e
def write_immutable(path:str|Path,r:dict[str,Any])->None:
    e=validate_record(r)
    if e:raise ValueError("Invalid result schema: "+", ".join(e))
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"w",encoding="utf-8") as f:
        json.dump(r,f,ensure_ascii=False,indent=2,sort_keys=True);f.write("\n")