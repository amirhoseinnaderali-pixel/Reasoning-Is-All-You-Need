from __future__ import annotations
import argparse,asyncio,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cttr_vps.hardening import preflight,hidden_path,load_protocol,load_manifest,canonical

def load_tasks(mode):
    path=ROOT/"benchmark/smoke_task.json" if mode=="smoke" else ROOT/"benchmark/materialized.json"
    if not path.exists():raise RuntimeError(f"Missing benchmark materialization: {path}")
    return json.loads(path.read_text())["tasks"]

def hidden_loader_for(task,mode):
    if mode=="smoke":
        data=json.loads((ROOT/"benchmark/smoke_task.json").read_text());rows={str(x["task_id"]):x["hidden_tests"] for x in data["tasks"]}
    else:
        data=json.loads(hidden_path().read_text());rows={str(x["task_id"]):x["hidden_tests"] for x in data["tasks"]}
    if str(task["task_id"]) not in rows:raise RuntimeError("Hidden-test artifact has no matching task")
    hidden=rows[str(task["task_id"])]
    if mode=="real":
        manifest=load_manifest();meta=next(x for x in manifest["tasks"] if str(x["task_id"])==str(task["task_id"]))
        expected=meta["hidden_hash"];actual=hashlib.sha256(canonical(hidden)).hexdigest()
        if actual!=expected:raise RuntimeError(f"Hidden-test hash mismatch for {task['task_id']}")
    return lambda hidden=hidden:hidden

async def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["validation","smoke","real"],required=True);p.add_argument("--method",choices=["single_pass","multi_sample","self_refinement","execution_refinement","cttr_vps"]);p.add_argument("--output-dir",default="results");a=p.parse_args()
    if a.mode=="validation":
        raise SystemExit(subprocess.run([sys.executable,"-m","pytest","-q","tests"],check=False).returncode)
    check=preflight("real" if a.mode=="real" else "validation")
    if a.mode=="real" and check["status"]!="PASS":print(json.dumps(check,indent=2));raise SystemExit(2)
    tasks=load_tasks(a.mode);protocol=load_protocol();seeds=[int(x) for x in protocol["seeds"]];methods=[a.method] if a.method else list(protocol["conditions"].keys())
    from cttr_vps.frozen_runner import run_real_or_smoke
    from cttr_vps.result_schema_v2 import write_immutable
    for task in tasks:
        for seed in seeds:
            for method in methods:
                loader=hidden_loader_for(task,a.mode)
                record,code=run_real_or_smoke(task,method,seed,a.mode,a.output_dir,hidden_loader=loader)
                out=Path(a.output_dir)/record["run_id"]/method;write_immutable(out/"result.json",record);(out/"final_code.cpp").write_text(code,encoding="utf-8")
    print("validation_only="+str(a.mode!="real"))
if __name__=="__main__":asyncio.run(main())
