from __future__ import annotations
import argparse,asyncio,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cttr_vps.hardening import preflight,hidden_path,load_protocol,load_manifest,canonical

def load_tasks(mode):
    path=ROOT/"benchmark/smoke_task.json" if mode=="smoke" else ROOT/"benchmark/materialized.json"
    if not path.exists():raise RuntimeError(f"Missing benchmark materialization: {path}")
    tasks=json.loads(path.read_text())["tasks"]
    if mode=="real":
        manifest=load_manifest();meta={str(x["task_id"]):x for x in manifest["tasks"]}
        for task in tasks:
            if str(task["task_id"]) not in meta:raise RuntimeError(f"Task not in frozen manifest: {task['task_id']}")
            if "hidden_tests" in task:raise RuntimeError("Frozen benchmark bundle must not contain hidden tests")
            actual=hashlib.sha256(canonical(task.get("visible_tests",[]))).hexdigest()
            if actual!=meta[str(task["task_id"])]["visible_hash"]:raise RuntimeError(f"Visible-test hash mismatch for {task['task_id']}")
    return tasks

def hidden_loader_for(task,mode):
    task_id=str(task["task_id"])
    if mode=="smoke":
        def load():
            data=json.loads((ROOT/"benchmark/smoke_task.json").read_text())
            rows={str(x["task_id"]):x["hidden_tests"] for x in data["tasks"]}
            return rows[task_id]
        return load
    def load():
        data=json.loads(hidden_path().read_text())
        rows={str(x["task_id"]):x["hidden_tests"] for x in data["tasks"]}
        if task_id not in rows:raise RuntimeError("Hidden-test artifact has no matching task")
        hidden=rows[task_id];manifest=load_manifest();meta=next(x for x in manifest["tasks"] if str(x["task_id"])==task_id)
        actual=hashlib.sha256(canonical(hidden)).hexdigest()
        if actual!=meta["hidden_hash"]:raise RuntimeError(f"Hidden-test hash mismatch for {task_id}")
        return hidden
    return load

async def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["validation","smoke","real"],required=True);p.add_argument("--method",choices=["single_pass","multi_sample","self_refinement","execution_refinement","cttr_vps"]);p.add_argument("--output-dir",default="results");a=p.parse_args()
    if a.mode=="validation":raise SystemExit(subprocess.run([sys.executable,"-m","pytest","-q","tests"],check=False).returncode)
    check=preflight(a.mode)
    if a.mode in {"smoke","real"} and check["status"]!="PASS":print(json.dumps(check,indent=2));raise SystemExit(2)
    tasks=load_tasks(a.mode);protocol=load_protocol();seeds=[int(x) for x in protocol["seeds"]];methods=[a.method] if a.method else list(protocol["conditions"].keys())
    from cttr_vps.frozen_runner import run_real_or_smoke
    from cttr_vps.result_schema_v2 import write_immutable
    for task in tasks:
        for seed in seeds:
            for method in methods:
                record,code=run_real_or_smoke(task,method,seed,a.mode,a.output_dir,hidden_loader=hidden_loader_for(task,a.mode))
                out=Path(a.output_dir)/record["run_id"]/method;write_immutable(out/"result.json",record);(out/"final_code.cpp").write_text(code,encoding="utf-8")
    print("validation_only="+str(a.mode!="real"))
if __name__=="__main__":asyncio.run(main())
