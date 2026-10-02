from __future__ import annotations
import argparse,asyncio,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cttr_vps.hardening import preflight,hidden_path,load_protocol,load_manifest

def load_tasks(mode):
    path=ROOT/"benchmark/smoke_task.json" if mode=="smoke" else ROOT/"benchmark/materialized.json"
    if not path.exists(): raise RuntimeError(f"Missing benchmark materialization: {path}")
    return json.loads(path.read_text())["tasks"]

def attach_hidden(task):
    data=json.loads(hidden_path().read_text())
    rows={str(x["task_id"]):x["hidden_tests"] for x in data["tasks"]}
    if str(task["task_id"]) not in rows: raise RuntimeError("Hidden-test artifact has no matching task")
    task=dict(task);task["hidden_tests"]=rows[str(task["task_id"])]
    return task

async def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["validation","smoke","real"],required=True)
    p.add_argument("--method",choices=["single_pass","multi_sample","self_refinement","execution_refinement","cttr_vps"])
    p.add_argument("--output-dir",default="results")
    a=p.parse_args()
    if a.mode=="validation":
        result=subprocess.run([sys.executable,"-m","pytest","-q","tests/test_hardening.py","tests/test_docker_policy.py","tests/test_result_immutability.py"],check=False)
        raise SystemExit(result.returncode)
    check=preflight("real" if a.mode=="real" else "validation")
    if a.mode=="real" and check["status"]!="PASS":
        print(json.dumps(check,indent=2));raise SystemExit(2)
    tasks=load_tasks(a.mode)
    if a.mode!="validation":
        tasks=[attach_hidden(t) for t in tasks]
    protocol=load_protocol();seeds=[int(x) for x in protocol["seeds"]]
    methods=[a.method] if a.method else list(protocol["conditions"].keys())
    from cttr_vps.frozen_runner import run_real_or_smoke
    from cttr_vps.result_schema_v2 import write_immutable
    for task in tasks:
        for seed in seeds:
            for method in methods:
                record,code=run_real_or_smoke(task,method,seed,a.mode,a.output_dir)
                out=Path(a.output_dir)/record["run_id"]/method
                write_immutable(out/"result.json",record)
                (out/"final_code.cpp").write_text(code,encoding="utf-8")
    print("validation_only="+str(a.mode!="real"))
if __name__=="__main__":asyncio.run(main())
