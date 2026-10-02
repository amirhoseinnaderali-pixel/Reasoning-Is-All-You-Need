from __future__ import annotations
import json,subprocess,tempfile,time
from pathlib import Path
from typing import Any

class DockerSandbox:
    def __init__(self,image:str,timeout_s:int=5,memory_mb:int=512,cpus:float=1.0,pids_limit:int=64):
        if "@" not in image:raise RuntimeError("Execution image must be digest pinned")
        self.image=image;self.timeout_s=timeout_s;self.memory_mb=memory_mb;self.cpus=cpus;self.pids_limit=pids_limit
    def _build_cmd(self,root:Path)->list[str]:
        wrapper="set -u; g++ -std=c++17 -O2 -Wall -Wextra /workspace/solution.cpp -o /workspace/solution 2>/workspace/compile.err || { cat /workspace/compile.err; exit 41; }; bash /workspace/runner.sh"
        return ["docker","run","--rm","--network","none","--read-only","--cap-drop","ALL","--security-opt","no-new-privileges:true","--cpus",str(self.cpus),"--memory",f"{self.memory_mb}m","--pids-limit",str(self.pids_limit),"--tmpfs","/tmp:rw,nosuid,nodev,size=64m","-v",f"{root}:/workspace:rw",self.image,"bash","-lc",wrapper]
    def evaluate(self,code:str,tests:list[dict[str,Any]])->dict[str,Any]:
        if not tests:return {"compiled":False,"tests_passed":0,"tests_failed":0,"total_tests":0,"all_passed":False,"failure_type":"no_tests","tests":[]}
        with tempfile.TemporaryDirectory(prefix="cttr-vps-") as td:
            root=Path(td);(root/"solution.cpp").write_text(code)
            lines=[]
            for i,t in enumerate(tests,1):
                inf=f"input_{i}.txt";exf=f"expected_{i}.txt"
                (root/inf).write_text(str(t.get("input","")));(root/exf).write_text(str(t.get("expected_output",t.get("output",""))))
                lines.append(f"{i}\t{inf}\t{exf}")
            (root/"manifest.tsv").write_text("\n".join(lines)+"\n")
            runner="""#!/usr/bin/env bash
set -u
: > /workspace/results.tsv
while IFS=$'\t' read -r idx input_file expected_file; do
  [ -z "$idx" ] && continue
  status="runtime_error"; rc=0
  if timeout 5s /workspace/solution < "/workspace/$input_file" > "/workspace/stdout_$idx.txt" 2> "/workspace/stderr_$idx.txt"; then
    rc=0
    status="completed"
  else
    rc=$?
    if [ "$rc" -eq 124 ]; then status="timeout"; else status="runtime_error"; fi
  fi
  printf '%s\t%s\t%s\n' "$idx" "$status" "$rc" >> /workspace/results.tsv
done < /workspace/manifest.tsv
cat /workspace/results.tsv
"""
            (root/"runner.sh").write_text(runner);(root/"runner.sh").chmod(0o755)
            start=time.perf_counter()
            try:p=subprocess.run(self._build_cmd(root),capture_output=True,text=True,timeout=self.timeout_s+20)
            except subprocess.TimeoutExpired:return {"compiled":True,"tests_passed":0,"tests_failed":len(tests),"total_tests":len(tests),"all_passed":False,"failure_type":"timeout","tests":[]}
            if p.returncode==41:return {"compiled":False,"tests_passed":0,"tests_failed":len(tests),"total_tests":len(tests),"all_passed":False,"failure_type":"compile_error","compile_error":p.stdout.strip(),"tests":[],"wall_ms":(time.perf_counter()-start)*1000}
            if p.returncode in (137,-9):return {"compiled":True,"tests_passed":0,"tests_failed":len(tests),"total_tests":len(tests),"all_passed":False,"failure_type":"memory_limit","tests":[],"wall_ms":(time.perf_counter()-start)*1000}
            if p.returncode!=0:return {"compiled":True,"tests_passed":0,"tests_failed":len(tests),"total_tests":len(tests),"all_passed":False,"failure_type":"sandbox_error","tests":[],"wall_ms":(time.perf_counter()-start)*1000}
            rows=[];status_by_index={}
            for line in p.stdout.splitlines():
                parts=line.split("\t")
                if len(parts)==3:status_by_index[int(parts[0])]=(parts[1],int(parts[2]))
            for i,t in enumerate(tests,1):
                status,rc=status_by_index.get(i,("sandbox_error",-1))
                actual=(root/f"stdout_{i}.txt").read_text(errors="replace").strip() if (root/f"stdout_{i}.txt").exists() else ""
                expected=str(t.get("expected_output",t.get("output",""))).strip()
                stderr=(root/f"stderr_{i}.txt").read_text(errors="replace") if (root/f"stderr_{i}.txt").exists() else ""
                passed=status=="completed" and rc==0 and actual==expected
                rows.append({"test_index":i,"passed":passed,"expected":expected,"actual":actual,"error":stderr,"returncode":rc,"timeout":status=="timeout"})
            passed=sum(1 for x in rows if x["passed"])
            failure="none" if passed==len(rows) else ("timeout" if any(x["timeout"] for x in rows) else ("runtime_error" if any(x["error"] and not x["passed"] for x in rows) and any(x["returncode"]!=0 for x in rows) else "wrong_output"))
            return {"compiled":True,"tests_passed":passed,"tests_failed":len(rows)-passed,"total_tests":len(rows),"all_passed":bool(rows) and passed==len(rows),"failure_type":failure,"tests":rows,"wall_ms":(time.perf_counter()-start)*1000}
