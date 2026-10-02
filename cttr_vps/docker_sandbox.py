from __future__ import annotations
import json, subprocess, tempfile, time
from pathlib import Path
from typing import Any
class DockerSandbox:
    def __init__(self,image:str,timeout_s:int=5,memory_mb:int=512,cpus:float=1.0,pids_limit:int=64):
        if '@' not in image: raise RuntimeError('Execution image must be digest pinned')
        self.image=image; self.timeout_s=timeout_s; self.memory_mb=memory_mb; self.cpus=cpus; self.pids_limit=pids_limit
    def evaluate(self,code:str,tests:list[dict[str,Any]])->dict[str,Any]:
        if not tests: return {'compiled':False,'tests_passed':0,'tests_failed':0,'total_tests':0,'all_passed':False,'failure_type':'no_tests','tests':[]}
        with tempfile.TemporaryDirectory(prefix='cttr-vps-') as td:
            root=Path(td); (root/'solution.cpp').write_text(code); (root/'tests.json').write_text(json.dumps(tests))
            wrapper='set -eu; g++ -std=c++17 -O2 -Wall -Wextra /workspace/solution.cpp -o /workspace/solution 2>/workspace/compile.err || { cat /workspace/compile.err; exit 41; }; python3 /workspace/runner.py'
            runner='import json,subprocess\nt=json.load(open("/workspace/tests.json"))\nr=[]\nfor i,x in enumerate(t,1):\n p=subprocess.run(["/workspace/solution"],input=str(x.get("input","")).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5)\n a=p.stdout.decode(errors="replace").strip(); e=str(x.get("expected_output",x.get("output",""))).strip()\n r.append({"test_index":i,"passed":p.returncode==0 and a==e,"actual":a,"expected":e,"stderr":p.stderr.decode(errors="replace"),"returncode":p.returncode})\nprint(json.dumps(r))\n'
            (root/'runner.py').write_text(runner)
            cmd=['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges:true','--cpus',str(self.cpus),'--memory',f'{self.memory_mb}m','--pids-limit',str(self.pids_limit),'--tmpfs','/tmp:rw,nosuid,nodev,size=64m','-v',f'{root}:/workspace:rw',self.image,'bash','-lc',wrapper]
            start=time.perf_counter()
            try: p=subprocess.run(cmd,capture_output=True,text=True,timeout=self.timeout_s+20)
            except subprocess.TimeoutExpired: return {'compiled':True,'tests_passed':0,'tests_failed':len(tests),'total_tests':len(tests),'all_passed':False,'failure_type':'timeout','tests':[]}
            if p.returncode==41: return {'compiled':False,'tests_passed':0,'tests_failed':len(tests),'total_tests':len(tests),'all_passed':False,'failure_type':'compile_error','tests':[],'wall_ms':(time.perf_counter()-start)*1000}
            if p.returncode!=0: return {'compiled':True,'tests_passed':0,'tests_failed':len(tests),'total_tests':len(tests),'all_passed':False,'failure_type':'sandbox_error','tests':[],'wall_ms':(time.perf_counter()-start)*1000}
            rows=json.loads(p.stdout.strip().splitlines()[-1]); passed=sum(1 for x in rows if x['passed'])
            failure='none' if passed==len(rows) else ('runtime_error' if any(x['returncode']!=0 for x in rows) else 'wrong_output')
            return {'compiled':True,'tests_passed':passed,'tests_failed':len(rows)-passed,'total_tests':len(rows),'all_passed':bool(rows) and passed==len(rows),'failure_type':failure,'tests':rows,'wall_ms':(time.perf_counter()-start)*1000}