from __future__ import annotations
import json,os,time
from collections.abc import Callable
from .hardening import canonical,config_hash,candidate_set_hash,environment_metadata,git_sha,load_manifest,load_protocol,run_id,sha256_bytes
from .docker_sandbox import DockerSandbox
from .model_adapter import BudgetExceeded,FrozenGemini,ModelCallError,extract_cpp

class Budget:
    def __init__(self,key,max_calls,max_tokens,max_debug,max_wall):
        self.api_key=key;self.max_calls=max_calls;self.max_tokens=max_tokens;self.max_debug=max_debug;self.max_wall=max_wall
        self.model_calls=0;self.debug_steps=0;self.token_usage=[];self.started=time.perf_counter()
    def tokens(self): return sum((x.get("total_tokens") or 0) for x in self.token_usage)
    def elapsed(self): return time.perf_counter()-self.started
    def check(self):
        if self.model_calls>=self.max_calls: raise BudgetExceeded("model-call budget exhausted")
        if self.tokens()>=self.max_tokens: raise BudgetExceeded("token budget exhausted")
        if self.elapsed()>=self.max_wall: raise BudgetExceeded("wall-clock budget exhausted")
    def check_debug(self):
        if self.debug_steps>=self.max_debug: raise BudgetExceeded("debug-step budget exhausted")
        self.check()

def ptxt(problem,role,prior="",feedback=""):
    return "Solve this C++17 task and return only source code.\nTASK:\n"+problem+"\nROLE:\n"+role+"\nPRIOR:\n"+prior+"\nVISIBLE EXECUTION FEEDBACK:\n"+feedback+"\nDo not use hidden tests."

def sandbox(protocol):
    e=protocol["execution"];return DockerSandbox(e["image"],e["timeout_s"],e["memory_mb"],e["cpus"],e["pids_limit"])

def failure_eval(failure_type,detail=""):
    return {"compiled":False,"tests_passed":0,"tests_failed":0,"total_tests":0,"all_passed":False,"failure_type":failure_type,"detail":detail,"tests":[]}

def run_condition(task,method,seed):
    protocol=load_protocol();cond=protocol["conditions"][method]
    key=os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEYS","").split(",")[0].strip()
    budget=Budget(key,cond["max_model_calls"],protocol["budgets"]["max_total_output_tokens"],cond.get("max_debug_steps",0),protocol["budgets"]["max_wall_time_seconds"])
    api=FrozenGemini(protocol["model"]["model_id"],protocol["model"]["generation"],budget);box=sandbox(protocol)
    generated=[];selection=[];events=[];failure_type=None;failure_detail="";start=time.perf_counter();final=""
    def call(role,prior="",feedback=""):
        budget.check();text,u=api.call(ptxt(task["problem"],role,prior,feedback),role);c=extract_cpp(text);generated.append(c);events.append(u|{"role":role});return c
    try:
        if method=="single_pass":
            final=call("single-pass generation");selection=[final]
        elif method=="multi_sample":
            for i in range(cond["candidate_count"]): call("independent sample "+str(i+1))
            evs=[box.evaluate(c,task["visible_tests"]) for c in generated];idx=max(range(len(evs)),key=lambda i:(evs[i].get("all_passed",False),evs[i].get("tests_passed",0),-i));final=generated[idx];selection=list(generated)
        elif method=="self_refinement":
            final=call("initial generation")
            for i in range(cond["refinement_rounds"]):
                budget.check_debug();budget.debug_steps+=1;text,u=api.call(ptxt(task["problem"],"self-refinement "+str(i+1),final),"self_refinement");final=extract_cpp(text);generated.append(final);events.append(u|{"round":i+1})
            selection=[final]
        elif method=="execution_refinement":
            final=call("initial generation")
            for i in range(cond["refinement_rounds"]):
                budget.check();ev=box.evaluate(final,task["visible_tests"]);events.append({"visible_evaluation":ev,"round":i+1})
                if ev.get("all_passed"):break
                budget.check_debug();budget.debug_steps+=1;text,u=api.call(ptxt(task["problem"],"execution-refinement "+str(i+1),final,json.dumps(ev)),"execution_refinement");final=extract_cpp(text);generated.append(final);events.append(u|{"round":i+1})
            selection=[final]
        elif method=="cttr_vps":
            plans=[]
            for i in range(cond["planning_calls"]):
                text,u=api.call(ptxt(task["problem"],"planning round "+str(i+1)),"planning");plans.append(text);events.append(u|{"round":i+1})
            plan="\n".join(plans)
            for i in range(cond["generation_calls"]):
                text,u=api.call(ptxt(task["problem"],"candidate generation",plan),"generation");generated.append(extract_cpp(text));events.append(u|{"round":i+1})
            dedup=[];seen=set()
            for c in generated:
                if c not in seen:seen.add(c);dedup.append(c)
            selection=dedup;evs=[box.evaluate(c,task["visible_tests"]) for c in selection];idx=max(range(len(evs)),key=lambda i:(evs[i].get("all_passed",False),evs[i].get("tests_passed",0),-i));final=selection[idx]
            for i in range(cond["refinement_rounds"]):
                budget.check();ev=box.evaluate(final,task["visible_tests"]);events.append({"visible_evaluation":ev,"round":i+1})
                if ev.get("all_passed"):break
                budget.check_debug();budget.debug_steps+=1;text,u=api.call(ptxt(task["problem"],"CTTR execution-debug "+str(i+1),final,json.dumps(ev)),"cttr_debug");final=extract_cpp(text);generated.append(final);events.append(u|{"round":i+1})
        else: raise ValueError(method)
        budget.check()
    except BudgetExceeded as exc:
        failure_type="budget_violation";failure_detail=str(exc)
    except ModelCallError as exc:
        failure_type="model_api_error";failure_detail=str(exc)
    except Exception as exc:
        failure_type="runner_error";failure_detail=str(exc)
    if failure_type:
        return final,selection,failure_eval(failure_type,failure_detail),budget,events,time.perf_counter()-start,failure_type,len(generated)
    visible=box.evaluate(final,task["visible_tests"])
    return final,selection,visible,budget,events,time.perf_counter()-start,None,len(generated)

def run_real_or_smoke(task,method,seed,mode,out_root,hidden_loader:Callable[[],list[dict]]|None=None):
    final,selection,visible,budget,events,wall,run_failure,generated_count=run_condition(task,method,seed)
    hidden=None;solved=False;hidden_hash=None
    visible_hash=sha256_bytes(canonical(task.get("visible_tests",[])))
    if not run_failure and mode in {"smoke","real"}:
        if hidden_loader is None:raise RuntimeError("Hidden-test loader is required for smoke/real")
        hidden_tests=hidden_loader()
        hidden_hash=sha256_bytes(canonical(hidden_tests))
        hidden=sandbox(load_protocol()).evaluate(final,hidden_tests)
    if mode=="real" and hidden is not None:solved=bool(hidden.get("all_passed"))
    protocol=load_protocol()
    status="VALIDATION_ONLY" if mode!="real" else "COMPLETED"
    if run_failure:status="BUDGET_VIOLATION" if run_failure=="budget_violation" else "FAILED"
    record={"schema_version":"2.0","experiment_id":"EXP-001","run_id":run_id(),"task_id":str(task["task_id"]),"seed":seed,"method":method,"candidate_set_hash":candidate_set_hash(selection),"config_hash":config_hash(protocol),"benchmark_hash":load_manifest().get("materialization_sha256"),"model_config_hash":config_hash(protocol["model"]),"model_revision":protocol["model"]["model_revision"],"candidate_count":len(selection),"generated_candidate_count":generated_count,"visible_test_hash":visible_hash,"hidden_test_hash":hidden_hash,"model_calls":budget.model_calls,"token_usage":budget.token_usage,"refinement_debug_steps":budget.debug_steps,"visible_evaluation":visible,"hidden_evaluation":hidden,"solved":solved,"failure_type":run_failure or (None if solved else (hidden or visible).get("failure_type")),"wall_clock_seconds":wall,"budget_usage":{"max_calls":budget.max_calls,"calls":budget.model_calls,"tokens":budget.tokens(),"max_tokens":budget.max_tokens,"max_debug_steps":budget.max_debug,"debug_steps":budget.debug_steps,"max_wall_time_seconds":budget.max_wall,"elapsed_seconds":budget.elapsed()},"environment":environment_metadata(),"git_sha":git_sha(),"execution_mode":mode,"status":status}
    return record,final
