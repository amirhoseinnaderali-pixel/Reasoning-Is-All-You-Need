from __future__ import annotations
from collections import defaultdict
from math import sqrt
def wilson(s,n,z=1.96):
    if n==0:return None
    p=s/n;d=1+z*z/n
    q=z*sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((p+z*z/(2*n)-q)/d,(p+z*z/(2*n)+q)/d)
def analyze(rows):
    groups=defaultdict(list)
    for r in rows: groups[r["method"]].append(r)
    out={"methods":{},"paired_vs_C0":[],"per_task_per_seed":[],"failure_modes":{}}
    for m,items in groups.items():
        solved=sum(bool(x.get("solved")) for x in items); compiled=sum(bool(x.get("visible_evaluation",{}).get("compiled")) for x in items)
        out["methods"][m]={"n":len(items),"solved":solved,"solved_rate":solved/len(items) if items else None,"uncertainty_95":wilson(solved,len(items)),"compilation_rate":compiled/len(items) if items else None,"avg_calls":sum(x.get("model_calls",0) for x in items)/len(items) if items else None}
        out["failure_modes"][m]={}
        for x in items: out["failure_modes"][m][x.get("failure_type")]=out["failure_modes"][m].get(x.get("failure_type"),0)+1
        for x in items: out["per_task_per_seed"].append({"task_id":x["task_id"],"seed":x["seed"],"method":m,"solved":x["solved"],"model_calls":x["model_calls"],"debug_steps":x["refinement_debug_steps"]})
    c0={(x["task_id"],x["seed"]):x for x in groups.get("single_pass",[])}
    for m,items in groups.items():
        if m=="single_pass":continue
        pairs=[(c0[(x["task_id"],x["seed"])],x) for x in items if (x["task_id"],x["seed"]) in c0]
        out["paired_vs_C0"].append({"method":m,"pairs":len(pairs),"both_solved":sum(a["solved"] and b["solved"] for a,b in pairs),"baseline_only":sum(a["solved"] and not b["solved"] for a,b in pairs),"method_only":sum((not a["solved"]) and b["solved"] for a,b in pairs)})
    return out
