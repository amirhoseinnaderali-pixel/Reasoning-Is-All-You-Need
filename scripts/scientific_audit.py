from pathlib import Path;import json;import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cttr_vps.hardening import preflight,load_protocol,load_manifest,config_hash
def main():
 p=preflight('real'); m=load_manifest(); c=load_protocol()
 checks={'benchmark_frozen':m.get('status')=='FROZEN','benchmark_materialized':bool(m.get('materialization_sha256')),'model_frozen':bool(c.get('model',{}).get('model_revision')),'runtime_frozen':'@' in c.get('execution',{}).get('image',''),'hidden_isolated':m.get('hidden_tests',{}).get('available_during_generation') is False,'immutable_results':c.get('reproducibility',{}).get('immutable_results') is True,'real_preflight':p['status']=='PASS'}
 out={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'preflight':p,'config_hash':config_hash(c)}
 print(json.dumps(out,indent=2));Path(ROOT/'audit').mkdir(exist_ok=True);Path(ROOT/'audit/scientific_audit.json').write_text(json.dumps(out,indent=2,sort_keys=True))
 raise SystemExit(0 if out['status']=='PASS' else 1)
if __name__=='__main__':main()