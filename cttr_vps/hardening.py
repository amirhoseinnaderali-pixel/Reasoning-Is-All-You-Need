from __future__ import annotations
import hashlib,json,os,platform,shutil,subprocess,uuid
from importlib.metadata import PackageNotFoundError,version
from pathlib import Path
from typing import Any
import yaml
ROOT=Path(__file__).resolve().parents[1];PROTOCOL=ROOT/"configs/exp001.yaml";MANIFEST=ROOT/"benchmark/manifest.json"
def canonical(value:Any)->bytes:return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sha256_bytes(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def sha256_file(path:Path)->str:return sha256_bytes(path.read_bytes())
def config_hash(config:dict[str,Any])->str:return sha256_bytes(canonical(config))
def load_protocol()->dict[str,Any]:
    data=yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(data,dict) or data.get("experiment_id")!="EXP-001":raise RuntimeError("EXP-001 protocol is missing or invalid")
    return data
def load_manifest()->dict[str,Any]:
    data=json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data,dict):raise RuntimeError("benchmark/manifest.json is invalid")
    return data
def require_frozen_benchmark(manifest:dict[str,Any])->str:
    if manifest.get("status")!="FROZEN":raise RuntimeError("Benchmark is not frozen")
    material=str(manifest.get("materialization_sha256") or "");tasks=manifest.get("tasks")
    if not material or not isinstance(tasks,list) or not tasks:raise RuntimeError("Benchmark materialization hash/task population is incomplete")
    bundle=ROOT/"benchmark/materialized.json"
    if not bundle.exists():raise RuntimeError("Frozen benchmark bundle is missing")
    if sha256_file(bundle)!=material:raise RuntimeError("Benchmark materialization hash mismatch")
    bundle_data=json.loads(bundle.read_text(encoding="utf-8"));bundle_tasks=bundle_data.get("tasks")
    if not isinstance(bundle_tasks,list) or len(bundle_tasks)!=len(tasks):raise RuntimeError("Benchmark task population mismatch")
    meta={str(x["task_id"]):x for x in tasks}
    for task in bundle_tasks:
        task_id=str(task.get("task_id"))
        if task_id not in meta:raise RuntimeError(f"Benchmark task missing from manifest: {task_id}")
        if "hidden_tests" in task:raise RuntimeError("Frozen benchmark bundle contains hidden tests")
        actual=sha256_bytes(canonical(task.get("visible_tests",[])))
        if actual!=meta[task_id].get("visible_hash"):raise RuntimeError(f"Visible-test hash mismatch for {task_id}")
    if manifest.get("task_count")!=len(tasks):raise RuntimeError("Benchmark task_count mismatch")
    for task in tasks:
        for key in ("task_id","visible_hash","hidden_hash"):
            if not task.get(key):raise RuntimeError(f"Benchmark task missing {key}")
    return material
def git_sha()->str:
    try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:return "UNAVAILABLE"
def dependency_lock_hash(protocol:dict[str,Any])->str:
    path=ROOT/protocol["dependencies"]["lock_file"]
    if sha256_file(path)!=protocol["dependencies"]["lock_sha256"]:raise RuntimeError("Dependency lock hash mismatch")
    return protocol["dependencies"]["lock_sha256"]
def dependency_versions()->dict[str,str|None]:
    out={}
    for pkg in ("google-genai","PyYAML","pytest"):
        try:out[pkg]=version(pkg)
        except PackageNotFoundError:out[pkg]=None
    return out
def environment_metadata()->dict[str,Any]:
    meta={"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine(),"docker":shutil.which("docker") or None,"dependencies":dependency_versions()}
    if meta["docker"]:
        try:meta["docker_version"]=subprocess.check_output(["docker","version","--format","{{.Server.Version}}"],text=True,stderr=subprocess.DEVNULL).strip()
        except Exception:meta["docker_version"]="UNAVAILABLE"
    return meta
def hidden_path()->Path:
    raw=os.getenv("CTTR_HIDDEN_TESTS_PATH","")
    if not raw:raise RuntimeError("CTTR_HIDDEN_TESTS_PATH is required for real execution")
    p=Path(raw).resolve()
    try:p.relative_to(ROOT.resolve())
    except ValueError:return p
    raise RuntimeError("Hidden tests must not live inside the repository")
def preflight(mode:str)->dict[str,Any]:
    protocol=load_protocol();manifest=load_manifest();failures=[]
    try:dependency_lock_hash(protocol)
    except Exception as exc:failures.append(str(exc))
    if mode=="real":
        try:require_frozen_benchmark(manifest)
        except Exception as exc:failures.append(str(exc))
    if mode in {"smoke","real"}:
        if not protocol.get("model",{}).get("model_revision"):failures.append("Model revision is not frozen")
        if "@" not in protocol.get("execution",{}).get("image",""):failures.append("Execution image is not digest pinned")
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEYS")):failures.append("Google credentials are missing")
        if shutil.which("docker") is None:failures.append("Docker is unavailable")
        elif environment_metadata().get("docker_version") in {None,"UNAVAILABLE"}:failures.append("Docker daemon is unavailable")
        if mode=="real":
            try:hidden_path()
            except Exception as exc:failures.append(str(exc))
            if not protocol.get("reproducibility",{}).get("immutable_results"):failures.append("Immutable results are not enabled")
    current_git=git_sha()
    if mode=="real" and current_git=="UNAVAILABLE":failures.append("Git SHA is unavailable")
    return {"status":"PASS" if not failures else "FAIL","mode":mode,"failures":failures,"git_sha":current_git,"config_hash":config_hash(protocol),"dependency_lock_hash":dependency_lock_hash(protocol) if not failures or "Dependency lock hash mismatch" not in failures else None,"environment":environment_metadata()}
def candidate_set_hash(candidates:list[str])->str:return sha256_bytes(canonical(candidates))
def run_id()->str:return str(uuid.uuid4())
