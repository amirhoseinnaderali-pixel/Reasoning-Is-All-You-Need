from __future__ import annotations
import hashlib, json, os, platform, shutil, subprocess, uuid
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "exp001.yaml"
MANIFEST = ROOT / "benchmark" / "manifest.json"

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical(config))

def load_protocol() -> dict[str, Any]:
    data = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("experiment_id") != "EXP-001":
        raise RuntimeError("EXP-001 protocol is missing or invalid")
    return data

def load_manifest() -> dict[str, Any]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("benchmark/manifest.json is invalid")
    return data

def require_frozen_benchmark(manifest: dict[str, Any]) -> str:
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("Benchmark is not frozen")
    material = manifest.get("materialization_sha256")
    tasks = manifest.get("tasks")
    if not material or not isinstance(tasks, list) or not tasks:
        raise RuntimeError("Benchmark materialization hash/task population is incomplete")
    for task in tasks:
        for key in ("task_id", "visible_hash", "hidden_hash"):
            if not task.get(key):
                raise RuntimeError(f"Benchmark task missing {key}")
    return str(material)

def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNAVAILABLE"

def environment_metadata() -> dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "docker": shutil.which("docker") or None}

def hidden_path() -> Path:
    raw = os.getenv("CTTR_HIDDEN_TESTS_PATH", "")
    if not raw:
        raise RuntimeError("CTTR_HIDDEN_TESTS_PATH is required for real execution")
    p = Path(raw).resolve()
    try:
        p.relative_to(ROOT.resolve())
    except ValueError:
        return p
    raise RuntimeError("Hidden tests must not live inside the repository")

def preflight(mode: str) -> dict[str, Any]:
    protocol = load_protocol()
    manifest = load_manifest()
    failures: list[str] = []
    if mode == "real":
        try:
            require_frozen_benchmark(manifest)
        except Exception as exc:
            failures.append(str(exc))
        if not protocol.get("model", {}).get("model_revision"):
            failures.append("Model revision is not frozen")
        if "@" not in protocol.get("execution", {}).get("image", ""):
            failures.append("Execution image is not digest pinned")
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEYS")):
            failures.append("Google credentials are missing")
        if shutil.which("docker") is None:
            failures.append("Docker is unavailable")
        try:
            hidden_path()
        except Exception as exc:
            failures.append(str(exc))
        if not protocol.get("reproducibility", {}).get("immutable_results"):
            failures.append("Immutable results are not enabled")
    current_git=git_sha()
    if mode=="real" and current_git=="UNAVAILABLE": failures.append("Git SHA is unavailable")
    return {"status": "PASS" if not failures else "FAIL", "mode": mode, "failures": failures,
            "git_sha": current_git, "config_hash": config_hash(protocol),
            "environment": environment_metadata()}

def candidate_set_hash(candidates: list[str]) -> str:
    return sha256_bytes(canonical(candidates))

def run_id() -> str:
    return str(uuid.uuid4())
