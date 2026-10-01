from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cttr_vps.config import load_yaml_config
from cttr_vps.methods import METHODS
from cttr_vps.results import write_json


def load_problem(config: dict) -> tuple[str, dict, list[dict]]:
    dataset_path = Path(config.get("dataset", {}).get("path", "ioi_multi_view.json"))
    index = int(config.get("dataset", {}).get("problem_index", 0))
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not 0 <= index < len(data):
        raise ValueError(f"problem_index {index} is outside dataset of length {len(data) if isinstance(data, list) else 'N/A'}")
    row = data[index]
    problem = row.get("implementation_view", row)
    plan = row.get("algorithm_view", row)
    tests = problem.get("samples", [])
    return json.dumps(plan, ensure_ascii=False, indent=2), row, tests


async def main_async(config_path: str, output_override: str | None) -> Path:
    config = load_yaml_config(config_path)
    method = str(config.get("method", "cttr_vps"))
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choices: {', '.join(METHODS)}")

    output_root = Path(output_override or config.get("output_dir", "results"))
    experiment_id = str(config.get("experiment_id") or f"{method}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}")
    output_dir = output_root / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    plan, problem, tests = load_problem(config)
    started = time.perf_counter()
    status = "completed"
    error = None
    payload = {}
    try:
        payload = await METHODS[method](plan, problem, tests, config, str(output_dir))
    except Exception as exc:
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
    elapsed = time.perf_counter() - started

    evaluations = payload.get("evaluations", [])
    final_eval = evaluations[-1] if evaluations else {}
    record = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "problem_id": str(problem.get("_meta", {}).get("uuid") or config.get("dataset", {}).get("problem_index", 0)),
        "method": method,
        "models": sorted({str(event.get("model")) for event in payload.get("events", []) if isinstance(event, dict) and event.get("model")}),
        "num_generation_attempts": int(config.get("samples", config.get("candidates_per_round", 1))),
        "num_refinement_rounds": int(config.get("refinement_rounds", config.get("generation_rounds", 0))),
        "candidate_count": len(payload.get("candidates", [])),
        "compiled": bool(final_eval.get("compiled", False)),
        "tests_passed": int(final_eval.get("tests_passed", 0)),
        "tests_failed": int(final_eval.get("tests_failed", len(tests))),
        "solved": bool(final_eval.get("solved", False)),
        "latency_seconds": final_eval.get("latency_seconds"),
        "model_calls": int(payload.get("model_calls", 0)),
        "verification_attempts": int(final_eval.get("verification_attempts", 0)),
        "tokens": None,
        "error_type": final_eval.get("error_type") or (error or {}).get("type"),
        "status": status,
        "error": error,
        "wall_time_seconds": elapsed,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "config": config,
        "events": payload.get("events", {}),
    }
    write_json(output_dir / "result.json", record)
    if payload.get("code"):
        (output_dir / "final_code.cpp").write_text(payload["code"], encoding="utf-8")
    if error:
        print(json.dumps({"status": status, "error": error, "result": str(output_dir / "result.json")}, indent=2))
        raise RuntimeError(error["message"])
    print(json.dumps({"status": status, "result": str(output_dir / "result.json"), "solved": record["solved"]}, indent=2))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CTTR-VPS experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    asyncio.run(main_async(args.config, args.output_dir))


if __name__ == "__main__":
    main()
