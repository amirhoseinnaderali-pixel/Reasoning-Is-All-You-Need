from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cttr_vps.config import load_yaml
from cttr_vps.methods import METHODS
from cttr_vps.result_schema import make_record
from cttr_vps.results import write_json


def load_problem(config: dict):
    dataset = Path(config.get("dataset", {}).get("path", "ioi_multi_view.json"))
    index = int(config.get("dataset", {}).get("problem_index", 0))
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not 0 <= index < len(rows):
        raise ValueError(f"Invalid problem_index={index}")
    return rows[index]


async def run(config_path: str, output_override: str | None = None):
    config = load_yaml(config_path)
    method = str(config["method"])
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    problem = load_problem(config)
    tests = problem.get("implementation_view", problem).get("samples", [])
    experiment_id = str(config.get("experiment_id") or f"{method}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}")
    root = Path(output_override or config.get("output_dir", "results")) / experiment_id
    root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    payload = await METHODS[method](problem, tests, config, str(root))
    wall = time.perf_counter() - started
    evaluation = payload["evaluations"][-1]

    record = make_record(
        experiment_id=experiment_id,
        problem_id=str(problem.get("_meta", {}).get("uuid", config.get("dataset", {}).get("problem_index"))),
        method=method,
        candidate_count=len(payload.get("candidates", [])),
        model_calls=int(payload.get("model_calls", 0)),
        wall_time_seconds=wall,
        evaluation=evaluation,
        config=config,
        events=payload.get("events", []),
        models=payload.get("models", []),
    )
    write_json(root / "result.json", record)
    if payload.get("code"):
        (root / "final_code.cpp").write_text(payload["code"], encoding="utf-8")
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = asyncio.run(run(args.config, args.output_dir))
    print(root)


if __name__ == "__main__":
    main()
