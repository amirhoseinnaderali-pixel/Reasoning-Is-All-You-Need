from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_results(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def pass_at_k(n: int, c: int, k: int) -> float | None:
    if n <= 0 or c < 0 or k <= 0 or k > n or c > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method", "unknown"))].append(row)

    methods = []
    for method, items in sorted(grouped.items()):
        solved = sum(bool(item.get("solved", False)) for item in items)
        calls = [int(item.get("model_calls", 0)) for item in items]
        walls = [item.get("wall_time_seconds") for item in items if item.get("wall_time_seconds") is not None]
        methods.append({
            "method": method,
            "problems": len(items),
            "solved": solved,
            "success_rate": solved / len(items),
            "avg_model_calls": sum(calls) / len(calls),
            "avg_wall_time_seconds": sum(walls) / len(walls) if walls else None,
        })
    return {"records": len(rows), "methods": methods}
