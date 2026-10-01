from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_result_files(root: str | Path) -> Iterable[Path]:
    base = Path(root)
    yield from sorted(base.rglob("result.json"))


def load_records(root: str | Path) -> list[dict[str, Any]]:
    records = []
    for path in iter_result_files(root):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimator for n sampled candidates and c correct ones."""
    if n <= 0 or c < 0 or k <= 0 or k > n or c > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("method", "unknown"))].append(record)

    methods = []
    for method, rows in sorted(groups.items()):
        solved = sum(bool(row.get("solved")) for row in rows)
        avg_calls = sum(int(row.get("model_calls", 0)) for row in rows) / len(rows)
        latencies = [row.get("latency_seconds") for row in rows if row.get("latency_seconds") is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        methods.append(
            {
                "method": method,
                "problems": len(rows),
                "solved": solved,
                "success_rate": solved / len(rows),
                "avg_calls": avg_calls,
                "avg_latency_seconds": avg_latency,
                "pass_at_1": pass_at_k(len(rows), solved, 1),
            }
        )

    return {"num_records": len(records), "methods": methods}


def markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Method | Problems | Solved | Success Rate | Avg Calls | Avg Latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("methods", []):
        latency = "N/A" if row["avg_latency_seconds"] is None else f'{row["avg_latency_seconds"]:.3f}'
        lines.append(
            f'| {row["method"]} | {row["problems"]} | {row["solved"]} | '
            f'{row["success_rate"]:.3f} | {row["avg_calls"]:.1f} | {latency} |'
        )
    return "\n".join(lines)
