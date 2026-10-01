from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cttr_vps.results import aggregate, load_records, markdown_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CTTR-VPS result.json files.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="results/summary.json")
    parser.add_argument("--markdown", default="results/summary.md")
    args = parser.parse_args()
    records = load_records(args.results)
    summary = aggregate(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(markdown_table(summary) + "\n", encoding="utf-8")
    print(markdown_table(summary))
    print(f"Records aggregated: {len(records)}")


if __name__ == "__main__":
    main()
