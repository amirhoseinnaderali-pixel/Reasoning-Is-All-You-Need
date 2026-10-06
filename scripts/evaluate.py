from __future__ import annotations

import argparse
import json
from pathlib import Path

from cttr_vps.results import aggregate, load_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    rows = load_results(args.results)
    summary = aggregate(rows)
    print(json.dumps(summary, indent=2))
    Path(args.results).mkdir(parents=True, exist_ok=True)
    Path(args.results, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
