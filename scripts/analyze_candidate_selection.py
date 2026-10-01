import argparse
import json
from pathlib import Path


def summarize(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    evaluations = data.get("evaluations", [])
    if not evaluations:
        raise SystemExit("No candidate evaluations found.")

    print(f"selected_candidate_index={data.get('selected_candidate_index')}")
    print(f"selection_rule={data.get('selection_rule')}")
    print(f"candidate_count={len(evaluations)}")
    for row in evaluations:
        print(
            f"candidate={row['candidate_index']} "
            f"compile={row['compile_ok']} "
            f"passed={row['passed_tests']}/{row['total_tests']} "
            f"all_passed={row['all_passed']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    summarize(parser.parse_args().input)
