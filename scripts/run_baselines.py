from __future__ import annotations

import argparse
import subprocess
import sys

CONFIGS = [
    "configs/baseline_single_pass.yaml",
    "configs/baseline_multi_sample.yaml",
    "configs/baseline_self_refine.yaml",
    "configs/baseline_verified.yaml",
    "configs/cttr_vps.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all configured baseline methods sequentially.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    failures = []
    for config in CONFIGS:
        command = [sys.executable, "scripts/run_experiment.py", "--config", config]
        if args.output_dir:
            command.extend(["--output-dir", args.output_dir])
        print("+", " ".join(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(config)
            if not args.continue_on_error:
                break
    if failures:
        raise SystemExit(f"Failed configurations: {', '.join(failures)}")


if __name__ == "__main__":
    main()
