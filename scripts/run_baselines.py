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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    for config in CONFIGS:
        result = subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", config], check=False)
        if result.returncode and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
