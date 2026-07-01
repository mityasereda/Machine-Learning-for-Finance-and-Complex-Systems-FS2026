"""
Coverage quantile sweep.

For each quantile in COVERAGE_LEVELS this script:
  1. Patches coverage_q in config.yaml
  2. Trains all strategies + runs backtests  (train.py)
  3. Runs calibration diagnostics            (calibration_diagnostics.py)
  4. Generates comparison plots + stats CSV  (comparison_presentation.py)
  5. Archives all outputs under sweep_results/q{q}/

Run from the momentum/ directory:
    python3 run_coverage_sweep.py
"""

import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime


def ts():
    return datetime.now().strftime("%H:%M:%S")

COVERAGE_LEVELS = [0.25, 0.50, 0.75, 0.90, 0.95]
CONFIG_PATH     = "config.yaml"
SWEEP_DIR       = "sweep_results_no_context_buffer"

OUTPUT_DIRS = [
    "backtest_rl_results",
    "calibration_results",
    "results",
    "models",
    "robust_models",
    "ball_models",
    "dynamic_radius_models",
]


def patch_config(q: float) -> None:
    with open(CONFIG_PATH) as f:
        text = f.read()
    text = re.sub(r"(coverage_q\s*:\s*)[\d.]+", rf"\g<1>{q}", text)
    with open(CONFIG_PATH, "w") as f:
        f.write(text)


def run(script: str) -> None:
    subprocess.run([sys.executable, script], check=True)


def archive(q: float) -> None:
    dest = os.path.join(SWEEP_DIR, f"q{int(q*100):02d}")
    os.makedirs(dest, exist_ok=True)
    for d in OUTPUT_DIRS:
        if os.path.isdir(d):
            target = os.path.join(dest, d)
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(d, target)
    shutil.copy(CONFIG_PATH, os.path.join(dest, "config.yaml"))
    print(f"  Archived -> {dest}")


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)

    sweep_start = time.time()

    for i, q in enumerate(COVERAGE_LEVELS, 1):
        label = f"q={q:.2f}"
        level_start = time.time()
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(COVERAGE_LEVELS)}] Coverage quantile: {label}  —  {ts()}")
        print(f"{'='*60}")

        print(f"  [1/4] Patching config ...  {ts()}")
        patch_config(q)

        print(f"  [2/4] Training + backtesting ...  {ts()}")
        run("train.py")

        print(f"  [3/4] Calibration diagnostics ...  {ts()}")
        run("calibration_diagnostics.py")

        print(f"  [4/4] Comparison plots ...  {ts()}")
        run("comparison_presentation.py")

        archive(q)
        elapsed = time.time() - level_start
        print(f"  Done in {elapsed/60:.1f} min  —  {ts()}")

    total = time.time() - sweep_start
    print(f"\nSweep complete in {total/60:.1f} min. Results in ./{SWEEP_DIR}/")


if __name__ == "__main__":
    main()
