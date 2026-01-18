#!/usr/bin/env python
"""Submit remaining SageMaker experiments sequentially."""

import subprocess
import sys
import time
from datetime import datetime

# Configuration
ROLE = "arn:aws:iam::931016744724:role/service-role/AmazonSagemaker-ExecutionRole-20250202P152458"
PROFILE = "personal"
REGION = "ap-northeast-2"

DATASETS = ["ECG200", "FordA", "ElectricDevices", "Wafer", "UWaveGestureLibraryAll"]
MODELS = ["ph_mlp", "inceptiontime"]
SEEDS = [42, 123, 456]

# Already completed
COMPLETED = [
    ("ECG200", "ph_mlp", 42),
    ("ECG200", "ph_mlp", 123),
]


def run_experiment(dataset: str, model: str, seed: int) -> bool:
    """Run a single experiment and wait for completion."""
    cmd = [
        "python", "infrastructure/sagemaker/run_benchmark.py",
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
        "--wait",
        "--role", ROLE,
        "--profile", PROFILE,
        "--region", REGION,
    ]

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {dataset}/{model}/seed{seed}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"[SUCCESS] {dataset}/{model}/seed{seed}")
            return True
        else:
            print(f"[FAILED] {dataset}/{model}/seed{seed}")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {dataset}/{model}/seed{seed}")
        return False


def main():
    experiments = [
        (d, m, s)
        for d in DATASETS
        for m in MODELS
        for s in SEEDS
        if (d, m, s) not in COMPLETED
    ]

    print(f"Total remaining experiments: {len(experiments)}")
    print(f"Estimated time: ~{len(experiments) * 3} minutes")

    successes, failures = 0, 0

    for i, (dataset, model, seed) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Submitting...")

        if run_experiment(dataset, model, seed):
            successes += 1
        else:
            failures += 1
            # Continue anyway

        time.sleep(5)  # Brief pause between jobs

    print(f"\n{'='*60}")
    print(f"Completed: {successes}/{len(experiments)} successes, {failures} failures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
