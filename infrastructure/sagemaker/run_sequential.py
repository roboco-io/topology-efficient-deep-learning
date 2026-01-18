#!/usr/bin/env python
"""Submit SageMaker experiments one at a time, waiting for completion."""

import subprocess
import sys
import time
from datetime import datetime

import boto3

# Configuration
ROLE = "arn:aws:iam::931016744724:role/service-role/AmazonSagemaker-ExecutionRole-20250202P152458"
PROFILE = "personal"
REGION = "ap-northeast-2"

DATASETS = ["ECG200", "FordA", "ElectricDevices", "Wafer", "UWaveGestureLibraryAll"]
MODELS = ["ph_mlp", "inceptiontime"]
SEEDS = [42, 123, 456]


def get_sagemaker_client():
    """Get SageMaker client."""
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    return session.client("sagemaker")


def get_completed_experiments(client) -> set:
    """Get list of completed experiments."""
    completed = set()
    paginator = client.get_paginator("list_training_jobs")
    for page in paginator.paginate(NameContains="track-a", StatusEquals="Completed"):
        for job in page["TrainingJobSummaries"]:
            name = job["TrainingJobName"]
            # Parse: track-a-{dataset}-{model}-seed{seed}-{timestamp}
            parts = name.split("-")
            if len(parts) >= 5 and "seed" in parts[-2]:
                dataset_raw = "-".join(parts[2:-2])
                model = parts[-2].replace("seed", "").replace(parts[-3], "")
                # This is complex, let's use a different approach
                pass
            completed.add(name)
    return completed


def get_running_jobs(client) -> list:
    """Get list of running jobs."""
    running = []
    paginator = client.get_paginator("list_training_jobs")
    for page in paginator.paginate(NameContains="track-a", StatusEquals="InProgress"):
        for job in page["TrainingJobSummaries"]:
            running.append(job["TrainingJobName"])
    return running


def wait_for_completion(client, job_name: str, max_wait: int = 3600) -> dict:
    """Wait for a job to complete and return status."""
    start = time.time()
    while time.time() - start < max_wait:
        response = client.describe_training_job(TrainingJobName=job_name)
        status = response["TrainingJobStatus"]
        if status in ["Completed", "Failed", "Stopped"]:
            return response
        time.sleep(30)  # Check every 30 seconds
    return {"TrainingJobStatus": "Timeout"}


def submit_job(dataset: str, model: str, seed: int) -> str:
    """Submit a single training job."""
    cmd = [
        "python", "infrastructure/sagemaker/run_benchmark.py",
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
        "--role", ROLE,
        "--profile", PROFILE,
        "--region", REGION,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Extract job name from output
    for line in result.stdout.split("\n"):
        if "Job:" in line:
            return line.split("Job:")[-1].strip()

    return None


def main():
    client = get_sagemaker_client()

    # Build experiment list
    experiments = [
        (d, m, s)
        for d in DATASETS
        for m in MODELS
        for s in SEEDS
    ]

    print(f"Total experiments: {len(experiments)}")

    # Check for running jobs
    running = get_running_jobs(client)
    if running:
        print(f"Found {len(running)} running job(s), waiting for completion...")
        for job_name in running:
            print(f"  Waiting for: {job_name}")
            response = wait_for_completion(client, job_name)
            status = response.get("TrainingJobStatus", "Unknown")
            f1 = None
            if status == "Completed":
                for metric in response.get("FinalMetricDataList", []):
                    if metric["MetricName"] == "final:f1":
                        f1 = metric["Value"]
            print(f"  Completed: {status}, F1={f1}")

    # Get completed job patterns (dataset-model-seed)
    completed_patterns = set()
    paginator = client.get_paginator("list_training_jobs")
    for page in paginator.paginate(NameContains="track-a", StatusEquals="Completed"):
        for job in page["TrainingJobSummaries"]:
            # Extract pattern from job name
            name = job["TrainingJobName"]
            # track-a-ecg200-ph-mlp-seed42-1768749386
            parts = name.rsplit("-", 1)[0]  # Remove timestamp
            completed_patterns.add(parts)

    # Run remaining experiments
    for i, (dataset, model, seed) in enumerate(experiments, 1):
        model_name = model.replace("_", "-")
        pattern = f"track-a-{dataset.lower()}-{model_name}-seed{seed}"

        if pattern in completed_patterns:
            print(f"[{i}/{len(experiments)}] {dataset}/{model}/seed{seed} - Already completed, skipping")
            continue

        print(f"\n[{i}/{len(experiments)}] {dataset}/{model}/seed{seed}")
        print(f"  Submitting job...")

        job_name = submit_job(dataset, model, seed)
        if not job_name:
            print(f"  Failed to submit!")
            continue

        print(f"  Job: {job_name}")
        print(f"  Waiting for completion...")

        response = wait_for_completion(client, job_name)
        status = response.get("TrainingJobStatus", "Unknown")
        f1 = None
        if status == "Completed":
            for metric in response.get("FinalMetricDataList", []):
                if metric["MetricName"] == "final:f1":
                    f1 = metric["Value"]

        print(f"  Status: {status}, F1={f1}")

        # Brief pause between jobs
        time.sleep(5)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
