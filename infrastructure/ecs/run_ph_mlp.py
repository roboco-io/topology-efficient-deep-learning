#!/usr/bin/env python
"""ECS Fargate Spot으로 PH-MLP 벤치마크 실행.

Usage:
    # ECR에 이미지 푸시 (최초 1회)
    ./build_and_push.sh

    # 단일 실험
    python run_ph_mlp.py --dataset ECG200 --seed 42

    # 전체 벤치마크 (병렬)
    python run_ph_mlp.py --all --parallel 4

    # Dry run
    python run_ph_mlp.py --all --dry-run
"""

import argparse
import json
import os
import time
from datetime import datetime

import boto3

# Configuration
AWS_PROFILE = os.environ.get("AWS_PROFILE", "personal")
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
CLUSTER = "topology-dl"
TASK_DEFINITION = "topology-dl-ph-mlp"
SUBNETS = []  # Will be auto-detected
SECURITY_GROUPS = []  # Will be auto-detected
S3_BUCKET = "sagemaker-ap-northeast-2-931016744724"

DATASETS = ["ECG200", "FordA", "ElectricDevices", "Wafer", "UWaveGestureLibraryAll"]
SEEDS = [42, 123, 456]


def get_default_vpc_config(ec2_client):
    """Get default VPC subnets and security groups."""
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise ValueError("No default VPC found")
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    subnets = ec2_client.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    subnet_ids = [s["SubnetId"] for s in subnets["Subnets"][:2]]  # Use first 2 subnets

    sgs = ec2_client.describe_security_groups(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": ["default"]}]
    )
    sg_ids = [sg["GroupId"] for sg in sgs["SecurityGroups"]]

    return subnet_ids, sg_ids


def run_task(ecs_client, dataset: str, seed: int, subnets: list, sgs: list, dry_run: bool = False):
    """Run a single ECS Fargate task."""
    task_name = f"ph-mlp-{dataset.lower()}-seed{seed}"

    container_overrides = {
        "containerOverrides": [{
            "name": "ph-mlp-trainer",
            "command": [
                "--dataset", dataset,
                "--seed", str(seed),
                "--data-dir", "/data/ucr",
                "--output-s3", f"s3://{S3_BUCKET}",
            ]
        }]
    }

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {task_name}")

    if dry_run:
        print(f"  [DRY RUN] Would run task: {task_name}")
        return {"taskArn": "dry-run", "status": "dry_run"}

    response = ecs_client.run_task(
        cluster=CLUSTER,
        taskDefinition=TASK_DEFINITION,
        count=1,
        launchType="FARGATE",
        platformVersion="LATEST",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": subnets,
                "securityGroups": sgs,
                "assignPublicIp": "ENABLED"
            }
        },
        overrides=container_overrides,
        capacityProviderStrategy=[
            {"capacityProvider": "FARGATE_SPOT", "weight": 1}
        ],
        tags=[
            {"key": "Project", "value": "topology-dl"},
            {"key": "Dataset", "value": dataset},
            {"key": "Seed", "value": str(seed)},
        ]
    )

    if response["tasks"]:
        task_arn = response["tasks"][0]["taskArn"]
        print(f"  Task ARN: {task_arn.split('/')[-1]}")
        return {"taskArn": task_arn, "status": "running"}
    else:
        failure = response.get("failures", [{}])[0]
        print(f"  [FAILED] {failure.get('reason', 'Unknown')}")
        return {"taskArn": None, "status": "failed", "reason": failure.get("reason")}


def wait_for_tasks(ecs_client, task_arns: list, timeout: int = 3600):
    """Wait for multiple tasks to complete."""
    print(f"\nWaiting for {len(task_arns)} tasks to complete...")
    start = time.time()

    while time.time() - start < timeout:
        response = ecs_client.describe_tasks(cluster=CLUSTER, tasks=task_arns)
        statuses = [t["lastStatus"] for t in response["tasks"]]

        running = sum(1 for s in statuses if s in ["PENDING", "RUNNING"])
        stopped = sum(1 for s in statuses if s == "STOPPED")

        print(f"  Running: {running}, Stopped: {stopped}/{len(task_arns)}")

        if running == 0:
            return response["tasks"]

        time.sleep(30)

    return None


def main():
    parser = argparse.ArgumentParser(description="ECS Fargate PH-MLP 벤치마크")
    parser.add_argument("--dataset", type=str, help="단일 데이터셋")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="전체 벤치마크")
    parser.add_argument("--parallel", type=int, default=1, help="동시 실행 수")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile", type=str, default=AWS_PROFILE)
    parser.add_argument("--region", type=str, default=AWS_REGION)
    args = parser.parse_args()

    # AWS clients
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    ecs = session.client("ecs")
    ec2 = session.client("ec2")

    # Get VPC config
    subnets, sgs = get_default_vpc_config(ec2)
    print(f"Subnets: {subnets}")
    print(f"Security Groups: {sgs}")

    # Build experiment list
    if args.all:
        experiments = [(d, s) for d in DATASETS for s in SEEDS]
    elif args.dataset:
        experiments = [(args.dataset, args.seed)]
    else:
        print("ERROR: --all 또는 --dataset 지정 필요")
        return

    print(f"\n총 {len(experiments)}개 실험")

    # Fargate Spot pricing estimate
    # 4 vCPU, 8GB: ~$0.07/hr (Spot)
    hours_per_exp = 0.25  # 15 min estimate
    cost = len(experiments) * hours_per_exp * 0.07
    print(f"예상 비용: ~${cost:.2f}")

    if args.dry_run:
        print("\n[DRY RUN] 실제 실행 없이 종료")
        return

    # Run experiments
    results = []
    pending_tasks = []

    for i, (dataset, seed) in enumerate(experiments):
        result = run_task(ecs, dataset, seed, subnets, sgs, args.dry_run)
        results.append({"dataset": dataset, "seed": seed, **result})

        if result["taskArn"] and result["status"] == "running":
            pending_tasks.append(result["taskArn"])

        # Respect parallel limit
        if len(pending_tasks) >= args.parallel:
            wait_for_tasks(ecs, pending_tasks)
            pending_tasks = []

        time.sleep(2)

    # Wait for remaining tasks
    if pending_tasks:
        wait_for_tasks(ecs, pending_tasks)

    print("\n" + "=" * 60)
    print("벤치마크 완료")
    print("=" * 60)

    # Save results
    with open("ecs_tasks.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Task 목록: ecs_tasks.json")


if __name__ == "__main__":
    main()
