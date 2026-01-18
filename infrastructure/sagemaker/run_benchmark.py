#!/usr/bin/env python
"""SageMaker Managed Spot Training으로 Track A 벤치마크 실행.

Usage:
    # 단일 실험
    python run_benchmark.py --dataset ECG200 --model ph_mlp

    # 전체 벤치마크
    python run_benchmark.py --all

    # Dry run (비용 확인)
    python run_benchmark.py --all --dry-run
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# 설정
DATASETS = ["ECG200", "FordA", "ElectricDevices", "Wafer", "UWaveGestureLibraryAll"]
MODELS = ["ph_mlp", "inceptiontime"]
SEEDS = [42, 123, 456]

# AWS 설정
AWS_PROFILE = os.environ.get("AWS_PROFILE", "personal")
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")


def get_sagemaker_session(profile: str, region: str):
    """SageMaker 세션 생성."""
    boto_session = boto3.Session(profile_name=profile, region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def upload_data(session, local_path: str, bucket: str, prefix: str):
    """데이터를 S3에 업로드."""
    s3_path = session.upload_data(
        path=local_path,
        bucket=bucket,
        key_prefix=prefix,
    )
    return s3_path


def create_estimator(
    session,
    role: str,
    instance_type: str = "ml.g4dn.xlarge",
    use_spot: bool = True,
    max_wait: int = 14400,  # 4 hours for spot
    max_run: int = 7200,    # 2 hours max training time
):
    """SageMaker PyTorch Estimator 생성."""
    estimator = PyTorch(
        entry_point="train_sagemaker.py",
        source_dir=str(Path(__file__).parent),
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version="2.0.0",
        py_version="py310",
        sagemaker_session=session,
        use_spot_instances=use_spot,
        max_wait=max_wait if use_spot else None,
        max_run=max_run,
        checkpoint_s3_uri=None,
        output_path=f"s3://{session.default_bucket()}/topology-dl/output",
        base_job_name="topology-dl-track-a",
        hyperparameters={},
        metric_definitions=[
            {"Name": "train:loss", "Regex": "loss=([0-9\\.]+)"},
            {"Name": "test:f1", "Regex": "f1=([0-9\\.]+)"},
            {"Name": "test:accuracy", "Regex": "Accuracy: ([0-9\\.]+)"},
            {"Name": "final:f1", "Regex": "F1 \\(macro\\): ([0-9\\.]+)"},
        ],
    )
    return estimator


def run_experiment(
    session,
    role: str,
    s3_data: str,
    dataset: str,
    model: str,
    seed: int,
    instance_type: str = "ml.g4dn.xlarge",
    use_spot: bool = True,
    wait: bool = False,
    dry_run: bool = False,
):
    """단일 실험 실행."""
    model_name = model.replace("_", "-")
    job_name = f"track-a-{dataset.lower()}-{model_name}-seed{seed}-{int(time.time())}"

    print(f"\n{'='*60}")
    print(f"Job: {job_name}")
    print(f"Dataset: {dataset}, Model: {model}, Seed: {seed}")
    print(f"Instance: {instance_type}, Spot: {use_spot}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Skipping actual job submission")
        return {"job_name": job_name, "status": "dry_run"}

    estimator = create_estimator(
        session=session,
        role=role,
        instance_type=instance_type,
        use_spot=use_spot,
    )

    estimator.set_hyperparameters(
        dataset=dataset,
        model=model,
        seed=seed,
        epochs=100,
        batch_size=32,
        lr=0.001,
        patience=10,
    )

    estimator.fit(
        inputs={"training": s3_data},
        job_name=job_name,
        wait=wait,
        logs=wait,
    )

    return {
        "job_name": job_name,
        "status": "submitted",
        "estimator": estimator,
    }


def main():
    parser = argparse.ArgumentParser(description="SageMaker Spot Training 벤치마크")
    parser.add_argument("--dataset", type=str, help="단일 데이터셋")
    parser.add_argument("--model", type=str, choices=["ph_mlp", "inceptiontime"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="전체 벤치마크 실행")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge")
    parser.add_argument("--no-spot", action="store_true", help="On-demand 인스턴스 사용")
    parser.add_argument("--wait", action="store_true", help="완료까지 대기")
    parser.add_argument("--sequential", action="store_true", help="순차 실행 (각 job 완료 대기)")
    parser.add_argument("--dry-run", action="store_true", help="실제 실행 없이 확인")
    parser.add_argument("--data-path", type=str, default="./data/ucr", help="로컬 데이터 경로")
    parser.add_argument("--role", type=str, help="SageMaker IAM Role ARN")
    parser.add_argument("--profile", type=str, default=AWS_PROFILE)
    parser.add_argument("--region", type=str, default=AWS_REGION)

    args = parser.parse_args()

    # SageMaker 세션
    print("Initializing SageMaker session...")
    session = get_sagemaker_session(args.profile, args.region)
    bucket = session.default_bucket()

    # IAM Role
    if args.role:
        role = args.role
    else:
        role = sagemaker.get_execution_role() if os.environ.get("SM_CURRENT_HOST") else None
        if not role:
            print("ERROR: --role 옵션으로 SageMaker Role ARN을 지정하세요.")
            print("  예: --role arn:aws:iam::123456789:role/SageMakerRole")
            return

    # 데이터 업로드
    print(f"\nUploading data to S3...")
    if not args.dry_run:
        s3_data = upload_data(
            session,
            local_path=args.data_path,
            bucket=bucket,
            prefix="topology-dl/data/ucr",
        )
        print(f"Data uploaded to: {s3_data}")
    else:
        s3_data = f"s3://{bucket}/topology-dl/data/ucr"
        print(f"[DRY RUN] Would upload to: {s3_data}")

    # 실험 목록 생성
    if args.all:
        experiments = [
            (d, m, s)
            for d in DATASETS
            for m in MODELS
            for s in SEEDS
        ]
    elif args.dataset and args.model:
        experiments = [(args.dataset, args.model, args.seed)]
    else:
        print("ERROR: --all 또는 --dataset/--model 옵션을 지정하세요.")
        return

    print(f"\n총 {len(experiments)}개 실험 예정")

    # 비용 추정
    # ml.g4dn.xlarge: ~$0.526/hr (On-demand), ~$0.16/hr (Spot)
    hours_per_exp = 0.5  # 예상
    on_demand_cost = len(experiments) * hours_per_exp * 0.526
    spot_cost = len(experiments) * hours_per_exp * 0.16

    print(f"\n예상 비용:")
    print(f"  On-demand: ${on_demand_cost:.2f}")
    print(f"  Spot (최대 70% 절감): ${spot_cost:.2f}")

    if args.dry_run:
        print("\n[DRY RUN] 실제 실행 없이 종료")
        return

    # 실험 실행
    results = []
    total = len(experiments)
    for idx, (dataset, model, seed) in enumerate(experiments, 1):
        print(f"\n[{idx}/{total}] Submitting experiment...")

        # 순차 실행 모드: 각 job 완료까지 대기
        should_wait = args.wait or args.sequential

        result = run_experiment(
            session=session,
            role=role,
            s3_data=s3_data,
            dataset=dataset,
            model=model,
            seed=seed,
            instance_type=args.instance_type,
            use_spot=not args.no_spot,
            wait=should_wait,
            dry_run=args.dry_run,
        )
        results.append(result)

        # Rate limiting
        time.sleep(2)

    # 요약
    print("\n" + "=" * 60)
    print("실험 제출 완료")
    print("=" * 60)
    print(f"총 {len(results)}개 job 제출")

    # Job 목록 저장
    jobs_file = Path("sagemaker_jobs.json")
    with open(jobs_file, "w") as f:
        json.dump([r for r in results if "estimator" not in r], f, indent=2)
    print(f"Job 목록 저장: {jobs_file}")

    print("\n모니터링:")
    print(f"  AWS Console: https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/jobs")
    print(f"  CLI: aws sagemaker list-training-jobs --profile {args.profile} --name-contains track-a")


if __name__ == "__main__":
    main()
