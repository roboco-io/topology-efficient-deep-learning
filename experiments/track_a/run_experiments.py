#!/usr/bin/env python
"""Track A 전체 실험 실행기.

Usage:
    # 모든 실험 실행
    python experiments/track_a/run_experiments.py --all

    # 특정 데이터셋만
    python experiments/track_a/run_experiments.py --datasets ECG200 FordA

    # 특정 모델만
    python experiments/track_a/run_experiments.py --models ph_mlp inceptiontime

    # Ablation 실험만
    python experiments/track_a/run_experiments.py --ablation
"""

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Track A 설정
DATASETS = [
    "ECG200",
    "FordA",
    "ElectricDevices",
    "Wafer",
    "UWaveGestureLibraryAll",
]

BASELINES = ["inceptiontime", "resnet", "fcn"]
PROPOSED = ["ph_mlp"]

SEEDS = [42, 123, 456]

# Ablation 설정
VECTORIZATIONS = ["persistence_landscape", "persistence_image", "statistics"]
HOMOLOGY_CONFIGS = [
    [0],        # H0만
    [0, 1],     # H0 + H1
]
DELAYS = [2, 3, 5]
DIMENSIONS = [2, 3, 4]


def run_single_experiment(
    dataset: str,
    model: str,
    seed: int,
    vectorization: str = "persistence_landscape",
    homology_dims: List[int] = [0, 1],
    delay: int = 5,
    dimension: int = 3,
    use_wandb: bool = False,
    dry_run: bool = False,
) -> Dict:
    """단일 실험 실행."""
    cmd = [
        sys.executable,
        "experiments/track_a/train.py",
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
        "--vectorization", vectorization,
        "--delay", str(delay),
        "--dimension", str(dimension),
        "--homology_dims", *[str(d) for d in homology_dims],
    ]

    if use_wandb:
        cmd.append("--use_wandb")

    print(f"\n{'='*60}")
    print(f"Running: {dataset} / {model} / seed={seed}")
    if model == "ph_mlp":
        print(f"  vectorization={vectorization}, homology={homology_dims}")
        print(f"  delay={delay}, dimension={dimension}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return {"status": "dry_run", "command": cmd}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        if result.returncode == 0:
            print("SUCCESS")
            return {"status": "success", "stdout": result.stdout[-500:]}
        else:
            print(f"FAILED: {result.stderr[-500:]}")
            return {"status": "failed", "stderr": result.stderr[-500:]}

    except Exception as e:
        print(f"ERROR: {e}")
        return {"status": "error", "error": str(e)}


def run_main_experiments(
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    use_wandb: bool = False,
    dry_run: bool = False,
) -> List[Dict]:
    """메인 실험 실행 (베이스라인 vs 제안 모델)."""
    results = []

    for dataset, model, seed in itertools.product(datasets, models, seeds):
        result = run_single_experiment(
            dataset=dataset,
            model=model,
            seed=seed,
            use_wandb=use_wandb,
            dry_run=dry_run,
        )
        result["config"] = {
            "dataset": dataset,
            "model": model,
            "seed": seed,
        }
        results.append(result)

    return results


def run_ablation_vectorization(
    datasets: List[str],
    seeds: List[int],
    use_wandb: bool = False,
    dry_run: bool = False,
) -> List[Dict]:
    """Ablation: Vectorization 방법 비교."""
    results = []

    for dataset, vectorization, seed in itertools.product(
        datasets, VECTORIZATIONS, seeds
    ):
        result = run_single_experiment(
            dataset=dataset,
            model="ph_mlp",
            seed=seed,
            vectorization=vectorization,
            use_wandb=use_wandb,
            dry_run=dry_run,
        )
        result["config"] = {
            "dataset": dataset,
            "model": "ph_mlp",
            "seed": seed,
            "ablation": "vectorization",
            "vectorization": vectorization,
        }
        results.append(result)

    return results


def run_ablation_homology(
    datasets: List[str],
    seeds: List[int],
    use_wandb: bool = False,
    dry_run: bool = False,
) -> List[Dict]:
    """Ablation: Homology 차수 비교."""
    results = []

    for dataset, homology_dims, seed in itertools.product(
        datasets, HOMOLOGY_CONFIGS, seeds
    ):
        result = run_single_experiment(
            dataset=dataset,
            model="ph_mlp",
            seed=seed,
            homology_dims=homology_dims,
            use_wandb=use_wandb,
            dry_run=dry_run,
        )
        result["config"] = {
            "dataset": dataset,
            "model": "ph_mlp",
            "seed": seed,
            "ablation": "homology",
            "homology_dims": homology_dims,
        }
        results.append(result)

    return results


def run_ablation_embedding(
    datasets: List[str],
    seeds: List[int],
    use_wandb: bool = False,
    dry_run: bool = False,
) -> List[Dict]:
    """Ablation: Embedding 파라미터 비교."""
    results = []

    for dataset, delay, dimension, seed in itertools.product(
        datasets, DELAYS, DIMENSIONS, seeds
    ):
        result = run_single_experiment(
            dataset=dataset,
            model="ph_mlp",
            seed=seed,
            delay=delay,
            dimension=dimension,
            use_wandb=use_wandb,
            dry_run=dry_run,
        )
        result["config"] = {
            "dataset": dataset,
            "model": "ph_mlp",
            "seed": seed,
            "ablation": "embedding",
            "delay": delay,
            "dimension": dimension,
        }
        results.append(result)

    return results


def count_experiments(
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    run_ablation: bool = False,
) -> Dict:
    """실험 수 계산."""
    main_count = len(datasets) * len(models) * len(seeds)

    if run_ablation:
        abl_vec = len(datasets) * len(VECTORIZATIONS) * len(seeds)
        abl_hom = len(datasets) * len(HOMOLOGY_CONFIGS) * len(seeds)
        abl_emb = len(datasets) * len(DELAYS) * len(DIMENSIONS) * len(seeds)
        total = main_count + abl_vec + abl_hom + abl_emb
        return {
            "main": main_count,
            "ablation_vectorization": abl_vec,
            "ablation_homology": abl_hom,
            "ablation_embedding": abl_emb,
            "total": total,
        }
    else:
        return {"main": main_count, "total": main_count}


def main():
    parser = argparse.ArgumentParser(description="Track A 전체 실험 실행")

    # 실험 범위
    parser.add_argument("--all", action="store_true", help="모든 실험 실행")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="데이터셋 목록",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=BASELINES + PROPOSED,
        help="모델 목록",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="시드 목록",
    )

    # Ablation
    parser.add_argument("--ablation", action="store_true", help="Ablation 실험 포함")
    parser.add_argument(
        "--ablation_only", action="store_true", help="Ablation 실험만 실행"
    )

    # 옵션
    parser.add_argument("--use_wandb", action="store_true", help="wandb 로깅 사용")
    parser.add_argument("--dry_run", action="store_true", help="실제 실행 없이 명령어만 출력")
    parser.add_argument(
        "--output",
        type=str,
        default="./results/track_a/experiment_log.json",
        help="실험 결과 로그 파일",
    )

    args = parser.parse_args()

    # 실험 수 계산
    counts = count_experiments(
        args.datasets,
        args.models,
        args.seeds,
        args.ablation or args.ablation_only,
    )

    print("=" * 60)
    print("Track A 실험 계획")
    print("=" * 60)
    print(f"데이터셋: {args.datasets}")
    print(f"모델: {args.models}")
    print(f"시드: {args.seeds}")
    print(f"\n실험 수:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - 실제 실행 없음]\n")

    all_results = []

    # 메인 실험
    if not args.ablation_only:
        print("\n>>> 메인 실험 실행")
        results = run_main_experiments(
            args.datasets,
            args.models,
            args.seeds,
            args.use_wandb,
            args.dry_run,
        )
        all_results.extend(results)

    # Ablation 실험
    if args.ablation or args.ablation_only:
        print("\n>>> Ablation: Vectorization")
        results = run_ablation_vectorization(
            args.datasets,
            args.seeds,
            args.use_wandb,
            args.dry_run,
        )
        all_results.extend(results)

        print("\n>>> Ablation: Homology")
        results = run_ablation_homology(
            args.datasets,
            args.seeds,
            args.use_wandb,
            args.dry_run,
        )
        all_results.extend(results)

        print("\n>>> Ablation: Embedding")
        results = run_ablation_embedding(
            args.datasets,
            args.seeds,
            args.use_wandb,
            args.dry_run,
        )
        all_results.extend(results)

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "counts": counts,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(log, f, indent=2)

    # 요약
    success = sum(1 for r in all_results if r.get("status") == "success")
    failed = sum(1 for r in all_results if r.get("status") == "failed")
    error = sum(1 for r in all_results if r.get("status") == "error")

    print("\n" + "=" * 60)
    print("실험 완료 요약")
    print("=" * 60)
    print(f"성공: {success}")
    print(f"실패: {failed}")
    print(f"에러: {error}")
    print(f"\n로그 저장: {output_path}")


if __name__ == "__main__":
    main()
