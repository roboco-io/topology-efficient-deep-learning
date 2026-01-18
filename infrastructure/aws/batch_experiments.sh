#!/bin/bash
# Track A 전체 실험을 AWS ECS에서 병렬 실행
#
# Usage:
#   ./batch_experiments.sh [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false

if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "[DRY RUN MODE]"
fi

# 데이터셋 목록
DATASETS=(
    "ECG200"
    "FordA"
    "ElectricDevices"
    "Wafer"
    "UWaveGestureLibraryAll"
)

# 모델 목록
MODELS=(
    "inceptiontime"
    "resnet"
    "fcn"
    "ph_mlp"
)

# 시드 목록
SEEDS=(42 123 456)

# 실험 카운터
TOTAL=0
LAUNCHED=0

echo "=========================================="
echo "Track A 배치 실험 실행"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            TOTAL=$((TOTAL + 1))

            echo "[$TOTAL] $dataset / $model / seed=$seed"

            if [ "$DRY_RUN" = true ]; then
                echo "  [SKIP - DRY RUN]"
            else
                "$SCRIPT_DIR/run_ecs_experiment.sh" \
                    --dataset "$dataset" \
                    --model "$model" \
                    --seed "$seed"

                LAUNCHED=$((LAUNCHED + 1))

                # 태스크 간 딜레이 (AWS API rate limiting 방지)
                sleep 2
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "배치 실험 완료"
echo "=========================================="
echo "총 실험 수: $TOTAL"
echo "실행된 태스크: $LAUNCHED"
echo ""
echo "모든 태스크 상태 확인:"
echo "  aws ecs list-tasks --profile personal --cluster topology-dl-cluster"
