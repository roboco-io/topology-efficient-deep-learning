#!/bin/bash
# Track A 실험을 AWS ECS/Fargate에서 실행
#
# 사전 요구사항:
# 1. AWS CLI 설정 (aws configure --profile personal)
# 2. ECR 리포지토리 생성
# 3. ECS 클러스터 생성
# 4. EFS 파일 시스템 (데이터/결과 저장용)
# 5. VPC, 서브넷, 보안 그룹 설정
#
# Usage:
#   ./run_ecs_experiment.sh --dataset ECG200 --model ph_mlp --seed 42

set -e

# 기본 설정
AWS_PROFILE="${AWS_PROFILE:-personal}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
CLUSTER_NAME="${CLUSTER_NAME:-topology-dl-cluster}"
TASK_DEFINITION="${TASK_DEFINITION:-topology-dl-track-a}"
SUBNET_IDS="${SUBNET_IDS:-}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"

# 인자 파싱
DATASET="ECG200"
MODEL="ph_mlp"
SEED="42"
VECTORIZATION="persistence_landscape"
USE_SPOT="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --vectorization)
            VECTORIZATION="$2"
            shift 2
            ;;
        --no-spot)
            USE_SPOT="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Track A 실험 실행 (AWS ECS/Fargate)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "Vectorization: $VECTORIZATION"
echo "Use Spot: $USE_SPOT"
echo "=========================================="

# 명령어 구성
COMMAND="[\"python\", \"experiments/track_a/train.py\", \"--dataset\", \"$DATASET\", \"--model\", \"$MODEL\", \"--seed\", \"$SEED\""

if [ "$MODEL" = "ph_mlp" ]; then
    COMMAND="$COMMAND, \"--vectorization\", \"$VECTORIZATION\""
fi

COMMAND="$COMMAND, \"--use_wandb\"]"

# 용량 제공자 설정
if [ "$USE_SPOT" = "true" ]; then
    CAPACITY_PROVIDER="FARGATE_SPOT"
else
    CAPACITY_PROVIDER="FARGATE"
fi

# ECS 태스크 실행
echo "Launching ECS task..."

TASK_ARN=$(aws ecs run-task \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --cluster "$CLUSTER_NAME" \
    --task-definition "$TASK_DEFINITION" \
    --capacity-provider-strategy "capacityProvider=$CAPACITY_PROVIDER,weight=1" \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
    --overrides "{\"containerOverrides\": [{\"name\": \"experiment\", \"command\": $COMMAND}]}" \
    --query 'tasks[0].taskArn' \
    --output text)

echo "Task launched: $TASK_ARN"
echo ""
echo "모니터링 명령어:"
echo "  aws ecs describe-tasks --profile $AWS_PROFILE --cluster $CLUSTER_NAME --tasks $TASK_ARN"
echo ""
echo "로그 확인:"
echo "  aws logs tail /ecs/topology-dl --profile $AWS_PROFILE --follow"
