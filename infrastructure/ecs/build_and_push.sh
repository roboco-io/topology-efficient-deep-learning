#!/bin/bash
# ECR에 Docker 이미지 빌드 및 푸시

set -e

AWS_PROFILE=${AWS_PROFILE:-personal}
AWS_REGION=${AWS_REGION:-ap-northeast-2}
ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
ECR_REPO="topology-dl"
IMAGE_TAG="latest"

echo "Account: $ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo "Repository: $ECR_REPO"

# ECR 로그인
aws ecr get-login-password --profile $AWS_PROFILE --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# ECR 리포지토리 생성 (없으면)
aws ecr describe-repositories --profile $AWS_PROFILE --region $AWS_REGION --repository-names $ECR_REPO 2>/dev/null || \
    aws ecr create-repository --profile $AWS_PROFILE --region $AWS_REGION --repository-name $ECR_REPO

# Docker 이미지 빌드
echo "Building Docker image..."
docker build -t $ECR_REPO:$IMAGE_TAG -f Dockerfile .

# 태그 및 푸시
FULL_IMAGE="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"
docker tag $ECR_REPO:$IMAGE_TAG $FULL_IMAGE
docker push $FULL_IMAGE

echo "Image pushed: $FULL_IMAGE"
