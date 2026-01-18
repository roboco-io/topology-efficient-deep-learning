# AWS ECS/Fargate 실험 환경 설정

Track A 실험을 AWS ECS/Fargate (Spot)에서 실행하기 위한 설정 가이드.

## 사전 요구사항

- AWS CLI 설치 및 설정
- Docker 설치
- AWS 계정 및 적절한 IAM 권한

## 1. AWS 프로파일 설정

```bash
aws configure --profile personal
# AWS Access Key ID
# AWS Secret Access Key
# Default region: ap-northeast-2
# Default output format: json
```

## 2. ECR 리포지토리 생성

```bash
# 리포지토리 생성
aws ecr create-repository \
    --profile personal \
    --repository-name topology-dl-experiments \
    --region ap-northeast-2

# ECR 로그인
aws ecr get-login-password --profile personal --region ap-northeast-2 | \
    docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-northeast-2.amazonaws.com

# 이미지 빌드 및 푸시
docker build -t topology-dl-experiments .
docker tag topology-dl-experiments:latest <ACCOUNT_ID>.dkr.ecr.ap-northeast-2.amazonaws.com/topology-dl-experiments:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-northeast-2.amazonaws.com/topology-dl-experiments:latest
```

## 3. ECS 클러스터 생성

```bash
aws ecs create-cluster \
    --profile personal \
    --cluster-name topology-dl-cluster \
    --capacity-providers FARGATE FARGATE_SPOT \
    --default-capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1
```

## 4. EFS 파일 시스템 (선택사항)

데이터셋과 결과를 영구 저장하려면 EFS를 사용.

```bash
# EFS 생성
aws efs create-file-system \
    --profile personal \
    --creation-token topology-dl-efs \
    --performance-mode generalPurpose

# 마운트 타겟 생성 (VPC 서브넷에 연결)
aws efs create-mount-target \
    --profile personal \
    --file-system-id fs-xxx \
    --subnet-id subnet-xxx \
    --security-groups sg-xxx
```

## 5. IAM 역할 생성

### Task Execution Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

필요한 정책:
- AmazonECSTaskExecutionRolePolicy
- CloudWatchLogsFullAccess (또는 제한된 로그 권한)

### Task Role

EFS 접근 권한:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "elasticfilesystem:ClientMount",
        "elasticfilesystem:ClientWrite"
      ],
      "Resource": "arn:aws:elasticfilesystem:ap-northeast-2:*:file-system/*"
    }
  ]
}
```

## 6. CloudWatch 로그 그룹 생성

```bash
aws logs create-log-group \
    --profile personal \
    --log-group-name /ecs/topology-dl
```

## 7. 태스크 정의 등록

환경 변수 대체 후 등록:

```bash
# 환경 변수 설정
export EXECUTION_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole"
export TASK_ROLE_ARN="arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole"
export ECR_IMAGE="ACCOUNT_ID.dkr.ecr.ap-northeast-2.amazonaws.com/topology-dl-experiments:latest"
export EFS_FILE_SYSTEM_ID="fs-xxx"
export WANDB_API_KEY="your-wandb-key"

# 태스크 정의 등록
envsubst < ecs-task-definition.json | aws ecs register-task-definition \
    --profile personal \
    --cli-input-json file:///dev/stdin
```

## 8. 실험 실행

### 단일 실험

```bash
./run_ecs_experiment.sh --dataset ECG200 --model ph_mlp --seed 42
```

### 배치 실험

```bash
# 모든 조합 실행
./batch_experiments.sh

# Dry run (실제 실행 없이 확인)
./batch_experiments.sh --dry-run
```

## 9. 모니터링

```bash
# 실행 중인 태스크 확인
aws ecs list-tasks --profile personal --cluster topology-dl-cluster

# 태스크 상태 확인
aws ecs describe-tasks \
    --profile personal \
    --cluster topology-dl-cluster \
    --tasks TASK_ARN

# 로그 확인
aws logs tail /ecs/topology-dl --profile personal --follow
```

## 비용 최적화

1. **Fargate Spot 사용**: 최대 70% 비용 절감
2. **적절한 리소스 크기**:
   - CPU: 2 vCPU (대부분의 실험에 충분)
   - Memory: 4 GB
3. **병렬 실행**: 여러 태스크를 동시에 실행하여 총 실행 시간 단축

## 문제 해결

### 태스크가 시작되지 않는 경우

1. CloudWatch 로그에서 오류 확인
2. 보안 그룹 설정 확인 (아웃바운드 허용 필요)
3. 서브넷이 인터넷 접근 가능한지 확인 (NAT Gateway 또는 Public IP)

### EFS 마운트 실패

1. 보안 그룹에서 NFS 포트 (2049) 허용
2. 마운트 타겟이 올바른 서브넷에 있는지 확인
