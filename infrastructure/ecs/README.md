# ECS Fargate Spot - PH-MLP Training

CPU 집약적인 PH-MLP 학습을 ECS Fargate Spot으로 실행합니다.

## 장점

- **비용 효율**: Fargate Spot ($0.07/hr for 4 vCPU) vs SageMaker GPU ($0.16/hr for T4)
- **병렬 처리**: joblib로 4 vCPU 활용
- **병렬 실행**: 여러 실험 동시 실행 가능

## 사전 요구사항

1. Docker 설치
2. AWS CLI 설정
3. ECS 클러스터 생성
4. IAM Role 생성

## 1. ECS 클러스터 생성

```bash
aws ecs create-cluster \
    --cluster-name topology-dl \
    --capacity-providers FARGATE_SPOT FARGATE \
    --default-capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1 \
    --profile personal
```

## 2. IAM Role 생성

### Task Execution Role
```bash
aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --profile personal

aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
    --profile personal
```

### Task Role (S3 접근용)
```bash
aws iam create-role \
    --role-name ecsTaskRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --profile personal

aws iam put-role-policy \
    --role-name ecsTaskRole \
    --policy-name S3Access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": ["arn:aws:s3:::sagemaker-*", "arn:aws:s3:::sagemaker-*/*"]
        }]
    }' \
    --profile personal
```

## 3. CloudWatch Log Group 생성

```bash
aws logs create-log-group \
    --log-group-name /ecs/topology-dl \
    --profile personal
```

## 4. Docker 이미지 빌드 및 푸시

```bash
chmod +x build_and_push.sh
./build_and_push.sh
```

## 5. 실험 실행

### 단일 실험
```bash
python run_ph_mlp.py --dataset ECG200 --seed 42
```

### 전체 벤치마크 (4개 병렬)
```bash
python run_ph_mlp.py --all --parallel 4
```

### Dry run
```bash
python run_ph_mlp.py --all --dry-run
```

## 비용 비교

| 설정 | 시간당 비용 | PH-MLP 1실험 | 15실험 |
|------|-----------|-------------|--------|
| SageMaker ml.g4dn.xlarge Spot | $0.16 | ~$0.04 | ~$0.60 |
| Fargate 4vCPU/8GB Spot | $0.07 | ~$0.02 | ~$0.30 |

병렬 실행 시 더 빠르고 저렴합니다.

## 파일 구조

```
infrastructure/ecs/
├── Dockerfile           # CPU 최적화 컨테이너
├── requirements.txt     # PyTorch CPU, joblib
├── train_ph_mlp.py     # 학습 스크립트 (병렬 PH)
├── run_ph_mlp.py       # 벤치마크 실행기
├── build_and_push.sh   # Docker 빌드/푸시
├── task-definition.json # ECS 태스크 정의
└── README.md
```
