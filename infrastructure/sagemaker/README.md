# SageMaker Managed Spot Training

Track A 벤치마크를 AWS SageMaker Spot Training으로 실행합니다.

## 장점

- **비용 절감**: Spot Instance로 최대 70-90% 비용 절감
- **GPU 가속**: ml.g4dn.xlarge (NVIDIA T4) 사용
- **자동 체크포인트**: Spot 중단 시 자동 복구
- **병렬 실행**: 여러 실험 동시 실행 가능

## 사전 요구사항

1. AWS CLI 설정
2. SageMaker IAM Role
3. 데이터셋 다운로드 완료

## 1. IAM Role 생성

SageMaker 실행에 필요한 IAM Role:

```bash
# Trust relationship
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

필요한 정책:
- `AmazonSageMakerFullAccess`
- S3 버킷 접근 권한

## 2. 데이터 업로드

```bash
# 로컬 데이터가 준비되어 있어야 함
python scripts/download_ucr.py
```

## 3. 실험 실행

### 단일 실험

```bash
python infrastructure/sagemaker/run_benchmark.py \
    --dataset ECG200 \
    --model inceptiontime \
    --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole \
    --profile personal
```

### 전체 벤치마크

```bash
# Dry run (비용 확인)
python infrastructure/sagemaker/run_benchmark.py \
    --all \
    --dry-run \
    --profile personal

# 실제 실행
python infrastructure/sagemaker/run_benchmark.py \
    --all \
    --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole \
    --profile personal
```

## 4. 모니터링

```bash
# Job 목록
aws sagemaker list-training-jobs \
    --profile personal \
    --name-contains track-a \
    --sort-by CreationTime \
    --sort-order Descending

# Job 상태
aws sagemaker describe-training-job \
    --profile personal \
    --training-job-name JOB_NAME
```

## 5. 결과 다운로드

```bash
# S3에서 결과 다운로드
aws s3 sync \
    s3://YOUR_BUCKET/topology-dl/output/ \
    ./results/sagemaker/ \
    --profile personal
```

## 비용 추정

| 인스턴스 | GPU | On-Demand | Spot (예상) |
|----------|-----|-----------|-------------|
| ml.g4dn.xlarge | T4 16GB | $0.526/hr | ~$0.16/hr |
| ml.g4dn.2xlarge | T4 16GB | $0.752/hr | ~$0.23/hr |
| ml.p3.2xlarge | V100 16GB | $3.06/hr | ~$0.92/hr |

전체 벤치마크 (5 데이터셋 × 2 모델 × 3 시드 = 30 실험):
- 예상 시간: 실험당 ~30분 = 15시간 (순차) / 1시간 (30개 병렬)
- Spot 비용: ~$2-5 (병렬 실행 시)

## 파일 구조

```
infrastructure/sagemaker/
├── train_sagemaker.py    # SageMaker 학습 스크립트
├── run_benchmark.py      # 벤치마크 실행 스크립트
├── requirements.txt      # 학습 환경 의존성
└── README.md
```
