# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

위상/구조 수학(TDA, Tensor Decomposition)을 활용한 딥러닝 효율성 검증 실험. 핵심 가설: "위상/구조 수학으로 계산량을 줄이거나, 같은 계산량에서 성능을 향상시킬 수 있다."

## 빌드 및 실행

```bash
# 환경 설정
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Track A 실험 (PH 기반 시계열 분류)
python experiments/track_a/train.py --model ph_mlp --dataset ECG200
python experiments/track_a/train.py --model inceptiontime --dataset ECG200  # 베이스라인

# 배치 실험 (전체 데이터셋 × 모델 × 시드)
python experiments/track_a/run_experiments.py --all            # 전체
python experiments/track_a/run_experiments.py --datasets ECG200 FordA --models ph_mlp inceptiontime
python experiments/track_a/run_experiments.py --ablation       # Ablation 포함
python experiments/track_a/run_experiments.py --dry_run        # 명령어만 확인

# SageMaker Spot Training
python infrastructure/sagemaker/run_benchmark.py --all --dry-run  # 비용 확인
python infrastructure/sagemaker/run_benchmark.py --dataset ECG200 --model ph_mlp --role <ARN>

# 테스트
pytest tests/
pytest tests/test_tda.py -v
pytest tests/test_tda.py::test_persistence -v

# 코드 포맷팅
black src/ experiments/
isort src/ experiments/
```

## 아키텍처

### 실험 트랙 구조
- **Track A** (완료, 가설 기각): Persistent Homology → 시계열 분류. PH-MLP가 InceptionTime 대비 8~32%p 낮은 성능
- **Track B** (구현됨, 미실험): Simplicial/Cell Complex → 그래프. 경계/코경계 메시지 패싱
- **Track C** (구현됨, 미실험): Tensor Decomposition → 범용. TT 분해로 가중치 압축

### 데이터 흐름 (Track A)

```
시계열 입력 → Takens Embedding → Persistence Diagram (H0/H1)
                                        ↓
                          Vectorization (landscape/image/stats)
                                        ↓
                                   PHMLP (2층)
                                        ↓
                                     분류
```

베이스라인 모델(InceptionTime, ResNet1D, FCN, CNN1D, GRU, TCN)은 원본 시계열을 직접 입력받음.

### 핵심 모듈

| 경로 | 역할 |
|------|------|
| `src/tda/` | TDA 파이프라인: embeddings → persistence (ripser/gudhi) → vectorization |
| `src/models/tda/ph_mlp.py` | PH 벡터 → MLP 분류기 |
| `src/models/baselines/` | InceptionTime, ResNet1D, FCN, CNN1D, GRU, TCN |
| `src/models/simplicial/simplicial_nn.py` | SimplicialConv: 경계/코경계 행렬 기반 메시지 패싱 |
| `src/models/tensor/tt_linear.py` | TTLinear: 가중치를 TT core로 분해, `compression_ratio()` 제공 |
| `src/data/ucr.py` | UCR Archive 로더 (TSV 형식, 레이블 0-indexed 자동 변환) |
| `src/utils/metrics.py` | `compute_metrics` (F1/AUROC) + `compute_efficiency_metrics` (params/latency/throughput) |

### 모델 공통 인터페이스

모든 모델은 `count_parameters() → int` 메서드를 구현. TTLinear은 추가로 `compression_ratio() → float` 제공.

### 실험 설정

- `configs/base.yaml`: AdamW, cosine scheduler, early stopping (patience=10), 시드 [42, 123, 456]
- `configs/track_{a,b,c}.yaml`: 트랙별 하이퍼파라미터
- `experiments/track_a/train.py`: 단일 실험. `--model` (ph_mlp|inceptiontime|resnet|fcn|cnn|gru|tcn), `--vectorization` (persistence_landscape|persistence_image|statistics)
- `experiments/track_a/run_experiments.py`: 배치 실행기. 데이터셋 5개 × 모델 × 시드 3개 조합, ablation (vectorization/homology/embedding 파라미터)

### 인프라

- 로컬, AWS ECS Fargate, SageMaker 중 선택
- AWS 사용시 **스팟 요금제 필수**. 병렬화 가능하면 여러 태스크 동시 실행
- `infrastructure/sagemaker/`: SageMaker Managed Spot Training (ml.g4dn.xlarge, ~$0.16/hr)
- `infrastructure/ecs/`: Dockerfile + ECS Fargate 태스크 정의
- 데이터는 `./data/ucr/` 에 UCR Archive TSV 형식으로 저장 (`scripts/download_ucr.py`로 다운로드)

### 결과 저장 구조

`results/track_a/{dataset}/{dataset}_{model}_seed{seed}/` 하위에:
- `metrics.json`: 성능 + 효율 지표
- `training_log.json`: 에폭별 로그
- `model.pt`: 모델 가중치

## 코드 컨벤션

- Python: Black, isort
- Type hints 권장
- Docstring: Google style

## Q&A 스킬

수학적 개념 학습을 위한 Q&A 자동 기록:

| 명령어 | 설명 |
|--------|------|
| `/qa <질문>` | 개념 설명 후 `docs/qa/` 에 저장 |
| `/qa-list [카테고리]` | 저장된 Q&A 목록 |
| `/qa-merge` | 모든 Q&A를 `docs/CONCEPTS.md`로 병합 |

카테고리: `topology`, `algebra`, `geometry`, `ml`, `statistics`, `general`
