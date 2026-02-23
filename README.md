# Topology-Efficient Deep Learning

위상/구조 수학을 활용한 딥러닝 모델의 계산 효율성 검증 실험 프로젝트

## 핵심 가설

> 위상/구조 수학을 사용하여 총 계산량(FLOPs, 메모리, 학습시간)을 줄이거나, 같은 계산량에서 성능을 향상시킬 수 있다.

## 실험 트랙

| 트랙 | 기법 | 도메인 | 목표 |
|------|------|--------|------|
| **A** | Persistent Homology | 시계열 | 위상 피처로 모델 크기 축소 |
| **B** | Simplicial/Cell Complex | 그래프 | 고차 관계로 전역 attention 회피 |
| **C** | Tensor Decomposition | 범용 | 가중치 압축으로 연산량 절감 |

## 프로젝트 구조

```
topology-efficient-deep-learning/
├── README.md
├── PRD.md                    # 상세 요구사항 문서
├── ideation.md               # 초기 아이디어 문서
├── configs/                  # 실험 설정
│   ├── track_a.yaml
│   ├── track_b.yaml
│   └── track_c.yaml
├── src/
│   ├── data/                 # 데이터 로더
│   ├── models/               # 모델 구현
│   │   ├── baselines/        # CNN, GRU, TCN, GAT 등
│   │   ├── tda/              # PH 기반 모델
│   │   ├── simplicial/       # Simplicial/Cell NN
│   │   └── tensor/           # 텐서 분해 모델
│   ├── tda/                  # TDA 유틸리티
│   │   ├── embeddings.py     # Takens, sliding window
│   │   ├── persistence.py    # PH 계산
│   │   └── vectorization.py  # Landscape, image, stats
│   └── utils/                # 공통 유틸리티
├── experiments/              # 실험 스크립트
│   ├── track_a/
│   ├── track_b/
│   └── track_c/
├── notebooks/                # 분석 및 시각화
├── results/                  # 실험 결과
├── docs/
│   └── qa/                   # Q&A 문서 저장소
└── scripts/
    └── merge_qa.py           # Q&A 병합 스크립트
```

## Q&A 학습 시스템

이 프로젝트는 수학적 개념 학습을 위한 Q&A 자동 기록 시스템을 포함합니다.

### 사용법

```bash
# 개념 설명 요청 및 자동 기록
/qa Persistent Homology가 뭐야?
/qa Takens embedding의 원리

# 저장된 Q&A 목록 확인
/qa-list
/qa-list topology

# 모든 Q&A를 하나의 문서로 병합
/qa-merge

# 스크립트로 병합 (CLI)
python scripts/merge_qa.py --output docs/CONCEPTS.md
```

### Q&A 카테고리

| 카테고리 | 설명 |
|----------|------|
| topology | 위상수학 (PH, Simplicial Complex 등) |
| algebra | 대수학 (Tensor Decomposition 등) |
| geometry | 기하학 (Manifold, Embedding 등) |
| ml | 머신러닝 |
| statistics | 통계학 |
| general | 기타 |

## 설치

```bash
# 저장소 클론
git clone <repository-url>
cd topology-efficient-deep-learning

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 주요 의존성

```
torch>=2.0
pytorch-lightning>=2.0
torch-geometric>=2.3
giotto-tda>=0.6
ripser>=0.6
gudhi>=3.8
wandb>=0.15
```

## 사용법

### Track A: Persistent Homology 실험

```bash
# 데이터 준비
python scripts/prepare_ucr.py --dataset ECG200

# PH 피처 추출
python src/tda/persistence.py \
    --input data/ECG200 \
    --method takens \
    --dim 3 \
    --delay 5

# 학습 실행
python experiments/track_a/train.py \
    --config configs/track_a.yaml \
    --model ph_mlp
```

### Track B: Simplicial NN 실험

```bash
# 고차 관계 구축
python src/data/build_simplicial.py \
    --input data/graph.json \
    --time_window 60 \
    --output data/simplicial.pt

# 학습 실행
python experiments/track_b/train.py \
    --config configs/track_b.yaml \
    --model simplicial_nn
```

### Track C: 텐서 분해 실험

```bash
# 텐서 분해 적용
python experiments/track_c/train.py \
    --config configs/track_c.yaml \
    --compression tt \
    --rank 16
```

## 실험 설정

### 공통 설정 (configs/base.yaml)

```yaml
training:
  optimizer: adamw
  scheduler: cosine
  early_stopping:
    patience: 10
    metric: val_loss
  seeds: [42, 123, 456]

evaluation:
  metrics:
    - accuracy
    - f1_macro
    - auroc
  efficiency:
    - params
    - flops
    - latency_ms
    - peak_vram_mb
```

### Track A 설정 예시

```yaml
tda:
  embedding: takens
  delay: 5
  dimension: 3
  homology: [0, 1]
  vectorization: persistence_landscape

model:
  type: mlp
  hidden_dims: [64, 32]
  dropout: 0.1
```

## 평가 지표

### 성능 지표
- 분류: F1, AUROC, Accuracy
- 예측: MAE, MSE, RMSE

### 효율 지표
- 학습 시간 (wall-clock)
- 추론 지연 (ms)
- 처리량 (samples/s)
- 파라미터 수
- Peak VRAM
- FLOPs

### 복합 지표
- 성능/추론시간
- 성능/FLOPs

## 성공 기준

| 트랙 | 기준 |
|------|------|
| A | 성능 ≤1%p 하락 + 추론시간 ≥30% 절감 |
| B | 고밀도 그래프에서 동일 성능 + VRAM/latency 절감 |
| C | 파라미터 30-70% 절감 + 성능 유지 |

---

## 실험 결과

### Track A: Persistent Homology 시계열 분류

**가설**: PH 요약 피처를 사용하면 더 작은 모델(PH-MLP)로 유사 성능을 달성하여 추론 비용을 절감할 수 있다.

#### 결과 요약

| Dataset | PH-MLP (F1) | InceptionTime (F1) | 차이 |
|---------|-------------|-------------------|------|
| ECG200 | 0.755 ± 0.008 | 0.839 ± 0.100 | -8.4%p |
| FordA | 0.641 ± 0.013 | 0.930 ± 0.034 | -28.9%p |
| ElectricDevices | 0.330 ± 0.017 | 0.652 ± 0.010 | -32.2%p |
| Wafer | 0.833 ± 0.016 | 0.963 ± 0.039 | -13.0%p |

#### 결론: **가설 기각**

- PH-MLP가 모든 데이터셋에서 InceptionTime 대비 **8~32%p 낮은 성능**
- 성공 기준(≤1%p 하락) 미달성

#### 원인 분석

1. **정보 손실**: Takens embedding → PH → Vectorization 과정에서 시간적 패턴 손실
2. **계산 비용**: 긴 시계열에서 PH 계산이 O(n³)으로 비효율적
3. **다중 클래스 한계**: PH 피처가 복잡한 분류 경계 표현에 부적합

#### 실험 환경

- AWS SageMaker Managed Spot Training (ml.g4dn.xlarge)
- 3개 시드 반복 (42, 123, 456)
- 총 비용: ~$0.40

> 상세 결과: [results/track_a_results.md](results/track_a_results.md)

---

## 결과 시각화

```bash
# 성능-비용 곡선 생성
python scripts/plot_results.py \
    --results results/track_a \
    --x latency_ms \
    --y f1_macro \
    --output figures/track_a_pareto.png
```

## 튜토리얼

- [바이브 코딩으로 딥러닝 연구하기](docs/tutorial-vibe-coding-dl-research.md): 이 프로젝트의 수행 이력을 기반으로, AI 코딩 어시스턴트와 서버리스 인프라를 활용하여 최소 비용($0.40)으로 딥러닝 가설을 검증하는 전 과정을 다루는 튜토리얼

## 참고 문헌

- [Persistent Homology for Time Series](https://arxiv.org/abs/1910.14424)
- [Simplicial Neural Networks](https://arxiv.org/abs/2010.03633)
- [Tensor Train Decomposition](https://arxiv.org/abs/1509.06569)
- [giotto-tda Documentation](https://giotto-ai.github.io/gtda-docs/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

## 라이선스

MIT License
