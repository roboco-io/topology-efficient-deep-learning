# Track A 실험 프로토콜

## 1. 실험 목표

**가설**: 원본 시계열 대신 PH 요약 피처를 사용하면, 더 작은 모델로 유사 성능을 달성하여 추론 비용을 절감할 수 있다.

**검증 시나리오** (둘 다 검증):
1. **효율 시나리오**: 성능 하락 ≤1%p (F1) + 추론시간 ≥30% 절감
2. **성능 시나리오**: 동일 추론시간에서 성능 +1~2%p 개선

---

## 2. 데이터셋

**선정 기준**: 다양성 우선 (길이/클래스수/도메인)

| 데이터셋 | 길이 | 클래스 | 도메인 | 선정 이유 |
|----------|------|--------|--------|-----------|
| ECG200 | 96 | 2 | 의료 | 짧은 길이, 2클래스, ECG |
| FordA | 500 | 2 | 센서 | 중간 길이, 노이즈 많음 |
| ElectricDevices | 96 | 7 | 센서 | 다중 클래스 |
| Wafer | 152 | 2 | 제조 | 불균형 데이터 |
| UWaveGestureLibraryAll | 945 | 8 | 모션 | 긴 길이, 다중 클래스 |

**성공 임계값**: 5/5 데이터셋에서 기준 충족

---

## 3. 베이스라인

**참조**: InceptionTime (Fawaz et al., 2020)

| 모델 | 구조 | 파라미터 수 (예상) |
|------|------|-------------------|
| InceptionTime | 6 Inception 모듈 | ~400K |
| ResNet-1D | 3 ResBlock | ~100K |
| FCN | 3 Conv + GAP | ~50K |

---

## 4. 제안 모델

```
시계열 → Takens Embedding (d=3, τ=auto) → PH (H0+H1) → Vectorization → MLP (2층)
```

**Vectorization 방법** (Ablation으로 비교):
1. Persistence Landscape (k=5)
2. Persistence Image (20x20)
3. Persistence Statistics (birth/death/lifetime 통계)

**MLP 구조**:
- Hidden: 64 → 32 → num_classes
- Activation: ReLU
- Dropout: 0.2

---

## 5. 공정 비교 조건

| 항목 | 설정 |
|------|------|
| Optimizer | AdamW |
| LR | 1e-3 (cosine decay) |
| Epochs | 100 (early stopping patience=10) |
| Batch size | 32 |
| Data split | 논문 기본 train/test split |

---

## 6. 측정 지표

**성능**:
- F1 Score (macro)
- Accuracy

**효율**:
- 추론시간 (ms/sample) - PH 포함/제외 둘 다 보고
- 파라미터 수
- FLOPs (fvcore)

**복합**:
- F1 / 추론시간
- F1 / 파라미터 수

---

## 7. 통계 검증

- **시드**: 3개 (42, 123, 456)
- **검정**: Paired t-test
- **유의수준**: p < 0.05
- **보고**: 평균 ± 표준편차

---

## 8. Ablation 설계

**우선순위** (순차 진행):

1. **Vectorization**: Landscape vs Image vs Stats
   - 최선 방법 선정 후 고정

2. **PH 차수**: H0만 vs H0+H1
   - 계산량 대비 효과 검증

3. **Embedding 방법**: Takens vs Sliding Window
   - 윈도우 크기: [16, 32, 64]
   - Takens 지연 차수: [2, 3, 4]

---

## 9. 실험 환경

| 항목 | 설정 |
|------|------|
| 컴퓨팅 | AWS ECS/Fargate (Spot) |
| AWS Profile | personal |
| 실험 추적 | wandb |
| HP 튜닝 | 고정값 우선, 필요시 Grid Search |
| 일정 | 기한 없음 |

---

## 10. 실패 시 대응

Track A가 5/5 데이터셋에서 성공하지 못할 경우:
- 성공한 데이터셋 특성 분석
- 조건부 결론 도출 ("이런 조건에서 유효")
- Track B/C 진행 여부는 결과 보고 후 결정

---

## 11. 실행 명령어

### 데이터 준비

```bash
# UCR 데이터셋 다운로드
python scripts/download_ucr.py --datasets ECG200 FordA ElectricDevices Wafer UWaveGestureLibraryAll
```

### 베이스라인 실행

```bash
# InceptionTime
python experiments/track_a/train.py \
    --model inceptiontime \
    --dataset ECG200 \
    --epochs 100 \
    --seed 42

# ResNet-1D
python experiments/track_a/train.py \
    --model resnet \
    --dataset ECG200 \
    --epochs 100 \
    --seed 42

# FCN
python experiments/track_a/train.py \
    --model fcn \
    --dataset ECG200 \
    --epochs 100 \
    --seed 42
```

### PH-MLP 실행

```bash
python experiments/track_a/train.py \
    --model ph_mlp \
    --dataset ECG200 \
    --vectorization persistence_landscape \
    --delay 5 \
    --dimension 3 \
    --homology_dims 0 1 \
    --epochs 100 \
    --seed 42
```

### 전체 실험 실행

```bash
# 모든 데이터셋, 모든 모델, 모든 시드
python experiments/track_a/run_experiments.py --all
```

---

## 12. 결과 저장 구조

```
results/track_a/
├── {dataset}/
│   ├── {model}_seed{seed}/
│   │   ├── config.yaml
│   │   ├── metrics.json
│   │   ├── model.pt
│   │   └── training_log.csv
│   └── summary.json
└── final_report.md
```

---

## 13. wandb 프로젝트 구조

- **Project**: `topology-efficient-dl`
- **Group**: `track_a`
- **Tags**: `[dataset_name, model_name, vectorization_method]`
- **Config**: 모든 하이퍼파라미터
- **Logged metrics**:
  - `train/loss`, `train/acc`
  - `val/loss`, `val/acc`, `val/f1`
  - `test/acc`, `test/f1`, `test/auroc`
  - `efficiency/params`, `efficiency/latency_ms`, `efficiency/throughput`
