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
python experiments/track_a/train.py --model cnn --dataset ECG200  # 베이스라인

# 테스트
pytest tests/
pytest tests/test_tda.py -v  # 단일 파일
pytest tests/test_tda.py::test_persistence -v  # 단일 테스트

# 코드 포맷팅
black src/ experiments/
isort src/ experiments/
```

## 아키텍처

### 실험 트랙 구조
- **Track A**: Persistent Homology → 시계열 분류. 원본 대신 위상 요약 피처로 작은 모델 사용
- **Track B**: Simplicial/Cell Complex → 그래프. 고차 관계로 전역 attention 회피
- **Track C**: Tensor Decomposition → 범용. 가중치를 TT/MPO로 압축

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

### 핵심 모듈

| 경로 | 역할 |
|------|------|
| `src/tda/embeddings.py` | Takens, sliding window embedding |
| `src/tda/persistence.py` | ripser/gudhi 백엔드로 PH 계산 |
| `src/tda/vectorization.py` | Persistence landscape/image/stats 변환 |
| `src/models/tda/ph_mlp.py` | PH 벡터 → MLP 분류기 |
| `src/models/simplicial/simplicial_nn.py` | 경계/코경계 기반 메시지 패싱 |
| `src/models/tensor/tt_linear.py` | Tensor Train Linear (가중치 압축) |
| `src/utils/metrics.py` | F1, AUROC + 효율 지표 (params, latency, throughput) |

### 설정 시스템
- `configs/base.yaml`: 공통 설정 (optimizer, scheduler, metrics)
- `configs/track_{a,b,c}.yaml`: 트랙별 하이퍼파라미터

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
