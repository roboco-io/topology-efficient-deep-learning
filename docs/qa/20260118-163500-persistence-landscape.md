---
date: 2026-01-18 16:35:00
tags: [persistence-landscape, vectorization, TDA, persistent-homology]
category: topology
---

# Q: Persistence Landscape가 뭐야?

## A: Persistence Diagram을 벡터 공간으로 변환하는 방법

Persistence Landscape는 Persistence Diagram을 함수 공간으로 변환하는 방법입니다. Bubenik (2015)이 제안했으며, 위상적 특징을 벡터화하여 머신러닝에 활용할 수 있게 합니다.

### 핵심 개념

1. **텐트 함수**: 각 위상적 특징(구멍, 연결 성분)을 삼각형으로 표현
   - 높이 = persistence (수명) = 특징의 중요도
   - 위치 = birth/death 시점

2. **k번째 Landscape**: 각 시점에서 k번째로 중요한 특징의 기여도
   - λ₁: 가장 두드러진 특징들의 윤곽
   - λ₂, λ₃, ...: 차순위 특징들

3. **이산화**: t를 균등 간격으로 샘플링하여 벡터로 변환

### 수학적 정의

Persistence Diagram의 각 점 (bᵢ, dᵢ)에 대해 텐트 함수를 정의:

```
Λᵢ(t) = max(0, min(t - bᵢ, dᵢ - t))
```

이 함수는 bᵢ에서 시작해 중간점에서 최댓값 (dᵢ - bᵢ)/2를 가지고, dᵢ에서 0으로 돌아가는 삼각형 모양입니다.

**k번째 Landscape** λₖ(t)는 모든 텐트 함수 값들 중 k번째로 큰 값:

```
λₖ(t) = k-th largest of {Λᵢ(t)}
```

시각화:
```
      λ₁(t)
       /\
      /  \      /\
     /    \    /  \
    /      \  /    \
   /        \/      \
  /                  \
 /____________λ₂(t)___\________→ t
```

### 장점

| 특성 | 설명 |
|------|------|
| **안정성** | 입력 데이터의 작은 변화 → Landscape의 작은 변화 |
| **벡터 공간** | 덧셈, 스칼라 곱, 평균 계산 가능 |
| **통계적 분석** | 여러 샘플의 평균 Landscape 계산 가능 |
| **고유성** | 원래 Diagram을 (거의) 복원 가능 |

### 다른 Vectorization과 비교

| 방법 | 출력 차원 | 계산 비용 | 정보 보존 |
|------|----------|----------|----------|
| Landscape | k × res | O(n log n) | 높음 |
| Persistence Image | res × res | O(n × res²) | 중간 |
| Statistics | ~10 | O(n) | 낮음 |

### 예시: ECG200 실험 결과

| Vectorization | F1 (macro) | Total Latency |
|---------------|------------|---------------|
| **Landscape** | **0.7478** | 3.05 ms |
| Image | 0.6081 | 39.45 ms |
| Statistics | 0.6938 | 1.61 ms |

Landscape가 가장 좋은 성능을 보인 이유:
1. 중요한 위상적 특징들의 순서와 크기 정보 유지
2. 1000차원은 MLP가 학습하기에 충분한 표현력 제공
3. Image 대비 10배 이상 빠른 계산

### 코드 위치

- `src/tda/vectorization.py`: `persistence_landscape()` 함수
- 파라미터: `num_landscapes=5`, `resolution=100`

### 관련 개념

- [[Persistence Diagram]]: Landscape의 입력
- [[Persistence Image]]: 대안적 벡터화 (2D 히트맵)
- [[Takens Embedding]]: 시계열 → 포인트 클라우드 변환

### 참고문헌

- Bubenik, P. (2015). "Statistical Topological Data Analysis using Persistence Landscapes." *JMLR*
