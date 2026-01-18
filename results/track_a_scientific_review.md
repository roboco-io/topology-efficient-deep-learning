# Track A 실험 과학적 리뷰

## 1. 연구 배경 및 문헌 맥락

### 1.1 Persistent Homology (PH) 기반 시계열 분류 현황

최신 문헌 조사 결과, PH/TDA 기반 시계열 분류 연구는 활발하지만 **InceptionTime 등 딥러닝 베이스라인과의 직접 비교는 거의 없다** [1][2].

| 연구 | 방법 | 성능 | 비교 대상 |
|------|------|------|-----------|
| arXiv 2025 [1] | δ-temporal motifs + clique complex | 92-100% acc | 그래프 기반 방법 |
| PMC 2024 [3] | VR filtration on fMRI | MCI 분류 성공 | 통계적 방법 |
| arXiv 2003.06462 | Persistence curves | UCR 일부 | 전통 ML |

**핵심 발견**: PH 방법은 주로 그래프, fMRI, 금융 이상탐지 등 **특수 도메인**에서 효과적이며, 일반 UCR 시계열에서 딥러닝 대비 성능은 보고된 바 없음.

### 1.2 알려진 PH 방법의 한계

문헌에서 보고된 한계점:
1. **계산 비용**: Vietoris-Rips filtration은 O(n³) 복잡도
2. **표현 의존성**: point cloud 구성 방법(Takens, sliding window)에 민감
3. **일반화 한계**: node label 없는 그래프에서 강점, 일반 시계열에서 약점
4. **도메인 특이성**: 다변량/구조화 데이터에 적합, 단변량 시계열에는 하이브리드 필요

---

## 2. 실험 설계 평가

### 2.1 방법론적 강점

| 항목 | 평가 |
|------|------|
| 데이터셋 다양성 | ✅ 길이(96~945), 클래스수(2~8), 도메인 다양 |
| 통계적 검증 | ✅ 3개 시드 반복 실험 |
| 공정 비교 | ✅ 동일 optimizer, LR, epochs |
| 베이스라인 선정 | ✅ InceptionTime (SOTA 베이스라인) |

### 2.2 방법론적 한계

| 항목 | 문제점 | 개선 방향 |
|------|--------|-----------|
| Embedding 방법 | Takens만 사용 (d=3, τ=5 고정) | Sliding window, 다양한 τ 탐색 |
| Vectorization | Persistence Landscape만 사용 | Persistence Image, Statistics 비교 |
| 서브샘플링 | 300 포인트로 고정 | 데이터셋별 적응적 설정 |
| PH 차수 | H0+H1 고정 | H0만 vs H0+H1 ablation |

### 2.3 문헌 대비 실험 설계 적절성

**Best Practices 준수 여부**:

| 문헌 권장사항 | 본 실험 | 평가 |
|---------------|---------|------|
| Filtration 선택: VR on sliding-window point clouds (dim 2-3, τ=1) | Takens (d=3, τ=5) | ⚠️ 부분 준수 |
| Domain knowledge 활용 | 미활용 | ❌ 미준수 |
| Wasserstein distance 사용 | Persistence Landscape | ⚠️ 대안 사용 |
| 하이브리드 접근 (Hilbert-Huang 등) | 미적용 | ❌ 미준수 |

---

## 3. 결과 분석

### 3.1 정량적 결과 해석

```
성능 격차 분석:
- ECG200:        -8.4%p  (짧은 시계열, 이진 분류 → 상대적 선방)
- Wafer:         -15.7%p (불균형 데이터 → 중간)
- FordA:         -28.9%p (긴 시계열 → 정보 손실)
- ElectricDevices: -32.2%p (다중 클래스 → PH 한계)
```

### 3.2 문헌 맥락에서의 해석

본 실험 결과는 문헌의 발견과 **일관**:

1. **ECG200 (F1=0.755)**: 유사 연구에서 ECG5000에 PH 적용 시 62% 정확도 보고 [2]. 본 실험 결과는 이보다 양호하며, 짧은 시계열에서 PH의 상대적 효과 확인.

2. **다중 클래스 한계**: ElectricDevices(7클래스)에서 F1=0.33은 문헌에서 언급된 "PH 피처가 복잡한 분류 경계 표현 어려움"과 일치.

3. **계산 비용**: UWaveGestureLibraryAll 타임아웃은 문헌의 "Filtrations scale poorly with data size" 경고와 일치.

### 3.3 InceptionTime 대비 성능 격차 원인

| 원인 | 근거 | 영향도 |
|------|------|--------|
| 정보 병목 | Takens → PH → Landscape 과정에서 시간적 미세 패턴 손실 | 높음 |
| 표현력 한계 | 2층 MLP (66K params) vs InceptionTime (1.4M params) | 중간 |
| 하이퍼파라미터 | τ=5, d=3 고정, 최적화 미수행 | 중간 |
| Vectorization | Persistence Landscape만 사용 | 낮음 |

---

## 4. 가설 검증 평가

### 4.1 원래 가설

> "PH 요약 피처를 사용하면 더 작은 모델로 유사 성능을 달성하여 추론 비용을 절감할 수 있다."

### 4.2 검증 결과

| 기준 | 목표 | 결과 | 판정 |
|------|------|------|------|
| 효율 시나리오 | ≤1%p F1 하락 + ≥30% 속도 개선 | 8~32%p 하락 | **기각** |
| 성능 시나리오 | 동일 속도에서 +1~2%p | 전 데이터셋 하락 | **기각** |

### 4.3 가설 실패의 근본 원인

문헌 기반 분석:

1. **가설의 전제 오류**: PH는 "요약"이 아닌 "위상적 관점 변환". 시계열의 시간적 패턴을 위상적 특징으로 변환하면 분류에 중요한 정보가 손실됨.

2. **도메인 불일치**: PH는 본질적으로 **형상(shape)** 분석에 강점. 시계열 분류는 **시간적 패턴** 인식이 핵심이라 PH가 불리.

3. **방법론적 미성숙**: 최신 문헌에서도 시계열에 PH 단독 적용보다 하이브리드(HHT+PH, Graph+PH) 권장.

---

## 5. 실험의 과학적 기여

### 5.1 긍정적 기여

1. **최초의 체계적 비교**: UCR 데이터셋에서 PH-MLP vs InceptionTime 직접 비교는 문헌에 없음. 본 실험이 **최초의 정량적 벤치마크** 제공.

2. **부정적 결과의 가치**: "PH 단독으로는 시계열 분류에 부적합"이라는 결론은 향후 연구 방향 설정에 기여.

3. **재현 가능성**: SageMaker 기반 실험 환경, 코드 공개로 재현 가능.

### 5.2 한계점

1. **Ablation 부족**: Vectorization, embedding 방법 비교 미수행
2. **하이브리드 미탐구**: PH+CNN, PH+RNN 등 결합 접근 미검토
3. **통계적 검정**: t-test 미수행 (p-value 미보고)

---

## 6. 권장 후속 연구

### 6.1 단기 (Track A 확장)

| 우선순위 | 연구 내용 | 근거 |
|----------|-----------|------|
| 1 | Vectorization ablation (Landscape vs Image vs Statistics) | 본 실험 미수행 |
| 2 | Takens 파라미터 최적화 (τ, d grid search) | 문헌 권장 |
| 3 | 하이브리드: PH 피처 + raw 시계열 결합 | 문헌 best practice |

### 6.2 중기 (방법론 개선)

| 연구 내용 | 기대 효과 |
|-----------|-----------|
| Hilbert-Huang Transform + PH (HHTPH) | 비정상 시계열 처리 개선 |
| Graph filtration on correlation matrix | 다변량 시계열 확장 |
| Persistence-aware neural network | End-to-end 학습 |

### 6.3 장기 (Track B/C 전환)

본 실험 결과를 바탕으로 **시계열보다 그래프 데이터**에서 TDA 적용이 더 유망:

- **Track B (Graph)**: Simplicial complex on graph가 문헌에서 효과 입증 [1]
- **Track C (Tensor)**: 모델 압축은 TDA와 독립적 접근 가능

---

## 7. 결론

### 7.1 과학적 결론

본 실험은 **"Persistent Homology 기반 피처 압축이 시계열 분류에서 딥러닝을 대체할 수 없다"**는 부정적 결과를 체계적으로 입증했다. 이는 문헌의 암묵적 경고("도메인 특이적 적용 필요")를 정량적으로 확인한 것이다.

### 7.2 실용적 함의

| 상황 | 권장 |
|------|------|
| 일반 시계열 분류 | InceptionTime 등 딥러닝 사용 |
| 짧은 시계열 + 해석 필요 | PH 피처 보조 사용 고려 |
| 그래프/네트워크 데이터 | TDA 적용 유망 |

### 7.3 최종 평가

| 평가 항목 | 점수 (5점) | 코멘트 |
|-----------|------------|--------|
| 가설 명확성 | 4 | 검증 가능한 정량적 기준 제시 |
| 실험 설계 | 3 | 다양한 데이터셋, 그러나 ablation 부족 |
| 통계적 엄밀성 | 3 | 3회 반복, t-test 미수행 |
| 문헌 정합성 | 4 | 결과가 기존 연구와 일관 |
| 재현 가능성 | 5 | 코드, 환경 문서화 완비 |
| **종합** | **3.8** | 유의미한 부정적 결과, 방법론 개선 여지 |

---

## References

[1] "Classification of Temporal Graphs Using Persistent Homology", arXiv 2025
[2] "Persistent Homology of Featured Time Series Data", AIMS Math 2024
[3] "Persistent Homology for MCI Classification", PMC 2024
[4] "Time Series Classification Algorithm Based on Persistent Homology" (HHTPH)
[5] Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification", 2020
