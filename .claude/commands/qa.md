# Q&A 기록 스킬

수학적/기술적 개념에 대한 질문과 답변을 자동으로 기록합니다.

## 사용법

```
/qa <질문 또는 개념>
```

## 예시

```
/qa Persistent Homology가 뭐야?
/qa Takens embedding의 수학적 원리
/qa Simplicial complex와 graph의 차이점
```

## 동작

$ARGUMENTS 파라미터로 전달된 질문/개념에 대해:

1. 개념을 명확하고 체계적으로 설명합니다
2. 필요시 수식, 예시, 다이어그램을 포함합니다
3. 설명 완료 후 `docs/qa/` 디렉토리에 마크다운 파일로 저장합니다

## 파일 저장 규칙

- 파일명: `YYYYMMDD-HHMMSS-<slug>.md` (예: `20250118-143052-persistent-homology.md`)
- 저장 위치: `docs/qa/`

## 파일 구조

```markdown
---
date: YYYY-MM-DD HH:MM:SS
tags: [관련 태그들]
category: <카테고리>
---

# Q: <질문>

## A: <답변 제목>

<상세 설명>

### 핵심 개념
- ...

### 수학적 정의 (해당시)
- ...

### 예시
- ...

### 관련 개념
- ...
```

## 카테고리 분류

- `topology`: 위상수학 (Persistent Homology, Simplicial Complex 등)
- `algebra`: 대수학 (Tensor, Matrix Decomposition 등)
- `geometry`: 기하학 (Manifold, Embedding 등)
- `ml`: 머신러닝 (Neural Network, Optimization 등)
- `statistics`: 통계학
- `general`: 기타

## 지침

1. **명확성**: 비전문가도 이해할 수 있도록 설명하되, 수학적 엄밀성 유지
2. **구조화**: 정의 → 직관적 설명 → 예시 → 응용 순서로 설명
3. **시각화**: 가능하면 Mermaid 다이어그램 (흑백) 사용
4. **연결성**: 관련 개념 링크 제공
5. **실용성**: 이 프로젝트(topology-efficient-deep-learning)와의 연관성 언급

## 실행 절차

1. 질문 분석 및 카테고리 분류
2. 체계적인 답변 작성
3. `docs/qa/` 디렉토리에 파일 저장
4. 저장 완료 메시지 출력

질문: $ARGUMENTS
