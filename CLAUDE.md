# Project Instructions

이 프로젝트는 위상/구조 수학을 활용한 딥러닝 효율성 검증 실험입니다.

## Q&A 스킬

이 프로젝트는 수학적 개념 학습을 위한 Q&A 자동 기록 시스템을 포함합니다.

### 사용 가능한 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `/qa <질문>` | 개념 설명 후 Q&A 문서로 저장 | `/qa Persistent Homology가 뭐야?` |
| `/qa-list [카테고리]` | 저장된 Q&A 목록 표시 | `/qa-list topology` |
| `/qa-merge` | 모든 Q&A를 하나의 문서로 병합 | `/qa-merge --by-category` |

### Q&A 카테고리

- `topology`: 위상수학 (Persistent Homology, Simplicial Complex 등)
- `algebra`: 대수학 (Tensor, Matrix Decomposition 등)
- `geometry`: 기하학 (Manifold, Embedding 등)
- `ml`: 머신러닝 (Neural Network, Optimization 등)
- `statistics`: 통계학
- `general`: 기타

### 파일 저장 위치

- 개별 Q&A: `docs/qa/YYYYMMDD-HHMMSS-<slug>.md`
- 병합 문서: `docs/CONCEPTS.md`

## 프로젝트 구조

```
├── configs/          # 실험 설정 (YAML)
├── src/
│   ├── data/         # 데이터 로더
│   ├── models/       # 모델 구현
│   ├── tda/          # TDA 유틸리티
│   └── utils/        # 공통 유틸리티
├── experiments/      # 실험 스크립트
├── docs/
│   └── qa/           # Q&A 문서 저장소
├── notebooks/        # 분석 노트북
└── results/          # 실험 결과
```

## 코드 스타일

- Python: Black, isort 사용
- Type hints 권장
- Docstring: Google style

## 실험 실행

```bash
# Track A: PH 기반 시계열 분류
python experiments/track_a/train.py --model ph_mlp --dataset ECG200

# 베이스라인 비교
python experiments/track_a/train.py --model cnn --dataset ECG200
```
