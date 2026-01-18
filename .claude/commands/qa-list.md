# Q&A 목록 스킬

저장된 Q&A 문서 목록을 표시합니다.

## 사용법

```
/qa-list [카테고리]
```

## 옵션

- 카테고리 미지정: 전체 목록 표시
- 카테고리 지정: 해당 카테고리만 표시 (topology, algebra, geometry, ml, statistics, general)

## 동작

1. `docs/qa/` 디렉토리 스캔
2. 각 파일의 메타데이터 추출
3. 테이블 형태로 목록 표시

## 출력 형식

```
## Q&A 목록

총 N개의 Q&A가 저장되어 있습니다.

| # | 날짜 | 카테고리 | 질문 | 파일 |
|---|------|----------|------|------|
| 1 | 2025-01-18 | topology | Persistent Homology가 뭐야? | 20250118-143052-persistent-homology.md |
| 2 | 2025-01-18 | algebra | Tensor Train 분해란? | 20250118-150023-tensor-train.md |
...

### 카테고리별 통계

| 카테고리 | 개수 |
|----------|------|
| topology | 5 |
| algebra | 3 |
| ml | 2 |
```

## 실행 절차

1. `docs/qa/` 디렉토리 확인
2. 파일 목록 수집 및 메타데이터 파싱
3. 테이블 형식으로 출력
4. 카테고리별 통계 출력

필터: $ARGUMENTS
