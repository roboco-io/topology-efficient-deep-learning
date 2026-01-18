# Q&A 병합 스킬

저장된 모든 Q&A 문서를 하나의 통합 문서로 병합합니다.

## 사용법

```
/qa-merge [옵션]
```

## 옵션

- `--by-category`: 카테고리별로 그룹화 (기본값)
- `--by-date`: 날짜순으로 정렬
- `--output <파일명>`: 출력 파일명 지정 (기본: `docs/CONCEPTS.md`)

## 동작

1. `docs/qa/` 디렉토리의 모든 Q&A 파일 스캔
2. 각 파일의 frontmatter에서 메타데이터 추출
3. 지정된 방식으로 정렬/그룹화
4. 통합 문서 생성

## 출력 구조

```markdown
# 개념 정리 (Concepts Reference)

> 이 문서는 프로젝트 진행 중 학습한 수학적/기술적 개념들을 정리한 것입니다.
> 자동 생성됨: YYYY-MM-DD HH:MM:SS

## 목차

- [Topology](#topology)
  - [Persistent Homology](#persistent-homology)
  - [Simplicial Complex](#simplicial-complex)
- [Algebra](#algebra)
  - [Tensor Decomposition](#tensor-decomposition)
...

---

## Topology

### Persistent Homology

**Q: Persistent Homology가 뭐야?**

<답변 내용>

---

### Simplicial Complex

**Q: Simplicial Complex란?**

<답변 내용>

---

## Algebra

...
```

## 카테고리 순서

1. topology
2. algebra
3. geometry
4. ml
5. statistics
6. general

## 실행 절차

1. `docs/qa/` 디렉토리 스캔
2. 각 파일의 frontmatter 파싱
3. 카테고리/날짜별 그룹화
4. 목차 생성
5. 통합 문서 작성
6. `docs/CONCEPTS.md`에 저장

옵션: $ARGUMENTS
