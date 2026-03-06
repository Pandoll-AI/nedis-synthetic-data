# Revision v0.5 — NEDIS 4.0 Migration + Clinical Coherence

Date: 2026-03-07

## Overview

NEDIS 4.0 변수명 전면 마이그레이션 완료 및 임상 일관성 3단계 개선을 수행합니다.

- **Phase 0-9**: 39개 Python 파일, ~2,097개 참조를 표본자료 변수명에서 NEDIS 4.0 DB 칼럼 ID로 전환
- **Phase A**: 구조적 결측값 보존 (co-missing 패턴 재현)
- **Phase B**: KTAS→주증상→진료과 조건부 생성 체인
- **Phase C**: KTAS 프로토콜 코드(ptmikpr1) 6자리 합성 생성

---

## NEDIS 4.0 변수명 마이그레이션 (Phase 0-9)

### 핵심 변경

| 구분 | 변경 전 | 변경 후 |
|------|---------|---------|
| 칼럼명 | `emorg_cd`, `vst_dt`, `pat_age_gr` | `ptmiemcd`, `ptmiindt`, `ptmibrtd` |
| 성별 값 | M/F | 1/2 |
| DB 접근 | 직접 쿼리 | DuckDB VIEW alias |

### 신규 모듈: `src/core/nedis4_converter.py`

- `SAMPLE_TO_NEDIS4`: 26쌍 매핑 딕셔너리 (단일 진실 소스)
- `create_source_view_sql()`: VIEW DDL 자동 생성
- `generate_synthetic_birthdates_vectorized()`: 연령군→합성 YYYYMMDD
- `generate_synthetic_zipcodes_vectorized()`: 시도코드→합성 12자리 우편번호

### 파이프라인 VIEW 자동 생성

`scripts/run_vectorized_pipeline.py`에 `_ensure_nedis4_view()` 추가:
- 소스 DB가 구 변수명 사용 시 자동으로 NEDIS 4.0 VIEW 생성
- `ptmiemcd` 칼럼 존재 여부로 판단

---

## Phase A: 구조적 결측값 보존

### 문제

원본 데이터에서 KTAS·주증상·진료과는 **구조적 동시 결측(co-missing)** 패턴을 가짐:
- KTAS 결측 시 주증상 98.0%, 진료과 99.9% 동시 결측
- 이전 합성기는 결측을 무시하여 분포가 오염됨

### 구현

**`src/analysis/pattern_analyzer.py`** — `analyze_missing_value_rates()`:
- 독립 결측률, 상관 그룹(KTAS→주증상+진료과), 조건부 결측률 분석
- SQL에서 `-` 값도 결측으로 판정 (`IN ('', '-')`)

**`src/vectorized/patient_generator.py`** — `_apply_missing_values()`:
1. KTAS 결측 마스크 생성 (primary_missing_rate ≈ 34%)
2. Co-missing: KTAS 결측 → ptmikpr1, ptmimnsy, ptmidept 동시 `-`
3. Residual: KTAS 유효 레코드에 조건부 결측 적용 (ptmimnsy 2.5%, ptmidept 7.3%)
4. Independent: ptmizipc(7.1%), ptmiemrt(0.1%), ptmiinmn(0.1%) 독립 적용

**패턴 분석 SQL 수정**: `AND ptmimnsy != '-'` 조건 추가하여 결측값이 유효값 분포에 포함되지 않도록 방지

### 결과

| 변수 | 원본 | 합성 | 차이 |
|------|------|------|------|
| ptmikts1 | 34.0% | 34.2% | +0.1% |
| ptmimnsy | 35.0% | 35.6% | +0.6% |
| ptmidept | 38.9% | 38.9% | 0.0% |
| ptmizipc | 7.1% | 7.1% | 0.0% |

---

## Phase B: KTAS-주증상-진료과 조건부 생성

### 문제

이전: KTAS, 주증상, 진료과를 **독립적**으로 생성 → 임상적으로 부자연스러운 조합 발생

### 구현

생성 체인: **KTAS → 주증상 → 진료과 → 치료결과**

**`src/analysis/pattern_analyzer.py`**:
- `analyze_chief_complaint_patterns()`: P(ptmimnsy | ptmikts1, ptmibrtd, ptmisexx)
  - 계층적 대안: ktas_age_sex → ktas_age → ktas → age_sex
- `analyze_department_patterns()`: P(ptmidept | ptmikts1, ptmimnsy)
  - 계층적 대안: ktas_symptom → ktas → sym_symptom → all

**`src/vectorized/patient_generator.py`**:
- `_conditional_symptom_generation()`: KTAS/연령/성별 조건부 주증상 벡터 생성
- `_conditional_department_generation()`: KTAS/주증상 조건부 진료과 벡터 생성
- 주증상·진료과가 `_generate_independent_clinical_attributes`에서 `_generate_conditional_clinical_attributes`로 이동

### 결과 (KTAS별 Top-3 일치율)

| KTAS | 주증상 Top-3 일치 | 진료과 Top-3 일치 |
|------|-------------------|-------------------|
| 1 | 2/3 | 2/3 |
| 2 | 3/3 | 3/3 |
| 3 | 3/3 | 3/3 |
| 4 | 3/3 | 3/3 |
| 5 | 3/3 | 3/3 |

Categorical score: 0.697 → **0.963** (+38%)

---

## Phase C: KTAS 프로토콜 코드 합성 생성

### 문제

NEDIS 4.0의 `ptmikpr1`은 6자리 KTAS 프로토콜 코드이나, 원본 표본(`ktas01`)은 정수 1~5로 축소됨.

### 구현

**코드 구조**: `[연령구분 1자리][대분류 1자리][중분류 1자리][소분류 2자리][감염코드 1자리]`

- 1단계: A(성인≥15세) / P(소아<15세) — 연령군 코드로 판단
- 2단계: 주증상(UMLS) → 신체계통 대분류 매핑 (`_SYMPTOM_MAJOR_CLASS`)
- 3-4단계: 주증상+KTAS 해시 기반 결정론적 생성
- 감염코드: 0(비감염 90%) / 1(감염 5%) / 9(미상 5%)

**DDL 변경**: `ptmikpr1 INTEGER` → `VARCHAR`

---

## 300K 검증 결과 요약

| 항목 | JSD / 차이 | 판정 |
|------|-----------|------|
| KTAS 등급 | JSD 0.000015 | 완벽 |
| 성별 | JSD 0.000000 | 완벽 |
| 연령군 | JSD 0.000002 | 완벽 |
| 내원수단 | JSD 0.000004 | 완벽 |
| 치료결과 | JSD 0.001189 | 양호 |
| 결측률 (6개) | 전 항목 ±0.6% | 완벽 |
| 활력징후 (6개) | 평균 차이 ≤0.2 | 완벽 |
| 시간대별 | 최대 0.1%p | 완벽 |
| 월별 | 최대 0.5%p | 양호 |

### 변경 파일 목록

**신규**:
- `src/core/nedis4_converter.py`

**주요 수정** (39개 Python + 1 YAML + 1 HTML):
- `src/analysis/pattern_analyzer.py` — 조건부 패턴 분석, 결측률 분석
- `src/vectorized/patient_generator.py` — 조건부 생성 체인, 결측 적용, KTAS 코드
- `src/clinical/dag_generator.py` — DDL 스키마 변경
- `scripts/run_vectorized_pipeline.py` — VIEW 자동 생성
- `src/temporal/comprehensive_time_gap_synthesizer.py` — NEDIS 4.0 칼럼 매핑
- 기타 34개 파일: 변수명 일괄 치환
